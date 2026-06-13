// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Render the in-process metrics registry (plus procfs deltas and the optional
//! eBPF syscall snapshot) into the nested JSON profile report.
//!
//! The report is `{target, engine, query, wall_ms, metrics, notes}` where
//! `metrics` is a flat map of numeric dotted keys grouped by section (`scan.*`,
//! `io.*`, `metadata.*`, `pruning.*`, `filter.*`, `decode.<encoding>.*`,
//! `pushdown_fallback.*`, `memory.*`, `cold.*`, and — with `--syscalls` —
//! `io.read_*`). `notes` is a parallel map of human-readable interpretations for
//! the non-obvious metrics (kept separate so `metrics` stays purely numeric).

use std::collections::BTreeMap;

use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use vortex::metrics::Metric;
use vortex::metrics::MetricValue;
#[cfg(feature = "profile-ebpf")]
use vortex::metrics::profile::hist_percentile;
use vortex::metrics::profile::procfs::ProcReading;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::host::collect::Snapshot;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::types::BIO_STAT_BYTES;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::types::BIO_STAT_READS;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::types::FUTEX_STAT_WAITS;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::types::NET_STAT_CONNECTIONS;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::types::NET_STAT_RETRANSMITS;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::types::READ_STAT_BYTES;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::types::READ_STAT_COUNT;

#[cfg(not(feature = "profile-ebpf"))]
type Snapshot = ();

/// Resolve a PMU [`ContextKey`](vortex_ebpf::types::ContextKey) to its metric
/// name: a decode's `id` is an encoding's interned symbol (resolved via the
/// process interner); other kinds format their instance index.
#[cfg(feature = "profile-ebpf")]
fn ctx_name(key: &vortex_ebpf::types::ContextKey) -> String {
    use vortex_ebpf::ContextKind;
    match ContextKind::from_u32(key.kind) {
        Some(ContextKind::Decode) => vortex_session::registry::Id::resolve_u64(key.id)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("decode#{}", key.id)),
        Some(ContextKind::Scan) => format!("scan[{}]", key.id),
        Some(ContextKind::Prune) => format!("prune.conjunct[{}]", key.id),
        Some(ContextKind::Filter) => format!("filter.conjunct[{}]", key.id),
        Some(ContextKind::IoWait) => format!("io_wait[{}]", key.id),
        Some(ContextKind::Fallback) => format!("fallback[{}]", key.id),
        None => format!("kind{}[{}]", key.kind, key.id),
    }
}

/// Reads below this size are counted as "tiny" (log2 bucket < 12 == < 4 KiB).
#[cfg(feature = "profile-ebpf")]
const TINY_READ_LOG2: usize = 12;

fn round4(x: f64) -> f64 {
    (x * 1.0e4).round() / 1.0e4
}

fn div(num: f64, den: f64) -> f64 {
    if den > 0.0 { num / den } else { 0.0 }
}

/// Look up the value of `key` among a metric's labels.
fn label<'a>(m: &'a Metric, key: &str) -> Option<&'a str> {
    m.labels()
        .iter()
        .find(|l| l.key() == key)
        .map(|l| l.value())
}

/// Sum the values of all counters named `name` (across labels).
fn counter_sum(metrics: &[Metric], name: &str) -> u64 {
    metrics
        .iter()
        .filter(|m| m.name() == name)
        .filter_map(|m| match m.value() {
            MetricValue::Counter(c) => Some(c.value()),
            _ => None,
        })
        .sum()
}

/// The counter named `name` carrying `label_key == label_val`, if any.
fn counter_for(metrics: &[Metric], name: &str, label_key: &str, label_val: &str) -> Option<u64> {
    metrics
        .iter()
        .find(|m| m.name() == name && label(m, label_key) == Some(label_val))
        .and_then(|m| match m.value() {
            MetricValue::Counter(c) => Some(c.value()),
            _ => None,
        })
}

/// The timer named `name` carrying `label_key == label_val`, as
/// `(count, total_ms, p50_ms, p99_ms)`.
fn timer_for(
    metrics: &[Metric],
    name: &str,
    label_key: &str,
    label_val: &str,
) -> Option<(u64, f64, f64, f64)> {
    metrics
        .iter()
        .filter(|m| m.name() == name && label(m, label_key) == Some(label_val))
        .filter_map(|m| match m.value() {
            MetricValue::Timer(t) => Some(t),
            _ => None,
        })
        .max_by_key(|t| t.count())
        .map(|t| {
            let ms =
                |d: Option<std::time::Duration>| d.map(|d| d.as_secs_f64() * 1.0e3).unwrap_or(0.0);
            (
                t.count() as u64,
                t.total().as_secs_f64() * 1.0e3,
                ms(t.quantile(0.50)),
                ms(t.quantile(0.99)),
            )
        })
}

/// Distinct values of `label_key` across metrics named `name`, in first-seen order.
fn label_values(metrics: &[Metric], name: &str, label_key: &str) -> Vec<String> {
    let mut seen = Vec::new();
    for m in metrics.iter().filter(|m| m.name() == name) {
        if let Some(v) = label(m, label_key)
            && !seen.iter().any(|s| s == v)
        {
            seen.push(v.to_string());
        }
    }
    seen
}

/// Build the `metrics` map.
fn metrics_map(
    metrics: &[Metric],
    before: &ProcReading,
    after: &ProcReading,
    syscalls: Option<&Snapshot>,
) -> Map<String, Value> {
    let mut m = Map::new();
    let mut put = |k: String, v: Value| {
        m.insert(k, v);
    };

    #[cfg(not(feature = "profile-ebpf"))]
    let _ = syscalls;

    // Scan lifecycle.
    let rows_out = counter_sum(metrics, "vortex.scan.rows_out");
    put("scan.rows_out".into(), json!(rows_out));
    if let Some((splits, _total, p50, p99)) = split_timer(metrics) {
        put("scan.splits".into(), json!(splits));
        put("scan.split_duration_p50_ms".into(), json!(round4(p50)));
        put("scan.split_duration_p99_ms".into(), json!(round4(p99)));
    }
    if let Some(peak) = gauge(metrics, "vortex.scan.split_peak") {
        // The gauge holds a small non-negative integer count (peak concurrency).
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let peak = peak as u64;
        put("scan.split_peak_concurrent".into(), json!(peak));
    }

    // Pruning, aggregated + per conjunct.
    let prune_in = counter_sum(metrics, "vortex.prune.rows_in");
    let prune_kept = counter_sum(metrics, "vortex.prune.rows_kept");
    put("pruning.rows_in".into(), json!(prune_in));
    put("pruning.rows_kept".into(), json!(prune_kept));
    put(
        "pruning.pruned_ratio".into(),
        json!(round4(if prune_in > 0 {
            1.0 - div(prune_kept as f64, prune_in as f64)
        } else {
            0.0
        })),
    );
    for c in label_values(metrics, "vortex.prune.rows_in", "conjunct") {
        emit_conjunct(&mut put, metrics, "pruning", "vortex.prune", &c);
    }

    // Filter, aggregated + per conjunct.
    let filt_in = counter_sum(metrics, "vortex.filter.rows_in");
    let filt_kept = counter_sum(metrics, "vortex.filter.rows_kept");
    put("filter.rows_in".into(), json!(filt_in));
    put("filter.rows_kept".into(), json!(filt_kept));
    put(
        "filter.selectivity".into(),
        json!(round4(div(filt_kept as f64, filt_in as f64))),
    );
    for c in label_values(metrics, "vortex.filter.rows_in", "conjunct") {
        emit_conjunct(&mut put, metrics, "filter", "vortex.filter", &c);
    }

    // IO.
    let segment_requests = counter_sum(metrics, "vortex.io.segment_requests");
    let logical_bytes = counter_sum(metrics, "vortex.io.segment_logical_bytes");
    let physical_bytes = counter_sum(metrics, "vortex.io.physical_bytes");
    let physical_reads = counter_sum(metrics, "io.requests.individual")
        + counter_sum(metrics, "io.requests.coalesced");
    put("io.segment_requests".into(), json!(segment_requests));
    put("io.logical_segment_bytes".into(), json!(logical_bytes));
    put("io.physical_reads".into(), json!(physical_reads));
    put("io.physical_read_bytes".into(), json!(physical_bytes));
    put(
        "io.read_amplification".into(),
        json!(round4(div(physical_bytes as f64, logical_bytes as f64))),
    );
    put(
        "io.coalescing_factor_avg".into(),
        json!(round4(div(segment_requests as f64, physical_reads as f64))),
    );
    if let Some((p50, p99)) = histogram_pcts(metrics, "vortex.io.segment_size") {
        put("io.segment_size_p50_bytes".into(), json!(round4(p50)));
        put("io.segment_size_p99_bytes".into(), json!(round4(p99)));
    }

    // Metadata.
    put(
        "metadata.footer_reads".into(),
        json!(counter_sum(metrics, "vortex.io.footer_reads")),
    );
    put(
        "metadata.footer_bytes".into(),
        json!(counter_sum(metrics, "vortex.io.footer_bytes")),
    );

    // Decode economics, per encoding.
    put(
        "decode.total_calls".into(),
        json!(counter_sum(metrics, "vortex.decode.calls")),
    );
    put(
        "decode.total_ms".into(),
        json!(round4(
            counter_sum(metrics, "vortex.decode.nanos") as f64 / 1.0e6
        )),
    );
    for enc in label_values(metrics, "vortex.decode.calls", "encoding") {
        let g = |name: &str| counter_for(metrics, name, "encoding", &enc).unwrap_or(0);
        let (calls, nanos, rows, bytes) = (
            g("vortex.decode.calls"),
            g("vortex.decode.nanos"),
            g("vortex.decode.rows"),
            g("vortex.decode.bytes"),
        );
        let secs = nanos as f64 / 1.0e9;
        put(format!("decode.{enc}.calls"), json!(calls));
        put(
            format!("decode.{enc}.ms"),
            json!(round4(nanos as f64 / 1.0e6)),
        );
        put(format!("decode.{enc}.rows"), json!(rows));
        put(format!("decode.{enc}.bytes"), json!(bytes));
        if secs > 0.0 {
            put(
                format!("decode.{enc}.rows_per_sec"),
                json!(round4(rows as f64 / secs)),
            );
            put(
                format!("decode.{enc}.mb_per_sec"),
                json!(round4(bytes as f64 / secs / 1.0e6)),
            );
        }
    }

    // Compute-on-encoded pushdown misses.
    put(
        "pushdown_fallback.total".into(),
        json!(counter_sum(metrics, "vortex.canonicalize_fallback")),
    );
    for m_ref in metrics
        .iter()
        .filter(|m| m.name() == "vortex.canonicalize_fallback")
    {
        if let (Some(parent), Some(child), MetricValue::Counter(c)) =
            (label(m_ref, "parent"), label(m_ref, "child"), m_ref.value())
        {
            put(
                format!("pushdown_fallback.{parent}=>{child}.count"),
                json!(c.value()),
            );
        }
    }

    // Optional eBPF layers; each section is emitted only if its layer was attached.
    #[cfg(feature = "profile-ebpf")]
    if let Some(snap) = syscalls {
        // Read-syscall layer (`--syscalls`).
        if let Some(r) = &snap.read {
            let count = r
                .read_stats
                .get(READ_STAT_COUNT as usize)
                .copied()
                .unwrap_or(0);
            let bytes = r
                .read_stats
                .get(READ_STAT_BYTES as usize)
                .copied()
                .unwrap_or(0);
            let tiny: u64 = r.read_hist.iter().take(TINY_READ_LOG2).sum();
            put("io.read_syscalls".into(), json!(count));
            put("io.read_syscall_bytes".into(), json!(bytes));
            put(
                "io.read_size_p50_bytes".into(),
                json!(hist_percentile(&r.read_hist, 0.50)),
            );
            put(
                "io.read_size_p99_bytes".into(),
                json!(hist_percentile(&r.read_hist, 0.99)),
            );
            put("io.tiny_reads".into(), json!(tiny));
            // What "tiny" means, so the threshold is self-documenting in the report.
            put(
                "io.tiny_read_threshold_bytes".into(),
                json!(1u64 << TINY_READ_LOG2),
            );
            // Per-bucket size breakdown of populated log2 buckets, keyed by the
            // bucket's lower bound in bytes. Makes the tiny-read source diagnosable:
            // a spike in `io.read_size_hist.64` is 64..128 B reads, not runtime noise.
            for (bucket, &cnt) in r.read_hist.iter().enumerate() {
                if cnt > 0 {
                    let lo = 1u64
                        .checked_shl(u32::try_from(bucket).unwrap_or(u32::MAX))
                        .unwrap_or(u64::MAX);
                    put(format!("io.read_size_hist.{lo}"), json!(cnt));
                }
            }
            // Bare read-syscall latency (ns buckets → µs): the kernel-side service
            // time, separate from in-process queueing on the blocking pool.
            put(
                "io.read_latency_p50_us".into(),
                json!(round4(hist_percentile(&r.rdlat_hist, 0.50) / 1.0e3)),
            );
            put(
                "io.read_latency_p99_us".into(),
                json!(round4(hist_percentile(&r.rdlat_hist, 0.99) / 1.0e3)),
            );
        }

        // PMU layer (`--pmu`).
        for (key, p) in &snap.pmu {
            let enc = ctx_name(key);
            put(format!("pmu.{enc}.cycle_samples"), json!(p.cycles));
            put(
                format!("pmu.{enc}.instruction_samples"),
                json!(p.instructions),
            );
            put(
                format!("pmu.{enc}.cache_miss_samples"),
                json!(p.cache_misses),
            );
            put(
                format!("pmu.{enc}.branch_miss_samples"),
                json!(p.branch_misses),
            );
            put(
                format!("pmu.{enc}.llc_load_miss_samples"),
                json!(p.llc_load_misses),
            );
            put(
                format!("pmu.{enc}.stalled_cycle_samples"),
                json!(p.stalled_cycles),
            );
        }

        // Block-layer device-read layer (`--bio`): ground truth past the page
        // cache. The block layer serves reads asynchronously (read-ahead,
        // writeback) and carries no reliable originating pid, so these are
        // *system-wide* for the run — absolute device reads/bytes and the
        // service-latency distribution, which is what answers "is the disk slow".
        // No ratio against the pid-scoped syscall/logical byte totals is emitted:
        // mixing system-wide device bytes with per-process bytes is unsound.
        // Latency buckets are log2 nanoseconds; report as milliseconds.
        if let Some(b) = &snap.bio {
            let dev_reads = b.stats.get(BIO_STAT_READS as usize).copied().unwrap_or(0);
            let dev_bytes = b.stats.get(BIO_STAT_BYTES as usize).copied().unwrap_or(0);
            put("bio.device_reads".into(), json!(dev_reads));
            put("bio.device_read_bytes".into(), json!(dev_bytes));
            put(
                "bio.read_latency_p50_ms".into(),
                json!(round4(hist_percentile(&b.lat_hist, 0.50) / 1.0e6)),
            );
            put(
                "bio.read_latency_p99_ms".into(),
                json!(round4(hist_percentile(&b.lat_hist, 0.99) / 1.0e6)),
            );
        }

        // Futex-contention layer (`--locks`): blocking lock-wait time (ns → µs).
        if let Some(l) = &snap.locks {
            let waits = l.stats.get(FUTEX_STAT_WAITS as usize).copied().unwrap_or(0);
            put("lock.futex_waits".into(), json!(waits));
            put(
                "lock.futex_wait_p50_us".into(),
                json!(round4(hist_percentile(&l.wait_hist, 0.50) / 1.0e3)),
            );
            put(
                "lock.futex_wait_p99_us".into(),
                json!(round4(hist_percentile(&l.wait_hist, 0.99) / 1.0e3)),
            );
        }

        // Network layer (`--net`, system-wide): TCP retransmits and connection
        // setup. Connect latency buckets are log2 nanoseconds (→ milliseconds).
        if let Some(n) = &snap.net {
            let retransmits = n
                .stats
                .get(NET_STAT_RETRANSMITS as usize)
                .copied()
                .unwrap_or(0);
            let connections = n
                .stats
                .get(NET_STAT_CONNECTIONS as usize)
                .copied()
                .unwrap_or(0);
            put("net.tcp_retransmits".into(), json!(retransmits));
            put("net.tcp_connections".into(), json!(connections));
            put(
                "net.connect_latency_p50_ms".into(),
                json!(round4(hist_percentile(&n.connect_lat_hist, 0.50) / 1.0e6)),
            );
            put(
                "net.connect_latency_p99_ms".into(),
                json!(round4(hist_percentile(&n.connect_lat_hist, 0.99) / 1.0e6)),
            );
        }

        // Off-CPU / scheduler layer (`--offcpu`): blocked time attributed to the
        // Vortex phase in flight, plus runqueue latency (ns buckets → µs).
        if let Some(o) = &snap.offcpu {
            let mut total_ns = 0u64;
            for (key, s) in &o.stats {
                let name = ctx_name(key);
                put(
                    format!("offcpu.{name}.ms"),
                    json!(round4(s.total_ns as f64 / 1.0e6)),
                );
                put(format!("offcpu.{name}.count"), json!(s.count));
                total_ns += s.total_ns;
            }
            put(
                "offcpu.total_ms".into(),
                json!(round4(total_ns as f64 / 1.0e6)),
            );
            put(
                "sched.runqueue_latency_p50_us".into(),
                json!(round4(hist_percentile(&o.runq_hist, 0.50) / 1.0e3)),
            );
            put(
                "sched.runqueue_latency_p99_us".into(),
                json!(round4(hist_percentile(&o.runq_hist, 0.99) / 1.0e3)),
            );
        }
    }

    // Memory + cold/warm (procfs deltas).
    put("memory.rss_peak_bytes".into(), json!(after.vm_hwm));
    put(
        "memory.rss_peak_delta_bytes".into(),
        json!(after.vm_hwm.saturating_sub(before.vm_rss)),
    );
    let cold_bytes = after.read_bytes.saturating_sub(before.read_bytes);
    let logical = after.rchar.saturating_sub(before.rchar);
    put("cold.storage_read_bytes".into(), json!(cold_bytes));
    put(
        "cold.page_cache_hit_ratio".into(),
        // Storage reads (block-aligned, with readahead) can exceed logical
        // `rchar`, so clamp into [0, 1].
        json!(round4(if logical > 0 {
            (1.0 - div(cold_bytes as f64, logical as f64)).clamp(0.0, 1.0)
        } else {
            0.0
        })),
    );
    put(
        "cold.major_faults".into(),
        json!(after.majflt.saturating_sub(before.majflt)),
    );

    m
}

/// Human-readable interpretations for the non-obvious metrics, keyed parallel to
/// the metric they explain. Kept out of `metrics` so that map stays purely
/// numeric (and diff-friendly). Currently explains the compute-on-encoded
/// pushdown misses, which are otherwise just an opaque `parent=>child` count.
fn notes_map(metrics: &[Metric]) -> Map<String, Value> {
    let mut n = Map::new();
    for m in metrics
        .iter()
        .filter(|m| m.name() == "vortex.canonicalize_fallback")
    {
        if let (Some(parent), Some(child), MetricValue::Counter(c)) =
            (label(m, "parent"), label(m, "child"), m.value())
        {
            n.insert(
                format!("pushdown_fallback.{parent}=>{child}"),
                json!(format!(
                    "{parent} could not push compute into {child}: the child was \
                     canonicalized {} time(s) before the operation ran (compute-on-encoded \
                     fallback). Lower is better; a non-zero count means the encoding was \
                     materialized instead of operated on in place.",
                    c.value()
                )),
            );
        }
    }
    n
}

/// Emit per-conjunct `rows_in`/`rows_kept`/`ms` (+ `selectivity` for filter).
fn emit_conjunct(
    put: &mut impl FnMut(String, Value),
    metrics: &[Metric],
    section: &str,
    metric_prefix: &str,
    conjunct: &str,
) {
    let rows_in = counter_for(
        metrics,
        &format!("{metric_prefix}.rows_in"),
        "conjunct",
        conjunct,
    )
    .unwrap_or(0);
    let rows_kept = counter_for(
        metrics,
        &format!("{metric_prefix}.rows_kept"),
        "conjunct",
        conjunct,
    )
    .unwrap_or(0);
    let ms = timer_for(
        metrics,
        &format!("{metric_prefix}.time"),
        "conjunct",
        conjunct,
    )
    .map(|(_, total, ..)| total)
    .unwrap_or(0.0);
    put(
        format!("{section}.conjunct.{conjunct}.rows_in"),
        json!(rows_in),
    );
    put(
        format!("{section}.conjunct.{conjunct}.rows_kept"),
        json!(rows_kept),
    );
    put(
        format!("{section}.conjunct.{conjunct}.ms"),
        json!(round4(ms)),
    );
    if section == "filter" {
        put(
            format!("filter.conjunct.{conjunct}.selectivity"),
            json!(round4(div(rows_kept as f64, rows_in as f64))),
        );
    }
}

/// The richest split-duration timer as `(count, total_ms, p50_ms, p99_ms)`.
fn split_timer(metrics: &[Metric]) -> Option<(u64, f64, f64, f64)> {
    metrics
        .iter()
        .filter(|m| m.name() == "vortex.scan.split_time")
        .filter_map(|m| match m.value() {
            MetricValue::Timer(t) => Some(t),
            _ => None,
        })
        .max_by_key(|t| t.count())
        .map(|t| {
            let ms =
                |d: Option<std::time::Duration>| d.map(|d| d.as_secs_f64() * 1.0e3).unwrap_or(0.0);
            (
                t.count() as u64,
                t.total().as_secs_f64() * 1.0e3,
                ms(t.quantile(0.50)),
                ms(t.quantile(0.99)),
            )
        })
}

/// The maximum value among gauges named `name`. (The same metric can be
/// registered by more than one file open; the peak high-water mark wins.)
fn gauge(metrics: &[Metric], name: &str) -> Option<f64> {
    metrics
        .iter()
        .filter(|m| m.name() == name)
        .filter_map(|m| match m.value() {
            MetricValue::Gauge(g) => Some(g.value()),
            _ => None,
        })
        .reduce(f64::max)
}

/// `(p50, p99)` of the richest histogram named `name` (the same metric can be
/// registered by more than one file open; the one with the most samples wins).
fn histogram_pcts(metrics: &[Metric], name: &str) -> Option<(f64, f64)> {
    metrics
        .iter()
        .filter(|m| m.name() == name)
        .filter_map(|m| match m.value() {
            MetricValue::Histogram(h) => Some(h),
            _ => None,
        })
        .max_by_key(|h| h.count())
        .map(|h| {
            (
                h.quantile(0.50).unwrap_or(0.0),
                h.quantile(0.99).unwrap_or(0.0),
            )
        })
}

/// Render the full nested report.
#[allow(clippy::too_many_arguments)]
pub fn render(
    target: &str,
    engine: &str,
    query: &str,
    wall_ms: f64,
    metrics: &[Metric],
    before: &ProcReading,
    after: &ProcReading,
    syscalls: Option<&Snapshot>,
) -> Value {
    // BTreeMap → sorted keys for stable, diff-friendly output.
    let map: BTreeMap<String, Value> = metrics_map(metrics, before, after, syscalls)
        .into_iter()
        .collect();
    let notes: BTreeMap<String, Value> = notes_map(metrics).into_iter().collect();
    json!({
        "target": target,
        "engine": engine,
        "query": query,
        "wall_ms": round4(wall_ms),
        "metrics": map,
        "notes": notes,
    })
}
