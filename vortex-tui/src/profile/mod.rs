// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vx profile` — format profiling integrated into `vx`.
//!
//! Format-semantic metrics (decode economics, pruning, filter, IO, splits,
//! pushdown misses) are collected **in-process** through Vortex's own
//! `vortex-metrics`: `vx` installs a [`ScanProfiler`] on the session, runs the
//! query, and reads the registry back. The base report needs no root, no eBPF
//! and no special build. `--syscalls` adds the eBPF read-syscall histogram and
//! `--pmu` adds per-encoding hardware counters (both need root; `--pmu`
//! additionally needs a `profile-pmu` build for the decode marker).

mod render;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::metrics::MetricsRegistry;
use vortex::metrics::profile::MetricsSessionExt;
use vortex::metrics::profile::ScanProfiler;
use vortex::metrics::profile::diff;
use vortex::metrics::profile::diff::DiffOutcome;
use vortex::metrics::profile::diff::HARD_COUNTERS_DOWN;
use vortex::metrics::profile::diff::HARD_COUNTERS_EXACT;
use vortex::metrics::profile::diff::HARD_COUNTERS_UP;
use vortex::metrics::profile::diff::is_dynamic_hard_counter;
use vortex::metrics::profile::procfs;
use vortex::session::VortexSession;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::host::probe::Probe;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::host::probe::ProbeOptions;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::host::probe::is_root;

use crate::datafusion_helper::execute_vortex_query;

/// `vx profile` arguments.
#[derive(Debug, clap::Parser)]
pub struct ProfileArgs {
    #[clap(subcommand)]
    command: ProfileCommand,
}

#[derive(Debug, clap::Subcommand)]
enum ProfileCommand {
    /// Profile a SQL query executed against a Vortex file with DataFusion.
    Query(QueryProfileArgs),
    /// Compare two saved profile reports. Exits with code 1 if a regression is detected.
    Diff(DiffArgs),
}

/// Arguments for `vx profile query`.
#[derive(Debug, clap::Parser)]
pub struct QueryProfileArgs {
    /// Path to the Vortex file. Registered as the table `data`.
    pub file: PathBuf,

    /// SQL query to execute against table `data`.
    #[arg(long, short)]
    pub sql: String,

    /// Write the JSON report here instead of stdout.
    #[arg(long)]
    pub json: Option<PathBuf>,

    /// Attach the eBPF read-syscall tracepoints (count + size histogram). Needs root.
    #[cfg(feature = "profile-ebpf")]
    #[arg(long)]
    pub syscalls: bool,

    /// Attach the eBPF block-layer read tracepoints: device reads, bytes, and
    /// service-latency distribution past the page cache. Needs root; the block
    /// layer carries no reliable originating pid, so these counts are system-wide
    /// for the run — use an otherwise-idle host.
    #[cfg(feature = "profile-ebpf")]
    #[arg(long)]
    pub bio: bool,

    /// Attach the eBPF blocking-futex tracepoints: lock-wait count and latency
    /// distribution (parking_lot/std mutexes, the registry lock, etc.). Pid-scoped.
    /// Needs root.
    #[cfg(feature = "profile-ebpf")]
    #[arg(long)]
    pub locks: bool,

    /// Attach the eBPF TCP tracepoints: retransmits and connection setup, for
    /// remote (object-store) reads. System-wide for the run. Needs root.
    #[cfg(feature = "profile-ebpf")]
    #[arg(long)]
    pub net: bool,

    /// Attach the eBPF scheduler tracepoints: off-CPU (blocked) time and runqueue
    /// latency, attributed to the Vortex phase in flight. Needs root and a
    /// `profile-pmu` build (the context markers maintain the per-thread stack).
    #[cfg(feature = "profile-ebpf")]
    #[arg(long)]
    pub offcpu: bool,

    /// Attach the eBPF per-encoding PMU samplers (cycles, instructions, cache /
    /// branch / LLC misses, stalled cycles). Needs root and a `profile-pmu` build;
    /// hardware counters may be unavailable on virtualized hosts.
    #[cfg(feature = "profile-ebpf")]
    #[arg(long)]
    pub pmu: bool,
}

/// Arguments for `vx profile diff`.
#[derive(Debug, clap::Parser)]
pub struct DiffArgs {
    /// Baseline profile report, produced by `vx profile query --json`.
    pub baseline: PathBuf,

    /// Candidate profile report, produced by `vx profile query --json`.
    pub candidate: PathBuf,

    /// Allowed fractional change before a hard counter is considered regressed.
    #[arg(long, default_value_t = 0.05)]
    pub tolerance: f64,
}

impl ProfileArgs {
    /// The file the profile operates on, for the launcher's existence check.
    pub fn file_path(&self) -> &PathBuf {
        match &self.command {
            ProfileCommand::Query(args) => &args.file,
            ProfileCommand::Diff(args) => &args.baseline,
        }
    }
}

/// Entry point for `vx profile`.
///
/// # Errors
///
/// Returns an error if the query fails or the optional eBPF layer cannot attach.
pub async fn exec_profile(session: &VortexSession, args: ProfileArgs) -> VortexResult<()> {
    match args.command {
        ProfileCommand::Query(q) => exec_query_profile(session, q).await,
        ProfileCommand::Diff(d) => exec_diff_profile(d),
    }
}

fn exec_diff_profile(args: DiffArgs) -> VortexResult<()> {
    if args.tolerance < 0.0 {
        vortex_bail!("--tolerance must be non-negative");
    }

    let base = diff::load_report(&args.baseline).map_err(|e| vortex_err!("{e:#}"))?;
    let cand = diff::load_report(&args.candidate).map_err(|e| vortex_err!("{e:#}"))?;
    let outcome = diff::diff(&base, &cand, args.tolerance);

    print_diff(&outcome, args.tolerance);

    if outcome.regressed() {
        vortex_bail!("profile diff detected deterministic-counter regression");
    }

    Ok(())
}

async fn exec_query_profile(session: &VortexSession, args: QueryProfileArgs) -> VortexResult<()> {
    let file_path = args
        .file
        .to_str()
        .ok_or_else(|| vortex_err!("Path is not valid UTF-8"))?;
    let target = args
        .file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path);

    // Install a profiler on the session: the executor (decode/pushdown), the
    // layout scan (prune/filter/splits/rows_out) and the file IO all register
    // into this one registry.
    let registry: Arc<dyn MetricsRegistry> = Arc::new(DefaultMetricsRegistry::default());
    let profiler = Arc::new(ScanProfiler::new(Arc::clone(&registry)));
    let session = session.clone().with_scan_profiler(profiler);

    // Optional eBPF layers, attached for the duration of the query.
    #[cfg(feature = "profile-ebpf")]
    let probe = {
        let opts = ProbeOptions {
            syscalls: args.syscalls,
            bio: args.bio,
            locks: args.locks,
            net: args.net,
            offcpu: args.offcpu,
            pmu: args.pmu,
        };
        if opts.any() {
            if !is_root() {
                let cmd: Vec<String> = std::env::args().collect();
                vortex_bail!(
                    "eBPF layers need root to load (CAP_BPF+CAP_PERFMON). Re-run with:\n    sudo -E {}",
                    cmd.join(" ")
                );
            }
            Some(Probe::attach(opts).map_err(|e| vortex_err!("{e:#}"))?)
        } else {
            None
        }
    };

    let before = procfs::read_self().map_err(|e| vortex_err!("{e:#}"))?;
    let start = Instant::now();

    execute_vortex_query(&session, file_path, &args.sql)
        .await
        .map_err(|e| vortex_err!("{e}"))?;

    let wall_ms = start.elapsed().as_secs_f64() * 1.0e3;
    let after = procfs::read_self().map_err(|e| vortex_err!("{e:#}"))?;

    // The `cold.*` section depends on page-cache warmth and is not reproducible
    // across runs. If the cache was not cold, warn so the numbers aren't trusted
    // blindly and point at how to get a comparable cold run.
    let cold_bytes = after.read_bytes.saturating_sub(before.read_bytes);
    let logical_bytes = after.rchar.saturating_sub(before.rchar);
    if logical_bytes > 0 && cold_bytes < logical_bytes {
        let hit = 1.0 - (cold_bytes as f64 / logical_bytes as f64);
        eprintln!(
            "vx profile: warning: page cache was ~{:.0}% warm; cold.* metrics are not \
             reproducible. For a cold run drop caches first (Linux, root):\n    \
             sync && echo 3 | sudo tee /proc/sys/vm/drop_caches",
            hit * 100.0
        );
    }

    #[cfg(feature = "profile-ebpf")]
    let syscall_snapshot = probe
        .map(|p| p.finish().map_err(|e| vortex_err!("{e:#}")))
        .transpose()?;
    #[cfg(not(feature = "profile-ebpf"))]
    let syscall_snapshot = None;

    let report = render::render(
        target,
        "datafusion",
        &args.sql,
        wall_ms,
        &registry.snapshot(),
        &before,
        &after,
        syscall_snapshot.as_ref(),
    );
    let pretty = serde_json::to_string_pretty(&report)
        .map_err(|e| vortex_err!("serializing report: {e}"))?;

    match &args.json {
        Some(path) => {
            std::fs::write(path, &pretty).map_err(|e| vortex_err!("writing {path:?}: {e}"))?;
            eprintln!("vx profile: wrote report to {}", path.display());
        }
        None => println!("{pretty}"),
    }
    Ok(())
}

fn print_diff(outcome: &DiffOutcome, tolerance: f64) {
    println!(
        "\n  {:<44} {:>14} {:>14} {:>10} {:<9} gate",
        "metric", "baseline", "candidate", "delta", "status"
    );
    println!(
        "  {:-<44} {:->14} {:->14} {:->10} {:-<9} {:-<10}",
        "", "", "", "", "", ""
    );

    for row in &outcome.rows {
        let status = if row.regressed { "REGRESS" } else { "ok" };
        println!(
            "  {:<44} {:>14.4} {:>14.4} {:>10} {:<9} {}",
            row.name,
            row.baseline,
            row.candidate,
            format_delta(row.delta),
            status,
            gate_name(&row.name),
        );
    }

    println!();
    if outcome.rows.is_empty() {
        println!("vx profile diff: no common hard counters found");
    } else if outcome.regressed() {
        println!(
            "vx profile diff: regression detected (tolerance {:.1}%)",
            tolerance * 100.0
        );
    } else {
        println!(
            "vx profile diff: no deterministic-counter regressions (tolerance {:.1}%)",
            tolerance * 100.0
        );
    }
}

fn format_delta(delta: f64) -> String {
    if delta.is_finite() {
        format!("{:+.1}%", delta * 100.0)
    } else if delta.is_sign_positive() {
        "+inf".to_string()
    } else {
        "-inf".to_string()
    }
}

fn gate_name(name: &str) -> &'static str {
    if HARD_COUNTERS_UP.contains(&name) || is_dynamic_hard_counter(name) {
        "increase"
    } else if HARD_COUNTERS_DOWN.contains(&name) {
        "decrease"
    } else if HARD_COUNTERS_EXACT.contains(&name) {
        "exact"
    } else {
        "unknown"
    }
}
