// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-process scan metrics, emitted through `vortex-metrics`.
//!
//! Pruning / filter selectivity (per conjunct), split timing and peak
//! concurrency, and the scan's output rows. Built once per scan from the
//! [`ScanProfiler`](vortex_metrics::profile::ScanProfiler)'s registry and shared
//! across split tasks via `Arc`; handles are registered lazily per conjunct and
//! reused (registering per call would lock the registry and bloat its snapshot).

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use vortex_metrics::Counter;
use vortex_metrics::Gauge;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;
use vortex_metrics::Timer;
use vortex_utils::aliases::dash_map::DashMap;

/// Per-conjunct rows-in / rows-kept / evaluation-time handles.
#[derive(Clone)]
struct ConjHandles {
    rows_in: Counter,
    rows_kept: Counter,
    time: Timer,
}

/// Format metrics for the layout scan path.
pub(crate) struct ScanMetrics {
    registry: Arc<dyn MetricsRegistry>,
    rows_out: Counter,
    split_time: Timer,
    /// In-flight split count, used to drive the peak-concurrency high-water mark.
    split_active: AtomicU64,
    split_peak: Gauge,
    prune: DashMap<u64, ConjHandles>,
    filter: DashMap<u64, ConjHandles>,
}

impl ScanMetrics {
    /// Register the per-scan handles into `registry`.
    pub(crate) fn new(registry: Arc<dyn MetricsRegistry>) -> Self {
        let rows_out = MetricBuilder::new(registry.as_ref()).counter("vortex.scan.rows_out");
        let split_time = MetricBuilder::new(registry.as_ref()).timer("vortex.scan.split_time");
        let split_peak = MetricBuilder::new(registry.as_ref()).gauge("vortex.scan.split_peak");
        Self {
            registry,
            rows_out,
            split_time,
            split_active: AtomicU64::new(0),
            split_peak,
            prune: DashMap::default(),
            filter: DashMap::default(),
        }
    }

    fn conj(&self, map: &DashMap<u64, ConjHandles>, kind: &str, conjunct: u64) -> ConjHandles {
        if let Some(h) = map.get(&conjunct) {
            return h.clone();
        }
        let label = conjunct.to_string();
        let builder =
            || MetricBuilder::new(self.registry.as_ref()).add_label("conjunct", label.clone());
        let handles = ConjHandles {
            rows_in: builder().counter(format!("vortex.{kind}.rows_in")),
            rows_kept: builder().counter(format!("vortex.{kind}.rows_kept")),
            time: builder().timer(format!("vortex.{kind}.time")),
        };
        map.insert(conjunct, handles.clone());
        handles
    }

    /// Record a stats-pruning evaluation of one conjunct.
    pub(crate) fn record_prune(
        &self,
        conjunct: u64,
        rows_in: u64,
        rows_kept: u64,
        elapsed: Duration,
    ) {
        let h = self.conj(&self.prune, "prune", conjunct);
        h.rows_in.add(rows_in);
        h.rows_kept.add(rows_kept);
        h.time.update(elapsed);
    }

    /// Record a post-decode filter evaluation of one conjunct.
    pub(crate) fn record_filter(
        &self,
        conjunct: u64,
        rows_in: u64,
        rows_kept: u64,
        elapsed: Duration,
    ) {
        let h = self.conj(&self.filter, "filter", conjunct);
        h.rows_in.add(rows_in);
        h.rows_kept.add(rows_kept);
        h.time.update(elapsed);
    }

    /// A split began executing; bumps the peak-concurrency high-water mark.
    pub(crate) fn split_begin(&self) {
        let active = self.split_active.fetch_add(1, Ordering::Relaxed) + 1;
        if (active as f64) > self.split_peak.value() {
            self.split_peak.set(active as f64);
        }
    }

    /// A split finished: record its duration and the rows it emitted (the scan's
    /// output rows are the sum across splits).
    pub(crate) fn split_end(&self, elapsed: Duration, rows_out: u64) {
        self.split_active.fetch_sub(1, Ordering::Relaxed);
        self.split_time.update(elapsed);
        self.rows_out.add(rows_out);
    }
}
