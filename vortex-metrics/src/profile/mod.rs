// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-process profiling support for `vx profile`.
//!
//! This module owns the semantic profiling context and deterministic report diff
//! logic. Kernel-side facts live in `vortex-ebpf`; core crates only emit small
//! metric updates through this API.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;

use crate::Counter;
use crate::MetricBuilder;
use crate::MetricsRegistry;

pub mod diff;
pub mod procfs;

/// Approximate a percentile from a log2 histogram (returns the bucket lower bound).
pub fn hist_percentile(hist: &[u64], q: f64) -> f64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let target = (q * total as f64).ceil();
    let mut cum = 0u64;
    for (bucket, &count) in hist.iter().enumerate() {
        cum += count;
        if cum as f64 >= target {
            return (1u64 << bucket) as f64;
        }
    }
    0.0
}

/// Per-encoding decode counters. `Counter` is `Arc`-backed, so cloning a handle
/// out of the cache is cheap and the adds are lock-free.
#[derive(Clone)]
struct DecodeHandles {
    calls: Counter,
    nanos: Counter,
    rows: Counter,
    bytes: Counter,
}

/// Shared profiling context for a scan: per-encoding decode economics and
/// pushdown-miss counts, all written into one [`MetricsRegistry`]. Published on
/// the session by a profiler (e.g. `vx profile`) and absent on normal scans.
pub struct ScanProfiler {
    registry: Arc<dyn MetricsRegistry>,
    decode: DashMap<String, DecodeHandles>,
    fallback: DashMap<(String, String), Counter>,
}

impl ScanProfiler {
    /// Create a profiler writing into `registry`.
    pub fn new(registry: Arc<dyn MetricsRegistry>) -> Self {
        Self {
            registry,
            decode: DashMap::default(),
            fallback: DashMap::default(),
        }
    }

    /// The underlying registry, so other crates can register their own metrics
    /// into the same place.
    pub fn registry(&self) -> Arc<dyn MetricsRegistry> {
        Arc::clone(&self.registry)
    }

    /// Record one decode of `encoding`: its wall time and the rows / own-buffer
    /// bytes produced.
    pub fn record_decode(&self, encoding: &str, elapsed: Duration, rows: u64, bytes: u64) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        // Fast path: a decode of an already-seen encoding (the common case) reads
        // the handles without allocating an owned key or holding the shard lock
        // across the atomic adds.
        let handles = match self.decode.get(encoding) {
            Some(handles) => handles.clone(),
            None => self
                .decode
                .entry(encoding.to_string())
                .or_insert_with(|| {
                    let make = |name: &'static str| {
                        MetricBuilder::new(self.registry.as_ref())
                            .add_label("encoding", encoding.to_string())
                            .counter(name)
                    };
                    DecodeHandles {
                        calls: make("vortex.decode.calls"),
                        nanos: make("vortex.decode.nanos"),
                        rows: make("vortex.decode.rows"),
                        bytes: make("vortex.decode.bytes"),
                    }
                })
                .clone(),
        };
        handles.calls.add(1);
        handles.nanos.add(nanos);
        handles.rows.add(rows);
        handles.bytes.add(bytes);
    }

    /// Record that `parent` forced `child` to canonicalize (a compute-on-encoded
    /// pushdown miss).
    pub fn record_fallback(&self, parent: &str, child: &str) {
        let key = (parent.to_string(), child.to_string());
        let counter = self.fallback.entry(key).or_insert_with(|| {
            MetricBuilder::new(self.registry.as_ref())
                .add_label("parent", parent.to_string())
                .add_label("child", child.to_string())
                .counter("vortex.canonicalize_fallback")
        });
        counter.add(1);
    }
}

impl std::fmt::Debug for ScanProfiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanProfiler").finish_non_exhaustive()
    }
}

/// Session var carrying the active [`ScanProfiler`].
struct ScanProfilerVar(Arc<ScanProfiler>);

impl std::fmt::Debug for ScanProfilerVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanProfilerVar").finish_non_exhaustive()
    }
}

impl SessionVar for ScanProfilerVar {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Publishes / reads the [`ScanProfiler`] on a [`VortexSession`]. A profiler
/// installs one before running a query; executors and scan/IO layers read it
/// back to register metrics into the same registry.
pub trait MetricsSessionExt {
    /// Install `profiler` so the scan emits format metrics.
    fn with_scan_profiler(self, profiler: Arc<ScanProfiler>) -> Self;

    /// The active profiler, if one was installed.
    fn scan_profiler(&self) -> Option<Arc<ScanProfiler>>;
}

impl MetricsSessionExt for VortexSession {
    fn with_scan_profiler(self, profiler: Arc<ScanProfiler>) -> Self {
        self.with_some(ScanProfilerVar(profiler))
    }

    fn scan_profiler(&self) -> Option<Arc<ScanProfiler>> {
        self.get_opt::<ScanProfilerVar>().map(|v| Arc::clone(&v.0))
    }
}
