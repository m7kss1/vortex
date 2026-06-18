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

/// Shared profiling context published on a [`VortexSession`] while profiling.
/// Owns the one registry every subsystem records into (scan / decode / IO today,
/// engine adapters later) plus the [`ScanProfiler`] derived from it.
pub struct ProfileContext {
    registry: Arc<dyn MetricsRegistry>,
    scan: Arc<ScanProfiler>,
}

impl ProfileContext {
    /// Build a context whose scan profiler shares `registry`. Upholds the
    /// invariant that [`registry`](Self::registry) and
    /// [`scan`](Self::scan)`.registry()` are the same registry, so every
    /// subsystem unifies into one snapshot.
    pub fn new(registry: Arc<dyn MetricsRegistry>) -> Self {
        let scan = Arc::new(ScanProfiler::new(Arc::clone(&registry)));
        Self { registry, scan }
    }

    /// The shared registry. Equal to [`scan`](Self::scan)`.registry()`.
    pub fn registry(&self) -> Arc<dyn MetricsRegistry> {
        Arc::clone(&self.registry)
    }

    /// The scan profiler.
    pub fn scan(&self) -> Arc<ScanProfiler> {
        Arc::clone(&self.scan)
    }
}

impl std::fmt::Debug for ProfileContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileContext").finish_non_exhaustive()
    }
}

/// Session var carrying the active [`ProfileContext`].
struct ProfileContextVar(Arc<ProfileContext>);

impl std::fmt::Debug for ProfileContextVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileContextVar").finish_non_exhaustive()
    }
}

impl SessionVar for ProfileContextVar {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Publishes / reads the [`ProfileContext`] on a [`VortexSession`]. A profiler
/// installs one before running a query; executors and scan / IO layers read the
/// [`ScanProfiler`] back to register metrics into the same registry.
pub trait MetricsSessionExt {
    /// Install `ctx` so the scan emits format metrics.
    fn with_profile_context(self, ctx: Arc<ProfileContext>) -> Self;

    /// The active profile context, if one was installed.
    fn profile_context(&self) -> Option<Arc<ProfileContext>>;

    /// The active scan profiler, if any. Resolved from the profile context, so
    /// existing scan / decode / IO hooks read it unchanged.
    fn scan_profiler(&self) -> Option<Arc<ScanProfiler>>;
}

impl MetricsSessionExt for VortexSession {
    fn with_profile_context(self, ctx: Arc<ProfileContext>) -> Self {
        self.with_some(ProfileContextVar(ctx))
    }

    fn profile_context(&self) -> Option<Arc<ProfileContext>> {
        self.get_opt::<ProfileContextVar>()
            .map(|v| Arc::clone(&v.0))
    }

    fn scan_profiler(&self) -> Option<Arc<ScanProfiler>> {
        self.profile_context().map(|c| c.scan())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_session::VortexSession;

    use super::MetricsSessionExt;
    use super::ProfileContext;
    use crate::DefaultMetricsRegistry;
    use crate::MetricsRegistry;

    fn context() -> Arc<ProfileContext> {
        let registry: Arc<dyn MetricsRegistry> = Arc::new(DefaultMetricsRegistry::default());
        Arc::new(ProfileContext::new(registry))
    }

    #[test]
    fn context_round_trips_through_session() {
        let ctx = context();
        let session = VortexSession::empty().with_profile_context(Arc::clone(&ctx));
        let read = session.profile_context().expect("context installed");
        assert!(Arc::ptr_eq(&read, &ctx));
    }

    #[test]
    fn scan_profiler_resolves_from_context() {
        let ctx = context();
        let session = VortexSession::empty().with_profile_context(Arc::clone(&ctx));
        let scan = session.scan_profiler().expect("scan profiler installed");
        assert!(Arc::ptr_eq(&scan, &ctx.scan()));
    }

    #[test]
    fn context_registry_is_scan_registry() {
        let ctx = context();
        assert!(Arc::ptr_eq(&ctx.registry(), &ctx.scan().registry()));
    }

    #[test]
    fn scan_profiler_absent_without_context() {
        assert!(VortexSession::empty().scan_profiler().is_none());
    }
}
