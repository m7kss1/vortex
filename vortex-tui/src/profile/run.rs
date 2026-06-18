// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Vortex-side profile run lifecycle: install a [`ProfileContext`] on the
//! session, bracket the engine execution with procfs and (optionally) eBPF, and
//! collect everything into a [`ProfileRunOutput`] for rendering. Engine-specific
//! introspection lives in [`super::engine`]; this module knows nothing about
//! which engine runs in between.

use std::sync::Arc;
use std::time::Instant;

use vortex::error::VortexResult;
#[cfg(feature = "profile-ebpf")]
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::metrics::Metric;
use vortex::metrics::profile::MetricsSessionExt;
use vortex::metrics::profile::ProfileContext;
use vortex::metrics::profile::procfs;
use vortex::metrics::profile::procfs::ProcReading;
use vortex::session::VortexSession;
/// The optional eBPF snapshot produced by a run. A unit type when built without
/// `profile-ebpf`, so the rest of the pipeline carries `Option<Snapshot>` in
/// every feature combination.
#[cfg(feature = "profile-ebpf")]
pub(super) use vortex_ebpf::host::collect::Snapshot;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::host::probe::Probe;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::host::probe::ProbeOptions;
#[cfg(feature = "profile-ebpf")]
use vortex_ebpf::host::probe::is_root;
#[cfg(not(feature = "profile-ebpf"))]
pub(super) type Snapshot = ();

/// The Vortex profile context plus the procfs / eBPF lifecycle around one engine
/// run.
pub(super) struct ProfileRun {
    context: Arc<ProfileContext>,
    before_procfs: ProcReading,
    start: Instant,
    #[cfg(feature = "profile-ebpf")]
    probe: Option<Probe>,
}

/// Everything a profile run collected, ready to render.
pub(super) struct ProfileRunOutput {
    pub(super) wall_ms: f64,
    pub(super) metrics: Vec<Metric>,
    pub(super) before_procfs: ProcReading,
    pub(super) after_procfs: ProcReading,
    pub(super) ebpf_snapshot: Option<Snapshot>,
}

impl ProfileRun {
    /// Install a fresh profile context on `session`, attach the requested eBPF
    /// probe (root-checked), read the procfs baseline and start the wall clock.
    /// Returns the profiled session the engine must run against.
    pub(super) fn begin(
        session: &VortexSession,
        #[cfg(feature = "profile-ebpf")] probe_opts: ProbeOptions,
    ) -> VortexResult<(VortexSession, Self)> {
        // One registry, shared by the executor (decode / pushdown), the layout
        // scan (prune / filter / splits / rows_out) and the file IO.
        let context = Arc::new(ProfileContext::new(Arc::new(
            DefaultMetricsRegistry::default(),
        )));
        let session = session.clone().with_profile_context(Arc::clone(&context));

        #[cfg(feature = "profile-ebpf")]
        let probe = attach_probe(probe_opts)?;

        let before_procfs = procfs::read_self().map_err(|e| vortex_err!("{e:#}"))?;
        let start = Instant::now();

        Ok((
            session,
            Self {
                context,
                before_procfs,
                start,
                #[cfg(feature = "profile-ebpf")]
                probe,
            },
        ))
    }

    /// Stop the wall clock, read procfs again, snapshot the registry and finish
    /// the eBPF probe.
    pub(super) fn finish(self) -> VortexResult<ProfileRunOutput> {
        let wall_ms = self.start.elapsed().as_secs_f64() * 1.0e3;
        let after_procfs = procfs::read_self().map_err(|e| vortex_err!("{e:#}"))?;
        let metrics = self.context.registry().snapshot();

        #[cfg(feature = "profile-ebpf")]
        let ebpf_snapshot = self
            .probe
            .map(|p| p.finish().map_err(|e| vortex_err!("{e:#}")))
            .transpose()?;
        #[cfg(not(feature = "profile-ebpf"))]
        let ebpf_snapshot = None;

        Ok(ProfileRunOutput {
            wall_ms,
            metrics,
            before_procfs: self.before_procfs,
            after_procfs,
            ebpf_snapshot,
        })
    }
}

/// Attach the eBPF probe for the requested layers, or `None` if no layer was
/// requested. Bails if a layer needs root and we are not root.
#[cfg(feature = "profile-ebpf")]
fn attach_probe(opts: ProbeOptions) -> VortexResult<Option<Probe>> {
    if !opts.any() {
        return Ok(None);
    }
    if !is_root() {
        let cmd: Vec<String> = std::env::args().collect();
        vortex_bail!(
            "eBPF layers need root to load (CAP_BPF+CAP_PERFMON). Re-run with:\n    sudo -E {}",
            cmd.join(" ")
        );
    }
    Ok(Some(Probe::attach(opts).map_err(|e| vortex_err!("{e:#}"))?))
}
