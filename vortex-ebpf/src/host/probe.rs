// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! High-level eBPF probe handles for `vx profile`, keeping `aya`/`libc` out of
//! the caller. Each probe loads the embedded object, attaches to the current
//! process, and is read once via `finish`.

use anyhow::Result;
use aya::Ebpf;

use crate::host::attach::attach_block_tracepoints;
use crate::host::attach::attach_context_uprobes;
use crate::host::attach::attach_lock_tracepoints;
use crate::host::attach::attach_net_tracepoints;
use crate::host::attach::attach_pmu_events;
use crate::host::attach::attach_read_tracepoints;
use crate::host::attach::attach_sched_tracepoints;
use crate::host::collect::Snapshot;
use crate::host::collect::snapshot;
use crate::host::loader;

/// Whether the current process is root (uid 0). Loading eBPF needs root
/// (`CAP_BPF`+`CAP_PERFMON`).
pub fn is_root() -> bool {
    // SAFETY: getuid is always safe and never fails.
    unsafe { libc::getuid() == 0 }
}

/// Which eBPF layers to attach. Each maps to one `vx profile` flag and is
/// independent; compose them freely (e.g. `--syscalls --bio`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbeOptions {
    /// Read-syscall tracepoints (`--syscalls`): count + size + latency, pid-scoped.
    pub syscalls: bool,
    /// Block-layer read tracepoints (`--bio`): device reads, bytes, latency.
    pub bio: bool,
    /// Blocking-futex tracepoints (`--locks`): lock-wait count + latency, pid-scoped.
    pub locks: bool,
    /// TCP tracepoints (`--net`): retransmits + connection setup. System-wide.
    pub net: bool,
    /// Scheduler tracepoints (`--offcpu`): off-CPU block time + runqueue latency,
    /// attributed to the in-flight context. Needs a `profile-pmu` workload so the
    /// context marker symbols (and thus the per-thread stack) exist.
    pub offcpu: bool,
    /// Context uprobes + per-context PMU sampling (`--pmu`). Needs a `profile-pmu`
    /// workload so the context marker symbols exist.
    pub pmu: bool,
}

impl ProbeOptions {
    /// Whether any layer is requested.
    pub fn any(&self) -> bool {
        self.syscalls || self.bio || self.locks || self.net || self.offcpu || self.pmu
    }
}

/// An attached set of eBPF probes scoped to this process; read it with
/// [`finish`](Self::finish).
pub struct Probe {
    ebpf: Ebpf,
    opts: ProbeOptions,
}

impl Probe {
    /// Load the embedded object and attach the requested layers to this process.
    pub fn attach(opts: ProbeOptions) -> Result<Self> {
        let mut ebpf = loader::load()?;
        let pid = std::process::id();
        if opts.syscalls {
            attach_read_tracepoints(&mut ebpf, pid)?;
        }
        if opts.locks {
            attach_lock_tracepoints(&mut ebpf, pid)?;
        }
        // Both PMU and off-CPU attribute to the per-thread context stack, so both
        // need the context-marker uprobes maintaining it.
        if opts.pmu || opts.offcpu {
            attach_context_uprobes(&mut ebpf, "/proc/self/exe", pid as libc::pid_t)?;
        }
        if opts.pmu {
            attach_pmu_events(&mut ebpf, pid)?;
        }
        if opts.offcpu {
            attach_sched_tracepoints(&mut ebpf)?;
        }
        if opts.bio {
            attach_block_tracepoints(&mut ebpf)?;
        }
        if opts.net {
            attach_net_tracepoints(&mut ebpf)?;
        }
        Ok(Self { ebpf, opts })
    }

    /// Read the kernel maps for the attached layers.
    pub fn finish(self) -> Result<Snapshot> {
        snapshot(&self.ebpf, &self.opts)
    }
}
