// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! High-level eBPF probe handles for `vx profile`, keeping `aya`/`libc` out of
//! the caller. Each probe loads the embedded object, attaches to the current
//! process, and is read once via `finish`.

use anyhow::Result;
use aya::Ebpf;

use crate::host::attach::attach_decode_uprobes;
use crate::host::attach::attach_read_tracepoints;
use crate::host::collect::Snapshot;
use crate::host::collect::snapshot;
use crate::host::loader;

/// Whether the current process is root (uid 0). Loading eBPF needs root
/// (`CAP_BPF`+`CAP_PERFMON`).
pub fn is_root() -> bool {
    // SAFETY: getuid is always safe and never fails.
    unsafe { libc::getuid() == 0 }
}

/// An attached set of eBPF probes scoped to this process; read it with
/// [`finish`](Self::finish).
pub struct Probe {
    ebpf: Ebpf,
}

impl Probe {
    /// Load the object and attach the read-syscall tracepoints to this process.
    pub fn attach_syscalls() -> Result<Self> {
        let mut ebpf = loader::load()?;
        attach_read_tracepoints(&mut ebpf, std::process::id())?;
        Ok(Self { ebpf })
    }

    /// Load the object and attach the read tracepoints **and** the decode
    /// uprobes (for per-encoding PMU) to this process. Requires the workload to
    /// be built with `profile-pmu` so the decode marker symbols exist.
    pub fn attach_syscalls_and_pmu() -> Result<Self> {
        let mut ebpf = loader::load()?;
        let pid = std::process::id();
        attach_read_tracepoints(&mut ebpf, pid)?;
        attach_decode_uprobes(&mut ebpf, "/proc/self/exe", pid as libc::pid_t)?;
        Ok(Self { ebpf })
    }

    /// Read the kernel maps.
    pub fn finish(self) -> Result<Snapshot> {
        snapshot(&self.ebpf)
    }
}
