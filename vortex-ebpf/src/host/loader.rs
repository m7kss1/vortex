// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loading the embedded eBPF object into the kernel.

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use aya::Ebpf;

use crate::host::ebpf_object;

/// Best-effort bump of `RLIMIT_MEMLOCK` for older kernels that account BPF
/// memory against it (no-op on memcg-accounted kernels ≥ 5.11).
fn bump_memlock_rlimit() {
    let rlim = libc::rlimit {
        rlim_cur: libc::RLIM_INFINITY,
        rlim_max: libc::RLIM_INFINITY,
    };
    // SAFETY: `setrlimit` with a valid rlimit pointer; failure is ignored.
    unsafe {
        libc::setrlimit(libc::RLIMIT_MEMLOCK, &raw const rlim);
    }
}

/// Load the embedded kernel programs. Requires `CAP_BPF`+`CAP_PERFMON` (or
/// root) and a build without `VORTEX_EBPF_SKIP_BPF`.
pub fn load() -> Result<Ebpf> {
    let object = ebpf_object();
    if object.is_empty() {
        bail!(
            "vortex-ebpf was built with VORTEX_EBPF_SKIP_BPF=1; \
             rebuild without it to enable profiling"
        );
    }
    bump_memlock_rlimit();
    Ebpf::load(object).context("loading eBPF object (root or CAP_BPF required)")
}
