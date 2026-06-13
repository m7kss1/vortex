// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host-side eBPF loader, probes, and snapshots.
//!
//! This module is available only with the `host` feature so core crates can use
//! the marker ABI without compiling `aya`, `libc`, or the BPF build script.

pub mod attach;
pub mod collect;
pub mod loader;
pub mod probe;

/// Shared BPF map layouts.
pub mod types {
    pub use crate::types::*;
}

/// The compiled eBPF object for the Vortex eBPF kernel programs.
///
/// Empty when the crate was built with `VORTEX_EBPF_SKIP_BPF=1`.
pub fn ebpf_object() -> &'static [u8] {
    aya::include_bytes_aligned!(concat!(env!("OUT_DIR"), "/vortex-ebpf-kern"))
}

mod pod {
    use crate::types::BioKey;
    use crate::types::ContextKey;
    use crate::types::ContextStack;
    use crate::types::OffCpuStart;
    use crate::types::OffCpuStats;
    use crate::types::PmuStats;

    // SAFETY: these are plain `#[repr(C)]` aggregates of explicitly sized
    // integers (padding fields are explicit and always zeroed), safe to read from
    // BPF maps as raw bytes.
    unsafe impl aya::Pod for ContextKey {}
    unsafe impl aya::Pod for ContextStack {}
    unsafe impl aya::Pod for PmuStats {}
    unsafe impl aya::Pod for BioKey {}
    unsafe impl aya::Pod for OffCpuStats {}
    unsafe impl aya::Pod for OffCpuStart {}
}

#[cfg(test)]
mod tests {
    use super::ebpf_object;

    #[test]
    fn embedded_object_is_elf() {
        let obj = ebpf_object();
        // Empty means VORTEX_EBPF_SKIP_BPF was set for this build.
        if !obj.is_empty() {
            assert_eq!(&obj[..4], b"\x7fELF");
        }
    }
}
