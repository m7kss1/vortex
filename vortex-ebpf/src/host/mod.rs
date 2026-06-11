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
    use crate::types::DecodeStack;
    use crate::types::EncKey;
    use crate::types::PmuStats;

    // SAFETY: these are plain `#[repr(C)]` aggregates of explicitly sized
    // integers/byte arrays with no invariants, safe to read from BPF maps as raw
    // bytes.
    unsafe impl aya::Pod for EncKey {}
    unsafe impl aya::Pod for DecodeStack {}
    unsafe impl aya::Pod for PmuStats {}
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
