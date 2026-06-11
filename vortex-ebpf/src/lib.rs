// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Vortex eBPF — PMU decode-context marker plus optional host collector.
//!
//! In its default configuration this crate exposes only two stable marker
//! symbols and a [`decode_scope`] RAII guard. That marker ABI lets the optional
//! eBPF PMU sampler attribute hardware-counter samples (cache misses,
//! instructions) to the **encoding** being decoded.
//!
//! The host-side loader and BPF map readers are compiled only
//! with the `host` feature. Everything else — decode economics
//! (calls/rows/bytes/time), pruning, filter, IO, splits — is collected
//! in-process through `vortex-metrics` and does not go through this crate.

#[cfg(feature = "host")]
pub mod host;
#[cfg(feature = "host")]
pub mod types;

/// The `vortex_ebpf_*` uprobe marker symbols.
///
/// An internal detail of [`decode_scope`]; instrument through that, not by
/// calling these. Each is `#[no_mangle] extern "C"` so the collector can attach
/// by a stable name and read arguments by calling convention, and
/// `#[inline(never)]` so the call site survives optimization.
#[doc(hidden)]
pub mod markers {
    use std::hint::black_box;

    /// A decode began on this thread. `arg`: `enc_ptr, enc_len` (the encoding-id
    /// string, read in-kernel via `bpf_probe_read_user`).
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub extern "C" fn vortex_ebpf_decode_begin(enc_ptr: *const u8, enc_len: u64) {
        black_box(enc_ptr);
        black_box(enc_len);
    }

    /// The decode finished on this thread.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub extern "C" fn vortex_ebpf_decode_end() {
        black_box(());
    }
}

/// RAII guard bracketing a decode for PMU attribution: the begin marker fires on
/// creation, the end marker on drop. Create it with [`decode_scope`].
#[must_use = "the decode scope ends immediately if the guard is dropped"]
pub struct DecodeScope(());

impl Drop for DecodeScope {
    fn drop(&mut self) {
        markers::vortex_ebpf_decode_end();
    }
}

/// Mark `encoding` as the encoding being decoded on this thread until the
/// returned guard is dropped, so the PMU sampler attributes samples to it.
pub fn decode_scope(encoding: &str) -> DecodeScope {
    markers::vortex_ebpf_decode_begin(encoding.as_ptr(), encoding.len() as u64);
    DecodeScope(())
}

#[cfg(test)]
mod tests {
    use super::decode_scope;

    #[test]
    fn scope_calls_markers() {
        // Smoke test: the symbols exist and the guard brackets without panicking.
        let scope = decode_scope("vortex.alp");
        drop(scope);
    }
}
