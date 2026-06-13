// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Vortex eBPF — one universal context marker plus optional host collector.
//!
//! In its default configuration this crate exposes a single pair of stable marker
//! symbols ([`markers::vortex_ebpf_ctx_push`] / [`markers::vortex_ebpf_ctx_pop`])
//! and the [`context_scope`] RAII guard. That one marker ABI lets the optional
//! eBPF samplers attribute kernel-side facts (hardware counters, and in future
//! off-CPU / lock / allocation time) to whatever Vortex work is in flight — a
//! decode of an encoding, a filter conjunct, a scan split — identified by a
//! [`ContextKind`] and an opaque `u64` instance id.
//!
//! A new phase needs only a new [`ContextKind`] variant and a `context_scope`
//! call at its boundary: no new symbols, uprobes, or kernel programs.
//!
//! The host-side loader and BPF map readers are compiled only with the `host`
//! feature. Everything else — decode economics (calls/rows/bytes/time), pruning,
//! filter, IO, splits — is collected in-process through `vortex-metrics` and does
//! not go through this crate.

#[cfg(feature = "host")]
pub mod host;
#[cfg(feature = "host")]
pub mod types;

/// What kind of work a context frame describes. The `u32` values are an ABI
/// shared with the kernel (`types::CTX_KIND_*`) and the host renderer; keep all
/// three in sync. A sampler attributes to the top of the per-thread stack.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContextKind {
    /// A scan split (row-range task).
    Scan = 0,
    /// Statistics pruning of one predicate conjunct.
    Prune = 1,
    /// Post-decode filter evaluation of one predicate conjunct.
    Filter = 2,
    /// Decoding one encoding.
    Decode = 3,
    /// Awaiting a segment read.
    IoWait = 4,
    /// Compute-on-encoded canonicalization fallback.
    Fallback = 5,
}

impl ContextKind {
    /// Map a raw `u32` (as stored in a [`ContextKey`](types::ContextKey)) back to
    /// a [`ContextKind`], or `None` if it is not a known discriminant.
    #[must_use]
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Scan),
            1 => Some(Self::Prune),
            2 => Some(Self::Filter),
            3 => Some(Self::Decode),
            4 => Some(Self::IoWait),
            5 => Some(Self::Fallback),
            _ => None,
        }
    }
}

/// The universal `vortex_ebpf_ctx_*` uprobe marker symbols.
///
/// An internal detail of [`context_scope`]; instrument through that, not by
/// calling these. Each is `#[no_mangle] extern "C"` so the collector can attach
/// by a stable name and read arguments by calling convention, and
/// `#[inline(never)]` so the call site survives optimization. Both arguments
/// arrive by register, so the kernel reads no user memory.
#[doc(hidden)]
pub mod markers {
    use std::hint::black_box;

    /// Work began on this thread: push `(kind, id)` onto the per-thread context
    /// stack.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub extern "C" fn vortex_ebpf_ctx_push(kind: u32, id: u64) {
        black_box(kind);
        black_box(id);
    }

    /// The most recently pushed work finished on this thread: pop the stack.
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub extern "C" fn vortex_ebpf_ctx_pop() {
        black_box(());
    }
}

/// RAII guard bracketing one unit of work for sample attribution: the push marker
/// fires on creation, the pop marker on drop. Create it with [`context_scope`].
#[must_use = "the context scope ends immediately if the guard is dropped"]
pub struct ContextScope(());

impl Drop for ContextScope {
    fn drop(&mut self) {
        markers::vortex_ebpf_ctx_pop();
    }
}

/// Mark `(kind, id)` as the work in flight on this thread until the returned
/// guard is dropped, so the eBPF samplers attribute to it. `id` is an opaque
/// instance key the host resolves per `kind` (e.g. an encoding's interned
/// symbol, or a conjunct index).
pub fn context_scope(kind: ContextKind, id: u64) -> ContextScope {
    markers::vortex_ebpf_ctx_push(kind as u32, id);
    ContextScope(())
}

/// Convenience wrapper for [`ContextKind::Decode`]: `id` is the encoding's
/// interned symbol (`vortex_session::registry::Id::as_u64`).
pub fn decode_scope(id: u64) -> ContextScope {
    context_scope(ContextKind::Decode, id)
}

#[cfg(test)]
mod tests {
    use super::ContextKind;
    use super::context_scope;
    use super::decode_scope;

    #[test]
    fn scope_calls_markers() {
        // Smoke test: the symbols exist and the guards bracket without panicking.
        let outer = context_scope(ContextKind::Scan, 7);
        let inner = decode_scope(42);
        drop(inner);
        drop(outer);
    }
}
