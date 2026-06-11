// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Map types shared between the Vortex eBPF kernel programs
//! (`vortex-ebpf/ebpf`) and this userspace collector.
//!
//! This file is the single source of truth for the BPF map layouts. The kernel
//! crate cannot take a cargo dependency on `vortex-ebpf` (this crate's
//! build script *builds* the kernel crate, which would be circular), so it
//! includes this exact file with `#[path]`. Everything here must therefore stay
//! `no_std`-clean: `#[repr(C)]` aggregates of explicitly-sized integers, no
//! `aya` references. The `aya::Pod` impls live in `crate::pod`, host-side only.
//!
//! The eBPF layer covers only the facts the kernel can see that userspace can't:
//! real read-syscall sizes (`READ_*`) and per-encoding hardware counters
//! (`PMU_STATS`, attributed via the `DECODE_STACK`). All format-semantic metrics
//! are collected in-process via `vortex-metrics`.

/// Index of the read-syscall count in the `READ_STATS` array map.
pub const READ_STAT_COUNT: u32 = 0;
/// Index of the read-syscall byte total in the `READ_STATS` array map.
pub const READ_STAT_BYTES: u32 = 1;
/// Number of entries in `READ_STATS`.
pub const READ_STATS_LEN: u32 = 2;

/// Number of log2 buckets in the `READ_HIST` size histogram (bucket `b` =
/// read sizes in `[2^b, 2^(b+1))`).
pub const READ_HIST_BUCKETS: u32 = 64;

/// Maximum encoding-id string length captured in-kernel for per-encoding PMU
/// attribution. Encoding ids (`vortex.alp`, `fastlanes.bitpacked`) are well
/// under this.
pub const ENC_NAME_LEN: usize = 32;

/// Maximum decode nesting depth tracked per thread. Vortex decodes recurse
/// (an operation decodes its encoded child, which decodes *its* child …), so the
/// kernel keeps a per-thread stack and attributes PMU samples to its top.
/// Bounded by the 512-byte BPF stack (`depth` + `enc_len` + `encs` must fit when
/// a zeroed entry is inserted): 8 × 32 B keeps it at 292 B with margin. Deeper
/// nesting clamps (samples attribute to the deepest tracked frame).
pub const DECODE_STACK_DEPTH: usize = 8;

/// Key of the `PMU_STATS` map: the fixed-width encoding-id bytes read from user
/// memory at the decode marker. Stable per encoding (same interned pointer), so
/// it groups samples correctly even though bytes past the name are arena content.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncKey(pub [u8; ENC_NAME_LEN]);

/// Per-thread decode stack: `decode_begin(enc)` pushes, `decode_end()` pops, and
/// a PMU sample attributes to `encs[depth - 1]`. Stored in `DECODE_STACK` keyed
/// by thread id.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DecodeStack {
    /// Number of frames currently on the stack.
    pub depth: u32,
    /// Real length of each frame's encoding-id string (for keying display).
    pub enc_len: [u32; DECODE_STACK_DEPTH],
    /// Each frame's fixed-width encoding-id bytes.
    pub encs: [[u8; ENC_NAME_LEN]; DECODE_STACK_DEPTH],
}

/// Per-encoding PMU sample counts, keyed by [`EncKey`] in `PMU_STATS`. Each field
/// counts perf-event samples attributed to an in-flight decode of that encoding;
/// the collector multiplies by the configured sample period.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PmuStats {
    /// CPU-cycle samples.
    pub cycles: u64,
    /// Retired-instruction samples.
    pub instructions: u64,
    /// Cache-miss samples.
    pub cache_misses: u64,
    /// Real length of this encoding's id string (for display).
    pub enc_len: u32,
    /// Padding to a `u64` boundary.
    pub _pad: u32,
}
