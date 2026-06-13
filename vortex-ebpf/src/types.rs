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
//! real read-syscall sizes (`READ_*`) and hardware counters (`PMU_STATS`)
//! attributed to the work in flight via a per-thread `CONTEXT_STACK` of
//! [`ContextKey`]s. All format-semantic metrics are collected in-process via
//! `vortex-metrics`.

/// Index of the read-syscall count in the `READ_STATS` array map.
pub const READ_STAT_COUNT: u32 = 0;
/// Index of the read-syscall byte total in the `READ_STATS` array map.
pub const READ_STAT_BYTES: u32 = 1;
/// Number of entries in `READ_STATS`.
pub const READ_STATS_LEN: u32 = 2;

/// Number of log2 buckets in the `READ_HIST` size histogram (bucket `b` =
/// read sizes in `[2^b, 2^(b+1))`).
pub const READ_HIST_BUCKETS: u32 = 64;

/// Index of the completed device-read count in the `BIO_STATS` array map.
pub const BIO_STAT_READS: u32 = 0;
/// Index of the completed device-read byte total in the `BIO_STATS` array map.
pub const BIO_STAT_BYTES: u32 = 1;
/// Number of entries in `BIO_STATS`.
pub const BIO_STATS_LEN: u32 = 2;

/// Number of log2 buckets in the `BIO_LAT_HIST` latency histogram (bucket `b` =
/// block-read service latencies in `[2^b, 2^(b+1))` nanoseconds).
pub const BIO_LAT_BUCKETS: u32 = 64;

/// Number of log2 buckets in every nanosecond-latency histogram (`RDLAT_HIST`,
/// `FUTEX_WAIT_HIST`, `RUNQ_HIST`, `CONNECT_LAT_HIST`): bucket `b` is latencies in
/// `[2^b, 2^(b+1))` ns.
pub const LAT_HIST_BUCKETS: u32 = 64;

/// Index of the futex-wait count in the `FUTEX_STATS` array map.
pub const FUTEX_STAT_WAITS: u32 = 0;
/// Number of entries in `FUTEX_STATS`.
pub const FUTEX_STATS_LEN: u32 = 1;

/// Index of the TCP retransmit count in the `NET_STATS` array map.
pub const NET_STAT_RETRANSMITS: u32 = 0;
/// Index of the new-TCP-connection count in the `NET_STATS` array map.
pub const NET_STAT_CONNECTIONS: u32 = 1;
/// Number of entries in `NET_STATS`.
pub const NET_STATS_LEN: u32 = 2;

/// Key of the `BIO_INFLIGHT` map: a block request identified by its device and
/// start sector, used to pair `block_rq_issue` with `block_rq_complete` and time
/// the device round-trip. `_pad` is explicit and always zero so the key has no
/// uninitialized bytes for BPF's byte-wise key comparison.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BioKey {
    /// Start sector of the request.
    pub sector: u64,
    /// Device number (`dev_t`).
    pub dev: u32,
    /// Zeroed padding to a 16-byte, alignment-clean key.
    pub _pad: u32,
}

/// Maximum context nesting depth tracked per thread. Vortex work nests (a scan
/// split runs a filter, which decodes an encoded child, which decodes *its*
/// child …), so the kernel keeps a per-thread stack and attributes samples to its
/// top. Each frame is a 16-byte [`ContextKey`]; 16 × 16 B + 8 B header = 264 B,
/// within the 512-byte BPF stack when a zeroed entry is inserted. Deeper nesting
/// clamps (samples attribute to the deepest tracked frame).
pub const CONTEXT_STACK_DEPTH: usize = 16;

/// Key of the sample maps (`PMU_STATS`, and future off-CPU / futex / alloc maps):
/// `(kind, id)` describing the work in flight. `id` is an opaque instance key the
/// producer chooses and the host resolves (for a decode it is the encoding's
/// interned symbol; for a conjunct/split/segment it is the index). Both halves
/// arrive by register at the marker, so the kernel never reads user memory.
///
/// `_pad` is explicit and always zero: BPF compares map keys by raw bytes, so the
/// struct must have no uninitialized padding.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContextKey {
    /// Opaque instance id (see struct docs).
    pub id: u64,
    /// `ContextKind` discriminant (`vortex_ebpf::ContextKind`), stored raw; the
    /// kernel never interprets it.
    pub kind: u32,
    /// Zeroed padding so the 16-byte key has no uninitialized bytes.
    pub _pad: u32,
}

/// Per-thread context stack: `ctx_push(kind, id)` pushes a [`ContextKey`],
/// `ctx_pop()` pops, and a sample attributes to `frames[depth - 1]`. Stored in
/// `CONTEXT_STACK` keyed by thread id.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContextStack {
    /// Number of frames currently on the stack.
    pub depth: u32,
    /// Zeroed padding so the `frames` array starts 8-byte aligned.
    pub _pad: u32,
    /// The frames; index `depth - 1` is the work currently in flight.
    pub frames: [ContextKey; CONTEXT_STACK_DEPTH],
}

/// PMU sample counts, keyed by [`ContextKey`] in `PMU_STATS`. Each field counts
/// perf-event samples attributed to the work in flight; the collector multiplies
/// by the configured sample period. The SIMD-heavy decode kernels (fastlanes
/// bit-packing, ALP, FSST) are bottlenecked on branch misprediction and LLC
/// misses, so those are tracked alongside the basics.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PmuStats {
    /// CPU-cycle samples.
    pub cycles: u64,
    /// Retired-instruction samples.
    pub instructions: u64,
    /// Cache-miss samples.
    pub cache_misses: u64,
    /// Branch-misprediction samples.
    pub branch_misses: u64,
    /// Last-level-cache load-miss samples.
    pub llc_load_misses: u64,
    /// Backend-stalled-cycle samples.
    pub stalled_cycles: u64,
}

/// Off-CPU time attributed to a [`ContextKey`] in `OFFCPU_STATS`: how long the
/// work in flight spent blocked (descheduled) rather than running.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OffCpuStats {
    /// Total nanoseconds spent off-CPU.
    pub total_ns: u64,
    /// Number of off-CPU episodes.
    pub count: u64,
}

/// Per-thread off-CPU start in `OFFCPU_START`: the timestamp a thread was
/// descheduled and the [`ContextKey`] it was working on at the time, so the
/// matching on-CPU switch can attribute the blocked time.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OffCpuStart {
    /// The work in flight when the thread went off-CPU.
    pub key: ContextKey,
    /// Timestamp (ns) the thread was descheduled.
    pub ts: u64,
}
