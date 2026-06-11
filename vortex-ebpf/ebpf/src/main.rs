// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex eBPF — kernel-side eBPF programs.
//!
//! Scoped to the facts userspace cannot see for itself:
//!
//! - **Read syscalls** (`trace_pread`/`trace_read`): count, byte total, and a
//!   log2 size histogram, pid-scoped via `TARGET_PID`.
//! - **Per-encoding PMU** (`pmu_*` perf_events): hardware-counter samples
//!   attributed to the encoding currently decoding on each thread, tracked by a
//!   per-thread **decode stack** maintained by the `decode_begin`/`decode_end`
//!   uprobes. Decodes nest, so the stack — not a single current value — is what
//!   keeps the parent attributed after a nested child finishes.
//!
//! All format-semantic metrics are collected in-process via `vortex-metrics`;
//! they do not appear here. Map layouts are shared with the collector by
//! `#[path]`-including `vortex-ebpf/src/types.rs`.

#![no_std]
#![no_main]

use aya_ebpf::helpers::bpf_get_current_pid_tgid;
use aya_ebpf::helpers::bpf_probe_read_user_buf;
use aya_ebpf::macros::map;
use aya_ebpf::macros::perf_event;
use aya_ebpf::macros::tracepoint;
use aya_ebpf::macros::uprobe;
use aya_ebpf::maps::Array;
use aya_ebpf::maps::HashMap;
use aya_ebpf::programs::PerfEventContext;
use aya_ebpf::programs::ProbeContext;
use aya_ebpf::programs::TracePointContext;
use types::DECODE_STACK_DEPTH;
use types::DecodeStack;
use types::ENC_NAME_LEN;
use types::EncKey;
use types::PmuStats;
use types::READ_HIST_BUCKETS;
use types::READ_STAT_BYTES;
use types::READ_STAT_COUNT;
use types::READ_STATS_LEN;

#[path = "../../src/types.rs"]
mod types;

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // BPF programs cannot panic; this exists only to satisfy `no_std`.
    loop {}
}

/// `count` field offset in the `sys_enter_pread64` / `sys_enter_read` tracepoint
/// (verified against the kernel's `.../format`).
const READ_COUNT_OFFSET: usize = 32;

/// Workload pid the collector set before attach; read tracepoints ignore other
/// processes. `0` means "all processes".
#[map]
static TARGET_PID: Array<u32> = Array::with_max_entries(1, 0);

/// Read-syscall counters: `[count, bytes]`.
#[map]
static READ_STATS: Array<u64> = Array::with_max_entries(READ_STATS_LEN, 0);

/// Log2 histogram of read-syscall sizes.
#[map]
static READ_HIST: Array<u64> = Array::with_max_entries(READ_HIST_BUCKETS, 0);

/// Per-thread decode stack: top is the encoding currently decoding, which PMU
/// samples attribute to.
#[map]
static DECODE_STACK: HashMap<u32, DecodeStack> = HashMap::with_max_entries(8192, 0);

/// Per-encoding PMU sample counts.
#[map]
static PMU_STATS: HashMap<EncKey, PmuStats> = HashMap::with_max_entries(256, 0);

#[inline(always)]
fn tid() -> u32 {
    bpf_get_current_pid_tgid() as u32
}

/// Whether the current thread belongs to the workload the collector is scoping to.
#[inline(always)]
fn in_target() -> bool {
    let tgid = (bpf_get_current_pid_tgid() >> 32) as u32;
    match TARGET_PID.get(0) {
        Some(&p) => p == 0 || p == tgid,
        None => true,
    }
}

#[inline(always)]
fn array_add(map: &Array<u64>, index: u32, delta: u64) {
    if let Some(p) = map.get_ptr_mut(index) {
        unsafe { *p += delta }
    }
}

/// floor(log2(v)); `v == 0` lands in bucket 0.
#[inline(always)]
fn log2_bucket(v: u64) -> u32 {
    if v == 0 {
        0
    } else {
        63u32.saturating_sub(v.leading_zeros())
    }
}

// ---------------------------------------------------------------------------
// Read-syscall tracepoints
// ---------------------------------------------------------------------------

#[inline(always)]
fn on_read(ctx: &TracePointContext) -> u32 {
    if !in_target() {
        return 0;
    }
    let count: u64 = unsafe { ctx.read_at(READ_COUNT_OFFSET) }.unwrap_or(0);
    array_add(&READ_STATS, READ_STAT_COUNT, 1);
    array_add(&READ_STATS, READ_STAT_BYTES, count);
    array_add(&READ_HIST, log2_bucket(count).min(READ_HIST_BUCKETS - 1), 1);
    0
}

#[tracepoint]
pub fn trace_pread(ctx: TracePointContext) -> u32 {
    on_read(&ctx)
}

#[tracepoint]
pub fn trace_read(ctx: TracePointContext) -> u32 {
    on_read(&ctx)
}

// ---------------------------------------------------------------------------
// Decode stack (for per-encoding PMU attribution)
// ---------------------------------------------------------------------------

/// An empty stack used to seed a thread's entry; small enough for the BPF stack.
const EMPTY_STACK: DecodeStack = DecodeStack {
    depth: 0,
    enc_len: [0; DECODE_STACK_DEPTH],
    encs: [[0; ENC_NAME_LEN]; DECODE_STACK_DEPTH],
};

#[uprobe]
pub fn decode_begin(ctx: ProbeContext) -> u32 {
    let enc_ptr: *const u8 = match ctx.arg(0) {
        Some(p) => p,
        None => return 1,
    };
    let enc_len: u64 = ctx.arg(1).unwrap_or(0);
    let t = tid();
    // Modify the map entry in place via a pointer — the whole `DecodeStack` is
    // larger than is comfortable to copy onto the 512-byte BPF stack.
    if DECODE_STACK.get_ptr_mut(&t).is_none() {
        let _ = DECODE_STACK.insert(&t, &EMPTY_STACK, 0);
    }
    if let Some(s) = DECODE_STACK.get_ptr_mut(&t) {
        unsafe {
            let d = (*s).depth as usize;
            if d < DECODE_STACK_DEPTH {
                // Fixed-width read straight into the map slot; stable per encoding
                // (bytes past the name are constant interner-arena content).
                let _ = bpf_probe_read_user_buf(enc_ptr, &mut (*s).encs[d]);
                (*s).enc_len[d] = enc_len.min(ENC_NAME_LEN as u64) as u32;
                (*s).depth += 1;
            }
        }
    }
    0
}

#[uprobe]
pub fn decode_end(_ctx: ProbeContext) -> u32 {
    if let Some(s) = DECODE_STACK.get_ptr_mut(&tid()) {
        unsafe { (*s).depth = (*s).depth.saturating_sub(1) }
    }
    0
}

// ---------------------------------------------------------------------------
// PMU sampling (attributed to the top of the decode stack, if any)
// ---------------------------------------------------------------------------

#[inline(always)]
fn pmu_sample(field: u32) -> u32 {
    if !in_target() {
        return 0;
    }
    // Read the top frame through the map pointer; copying the whole stack would
    // be heavy on the BPF stack.
    let s = match DECODE_STACK.get_ptr_mut(&tid()) {
        Some(s) => s,
        None => return 0,
    };
    let depth = unsafe { (*s).depth };
    if depth == 0 {
        return 0;
    }
    let top = (depth as usize - 1).min(DECODE_STACK_DEPTH - 1);
    let key = EncKey(unsafe { (*s).encs[top] });
    let enc_len = unsafe { (*s).enc_len[top] };
    if PMU_STATS.get_ptr_mut(&key).is_none() {
        let _ = PMU_STATS.insert(
            &key,
            &PmuStats {
                cycles: 0,
                instructions: 0,
                cache_misses: 0,
                enc_len,
                _pad: 0,
            },
            0,
        );
    }
    if let Some(p) = PMU_STATS.get_ptr_mut(&key) {
        unsafe {
            match field {
                0 => (*p).cycles += 1,
                1 => (*p).instructions += 1,
                _ => (*p).cache_misses += 1,
            }
        }
    }
    0
}

#[perf_event]
pub fn pmu_cycles(_ctx: PerfEventContext) -> u32 {
    pmu_sample(0)
}

#[perf_event]
pub fn pmu_instructions(_ctx: PerfEventContext) -> u32 {
    pmu_sample(1)
}

#[perf_event]
pub fn pmu_cache_misses(_ctx: PerfEventContext) -> u32 {
    pmu_sample(2)
}
