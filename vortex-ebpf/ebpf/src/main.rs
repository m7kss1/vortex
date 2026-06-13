// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex eBPF — kernel-side eBPF programs.
//!
//! Scoped to the facts userspace cannot see for itself:
//!
//! - **Read syscalls** (`trace_pread`/`trace_read`): count, byte total, and a
//!   log2 size histogram, pid-scoped via `TARGET_PID`.
//! - **PMU** (`pmu_*` perf_events): hardware-counter samples attributed to the
//!   work currently in flight on each thread, tracked by a per-thread **context
//!   stack** maintained by the `ctx_push`/`ctx_pop` uprobes. Work nests (a split
//!   runs a filter that decodes a child that decodes *its* child …), so the
//!   stack — not a single current value — keeps the parent attributed after a
//!   nested child finishes.
//!
//! All format-semantic metrics are collected in-process via `vortex-metrics`;
//! they do not appear here. Map layouts are shared with the collector by
//! `#[path]`-including `vortex-ebpf/src/types.rs`.

#![no_std]
#![no_main]

use aya_ebpf::helpers::bpf_get_current_pid_tgid;
use aya_ebpf::helpers::bpf_ktime_get_ns;
use aya_ebpf::macros::map;
use aya_ebpf::macros::perf_event;
use aya_ebpf::macros::tracepoint;
use aya_ebpf::macros::uprobe;
use aya_ebpf::maps::Array;
use aya_ebpf::maps::HashMap;
use aya_ebpf::programs::PerfEventContext;
use aya_ebpf::programs::ProbeContext;
use aya_ebpf::programs::TracePointContext;
use types::BIO_LAT_BUCKETS;
use types::BIO_STAT_BYTES;
use types::BIO_STAT_READS;
use types::BIO_STATS_LEN;
use types::BioKey;
use types::CONTEXT_STACK_DEPTH;
use types::ContextKey;
use types::ContextStack;
use types::FUTEX_STAT_WAITS;
use types::FUTEX_STATS_LEN;
use types::LAT_HIST_BUCKETS;
use types::NET_STAT_CONNECTIONS;
use types::NET_STAT_RETRANSMITS;
use types::NET_STATS_LEN;
use types::OffCpuStart;
use types::OffCpuStats;
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

/// Field offsets in the `block:block_rq_issue` / `block:block_rq_complete`
/// tracepoints (verified against the kernel's `.../format`; identical layout for
/// the shared fields). `rwbs[0]` is `'R'` for reads (incl. read-ahead).
const BIO_DEV_OFFSET: usize = 8;
const BIO_SECTOR_OFFSET: usize = 16;
const BIO_NR_SECTOR_OFFSET: usize = 24;
const BIO_RWBS_OFFSET: usize = 32;
/// A `sector_t` is a 512-byte sector.
const SECTOR_BYTES: u64 = 512;

/// `op` field offset in `syscalls:sys_enter_futex` (verified against the kernel's
/// `.../format`). Low 7 bits are the futex command; `FUTEX_WAIT`(0) and
/// `FUTEX_WAIT_BITSET`(9) are the blocking ops a lock contends on.
const FUTEX_OP_OFFSET: usize = 24;
const FUTEX_CMD_MASK: u64 = 0x7f;
const FUTEX_WAIT: u64 = 0;
const FUTEX_WAIT_BITSET: u64 = 9;

/// Field offsets in `sock:inet_sock_set_state` (verified against the kernel's
/// `.../format`).
const SOCK_SKADDR_OFFSET: usize = 8;
const SOCK_NEWSTATE_OFFSET: usize = 20;
const SOCK_PROTOCOL_OFFSET: usize = 30;
/// `IPPROTO_TCP`.
const IPPROTO_TCP: u16 = 6;
/// TCP states we pair for connect latency.
const TCP_ESTABLISHED: i32 = 1;
const TCP_SYN_SENT: i32 = 2;

/// Field offsets in `sched:sched_switch` / `sched:sched_wakeup` (verified against
/// the kernel's `.../format`). `prev_state & 0xff == 0` is `TASK_RUNNING` — a
/// preemption, not a blocking sleep.
const SCHED_PREV_PID_OFFSET: usize = 24;
const SCHED_PREV_STATE_OFFSET: usize = 32;
const SCHED_NEXT_PID_OFFSET: usize = 56;
const SCHED_WAKEUP_PID_OFFSET: usize = 24;
const TASK_STATE_MASK: i64 = 0xff;

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

/// Per-thread context stack: top is the work currently in flight, which PMU
/// samples attribute to.
#[map]
static CONTEXT_STACK: HashMap<u32, ContextStack> = HashMap::with_max_entries(8192, 0);

/// PMU sample counts keyed by the in-flight `(kind, id)`.
#[map]
static PMU_STATS: HashMap<ContextKey, PmuStats> = HashMap::with_max_entries(256, 0);

/// In-flight block reads: `(dev, sector)` → issue timestamp (ns), to pair a
/// `block_rq_issue` with its `block_rq_complete` and time the device round-trip.
#[map]
static BIO_INFLIGHT: HashMap<BioKey, u64> = HashMap::with_max_entries(16384, 0);

/// Completed device-read counters: `[reads, bytes]`.
#[map]
static BIO_STATS: Array<u64> = Array::with_max_entries(BIO_STATS_LEN, 0);

/// Log2 histogram of block-read service latencies (nanoseconds).
#[map]
static BIO_LAT_HIST: Array<u64> = Array::with_max_entries(BIO_LAT_BUCKETS, 0);

/// Per-thread `read`/`pread64` syscall entry timestamp, to time the syscall.
#[map]
static RD_START: HashMap<u32, u64> = HashMap::with_max_entries(8192, 0);

/// Log2 histogram of `read`/`pread64` syscall latencies (nanoseconds).
#[map]
static RDLAT_HIST: Array<u64> = Array::with_max_entries(LAT_HIST_BUCKETS, 0);

/// Per-thread blocking-futex entry timestamp, to time lock waits.
#[map]
static FUTEX_START: HashMap<u32, u64> = HashMap::with_max_entries(8192, 0);

/// Futex-wait counters: `[waits]`.
#[map]
static FUTEX_STATS: Array<u64> = Array::with_max_entries(FUTEX_STATS_LEN, 0);

/// Log2 histogram of blocking-futex wait latencies (nanoseconds).
#[map]
static FUTEX_WAIT_HIST: Array<u64> = Array::with_max_entries(LAT_HIST_BUCKETS, 0);

/// Network counters: `[tcp_retransmits, tcp_connections]`. System-wide.
#[map]
static NET_STATS: Array<u64> = Array::with_max_entries(NET_STATS_LEN, 0);

/// In-flight TCP connects: socket address → `SYN_SENT` timestamp (ns).
#[map]
static CONNECT_START: HashMap<u64, u64> = HashMap::with_max_entries(4096, 0);

/// Log2 histogram of TCP connect latencies (`SYN_SENT`→`ESTABLISHED`, ns).
#[map]
static CONNECT_LAT_HIST: Array<u64> = Array::with_max_entries(LAT_HIST_BUCKETS, 0);

/// Off-CPU (blocked) time attributed to the in-flight `(kind, id)`.
#[map]
static OFFCPU_STATS: HashMap<ContextKey, OffCpuStats> = HashMap::with_max_entries(256, 0);

/// Per-thread off-CPU start: when a phase-active thread was descheduled.
#[map]
static OFFCPU_START: HashMap<u32, OffCpuStart> = HashMap::with_max_entries(8192, 0);

/// Per-thread wakeup/preemption timestamp, to time runqueue latency.
#[map]
static WAKEUP_TS: HashMap<u32, u64> = HashMap::with_max_entries(8192, 0);

/// Log2 histogram of runqueue latency (wakeup→run, nanoseconds) for phase-active
/// threads.
#[map]
static RUNQ_HIST: Array<u64> = Array::with_max_entries(LAT_HIST_BUCKETS, 0);

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
    // Stamp entry so the matching exit can time the bare syscall (queue time is
    // not included here — this is the kernel-side read latency).
    let t = tid();
    let now = unsafe { bpf_ktime_get_ns() };
    let _ = RD_START.insert(&t, &now, 0);
    0
}

/// Record the latency of a `read`/`pread64` whose entry we stamped.
#[inline(always)]
fn on_read_exit() -> u32 {
    let t = tid();
    if let Some(&start) = unsafe { RD_START.get(&t) } {
        let _ = RD_START.remove(&t);
        let latency = unsafe { bpf_ktime_get_ns() }.saturating_sub(start);
        array_add(&RDLAT_HIST, log2_bucket(latency).min(LAT_HIST_BUCKETS - 1), 1);
    }
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

#[tracepoint]
pub fn trace_pread_exit(_ctx: TracePointContext) -> u32 {
    on_read_exit()
}

#[tracepoint]
pub fn trace_read_exit(_ctx: TracePointContext) -> u32 {
    on_read_exit()
}

// ---------------------------------------------------------------------------
// Futex contention (lock-wait time, `--locks`)
// ---------------------------------------------------------------------------

#[tracepoint]
pub fn trace_futex_enter(ctx: TracePointContext) -> u32 {
    if !in_target() {
        return 0;
    }
    let op: u64 = unsafe { ctx.read_at(FUTEX_OP_OFFSET) }.unwrap_or(0);
    let cmd = op & FUTEX_CMD_MASK;
    // Only the blocking waits represent contention; ignore WAKE and friends.
    if cmd != FUTEX_WAIT && cmd != FUTEX_WAIT_BITSET {
        return 0;
    }
    let t = tid();
    let now = unsafe { bpf_ktime_get_ns() };
    let _ = FUTEX_START.insert(&t, &now, 0);
    0
}

#[tracepoint]
pub fn trace_futex_exit(_ctx: TracePointContext) -> u32 {
    let t = tid();
    if let Some(&start) = unsafe { FUTEX_START.get(&t) } {
        let _ = FUTEX_START.remove(&t);
        let waited = unsafe { bpf_ktime_get_ns() }.saturating_sub(start);
        array_add(&FUTEX_STATS, FUTEX_STAT_WAITS, 1);
        array_add(&FUTEX_WAIT_HIST, log2_bucket(waited).min(LAT_HIST_BUCKETS - 1), 1);
    }
    0
}

// ---------------------------------------------------------------------------
// Network: TCP retransmits + connection setup (`--net`, system-wide)
// ---------------------------------------------------------------------------

#[tracepoint]
pub fn trace_tcp_retransmit(_ctx: TracePointContext) -> u32 {
    array_add(&NET_STATS, NET_STAT_RETRANSMITS, 1);
    0
}

#[tracepoint]
pub fn trace_inet_sock_set_state(ctx: TracePointContext) -> u32 {
    let protocol: u16 = unsafe { ctx.read_at(SOCK_PROTOCOL_OFFSET) }.unwrap_or(0);
    if protocol != IPPROTO_TCP {
        return 0;
    }
    let newstate: i32 = unsafe { ctx.read_at(SOCK_NEWSTATE_OFFSET) }.unwrap_or(0);
    let skaddr: u64 = unsafe { ctx.read_at(SOCK_SKADDR_OFFSET) }.unwrap_or(0);
    let now = unsafe { bpf_ktime_get_ns() };
    if newstate == TCP_SYN_SENT {
        let _ = CONNECT_START.insert(&skaddr, &now, 0);
    } else if newstate == TCP_ESTABLISHED {
        array_add(&NET_STATS, NET_STAT_CONNECTIONS, 1);
        if let Some(&start) = unsafe { CONNECT_START.get(&skaddr) } {
            let _ = CONNECT_START.remove(&skaddr);
            let latency = now.saturating_sub(start);
            array_add(&CONNECT_LAT_HIST, log2_bucket(latency).min(LAT_HIST_BUCKETS - 1), 1);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Block-layer device reads (ground truth past the page cache)
// ---------------------------------------------------------------------------

/// Read the `(dev, sector)` key common to both block tracepoints.
#[inline(always)]
fn bio_key(ctx: &TracePointContext) -> BioKey {
    let dev: u32 = unsafe { ctx.read_at(BIO_DEV_OFFSET) }.unwrap_or(0);
    let sector: u64 = unsafe { ctx.read_at(BIO_SECTOR_OFFSET) }.unwrap_or(0);
    BioKey {
        sector,
        dev,
        _pad: 0,
    }
}

/// Record the issue time of a block **read** so the matching completion can time
/// it. Not pid-scoped: the block layer serves reads asynchronously (read-ahead,
/// writeback), so these counters are system-wide for the run — meaningful on an
/// otherwise-idle profiling host.
#[tracepoint]
pub fn block_issue(ctx: TracePointContext) -> u32 {
    let rwbs0: u8 = unsafe { ctx.read_at(BIO_RWBS_OFFSET) }.unwrap_or(0);
    // 'R' covers reads and read-ahead ("R", "RA"); writes/flush/discard differ.
    if rwbs0 != b'R' {
        return 0;
    }
    let key = bio_key(&ctx);
    let now = unsafe { bpf_ktime_get_ns() };
    let _ = BIO_INFLIGHT.insert(&key, &now, 0);
    0
}

#[tracepoint]
pub fn block_complete(ctx: TracePointContext) -> u32 {
    let key = bio_key(&ctx);
    let issued = match unsafe { BIO_INFLIGHT.get(&key) } {
        Some(&ts) => ts,
        // Not a read we tracked (write, or issued before we attached).
        None => return 0,
    };
    let _ = BIO_INFLIGHT.remove(&key);
    let latency = unsafe { bpf_ktime_get_ns() }.saturating_sub(issued);
    let nr_sector: u32 = unsafe { ctx.read_at(BIO_NR_SECTOR_OFFSET) }.unwrap_or(0);
    array_add(&BIO_STATS, BIO_STAT_READS, 1);
    array_add(&BIO_STATS, BIO_STAT_BYTES, nr_sector as u64 * SECTOR_BYTES);
    array_add(&BIO_LAT_HIST, log2_bucket(latency).min(BIO_LAT_BUCKETS - 1), 1);
    0
}

// ---------------------------------------------------------------------------
// Context stack (for sample attribution)
// ---------------------------------------------------------------------------

/// An empty frame used to seed map slots.
const EMPTY_KEY: ContextKey = ContextKey {
    id: 0,
    kind: 0,
    _pad: 0,
};

/// An empty stack used to seed a thread's entry; small enough for the BPF stack.
const EMPTY_STACK: ContextStack = ContextStack {
    depth: 0,
    _pad: 0,
    frames: [EMPTY_KEY; CONTEXT_STACK_DEPTH],
};

#[uprobe]
pub fn ctx_push(ctx: ProbeContext) -> u32 {
    let kind: u32 = match ctx.arg(0) {
        Some(k) => k,
        None => return 1,
    };
    let id: u64 = ctx.arg(1).unwrap_or(0);
    let t = tid();
    // Modify the map entry in place via a pointer — the whole `ContextStack` is
    // larger than is comfortable to copy onto the 512-byte BPF stack.
    if CONTEXT_STACK.get_ptr_mut(&t).is_none() {
        let _ = CONTEXT_STACK.insert(&t, &EMPTY_STACK, 0);
    }
    if let Some(s) = CONTEXT_STACK.get_ptr_mut(&t) {
        unsafe {
            let d = (*s).depth as usize;
            if d < CONTEXT_STACK_DEPTH {
                // Register-passed `(kind, id)` — no user-memory read. `_pad` stays
                // zero so the key has no uninitialized bytes for map comparison.
                (*s).frames[d] = ContextKey { id, kind, _pad: 0 };
                (*s).depth += 1;
            }
        }
    }
    0
}

#[uprobe]
pub fn ctx_pop(_ctx: ProbeContext) -> u32 {
    if let Some(s) = CONTEXT_STACK.get_ptr_mut(&tid()) {
        unsafe { (*s).depth = (*s).depth.saturating_sub(1) }
    }
    0
}

// ---------------------------------------------------------------------------
// Off-CPU + runqueue latency, attributed to the in-flight context (`--offcpu`)
// ---------------------------------------------------------------------------

/// The `(kind, id)` on the top of `tid`'s context stack, if it is doing Vortex
/// work. Scopes the scheduler probes to the workload's phase-active threads.
#[inline(always)]
fn top_context(t: u32) -> Option<ContextKey> {
    let s = unsafe { CONTEXT_STACK.get(&t) }?;
    let depth = s.depth;
    if depth == 0 {
        return None;
    }
    let top = (depth as usize - 1).min(CONTEXT_STACK_DEPTH - 1);
    Some(s.frames[top])
}

#[inline(always)]
fn offcpu_add(key: &ContextKey, delta: u64) {
    if OFFCPU_STATS.get_ptr_mut(key).is_none() {
        let _ = OFFCPU_STATS.insert(key, &OffCpuStats::default(), 0);
    }
    if let Some(p) = OFFCPU_STATS.get_ptr_mut(key) {
        unsafe {
            (*p).total_ns += delta;
            (*p).count += 1;
        }
    }
}

#[tracepoint]
pub fn trace_sched_wakeup(ctx: TracePointContext) -> u32 {
    let pid: u32 = unsafe { ctx.read_at(SCHED_WAKEUP_PID_OFFSET) }.unwrap_or(0);
    // Only time threads currently doing Vortex work.
    if top_context(pid).is_some() {
        let now = unsafe { bpf_ktime_get_ns() };
        let _ = WAKEUP_TS.insert(&pid, &now, 0);
    }
    0
}

#[tracepoint]
pub fn trace_sched_switch(ctx: TracePointContext) -> u32 {
    let prev: u32 = unsafe { ctx.read_at(SCHED_PREV_PID_OFFSET) }.unwrap_or(0);
    let next: u32 = unsafe { ctx.read_at(SCHED_NEXT_PID_OFFSET) }.unwrap_or(0);
    let now = unsafe { bpf_ktime_get_ns() };

    // `next` comes on-CPU: account any off-CPU block and runqueue wait we timed.
    if let Some(start) = unsafe { OFFCPU_START.get(&next) }.copied() {
        let _ = OFFCPU_START.remove(&next);
        offcpu_add(&start.key, now.saturating_sub(start.ts));
    }
    if let Some(&wake) = unsafe { WAKEUP_TS.get(&next) } {
        let _ = WAKEUP_TS.remove(&next);
        array_add(
            &RUNQ_HIST,
            log2_bucket(now.saturating_sub(wake)).min(LAT_HIST_BUCKETS - 1),
            1,
        );
    }

    // `prev` leaves the CPU: if it was doing Vortex work, time the gap. A still
    // runnable `prev` (TASK_RUNNING) was preempted — runqueue wait, not a block;
    // a non-running state is a real off-CPU block (sleep/IO/lock).
    if let Some(key) = top_context(prev) {
        let prev_state: i64 = unsafe { ctx.read_at(SCHED_PREV_STATE_OFFSET) }.unwrap_or(0);
        if prev_state & TASK_STATE_MASK == 0 {
            let _ = WAKEUP_TS.insert(&prev, &now, 0);
        } else {
            let _ = OFFCPU_START.insert(&prev, &OffCpuStart { key, ts: now }, 0);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// PMU sampling (attributed to the top of the context stack, if any)
// ---------------------------------------------------------------------------

#[inline(always)]
fn pmu_sample(field: u32) -> u32 {
    if !in_target() {
        return 0;
    }
    // Read the top frame through the map pointer; copying the whole stack would
    // be heavy on the BPF stack.
    let s = match CONTEXT_STACK.get_ptr_mut(&tid()) {
        Some(s) => s,
        None => return 0,
    };
    let depth = unsafe { (*s).depth };
    if depth == 0 {
        return 0;
    }
    let top = (depth as usize - 1).min(CONTEXT_STACK_DEPTH - 1);
    let key = unsafe { (*s).frames[top] };
    if PMU_STATS.get_ptr_mut(&key).is_none() {
        let _ = PMU_STATS.insert(&key, &PmuStats::default(), 0);
    }
    if let Some(p) = PMU_STATS.get_ptr_mut(&key) {
        unsafe {
            match field {
                0 => (*p).cycles += 1,
                1 => (*p).instructions += 1,
                2 => (*p).cache_misses += 1,
                3 => (*p).branch_misses += 1,
                4 => (*p).llc_load_misses += 1,
                _ => (*p).stalled_cycles += 1,
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

#[perf_event]
pub fn pmu_branch_misses(_ctx: PerfEventContext) -> u32 {
    pmu_sample(3)
}

#[perf_event]
pub fn pmu_llc_misses(_ctx: PerfEventContext) -> u32 {
    pmu_sample(4)
}

#[perf_event]
pub fn pmu_stalled(_ctx: PerfEventContext) -> u32 {
    pmu_sample(5)
}
