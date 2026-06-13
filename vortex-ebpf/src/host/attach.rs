// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Attaching the slimmed kernel programs: the read-syscall tracepoints
//! (`--syscalls`) and the context uprobes that drive PMU attribution (`--pmu`).

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use aya::Ebpf;
use aya::maps::Array as BpfArray;
use aya::programs::PerfEvent;
use aya::programs::TracePoint;
use aya::programs::UProbe;
use aya::programs::perf_event::PerfEventScope;
use aya::programs::perf_event::PerfTypeId;
use aya::programs::perf_event::SamplePolicy;

/// (eBPF program, marker symbol) pairs that maintain the per-thread context stack
/// the PMU sampler reads. Present in the workload only when built with
/// `profile-pmu`.
const CONTEXT_PROBES: &[(&str, &str)] = &[
    ("ctx_push", "vortex_ebpf_ctx_push"),
    ("ctx_pop", "vortex_ebpf_ctx_pop"),
];

/// (eBPF program, tracepoint category, tracepoint name) attached for `--syscalls`:
/// read-syscall entry (count + size) and exit (latency). Pid-scoped.
const TRACEPOINTS: &[(&str, &str, &str)] = &[
    ("trace_pread", "syscalls", "sys_enter_pread64"),
    ("trace_read", "syscalls", "sys_enter_read"),
    ("trace_pread_exit", "syscalls", "sys_exit_pread64"),
    ("trace_read_exit", "syscalls", "sys_exit_read"),
];

/// (eBPF program, tracepoint category, tracepoint name) attached for `--bio`:
/// the block-layer read round-trip past the page cache.
const BLOCK_TRACEPOINTS: &[(&str, &str, &str)] = &[
    ("block_issue", "block", "block_rq_issue"),
    ("block_complete", "block", "block_rq_complete"),
];

/// (eBPF program, tracepoint category, tracepoint name) attached for `--locks`:
/// blocking-futex wait time. Pid-scoped.
const LOCK_TRACEPOINTS: &[(&str, &str, &str)] = &[
    ("trace_futex_enter", "syscalls", "sys_enter_futex"),
    ("trace_futex_exit", "syscalls", "sys_exit_futex"),
];

/// (eBPF program, tracepoint category, tracepoint name) attached for `--net`:
/// TCP retransmits and connection setup. System-wide.
const NET_TRACEPOINTS: &[(&str, &str, &str)] = &[
    ("trace_tcp_retransmit", "tcp", "tcp_retransmit_skb"),
    ("trace_inet_sock_set_state", "sock", "inet_sock_set_state"),
];

/// (eBPF program, tracepoint category, tracepoint name) attached for `--offcpu`:
/// off-CPU block time and runqueue latency, scoped in-kernel to phase-active
/// threads via the context stack.
const SCHED_TRACEPOINTS: &[(&str, &str, &str)] = &[
    ("trace_sched_switch", "sched", "sched_switch"),
    ("trace_sched_wakeup", "sched", "sched_wakeup"),
];

/// One perf-event sample per this many hardware events. Coarse enough to keep
/// overhead low; the collector reports raw sample counts (advisory, never gated).
const PMU_SAMPLE_PERIOD: u64 = 100_000;

/// Scope the read tracepoints to `pid` via the `TARGET_PID` map, then attach
/// them. The tracepoints fire process-wide; the in-kernel filter does the
/// scoping.
pub fn attach_read_tracepoints(ebpf: &mut Ebpf, pid: u32) -> Result<()> {
    set_target_pid(ebpf, pid)?;
    attach_tracepoints(ebpf, TRACEPOINTS)
}

/// Attach the blocking-futex tracepoints for `--locks`, scoped to `pid`.
pub fn attach_lock_tracepoints(ebpf: &mut Ebpf, pid: u32) -> Result<()> {
    set_target_pid(ebpf, pid)?;
    attach_tracepoints(ebpf, LOCK_TRACEPOINTS)
}

/// Attach the block-layer read tracepoints for `--bio`. Not pid-scoped: the block
/// layer serves reads asynchronously, so the counters are system-wide for the run.
pub fn attach_block_tracepoints(ebpf: &mut Ebpf) -> Result<()> {
    attach_tracepoints(ebpf, BLOCK_TRACEPOINTS)
}

/// Attach the TCP tracepoints for `--net`. System-wide for the run.
pub fn attach_net_tracepoints(ebpf: &mut Ebpf) -> Result<()> {
    attach_tracepoints(ebpf, NET_TRACEPOINTS)
}

/// Attach the scheduler tracepoints for `--offcpu`. Scoping is in-kernel (only
/// threads with an active context frame), so no `TARGET_PID` is needed.
pub fn attach_sched_tracepoints(ebpf: &mut Ebpf) -> Result<()> {
    attach_tracepoints(ebpf, SCHED_TRACEPOINTS)
}

/// Attach the per-context PMU samplers for `--pmu`, scoped to `pid`. Each counter
/// is best-effort: hardware counters are often unavailable on virtualized hosts
/// (e.g. many cloud instances), so a counter that cannot be opened is skipped
/// with a warning rather than failing the whole run.
pub fn attach_pmu_events(ebpf: &mut Ebpf, pid: u32) -> Result<()> {
    // (program, perf type, config). Configs are the standard `perf_event_open`
    // hardware-counter ids; LLC is a `HW_CACHE` triple (LL | READ<<8 | MISS<<16).
    let counters: [(&str, PerfTypeId, u64); 6] = [
        ("pmu_cycles", PerfTypeId::Hardware, 0),
        ("pmu_instructions", PerfTypeId::Hardware, 1),
        ("pmu_cache_misses", PerfTypeId::Hardware, 3),
        ("pmu_branch_misses", PerfTypeId::Hardware, 5),
        ("pmu_stalled", PerfTypeId::Hardware, 8),
        ("pmu_llc_misses", PerfTypeId::HwCache, 0x1_0002),
    ];
    for (prog, perf_type, config) in counters {
        let pe: &mut PerfEvent = ebpf
            .program_mut(prog)
            .with_context(|| format!("eBPF program {prog} missing"))?
            .try_into()
            .with_context(|| format!("{prog} is not a PerfEvent"))?;
        pe.load().with_context(|| format!("loading {prog}"))?;
        if let Err(e) = pe.attach(
            perf_type,
            config,
            PerfEventScope::OneProcessAnyCpu { pid },
            SamplePolicy::Period(PMU_SAMPLE_PERIOD),
            true,
        ) {
            eprintln!("vx profile: PMU counter {prog} unavailable ({e}); skipping");
        }
    }
    Ok(())
}

/// Scope the in-kernel pid filter (`TARGET_PID`) used by the pid-scoped layers
/// (read syscalls, futex). Idempotent across layers.
fn set_target_pid(ebpf: &mut Ebpf, pid: u32) -> Result<()> {
    let mut target_pid: BpfArray<_, u32> = BpfArray::try_from(
        ebpf.map_mut("TARGET_PID")
            .context("TARGET_PID map missing")?,
    )?;
    target_pid.set(0, pid, 0).context("setting TARGET_PID")?;
    Ok(())
}

fn attach_tracepoints(ebpf: &mut Ebpf, tracepoints: &[(&str, &str, &str)]) -> Result<()> {
    for (prog, category, name) in tracepoints {
        let tp: &mut TracePoint = ebpf
            .program_mut(prog)
            .with_context(|| format!("eBPF program {prog} missing"))?
            .try_into()
            .with_context(|| format!("{prog} is not a TracePoint"))?;
        tp.load().with_context(|| format!("loading {prog}"))?;
        tp.attach(category, name)
            .with_context(|| format!("attaching {prog} to {category}:{name}"))?;
    }
    Ok(())
}

/// Attach the context push/pop uprobes to `target`, scoped to `pid`, so the
/// kernel can track the per-thread context stack for PMU attribution. Fails if
/// the marker symbols are absent — i.e. the workload was not built with
/// `profile-pmu`.
pub fn attach_context_uprobes(
    ebpf: &mut Ebpf,
    target: impl AsRef<Path>,
    pid: libc::pid_t,
) -> Result<()> {
    let target = target.as_ref();
    for (prog_name, symbol) in CONTEXT_PROBES {
        let prog: &mut UProbe = ebpf
            .program_mut(prog_name)
            .with_context(|| format!("eBPF program {prog_name} missing"))?
            .try_into()
            .with_context(|| format!("{prog_name} is not a UProbe"))?;
        prog.load()
            .with_context(|| format!("loading {prog_name}"))?;
        prog.attach(Some(*symbol), 0, target, Some(pid))
            .with_context(|| {
                format!("attaching {symbol} (was the workload built with `profile-pmu`?)")
            })?;
    }
    Ok(())
}
