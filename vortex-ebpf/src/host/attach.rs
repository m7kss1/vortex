// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Attaching the slimmed kernel programs: the read-syscall tracepoints
//! (`--syscalls`) and the decode uprobes that drive per-encoding PMU
//! attribution (`--pmu`).

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use aya::Ebpf;
use aya::maps::Array as BpfArray;
use aya::programs::TracePoint;
use aya::programs::UProbe;

/// (eBPF program, marker symbol) pairs that maintain the per-thread decode stack
/// the PMU sampler reads. Present in the workload only when built with
/// `profile-pmu`.
const DECODE_PROBES: &[(&str, &str)] = &[
    ("decode_begin", "vortex_ebpf_decode_begin"),
    ("decode_end", "vortex_ebpf_decode_end"),
];

/// (eBPF program, tracepoint category, tracepoint name) attached for `--syscalls`.
const TRACEPOINTS: &[(&str, &str, &str)] = &[
    ("trace_pread", "syscalls", "sys_enter_pread64"),
    ("trace_read", "syscalls", "sys_enter_read"),
];

/// Scope the read tracepoints to `pid` via the `TARGET_PID` map, then attach
/// them. The tracepoints fire process-wide; the in-kernel filter does the
/// scoping.
pub fn attach_read_tracepoints(ebpf: &mut Ebpf, pid: u32) -> Result<()> {
    {
        let mut target_pid: BpfArray<_, u32> = BpfArray::try_from(
            ebpf.map_mut("TARGET_PID")
                .context("TARGET_PID map missing")?,
        )?;
        target_pid.set(0, pid, 0).context("setting TARGET_PID")?;
    }
    for (prog, category, name) in TRACEPOINTS {
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

/// Attach the decode begin/end uprobes to `target`, scoped to `pid`, so the
/// kernel can track the per-thread decode stack for PMU attribution. Fails if
/// the marker symbols are absent — i.e. the workload was not built with
/// `profile-pmu`.
pub fn attach_decode_uprobes(
    ebpf: &mut Ebpf,
    target: impl AsRef<Path>,
    pid: libc::pid_t,
) -> Result<()> {
    let target = target.as_ref();
    for (prog_name, symbol) in DECODE_PROBES {
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
