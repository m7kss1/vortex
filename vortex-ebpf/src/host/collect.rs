// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading the eBPF maps after a profiled run. Only the kernel-side facts live
//! here — read-syscall sizes and PMU samples keyed by the in-flight
//! [`ContextKey`]; all format-semantic metrics come from the in-process
//! `vortex-metrics` registry instead. The host renderer resolves a `ContextKey`
//! to a human name (it owns the `kind`→name policy and the encoding interner).

use anyhow::Context;
use anyhow::Result;
use aya::Ebpf;
use aya::maps::Array as BpfArray;
use aya::maps::HashMap as BpfHashMap;
use aya::maps::Map;

use crate::host::probe::ProbeOptions;
use crate::types::ContextKey;
use crate::types::OffCpuStats;
use crate::types::PmuStats;

/// Read-syscall layer (`--syscalls`).
#[derive(Debug, Default)]
pub struct ReadSnapshot {
    /// `[count, bytes]` of read syscalls (see `READ_STAT_*` indices).
    pub read_stats: Vec<u64>,
    /// Log2 histogram of read-syscall sizes.
    pub read_hist: Vec<u64>,
    /// Log2 histogram of read-syscall latencies (nanoseconds).
    pub rdlat_hist: Vec<u64>,
}

/// Block-layer device-read layer (`--bio`).
#[derive(Debug, Default)]
pub struct BioSnapshot {
    /// `[reads, bytes]` completed to block devices (see `BIO_STAT_*` indices).
    pub stats: Vec<u64>,
    /// Log2 histogram of block-read service latencies (nanoseconds).
    pub lat_hist: Vec<u64>,
}

/// Futex-contention layer (`--locks`).
#[derive(Debug, Default)]
pub struct LocksSnapshot {
    /// `[waits]` blocking-futex count (see `FUTEX_STAT_*` indices).
    pub stats: Vec<u64>,
    /// Log2 histogram of blocking-futex wait latencies (nanoseconds).
    pub wait_hist: Vec<u64>,
}

/// Network layer (`--net`).
#[derive(Debug, Default)]
pub struct NetSnapshot {
    /// `[retransmits, connections]` (see `NET_STAT_*` indices).
    pub stats: Vec<u64>,
    /// Log2 histogram of TCP connect latencies (nanoseconds).
    pub connect_lat_hist: Vec<u64>,
}

/// Off-CPU / scheduler layer (`--offcpu`).
#[derive(Debug, Default)]
pub struct OffCpuSnapshot {
    /// Off-CPU block time per in-flight `(kind, id)`, sorted by total descending.
    /// The renderer resolves each key to a name.
    pub stats: Vec<(ContextKey, OffCpuStats)>,
    /// Log2 histogram of runqueue latency (nanoseconds).
    pub runq_hist: Vec<u64>,
}

/// Everything read out of the eBPF maps at the end of a run. Each layer is
/// present only if it was attached, so the renderer emits a section's keys only
/// when its layer ran.
#[derive(Debug, Default)]
pub struct Snapshot {
    /// Read-syscall layer, present with `--syscalls`.
    pub read: Option<ReadSnapshot>,
    /// PMU sample counts keyed by the in-flight `(kind, id)`, sorted by cycle
    /// samples descending; empty unless the PMU layer ran. The renderer resolves
    /// each key to a name.
    pub pmu: Vec<(ContextKey, PmuStats)>,
    /// Block-layer device-read layer, present with `--bio`.
    pub bio: Option<BioSnapshot>,
    /// Futex-contention layer, present with `--locks`.
    pub locks: Option<LocksSnapshot>,
    /// Network layer, present with `--net`.
    pub net: Option<NetSnapshot>,
    /// Off-CPU / scheduler layer, present with `--offcpu`.
    pub offcpu: Option<OffCpuSnapshot>,
}

fn read_u64_array(ebpf: &Ebpf, name: &str) -> Result<Vec<u64>> {
    let map = ebpf
        .map(name)
        .with_context(|| format!("{name} map missing"))?;
    let arr: BpfArray<&aya::maps::MapData, u64> =
        BpfArray::try_from(map).with_context(|| format!("{name} is not an Array<u64>"))?;
    Ok(arr.iter().collect::<Result<Vec<_>, _>>()?)
}

/// Read the maps for the layers `opts` attached. Non-destructive: maps keep
/// counting, so call after the window of interest.
pub fn snapshot(ebpf: &Ebpf, opts: &ProbeOptions) -> Result<Snapshot> {
    let read = opts
        .syscalls
        .then(|| -> Result<ReadSnapshot> {
            Ok(ReadSnapshot {
                read_stats: read_u64_array(ebpf, "READ_STATS")?,
                read_hist: read_u64_array(ebpf, "READ_HIST")?,
                rdlat_hist: read_u64_array(ebpf, "RDLAT_HIST")?,
            })
        })
        .transpose()?;

    let mut pmu = Vec::new();
    if opts.pmu {
        let pmu_map: &Map = ebpf.map("PMU_STATS").context("PMU_STATS map missing")?;
        let pmu_map: BpfHashMap<&aya::maps::MapData, ContextKey, PmuStats> =
            BpfHashMap::try_from(pmu_map).context("PMU_STATS has unexpected key/value")?;
        for entry in pmu_map.iter() {
            let (key, stats) = entry.context("iterating PMU_STATS")?;
            pmu.push((key, stats));
        }
        pmu.sort_by(|a, b| b.1.cycles.cmp(&a.1.cycles));
    }

    let bio = opts
        .bio
        .then(|| -> Result<BioSnapshot> {
            Ok(BioSnapshot {
                stats: read_u64_array(ebpf, "BIO_STATS")?,
                lat_hist: read_u64_array(ebpf, "BIO_LAT_HIST")?,
            })
        })
        .transpose()?;

    let locks = opts
        .locks
        .then(|| -> Result<LocksSnapshot> {
            Ok(LocksSnapshot {
                stats: read_u64_array(ebpf, "FUTEX_STATS")?,
                wait_hist: read_u64_array(ebpf, "FUTEX_WAIT_HIST")?,
            })
        })
        .transpose()?;

    let net = opts
        .net
        .then(|| -> Result<NetSnapshot> {
            Ok(NetSnapshot {
                stats: read_u64_array(ebpf, "NET_STATS")?,
                connect_lat_hist: read_u64_array(ebpf, "CONNECT_LAT_HIST")?,
            })
        })
        .transpose()?;

    let offcpu = opts
        .offcpu
        .then(|| -> Result<OffCpuSnapshot> {
            let map: &Map = ebpf
                .map("OFFCPU_STATS")
                .context("OFFCPU_STATS map missing")?;
            let map: BpfHashMap<&aya::maps::MapData, ContextKey, OffCpuStats> =
                BpfHashMap::try_from(map).context("OFFCPU_STATS has unexpected key/value")?;
            let mut stats = Vec::new();
            for entry in map.iter() {
                let (key, s) = entry.context("iterating OFFCPU_STATS")?;
                stats.push((key, s));
            }
            stats.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
            Ok(OffCpuSnapshot {
                stats,
                runq_hist: read_u64_array(ebpf, "RUNQ_HIST")?,
            })
        })
        .transpose()?;

    Ok(Snapshot {
        read,
        pmu,
        bio,
        locks,
        net,
        offcpu,
    })
}
