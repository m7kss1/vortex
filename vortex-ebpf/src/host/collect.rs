// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading the eBPF maps after a profiled run. Only the kernel-side facts live
//! here — read-syscall sizes and per-encoding PMU samples; all format-semantic
//! metrics come from the in-process `vortex-metrics` registry instead.

use anyhow::Context;
use anyhow::Result;
use aya::Ebpf;
use aya::maps::Array as BpfArray;
use aya::maps::HashMap as BpfHashMap;
use aya::maps::Map;

use crate::types::EncKey;
use crate::types::PmuStats;

/// Everything read out of the eBPF maps at the end of a run.
#[derive(Debug, Default)]
pub struct Snapshot {
    /// `[count, bytes]` of read syscalls (see `READ_STAT_*` indices).
    pub read_stats: Vec<u64>,
    /// Log2 histogram of read-syscall sizes.
    pub read_hist: Vec<u64>,
    /// Per-encoding PMU sample counts, sorted by cycle samples descending.
    pub pmu: Vec<(String, PmuStats)>,
}

/// Render the display name of an [`EncKey`]: its first `len` bytes.
fn enc_display(key: &EncKey, len: u32) -> String {
    let end = (len as usize).min(key.0.len());
    String::from_utf8_lossy(&key.0[..end]).into_owned()
}

fn read_u64_array(ebpf: &Ebpf, name: &str) -> Result<Vec<u64>> {
    let map = ebpf
        .map(name)
        .with_context(|| format!("{name} map missing"))?;
    let arr: BpfArray<&aya::maps::MapData, u64> =
        BpfArray::try_from(map).with_context(|| format!("{name} is not an Array<u64>"))?;
    Ok(arr.iter().collect::<Result<Vec<_>, _>>()?)
}

/// Read all maps. Non-destructive: maps keep counting, so call after the window
/// of interest.
pub fn snapshot(ebpf: &Ebpf) -> Result<Snapshot> {
    let pmu_map: &Map = ebpf.map("PMU_STATS").context("PMU_STATS map missing")?;
    let pmu_map: BpfHashMap<&aya::maps::MapData, EncKey, PmuStats> =
        BpfHashMap::try_from(pmu_map).context("PMU_STATS has unexpected key/value")?;
    let mut pmu = Vec::new();
    for entry in pmu_map.iter() {
        let (key, stats) = entry.context("iterating PMU_STATS")?;
        pmu.push((enc_display(&key, stats.enc_len), stats));
    }
    pmu.sort_by(|a, b| b.1.cycles.cmp(&a.1.cycles));

    Ok(Snapshot {
        read_stats: read_u64_array(ebpf, "READ_STATS")?,
        read_hist: read_u64_array(ebpf, "READ_HIST")?,
        pmu,
    })
}
