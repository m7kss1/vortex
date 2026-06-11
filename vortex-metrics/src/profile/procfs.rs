// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cold/warm-cache and memory readings from procfs — free, exact, no eBPF.
//!
//! Take a [`ProcReading`] before and after the profiled window and subtract:
//! `read_bytes` delta is bytes actually fetched from storage (cold bytes),
//! `rchar` delta is logical read bytes (so `1 - read_bytes/rchar` approximates
//! the page-cache hit ratio), `majflt` delta counts major faults, and `vm_hwm`
//! is the peak-RSS high-water mark (compare to the before-reading's current
//! RSS for the workload's contribution).

use std::fs;

use anyhow::Context;
use anyhow::Result;

/// A point-in-time reading of this process's IO and memory counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcReading {
    /// Bytes fetched from the storage layer (`/proc/self/io` `read_bytes`).
    pub read_bytes: u64,
    /// Logical bytes read by syscalls (`/proc/self/io` `rchar`).
    pub rchar: u64,
    /// Major page faults so far (`/proc/self/stat` field 12).
    pub majflt: u64,
    /// Peak resident set size in bytes (`/proc/self/status` `VmHWM`).
    pub vm_hwm: u64,
    /// Current resident set size in bytes (`/proc/self/status` `VmRSS`).
    pub vm_rss: u64,
}

fn field_after(haystack: &str, key: &str) -> Option<u64> {
    let line = haystack.lines().find(|l| l.starts_with(key))?;
    line[key.len()..]
        .trim_start_matches(':')
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Read the current [`ProcReading`] for this process.
pub fn read_self() -> Result<ProcReading> {
    let io = fs::read_to_string("/proc/self/io").context("reading /proc/self/io")?;
    let status = fs::read_to_string("/proc/self/status").context("reading /proc/self/status")?;
    let stat = fs::read_to_string("/proc/self/stat").context("reading /proc/self/stat")?;

    // stat: fields after the parenthesised comm; majflt is field 12 (1-based).
    let after_comm = stat
        .rsplit_once(") ")
        .map(|(_, rest)| rest)
        .context("malformed /proc/self/stat")?;
    let majflt = after_comm
        .split_whitespace()
        .nth(9) // field 12 overall; 10th token after the comm field
        .and_then(|v| v.parse().ok())
        .context("majflt missing from /proc/self/stat")?;

    Ok(ProcReading {
        read_bytes: field_after(&io, "read_bytes").context("read_bytes missing")?,
        rchar: field_after(&io, "rchar").context("rchar missing")?,
        majflt,
        vm_hwm: field_after(&status, "VmHWM").context("VmHWM missing")? * 1024,
        vm_rss: field_after(&status, "VmRSS").context("VmRSS missing")? * 1024,
    })
}

#[cfg(test)]
mod tests {
    use super::read_self;

    #[test]
    fn read_self_is_sane() -> anyhow::Result<()> {
        let r = read_self()?;
        assert!(r.vm_rss > 0);
        assert!(r.vm_hwm >= r.vm_rss);
        assert!(r.rchar > 0);
        Ok(())
    }
}
