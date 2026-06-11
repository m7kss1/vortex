// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic-counter diff gate over two profile reports.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;

/// Counters where an *increase* beyond tolerance is a regression.
pub const HARD_COUNTERS_UP: &[&str] = &[
    "io.read_syscalls",
    "io.read_syscall_bytes",
    "io.tiny_reads",
    "io.read_amplification",
    "io.physical_reads",
    "io.physical_read_bytes",
    "io.segment_requests",
    "metadata.footer_reads",
    "metadata.footer_bytes",
    "decode.total_calls",
    "pushdown_fallback.total",
];

/// Counters where a *decrease* beyond tolerance is a regression (less work
/// skipped than before).
pub const HARD_COUNTERS_DOWN: &[&str] = &["pruning.pruned_ratio"];

/// Correctness counters: *any* change beyond tolerance is a regression.
/// These must be identical across runs on the same query+data.
pub const HARD_COUNTERS_EXACT: &[&str] = &["scan.rows_out", "filter.rows_kept"];

/// One compared counter in a [`DiffOutcome`].
#[derive(Debug, Clone)]
pub struct DiffRow {
    /// Counter name.
    pub name: String,
    /// Baseline value.
    pub baseline: f64,
    /// Candidate value.
    pub candidate: f64,
    /// Fractional change, positive = increase.
    pub delta: f64,
    /// Whether this row regressed beyond tolerance.
    pub regressed: bool,
}

/// Result of comparing two reports.
#[derive(Debug, Clone, Default)]
pub struct DiffOutcome {
    /// All compared counters present in both reports.
    pub rows: Vec<DiffRow>,
}

impl DiffOutcome {
    /// Whether any hard counter regressed.
    pub fn regressed(&self) -> bool {
        self.rows.iter().any(|r| r.regressed)
    }
}

/// Load the flat `metrics` map out of a saved nested JSON profile report.
pub fn load_report(path: impl AsRef<Path>) -> Result<BTreeMap<String, f64>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text).context("parsing report")?;
    let metrics = value
        .get("metrics")
        .and_then(|m| m.as_object())
        .context("report has no `metrics` object")?;
    Ok(metrics
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v)))
        .collect())
}

/// Compare two loaded reports against the hard-counter gate.
pub fn diff(
    base: &BTreeMap<String, f64>,
    cand: &BTreeMap<String, f64>,
    tolerance: f64,
) -> DiffOutcome {
    let mut rows = Vec::new();
    let mut push = |name: &str, increase_bad: bool, decrease_bad: bool| {
        let (Some(&b), Some(&c)) = (base.get(name), cand.get(name)) else {
            return;
        };
        let delta = if b != 0.0 {
            (c - b) / b
        } else if c == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        let regressed = (increase_bad && delta > tolerance) || (decrease_bad && delta < -tolerance);
        rows.push(DiffRow {
            name: name.to_string(),
            baseline: b,
            candidate: c,
            delta,
            regressed,
        });
    };
    for name in HARD_COUNTERS_UP {
        push(name, true, false);
    }
    for name in HARD_COUNTERS_DOWN {
        push(name, false, true);
    }
    for name in HARD_COUNTERS_EXACT {
        push(name, true, true);
    }
    DiffOutcome { rows }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::diff;

    fn report(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn gates_on_direction() {
        let base = report(&[
            ("decode.total_calls", 100.0),
            ("pruning.pruned_ratio", 0.6),
            ("scan.rows_out", 1000.0),
        ]);
        let more_calls = report(&[
            ("decode.total_calls", 120.0),
            ("pruning.pruned_ratio", 0.6),
            ("scan.rows_out", 1000.0),
        ]);
        assert!(diff(&base, &more_calls, 0.05).regressed());
        let less_pruning = report(&[
            ("decode.total_calls", 100.0),
            ("pruning.pruned_ratio", 0.4),
            ("scan.rows_out", 1000.0),
        ]);
        assert!(diff(&base, &less_pruning, 0.05).regressed());
        let better = report(&[
            ("decode.total_calls", 90.0),
            ("pruning.pruned_ratio", 0.7),
            ("scan.rows_out", 1000.0),
        ]);
        assert!(!diff(&base, &better, 0.05).regressed());
    }
}
