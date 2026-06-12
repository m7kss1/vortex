// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic-counter diff gate over two profile reports.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
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

/// Whether `key` is a dynamic per-encoding / per-pair counter gated as
/// increase-bad. The fixed [`HARD_COUNTERS_UP`] totals (`decode.total_calls`,
/// `pushdown_fallback.total`) can stay flat while work shifts between encodings,
/// so we additionally gate every `decode.<encoding>.calls` and
/// `pushdown_fallback.<parent>=><child>.count` present in both reports. This is
/// what surfaces a pushdown regression that forces extra decodes of one
/// encoding without changing the grand total.
pub fn is_dynamic_hard_counter(key: &str) -> bool {
    (key.starts_with("decode.") && key.ends_with(".calls") && key != "decode.total_calls")
        || (key.starts_with("pushdown_fallback.") && key.ends_with(".count"))
}

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
    let mut seen = BTreeSet::new();
    let mut push = |name: &str, increase_bad: bool, decrease_bad: bool, tol: f64| {
        if !seen.insert(name.to_string()) {
            return;
        }
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
        let regressed = (increase_bad && delta > tol) || (decrease_bad && delta < -tol);
        rows.push(DiffRow {
            name: name.to_string(),
            baseline: b,
            candidate: c,
            delta,
            regressed,
        });
    };
    for name in HARD_COUNTERS_UP {
        push(name, true, false, tolerance);
    }
    for name in HARD_COUNTERS_DOWN {
        push(name, false, true, tolerance);
    }
    // Correctness counters must stay identical across runs on the same query+data,
    // so any change is a regression regardless of the perf tolerance.
    for name in HARD_COUNTERS_EXACT {
        push(name, true, true, 0.0);
    }
    // Dynamic per-encoding / per-pair counters present in both reports.
    let dynamic: Vec<String> = base
        .keys()
        .filter(|k| cand.contains_key(*k) && is_dynamic_hard_counter(k.as_str()))
        .cloned()
        .collect();
    for name in &dynamic {
        push(name, true, false, tolerance);
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

    #[test]
    fn exact_counters_ignore_tolerance() {
        let base = report(&[("scan.rows_out", 1000.0), ("filter.rows_kept", 500.0)]);
        // A 0.4% drift — well under the 5% tolerance — must still regress, because
        // these counters must be identical for the same query+data.
        let drifted = report(&[("scan.rows_out", 1004.0), ("filter.rows_kept", 500.0)]);
        assert!(diff(&base, &drifted, 0.05).regressed());
        // Identical values do not regress.
        assert!(!diff(&base, &base, 0.05).regressed());
    }

    #[test]
    fn gates_per_encoding_decode_and_fallback() {
        // The total is unchanged but work shifted onto pco: the per-encoding
        // counter and the per-pair fallback must still flag the regression.
        let base = report(&[
            ("decode.total_calls", 100.0),
            ("decode.vortex.pco.calls", 20.0),
            ("pushdown_fallback.vortex.filter=>vortex.pco.count", 5.0),
        ]);
        let cand = report(&[
            ("decode.total_calls", 100.0),
            ("decode.vortex.pco.calls", 40.0),
            ("pushdown_fallback.vortex.filter=>vortex.pco.count", 11.0),
        ]);
        let outcome = diff(&base, &cand, 0.05);
        assert!(outcome.regressed());
        assert!(
            outcome
                .rows
                .iter()
                .any(|r| r.name == "decode.vortex.pco.calls" && r.regressed)
        );
        assert!(
            outcome.rows.iter().any(|r| r.name
                == "pushdown_fallback.vortex.filter=>vortex.pco.count"
                && r.regressed)
        );
        // The fixed total is gated exactly once, not duplicated by the pattern.
        assert_eq!(
            outcome
                .rows
                .iter()
                .filter(|r| r.name == "decode.total_calls")
                .count(),
            1
        );
    }
}
