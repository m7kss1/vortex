// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Query-engine adapters for `vx profile`. Today only DataFusion; the closed
//! [`QueryEngine`] enum is the extension point for future engines (e.g. DuckDB).
//! Vortex format instrumentation stays in the Vortex crates — an engine adapter
//! only drives the query and reports its own engine-internal introspection.

use serde_json::Map;
use serde_json::Value;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;

use crate::datafusion_helper::execute_vortex_query;

/// Inputs an engine needs to run one profiled query.
pub(super) struct EngineRunInput<'a> {
    /// The profiled session. A new engine adapter **must** thread this into the
    /// Vortex scan path, or `scan_profiler()` stays `None` and no format metrics
    /// are recorded.
    pub(super) session: &'a VortexSession,
    pub(super) file_path: &'a str,
    pub(super) sql: &'a str,
}

/// Engine-specific profiling output. `details` is rendered under the report's
/// `engine_details` object and is **not** compared by `vx profile diff`, so the
/// flat `metrics` map stays purely Vortex-format.
pub(super) struct EngineRunOutput {
    pub(super) engine_name: &'static str,
    pub(super) details: Map<String, Value>,
}

impl EngineRunOutput {
    /// An output with no engine-specific details (the DataFusion case today).
    pub(super) fn empty(engine_name: &'static str) -> Self {
        Self {
            engine_name,
            details: Map::new(),
        }
    }
}

/// The query engines `vx profile` can drive. A closed set so adding an engine
/// (e.g. DuckDB) is a new variant the compiler forces every `match` to handle.
pub(super) enum QueryEngine {
    DataFusion,
}

impl QueryEngine {
    /// Stable engine name recorded in the report.
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::DataFusion => "datafusion",
        }
    }

    /// Run `input`'s query on this engine.
    ///
    /// Contract for future engines: thread `input.session` into the Vortex scan
    /// path so the installed profiler observes the scan. DuckDB currently uses a
    /// global `LazyLock<VortexSession>`, so wiring a per-run profiled session is
    /// non-trivial follow-up work, not a drop-in.
    pub(super) async fn run(&self, input: EngineRunInput<'_>) -> VortexResult<EngineRunOutput> {
        match self {
            Self::DataFusion => {
                execute_vortex_query(input.session, input.file_path, input.sql)
                    .await
                    .map_err(|e| vortex_err!("{e}"))?;
                Ok(EngineRunOutput::empty(self.name()))
            }
        }
    }
}
