// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiled program produced by [`super::compile`] and consumed by [`super::execute_program`].

use std::sync::Arc;

use super::Opcode;
use super::RegId;
use crate::dtype::DType;

/// A compiled, linear representation of an [`crate::expr::Expression`].
///
/// Built once per (expression, schema) pair and executed many times (once per batch). The
/// executor walks `opcodes` linearly, dispatching each through a handler table indexed by the
/// opcode's tag (LEE-style threaded dispatch).
///
/// ## Caching
///
/// Programs compiled via the Phase A/B encoding-aware path may contain
/// [`super::Opcode::LoadCapture`] opcodes that embed arrays extracted from the compile-time
/// scope. Such programs have `cacheable = false` and must not be shared across batches.
/// Programs without captures (`cacheable = true`) are safe to share via [`super::ProgramCache`].
#[derive(Clone, Debug)]
pub struct ExprProgram {
    /// Flat list of opcodes executed linearly by the dispatcher.
    pub opcodes: Vec<Opcode>,
    /// Number of registers the executor must allocate before running.
    pub num_regs: u16,
    /// The register holding the program's final result. The `Return` opcode references it.
    pub result_reg: RegId,
    /// dtype of the scope this program expects as input. Kept for future type-checking use.
    pub scope_dtype: Arc<DType>,
    /// Whether this program is safe to cache and reuse across different batches with the same
    /// encoding structure. Programs containing [`super::Opcode::LoadCapture`] opcodes embed
    /// batch-specific arrays and must not be cached.
    pub cacheable: bool,
}
