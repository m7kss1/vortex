// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension trait providing high-level LEE entry points on [`ArrayRef`].

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::VortexSessionExecute as _;
use crate::expr::Expression;
use crate::lee::cache::ProgramCacheSessionExt as _;
use crate::lee::execute_mask_program;
use crate::lee::execute_program;

/// High-level expression evaluation methods on [`ArrayRef`].
///
/// All methods compile the expression once (cached by [`crate::lee::ProgramCache`]) and
/// execute it against `self` as the scope array.
pub trait ArrayRefLeeExt {
    /// Evaluate `expr` over `self` and return the result array.
    ///
    /// Compiles `expr` against `self` (using the session-scoped [`crate::lee::ProgramCache`]),
    /// then runs the resulting [`ExprProgram`] and returns the output array.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or execution fails.
    fn execute_expr(&self, expr: &Expression, session: &VortexSession) -> VortexResult<ArrayRef>;

    /// Evaluate `expr` over `self` and return a boolean [`Mask`].
    ///
    /// Equivalent to `execute_expr_mask_with_input` with an all-true input mask. Use this when
    /// every row should be evaluated.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or execution fails.
    fn execute_expr_mask(
        &self,
        expr: &Expression,
        session: &VortexSession,
    ) -> VortexResult<Mask>;

    /// Evaluate `expr` over `self`, intersecting the result with `input_mask`.
    ///
    /// Only rows that are `true` in `input_mask` are considered; the output mask has the same
    /// length as `input_mask`. This is the preferred form when prior pruning has already
    /// reduced the live-row set.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or execution fails.
    fn execute_expr_mask_with_input(
        &self,
        expr: &Expression,
        input_mask: &Mask,
        session: &VortexSession,
    ) -> VortexResult<Mask>;
}

impl ArrayRefLeeExt for ArrayRef {
    fn execute_expr(&self, expr: &Expression, session: &VortexSession) -> VortexResult<ArrayRef> {
        let program = session.compile_expr(expr, self)?;
        let mut ctx = session.create_execution_ctx();
        execute_program(&program, self, &mut ctx)
    }

    fn execute_expr_mask(
        &self,
        expr: &Expression,
        session: &VortexSession,
    ) -> VortexResult<Mask> {
        let input_mask = Mask::new_true(self.len());
        self.execute_expr_mask_with_input(expr, &input_mask, session)
    }

    fn execute_expr_mask_with_input(
        &self,
        expr: &Expression,
        input_mask: &Mask,
        session: &VortexSession,
    ) -> VortexResult<Mask> {
        let program = session.compile_expr(expr, self)?;
        let mut ctx = session.create_execution_ctx();
        execute_mask_program(&program, self, input_mask, &mut ctx)
    }
}

