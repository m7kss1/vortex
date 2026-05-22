// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::root::Root;

impl ArrayRef {
    /// Apply the expression to this array, producing a new lazy [`ScalarFnArray`] tree.
    ///
    /// # Prefer [`ArrayRefLeeExt::execute_expr`]
    ///
    /// This method returns a lazily-evaluated array tree. Callers that simply want to evaluate
    /// an expression should use the session-aware LEE entry points instead:
    ///
    /// ```rust,ignore
    /// use vortex_array::lee::ArrayRefLeeExt as _;
    /// let result = array.execute_expr(&expr, &session)?;
    /// ```
    ///
    /// `apply` is kept for internal code paths that need to inspect the array tree before
    /// evaluation (e.g. `substitute_row_count` for zone-map pruning).
    #[doc(hidden)]
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        // If the expression is a root, return self.
        if expr.is::<Root>() {
            return Ok(self);
        }

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply(e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let array =
            ScalarFnArray::try_new(expr.scalar_fn().clone(), children, self.len())?.into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }
}
