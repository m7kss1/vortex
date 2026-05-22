// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped [`ProgramCache`] for compiled [`ExprProgram`]s.
//!
//! The cache is registered in the [`VortexSession`] as a [`SessionVar`] and keyed by
//! `(expression, scope dtype, scope encoding)`. Two scope arrays with the same dtype but
//! different top-level encodings get distinct cache entries because encoding-aware Phase B
//! rules may produce different opcode streams.

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_error::VortexResult;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::ArrayRef;
use crate::array::ArrayId;
use crate::dtype::DType;
use crate::expr::ExactExpr;
use crate::expr::Expression;
use crate::lee::ExprProgram;
use crate::lee::compile;

/// Session-scoped cache for compiled [`ExprProgram`]s.
///
/// Add to a session with `session.with::<ProgramCache>()`. Programs are compiled once
/// per `(expression, scope dtype, scope encoding)` triple and shared across threads.
///
/// # Thread safety
///
/// `ProgramCache` is `Send + Sync`: the inner [`DashMap`] is concurrency-safe and the
/// [`OnceLock`] per entry ensures compile happens exactly once even under contention.
#[derive(Default)]
pub struct ProgramCache {
    inner: DashMap<CacheKey, Arc<OnceLock<Arc<ExprProgram>>>>,
}

impl Debug for ProgramCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramCache")
            .field("entries", &self.inner.len())
            .finish()
    }
}

impl SessionVar for ProgramCache {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ProgramCache {
    /// Look up or compile a program for `(expr, scope)`.
    ///
    /// If no program has been compiled for this `(expression, scope dtype, scope encoding)`
    /// triple, compiles one and stores it. Returns a shared [`Arc`] to the compiled program.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`compile`].
    pub fn get_or_compile(
        &self,
        expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<Arc<ExprProgram>> {
        let key = CacheKey {
            expr: ExactExpr(expr.clone()),
            scope_dtype: scope.dtype().clone(),
            scope_encoding: scope.encoding_id(),
        };

        // Fast path: read-only cache probe. `DashMap::get` takes a shared lock on the shard and
        // does not insert, so cacheable programs already compiled by another thread are returned
        // immediately without taking the write lock.
        if let Some(cell) = self.inner.get(&key)
            && let Some(program) = cell.get()
        {
            return Ok(Arc::clone(program));
        }

        // Cache miss (or non-cacheable): compile now.
        let compiled = Arc::new(compile(expr, scope)?);

        // Programs containing `LoadCapture` opcodes embed batch-specific arrays extracted by the
        // encoding optimizer (e.g. dict values, dict codes). They are valid only for the scope
        // they were compiled from; do not insert them into the cache.
        if !compiled.cacheable {
            return Ok(compiled);
        }

        // Insert into the cache. `OnceLock::set` is a no-op if another thread raced and set it
        // first; `cell.get()` then returns whichever value won.
        let cell = self
            .inner
            .entry(key)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone();
        drop(cell.set(Arc::clone(&compiled)));
        Ok(cell.get().cloned().unwrap_or(compiled))
    }
}

/// Extension trait for accessing [`ProgramCache`] from a [`VortexSession`].
pub trait ProgramCacheSessionExt {
    /// Get or compile a program, using the session-scoped [`ProgramCache`].
    fn compile_expr(&self, expr: &Expression, scope: &ArrayRef) -> VortexResult<Arc<ExprProgram>>;
}

impl ProgramCacheSessionExt for VortexSession {
    fn compile_expr(&self, expr: &Expression, scope: &ArrayRef) -> VortexResult<Arc<ExprProgram>> {
        self.get::<ProgramCache>().get_or_compile(expr, scope)
    }
}

/// Cache key: expression identity + scope encoding fingerprint.
#[derive(Eq, PartialEq, Hash)]
struct CacheKey {
    expr: ExactExpr,
    scope_dtype: DType,
    scope_encoding: ArrayId,
}
