// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compile an [`Expression`] into a linear [`ExprProgram`].
//!
//! ## Overview
//!
//! Compilation runs in two stages:
//!
//! **Phase A + B — encoding-aware tree construction** (`scope.apply(expr)`):
//!
//! 1. [`crate::ArrayRef::apply`] walks the expression and builds a lazy [`crate::arrays::ScalarFnArray`]
//!    tree in which each `Root` node is replaced by the live scope array and each `Literal` by a
//!    [`crate::arrays::ConstantArray`].
//! 2. At each tree node [`crate::optimizer::ArrayOptimizer::optimize`] fires the encoding-specific
//!    `reduce` / `reduce_parent` rules registered on vtables (e.g.
//!    `DictionaryScalarFnValuesPushDownRule`, `StructGetItemRule`, …). This resolves struct field
//!    accesses to their concrete column arrays and, for dictionary-encoded columns, rewrites a
//!    scalar function applied to the full column into a `take(fn(dict_values), codes)` pair — the
//!    headline encoding-pushdown win.
//!
//! **Phase C — lower to opcodes** ([`lower_tree`]):
//!
//! Walks the optimized `ArrayRef` tree and emits flat opcodes:
//!
//! | Tree node | Opcode(s) emitted |
//! |---|---|
//! | The scope array itself (`Arc::ptr_eq`) | `LoadScope` (register shared) |
//! | [`crate::arrays::ConstantArray`] with `len == scope.len()` | `LoadConst` (cacheable) |
//! | `ConstantArray` with `len != scope.len()` | `LoadCapture` (per-batch capture) |
//! | Non-nullable bool `AND` chain | `AllocBool(true)` + N×`AndInto` |
//! | Non-nullable bool `OR` chain | `AllocBool(false)` + N×`OrInto` |
//! | Non-nullable bool `NOT` | `AllocBool(false)` + `OrInto` + `NotInto` |
//! | `CaseWhen` with ELSE | `lower(ELSE)` + reverse(`CaseMerge`) per pair |
//! | `ScalarFnArray` (generic) | recurse children → `Call` |
//! | Any other `ArrayRef` (sub-array extracted by optimizer) | `LoadCapture` |
//!
//! Programs that emit at least one `LoadCapture` set `cacheable = false` and are recompiled
//! for each batch. Programs without captures (e.g. plain primitive columns) remain `cacheable =
//! true` and are shared via [`super::ProgramCache`] across all batches of the same schema.
//!
//! ## Register allocation
//!
//! `CompileCtx::alloc_reg()` issues a new register id and `free_reg()` returns it to a free list
//! so that ids are recycled across AND/OR leaves and CASE WHEN branches, keeping `num_regs` small.
//! `scope_reg` (for `LoadScope`) and `capture_reg_map` (for `LoadCapture`) deduplicate repeated
//! references to the same underlying array, avoiding redundant loads.

use std::collections::HashMap;
use std::sync::Arc;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::ExprProgram;
use super::Opcode;
use super::RegId;
use crate::ArrayRef;
use crate::arrays::Constant;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::case_when::CaseWhen;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::operators::Operator;

/// Compile an expression into an executable [`ExprProgram`] for the given scope array.
///
/// Internally this runs the full Phase A+B+C pipeline:
/// 1. **Phase A** — calls [`crate::ArrayRef::apply`] to build a `ScalarFnArray` tree with the
///    scope bound at every `Root` node.
/// 2. **Phase B** — the encoding-aware `ArrayOptimizer` rules already fire inside `apply()` at
///    each node, pushing scalar functions through dict / runend / struct encodings.
/// 3. **Phase C** — [`lower_tree`] walks the optimised tree and emits a flat `Vec<Opcode>`.
///
/// The returned program's [`ExprProgram::cacheable`] flag indicates whether it is safe to
/// share across batches. Programs that contain [`Opcode::LoadCapture`] opcodes (i.e. those where
/// the optimizer extracted sub-arrays from the scope) must be recompiled per batch.
///
/// # Arguments
///
/// * `expr` — the expression to compile
/// * `scope` — the array this program will be executed against; its encoding structure drives
///   the encoding-aware optimisation rules.
///
/// # Errors
///
/// Returns an error if `apply()` fails (e.g. type mismatch), if `optimize()` fails, or if the
/// register count overflows `u16` (more than 65 535 live registers — practically unreachable).
pub fn compile(expr: &Expression, scope: &ArrayRef) -> VortexResult<ExprProgram> {
    // Phase A + B: build the encoding-optimised ScalarFnArray tree.
    let tree = scope.clone().apply(expr)?;

    // Phase C: lower the optimised tree to a flat opcode list.
    let scope_dtype = scope.dtype();
    let mut ctx = CompileCtx {
        opcodes: Vec::new(),
        next_reg: 0,
        scope_dtype,
        scope: scope.clone(),
        scope_reg: None,
        free_list: SmallVec::new(),
        capture_map: HashMap::new(),
        has_captures: false,
    };
    let result_reg = lower_tree(&tree, &mut ctx)?;
    ctx.opcodes.push(Opcode::Return { src: result_reg });
    Ok(ExprProgram {
        opcodes: ctx.opcodes,
        num_regs: ctx.next_reg,
        result_reg,
        scope_dtype: Arc::new(scope_dtype.clone()),
        cacheable: !ctx.has_captures,
    })
}

struct CompileCtx<'a> {
    opcodes: Vec<Opcode>,
    /// Next register id to allocate when the free list is empty.
    next_reg: u16,
    scope_dtype: &'a DType,
    /// The compile-time scope array, used for pointer-equality checks and capture dedup.
    scope: ArrayRef,
    /// Shared register for the single `LoadScope` opcode. All occurrences of the scope array
    /// in the tree reuse this register without emitting a second `LoadScope`.
    scope_reg: Option<RegId>,
    /// Dead registers available for reuse. `alloc_reg` pops from here before bumping `next_reg`,
    /// so register ids are recycled across AND/OR leaf evaluations and CASE WHEN branches.
    free_list: SmallVec<[RegId; 8]>,
    /// Deduplication map for `LoadCapture` opcodes. Maps the array's pointer address to the
    /// register already holding it, so the same sub-array (e.g. dict_values referenced by two
    /// conjuncts) is loaded only once.
    capture_map: HashMap<usize, RegId>,
    /// Set to `true` the first time a `LoadCapture` opcode is emitted. Causes the resulting
    /// `ExprProgram` to be marked non-cacheable.
    has_captures: bool,
}

impl CompileCtx<'_> {
    /// Allocate the next available register, recycling dead registers first.
    fn alloc_reg(&mut self) -> VortexResult<RegId> {
        if let Some(id) = self.free_list.pop() {
            return Ok(id);
        }
        let Some(next) = self.next_reg.checked_add(1) else {
            vortex_bail!("lee: too many registers in a single program (overflow at u16)");
        };
        let id = RegId::new(self.next_reg);
        self.next_reg = next;
        Ok(id)
    }

    /// Mark a register as dead and eligible for reuse.
    ///
    /// The scope register is never freed — it must remain valid throughout the program because
    /// it is read by every `Call` opcode that accesses a scope column.
    fn free_reg(&mut self, id: RegId) {
        if Some(id) == self.scope_reg {
            return;
        }
        self.free_list.push(id);
    }
}

/// True iff `array` has type `DType::Bool(NonNullable)`.
fn is_non_nullable_bool_array(array: &ArrayRef) -> bool {
    matches!(array.dtype(), DType::Bool(Nullability::NonNullable))
}

/// Emit `LoadCapture` for `array` (or reuse an existing capture register), marking the program
/// as non-cacheable.
fn emit_capture(array: &ArrayRef, ctx: &mut CompileCtx<'_>) -> VortexResult<RegId> {
    let ptr = array.addr();
    if let Some(&existing) = ctx.capture_map.get(&ptr) {
        return Ok(existing);
    }
    let dst = ctx.alloc_reg()?;
    ctx.opcodes.push(Opcode::LoadCapture {
        dst,
        array: array.clone(),
    });
    ctx.has_captures = true;
    ctx.capture_map.insert(ptr, dst);
    Ok(dst)
}

/// Recursively lower an optimised `ArrayRef` tree, emitting opcodes into `ctx` and returning
/// the register that will hold the array's value at runtime.
///
/// The tree was produced by [`crate::ArrayRef::apply`] + encoding-aware optimisation. Leaves
/// are either the scope array itself, `ConstantArray` nodes (literals), `ScalarFnArray` nodes
/// (unevaluated scalar functions), or raw sub-arrays extracted by the optimizer (e.g.
/// `dict_values`, `dict_codes`).
fn lower_tree(array: &ArrayRef, ctx: &mut CompileCtx<'_>) -> VortexResult<RegId> {
    // ── 1. Scope identity: the scope array appears directly as a leaf when the expression is
    //        `root()` or when a scalar function reduces to the scope unchanged.
    if ArrayRef::ptr_eq(array, &ctx.scope.clone()) {
        if let Some(existing) = ctx.scope_reg {
            return Ok(existing);
        }
        let dst = ctx.alloc_reg()?;
        ctx.opcodes.push(Opcode::LoadScope { dst });
        ctx.scope_reg = Some(dst);
        return Ok(dst);
    }

    // ── 2. ConstantArray: a literal scalar, possibly broadcast to an arbitrary length.
    //        If it has the same length as the scope we can use `LoadConst` (the handler
    //        re-creates the array at the scope's runtime length). Otherwise the length is
    //        encoding-specific (e.g. dict n_values) and we must capture it as-is.
    if let Some(const_array) = array.as_opt::<Constant>() {
        let dst = ctx.alloc_reg()?;
        if array.len() == ctx.scope.len() {
            ctx.opcodes.push(Opcode::LoadConst {
                dst,
                scalar: const_array.scalar().clone(),
            });
        } else {
            // Length differs from scope (e.g. after dict-values pushdown). Capture the
            // pre-sized array directly; `h_load_capture` will hand it to `Call` unchanged.
            ctx.opcodes.push(Opcode::LoadCapture {
                dst,
                array: array.clone(),
            });
            ctx.has_captures = true;
            ctx.capture_map.insert(array.addr(), dst);
        }
        return Ok(dst);
    }

    // ── 3. ScalarFnArray: an unevaluated scalar function node with zero or more children.
    //        Detect special patterns (AND/OR chains, NOT, CaseWhen) before the generic Call path.
    if let Some(sfn) = array.as_opt::<ScalarFn>() {
        let scalar_fn = sfn.scalar_fn().clone();

        // AND chain over non-nullable bools → AllocBool(true) + N×AndInto.
        if matches!(scalar_fn.as_opt::<Binary>(), Some(Operator::And)) {
            let mut leaves: SmallVec<[ArrayRef; 8]> = SmallVec::new();
            collect_and_leaves_tree(array, &mut leaves);
            if !leaves.is_empty() {
                let dst = ctx.alloc_reg()?;
                ctx.opcodes.push(Opcode::AllocBool { dst, init: true });
                for leaf in &leaves {
                    let leaf_reg = lower_tree(leaf, ctx)?;
                    ctx.opcodes.push(Opcode::AndInto { dst, src: leaf_reg });
                    ctx.free_reg(leaf_reg);
                }
                return Ok(dst);
            }
        }

        // OR chain over non-nullable bools → AllocBool(false) + N×OrInto.
        if matches!(scalar_fn.as_opt::<Binary>(), Some(Operator::Or)) {
            let mut leaves: SmallVec<[ArrayRef; 8]> = SmallVec::new();
            collect_or_leaves_tree(array, &mut leaves);
            if !leaves.is_empty() {
                let dst = ctx.alloc_reg()?;
                ctx.opcodes.push(Opcode::AllocBool { dst, init: false });
                for leaf in &leaves {
                    let leaf_reg = lower_tree(leaf, ctx)?;
                    ctx.opcodes.push(Opcode::OrInto { dst, src: leaf_reg });
                    ctx.free_reg(leaf_reg);
                }
                return Ok(dst);
            }
        }

        // NOT over a non-nullable bool child.
        if scalar_fn.is::<Not>()
            && sfn.nchildren() == 1
            && is_non_nullable_bool_array(sfn.child_at(0))
        {
            let child_reg = lower_tree(sfn.child_at(0), ctx)?;
            let dst = ctx.alloc_reg()?;
            ctx.opcodes.push(Opcode::AllocBool { dst, init: false });
            ctx.opcodes.push(Opcode::OrInto {
                dst,
                src: child_reg,
            });
            ctx.free_reg(child_reg);
            ctx.opcodes.push(Opcode::NotInto { reg: dst });
            return Ok(dst);
        }

        // CaseWhen with an explicit ELSE clause → CaseMerge opcodes (reverse pair order).
        if let Some(opts) = scalar_fn.as_opt::<CaseWhen>() {
            let num_pairs = opts.num_when_then_pairs as usize;
            let has_else = opts.has_else;

            let dst = if has_else {
                lower_tree(sfn.child_at(num_pairs * 2), ctx)?
            } else {
                let result_dtype = array.dtype().clone();
                let d = ctx.alloc_reg()?;
                ctx.opcodes.push(Opcode::LoadConst {
                    dst: d,
                    scalar: crate::scalar::Scalar::null(result_dtype),
                });
                d
            };

            for i in (0..num_pairs).rev() {
                let cond_reg = lower_tree(sfn.child_at(i * 2), ctx)?;
                let value_reg = lower_tree(sfn.child_at(i * 2 + 1), ctx)?;
                ctx.opcodes.push(Opcode::CaseMerge {
                    dst,
                    value: value_reg,
                    cond: cond_reg,
                });
                ctx.free_reg(value_reg);
                ctx.free_reg(cond_reg);
            }
            return Ok(dst);
        }

        // Generic ScalarFnArray: lower children left-to-right, emit a Call opcode.
        let mut arg_regs: SmallVec<[RegId; 4]> = SmallVec::new();
        for i in 0..sfn.nchildren() {
            arg_regs.push(lower_tree(sfn.child_at(i), ctx)?);
        }
        let dst = ctx.alloc_reg()?;
        ctx.opcodes.push(Opcode::Call {
            dst,
            scalar_fn,
            args: arg_regs,
        });
        return Ok(dst);
    }

    // ── 4. Any other array: a sub-array extracted from the scope by the encoding optimizer
    //        (e.g. `dict_values`, `dict_codes`, or a fully-evaluated sub-expression). Capture
    //        it by pointer, deduplicating if the same array appears in multiple conjuncts.
    emit_capture(array, ctx)
}

/// Flatten a non-nullable-bool AND tree into a flat list of leaf arrays.
///
/// Only descends into `ScalarFnArray(Binary(And))` nodes whose children are both non-nullable
/// bool. Any mixed-nullability subtree is treated as an opaque leaf.
fn collect_and_leaves_tree(array: &ArrayRef, out: &mut SmallVec<[ArrayRef; 8]>) {
    if let Some(sfn) = array.as_opt::<ScalarFn>() {
        if matches!(sfn.scalar_fn().as_opt::<Binary>(), Some(Operator::And))
            && sfn.nchildren() == 2
            && is_non_nullable_bool_array(sfn.child_at(0))
            && is_non_nullable_bool_array(sfn.child_at(1))
        {
            collect_and_leaves_tree(sfn.child_at(0), out);
            collect_and_leaves_tree(sfn.child_at(1), out);
            return;
        }
    }
    if is_non_nullable_bool_array(array) {
        out.push(array.clone());
    }
}

/// Flatten a non-nullable-bool OR tree into a flat list of leaf arrays.
fn collect_or_leaves_tree(array: &ArrayRef, out: &mut SmallVec<[ArrayRef; 8]>) {
    if let Some(sfn) = array.as_opt::<ScalarFn>() {
        if matches!(sfn.scalar_fn().as_opt::<Binary>(), Some(Operator::Or))
            && sfn.nchildren() == 2
            && is_non_nullable_bool_array(sfn.child_at(0))
            && is_non_nullable_bool_array(sfn.child_at(1))
        {
            collect_or_leaves_tree(sfn.child_at(0), out);
            collect_or_leaves_tree(sfn.child_at(1), out);
            return;
        }
    }
    if is_non_nullable_bool_array(array) {
        out.push(array.clone());
    }
}
