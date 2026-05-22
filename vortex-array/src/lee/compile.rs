// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compile an [`Expression`] into a linear [`ExprProgram`].
//!
//! Compilation runs in two stages:
//!
//! 1. **Encoding-aware tree construction** (`scope.apply(expr)`): builds a lazy
//!    [`crate::arrays::ScalarFnArray`] tree with the scope bound at every `Root` node, then fires
//!    encoding-specific optimizer rules at each node (dict push-down, struct field resolution, …).
//!
//! 2. **Lowering to opcodes** ([`lower_tree`]): walks the optimised tree and emits flat opcodes:
//!
//! | Tree node | Opcodes emitted |
//! |---|---|
//! | The scope array (`Arc::ptr_eq`) | `LoadScope` (register shared) |
//! | [`crate::arrays::ConstantArray`] with `len == scope.len()` | `LoadConst` (cacheable) |
//! | `ConstantArray` with `len != scope.len()` | `LoadCapture` (per-batch) |
//! | All-constant `ScalarFnArray` children | `LoadConst` (folded at compile time) |
//! | Non-nullable bool `AND` chain | `AllocBool(true)` + N×`AndInto` |
//! | Non-nullable bool `OR` chain | `AllocBool(false)` + N×`OrInto` |
//! | Non-nullable bool `NOT` | `AllocBool(false)` + `OrInto` + `NotInto` |
//! | Nullable bool `AND` chain | `AllocNullableBool(true)` + N×`AndIntoNullable` |
//! | Nullable bool `OR` chain | `AllocNullableBool(false)` + N×`OrIntoNullable` |
//! | Nullable bool `NOT` | `AllocNullableBool(false)` + `OrIntoNullable` + `NotIntoNullable` |
//! | `CaseWhen` with ELSE | `lower(ELSE)` + reverse(`CaseMerge`) per pair |
//! | `ScalarFnArray` (generic) | recurse children → `Call` |
//! | Any other `ArrayRef` (optimizer-extracted sub-array) | `LoadCapture` |
//!
//! Programs that emit at least one `LoadCapture` set `cacheable = false` and are recompiled
//! per batch. Programs without captures remain `cacheable = true` and are shared via
//! [`super::ProgramCache`] across all batches of the same schema.

use std::sync::Arc;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use super::ExprProgram;
use super::Opcode;
use super::RegId;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::fns::binary::Binary;
use vortex_session::VortexSession;
use crate::scalar_fn::fns::case_when::CaseWhen;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::operators::Operator;

/// Compile an expression into an executable [`ExprProgram`] for the given scope array.
///
/// Runs the full encoding-aware pipeline: builds an optimised `ScalarFnArray` tree via
/// `scope.apply(expr)` (which fires encoding-specific optimizer rules at each node), then lowers
/// the tree to a flat opcode list.
///
/// The returned program's [`ExprProgram::cacheable`] flag indicates whether it is safe to share
/// across batches. Programs containing [`Opcode::LoadCapture`] opcodes must be recompiled per
/// batch; all others are safe to share via [`super::ProgramCache`].
pub fn compile(expr: &Expression, scope: &ArrayRef) -> VortexResult<ExprProgram> {
    let tree = scope.clone().apply(expr)?;

    let scope_dtype = scope.dtype();
    let mut ctx = CompileCtx {
        opcodes: Vec::new(),
        next_reg: 0,
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

struct CompileCtx {
    opcodes: Vec<Opcode>,
    next_reg: u16,
    scope: ArrayRef,
    /// Shared register for the single `LoadScope` opcode; reused by every occurrence of the
    /// scope array in the tree without emitting a second `LoadScope`.
    scope_reg: Option<RegId>,
    /// Dead registers available for reuse. Recycled across AND/OR leaf evaluations and CASE WHEN
    /// branches to keep `num_regs` small.
    free_list: SmallVec<[RegId; 8]>,
    /// Maps pointer address → register for `LoadCapture` deduplication: the same sub-array
    /// extracted by the optimizer (e.g. `dict_values`) is loaded only once.
    capture_map: HashMap<usize, RegId>,
    has_captures: bool,
}

impl CompileCtx {
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
    /// The scope register is never freed: it must remain valid for the lifetime of the program
    /// because all `Call` opcodes that access scope columns read from it.
    fn free_reg(&mut self, id: RegId) {
        if Some(id) == self.scope_reg {
            return;
        }
        self.free_list.push(id);
    }
}

fn is_bool_array(array: &ArrayRef) -> bool {
    matches!(array.dtype(), DType::Bool(_))
}

fn is_nullable_bool_array(array: &ArrayRef) -> bool {
    matches!(array.dtype(), DType::Bool(Nullability::Nullable))
}

/// Returns true if `array` is a `ConstantArray` with the same length as `scope`.
///
/// Such nodes can be constant-folded at compile time: the scalar value is scope-length-independent
/// and the handler can broadcast it at runtime with a plain `LoadConst`.
fn is_foldable_constant(array: &ArrayRef, scope_len: usize) -> bool {
    array.as_opt::<Constant>().is_some() && array.len() == scope_len
}

/// Emit `LoadCapture` for `array` (or reuse an existing capture register), marking the program
/// as non-cacheable.
fn emit_capture(array: &ArrayRef, ctx: &mut CompileCtx) -> VortexResult<RegId> {
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

/// Attempt to constant-fold a `ScalarFnArray` node whose children are all scope-length constants.
///
/// Executes the function with 1-row constant inputs, extracts the scalar result, and returns it.
/// Using a 1-row execution is valid because every input row produces the same output when all
/// inputs are constants.
///
/// Returns `None` if any child is not a foldable constant or if execution fails.
fn try_constant_fold(sfn: ArrayView<'_, ScalarFn>, scope_len: usize) -> Option<Scalar> {
    if sfn.nchildren() == 0 {
        return None;
    }
    // All children must be scope-length ConstantArrays.
    for i in 0..sfn.nchildren() {
        if !is_foldable_constant(sfn.child_at(i), scope_len) {
            return None;
        }
    }

    // Execute with 1-row versions of each constant to produce a single-row result.
    struct ConstArgs(SmallVec<[ArrayRef; 4]>);
    impl ExecutionArgs for ConstArgs {
        fn get(&self, index: usize) -> VortexResult<ArrayRef> {
            self.0.get(index).cloned().ok_or_else(|| {
                vortex_error::vortex_err!("index {} out of bounds", index)
            })
        }
        fn num_inputs(&self) -> usize {
            self.0.len()
        }
        fn row_count(&self) -> usize {
            1
        }
    }

    let mut inputs: SmallVec<[ArrayRef; 4]> = SmallVec::new();
    for i in 0..sfn.nchildren() {
        let c = sfn.child_at(i).as_opt::<Constant>()?;
        inputs.push(ConstantArray::new(c.scalar().clone(), 1).into_array());
    }

    let mut ctx = ExecutionCtx::new(VortexSession::empty());
    let result = sfn.scalar_fn().execute(&ConstArgs(inputs), &mut ctx).ok()?;

    // Extract the scalar at row 0 from the 1-row result.
    result.execute_scalar(0, &mut ctx).ok()
}

fn lower_tree(array: &ArrayRef, ctx: &mut CompileCtx) -> VortexResult<RegId> {
    if ArrayRef::ptr_eq(array, &ctx.scope.clone()) {
        if let Some(existing) = ctx.scope_reg {
            return Ok(existing);
        }
        let dst = ctx.alloc_reg()?;
        ctx.opcodes.push(Opcode::LoadScope { dst });
        ctx.scope_reg = Some(dst);
        return Ok(dst);
    }

    if let Some(const_array) = array.as_opt::<Constant>() {
        let dst = ctx.alloc_reg()?;
        if array.len() == ctx.scope.len() {
            ctx.opcodes.push(Opcode::LoadConst {
                dst,
                scalar: const_array.scalar().clone(),
            });
        } else {
            // Length differs from scope (e.g. after dict-values pushdown). Capture the
            // pre-sized array directly.
            ctx.opcodes.push(Opcode::LoadCapture {
                dst,
                array: array.clone(),
            });
            ctx.has_captures = true;
            ctx.capture_map.insert(array.addr(), dst);
        }
        return Ok(dst);
    }

    if let Some(sfn) = array.as_opt::<ScalarFn>() {
        let scalar_fn = sfn.scalar_fn().clone();

        // AND chain: collect bool leaves, filter out identity constants (TRUE), short-circuit on
        // absorber (FALSE). Mixed nullable/non-nullable → Kleene path; all non-nullable → fast path.
        if matches!(scalar_fn.as_opt::<Binary>(), Some(Operator::And)) {
            let mut leaves: SmallVec<[ArrayRef; 8]> = SmallVec::new();
            collect_bool_and_leaves_tree(array, &mut leaves);
            if !leaves.is_empty() {
                // Short-circuit: any constant FALSE absorbs the whole AND.
                if leaves.iter().any(|l| {
                    l.as_opt::<Constant>()
                        .and_then(|c| c.scalar().as_bool_opt().and_then(|b| b.value()))
                        .is_some_and(|v| !v)
                }) {
                    let dst = ctx.alloc_reg()?;
                    ctx.opcodes.push(Opcode::LoadConst {
                        dst,
                        scalar: Scalar::from(false),
                    });
                    return Ok(dst);
                }
                // Identity: drop constant TRUE leaves.
                leaves.retain(|l| {
                    !l.as_opt::<Constant>()
                        .and_then(|c| c.scalar().as_bool_opt().and_then(|b| b.value()))
                        .is_some_and(|v| v)
                });
                // If all leaves were identity-eliminated, the result is TRUE.
                if leaves.is_empty() {
                    let dst = ctx.alloc_reg()?;
                    ctx.opcodes.push(Opcode::LoadConst {
                        dst,
                        scalar: Scalar::from(true),
                    });
                    return Ok(dst);
                }
                let any_nullable = leaves.iter().any(is_nullable_bool_array);
                let dst = ctx.alloc_reg()?;
                if any_nullable {
                    ctx.opcodes
                        .push(Opcode::AllocNullableBool { dst, init: true });
                    for leaf in &leaves {
                        let leaf_reg = lower_tree(leaf, ctx)?;
                        ctx.opcodes
                            .push(Opcode::AndIntoNullable { dst, src: leaf_reg });
                        ctx.free_reg(leaf_reg);
                    }
                } else {
                    ctx.opcodes.push(Opcode::AllocBool { dst, init: true });
                    for leaf in &leaves {
                        let leaf_reg = lower_tree(leaf, ctx)?;
                        ctx.opcodes.push(Opcode::AndInto { dst, src: leaf_reg });
                        ctx.free_reg(leaf_reg);
                    }
                }
                return Ok(dst);
            }
        }

        // OR chain: filter out identity constants (FALSE), short-circuit on absorber (TRUE).
        if matches!(scalar_fn.as_opt::<Binary>(), Some(Operator::Or)) {
            let mut leaves: SmallVec<[ArrayRef; 8]> = SmallVec::new();
            collect_bool_or_leaves_tree(array, &mut leaves);
            if !leaves.is_empty() {
                // Short-circuit: any constant TRUE absorbs the whole OR.
                if leaves.iter().any(|l| {
                    l.as_opt::<Constant>()
                        .and_then(|c| c.scalar().as_bool_opt().and_then(|b| b.value()))
                        .is_some_and(|v| v)
                }) {
                    let dst = ctx.alloc_reg()?;
                    ctx.opcodes.push(Opcode::LoadConst {
                        dst,
                        scalar: Scalar::from(true),
                    });
                    return Ok(dst);
                }
                // Identity: drop constant FALSE leaves.
                leaves.retain(|l| {
                    l.as_opt::<Constant>()
                        .and_then(|c| c.scalar().as_bool_opt().and_then(|b| b.value()))
                        .is_none_or(|v| v)
                });
                if leaves.is_empty() {
                    let dst = ctx.alloc_reg()?;
                    ctx.opcodes.push(Opcode::LoadConst {
                        dst,
                        scalar: Scalar::from(false),
                    });
                    return Ok(dst);
                }
                let any_nullable = leaves.iter().any(is_nullable_bool_array);
                let dst = ctx.alloc_reg()?;
                if any_nullable {
                    ctx.opcodes
                        .push(Opcode::AllocNullableBool { dst, init: false });
                    for leaf in &leaves {
                        let leaf_reg = lower_tree(leaf, ctx)?;
                        ctx.opcodes
                            .push(Opcode::OrIntoNullable { dst, src: leaf_reg });
                        ctx.free_reg(leaf_reg);
                    }
                } else {
                    ctx.opcodes.push(Opcode::AllocBool { dst, init: false });
                    for leaf in &leaves {
                        let leaf_reg = lower_tree(leaf, ctx)?;
                        ctx.opcodes.push(Opcode::OrInto { dst, src: leaf_reg });
                        ctx.free_reg(leaf_reg);
                    }
                }
                return Ok(dst);
            }
        }

        if scalar_fn.is::<Not>() && sfn.nchildren() == 1 && is_bool_array(sfn.child_at(0)) {
            let child = sfn.child_at(0);
            let nullable = is_nullable_bool_array(child);
            let child_reg = lower_tree(child, ctx)?;
            let dst = ctx.alloc_reg()?;
            if nullable {
                ctx.opcodes
                    .push(Opcode::AllocNullableBool { dst, init: false });
                ctx.opcodes.push(Opcode::OrIntoNullable {
                    dst,
                    src: child_reg,
                });
                ctx.free_reg(child_reg);
                ctx.opcodes.push(Opcode::NotIntoNullable { reg: dst });
            } else {
                ctx.opcodes.push(Opcode::AllocBool { dst, init: false });
                ctx.opcodes.push(Opcode::OrInto {
                    dst,
                    src: child_reg,
                });
                ctx.free_reg(child_reg);
                ctx.opcodes.push(Opcode::NotInto { reg: dst });
            }
            return Ok(dst);
        }

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
                    scalar: Scalar::null(result_dtype),
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

        // Constant folding: if all children are scope-length constants, execute now and emit
        // LoadConst. This handles expressions like `5 > 3`, `NOT FALSE`, etc.
        if let Some(scalar) = try_constant_fold(sfn, ctx.scope.len()) {
            let dst = ctx.alloc_reg()?;
            ctx.opcodes.push(Opcode::LoadConst { dst, scalar });
            return Ok(dst);
        }

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

    emit_capture(array, ctx)
}

/// Flatten a bool AND tree into leaf arrays, stopping at non-AND or non-bool nodes.
fn collect_bool_and_leaves_tree(array: &ArrayRef, out: &mut SmallVec<[ArrayRef; 8]>) {
    if let Some(sfn) = array.as_opt::<ScalarFn>()
        && matches!(sfn.scalar_fn().as_opt::<Binary>(), Some(Operator::And))
        && sfn.nchildren() == 2
        && is_bool_array(sfn.child_at(0))
        && is_bool_array(sfn.child_at(1))
    {
        collect_bool_and_leaves_tree(sfn.child_at(0), out);
        collect_bool_and_leaves_tree(sfn.child_at(1), out);
        return;
    }
    if is_bool_array(array) {
        out.push(array.clone());
    }
}

/// Flatten a bool OR tree into leaf arrays, stopping at non-OR or non-bool nodes.
fn collect_bool_or_leaves_tree(array: &ArrayRef, out: &mut SmallVec<[ArrayRef; 8]>) {
    if let Some(sfn) = array.as_opt::<ScalarFn>()
        && matches!(sfn.scalar_fn().as_opt::<Binary>(), Some(Operator::Or))
        && sfn.nchildren() == 2
        && is_bool_array(sfn.child_at(0))
        && is_bool_array(sfn.child_at(1))
    {
        collect_bool_or_leaves_tree(sfn.child_at(0), out);
        collect_bool_or_leaves_tree(sfn.child_at(1), out);
        return;
    }
    if is_bool_array(array) {
        out.push(array.clone());
    }
}
