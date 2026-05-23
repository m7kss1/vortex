// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Threaded-dispatch executor for [`ExprProgram`].
//!
//! Each opcode is handled by a self-contained function with the signature
//! `fn(&mut ExecState) -> VortexResult<HandlerCtrl>`. Handlers are looked up through a static
//! `HANDLERS` table indexed by [`OpTag`]; there is no centralised match on opcodes.
//!
//! On stable Rust this gives us one indirect call per opcode plus a tight outer loop. When
//! `become` / `musttail` lands, the outer loop can be removed and each handler can tail-call its
//! successor directly — equivalent to PG's computed-goto `TEEO_NEXT` macro.

use std::ops::BitAnd;

use smallvec::SmallVec;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use super::ExprProgram;
use super::OutputRegister;
use super::RegId;
use super::opcode::OpTag;
use super::opcode::Opcode;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::bool::BoolArrayExt;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::fns::zip::zip_impl;

/// Control-flow signal returned by each handler back to the dispatch loop.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum HandlerCtrl {
    Next,
    Return,
}

/// [`ExecutionArgs`] backed by an inline `SmallVec<[ArrayRef; 4]>`.
///
/// `into_vec()` on a `SmallVec` always heap-allocates even when the data fits inline. This type
/// avoids that allocation for the common case of ≤4 scalar function arguments.
struct SmallVecExecutionArgs {
    inputs: SmallVec<[ArrayRef; 4]>,
    row_count: usize,
}

impl SmallVecExecutionArgs {
    fn new(inputs: SmallVec<[ArrayRef; 4]>, row_count: usize) -> Self {
        Self { inputs, row_count }
    }
}

impl ExecutionArgs for SmallVecExecutionArgs {
    fn get(&self, index: usize) -> VortexResult<ArrayRef> {
        self.inputs.get(index).cloned().ok_or_else(|| {
            vortex_error::vortex_err!(
                "Input index {} out of bounds (num_inputs={})",
                index,
                self.inputs.len()
            )
        })
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn row_count(&self) -> usize {
        self.row_count
    }
}

/// Execution state shared between handlers — closely modelled on LEE's `TurboExprState`.
struct ExecState<'a> {
    program: &'a ExprProgram,
    regs: &'a mut [OutputRegister],
    scope: &'a ArrayRef,
    ctx: &'a mut ExecutionCtx,
    input_mask: Option<&'a Mask>,
    mask_seeded: bool,
    pc: usize,
    result_src: Option<RegId>,
}

/// Handler function pointer type — all opcode handlers share this signature so they can be
/// stored uniformly in `HANDLERS`.
type Handler = fn(&mut ExecState<'_>) -> VortexResult<HandlerCtrl>;

/// Static handler table — indexed by `OpTag as usize`. Order **must** match the discriminants
/// in [`OpTag`].
static HANDLERS: [Handler; OpTag::COUNT] = [
    h_load_scope,          // 0:  LoadScope
    h_load_const,          // 1:  LoadConst
    h_call,                // 2:  Call
    h_alloc_bool,          // 3:  AllocBool
    h_and_into,            // 4:  AndInto
    h_or_into,             // 5:  OrInto
    h_not_into,            // 6:  NotInto
    h_return,              // 7:  Return
    h_case_merge,          // 8:  CaseMerge
    h_load_capture,        // 9:  LoadCapture
    h_alloc_nullable_bool, // 10: AllocNullableBool
    h_and_into_nullable,   // 11: AndIntoNullable
    h_or_into_nullable,    // 12: OrIntoNullable
    h_not_into_nullable,   // 13: NotIntoNullable
];

/// Execute a compiled program against the given scope array, returning the result array.
pub fn execute_program(
    program: &ExprProgram,
    scope: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (mut result, _) = run_program(program, scope, None, ctx)?;
    result.take_or_freeze()
}

/// Execute a compiled boolean program as a scan filter and return the final mask.
///
/// If the program's result register is an in-place Bool scratch buffer, this avoids freezing it
/// into a `BoolArray` and then converting that array back into a `Mask`. For top-level AND
/// reductions the initial `AllocBool(true)` is seeded from `input_mask`, so the final input-mask
/// intersection is fused into the in-place reduction.
pub fn execute_mask_program(
    program: &ExprProgram,
    scope: &ArrayRef,
    input_mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    let (result, mask_seeded) = run_program(program, scope, Some(input_mask), ctx)?;
    let result_mask = match result {
        OutputRegister::View(array) => array.execute::<Mask>(ctx)?,
        OutputRegister::Bool(buf) => Mask::from_buffer(buf.freeze()),
        // Kleene nulls don't pass the filter: mask = values & validity (NULL → false).
        OutputRegister::NullableBool { values, validity } => {
            Mask::from_buffer(values.freeze() & validity.freeze())
        }
        OutputRegister::Empty => vortex_bail!("lee: Return handler produced an empty register"),
    };

    if mask_seeded {
        Ok(result_mask)
    } else {
        Ok(input_mask.bitand(&result_mask))
    }
}

fn run_program(
    program: &ExprProgram,
    scope: &ArrayRef,
    input_mask: Option<&Mask>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(OutputRegister, bool)> {
    let mut regs: Vec<OutputRegister> = (0..program.num_regs as usize)
        .map(|_| OutputRegister::default())
        .collect();

    let mut state = ExecState {
        program,
        regs: regs.as_mut_slice(),
        scope,
        ctx,
        input_mask,
        mask_seeded: false,
        pc: 0,
        result_src: None,
    };

    // Threaded loop: one indirect call per opcode, no centralised match on `Opcode`.
    loop {
        if state.pc >= state.program.opcodes.len() {
            vortex_bail!("lee: ran off the end of program without a Return opcode");
        }
        let tag = state.program.opcodes[state.pc].tag();
        match HANDLERS[tag as usize](&mut state)? {
            HandlerCtrl::Next => state.pc += 1,
            HandlerCtrl::Return => {
                let src = state.result_src.ok_or_else(|| {
                    vortex_error::vortex_err!(
                        "lee: Return handler did not identify a result register"
                    )
                })?;
                return Ok((
                    std::mem::take(&mut state.regs[src.idx()]),
                    state.mask_seeded,
                ));
            }
        }
    }
}

// -------------------------------------------------------------------------------------
// Handlers — one per opcode variant. Kept #[inline(always)] so the threaded loop stays
// as tight as possible and the destructuring in each handler is elided.
// -------------------------------------------------------------------------------------

#[inline(always)]
fn h_load_scope(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::LoadScope { dst } = &s.program.opcodes[s.pc] else {
        unreachable!("h_load_scope dispatched on non-LoadScope opcode");
    };
    let dst = *dst;
    s.regs[dst.idx()].assign_view(s.scope.clone());
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_load_const(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::LoadConst { dst, scalar } = &s.program.opcodes[s.pc] else {
        unreachable!("h_load_const dispatched on non-LoadConst opcode");
    };
    let dst = *dst;
    let scalar = scalar.clone();
    let len = s.scope.len();
    s.regs[dst.idx()].assign_view(ConstantArray::new(scalar, len).into_array());
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_call(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::Call {
        dst,
        scalar_fn,
        args,
    } = &s.program.opcodes[s.pc]
    else {
        unreachable!("h_call dispatched on non-Call opcode");
    };
    let dst = *dst;
    let scalar_fn = scalar_fn.clone();

    let mut arg_arrays: SmallVec<[ArrayRef; 4]> = SmallVec::with_capacity(args.len());
    for r in args.iter() {
        arg_arrays.push(s.regs[r.idx()].as_array_ref()?.clone());
    }

    let row_count = s.scope.len();
    scalar_fn.execute_into(
        &SmallVecExecutionArgs::new(arg_arrays, row_count),
        &mut s.regs[dst.idx()],
        s.ctx,
    )?;
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_alloc_bool(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::AllocBool { dst, init } = &s.program.opcodes[s.pc] else {
        unreachable!("h_alloc_bool dispatched on non-AllocBool opcode");
    };
    let dst = *dst;
    let init = *init;
    if init
        && dst == s.program.result_reg
        && let Some(mask) = s.input_mask
    {
        s.regs[dst.idx()].assign_bool_from_mask(mask);
        s.mask_seeded = true;
    } else {
        let len = s.scope.len();
        s.regs[dst.idx()].assign_bool_filled(len, init);
    }
    Ok(HandlerCtrl::Next)
}

/// Shared word-by-word merge of `src` into `dst` (a Bool scratch buffer). `src` may be
/// either a canonical bool array view or another in-flight Bool scratch register.
#[inline(always)]
fn merge_bool_into(s: &mut ExecState<'_>, dst: RegId, src: RegId, op: ByteOp) -> VortexResult<()> {
    // Materialise/copy the source bits before mutably borrowing `dst`; this also supports nested
    // reductions where `src` is itself a Bool scratch register, e.g. `(a AND b) AND c`.
    let src_bits = match &mut s.regs[src.idx()] {
        OutputRegister::View(array) => array.clone().execute::<BoolArray>(s.ctx)?.into_bit_buffer(),
        OutputRegister::Bool(buf) => buf.clone().freeze(),
        OutputRegister::NullableBool { .. } => {
            vortex_bail!(
                "lee: non-nullable AndInto/OrInto received a NullableBool source; \
                 use AndIntoNullable/OrIntoNullable for nullable operands"
            )
        }
        OutputRegister::Empty => vortex_bail!("lee: attempted to merge from empty register"),
    };
    let src_len = src_bits.len();

    let dst_buf = s.regs[dst.idx()].as_bool_mut()?;
    if dst_buf.len() != src_len {
        vortex_bail!(
            "lee: bool merge length mismatch (dst {} vs src {})",
            dst_buf.len(),
            src_len
        );
    }

    let dst_offset = dst_buf.offset();
    let dst_bytes = dst_buf.as_mut_slice();
    if src_bits.offset() == 0 && dst_offset == 0 {
        let src_bytes = src_bits.inner().as_slice();
        for (dst_byte, src_byte) in dst_bytes.iter_mut().zip(src_bytes.iter().copied()) {
            *dst_byte = match op {
                ByteOp::And => *dst_byte & src_byte,
                ByteOp::Or => *dst_byte | src_byte,
            };
        }
        return Ok(());
    }

    // Fallback for sliced/un-byte-aligned buffers. This is uncommon in the dense scan path but
    // keeps the opcode correct for general expression execution.
    for byte_idx in 0..dst_bytes.len() {
        let bit_start = byte_idx * 8;
        let bit_end = (bit_start + 8).min(src_len);
        let mut src_byte: u8 = 0;
        for bit in bit_start..bit_end {
            if src_bits.value(bit) {
                src_byte |= 1 << (bit - bit_start);
            }
        }
        dst_bytes[byte_idx] = match op {
            ByteOp::And => dst_bytes[byte_idx] & src_byte,
            ByteOp::Or => dst_bytes[byte_idx] | src_byte,
        };
    }
    Ok(())
}

#[derive(Copy, Clone)]
enum ByteOp {
    And,
    Or,
}

#[inline(always)]
fn h_and_into(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::AndInto { dst, src } = &s.program.opcodes[s.pc] else {
        unreachable!("h_and_into dispatched on non-AndInto opcode");
    };
    let (dst, src) = (*dst, *src);
    merge_bool_into(s, dst, src, ByteOp::And)?;
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_or_into(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::OrInto { dst, src } = &s.program.opcodes[s.pc] else {
        unreachable!("h_or_into dispatched on non-OrInto opcode");
    };
    let (dst, src) = (*dst, *src);
    merge_bool_into(s, dst, src, ByteOp::Or)?;
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_not_into(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::NotInto { reg } = &s.program.opcodes[s.pc] else {
        unreachable!("h_not_into dispatched on non-NotInto opcode");
    };
    let reg = *reg;
    let buf = s.regs[reg.idx()].as_bool_mut()?;
    let bit_len = buf.len();
    let bytes = buf.as_mut_slice();
    // Flip whole bytes; the trailing partial byte's high bits are above `bit_len` and are
    // ignored by every downstream `value(idx)` read.
    if let Some(last_idx) = bytes.len().checked_sub(1) {
        for b in bytes[..last_idx].iter_mut() {
            *b = !*b;
        }
        // For the last byte, only flip the bits below bit_len to keep the trailing bits
        // canonically zero — this matters when freezing into a `BoolArray` whose internal
        // invariants rely on padding bits being zero.
        let valid_in_last = bit_len - last_idx * 8;
        let mask: u8 = if valid_in_last >= 8 {
            0xFF
        } else {
            (1u8 << valid_in_last) - 1
        };
        bytes[last_idx] = (!bytes[last_idx]) & mask;
    }
    Ok(()).map(|_| HandlerCtrl::Next)
}

/// Phase 3 CASE WHEN handler: `dst[i] = value[i] if cond[i] else dst[i]`.
///
/// Implements one step of the reverse-order CASE WHEN compilation. The `cond` register holds a
/// non-nullable bool array (WHEN condition); `value` holds the THEN result; `dst` is the running
/// accumulator (initialised from the ELSE branch). Using Arrow's `zip` operation, rows where
/// `cond` is true get the value from `value`, others keep their current `dst` value.
///
/// After all `CaseMerge` opcodes run in reverse pair order, `dst` contains the CASE WHEN result
/// with correct first-match-wins semantics (the first pair's write arrives last and wins).
#[inline(always)]
fn h_case_merge(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::CaseMerge { dst, value, cond } = &s.program.opcodes[s.pc] else {
        unreachable!("h_case_merge dispatched on non-CaseMerge opcode");
    };
    let (dst, value, cond) = (*dst, *value, *cond);

    let value_array = s.regs[value.idx()].as_array_ref()?.clone();
    let dst_array = s.regs[dst.idx()].as_array_ref()?.clone();
    let cond_array = s.regs[cond.idx()].as_array_ref()?.clone();

    // Convert the WHEN condition to a Mask, then zip: dst[i] = value[i] if cond[i] else dst[i].
    let cond_mask = cond_array.execute::<Mask>(s.ctx)?;
    let merged = zip_impl(&value_array, &dst_array, &cond_mask)?;
    s.regs[dst.idx()].assign_view(merged);

    Ok(HandlerCtrl::Next)
}

/// Shared Kleene byte-level merge of `src` into `dst` (a NullableBool scratch register).
///
/// Implements Kleene three-valued logic for AND and OR per byte, without allocating.
///
/// AND formula (per byte):
/// - `dst_val  = dst_val & src_val`
/// - `dst_valid = (dst_valid & src_valid) | (dst_valid & !dst_val_old) | (src_valid & !src_val)`
///
/// OR formula (per byte):
/// - `dst_val  = dst_val | src_val`
/// - `dst_valid = (dst_valid & src_valid) | (dst_valid & dst_val_old) | (src_valid & src_val)`
///
/// When `src` is non-nullable (src_valid is all-`0xFF`), the formulas simplify:
/// - AND: `dst_valid = dst_valid | !src_val`
/// - OR:  `dst_valid = dst_valid | src_val`
#[inline(always)]
fn merge_nullable_bool_into(
    s: &mut ExecState<'_>,
    dst: RegId,
    src: RegId,
    op: NullableByteOp,
) -> VortexResult<()> {
    // Extract src bits before mutably borrowing dst. Both values and validity are needed.
    // `src_validity = None` means all-valid (non-nullable source).
    let (src_values, src_validity): (BitBuffer, Option<BitBuffer>) = match &s.regs[src.idx()] {
        OutputRegister::View(array) => {
            let bool_arr = array.clone().execute::<BoolArray>(s.ctx)?;
            // Use the BoolArrayExt trait method explicitly to avoid ambiguity with
            // TypedArrayRef::validity() which returns VortexResult<Validity>.
            let validity = BoolArrayExt::validity(&bool_arr);
            let validity_bits: Option<BitBuffer> = if validity.no_nulls() {
                None
            } else {
                let mask = validity.execute_mask(bool_arr.len(), s.ctx)?;
                Some(match mask {
                    Mask::AllTrue(len) => BitBuffer::new_set(len),
                    Mask::AllFalse(len) => BitBuffer::new_unset(len),
                    Mask::Values(mv) => mv.bit_buffer().clone(),
                })
            };
            (bool_arr.into_bit_buffer(), validity_bits)
        }
        OutputRegister::Bool(buf) => (buf.clone().freeze(), None),
        OutputRegister::NullableBool { values, validity } => {
            (values.clone().freeze(), Some(validity.clone().freeze()))
        }
        OutputRegister::Empty => vortex_bail!("lee: attempted to merge from empty register"),
    };
    let src_len = src_values.len();

    // Mutably borrow dst (NullableBool) — safe because src != dst for AND/OR opcodes.
    let (dst_values, dst_validity) = match &mut s.regs[dst.idx()] {
        OutputRegister::NullableBool { values, validity } => (values, validity),
        _ => vortex_bail!("lee: nullable merge target is not a NullableBool register"),
    };

    if dst_values.len() != src_len {
        vortex_bail!(
            "lee: nullable bool merge length mismatch (dst {} vs src {})",
            dst_values.len(),
            src_len
        );
    }

    // Fast path: both buffers are byte-aligned (offset == 0).
    if src_values.offset() == 0 && dst_values.offset() == 0 {
        let src_val_bytes = src_values.inner().as_slice();
        let dst_val_bytes = dst_values.as_mut_slice();
        let dst_valid_bytes = dst_validity.as_mut_slice();

        match src_validity {
            None => {
                // Non-nullable source: src_valid = 0xFF per byte.
                // AND simplified: dst_valid = dst_valid | !src_val
                // OR  simplified: dst_valid = dst_valid | src_val
                for i in 0..dst_val_bytes.len() {
                    let dv = dst_val_bytes[i];
                    let sv = src_val_bytes[i];
                    let dp = dst_valid_bytes[i];
                    match op {
                        NullableByteOp::And => {
                            dst_val_bytes[i] = dv & sv;
                            dst_valid_bytes[i] = dp | !sv;
                        }
                        NullableByteOp::Or => {
                            dst_val_bytes[i] = dv | sv;
                            dst_valid_bytes[i] = dp | sv;
                        }
                    }
                }
            }
            Some(src_valid_buf) => {
                let src_valid_bytes = src_valid_buf.inner().as_slice();
                for i in 0..dst_val_bytes.len() {
                    let dv = dst_val_bytes[i];
                    let sv = src_val_bytes[i];
                    let dp = dst_valid_bytes[i];
                    let sp = src_valid_bytes[i];
                    match op {
                        NullableByteOp::And => {
                            dst_val_bytes[i] = dv & sv;
                            dst_valid_bytes[i] = (dp & sp) | (dp & !dv) | (sp & !sv);
                        }
                        NullableByteOp::Or => {
                            dst_val_bytes[i] = dv | sv;
                            dst_valid_bytes[i] = (dp & sp) | (dp & dv) | (sp & sv);
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    // Fallback for sliced/unaligned buffers: bit-by-bit.
    for bit in 0..src_len {
        let sv = src_values.value(bit);
        let sp = src_validity.as_ref().is_none_or(|v| v.value(bit));
        let dv = dst_values.value(bit);
        let dp = dst_validity.value(bit);
        let (new_val, new_valid) = match op {
            NullableByteOp::And => (dv & sv, (dp & sp) | (dp & !dv) | (sp & !sv)),
            NullableByteOp::Or => (dv | sv, (dp & sp) | (dp & dv) | (sp & sv)),
        };
        if new_val {
            dst_values.set(bit);
        } else {
            dst_values.unset(bit);
        }
        if new_valid {
            dst_validity.set(bit);
        } else {
            dst_validity.unset(bit);
        }
    }
    Ok(())
}

#[derive(Copy, Clone)]
enum NullableByteOp {
    And,
    Or,
}

#[inline(always)]
fn h_alloc_nullable_bool(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::AllocNullableBool { dst, init } = &s.program.opcodes[s.pc] else {
        unreachable!("h_alloc_nullable_bool dispatched on non-AllocNullableBool opcode");
    };
    let dst = *dst;
    let init = *init;
    if init
        && dst == s.program.result_reg
        && let Some(mask) = s.input_mask
    {
        s.regs[dst.idx()].assign_nullable_bool_from_mask(mask);
        s.mask_seeded = true;
    } else {
        let len = s.scope.len();
        s.regs[dst.idx()].assign_nullable_bool_filled(len, init);
    }
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_and_into_nullable(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::AndIntoNullable { dst, src } = &s.program.opcodes[s.pc] else {
        unreachable!("h_and_into_nullable dispatched on non-AndIntoNullable opcode");
    };
    let (dst, src) = (*dst, *src);
    merge_nullable_bool_into(s, dst, src, NullableByteOp::And)?;
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_or_into_nullable(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::OrIntoNullable { dst, src } = &s.program.opcodes[s.pc] else {
        unreachable!("h_or_into_nullable dispatched on non-OrIntoNullable opcode");
    };
    let (dst, src) = (*dst, *src);
    merge_nullable_bool_into(s, dst, src, NullableByteOp::Or)?;
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_not_into_nullable(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::NotIntoNullable { reg } = &s.program.opcodes[s.pc] else {
        unreachable!("h_not_into_nullable dispatched on non-NotIntoNullable opcode");
    };
    let reg = *reg;
    let (values, _validity) = s.regs[reg.idx()].as_nullable_bool_mut()?;
    // Flip values in place (same logic as h_not_into); validity is unchanged: NOT NULL = NULL.
    let bit_len = values.len();
    let bytes = values.as_mut_slice();
    if let Some(last_idx) = bytes.len().checked_sub(1) {
        for b in bytes[..last_idx].iter_mut() {
            *b = !*b;
        }
        let valid_in_last = bit_len - last_idx * 8;
        let mask: u8 = if valid_in_last >= 8 {
            0xFF
        } else {
            (1u8 << valid_in_last) - 1
        };
        bytes[last_idx] = (!bytes[last_idx]) & mask;
    }
    Ok(HandlerCtrl::Next)
}

/// Phase A/B: load a pre-captured array into the destination register.
///
/// The array was extracted from the scope at compile time by the encoding optimizer (e.g. the
/// dict values or codes sub-array). It is embedded directly in the opcode, so this handler is
/// a simple `Arc` clone — no data movement.
#[inline(always)]
fn h_load_capture(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::LoadCapture { dst, array } = &s.program.opcodes[s.pc] else {
        unreachable!("h_load_capture dispatched on non-LoadCapture opcode");
    };
    let dst = *dst;
    s.regs[dst.idx()].assign_view(array.clone());
    Ok(HandlerCtrl::Next)
}

#[inline(always)]
fn h_return(s: &mut ExecState<'_>) -> VortexResult<HandlerCtrl> {
    let Opcode::Return { src } = &s.program.opcodes[s.pc] else {
        unreachable!("h_return dispatched on non-Return opcode");
    };
    let src = *src;
    s.result_src = Some(src);
    Ok(HandlerCtrl::Return)
}

