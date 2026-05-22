// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opcode set for the linear expression engine.
//!
//! Phase 0 implemented the minimum needed to be equivalent to
//! [`crate::ArrayRef::apply`] + `execute::<ArrayRef>`:
//!
//! - [`Opcode::LoadScope`] — analogue of `Root`
//! - [`Opcode::LoadConst`] — analogue of `Literal`
//! - [`Opcode::Call`] — generic fallback; delegates to [`crate::scalar_fn::ScalarFnRef::execute`]
//! - [`Opcode::Return`] — terminate program, yield the value in the given register
//!
//! Phase 1 adds **stateful** bool opcodes that operate in place on an
//! [`super::OutputRegister::Bool`] scratch buffer, eliminating per-step `BoolArray`
//! allocations for predicates of the form `e1 AND e2 (… AND eN)` and `NOT e`:
//!
//! - [`Opcode::AllocBool`] — initialise a destination Bool register with `init` bits (AND
//!   wants `true`, OR wants `false`).
//! - [`Opcode::AndInto`] — `regs[dst] &= regs[src].as_bool_bits()` word-by-word.
//! - [`Opcode::OrInto`]  — `regs[dst] |= regs[src].as_bool_bits()` word-by-word.
//! - [`Opcode::NotInto`] — flips bits in `regs[reg]` in place (XOR with 0xFF…).
//!
//! All Phase 1 stateful opcodes are *strictly non-nullable*: they look at values only and
//! drop validity. The compiler only emits them when both operands have
//! `DType::Bool(NonNullable)`; otherwise the generic `Call` path is used, preserving exact
//! Kleene semantics of the legacy tree walker.
//!
//! Phase 3 adds [`Opcode::CaseMerge`] for LEE-style CASE WHEN evaluation. It implements
//! a `zip`-style conditional write: `dst[i] = value[i] if cond[i] else dst[i]`. The compiler
//! detects `CaseWhen` scalar functions and emits: compile ELSE → `dst`, then for each
//! WHEN/THEN pair in **reverse** order emit `CaseMerge { dst, value: then_reg, cond: when_reg }`.
//! Reverse order gives correct first-match-wins semantics: the first pair's write arrives last
//! and wins over any later pair.
//!
//! Each variant carries a discriminant exposed via [`Opcode::tag`]; the executor indexes its
//! `HANDLERS` table by this tag (LEE-style threaded dispatch).

use smallvec::SmallVec;

use super::register::RegId;
use crate::ArrayRef;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnRef;

/// Tag identifying an [`Opcode`] variant, used as an index into the executor's handler table.
///
/// Order **must** match the order of handlers in `execute::HANDLERS`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum OpTag {
    LoadScope = 0,
    LoadConst = 1,
    Call = 2,
    AllocBool = 3,
    AndInto = 4,
    OrInto = 5,
    NotInto = 6,
    Return = 7,
    /// Phase 3: conditional write for CASE WHEN. `dst[i] = value[i] if cond[i] else dst[i]`.
    CaseMerge = 8,
    /// Phase A/B: load a pre-captured array extracted from the scope at compile time (e.g. a
    /// dict values or codes sub-array produced by the encoding optimizer). Unlike [`LoadScope`]
    /// this bypasses the runtime scope argument; it holds the array object itself.
    LoadCapture = 9,
}

impl OpTag {
    /// Number of distinct opcode tags. Used to size the executor's handler table.
    pub const COUNT: usize = 10;
}

/// A single opcode in an [`super::ExprProgram`].
#[derive(Debug, Clone)]
pub enum Opcode {
    /// `dst <- input scope array`. Stateless — assigns a view, no copy.
    LoadScope { dst: RegId },

    /// `dst <- ConstantArray::new(scalar, scope.len())`. Stateless.
    LoadConst { dst: RegId, scalar: Scalar },

    /// `dst <- scalar_fn(args)`. Phase 0 fallback: delegates to [`ScalarFnRef::execute`] and
    /// stores the returned `ArrayRef` as a register view. Used for any expression that doesn't
    /// fit a more specialised opcode.
    Call {
        dst: RegId,
        scalar_fn: ScalarFnRef,
        args: SmallVec<[RegId; 4]>,
    },

    /// Initialise `dst` as a non-nullable Bool scratch buffer filled with `init`. This is the
    /// AND/OR reduction identity: `init = true` for AND-chains, `init = false` for OR-chains.
    AllocBool { dst: RegId, init: bool },

    /// `regs[dst].bool_buf &= regs[src].as_bool_bits()`, in place, word by word. Both
    /// operands are assumed non-nullable.
    AndInto { dst: RegId, src: RegId },

    /// `regs[dst].bool_buf |= regs[src].as_bool_bits()`, in place, word by word. Both
    /// operands are assumed non-nullable.
    OrInto { dst: RegId, src: RegId },

    /// Flip every bit in `regs[reg].bool_buf`, in place. Operand is assumed non-nullable.
    NotInto { reg: RegId },

    /// Terminate the program; the executor returns the value of `src`. A `Bool` register is
    /// frozen into a non-nullable `BoolArray` at this point.
    Return { src: RegId },

    /// Phase 3: conditional write for CASE WHEN (LEE-style zip).
    ///
    /// Semantics: `regs[dst][i] = regs[value][i] if regs[cond][i] else regs[dst][i]`.
    ///
    /// The compiler emits one `CaseMerge` per WHEN/THEN pair in **reverse** pair order so that
    /// the first WHEN clause's write arrives last and wins (first-match-wins semantics).
    CaseMerge {
        /// Accumulator register (ELSE result or previous CaseMerge output). Updated in place.
        dst: RegId,
        /// THEN value for this branch.
        value: RegId,
        /// WHEN condition (a non-nullable bool array).
        cond: RegId,
    },

    /// Phase A/B: load a pre-captured [`ArrayRef`] into `dst`.
    ///
    /// Used by the encoding-aware compile path when the [`crate::optimizer::ArrayOptimizer`]
    /// rewrites the expression tree and extracts sub-arrays from the scope (e.g. the values or
    /// codes array of a `DictArray`). The captured array was extracted at compile time from the
    /// current scope and is valid for this batch only; programs containing this opcode are marked
    /// non-cacheable.
    LoadCapture { dst: RegId, array: ArrayRef },
}

impl Opcode {
    /// Returns the `OpTag` discriminant for handler-table dispatch.
    #[inline(always)]
    pub fn tag(&self) -> OpTag {
        match self {
            Opcode::LoadScope { .. } => OpTag::LoadScope,
            Opcode::LoadConst { .. } => OpTag::LoadConst,
            Opcode::Call { .. } => OpTag::Call,
            Opcode::AllocBool { .. } => OpTag::AllocBool,
            Opcode::AndInto { .. } => OpTag::AndInto,
            Opcode::OrInto { .. } => OpTag::OrInto,
            Opcode::NotInto { .. } => OpTag::NotInto,
            Opcode::Return { .. } => OpTag::Return,
            Opcode::CaseMerge { .. } => OpTag::CaseMerge,
            Opcode::LoadCapture { .. } => OpTag::LoadCapture,
        }
    }
}
