// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opcode set for the linear expression engine.
//!
//! Two families of bool opcodes operate in place on scratch registers, eliminating per-step
//! `BoolArray` allocations:
//!
//! - **Non-nullable** ([`OutputRegister::Bool`]): `AllocBool` / `AndInto` / `OrInto` / `NotInto`.
//!   Compiler emits these when all bool operands are `DType::Bool(NonNullable)`.
//! - **Kleene nullable** ([`OutputRegister::NullableBool`]): `AllocNullableBool` /
//!   `AndIntoNullable` / `OrIntoNullable` / `NotIntoNullable`. Implements SQL three-valued logic
//!   (NULL AND FALSE = FALSE; NULL OR TRUE = TRUE; NOT NULL = NULL) over paired values+validity
//!   bit-buffers. Emitted when any bool operand is `DType::Bool(Nullable)`.
//!
//! `CaseMerge` implements `dst[i] = value[i] if cond[i] else dst[i]`. The compiler emits pairs
//! in reverse WHEN order so that the first WHEN clause's write arrives last (first-match-wins).
//!
//! Each variant carries a discriminant exposed via [`Opcode::tag`]; the executor indexes its
//! `HANDLERS` table by this tag.

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

    /// Phase 4: initialise `dst` as a nullable Bool scratch buffer. `values` is filled with
    /// `init`; `validity` is all-true (every row starts as a known non-null value).
    AllocNullableBool = 10,

    /// Phase 4: Kleene AND-merge of `src` into `dst` (a NullableBool register).
    ///
    /// `dst_values &= src_values`. Validity follows three-valued logic:
    /// `dst_valid = (dst_valid & src_valid) | (dst_valid & !dst_val) | (src_valid & !src_val)`.
    AndIntoNullable = 11,

    /// Phase 4: Kleene OR-merge of `src` into `dst` (a NullableBool register).
    ///
    /// `dst_values |= src_values`. Validity follows three-valued logic:
    /// `dst_valid = (dst_valid & src_valid) | (dst_valid & dst_val) | (src_valid & src_val)`.
    OrIntoNullable = 12,

    /// Phase 4: flip the values bits of `reg` (a NullableBool register) in place.
    /// Validity is unchanged: NOT NULL = NULL.
    NotIntoNullable = 13,
}

impl OpTag {
    /// Number of distinct opcode tags. Used to size the executor's handler table.
    pub const COUNT: usize = 14;
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

    /// Phase 4: initialise `dst` as a NullableBool scratch buffer with `values` = `init` and
    /// `validity` = all-true.
    AllocNullableBool { dst: RegId, init: bool },

    /// Phase 4: Kleene AND-merge of `src` into `dst` (a NullableBool register).
    AndIntoNullable { dst: RegId, src: RegId },

    /// Phase 4: Kleene OR-merge of `src` into `dst` (a NullableBool register).
    OrIntoNullable { dst: RegId, src: RegId },

    /// Phase 4: flip the `values` bits of `reg` (a NullableBool register) in place.
    NotIntoNullable { reg: RegId },
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
            Opcode::AllocNullableBool { .. } => OpTag::AllocNullableBool,
            Opcode::AndIntoNullable { .. } => OpTag::AndIntoNullable,
            Opcode::OrIntoNullable { .. } => OpTag::OrIntoNullable,
            Opcode::NotIntoNullable { .. } => OpTag::NotIntoNullable,
        }
    }
}
