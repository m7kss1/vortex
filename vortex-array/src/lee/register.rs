// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Register-based scratch storage for the linear expression engine.
//!
//! LEE keeps inter-opcode values in registers. In Phase 0 the only inhabited variant was
//! [`OutputRegister::View`] (a cheap reference to an existing [`ArrayRef`]). Phase 1 adds
//! [`OutputRegister::Bool`], an owned mutable bit-buffer used by stateful opcodes
//! (`AllocBool`, `AndInto`, `OrInto`, `NotInto`) to AND/OR/flip bits **in place** without
//! materialising a new `BoolArray` between steps.
//!
//! Phase 1 keeps `Bool` registers strictly *non-nullable*: a bare [`BitBufferMut`] with no
//! validity buffer alongside it. The compiler only emits the stateful Bool opcodes when both
//! operands produce `DType::Bool(NonNullable)`; anything that could be nullable falls back to
//! the generic `Call` opcode (Phase 0 path) which preserves Kleene-AND semantics. A future
//! phase will introduce `OutputRegister::NullableBool { values, validity }` to make the
//! in-place path work for nullable inputs too.

use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::dtype::Nullability;
use crate::validity::Validity;

/// Identifier for a register inside an [`super::ExprProgram`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct RegId(u16);

impl RegId {
    pub const fn new(idx: u16) -> Self {
        Self(idx)
    }
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// Storage for a single register during program execution.
#[derive(Default)]
pub enum OutputRegister {
    /// Uninitialised slot.
    #[default]
    Empty,

    /// Cheap reference to an existing array. Assignment is an `Arc<Array>` clone.
    View(ArrayRef),

    /// Owned, mutable non-nullable boolean scratch buffer. Used as the destination of in-place
    /// bool bitops (`AllocBool`/`AndInto`/`OrInto`/`NotInto`). At the end of a program a `Bool`
    /// register is frozen into a `BoolArray` view (see [`Self::take_or_freeze`]).
    Bool(BitBufferMut),
}

impl OutputRegister {
    /// Replace the register's contents with a view onto the given array.
    pub fn assign_view(&mut self, array: ArrayRef) {
        *self = OutputRegister::View(array);
    }

    /// Initialise the register as a non-nullable Bool scratch buffer filled with `init`.
    ///
    /// Used by `AllocBool` to set up the destination of an in-place bool reduction (AND wants
    /// `init = true`, OR wants `init = false`).
    pub fn assign_bool_filled(&mut self, len: usize, init: bool) {
        *self = OutputRegister::Bool(BitBufferMut::full(init, len));
    }

    /// Initialise the register as a Bool scratch buffer copied from an input mask.
    pub fn assign_bool_from_mask(&mut self, mask: &Mask) {
        *self = OutputRegister::Bool(BitBufferMut::copy_from(&mask.to_bit_buffer()));
    }

    /// Borrow the register's contents as a non-mutable [`ArrayRef`], or error if it is in a
    /// state that has no array form (e.g. an in-flight `Bool` scratch buffer that has not been
    /// frozen yet, or an uninitialised slot).
    pub fn as_array_ref(&self) -> VortexResult<&ArrayRef> {
        match self {
            OutputRegister::View(a) => Ok(a),
            OutputRegister::Bool(_) => {
                vortex_bail!(
                    "lee: cannot read an in-flight Bool scratch register as ArrayRef; \
                     freeze it via FreezeBool first or only use it with stateful bool opcodes"
                )
            }
            OutputRegister::Empty => vortex_bail!("attempted to read from uninitialised register"),
        }
    }

    /// Borrow the register as a mutable Bool scratch buffer.
    pub fn as_bool_mut(&mut self) -> VortexResult<&mut BitBufferMut> {
        match self {
            OutputRegister::Bool(b) => Ok(b),
            OutputRegister::View(_) => vortex_bail!(
                "lee: register holds an array view, not a mutable Bool scratch buffer"
            ),
            OutputRegister::Empty => {
                vortex_bail!("lee: cannot mutate uninitialised register as Bool")
            }
        }
    }

    /// Take ownership of the register's contents, leaving it [`OutputRegister::Empty`].
    ///
    /// Bool scratch registers are frozen into a non-nullable [`BoolArray`] view at this point.
    pub fn take_or_freeze(&mut self) -> VortexResult<ArrayRef> {
        match std::mem::take(self) {
            OutputRegister::View(a) => Ok(a),
            OutputRegister::Bool(buf) => Ok(BoolArray::new(
                buf.freeze(),
                Validity::from(Nullability::NonNullable),
            )
            .into_array()),
            OutputRegister::Empty => vortex_bail!("attempted to take from uninitialised register"),
        }
    }
}
