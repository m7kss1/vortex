// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Register-based scratch storage for the linear expression engine.
//!
//! Three inhabited variants:
//!
//! - [`OutputRegister::View`] — a cheap `Arc` clone of an existing array; assignment is O(1).
//! - [`OutputRegister::Bool`] — owned mutable bit-buffer for in-place non-nullable bool
//!   reductions (`AllocBool` / `AndInto` / `OrInto` / `NotInto`). Avoids allocating a new
//!   `BoolArray` between conjuncts.
//! - [`OutputRegister::NullableBool`] — paired values + validity bit-buffers for Kleene
//!   three-valued bool reductions (`AllocNullableBool` / `AndIntoNullable` / `OrIntoNullable` /
//!   `NotIntoNullable`). Validity tracks whether each row's result is known (non-null).

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

    /// Owned, mutable nullable boolean scratch buffer: separate values and validity bit-buffers.
    ///
    /// Used by Phase 4 Kleene opcodes (`AllocNullableBool`/`AndIntoNullable`/`OrIntoNullable`/
    /// `NotIntoNullable`). `values` holds the boolean values; `validity` holds a `1` for
    /// non-null rows and `0` for null rows. At the end of a program this is frozen into a
    /// nullable `BoolArray`.
    NullableBool {
        values: BitBufferMut,
        validity: BitBufferMut,
    },
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

    /// Initialise the register as a NullableBool scratch buffer filled with `init` values and
    /// all-valid validity (every row starts as a known non-null value).
    pub fn assign_nullable_bool_filled(&mut self, len: usize, init: bool) {
        *self = OutputRegister::NullableBool {
            values: BitBufferMut::full(init, len),
            validity: BitBufferMut::full(true, len),
        };
    }

    /// Initialise the register as a NullableBool scratch buffer seeded from the input mask.
    ///
    /// `values` is copied from the mask bits; `validity` is all-true because rows excluded by
    /// the mask are never read — they are already absent from the scan window.
    pub fn assign_nullable_bool_from_mask(&mut self, mask: &Mask) {
        let len = mask.len();
        *self = OutputRegister::NullableBool {
            values: BitBufferMut::copy_from(&mask.to_bit_buffer()),
            validity: BitBufferMut::full(true, len),
        };
    }

    /// Borrow the register as a pair of mutable nullable-bool scratch buffers `(values, validity)`.
    pub fn as_nullable_bool_mut(&mut self) -> VortexResult<(&mut BitBufferMut, &mut BitBufferMut)> {
        match self {
            OutputRegister::NullableBool { values, validity } => Ok((values, validity)),
            OutputRegister::Bool(_) => {
                vortex_bail!("lee: register holds a non-nullable Bool buffer, not a NullableBool")
            }
            OutputRegister::View(_) => {
                vortex_bail!("lee: register holds an array view, not a NullableBool scratch buffer")
            }
            OutputRegister::Empty => {
                vortex_bail!("lee: cannot mutate uninitialised register as NullableBool")
            }
        }
    }

    /// Borrow the register's contents as a non-mutable [`ArrayRef`], or error if it is in a
    /// state that has no array form (e.g. an in-flight Bool scratch buffer or uninitialised slot).
    pub fn as_array_ref(&self) -> VortexResult<&ArrayRef> {
        match self {
            OutputRegister::View(a) => Ok(a),
            OutputRegister::Bool(_) => {
                vortex_bail!(
                    "lee: cannot read an in-flight Bool scratch register as ArrayRef; \
                     freeze it via FreezeBool first or only use it with stateful bool opcodes"
                )
            }
            OutputRegister::NullableBool { .. } => {
                vortex_bail!(
                    "lee: cannot read an in-flight NullableBool scratch register as ArrayRef; \
                     only use it with nullable stateful bool opcodes"
                )
            }
            OutputRegister::Empty => vortex_bail!("attempted to read from uninitialised register"),
        }
    }

    /// Borrow the register as a mutable Bool scratch buffer.
    pub fn as_bool_mut(&mut self) -> VortexResult<&mut BitBufferMut> {
        match self {
            OutputRegister::Bool(b) => Ok(b),
            OutputRegister::View(_) => {
                vortex_bail!("lee: register holds an array view, not a mutable Bool scratch buffer")
            }
            OutputRegister::NullableBool { .. } => vortex_bail!(
                "lee: register holds a NullableBool buffer; \
                 use as_nullable_bool_mut for nullable bool operations"
            ),
            OutputRegister::Empty => {
                vortex_bail!("lee: cannot mutate uninitialised register as Bool")
            }
        }
    }

    /// Take ownership of the register's contents, leaving it [`OutputRegister::Empty`].
    ///
    /// Bool scratch registers are frozen into a non-nullable [`BoolArray`]; NullableBool scratch
    /// registers are frozen into a nullable [`BoolArray`] using the accumulated validity buffer.
    pub fn take_or_freeze(&mut self) -> VortexResult<ArrayRef> {
        match std::mem::take(self) {
            OutputRegister::View(a) => Ok(a),
            OutputRegister::Bool(buf) => Ok(BoolArray::new(
                buf.freeze(),
                Validity::from(Nullability::NonNullable),
            )
            .into_array()),
            OutputRegister::NullableBool { values, validity } => {
                let values_buf = values.freeze();
                let validity_buf = validity.freeze();
                let validity_array =
                    BoolArray::new(validity_buf, Validity::NonNullable).into_array();
                Ok(BoolArray::new(values_buf, Validity::Array(validity_array)).into_array())
            }
            OutputRegister::Empty => vortex_bail!("attempted to take from uninitialised register"),
        }
    }
}
