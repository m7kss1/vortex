// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Linear Expression Engine (LEE) — the single expression evaluation engine for Vortex.
//!
//! # Architecture
//!
//! An [`Expression`](crate::expr::Expression) is compiled once into an [`ExprProgram`] — a flat
//! `Vec<Opcode>` with a register count — and then executed against each scope array:
//!
//! 1. **Compile** — [`compile`] builds a `ScalarFnArray` tree (Phase A), runs encoding-aware
//!    optimizer rules (Phase B), then lowers the tree to a flat opcode stream (Phase C).
//! 2. **Execute** — [`execute_program`] / [`execute_mask_program`] walk the opcode list and
//!    dispatch each opcode through a `HANDLERS` function-pointer table (threaded dispatch).
//!    Each handler reads inputs from registers, writes its result in-place, and returns.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use vortex_array::lee::{compile, execute_mask_program};
//! use vortex_array::session::ArraySession;
//! use vortex_session::VortexSession;
//! use vortex_mask::Mask;
//!
//! let session = VortexSession::empty().with::<ArraySession>();
//! let program = compile(&expr, &scope)?;
//! let mut ctx = session.create_execution_ctx();
//! let mask = execute_mask_program(&program, &scope, &Mask::new_true(scope.len()), &mut ctx)?;
//! ```

pub mod cache;
mod compile;
mod execute;
mod ext;
mod opcode;
mod program;
mod register;

pub use cache::ProgramCache;
pub use cache::ProgramCacheSessionExt;
pub use compile::compile;
pub use execute::execute_mask_program;
pub use execute::execute_program;
pub use ext::ArrayRefLeeExt;
pub(crate) use opcode::Opcode;
pub use program::ExprProgram;
pub(crate) use register::OutputRegister;
pub(crate) use register::RegId;
