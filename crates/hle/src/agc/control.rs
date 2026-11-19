//! Command-buffer control flow (branches, conditional execution, calls)
//! and the `*Patch*` functions that fill in packets later.
//! Ported from SharpEmu (GPL-2.0-or-later); see `docs/gpu.md`.

use crate::Function;

pub(super) static FUNCTIONS: &[Function] = &[];

pub(super) static STUBS: &[Function] = &[];
