//! AMD GFX10.1 shader machine code, as the PS5 GPU (gfx1013) runs it: the
//! `.ags` container, a decoder, a printer in LLVM's syntax and a translator to
//! Metal Shading Language.

pub mod ags;
pub mod decode;
mod opcodes;
pub mod opinfo;
pub mod print;
pub mod translate;

pub use decode::{DecodeError, Inst, decode, decode_all};
pub use print::{Syntax, print, print_checked};
pub use translate::{Request, Resource, Source, Translated, translate};
