//! Reads Orbis executables and modules. See `docs/loader.md`.

pub mod elf;
mod error;
pub mod nid;

pub use elf::ElfFile;
pub use error::Error;

pub type Result<T> = std::result::Result<T, Error>;
