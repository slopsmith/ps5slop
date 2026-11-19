//! Reads Orbis executables and modules. See `docs/loader.md`.

pub mod dynamic;
pub mod elf;
mod error;
pub mod image;
pub mod nid;

pub use dynamic::Dynamic;
pub use elf::ElfFile;
pub use error::Error;
pub use image::Image;

pub type Result<T> = std::result::Result<T, Error>;
