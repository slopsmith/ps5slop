//! Reads fake-signed PS4 packages: the header, the metadata stored in the clear,
//! and (with the package's passcode) the files in its PFS image.
//!
//! Formats and their sources are described in `docs/packages.md`.

mod error;
pub mod keys;
mod package;
mod read;
mod xts;

pub use error::Error;
pub use package::{Entry, Header, Package, entry_id};
pub use read::{FileSlice, ReadAt};
pub use xts::XtsReader;

pub type Result<T> = std::result::Result<T, Error>;
