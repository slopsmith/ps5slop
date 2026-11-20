//! Reads fake-signed PS4 packages and PS5 packages whose image is stored in the
//! clear: the header, the metadata stored in the clear, and the files in the
//! PFS image (for PS4, with the package's passcode).
//!
//! Formats and their sources are described in `docs/packages.md`.

mod error;
mod extract;
mod inner;
pub mod keys;
mod kraken;
pub mod naps;
mod package;
pub mod pfs;
mod pfsc;
mod read;
pub mod title;
mod xts;

pub use error::Error;
pub use extract::Extracted;
pub use inner::InnerImage;
pub use package::{Entry, Format, Header, Package, entry_id};
pub use pfsc::PfscReader;
pub use read::{FileSlice, ReadAt};
pub use xts::XtsReader;

pub type Result<T> = std::result::Result<T, Error>;
