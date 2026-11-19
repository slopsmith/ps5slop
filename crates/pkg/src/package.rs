use std::fs::File;
use std::path::Path;

use crate::error::invalid;
use crate::read::{FileSlice, ReadAt, u32_be, u64_be};
use crate::{Error, Result};

const MAGIC: [u8; 4] = *b"\x7fCNT";
const HEADER_SIZE: usize = 0x1000;
const ENTRY_SIZE: usize = 0x20;
const ENTRY_ENCRYPTED: u32 = 0x8000_0000;

/// Entry ids with a known meaning (psdevwiki, "PKG files").
pub mod entry_id {
    pub const DIGESTS: u32 = 0x1;
    pub const ENTRY_KEYS: u32 = 0x10;
    pub const IMAGE_KEY: u32 = 0x20;
    pub const GENERAL_DIGESTS: u32 = 0x80;
    pub const METADATA: u32 = 0x100;
    pub const ENTRY_NAMES: u32 = 0x200;
    pub const PARAM_SFO: u32 = 0x1000;
    pub const ICON0_PNG: u32 = 0x1200;
    pub const PIC0_PNG: u32 = 0x1220;
    pub const PIC1_PNG: u32 = 0x1006;
}

/// The fields of the package header that we use. All are big-endian on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub content_id: String,
    pub content_type: u32,
    pub content_flags: u32,
    pub pfs_flags: u64,
    pub pfs_image_offset: u64,
    pub pfs_image_size: u64,
    pub package_size: u64,
    entry_count: u32,
    entry_table_offset: u32,
}

impl Header {
    pub fn parse(b: &[u8]) -> Result<Self> {
        if b.len() < HEADER_SIZE {
            return Err(invalid("package header is truncated"));
        }
        if b[..4] != MAGIC {
            return Err(invalid("not a package (bad magic)"));
        }
        let id = &b[0x40..0x64];
        let id = &id[..id.iter().position(|&c| c == 0).unwrap_or(id.len())];
        let content_id = std::str::from_utf8(id)
            .map_err(|_| invalid("content ID is not ASCII"))?
            .to_owned();
        Ok(Self {
            content_id,
            content_type: u32_be(b, 0x74),
            content_flags: u32_be(b, 0x78),
            pfs_flags: u64_be(b, 0x408),
            pfs_image_offset: u64_be(b, 0x410),
            pfs_image_size: u64_be(b, 0x418),
            package_size: u64_be(b, 0x430),
            entry_count: u32_be(b, 0x10),
            entry_table_offset: u32_be(b, 0x18),
        })
    }
}

/// One row of the entry table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub id: u32,
    pub flags1: u32,
    pub flags2: u32,
    pub offset: u32,
    pub size: u32,
}

impl Entry {
    fn parse(b: &[u8]) -> Self {
        Self {
            id: u32_be(b, 0x00),
            flags1: u32_be(b, 0x08),
            flags2: u32_be(b, 0x0c),
            offset: u32_be(b, 0x10),
            size: u32_be(b, 0x14),
        }
    }

    pub fn is_encrypted(&self) -> bool {
        self.flags1 & ENTRY_ENCRYPTED != 0
    }

    /// The derived key an encrypted entry uses (psdevwiki: "encryption key index" in flags2).
    pub fn key_index(&self) -> u32 {
        (self.flags2 >> 12) & 0xf
    }
}

pub struct Package {
    source: FileSlice,
    header: Header,
    entries: Vec<Entry>,
}

impl Package {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_source(FileSlice::whole(File::open(path)?)?)
    }

    pub fn from_source(source: FileSlice) -> Result<Self> {
        let header = Header::parse(&source.read_vec(0, HEADER_SIZE.min(source.len() as usize))?)?;
        let table_len = header.entry_count as usize * ENTRY_SIZE;
        let table = source.read_vec(header.entry_table_offset.into(), table_len)?;
        let entries = table
            .as_chunks::<ENTRY_SIZE>()
            .0
            .iter()
            .map(|e| Entry::parse(e))
            .collect();
        Ok(Self {
            source,
            header,
            entries,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn entry(&self, id: u32) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// The bytes of an entry stored in the clear.
    pub fn read_entry(&self, entry: &Entry) -> Result<Vec<u8>> {
        if entry.is_encrypted() {
            return Err(Error::Unsupported(format!(
                "entry {:#x} is encrypted with derived key {}",
                entry.id,
                entry.key_index()
            )));
        }
        self.source
            .read_vec(entry.offset.into(), entry.size as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_bytes() -> Vec<u8> {
        let mut b = vec![0u8; HEADER_SIZE];
        b[..4].copy_from_slice(&MAGIC);
        b[0x10..0x14].copy_from_slice(&2u32.to_be_bytes());
        b[0x18..0x1c].copy_from_slice(&0x1000u32.to_be_bytes());
        b[0x40..0x64].copy_from_slice(b"UP0000-TEST00000_00-TESTTESTTESTTEST");
        b[0x74..0x78].copy_from_slice(&0x1au32.to_be_bytes());
        b[0x408..0x410].copy_from_slice(&0x8000_0000_0000_03ccu64.to_be_bytes());
        b[0x410..0x418].copy_from_slice(&0x80000u64.to_be_bytes());
        b[0x418..0x420].copy_from_slice(&0x10a0000u64.to_be_bytes());
        b
    }

    #[test]
    fn parses_header_fields_big_endian() {
        let h = Header::parse(&header_bytes()).unwrap();
        assert_eq!(h.content_id, "UP0000-TEST00000_00-TESTTESTTESTTEST");
        assert_eq!(h.content_type, 0x1a);
        assert_eq!(h.pfs_flags, 0x8000_0000_0000_03cc);
        assert_eq!(h.pfs_image_offset, 0x80000);
        assert_eq!(h.pfs_image_size, 0x10a0000);
        assert_eq!((h.entry_count, h.entry_table_offset), (2, 0x1000));
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        let mut b = header_bytes();
        b[0] = 0;
        assert!(matches!(Header::parse(&b), Err(Error::Invalid(_))));
        assert!(matches!(
            Header::parse(&header_bytes()[..0x800]),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn entry_flags_give_encryption_and_key_index() {
        let mut raw = [0u8; ENTRY_SIZE];
        raw[..4].copy_from_slice(&0x401u32.to_be_bytes());
        raw[8..12].copy_from_slice(&0x8000_0000u32.to_be_bytes());
        raw[12..16].copy_from_slice(&0x2000u32.to_be_bytes());
        let e = Entry::parse(&raw);
        assert!(e.is_encrypted());
        assert_eq!(e.key_index(), 2);

        raw[8..12].copy_from_slice(&0u32.to_be_bytes());
        assert!(!Entry::parse(&raw).is_encrypted());
    }
}
