//! `naps_pkg_layout.dat`, the map from a PS5 inner image's logical blocks to
//! the bytes stored in `pfs_image.dat`. Section order and record sizes come
//! from pkg-to-anyps5's wiki and LibProsperoPKG's docs; the record fields were
//! worked out on two packages (see `docs/packages.md`).

use std::collections::HashMap;

use crate::Result;
use crate::error::invalid;
use crate::read::{u32_le, u64_le};

/// Logical block size of a PS5 inner image; also the most one record covers.
pub const BLOCK_SIZE: u64 = 0x40000;
/// The two halves of a block are compressed separately.
pub const CHUNK_SIZE: u64 = 0x20000;

const HEADER_SIZE: usize = 16;
const OUTER_DIGEST_SIZE: usize = 8;
const FILE_ENTRY_SIZE: usize = 6;
const U2C_SIZE: usize = 10;
const U2C_BLOCKS: u64 = 8;
const RECORD_SIZE: usize = 9;
/// Records start four bytes into the CblockInfo section.
const RECORD_SKEW: usize = 4;
const FILE_KIND_SPARSE: u8 = 0x40;
/// Bits of an anchor's size field that hold stored offset bits 34 and up.
const ANCHOR_HIGH_BITS: u32 = 0xff;

const FLAG_LZ1: u32 = 0x4;
const FLAG_SUB1: u32 = 0x2;
const FLAG_LZ2: u32 = 0x20;
const FLAG_SUB2: u32 = 0x10;

/// One row of the file-offset table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileEntry {
    /// Where the region starts in the logical (decompressed) image.
    pub offset: u64,
    /// `0x40` for zero-filled regions and on the last row, which holds the
    /// mount size; `0` for regions with stored data.
    pub kind: u8,
}

/// One CblockInfo record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    /// Sets the base for the records after it instead of describing data.
    pub anchor: bool,
    /// Where the record's bytes start in `pfs_image.dat`.
    pub offset: u64,
    /// Where the piece starts within its logical block.
    pub start: u32,
    /// Stored size of the first 128 KiB chunk.
    pub chunk1: u32,
    pub flags: u32,
    low16: u16,
}

/// How one half of a block is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    /// Stored as is, or as a single entropy-coded array.
    Plain,
    /// An LZ chunk; `sub` selects literals stored as differences.
    Lz { sub: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// The header as stored; [`Layout::compression`] and the counts are decoded from it.
    pub header: [u8; HEADER_SIZE],
    pub file_count: u32,
    pub logical_blocks: u64,
    pub outer_blocks: u64,
    /// The file-offset table, with the mount-size row last.
    pub files: Vec<FileEntry>,
    /// For each group of eight logical blocks: the index of its first
    /// record, then the offsets of the other seven from it.
    pub u2c: Vec<(u32, [u8; 7])>,
    pub records: Vec<Record>,
    /// Stored size of each piece that starts at a given offset.
    sizes: HashMap<u64, u32>,
}

impl Layout {
    pub fn parse(b: &[u8]) -> Result<Self> {
        let header: [u8; HEADER_SIZE] = b
            .get(..HEADER_SIZE)
            .ok_or_else(|| invalid("NAPS layout is truncated"))?
            .try_into()
            .unwrap();
        let file_count = u32_le(&header, 0) & 0xff_ffff;
        let logical_blocks = u64::from(u32_le(&header, 4));
        let outer_blocks = u64_le(&header, 8) & 0xff_ffff;

        let mut at = HEADER_SIZE + outer_blocks as usize * OUTER_DIGEST_SIZE;
        let rows = file_count as usize + 1;
        let table = b
            .get(at..at + rows * FILE_ENTRY_SIZE)
            .ok_or_else(|| invalid("NAPS file-offset table runs past the end"))?;
        let files: Vec<FileEntry> = table
            .as_chunks::<FILE_ENTRY_SIZE>()
            .0
            .iter()
            .map(|e| FileEntry {
                offset: u64_le(&[&e[..5], &[0; 3]].concat(), 0),
                kind: e[5],
            })
            .collect();
        let end = files.last().unwrap();
        let last_block = logical_blocks.saturating_sub(1) * BLOCK_SIZE;
        if end.kind != FILE_KIND_SPARSE
            || end.offset <= last_block
            || end.offset > logical_blocks * BLOCK_SIZE
        {
            return Err(invalid(
                "NAPS file-offset table does not end at the mount size",
            ));
        }
        if files.windows(2).any(|w| w[1].offset < w[0].offset) {
            return Err(invalid("NAPS file offsets are out of order"));
        }
        at += rows * FILE_ENTRY_SIZE;

        let groups = logical_blocks.div_ceil(U2C_BLOCKS) as usize;
        let u2c_bytes = b
            .get(at..at + groups * U2C_SIZE)
            .ok_or_else(|| invalid("NAPS u2c table runs past the end"))?;
        let u2c: Vec<(u32, [u8; 7])> = u2c_bytes
            .as_chunks::<U2C_SIZE>()
            .0
            .iter()
            .map(|e| {
                (
                    u32_le(&[&e[..3], &[0]].concat(), 0),
                    e[3..].try_into().unwrap(),
                )
            })
            .collect();
        if u2c.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err(invalid("NAPS u2c table is out of order"));
        }
        at += groups * U2C_SIZE;

        let count = (b.len() - at) / RECORD_SIZE;
        let records = parse_records(b, at, count);
        let last = u2c.last().map_or(0, |(base, d)| base + u32::from(d[6]));
        if last as usize >= records.len() {
            return Err(invalid("NAPS u2c table points past the records"));
        }
        let sizes = piece_sizes(&records);
        Ok(Self {
            header,
            file_count,
            logical_blocks,
            outer_blocks,
            files,
            u2c,
            records,
            sizes,
        })
    }

    /// The codec byte of the header: 2 in every package seen (both Kraken).
    pub fn compression(&self) -> u8 {
        self.header[3]
    }

    /// The size of the logical image the files are laid out in. It need not
    /// fill its last 256 KiB block.
    pub fn mount_size(&self) -> u64 {
        self.files.last().unwrap().offset
    }

    /// The index of the first record of a logical block.
    pub fn first_record(&self, block: u64) -> Option<u32> {
        let (base, deltas) = self.u2c.get((block / U2C_BLOCKS) as usize)?;
        match block % U2C_BLOCKS {
            0 => Some(*base),
            i => Some(base + u32::from(deltas[i as usize - 1])),
        }
    }

    /// The region of the file-offset table that holds a logical offset.
    pub fn region_at(&self, offset: u64) -> Option<usize> {
        let i = self.files.partition_point(|f| f.offset <= offset);
        (i > 0 && i < self.files.len()).then(|| i - 1)
    }

    /// The bounds of a region and whether it is zero-filled.
    pub fn region(&self, index: usize) -> (u64, u64, bool) {
        let f = self.files[index];
        (
            f.offset,
            self.files[index + 1].offset,
            f.kind == FILE_KIND_SPARSE,
        )
    }

    /// The record of the piece that starts at a logical offset.
    pub fn piece(&self, logical: u64) -> Option<&Record> {
        let block = logical / BLOCK_SIZE;
        let start = (logical % BLOCK_SIZE) as u32;
        let first = self.first_record(block)? as usize;
        let last = match self.first_record(block + 1) {
            Some(n) => n as usize,
            None => self.records.len(),
        };
        self.records
            .get(first..last.min(self.records.len()))?
            .iter()
            .find(|r| !r.anchor && r.start == start)
    }

    /// The stored size of a piece, when the records give it.
    pub fn piece_size(&self, record: &Record) -> Option<u32> {
        self.sizes.get(&record.offset).copied()
    }

    /// How each half of a piece is stored.
    pub fn chunk_kinds(record: &Record) -> (ChunkKind, ChunkKind) {
        let kind = |lz, sub| {
            if record.flags & lz != 0 {
                ChunkKind::Lz {
                    sub: record.flags & sub != 0,
                }
            } else {
                ChunkKind::Plain
            }
        };
        (kind(FLAG_LZ1, FLAG_SUB1), kind(FLAG_LZ2, FLAG_SUB2))
    }
}

/// Reads the records and resolves each one's stored offset: anchors carry it
/// whole, the rest only its low 18 bits.
fn parse_records(b: &[u8], at: usize, count: usize) -> Vec<Record> {
    let mut out = Vec::with_capacity(count);
    let mut reference = 0u64;
    for r in 0..count {
        let from = at + r * RECORD_SIZE + RECORD_SKEW;
        let mut raw = [0u8; 16];
        let avail = b.len().saturating_sub(from).min(RECORD_SIZE);
        raw[..avail].copy_from_slice(&b[from..from + avail]);
        let v = u128::from_le_bytes(raw);
        let low16 = (v & 0xffff) as u16;
        let nibble = ((v >> 16) & 0xf) as u32;
        let start = ((v >> 20) & 0x3_ffff) as u32;
        let size_field = ((v >> 38) & 0x1_ffff) as u32;
        let flags = ((v >> 55) & 0x1_ffff) as u32;
        let anchor = nibble & 4 != 0;
        let offset = if anchor {
            (u64::from(size_field & ANCHOR_HIGH_BITS) << 34)
                | (u64::from(start) << 16)
                | u64::from(low16)
        } else {
            let low = (u64::from(nibble & 3) << 16) | u64::from(low16);
            let base = reference.saturating_sub(0x10000);
            let mut x = (base & !0x3_ffff) | low;
            if x < base {
                x += 0x4_0000;
            }
            x
        };
        reference = if anchor {
            offset
        } else {
            offset + u64::from(size_field) + 1
        };
        out.push(Record {
            anchor,
            offset,
            start,
            chunk1: size_field + 1,
            flags,
            low16,
        });
    }
    out
}

/// Stored piece sizes; an anchor after a piece gives its end's low 16 bits,
/// bounded by the next piece stored after it.
fn piece_sizes(records: &[Record]) -> HashMap<u64, u32> {
    let mut sizes = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        let Some(next) = records.get(i + 1) else {
            break;
        };
        if r.anchor {
            continue;
        }
        let end = if next.anchor {
            let low = r.offset + u64::from(r.chunk1).min(CHUNK_SIZE);
            let mut end = (low & !0xffff) | u64::from(next.low16);
            if end < low {
                end += 0x1_0000;
            }
            let bound = records[i + 2..]
                .iter()
                .find(|x| !x.anchor && x.offset > r.offset)
                .map(|x| x.offset)
                .filter(|&o| o <= r.offset + BLOCK_SIZE + 0x1_0000);
            if let Some(bound) = bound {
                while end + 0x1_0000 <= bound {
                    end += 0x1_0000;
                }
            }
            end
        } else {
            next.offset
        };
        if let Some(d) = end
            .checked_sub(r.offset)
            .filter(|d| (1..=BLOCK_SIZE).contains(d))
        {
            sizes.entry(r.offset).or_insert(d as u32);
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(files: &[u64], blocks: u32, outer: u32) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_SIZE];
        b[..4].copy_from_slice(&(files.len() as u32 | 2 << 24).to_le_bytes());
        b[4..8].copy_from_slice(&blocks.to_le_bytes());
        b[8..12].copy_from_slice(&outer.to_le_bytes());
        b.resize(b.len() + outer as usize * OUTER_DIGEST_SIZE, 0);
        for &f in files {
            b.extend(&f.to_le_bytes()[..5]);
            b.push(0);
        }
        b.extend(&(u64::from(blocks) * BLOCK_SIZE).to_le_bytes()[..5]);
        b.push(FILE_KIND_SPARSE);
        let mut next = 1u32;
        for _ in 0..u64::from(blocks).div_ceil(U2C_BLOCKS) {
            b.extend(&next.to_le_bytes()[..3]);
            b.extend([1, 2, 3, 4, 5, 6, 7]);
            next += 8;
        }
        b.resize(b.len() + next as usize * RECORD_SIZE, 0);
        b
    }

    /// Writes a record's 72 bits at its (skewed) place in the table.
    fn put_record(b: &mut [u8], table: usize, r: usize, v: u128) {
        let at = table + r * RECORD_SIZE + RECORD_SKEW;
        let bytes = v.to_le_bytes();
        let n = (b.len() - at).min(RECORD_SIZE);
        b[at..at + n].copy_from_slice(&bytes[..n]);
    }

    fn record(anchor_or_low18: (bool, u64), start: u32, chunk1: u32, flags: u32) -> u128 {
        let (anchor, value) = anchor_or_low18;
        let (low16, nibble, field_start, size_field) = if anchor {
            (
                value & 0xffff,
                4,
                (value >> 16) & 0x3_ffff,
                (value >> 34) & u64::from(ANCHOR_HIGH_BITS),
            )
        } else {
            (
                value & 0xffff,
                (value >> 16) & 3,
                u64::from(start),
                u64::from(chunk1 - 1),
            )
        };
        u128::from(low16)
            | u128::from(nibble) << 16
            | u128::from(field_start) << 20
            | u128::from(size_field) << 38
            | u128::from(flags) << 55
    }

    #[test]
    fn parses_counts_tables_and_block_lookup() {
        let l = Layout::parse(&layout(&[0, 0x31d0, 0x31da], 10, 3)).unwrap();
        assert_eq!((l.file_count, l.logical_blocks, l.outer_blocks), (3, 10, 3));
        assert_eq!(l.compression(), 2);
        assert_eq!(l.mount_size(), 10 * BLOCK_SIZE);
        assert_eq!(l.files[1].offset, 0x31d0);
        assert_eq!(l.first_record(0), Some(1));
        assert_eq!(l.first_record(3), Some(4));
        assert_eq!(l.region_at(0x31d5), Some(1));
        assert_eq!(l.region(2), (0x31da, 10 * BLOCK_SIZE, false));
    }

    #[test]
    fn resolves_offsets_from_anchors_and_low_bits() {
        let mut b = layout(&[0], 3, 0);
        let table = b.len() - 9 * RECORD_SIZE;
        put_record(&mut b, table, 0, record((true, 0x4_0001_0000), 0, 1, 0));
        put_record(&mut b, table, 1, record((false, 0x1_2000), 0, 0x100, 0x5));
        put_record(&mut b, table, 2, record((false, 0x1_2100), 0, 0x200, 0x5));
        put_record(&mut b, table, 3, record((false, 0x0_0100), 0x10, 0x10, 0x5));
        let l = Layout::parse(&b).unwrap();
        assert!(l.records[0].anchor);
        assert_eq!(l.records[0].offset, 0x4_0001_0000);
        assert_eq!(l.records[1].offset, 0x4_0001_2000);
        assert_eq!(l.records[2].offset, 0x4_0001_2100);
        assert_eq!(l.records[3].offset, 0x4_0004_0100);
        assert_eq!(l.piece_size(&l.records[1]), Some(0x100));
        assert_eq!(l.piece(0).map(|r| r.offset), Some(0x4_0001_2000));
        let kinds = Layout::chunk_kinds(&l.records[1]);
        assert_eq!(kinds, (ChunkKind::Lz { sub: false }, ChunkKind::Plain));
    }

    #[test]
    fn reads_anchors_past_64_gib() {
        let mut b = layout(&[0], 3, 0);
        let table = b.len() - 9 * RECORD_SIZE;
        put_record(&mut b, table, 0, record((true, 0x18_0a87_0000), 0, 1, 0));
        put_record(&mut b, table, 1, record((false, 0x7_2000), 0, 0x100, 0x5));
        let l = Layout::parse(&b).unwrap();
        assert_eq!(l.records[0].offset, 0x18_0a87_0000);
        assert_eq!(l.records[1].offset, 0x18_0a87_2000);
    }

    #[test]
    fn mount_size_may_end_inside_the_last_block() {
        let mut b = layout(&[0, 0x31d0], 3, 0);
        let last = HEADER_SIZE + 2 * FILE_ENTRY_SIZE;
        let short = 2 * BLOCK_SIZE + CHUNK_SIZE;
        b[last..last + 5].copy_from_slice(&short.to_le_bytes()[..5]);
        let l = Layout::parse(&b).unwrap();
        assert_eq!(l.mount_size(), short);
        assert_eq!(l.region(1), (0x31d0, short, false));
        b[last..last + 5].copy_from_slice(&(2 * BLOCK_SIZE).to_le_bytes()[..5]);
        assert!(Layout::parse(&b).is_err());
    }

    #[test]
    fn rejects_a_table_that_misses_the_mount_size() {
        let mut b = layout(&[0], 2, 0);
        b[HEADER_SIZE + 6 + 5] = 0;
        assert!(Layout::parse(&b).is_err());
        assert!(Layout::parse(&b[..10]).is_err());
    }
}
