//! The PFS file system inside packages. Layout facts come from LibOrbisPkg's
//! `PfsStructs.cs` and `PfsReader.cs` (read for facts, see `docs/references.md`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::invalid;
use crate::keys::PfsKeys;
use crate::pfsc::PfscReader;
use crate::read::{ReadAt, u16_le, u32_le, u64_le};
use crate::xts::{SECTOR_SIZE, XtsReader};
use crate::{Error, Result};

const MAGIC: u64 = 20130315;
const SEED_OFFSET: usize = 0x370;
const SUPERBLOCK_READ: usize = 0x400;

const MODE_SIGNED: u16 = 0x1;
const MODE_64BIT: u16 = 0x2;
const MODE_ENCRYPTED: u16 = 0x4;

const INODE_UNSIGNED_SIZE: usize = 0xa8;
const INODE_SIGNED_SIZE: usize = 0x2c8;
const INODE_FLAG_COMPRESSED: u32 = 0x1;
const DIRECT_BLOCKS: usize = 12;
const INDIRECT_BLOCKS: usize = 5;
const SIGNED_POINTER: usize = 36;
const NO_BLOCK: u32 = u32::MAX;

const DIRENT_FILE: u32 = 2;
const DIRENT_DIR: u32 = 3;
const DIRENT_HEADER: usize = 16;

/// The superroot directory; user files live in its `uroot` child.
const SUPERROOT: u32 = 0;

type Source = Arc<dyn ReadAt + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superblock {
    pub mode: u16,
    pub block_size: u32,
    pub dinode_count: u64,
    pub dinode_block_count: u64,
    pub seed: [u8; 16],
}

impl Superblock {
    fn parse(b: &[u8]) -> Result<Self> {
        if u64_le(b, 0) != 1 || u64_le(b, 8) != MAGIC {
            return Err(invalid("not a PFS image (bad version or magic)"));
        }
        let block_size = u32_le(b, 0x20);
        if block_size == 0 || u64::from(block_size) % SECTOR_SIZE != 0 {
            return Err(invalid(format!(
                "PFS block size {block_size:#x} is not a multiple of 0x1000"
            )));
        }
        Ok(Self {
            mode: u16_le(b, 0x1c),
            block_size,
            dinode_count: u64_le(b, 0x30),
            dinode_block_count: u64_le(b, 0x40),
            seed: b[SEED_OFFSET..SEED_OFFSET + 16].try_into().unwrap(),
        })
    }

    pub fn is_signed(&self) -> bool {
        self.mode & MODE_SIGNED != 0
    }

    pub fn is_encrypted(&self) -> bool {
        self.mode & MODE_ENCRYPTED != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inode {
    pub flags: u32,
    /// Bytes stored in the file's blocks.
    pub size: u64,
    /// Named after LibOrbisPkg's field. For the PFSC file in vic's package it
    /// holds the decompressed length, which is smaller than `size`.
    pub size_compressed: u64,
    pub blocks: u32,
    direct: [u32; DIRECT_BLOCKS],
    indirect: [u32; INDIRECT_BLOCKS],
}

impl Inode {
    fn parse(b: &[u8], signed: bool) -> Self {
        let stride = if signed { SIGNED_POINTER } else { 4 };
        let skip = if signed { 32 } else { 0 };
        let ptr = |i: usize| u32_le(b, 0x64 + i * stride + skip);
        Self {
            flags: u32_le(b, 0x04),
            size: u64_le(b, 0x08),
            size_compressed: u64_le(b, 0x10),
            blocks: u32_le(b, 0x60),
            direct: core::array::from_fn(ptr),
            indirect: core::array::from_fn(|i| ptr(DIRECT_BLOCKS + i)),
        }
    }

    pub fn is_compressed(&self) -> bool {
        self.flags & INODE_FLAG_COMPRESSED != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    File,
    Dir,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub inode: u32,
    pub kind: Kind,
}

pub struct Pfs {
    source: Source,
    superblock: Superblock,
    inodes: Vec<Inode>,
}

impl Pfs {
    /// Opens a PFS image. Encrypted images need the package's EKPFS, and
    /// `new_crypt` selects the variant key schedule (see [`PfsKeys::derive`]).
    pub fn open(raw: Source, ekpfs: Option<&[u8; 32]>, new_crypt: bool) -> Result<Self> {
        let superblock = Superblock::parse(&raw.read_vec(0, SUPERBLOCK_READ)?)?;
        if superblock.mode & MODE_64BIT != 0 {
            return Err(Error::Unsupported("PFS images with 64-bit inodes".into()));
        }
        let source: Source = if superblock.is_encrypted() {
            let ekpfs =
                ekpfs.ok_or_else(|| invalid("PFS image is encrypted but no key was given"))?;
            let keys = PfsKeys::derive(ekpfs, &superblock.seed, new_crypt);
            let first = u64::from(superblock.block_size) / SECTOR_SIZE;
            Arc::new(XtsReader::new(raw, &keys, first))
        } else {
            raw
        };
        let mut pfs = Self {
            source,
            superblock,
            inodes: Vec::new(),
        };
        pfs.inodes = pfs.read_inodes()?;
        Ok(pfs)
    }

    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    fn block_size(&self) -> u64 {
        self.superblock.block_size.into()
    }

    fn read_inodes(&self) -> Result<Vec<Inode>> {
        let signed = self.superblock.is_signed();
        let size = if signed {
            INODE_SIGNED_SIZE
        } else {
            INODE_UNSIGNED_SIZE
        };
        let per_block = self.block_size() as usize / size;
        let count = self.superblock.dinode_count;
        if count > self.superblock.dinode_block_count * per_block as u64 {
            return Err(invalid("PFS inode count does not fit its inode blocks"));
        }
        let mut inodes = Vec::with_capacity(count as usize);
        for block in 0..self.superblock.dinode_block_count {
            let data = self
                .source
                .read_vec((1 + block) * self.block_size(), self.block_size() as usize)?;
            for chunk in data.chunks_exact(size).take(per_block) {
                if inodes.len() as u64 == count {
                    return Ok(inodes);
                }
                inodes.push(Inode::parse(chunk, signed));
            }
        }
        Ok(inodes)
    }

    pub fn inode(&self, number: u32) -> Result<&Inode> {
        self.inodes
            .get(number as usize)
            .ok_or_else(|| invalid(format!("inode {number} does not exist")))
    }

    /// The data blocks of an inode, in order.
    fn block_list(&self, inode: &Inode) -> Result<Vec<u32>> {
        let n = inode.blocks as usize;
        if n <= 1 || inode.direct[1] == NO_BLOCK {
            return Ok((0..inode.blocks).map(|i| inode.direct[0] + i).collect());
        }
        if !self.superblock.is_signed() {
            return Err(Error::Unsupported(
                "non-contiguous files in an unsigned PFS".into(),
            ));
        }
        let mut list: Vec<u32> = inode.direct.iter().copied().take(n).collect();
        for (level, &ib) in (1..).zip(&inode.indirect) {
            if list.len() >= n {
                break;
            }
            self.collect_indirect(ib, level, n, &mut list)?;
        }
        if list.len() < n {
            return Err(invalid("PFS file has more blocks than its pointers reach"));
        }
        Ok(list)
    }

    /// Appends the blocks reachable through an indirect block of `level`
    /// (1 points at data blocks, 2 at level-1 blocks, ...).
    fn collect_indirect(
        &self,
        block: u32,
        level: u32,
        want: usize,
        list: &mut Vec<u32>,
    ) -> Result<()> {
        let data = self.source.read_vec(
            u64::from(block) * self.block_size(),
            self.block_size() as usize,
        )?;
        for ptr in data.as_chunks::<SIGNED_POINTER>().0 {
            if list.len() >= want {
                break;
            }
            let target = u32_le(ptr, 32);
            if level == 1 {
                list.push(target);
            } else {
                self.collect_indirect(target, level - 1, want, list)?;
            }
        }
        Ok(())
    }

    /// The bytes an inode stores, as laid out on disk (still compressed for
    /// compressed files).
    fn stored(&self, inode: &Inode, len: u64) -> Result<BlockFile> {
        let blocks = self.block_list(inode)?;
        if (blocks.len() as u64) * self.block_size() < len {
            return Err(invalid("PFS file is longer than its blocks"));
        }
        Ok(BlockFile {
            source: self.source.clone(),
            blocks,
            block_size: self.block_size(),
            len,
        })
    }

    /// The contents of a file, decompressed if it is a PFSC file.
    pub fn open_file(&self, number: u32) -> Result<Source> {
        let inode = self.inode(number)?;
        let stored = self.stored(inode, inode.size)?;
        if inode.is_compressed() && inode.size_compressed != inode.size {
            Ok(Arc::new(PfscReader::new(stored)?))
        } else {
            Ok(Arc::new(stored))
        }
    }

    pub fn read_dir(&self, number: u32) -> Result<Vec<DirEntry>> {
        let inode = self.inode(number)?;
        let mut entries = Vec::new();
        for block in self.block_list(inode)? {
            let data = self.source.read_vec(
                u64::from(block) * self.block_size(),
                self.block_size() as usize,
            )?;
            let mut at = 0;
            while at + DIRENT_HEADER <= data.len() {
                let ino = u32_le(&data, at);
                let kind = u32_le(&data, at + 4);
                let name_len = u32_le(&data, at + 8) as usize;
                let ent_size = u32_le(&data, at + 12) as usize;
                if ent_size == 0 {
                    break;
                }
                let name = data
                    .get(at + DIRENT_HEADER..at + DIRENT_HEADER + name_len)
                    .ok_or_else(|| invalid("PFS directory entry name runs past its block"))?;
                let name = String::from_utf8(name.to_vec())
                    .map_err(|_| invalid("PFS file name is not UTF-8"))?;
                let kind = match kind {
                    DIRENT_FILE => Some(Kind::File),
                    DIRENT_DIR => Some(Kind::Dir),
                    _ => None,
                };
                if let Some(kind) = kind {
                    entries.push(DirEntry {
                        name,
                        inode: ino,
                        kind,
                    });
                }
                at += ent_size;
            }
        }
        Ok(entries)
    }

    /// The `uroot` directory, where a package's files start.
    pub fn uroot(&self) -> Result<u32> {
        self.read_dir(SUPERROOT)?
            .into_iter()
            .find(|e| e.name == "uroot" && e.kind == Kind::Dir)
            .map(|e| e.inode)
            .ok_or_else(|| invalid("PFS image has no uroot"))
    }

    /// Looks a file up by a `/`-separated path below `uroot`.
    pub fn lookup(&self, path: &str) -> Result<DirEntry> {
        let mut current = DirEntry {
            name: String::new(),
            inode: self.uroot()?,
            kind: Kind::Dir,
        };
        for part in path.split('/').filter(|p| !p.is_empty()) {
            if current.kind != Kind::Dir {
                return Err(invalid(format!("{} is not a directory", current.name)));
            }
            current = self
                .read_dir(current.inode)?
                .into_iter()
                .find(|e| e.name == part)
                .ok_or_else(|| invalid(format!("no {path} in the PFS image")))?;
        }
        Ok(current)
    }

    /// Every file below `uroot`, with its path relative to it.
    pub fn files(&self) -> Result<Vec<(PathBuf, u32)>> {
        let mut out = Vec::new();
        self.walk(self.uroot()?, Path::new(""), 0, &mut out)?;
        Ok(out)
    }

    fn walk(
        &self,
        dir: u32,
        prefix: &Path,
        depth: u32,
        out: &mut Vec<(PathBuf, u32)>,
    ) -> Result<()> {
        if depth > 64 {
            return Err(invalid("PFS directories nest too deep (a cycle?)"));
        }
        for e in self.read_dir(dir)? {
            if e.name.contains('/') || e.name == "." || e.name == ".." || e.name.is_empty() {
                return Err(invalid(format!(
                    "unsafe file name {:?} in the PFS image",
                    e.name
                )));
            }
            let path = prefix.join(&e.name);
            match e.kind {
                Kind::File => out.push((path, e.inode)),
                Kind::Dir => self.walk(e.inode, &path, depth + 1, out)?,
            }
        }
        Ok(())
    }
}

/// A file's stored bytes, gathered from its blocks.
struct BlockFile {
    source: Source,
    blocks: Vec<u32>,
    block_size: u64,
    len: u64,
}

impl ReadAt for BlockFile {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(buf.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(invalid("read past the end of a PFS file"));
        }
        let mut done = 0;
        while done < buf.len() {
            let pos = offset + done as u64;
            let index = (pos / self.block_size) as usize;
            let within = pos % self.block_size;
            let n = ((self.block_size - within) as usize).min(buf.len() - done);
            let at = u64::from(self.blocks[index]) * self.block_size + within;
            self.source.read_exact_at(at, &mut buf[done..done + n])?;
            done += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: usize = 0x1000;

    /// A small PFS image, built block by block.
    struct Image {
        blocks: Vec<Vec<u8>>,
        signed: bool,
        inodes: Vec<Vec<u8>>,
    }

    impl Image {
        fn new(signed: bool) -> Self {
            Self {
                blocks: vec![vec![0; BS], vec![0; BS]],
                signed,
                inodes: Vec::new(),
            }
        }

        fn alloc(&mut self, data: &[u8]) -> u32 {
            let first = self.blocks.len() as u32;
            for chunk in data.chunks(BS) {
                let mut b = chunk.to_vec();
                b.resize(BS, 0);
                self.blocks.push(b);
            }
            first
        }

        fn inode(&mut self, flags: u32, size: u64, blocks: &[u32], indirect: &[u32]) -> u32 {
            let len = if self.signed {
                INODE_SIGNED_SIZE
            } else {
                INODE_UNSIGNED_SIZE
            };
            let stride = if self.signed { SIGNED_POINTER } else { 4 };
            let skip = if self.signed { 32 } else { 0 };
            let mut b = vec![0u8; len];
            b[4..8].copy_from_slice(&flags.to_le_bytes());
            b[8..16].copy_from_slice(&size.to_le_bytes());
            b[16..24].copy_from_slice(&size.to_le_bytes());
            b[0x60..0x64].copy_from_slice(&(blocks.len() as u32).to_le_bytes());
            for i in 0..DIRECT_BLOCKS + INDIRECT_BLOCKS {
                let v = if i < DIRECT_BLOCKS {
                    blocks.get(i).copied()
                } else {
                    indirect.get(i - DIRECT_BLOCKS).copied()
                };
                let at = 0x64 + i * stride + skip;
                b[at..at + 4].copy_from_slice(&v.unwrap_or(NO_BLOCK).to_le_bytes());
            }
            if blocks.len() > 1 && !self.signed {
                b[0x64 + 4..0x64 + 8].copy_from_slice(&NO_BLOCK.to_le_bytes());
            }
            self.inodes.push(b);
            self.inodes.len() as u32 - 1
        }

        fn file(&mut self, data: &[u8]) -> u32 {
            let first = self.alloc(data);
            let n = data.len().div_ceil(BS) as u32;
            let blocks: Vec<u32> = (first..first + n).collect();
            self.inode(0, data.len() as u64, &blocks, &[])
        }

        fn dir(&mut self, entries: &[(&str, u32, u32)]) -> u32 {
            let mut d = Vec::new();
            for (name, ino, kind) in entries {
                let size = (DIRENT_HEADER + name.len() + 1).next_multiple_of(8);
                let mut e = vec![0u8; size];
                e[0..4].copy_from_slice(&ino.to_le_bytes());
                e[4..8].copy_from_slice(&kind.to_le_bytes());
                e[8..12].copy_from_slice(&(name.len() as u32).to_le_bytes());
                e[12..16].copy_from_slice(&(size as u32).to_le_bytes());
                e[16..16 + name.len()].copy_from_slice(name.as_bytes());
                d.extend(e);
            }
            let block = self.alloc(&d);
            self.inode(0, BS as u64, &[block], &[])
        }

        fn finish(mut self, inode_order: &[u32]) -> Vec<u8> {
            let mut table = Vec::new();
            for &i in inode_order {
                table.extend_from_slice(&self.inodes[i as usize]);
            }
            assert!(table.len() <= BS, "test image needs one inode block");
            table.resize(BS, 0);
            self.blocks[1] = table;
            let sb = &mut self.blocks[0];
            sb[0..8].copy_from_slice(&1u64.to_le_bytes());
            sb[8..16].copy_from_slice(&MAGIC.to_le_bytes());
            let mode: u16 = 0x8 | if self.signed { MODE_SIGNED } else { 0 };
            sb[0x1c..0x1e].copy_from_slice(&mode.to_le_bytes());
            sb[0x20..0x24].copy_from_slice(&(BS as u32).to_le_bytes());
            sb[0x30..0x38].copy_from_slice(&(inode_order.len() as u64).to_le_bytes());
            sb[0x40..0x48].copy_from_slice(&1u64.to_le_bytes());
            self.blocks.concat()
        }
    }

    fn open(img: Vec<u8>) -> Pfs {
        Pfs::open(Arc::new(img), None, false).unwrap()
    }

    /// Inodes are created leaves first, so `finish` puts them in the order the
    /// directory entries name them.
    fn tree(signed: bool) -> (Vec<u8>, Vec<u8>) {
        let big: Vec<u8> = (0..BS * 2 + 10).map(|i| (i % 253) as u8).collect();
        let mut img = Image::new(signed);
        let a = img.file(b"hello");
        let b = img.file(&big);
        let sub = img.dir(&[("b.bin", 4, DIRENT_FILE)]);
        let uroot = img.dir(&[
            (".", 1, 4),
            ("a.txt", 2, DIRENT_FILE),
            ("sub", 3, DIRENT_DIR),
        ]);
        let superroot = img.dir(&[("uroot", 1, DIRENT_DIR)]);
        (img.finish(&[superroot, uroot, a, sub, b]), big)
    }

    #[test]
    fn walks_uroot_and_reads_files() {
        let (img, big) = tree(false);
        let pfs = open(img);
        let mut files = pfs.files().unwrap();
        files.sort();
        assert_eq!(
            files,
            [(PathBuf::from("a.txt"), 2), (PathBuf::from("sub/b.bin"), 4)]
        );
        assert_eq!(pfs.open_file(2).unwrap().read_vec(0, 5).unwrap(), b"hello");
        let f = pfs.open_file(4).unwrap();
        assert_eq!(f.len(), big.len() as u64);
        assert_eq!(f.read_vec(0, big.len()).unwrap(), big);
    }

    #[test]
    fn looks_up_paths_below_uroot() {
        let (img, _) = tree(true);
        let pfs = open(img);
        assert_eq!(pfs.lookup("sub/b.bin").unwrap().inode, 4);
        assert!(pfs.lookup("sub/missing").is_err());
        assert!(pfs.lookup("a.txt/x").is_err());
    }

    #[test]
    fn follows_indirect_blocks_in_signed_images() {
        let mut img = Image::new(true);
        let n = DIRECT_BLOCKS + 2;
        let data: Vec<u8> = (0..n * BS).map(|i| (i / BS) as u8).collect();
        let first = img.alloc(&data);
        // Store the blocks in reverse so the list is not contiguous.
        let mut blocks: Vec<u32> = (first..first + n as u32).rev().collect();
        let tail = blocks.split_off(DIRECT_BLOCKS);
        let mut ptrs = Vec::new();
        for t in &tail {
            ptrs.extend([0u8; 32]);
            ptrs.extend(t.to_le_bytes());
        }
        let ib = img.alloc(&ptrs);
        let mut all = blocks.clone();
        all.extend(&tail);
        let file = img.inode(0, (n * BS) as u64, &all, &[ib]);
        let uroot = img.dir(&[("f", 2, DIRENT_FILE)]);
        let superroot = img.dir(&[("uroot", 1, DIRENT_DIR)]);
        let pfs = open(img.finish(&[superroot, uroot, file]));
        let f = pfs.open_file(2).unwrap();
        let read = f.read_vec(0, n * BS).unwrap();
        let expected: Vec<u8> = (0..n)
            .rev()
            .flat_map(|b| std::iter::repeat_n(b as u8, BS))
            .collect();
        assert_eq!(read, expected);
    }

    #[test]
    fn rejects_unsafe_names() {
        let mut img = Image::new(false);
        let a = img.file(b"x");
        let uroot = img.dir(&[("..", 0, DIRENT_FILE), ("ok", 2, DIRENT_FILE)]);
        let superroot = img.dir(&[("uroot", 1, DIRENT_DIR)]);
        let pfs = open(img.finish(&[superroot, uroot, a]));
        assert!(pfs.files().is_err());
    }

    #[test]
    fn encrypted_images_need_a_key() {
        let (mut img, _) = tree(false);
        img[0x1c] |= MODE_ENCRYPTED as u8;
        assert!(Pfs::open(Arc::new(img), None, false).is_err());
    }
}
