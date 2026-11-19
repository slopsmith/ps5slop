//! ELF files, plain or inside a fake-signed SELF container. Layout facts are
//! in `docs/loader.md`.

use crate::error::invalid;
use crate::{Error, Result};

pub const ET_SCE_EXEC: u16 = 0xfe00;
pub const ET_SCE_STUBLIB: u16 = 0xfe0c;
pub const ET_SCE_DYNEXEC: u16 = 0xfe10;
pub const ET_SCE_DYNAMIC: u16 = 0xfe18;

pub const PT_LOAD: u32 = 1;
pub const PT_DYNAMIC: u32 = 2;
pub const PT_INTERP: u32 = 3;
pub const PT_TLS: u32 = 7;
pub const PT_SCE_RELA: u32 = 0x6000_0000;
pub const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;
pub const PT_SCE_PROCPARAM: u32 = 0x6100_0001;
pub const PT_SCE_MODULE_PARAM: u32 = 0x6100_0002;
pub const PT_SCE_RELRO: u32 = 0x6100_0010;
pub const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
pub const PT_SCE_COMMENT: u32 = 0x6fff_ff00;
pub const PT_SCE_LIBVERSION: u32 = 0x6fff_ff01;

pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

pub const EM_X86_64: u16 = 62;
pub const ELF_HEADER_SIZE: usize = 0x40;
pub const PHDR_SIZE: usize = 0x38;

const SELF_MAGIC_PS4: u32 = 0x1d3d_154f;
const SELF_MAGIC_PS5: u32 = 0xeef5_1454;
const SELF_HEADER_SIZE: usize = 0x20;
const SELF_SEGMENT_SIZE: usize = 0x20;
const SELF_BLOCKED: u64 = 0x800;
const SELF_ENCRYPTED: u64 = 0x2;
const SELF_COMPRESSED: u64 = 0x8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElfHeader {
    pub e_type: u16,
    pub entry: u64,
    phoff: u64,
    phnum: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramHeader {
    pub p_type: u32,
    pub flags: u32,
    pub offset: u64,
    pub vaddr: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub align: u64,
}

impl ProgramHeader {
    fn parse(b: &[u8]) -> Self {
        Self {
            p_type: u32_le(b, 0x00),
            flags: u32_le(b, 0x04),
            offset: u64_le(b, 0x08),
            vaddr: u64_le(b, 0x10),
            filesz: u64_le(b, 0x20),
            memsz: u64_le(b, 0x28),
            align: u64_le(b, 0x30),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Elf,
    /// A fake-signed SELF: plaintext segments behind a SELF header.
    SelfPs4,
    SelfPs5,
}

#[derive(Debug, Clone, Copy)]
struct SelfSegment {
    flags: u64,
    file_offset: u64,
    file_size: u64,
}

impl SelfSegment {
    fn program_header(&self) -> usize {
        ((self.flags >> 20) & 0xfff) as usize
    }
}

/// An executable or module, with its program headers and a way to get at each
/// segment's file contents.
pub struct ElfFile {
    data: Vec<u8>,
    container: Container,
    header: ElfHeader,
    phdrs: Vec<ProgramHeader>,
    self_segments: Vec<SelfSegment>,
}

impl ElfFile {
    pub fn parse(data: Vec<u8>) -> Result<Self> {
        let magic = data
            .get(..4)
            .map(|m| u32::from_le_bytes(m.try_into().unwrap()))
            .ok_or_else(|| invalid("file is too short"))?;
        let (container, elf_at, self_segments) = match magic {
            SELF_MAGIC_PS4 | SELF_MAGIC_PS5 => {
                let header = get(&data, 0, SELF_HEADER_SIZE)?;
                let count = u16_le(header, 0x18) as usize;
                let table = get(&data, SELF_HEADER_SIZE, count * SELF_SEGMENT_SIZE)?;
                let segments = table
                    .as_chunks::<SELF_SEGMENT_SIZE>()
                    .0
                    .iter()
                    .map(|s| SelfSegment {
                        flags: u64_le(s, 0),
                        file_offset: u64_le(s, 8),
                        file_size: u64_le(s, 16),
                    })
                    .collect();
                let container = if magic == SELF_MAGIC_PS4 {
                    Container::SelfPs4
                } else {
                    Container::SelfPs5
                };
                (
                    container,
                    SELF_HEADER_SIZE + count * SELF_SEGMENT_SIZE,
                    segments,
                )
            }
            _ => (Container::Elf, 0, Vec::new()),
        };

        let eh = get(&data, elf_at, ELF_HEADER_SIZE)?;
        if &eh[..4] != b"\x7fELF" {
            return Err(invalid("not an ELF file"));
        }
        if eh[4] != 2 || eh[5] != 1 {
            return Err(Error::Unsupported("only 64-bit little-endian ELF".into()));
        }
        if u16_le(eh, 0x12) != EM_X86_64 {
            return Err(Error::Unsupported(format!(
                "machine {} is not x86-64",
                u16_le(eh, 0x12)
            )));
        }
        if u16_le(eh, 0x36) as usize != PHDR_SIZE {
            return Err(invalid("unexpected program header size"));
        }
        let header = ElfHeader {
            e_type: u16_le(eh, 0x10),
            entry: u64_le(eh, 0x18),
            phoff: u64_le(eh, 0x20),
            phnum: u16_le(eh, 0x38),
        };
        let phoff =
            usize::try_from(header.phoff).map_err(|_| invalid("program headers out of range"))?;
        let table = get(&data, elf_at + phoff, header.phnum as usize * PHDR_SIZE)?;
        let phdrs = table
            .as_chunks::<PHDR_SIZE>()
            .0
            .iter()
            .map(|p| ProgramHeader::parse(p))
            .collect();
        Ok(Self {
            data,
            container,
            header,
            phdrs,
            self_segments,
        })
    }

    pub fn container(&self) -> Container {
        self.container
    }

    pub fn header(&self) -> &ElfHeader {
        &self.header
    }

    pub fn program_headers(&self) -> &[ProgramHeader] {
        &self.phdrs
    }

    /// The file bytes of program header `index` (`p_filesz` of them).
    pub fn segment_data(&self, index: usize) -> Result<&[u8]> {
        let ph = self
            .phdrs
            .get(index)
            .ok_or_else(|| invalid(format!("no program header {index}")))?;
        let len = usize::try_from(ph.filesz).map_err(|_| invalid("segment too large"))?;
        if len == 0 {
            return Ok(&[]);
        }
        let start = match self.container {
            Container::Elf => ph.offset,
            Container::SelfPs4 | Container::SelfPs5 => {
                // Some headers (PT_DYNAMIC) have no SELF segment of their own.
                let (seg, owner) = self
                    .self_segments
                    .iter()
                    .filter(|s| s.flags & SELF_BLOCKED != 0)
                    .filter_map(|s| Some((s, self.phdrs.get(s.program_header())?)))
                    .find(|(_, o)| {
                        ph.offset >= o.offset && ph.offset + ph.filesz <= o.offset + o.filesz
                    })
                    .ok_or_else(|| {
                        invalid(format!("SELF has no data for program header {index}"))
                    })?;
                if seg.flags & (SELF_ENCRYPTED | SELF_COMPRESSED) != 0 {
                    return Err(Error::Unsupported(format!(
                        "SELF segment for program header {index} is encrypted or compressed; only fake-signed files are supported"
                    )));
                }
                let within = ph.offset - owner.offset;
                if seg.file_size < within + ph.filesz {
                    return Err(invalid(format!(
                        "SELF segment for program header {index} is short"
                    )));
                }
                seg.file_offset + within
            }
        };
        let start = usize::try_from(start).map_err(|_| invalid("segment offset out of range"))?;
        get(&self.data, start, len)
    }

    /// The first program header of a type, with its index.
    pub fn find(&self, p_type: u32) -> Option<(usize, &ProgramHeader)> {
        self.phdrs
            .iter()
            .enumerate()
            .find(|(_, p)| p.p_type == p_type)
    }
}

fn get(data: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    at.checked_add(len)
        .and_then(|end| data.get(at..end))
        .ok_or_else(|| {
            invalid(format!(
                "{len} bytes at {at:#x} are past the end of the file"
            ))
        })
}

pub(crate) fn u16_le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}

pub(crate) fn u32_le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

pub(crate) fn u64_le(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// One segment for a test file: type, flags, vaddr, contents, memsz.
    pub struct Seg {
        pub p_type: u32,
        pub flags: u32,
        pub vaddr: u64,
        pub data: Vec<u8>,
        pub memsz: u64,
    }

    /// Builds a plain ELF whose segment contents follow the headers.
    pub fn build_elf(e_type: u16, entry: u64, segs: &[Seg]) -> Vec<u8> {
        let mut f = vec![0u8; ELF_HEADER_SIZE + segs.len() * PHDR_SIZE];
        f[..4].copy_from_slice(b"\x7fELF");
        f[4] = 2;
        f[5] = 1;
        f[6] = 1;
        f[0x10..0x12].copy_from_slice(&e_type.to_le_bytes());
        f[0x12..0x14].copy_from_slice(&EM_X86_64.to_le_bytes());
        f[0x18..0x20].copy_from_slice(&entry.to_le_bytes());
        f[0x20..0x28].copy_from_slice(&(ELF_HEADER_SIZE as u64).to_le_bytes());
        f[0x34..0x36].copy_from_slice(&(ELF_HEADER_SIZE as u16).to_le_bytes());
        f[0x36..0x38].copy_from_slice(&(PHDR_SIZE as u16).to_le_bytes());
        f[0x38..0x3a].copy_from_slice(&(segs.len() as u16).to_le_bytes());
        for (i, s) in segs.iter().enumerate() {
            let offset = f.len() as u64;
            f.extend_from_slice(&s.data);
            let p = &mut f[ELF_HEADER_SIZE + i * PHDR_SIZE..][..PHDR_SIZE];
            p[0..4].copy_from_slice(&s.p_type.to_le_bytes());
            p[4..8].copy_from_slice(&s.flags.to_le_bytes());
            p[8..16].copy_from_slice(&offset.to_le_bytes());
            p[16..24].copy_from_slice(&s.vaddr.to_le_bytes());
            p[24..32].copy_from_slice(&s.vaddr.to_le_bytes());
            p[32..40].copy_from_slice(&(s.data.len() as u64).to_le_bytes());
            p[40..48].copy_from_slice(&s.memsz.to_le_bytes());
            p[48..56].copy_from_slice(&0x4000u64.to_le_bytes());
        }
        f
    }

    /// Wraps an ELF built by `build_elf` in a fake-signed SELF, with one
    /// blocked segment per program header. `extra_flags` is ORed into each.
    pub fn wrap_self(elf: &[u8], extra_flags: u64) -> Vec<u8> {
        let n = ElfFile::parse(elf.to_vec()).unwrap().phdrs.len();
        wrap(elf, n, extra_flags)
    }

    /// Like `wrap_self`, but only the first `blocked` program headers get a
    /// SELF segment.
    pub fn wrap_self_first(elf: &[u8], blocked: usize) -> Vec<u8> {
        wrap(elf, blocked, 0)
    }

    fn wrap(elf: &[u8], blocked: usize, extra_flags: u64) -> Vec<u8> {
        let file = ElfFile::parse(elf.to_vec()).unwrap();
        let n = blocked;
        let headers_end = ELF_HEADER_SIZE + file.phdrs.len() * PHDR_SIZE;
        let mut out = vec![0u8; SELF_HEADER_SIZE + n * SELF_SEGMENT_SIZE];
        out[..4].copy_from_slice(&SELF_MAGIC_PS4.to_le_bytes());
        out[0x18..0x1a].copy_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&elf[..headers_end]);
        for (i, ph) in file.phdrs.iter().enumerate().take(n) {
            let at = out.len() as u64;
            out.extend_from_slice(file.segment_data(i).unwrap());
            let flags = SELF_BLOCKED | ((i as u64) << 20) | extra_flags;
            let s = &mut out[SELF_HEADER_SIZE + i * SELF_SEGMENT_SIZE..][..SELF_SEGMENT_SIZE];
            s[0..8].copy_from_slice(&flags.to_le_bytes());
            s[8..16].copy_from_slice(&at.to_le_bytes());
            s[16..24].copy_from_slice(&ph.filesz.to_le_bytes());
            s[24..32].copy_from_slice(&ph.memsz.to_le_bytes());
        }
        out
    }

    fn sample() -> Vec<u8> {
        build_elf(
            ET_SCE_DYNEXEC,
            0x1234,
            &[
                Seg {
                    p_type: PT_LOAD,
                    flags: PF_R | PF_X,
                    vaddr: 0,
                    data: b"code".to_vec(),
                    memsz: 4,
                },
                Seg {
                    p_type: PT_LOAD,
                    flags: PF_R | PF_W,
                    vaddr: 0x4000,
                    data: b"data!".to_vec(),
                    memsz: 0x100,
                },
            ],
        )
    }

    #[test]
    fn reads_plain_elf_segments() {
        let f = ElfFile::parse(sample()).unwrap();
        assert_eq!(f.container(), Container::Elf);
        assert_eq!(f.header().e_type, ET_SCE_DYNEXEC);
        assert_eq!(f.header().entry, 0x1234);
        assert_eq!(f.program_headers()[1].memsz, 0x100);
        assert_eq!(f.segment_data(0).unwrap(), b"code");
        assert_eq!(f.segment_data(1).unwrap(), b"data!");
    }

    #[test]
    fn reads_the_same_segments_through_a_self() {
        let f = ElfFile::parse(wrap_self(&sample(), 0)).unwrap();
        assert_eq!(f.container(), Container::SelfPs4);
        assert_eq!(f.program_headers().len(), 2);
        assert_eq!(f.segment_data(0).unwrap(), b"code");
        assert_eq!(f.segment_data(1).unwrap(), b"data!");
    }

    #[test]
    fn finds_segments_nested_in_another_through_a_self() {
        let mut elf = sample();
        // Add a third header covering the last 3 bytes of segment 1 ("ta!").
        let f = ElfFile::parse(elf.clone()).unwrap();
        let inner = ProgramHeader {
            offset: f.phdrs[1].offset + 2,
            filesz: 3,
            ..f.phdrs[1]
        };
        elf[0x38..0x3a].copy_from_slice(&3u16.to_le_bytes());
        let mut ph = vec![0u8; PHDR_SIZE];
        ph[0..4].copy_from_slice(&PT_DYNAMIC.to_le_bytes());
        ph[8..16].copy_from_slice(&inner.offset.to_le_bytes());
        ph[32..40].copy_from_slice(&inner.filesz.to_le_bytes());
        // Header table grows by one entry: shift segment data offsets.
        let at = ELF_HEADER_SIZE + 2 * PHDR_SIZE;
        elf.splice(at..at, ph);
        for i in 0..3 {
            let o = ELF_HEADER_SIZE + i * PHDR_SIZE + 8;
            let v = u64_le(&elf, o) + PHDR_SIZE as u64;
            elf[o..o + 8].copy_from_slice(&v.to_le_bytes());
        }
        let plain = ElfFile::parse(elf.clone()).unwrap();
        assert_eq!(plain.segment_data(2).unwrap(), b"ta!");
        let wrapped = ElfFile::parse(wrap_self_first(&elf, 2)).unwrap();
        assert_eq!(wrapped.segment_data(2).unwrap(), b"ta!");
        assert_eq!(wrapped.segment_data(1).unwrap(), b"data!");
    }

    #[test]
    fn refuses_encrypted_self_segments() {
        let f = ElfFile::parse(wrap_self(&sample(), SELF_ENCRYPTED)).unwrap();
        assert!(matches!(f.segment_data(0), Err(Error::Unsupported(_))));
    }

    #[test]
    fn refuses_other_machines_and_truncated_files() {
        let mut elf = sample();
        elf[0x12] = 3;
        assert!(matches!(ElfFile::parse(elf), Err(Error::Unsupported(_))));
        assert!(ElfFile::parse(sample()[..0x30].to_vec()).is_err());
        assert!(ElfFile::parse(b"\x7fELX".repeat(32)).is_err());
    }
}
