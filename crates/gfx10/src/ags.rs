//! The `.ags` shader container: a small ELF64 file (machine `EM_AMDGPU`)
//! whose sections hold the machine code and the AGC shader header. Layout as
//! observed in Balatro's shaders; see `docs/shaders.md`.

/// Magic and version at the start of the AGC shader header.
pub const HEADER_MAGIC: u32 = 0x3433_3231;
pub const HEADER_VERSION: u32 = 0x18;

/// Shader stage, from the header byte at 0x5A.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Pixel,
    /// The NGG geometry stage, which runs vertex shaders on GFX10.
    Geometry,
    Other(u8),
}

/// A parsed `.ags` file.
#[derive(Debug)]
pub struct Ags<'a> {
    /// The `.shader_text` section: code, then a trailer starting with `sl00`.
    pub text: &'a [u8],
    /// The `.shader_header` section, which starts with the AGC shader header.
    pub header: &'a [u8],
    pub metadata: &'a [u8],
    pub cfg: &'a [u8],
}

/// Why a file is not an `.ags` container.
#[derive(Debug, PartialEq, Eq)]
pub enum AgsError {
    NotElf,
    NotAmdgpu,
    Truncated,
    MissingSection(&'static str),
    BadHeader,
}

const EM_AMDGPU: u16 = 0xE0;

fn u16_at(b: &[u8], at: usize) -> Result<u16, AgsError> {
    let s = b.get(at..at + 2).ok_or(AgsError::Truncated)?;
    Ok(u16::from_le_bytes(s.try_into().unwrap()))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, AgsError> {
    let s = b.get(at..at + 4).ok_or(AgsError::Truncated)?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, AgsError> {
    let s = b.get(at..at + 8).ok_or(AgsError::Truncated)?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}

impl<'a> Ags<'a> {
    /// Parses an `.ags` file.
    pub fn parse(file: &'a [u8]) -> Result<Self, AgsError> {
        if file.get(..4) != Some(b"\x7fELF".as_slice()) || file.get(4) != Some(&2) {
            return Err(AgsError::NotElf);
        }
        if u16_at(file, 0x12)? != EM_AMDGPU {
            return Err(AgsError::NotAmdgpu);
        }
        let shoff = u64_at(file, 0x28)? as usize;
        let shentsize = u16_at(file, 0x3A)? as usize;
        let shnum = u16_at(file, 0x3C)? as usize;
        let shstrndx = u16_at(file, 0x3E)? as usize;
        let section = |i: usize| -> Result<(u32, &'a [u8]), AgsError> {
            let h = shoff + i * shentsize;
            let name = u32_at(file, h)?;
            let off = u64_at(file, h + 0x18)? as usize;
            let size = u64_at(file, h + 0x20)? as usize;
            let data = file.get(off..off + size).ok_or(AgsError::Truncated)?;
            Ok((name, data))
        };
        let (_, strtab) = section(shstrndx)?;
        let find = |want: &'static str| -> Result<&'a [u8], AgsError> {
            for i in 0..shnum {
                let (name, data) = section(i)?;
                let rest = strtab.get(name as usize..).ok_or(AgsError::Truncated)?;
                let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
                if &rest[..end] == want.as_bytes() {
                    return Ok(data);
                }
            }
            Err(AgsError::MissingSection(want))
        };
        let ags = Ags {
            text: find(".shader_text")?,
            header: find(".shader_header")?,
            metadata: find(".metadata")?,
            cfg: find(".cfg")?,
        };
        if u32_at(ags.header, 0)? != HEADER_MAGIC || u32_at(ags.header, 4)? != HEADER_VERSION {
            return Err(AgsError::BadHeader);
        }
        Ok(ags)
    }

    /// The machine code: `.shader_text` up to the `sl00` trailer.
    pub fn code(&self) -> &'a [u8] {
        let end = self
            .text
            .windows(4)
            .position(|w| w == b"sl00")
            .unwrap_or(self.text.len());
        &self.text[..end & !3]
    }

    /// The machine code as little-endian dwords.
    pub fn code_words(&self) -> Vec<u32> {
        self.code()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    pub fn stage(&self) -> Stage {
        match self.header.get(0x5A).copied().unwrap_or(0xFF) {
            0 => Stage::Compute,
            1 => Stage::Pixel,
            2 => Stage::Geometry,
            n => Stage::Other(n),
        }
    }

    /// Follows the self-relative offset stored at header offset `field`.
    pub fn header_table(&self, field: usize) -> Option<usize> {
        let rel = u64_at(self.header, field).ok()?;
        (rel != 0).then(|| field + rel as usize)
    }

    /// `(register, value)` pairs of a register table: `CX_REGISTERS` or `SH_REGISTERS`.
    pub fn registers(&self, field: usize, count: usize) -> Vec<(u32, u32)> {
        let Some(at) = self.header_table(field) else {
            return Vec::new();
        };
        (0..count)
            .filter_map(|i| {
                Some((
                    u32_at(self.header, at + 8 * i).ok()?,
                    u32_at(self.header, at + 8 * i + 4).ok()?,
                ))
            })
            .collect()
    }

    /// The SH registers: program address, resources and checksum.
    pub fn sh_registers(&self) -> Vec<(u32, u32)> {
        let n = self.header.get(0x5C).copied().unwrap_or(0) as usize;
        self.registers(0x20, n)
    }

    /// The context registers the shader needs set.
    pub fn cx_registers(&self) -> Vec<(u32, u32)> {
        let n = self.header.get(0x5B).copied().unwrap_or(0) as usize;
        self.registers(0x18, n)
    }

    /// Whether the shader runs in wave64, from the PS or NGG register that says so.
    pub fn wave64(&self) -> bool {
        let cx = self.cx_registers();
        match self.stage() {
            Stage::Pixel => cx
                .iter()
                .find(|r| r.0 == SPI_PS_IN_CONTROL)
                .is_none_or(|r| r.1 & 1 << 15 == 0),
            _ => true,
        }
    }
}

impl Ags<'_> {
    /// The translation request this file's registers describe, for `code`.
    pub fn request<'c>(&self, code: &'c [u32]) -> crate::translate::Request<'c> {
        let sh = self.sh_registers();
        let cx = self.cx_registers();
        let reg = |list: &[(u32, u32)], r: u32| list.iter().find(|x| x.0 == r).map_or(0, |x| x.1);
        let stage = match self.stage() {
            Stage::Pixel => crate::translate::Stage::Pixel,
            Stage::Compute => crate::translate::Stage::Compute,
            _ => crate::translate::Stage::Vertex,
        };
        let rsrc2 = match stage {
            crate::translate::Stage::Pixel => reg(&sh, SPI_SHADER_PGM_RSRC2_PS),
            _ => reg(&sh, SPI_SHADER_PGM_RSRC2_GS),
        };
        crate::translate::Request {
            stage,
            code,
            user_sgprs: rsrc2 >> 1 & 0x1F,
            ngg: self.stage() == Stage::Geometry,
            ps_input_ena: reg(&cx, SPI_PS_INPUT_ENA),
            ps_input_addr: reg(&cx, SPI_PS_INPUT_ADDR),
            ps_input_cntl: &[],
            col_format: reg(&cx, SPI_SHADER_COL_FORMAT),
            compute_rsrc2: 0,
            threads: [64, 1, 1],
        }
    }
}

/// SH registers (offsets from 0x2C00), from Mesa's register tables.
pub const SPI_SHADER_PGM_RSRC2_PS: u32 = 0x0B;
pub const SPI_SHADER_PGM_RSRC2_GS: u32 = 0x8B;
/// Context registers (offsets from 0xA000), from Mesa's register tables.
pub const SPI_PS_INPUT_ENA: u32 = 0x1B3;
pub const SPI_PS_INPUT_ADDR: u32 = 0x1B4;
pub const SPI_SHADER_COL_FORMAT: u32 = 0x1C5;

/// `SPI_PS_IN_CONTROL`, a context register (offset from 0xA000).
pub const SPI_PS_IN_CONTROL: u32 = 0x1B6;
