//! Decoder for 64-bit mode. Encodings follow the Intel SDM, volume 2.

pub const MAX_LEN: usize = 15;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Map {
    #[default]
    Primary,
    M0F,
    M0F38,
    M0F3A,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rep {
    #[default]
    None,
    /// `F3`
    Rep,
    /// `F2`
    Repne,
}

/// The only segment overrides that mean anything in 64-bit mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seg {
    Fs,
    Gs,
}

/// A memory operand: `seg + base + index * scale + disp`, or `next_rip + disp`
/// when `rip_relative`. With `addr32`, the sum (before the segment base) is
/// truncated to 32 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8,
    pub disp: i32,
    pub rip_relative: bool,
    pub addr32: bool,
    pub seg: Option<Seg>,
    /// The SIB index field with REX.X, even when it is 4: a vector index for
    /// VSIB addressing (gathers), where 4 does not mean "no index".
    pub vsib_index: Option<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Insn {
    pub map: Map,
    pub op: u8,
    pub len: u8,
    /// The REX byte, or 0 if there was none.
    pub rex: u8,
    pub opsize_prefix: bool,
    /// `67`: 32-bit addressing.
    pub addr32: bool,
    pub rep: Rep,
    pub lock: bool,
    pub seg: Option<Seg>,
    pub modrm: Option<u8>,
    /// ModRM `reg` with REX.R.
    pub reg: u8,
    /// ModRM `rm` with REX.B, for register operands; the low opcode bits with
    /// REX.B for `+r` encodings.
    pub rm: u8,
    pub mem: Option<Mem>,
    /// The immediate, sign-extended from its encoded size (zero-extended for
    /// `mov r64, imm64` and absolute addresses).
    pub imm: i64,
    /// ENTER's second immediate.
    pub imm2: u8,
    /// Set for VEX-encoded instructions.
    pub vex: Option<Vex>,
}

/// The fields a VEX prefix adds beyond REX and the implied 66/F3/F2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vex {
    /// 256-bit operation (VEX.L).
    pub l: bool,
    /// The extra source register (VEX.vvvv, already un-inverted).
    pub vvvv: u8,
}

impl Insn {
    pub fn rex_w(&self) -> bool {
        self.rex & 8 != 0
    }

    /// ModRM `reg` without REX.R: the opcode extension of group instructions.
    pub fn ext(&self) -> u8 {
        self.reg & 7
    }

    /// Operand size in bytes for a full-size operation: 8 with REX.W, else 2
    /// with `66`, else 4.
    pub fn osize(&self) -> u8 {
        if self.rex_w() {
            8
        } else if self.opsize_prefix {
            2
        } else {
            4
        }
    }

    /// Operand size for instructions that default to 64 bits (push, pop,
    /// near branches): 2 with `66`, else 8.
    pub fn osize64(&self) -> u8 {
        if self.opsize_prefix && !self.rex_w() {
            2
        } else {
            8
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError<E> {
    /// Longer than 15 bytes.
    TooLong,
    /// An opcode that does not exist in 64-bit mode.
    Invalid,
    /// A valid encoding we do not decode yet (VEX, EVEX, XOP, 3DNow!).
    Unsupported,
    Fetch(E),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Imm {
    None,
    B,
    W,
    /// 16 or 32 bits by operand size; 32 bits sign-extended with REX.W.
    Z,
    /// 16, 32 or 64 bits by operand size (`mov r, imm`).
    V,
    /// A 64-bit (or with `67`, 32-bit) absolute address (`mov al, [moffs]`).
    Moffs,
    /// ENTER: 16 then 8 bits.
    Enter,
    /// Two bytes, in `imm` and `imm2` (SSE4a `extrq`/`insertq`).
    BB,
}

fn primary(op: u8) -> Option<(bool, Imm)> {
    use Imm::*;
    Some(match op {
        0x06 | 0x07 | 0x0e | 0x16 | 0x17 | 0x1e | 0x1f | 0x27 | 0x2f | 0x37 | 0x3f => {
            return Option::None;
        }
        0x00..=0x3f => match op & 7 {
            0..=3 => (true, None),
            4 => (false, B),
            _ => (false, Z),
        },
        0x50..=0x5f => (false, None),
        0x63 => (true, None),
        0x68 => (false, Z),
        0x69 => (true, Z),
        0x6a => (false, B),
        0x6b => (true, B),
        0x6c..=0x6f => (false, None),
        0x70..=0x7f => (false, B),
        0x80 | 0x83 => (true, B),
        0x81 => (true, Z),
        0x84..=0x8f => (true, None),
        0x90..=0x99 | 0x9b..=0x9f => (false, None),
        0xa0..=0xa3 => (false, Moffs),
        0xa4..=0xa7 | 0xaa..=0xaf => (false, None),
        0xa8 => (false, B),
        0xa9 => (false, Z),
        0xb0..=0xb7 => (false, B),
        0xb8..=0xbf => (false, V),
        0xc0 | 0xc1 | 0xc6 => (true, B),
        0xc7 => (true, Z),
        0xc2 | 0xca => (false, W),
        0xc3 | 0xc9 | 0xcb | 0xcc | 0xcf => (false, None),
        0xc8 => (false, Enter),
        0xcd => (false, B),
        0xd0..=0xd3 | 0xd8..=0xdf => (true, None),
        0xd7 => (false, None),
        0xe0..=0xe7 => (false, B),
        0xe8 | 0xe9 => (false, Z),
        0xeb => (false, B),
        0xec..=0xef | 0xf1 | 0xf4 | 0xf5 | 0xf8..=0xfd => (false, None),
        0xf6 | 0xf7 | 0xfe | 0xff => (true, None),
        _ => return Option::None,
    })
}

fn map0f(op: u8) -> Option<(bool, Imm)> {
    use Imm::*;
    Some(match op {
        0x04
        | 0x0a
        | 0x0c
        | 0x0f
        | 0x24..=0x27
        | 0x36
        | 0x39..=0x3f
        | 0x7a
        | 0x7b
        | 0xa6
        | 0xa7
        | 0xff => return Option::None,
        0x05..=0x09
        | 0x0b
        | 0x0e
        | 0x30..=0x35
        | 0x37
        | 0x77
        | 0xa0..=0xa2
        | 0xa8..=0xaa
        | 0xc8..=0xcf => (false, None),
        0x80..=0x8f => (false, Z),
        0x70..=0x73 | 0xa4 | 0xac | 0xba | 0xc2 | 0xc4..=0xc6 => (true, B),
        _ => (true, None),
    })
}

/// Decodes one instruction. `fetch(i)` returns the instruction's byte `i`.
pub fn decode<E>(mut fetch: impl FnMut(usize) -> Result<u8, E>) -> Result<Insn, DecodeError<E>> {
    let mut i = 0usize;
    let mut next = |i: &mut usize| -> Result<u8, DecodeError<E>> {
        if *i >= MAX_LEN {
            return Err(DecodeError::TooLong);
        }
        let b = fetch(*i).map_err(DecodeError::Fetch)?;
        *i += 1;
        Ok(b)
    };
    let mut insn = Insn::default();
    let mut addr32 = false;
    let mut b = next(&mut i)?;
    // A REX byte only counts directly before the opcode; any prefix after it voids it.
    loop {
        match b {
            0xf0 => insn.lock = true,
            0xf2 => insn.rep = Rep::Repne,
            0xf3 => insn.rep = Rep::Rep,
            0x26 | 0x2e | 0x36 | 0x3e => {}
            0x64 => insn.seg = Some(Seg::Fs),
            0x65 => insn.seg = Some(Seg::Gs),
            0x66 => insn.opsize_prefix = true,
            0x67 => addr32 = true,
            0x40..=0x4f => {
                insn.rex = b;
                b = next(&mut i)?;
                continue;
            }
            _ => break,
        }
        insn.rex = 0;
        b = next(&mut i)?;
    }
    let mut vex_map = None;
    if matches!(b, 0xc4 | 0xc5) {
        if insn.rex != 0 || insn.opsize_prefix || insn.rep != Rep::None || insn.lock {
            return Err(DecodeError::Invalid);
        }
        let b1 = next(&mut i)?;
        let (rxb, map, b2) = if b == 0xc5 {
            ((!b1 >> 7) << 2, 1, b1)
        } else {
            (((!b1) >> 5) & 7, b1 & 0x1f, next(&mut i)?)
        };
        let w = if b == 0xc4 { b2 >> 7 } else { 0 };
        insn.rex = 0x40 | w << 3 | rxb;
        insn.vex = Some(Vex {
            l: b2 & 4 != 0,
            vvvv: (!b2 >> 3) & 0xf,
        });
        match b2 & 3 {
            1 => insn.opsize_prefix = true,
            2 => insn.rep = Rep::Rep,
            3 => insn.rep = Rep::Repne,
            _ => {}
        }
        vex_map = Some(match map {
            1 => Map::M0F,
            2 => Map::M0F38,
            3 => Map::M0F3A,
            _ => return Err(DecodeError::Invalid),
        });
        b = next(&mut i)?;
    } else if b == 0x62 {
        return Err(DecodeError::Unsupported);
    }
    let r = (insn.rex >> 2) & 1;
    let x = (insn.rex >> 1) & 1;
    let rb = insn.rex & 1;

    let (has_modrm, imm) = if let Some(map) = vex_map {
        insn.map = map;
        match map {
            Map::M0F => map0f(b).ok_or(DecodeError::Invalid)?,
            Map::M0F38 => (true, Imm::None),
            _ => (true, Imm::B),
        }
    } else if b == 0x0f {
        b = next(&mut i)?;
        match b {
            0x38 => {
                insn.map = Map::M0F38;
                b = next(&mut i)?;
                (true, Imm::None)
            }
            0x3a => {
                insn.map = Map::M0F3A;
                b = next(&mut i)?;
                (true, Imm::B)
            }
            0x0f => return Err(DecodeError::Unsupported),
            _ => {
                insn.map = Map::M0F;
                map0f(b).ok_or(DecodeError::Invalid)?
            }
        }
    } else {
        primary(b).ok_or(DecodeError::Invalid)?
    };
    insn.op = b;
    insn.rm = (b & 7) | (rb << 3);

    if has_modrm {
        let m = next(&mut i)?;
        insn.modrm = Some(m);
        let md = m >> 6;
        insn.reg = ((m >> 3) & 7) | (r << 3);
        if insn.map == Map::Primary && b == 0x8f && insn.ext() != 0 {
            return Err(DecodeError::Unsupported);
        }
        let rm = m & 7;
        if md == 3 {
            insn.rm = rm | (rb << 3);
        } else {
            let mut mem = Mem {
                addr32,
                seg: insn.seg,
                scale: 1,
                ..Mem::default()
            };
            let mut disp_len = match md {
                1 => 1,
                2 => 4,
                _ => 0,
            };
            if rm == 4 {
                let sib = next(&mut i)?;
                let index = ((sib >> 3) & 7) | (x << 3);
                mem.vsib_index = Some(index);
                if index != 4 {
                    mem.index = Some(index);
                }
                mem.scale = 1 << (sib >> 6);
                let base = sib & 7;
                if base == 5 && md == 0 {
                    disp_len = 4;
                } else {
                    mem.base = Some(base | (rb << 3));
                }
            } else if rm == 5 && md == 0 {
                mem.rip_relative = true;
                disp_len = 4;
            } else {
                mem.base = Some(rm | (rb << 3));
            }
            mem.disp = match disp_len {
                1 => next(&mut i)? as i8 as i32,
                4 => {
                    let mut v = 0u32;
                    for k in 0..4 {
                        v |= (next(&mut i)? as u32) << (8 * k);
                    }
                    v as i32
                }
                _ => 0,
            };
            insn.mem = Some(mem);
        }
    }

    let mut read = |n: usize, i: &mut usize| -> Result<u64, DecodeError<E>> {
        let mut v = 0u64;
        for k in 0..n {
            v |= (next(i)? as u64) << (8 * k);
        }
        Ok(v)
    };
    let sext = |v: u64, bytes: usize| -> i64 {
        let shift = 64 - 8 * bytes as u32;
        ((v << shift) as i64) >> shift
    };
    let imm = match (insn.map, b) {
        (Map::Primary, 0xf6) if insn.ext() < 2 => Imm::B,
        (Map::Primary, 0xf7) if insn.ext() < 2 => Imm::Z,
        (Map::M0F, 0x78) if insn.vex.is_none() && (insn.opsize_prefix || insn.rep == Rep::Repne) => Imm::BB,
        _ => imm,
    };
    let branch = matches!(
        (insn.map, b),
        (Map::Primary, 0xe8 | 0xe9) | (Map::M0F, 0x80..=0x8f)
    );
    match imm {
        Imm::None => {}
        Imm::B => insn.imm = sext(read(1, &mut i)?, 1),
        Imm::W => insn.imm = read(2, &mut i)? as i64,
        Imm::Z => {
            let n = if insn.opsize_prefix && !insn.rex_w() && !branch {
                2
            } else {
                4
            };
            insn.imm = sext(read(n, &mut i)?, n);
        }
        Imm::V => insn.imm = read(insn.osize() as usize, &mut i)? as i64,
        Imm::Moffs => insn.imm = read(if addr32 { 4 } else { 8 }, &mut i)? as i64,
        Imm::Enter => {
            insn.imm = read(2, &mut i)? as i64;
            insn.imm2 = read(1, &mut i)? as u8;
        }
        Imm::BB => {
            insn.imm = read(1, &mut i)? as i64;
            insn.imm2 = read(1, &mut i)? as u8;
        }
    }
    insn.addr32 = addr32;
    insn.len = i as u8;
    Ok(insn)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes `bytes`, the encoding of `_asm` as llvm-mc emits it.
    fn dec(_asm: &str, bytes: &[u8]) -> Insn {
        decode(|i| bytes.get(i).copied().ok_or(())).unwrap()
    }

    #[test]
    fn decodes_register_and_rex_operands() {
        let i = dec("mov r9, rax", &[0x49, 0x89, 0xc1]);
        assert_eq!((i.op, i.len, i.reg, i.rm, i.osize()), (0x89, 3, 0, 9, 8));
        assert!(i.mem.is_none());
    }

    #[test]
    fn decodes_sib_with_displacement() {
        let i = dec(
            "lea rax, [rbx + r12*4 + 0x10]",
            &[0x4a, 0x8d, 0x44, 0xa3, 0x10],
        );
        let m = i.mem.unwrap();
        assert_eq!(
            (m.base, m.index, m.scale, m.disp),
            (Some(3), Some(12), 4, 0x10)
        );
        assert_eq!(i.len, 5);
    }

    #[test]
    fn decodes_rip_relative_and_disp32_without_base() {
        let m = dec("mov eax, [rip - 8]", &[0x8b, 0x05, 0xf8, 0xff, 0xff, 0xff])
            .mem
            .unwrap();
        assert!(m.rip_relative && m.base.is_none());
        assert_eq!(m.disp, -8);
        let m = dec(
            "mov eax, [0x1000]",
            &[0x8b, 0x04, 0x25, 0x00, 0x10, 0x00, 0x00],
        )
        .mem
        .unwrap();
        assert!(!m.rip_relative && m.base.is_none() && m.index.is_none());
        assert_eq!(m.disp, 0x1000);
    }

    #[test]
    fn r13_base_needs_an_explicit_displacement() {
        let m = dec("mov rax, [r13]", &[0x49, 0x8b, 0x45, 0x00])
            .mem
            .unwrap();
        assert_eq!((m.base, m.disp), (Some(13), 0));
    }

    #[test]
    fn decodes_immediates_by_size() {
        let i = dec(
            "mov rax, 0x0807060504030201",
            &[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8],
        );
        assert_eq!((i.imm as u64, i.len), (0x0807_0605_0403_0201, 10));
        assert_eq!(dec("add rax, -1", &[0x48, 0x83, 0xc0, 0xff]).imm, -1);
        let i = dec("mov word [rax], 0x1234", &[0x66, 0xc7, 0x00, 0x34, 0x12]);
        assert_eq!((i.imm, i.len), (0x1234, 5));
        assert_eq!(
            dec(
                "test eax, 0x12345678",
                &[0xf7, 0xc0, 0x78, 0x56, 0x34, 0x12]
            )
            .imm,
            0x1234_5678
        );
        assert_eq!(dec("not eax", &[0xf7, 0xd0]).len, 2);
    }

    #[test]
    fn decodes_two_byte_maps_and_prefixes() {
        let i = dec("endbr64", &[0xf3, 0x0f, 0x1e, 0xfa]);
        assert_eq!((i.map, i.op, i.rep, i.len), (Map::M0F, 0x1e, Rep::Rep, 4));
        let i = dec("jne +0x10", &[0x0f, 0x85, 0x10, 0, 0, 0]);
        assert_eq!((i.imm, i.len), (0x10, 6));
        let i = dec(
            "mov rax, fs:[0]",
            &[0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0],
        );
        assert_eq!(i.mem.unwrap().seg, Some(Seg::Fs));
    }

    #[test]
    fn a_prefix_after_rex_voids_the_rex() {
        let i = dec(
            "rex.w; mov eax, ecx with 66 after rex",
            &[0x48, 0x66, 0x89, 0xc8],
        );
        assert_eq!((i.rex, i.osize(), i.len), (0, 2, 4));
    }

    #[test]
    fn decodes_two_and_three_byte_vex() {
        let i = dec("vzeroupper", &[0xc5, 0xf8, 0x77]);
        assert_eq!(
            (i.map, i.op, i.len, i.vex),
            (Map::M0F, 0x77, 3, Some(Vex { l: false, vvvv: 0 }))
        );
        let i = dec("vaddps ymm1, ymm2, ymm3", &[0xc5, 0xec, 0x58, 0xcb]);
        assert_eq!(
            (i.reg, i.rm, i.vex.unwrap().vvvv, i.vex.unwrap().l),
            (1, 3, 2, true)
        );
        let i = dec(
            "vpaddq ymm9, ymm10, ymmword ptr [r11]",
            &[0xc4, 0x41, 0x2d, 0xd4, 0x0b],
        );
        assert_eq!(
            (
                i.reg,
                i.vex.unwrap().vvvv,
                i.mem.unwrap().base,
                i.opsize_prefix
            ),
            (9, 10, Some(11), true)
        );
        let i = dec(
            "vfmadd231ps xmm0, xmm1, xmm2",
            &[0xc4, 0xe2, 0x71, 0xb8, 0xc2],
        );
        assert_eq!((i.map, i.op, i.vex.unwrap().vvvv), (Map::M0F38, 0xb8, 1));
        let i = dec(
            "vpermq ymm0, ymm1, 0x1b",
            &[0xc4, 0xe3, 0xfd, 0x00, 0xc1, 0x1b],
        );
        assert_eq!(
            (i.map, i.imm, i.rex_w(), i.len),
            (Map::M0F3A, 0x1b, true, 6)
        );
    }

    #[test]
    fn rejects_invalid_unsupported_and_too_long_instructions() {
        assert_eq!(
            decode(|i| [0x06].get(i).copied().ok_or(())),
            Err(DecodeError::Invalid)
        );
        assert_eq!(decode(|_| Ok::<u8, ()>(0x66)), Err(DecodeError::TooLong));
        assert_eq!(
            decode(|i| [0x62, 0xf1, 0x7c, 0x48, 0x58, 0xc1]
                .get(i)
                .copied()
                .ok_or(())),
            Err(DecodeError::Unsupported)
        );
    }
}
