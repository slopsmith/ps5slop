//! Decodes GFX10.1 machine code into [`Inst`]s.

use crate::opinfo::{Fmt, OpInfo, lookup};

/// Operand codes of the 9-bit source field that have a name of their own.
pub mod src {
    pub const VCC_LO: u16 = 106;
    pub const VCC_HI: u16 = 107;
    pub const M0: u16 = 124;
    pub const NULL: u16 = 125;
    pub const EXEC_LO: u16 = 126;
    pub const EXEC_HI: u16 = 127;
    pub const ZERO: u16 = 128;
    pub const DPP8: u16 = 233;
    pub const DPP8_FI: u16 = 234;
    pub const SDWA: u16 = 249;
    pub const DPP16: u16 = 250;
    pub const VCCZ: u16 = 251;
    pub const EXECZ: u16 = 252;
    pub const SCC: u16 = 253;
    pub const LDS_DIRECT: u16 = 254;
    pub const LITERAL: u16 = 255;
    pub const VGPR0: u16 = 256;
}

/// How a vector ALU instruction was encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VEnc {
    /// VOP1, VOP2 or VOPC in their 32-bit encoding.
    E32,
    /// VOP3 (also VOP1, VOP2 and VOPC promoted to it).
    E64,
    Sdwa,
    Dpp16,
    Dpp8,
    /// VOP3P.
    P,
}

/// Sub-dword addressing fields (the SDWA dword).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sdwa {
    pub dst_sel: u8,
    pub dst_unused: u8,
    pub src_sel: [u8; 2],
    pub sext: [bool; 2],
}

/// Data-parallel fields (the DPP dword).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dpp {
    pub ctrl: u16,
    pub fi: bool,
    pub bound_ctrl: bool,
    pub bank_mask: u8,
    pub row_mask: u8,
    /// DPP8 lane selects, 3 bits each.
    pub sel8: u32,
}

/// A vector ALU instruction in any of its encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Valu {
    pub enc: VEnc,
    pub vdst: u8,
    /// Scalar destination: VOP3b, VOPC in VOP3 or SDWA form. `None` means VCC or nothing.
    pub sdst: Option<u8>,
    /// Source operand codes; 256 and up are VGPRs.
    pub src: [u16; 3],
    pub abs: u8,
    pub neg: u8,
    pub neg_hi: u8,
    pub opsel: u8,
    pub opsel_hi: u8,
    pub clamp: bool,
    pub omod: u8,
    pub sdwa: Sdwa,
    pub dpp: Dpp,
}

/// Fields of each format, named as in AMD's ISA guide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Sop2 {
        sdst: u8,
        ssrc0: u16,
        ssrc1: u16,
    },
    Sopk {
        sdst: u8,
        simm16: u16,
    },
    Sop1 {
        sdst: u8,
        ssrc0: u16,
    },
    Sopc {
        ssrc0: u16,
        ssrc1: u16,
    },
    Sopp {
        simm16: u16,
    },
    Smem {
        sdata: u8,
        sbase: u8,
        glc: bool,
        dlc: bool,
        offset: i32,
        soffset: u8,
    },
    Valu(Valu),
    Vintrp {
        vdst: u8,
        vsrc: u8,
        attr: u8,
        chan: u8,
    },
    Ds {
        offset0: u8,
        offset1: u8,
        gds: bool,
        addr: u8,
        data0: u8,
        data1: u8,
        vdst: u8,
    },
    Mubuf(Buf),
    Mtbuf(Buf),
    Mimg(Mimg),
    Exp {
        en: u8,
        target: u8,
        compr: bool,
        done: bool,
        vm: bool,
        vsrc: [u8; 4],
    },
    Flat(Flat),
}

/// MUBUF and MTBUF fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf {
    pub offset: u16,
    pub offen: bool,
    pub idxen: bool,
    pub glc: bool,
    pub dlc: bool,
    pub slc: bool,
    pub lds: bool,
    pub tfe: bool,
    pub format: u8,
    pub vaddr: u8,
    pub vdata: u8,
    pub srsrc: u8,
    pub soffset: u8,
}

/// MIMG fields. `vaddr` holds the address VGPRs: the first, then the NSA bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mimg {
    pub nsa: u8,
    pub dim: u8,
    pub dlc: bool,
    pub dmask: u8,
    pub unorm: bool,
    pub glc: bool,
    pub slc: bool,
    pub r128: bool,
    pub tfe: bool,
    pub lwe: bool,
    pub a16: bool,
    pub d16: bool,
    pub vaddr: [u8; 13],
    pub vdata: u8,
    pub srsrc: u8,
    pub ssamp: u8,
}

/// FLAT, GLOBAL and SCRATCH fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flat {
    pub offset: i16,
    pub dlc: bool,
    pub glc: bool,
    pub slc: bool,
    pub lds: bool,
    pub addr: u8,
    pub data: u8,
    pub saddr: u8,
    pub nv: bool,
    pub vdst: u8,
}

/// One decoded instruction.
#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub op: &'static OpInfo,
    pub kind: Kind,
    pub literal: Option<u32>,
    /// Length in dwords, literal included.
    pub words: u8,
}

/// Why a word sequence did not decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The instruction runs past the end of the input.
    Truncated,
    /// An encoding gfx1013 does not have; holds the first word.
    Invalid(u32),
}

fn bits(w: u32, lo: u32, n: u32) -> u32 {
    (w >> lo) & ((1u64 << n) - 1) as u32
}

fn bit(w: u32, b: u32) -> bool {
    (w >> b) & 1 != 0
}

struct Reader<'a> {
    words: &'a [u32],
}

impl Reader<'_> {
    fn get(&self, i: usize) -> Result<u32, DecodeError> {
        self.words.get(i).copied().ok_or(DecodeError::Truncated)
    }
}

/// Decodes the instruction at the start of `words`.
pub fn decode(words: &[u32]) -> Result<Inst, DecodeError> {
    let r = Reader { words };
    let w = r.get(0)?;
    let bad = DecodeError::Invalid(w);
    let op = |fmt, code: u32| lookup(fmt, code as u16).ok_or(bad);
    let top9 = w >> 23;
    let top6 = w >> 26;

    let (op, kind, mut len) = if top9 == 0x17D {
        let kind = Kind::Sop1 {
            sdst: bits(w, 16, 7) as u8,
            ssrc0: bits(w, 0, 8) as u16,
        };
        (op(Fmt::Sop1, bits(w, 8, 8))?, kind, 1)
    } else if top9 == 0x17E {
        let kind = Kind::Sopc {
            ssrc0: bits(w, 0, 8) as u16,
            ssrc1: bits(w, 8, 8) as u16,
        };
        (op(Fmt::Sopc, bits(w, 16, 7))?, kind, 1)
    } else if top9 == 0x17F {
        (
            op(Fmt::Sopp, bits(w, 16, 7))?,
            Kind::Sopp { simm16: w as u16 },
            1,
        )
    } else if w >> 28 == 0xB {
        let kind = Kind::Sopk {
            sdst: bits(w, 16, 7) as u8,
            simm16: w as u16,
        };
        (op(Fmt::Sopk, bits(w, 23, 5))?, kind, 1)
    } else if w >> 30 == 0b10 {
        let kind = Kind::Sop2 {
            sdst: bits(w, 16, 7) as u8,
            ssrc0: bits(w, 0, 8) as u16,
            ssrc1: bits(w, 8, 8) as u16,
        };
        (op(Fmt::Sop2, bits(w, 23, 7))?, kind, 1)
    } else if w >> 31 == 0 {
        return decode_vop_short(&r, w);
    } else {
        match top6 {
            0x3D => {
                let w1 = r.get(1)?;
                let offset = ((bits(w1, 0, 21) << 11) as i32) >> 11;
                let kind = Kind::Smem {
                    sbase: (bits(w, 0, 6) << 1) as u8,
                    sdata: bits(w, 6, 7) as u8,
                    dlc: bit(w, 14),
                    glc: bit(w, 16),
                    offset,
                    soffset: bits(w1, 25, 7) as u8,
                };
                (op(Fmt::Smem, bits(w, 18, 8))?, kind, 2)
            }
            0x35 => return decode_vop3(&r, w),
            0x33 => return decode_vop3p(&r, w),
            0x32 => {
                let kind = Kind::Vintrp {
                    vsrc: bits(w, 0, 8) as u8,
                    chan: bits(w, 8, 2) as u8,
                    attr: bits(w, 10, 6) as u8,
                    vdst: bits(w, 18, 8) as u8,
                };
                (op(Fmt::Vintrp, bits(w, 16, 2))?, kind, 1)
            }
            0x36 => {
                let w1 = r.get(1)?;
                let kind = Kind::Ds {
                    offset0: bits(w, 0, 8) as u8,
                    offset1: bits(w, 8, 8) as u8,
                    gds: bit(w, 17),
                    addr: bits(w1, 0, 8) as u8,
                    data0: bits(w1, 8, 8) as u8,
                    data1: bits(w1, 16, 8) as u8,
                    vdst: bits(w1, 24, 8) as u8,
                };
                (op(Fmt::Ds, bits(w, 18, 8))?, kind, 2)
            }
            0x38 | 0x3A => {
                let w1 = r.get(1)?;
                let mtbuf = top6 == 0x3A;
                let b = Buf {
                    offset: bits(w, 0, 12) as u16,
                    offen: bit(w, 12),
                    idxen: bit(w, 13),
                    glc: bit(w, 14),
                    dlc: bit(w, 15),
                    lds: !mtbuf && bit(w, 16),
                    format: if mtbuf { bits(w, 19, 7) as u8 } else { 0 },
                    vaddr: bits(w1, 0, 8) as u8,
                    vdata: bits(w1, 8, 8) as u8,
                    srsrc: (bits(w1, 16, 5) << 2) as u8,
                    slc: bit(w1, 22),
                    tfe: bit(w1, 23),
                    soffset: bits(w1, 24, 8) as u8,
                };
                if mtbuf {
                    let code = bits(w, 16, 3) | bits(w1, 21, 1) << 3;
                    (op(Fmt::Mtbuf, code)?, Kind::Mtbuf(b), 2)
                } else {
                    if bit(w1, 21) {
                        return Err(bad);
                    }
                    let code = bits(w, 18, 7) | bits(w, 25, 1) << 7;
                    (op(Fmt::Mubuf, code)?, Kind::Mubuf(b), 2)
                }
            }
            0x3C => {
                let w1 = r.get(1)?;
                let nsa = bits(w, 1, 2) as u8;
                let mut vaddr = [0u8; 13];
                vaddr[0] = bits(w1, 0, 8) as u8;
                for i in 0..nsa as usize {
                    let wn = r.get(2 + i)?;
                    for j in 0..4 {
                        vaddr[1 + i * 4 + j] = (wn >> (8 * j)) as u8;
                    }
                }
                let m = Mimg {
                    nsa,
                    dim: bits(w, 3, 3) as u8,
                    dlc: bit(w, 7),
                    dmask: bits(w, 8, 4) as u8,
                    unorm: bit(w, 12),
                    glc: bit(w, 13),
                    r128: bit(w, 15),
                    tfe: bit(w, 16),
                    lwe: bit(w, 17),
                    slc: bit(w, 25),
                    vaddr,
                    vdata: bits(w1, 8, 8) as u8,
                    srsrc: (bits(w1, 16, 5) << 2) as u8,
                    ssamp: (bits(w1, 21, 5) << 2) as u8,
                    a16: bit(w1, 30),
                    d16: bit(w1, 31),
                };
                let code = sample_adjust_base(bits(w, 18, 7) | bits(w, 0, 1) << 7);
                (op(Fmt::Mimg, code)?, Kind::Mimg(m), 2 + nsa)
            }
            0x3E => {
                let w1 = r.get(1)?;
                let kind = Kind::Exp {
                    en: bits(w, 0, 4) as u8,
                    target: bits(w, 4, 6) as u8,
                    compr: bit(w, 10),
                    done: bit(w, 11),
                    vm: bit(w, 12),
                    vsrc: w1.to_le_bytes(),
                };
                (op(Fmt::Exp, 0)?, kind, 2)
            }
            0x37 => {
                let w1 = r.get(1)?;
                let fmt = match bits(w, 14, 2) {
                    0 => Fmt::Flat,
                    1 => Fmt::Scratch,
                    2 => Fmt::Global,
                    _ => return Err(bad),
                };
                let off = bits(w, 0, 12) as i16;
                let f = Flat {
                    offset: if fmt == Fmt::Flat {
                        off
                    } else {
                        (off << 4) >> 4
                    },
                    dlc: bit(w, 12),
                    lds: bit(w, 13),
                    glc: bit(w, 16),
                    slc: bit(w, 17),
                    addr: bits(w1, 0, 8) as u8,
                    data: bits(w1, 8, 8) as u8,
                    saddr: bits(w1, 16, 7) as u8,
                    nv: bit(w1, 23),
                    vdst: bits(w1, 24, 8) as u8,
                };
                (op(fmt, bits(w, 18, 7))?, Kind::Flat(f), 2)
            }
            _ => return Err(bad),
        }
    };
    let mut literal = None;
    let wants_literal = match kind {
        Kind::Sop2 { ssrc0, ssrc1, .. } | Kind::Sopc { ssrc0, ssrc1 } => {
            ssrc0 == src::LITERAL || ssrc1 == src::LITERAL
        }
        Kind::Sop1 { ssrc0, .. } => ssrc0 == src::LITERAL,
        Kind::Sopk { .. } => op.name == "s_setreg_imm32_b32",
        _ => false,
    };
    if wants_literal {
        literal = Some(r.get(len as usize)?);
        len += 1;
    }
    Ok(Inst {
        op,
        kind,
        literal,
        words: len,
    })
}

/// The PS5's sampler-adjust image sample forms (`0xA0` to `0xBE`, not in
/// public RDNA2 docs) as their base sample instruction, `0x80` lower: same
/// operands, only reserved sampler bits differ (SharpEmu-mac's description).
fn sample_adjust_base(code: u32) -> u32 {
    match code {
        0xa0 | 0xa1 | 0xa5 | 0xa6 | 0xa8 | 0xa9 | 0xad | 0xae | 0xb0 | 0xb1 | 0xb5 | 0xb6 | 0xb8 | 0xb9 | 0xbd
        | 0xbe => code - 0x80,
        _ => code,
    }
}

fn empty_valu(enc: VEnc) -> Valu {
    Valu {
        enc,
        vdst: 0,
        sdst: None,
        src: [0; 3],
        abs: 0,
        neg: 0,
        neg_hi: 0,
        opsel: 0,
        opsel_hi: 0,
        clamp: false,
        omod: 0,
        sdwa: Sdwa::default(),
        dpp: Dpp::default(),
    }
}

// Opcodes that always carry a literal constant K in their 32-bit form.
fn has_k(op: &OpInfo) -> bool {
    op.srcs.contains(&crate::opinfo::Ty::Imm)
}

fn decode_vop_short(r: &Reader, w: u32) -> Result<Inst, DecodeError> {
    let bad = DecodeError::Invalid(w);
    let src0 = bits(w, 0, 9) as u16;
    let (op, mut v) = match w >> 25 {
        0x3F => {
            let op = lookup(Fmt::Vop1, bits(w, 9, 8) as u16).ok_or(bad)?;
            let mut v = empty_valu(VEnc::E32);
            v.vdst = bits(w, 17, 8) as u8;
            v.src[0] = src0;
            (op, v)
        }
        0x3E => {
            let op = lookup(Fmt::Vopc, bits(w, 17, 8) as u16).ok_or(bad)?;
            let mut v = empty_valu(VEnc::E32);
            v.src = [src0, src::VGPR0 + bits(w, 9, 8) as u16, 0];
            (op, v)
        }
        code => {
            let op = lookup(Fmt::Vop2, code as u16).ok_or(bad)?;
            if code == 0 && w != 0 {
                return Err(bad);
            }
            let mut v = empty_valu(VEnc::E32);
            v.vdst = bits(w, 17, 8) as u8;
            v.src = [src0, src::VGPR0 + bits(w, 9, 8) as u16, 0];
            (op, v)
        }
    };
    let mut len = 1;
    let mut literal = None;
    match src0 {
        src::SDWA => {
            let x = r.get(1)?;
            len = 2;
            v.enc = VEnc::Sdwa;
            v.src[0] = bits(x, 0, 8) as u16 + if bit(x, 23) { 0 } else { src::VGPR0 };
            if op.fmt != Fmt::Vop1 && bit(x, 31) {
                v.src[1] -= src::VGPR0;
            }
            v.sdwa = Sdwa {
                dst_sel: bits(x, 8, 3) as u8,
                dst_unused: bits(x, 11, 2) as u8,
                src_sel: [bits(x, 16, 3) as u8, bits(x, 24, 3) as u8],
                sext: [bit(x, 19), bit(x, 27)],
            };
            if op.fmt == Fmt::Vopc {
                if bit(x, 15) {
                    v.sdst = Some(bits(x, 8, 7) as u8);
                }
                v.sdwa.dst_sel = 0;
                v.sdwa.dst_unused = 0;
            } else {
                v.clamp = bit(x, 13);
                v.omod = bits(x, 14, 2) as u8;
            }
            v.neg = bit(x, 20) as u8 | (bit(x, 28) as u8) << 1;
            v.abs = bit(x, 21) as u8 | (bit(x, 29) as u8) << 1;
        }
        src::DPP16 => {
            let x = r.get(1)?;
            len = 2;
            v.enc = VEnc::Dpp16;
            v.src[0] = src::VGPR0 + bits(x, 0, 8) as u16;
            v.dpp = Dpp {
                ctrl: bits(x, 8, 9) as u16,
                fi: bit(x, 18),
                bound_ctrl: bit(x, 19),
                bank_mask: bits(x, 24, 4) as u8,
                row_mask: bits(x, 28, 4) as u8,
                sel8: 0,
            };
            v.neg = bit(x, 20) as u8 | (bit(x, 22) as u8) << 1;
            v.abs = bit(x, 21) as u8 | (bit(x, 23) as u8) << 1;
        }
        src::DPP8 | src::DPP8_FI => {
            let x = r.get(1)?;
            len = 2;
            v.enc = VEnc::Dpp8;
            v.src[0] = src::VGPR0 + bits(x, 0, 8) as u16;
            v.dpp.sel8 = x >> 8;
            v.dpp.fi = src0 == src::DPP8_FI;
        }
        src::LITERAL => {
            literal = Some(r.get(1)?);
            len = 2;
        }
        _ => {}
    }
    if has_k(op) {
        if v.enc != VEnc::E32 {
            return Err(bad);
        }
        if literal.is_none() {
            literal = Some(r.get(1)?);
            len = 2;
        }
    }
    Ok(Inst {
        op,
        kind: Kind::Valu(v),
        literal,
        words: len,
    })
}

/// Finds what a VOP3 opcode number means: a VOP3-only opcode or a promoted VOP1, VOP2 or VOPC one.
pub fn vop3_op(code: u16) -> Option<&'static OpInfo> {
    if let Some(op) = lookup(Fmt::Vop3, code) {
        return Some(op);
    }
    let op = match code {
        0..=0xFF => lookup(Fmt::Vopc, code),
        0x100..=0x13F => lookup(Fmt::Vop2, code - 0x100),
        0x180..=0x1FF => lookup(Fmt::Vop1, code - 0x180),
        0x200..=0x202 => lookup(Fmt::Vintrp, code - 0x200),
        _ => None,
    }?;
    let e32_only = matches!(
        op.name,
        "v_readfirstlane_b32" | "v_swap_b32" | "v_swaprel_b32" | "v_pk_fmac_f16" | "v_illegal"
    );
    (!has_k(op) && !e32_only).then_some(op)
}

/// Whether a VOP3 opcode uses the VOP3b layout (a scalar destination in place of abs/opsel).
pub fn is_vop3b(op: &OpInfo) -> bool {
    matches!(op.fmt, Fmt::Vop2 | Fmt::Vop3)
        && op.defs.len() == 2
        && op.defs[1] == crate::opinfo::Ty::Lane
}

fn literal_if_used(r: &Reader, srcs: &[u16]) -> Result<(Option<u32>, u8), DecodeError> {
    if srcs.contains(&src::LITERAL) {
        Ok((Some(r.get(2)?), 3))
    } else {
        Ok((None, 2))
    }
}

fn decode_vop3(r: &Reader, w: u32) -> Result<Inst, DecodeError> {
    let w1 = r.get(1)?;
    let op = vop3_op(bits(w, 16, 10) as u16).ok_or(DecodeError::Invalid(w))?;
    let mut v = empty_valu(VEnc::E64);
    v.vdst = bits(w, 0, 8) as u8;
    if is_vop3b(op) {
        v.sdst = Some(bits(w, 8, 7) as u8);
    } else {
        v.abs = bits(w, 8, 3) as u8;
        v.opsel = bits(w, 11, 4) as u8;
    }
    v.clamp = bit(w, 15);
    v.src = [
        bits(w1, 0, 9) as u16,
        bits(w1, 9, 9) as u16,
        bits(w1, 18, 9) as u16,
    ];
    v.omod = bits(w1, 27, 2) as u8;
    v.neg = bits(w1, 29, 3) as u8;
    let n = op.srcs.len().min(3);
    let first = if op.name.starts_with("v_interp_") {
        1
    } else {
        0
    };
    let (literal, words) = literal_if_used(r, &v.src[first..n])?;
    Ok(Inst {
        op,
        kind: Kind::Valu(v),
        literal,
        words,
    })
}

fn decode_vop3p(r: &Reader, w: u32) -> Result<Inst, DecodeError> {
    let w1 = r.get(1)?;
    let op = lookup(Fmt::Vop3p, bits(w, 16, 7) as u16).ok_or(DecodeError::Invalid(w))?;
    let mut v = empty_valu(VEnc::P);
    v.vdst = bits(w, 0, 8) as u8;
    v.neg_hi = bits(w, 8, 3) as u8;
    v.opsel = bits(w, 11, 3) as u8;
    v.opsel_hi = bits(w1, 27, 2) as u8 | (bit(w, 14) as u8) << 2;
    v.clamp = bit(w, 15);
    v.src = [
        bits(w1, 0, 9) as u16,
        bits(w1, 9, 9) as u16,
        bits(w1, 18, 9) as u16,
    ];
    v.neg = bits(w1, 29, 3) as u8;
    let n = op.srcs.len().min(3);
    let (literal, words) = literal_if_used(r, &v.src[..n])?;
    Ok(Inst {
        op,
        kind: Kind::Valu(v),
        literal,
        words,
    })
}

/// Instructions with their dword offsets, and where decoding stopped with an error.
pub type Decoded = (Vec<(usize, Inst)>, Option<(usize, DecodeError)>);

/// Decodes a whole program, stopping at the first error.
pub fn decode_all(words: &[u32]) -> Decoded {
    let mut out = Vec::new();
    let mut pc = 0;
    while pc < words.len() {
        match decode(&words[pc..]) {
            Ok(i) => {
                let n = i.words as usize;
                out.push((pc, i));
                pc += n;
            }
            Err(e) => return (out, Some((pc, e))),
        }
    }
    (out, None)
}
