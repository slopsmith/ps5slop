//! Compact opcode keys and readable mnemonic names for JIT fallback reports.

use crate::decode::{Insn, Map, Rep};

/// Grouping key for a decoded x86-64 instruction in fallback and profile reports.
///
/// Only records ModRM `ext` for actual group opcodes, and `mem` only where the
/// memory and register forms are distinct instructions, so different register
/// choices collapse into one row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpcodeKey {
    pub map: u8,
    pub op: u8,
    /// Group extension `0..=7`, or `0xff` when ModRM `reg` is a register operand.
    pub ext: u8,
    /// Mandatory / size prefix: 0 none, 1 `66`, 2 `F3`, 3 `F2`.
    pub pfx: u8,
    /// 0 legacy, 1 VEX.128, 2 VEX.256.
    pub vex: u8,
    pub w: bool,
    pub mem: bool,
    pub lock: bool,
}

impl OpcodeKey {
    pub fn from_insn(i: &Insn) -> Self {
        let map = match i.map {
            Map::Primary => 0,
            Map::M0F => 1,
            Map::M0F38 => 2,
            Map::M0F3A => 3,
        };
        let pfx = match (i.rep, i.opsize_prefix) {
            (Rep::Rep, _) => 2,
            (Rep::Repne, _) => 3,
            (Rep::None, true) => 1,
            (Rep::None, false) => 0,
        };
        let vex = match i.vex {
            None => 0,
            Some(v) if v.l => 2,
            Some(_) => 1,
        };
        let ext = if i.modrm.is_some() && has_ext(map, i.op, pfx, vex != 0) {
            i.ext()
        } else {
            0xff
        };
        let mem = mem_distinguishes(map, i.op) && i.mem.is_some();
        Self {
            map,
            op: i.op,
            ext,
            pfx,
            vex,
            w: i.rex_w(),
            mem,
            lock: i.lock,
        }
    }

    /// Formats the key as `mnemonic (encoding)`, or the encoding alone if unknown.
    pub fn describe(&self) -> String {
        let enc = self.encoding();
        match self.mnemonic() {
            Some(name) => format!("{name} ({enc})"),
            None => enc,
        }
    }

    /// Compact x86 encoding string, e.g. `VEX.256.66.0F38 16` or `80 /2`.
    pub fn encoding(&self) -> String {
        let mut s = String::new();
        if self.lock {
            s.push_str("lock ");
        }
        let pfx_str = match self.pfx {
            1 => "66",
            2 => "F3",
            3 => "F2",
            _ => "",
        };
        if self.vex != 0 {
            s.push_str(if self.vex == 2 { "VEX.256" } else { "VEX.128" });
            if !pfx_str.is_empty() {
                s.push('.');
                s.push_str(pfx_str);
            }
            s.push('.');
            s.push_str(match self.map {
                1 => "0F",
                2 => "0F38",
                3 => "0F3A",
                _ => "P",
            });
            if self.w {
                s.push_str(".W1");
            }
            s.push(' ');
        } else {
            if !pfx_str.is_empty() {
                s.push_str(pfx_str);
                s.push(' ');
            }
            if self.w {
                s.push_str("REX.W ");
            }
            match self.map {
                1 => s.push_str("0F "),
                2 => s.push_str("0F 38 "),
                3 => s.push_str("0F 3A "),
                _ => {}
            }
        }
        use std::fmt::Write;
        let _ = write!(s, "{:02X}", self.op);
        if self.ext != 0xff {
            let _ = write!(s, " /{}", self.ext);
        }
        if self.mem {
            s.push_str(" m");
        }
        s
    }

    /// Best-effort Intel mnemonic for reporting.
    pub fn mnemonic(&self) -> Option<&'static str> {
        let v = self.vex != 0;
        let p = self.pfx;
        let w = self.w;
        Some(match self.map {
            0 => match self.op {
                0x00..=0x05 => "add",
                0x08..=0x0d => "or",
                0x10..=0x15 => "adc",
                0x18..=0x1d => "sbb",
                0x20..=0x25 => "and",
                0x28..=0x2d => "sub",
                0x30..=0x35 => "xor",
                0x38..=0x3d => "cmp",
                0x50..=0x57 => "push",
                0x58..=0x5f => "pop",
                0x63 => "movsxd",
                0x68 | 0x6a => "push",
                0x69 | 0x6b => "imul",
                0x70..=0x7f => "jcc",
                0x80..=0x83 => match self.ext {
                    0 => "add",
                    1 => "or",
                    2 => "adc",
                    3 => "sbb",
                    4 => "and",
                    5 => "sub",
                    6 => "xor",
                    7 => "cmp",
                    _ => return None,
                },
                0x84 | 0x85 => "test",
                0x86 | 0x87 => "xchg",
                0x88..=0x8c | 0x8e => "mov",
                0x8d => "lea",
                0x8f if self.ext == 0 => "pop",
                0x90 if p == 2 => "pause",
                0x90 => "nop",
                0x91..=0x97 => "xchg",
                0x98 => match (w, p) {
                    (true, _) => "cdqe",
                    (false, 1) => "cbw",
                    _ => "cwde",
                },
                0x99 => match (w, p) {
                    (true, _) => "cqo",
                    (false, 1) => "cwd",
                    _ => "cdq",
                },
                0x9b => "fwait",
                0x9c => "pushf",
                0x9d => "popf",
                0x9e => "sahf",
                0x9f => "lahf",
                0xa0..=0xa3 => "mov",
                0xa4 | 0xa5 => "movs",
                0xa6 | 0xa7 => "cmps",
                0xa8 | 0xa9 => "test",
                0xaa | 0xab => "stos",
                0xac | 0xad => "lods",
                0xae | 0xaf => "scas",
                0xb0..=0xbf => "mov",
                0xc0 | 0xc1 | 0xd0..=0xd3 => match self.ext {
                    0 => "rol",
                    1 => "ror",
                    2 => "rcl",
                    3 => "rcr",
                    4 | 6 => "shl",
                    5 => "shr",
                    7 => "sar",
                    _ => return None,
                },
                0xc2 | 0xc3 => "ret",
                0xc6 | 0xc7 if self.ext == 0 => "mov",
                0xc8 => "enter",
                0xc9 => "leave",
                0xcc => "int3",
                0xcd => "int",
                0xd8..=0xdf => x87_mnemonic(self.op, self.ext, self.mem)?,
                0xe0 => "loopne",
                0xe1 => "loope",
                0xe2 => "loop",
                0xe3 => "jrcxz",
                0xe8 => "call",
                0xe9 | 0xeb => "jmp",
                0xf5 => "cmc",
                0xf6 | 0xf7 => match self.ext {
                    0 | 1 => "test",
                    2 => "not",
                    3 => "neg",
                    4 => "mul",
                    5 => "imul",
                    6 => "div",
                    7 => "idiv",
                    _ => return None,
                },
                0xf8 => "clc",
                0xf9 => "stc",
                0xfc => "cld",
                0xfd => "std",
                0xfe | 0xff => match self.ext {
                    0 => "inc",
                    1 => "dec",
                    2 => "call",
                    4 => "jmp",
                    6 => "push",
                    _ => return None,
                },
                _ => return None,
            },
            1 => match self.op {
                0x0b => "ud2",
                0x0d => "prefetchw",
                0x10 | 0x11 => match (v, p) {
                    (false, 0) => "movups",
                    (false, 1) => "movupd",
                    (false, 2) => "movss",
                    (false, 3) => "movsd",
                    (true, 0) => "vmovups",
                    (true, 1) => "vmovupd",
                    (true, 2) => "vmovss",
                    (true, 3) => "vmovsd",
                    _ => return None,
                },
                0x12 => match (v, p, self.mem) {
                    (false, 0, true) => "movlps",
                    (false, 0, false) => "movhlps",
                    (false, 1, _) => "movlpd",
                    (false, 2, _) => "movsldup",
                    (false, 3, _) => "movddup",
                    (true, 0, true) => "vmovlps",
                    (true, 0, false) => "vmovhlps",
                    (true, 1, _) => "vmovlpd",
                    (true, 2, _) => "vmovsldup",
                    (true, 3, _) => "vmovddup",
                    _ => return None,
                },
                0x13 => match (v, p) {
                    (false, 0) => "movlps",
                    (false, 1) => "movlpd",
                    (true, 0) => "vmovlps",
                    (true, 1) => "vmovlpd",
                    _ => return None,
                },
                0x14 => pick_vp(v, p, "unpcklps", "unpcklpd", "vunpcklps", "vunpcklpd")?,
                0x15 => pick_vp(v, p, "unpckhps", "unpckhpd", "vunpckhps", "vunpckhpd")?,
                0x16 => match (v, p, self.mem) {
                    (false, 0, true) => "movhps",
                    (false, 0, false) => "movlhps",
                    (false, 1, _) => "movhpd",
                    (false, 2, _) => "movshdup",
                    (true, 0, true) => "vmovhps",
                    (true, 0, false) => "vmovlhps",
                    (true, 1, _) => "vmovhpd",
                    (true, 2, _) => "vmovshdup",
                    _ => return None,
                },
                0x17 => pick_vp(v, p, "movhps", "movhpd", "vmovhps", "vmovhpd")?,
                0x18 => "prefetch",
                0x1f => "nop",
                0x28 | 0x29 => pick_vp(v, p, "movaps", "movapd", "vmovaps", "vmovapd")?,
                0x2a => match (v, p) {
                    (false, 0) => "cvtpi2ps",
                    (false, 1) => "cvtpi2pd",
                    (false, 2) => "cvtsi2ss",
                    (false, 3) => "cvtsi2sd",
                    (true, 2) => "vcvtsi2ss",
                    (true, 3) => "vcvtsi2sd",
                    _ => return None,
                },
                0x2b => match (v, p) {
                    (false, 0) => "movntps",
                    (false, 1) => "movntpd",
                    (false, 2) => "movntss",
                    (false, 3) => "movntsd",
                    (true, 0) => "vmovntps",
                    (true, 1) => "vmovntpd",
                    _ => return None,
                },
                0x2c => match (v, p) {
                    (false, 2) => "cvttss2si",
                    (false, 3) => "cvttsd2si",
                    (true, 2) => "vcvttss2si",
                    (true, 3) => "vcvttsd2si",
                    _ => return None,
                },
                0x2d => match (v, p) {
                    (false, 2) => "cvtss2si",
                    (false, 3) => "cvtsd2si",
                    (true, 2) => "vcvtss2si",
                    (true, 3) => "vcvtsd2si",
                    _ => return None,
                },
                0x2e => pick_vp(v, p, "ucomiss", "ucomisd", "vucomiss", "vucomisd")?,
                0x2f => pick_vp(v, p, "comiss", "comisd", "vcomiss", "vcomisd")?,
                0x31 => "rdtsc",
                0x40..=0x4f => "cmovcc",
                0x50 => pick_vp(v, p, "movmskps", "movmskpd", "vmovmskps", "vmovmskpd")?,
                0x51 => pick_fp4(
                    v, p, "sqrtps", "sqrtpd", "sqrtss", "sqrtsd", "vsqrtps", "vsqrtpd", "vsqrtss",
                    "vsqrtsd",
                )?,
                0x52 => match (v, p) {
                    (false, 0) => "rsqrtps",
                    (false, 2) => "rsqrtss",
                    (true, 0) => "vrsqrtps",
                    (true, 2) => "vrsqrtss",
                    _ => return None,
                },
                0x53 => match (v, p) {
                    (false, 0) => "rcpps",
                    (false, 2) => "rcpss",
                    (true, 0) => "vrcpps",
                    (true, 2) => "vrcpss",
                    _ => return None,
                },
                0x54 => pick_vp(v, p, "andps", "andpd", "vandps", "vandpd")?,
                0x55 => pick_vp(v, p, "andnps", "andnpd", "vandnps", "vandnpd")?,
                0x56 => pick_vp(v, p, "orps", "orpd", "vorps", "vorpd")?,
                0x57 => pick_vp(v, p, "xorps", "xorpd", "vxorps", "vxorpd")?,
                0x58 => pick_fp4(
                    v, p, "addps", "addpd", "addss", "addsd", "vaddps", "vaddpd", "vaddss",
                    "vaddsd",
                )?,
                0x59 => pick_fp4(
                    v, p, "mulps", "mulpd", "mulss", "mulsd", "vmulps", "vmulpd", "vmulss",
                    "vmulsd",
                )?,
                0x5a => pick_fp4(
                    v,
                    p,
                    "cvtps2pd",
                    "cvtpd2ps",
                    "cvtss2sd",
                    "cvtsd2ss",
                    "vcvtps2pd",
                    "vcvtpd2ps",
                    "vcvtss2sd",
                    "vcvtsd2ss",
                )?,
                0x5b => match (v, p) {
                    (false, 0) => "cvtdq2ps",
                    (false, 1) => "cvtps2dq",
                    (false, 2) => "cvttps2dq",
                    (true, 0) => "vcvtdq2ps",
                    (true, 1) => "vcvtps2dq",
                    (true, 2) => "vcvttps2dq",
                    _ => return None,
                },
                0x5c => pick_fp4(
                    v, p, "subps", "subpd", "subss", "subsd", "vsubps", "vsubpd", "vsubss",
                    "vsubsd",
                )?,
                0x5d => pick_fp4(
                    v, p, "minps", "minpd", "minss", "minsd", "vminps", "vminpd", "vminss",
                    "vminsd",
                )?,
                0x5e => pick_fp4(
                    v, p, "divps", "divpd", "divss", "divsd", "vdivps", "vdivpd", "vdivss",
                    "vdivsd",
                )?,
                0x5f => pick_fp4(
                    v, p, "maxps", "maxpd", "maxss", "maxsd", "vmaxps", "vmaxpd", "vmaxss",
                    "vmaxsd",
                )?,
                0x6e => {
                    if v {
                        if w { "vmovq" } else { "vmovd" }
                    } else if w {
                        "movq"
                    } else {
                        "movd"
                    }
                }
                0x6f => match (v, p) {
                    (false, 1) => "movdqa",
                    (false, 2) => "movdqu",
                    (true, 1) => "vmovdqa",
                    (true, 2) => "vmovdqu",
                    _ => return None,
                },
                0x70 => match (v, p) {
                    (false, 1) => "pshufd",
                    (false, 2) => "pshufhw",
                    (false, 3) => "pshuflw",
                    (true, 1) => "vpshufd",
                    (true, 2) => "vpshufhw",
                    (true, 3) => "vpshuflw",
                    _ => return None,
                },
                0x71 => match self.ext {
                    2 => {
                        if v {
                            "vpsrlw"
                        } else {
                            "psrlw"
                        }
                    }
                    4 => {
                        if v {
                            "vpsraw"
                        } else {
                            "psraw"
                        }
                    }
                    6 => {
                        if v {
                            "vpsllw"
                        } else {
                            "psllw"
                        }
                    }
                    _ => return None,
                },
                0x72 => match self.ext {
                    2 => {
                        if v {
                            "vpsrld"
                        } else {
                            "psrld"
                        }
                    }
                    4 => {
                        if v {
                            "vpsrad"
                        } else {
                            "psrad"
                        }
                    }
                    6 => {
                        if v {
                            "vpslld"
                        } else {
                            "pslld"
                        }
                    }
                    _ => return None,
                },
                0x73 => match self.ext {
                    2 => {
                        if v {
                            "vpsrlq"
                        } else {
                            "psrlq"
                        }
                    }
                    3 => {
                        if v {
                            "vpsrldq"
                        } else {
                            "psrldq"
                        }
                    }
                    6 => {
                        if v {
                            "vpsllq"
                        } else {
                            "psllq"
                        }
                    }
                    7 => {
                        if v {
                            "vpslldq"
                        } else {
                            "pslldq"
                        }
                    }
                    _ => return None,
                },
                0x77 => {
                    if v {
                        if self.vex == 2 {
                            "vzeroall"
                        } else {
                            "vzeroupper"
                        }
                    } else {
                        "emms"
                    }
                }
                0x78 | 0x79 => match (v, p) {
                    (false, 1) => "extrq",
                    (false, 3) => "insertq",
                    _ => return None,
                },
                0x7c => match (v, p) {
                    (false, 1) => "haddpd",
                    (false, 3) => "haddps",
                    (true, 1) => "vhaddpd",
                    (true, 3) => "vhaddps",
                    _ => return None,
                },
                0x7d => match (v, p) {
                    (false, 1) => "hsubpd",
                    (false, 3) => "hsubps",
                    (true, 1) => "vhsubpd",
                    (true, 3) => "vhsubps",
                    _ => return None,
                },
                0x7e => match (v, p) {
                    (false, 1) => {
                        if w {
                            "movq"
                        } else {
                            "movd"
                        }
                    }
                    (false, 2) => "movq",
                    (true, 1) => {
                        if w {
                            "vmovq"
                        } else {
                            "vmovd"
                        }
                    }
                    (true, 2) => "vmovq",
                    _ => return None,
                },
                0x7f => match (v, p) {
                    (false, 1) => "movdqa",
                    (false, 2) => "movdqu",
                    (true, 1) => "vmovdqa",
                    (true, 2) => "vmovdqu",
                    _ => return None,
                },
                0x80..=0x8f => "jcc",
                0x90..=0x9f => "setcc",
                0xa2 => "cpuid",
                0xa3 => "bt",
                0xa4 | 0xa5 => "shld",
                0xab => "bts",
                0xac | 0xad => "shrd",
                0xae => match (self.ext, self.mem, p) {
                    (0, true, 0) => "fxsave",
                    (1, true, 0) => "fxrstor",
                    (2, true, 0) => {
                        if v {
                            "vldmxcsr"
                        } else {
                            "ldmxcsr"
                        }
                    }
                    (3, true, 0) => {
                        if v {
                            "vstmxcsr"
                        } else {
                            "stmxcsr"
                        }
                    }
                    (4, true, 0) => "xsave",
                    (5, true, 0) => "xrstor",
                    (5, false, 0) => "lfence",
                    (6, false, 0) => "mfence",
                    (7, false, 0) => "sfence",
                    (7, true, 0) => "clflush",
                    (7, true, 1) => "clflushopt",
                    _ => return None,
                },
                0xaf => "imul",
                0xb0 | 0xb1 => "cmpxchg",
                0xb3 => "btr",
                0xb6 | 0xb7 => "movzx",
                0xb8 if p == 2 => "popcnt",
                0xba => match self.ext {
                    4 => "bt",
                    5 => "bts",
                    6 => "btr",
                    7 => "btc",
                    _ => return None,
                },
                0xbb => "btc",
                0xbc => {
                    if p == 2 {
                        "tzcnt"
                    } else {
                        "bsf"
                    }
                }
                0xbd => {
                    if p == 2 {
                        "lzcnt"
                    } else {
                        "bsr"
                    }
                }
                0xbe | 0xbf => "movsx",
                0xc0 | 0xc1 => "xadd",
                0xc2 => pick_fp4(
                    v, p, "cmpps", "cmppd", "cmpss", "cmpsd", "vcmpps", "vcmppd", "vcmpss",
                    "vcmpsd",
                )?,
                0xc3 => "movnti",
                0xc4 => {
                    if v {
                        "vpinsrw"
                    } else {
                        "pinsrw"
                    }
                }
                0xc5 => {
                    if v {
                        "vpextrw"
                    } else {
                        "pextrw"
                    }
                }
                0xc6 => pick_vp(v, p, "shufps", "shufpd", "vshufps", "vshufpd")?,
                0xc7 => match (self.ext, self.mem) {
                    (1, true) => {
                        if w {
                            "cmpxchg16b"
                        } else {
                            "cmpxchg8b"
                        }
                    }
                    (6, false) => "rdrand",
                    (7, false) => "rdseed",
                    _ => return None,
                },
                0xc8..=0xcf => "bswap",
                0xd0 => match (v, p) {
                    (false, 1) => "addsubpd",
                    (false, 3) => "addsubps",
                    (true, 1) => "vaddsubpd",
                    (true, 3) => "vaddsubps",
                    _ => return None,
                },
                0xd6 => {
                    if v {
                        "vmovq"
                    } else {
                        "movq"
                    }
                }
                0xd7 => {
                    if v {
                        "vpmovmskb"
                    } else {
                        "pmovmskb"
                    }
                }
                0xe6 => match (v, p) {
                    (false, 1) => "cvttpd2dq",
                    (false, 2) => "cvtdq2pd",
                    (false, 3) => "cvtpd2dq",
                    (true, 1) => "vcvttpd2dq",
                    (true, 2) => "vcvtdq2pd",
                    (true, 3) => "vcvtpd2dq",
                    _ => return None,
                },
                0xe7 => {
                    if v {
                        "vmovntdq"
                    } else {
                        "movntdq"
                    }
                }
                0xf0 if p == 3 => {
                    if v {
                        "vlddqu"
                    } else {
                        "lddqu"
                    }
                }
                _ => return None,
            },
            2 => match self.op {
                0x00 => {
                    if v {
                        "vpshufb"
                    } else {
                        "pshufb"
                    }
                }
                0x01 => {
                    if v {
                        "vphaddw"
                    } else {
                        "phaddw"
                    }
                }
                0x02 => {
                    if v {
                        "vphaddd"
                    } else {
                        "phaddd"
                    }
                }
                0x03 => {
                    if v {
                        "vphaddsw"
                    } else {
                        "phaddsw"
                    }
                }
                0x04 => {
                    if v {
                        "vpmaddubsw"
                    } else {
                        "pmaddubsw"
                    }
                }
                0x05 => {
                    if v {
                        "vphsubw"
                    } else {
                        "phsubw"
                    }
                }
                0x06 => {
                    if v {
                        "vphsubd"
                    } else {
                        "phsubd"
                    }
                }
                0x07 => {
                    if v {
                        "vphsubsw"
                    } else {
                        "phsubsw"
                    }
                }
                0x08 => {
                    if v {
                        "vpsignb"
                    } else {
                        "psignb"
                    }
                }
                0x09 => {
                    if v {
                        "vpsignw"
                    } else {
                        "psignw"
                    }
                }
                0x0a => {
                    if v {
                        "vpsignd"
                    } else {
                        "psignd"
                    }
                }
                0x0b => {
                    if v {
                        "vpmulhrsw"
                    } else {
                        "pmulhrsw"
                    }
                }
                0x0c => "vpermilps",
                0x0d => "vpermilpd",
                0x0e => "vtestps",
                0x0f => "vtestpd",
                0x10 => "pblendvb",
                0x13 => "vcvtph2ps",
                0x14 => "blendvps",
                0x15 => "blendvpd",
                0x16 => "vpermps",
                0x17 => {
                    if v {
                        "vptest"
                    } else {
                        "ptest"
                    }
                }
                0x18 => "vbroadcastss",
                0x19 => "vbroadcastsd",
                0x1a => "vbroadcastf128",
                0x1c => {
                    if v {
                        "vpabsb"
                    } else {
                        "pabsb"
                    }
                }
                0x1d => {
                    if v {
                        "vpabsw"
                    } else {
                        "pabsw"
                    }
                }
                0x1e => {
                    if v {
                        "vpabsd"
                    } else {
                        "pabsd"
                    }
                }
                0x20 => {
                    if v {
                        "vpmovsxbw"
                    } else {
                        "pmovsxbw"
                    }
                }
                0x21 => {
                    if v {
                        "vpmovsxbd"
                    } else {
                        "pmovsxbd"
                    }
                }
                0x22 => {
                    if v {
                        "vpmovsxbq"
                    } else {
                        "pmovsxbq"
                    }
                }
                0x23 => {
                    if v {
                        "vpmovsxwd"
                    } else {
                        "pmovsxwd"
                    }
                }
                0x24 => {
                    if v {
                        "vpmovsxwq"
                    } else {
                        "pmovsxwq"
                    }
                }
                0x25 => {
                    if v {
                        "vpmovsxdq"
                    } else {
                        "pmovsxdq"
                    }
                }
                0x28 => {
                    if v {
                        "vpmuldq"
                    } else {
                        "pmuldq"
                    }
                }
                0x29 => {
                    if v {
                        "vpcmpeqq"
                    } else {
                        "pcmpeqq"
                    }
                }
                0x2a => {
                    if v {
                        "vmovntdqa"
                    } else {
                        "movntdqa"
                    }
                }
                0x2b => {
                    if v {
                        "vpackusdw"
                    } else {
                        "packusdw"
                    }
                }
                0x2c => "vmaskmovps",
                0x2d => "vmaskmovpd",
                0x2e => "vmaskmovps",
                0x2f => "vmaskmovpd",
                0x30 => {
                    if v {
                        "vpmovzxbw"
                    } else {
                        "pmovzxbw"
                    }
                }
                0x31 => {
                    if v {
                        "vpmovzxbd"
                    } else {
                        "pmovzxbd"
                    }
                }
                0x32 => {
                    if v {
                        "vpmovzxbq"
                    } else {
                        "pmovzxbq"
                    }
                }
                0x33 => {
                    if v {
                        "vpmovzxwd"
                    } else {
                        "pmovzxwd"
                    }
                }
                0x34 => {
                    if v {
                        "vpmovzxwq"
                    } else {
                        "pmovzxwq"
                    }
                }
                0x35 => {
                    if v {
                        "vpmovzxdq"
                    } else {
                        "pmovzxdq"
                    }
                }
                0x36 => "vpermd",
                0x37 => {
                    if v {
                        "vpcmpgtq"
                    } else {
                        "pcmpgtq"
                    }
                }
                0x38 => {
                    if v {
                        "vpminsb"
                    } else {
                        "pminsb"
                    }
                }
                0x39 => {
                    if v {
                        "vpminsd"
                    } else {
                        "pminsd"
                    }
                }
                0x3a => {
                    if v {
                        "vpminuw"
                    } else {
                        "pminuw"
                    }
                }
                0x3b => {
                    if v {
                        "vpminud"
                    } else {
                        "pminud"
                    }
                }
                0x3c => {
                    if v {
                        "vpmaxsb"
                    } else {
                        "pmaxsb"
                    }
                }
                0x3d => {
                    if v {
                        "vpmaxsd"
                    } else {
                        "pmaxsd"
                    }
                }
                0x3e => {
                    if v {
                        "vpmaxuw"
                    } else {
                        "pmaxuw"
                    }
                }
                0x3f => {
                    if v {
                        "vpmaxud"
                    } else {
                        "pmaxud"
                    }
                }
                0x40 => {
                    if v {
                        "vpmulld"
                    } else {
                        "pmulld"
                    }
                }
                0x41 => {
                    if v {
                        "vphminposuw"
                    } else {
                        "phminposuw"
                    }
                }
                0x45 => {
                    if w {
                        "vpsrlvq"
                    } else {
                        "vpsrlvd"
                    }
                }
                0x46 => "vpsravd",
                0x47 => {
                    if w {
                        "vpsllvq"
                    } else {
                        "vpsllvd"
                    }
                }
                0x58 => "vpbroadcastd",
                0x59 => "vpbroadcastq",
                0x5a => "vbroadcasti128",
                0x78 => "vpbroadcastb",
                0x79 => "vpbroadcastw",
                0x8c | 0x8e => {
                    if w {
                        "vpmaskmovq"
                    } else {
                        "vpmaskmovd"
                    }
                }
                0x90 => {
                    if w {
                        "vpgatherdq"
                    } else {
                        "vpgatherdd"
                    }
                }
                0x91 => {
                    if w {
                        "vpgatherqq"
                    } else {
                        "vpgatherqd"
                    }
                }
                0x92 => {
                    if w {
                        "vgatherdpd"
                    } else {
                        "vgatherdps"
                    }
                }
                0x93 => {
                    if w {
                        "vgatherqpd"
                    } else {
                        "vgatherqps"
                    }
                }
                0x96..=0xbf if v => fma_mnemonic(self.op, w)?,
                0xd8..=0xdf => "aes",
                0xf0 | 0xf1 => match p {
                    3 => "crc32",
                    0 | 1 => "movbe",
                    _ => return None,
                },
                0xf2 => "andn",
                0xf3 => match self.ext {
                    1 => "blsr",
                    2 => "blsmsk",
                    3 => "blsi",
                    _ => return None,
                },
                0xf5 => match p {
                    0 => "bzhi",
                    2 => "pext",
                    3 => "pdep",
                    _ => return None,
                },
                0xf6 => match (v, p) {
                    (true, 3) => "mulx",
                    (false, 1) => "adcx",
                    (false, 2) => "adox",
                    _ => return None,
                },
                0xf7 => match p {
                    0 => "bextr",
                    1 => "shlx",
                    2 => "sarx",
                    3 => "shrx",
                    _ => return None,
                },
                _ => return None,
            },
            3 => match self.op {
                0x00 => "vpermq",
                0x01 => "vpermpd",
                0x02 => "vpblendd",
                0x04 => "vpermilps",
                0x05 => "vpermilpd",
                0x06 => "vperm2f128",
                0x08 => {
                    if v {
                        "vroundps"
                    } else {
                        "roundps"
                    }
                }
                0x09 => {
                    if v {
                        "vroundpd"
                    } else {
                        "roundpd"
                    }
                }
                0x0a => {
                    if v {
                        "vroundss"
                    } else {
                        "roundss"
                    }
                }
                0x0b => {
                    if v {
                        "vroundsd"
                    } else {
                        "roundsd"
                    }
                }
                0x0c => {
                    if v {
                        "vblendps"
                    } else {
                        "blendps"
                    }
                }
                0x0d => {
                    if v {
                        "vblendpd"
                    } else {
                        "blendpd"
                    }
                }
                0x0e => {
                    if v {
                        "vpblendw"
                    } else {
                        "pblendw"
                    }
                }
                0x0f => {
                    if v {
                        "vpalignr"
                    } else {
                        "palignr"
                    }
                }
                0x14 => {
                    if v {
                        "vpextrb"
                    } else {
                        "pextrb"
                    }
                }
                0x15 => {
                    if v {
                        "vpextrw"
                    } else {
                        "pextrw"
                    }
                }
                0x16 => {
                    if v {
                        if w { "vpextrq" } else { "vpextrd" }
                    } else if w {
                        "pextrq"
                    } else {
                        "pextrd"
                    }
                }
                0x17 => {
                    if v {
                        "vextractps"
                    } else {
                        "extractps"
                    }
                }
                0x18 => "vinsertf128",
                0x19 => "vextractf128",
                0x1d => "vcvtps2ph",
                0x20 => {
                    if v {
                        "vpinsrb"
                    } else {
                        "pinsrb"
                    }
                }
                0x21 => {
                    if v {
                        "vinsertps"
                    } else {
                        "insertps"
                    }
                }
                0x22 => {
                    if v {
                        if w { "vpinsrq" } else { "vpinsrd" }
                    } else if w {
                        "pinsrq"
                    } else {
                        "pinsrd"
                    }
                }
                0x38 => "vinserti128",
                0x39 => "vextracti128",
                0x40 => {
                    if v {
                        "vdpps"
                    } else {
                        "dpps"
                    }
                }
                0x41 => {
                    if v {
                        "vdppd"
                    } else {
                        "dppd"
                    }
                }
                0x42 => {
                    if v {
                        "vmpsadbw"
                    } else {
                        "mpsadbw"
                    }
                }
                0x44 => {
                    if v {
                        "vpclmulqdq"
                    } else {
                        "pclmulqdq"
                    }
                }
                0x46 => "vperm2i128",
                0x4a => "vblendvps",
                0x4b => "vblendvpd",
                0x4c => "vpblendvb",
                0x60 => {
                    if v {
                        "vpcmpestrm"
                    } else {
                        "pcmpestrm"
                    }
                }
                0x61 => {
                    if v {
                        "vpcmpestri"
                    } else {
                        "pcmpestri"
                    }
                }
                0x62 => {
                    if v {
                        "vpcmpistrm"
                    } else {
                        "pcmpistrm"
                    }
                }
                0x63 => {
                    if v {
                        "vpcmpistri"
                    } else {
                        "pcmpistri"
                    }
                }
                0xdf => {
                    if v {
                        "vaeskeygenassist"
                    } else {
                        "aeskeygenassist"
                    }
                }
                0xf0 => "rorx",
                _ => return None,
            },
            _ => return None,
        })
    }
}

fn has_ext(map: u8, op: u8, pfx: u8, vex: bool) -> bool {
    match map {
        0 => matches!(
            op,
            0x80..=0x83 | 0x8f | 0xc0 | 0xc1 | 0xc6 | 0xc7 | 0xd0..=0xd3 | 0xd8..=0xdf | 0xf6 | 0xf7 | 0xfe | 0xff
        ),
        1 => {
            matches!(op, 0x00 | 0x01 | 0x18 | 0x71..=0x73 | 0xae | 0xba | 0xc7)
                || (op == 0x78 && !vex && pfx == 1)
        }
        2 => op == 0xf3,
        _ => false,
    }
}

fn mem_distinguishes(map: u8, op: u8) -> bool {
    match map {
        0 => (0xd8..=0xdf).contains(&op),
        1 => matches!(op, 0x01 | 0x12 | 0x16 | 0xae | 0xc7),
        _ => false,
    }
}

fn pick_vp(
    v: bool,
    p: u8,
    ps: &'static str,
    pd: &'static str,
    vps: &'static str,
    vpd: &'static str,
) -> Option<&'static str> {
    match (v, p) {
        (false, 0) => Some(ps),
        (false, 1) => Some(pd),
        (true, 0) => Some(vps),
        (true, 1) => Some(vpd),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn pick_fp4(
    v: bool,
    p: u8,
    ps: &'static str,
    pd: &'static str,
    ss: &'static str,
    sd: &'static str,
    vps: &'static str,
    vpd: &'static str,
    vss: &'static str,
    vsd: &'static str,
) -> Option<&'static str> {
    match (v, p) {
        (false, 0) => Some(ps),
        (false, 1) => Some(pd),
        (false, 2) => Some(ss),
        (false, 3) => Some(sd),
        (true, 0) => Some(vps),
        (true, 1) => Some(vpd),
        (true, 2) => Some(vss),
        (true, 3) => Some(vsd),
        _ => None,
    }
}

fn fma_mnemonic(op: u8, w: bool) -> Option<&'static str> {
    Some(match (op & 0xf, w) {
        (0x6, false) => "vfmaddsubps",
        (0x6, true) => "vfmaddsubpd",
        (0x7, false) => "vfmsubaddps",
        (0x7, true) => "vfmsubaddpd",
        (0x8, false) => "vfmaddps",
        (0x8, true) => "vfmaddpd",
        (0x9, false) => "vfmaddss",
        (0x9, true) => "vfmaddsd",
        (0xa, false) => "vfmsubps",
        (0xa, true) => "vfmsubpd",
        (0xb, false) => "vfmsubss",
        (0xb, true) => "vfmsubsd",
        (0xc, false) => "vfnmaddps",
        (0xc, true) => "vfnmaddpd",
        (0xd, false) => "vfnmaddss",
        (0xd, true) => "vfnmaddsd",
        (0xe, false) => "vfnmsubps",
        (0xe, true) => "vfnmsubpd",
        (0xf, false) => "vfnmsubss",
        (0xf, true) => "vfnmsubsd",
        _ => return None,
    })
}

fn x87_mnemonic(op: u8, ext: u8, mem: bool) -> Option<&'static str> {
    Some(match (op, ext, mem) {
        (0xd8 | 0xdc, 0, _) => "fadd",
        (0xd8 | 0xdc, 1, _) => "fmul",
        (0xd8 | 0xdc, 2, _) => "fcom",
        (0xd8 | 0xdc, 3, _) => "fcomp",
        (0xd8 | 0xdc, 4, _) => "fsub",
        (0xd8 | 0xdc, 5, _) => "fsubr",
        (0xd8 | 0xdc, 6, _) => "fdiv",
        (0xd8 | 0xdc, 7, _) => "fdivr",
        (0xd9, 0, _) => "fld",
        (0xd9, 1, false) => "fxch",
        (0xd9, 2, true) => "fst",
        (0xd9, 3, true) => "fstp",
        (0xd9, 5, true) => "fldcw",
        (0xd9, 7, true) => "fnstcw",
        (0xda | 0xde, 0, true) => "fiadd",
        (0xda | 0xde, 1, true) => "fimul",
        (0xda | 0xde, 4, true) => "fisub",
        (0xda | 0xde, 5, true) => "fisubr",
        (0xda | 0xde, 6, true) => "fidiv",
        (0xda | 0xde, 7, true) => "fidivr",
        (0xdb | 0xdf, 0, true) => "fild",
        (0xdb | 0xdf, 1, true) => "fisttp",
        (0xdb | 0xdf, 2, true) => "fist",
        (0xdb | 0xdf, 3, true) => "fistp",
        (0xdb, 5, true) => "fld",
        (0xdb, 7, true) => "fstp",
        (0xdd, 0, true) => "fld",
        (0xdd, 1, true) => "fisttp",
        (0xdd, 2, true) => "fst",
        (0xdd, 3, true) => "fstp",
        (0xdd, 7, true) => "fnstsw",
        (0xdf, 4, false) => "fnstsw",
        (0xdf, 5, true) => "fild",
        (0xdf, 7, true) => "fistp",
        _ => "x87",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::decode;

    fn key_of(bytes: &[u8]) -> OpcodeKey {
        let i = decode(|k| bytes.get(k).copied().ok_or(())).expect("decodes");
        OpcodeKey::from_insn(&i)
    }

    #[test]
    fn non_group_instructions_collapse_across_reg_fields() {
        // vcvtph2ps ymm0, xmm1  vs  vcvtph2ps ymm5, xmm3
        let a = key_of(&[0xc4, 0xe2, 0x7d, 0x13, 0xc1]);
        let b = key_of(&[0xc4, 0xe2, 0x7d, 0x13, 0xeb]);
        assert_eq!(a, b);
        assert_eq!(a.ext, 0xff);
        assert_eq!(a.describe(), "vcvtph2ps (VEX.256.66.0F38 13)");
    }

    #[test]
    fn group_and_hot_opcodes_format_with_readable_names() {
        assert_eq!(key_of(&[0x80, 0xd0, 0x01]).describe(), "adc (80 /2)");
        assert_eq!(
            key_of(&[0x66, 0x83, 0xd8, 0x01]).describe(),
            "sbb (66 83 /3)"
        );
        assert_eq!(key_of(&[0xf6, 0xf1]).describe(), "div (F6 /6)");
        assert_eq!(key_of(&[0x66, 0xf7, 0xf9]).describe(), "idiv (66 F7 /7)");
        assert_eq!(
            key_of(&[0xc4, 0xe3, 0x7d, 0x1d, 0xc1, 0x00]).describe(),
            "vcvtps2ph (VEX.256.66.0F3A 1D)"
        );
        assert_eq!(
            key_of(&[0xc4, 0xe2, 0x6d, 0x16, 0xc1]).describe(),
            "vpermps (VEX.256.66.0F38 16)"
        );
        assert_eq!(
            key_of(&[0xc4, 0xe2, 0x6d, 0x36, 0xc1]).describe(),
            "vpermd (VEX.256.66.0F38 36)"
        );
        assert_eq!(
            key_of(&[0x66, 0x0f, 0x3a, 0x63, 0xc1, 0x08]).describe(),
            "pcmpistri (66 0F 3A 63)"
        );
        assert_eq!(
            key_of(&[0x66, 0x0f, 0x78, 0xc0, 0x04, 0x02]).describe(),
            "extrq (66 0F 78 /0)"
        );
        assert_eq!(
            key_of(&[0xf2, 0x0f, 0x79, 0xc1]).describe(),
            "insertq (F2 0F 79)"
        );
    }
}
