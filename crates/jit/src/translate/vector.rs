//! SSE, AVX and AVX2 integer instructions, and BMI, on NEON. YMM registers
//! live in the vCPU's context block (x25) and are loaded per instruction; a
//! 256-bit operation runs as two 128-bit halves. Many NEON choices follow
//! 86slopper's `translate/sse.rs`.

use a64::{Fcm, Logic, Narrow, Reg, Size, V, Width, r};
use x86::decode::{Insn, Map, Rep};

use super::{A, ADDR, B, FlagUse, T, Translator};

/// The vCPU's context block: YMM`n` at `32 * n`, its upper half 16 further.
pub const CTX: Reg = Reg(25);

const XA: V = V(0);
const XB: V = V(1);
const XR: V = V(2);
const XT: V = V(3);
const XS: V = V(4);
const XU: V = V(5);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pfx {
    None,
    P66,
    F3,
    F2,
}

fn pfx(i: &Insn) -> Pfx {
    match (i.rep, i.opsize_prefix) {
        (Rep::Rep, _) => Pfx::F3,
        (Rep::Repne, _) => Pfx::F2,
        (Rep::None, true) => Pfx::P66,
        (Rep::None, false) => Pfx::None,
    }
}

/// The opcode with its map: `0x1xx` for `0F xx`, `0x2xx` for `0F 38 xx`,
/// `0x3xx` for `0F 3A xx`.
fn key(i: &Insn) -> u16 {
    let map = match i.map {
        Map::Primary => 0,
        Map::M0F => 1,
        Map::M0F38 => 2,
        Map::M0F3A => 3,
    };
    map << 8 | i.op as u16
}

/// NEON's "three same" form of SSE2 integer opcodes: (U, size, opcode).
/// The table is 86slopper's.
fn simd_three(op: u16) -> Option<(bool, u32, u32)> {
    Some(match op {
        0x1fc => (false, 0, 0b10000),
        0x1fd => (false, 1, 0b10000),
        0x1fe => (false, 2, 0b10000),
        0x1d4 => (false, 3, 0b10000),
        0x1f8 => (true, 0, 0b10000),
        0x1f9 => (true, 1, 0b10000),
        0x1fa => (true, 2, 0b10000),
        0x1fb => (true, 3, 0b10000),
        0x1ec => (false, 0, 0b00001),
        0x1ed => (false, 1, 0b00001),
        0x1dc => (true, 0, 0b00001),
        0x1dd => (true, 1, 0b00001),
        0x1e8 => (false, 0, 0b00101),
        0x1e9 => (false, 1, 0b00101),
        0x1d8 => (true, 0, 0b00101),
        0x1d9 => (true, 1, 0b00101),
        0x174 => (true, 0, 0b10001),
        0x175 => (true, 1, 0b10001),
        0x176 => (true, 2, 0b10001),
        0x164 => (false, 0, 0b00110),
        0x165 => (false, 1, 0b00110),
        0x166 => (false, 2, 0b00110),
        0x1d5 => (false, 1, 0b10011),
        0x1e0 => (true, 0, 0b00010),
        0x1e3 => (true, 1, 0b00010),
        0x1da => (true, 0, 0b01101),
        0x1de => (true, 0, 0b01100),
        0x1ea => (false, 1, 0b01101),
        0x1ee => (false, 1, 0b01100),
        0x229 => (true, 3, 0b10001),
        0x237 => (false, 3, 0b00110),
        0x240 => (false, 2, 0b10011),
        0x238 => (false, 0, 0b01101),
        0x239 => (false, 2, 0b01101),
        0x23a => (true, 1, 0b01101),
        0x23b => (true, 2, 0b01101),
        0x23c => (false, 0, 0b01100),
        0x23d => (false, 2, 0b01100),
        0x23e => (true, 1, 0b01100),
        0x23f => (true, 2, 0b01100),
        _ => return None,
    })
}

/// Two-source integer operations done per 128-bit half.
fn packed_int(op: u16) -> bool {
    matches!(op, 0x160..=0x16d | 0x1db | 0x1df | 0x1eb | 0x1ef | 0x1e4 | 0x1e5 | 0x1f4..=0x1f6 | 0x200 | 0x204 | 0x22b)
        || simd_three(op).is_some()
}

/// Scalar FMA (`vfmadd`, `vfmsub`, `vfnmadd`, `vfnmsub` in their 132, 213
/// and 231 forms).
fn fma_scalar(op: u16) -> bool {
    matches!(op >> 4, 0x29..=0x2b) && matches!(op & 0xf, 0x9 | 0xb | 0xd | 0xf)
}

/// Packed FMA (the same families as [`fma_scalar`], on whole vectors).
fn fma_packed(op: u16) -> bool {
    matches!(op >> 4, 0x29..=0x2b) && matches!(op & 0xf, 0x8 | 0xa | 0xc | 0xe)
}

/// Whether the vector module handles `i`, and its flag use.
pub(super) fn classify(i: &Insn) -> Option<(bool, FlagUse)> {
    let none = FlagUse {
        reads: false,
        writes: false,
    };
    let writes = FlagUse {
        reads: false,
        writes: true,
    };
    let p = pfx(i);
    let wide = i.vex.is_some_and(|v| v.l);
    let op = key(i);
    if i.lock || i.seg == Some(x86::decode::Seg::Gs) {
        return None;
    }
    let ok = match op {
        0x158 | 0x159 | 0x15c..=0x15f => true,
        0x110 | 0x111 if matches!(p, Pfx::F3 | Pfx::F2) => true,
        0x12e | 0x12f => matches!(p, Pfx::None | Pfx::P66),
        0x12a | 0x12c | 0x12d => matches!(p, Pfx::F3 | Pfx::F2),
        0x15a => true,
        0x1e7 => p == Pfx::P66 && i.mem.is_some(),
        0x12b => matches!(p, Pfx::None | Pfx::P66) && i.mem.is_some(),
        0x112 | 0x116 if p == Pfx::F3 => true,
        0x112 if p == Pfx::F2 => true,
        0x112 | 0x116 => !wide && (p == Pfx::None || p == Pfx::P66 && i.mem.is_some()),
        0x113 | 0x117 => !wide && matches!(p, Pfx::None | Pfx::P66) && i.mem.is_some(),
        _ if fma_scalar(op) || fma_packed(op) => i.vex.is_some() && p == Pfx::P66,
        0x217 => p == Pfx::P66,
        0x20e | 0x20f => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x2f6 => i.vex.is_some() && p == Pfx::F2,
        0x321 => p == Pfx::P66 && !wide,
        0x304 | 0x305 => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x318 | 0x319 | 0x338 | 0x339 => wide && p == Pfx::P66 && !i.rex_w(),
        0x15b => true,
        0x34a..=0x34c => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x16f | 0x17f => matches!(p, Pfx::P66 | Pfx::F3),
        0x110 | 0x111 | 0x128 | 0x129 => matches!(p, Pfx::None | Pfx::P66),
        0x154..=0x157 => matches!(p, Pfx::None | Pfx::P66),
        0x170 => p != Pfx::None,
        0x171..=0x173 => {
            p == Pfx::P66
                && i.mem.is_none()
                && matches!((op, i.ext()), (0x173, 2 | 3 | 6 | 7) | (_, 2 | 4 | 6))
        }
        0x177 => i.vex.is_some() && p == Pfx::None,
        0x16e => p == Pfx::P66 && !wide,
        0x17e => (p == Pfx::P66 || p == Pfx::F3) && !wide,
        0x1d6 => p == Pfx::P66 && !wide,
        0x1d7 => p == Pfx::P66 && i.mem.is_none(),
        0x308..=0x30d => p == Pfx::P66,
        0x1c2 | 0x151 => true,
        0x2f3 => i.vex.is_some() && p == Pfx::None && (1..=3).contains(&i.ext()),
        _ if packed_int(op) => p == Pfx::P66 && (op != 0x200 || i.map == Map::M0F38),
        0x258 | 0x259 | 0x278 | 0x279 | 0x218 => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x219 => i.vex.is_some_and(|v| v.l) && p == Pfx::P66 && !i.rex_w(),
        0x245 | 0x247 => i.vex.is_some() && p == Pfx::P66,
        0x246 => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x2f2 => i.vex.is_some() && p == Pfx::None,
        0x2f7 => i.vex.is_some(),
        0x2f5 => i.vex.is_some() && p != Pfx::P66,
        0x22a => p == Pfx::P66 && i.mem.is_some(),
        0x1e6 => p != Pfx::None,
        0x1c6 => matches!(p, Pfx::None | Pfx::P66),
        0x300 | 0x301 => wide && p == Pfx::P66 && i.rex_w(),
        0x302 => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x320 | 0x322 => p == Pfx::P66 && !wide,
        0x220..=0x225 | 0x230..=0x235 => p == Pfx::P66,
        0x314..=0x317 => p == Pfx::P66 && !wide,
        0x30e => p == Pfx::P66,
        0x1c4 => p == Pfx::P66 && !wide,
        0x21a | 0x25a => wide && p == Pfx::P66 && !i.rex_w() && i.mem.is_some(),
        0x306 | 0x346 => wide && p == Pfx::P66,
        0x1c5 => p == Pfx::P66 && i.mem.is_none() && !wide,
        0x114 | 0x115 => matches!(p, Pfx::None | Pfx::P66),
        0x3f0 => i.vex.is_some() && p == Pfx::F2,
        0x17c | 0x17d | 0x1d0 => matches!(p, Pfx::P66 | Pfx::F2),
        0x150 => matches!(p, Pfx::None | Pfx::P66) && i.mem.is_none(),
        0x152 | 0x153 => matches!(p, Pfx::None | Pfx::F3),
        0x1d1..=0x1d3 | 0x1e1 | 0x1e2 | 0x1f1..=0x1f3 => p == Pfx::P66,
        0x213 | 0x31d => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x216 | 0x236 => wide && p == Pfx::P66 && !i.rex_w(),
        0x360..=0x363 => p == Pfx::P66 && !wide,
        0x178 => {
            i.vex.is_none() && i.mem.is_none() && ((p == Pfx::P66 && i.ext() == 0) || p == Pfx::F2)
        }
        0x179 => i.vex.is_none() && i.mem.is_none() && matches!(p, Pfx::P66 | Pfx::F2),
        0x1ae => p == Pfx::None && i.mem.is_some() && matches!(i.ext(), 2 | 3),
        0x30f => p == Pfx::P66,
        0x20c => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x22c..=0x22f => i.vex.is_some() && p == Pfx::P66 && !i.rex_w() && i.mem.is_some(),
        0x28c | 0x28e => i.vex.is_some() && p == Pfx::P66 && i.mem.is_some(),
        _ => false,
    };
    if !ok || (i.vex.is_none() && wide) {
        return None;
    }
    Some(match op {
        0x12e | 0x12f => (false, writes),
        0x2f2 | 0x217 | 0x20e | 0x20f | 0x2f3 => (false, writes),
        0x2f7 | 0x2f5 if p == Pfx::None => (false, writes),
        0x360..=0x363 => (false, writes),
        _ => (false, none),
    })
}

const FCVTL: u32 = 0x0e61_7800;
/// FCVTL Vd.4S, Vn.4H (| Q for FCVTL2 Vd.4S, Vn.8H).
const FCVTL_4S: u32 = 0x0e21_7800;
/// SSHLL Vd.2D, Vn.2S, #0.
const SSHLL_2D: u32 = 0x0f20_a400;
/// SSHLL and USHLL by 0 of the lower half, without the element size bit.
const SSHLL: u32 = 0x0f00_a400;
const USHLL: u32 = 0x2f00_a400;
const FCVTN: u32 = 0x0e61_6800;
/// FCVTN Vd.4H, Vn.4S (| Q for FCVTN2 Vd.8H, Vn.4S).
const FCVTN_4H: u32 = 0x0e21_6800;
/// The Q bit: FCVTL2, FCVTN2, SSHLL2 and XTN2 work on the upper half.
const Q: u32 = 1 << 30;
/// XTN Vd.2S, Vn.2D.
const XTN_2S: u32 = 0x0ea1_2800;
/// FMLA Vd.4S, Vn.4S, Vm.4S (bit 23 makes it FMLS, bit 22 .2D).
const FMLA_V: u32 = 0x4e20_cc00;
/// FNEG Vd.4S, Vn.4S (bit 22 makes it .2D).
const FNEG_V: u32 = 0x6ea0_f800;

fn vregs(d: V, n: V) -> u32 {
    (n.0 as u32) << 5 | d.0 as u32
}

/// UMOV Xd, Vn.D[1].
fn umov_d1(d: Reg, n: V) -> u32 {
    0x4e18_3c00 | (n.0 as u32) << 5 | d.0 as u32
}

/// The `fp1` operation for ROUNDSS/ROUNDSD's immediate: FRINTI (FPCR's mode,
/// which follows MXCSR.RC) with bit 2, else FRINTN, FRINTM, FRINTP or FRINTZ.
fn frint_scalar(imm: u32) -> u32 {
    if imm & 4 != 0 {
        15
    } else {
        [8, 10, 9, 11][(imm & 3) as usize]
    }
}

/// The vector FRINT on four singles for ROUNDPS's immediate (bit 22 makes
/// it .2D), chosen as [`frint_scalar`] does.
fn frint_vector(imm: u32) -> u32 {
    if imm & 4 != 0 {
        0x6ea1_9800
    } else {
        [0x4e21_8800, 0x4e21_9800, 0x4ea1_8800, 0x4ea1_9800][(imm & 3) as usize]
    }
}

fn ymm(n: u8, half: u32) -> i32 {
    (n as u32 * 32 + half * 16) as i32
}

impl Translator {
    /// Translates `i`, which [`classify`] accepted.
    pub(super) fn vector(&mut self, i: &Insn, next: u64, flags: bool) {
        let op = key(i);
        let p = pfx(i);
        let vex = i.vex.is_some();
        let scalar = matches!(p, Pfx::F3 | Pfx::F2)
            && matches!(
                op,
                0x110 | 0x111 | 0x12a | 0x158 | 0x159 | 0x15a | 0x15c..=0x15f | 0x1c2 | 0x151
            )
            || fma_scalar(op)
            || matches!(op, 0x30a | 0x30b);
        let halves = if i.vex.is_some_and(|v| v.l) && !scalar {
            2
        } else {
            1
        };
        let wide = i.vex.is_some_and(|v| v.l);
        let src1 = i.vex.map_or(i.reg, |v| v.vvvv);
        if let Some(m) = i.mem {
            self.addr(&m, next);
        }
        match op {
            0x1ae => return self.mxcsr(i),
            0x22c..=0x22f | 0x28c | 0x28e => return self.masked_move(i, op, halves),
            0x30f => {
                let n = i.imm as u32 & 0xff;
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    let out = match n {
                        0 => XB,
                        1..=15 => {
                            self.a.ext(XR, XB, XA, n);
                            XR
                        }
                        16 => XA,
                        _ => {
                            self.a.vec3(true, true, 0, 0b00011, XT, XT, XT);
                            if n < 32 {
                                self.a.ext(XT, XA, XT, n - 16);
                            }
                            XT
                        }
                    };
                    self.a.str_q(out, CTX, ymm(i.reg, h));
                }
            }
            0x20c => {
                let offsets = V(6);
                let a = &mut self.a;
                a.mov_imm(Size::W, T, 3);
                a.dup_gen(2, XS, T);
                a.mov_imm(Size::W, T, 0x0404_0404);
                a.dup_gen(2, XU, T);
                a.mov_imm(Size::W, T, 0x0302_0100);
                a.dup_gen(2, offsets, T);
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    let a = &mut self.a;
                    a.vec3(true, false, 0, 0b00011, XB, XB, XS);
                    a.vec3(true, false, 2, 0b10011, XB, XB, XU);
                    a.vec3(true, false, 2, 0b10000, XB, XB, offsets);
                    a.tbl(XR, XA, XB);
                    a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x2f2 | 0x2f6 | 0x2f7 | 0x3f0 => return self.bmi(i, op, p, flags),
            0x2f5 => return self.bmi2_bits(i, p, flags),
            0x2f3 => return self.bmi1_lowest(i, flags),
            0x213 => {
                self.convert_source(i, wide, XB);
                self.a.emit(FCVTL_4S | vregs(XR, XB));
                self.a.str_q(XR, CTX, ymm(i.reg, 0));
                if wide {
                    self.a.emit(FCVTL_4S | Q | vregs(XR, XB));
                    self.a.str_q(XR, CTX, ymm(i.reg, 1));
                    return;
                }
            }
            0x31d => return self.vcvtps2ph(i, wide),
            0x216 | 0x236 => return self.vperm_d(i, src1),
            0x360..=0x363 => return self.string_compare(i, op, vex, flags),
            0x178 | 0x179 => return self.extrq_insertq(i, op, p),
            0x151 if scalar => {
                let double = p == Pfx::F2;
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.scalar_src(i, double, XB);
                self.a.fp1(double, 3, XR, XB);
                self.a.ins(if double { 3 } else { 2 }, XA, 0, XR, 0);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x151 => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.a.fsqrt_v(p == Pfx::P66, XR, XB);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x30a | 0x30b => {
                let double = op == 0x30b;
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.scalar_src(i, double, XB);
                self.a.fp1(double, frint_scalar(i.imm as u32), XR, XB);
                self.a.ins(if double { 3 } else { 2 }, XA, 0, XR, 0);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x308 | 0x309 => {
                let word = frint_vector(i.imm as u32) | ((op == 0x309) as u32) << 22;
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.a.emit(word | vregs(XR, XB));
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x30c | 0x30d => {
                let (lanes, size) = if op == 0x30c { (4, 2) } else { (2, 3) };
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    for k in 0..lanes {
                        if i.imm as u32 >> (lanes * h + k) & 1 != 0 {
                            self.a.ins(size, XA, k, XB, k);
                        }
                    }
                    self.a.str_q(XA, CTX, ymm(i.reg, h));
                }
            }
            0x1c2 if scalar => {
                let double = p == Pfx::F2;
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.scalar_src(i, double, XB);
                self.compare(i, double, false);
                self.a.ins(if double { 3 } else { 2 }, XA, 0, XR, 0);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x1c2 => {
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    self.compare(i, p == Pfx::P66, true);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x112 if p == Pfx::F2 => {
                for h in 0..halves {
                    match i.mem {
                        Some(_) if halves == 1 => self.a.ldr_d(XB, ADDR, 0),
                        _ => self.rm_half(i, h, XB),
                    }
                    self.a.dup_elem(3, XR, XB);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x1e6 if p == Pfx::F3 => {
                self.convert_source(i, wide, XB);
                self.a.emit(SSHLL_2D | vregs(XR, XB));
                self.a.scvtf_v(true, XR, XR);
                self.a.str_q(XR, CTX, ymm(i.reg, 0));
                if wide {
                    self.a.emit(SSHLL_2D | Q | vregs(XR, XB));
                    self.a.scvtf_v(true, XR, XR);
                    self.a.str_q(XR, CTX, ymm(i.reg, 1));
                    return;
                }
            }
            0x1e6 => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.a.frint32_v(p == Pfx::F2, true, XT, XB);
                    self.a.fcvtzs_v(true, XT, XT);
                    self.a.emit(XTN_2S | (h * Q) | vregs(XR, XT));
                }
                self.a.str_q(XR, CTX, ymm(i.reg, 0));
                if wide {
                    self.zero_upper(i.reg);
                    return;
                }
            }
            0x1c6 => {
                let imm = i.imm as u32 & 0xff;
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    if p == Pfx::None {
                        for k in 0..4 {
                            let from = if k < 2 { XA } else { XB };
                            self.a.ins(2, XR, k, from, (imm >> (2 * k)) & 3);
                        }
                    } else {
                        self.a.ins(3, XR, 0, XA, (imm >> (2 * h)) & 1);
                        self.a.ins(3, XR, 1, XB, (imm >> (2 * h + 1)) & 1);
                    }
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x302 => {
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    for k in 0..4 {
                        if i.imm as u32 >> (4 * h + k) & 1 != 0 {
                            self.a.ins(2, XA, k, XB, k);
                        }
                    }
                    self.a.str_q(XA, CTX, ymm(i.reg, h));
                }
            }
            0x320 | 0x322 => {
                let (w, bytes, lane) = match (op, i.rex_w()) {
                    (0x320, _) => (Width::B, 1, i.imm as i32 & 15),
                    (_, true) => (Width::X, 8, i.imm as i32 & 1),
                    _ => (Width::W, 4, i.imm as i32 & 3),
                };
                match i.mem {
                    Some(_) => self.a.ldr(w, A, ADDR, 0),
                    None => self.a.mov(Size::X, A, r(i.rm)),
                }
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
                self.a.str(w, A, CTX, ymm(i.reg, 0) + bytes * lane);
            }
            0x220..=0x225 | 0x230..=0x235 => {
                let (from, to) = match op & 0xf {
                    0 => (0, 1),
                    1 => (0, 2),
                    2 => (0, 3),
                    3 => (1, 2),
                    4 => (1, 3),
                    _ => (2, 3),
                };
                let steps = to - from;
                let part = 8 >> (steps - 1);
                let base = if op >= 0x230 { USHLL } else { SSHLL };
                if i.mem.is_none() {
                    self.a.ldr_q(XS, CTX, ymm(i.rm, 0));
                }
                for h in 0..halves {
                    match i.mem {
                        Some(_) => {
                            let w = [Width::H, Width::W, Width::X][(3 - steps) as usize];
                            self.a.ldr(w, A, ADDR, h as i32 * part);
                            self.a.fmov_dx(XB, A);
                        }
                        None => self.a.ext(XB, XS, XS, h * part as u32),
                    }
                    for size in from..to {
                        self.a.emit(base | 1 << (19 + size) | vregs(XB, XB));
                    }
                    self.a.str_q(XB, CTX, ymm(i.reg, h));
                }
            }
            0x314..=0x317 => {
                let (w, bytes, lane) = match op {
                    0x314 => (Width::B, 1, i.imm as i32 & 15),
                    0x315 => (Width::H, 2, i.imm as i32 & 7),
                    0x316 if i.rex_w() => (Width::X, 8, i.imm as i32 & 1),
                    _ => (Width::W, 4, i.imm as i32 & 3),
                };
                self.a.ldr(w, A, CTX, ymm(i.reg, 0) + bytes * lane);
                match i.mem {
                    Some(_) => self.a.str(w, A, ADDR, 0),
                    None if w == Width::X => self.a.mov(Size::X, r(i.rm), A),
                    None => self.a.mov(Size::W, r(i.rm), A),
                }
                return;
            }
            0x1c4 => {
                match i.mem {
                    Some(_) => self.a.ldr(Width::H, A, ADDR, 0),
                    None => self.a.mov(Size::X, A, r(i.rm)),
                }
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
                self.a
                    .str(Width::H, A, CTX, ymm(i.reg, 0) + 2 * (i.imm as i32 & 7));
            }
            0x30e => {
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    for k in 0..8 {
                        if i.imm as u32 >> k & 1 != 0 {
                            self.a.ins(1, XA, k, XB, k);
                        }
                    }
                    self.a.str_q(XA, CTX, ymm(i.reg, h));
                }
            }
            0x21a | 0x25a => {
                self.a.ldr_q(XA, ADDR, 0);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
                self.a.str_q(XA, CTX, ymm(i.reg, 1));
                return;
            }
            0x306 | 0x346 => {
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.a.ldr_q(XB, CTX, ymm(src1, 1));
                self.rm_half(i, 0, XT);
                self.rm_half(i, 1, XS);
                for h in 0..2 {
                    let sel = (i.imm as u32 >> (4 * h)) & 0xf;
                    let out = if h == 0 { XR } else { XU };
                    if sel & 8 != 0 {
                        self.a.vec3(true, true, 0, 0b00011, out, out, out);
                    } else {
                        let from = [XA, XB, XT, XS][(sel & 3) as usize];
                        self.a.vec3(true, false, 2, 0b00011, out, from, from);
                    }
                }
                self.a.str_q(XR, CTX, ymm(i.reg, 0));
                self.a.str_q(XU, CTX, ymm(i.reg, 1));
                return;
            }
            0x1c5 => {
                self.a.ldr_q(XB, CTX, ymm(i.rm, 0));
                self.a.umov_h(A, XB, i.imm as u32 & 7);
                self.a.mov(Size::X, r(i.reg), A);
                return;
            }
            0x114 | 0x115 => {
                let size = if p == Pfx::P66 { 3 } else { 2 };
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    self.a.zip(true, size, op == 0x115, XR, XA, XB);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x300 | 0x301 => {
                self.rm_half(i, 0, XA);
                self.rm_half(i, 1, XB);
                for (h, out) in [(0, XR), (1, XT)] {
                    for k in 0..2 {
                        let n = (i.imm as u32 >> (2 * (2 * h + k))) & 3;
                        self.a.ins(3, out, k, if n < 2 { XA } else { XB }, n & 1);
                    }
                }
                self.a.str_q(XR, CTX, ymm(i.reg, 0));
                self.a.str_q(XT, CTX, ymm(i.reg, 1));
                return;
            }
            0x15a if matches!(p, Pfx::F3 | Pfx::F2) => {
                let double = p == Pfx::F2;
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.scalar_src(i, double, XB);
                self.a.fp1(double, if double { 4 } else { 5 }, XR, XB);
                self.a.ins(if double { 2 } else { 3 }, XA, 0, XR, 0);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x15a if p == Pfx::None => {
                self.convert_source(i, wide, XB);
                self.a.emit(FCVTL | vregs(XR, XB));
                self.a.str_q(XR, CTX, ymm(i.reg, 0));
                if wide {
                    self.a.emit(FCVTL | Q | vregs(XR, XB));
                    self.a.str_q(XR, CTX, ymm(i.reg, 1));
                    return;
                }
            }
            0x15a => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.a.emit(FCVTN | (h * Q) | vregs(XR, XB));
                }
                self.a.str_q(XR, CTX, ymm(i.reg, 0));
                if wide {
                    self.zero_upper(i.reg);
                    return;
                }
            }
            0x12c | 0x12d => return self.convert_to_int(i, op, p),
            _ if fma_packed(op) => {
                let double = i.rex_w();
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(i.reg, h));
                    self.a.ldr_q(XB, CTX, ymm(src1, h));
                    self.rm_half(i, h, XT);
                    let (n, m, acc) = match op >> 4 {
                        0x29 => (XA, XT, XB),
                        0x2a => (XB, XA, XT),
                        _ => (XB, XT, XA),
                    };
                    self.a.vec3(true, false, 2, 0b00011, XR, acc, acc);
                    if matches!(op & 0xf, 0xa | 0xe) {
                        self.a.emit(FNEG_V | (double as u32) << 22 | vregs(XR, XR));
                    }
                    let fmls = matches!(op & 0xf, 0xc | 0xe) as u32;
                    self.a.emit(
                        FMLA_V
                            | fmls << 23
                            | (double as u32) << 22
                            | (m.0 as u32) << 16
                            | vregs(XR, n),
                    );
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            _ if fma_scalar(op) => {
                let double = i.rex_w();
                self.a.ldr_q(XA, CTX, ymm(i.reg, 0));
                self.a.ldr_q(XB, CTX, ymm(src1, 0));
                self.scalar_src(i, double, XT);
                let (n, m, acc) = match op >> 4 {
                    0x29 => (XA, XT, XB),
                    0x2a => (XB, XA, XT),
                    _ => (XB, XT, XA),
                };
                let (o1, o0) = match op & 0xf {
                    0x9 => (0, 0),
                    0xb => (1, 1),
                    0xd => (0, 1),
                    _ => (1, 0),
                };
                self.a.emit(
                    0x1f00_0000
                        | (double as u32) << 22
                        | o1 << 21
                        | (m.0 as u32) << 16
                        | o0 << 15
                        | (acc.0 as u32) << 10
                        | vregs(XR, n),
                );
                self.a.ins(if double { 3 } else { 2 }, XA, 0, XR, 0);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x217 | 0x20e | 0x20f => {
                if flags {
                    let sign = match op {
                        0x20e => 32,
                        0x20f => 64,
                        _ => 0,
                    };
                    self.ptest(i, halves, sign);
                }
                return;
            }
            0x321 => {
                let imm = i.imm as u32 & 0xff;
                let to = (imm >> 4) & 3;
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                match i.mem {
                    Some(_) => {
                        self.a.ldr_s(XB, ADDR, 0);
                        self.a.ins(2, XA, to, XB, 0);
                    }
                    None => {
                        self.a.ldr_q(XB, CTX, ymm(i.rm, 0));
                        self.a.ins(2, XA, to, XB, imm >> 6);
                    }
                }
                if imm & 0xf != 0 {
                    self.a.vec3(true, true, 0, 0b00011, XT, XT, XT);
                    for k in (0..4).filter(|k| imm >> k & 1 != 0) {
                        self.a.ins(2, XA, k, XT, 0);
                    }
                }
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x305 => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    for k in 0..2 {
                        self.a.ins(3, XR, k, XB, (i.imm as u32 >> (2 * h + k)) & 1);
                    }
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x319 | 0x339 => {
                self.a.ldr_q(XA, CTX, ymm(i.reg, i.imm as u32 & 1));
                match i.mem {
                    Some(_) => self.a.str_q(XA, ADDR, 0),
                    None => {
                        self.a.str_q(XA, CTX, ymm(i.rm, 0));
                        self.zero_upper(i.rm);
                    }
                }
                return;
            }
            0x318 | 0x338 => {
                let h = i.imm as u32 & 1;
                self.a.ldr_q(XA, CTX, ymm(src1, 1 - h));
                self.rm_half(i, 0, XB);
                self.a.str_q(XA, CTX, ymm(i.reg, 1 - h));
                self.a.str_q(XB, CTX, ymm(i.reg, h));
                return;
            }
            0x112 | 0x116 if p == Pfx::F3 => {
                let odd = (op == 0x116) as u32;
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    for k in 0..4 {
                        self.a.ins(2, XR, k, XB, (k & 2) | odd);
                    }
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x112 | 0x116 => {
                let lane = (op == 0x116) as u32;
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                match i.mem {
                    Some(_) => {
                        self.a.ldr_d(XB, ADDR, 0);
                        self.a.ins(3, XA, lane, XB, 0);
                    }
                    None => {
                        self.a.ldr_q(XB, CTX, ymm(i.rm, 0));
                        self.a.ins(3, XA, lane, XB, 1 - lane);
                    }
                }
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x113 | 0x117 => {
                let at = ymm(i.reg, 0) + 8 * (op == 0x117) as i32;
                self.a.ldr(Width::X, A, CTX, at);
                self.a.str(Width::X, A, ADDR, 0);
                return;
            }
            0x158 | 0x159 | 0x15c..=0x15f if matches!(p, Pfx::F3 | Pfx::F2) => {
                let double = p == Pfx::F2;
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.scalar_src(i, double, XB);
                let fp2 = match op {
                    0x159 => 0,
                    0x15e => 1,
                    0x158 => 2,
                    0x15c => 3,
                    0x15f => 4,
                    _ => 5,
                };
                self.a.fp2(double, fp2, XA, XA, XB);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x158 | 0x159 | 0x15c..=0x15f => {
                let d = (p == Pfx::P66) as u32;
                let (u, size, opcode) = match op {
                    0x158 => (false, d, 0b11010),
                    0x15c => (false, 2 | d, 0b11010),
                    0x159 => (true, d, 0b11011),
                    0x15e => (true, d, 0b11111),
                    0x15f => (false, d, 0b11110),
                    _ => (false, 2 | d, 0b11110),
                };
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    self.a.vec3(true, u, size, opcode, XR, XA, XB);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x110 | 0x111 if matches!(p, Pfx::F3 | Pfx::F2) => {
                let double = p == Pfx::F2;
                let lane = if double { 3 } else { 2 };
                match (op, i.mem.is_some()) {
                    (0x110, true) => {
                        self.scalar_src(i, double, XB);
                        self.a.str_q(XB, CTX, ymm(i.reg, 0));
                    }
                    (0x111, true) => {
                        let w = if double { Width::X } else { Width::W };
                        self.a.ldr(w, A, CTX, ymm(i.reg, 0));
                        self.a.str(w, A, ADDR, 0);
                        return;
                    }
                    _ => {
                        let (dst, low) = if op == 0x110 {
                            (i.reg, i.rm)
                        } else {
                            (i.rm, i.reg)
                        };
                        let upper = if vex { src1 } else { dst };
                        self.a.ldr_q(XA, CTX, ymm(upper, 0));
                        self.a.ldr_q(XB, CTX, ymm(low, 0));
                        self.a.ins(lane, XA, 0, XB, 0);
                        self.a.str_q(XA, CTX, ymm(dst, 0));
                        if vex {
                            self.zero_upper(dst);
                        }
                        return;
                    }
                }
            }
            0x12e | 0x12f => {
                let double = p == Pfx::P66;
                self.a.ldr_q(XA, CTX, ymm(i.reg, 0));
                self.scalar_src(i, double, XB);
                self.a.fcmp(double, false, XA, XB);
                self.compare_flags();
                return;
            }
            0x12a => {
                let double = p == Pfx::F2;
                let size = if i.rex_w() { Size::X } else { Size::W };
                match i.mem {
                    Some(_) => self
                        .a
                        .ldr(if i.rex_w() { Width::X } else { Width::W }, A, ADDR, 0),
                    None => self.a.mov(size, A, r(i.rm)),
                }
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.a.scvtf(double, size, XA, A);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x15b => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    match p {
                        Pfx::None => self.a.scvtf_v(false, XR, XB),
                        _ => {
                            self.a.frint32_v(p == Pfx::P66, false, XR, XB);
                            self.a.fcvtzs_v(false, XR, XR);
                        }
                    }
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x34a..=0x34c => {
                let esize = match op {
                    0x34a => 32,
                    0x34b => 64,
                    _ => 8,
                };
                let mask = (i.imm as u8 >> 4) & 0xf;
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    self.a.ldr_q(XT, CTX, ymm(mask, h));
                    self.a.shr_v(true, true, esize, XR, XT, esize - 1);
                    self.a.vec3(true, true, 1, 0b00011, XR, XB, XA);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x258 | 0x259 | 0x278 | 0x279 | 0x218 | 0x219 => {
                let size = match op {
                    0x278 => 0,
                    0x279 => 1,
                    0x258 | 0x218 => 2,
                    _ => 3,
                };
                match i.mem {
                    Some(_) => {
                        let w = [Width::B, Width::H, Width::W, Width::X][size as usize];
                        self.a.ldr(w, A, ADDR, 0);
                        self.a.dup_gen(size, XR, A);
                    }
                    None => {
                        self.a.ldr_q(XB, CTX, ymm(i.rm, 0));
                        self.a.dup_elem(size, XR, XB);
                    }
                }
                for h in 0..halves {
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x245..=0x247 => {
                let size = if i.rex_w() { 3 } else { 2 };
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    self.variable_shift(op, size);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x177 => {
                let all = i.vex.is_some_and(|v| v.l);
                for n in 0..16 {
                    if all {
                        self.a.stp(Size::X, Reg(31), Reg(31), CTX, ymm(n, 0));
                    }
                    self.a.stp(Size::X, Reg(31), Reg(31), CTX, ymm(n, 1));
                }
                return;
            }
            0x16f | 0x128 | 0x110 | 0x22a => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.a.str_q(XB, CTX, ymm(i.reg, h));
                }
            }
            0x17f | 0x129 | 0x111 | 0x1e7 | 0x12b => {
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(i.reg, h));
                    match i.mem {
                        Some(_) => self.a.str_q(XA, ADDR, 16 * h as i32),
                        None => self.a.str_q(XA, CTX, ymm(i.rm, h)),
                    }
                }
                if i.mem.is_none() && vex && halves == 1 {
                    self.zero_upper(i.rm);
                }
                return;
            }
            0x154..=0x157 => {
                let logic = [0x1db, 0x1df, 0x1eb, 0x1ef][(op - 0x154) as usize];
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    self.packed(logic);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x170 if p != Pfx::P66 => {
                let high = (p == Pfx::F3) as u32 * 4;
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.a.vec3(true, false, 2, 0b00011, XR, XB, XB);
                    for k in 0..4 {
                        let from = high + ((i.imm as u32 >> (2 * k)) & 3);
                        self.a.ins(1, XR, high + k, XB, from);
                    }
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x170 | 0x304 => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    for k in 0..4 {
                        self.a.ins(2, XR, k, XB, (i.imm as u32 >> (2 * k)) & 3);
                    }
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x171..=0x173 => {
                let dst = if vex { src1 } else { i.rm };
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(i.rm, h));
                    self.shift_imm(op, i.ext(), i.imm as u32 & 0xff);
                    self.a.str_q(XA, CTX, ymm(dst, h));
                }
                if vex && halves == 1 {
                    self.zero_upper(dst);
                }
                return;
            }
            0x16e => {
                let w = if i.rex_w() { Width::X } else { Width::W };
                match i.mem {
                    Some(_) => self.a.ldr(w, A, ADDR, 0),
                    None if w == Width::X => self.a.mov(Size::X, A, r(i.rm)),
                    None => self.a.mov(Size::W, A, r(i.rm)),
                }
                self.a.stp(Size::X, A, Reg(31), CTX, ymm(i.reg, 0));
                if !vex {
                    return;
                }
            }
            0x17e if p == Pfx::P66 => {
                let w = if i.rex_w() { Width::X } else { Width::W };
                self.a.ldr(w, A, CTX, ymm(i.reg, 0));
                match i.mem {
                    Some(_) => self.a.str(w, A, ADDR, 0),
                    None => self
                        .a
                        .mov(if w == Width::X { Size::X } else { Size::W }, r(i.rm), A),
                }
                return;
            }
            0x17e => {
                match i.mem {
                    Some(_) => self.a.ldr(Width::X, A, ADDR, 0),
                    None => self.a.ldr(Width::X, A, CTX, ymm(i.rm, 0)),
                }
                self.a.stp(Size::X, A, Reg(31), CTX, ymm(i.reg, 0));
            }
            0x1d6 => {
                self.a.ldr(Width::X, A, CTX, ymm(i.reg, 0));
                match i.mem {
                    Some(_) => {
                        self.a.str(Width::X, A, ADDR, 0);
                        return;
                    }
                    None => self.a.stp(Size::X, A, Reg(31), CTX, ymm(i.rm, 0)),
                }
                if vex {
                    self.zero_upper(i.rm);
                }
                return;
            }
            0x17c | 0x17d | 0x1d0 => {
                let d = (p == Pfx::P66) as u32;
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    let a = &mut self.a;
                    match op {
                        0x17c => a.vec3(true, true, d, 0b11010, XR, XA, XB),
                        0x17d => {
                            a.uzp(true, 2 + d, false, XT, XA, XB);
                            a.uzp(true, 2 + d, true, XS, XA, XB);
                            a.vec3(true, false, 2 | d, 0b11010, XR, XT, XS);
                        }
                        _ => {
                            a.vec3(true, false, 2 | d, 0b11010, XR, XA, XB);
                            a.vec3(true, false, d, 0b11010, XT, XA, XB);
                            if d == 1 {
                                a.ins(3, XR, 1, XT, 1);
                            } else {
                                a.ins(2, XR, 1, XT, 1);
                                a.ins(2, XR, 3, XT, 3);
                            }
                        }
                    }
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x150 => {
                let double = p == Pfx::P66;
                let lanes = if double { 2 } else { 4 };
                for h in 0..halves {
                    let a = &mut self.a;
                    a.ldr_q(XB, CTX, ymm(i.rm, h));
                    if double {
                        a.shr_v(true, false, 64, XR, XB, 63);
                    } else {
                        a.shr_v(true, false, 32, XR, XB, 31);
                        a.usra(true, 64, XR, XR, 31);
                    }
                    a.umov_b(B, XR, 0);
                    a.umov_b(T, XR, 8);
                    a.logic_shift(
                        Logic::Orr,
                        false,
                        Size::W,
                        B,
                        B,
                        T,
                        a64::Shift::Lsl,
                        lanes / 2,
                    );
                    if h == 0 {
                        a.mov(Size::W, A, B);
                    } else {
                        a.logic_shift(Logic::Orr, false, Size::W, A, A, B, a64::Shift::Lsl, lanes);
                    }
                }
                self.a.mov(Size::X, r(i.reg), A);
                return;
            }
            0x152 | 0x153 if p == Pfx::F3 => {
                self.a.ldr_q(XA, CTX, ymm(src1, 0));
                self.scalar_src(i, false, XB);
                self.reciprocal(op == 0x152);
                self.a.ins(2, XA, 0, XR, 0);
                self.a.str_q(XA, CTX, ymm(i.reg, 0));
            }
            0x152 | 0x153 => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.reciprocal(op == 0x152);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x1d1..=0x1d3 | 0x1e1 | 0x1e2 | 0x1f1..=0x1f3 => {
                let size = (op & 3) as u32;
                let right = op < 0x1f0;
                match i.mem {
                    Some(_) => self.a.ldr(Width::X, B, ADDR, 0),
                    None => self.a.ldr(Width::X, B, CTX, ymm(i.rm, 0)),
                }
                let a = &mut self.a;
                a.lsr_imm(Size::X, T, B, 3 + size);
                a.clz(Size::X, T, T);
                a.lsr_imm(Size::X, T, T, 6);
                a.sub_imm(Size::X, T, T, 1);
                a.bic(Size::X, B, B, T);
                assert!(a.logic_imm(Logic::And, Size::X, T, T, 8 << size));
                a.orr(Size::X, B, B, T);
                if right {
                    a.neg(Size::X, B, B);
                }
                a.dup_gen(size, XB, B);
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.a
                        .vec3(true, op & 0xf0 != 0xe0, size, 0b01000, XR, XA, XB);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
            0x1d7 => {
                for h in 0..halves {
                    let a = &mut self.a;
                    a.ldr_q(XB, CTX, ymm(i.rm, h));
                    a.shr_v(true, false, 8, XR, XB, 7);
                    a.usra(true, 16, XR, XR, 7);
                    a.usra(true, 32, XR, XR, 14);
                    a.usra(true, 64, XR, XR, 28);
                    a.umov_b(B, XR, 0);
                    a.umov_b(T, XR, 8);
                    a.logic_shift(Logic::Orr, false, Size::W, B, B, T, a64::Shift::Lsl, 8);
                    if h == 0 {
                        a.mov(Size::W, A, B);
                    } else {
                        a.logic_shift(Logic::Orr, false, Size::W, A, A, B, a64::Shift::Lsl, 16);
                    }
                }
                self.a.mov(Size::X, r(i.reg), A);
                return;
            }
            _ => {
                for h in 0..halves {
                    self.a.ldr_q(XA, CTX, ymm(src1, h));
                    self.rm_half(i, h, XB);
                    self.packed(op);
                    self.a.str_q(XR, CTX, ymm(i.reg, h));
                }
            }
        }
        if vex && halves == 1 {
            self.zero_upper(i.reg);
        }
    }

    /// CVTTSS2SI, CVTTSD2SI, CVTSS2SI and CVTSD2SI. NaN and results out of
    /// range give x86's "integer indefinite" (only the sign bit set).
    fn convert_to_int(&mut self, i: &Insn, op: u16, p: Pfx) {
        let double = p == Pfx::F2;
        let wide = i.rex_w();
        self.scalar_src(i, double, XB);
        let (old, a) = (super::OLD_NZCV, &mut self.a);
        if op == 0x12d {
            a.fp1(double, 14, XB, XB);
        }
        a.mrs(old, a64::sysreg::NZCV);
        a.fcvtzs(double, Size::X, A, XB);
        a.mov_imm(Size::X, super::U, if wide { 1 << 63 } else { 1 << 31 });
        if wide {
            a.mov_imm(Size::X, B, i64::MAX as u64);
            a.cmp(Size::X, A, B);
            a.csel(Size::X, A, super::U, A, a64::Cond::Eq);
        } else {
            a.sxtw(B, A);
            a.cmp(Size::X, A, B);
            a.csel(Size::X, A, super::U, A, a64::Cond::Ne);
        }
        a.fcmp(double, false, XB, XB);
        a.csel(Size::X, A, super::U, A, a64::Cond::Vs);
        a.msr(a64::sysreg::NZCV, old);
        a.mov(if wide { Size::X } else { Size::W }, r(i.reg), A);
    }

    /// PTEST: ZF = (dst AND src) is zero, CF = (src AND NOT dst) is zero,
    /// the other arithmetic flags clear. VTESTPS/PD (`sign` 32 or 64) look
    /// only at each element's sign bit.
    fn ptest(&mut self, i: &Insn, halves: u32, sign: u32) {
        for h in 0..halves {
            let (and, andn) = if h == 0 { (XR, XU) } else { (XT, XS) };
            self.a.ldr_q(XA, CTX, ymm(i.reg, h));
            self.rm_half(i, h, XB);
            self.a.vec3(true, false, 0, 0b00011, and, XA, XB);
            self.a.vec3(true, false, 1, 0b00011, andn, XB, XA);
            if h == 1 {
                self.a.vec3(true, false, 2, 0b00011, XR, XR, XT);
                self.a.vec3(true, false, 2, 0b00011, XU, XU, XS);
            }
        }
        if sign != 0 {
            self.a.shr_v(true, false, sign, XR, XR, sign - 1);
            self.a.shr_v(true, false, sign, XU, XU, sign - 1);
        }
        let (u, a) = (super::U, &mut self.a);
        a.fmov_xd(A, XR);
        a.emit(umov_d1(B, XR));
        a.orr(Size::X, A, A, B);
        a.fmov_xd(u, XU);
        a.emit(umov_d1(B, XU));
        a.orr(Size::X, u, u, B);
        a.cmp_imm(Size::X, A, 0);
        a.cset(Size::X, T, a64::Cond::Eq);
        a.cmp_imm(Size::X, u, 0);
        a.cset(Size::X, B, a64::Cond::Ne);
        a.lsl_imm(Size::X, T, T, 30);
        a.logic_shift(Logic::Orr, false, Size::X, T, T, B, a64::Shift::Lsl, 29);
        a.msr(a64::sysreg::NZCV, T);
        a.mov_imm(Size::X, super::PF_RESULT, 1);
        a.mov_imm(Size::X, super::AF_AUX, 1);
    }

    /// CMPPS/CMPPD/CMPSS/CMPSD: XR = all ones where XA `predicate` XB holds.
    fn compare(&mut self, i: &Insn, double: bool, vector: bool) {
        let pred = i.imm as u8 & if i.vex.is_some() { 0x1f } else { 7 };
        let a = &mut self.a;
        let fcm = |a: &mut a64::Asm, kind, rd, rn, rm| {
            if vector {
                a.fcm_v(kind, double, rd, rn, rm);
            } else {
                a.fcm(kind, double, rd, rn, rm);
            }
        };
        let base = pred & 7;
        match base {
            0 | 4 => fcm(a, Fcm::Eq, XR, XA, XB),
            1 | 5 => fcm(a, Fcm::Gt, XR, XB, XA),
            2 | 6 => fcm(a, Fcm::Ge, XR, XB, XA),
            _ => {}
        }
        if matches!(base, 3 | 7) || pred & 8 != 0 {
            fcm(a, Fcm::Eq, XT, XA, XA);
            fcm(a, Fcm::Eq, XS, XB, XB);
            a.vec3(true, false, 0, 0b00011, XT, XT, XS);
        }
        if matches!(base, 3 | 7) {
            a.vec3(true, false, 2, 0b00011, XR, XT, XT);
        }
        if matches!(base, 3..=6) {
            a.not_v(XR, XR);
        }
        if pred & 8 != 0 {
            a.vec3(true, true, 0, 0b00011, XR, XR, XT);
            a.not_v(XR, XR);
        }
    }

    /// `dst` = half `h` of the r/m operand (memory at x19, or a register).
    fn rm_half(&mut self, i: &Insn, h: u32, dst: V) {
        match i.mem {
            Some(_) => self.a.ldr_q(dst, ADDR, 16 * h as i32),
            None => self.a.ldr_q(dst, CTX, ymm(i.rm, h)),
        }
    }

    /// `dst` = the 64-bit source of a widening conversion, or 128 bits for
    /// its 256-bit form.
    fn convert_source(&mut self, i: &Insn, wide: bool, dst: V) {
        match i.mem {
            Some(_) if !wide => self.a.ldr_d(dst, ADDR, 0),
            _ => self.rm_half(i, 0, dst),
        }
    }

    /// `dst` = the scalar r/m operand: 4 or 8 bytes from memory (the rest
    /// zero), or the whole register.
    fn scalar_src(&mut self, i: &Insn, double: bool, dst: V) {
        match i.mem {
            Some(_) if double => self.a.ldr_d(dst, ADDR, 0),
            Some(_) => self.a.ldr_s(dst, ADDR, 0),
            None => self.a.ldr_q(dst, CTX, ymm(i.rm, 0)),
        }
    }

    /// x86's flags after UCOMISS/COMISS from the FCMP just made: ZF, PF, CF
    /// as the comparison says (all three when unordered), OF, SF, AF clear.
    pub(super) fn compare_flags(&mut self) {
        let a = &mut self.a;
        a.cset(Size::X, T, a64::Cond::Ge);
        a.cset(Size::X, super::U, a64::Cond::Eq);
        a.cset(Size::X, B, a64::Cond::Vs);
        a.orr(Size::X, super::U, super::U, B);
        a.lsl_imm(Size::X, T, T, 29);
        a.logic_shift(
            Logic::Orr,
            false,
            Size::X,
            T,
            T,
            super::U,
            a64::Shift::Lsl,
            30,
        );
        a.msr(a64::sysreg::NZCV, T);
        assert!(a.logic_imm(Logic::Eor, Size::X, super::PF_RESULT, B, 1));
        a.mov(Size::X, super::AF_AUX, super::PF_RESULT);
    }

    /// XR = RCPPS (or RSQRTPS) of the four singles in XB, as the
    /// interpreter computes them: exact, NaNs quieted, denormals and zeros
    /// giving infinity of their sign, other negatives the default NaN.
    fn reciprocal(&mut self, sqrt: bool) {
        const CMEQ_ZERO: u32 = 0x4ea0_9800;
        const CMLT_ZERO: u32 = 0x4ea0_a800;
        let a = &mut self.a;
        let splat = |a: &mut a64::Asm, v: V, imm: u64| {
            a.mov_imm(Size::W, T, imm);
            a.dup_gen(2, v, T);
        };
        splat(a, XT, 0x3f80_0000);
        if sqrt {
            a.fsqrt_v(false, XS, XB);
            a.vec3(true, true, 0, 0b11111, XR, XT, XS);
            a.fcm_v(Fcm::Eq, false, XU, XB, XB);
            a.emit(CMLT_ZERO | vregs(XS, XB));
            a.vec3(true, false, 0, 0b00011, XU, XU, XS);
            splat(a, XT, 0xffc0_0000);
            a.vec3(true, true, 2, 0b00011, XR, XT, XU);
        } else {
            a.vec3(true, true, 0, 0b11111, XR, XT, XB);
        }
        splat(a, XS, 0x7f80_0000);
        a.vec3(true, false, 0, 0b00011, XU, XB, XS);
        a.emit(CMEQ_ZERO | vregs(XU, XU));
        splat(a, XT, 0x8000_0000);
        a.vec3(true, false, 0, 0b00011, XT, XB, XT);
        a.vec3(true, false, 2, 0b00011, XT, XT, XS);
        a.vec3(true, true, 2, 0b00011, XR, XT, XU);
    }

    fn zero_upper(&mut self, n: u8) {
        self.a.stp(Size::X, Reg(31), Reg(31), CTX, ymm(n, 1));
    }

    /// XR = XA `op` XB, one 128-bit half of a packed integer operation.
    fn packed(&mut self, op: u16) {
        let a = &mut self.a;
        match op {
            0x1db => a.vec3(true, false, 0, 0b00011, XR, XA, XB),
            0x1df => a.vec3(true, false, 1, 0b00011, XR, XB, XA),
            0x1eb => a.vec3(true, false, 2, 0b00011, XR, XA, XB),
            0x1ef => a.vec3(true, true, 0, 0b00011, XR, XA, XB),
            0x160..=0x162 => a.zip(true, (op - 0x160) as u32, false, XR, XA, XB),
            0x16c => a.zip(true, 3, false, XR, XA, XB),
            0x168..=0x16a => a.zip(true, (op - 0x168) as u32, true, XR, XA, XB),
            0x16d => a.zip(true, 3, true, XR, XA, XB),
            0x163 | 0x167 | 0x16b | 0x22b => {
                let (kind, size) = match op {
                    0x163 => (Narrow::Sqxtn, 0),
                    0x167 => (Narrow::Sqxtun, 0),
                    0x22b => (Narrow::Sqxtun, 1),
                    _ => (Narrow::Sqxtn, 1),
                };
                a.narrow(kind, size, XR, XA);
                a.narrow_hi(kind, size, XR, XB);
            }
            0x1e4 | 0x1e5 => {
                a.mull(op == 0x1e5, 1, XR, XA, XB);
                a.mull_hi(op == 0x1e5, 1, XT, XA, XB);
                a.uzp(true, 1, true, XR, XR, XT);
            }
            0x1f5 => {
                a.mull(true, 1, XR, XA, XB);
                a.mull_hi(true, 1, XT, XA, XB);
                a.vec3(true, false, 2, 0b10111, XR, XR, XT);
            }
            0x1f4 => {
                a.narrow(Narrow::Xtn, 2, XR, XA);
                a.narrow(Narrow::Xtn, 2, XT, XB);
                a.mull(false, 2, XR, XR, XT);
            }
            0x1f6 => {
                a.vec3(true, true, 0, 0b01110, XR, XA, XB);
                a.uaddlp(true, 0, XR, XR);
                a.uaddlp(true, 1, XR, XR);
                a.uaddlp(true, 2, XR, XR);
            }
            0x204 => {
                a.emit(USHLL | 1 << 19 | vregs(XT, XA));
                a.emit(USHLL | 1 << 19 | Q | vregs(XS, XA));
                a.emit(SSHLL | 1 << 19 | vregs(XR, XB));
                a.emit(SSHLL | 1 << 19 | Q | vregs(XU, XB));
                a.vec3(true, false, 1, 0b10011, XT, XT, XR);
                a.vec3(true, false, 1, 0b10011, XS, XS, XU);
                a.uzp(true, 1, false, XR, XT, XS);
                a.uzp(true, 1, true, XU, XT, XS);
                a.vec3(true, false, 1, 0b00001, XR, XR, XU);
            }
            0x200 => {
                a.movi_b(XT, 0x8f);
                a.vec3(true, false, 0, 0b00011, XT, XB, XT);
                a.tbl(XR, XA, XT);
            }
            _ => {
                let (u, size, opcode) = simd_three(op).expect("classified");
                a.vec3(true, u, size, opcode, XR, XA, XB);
            }
        }
    }

    /// VPSRLV, VPSRAV or VPSLLV: XR = XA shifted by the counts in XB. NEON uses
    /// only a count's low byte, so counts of the width or more are masked.
    fn variable_shift(&mut self, op: u16, size: u32) {
        let esize = 8u64 << size;
        let a = &mut self.a;
        a.mov_imm(Size::X, A, esize - 1);
        a.dup_gen(size, XT, A);
        if op == 0x246 {
            a.vec3(true, true, size, 0b01101, XS, XB, XT);
            a.neg_v(size, XS, XS);
            a.vec3(true, false, size, 0b01000, XR, XA, XS);
            return;
        }
        a.vec3(true, true, size, 0b00110, XT, XB, XT);
        if op == 0x245 {
            a.neg_v(size, XS, XB);
        } else {
            a.vec3(true, false, 2, 0b00011, XS, XB, XB);
        }
        a.vec3(true, true, size, 0b01000, XR, XA, XS);
        a.vec3(true, false, 1, 0b00011, XR, XR, XT);
    }

    /// XA = XA shifted by `n` as group 12/13/14 `ext` says, per 128-bit half.
    fn shift_imm(&mut self, op: u16, ext: u8, n: u32) {
        let esize = match op {
            0x171 => 16,
            0x172 => 32,
            _ => 64,
        };
        let a = &mut self.a;
        match (op, ext) {
            (0x173, 3 | 7) if n >= 16 => a.vec3(true, true, 0, 0b00011, XA, XA, XA),
            (0x173, 3 | 7) if n > 0 => {
                a.vec3(true, true, 0, 0b00011, XT, XT, XT);
                if ext == 3 {
                    a.ext(XA, XA, XT, n)
                } else {
                    a.ext(XA, XT, XA, 16 - n)
                }
            }
            (_, 2 | 6) if n >= esize => a.vec3(true, true, 0, 0b00011, XA, XA, XA),
            (_, 6) if n > 0 => a.shl_v(true, esize, XA, XA, n),
            (_, 2) if n > 0 => a.shr_v(true, false, esize, XA, XA, n),
            (_, 4) if n > 0 => a.shr_v(true, true, esize, XA, XA, n.min(esize - 1).max(1)),
            _ => {}
        }
    }

    /// BZHI, PDEP and PEXT. PDEP and PEXT walk the mask's set bits in a loop
    /// and leave NZCV alone.
    fn bmi2_bits(&mut self, i: &Insn, p: Pfx, flags: bool) {
        let size = if i.rex_w() { 8 } else { 4 };
        let s = if size == 8 { Size::X } else { Size::W };
        let v = r(i.vex.expect("classified").vvvv);
        let (u, low, tmp) = (super::U, super::OLD_NZCV, super::OLD_AF);
        match i.mem {
            Some(_) => self
                .a
                .ldr(if size == 8 { Width::X } else { Width::W }, A, ADDR, 0),
            None => self.a.mov(s, A, r(i.rm)),
        }
        let a = &mut self.a;
        if p == Pfx::None {
            a.ubfx(Size::X, B, v, 0, 8);
            a.mov_imm(Size::X, u, 1);
            a.lslv(Size::X, u, u, B);
            a.sub_imm(Size::X, u, u, 1);
            a.and(Size::X, T, A, u);
            a.cmp_imm(Size::X, B, size as u32 * 8);
            a.csel(s, T, A, T, a64::Cond::Hs);
            if flags {
                a.cset(Size::X, tmp, a64::Cond::Lo);
                a.tst(s, T, T);
                a.mrs(B, a64::sysreg::NZCV);
                a.bfi(Size::X, B, tmp, 29, 1);
                a.msr(a64::sysreg::NZCV, B);
                a.mov_imm(Size::X, super::PF_RESULT, 1);
                a.mov_imm(Size::X, super::AF_AUX, 1);
            }
            a.mov(s, r(i.reg), T);
            return;
        }
        let pext = p == Pfx::F3;
        let (top, skip, done) = (a.label(), a.label(), a.label());
        a.mov(s, B, v);
        a.mov_imm(Size::X, T, 0);
        a.mov_imm(Size::X, u, 1);
        a.bind(top);
        a.cbz(Size::X, A, done);
        a.neg(Size::X, low, A);
        a.and(Size::X, low, A, low);
        a.and(Size::X, tmp, B, if pext { low } else { u });
        a.cbz(Size::X, tmp, skip);
        a.orr(Size::X, T, T, if pext { u } else { low });
        a.bind(skip);
        a.add(Size::X, u, u, u);
        a.bic(Size::X, A, A, low);
        a.b(top);
        a.bind(done);
        a.mov(s, r(i.reg), T);
    }

    /// BLSR, BLSMSK and BLSI: ZF and SF from the result, CF from the
    /// source, the other arithmetic flags clear.
    fn bmi1_lowest(&mut self, i: &Insn, flags: bool) {
        let s = if i.rex_w() { Size::X } else { Size::W };
        let v = r(i.vex.expect("classified").vvvv);
        match i.mem {
            Some(_) => self
                .a
                .ldr(if i.rex_w() { Width::X } else { Width::W }, A, ADDR, 0),
            None => self.a.mov(s, A, r(i.rm)),
        }
        let a = &mut self.a;
        a.sub_imm(s, B, A, 1);
        match i.ext() {
            1 => a.and(s, T, A, B),
            2 => a.eor(s, T, A, B),
            _ => {
                a.neg(s, T, A);
                a.and(s, T, A, T);
            }
        }
        if flags {
            a.cmp_imm(s, A, 0);
            a.cset(
                Size::X,
                B,
                if i.ext() == 3 {
                    a64::Cond::Eq
                } else {
                    a64::Cond::Ne
                },
            );
            a.tst(s, T, T);
            a.mrs(super::U, a64::sysreg::NZCV);
            a.bfi(Size::X, super::U, B, 29, 1);
            a.msr(a64::sysreg::NZCV, super::U);
            a.mov_imm(Size::X, super::PF_RESULT, 1);
            a.mov_imm(Size::X, super::AF_AUX, 1);
        }
        a.mov(s, v, T);
    }

    /// ANDN, BEXTR, SHLX/SHRX/SARX, RORX and MULX on general registers.
    fn bmi(&mut self, i: &Insn, op: u16, p: Pfx, flags: bool) {
        let size = if i.rex_w() { 8 } else { 4 };
        let s = if size == 8 { Size::X } else { Size::W };
        let bits = size * 8;
        let v = i.vex.expect("classified").vvvv;
        match i.mem {
            Some(_) => self
                .a
                .ldr(if size == 8 { Width::X } else { Width::W }, A, ADDR, 0),
            None => self.a.mov(s, A, r(i.rm)),
        }
        let a = &mut self.a;
        if op == 0x2f6 {
            if size == 8 {
                a.mul(Size::X, T, r(2), A);
                a.umulh(super::U, r(2), A);
            } else {
                a.umull(T, r(2), A);
                a.lsr_imm(Size::X, super::U, T, 32);
            }
            a.mov(s, r(v), T);
            a.mov(s, r(i.reg), super::U);
            return;
        }
        match (op, p) {
            (0x2f2, _) => a.bic(s, T, A, r(v)),
            (0x2f7, Pfx::None) => {
                let (done, zero) = (a.label(), a.label());
                a.ubfx(Size::X, B, r(v), 0, 8);
                a.cmp_imm(Size::X, B, bits);
                a.b_cond(a64::Cond::Hs, zero);
                a.lsrv(s, T, A, B);
                a.ubfx(Size::X, B, r(v), 8, 8);
                a.cmp_imm(Size::X, B, 64);
                a.b_cond(a64::Cond::Hs, done);
                a.mov_imm(Size::X, super::U, 1);
                a.lslv(Size::X, super::U, super::U, B);
                a.sub_imm(Size::X, super::U, super::U, 1);
                a.and(s, T, T, super::U);
                a.b(done);
                a.bind(zero);
                a.mov_imm(Size::X, T, 0);
                a.bind(done);
                a.mov(s, T, T);
            }
            (0x2f7, Pfx::P66) => a.lslv(s, T, A, r(v)),
            (0x2f7, Pfx::F3) => a.asrv(s, T, A, r(v)),
            (0x2f7, _) => a.lsrv(s, T, A, r(v)),
            _ => a.ror_imm(s, T, A, i.imm as u32 & (bits - 1)),
        }
        if flags && matches!((op, p), (0x2f2, _) | (0x2f7, Pfx::None)) {
            a.tst(s, T, T);
            if op == 0x2f7 {
                a.mrs(B, a64::sysreg::NZCV);
                assert!(a.logic_imm(Logic::And, Size::W, B, B, 1 << 30));
                a.msr(a64::sysreg::NZCV, B);
            }
            a.cfinv();
            a.mov_imm(Size::X, super::PF_RESULT, 1);
            a.mov_imm(Size::X, super::AF_AUX, 1);
        }
        a.mov(s, r(i.reg), T);
    }

    /// VCVTPS2PH: four or eight singles in `ymm[i.reg]` to IEEE half floats in
    /// `xmm[i.rm]` or memory, temporarily switching FPCR.RMode unless imm[2] is set.
    fn vcvtps2ph(&mut self, i: &Insn, wide: bool) {
        let imm = i.imm as u32;
        let override_rc = imm & 4 == 0;
        let a = &mut self.a;
        a.ldr_q(XA, CTX, ymm(i.reg, 0));
        if wide {
            a.ldr_q(XB, CTX, ymm(i.reg, 1));
        }
        if override_rc {
            let rc = imm & 3;
            let arm_rc = ((rc & 1) << 1) | (rc >> 1);
            a.mrs(A, a64::sysreg::FPCR);
            a.mov(Size::X, B, A);
            a.mov_imm(Size::X, T, arm_rc as u64);
            a.bfi(Size::X, B, T, 22, 2);
            a.msr(a64::sysreg::FPCR, B);
        }
        a.emit(FCVTN_4H | vregs(XR, XA));
        if wide {
            a.emit(FCVTN_4H | Q | vregs(XR, XB));
        }
        if override_rc {
            a.msr(a64::sysreg::FPCR, A);
        }
        match i.mem {
            Some(_) if wide => a.str_q(XR, ADDR, 0),
            Some(_) => a.str_d(XR, ADDR, 0),
            None => {
                a.str_q(XR, CTX, ymm(i.rm, 0));
                self.zero_upper(i.rm);
            }
        }
    }

    /// LDMXCSR and STMXCSR (and their VEX forms) on the MXCSR slot of the
    /// context block. A load sets FPCR as `engine::fpcr` does and clears
    /// FPSR, whose flags MXCSR now replaces; a value with reserved bits set
    /// goes to the interpreter, which raises #GP. A store adds the flags
    /// FPSR gathered since.
    fn mxcsr(&mut self, i: &Insn) {
        const MXCSR: i32 = crate::layout::MXCSR as i32;
        const FPCR_SSE: i32 = crate::layout::FPCR_SSE as i32;
        if i.ext() == 2 {
            let bail = self.bail();
            let a = &mut self.a;
            a.ldr(Width::W, A, ADDR, 0);
            a.lsr_imm(Size::W, T, A, 16);
            a.cbnz(Size::W, T, bail);
            a.mov_imm(Size::X, B, 0b110);
            for (from, to) in [(6, 0), (15, 24), (13, 23), (14, 22)] {
                a.ubfx(Size::X, T, A, from, 1);
                a.logic_shift(Logic::Orr, false, Size::X, B, B, T, a64::Shift::Lsl, to);
            }
            a.msr(a64::sysreg::FPCR, B);
            a.msr(a64::sysreg::FPSR, Reg(31));
            a.str(Width::X, B, CTX, FPCR_SSE);
            a.str(Width::X, A, CTX, MXCSR);
        } else {
            let a = &mut self.a;
            a.mrs(T, a64::sysreg::FPSR);
            a.ldr(Width::X, A, CTX, MXCSR);
            assert!(a.logic_imm(Logic::And, Size::X, B, T, 1));
            a.orr(Size::X, A, A, B);
            a.ubfx(Size::X, B, T, 7, 1);
            a.logic_shift(Logic::Orr, false, Size::X, A, A, B, a64::Shift::Lsl, 1);
            assert!(a.logic_imm(Logic::And, Size::X, B, T, 0x1e));
            a.logic_shift(Logic::Orr, false, Size::X, A, A, B, a64::Shift::Lsl, 1);
            a.str(Width::W, A, ADDR, 0);
        }
    }

    /// VMASKMOVPS/PD and VPMASKMOVD/Q, loads and stores. Each lane whose
    /// mask sign bit is set is moved on its own, so masked-off lanes never
    /// touch memory; a load builds the whole result before writing it.
    fn masked_move(&mut self, i: &Insn, op: u16, halves: u32) {
        let double = matches!(op, 0x22d | 0x22f) || (op >= 0x28c && i.rex_w());
        let store = matches!(op, 0x22e | 0x22f | 0x28e);
        let (size, bytes, w) = if double {
            (3, 8, Width::X)
        } else {
            (2, 4, Width::W)
        };
        let lanes = 16 / bytes;
        let mask = i.vex.expect("VEX only").vvvv;
        for h in 0..halves {
            let out = if h == 0 { XR } else { XT };
            if !store {
                self.a.vec3(true, true, 0, 0b00011, out, out, out);
            }
            for k in 0..lanes {
                let lane = ymm(mask, h) + (bytes * k) as i32;
                let at = (16 * h + bytes * k) as i32;
                let skip = self.a.label();
                self.a.ldr(w, A, CTX, lane);
                self.a.tbz(A, bytes * 8 - 1, skip);
                if store {
                    self.a.ldr(w, A, CTX, ymm(i.reg, h) + (bytes * k) as i32);
                    self.a.str(w, A, ADDR, at);
                } else {
                    self.a.ldr(w, A, ADDR, at);
                    self.a.ins_gen(size, out, k, A);
                }
                self.a.bind(skip);
            }
        }
        if store {
            return;
        }
        self.a.str_q(XR, CTX, ymm(i.reg, 0));
        if halves == 2 {
            self.a.str_q(XT, CTX, ymm(i.reg, 1));
        } else {
            self.zero_upper(i.reg);
        }
    }

    /// VPERMPS and VPERMD: 8 x 32-bit cross-lane permute via 2-register TBL.
    fn vperm_d(&mut self, i: &Insn, src1: u8) {
        self.rm_half(i, 0, XA);
        self.rm_half(i, 1, XB);
        let a = &mut self.a;
        let mask7 = V(6);
        a.mov_imm(Size::W, T, 7);
        a.dup_gen(2, mask7, T);
        a.mov_imm(Size::W, T, 0x0404_0404);
        a.dup_gen(2, XS, T);
        a.mov_imm(Size::W, T, 0x0302_0100);
        a.dup_gen(2, XU, T);
        for (h, out) in [(0, XR), (1, XT)] {
            a.ldr_q(XT, CTX, ymm(src1, h));
            a.vec3(true, false, 0, 0b00011, XT, XT, mask7);
            a.vec3(true, false, 2, 0b10011, XT, XT, XS);
            a.vec3(true, false, 2, 0b10000, XT, XT, XU);
            a.tbl2(out, XA, XT);
        }
        a.str_q(XR, CTX, ymm(i.reg, 0));
        a.str_q(XT, CTX, ymm(i.reg, 1));
    }

    /// SSE4a EXTRQ and INSERTQ on the low 64 bits of an XMM register.
    fn extrq_insertq(&mut self, i: &Insn, op: u16, p: Pfx) {
        let d = if op == 0x178 && p == Pfx::P66 {
            i.rm
        } else {
            i.reg
        };
        let a = &mut self.a;
        a.ldr(Width::X, A, CTX, ymm(d, 0));
        if op == 0x178 {
            let len = i.imm as u32 & 63;
            let idx = i.imm2 as u32 & 63;
            let m = if len == 0 {
                u64::MAX
            } else {
                (1u64 << len) - 1
            };
            if p == Pfx::P66 {
                if idx > 0 {
                    a.lsr_imm(Size::X, A, A, idx);
                }
                if len != 0 {
                    a.mov_imm(Size::X, T, m);
                    a.and(Size::X, A, A, T);
                }
            } else {
                a.ldr(Width::X, B, CTX, ymm(i.rm, 0));
                let field = m << idx;
                a.mov_imm(Size::X, T, field);
                a.bic(Size::X, A, A, T);
                if idx > 0 {
                    a.lsl_imm(Size::X, B, B, idx);
                }
                a.and(Size::X, B, B, T);
                a.orr(Size::X, A, A, B);
            }
        } else {
            let (u, mreg) = (super::U, super::OLD_NZCV);
            if p == Pfx::P66 {
                a.ldr(Width::X, B, CTX, ymm(i.rm, 0));
                a.ubfx(Size::X, T, B, 0, 6);
                a.ubfx(Size::X, u, B, 8, 6);
                // u64::MAX >> ((64 - len) & 63) gives u64::MAX for len == 0.
                a.mov_imm(Size::X, mreg, u64::MAX);
                a.neg(Size::X, T, T);
                a.lsrv(Size::X, mreg, mreg, T);
                a.lsrv(Size::X, A, A, u);
                a.and(Size::X, A, A, mreg);
            } else {
                a.ldr(Width::X, B, CTX, ymm(i.rm, 0));
                a.ldr(Width::X, super::OLD_AF, CTX, ymm(i.rm, 0) + 8);
                a.ubfx(Size::X, T, super::OLD_AF, 0, 6);
                a.ubfx(Size::X, u, super::OLD_AF, 8, 6);
                a.mov_imm(Size::X, mreg, u64::MAX);
                a.neg(Size::X, T, T);
                a.lsrv(Size::X, mreg, mreg, T);
                a.lslv(Size::X, T, mreg, u);
                a.bic(Size::X, A, A, T);
                a.lslv(Size::X, B, B, u);
                a.and(Size::X, B, B, T);
                a.orr(Size::X, A, A, B);
            }
        }
        a.str(Width::X, A, CTX, ymm(d, 0));
    }

    /// PCMPESTRM, PCMPESTRI, PCMPISTRM and PCMPISTRI (and VEX 128-bit forms).
    fn string_compare(&mut self, i: &Insn, op: u16, vex: bool, flags: bool) {
        let imm = i.imm as u8;
        let words = imm & 1 != 0;
        let signed = imm & 2 != 0;
        let agg = (imm >> 2) & 3;
        let pol = (imm >> 4) & 3;
        let ctrl = imm & 0x40 != 0;
        let explicit = op <= 0x361;
        let index_out = op & 1 == 1;
        let n: u64 = if words { 8 } else { 16 };
        let all: u64 = (1u64 << n) - 1;
        let sz: u32 = if words { 1 } else { 0 };
        let elem_bytes: u32 = if words { 2 } else { 1 };
        let cmeq_zero: u32 = 0x4e20_9800 | (sz << 22);

        self.a.ldr_q(XA, CTX, ymm(i.reg, 0));
        self.rm_half(i, 0, XB);

        let (u, old_nzcv, old_af) = (super::U, super::OLD_NZCV, super::OLD_AF);
        let a = &mut self.a;
        // XU holds per-lane bit weights (1, 2, 4, ..., 128) for bitmask extraction/expansion.
        if words {
            a.mov_imm(Size::X, T, 0x0008_0004_0002_0001);
            a.fmov_dx(XU, T);
            a.mov_imm(Size::X, T, 0x0080_0040_0020_0010);
            a.fmov_dx(XT, T);
            a.ins_d1(XU, XT);
        } else {
            a.mov_imm(Size::X, T, 0x8040_2010_0804_0201);
            a.dup_gen(3, XU, T);
        }

        let bitmask = |a: &mut a64::Asm, dst: Reg, src: V| {
            a.vec3(true, false, 0, 0b00011, XT, src, XU);
            for _ in 0..3 {
                a.vec3(true, false, sz, 0b10111, XT, XT, XT);
            }
            a.umov_h(dst, XT, 0);
        };

        // Compute lengths la -> A and lb -> B, each in 0..=n.
        if explicit {
            for (dst, src) in [(A, r(0)), (B, r(2))] {
                if i.rex_w() {
                    a.mov(Size::X, dst, src);
                } else {
                    a.sxtw(dst, src);
                }
                a.asr_imm(Size::X, T, dst, 63);
                a.eor(Size::X, dst, dst, T);
                a.sub(Size::X, dst, dst, T);
                a.mov_imm(Size::X, T, n);
                a.cmp(Size::X, dst, T);
                a.csel(Size::X, dst, T, dst, a64::Cond::Hs);
            }
        } else {
            a.mov_imm(Size::X, old_af, 1 << n);
            for (dst, v) in [(A, XA), (B, XB)] {
                a.emit(cmeq_zero | vregs(XR, v));
                bitmask(a, T, XR);
                a.orr(Size::X, T, T, old_af);
                a.rbit(Size::X, T, T);
                a.clz(Size::X, dst, T);
            }
        }

        // ADDR = mask_b = (1 << lb) - 1.
        a.mov_imm(Size::X, ADDR, 1);
        a.lslv(Size::X, ADDR, ADDR, B);
        a.sub_imm(Size::X, ADDR, ADDR, 1);

        // Compute n-bit result mask -> U according to aggregation mode.
        match agg {
            0 => {
                let (top, done) = (a.label(), a.label());
                a.vec3(true, true, 0, 0b00011, XR, XR, XR);
                a.mov(Size::X, old_af, A);
                a.cbz(Size::X, old_af, done);
                a.bind(top);
                a.dup_elem(sz, XT, XA);
                a.vec3(true, true, sz, 0b10001, XT, XB, XT);
                a.vec3(true, false, 2, 0b00011, XR, XR, XT);
                a.ext(XA, XA, XA, elem_bytes);
                a.sub_imm(Size::X, old_af, old_af, 1);
                a.cbnz(Size::X, old_af, top);
                a.bind(done);
                bitmask(a, u, XR);
                a.and(Size::X, u, u, ADDR);
            }
            1 => {
                let (top, done) = (a.label(), a.label());
                a.vec3(true, true, 0, 0b00011, XR, XR, XR);
                a.lsr_imm(Size::X, old_af, A, 1);
                a.cbz(Size::X, old_af, done);
                a.bind(top);
                a.dup_elem(sz, XT, XA);
                a.ext(XA, XA, XA, elem_bytes);
                a.dup_elem(sz, XS, XA);
                a.ext(XA, XA, XA, elem_bytes);
                a.vec3(true, !signed, sz, 0b00111, XT, XB, XT);
                a.vec3(true, !signed, sz, 0b00111, XS, XS, XB);
                a.vec3(true, false, 0, 0b00011, XT, XT, XS);
                a.vec3(true, false, 2, 0b00011, XR, XR, XT);
                a.sub_imm(Size::X, old_af, old_af, 1);
                a.cbnz(Size::X, old_af, top);
                a.bind(done);
                bitmask(a, u, XR);
                a.and(Size::X, u, u, ADDR);
            }
            2 => {
                a.vec3(true, true, sz, 0b10001, XR, XA, XB);
                bitmask(a, u, XR);
                a.mov_imm(Size::X, old_af, 1);
                a.lslv(Size::X, old_af, old_af, A);
                a.sub_imm(Size::X, old_af, old_af, 1);
                a.and(Size::X, T, old_af, ADDR);
                a.and(Size::X, u, u, T);
                a.orr(Size::X, T, old_af, ADDR);
                a.mov_imm(Size::X, old_nzcv, all);
                a.bic(Size::X, T, old_nzcv, T);
                a.orr(Size::X, u, u, T);
            }
            _ => {
                let (top, done) = (a.label(), a.label());
                a.mov_imm(Size::X, u, all);
                a.cbz(Size::X, A, done);
                a.mov_imm(Size::X, old_af, 0);
                a.mov_imm(Size::X, old_nzcv, !all);
                a.bind(top);
                a.dup_elem(sz, XT, XA);
                a.vec3(true, true, sz, 0b10001, XR, XB, XT);
                bitmask(a, T, XR);
                a.and(Size::X, T, T, ADDR);
                a.orr(Size::X, T, T, old_nzcv);
                a.lsrv(Size::X, T, T, old_af);
                a.and(Size::X, u, u, T);
                a.ext(XA, XA, XA, elem_bytes);
                a.add_imm(Size::X, old_af, old_af, 1);
                a.cmp(Size::X, old_af, A);
                a.b_cond(a64::Cond::Lo, top);
                a.bind(done);
            }
        }

        // Polarity adjustment.
        match pol {
            1 => {
                a.mov_imm(Size::X, T, all);
                a.eor(Size::X, u, u, T);
            }
            3 => a.eor(Size::X, u, u, ADDR),
            _ => {}
        }

        // Write destination: RCX for *I, XMM0 (zeroing upper if VEX) for *M.
        if index_out {
            let (zero, out_done) = (a.label(), a.label());
            a.cbz(Size::X, u, zero);
            if ctrl {
                a.clz(Size::W, T, u);
                a.mov_imm(Size::X, old_af, 31);
                a.sub(Size::X, r(1), old_af, T);
            } else {
                a.rbit(Size::W, T, u);
                a.clz(Size::W, r(1), T);
            }
            a.b(out_done);
            a.bind(zero);
            a.mov_imm(Size::X, r(1), n);
            a.bind(out_done);
        } else {
            if !ctrl {
                a.stp(Size::X, u, a64::ZR, CTX, ymm(0, 0));
            } else if !words {
                a.dup_gen(0, XT, u);
                a.lsr_imm(Size::X, T, u, 8);
                a.dup_gen(0, XS, T);
                a.ins(3, XT, 1, XS, 0);
                a.vec3(true, false, 0, 0b10001, XT, XT, XU);
                a.str_q(XT, CTX, ymm(0, 0));
            } else {
                a.dup_gen(1, XT, u);
                a.vec3(true, false, 1, 0b10001, XT, XT, XU);
                a.str_q(XT, CTX, ymm(0, 0));
            }
            if vex {
                self.zero_upper(0);
            }
        }

        if flags {
            let a = &mut self.a;
            a.mov_imm(Size::X, old_nzcv, 0);
            a.cmp_imm(Size::X, A, n as u32);
            a.cset(Size::X, T, a64::Cond::Lo);
            a.bfi(Size::X, old_nzcv, T, 31, 1);
            a.cmp_imm(Size::X, B, n as u32);
            a.cset(Size::X, T, a64::Cond::Lo);
            a.bfi(Size::X, old_nzcv, T, 30, 1);
            a.cmp_imm(Size::X, u, 0);
            a.cset(Size::X, T, a64::Cond::Eq);
            a.bfi(Size::X, old_nzcv, T, 29, 1);
            a.bfi(Size::X, old_nzcv, u, 28, 1);
            a.msr(a64::sysreg::NZCV, old_nzcv);
            a.mov_imm(Size::X, super::PF_RESULT, 1);
            a.mov_imm(Size::X, super::AF_AUX, 1);
        }
    }
}
