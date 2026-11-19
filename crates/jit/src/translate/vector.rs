//! SSE, AVX and AVX2 integer instructions, and BMI, on NEON. YMM registers
//! live in the vCPU's context block (x25) and are loaded per instruction; a
//! 256-bit operation runs as two 128-bit halves. Many NEON choices follow
//! 86slopper's `translate/sse.rs`.

use a64::{Logic, Narrow, Reg, Size, V, Width, r};
use x86::decode::{Insn, Map, Rep};

use super::{A, ADDR, B, FlagUse, T, Translator};

/// The vCPU's context block: YMM`n` at `32 * n`, its upper half 16 further.
pub const CTX: Reg = Reg(25);

const XA: V = V(0);
const XB: V = V(1);
const XR: V = V(2);
const XT: V = V(3);
const XS: V = V(4);

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
        _ => return None,
    })
}

/// Two-source integer operations done per 128-bit half.
fn packed_int(op: u16) -> bool {
    matches!(op, 0x160..=0x16d | 0x1db | 0x1df | 0x1eb | 0x1ef | 0x1e4 | 0x1e5 | 0x1f4..=0x1f6 | 0x200)
        || simd_three(op).is_some()
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
        0x16f | 0x17f => matches!(p, Pfx::P66 | Pfx::F3),
        0x110 | 0x111 | 0x128 | 0x129 => matches!(p, Pfx::None | Pfx::P66),
        0x154..=0x157 => matches!(p, Pfx::None | Pfx::P66),
        0x170 => p == Pfx::P66,
        0x171..=0x173 => {
            p == Pfx::P66
                && i.mem.is_none()
                && matches!((op, i.ext()), (0x173, 2 | 3 | 6 | 7) | (_, 2 | 4 | 6))
        }
        0x177 => i.vex.is_some() && p == Pfx::None,
        0x16e => p == Pfx::P66 && !wide,
        0x17e => (p == Pfx::P66 || p == Pfx::F3) && !wide,
        0x1d6 => p == Pfx::P66 && !wide,
        0x1d7 => p == Pfx::P66 && i.mem.is_none() && !wide,
        _ if packed_int(op) => p == Pfx::P66 && (op != 0x200 || i.map == Map::M0F38),
        0x258 | 0x259 | 0x278 | 0x279 => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x245 | 0x247 => i.vex.is_some() && p == Pfx::P66,
        0x246 => i.vex.is_some() && p == Pfx::P66 && !i.rex_w(),
        0x2f2 => i.vex.is_some() && p == Pfx::None,
        0x2f7 => i.vex.is_some(),
        0x3f0 => i.vex.is_some() && p == Pfx::F2,
        _ => false,
    };
    if !ok || (i.vex.is_none() && wide) {
        return None;
    }
    Some(match op {
        0x2f2 => (false, writes),
        0x2f7 if p == Pfx::None => (false, writes),
        _ => (false, none),
    })
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
        let halves = if i.vex.is_some_and(|v| v.l) { 2 } else { 1 };
        let src1 = i.vex.map_or(i.reg, |v| v.vvvv);
        if let Some(m) = i.mem {
            self.addr(&m, next);
        }
        match op {
            0x2f2 | 0x2f7 | 0x3f0 => return self.bmi(i, op, p, flags),
            0x258 | 0x259 | 0x278 | 0x279 => {
                let size = match op {
                    0x278 => 0,
                    0x279 => 1,
                    0x258 => 2,
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
            0x16f | 0x128 | 0x110 => {
                for h in 0..halves {
                    self.rm_half(i, h, XB);
                    self.a.str_q(XB, CTX, ymm(i.reg, h));
                }
            }
            0x17f | 0x129 | 0x111 => {
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
            0x170 => {
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
            0x1d7 => {
                self.a.ldr_q(XB, CTX, ymm(i.rm, 0));
                self.a.shr_v(true, false, 8, XR, XB, 7);
                self.a.usra(true, 16, XR, XR, 7);
                self.a.usra(true, 32, XR, XR, 14);
                self.a.usra(true, 64, XR, XR, 28);
                self.a.umov_b(A, XR, 0);
                self.a.umov_b(B, XR, 8);
                self.a
                    .logic_shift(Logic::Orr, false, Size::W, A, A, B, a64::Shift::Lsl, 8);
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

    /// `dst` = half `h` of the r/m operand (memory at x19, or a register).
    fn rm_half(&mut self, i: &Insn, h: u32, dst: V) {
        match i.mem {
            Some(_) => self.a.ldr_q(dst, ADDR, 16 * h as i32),
            None => self.a.ldr_q(dst, CTX, ymm(i.rm, h)),
        }
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
            0x163 | 0x167 | 0x16b => {
                let (kind, size) = match op {
                    0x163 => (Narrow::Sqxtn, 0),
                    0x167 => (Narrow::Sqxtun, 0),
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

    /// ANDN, BEXTR, SHLX/SHRX/SARX and RORX on general registers.
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
}
