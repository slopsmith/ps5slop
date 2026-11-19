//! The x87 floating-point unit. Values are kept as `f64`, so 80-bit
//! operands are rounded to double precision when loaded. Exceptions are
//! never raised; of their status bits only the invalid operation (IE) one
//! is tracked.

use super::{Ctx, Loc, Res};
use crate::cpu::flags::{AF, CF, OF, PF, SF, ZF};
use crate::cpu::reg::RAX;

const IE: u16 = 1;
const C0: u16 = 1 << 8;
const C1: u16 = 1 << 9;
const C2: u16 = 1 << 10;
const C3: u16 = 1 << 14;
const TOP_SHIFT: u16 = 11;

/// `f64` as an 80-bit extended value: the significand (with its explicit
/// integer bit) and the sign and exponent word.
pub(super) fn to_extended(v: f64) -> (u64, u16) {
    let bits = v.to_bits();
    let sign = ((bits >> 63) as u16) << 15;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1 << 52) - 1);
    match exp {
        0 if frac == 0 => (0, sign),
        0 => {
            let shift = frac.leading_zeros();
            (frac << shift, sign | (15372 - shift as i32) as u16)
        }
        0x7ff => (1 << 63 | frac << 11, sign | 0x7fff),
        _ => (1 << 63 | frac << 11, sign | (exp - 1023 + 16383) as u16),
    }
}

/// An 80-bit extended value rounded to the nearest `f64`.
pub(super) fn from_extended(mant: u64, se: u16) -> f64 {
    let sign = ((se >> 15) as u64) << 63;
    let mut e = (se & 0x7fff) as i32;
    if e != 0 && mant >> 63 == 0 {
        return f64::NAN;
    }
    if e == 0x7fff {
        let frac = (mant >> 11) & ((1 << 52) - 1);
        return f64::from_bits(match mant << 1 {
            0 => sign | 0x7ff << 52,
            _ => sign | 0x7ff << 52 | frac | 1 << 51,
        });
    }
    if mant == 0 {
        return f64::from_bits(sign);
    }
    let mut mant = mant;
    e = e.max(1);
    let shift = mant.leading_zeros() as i32;
    mant <<= shift;
    e -= shift;
    let field = e - 16383 + 1023;
    if field >= 1 {
        let mut keep = mant >> 11;
        let rem = mant & 0x7ff;
        if rem > 0x400 || (rem == 0x400 && keep & 1 == 1) {
            keep += 1;
        }
        let mut field = field as u64;
        if keep == 1 << 53 {
            keep >>= 1;
            field += 1;
        }
        if field >= 0x7ff {
            return f64::from_bits(sign | 0x7ff << 52);
        }
        return f64::from_bits(sign | field << 52 | (keep & ((1 << 52) - 1)));
    }
    let s = (12 - field) as u32;
    if s > 64 {
        return f64::from_bits(sign);
    }
    let (mut keep, rem) = if s == 64 {
        (0, mant)
    } else {
        (mant >> s, mant & ((1 << s) - 1))
    };
    let half = 1u64 << (s - 1);
    if rem > half || (rem == half && keep & 1 == 1) {
        keep += 1;
    }
    f64::from_bits(sign | keep)
}

/// Rounds `v` to an integer the way rounding control `rc` says.
fn round(v: f64, rc: u16) -> f64 {
    match rc & 3 {
        0 => v.round_ties_even(),
        1 => v.floor(),
        2 => v.ceil(),
        _ => v.trunc(),
    }
}

/// `v` as a `size`-byte integer, or the integer indefinite value when it
/// does not fit.
fn to_int(v: f64, size: u8) -> u64 {
    let bits = size as u32 * 8;
    let limit = 2f64.powi(bits as i32 - 1);
    if v.is_nan() || v < -limit || v >= limit {
        return 1 << (bits - 1);
    }
    (v as i64 as u64) & (u64::MAX >> (64 - bits))
}

/// `v * 2^n` without overflowing in an intermediate power of two.
fn scale(mut v: f64, n: f64) -> f64 {
    let mut n = n.clamp(-4000.0, 4000.0) as i32;
    while n != 0 {
        let step = n.clamp(-1000, 1000);
        v *= 2f64.powi(step);
        n -= step;
    }
    v
}

/// The exponent and significand (in [1, 2)) of a finite, nonzero `v`.
fn extract(v: f64) -> (f64, f64) {
    let (v, bias) = if v.abs() < f64::MIN_POSITIVE {
        (v * 2f64.powi(64), 64)
    } else {
        (v, 0)
    };
    let bits = v.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023 - bias;
    let sig = f64::from_bits((bits & !(0x7ff << 52)) | 1023 << 52);
    (e as f64, sig)
}

/// The condition codes C3, C2 and C0 for comparing `a` with `b`.
fn compare(a: f64, b: f64) -> u16 {
    match a.partial_cmp(&b) {
        None => C3 | C2 | C0,
        Some(std::cmp::Ordering::Less) => C0,
        Some(std::cmp::Ordering::Equal) => C3,
        Some(std::cmp::Ordering::Greater) => 0,
    }
}

/// The result of arithmetic operation `op` (a ModRM reg field) on `a`
/// (the destination) and `b`.
fn arith(op: u8, a: f64, b: f64) -> f64 {
    match op {
        0 => a + b,
        1 => a * b,
        4 => a - b,
        5 => b - a,
        6 => a / b,
        _ => b / a,
    }
}

impl Ctx<'_> {
    fn top(&self) -> usize {
        ((self.cpu.fsw >> TOP_SHIFT) & 7) as usize
    }

    fn set_top(&mut self, t: usize) {
        self.cpu.fsw = (self.cpu.fsw & !(7 << TOP_SHIFT)) | ((t as u16 & 7) << TOP_SHIFT);
    }

    fn phys(&self, i: usize) -> usize {
        (self.top() + i) & 7
    }

    fn st(&self, i: usize) -> f64 {
        f64::from_bits(self.cpu.st[self.phys(i)])
    }

    fn set_st(&mut self, i: usize, v: f64) {
        let p = self.phys(i);
        self.cpu.st[p] = v.to_bits();
        self.cpu.ftag |= 1 << p;
    }

    fn fpush(&mut self, v: f64) {
        self.set_top(self.top().wrapping_sub(1));
        self.set_st(0, v);
    }

    fn fpop(&mut self) {
        self.cpu.ftag &= !(1 << self.top());
        self.set_top(self.top() + 1);
    }

    /// Writes a computed result to ST(`i`); a NaN computed from operands
    /// that were not NaN is an invalid operation.
    fn set_result(&mut self, i: usize, v: f64, inputs: &[f64]) {
        if v.is_nan() && !inputs.iter().any(|x| x.is_nan()) {
            self.cpu.fsw |= IE;
        }
        self.set_st(i, v);
    }

    fn set_cc(&mut self, cc: u16) {
        self.cpu.fsw = (self.cpu.fsw & !(C0 | C1 | C2 | C3)) | cc;
    }

    /// FCOMI and friends: the comparison in ZF, PF and CF.
    fn compare_flags(&mut self, a: f64, b: f64) {
        let cc = compare(a, b);
        let mut f = 0;
        if cc & C3 != 0 {
            f |= ZF;
        }
        if cc & C2 != 0 {
            f |= PF;
        }
        if cc & C0 != 0 {
            f |= CF;
        }
        self.cpu.set_arith(f & !(OF | SF | AF));
        self.cpu.fsw &= !C1;
    }

    fn rc(&self) -> u16 {
        (self.cpu.fcw >> 10) & 3
    }

    fn load_f32(&self, a: u64) -> Res<f64> {
        Ok(f32::from_bits(self.load(a, 4)? as u32) as f64)
    }

    fn load_f64(&self, a: u64) -> Res<f64> {
        Ok(f64::from_bits(self.load(a, 8)?))
    }

    fn load_f80(&self, a: u64) -> Res<f64> {
        Ok(from_extended(self.load(a, 8)?, self.load(a + 8, 2)? as u16))
    }

    fn store_f80(&mut self, a: u64, v: f64) -> Res<()> {
        let (mant, se) = to_extended(v);
        self.store(a, 8, mant)?;
        self.store(a + 8, 2, se as u64)
    }

    fn load_int(&self, a: u64, size: u8) -> Res<f64> {
        let v = self.load(a, size)?;
        let shift = 64 - 8 * size as u32;
        Ok(((v << shift) as i64 >> shift) as f64)
    }

    /// FIST/FISTP (`rc` from the control word) and FISTTP (truncating).
    fn store_int(&mut self, a: u64, size: u8, truncate: bool) -> Res<()> {
        let rc = if truncate { 3 } else { self.rc() };
        let v = round(self.st(0), rc);
        let int = to_int(v, size);
        if int == 1 << (size as u32 * 8 - 1) && v != -(2f64.powi(size as i32 * 8 - 1)) {
            self.cpu.fsw |= IE;
        }
        self.store(a, size, int)
    }

    /// The 16-bit tag word: 2 bits per register (valid, zero, special,
    /// empty).
    pub(super) fn full_tag_word(&self) -> u16 {
        (0..8).fold(0, |w, p| {
            let v = f64::from_bits(self.cpu.st[p]);
            let tag = if self.cpu.ftag & (1 << p) == 0 {
                3
            } else if v == 0.0 {
                1
            } else if !v.is_normal() {
                2
            } else {
                0
            };
            w | tag << (2 * p)
        })
    }

    fn fninit(&mut self) {
        self.cpu.fcw = crate::cpu::FCW_DEFAULT;
        self.cpu.fsw = 0;
        self.cpu.ftag = 0;
    }

    /// The size of one FNSTENV field: 2 bytes with an operand-size prefix
    /// (a 14-byte environment), else 4 (28 bytes).
    fn env_field(&self) -> u64 {
        if self.insn.opsize_prefix { 2 } else { 4 }
    }

    /// FNSTENV's environment (instruction and operand pointers are stored as
    /// zero); returns its size.
    fn store_env(&mut self, a: u64) -> Res<u64> {
        let n = self.env_field();
        let words = [self.cpu.fcw, self.cpu.fsw, self.full_tag_word()];
        for (k, w) in words.iter().enumerate() {
            self.store(a + n * k as u64, n as u8, 0xffff_0000 | *w as u64)?;
        }
        for k in 3..7 {
            self.store(a + n * k, n as u8, 0)?;
        }
        Ok(7 * n)
    }

    fn load_env(&mut self, a: u64) -> Res<u64> {
        let n = self.env_field();
        self.cpu.fcw = self.load(a, 2)? as u16;
        self.cpu.fsw = self.load(a + n, 2)? as u16;
        let tags = self.load(a + 2 * n, 2)?;
        self.cpu.ftag = (0..8).fold(0, |t, p| match (tags >> (2 * p)) & 3 {
            3 => t,
            _ => t | 1 << p,
        });
        Ok(7 * n)
    }

    pub(super) fn x87(&mut self) -> Res<()> {
        let i = self.insn;
        let ext = i.ext();
        match self.rm() {
            Loc::Mem(a) => self.x87_memory(i.op, ext, a),
            Loc::Reg(_) => self.x87_register(i.op, ext, (i.modrm.unwrap_or(0) & 7) as usize),
        }
    }

    /// Operations on ST(0) and a memory operand, or ST(0) with ST(i)
    /// (D8 forms): add, mul, com, comp, sub, subr, div, divr.
    fn arith_st0(&mut self, op: u8, v: f64) {
        match op {
            2 | 3 => {
                let cc = compare(self.st(0), v);
                self.set_cc(cc);
                if op == 3 {
                    self.fpop();
                }
            }
            _ => {
                let a = self.st(0);
                self.set_result(0, arith(op, a, v), &[a, v]);
            }
        }
    }

    /// DC and DE register forms: ST(i) = ST(i) op ST(0), where the
    /// encodings of the reversed subtractions and divisions swap.
    fn arith_sti(&mut self, op: u8, r: usize, pop: bool) {
        let op = match op {
            4 => 5,
            5 => 4,
            6 => 7,
            7 => 6,
            o => o,
        };
        let (a, b) = (self.st(r), self.st(0));
        self.set_result(r, arith(op, a, b), &[a, b]);
        if pop {
            self.fpop();
        }
    }

    fn x87_memory(&mut self, op: u8, ext: u8, a: u64) -> Res<()> {
        match (op, ext) {
            (0xd8, _) => {
                let v = self.load_f32(a)?;
                self.arith_st0(ext, v);
            }
            (0xda, _) => {
                let v = self.load_int(a, 4)?;
                self.arith_st0(ext, v);
            }
            (0xdc, _) => {
                let v = self.load_f64(a)?;
                self.arith_st0(ext, v);
            }
            (0xde, _) => {
                let v = self.load_int(a, 2)?;
                self.arith_st0(ext, v);
            }
            (0xd9, 0) => {
                let v = self.load_f32(a)?;
                self.fpush(v);
            }
            (0xd9, 2 | 3) => {
                self.store(a, 4, (self.st(0) as f32).to_bits() as u64)?;
                if ext == 3 {
                    self.fpop();
                }
            }
            (0xd9, 4) => _ = self.load_env(a)?,
            (0xd9, 5) => self.cpu.fcw = self.load(a, 2)? as u16,
            (0xd9, 6) => {
                _ = self.store_env(a)?;
                self.cpu.fcw |= 0x3f;
            }
            (0xd9, 7) => self.store(a, 2, self.cpu.fcw as u64)?,
            (0xdb, 0) => {
                let v = self.load_int(a, 4)?;
                self.fpush(v);
            }
            (0xdb, 1..=3) => {
                self.store_int(a, 4, ext == 1)?;
                if ext != 2 {
                    self.fpop();
                }
            }
            (0xdb, 5) => {
                let v = self.load_f80(a)?;
                self.fpush(v);
            }
            (0xdb, 7) => {
                self.store_f80(a, self.st(0))?;
                self.fpop();
            }
            (0xdd, 0) => {
                let v = self.load_f64(a)?;
                self.fpush(v);
            }
            (0xdd, 1) => {
                self.store_int(a, 8, true)?;
                self.fpop();
            }
            (0xdd, 2 | 3) => {
                self.store(a, 8, self.st(0).to_bits())?;
                if ext == 3 {
                    self.fpop();
                }
            }
            (0xdd, 4) => {
                let regs = a + self.load_env(a)?;
                for k in 0..8 {
                    let v = self.load_f80(regs + 10 * k as u64)?;
                    let p = self.phys(k);
                    self.cpu.st[p] = v.to_bits();
                }
            }
            (0xdd, 6) => {
                let regs = a + self.store_env(a)?;
                for k in 0..8 {
                    self.store_f80(regs + 10 * k as u64, self.st(k))?;
                }
                self.fninit();
            }
            (0xdd, 7) => self.store(a, 2, self.cpu.fsw as u64)?,
            (0xdf, 0) => {
                let v = self.load_int(a, 2)?;
                self.fpush(v);
            }
            (0xdf, 1..=3) => {
                self.store_int(a, 2, ext == 1)?;
                if ext != 2 {
                    self.fpop();
                }
            }
            (0xdf, 5) => {
                let v = self.load_int(a, 8)?;
                self.fpush(v);
            }
            (0xdf, 7) => {
                self.store_int(a, 8, false)?;
                self.fpop();
            }
            _ => return self.unimplemented(),
        }
        Ok(())
    }

    fn x87_register(&mut self, op: u8, ext: u8, r: usize) -> Res<()> {
        let flags = |c: &Self, cc: u8| match cc {
            0 => c.cpu.flag(CF),
            1 => c.cpu.flag(ZF),
            2 => c.cpu.flag(CF) || c.cpu.flag(ZF),
            _ => c.cpu.flag(PF),
        };
        match (op, ext) {
            (0xd8, _) => self.arith_st0(ext, self.st(r)),
            (0xdc, 2 | 3) => self.arith_st0(ext, self.st(r)),
            (0xdc, _) => self.arith_sti(ext, r, false),
            (0xde, 3) if r == 1 => {
                let cc = compare(self.st(0), self.st(1));
                self.set_cc(cc);
                self.fpop();
                self.fpop();
            }
            (0xde, 0 | 1 | 4..=7) => self.arith_sti(ext, r, true),
            (0xd9, 0) => self.fpush(self.st(r)),
            (0xd9, 1) => {
                let (a, b) = (self.st(0), self.st(r));
                self.set_st(0, b);
                self.set_st(r, a);
            }
            (0xd9, 2) if r == 0 => {}
            (0xd9, 4) => match r {
                0 => self.set_st(0, -self.st(0)),
                1 => self.set_st(0, self.st(0).abs()),
                4 => {
                    let cc = compare(self.st(0), 0.0);
                    self.set_cc(cc);
                }
                5 => {
                    let v = self.st(0);
                    let class = if self.cpu.ftag & (1 << self.top()) == 0 {
                        C3 | C0
                    } else if v.is_nan() {
                        C0
                    } else if v.is_infinite() {
                        C2 | C0
                    } else if v == 0.0 {
                        C3
                    } else if v.is_subnormal() {
                        C3 | C2
                    } else {
                        C2
                    };
                    let sign = if v.is_sign_negative() { C1 } else { 0 };
                    self.set_cc(class | sign);
                }
                _ => return self.unimplemented(),
            },
            (0xd9, 5) => {
                use std::f64::consts::{LN_2, LOG2_10, LOG2_E, LOG10_2, PI};
                let v = match r {
                    0 => 1.0,
                    1 => LOG2_10,
                    2 => LOG2_E,
                    3 => PI,
                    4 => LOG10_2,
                    5 => LN_2,
                    6 => 0.0,
                    _ => return self.unimplemented(),
                };
                self.fpush(v);
            }
            (0xd9, 6) => {
                let (a, b) = (self.st(0), self.st(1));
                match r {
                    0 => self.set_result(0, (a * std::f64::consts::LN_2).exp_m1(), &[a]),
                    1 => {
                        self.set_result(1, b * a.log2(), &[a, b]);
                        self.fpop();
                    }
                    2 => {
                        self.set_result(0, a.tan(), &[a]);
                        self.fpush(1.0);
                        self.cpu.fsw &= !C2;
                    }
                    3 => {
                        self.set_st(1, b.atan2(a));
                        self.fpop();
                    }
                    4 => {
                        if a == 0.0 {
                            self.set_st(0, f64::NEG_INFINITY);
                            self.fpush(a);
                        } else if !a.is_finite() {
                            self.set_st(0, a.abs());
                            self.fpush(a);
                        } else {
                            let (e, sig) = extract(a);
                            self.set_st(0, e);
                            self.fpush(sig);
                        }
                    }
                    5 => self.partial_remainder(true),
                    6 => {
                        self.set_top(self.top().wrapping_sub(1));
                        self.cpu.fsw &= !C1;
                    }
                    _ => {
                        self.set_top(self.top() + 1);
                        self.cpu.fsw &= !C1;
                    }
                }
            }
            (0xd9, 7) => {
                let (a, b) = (self.st(0), self.st(1));
                match r {
                    0 => self.partial_remainder(false),
                    1 => {
                        self.set_result(1, b * (a.ln_1p() / std::f64::consts::LN_2), &[a, b]);
                        self.fpop();
                    }
                    2 => self.set_result(0, a.sqrt(), &[a]),
                    3 => {
                        let (s, c) = a.sin_cos();
                        self.set_result(0, s, &[a]);
                        self.fpush(c);
                        self.cpu.fsw &= !C2;
                    }
                    4 => self.set_st(0, round(a, self.rc())),
                    5 => {
                        let v = if b.is_infinite() {
                            a * if b > 0.0 { f64::INFINITY } else { 0.0 }
                        } else {
                            scale(a, b.trunc())
                        };
                        self.set_result(0, v, &[a, b]);
                    }
                    6 => {
                        self.set_result(0, a.sin(), &[a]);
                        self.cpu.fsw &= !C2;
                    }
                    _ => {
                        self.set_result(0, a.cos(), &[a]);
                        self.cpu.fsw &= !C2;
                    }
                }
            }
            (0xda, 0..=3) => {
                if flags(self, ext) {
                    self.set_st(0, self.st(r));
                }
            }
            (0xda, 5) if r == 1 => {
                let cc = compare(self.st(0), self.st(1));
                self.set_cc(cc);
                self.fpop();
                self.fpop();
            }
            (0xdb, 0..=3) => {
                if !flags(self, ext) {
                    self.set_st(0, self.st(r));
                }
            }
            (0xdb, 4) => match r {
                2 => self.cpu.fsw &= 0x7f00,
                3 => self.fninit(),
                0 | 1 | 4 => {}
                _ => return self.unimplemented(),
            },
            (0xdb, 5 | 6) => self.compare_flags(self.st(0), self.st(r)),
            (0xdd, 0) => {
                let p = self.phys(r);
                self.cpu.ftag &= !(1 << p);
            }
            (0xdd, 2 | 3) => {
                self.set_st(r, self.st(0));
                if ext == 3 {
                    self.fpop();
                }
            }
            (0xdd, 4 | 5) => {
                let cc = compare(self.st(0), self.st(r));
                self.set_cc(cc);
                if ext == 5 {
                    self.fpop();
                }
            }
            (0xdf, 4) if r == 0 => self.set(RAX as u8, 2, self.cpu.fsw as u64),
            (0xdf, 5 | 6) => {
                self.compare_flags(self.st(0), self.st(r));
                self.fpop();
            }
            _ => return self.unimplemented(),
        }
        Ok(())
    }

    /// FPREM (truncating quotient) and FPREM1 (rounded): ST(0) becomes the
    /// remainder of ST(0) / ST(1), with the low quotient bits in C0, C3, C1.
    fn partial_remainder(&mut self, nearest: bool) {
        let (a, b) = (self.st(0), self.st(1));
        let q = if b.is_infinite() && a.is_finite() {
            0.0
        } else if nearest {
            (a / b).round_ties_even()
        } else {
            (a / b).trunc()
        };
        let r = match (q, nearest) {
            (0.0, _) if b.is_infinite() && a.is_finite() => a,
            (_, true) => match a - q * b {
                0.0 => 0.0f64.copysign(a),
                r => r,
            },
            (_, false) => a % b,
        };
        self.set_result(0, r, &[a, b]);
        let q = if q.is_finite() { q.abs() as u64 } else { 0 };
        let mut cc = 0;
        if q & 4 != 0 {
            cc |= C0;
        }
        if q & 2 != 0 {
            cc |= C3;
        }
        if q & 1 != 0 {
            cc |= C1;
        }
        self.set_cc(cc);
    }
}
