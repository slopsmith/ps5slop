//! SSE and SSE2: XMM moves, logic, integer SIMD and floating point.
//! Semantics follow the Intel SDM, volume 2; NaN and conversion results
//! follow its rules for SSE (first-operand NaN propagation, the "real
//! indefinite" NaN for invalid operations, the "integer indefinite" for
//! conversions out of range).

use super::*;

/// MXCSR's denormals-are-zero and flush-to-zero bits.
const MXCSR_DAZ: u32 = 1 << 6;
const MXCSR_FTZ: u32 = 1 << 15;

/// The x86 default NaNs produced by invalid operations.
const F32_INDEFINITE: u32 = 0xffc0_0000;
const F64_INDEFINITE: u64 = 0xfff8_0000_0000_0000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pfx {
    None,
    P66,
    F3,
    F2,
}

/// Packed lanes as a little-endian `u128`.
fn lanes<const N: usize>(v: u128) -> [u64; N] {
    let bits = 128 / N;
    let m = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    core::array::from_fn(|i| (v >> (i * bits)) as u64 & m)
}

fn join<const N: usize>(l: [u64; N]) -> u128 {
    let bits = 128 / N;
    let m = if bits == 64 {
        u64::MAX as u128
    } else {
        (1u128 << bits) - 1
    };
    l.iter()
        .enumerate()
        .fold(0, |acc, (i, &x)| acc | (x as u128 & m) << (i * bits))
}

fn map2<const N: usize>(a: u128, b: u128, f: impl Fn(u64, u64) -> u64) -> u128 {
    let (a, b) = (lanes::<N>(a), lanes::<N>(b));
    join::<N>(core::array::from_fn(|i| f(a[i], b[i])))
}

fn sx(v: u64, bits: u32) -> i64 {
    ((v << (64 - bits)) as i64) >> (64 - bits)
}

fn sat_s(v: i64, bits: u32) -> u64 {
    let max = (1i64 << (bits - 1)) - 1;
    v.clamp(-max - 1, max) as u64
}

fn sat_u(v: i64, bits: u32) -> u64 {
    v.clamp(0, (1i64 << bits) - 1) as u64
}

fn quiet32(a: u32) -> u32 {
    a | 0x0040_0000
}

fn quiet64(a: u64) -> u64 {
    a | 0x0008_0000_0000_0000
}

/// An SSE binary operation on f32 with x86 NaN rules.
fn f32op(a: u32, b: u32, f: impl Fn(f32, f32) -> f32) -> u32 {
    let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
    if fa.is_nan() {
        return quiet32(a);
    }
    if fb.is_nan() {
        return quiet32(b);
    }
    let r = f(fa, fb);
    if r.is_nan() {
        F32_INDEFINITE
    } else {
        r.to_bits()
    }
}

fn f64op(a: u64, b: u64, f: impl Fn(f64, f64) -> f64) -> u64 {
    let (fa, fb) = (f64::from_bits(a), f64::from_bits(b));
    if fa.is_nan() {
        return quiet64(a);
    }
    if fb.is_nan() {
        return quiet64(b);
    }
    let r = f(fa, fb);
    if r.is_nan() {
        F64_INDEFINITE
    } else {
        r.to_bits()
    }
}

/// MIN/MAX: the second operand wins when either is NaN or both are zero.
fn minmax32(a: u32, b: u32, max: bool) -> u32 {
    let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
    let pick_a = if max { fa > fb } else { fa < fb };
    if pick_a { a } else { b }
}

fn minmax64(a: u64, b: u64, max: bool) -> u64 {
    let (fa, fb) = (f64::from_bits(a), f64::from_bits(b));
    let pick_a = if max { fa > fb } else { fa < fb };
    if pick_a { a } else { b }
}

fn sqrt32(a: u32) -> u32 {
    let f = f32::from_bits(a);
    if f.is_nan() {
        quiet32(a)
    } else if f < 0.0 {
        F32_INDEFINITE
    } else {
        f.sqrt().to_bits()
    }
}

fn sqrt64(a: u64) -> u64 {
    let f = f64::from_bits(a);
    if f.is_nan() {
        quiet64(a)
    } else if f < 0.0 {
        F64_INDEFINITE
    } else {
        f.sqrt().to_bits()
    }
}

fn f64_to_f32(a: u64) -> u32 {
    let f = f64::from_bits(a);
    if f.is_nan() {
        let sign = ((a >> 63) as u32) << 31;
        let payload = ((a >> 29) as u32) & 0x003f_ffff;
        sign | 0x7fc0_0000 | payload
    } else {
        (f as f32).to_bits()
    }
}

fn f32_to_f64(a: u32) -> u64 {
    let f = f32::from_bits(a);
    if f.is_nan() {
        let sign = ((a >> 31) as u64) << 63;
        let payload = ((a & 0x003f_ffff) as u64) << 29;
        sign | 0x7ff8_0000_0000_0000 | payload
    } else {
        (f as f64).to_bits()
    }
}

/// Rounds per MXCSR.RC (or truncates), then converts to a signed integer of
/// `bits`; out of range and NaN give the integer indefinite (most negative).
fn to_int(f: f64, rc: u32, truncate: bool, bits: u32) -> u64 {
    let r = if truncate {
        f.trunc()
    } else {
        match rc {
            0 => f.round_ties_even(),
            1 => f.floor(),
            2 => f.ceil(),
            _ => f.trunc(),
        }
    };
    let limit = 2f64.powi(bits as i32 - 1);
    let indefinite = 1u64 << (bits - 1);
    if r.is_nan() || r >= limit || r < -limit {
        indefinite
    } else {
        (r as i64) as u64 & alu::mask((bits / 8) as u8)
    }
}

/// CMPPS/CMPSS predicate `p` (0-7).
fn cmp_pred(a: f64, b: f64, p: u8) -> bool {
    let un = a.is_nan() || b.is_nan();
    match p & 7 {
        0 => a == b,
        1 => a < b,
        2 => a <= b,
        3 => un,
        4 => a != b || un,
        5 => a >= b || un,
        6 => a > b || un,
        _ => !un,
    }
}

impl Ctx<'_> {
    fn pfx(&self) -> Pfx {
        match (self.insn.rep, self.insn.opsize_prefix) {
            (Rep::Rep, _) => Pfx::F3,
            (Rep::Repne, _) => Pfx::F2,
            (Rep::None, true) => Pfx::P66,
            (Rep::None, false) => Pfx::None,
        }
    }

    /// The current 128-bit half of register `r`.
    pub(super) fn xget(&self, r: u8) -> u128 {
        let x = self.cpu.xmm[r as usize];
        let h = 2 * self.half as usize;
        x[h] as u128 | (x[h + 1] as u128) << 64
    }

    /// Writes the current half of register `r`. A VEX.128 write also clears
    /// the upper half; a legacy SSE write keeps it.
    pub(super) fn xset(&mut self, r: u8, v: u128) {
        let h = 2 * self.half as usize;
        let clear_upper = self.insn.vex.is_some() && !self.wide;
        let x = &mut self.cpu.xmm[r as usize];
        x[h] = v as u64;
        x[h + 1] = (v >> 64) as u64;
        if clear_upper {
            x[2] = 0;
            x[3] = 0;
        }
    }

    /// The first source: VEX.vvvv, or the destination for legacy SSE.
    pub(super) fn src1(&self) -> u128 {
        match self.insn.vex {
            Some(v) => self.xget(v.vvvv),
            None => self.xget(self.insn.reg),
        }
    }

    /// Whether a memory operand must be aligned: legacy SSE requires it of
    /// most 128-bit operands, VEX only of the explicitly aligned moves.
    fn must_align(&self, legacy: bool) -> bool {
        let i = self.insn;
        match i.vex {
            None => legacy,
            Some(_) => {
                legacy
                    && (matches!(i.op, 0x28 | 0x29 | 0x2b | 0xe7)
                        || (matches!(i.op, 0x6f | 0x7f) && i.opsize_prefix))
            }
        }
    }

    fn check_align(&self, addr: u64, aligned: bool) -> Res<()> {
        let width = if self.wide { 32 } else { 16 };
        if self.must_align(aligned) && !addr.is_multiple_of(width) {
            return Err(CpuFault::GeneralProtection);
        }
        Ok(())
    }

    fn load128(&self, addr: u64, aligned: bool) -> Res<u128> {
        self.check_align(addr, aligned)?;
        let mut b = [0u8; 16];
        self.mem.read(addr + 16 * self.half as u64, &mut b)?;
        Ok(u128::from_le_bytes(b))
    }

    fn store128(&mut self, addr: u64, v: u128, aligned: bool) -> Res<()> {
        self.check_align(addr, aligned)?;
        self.mem
            .write(addr + 16 * self.half as u64, &v.to_le_bytes())?;
        Ok(())
    }

    /// The `xmm/m128` source operand.
    fn src128(&self, aligned: bool) -> Res<u128> {
        match self.rm() {
            Loc::Reg(r) => Ok(self.xget(r)),
            Loc::Mem(a) => self.load128(a, aligned),
        }
    }

    /// The low `bytes` of the `xmm/m` source operand (scalar forms).
    fn src_low(&self, bytes: u8) -> Res<u64> {
        match self.rm() {
            Loc::Reg(r) => Ok(self.xget(r) as u64 & alu::mask(bytes)),
            Loc::Mem(a) => self.load(a + 16 * self.half as u64, bytes),
        }
    }

    /// Writes the low `bytes` of register `r`; the rest comes from `r` itself
    /// (legacy SSE) or from VEX.vvvv.
    fn set_low(&mut self, r: u8, bytes: u8, v: u64) {
        let m = alu::mask(bytes) as u128;
        let base = match self.insn.vex {
            Some(x) => self.xget(x.vvvv),
            None => self.xget(r),
        };
        self.xset(r, (base & !m) | (v as u128 & m));
    }

    /// Runs `i` if it is an SSE/SSE2 instruction or its VEX form that we
    /// implement; returns whether it was one.
    pub(super) fn sse(&mut self) -> Res<bool> {
        let i = self.insn;
        let Some(vex) = i.vex else {
            return self.sse_op();
        };
        if i.map == Map::M0F && i.op == 0x77 {
            for x in self.cpu.xmm.iter_mut() {
                if vex.l {
                    *x = [0; 4];
                } else {
                    x[2] = 0;
                    x[3] = 0;
                }
            }
            return Ok(true);
        }
        if !vex.l || scalar(i.op, self.pfx()) {
            return self.sse_op();
        }
        if !in_lane(i.op, self.pfx()) {
            return Ok(false);
        }
        self.wide = true;
        let first = self.sse_op();
        let result = match first {
            Ok(true) => {
                self.half = 1;
                self.sse_op()
            }
            other => other,
        };
        self.half = 0;
        self.wide = false;
        result
    }

    fn sse_op(&mut self) -> Res<bool> {
        let i = self.insn;
        let p = self.pfx();
        let d = i.reg;
        let op = i.op;
        let mmx = p == Pfx::None && matches!(op, 0x60..=0x7f | 0xd0..=0xff) && !matches!(op, 0x77);
        if mmx {
            return Ok(false);
        }
        match (p, op) {
            (Pfx::None | Pfx::P66, 0x10) => {
                let v = self.src128(false)?;
                self.xset(d, v);
            }
            (Pfx::None | Pfx::P66, 0x11) => {
                let v = self.xget(d);
                match self.rm() {
                    Loc::Reg(r) => self.xset(r, v),
                    Loc::Mem(a) => self.store128(a, v, false)?,
                }
            }
            (Pfx::F3 | Pfx::F2, 0x10) => {
                let n = if p == Pfx::F3 { 4 } else { 8 };
                match self.rm() {
                    Loc::Reg(r) => {
                        let v = self.xget(r) as u64;
                        self.set_low(d, n, v);
                    }
                    Loc::Mem(a) => {
                        let v = self.load(a, n)?;
                        self.xset(d, v as u128);
                    }
                }
            }
            (Pfx::F3 | Pfx::F2, 0x11) => {
                let n = if p == Pfx::F3 { 4 } else { 8 };
                let v = self.xget(d) as u64;
                match self.rm() {
                    Loc::Reg(r) => self.set_low(r, n, v),
                    Loc::Mem(a) => self.store(a, n, v)?,
                }
            }
            (Pfx::None | Pfx::P66, 0x12 | 0x16) => {
                let high = op == 0x16;
                let cur = self.src1();
                let v = match self.rm() {
                    Loc::Reg(r) if p == Pfx::None => {
                        let s = self.xget(r);
                        if high { s << 64 } else { s >> 64 }
                    }
                    Loc::Reg(_) => return Err(CpuFault::InvalidOpcode),
                    Loc::Mem(a) => {
                        let m = self.load(a, 8)? as u128;
                        if high { m << 64 } else { m }
                    }
                };
                let keep = if high {
                    u64::MAX as u128
                } else {
                    (u64::MAX as u128) << 64
                };
                self.xset(d, (cur & keep) | (v & !keep));
            }
            (Pfx::None | Pfx::P66, 0x13 | 0x17) => {
                let Loc::Mem(a) = self.rm() else {
                    return Err(CpuFault::InvalidOpcode);
                };
                let v = self.xget(d);
                self.store(
                    a,
                    8,
                    if op == 0x17 {
                        (v >> 64) as u64
                    } else {
                        v as u64
                    },
                )?;
            }
            (Pfx::F2, 0x12) => {
                let v = self.src_low(8)? as u128;
                self.xset(d, v | v << 64);
            }
            (Pfx::F3, 0x12 | 0x16) => {
                let s = lanes::<4>(self.src128(true)?);
                let (x, y) = if op == 0x12 {
                    (s[0], s[2])
                } else {
                    (s[1], s[3])
                };
                self.xset(d, join::<4>([x, x, y, y]));
            }
            (Pfx::None, 0x14 | 0x15) => {
                let (a, b) = (lanes::<4>(self.src1()), lanes::<4>(self.src128(true)?));
                let r = if op == 0x14 {
                    [a[0], b[0], a[1], b[1]]
                } else {
                    [a[2], b[2], a[3], b[3]]
                };
                self.xset(d, join::<4>(r));
            }
            (Pfx::P66, 0x14 | 0x15) => {
                let (a, b) = (lanes::<2>(self.src1()), lanes::<2>(self.src128(true)?));
                let r = if op == 0x14 {
                    [a[0], b[0]]
                } else {
                    [a[1], b[1]]
                };
                self.xset(d, join::<2>(r));
            }
            (Pfx::None | Pfx::P66, 0x28) => {
                let v = self.src128(true)?;
                self.xset(d, v);
            }
            (Pfx::None | Pfx::P66, 0x29 | 0x2b) | (Pfx::P66, 0x7f | 0xe7) => {
                let v = self.xget(d);
                match self.rm() {
                    Loc::Reg(r) if op == 0x29 || op == 0x7f => self.xset(r, v),
                    Loc::Reg(_) => return Err(CpuFault::InvalidOpcode),
                    Loc::Mem(a) => self.store128(a, v, true)?,
                }
            }
            (Pfx::F3, 0x7f) => {
                let v = self.xget(d);
                match self.rm() {
                    Loc::Reg(r) => self.xset(r, v),
                    Loc::Mem(a) => self.store128(a, v, false)?,
                }
            }
            (Pfx::F3 | Pfx::F2, 0x2a) => {
                let n = if i.rex_w() { 8 } else { 4 };
                let v = alu::sext(self.read(self.rm(), n)?, n) as i64;
                if p == Pfx::F3 {
                    self.set_low(d, 4, (v as f32).to_bits() as u64);
                } else {
                    self.set_low(d, 8, (v as f64).to_bits());
                }
            }
            (Pfx::F3 | Pfx::F2, 0x2c | 0x2d) => {
                let f = if p == Pfx::F3 {
                    f32::from_bits(self.src_low(4)? as u32) as f64
                } else {
                    f64::from_bits(self.src_low(8)?)
                };
                let bits = if i.rex_w() { 64 } else { 32 };
                let v = to_int(f, (self.cpu.mxcsr >> 13) & 3, op == 0x2c, bits);
                self.set(d, bits as u8 / 8, v);
            }
            (Pfx::None | Pfx::P66, 0x2e | 0x2f) => {
                let (a, b) = if p == Pfx::None {
                    let a = f32::from_bits(self.xget(d) as u32) as f64;
                    (a, f32::from_bits(self.src_low(4)? as u32) as f64)
                } else {
                    (
                        f64::from_bits(self.xget(d) as u64),
                        f64::from_bits(self.src_low(8)?),
                    )
                };
                let f = if a.is_nan() || b.is_nan() {
                    ZF | PF | CF
                } else if a < b {
                    CF
                } else if a == b {
                    ZF
                } else {
                    0
                };
                self.cpu.set_arith(f);
            }
            (Pfx::None | Pfx::P66, 0x50) => {
                let v = self.xget(i.rm);
                let m = if p == Pfx::None {
                    (0..4).fold(0, |acc, k| acc | ((v >> (32 * k + 31)) as u64 & 1) << k)
                } else {
                    (0..2).fold(0, |acc, k| acc | ((v >> (64 * k + 63)) as u64 & 1) << k)
                };
                self.set(d, 4, m);
            }
            (_, 0x51 | 0x58 | 0x59 | 0x5c..=0x5f) => self.float_arith(p)?,
            (_, 0x54..=0x57) if matches!(p, Pfx::None | Pfx::P66) => {
                let (a, b) = (self.src1(), self.src128(true)?);
                let r = match op {
                    0x54 => a & b,
                    0x55 => !a & b,
                    0x56 => a | b,
                    _ => a ^ b,
                };
                self.xset(d, r);
            }
            (_, 0x5a) => match p {
                Pfx::None => {
                    let s = lanes::<4>(self.src_low(8)? as u128);
                    self.xset(
                        d,
                        join::<2>([f32_to_f64(s[0] as u32), f32_to_f64(s[1] as u32)]),
                    );
                }
                Pfx::P66 => {
                    let s = lanes::<2>(self.src128(true)?);
                    self.xset(
                        d,
                        join::<4>([f64_to_f32(s[0]) as u64, f64_to_f32(s[1]) as u64, 0, 0]),
                    );
                }
                Pfx::F3 => {
                    let v = f32_to_f64(self.src_low(4)? as u32);
                    self.set_low(d, 8, v);
                }
                Pfx::F2 => {
                    let v = f64_to_f32(self.src_low(8)?);
                    self.set_low(d, 4, v as u64);
                }
            },
            (_, 0x5b) if p != Pfx::F2 => {
                let s = lanes::<4>(self.src128(true)?);
                let rc = (self.cpu.mxcsr >> 13) & 3;
                let r = s.map(|x| match p {
                    Pfx::None => (sx(x, 32) as f32).to_bits() as u64,
                    _ => to_int(f32::from_bits(x as u32) as f64, rc, p == Pfx::F3, 32),
                });
                self.xset(d, join::<4>(r));
            }
            (Pfx::P66 | Pfx::F3 | Pfx::F2, 0xe6) => {
                let rc = (self.cpu.mxcsr >> 13) & 3;
                if p == Pfx::F3 {
                    let s = lanes::<4>(self.src_low(8)? as u128);
                    self.xset(
                        d,
                        join::<2>([
                            (sx(s[0], 32) as f64).to_bits(),
                            (sx(s[1], 32) as f64).to_bits(),
                        ]),
                    );
                } else {
                    let s = lanes::<2>(self.src128(true)?);
                    let c = |x| to_int(f64::from_bits(x), rc, p == Pfx::P66, 32);
                    self.xset(d, join::<4>([c(s[0]), c(s[1]), 0, 0]));
                }
            }
            (Pfx::P66, 0x60..=0x6d) => self.unpack_pack()?,
            (Pfx::P66, 0x6e) => {
                let n = if i.rex_w() { 8 } else { 4 };
                let v = self.read(self.rm(), n)?;
                self.xset(d, v as u128);
            }
            (Pfx::P66, 0x7e) => {
                let n = if i.rex_w() { 8 } else { 4 };
                let v = self.xget(d) as u64 & alu::mask(n);
                self.write(self.rm(), n, v)?;
            }
            (Pfx::F3, 0x7e) => {
                let v = self.src_low(8)?;
                self.xset(d, v as u128);
            }
            (Pfx::P66, 0xd6) => {
                let v = self.xget(d) as u64;
                match self.rm() {
                    Loc::Reg(r) => self.xset(r, v as u128),
                    Loc::Mem(a) => self.store(a, 8, v)?,
                }
            }
            (Pfx::P66 | Pfx::F3, 0x6f) => {
                let v = self.src128(p == Pfx::P66)?;
                self.xset(d, v);
            }
            (Pfx::P66 | Pfx::F3 | Pfx::F2, 0x70) => {
                let s = self.src128(true)?;
                let imm = self.imm() as u8;
                let sel = |l: &[u64], k: usize| l[((imm >> (2 * k)) & 3) as usize];
                let r = match p {
                    Pfx::P66 => {
                        let l = lanes::<4>(s);
                        join::<4>(core::array::from_fn(|k| sel(&l, k)))
                    }
                    Pfx::F3 => {
                        let l = lanes::<8>(s);
                        let hi = [l[4], l[5], l[6], l[7]];
                        let mut o = l;
                        for k in 0..4 {
                            o[4 + k] = sel(&hi, k);
                        }
                        join::<8>(o)
                    }
                    _ => {
                        let l = lanes::<8>(s);
                        let lo = [l[0], l[1], l[2], l[3]];
                        let mut o = l;
                        for (k, slot) in o.iter_mut().take(4).enumerate() {
                            *slot = sel(&lo, k);
                        }
                        join::<8>(o)
                    }
                };
                self.xset(d, r);
            }
            (Pfx::P66, 0x71..=0x73) => {
                let count = self.imm() as u8 as u64;
                let v = self.xget(i.rm);
                let r = i.vex.map_or(i.rm, |x| x.vvvv);
                let res = match (op, i.ext()) {
                    (0x71, 2) => self.shift_lanes::<8>(v, count, 0),
                    (0x71, 4) => self.shift_lanes::<8>(v, count, 1),
                    (0x71, 6) => self.shift_lanes::<8>(v, count, 2),
                    (0x72, 2) => self.shift_lanes::<4>(v, count, 0),
                    (0x72, 4) => self.shift_lanes::<4>(v, count, 1),
                    (0x72, 6) => self.shift_lanes::<4>(v, count, 2),
                    (0x73, 2) => self.shift_lanes::<2>(v, count, 0),
                    (0x73, 6) => self.shift_lanes::<2>(v, count, 2),
                    (0x73, 3) => {
                        if count > 15 {
                            0
                        } else {
                            v >> (8 * count)
                        }
                    }
                    (0x73, 7) => {
                        if count > 15 {
                            0
                        } else {
                            v << (8 * count)
                        }
                    }
                    _ => return Err(CpuFault::InvalidOpcode),
                };
                self.xset(r, res);
            }
            (Pfx::P66, 0xd1 | 0xd2 | 0xd3 | 0xe1 | 0xe2 | 0xf1 | 0xf2 | 0xf3) => {
                let half = std::mem::replace(&mut self.half, 0);
                let count = self.src128(true);
                self.half = half;
                let count = count? as u64;
                let v = self.src1();
                let kind = match op & 0xf0 {
                    0xd0 => 0,
                    0xe0 => 1,
                    _ => 2,
                };
                let res = match op & 0xf {
                    1 => self.shift_lanes::<8>(v, count, kind),
                    2 => self.shift_lanes::<4>(v, count, kind),
                    _ => self.shift_lanes::<2>(v, count, kind),
                };
                self.xset(d, res);
            }
            (
                Pfx::P66,
                0x74..=0x76
                | 0xd4
                | 0xd5
                | 0xd8..=0xe0
                | 0xe3..=0xe5
                | 0xe8..=0xef
                | 0xf4..=0xf6
                | 0xf8..=0xfe,
            ) => {
                let (a, b) = (self.src1(), self.src128(true)?);
                self.xset(d, int_op(op, a, b));
            }
            (Pfx::P66, 0xd7) => {
                let v = self.xget(i.rm);
                let m = (0..16).fold(0u64, |acc, k| acc | ((v >> (8 * k + 7)) as u64 & 1) << k);
                self.set(d, 4, m);
            }
            (Pfx::P66, 0xc4) => {
                let v = self.read(self.rm(), 2)?;
                let k = (self.imm() & 7) as usize;
                let mut l = lanes::<8>(self.src1());
                l[k] = v;
                self.xset(d, join::<8>(l));
            }
            (Pfx::P66, 0xc5) => {
                let k = (self.imm() & 7) as usize;
                let v = lanes::<8>(self.xget(i.rm))[k];
                self.set(d, 4, v);
            }
            (Pfx::None, 0xc6) => {
                let (a, b) = (lanes::<4>(self.src1()), lanes::<4>(self.src128(true)?));
                let imm = self.imm() as u8;
                let s = |l: &[u64; 4], k: u8| l[((imm >> (2 * k)) & 3) as usize];
                self.xset(d, join::<4>([s(&a, 0), s(&a, 1), s(&b, 2), s(&b, 3)]));
            }
            (Pfx::P66, 0xc6) => {
                let (a, b) = (lanes::<2>(self.src1()), lanes::<2>(self.src128(true)?));
                let imm = self.imm() >> (2 * self.half);
                self.xset(
                    d,
                    join::<2>([a[(imm & 1) as usize], b[((imm >> 1) & 1) as usize]]),
                );
            }
            (_, 0xc2) => {
                let pred = self.imm() as u8;
                let a = self.src1();
                match p {
                    Pfx::None | Pfx::P66 => {
                        let b = self.src128(true)?;
                        let r = if p == Pfx::None {
                            map2::<4>(a, b, |x, y| {
                                let t = cmp_pred(
                                    f32::from_bits(x as u32) as f64,
                                    f32::from_bits(y as u32) as f64,
                                    pred,
                                );
                                if t { u32::MAX as u64 } else { 0 }
                            })
                        } else {
                            map2::<2>(a, b, |x, y| {
                                if cmp_pred(f64::from_bits(x), f64::from_bits(y), pred) {
                                    u64::MAX
                                } else {
                                    0
                                }
                            })
                        };
                        self.xset(d, r);
                    }
                    Pfx::F3 => {
                        let b = self.src_low(4)?;
                        let t = cmp_pred(
                            f32::from_bits(a as u32) as f64,
                            f32::from_bits(b as u32) as f64,
                            pred,
                        );
                        self.set_low(d, 4, if t { u32::MAX as u64 } else { 0 });
                    }
                    Pfx::F2 => {
                        let b = self.src_low(8)?;
                        let t = cmp_pred(f64::from_bits(a as u64), f64::from_bits(b), pred);
                        self.set_low(d, 8, if t { u64::MAX } else { 0 });
                    }
                }
            }
            (Pfx::None, 0xc3) => {
                let Loc::Mem(a) = self.rm() else {
                    return Err(CpuFault::InvalidOpcode);
                };
                let n = if i.rex_w() { 8 } else { 4 };
                self.store(a, n, self.get(d, n))?;
            }
            (Pfx::None, 0xae) => match (i.ext(), self.rm()) {
                (2, Loc::Mem(a)) => {
                    let v = self.load(a, 4)? as u32;
                    if v & 0xffff_0000 != 0 {
                        return Err(CpuFault::GeneralProtection);
                    }
                    self.cpu.mxcsr = v;
                }
                (3, Loc::Mem(a)) => self.store(a, 4, self.cpu.mxcsr as u64)?,
                (5..=7, Loc::Reg(_)) | (7, Loc::Mem(_)) => {}
                _ => return Ok(false),
            },
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Lane shifts: `kind` 0 logical right, 1 arithmetic right, 2 left.
    fn shift_lanes<const N: usize>(&self, v: u128, count: u64, kind: u8) -> u128 {
        let bits = (128 / N) as u64;
        let l = lanes::<N>(v);
        join::<N>(l.map(|x| match kind {
            0 => {
                if count >= bits {
                    0
                } else {
                    x >> count
                }
            }
            1 => (sx(x, bits as u32) >> count.min(bits - 1)) as u64,
            _ => {
                if count >= bits {
                    0
                } else {
                    x << count
                }
            }
        }))
    }

    fn float_arith(&mut self, p: Pfx) -> Res<()> {
        let op = self.insn.op;
        let d = self.insn.reg;
        let daz = self.cpu.mxcsr & MXCSR_DAZ != 0;
        let ftz = self.cpu.mxcsr & MXCSR_FTZ != 0;
        let f32f = |a: u32, b: u32| -> u32 {
            let zero_in = |x: u32| {
                if daz && x & 0x7f80_0000 == 0 {
                    x & 0x8000_0000
                } else {
                    x
                }
            };
            let (a, b) = (zero_in(a), zero_in(b));
            let r = match op {
                0x51 => sqrt32(b),
                0x58 => f32op(a, b, |x, y| x + y),
                0x59 => f32op(a, b, |x, y| x * y),
                0x5c => f32op(a, b, |x, y| x - y),
                0x5d => minmax32(a, b, false),
                0x5e => f32op(a, b, |x, y| x / y),
                _ => minmax32(a, b, true),
            };
            if ftz && r & 0x7f80_0000 == 0 {
                r & 0x8000_0000
            } else {
                r
            }
        };
        let f64f = |a: u64, b: u64| -> u64 {
            let exp = 0x7ff0_0000_0000_0000u64;
            let sign = 1u64 << 63;
            let zero_in = |x: u64| if daz && x & exp == 0 { x & sign } else { x };
            let (a, b) = (zero_in(a), zero_in(b));
            let r = match op {
                0x51 => sqrt64(b),
                0x58 => f64op(a, b, |x, y| x + y),
                0x59 => f64op(a, b, |x, y| x * y),
                0x5c => f64op(a, b, |x, y| x - y),
                0x5d => minmax64(a, b, false),
                0x5e => f64op(a, b, |x, y| x / y),
                _ => minmax64(a, b, true),
            };
            if ftz && r & exp == 0 { r & sign } else { r }
        };
        let a = self.src1();
        match p {
            Pfx::None => {
                let b = self.src128(true)?;
                self.xset(d, map2::<4>(a, b, |x, y| f32f(x as u32, y as u32) as u64));
            }
            Pfx::P66 => {
                let b = self.src128(true)?;
                self.xset(d, map2::<2>(a, b, f64f));
            }
            Pfx::F3 => {
                let b = self.src_low(4)? as u32;
                self.set_low(d, 4, f32f(a as u32, b) as u64);
            }
            Pfx::F2 => {
                let b = self.src_low(8)?;
                self.set_low(d, 8, f64f(a as u64, b));
            }
        }
        Ok(())
    }

    fn unpack_pack(&mut self) -> Res<()> {
        let op = self.insn.op;
        let d = self.insn.reg;
        let (a, b) = (self.src1(), self.src128(true)?);
        let unpack = |n: usize, high: bool| -> u128 {
            let bits = 128 / n;
            let m = if bits == 64 {
                u64::MAX as u128
            } else {
                (1u128 << bits) - 1
            };
            let base = if high { n / 2 } else { 0 };
            (0..n / 2).fold(0u128, |acc, k| {
                let x = (a >> ((base + k) * bits)) & m;
                let y = (b >> ((base + k) * bits)) & m;
                acc | x << (2 * k * bits) | y << ((2 * k + 1) * bits)
            })
        };
        let r = match op {
            0x60 => unpack(16, false),
            0x61 => unpack(8, false),
            0x62 => unpack(4, false),
            0x68 => unpack(16, true),
            0x69 => unpack(8, true),
            0x6a => unpack(4, true),
            0x6c => unpack(2, false),
            0x6d => unpack(2, true),
            0x63 | 0x67 => {
                let (la, lb) = (lanes::<8>(a), lanes::<8>(b));
                let s = |x: u64| {
                    if op == 0x63 {
                        sat_s(sx(x, 16), 8)
                    } else {
                        sat_u(sx(x, 16), 8)
                    }
                };
                let mut o = [0u64; 16];
                for k in 0..8 {
                    o[k] = s(la[k]);
                    o[8 + k] = s(lb[k]);
                }
                join::<16>(o)
            }
            0x6b => {
                let (la, lb) = (lanes::<4>(a), lanes::<4>(b));
                let mut o = [0u64; 8];
                for k in 0..4 {
                    o[k] = sat_s(sx(la[k], 32), 16);
                    o[4 + k] = sat_s(sx(lb[k], 32), 16);
                }
                join::<8>(o)
            }
            0x64 => map2::<16>(a, b, |x, y| if sx(x, 8) > sx(y, 8) { 0xff } else { 0 }),
            0x65 => map2::<8>(a, b, |x, y| if sx(x, 16) > sx(y, 16) { 0xffff } else { 0 }),
            _ => map2::<4>(a, b, |x, y| {
                if sx(x, 32) > sx(y, 32) {
                    0xffff_ffff
                } else {
                    0
                }
            }),
        };
        self.xset(d, r);
        Ok(())
    }
}

/// Whether `0F op` is a scalar operation, for which VEX.L is ignored.
fn scalar(op: u8, p: Pfx) -> bool {
    match p {
        Pfx::F3 | Pfx::F2 => matches!(
            op,
            0x10 | 0x11 | 0x2a | 0x2c | 0x2d | 0x51 | 0x58 | 0x59 | 0x5a | 0x5c..=0x5f | 0xc2
        ),
        _ => matches!(op, 0x2e | 0x2f),
    }
}

/// Whether a 256-bit VEX form of `0F op` works on each 128-bit half on its own.
fn in_lane(op: u8, p: Pfx) -> bool {
    let packed = matches!(p, Pfx::None | Pfx::P66);
    match op {
        0x10
        | 0x11
        | 0x14
        | 0x15
        | 0x28
        | 0x29
        | 0x2b
        | 0x51
        | 0x54..=0x59
        | 0x5c..=0x5f
        | 0xc2
        | 0xc6 => packed,
        0x12 | 0x16 => matches!(p, Pfx::F3 | Pfx::F2),
        0x5b => p != Pfx::F2,
        0x6f | 0x7f => matches!(p, Pfx::P66 | Pfx::F3),
        0x70 => p != Pfx::None,
        0x60..=0x6d
        | 0x71..=0x76
        | 0xd1..=0xd5
        | 0xd8..=0xe5
        | 0xe7..=0xef
        | 0xf1..=0xf6
        | 0xf8..=0xfe => p == Pfx::P66,
        _ => false,
    }
}

/// The two-operand integer operations of the `66 0F xx` map.
fn int_op(op: u8, a: u128, b: u128) -> u128 {
    match op {
        0x74 => map2::<16>(a, b, |x, y| if x == y { 0xff } else { 0 }),
        0x75 => map2::<8>(a, b, |x, y| if x == y { 0xffff } else { 0 }),
        0x76 => map2::<4>(a, b, |x, y| if x == y { 0xffff_ffff } else { 0 }),
        0xd4 => map2::<2>(a, b, |x, y| x.wrapping_add(y)),
        0xd5 => map2::<8>(a, b, |x, y| (sx(x, 16) * sx(y, 16)) as u64),
        0xd8 => map2::<16>(a, b, |x, y| x.saturating_sub(y)),
        0xd9 => map2::<8>(a, b, |x, y| x.saturating_sub(y)),
        0xda => map2::<16>(a, b, |x, y| x.min(y)),
        0xdb => a & b,
        0xdc => map2::<16>(a, b, |x, y| (x + y).min(0xff)),
        0xdd => map2::<8>(a, b, |x, y| (x + y).min(0xffff)),
        0xde => map2::<16>(a, b, |x, y| x.max(y)),
        0xdf => !a & b,
        0xe0 => map2::<16>(a, b, |x, y| (x + y + 1) >> 1),
        0xe3 => map2::<8>(a, b, |x, y| (x + y + 1) >> 1),
        0xe4 => map2::<8>(a, b, |x, y| (x * y) >> 16),
        0xe5 => map2::<8>(a, b, |x, y| ((sx(x, 16) * sx(y, 16)) >> 16) as u64),
        0xe8 => map2::<16>(a, b, |x, y| sat_s(sx(x, 8) - sx(y, 8), 8)),
        0xe9 => map2::<8>(a, b, |x, y| sat_s(sx(x, 16) - sx(y, 16), 16)),
        0xea => map2::<8>(a, b, |x, y| sx(x, 16).min(sx(y, 16)) as u64),
        0xeb => a | b,
        0xec => map2::<16>(a, b, |x, y| sat_s(sx(x, 8) + sx(y, 8), 8)),
        0xed => map2::<8>(a, b, |x, y| sat_s(sx(x, 16) + sx(y, 16), 16)),
        0xee => map2::<8>(a, b, |x, y| sx(x, 16).max(sx(y, 16)) as u64),
        0xef => a ^ b,
        0xf4 => map2::<2>(a, b, |x, y| (x & 0xffff_ffff) * (y & 0xffff_ffff)),
        0xf5 => {
            let (la, lb) = (lanes::<8>(a), lanes::<8>(b));
            join::<4>(core::array::from_fn(|k| {
                let s = sx(la[2 * k], 16) * sx(lb[2 * k], 16)
                    + sx(la[2 * k + 1], 16) * sx(lb[2 * k + 1], 16);
                s as u64
            }))
        }
        0xf6 => {
            let (la, lb) = (lanes::<16>(a), lanes::<16>(b));
            let sad = |r: std::ops::Range<usize>| r.map(|k| la[k].abs_diff(lb[k])).sum::<u64>();
            join::<2>([sad(0..8), sad(8..16)])
        }
        0xf8 => map2::<16>(a, b, |x, y| x.wrapping_sub(y)),
        0xf9 => map2::<8>(a, b, |x, y| x.wrapping_sub(y)),
        0xfa => map2::<4>(a, b, |x, y| x.wrapping_sub(y)),
        0xfb => map2::<2>(a, b, |x, y| x.wrapping_sub(y)),
        0xfc => map2::<16>(a, b, |x, y| x.wrapping_add(y)),
        0xfd => map2::<8>(a, b, |x, y| x.wrapping_add(y)),
        _ => map2::<4>(a, b, |x, y| x.wrapping_add(y)),
    }
}
