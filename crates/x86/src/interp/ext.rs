//! The `0F38` and `0F3A` maps: SSSE3, SSE4.1, SSE4.2's CRC32, MOVBE, the
//! AVX/AVX2-only vector instructions, FMA, and BMI1/BMI2 on general
//! registers. Semantics follow the Intel SDM, volume 2.

use super::sse::{Pfx, join, lanes, map2, sat_s, sat_u, sx};
use super::*;

/// What runs when the instruction is not one we implement.
fn none() -> Res<bool> {
    Ok(false)
}

fn sign_bit(v: u64, bits: u32) -> bool {
    v >> (bits - 1) & 1 != 0
}

/// IEEE half precision to single precision.
fn f16_to_f32(h: u16) -> u32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    match exp {
        0 if frac == 0 => sign,
        0 => {
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e as u32) << 23 | (f & 0x3ff) << 13
        }
        0x1f => sign | 0x7f80_0000 | frac << 13 | if frac != 0 { 0x0040_0000 } else { 0 },
        _ => sign | (exp + 127 - 15) << 23 | frac << 13,
    }
}

/// Single precision to half precision with rounding `rc` (0 nearest-even,
/// 1 down, 2 up, 3 toward zero).
fn f32_to_f16(x: u32, rc: u32) -> u16 {
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let frac = x & 0x7f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if frac != 0 {
                0x200 | (frac >> 13) as u16
            } else {
                0
            };
    }
    let negative = sign != 0;
    // Value = mantissa * 2^(e - 23), mantissa including the hidden bit.
    let (mant, e) = if exp == 0 {
        (frac, -126)
    } else {
        (frac | 0x80_0000, exp - 127)
    };
    if mant == 0 {
        return sign;
    }
    let half_exp = e + 15;
    let shift = if half_exp >= 1 {
        13
    } else {
        13 + (1 - half_exp)
    } as u32;
    let round = |m: u64, shift: u32| -> u64 {
        if shift >= 64 {
            let lost = m != 0;
            return match rc {
                1 if negative && lost => 1,
                2 if !negative && lost => 1,
                _ => 0,
            };
        }
        let kept = m >> shift;
        let rem = m & ((1u64 << shift) - 1);
        let halfway = 1u64 << (shift - 1);
        let up = match rc {
            0 => rem > halfway || (rem == halfway && kept & 1 == 1),
            1 => negative && rem != 0,
            2 => !negative && rem != 0,
            _ => false,
        };
        kept + up as u64
    };
    let m = round(mant as u64, shift);
    if half_exp >= 1 {
        let (mut m, mut he) = (m, half_exp);
        if m >= 0x800 {
            m >>= 1;
            he += 1;
        }
        if he >= 0x1f {
            let to_inf = match rc {
                0 => true,
                1 => negative,
                2 => !negative,
                _ => false,
            };
            return sign | if to_inf { 0x7c00 } else { 0x7bff };
        }
        sign | (he as u16) << 10 | (m as u16 & 0x3ff)
    } else {
        sign | m as u16
    }
}

fn fma32(a: f32, b: f32, c: f32) -> u32 {
    for v in [a, b, c] {
        if v.is_nan() {
            return v.to_bits() | 0x0040_0000;
        }
    }
    let r = a.mul_add(b, c);
    if r.is_nan() { 0xffc0_0000 } else { r.to_bits() }
}

fn fma64(a: f64, b: f64, c: f64) -> u64 {
    for v in [a, b, c] {
        if v.is_nan() {
            return v.to_bits() | 0x0008_0000_0000_0000;
        }
    }
    let r = a.mul_add(b, c);
    if r.is_nan() {
        0xfff8_0000_0000_0000
    } else {
        r.to_bits()
    }
}

/// CRC-32C (Castagnoli), as SSE4.2's CRC32 computes it, over `bytes` of `v`.
fn crc32c(mut crc: u32, v: u64, bytes: u8) -> u32 {
    for i in 0..bytes {
        crc ^= (v >> (8 * i)) as u8 as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x82f6_3b78
            } else {
                crc >> 1
            };
        }
    }
    crc
}

fn round_f(v: f64, mode: u32) -> f64 {
    match mode {
        0 => v.round_ties_even(),
        1 => v.floor(),
        2 => v.ceil(),
        _ => v.trunc(),
    }
}

impl Ctx<'_> {
    /// Both 128-bit halves of YMM register `r`.
    fn yget(&self, r: u8) -> [u128; 2] {
        let x = self.cpu.xmm[r as usize];
        [
            x[0] as u128 | (x[1] as u128) << 64,
            x[2] as u128 | (x[3] as u128) << 64,
        ]
    }

    /// Writes YMM register `r`; a 128-bit result clears the upper half.
    fn yset(&mut self, r: u8, v: [u128; 2]) {
        let x = &mut self.cpu.xmm[r as usize];
        *x = [
            v[0] as u64,
            (v[0] >> 64) as u64,
            v[1] as u64,
            (v[1] >> 64) as u64,
        ];
    }

    fn wide256(&self) -> bool {
        self.insn.vex.is_some_and(|v| v.l)
    }

    fn vvvv(&self) -> u8 {
        self.insn.vex.map_or(0, |v| v.vvvv)
    }

    /// The `xmm/m128` or `ymm/m256` source, whole.
    fn ysrc(&self) -> Res<[u128; 2]> {
        match self.rm() {
            Loc::Reg(r) => Ok(self.yget(r)),
            Loc::Mem(a) => {
                let lo = self.load128(a, false)?;
                let hi = if self.wide256() {
                    self.mem_u128(a + 16)?
                } else {
                    0
                };
                Ok([lo, hi])
            }
        }
    }

    fn mem_u128(&self, addr: u64) -> Res<u128> {
        let mut b = [0u8; 16];
        self.mem.read(addr, &mut b)?;
        Ok(u128::from_le_bytes(b))
    }

    /// Writes the destination of a cross-lane operation: 256 bits with
    /// VEX.L, else 128 (clearing the upper half for VEX, keeping it for SSE).
    fn put(&mut self, r: u8, v: [u128; 2]) {
        if self.wide256() || self.insn.vex.is_some() {
            let hi = if self.wide256() { v[1] } else { 0 };
            self.yset(r, [v[0], hi]);
        } else {
            self.xset(r, v[0]);
        }
    }

    /// Runs `i` if it is in the `0F38` or `0F3A` map and we implement it.
    pub(super) fn ext(&mut self) -> Res<bool> {
        let i = self.insn;
        if i.vex.is_some() && self.bmi()? {
            return Ok(true);
        }
        if i.vex.is_none() && i.map == Map::M0F38 && matches!(i.op, 0xf0 | 0xf1) {
            return self.movbe_crc32();
        }
        if self.pfx() != Pfx::P66 {
            return none();
        }
        if i.map == Map::M0F38 && i.op == 0x17 {
            return self.ptest();
        }
        if self.cross_lane()? {
            return Ok(true);
        }
        if self.wide256() {
            if !self.in_lane_ext() {
                return none();
            }
            return self.by_halves(Self::ext_op);
        }
        self.ext_op()
    }

    fn in_lane_ext(&self) -> bool {
        let op = self.insn.op;
        match self.insn.map {
            Map::M0F38 => {
                matches!(
                    op,
                    0x00..=0x0e | 0x1c..=0x1e | 0x28..=0x2b | 0x37..=0x40 | 0x45..=0x47 | 0x96..=0x98 | 0x9a | 0x9c | 0x9e
                        | 0xa6..=0xa8 | 0xaa | 0xac | 0xae | 0xb6..=0xb8 | 0xba | 0xbc | 0xbe
                ) && op != 0x0f
            }
            _ => {
                matches!(op, 0x02 | 0x04 | 0x05 | 0x08 | 0x09 | 0x0c..=0x0f | 0x40 | 0x42 | 0x4a..=0x4c)
            }
        }
    }

    /// In-lane vector operations, on the current 128-bit half.
    fn ext_op(&mut self) -> Res<bool> {
        let i = self.insn;
        let d = i.reg;
        let op = i.op;
        let vex = i.vex.is_some();
        let w = i.rex_w();
        let half = self.half as u32;
        if i.map == Map::M0F3A && (0x60..=0x63).contains(&op) {
            return self.string_compare();
        }
        if i.map == Map::M0F38 {
            if (0x96..=0xbf).contains(&op) {
                return self.fma();
            }
            if op == 0x2a {
                let Loc::Mem(addr) = self.rm() else {
                    return Err(CpuFault::InvalidOpcode);
                };
                let v = self.load128(addr, !vex)?;
                self.xset(d, v);
                return Ok(true);
            }
            let (a, b) = (self.src1(), self.src128(!vex)?);
            let r = match op {
                0x00 => {
                    let (la, lb) = (lanes::<16>(a), lanes::<16>(b));
                    join::<16>(core::array::from_fn(|k| {
                        if lb[k] & 0x80 != 0 {
                            0
                        } else {
                            la[(lb[k] & 15) as usize]
                        }
                    }))
                }
                0x01..=0x03 | 0x05..=0x07 => {
                    let sub = op >= 5;
                    let words = op & 3 != 2;
                    let sat = op & 3 == 3;
                    let pair = |x: u64, y: u64, bits: u32| -> u64 {
                        let (x, y) = (sx(x, bits), sx(y, bits));
                        let r = if sub { x - y } else { x + y };
                        if sat { sat_s(r, bits) } else { r as u64 }
                    };
                    if words {
                        let (la, lb) = (lanes::<8>(a), lanes::<8>(b));
                        join::<8>(core::array::from_fn(|k| {
                            let s = if k < 4 { &la } else { &lb };
                            let j = (k % 4) * 2;
                            pair(s[j], s[j + 1], 16)
                        }))
                    } else {
                        let (la, lb) = (lanes::<4>(a), lanes::<4>(b));
                        join::<4>(core::array::from_fn(|k| {
                            let s = if k < 2 { &la } else { &lb };
                            let j = (k % 2) * 2;
                            pair(s[j], s[j + 1], 32)
                        }))
                    }
                }
                0x04 => {
                    let (la, lb) = (lanes::<16>(a), lanes::<16>(b));
                    join::<8>(core::array::from_fn(|k| {
                        let p = |j: usize| la[j] as i64 * sx(lb[j], 8);
                        sat_s(p(2 * k) + p(2 * k + 1), 16)
                    }))
                }
                0x08..=0x0a => {
                    let bits = 8 << (op - 8);
                    let sign = |x: u64, y: u64| match sx(y, bits).signum() {
                        -1 => (-sx(x, bits)) as u64,
                        0 => 0,
                        _ => x,
                    };
                    match op {
                        0x08 => map2::<16>(a, b, sign),
                        0x09 => map2::<8>(a, b, sign),
                        _ => map2::<4>(a, b, sign),
                    }
                }
                0x0b => map2::<8>(a, b, |x, y| {
                    ((((sx(x, 16) * sx(y, 16)) >> 14) + 1) >> 1) as u64
                }),
                0x0c if vex => {
                    let (la, lb) = (lanes::<4>(a), lanes::<4>(b));
                    join::<4>(core::array::from_fn(|k| la[(lb[k] & 3) as usize]))
                }
                0x0d if vex => {
                    let (la, lb) = (lanes::<2>(a), lanes::<2>(b));
                    join::<2>(core::array::from_fn(|k| la[((lb[k] >> 1) & 1) as usize]))
                }
                0x0e if vex => {
                    return Ok(false);
                }
                0x1c => map2::<16>(0, b, |_, y| sx(y, 8).unsigned_abs()),
                0x1d => map2::<8>(0, b, |_, y| sx(y, 16).unsigned_abs()),
                0x1e => map2::<4>(0, b, |_, y| sx(y, 32).unsigned_abs()),
                0x28 => map2::<2>(a, b, |x, y| (sx(x, 32) * sx(y, 32)) as u64),
                0x29 => map2::<2>(a, b, |x, y| if x == y { u64::MAX } else { 0 }),
                0x2b => {
                    let (la, lb) = (lanes::<4>(a), lanes::<4>(b));
                    join::<8>(core::array::from_fn(|k| {
                        let v = if k < 4 { la[k] } else { lb[k - 4] };
                        sat_u(sx(v, 32), 16)
                    }))
                }
                0x37 => map2::<2>(
                    a,
                    b,
                    |x, y| if (x as i64) > (y as i64) { u64::MAX } else { 0 },
                ),
                0x38 => map2::<16>(a, b, |x, y| sx(x, 8).min(sx(y, 8)) as u64),
                0x39 => map2::<4>(a, b, |x, y| sx(x, 32).min(sx(y, 32)) as u64),
                0x3a => map2::<8>(a, b, |x, y| x.min(y)),
                0x3b => map2::<4>(a, b, |x, y| x.min(y)),
                0x3c => map2::<16>(a, b, |x, y| sx(x, 8).max(sx(y, 8)) as u64),
                0x3d => map2::<4>(a, b, |x, y| sx(x, 32).max(sx(y, 32)) as u64),
                0x3e => map2::<8>(a, b, |x, y| x.max(y)),
                0x3f => map2::<4>(a, b, |x, y| x.max(y)),
                0x40 => map2::<4>(a, b, |x, y| x.wrapping_mul(y)),
                0x45 | 0x47 if vex => {
                    let left = op == 0x47;
                    let f = |x: u64, n: u64, bits: u64| {
                        if n >= bits {
                            0
                        } else if left {
                            x << n
                        } else {
                            x >> n
                        }
                    };
                    if w {
                        map2::<2>(a, b, |x, n| f(x, n, 64))
                    } else {
                        map2::<4>(a, b, |x, n| f(x, n, 32))
                    }
                }
                0x46 if vex && !w => map2::<4>(a, b, |x, n| (sx(x, 32) >> n.min(31)) as u64),
                0x41 if !vex || !self.wide => {
                    let l = lanes::<8>(b);
                    let (idx, min) = l
                        .iter()
                        .enumerate()
                        .min_by_key(|&(k, &v)| (v, k))
                        .map(|(k, &v)| (k, v))
                        .unwrap();
                    min as u128 | (idx as u128) << 16
                }
                0x10 | 0x14 | 0x15 if !vex => {
                    let mask = self.xget(0);
                    let (bits, n) = match op {
                        0x10 => (8, 16),
                        0x14 => (32, 4),
                        _ => (64, 2),
                    };
                    let width = 128 / n;
                    let m = if width == 128 {
                        u128::MAX
                    } else {
                        (1u128 << width) - 1
                    };
                    (0..n).fold(0u128, |acc, k| {
                        let take_b = (mask >> (k * width + bits - 1)) & 1 != 0;
                        let src = if take_b { b } else { a };
                        acc | ((src >> (k * width)) & m) << (k * width)
                    })
                }
                0x17 => {
                    let (x, y) = if vex {
                        (self.yget(d), self.ysrc()?)
                    } else {
                        ([self.xget(d), 0], [b, 0])
                    };
                    let z = x[0] & y[0] == 0 && x[1] & y[1] == 0;
                    let c = !x[0] & y[0] == 0 && !x[1] & y[1] == 0;
                    self.cpu
                        .set_arith(if z { ZF } else { 0 } | if c { CF } else { 0 });
                    return Ok(true);
                }
                _ => return none(),
            };
            self.xset(d, r);
            return Ok(true);
        }

        let imm = self.imm() as u32;
        if matches!(op, 0x14..=0x17 | 0x20..=0x22) {
            return if self.wide {
                none()
            } else {
                self.extract_insert()
            };
        }
        if matches!(op, 0x0a | 0x0b) {
            return self.round_scalar(imm);
        }
        let (a, b) = (self.src1(), self.src128(!vex)?);
        let r = match op {
            0x02 if vex => {
                let sel = imm >> (4 * half);
                let (la, lb) = (lanes::<4>(a), lanes::<4>(b));
                join::<4>(core::array::from_fn(|k| {
                    if sel >> k & 1 != 0 { lb[k] } else { la[k] }
                }))
            }
            0x04 if vex => {
                let l = lanes::<4>(b);
                join::<4>(core::array::from_fn(|k| l[((imm >> (2 * k)) & 3) as usize]))
            }
            0x05 if vex => {
                let l = lanes::<2>(b);
                let sel = imm >> (2 * half);
                join::<2>(core::array::from_fn(|k| l[((sel >> k) & 1) as usize]))
            }
            0x08 | 0x09 => {
                let mode = if imm & 4 != 0 {
                    (self.cpu.mxcsr >> 13) & 3
                } else {
                    imm & 3
                };
                if op == 0x08 {
                    map2::<4>(0, b, |_, y| {
                        let f = f32::from_bits(y as u32);
                        if f.is_nan() {
                            (y as u32 | 0x0040_0000) as u64
                        } else {
                            (round_f(f as f64, mode) as f32).to_bits() as u64
                        }
                    })
                } else {
                    map2::<2>(0, b, |_, y| {
                        let f = f64::from_bits(y);
                        if f.is_nan() {
                            y | 0x0008_0000_0000_0000
                        } else {
                            round_f(f, mode).to_bits()
                        }
                    })
                }
            }
            0x0c => {
                let sel = imm >> (4 * half);
                let (la, lb) = (lanes::<4>(a), lanes::<4>(b));
                join::<4>(core::array::from_fn(|k| {
                    if sel >> k & 1 != 0 { lb[k] } else { la[k] }
                }))
            }
            0x0d => {
                let sel = imm >> (2 * half);
                let (la, lb) = (lanes::<2>(a), lanes::<2>(b));
                join::<2>(core::array::from_fn(|k| {
                    if sel >> k & 1 != 0 { lb[k] } else { la[k] }
                }))
            }
            0x0e => {
                let (la, lb) = (lanes::<8>(a), lanes::<8>(b));
                join::<8>(core::array::from_fn(|k| {
                    if imm >> k & 1 != 0 { lb[k] } else { la[k] }
                }))
            }
            0x0f => {
                let n = imm as usize;
                let wide = (a, b);
                let shifted = |src_hi: u128, src_lo: u128| -> u128 {
                    if n >= 32 {
                        0
                    } else if n >= 16 {
                        src_hi >> (8 * (n - 16))
                    } else if n == 0 {
                        src_lo
                    } else {
                        src_lo >> (8 * n) | src_hi << (8 * (16 - n))
                    }
                };
                shifted(wide.0, wide.1)
            }
            0x40 => {
                let (la, lb) = (lanes::<4>(a), lanes::<4>(b));
                let p = |k: usize| {
                    if imm >> (4 + k) & 1 != 0 {
                        f32::from_bits(la[k] as u32) * f32::from_bits(lb[k] as u32)
                    } else {
                        0.0
                    }
                };
                let sum = (p(0) + p(1)) + (p(2) + p(3));
                join::<4>(core::array::from_fn(|k| {
                    if imm >> k & 1 != 0 {
                        sum.to_bits() as u64
                    } else {
                        0
                    }
                }))
            }
            0x41 if !self.wide => {
                let (la, lb) = (lanes::<2>(a), lanes::<2>(b));
                let p = |k: usize| {
                    if imm >> (4 + k) & 1 != 0 {
                        f64::from_bits(la[k]) * f64::from_bits(lb[k])
                    } else {
                        0.0
                    }
                };
                let sum = p(0) + p(1);
                join::<2>(core::array::from_fn(|k| {
                    if imm >> k & 1 != 0 { sum.to_bits() } else { 0 }
                }))
            }
            0x42 => {
                let sel = imm >> (3 * half);
                let (la, lb) = (lanes::<16>(a), lanes::<16>(b));
                let aoff = ((sel >> 2) & 1) as usize * 4;
                let boff = (sel & 3) as usize * 4;
                join::<8>(core::array::from_fn(|k| {
                    (0..4)
                        .map(|j| la[aoff + k + j].abs_diff(lb[boff + j]))
                        .sum()
                }))
            }
            0x44 if !self.wide => {
                let x = lanes::<2>(a)[(imm & 1) as usize];
                let y = lanes::<2>(b)[((imm >> 4) & 1) as usize];
                (0..64).fold(0u128, |acc, k| {
                    if y >> k & 1 != 0 {
                        acc ^ (x as u128) << k
                    } else {
                        acc
                    }
                })
            }
            0x4a..=0x4c if vex => {
                let mask = self.xget((imm >> 4 & 0xf) as u8);
                let (bits, n) = match op {
                    0x4a => (32, 4),
                    0x4b => (64, 2),
                    _ => (8, 16),
                };
                let width = 128 / n;
                let m = if width == 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                (0..n).fold(0u128, |acc, k| {
                    let take_b = (mask >> (k * width + bits - 1)) & 1 != 0;
                    acc | ((if take_b { b } else { a }) >> (k * width) & m) << (k * width)
                })
            }
            _ => return none(),
        };
        self.xset(d, r);
        Ok(true)
    }

    /// ROUNDSS and ROUNDSD.
    fn round_scalar(&mut self, imm: u32) -> Res<bool> {
        let d = self.insn.reg;
        let mode = if imm & 4 != 0 {
            (self.cpu.mxcsr >> 13) & 3
        } else {
            imm & 3
        };
        let (bytes, v) = if self.insn.op == 0x0a {
            let y = self.src_low(4)?;
            let f = f32::from_bits(y as u32);
            let v = if f.is_nan() {
                (y as u32 | 0x0040_0000) as u64
            } else {
                (round_f(f as f64, mode) as f32).to_bits() as u64
            };
            (4, v)
        } else {
            let y = self.src_low(8)?;
            let f = f64::from_bits(y);
            (
                8,
                if f.is_nan() {
                    y | 0x0008_0000_0000_0000
                } else {
                    round_f(f, mode).to_bits()
                },
            )
        };
        self.set_low(d, bytes, v);
        Ok(true)
    }

    /// PEXTR*, EXTRACTPS, PINSR*, INSERTPS.
    fn extract_insert(&mut self) -> Res<bool> {
        let i = self.insn;
        let d = i.reg;
        let imm = self.imm() as u32;
        let w = i.rex_w();
        match i.op {
            0x14..=0x17 => {
                let v = self.xget(d);
                let (bytes, x) = match i.op {
                    0x14 => (1, (v >> (8 * (imm & 15))) as u64 & 0xff),
                    0x15 => (2, (v >> (16 * (imm & 7))) as u64 & 0xffff),
                    0x16 if w => (8, (v >> (64 * (imm & 1))) as u64),
                    _ => (4, (v >> (32 * (imm & 3))) as u64 & 0xffff_ffff),
                };
                match self.rm() {
                    Loc::Reg(r) => self.set(r, if bytes == 8 { 8 } else { 4 }, x),
                    Loc::Mem(a) => self.store(a, bytes, x)?,
                }
            }
            0x20 | 0x22 => {
                let (bytes, slot) = match i.op {
                    0x20 => (1, imm & 15),
                    _ if w => (8, imm & 1),
                    _ => (4, imm & 3),
                };
                let x = match self.rm() {
                    Loc::Reg(r) => self.get(r, if bytes == 8 { 8 } else { 4 }) & alu::mask(bytes),
                    Loc::Mem(a) => self.load(a, bytes)?,
                };
                let shift = 8 * bytes as u32 * slot;
                let m = (alu::mask(bytes) as u128) << shift;
                let v = (self.src1() & !m) | (x as u128) << shift;
                self.xset(d, v);
            }
            0x21 => {
                let src = match self.rm() {
                    Loc::Reg(r) => lanes::<4>(self.xget(r))[((imm >> 6) & 3) as usize],
                    Loc::Mem(a) => self.load(a, 4)?,
                };
                let mut l = lanes::<4>(self.src1());
                l[((imm >> 4) & 3) as usize] = src;
                for (k, slot) in l.iter_mut().enumerate() {
                    if imm >> k & 1 != 0 {
                        *slot = 0;
                    }
                }
                self.xset(d, join::<4>(l));
            }
            _ => return none(),
        }
        Ok(true)
    }

    /// FMA3: `0F38 96..BF`, packed forms on the current half.
    fn fma(&mut self) -> Res<bool> {
        let i = self.insn;
        let op = i.op;
        let d = i.reg;
        let scalar = op & 1 == 1 && op & 0xf >= 9;
        let kind = op & 0xf;
        let order = op >> 4;
        let w = i.rex_w();
        let (x, y) = (self.xget(d), self.src1());
        let z = if scalar {
            self.src_low(if w { 8 } else { 4 })? as u128
        } else {
            self.src128(false)?
        };
        // (multiplicand, multiplier, addend) per 132 / 213 / 231 operand order.
        let (m1, m2, add) = match order {
            0x9 => (x, z, y),
            0xa => (y, x, z),
            _ => (y, z, x),
        };
        let lane = |k: usize| -> (bool, bool) {
            match kind {
                0x6 => (k.is_multiple_of(2), false),
                0x7 => (k % 2 == 1, false),
                0x8 | 0x9 => (false, false),
                0xa | 0xb => (true, false),
                0xc | 0xd => (false, true),
                _ => (true, true),
            }
        };
        let r = if w {
            let (a, b, c) = (lanes::<2>(m1), lanes::<2>(m2), lanes::<2>(add));
            join::<2>(core::array::from_fn(|k| {
                let (sub, neg) = lane(k);
                let p = if neg {
                    -f64::from_bits(a[k])
                } else {
                    f64::from_bits(a[k])
                };
                let c = if sub {
                    -f64::from_bits(c[k])
                } else {
                    f64::from_bits(c[k])
                };
                fma64(p, f64::from_bits(b[k]), c)
            }))
        } else {
            let (a, b, c) = (lanes::<4>(m1), lanes::<4>(m2), lanes::<4>(add));
            join::<4>(core::array::from_fn(|k| {
                let (sub, neg) = lane(k);
                let p = f32::from_bits(a[k] as u32);
                let p = if neg { -p } else { p };
                let c = f32::from_bits(c[k] as u32);
                let c = if sub { -c } else { c };
                fma32(p, f32::from_bits(b[k] as u32), c) as u64
            }))
        };
        if scalar {
            let bytes = if w { 8 } else { 4 };
            let m = alu::mask(bytes) as u128;
            self.xset(d, (x & !m) | (r & m));
        } else {
            self.xset(d, r);
        }
        Ok(true)
    }

    /// Operations that move data between 128-bit halves or change width.
    fn cross_lane(&mut self) -> Res<bool> {
        let i = self.insn;
        let d = i.reg;
        let op = i.op;
        let vex = i.vex.is_some();
        let w = i.rex_w();
        let wide = self.wide256();
        let elems = |bits: usize| if wide { 256 / bits } else { 128 / bits };
        let to_y = |v: Vec<u64>, bits: usize| -> [u128; 2] {
            let mut out = [0u128; 2];
            for (k, &x) in v.iter().enumerate() {
                let pos = k * bits;
                let m = if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                out[pos / 128] |= ((x & m) as u128) << (pos % 128);
            }
            out
        };
        let from_y = |v: [u128; 2], bits: usize, n: usize| -> Vec<u64> {
            let m = if bits == 64 {
                u64::MAX as u128
            } else {
                (1u128 << bits) - 1
            };
            (0..n)
                .map(|k| ((v[k * bits / 128] >> ((k * bits) % 128)) & m) as u64)
                .collect()
        };
        if i.map == Map::M0F38 {
            match op {
                0x20..=0x25 | 0x30..=0x35 => {
                    let signed = op < 0x30;
                    let (from, to) = match op & 0xf {
                        0 => (8, 16),
                        1 => (8, 32),
                        2 => (8, 64),
                        3 => (16, 32),
                        4 => (16, 64),
                        _ => (32, 64),
                    };
                    let n = elems(to);
                    let bytes = n * from / 8;
                    let src = match self.rm() {
                        Loc::Reg(r) => self.xget(r),
                        Loc::Mem(a) => {
                            let mut b = [0u8; 16];
                            self.mem.read(a, &mut b[..bytes])?;
                            u128::from_le_bytes(b)
                        }
                    };
                    let v = from_y([src, 0], from, n)
                        .into_iter()
                        .map(|x| if signed { sx(x, from as u32) as u64 } else { x })
                        .collect();
                    self.put(d, to_y(v, to));
                }
                0x13 if vex && !w => {
                    let n = elems(32);
                    let src = match self.rm() {
                        Loc::Reg(r) => self.xget(r),
                        Loc::Mem(a) => {
                            let mut b = [0u8; 16];
                            self.mem.read(a, &mut b[..n * 2])?;
                            u128::from_le_bytes(b)
                        }
                    };
                    let v = from_y([src, 0], 16, n)
                        .into_iter()
                        .map(|h| f16_to_f32(h as u16) as u64)
                        .collect();
                    self.put(d, to_y(v, 32));
                }
                0x16 | 0x36 if vex && wide && !w => {
                    let idx = from_y(self.yget(self.vvvv()), 32, 8);
                    let data = from_y(self.ysrc()?, 32, 8);
                    let v = idx.iter().map(|&k| data[(k & 7) as usize]).collect();
                    self.put(d, to_y(v, 32));
                }
                0x18 | 0x19 | 0x58 | 0x59 | 0x78 | 0x79
                    if vex && !w || matches!(op, 0x19 | 0x59) && vex =>
                {
                    let bits = match op {
                        0x78 => 8,
                        0x79 => 16,
                        0x18 | 0x58 => 32,
                        _ => 64,
                    };
                    let x = match self.rm() {
                        Loc::Reg(r) => self.xget(r) as u64,
                        Loc::Mem(a) => self.load(a, (bits / 8) as u8)?,
                    };
                    self.put(d, to_y(vec![x; elems(bits)], bits));
                }
                0x1a | 0x5a if vex && wide && !w => {
                    let Loc::Mem(a) = self.rm() else {
                        return Err(CpuFault::InvalidOpcode);
                    };
                    let v = self.mem_u128(a)?;
                    self.yset(d, [v, v]);
                }
                0x2c | 0x2d | 0x8c if vex && (op != 0x8c || true) => {
                    let bits = if op == 0x2d || (op == 0x8c && w) {
                        64
                    } else {
                        32
                    };
                    let Loc::Mem(a) = self.rm() else {
                        return Err(CpuFault::InvalidOpcode);
                    };
                    let n = elems(bits);
                    let mask = from_y(self.yget(self.vvvv()), bits, n);
                    let mut v = vec![0u64; n];
                    for k in 0..n {
                        if sign_bit(mask[k], bits as u32) {
                            v[k] = self.load(a + (k * bits / 8) as u64, (bits / 8) as u8)?;
                        }
                    }
                    self.put(d, to_y(v, bits));
                }
                0x2e | 0x2f | 0x8e if vex => {
                    let bits = if op == 0x2f || (op == 0x8e && w) {
                        64
                    } else {
                        32
                    };
                    let Loc::Mem(a) = self.rm() else {
                        return Err(CpuFault::InvalidOpcode);
                    };
                    let n = elems(bits);
                    let mask = from_y(self.yget(self.vvvv()), bits, n);
                    let data = from_y(self.yget(d), bits, n);
                    for k in 0..n {
                        if sign_bit(mask[k], bits as u32) {
                            self.store(a + (k * bits / 8) as u64, (bits / 8) as u8, data[k])?;
                        }
                    }
                }
                0x0e | 0x0f if vex => {
                    let bits = if op == 0x0e { 32 } else { 64 };
                    let (x, y) = (self.yget(self.insn.reg), self.ysrc()?);
                    let width = if wide { 2 } else { 1 };
                    let signs = |v: [u128; 2]| -> [u128; 2] {
                        let mut m = 0u128;
                        for k in 0..128 / bits {
                            m |= 1u128 << (k * bits + bits - 1);
                        }
                        [v[0] & m, v[1] & m]
                    };
                    let (x, y) = (signs(x), signs(y));
                    let z = (0..width).all(|h| x[h] & y[h] == 0);
                    let c = (0..width).all(|h| !x[h] & y[h] == 0);
                    self.cpu
                        .set_arith(if z { ZF } else { 0 } | if c { CF } else { 0 });
                }
                0x90..=0x93 if vex => return self.gather(),
                0xd7 => return none(),
                0x0c
                | 0x0d
                | 0x00..=0x0b
                | 0x1c..=0x1e
                | 0x28..=0x2b
                | 0x37..=0x47
                | 0x96..=0xbf
                | 0x10
                | 0x14
                | 0x15
                | 0x17 => return Ok(false),
                _ => return Ok(false),
            }
            return Ok(true);
        }
        let imm = self.imm() as u32;
        match op {
            0x00 | 0x01 if vex && wide && w => {
                let data = from_y(self.ysrc()?, 64, 4);
                let v = (0..4)
                    .map(|k| data[((imm >> (2 * k)) & 3) as usize])
                    .collect();
                self.put(d, to_y(v, 64));
            }
            0x06 | 0x46 if vex && wide => {
                let (a, b) = (self.yget(self.vvvv()), self.ysrc()?);
                let pick = |sel: u32| -> u128 {
                    if sel & 8 != 0 {
                        return 0;
                    }
                    match sel & 3 {
                        0 => a[0],
                        1 => a[1],
                        2 => b[0],
                        _ => b[1],
                    }
                };
                self.yset(d, [pick(imm), pick(imm >> 4)]);
            }
            0x18 | 0x38 if vex && wide => {
                let mut v = self.yget(self.vvvv());
                let src = match self.rm() {
                    Loc::Reg(r) => self.xget(r),
                    Loc::Mem(a) => self.mem_u128(a)?,
                };
                v[(imm & 1) as usize] = src;
                self.yset(d, v);
            }
            0x19 | 0x39 if vex && wide => {
                let v = self.yget(d)[(imm & 1) as usize];
                match self.rm() {
                    Loc::Reg(r) => self.yset(r, [v, 0]),
                    Loc::Mem(a) => self.mem.write(a, &v.to_le_bytes())?,
                }
            }
            0x1d if vex && !w => {
                let n = elems(32);
                let rc = if imm & 4 != 0 {
                    (self.cpu.mxcsr >> 13) & 3
                } else {
                    imm & 3
                };
                let src = from_y(self.yget(d), 32, n);
                let halves: Vec<u64> = src
                    .iter()
                    .map(|&x| f32_to_f16(x as u32, rc) as u64)
                    .collect();
                let packed = to_y(halves, 16)[0];
                match self.rm() {
                    Loc::Reg(r) => self.yset(r, [packed, 0]),
                    Loc::Mem(a) => {
                        let bytes = packed.to_le_bytes();
                        self.mem.write(a, &bytes[..n * 2])?;
                    }
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// VPGATHER*/VGATHER*: VSIB loads of the elements whose mask (VEX.vvvv)
    /// sign bit is set; the mask register ends up all zero.
    fn gather(&mut self) -> Res<bool> {
        let i = self.insn;
        let Some(m) = i.mem else {
            return Err(CpuFault::InvalidOpcode);
        };
        let Some(index_reg) = m.vsib_index else {
            return Err(CpuFault::InvalidOpcode);
        };
        let w = i.rex_w();
        let wide = self.wide256();
        let qword_index = i.op & 1 == 1;
        let data_bits = if w { 64 } else { 32 };
        let index_bits = if qword_index { 64 } else { 32 };
        let n = if wide { 256 } else { 128 } / data_bits.max(index_bits);
        let base = m
            .base
            .map_or(0, |b| self.cpu.regs[b as usize])
            .wrapping_add(m.disp as i64 as u64);
        let base = match m.seg {
            Some(Seg::Fs) => base.wrapping_add(self.cpu.fs_base),
            Some(Seg::Gs) => base.wrapping_add(self.cpu.gs_base),
            None => base,
        };
        let idx = self.yget(index_reg);
        let mask_reg = self.vvvv();
        let get = |v: [u128; 2], bits: usize, k: usize| -> u64 {
            let m = if bits == 64 {
                u64::MAX as u128
            } else {
                (1u128 << bits) - 1
            };
            ((v[k * bits / 128] >> ((k * bits) % 128)) & m) as u64
        };
        let set = |v: &mut [u128; 2], bits: usize, k: usize, x: u64| {
            let m = if bits == 64 {
                u64::MAX as u128
            } else {
                (1u128 << bits) - 1
            };
            let (lane, pos) = (k * bits / 128, (k * bits) % 128);
            v[lane] = (v[lane] & !(m << pos)) | ((x as u128 & m) << pos);
        };
        let mut dst = self.yget(i.reg);
        let mut mask = self.yget(mask_reg);
        for k in 0..n {
            if !sign_bit(get(mask, data_bits, k), data_bits as u32) {
                continue;
            }
            let ix = get(idx, index_bits, k);
            let ix = if qword_index { ix } else { sx(ix, 32) as u64 };
            let addr = base.wrapping_add(ix.wrapping_mul(m.scale as u64));
            let x = self.load(addr, (data_bits / 8) as u8)?;
            set(&mut dst, data_bits, k, x);
            set(&mut mask, data_bits, k, 0);
            self.yset(i.reg, dst);
            self.yset(mask_reg, mask);
        }
        let used = n * data_bits;
        let clear = |v: &mut [u128; 2]| {
            if used <= 128 {
                if used < 128 {
                    v[0] &= (1u128 << used) - 1;
                }
                v[1] = 0;
            }
        };
        clear(&mut dst);
        self.yset(i.reg, dst);
        self.yset(mask_reg, [0, 0]);
        Ok(true)
    }

    /// MOVBE (`0F38 F0/F1`) and SSE4.2's CRC32 (`F2 0F38 F0/F1`).
    fn movbe_crc32(&mut self) -> Res<bool> {
        let i = self.insn;
        if i.rep == Rep::Repne {
            let size = if i.op == 0xf0 { 1 } else { i.osize() };
            let v = self.read(self.rm(), size)?;
            let crc = crc32c(self.get(i.reg, 4) as u32, v, size);
            self.set(i.reg, 4, crc as u64);
            return Ok(true);
        }
        let Loc::Mem(a) = self.rm() else {
            return Err(CpuFault::InvalidOpcode);
        };
        let size = i.osize();
        let swap = |v: u64| match size {
            2 => (v as u16).swap_bytes() as u64,
            4 => (v as u32).swap_bytes() as u64,
            _ => v.swap_bytes(),
        };
        if i.op == 0xf0 {
            let v = self.load(a, size)?;
            self.set(i.reg, size, swap(v));
        } else {
            let v = self.get(i.reg, size);
            self.store(a, size, swap(v))?;
        }
        Ok(true)
    }

    /// BMI1 and BMI2 on general registers (VEX-encoded).
    fn bmi(&mut self) -> Res<bool> {
        let i = self.insn;
        let size = if i.rex_w() { 8 } else { 4 };
        let bits = size as u32 * 8;
        let v = self.vvvv();
        let p = self.pfx();
        let flags_zs = |r: u64| -> u64 { alu::szp(r, size) & (ZF | SF) };
        match (i.map, i.op, p) {
            (Map::M0F38, 0xf2, Pfx::None) => {
                let r = !self.get(v, size) & self.read(self.rm(), size)? & alu::mask(size);
                self.cpu.set_arith(flags_zs(r));
                self.set(i.reg, size, r);
            }
            (Map::M0F38, 0xf3, Pfx::None) if (1..=3).contains(&i.ext()) => {
                let s = self.read(self.rm(), size)?;
                let (r, cf) = match i.ext() {
                    1 => (s & s.wrapping_sub(1), s == 0),
                    2 => (s ^ s.wrapping_sub(1), s == 0),
                    _ => (s.isolate_lowest_one(), s != 0),
                };
                let r = r & alu::mask(size);
                self.cpu.set_arith(flags_zs(r) | if cf { CF } else { 0 });
                self.set(v, size, r);
            }
            (Map::M0F38, 0xf5, Pfx::None) => {
                let s = self.read(self.rm(), size)?;
                let n = (self.get(v, size) & 0xff) as u32;
                let r = if n >= bits { s } else { s & ((1u64 << n) - 1) };
                self.cpu
                    .set_arith(flags_zs(r) | if n > bits - 1 { CF } else { 0 });
                self.set(i.reg, size, r);
            }
            (Map::M0F38, 0xf5, Pfx::F2 | Pfx::F3) => {
                let (src, mask) = (self.get(v, size), self.read(self.rm(), size)?);
                let mut r = 0u64;
                let mut k = 0;
                for b in 0..bits {
                    if mask >> b & 1 != 0 {
                        if p == Pfx::F2 {
                            r |= (src >> k & 1) << b;
                        } else {
                            r |= (src >> b & 1) << k;
                        }
                        k += 1;
                    }
                }
                self.set(i.reg, size, r);
            }
            (Map::M0F38, 0xf6, Pfx::F2) => {
                let (a, b) = (self.get(RDX as u8, size), self.read(self.rm(), size)?);
                let prod = a as u128 * b as u128;
                self.set(v, size, prod as u64 & alu::mask(size));
                self.set(i.reg, size, (prod >> bits) as u64 & alu::mask(size));
            }
            (Map::M0F38, 0xf7, Pfx::None) => {
                let s = self.read(self.rm(), size)?;
                let ctl = self.get(v, size);
                let (start, len) = ((ctl & 0xff) as u32, ((ctl >> 8) & 0xff) as u32);
                let r = if start >= bits {
                    0
                } else {
                    let x = s >> start;
                    if len >= 64 {
                        x
                    } else {
                        x & ((1u64 << len) - 1)
                    }
                } & alu::mask(size);
                self.cpu.set_arith(alu::szp(r, size) & ZF);
                self.set(i.reg, size, r);
            }
            (Map::M0F38, 0xf7, Pfx::P66 | Pfx::F3 | Pfx::F2) => {
                let s = self.read(self.rm(), size)?;
                let n = (self.get(v, size) & (bits as u64 - 1)) as u32;
                let r = match p {
                    Pfx::P66 => s << n,
                    Pfx::F3 => (sx(s, bits) >> n) as u64,
                    _ => s >> n,
                } & alu::mask(size);
                self.set(i.reg, size, r);
            }
            (Map::M0F3A, 0xf0, Pfx::F2) => {
                let s = self.read(self.rm(), size)?;
                let n = (self.imm() as u32) & (bits - 1);
                let r = if n == 0 {
                    s
                } else {
                    ((s >> n) | (s << (bits - n))) & alu::mask(size)
                };
                self.set(i.reg, size, r);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// PCMPESTRM, PCMPESTRI, PCMPISTRM and PCMPISTRI, as the SDM's "Packed
    /// String Compare" describes them.
    fn string_compare(&mut self) -> Res<bool> {
        let i = self.insn;
        let imm = self.imm() as u8;
        let words = imm & 1 != 0;
        let signed = imm & 2 != 0;
        let n = if words { 8 } else { 16 };
        let a = self.xget(i.reg);
        let b = self.src128(false)?;
        let elem = |v: u128, k: usize| -> i64 {
            let raw = if words {
                (v >> (16 * k)) as u16 as u64
            } else {
                (v >> (8 * k)) as u8 as u64
            };
            match (signed, words) {
                (true, true) => raw as u16 as i16 as i64,
                (true, false) => raw as u8 as i8 as i64,
                _ => raw as i64,
            }
        };
        let explicit = i.op <= 0x61;
        let len = |reg: usize, v: u128| -> usize {
            if explicit {
                let raw = self.get(reg as u8, if i.rex_w() { 8 } else { 4 });
                let l = if i.rex_w() {
                    raw as i64
                } else {
                    raw as i32 as i64
                };
                (l.unsigned_abs() as usize).min(n)
            } else {
                (0..n).find(|&k| elem(v, k) == 0).unwrap_or(n)
            }
        };
        let (la, lb) = (len(RAX, a), len(RDX, b));
        let agg = (imm >> 2) & 3;
        let bool_res = |j: usize, k: usize| -> bool {
            let (aj, bk) = (j < la, k < lb);
            match agg {
                0 | 1 if !aj || !bk => false,
                2 if !aj && !bk => true,
                2 if !aj || !bk => false,
                3 if !aj => true,
                3 if !bk => false,
                1 if j.is_multiple_of(2) => elem(b, k) >= elem(a, j),
                1 => elem(b, k) <= elem(a, j),
                _ => elem(b, k) == elem(a, j),
            }
        };
        let mut res = 0u32;
        for k in 0..n {
            let bit = match agg {
                0 => (0..n).any(|j| bool_res(j, k)),
                1 => (0..n)
                    .step_by(2)
                    .any(|j| bool_res(j, k) && bool_res(j + 1, k)),
                2 => bool_res(k, k),
                _ => (0..n - k).all(|j| bool_res(j, k + j)),
            };
            res |= (bit as u32) << k;
        }
        let all = (1u32 << n) - 1;
        let res = match (imm >> 4) & 3 {
            1 => !res & all,
            3 => res ^ ((1u32 << lb) - 1),
            _ => res,
        };
        if i.op & 1 == 1 {
            let index = if res == 0 {
                n as u32
            } else if imm & 0x40 != 0 {
                31 - res.leading_zeros()
            } else {
                res.trailing_zeros()
            };
            self.cpu.regs[RCX] = index as u64;
        } else {
            let mask = if imm & 0x40 == 0 {
                res as u128
            } else {
                (0..n).fold(0u128, |m, k| {
                    let ones: u128 = if words { 0xffff } else { 0xff };
                    if res >> k & 1 != 0 {
                        m | ones << (k * if words { 16 } else { 8 })
                    } else {
                        m
                    }
                })
            };
            self.xset(0, mask);
        }
        let mut f = 0;
        if res != 0 {
            f |= CF;
        }
        if lb < n {
            f |= ZF;
        }
        if la < n {
            f |= SF;
        }
        if res & 1 != 0 {
            f |= OF;
        }
        self.cpu.set_arith(f);
        Ok(true)
    }

    /// PTEST/VPTEST: ZF when `a AND b` is all zero, CF when `(NOT a) AND b`
    /// is, over the whole operand; the other arithmetic flags clear.
    fn ptest(&mut self) -> Res<bool> {
        let i = self.insn;
        let wide = i.vex.is_some_and(|v| v.l);
        let a = self.cpu.xmm[i.reg as usize];
        let b: [u64; 4] = match self.rm() {
            Loc::Mem(addr) => {
                let mut v = [0u64; 4];
                for (k, lane) in v.iter_mut().enumerate().take(if wide { 4 } else { 2 }) {
                    *lane = self.load(addr + 8 * k as u64, 8)?;
                }
                v
            }
            Loc::Reg(r) => self.cpu.xmm[r as usize],
        };
        let n = if wide { 4 } else { 2 };
        let zf = (0..n).all(|k| a[k] & b[k] == 0);
        let cf = (0..n).all(|k| !a[k] & b[k] == 0);
        self.cpu
            .set_arith(if zf { ZF } else { 0 } | if cf { CF } else { 0 });
        Ok(true)
    }
}
