//! The SHA extensions, AES-NI and PCLMULQDQ, as the Intel SDM volume 2
//! defines them. AES's tables are computed from FIPS-197's definitions.

use super::sse::Pfx;
use super::*;

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut r = 0;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        a = (a << 1) ^ if a & 0x80 != 0 { 0x1b } else { 0 };
        b >>= 1;
    }
    r
}

const fn sbox() -> [u8; 256] {
    let mut s = [0u8; 256];
    let mut x = 0;
    while x < 256 {
        let mut inv = 0u8;
        if x != 0 {
            let mut y = 1;
            while y < 256 {
                if gf_mul(x as u8, y as u8) == 1 {
                    inv = y as u8;
                    break;
                }
                y += 1;
            }
        }
        s[x] = inv
            ^ inv.rotate_left(1)
            ^ inv.rotate_left(2)
            ^ inv.rotate_left(3)
            ^ inv.rotate_left(4)
            ^ 0x63;
        x += 1;
    }
    s
}

const SBOX: [u8; 256] = sbox();

const INV_SBOX: [u8; 256] = {
    let mut inv = [0u8; 256];
    let mut x = 0;
    while x < 256 {
        inv[SBOX[x] as usize] = x as u8;
        x += 1;
    }
    inv
};

/// Byte `r + 4c` of the state is row `r`, column `c`.
fn shift_rows(s: [u8; 16], inverse: bool) -> [u8; 16] {
    core::array::from_fn(|k| {
        let (r, c) = (k % 4, k / 4);
        let from = if inverse {
            (c + 4 - r) % 4
        } else {
            (c + r) % 4
        };
        s[r + 4 * from]
    })
}

fn mix_columns(s: [u8; 16], inverse: bool) -> [u8; 16] {
    let m: [u8; 4] = if inverse {
        [14, 11, 13, 9]
    } else {
        [2, 3, 1, 1]
    };
    core::array::from_fn(|k| {
        let (r, c) = (k % 4, k / 4);
        (0..4).fold(0, |acc, j| acc ^ gf_mul(m[(j + 4 - r) % 4], s[j + 4 * c]))
    })
}

/// One AES round of `op` (`0F38 DC..DF`) on `state` with `key`.
fn aes_round(op: u8, state: u128, key: u128) -> u128 {
    let decrypt = op >= 0xde;
    let last = op & 1 == 1;
    let box_ = if decrypt { &INV_SBOX } else { &SBOX };
    let mut s = shift_rows(state.to_le_bytes(), decrypt).map(|b| box_[b as usize]);
    if !last {
        s = mix_columns(s, decrypt);
    }
    u128::from_le_bytes(s) ^ key
}

fn sub_word(w: u32) -> u32 {
    u32::from_le_bytes(w.to_le_bytes().map(|b| SBOX[b as usize]))
}

fn keygen_assist(src: u128, rcon: u32) -> u128 {
    let x = |k: u32| (src >> (32 * k)) as u32;
    let (s1, s3) = (sub_word(x(1)), sub_word(x(3)));
    let words = [s1, s1.rotate_right(8) ^ rcon, s3, s3.rotate_right(8) ^ rcon];
    words_to(words)
}

/// The four dwords of `v`, lowest first.
fn words(v: u128) -> [u32; 4] {
    core::array::from_fn(|k| (v >> (32 * k)) as u32)
}

fn words_to(w: [u32; 4]) -> u128 {
    w.iter()
        .enumerate()
        .fold(0, |acc, (k, &x)| acc | (x as u128) << (32 * k))
}

fn sha1rnds4(a: u128, b: u128, func: u32) -> u128 {
    let [d, c, bb, aa] = words(a);
    let [w3, w2, w1, w0] = words(b);
    let (f, k): (fn(u32, u32, u32) -> u32, u32) = match func & 3 {
        0 => (|b, c, d| (b & c) ^ (!b & d), 0x5a82_7999),
        1 => (|b, c, d| b ^ c ^ d, 0x6ed9_eba1),
        2 => (|b, c, d| (b & c) ^ (b & d) ^ (c & d), 0x8f1b_bcdc),
        _ => (|b, c, d| b ^ c ^ d, 0xca62_c1d6),
    };
    let (mut a, mut b, mut c, mut d, mut e) = (aa, bb, c, d, 0u32);
    for w in [w0, w1, w2, w3] {
        let next = f(b, c, d)
            .wrapping_add(a.rotate_left(5))
            .wrapping_add(w)
            .wrapping_add(e)
            .wrapping_add(k);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = next;
    }
    words_to([d, c, b, a])
}

fn sha1nexte(a: u128, b: u128) -> u128 {
    let t = words(a)[3].rotate_left(30);
    let mut w = words(b);
    w[3] = w[3].wrapping_add(t);
    words_to(w)
}

fn sha1msg1(a: u128, b: u128) -> u128 {
    let [w3, w2, w1, w0] = words(a);
    let [_, _, w5, w4] = words(b);
    words_to([w5 ^ w3, w4 ^ w2, w3 ^ w1, w2 ^ w0])
}

fn sha1msg2(a: u128, b: u128) -> u128 {
    let [s0, s1, s2, s3] = words(a);
    let [w15, w14, w13, _] = words(b);
    let w16 = (s3 ^ w13).rotate_left(1);
    let w17 = (s2 ^ w14).rotate_left(1);
    let w18 = (s1 ^ w15).rotate_left(1);
    let w19 = (s0 ^ w16).rotate_left(1);
    words_to([w19, w18, w17, w16])
}

fn sha256rnds2(a: u128, b: u128, wk: u128) -> u128 {
    let [h, g, d, c] = words(a);
    let [f, e, bb, aa] = words(b);
    let wk = words(wk);
    let (mut a, mut b, mut c, mut d) = (aa, bb, c, d);
    let (mut e, mut f, mut g, mut h) = (e, f, g, h);
    for &w in &wk[..2] {
        let ch = (e & f) ^ (!e & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let t = ch.wrapping_add(s1).wrapping_add(w).wrapping_add(h);
        let next_a = t.wrapping_add(maj).wrapping_add(s0);
        let next_e = t.wrapping_add(d);
        h = g;
        g = f;
        f = e;
        e = next_e;
        d = c;
        c = b;
        b = a;
        a = next_a;
    }
    words_to([f, e, b, a])
}

fn sha256msg1(a: u128, b: u128) -> u128 {
    let s0 = |w: u32| w.rotate_right(7) ^ w.rotate_right(18) ^ (w >> 3);
    let [w0, w1, w2, w3] = words(a);
    let w4 = words(b)[0];
    words_to([
        w0.wrapping_add(s0(w1)),
        w1.wrapping_add(s0(w2)),
        w2.wrapping_add(s0(w3)),
        w3.wrapping_add(s0(w4)),
    ])
}

fn sha256msg2(a: u128, b: u128) -> u128 {
    let s1 = |w: u32| w.rotate_right(17) ^ w.rotate_right(19) ^ (w >> 10);
    let [x0, x1, x2, x3] = words(a);
    let [_, _, w14, w15] = words(b);
    let w16 = x0.wrapping_add(s1(w14));
    let w17 = x1.wrapping_add(s1(w15));
    let w18 = x2.wrapping_add(s1(w16));
    let w19 = x3.wrapping_add(s1(w17));
    words_to([w16, w17, w18, w19])
}

fn clmul(a: u64, b: u64) -> u128 {
    (0..64)
        .filter(|k| b >> k & 1 != 0)
        .fold(0, |acc, k| acc ^ (a as u128) << k)
}

impl Ctx<'_> {
    /// Runs `i` if it is a SHA or AES instruction.
    pub(super) fn crypto(&mut self) -> Res<Option<bool>> {
        let i = self.insn;
        let sha = i.vex.is_none()
            && self.pfx() == Pfx::None
            && match i.map {
                Map::M0F38 => (0xc8..=0xcd).contains(&i.op),
                Map::M0F3A => i.op == 0xcc,
                _ => false,
            };
        if sha {
            let (a, b) = (self.xget(i.reg), self.src128(true)?);
            let r = match (i.map, i.op) {
                (Map::M0F3A, _) => sha1rnds4(a, b, self.imm() as u32),
                (_, 0xc8) => sha1nexte(a, b),
                (_, 0xc9) => sha1msg1(a, b),
                (_, 0xca) => sha1msg2(a, b),
                (_, 0xcb) => sha256rnds2(a, b, self.xget(0)),
                (_, 0xcc) => sha256msg1(a, b),
                _ => sha256msg2(a, b),
            };
            self.xset(i.reg, r);
            return Ok(Some(true));
        }
        let aes_or_clmul = self.pfx() == Pfx::P66
            && match i.map {
                Map::M0F38 => (0xdb..=0xdf).contains(&i.op),
                Map::M0F3A => matches!(i.op, 0x44 | 0xdf),
                _ => false,
            };
        if !aes_or_clmul {
            return Ok(None);
        }
        // Zen 2 has neither 256-bit VAES nor VPCLMULQDQ.
        if i.vex.is_some_and(|v| v.l) {
            return Err(CpuFault::InvalidOpcode);
        }
        let src = self.src128(i.vex.is_none())?;
        let r = match (i.map, i.op) {
            (Map::M0F3A, 0x44) => {
                let imm = self.imm();
                let a = (self.src1() >> (64 * (imm & 1))) as u64;
                let b = (src >> (64 * (imm >> 4 & 1))) as u64;
                clmul(a, b)
            }
            (Map::M0F3A, _) => keygen_assist(src, self.imm() as u8 as u32),
            (_, 0xdb) => u128::from_le_bytes(mix_columns(src.to_le_bytes(), true)),
            (_, op) => aes_round(op, self.src1(), src),
        };
        self.xset(i.reg, r);
        Ok(Some(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbox_matches_fips_197_examples() {
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(INV_SBOX[0xed], 0x53);
    }

    #[test]
    fn aes_128_encrypts_the_fips_197_example() {
        let key = 0x000102030405060708090a0b0c0d0e0fu128.swap_bytes();
        let plain = 0x00112233445566778899aabbccddeeffu128.swap_bytes();
        let mut keys = vec![key];
        for rcon in [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36] {
            let prev = *keys.last().unwrap();
            let assist = words(keygen_assist(prev, rcon))[3];
            let mut w = words(prev);
            w[0] ^= assist;
            for k in 1..4 {
                w[k] ^= w[k - 1];
            }
            keys.push(words_to(w));
        }
        let mut s = plain ^ keys[0];
        for k in &keys[1..10] {
            s = aes_round(0xdc, s, *k);
        }
        s = aes_round(0xdd, s, keys[10]);
        assert_eq!(s, 0x69c4e0d86a7b0430d8cdb78070b4c55au128.swap_bytes());
    }
}
