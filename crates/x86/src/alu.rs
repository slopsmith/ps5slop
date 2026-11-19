//! Integer arithmetic with eagerly computed flags. Sizes are in bytes (1, 2,
//! 4 or 8); every function returns the result and the six arithmetic flags.

use crate::cpu::flags::*;

pub fn mask(size: u8) -> u64 {
    if size == 8 {
        u64::MAX
    } else {
        (1u64 << (size * 8)) - 1
    }
}

pub fn sign(size: u8) -> u64 {
    1u64 << (size * 8 - 1)
}

/// Sign-extends the low `size` bytes of `v`.
pub fn sext(v: u64, size: u8) -> u64 {
    let shift = 64 - size as u32 * 8;
    (((v << shift) as i64) >> shift) as u64
}

pub fn szp(r: u64, size: u8) -> u64 {
    let mut f = 0;
    if r & mask(size) == 0 {
        f |= ZF;
    }
    if r & sign(size) != 0 {
        f |= SF;
    }
    if (r as u8).count_ones().is_multiple_of(2) {
        f |= PF;
    }
    f
}

pub fn add(a: u64, b: u64, carry: bool, size: u8) -> (u64, u64) {
    let m = mask(size);
    let (a, b) = (a & m, b & m);
    let wide = a as u128 + b as u128 + carry as u128;
    let r = wide as u64 & m;
    let mut f = szp(r, size);
    if wide > m as u128 {
        f |= CF;
    }
    if (a ^ b ^ r) & 0x10 != 0 {
        f |= AF;
    }
    if (a ^ r) & (b ^ r) & sign(size) != 0 {
        f |= OF;
    }
    (r, f)
}

pub fn sub(a: u64, b: u64, borrow: bool, size: u8) -> (u64, u64) {
    let m = mask(size);
    let (a, b) = (a & m, b & m);
    let r = a.wrapping_sub(b).wrapping_sub(borrow as u64) & m;
    let mut f = szp(r, size);
    if (a as u128) < b as u128 + borrow as u128 {
        f |= CF;
    }
    if (a ^ b ^ r) & 0x10 != 0 {
        f |= AF;
    }
    if (a ^ b) & (a ^ r) & sign(size) != 0 {
        f |= OF;
    }
    (r, f)
}

pub fn logic(r: u64, size: u8) -> (u64, u64) {
    let r = r & mask(size);
    (r, szp(r, size))
}

/// The eight group-1 operations, in ModRM `reg` order: ADD OR ADC SBB AND
/// SUB XOR CMP. `cf` is the incoming carry.
pub fn group1(op: u8, a: u64, b: u64, cf: bool, size: u8) -> (u64, u64) {
    match op {
        0 => add(a, b, false, size),
        1 => logic(a | b, size),
        2 => add(a, b, cf, size),
        3 => sub(a, b, cf, size),
        4 => logic(a & b, size),
        5 | 7 => sub(a, b, false, size),
        _ => logic(a ^ b, size),
    }
}

/// INC and DEC leave CF alone: the caller keeps `old` CF.
pub fn inc_dec(a: u64, dec: bool, old: u64, size: u8) -> (u64, u64) {
    let (r, f) = if dec {
        sub(a, 1, false, size)
    } else {
        add(a, 1, false, size)
    };
    (r, (f & !CF) | (old & CF))
}

/// The group-2 shifts and rotates, in ModRM `reg` order: ROL ROR RCL RCR
/// SHL SHR SAL SAR. `count` is already masked. Flags the operation does
/// not define keep their `old` value; with a zero count nothing changes.
pub fn shift(op: u8, a: u64, count: u32, old: u64, size: u8) -> (u64, u64) {
    let bits = size as u32 * 8;
    let m = mask(size);
    let a = a & m;
    if count == 0 {
        return (a, old);
    }
    let msb = |v: u64| v & sign(size) != 0;
    let cf_in = old & CF != 0;
    let mut f = old;
    let set = |f: &mut u64, flag: u64, on: bool| {
        if on { *f |= flag } else { *f &= !flag }
    };
    let r = match op {
        0 => {
            let c = count % bits;
            let r = if c == 0 {
                a
            } else {
                ((a << c) | (a >> (bits - c))) & m
            };
            set(&mut f, CF, r & 1 != 0);
            set(&mut f, OF, msb(r) != (r & 1 != 0));
            r
        }
        1 => {
            let c = count % bits;
            let r = if c == 0 {
                a
            } else {
                ((a >> c) | (a << (bits - c))) & m
            };
            set(&mut f, CF, msb(r));
            set(&mut f, OF, msb(r) != (r & sign(size) >> 1 != 0));
            r
        }
        2 | 3 => {
            let c = count % (bits + 1);
            let mut r = a;
            let mut carry = cf_in;
            for _ in 0..c {
                if op == 2 {
                    let out = msb(r);
                    r = ((r << 1) | carry as u64) & m;
                    carry = out;
                } else {
                    let out = r & 1 != 0;
                    r = (r >> 1) | if carry { sign(size) } else { 0 };
                    carry = out;
                }
            }
            set(&mut f, CF, carry);
            if op == 2 {
                set(&mut f, OF, msb(r) != carry);
            } else {
                set(&mut f, OF, msb(r) != (r & sign(size) >> 1 != 0));
            }
            r
        }
        4 | 6 => {
            let r = if count >= bits { 0 } else { (a << count) & m };
            let out = count <= bits && (a >> (bits - count)) & 1 != 0;
            f = (f & !(ZF | SF | PF | CF | OF)) | szp(r, size);
            set(&mut f, CF, out);
            set(&mut f, OF, msb(r) != out);
            r
        }
        5 => {
            let r = if count >= bits { 0 } else { a >> count };
            let out = count <= bits && (a >> (count - 1)) & 1 != 0;
            f = (f & !(ZF | SF | PF | CF | OF)) | szp(r, size);
            set(&mut f, CF, out);
            set(&mut f, OF, msb(a));
            r
        }
        _ => {
            let s = sext(a, size) as i64;
            let c = count.min(63);
            let r = (s >> c) as u64 & m;
            let out = (s >> (c - 1).min(63)) & 1 != 0;
            f = (f & !(ZF | SF | PF | CF | OF)) | szp(r, size);
            set(&mut f, CF, out);
            r
        }
    };
    (r, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sets_carry_and_overflow_per_size() {
        assert_eq!(add(0xff, 1, false, 1), (0, ZF | CF | AF | PF));
        assert_eq!(add(0x7f, 1, false, 1).1 & (OF | SF), OF | SF);
        let (r, f) = add(u64::MAX, 1, false, 8);
        assert_eq!(r, 0);
        assert!(f & CF != 0 && f & OF == 0);
        assert_eq!(add(0x7fff_ffff, 1, false, 4).1 & OF, OF);
    }

    #[test]
    fn sub_borrows_and_overflows() {
        let (r, f) = sub(0, 1, false, 4);
        assert_eq!(r, 0xffff_ffff);
        assert!(f & CF != 0 && f & SF != 0 && f & OF == 0);
        assert_eq!(sub(0x8000_0000_0000_0000, 1, false, 8).1 & OF, OF);
        assert_eq!(sub(5, 5, false, 2).1 & (ZF | CF), ZF);
    }

    #[test]
    fn inc_keeps_carry() {
        assert_eq!(inc_dec(0xff, false, CF, 1).1 & CF, CF);
        assert_eq!(inc_dec(0, true, 0, 1), (0xff, SF | AF | PF));
    }

    #[test]
    fn shifts_report_the_last_bit_out() {
        let (r, f) = shift(4, 0x80, 1, 0, 1);
        assert_eq!(r, 0);
        assert!(f & CF != 0 && f & ZF != 0 && f & OF != 0);
        let (r, f) = shift(5, 0b101, 1, 0, 4);
        assert_eq!((r, f & CF), (0b10, CF));
        assert_eq!(shift(7, 0x8000_0000, 4, 0, 4).0, 0xf800_0000);
        assert_eq!(shift(7, 0x8000_0000_0000_0000, 63, 0, 8).0, u64::MAX);
    }

    #[test]
    fn zero_count_changes_nothing() {
        assert_eq!(shift(4, 0x1234, 0, CF | ZF, 2), (0x1234, CF | ZF));
    }

    #[test]
    fn rotates_wrap_around() {
        assert_eq!(shift(0, 0x81, 1, 0, 1).0, 0x03);
        assert_eq!(shift(1, 0x01, 1, 0, 1), (0x80, CF | OF));
        assert_eq!(shift(2, 0x80, 1, 0, 1), (0x00, CF | OF));
        assert_eq!(shift(3, 0x01, 1, CF, 1).0, 0x80);
    }
}
