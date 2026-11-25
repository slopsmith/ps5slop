//! x87 on scalar doubles, the way the interpreter models it: values are
//! `f64`, TOP lives in the status word, a tag bit per physical register says
//! it holds a value, and of the exception bits only IE is tracked. Blocks
//! run x87 arithmetic with the host's default FPCR (as the interpreter's own
//! `f64` arithmetic does) and keep FPSR around it, so MXCSR sees nothing.
//! Nothing here touches NZCV except FCOMI and FUCOMI, which define it.

use a64::{Cond, Fcm, Logic, Reg, Shift, Size, V, Width};
use x86::decode::{Insn, Map};

use super::vector::CTX;
use super::{A, ADDR, B, FlagUse, OLD_AF, OLD_NZCV, T, Translator, U};
use crate::layout;

const ST: i32 = layout::X87_ST as i32;
const FSW: i32 = layout::X87_FSW as i32;
const FCW: i32 = layout::X87_FCW as i32;
const FTAG: i32 = layout::X87_FTAG as i32;
const FPCR_SSE: i32 = layout::FPCR_SSE as i32;
const FPSR_KEPT: i32 = layout::FPSR_KEPT as i32;

/// The status word and TOP while an x87 instruction is translated.
const F: Reg = OLD_NZCV;
const TOP: Reg = OLD_AF;

const VA: V = V(0);
const VB: V = V(1);
const VR: V = V(2);
const VT: V = V(3);

/// FMUL, FDIV, FADD, FSUB for `fp2`.
const MUL: u32 = 0;
const DIV: u32 = 1;
const ADD: u32 = 2;
const SUB: u32 = 3;

/// Whether this module handles `i`, and its flag use.
pub(super) fn classify(i: &Insn) -> Option<(bool, FlagUse)> {
    if i.map != Map::Primary || !(0xd8..=0xdf).contains(&i.op) {
        return None;
    }
    let none = FlagUse {
        reads: false,
        writes: false,
    };
    let ext = i.ext();
    let r = i.modrm? & 7;
    let arith = matches!(ext, 0 | 1 | 4..=7);
    let ok = match (i.op, i.mem.is_some()) {
        (0xd8 | 0xda | 0xdc | 0xde, true) => arith,
        (0xd9, true) => {
            matches!(ext, 0 | 2 | 3 | 5 | 7) || (matches!(ext, 4 | 6) && !i.opsize_prefix)
        }
        (0xdb, true) => matches!(ext, 0..=3 | 5 | 7),
        (0xdd, true) => matches!(ext, 0..=3 | 7),
        (0xdf, true) => matches!(ext, 0..=3 | 5 | 7),
        (0xd8 | 0xdc | 0xde, false) => arith,
        (0xd9, false) => match ext {
            0 | 1 => true,
            4 => r <= 1,
            5 => r <= 6,
            7 => r == 2 || r == 5,
            _ => false,
        },
        (0xda, false) => ext <= 3,
        (0xdb, false) => ext <= 3 || matches!(ext, 5 | 6),
        (0xdd, false) => matches!(ext, 2 | 3),
        (0xdf, false) => matches!(ext, 5 | 6) || (ext == 4 && r == 0),
        _ => false,
    };
    if !ok {
        return None;
    }
    let flags = match (i.op, i.mem.is_some(), ext) {
        (0xda | 0xdb, false, 0..=3) => FlagUse {
            reads: true,
            writes: false,
        },
        (0xdb | 0xdf, false, 5 | 6) => FlagUse {
            reads: false,
            writes: true,
        },
        _ => none,
    };
    Some((false, flags))
}

/// The result of arithmetic operation `op` (a ModRM reg field) on the
/// destination `a` and `b`, as `fp2`'s operation and operand order.
fn arith(op: u8, a: V, b: V) -> (u32, V, V) {
    match op {
        0 => (ADD, a, b),
        1 => (MUL, a, b),
        4 => (SUB, a, b),
        5 => (SUB, b, a),
        6 => (DIV, a, b),
        _ => (DIV, b, a),
    }
}

impl Translator {
    /// Switches to the host's default FPCR for x87 arithmetic, keeping FPSR.
    fn enter_x87(&mut self) {
        if self.x87_mode {
            return;
        }
        self.x87_mode = true;
        self.a.mrs(T, a64::sysreg::FPSR);
        self.a.str(Width::X, T, CTX, FPSR_KEPT);
        self.a.msr(a64::sysreg::FPCR, Reg(31));
    }

    /// Puts back the FPCR that MXCSR gives and the FPSR kept by `enter_x87`;
    /// uses only `T`, and leaves NZCV alone.
    pub(super) fn leave_x87(&mut self) {
        if self.x87_mode {
            self.x87_mode = false;
            self.restore_fp();
        }
    }

    pub(super) fn restore_fp(&mut self) {
        self.a.ldr(Width::X, T, CTX, FPCR_SSE);
        self.a.msr(a64::sysreg::FPCR, T);
        self.a.ldr(Width::X, T, CTX, FPSR_KEPT);
        self.a.msr(a64::sysreg::FPSR, T);
    }

    /// `dst` = the physical register number of ST(`i`).
    fn phys(&mut self, dst: Reg, i: u8) {
        self.a.add_imm(Size::X, dst, TOP, i as u32);
        assert!(self.a.logic_imm(Logic::And, Size::X, dst, dst, 7));
    }

    /// `dst` = the context address of ST(`i`), less `ST`.
    fn st_slot(&mut self, dst: Reg, i: u8) {
        self.phys(dst, i);
        self.a.add_shift(Size::X, dst, CTX, dst, Shift::Lsl, 3);
    }

    fn load_st(&mut self, v: V, i: u8) {
        self.st_slot(U, i);
        self.a.ldr_d(v, U, ST);
    }

    /// ST(`i`) = `v`, marking the register as holding a value.
    fn store_st(&mut self, v: V, i: u8) {
        self.st_slot(U, i);
        self.a.str_d(v, U, ST);
        self.phys(U, i);
        self.tag(U, true);
    }

    /// Sets or clears the tag bit of the physical register in `p`.
    fn tag(&mut self, p: Reg, set: bool) {
        let a = &mut self.a;
        a.mov_imm(Size::X, A, 1);
        a.lslv(Size::X, A, A, p);
        a.ldr(Width::X, B, CTX, FTAG);
        if set {
            a.orr(Size::X, B, B, A);
        } else {
            a.bic(Size::X, B, B, A);
        }
        a.str(Width::X, B, CTX, FTAG);
    }

    fn set_top_field(&mut self) {
        self.a.bfi(Size::X, F, TOP, 11, 3);
    }

    fn push(&mut self, v: V) {
        self.a.sub_imm(Size::X, TOP, TOP, 1);
        assert!(self.a.logic_imm(Logic::And, Size::X, TOP, TOP, 7));
        self.set_top_field();
        self.store_st(v, 0);
    }

    fn pop(&mut self) {
        self.tag(TOP, false);
        self.a.add_imm(Size::X, TOP, TOP, 1);
        assert!(self.a.logic_imm(Logic::And, Size::X, TOP, TOP, 7));
        self.set_top_field();
    }

    /// Sets IE when `res` is NaN but none of `inputs` is.
    fn invalid_if_nan(&mut self, res: V, inputs: &[V]) {
        self.a.fcm(Fcm::Eq, true, VT, res, res);
        self.a.fmov_xd(B, VT);
        self.a.mvn(Size::X, B, B);
        for &v in inputs {
            self.a.fcm(Fcm::Eq, true, VT, v, v);
            self.a.fmov_xd(A, VT);
            self.a.and(Size::X, B, B, A);
        }
        assert!(self.a.logic_imm(Logic::And, Size::X, B, B, 1));
        self.a.orr(Size::X, F, F, B);
    }

    /// Translates `i`, which [`classify`] accepted; memory operands are at x19.
    pub(super) fn x87(&mut self, i: &Insn, next: u64) {
        self.enter_x87();
        if let Some(m) = i.mem {
            self.addr(&m, next);
        }
        self.a.ldr(Width::X, F, CTX, FSW);
        self.a.ubfx(Size::X, TOP, F, 11, 3);
        let ext = i.ext();
        let r = i.modrm.unwrap_or(0) & 7;
        match (i.op, i.mem.is_some()) {
            (_, true) => self.x87_memory(i.op, ext),
            (_, false) => self.x87_register(i.op, ext, r),
        }
        self.a.str(Width::X, F, CTX, FSW);
    }

    /// `VB` = the memory operand of an arithmetic instruction, as a double.
    fn arith_operand(&mut self, op: u8) {
        match op {
            0xd8 => {
                self.a.ldr_s(VB, ADDR, 0);
                self.a.fp1(false, 5, VB, VB);
            }
            0xdc => self.a.ldr_d(VB, ADDR, 0),
            0xda => {
                self.a.ldrs(Width::W, Size::X, A, ADDR, 0);
                self.a.scvtf(true, Size::X, VB, A);
            }
            _ => {
                self.a.ldrs(Width::H, Size::X, A, ADDR, 0);
                self.a.scvtf(true, Size::X, VB, A);
            }
        }
    }

    /// ST(0) = ST(0) `op` VB.
    fn arith_st0(&mut self, op: u8) {
        self.load_st(VA, 0);
        let (f, n, m) = arith(op, VA, VB);
        self.a.fp2(true, f, VR, n, m);
        self.invalid_if_nan(VR, &[VA, VB]);
        self.store_st(VR, 0);
    }

    fn x87_memory(&mut self, op: u8, ext: u8) {
        match (op, ext) {
            (0xd8 | 0xda | 0xdc | 0xde, _) => {
                self.arith_operand(op);
                self.arith_st0(ext);
            }
            (0xd9, 0) => {
                self.a.ldr_s(VA, ADDR, 0);
                self.a.fp1(false, 5, VA, VA);
                self.push(VA);
            }
            (0xdd, 0) => {
                self.a.ldr_d(VA, ADDR, 0);
                self.push(VA);
            }
            (0xdb, 0) | (0xdf, 0) | (0xdf, 5) => {
                match (op, ext) {
                    (0xdb, _) => self.a.ldrs(Width::W, Size::X, A, ADDR, 0),
                    (_, 0) => self.a.ldrs(Width::H, Size::X, A, ADDR, 0),
                    _ => self.a.ldr(Width::X, A, ADDR, 0),
                }
                self.a.scvtf(true, Size::X, VA, A);
                self.push(VA);
            }
            (0xdb, 5) => {
                self.load_f80();
                self.push(VA);
            }
            (0xd9, 2 | 3) | (0xdd, 2 | 3) => {
                self.load_st(VA, 0);
                if op == 0xd9 {
                    self.a.fp1(true, 4, VA, VA);
                    self.a.str_s(VA, ADDR, 0);
                } else {
                    self.a.str_d(VA, ADDR, 0);
                }
                if ext == 3 {
                    self.pop();
                }
            }
            (0xdb, 7) => {
                self.load_st(VA, 0);
                self.store_f80();
                self.pop();
            }
            (0xdb | 0xdd | 0xdf, 1..=3) | (0xdf, 7) => {
                let size = match (op, ext) {
                    (0xdb, _) => 4,
                    (0xdd, _) | (_, 7) => 8,
                    _ => 2,
                };
                self.store_int(size, ext == 1 || op == 0xdd);
                if ext != 2 {
                    self.pop();
                }
            }
            (0xd9, 4) => self.load_env(),
            (0xd9, 6) => self.store_env(),
            (0xd9, 5) => {
                self.a.ldr(Width::H, A, ADDR, 0);
                self.a.str(Width::X, A, CTX, FCW);
            }
            (0xd9, 7) => {
                self.a.ldr(Width::X, A, CTX, FCW);
                self.a.str(Width::H, A, ADDR, 0);
            }
            (0xdd, 7) => self.a.str(Width::H, F, ADDR, 0),
            _ => unreachable!("classify accepted an x87 form it does not handle"),
        }
    }

    fn x87_register(&mut self, op: u8, ext: u8, r: u8) {
        match (op, ext) {
            (0xd8, _) => {
                self.load_st(VB, r);
                self.arith_st0(ext);
            }
            (0xdc | 0xde, _) => {
                let swapped = match ext {
                    4 => 5,
                    5 => 4,
                    6 => 7,
                    7 => 6,
                    o => o,
                };
                self.load_st(VA, r);
                self.load_st(VB, 0);
                let (f, n, m) = arith(swapped, VA, VB);
                self.a.fp2(true, f, VR, n, m);
                self.invalid_if_nan(VR, &[VA, VB]);
                self.store_st(VR, r);
                if op == 0xde {
                    self.pop();
                }
            }
            (0xd9, 0) => {
                self.load_st(VA, r);
                self.push(VA);
            }
            (0xd9, 1) => {
                self.load_st(VA, 0);
                self.load_st(VB, r);
                self.store_st(VB, 0);
                self.store_st(VA, r);
            }
            (0xd9, 4) => {
                self.load_st(VA, 0);
                self.a.fp1(true, if r == 0 { 2 } else { 1 }, VA, VA);
                self.store_st(VA, 0);
            }
            (0xd9, 5) => {
                use std::f64::consts::{LN_2, LOG2_10, LOG2_E, LOG10_2, PI};
                let v = [1.0, LOG2_10, LOG2_E, PI, LOG10_2, LN_2, 0.0][r as usize];
                self.a.mov_imm(Size::X, A, f64::to_bits(v));
                self.a.fmov_dx(VA, A);
                self.push(VA);
            }
            (0xd9, 7) if r == 2 => {
                self.load_st(VA, 0);
                self.a.fp1(true, 3, VR, VA);
                self.invalid_if_nan(VR, &[VA]);
                self.store_st(VR, 0);
            }
            (0xd9, 7) => self.fscale(),
            (0xda | 0xdb, 3) => {
                let skip = self.a.label();
                self.parity_branch(0xa, op == 0xda, skip);
                self.load_st(VA, r);
                self.store_st(VA, 0);
                self.a.bind(skip);
            }
            (0xda | 0xdb, _) if ext <= 2 => {
                let c = match ext {
                    0 => Cond::Lo,
                    1 => Cond::Eq,
                    _ => Cond::Ls,
                };
                let c = if op == 0xdb { c.invert() } else { c };
                let skip = self.a.label();
                self.a.b_cond(c.invert(), skip);
                self.load_st(VA, r);
                self.store_st(VA, 0);
                self.a.bind(skip);
            }
            (0xdb | 0xdf, 5 | 6) => {
                self.load_st(VA, 0);
                self.load_st(VB, r);
                self.a.fcmp(true, false, VA, VB);
                self.compare_flags();
                assert!(self.a.logic_imm(Logic::And, Size::X, F, F, !(1u64 << 9)));
                if op == 0xdf {
                    self.pop();
                }
            }
            (0xdd, 2 | 3) => {
                self.load_st(VA, 0);
                self.store_st(VA, r);
                if ext == 3 {
                    self.pop();
                }
            }
            (0xdf, 4) => self.a.bfi(Size::X, Reg(0), F, 0, 16),
            _ => unreachable!("classify accepted an x87 form it does not handle"),
        }
    }

    /// FSCALE: ST(0) * 2^trunc(ST(1)), the exponent clamped to +-4000 and
    /// applied in steps of at most 1000, as the interpreter does.
    fn fscale(&mut self) {
        let (inf, top, done, store) = (
            self.a.label(),
            self.a.label(),
            self.a.label(),
            self.a.label(),
        );
        self.load_st(VA, 0);
        self.load_st(VB, 1);
        let a = &mut self.a;
        a.fmov_xd(A, VB);
        a.lsl_imm(Size::X, B, A, 1);
        a.mov_imm(Size::X, T, 0xffe0_0000_0000_0000);
        a.eor(Size::X, B, B, T);
        a.cbz(Size::X, B, inf);
        a.fp1(true, 11, VT, VB);
        a.mov_imm(Size::X, T, (-4000f64).to_bits());
        a.fmov_dx(VR, T);
        a.fp2(true, 4, VT, VT, VR);
        a.mov_imm(Size::X, T, 4000f64.to_bits());
        a.fmov_dx(VR, T);
        a.fp2(true, 5, VT, VT, VR);
        a.ins(3, VR, 0, VA, 0);
        a.bind(top);
        a.fcvtzs(true, Size::X, T, VT);
        a.cbz(Size::X, T, done);
        a.mov_imm(Size::X, T, (-1000f64).to_bits());
        a.fmov_dx(V(4), T);
        a.fp2(true, 4, V(4), VT, V(4));
        a.mov_imm(Size::X, T, 1000f64.to_bits());
        a.fmov_dx(V(5), T);
        a.fp2(true, 5, V(4), V(4), V(5));
        a.fcvtzs(true, Size::X, T, V(4));
        a.add_imm(Size::X, T, T, 1023);
        a.lsl_imm(Size::X, T, T, 52);
        a.fmov_dx(V(5), T);
        a.fp2(true, MUL, VR, VR, V(5));
        a.fp2(true, SUB, VT, VT, V(4));
        a.b(top);
        a.bind(inf);
        a.mov_imm(Size::X, T, f64::INFINITY.to_bits());
        a.tbz(A, 63, store);
        a.mov_imm(Size::X, T, 0);
        a.bind(store);
        a.fmov_dx(VT, T);
        a.fp2(true, MUL, VR, VA, VT);
        a.bind(done);
        self.invalid_if_nan(VR, &[VA, VB]);
        self.store_st(VR, 0);
    }

    /// FIST, FISTP and FISTTP (`truncate`) of `size` bytes.
    fn store_int(&mut self, size: u8, truncate: bool) {
        self.load_st(VA, 0);
        let (rounded, out, stored) = (self.a.label(), self.a.label(), self.a.label());
        if truncate {
            self.a.fp1(true, 11, VA, VA);
        } else {
            let (up_or_zero, zero, down) = (self.a.label(), self.a.label(), self.a.label());
            self.a.ldr(Width::X, A, CTX, FCW);
            self.a.tbnz(A, 11, up_or_zero);
            self.a.tbnz(A, 10, down);
            self.a.fp1(true, 8, VA, VA);
            self.a.b(rounded);
            self.a.bind(down);
            self.a.fp1(true, 10, VA, VA);
            self.a.b(rounded);
            self.a.bind(up_or_zero);
            self.a.tbnz(A, 10, zero);
            self.a.fp1(true, 9, VA, VA);
            self.a.b(rounded);
            self.a.bind(zero);
            self.a.fp1(true, 11, VA, VA);
        }
        self.a.bind(rounded);
        let limit = 2f64.powi(size as i32 * 8 - 1);
        let a = &mut self.a;
        a.mov_imm(Size::X, T, (-limit).to_bits());
        a.fmov_dx(VB, T);
        a.fcm(Fcm::Ge, true, VR, VA, VB);
        a.mov_imm(Size::X, T, limit.to_bits());
        a.fmov_dx(VB, T);
        a.fcm(Fcm::Gt, true, VT, VB, VA);
        a.fmov_xd(A, VR);
        a.fmov_xd(B, VT);
        a.and(Size::X, A, A, B);
        a.cbz(Size::X, A, out);
        a.fcvtzs(true, Size::X, A, VA);
        a.b(stored);
        a.bind(out);
        a.mov_imm(Size::X, A, 1u64 << (size as u32 * 8 - 1));
        assert!(a.logic_imm(Logic::Orr, Size::X, F, F, 1));
        a.bind(stored);
        let w = match size {
            2 => Width::H,
            4 => Width::W,
            _ => Width::X,
        };
        a.str(w, A, ADDR, 0);
    }

    /// FLDENV of the 28-byte environment at x19: the control and status
    /// words, and a tag bit for each register whose 2-bit tag is not empty.
    fn load_env(&mut self) {
        let a = &mut self.a;
        a.ldr(Width::H, A, ADDR, 0);
        a.ldr(Width::H, F, ADDR, 4);
        a.ldr(Width::H, T, ADDR, 8);
        a.logic_shift(Logic::And, false, Size::X, U, T, T, Shift::Lsr, 1);
        for (shift, keep) in [
            (0, 0x5555_5555_5555_5555),
            (1, 0x3333_3333_3333_3333),
            (2, 0x0f0f_0f0f_0f0f_0f0f),
            (4, 0x00ff_00ff_00ff_00ff),
        ] {
            if shift != 0 {
                a.logic_shift(Logic::Orr, false, Size::X, U, U, U, Shift::Lsr, shift);
            }
            assert!(a.logic_imm(Logic::And, Size::X, U, U, keep));
        }
        assert!(a.logic_imm(Logic::Eor, Size::X, U, U, 0xff));
        a.str(Width::X, A, CTX, FCW);
        a.str(Width::X, U, CTX, FTAG);
    }

    /// FNSTENV of the 28-byte environment at x19 (the instruction and
    /// operand pointers as zeros), then masks every exception.
    fn store_env(&mut self) {
        let a = &mut self.a;
        a.ldr(Width::X, U, CTX, FTAG);
        a.mov_imm(Size::X, T, 0);
        for p in 0..8 {
            let put = a.label();
            a.mov_imm(Size::X, B, 3);
            a.tbz(U, p, put);
            a.ldr(Width::X, A, CTX, ST + 8 * p as i32);
            a.lsl_imm(Size::X, A, A, 1);
            a.mov_imm(Size::X, B, 1);
            a.cbz(Size::X, A, put);
            a.lsr_imm(Size::X, A, A, 53);
            a.mov_imm(Size::X, B, 2);
            a.cbz(Size::X, A, put);
            a.add_imm(Size::X, A, A, 1);
            a.tbnz(A, 11, put);
            a.mov_imm(Size::X, B, 0);
            a.bind(put);
            a.logic_shift(Logic::Orr, false, Size::X, T, T, B, Shift::Lsl, 2 * p);
        }
        a.ldr(Width::X, A, CTX, FCW);
        for (k, v) in [A, F, T].into_iter().enumerate() {
            assert!(a.logic_imm(Logic::Orr, Size::W, B, v, 0xffff_0000));
            a.str(Width::W, B, ADDR, 4 * k as i32);
        }
        for k in 3..7 {
            a.str(Width::W, Reg(31), ADDR, 4 * k);
        }
        assert!(a.logic_imm(Logic::Orr, Size::X, A, A, 0x3f));
        a.str(Width::X, A, CTX, FCW);
    }

    /// `VA` = the 80-bit value at x19 rounded to the nearest double, as the
    /// interpreter's `from_extended` does: unnormals give the default NaN,
    /// NaNs keep their sign and top payload bits, and results too large or
    /// too small become infinities, subnormals or zeros.
    fn load_f80(&mut self) {
        let [
            denormal,
            special,
            nan,
            normalized,
            tiny,
            round,
            inf,
            zero,
            sign,
            done,
        ] = std::array::from_fn(|_| self.a.label());
        let a = &mut self.a;
        a.ldr(Width::X, A, ADDR, 0);
        a.ldr(Width::H, B, ADDR, 8);
        a.ubfx(Size::X, T, B, 0, 15);
        a.lsr_imm(Size::X, B, B, 15);
        a.cbz(Size::X, T, denormal);
        a.tbz(A, 63, nan);
        assert!(a.logic_imm(Logic::Eor, Size::X, U, T, 0x7fff));
        a.cbz(Size::X, U, special);

        // A: the significand with its top bit set; T: its biased exponent,
        // which may be zero or negative after normalising a denormal.
        a.bind(normalized);
        a.mov_imm(Size::X, U, 15361);
        a.sub(Size::X, U, U, T);
        a.tbz(U, 63, tiny);
        a.neg(Size::X, T, U);
        a.sub_imm(Size::X, U, T, 0x7ff);
        a.tbz(U, 63, inf);
        a.mov_imm(Size::X, U, 0);
        a.b(round);
        a.bind(tiny);
        a.sub_imm(Size::X, T, U, 54);
        a.tbz(T, 63, zero);
        a.mov_imm(Size::X, T, 0);

        // Rounds A to nearest even after dropping 11 + U bits, then adds
        // the exponent in T: the significand's top bit carries into it.
        a.bind(round);
        a.add_imm(Size::X, U, U, 10);
        a.lsrv(Size::X, ADDR, A, U);
        a.neg(Size::X, U, U);
        a.lslv(Size::X, A, A, U);
        a.neg(Size::X, U, A);
        a.orr(Size::X, A, A, U);
        a.lsr_imm(Size::X, A, A, 63);
        a.logic_shift(Logic::Orr, false, Size::X, A, A, ADDR, Shift::Lsr, 1);
        a.and(Size::X, A, A, ADDR);
        assert!(a.logic_imm(Logic::And, Size::X, A, A, 1));
        a.add_shift(Size::X, A, A, ADDR, Shift::Lsr, 1);
        a.add_shift(Size::X, A, A, T, Shift::Lsl, 52);
        a.lsr_imm(Size::X, U, A, 52);
        a.sub_imm(Size::X, U, U, 0x7ff);
        a.tbz(U, 63, inf);
        a.b(sign);

        a.bind(denormal);
        a.cbz(Size::X, A, zero);
        a.clz(Size::X, U, A);
        a.lslv(Size::X, A, A, U);
        a.mov_imm(Size::X, T, 1);
        a.sub(Size::X, T, T, U);
        a.b(normalized);

        a.bind(special);
        a.lsl_imm(Size::X, U, A, 1);
        a.cbz(Size::X, U, inf);
        a.ubfx(Size::X, A, A, 11, 52);
        assert!(a.logic_imm(Logic::Orr, Size::X, A, A, 0x7ff8 << 48));
        a.b(sign);

        a.bind(nan);
        a.mov_imm(Size::X, A, f64::NAN.to_bits());
        a.b(done);

        a.bind(inf);
        a.mov_imm(Size::X, A, f64::INFINITY.to_bits());
        a.b(sign);

        a.bind(zero);
        a.mov_imm(Size::X, A, 0);
        a.bind(sign);
        a.logic_shift(Logic::Orr, false, Size::X, A, A, B, Shift::Lsl, 63);
        a.bind(done);
        a.fmov_dx(VA, A);
    }

    /// Stores `VA` at x19 as an 80-bit value.
    fn store_f80(&mut self) {
        let (low, special, write) = (self.a.label(), self.a.label(), self.a.label());
        let a = &mut self.a;
        a.fmov_xd(A, VA);
        a.ubfx(Size::X, T, A, 52, 11);
        a.lsr_imm(Size::X, B, A, 63);
        a.lsl_imm(Size::X, B, B, 15);
        a.cbz(Size::X, T, low);
        a.lsl_imm(Size::X, U, A, 11);
        assert!(a.logic_imm(Logic::Orr, Size::X, U, U, 1 << 63));
        assert!(a.logic_imm(Logic::Eor, Size::X, A, T, 0x7ff));
        a.cbz(Size::X, A, special);
        a.mov_imm(Size::X, A, 15360);
        a.add(Size::X, T, T, A);
        a.orr(Size::X, B, B, T);
        a.b(write);
        a.bind(special);
        assert!(a.logic_imm(Logic::Orr, Size::X, B, B, 0x7fff));
        a.b(write);
        a.bind(low);
        a.lsl_imm(Size::X, U, A, 1);
        a.cbz(Size::X, U, write);
        a.ubfx(Size::X, A, A, 0, 52);
        a.clz(Size::X, T, A);
        a.lslv(Size::X, U, A, T);
        a.mov_imm(Size::X, A, 15372);
        a.sub(Size::X, A, A, T);
        a.orr(Size::X, B, B, A);
        a.bind(write);
        a.str(Width::X, U, ADDR, 0);
        a.str(Width::H, B, ADDR, 8);
    }
}
