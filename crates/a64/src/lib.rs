//! A small ARM64 (A64) assembler for generating JIT code, following the
//! encodings in the Arm Architecture Reference Manual (DDI 0487), chapter C4.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg(pub u8);

/// Register 31 reads as zero in most data-processing instructions.
pub const ZR: Reg = Reg(31);
/// Register 31 is the stack pointer in address bases and add/sub immediate.
pub const SP: Reg = Reg(31);

/// A SIMD&FP register (V0..V31).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V(pub u8);

pub const fn r(n: u8) -> Reg {
    Reg(n)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    W,
    X,
}

impl Size {
    fn sf(self) -> u32 {
        match self {
            Size::W => 0,
            Size::X => 1 << 31,
        }
    }

    fn bits(self) -> u32 {
        match self {
            Size::W => 32,
            Size::X => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    Eq = 0,
    Ne,
    Hs,
    Lo,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
}

impl Cond {
    pub fn invert(self) -> Cond {
        assert!(self != Cond::Al);
        // SAFETY: flipping bit 0 of a value in 0..=13 stays in 0..=13.
        unsafe { std::mem::transmute(self as u8 ^ 1) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extend {
    Uxtb = 0,
    Uxth = 1,
    Uxtw = 2,
    Uxtx = 3,
    Sxtb = 4,
    Sxth = 5,
    Sxtw = 6,
    Sxtx = 7,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fcm {
    Eq,
    Ge,
    Gt,
    /// |rn| > |rm|.
    Agt,
}

impl Fcm {
    fn scalar(self) -> u32 {
        match self {
            Fcm::Eq => 0x5e20_e400,
            Fcm::Ge => 0x7e20_e400,
            Fcm::Gt => 0x7ea0_e400,
            Fcm::Agt => 0x7ea0_ec00,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logic {
    And,
    Orr,
    Eor,
    Ands,
}

impl Logic {
    fn opc(self) -> u32 {
        match self {
            Logic::And => 0,
            Logic::Orr => 1,
            Logic::Eor => 2,
            Logic::Ands => 3,
        }
    }
}

/// Load/store access width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    B,
    H,
    W,
    X,
}

impl Width {
    fn size(self) -> u32 {
        match self {
            Width::B => 0,
            Width::H => 1,
            Width::W => 2,
            Width::X => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(usize);

#[derive(Clone, Copy)]
enum Fix {
    Imm26,
    Imm19,
    Imm14,
    Adr,
}

/// Encodes `imm` as an A64 logical (bitmask) immediate, returning the
/// N:immr:imms field, or `None` if it has no such encoding.
pub fn encode_bitmask(imm: u64, size: Size) -> Option<u32> {
    let width = size.bits();
    let imm = if width == 32 {
        if imm >> 32 != 0 {
            return None;
        }
        imm | imm << 32
    } else {
        imm
    };
    if imm == 0 || imm == u64::MAX {
        return None;
    }
    let mut esize = 64u32;
    while esize > 2 {
        let half = esize / 2;
        let mask = (1u64 << half) - 1;
        if (imm & mask) != ((imm >> half) & mask) {
            break;
        }
        esize = half;
    }
    let emask = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let elem = imm & emask;
    let ones = elem.count_ones();
    let rot = (0..esize).find(|&r| {
        let rotated = ((elem >> r) | (elem << ((esize - r) % esize))) & emask;
        rotated == (1u64 << ones) - 1
    })?;
    let immr = (esize - rot) % esize;
    let n = (esize == 64) as u32;
    let imms = ((!(esize * 2 - 1)) & 0x3f) | (ones - 1);
    if width == 32 && n == 1 {
        return None;
    }
    Some(n << 12 | immr << 6 | (imms & 0x3f))
}

#[derive(Default)]
pub struct Asm {
    pub code: Vec<u32>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, Label, Fix)>,
}

fn reg(r: Reg) -> u32 {
    debug_assert!(r.0 < 32);
    r.0 as u32
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit(&mut self, word: u32) {
        self.code.push(word);
    }

    pub fn pos(&self) -> usize {
        self.code.len()
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, l: Label) {
        assert!(self.labels[l.0].is_none(), "label bound twice");
        self.labels[l.0] = Some(self.code.len());
    }

    /// Drops the code from `pos` on, with its label references and bindings.
    pub fn truncate(&mut self, pos: usize) {
        self.code.truncate(pos);
        self.fixups.retain(|&(at, _, _)| at < pos);
        for l in &mut self.labels {
            if l.is_some_and(|at| at > pos) {
                *l = None;
            }
        }
    }

    fn branch_to(&mut self, word: u32, l: Label, fix: Fix) {
        self.fixups.push((self.code.len(), l, fix));
        self.emit(word);
    }

    /// Resolves label references and returns the code. Panics on unbound
    /// labels or out-of-range branches.
    pub fn finish(mut self) -> Vec<u32> {
        for &(at, l, fix) in &self.fixups {
            let target = self.labels[l.0].unwrap_or_else(|| {
                panic!("unbound label {} at word {at} of {}", l.0, self.code.len())
            });
            let delta = target as i64 - at as i64;
            let w = &mut self.code[at];
            match fix {
                Fix::Imm26 => {
                    assert!((-(1 << 25)..(1 << 25)).contains(&delta));
                    *w |= (delta as u32) & 0x03ff_ffff;
                }
                Fix::Imm19 => {
                    assert!((-(1 << 18)..(1 << 18)).contains(&delta));
                    *w |= ((delta as u32) & 0x7ffff) << 5;
                }
                Fix::Imm14 => {
                    assert!((-(1 << 13)..(1 << 13)).contains(&delta));
                    *w |= ((delta as u32) & 0x3fff) << 5;
                }
                Fix::Adr => {
                    let bytes = delta * 4;
                    assert!((-(1 << 20)..(1 << 20)).contains(&bytes));
                    let b = bytes as u32;
                    *w |= (b & 3) << 29 | ((b >> 2) & 0x7ffff) << 5;
                }
            }
        }
        self.code
    }

    fn addsub_imm(&mut self, sz: Size, op: u32, s: u32, rd: Reg, rn: Reg, imm: u32) {
        let (imm12, sh) = if imm < 4096 {
            (imm, 0)
        } else {
            assert!(
                imm & 0xfff == 0 && imm >> 12 < 4096,
                "add/sub immediate {imm:#x} not encodable"
            );
            (imm >> 12, 1)
        };
        self.emit(
            sz.sf()
                | op << 30
                | s << 29
                | 0x1100_0000
                | sh << 22
                | imm12 << 10
                | reg(rn) << 5
                | reg(rd),
        );
    }

    pub fn add_imm(&mut self, sz: Size, rd: Reg, rn: Reg, imm: u32) {
        self.addsub_imm(sz, 0, 0, rd, rn, imm)
    }

    pub fn adds_imm(&mut self, sz: Size, rd: Reg, rn: Reg, imm: u32) {
        self.addsub_imm(sz, 0, 1, rd, rn, imm)
    }

    pub fn sub_imm(&mut self, sz: Size, rd: Reg, rn: Reg, imm: u32) {
        self.addsub_imm(sz, 1, 0, rd, rn, imm)
    }

    pub fn subs_imm(&mut self, sz: Size, rd: Reg, rn: Reg, imm: u32) {
        self.addsub_imm(sz, 1, 1, rd, rn, imm)
    }

    pub fn cmp_imm(&mut self, sz: Size, rn: Reg, imm: u32) {
        self.subs_imm(sz, ZR, rn, imm)
    }

    pub fn cmn_imm(&mut self, sz: Size, rn: Reg, imm: u32) {
        self.adds_imm(sz, ZR, rn, imm)
    }

    #[allow(clippy::too_many_arguments)]
    fn addsub_shifted(
        &mut self,
        sz: Size,
        op: u32,
        s: u32,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: Shift,
        amount: u32,
    ) {
        assert!(shift != Shift::Ror && amount < sz.bits());
        self.emit(
            sz.sf()
                | op << 30
                | s << 29
                | 0x0b00_0000
                | (shift as u32) << 22
                | reg(rm) << 16
                | amount << 10
                | reg(rn) << 5
                | reg(rd),
        );
    }

    pub fn add(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.addsub_shifted(sz, 0, 0, rd, rn, rm, Shift::Lsl, 0)
    }

    pub fn adds(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.addsub_shifted(sz, 0, 1, rd, rn, rm, Shift::Lsl, 0)
    }

    pub fn sub(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.addsub_shifted(sz, 1, 0, rd, rn, rm, Shift::Lsl, 0)
    }

    pub fn subs(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.addsub_shifted(sz, 1, 1, rd, rn, rm, Shift::Lsl, 0)
    }

    pub fn add_shift(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, shift: Shift, amount: u32) {
        self.addsub_shifted(sz, 0, 0, rd, rn, rm, shift, amount)
    }

    pub fn adds_shift(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, shift: Shift, amount: u32) {
        self.addsub_shifted(sz, 0, 1, rd, rn, rm, shift, amount)
    }

    pub fn sub_shift(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, shift: Shift, amount: u32) {
        self.addsub_shifted(sz, 1, 0, rd, rn, rm, shift, amount)
    }

    pub fn subs_shift(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, shift: Shift, amount: u32) {
        self.addsub_shifted(sz, 1, 1, rd, rn, rm, shift, amount)
    }

    pub fn cmp(&mut self, sz: Size, rn: Reg, rm: Reg) {
        self.subs(sz, ZR, rn, rm)
    }

    pub fn cmn(&mut self, sz: Size, rn: Reg, rm: Reg) {
        self.adds(sz, ZR, rn, rm)
    }

    pub fn neg(&mut self, sz: Size, rd: Reg, rm: Reg) {
        self.sub(sz, rd, ZR, rm)
    }

    pub fn negs(&mut self, sz: Size, rd: Reg, rm: Reg) {
        self.subs(sz, rd, ZR, rm)
    }

    /// Add/sub with extended register; `rn` may be SP.
    #[allow(clippy::too_many_arguments)]
    pub fn addsub_ext(
        &mut self,
        sz: Size,
        sub: bool,
        s: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        ext: Extend,
        amount: u32,
    ) {
        assert!(amount <= 4);
        self.emit(
            sz.sf()
                | (sub as u32) << 30
                | (s as u32) << 29
                | 0x0b20_0000
                | reg(rm) << 16
                | (ext as u32) << 13
                | amount << 10
                | reg(rn) << 5
                | reg(rd),
        );
    }

    fn adc_like(&mut self, sz: Size, op: u32, s: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(
            sz.sf() | op << 30 | s << 29 | 0x1a00_0000 | reg(rm) << 16 | reg(rn) << 5 | reg(rd),
        );
    }

    pub fn adc(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.adc_like(sz, 0, 0, rd, rn, rm)
    }

    pub fn adcs(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.adc_like(sz, 0, 1, rd, rn, rm)
    }

    pub fn sbc(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.adc_like(sz, 1, 0, rd, rn, rm)
    }

    pub fn sbcs(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.adc_like(sz, 1, 1, rd, rn, rm)
    }

    /// Logical operation with an immediate. Returns false, emitting nothing,
    /// when `imm` is not a bitmask immediate.
    #[must_use]
    pub fn logic_imm(&mut self, op: Logic, sz: Size, rd: Reg, rn: Reg, imm: u64) -> bool {
        let Some(enc) = encode_bitmask(imm, sz) else {
            return false;
        };
        self.emit(sz.sf() | op.opc() << 29 | 0x1200_0000 | enc << 10 | reg(rn) << 5 | reg(rd));
        true
    }

    /// Logical operation on a shifted register; `invert` gives BIC/ORN/EON/BICS.
    #[allow(clippy::too_many_arguments)]
    pub fn logic_shift(
        &mut self,
        op: Logic,
        invert: bool,
        sz: Size,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: Shift,
        amount: u32,
    ) {
        assert!(amount < sz.bits());
        self.emit(
            sz.sf()
                | op.opc() << 29
                | 0x0a00_0000
                | (shift as u32) << 22
                | (invert as u32) << 21
                | reg(rm) << 16
                | amount << 10
                | reg(rn) << 5
                | reg(rd),
        );
    }

    pub fn logic(&mut self, op: Logic, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.logic_shift(op, false, sz, rd, rn, rm, Shift::Lsl, 0)
    }

    pub fn and(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.logic(Logic::And, sz, rd, rn, rm)
    }

    pub fn orr(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.logic(Logic::Orr, sz, rd, rn, rm)
    }

    pub fn eor(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.logic(Logic::Eor, sz, rd, rn, rm)
    }

    pub fn ands(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.logic(Logic::Ands, sz, rd, rn, rm)
    }

    pub fn bic(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.logic_shift(Logic::And, true, sz, rd, rn, rm, Shift::Lsl, 0)
    }

    pub fn tst(&mut self, sz: Size, rn: Reg, rm: Reg) {
        self.ands(sz, ZR, rn, rm)
    }

    pub fn mvn(&mut self, sz: Size, rd: Reg, rm: Reg) {
        self.logic_shift(Logic::Orr, true, sz, rd, ZR, rm, Shift::Lsl, 0)
    }

    /// Register move; register 31 means ZR here, so use `mov_sp` for SP.
    pub fn mov(&mut self, sz: Size, rd: Reg, rm: Reg) {
        self.orr(sz, rd, ZR, rm)
    }

    pub fn mov_sp(&mut self, rd: Reg, rn: Reg) {
        self.add_imm(Size::X, rd, rn, 0)
    }

    fn movw(&mut self, sz: Size, opc: u32, rd: Reg, imm16: u32, hw: u32) {
        assert!(imm16 < 0x10000 && hw * 16 < sz.bits());
        self.emit(sz.sf() | opc << 29 | 0x1280_0000 | hw << 21 | imm16 << 5 | reg(rd));
    }

    pub fn movz(&mut self, sz: Size, rd: Reg, imm16: u32, hw: u32) {
        self.movw(sz, 2, rd, imm16, hw)
    }

    pub fn movn(&mut self, sz: Size, rd: Reg, imm16: u32, hw: u32) {
        self.movw(sz, 0, rd, imm16, hw)
    }

    pub fn movk(&mut self, sz: Size, rd: Reg, imm16: u32, hw: u32) {
        self.movw(sz, 3, rd, imm16, hw)
    }

    /// Loads a constant using the shortest MOVZ/MOVN/ORR-immediate/MOVK sequence.
    pub fn mov_imm(&mut self, sz: Size, rd: Reg, imm: u64) {
        let imm = if sz == Size::W {
            imm & 0xffff_ffff
        } else {
            imm
        };
        let chunks = (sz.bits() / 16) as usize;
        let part = |i: usize| ((imm >> (16 * i)) & 0xffff) as u32;
        let zeros = (0..chunks).filter(|&i| part(i) == 0).count();
        let ones = (0..chunks).filter(|&i| part(i) == 0xffff).count();
        if zeros < chunks - 1 && ones < chunks - 1 && encode_bitmask(imm, sz).is_some() {
            let ok = self.logic_imm(Logic::Orr, sz, rd, ZR, imm);
            debug_assert!(ok);
            return;
        }
        let inverted = ones > zeros;
        let skip = if inverted { 0xffff } else { 0 };
        let mut first = true;
        for i in 0..chunks {
            let p = part(i);
            if p == skip {
                continue;
            }
            if first {
                if inverted {
                    self.movn(sz, rd, p ^ 0xffff, i as u32);
                } else {
                    self.movz(sz, rd, p, i as u32);
                }
                first = false;
            } else {
                self.movk(sz, rd, p, i as u32);
            }
        }
        if first {
            if inverted {
                self.movn(sz, rd, 0, 0);
            } else {
                self.movz(sz, rd, 0, 0);
            }
        }
    }

    fn bitfield(&mut self, sz: Size, opc: u32, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        let n = (sz == Size::X) as u32;
        assert!(immr < sz.bits() && imms < sz.bits());
        self.emit(
            sz.sf()
                | opc << 29
                | 0x1300_0000
                | n << 22
                | immr << 16
                | imms << 10
                | reg(rn) << 5
                | reg(rd),
        );
    }

    pub fn sbfm(&mut self, sz: Size, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.bitfield(sz, 0, rd, rn, immr, imms)
    }

    pub fn bfm(&mut self, sz: Size, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.bitfield(sz, 1, rd, rn, immr, imms)
    }

    pub fn ubfm(&mut self, sz: Size, rd: Reg, rn: Reg, immr: u32, imms: u32) {
        self.bitfield(sz, 2, rd, rn, immr, imms)
    }

    pub fn lsl_imm(&mut self, sz: Size, rd: Reg, rn: Reg, sh: u32) {
        let b = sz.bits();
        self.ubfm(sz, rd, rn, (b - sh) % b, b - 1 - sh)
    }

    pub fn lsr_imm(&mut self, sz: Size, rd: Reg, rn: Reg, sh: u32) {
        self.ubfm(sz, rd, rn, sh, sz.bits() - 1)
    }

    pub fn asr_imm(&mut self, sz: Size, rd: Reg, rn: Reg, sh: u32) {
        self.sbfm(sz, rd, rn, sh, sz.bits() - 1)
    }

    pub fn ror_imm(&mut self, sz: Size, rd: Reg, rn: Reg, sh: u32) {
        self.extr(sz, rd, rn, rn, sh)
    }

    pub fn ubfx(&mut self, sz: Size, rd: Reg, rn: Reg, lsb: u32, width: u32) {
        self.ubfm(sz, rd, rn, lsb, lsb + width - 1)
    }

    pub fn sbfx(&mut self, sz: Size, rd: Reg, rn: Reg, lsb: u32, width: u32) {
        self.sbfm(sz, rd, rn, lsb, lsb + width - 1)
    }

    /// Inserts the low `width` bits of `rn` into `rd` at bit `lsb`.
    pub fn bfi(&mut self, sz: Size, rd: Reg, rn: Reg, lsb: u32, width: u32) {
        let b = sz.bits();
        self.bfm(sz, rd, rn, (b - lsb) % b, width - 1)
    }

    /// Copies bits `lsb..lsb + width` of `rn` into the low bits of `rd`.
    pub fn bfxil(&mut self, sz: Size, rd: Reg, rn: Reg, lsb: u32, width: u32) {
        self.bfm(sz, rd, rn, lsb, lsb + width - 1)
    }

    pub fn uxtb(&mut self, rd: Reg, rn: Reg) {
        self.ubfm(Size::W, rd, rn, 0, 7)
    }

    pub fn uxth(&mut self, rd: Reg, rn: Reg) {
        self.ubfm(Size::W, rd, rn, 0, 15)
    }

    pub fn sxtb(&mut self, sz: Size, rd: Reg, rn: Reg) {
        self.sbfm(sz, rd, rn, 0, 7)
    }

    pub fn sxth(&mut self, sz: Size, rd: Reg, rn: Reg) {
        self.sbfm(sz, rd, rn, 0, 15)
    }

    pub fn sxtw(&mut self, rd: Reg, rn: Reg) {
        self.sbfm(Size::X, rd, rn, 0, 31)
    }

    pub fn extr(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, lsb: u32) {
        let n = (sz == Size::X) as u32;
        assert!(lsb < sz.bits());
        self.emit(
            sz.sf() | 0x1380_0000 | n << 22 | reg(rm) << 16 | lsb << 10 | reg(rn) << 5 | reg(rd),
        );
    }

    fn dp2(&mut self, sz: Size, opcode: u32, rd: Reg, rn: Reg, rm: Reg) {
        self.emit(sz.sf() | 0x1ac0_0000 | reg(rm) << 16 | opcode << 10 | reg(rn) << 5 | reg(rd));
    }

    pub fn udiv(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.dp2(sz, 0b000010, rd, rn, rm)
    }

    pub fn sdiv(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.dp2(sz, 0b000011, rd, rn, rm)
    }

    pub fn lslv(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.dp2(sz, 0b001000, rd, rn, rm)
    }

    pub fn lsrv(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.dp2(sz, 0b001001, rd, rn, rm)
    }

    pub fn asrv(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.dp2(sz, 0b001010, rd, rn, rm)
    }

    pub fn rorv(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.dp2(sz, 0b001011, rd, rn, rm)
    }

    fn dp1(&mut self, sz: Size, opcode: u32, rd: Reg, rn: Reg) {
        self.emit(sz.sf() | 0x5ac0_0000 | opcode << 10 | reg(rn) << 5 | reg(rd));
    }

    pub fn rbit(&mut self, sz: Size, rd: Reg, rn: Reg) {
        self.dp1(sz, 0, rd, rn)
    }

    pub fn rev16(&mut self, sz: Size, rd: Reg, rn: Reg) {
        self.dp1(sz, 1, rd, rn)
    }

    /// Reverses the byte order of the whole register.
    pub fn rev(&mut self, sz: Size, rd: Reg, rn: Reg) {
        self.dp1(sz, if sz == Size::W { 2 } else { 3 }, rd, rn)
    }

    pub fn clz(&mut self, sz: Size, rd: Reg, rn: Reg) {
        self.dp1(sz, 4, rd, rn)
    }

    #[allow(clippy::too_many_arguments)]
    fn dp3(&mut self, sz: Size, op31: u32, o0: u32, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        self.emit(
            sz.sf()
                | 0x1b00_0000
                | op31 << 21
                | reg(rm) << 16
                | o0 << 15
                | reg(ra) << 10
                | reg(rn) << 5
                | reg(rd),
        );
    }

    pub fn madd(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        self.dp3(sz, 0, 0, rd, rn, rm, ra)
    }

    pub fn msub(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        self.dp3(sz, 0, 1, rd, rn, rm, ra)
    }

    pub fn mul(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg) {
        self.madd(sz, rd, rn, rm, ZR)
    }

    /// Signed 32x32 to 64-bit multiply.
    pub fn smull(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.dp3(Size::X, 0b001, 0, rd, rn, rm, ZR)
    }

    /// Unsigned 32x32 to 64-bit multiply.
    pub fn umull(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.dp3(Size::X, 0b101, 0, rd, rn, rm, ZR)
    }

    pub fn smulh(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.dp3(Size::X, 0b010, 0, rd, rn, rm, ZR)
    }

    pub fn umulh(&mut self, rd: Reg, rn: Reg, rm: Reg) {
        self.dp3(Size::X, 0b110, 0, rd, rn, rm, ZR)
    }

    #[allow(clippy::too_many_arguments)]
    fn csel_like(&mut self, sz: Size, op: u32, o2: u32, rd: Reg, rn: Reg, rm: Reg, c: Cond) {
        self.emit(
            sz.sf()
                | op << 30
                | 0x1a80_0000
                | reg(rm) << 16
                | (c as u32) << 12
                | o2 << 10
                | reg(rn) << 5
                | reg(rd),
        );
    }

    pub fn csel(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, c: Cond) {
        self.csel_like(sz, 0, 0, rd, rn, rm, c)
    }

    pub fn csinc(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, c: Cond) {
        self.csel_like(sz, 0, 1, rd, rn, rm, c)
    }

    pub fn csinv(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, c: Cond) {
        self.csel_like(sz, 1, 0, rd, rn, rm, c)
    }

    pub fn csneg(&mut self, sz: Size, rd: Reg, rn: Reg, rm: Reg, c: Cond) {
        self.csel_like(sz, 1, 1, rd, rn, rm, c)
    }

    /// Sets `rd` to 1 when `c` holds, else 0.
    pub fn cset(&mut self, sz: Size, rd: Reg, c: Cond) {
        self.csinc(sz, rd, ZR, ZR, c.invert())
    }

    /// Sets `rd` to all ones when `c` holds, else 0.
    pub fn csetm(&mut self, sz: Size, rd: Reg, c: Cond) {
        self.csinv(sz, rd, ZR, ZR, c.invert())
    }

    pub fn cinc(&mut self, sz: Size, rd: Reg, rn: Reg, c: Cond) {
        self.csinc(sz, rd, rn, rn, c.invert())
    }

    #[allow(clippy::too_many_arguments)]
    fn ccmp_like(
        &mut self,
        sz: Size,
        op: u32,
        imm: bool,
        rn: Reg,
        rm_or_imm: u32,
        nzcv: u32,
        c: Cond,
    ) {
        assert!(nzcv < 16 && rm_or_imm < 32);
        self.emit(
            sz.sf()
                | op << 30
                | 1 << 29
                | 0x1a40_0000
                | rm_or_imm << 16
                | (c as u32) << 12
                | (imm as u32) << 11
                | reg(rn) << 5
                | nzcv,
        );
    }

    pub fn ccmp(&mut self, sz: Size, rn: Reg, rm: Reg, nzcv: u32, c: Cond) {
        self.ccmp_like(sz, 1, false, rn, reg(rm), nzcv, c)
    }

    pub fn ccmp_imm(&mut self, sz: Size, rn: Reg, imm5: u32, nzcv: u32, c: Cond) {
        self.ccmp_like(sz, 1, true, rn, imm5, nzcv, c)
    }

    pub fn ccmn_imm(&mut self, sz: Size, rn: Reg, imm5: u32, nzcv: u32, c: Cond) {
        self.ccmp_like(sz, 0, true, rn, imm5, nzcv, c)
    }

    pub fn cfinv(&mut self) {
        self.emit(0xd500_401f)
    }

    pub fn axflag(&mut self) {
        self.emit(0xd500_405f)
    }

    pub fn xaflag(&mut self) {
        self.emit(0xd500_403f)
    }

    /// Rotates Xn right by `shift` and inserts its low four bits into NZCV
    /// for the flags selected by `mask`.
    pub fn rmif(&mut self, rn: Reg, shift: u32, mask: u32) {
        assert!(shift < 64 && mask < 16);
        self.emit(0xba00_0400 | shift << 15 | reg(rn) << 5 | mask)
    }

    pub fn setf8(&mut self, rn: Reg) {
        self.emit(0x3a00_080d | reg(rn) << 5)
    }

    pub fn setf16(&mut self, rn: Reg) {
        self.emit(0x3a00_480d | reg(rn) << 5)
    }

    fn ldst_opc(load: bool, signed: bool, dst64: bool) -> u32 {
        match (load, signed, dst64) {
            (false, _, _) => 0,
            (true, false, _) => 1,
            (true, true, true) => 2,
            (true, true, false) => 3,
        }
    }

    /// Load or store at `[rn, #offset]`, using LDUR/STUR when the offset is
    /// negative or unaligned.
    #[allow(clippy::too_many_arguments)]
    pub fn ldst(
        &mut self,
        load: bool,
        w: Width,
        signed: bool,
        dst64: bool,
        rt: Reg,
        rn: Reg,
        offset: i32,
    ) {
        let size = w.size();
        let opc = Self::ldst_opc(load, signed, dst64);
        let scale = 1i32 << size;
        if offset >= 0 && offset % scale == 0 && offset / scale < 4096 {
            let imm12 = (offset / scale) as u32;
            self.emit(size << 30 | 0x3900_0000 | opc << 22 | imm12 << 10 | reg(rn) << 5 | reg(rt));
        } else {
            assert!(
                (-256..256).contains(&offset),
                "load/store offset {offset} out of range"
            );
            let imm9 = (offset as u32) & 0x1ff;
            self.emit(size << 30 | 0x3800_0000 | opc << 22 | imm9 << 12 | reg(rn) << 5 | reg(rt));
        }
    }

    pub fn ldr(&mut self, w: Width, rt: Reg, rn: Reg, offset: i32) {
        self.ldst(true, w, false, w == Width::X, rt, rn, offset)
    }

    pub fn str(&mut self, w: Width, rt: Reg, rn: Reg, offset: i32) {
        self.ldst(false, w, false, false, rt, rn, offset)
    }

    /// Sign-extending load (LDRSB/LDRSH/LDRSW) into a W or X register.
    pub fn ldrs(&mut self, w: Width, dst: Size, rt: Reg, rn: Reg, offset: i32) {
        assert!(w != Width::X && !(w == Width::W && dst == Size::W));
        self.ldst(true, w, true, dst == Size::X, rt, rn, offset)
    }

    /// Load or store with base writeback: pre-index when `pre`, else post-index.
    pub fn ldst_index(&mut self, load: bool, w: Width, rt: Reg, rn: Reg, offset: i32, pre: bool) {
        assert!((-256..256).contains(&offset));
        let opc = Self::ldst_opc(load, false, false);
        let imm9 = (offset as u32) & 0x1ff;
        let mode = if pre { 3 } else { 1 };
        self.emit(
            w.size() << 30
                | 0x3800_0000
                | opc << 22
                | imm9 << 12
                | mode << 10
                | reg(rn) << 5
                | reg(rt),
        );
    }

    /// Load or store at `rn + extend(rm)`, shifted by the access size when `scaled`.
    #[allow(clippy::too_many_arguments)]
    pub fn ldst_reg(
        &mut self,
        load: bool,
        w: Width,
        signed: bool,
        dst64: bool,
        rt: Reg,
        rn: Reg,
        rm: Reg,
        ext: Extend,
        scaled: bool,
    ) {
        assert!(matches!(
            ext,
            Extend::Uxtw | Extend::Uxtx | Extend::Sxtw | Extend::Sxtx
        ));
        let opc = Self::ldst_opc(load, signed, dst64);
        self.emit(
            w.size() << 30
                | 0x3820_0800
                | opc << 22
                | reg(rm) << 16
                | (ext as u32) << 13
                | (scaled as u32) << 12
                | reg(rn) << 5
                | reg(rt),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn pair(&mut self, load: bool, sz: Size, rt: Reg, rt2: Reg, rn: Reg, offset: i32, mode: u32) {
        let scale = if sz == Size::X { 8 } else { 4 };
        assert!(offset % scale == 0 && (-64..64).contains(&(offset / scale)));
        let opc = if sz == Size::X { 2 } else { 0 };
        let imm7 = ((offset / scale) as u32) & 0x7f;
        self.emit(
            opc << 30
                | 0x2800_0000
                | mode << 23
                | (load as u32) << 22
                | imm7 << 15
                | reg(rt2) << 10
                | reg(rn) << 5
                | reg(rt),
        );
    }

    pub fn ldp(&mut self, sz: Size, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        self.pair(true, sz, rt, rt2, rn, offset, 2)
    }

    pub fn stp(&mut self, sz: Size, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        self.pair(false, sz, rt, rt2, rn, offset, 2)
    }

    pub fn ldp_post(&mut self, sz: Size, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        self.pair(true, sz, rt, rt2, rn, offset, 1)
    }

    pub fn stp_pre(&mut self, sz: Size, rt: Reg, rt2: Reg, rn: Reg, offset: i32) {
        self.pair(false, sz, rt, rt2, rn, offset, 3)
    }

    #[allow(clippy::too_many_arguments)]
    fn excl(&mut self, w: Width, o2: u32, l: u32, o1: u32, o0: u32, rs: Reg, rt: Reg, rn: Reg) {
        self.emit(
            w.size() << 30
                | 0x0800_0000
                | o2 << 23
                | l << 22
                | o1 << 21
                | reg(rs) << 16
                | o0 << 15
                | 31 << 10
                | reg(rn) << 5
                | reg(rt),
        );
    }

    pub fn ldar(&mut self, w: Width, rt: Reg, rn: Reg) {
        self.excl(w, 1, 1, 0, 1, ZR, rt, rn)
    }

    pub fn stlr(&mut self, w: Width, rt: Reg, rn: Reg) {
        self.excl(w, 1, 0, 0, 1, ZR, rt, rn)
    }

    pub fn ldaxr(&mut self, w: Width, rt: Reg, rn: Reg) {
        self.excl(w, 0, 1, 0, 1, ZR, rt, rn)
    }

    /// Store-release exclusive; `rs` receives 0 on success.
    pub fn stlxr(&mut self, w: Width, rs: Reg, rt: Reg, rn: Reg) {
        self.excl(w, 0, 0, 0, 1, rs, rt, rn)
    }

    /// Compares `rs` with `[rn]` and stores `rt` if equal; `rs` receives the old value.
    pub fn casal(&mut self, w: Width, rs: Reg, rt: Reg, rn: Reg) {
        self.excl(w, 1, 1, 1, 1, rs, rt, rn)
    }

    fn lse(&mut self, w: Width, opc: u32, o3: u32, rs: Reg, rt: Reg, rn: Reg) {
        self.emit(
            w.size() << 30
                | 0x38e0_0000
                | reg(rs) << 16
                | o3 << 15
                | opc << 12
                | reg(rn) << 5
                | reg(rt),
        );
    }

    /// Atomically adds `rs` to `[rn]`; `rt` receives the old value.
    pub fn ldaddal(&mut self, w: Width, rs: Reg, rt: Reg, rn: Reg) {
        self.lse(w, 0b000, 0, rs, rt, rn)
    }

    pub fn ldclral(&mut self, w: Width, rs: Reg, rt: Reg, rn: Reg) {
        self.lse(w, 0b001, 0, rs, rt, rn)
    }

    pub fn ldeoral(&mut self, w: Width, rs: Reg, rt: Reg, rn: Reg) {
        self.lse(w, 0b010, 0, rs, rt, rn)
    }

    pub fn ldsetal(&mut self, w: Width, rs: Reg, rt: Reg, rn: Reg) {
        self.lse(w, 0b011, 0, rs, rt, rn)
    }

    pub fn swpal(&mut self, w: Width, rs: Reg, rt: Reg, rn: Reg) {
        self.lse(w, 0b000, 1, rs, rt, rn)
    }

    pub fn b(&mut self, l: Label) {
        self.branch_to(0x1400_0000, l, Fix::Imm26)
    }

    pub fn bl(&mut self, l: Label) {
        self.branch_to(0x9400_0000, l, Fix::Imm26)
    }

    /// B to a target `words` instructions away, for code outside this buffer.
    pub fn b_offset(&mut self, words: i32) {
        assert!((-(1 << 25)..(1 << 25)).contains(&words));
        self.emit(0x1400_0000 | (words as u32 & 0x03ff_ffff))
    }

    /// ADRP: `rd` gets the address of the 4 KiB page `pages` pages from this
    /// instruction's page.
    pub fn adrp_offset(&mut self, rd: Reg, pages: i64) {
        assert!((-(1 << 20)..(1 << 20)).contains(&pages));
        let p = pages as u32;
        self.emit(0x9000_0000 | (p & 3) << 29 | ((p >> 2) & 0x7ffff) << 5 | reg(rd))
    }

    /// BL to a target `words` instructions away, for code outside this buffer.
    pub fn bl_offset(&mut self, words: i32) {
        assert!((-(1 << 25)..(1 << 25)).contains(&words));
        self.emit(0x9400_0000 | (words as u32 & 0x03ff_ffff))
    }

    pub fn b_cond(&mut self, c: Cond, l: Label) {
        self.branch_to(0x5400_0000 | c as u32, l, Fix::Imm19)
    }

    pub fn cbz(&mut self, sz: Size, rt: Reg, l: Label) {
        self.branch_to(sz.sf() | 0x3400_0000 | reg(rt), l, Fix::Imm19)
    }

    pub fn cbnz(&mut self, sz: Size, rt: Reg, l: Label) {
        self.branch_to(sz.sf() | 0x3500_0000 | reg(rt), l, Fix::Imm19)
    }

    pub fn tbz(&mut self, rt: Reg, bit: u32, l: Label) {
        assert!(bit < 64);
        self.branch_to(
            (bit >> 5) << 31 | 0x3600_0000 | (bit & 31) << 19 | reg(rt),
            l,
            Fix::Imm14,
        )
    }

    pub fn tbnz(&mut self, rt: Reg, bit: u32, l: Label) {
        assert!(bit < 64);
        self.branch_to(
            (bit >> 5) << 31 | 0x3700_0000 | (bit & 31) << 19 | reg(rt),
            l,
            Fix::Imm14,
        )
    }

    pub fn br(&mut self, rn: Reg) {
        self.emit(0xd61f_0000 | reg(rn) << 5)
    }

    pub fn blr(&mut self, rn: Reg) {
        self.emit(0xd63f_0000 | reg(rn) << 5)
    }

    pub fn ret(&mut self) {
        self.emit(0xd65f_03c0)
    }

    /// RET to `rn`: a branch the CPU predicts from its return stack.
    pub fn ret_to(&mut self, rn: Reg) {
        self.emit(0xd65f_0000 | reg(rn) << 5)
    }

    pub fn eret(&mut self) {
        self.emit(0xd69f_03e0)
    }

    pub fn adr(&mut self, rd: Reg, l: Label) {
        self.branch_to(0x1000_0000 | reg(rd), l, Fix::Adr)
    }

    pub fn svc(&mut self, imm: u16) {
        self.emit(0xd400_0001 | (imm as u32) << 5)
    }

    pub fn hvc(&mut self, imm: u16) {
        self.emit(0xd400_0002 | (imm as u32) << 5)
    }

    pub fn brk(&mut self, imm: u16) {
        self.emit(0xd420_0000 | (imm as u32) << 5)
    }

    pub fn nop(&mut self) {
        self.emit(0xd503_201f)
    }

    pub fn isb(&mut self) {
        self.emit(0xd503_3fdf)
    }

    /// DMB with CRm `option`: 0xF = SY, 0xB = ISH, 0x9 = ISHLD, 0xA = ISHST.
    pub fn dmb(&mut self, option: u32) {
        self.emit(0xd503_30bf | (option & 15) << 8)
    }

    pub fn dsb(&mut self, option: u32) {
        self.emit(0xd503_309f | (option & 15) << 8)
    }

    /// Reads a system register given as op0:op1:CRn:CRm:op2 packed into 16
    /// bits, the same packing as Hypervisor.framework's `HV_SYS_REG_*`.
    pub fn mrs(&mut self, rt: Reg, sysreg: u16) {
        self.emit(0xd530_0000 | (sysreg as u32 & 0x7fff) << 5 | reg(rt))
    }

    /// Writes a system register; see `mrs` for the encoding of `sysreg`.
    pub fn msr(&mut self, sysreg: u16, rt: Reg) {
        self.emit(0xd510_0000 | (sysreg as u32 & 0x7fff) << 5 | reg(rt))
    }

    /// SYS instruction (AT, IC, DC, TLBI aliases).
    pub fn sys(&mut self, op1: u32, crn: u32, crm: u32, op2: u32, rt: Reg) {
        assert!(op1 < 8 && crn < 16 && crm < 16 && op2 < 8);
        self.emit(0xd508_0000 | op1 << 16 | crn << 12 | crm << 8 | op2 << 5 | reg(rt))
    }

    /// Translates the address in `rt` as an EL1 read into PAR_EL1.
    pub fn at_s1e1r(&mut self, rt: Reg) {
        self.sys(0, 7, 8, 0, rt)
    }

    pub fn ic_iallu(&mut self) {
        self.sys(0, 7, 5, 0, ZR)
    }

    pub fn tlbi_vmalle1(&mut self) {
        self.sys(0, 8, 7, 0, ZR)
    }

    /// Invalidates EL1 TLB entries for the page whose VA >> 12 is in `rt`.
    pub fn tlbi_vae1(&mut self, rt: Reg) {
        self.sys(0, 8, 7, 1, rt)
    }

    /// Like `tlbi_vae1`, for every ASID.
    pub fn tlbi_vaae1(&mut self, rt: Reg) {
        self.sys(0, 8, 7, 3, rt)
    }

    /// TLBI operation given as op1:CRm:op2 packed into 10 bits.
    pub fn tlbi(&mut self, op1_crm_op2: u32, rt: Reg) {
        self.emit(0xd508_8000 | op1_crm_op2 << 5 | reg(rt))
    }
}

pub mod sysreg {
    pub const SCTLR_EL1: u16 = 0xc080;
    pub const TTBR0_EL1: u16 = 0xc100;
    pub const TTBR1_EL1: u16 = 0xc101;
    pub const TCR_EL1: u16 = 0xc102;
    pub const SPSR_EL1: u16 = 0xc200;
    pub const ELR_EL1: u16 = 0xc201;
    pub const SP_EL0: u16 = 0xc208;
    pub const ESR_EL1: u16 = 0xc290;
    pub const FAR_EL1: u16 = 0xc300;
    pub const PAR_EL1: u16 = 0xc3a0;
    pub const MAIR_EL1: u16 = 0xc510;
    pub const VBAR_EL1: u16 = 0xc600;
    pub const TPIDR_EL1: u16 = 0xc684;
    pub const CTR_EL0: u16 = 0xd801;
    pub const NZCV: u16 = 0xda10;
    pub const FPCR: u16 = 0xda20;
    pub const FPSR: u16 = 0xda21;
    pub const TPIDR_EL0: u16 = 0xde82;
    pub const CNTVCT_EL0: u16 = 0xdf02;
}

/// Advanced SIMD. `q` selects the 128-bit form; `size` is the element size
/// (0 = B, 1 = H, 2 = S, 3 = D).
impl Asm {
    pub fn ldr_d(&mut self, rt: V, rn: Reg, offset: i32) {
        assert!(offset >= 0 && offset % 8 == 0 && offset < 8 << 12);
        self.emit(0xfd40_0000 | ((offset / 8) as u32) << 10 | (rn.0 as u32) << 5 | rt.0 as u32);
    }

    pub fn str_d(&mut self, rt: V, rn: Reg, offset: i32) {
        assert!(offset >= 0 && offset % 8 == 0 && offset < 8 << 12);
        self.emit(0xfd00_0000 | ((offset / 8) as u32) << 10 | (rn.0 as u32) << 5 | rt.0 as u32);
    }

    pub fn fmov_dx(&mut self, rd: V, rn: Reg) {
        self.emit(0x9e67_0000 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    pub fn fmov_xd(&mut self, rd: Reg, rn: V) {
        self.emit(0x9e66_0000 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// Scalar two-source FP: `op` is 0 FMUL, 1 FDIV, 2 FADD, 3 FSUB, 4 FMAX, 5 FMIN.
    pub fn fp2(&mut self, double: bool, op: u32, rd: V, rn: V, rm: V) {
        self.emit(
            0x1e20_0800
                | (double as u32) << 22
                | (rm.0 as u32) << 16
                | op << 12
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    /// Scalar one-source FP: `op` is 1 FABS, 2 FNEG, 3 FSQRT, 4/5 FCVT to
    /// single/double, 8 FRINTN, 9 FRINTP, 10 FRINTM, 11 FRINTZ, 14 FRINTX.
    pub fn fp1(&mut self, double: bool, op: u32, rd: V, rn: V) {
        self.emit(
            0x1e20_4000 | (double as u32) << 22 | op << 15 | (rn.0 as u32) << 5 | rd.0 as u32,
        );
    }

    /// Scalar FP compare into all ones or zeros in `rd`; NZCV is left alone.
    pub fn fcm(&mut self, kind: Fcm, double: bool, rd: V, rn: V, rm: V) {
        self.emit(
            kind.scalar()
                | (double as u32) << 22
                | (rm.0 as u32) << 16
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    /// The vector form of `fcm`, on four singles or two doubles.
    pub fn fcm_v(&mut self, kind: Fcm, double: bool, rd: V, rn: V, rm: V) {
        self.emit(
            kind.scalar() & !0x1000_0000
                | 1 << 30
                | (double as u32) << 22
                | (rm.0 as u32) << 16
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    pub fn ldr_s(&mut self, rt: V, rn: Reg, offset: i32) {
        assert!(offset >= 0 && offset % 4 == 0 && offset < 4 << 12);
        self.emit(0xbd40_0000 | ((offset / 4) as u32) << 10 | (rn.0 as u32) << 5 | rt.0 as u32);
    }

    pub fn str_s(&mut self, rt: V, rn: Reg, offset: i32) {
        assert!(offset >= 0 && offset % 4 == 0 && offset < 4 << 12);
        self.emit(0xbd00_0000 | ((offset / 4) as u32) << 10 | (rn.0 as u32) << 5 | rt.0 as u32);
    }

    pub fn ldr_q(&mut self, rt: V, rn: Reg, offset: i32) {
        assert!(offset >= 0 && offset % 16 == 0 && offset < 16 << 12);
        self.emit(0x3dc0_0000 | ((offset / 16) as u32) << 10 | (rn.0 as u32) << 5 | rt.0 as u32);
    }

    pub fn str_q(&mut self, rt: V, rn: Reg, offset: i32) {
        assert!(offset >= 0 && offset % 16 == 0 && offset < 16 << 12);
        self.emit(0x3d80_0000 | ((offset / 16) as u32) << 10 | (rn.0 as u32) << 5 | rt.0 as u32);
    }

    /// INS: element `i` of `rd` = element `j` of `rn`, elements of 1 << `size` bytes.
    pub fn ins(&mut self, size: u32, rd: V, i: u32, rn: V, j: u32) {
        let imm5 = (i << (size + 1)) | 1 << size;
        self.emit(0x6e00_0400 | imm5 << 16 | (j << size) << 11 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    pub fn not_v(&mut self, rd: V, rn: V) {
        self.emit(0x6e20_5800 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// FSQRT on four singles or two doubles.
    pub fn fsqrt_v(&mut self, double: bool, rd: V, rn: V) {
        self.emit(0x6ea1_f800 | (double as u32) << 22 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// FMADD (`sub` false: ra + rn * rm) or FMSUB (ra - rn * rm), fused, on doubles.
    pub fn fmadd_d(&mut self, sub: bool, rd: V, rn: V, rm: V, ra: V) {
        self.emit(
            0x1f40_0000
                | (rm.0 as u32) << 16
                | (sub as u32) << 15
                | (ra.0 as u32) << 10
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    /// FCMP, or FCMPE (`signal`), of `rn` with `rm`.
    pub fn fcmp(&mut self, double: bool, signal: bool, rn: V, rm: V) {
        self.emit(
            0x1e20_2000
                | (double as u32) << 22
                | (rm.0 as u32) << 16
                | (rn.0 as u32) << 5
                | (signal as u32) << 4,
        );
    }

    /// SCVTF from a W or X register.
    pub fn scvtf(&mut self, double: bool, size: Size, rd: V, rn: Reg) {
        self.emit(
            0x1e22_0000 | size.sf() | (double as u32) << 22 | (rn.0 as u32) << 5 | rd.0 as u32,
        );
    }

    /// FCVTZS to a W or X register.
    pub fn fcvtzs(&mut self, double: bool, size: Size, rd: Reg, rn: V) {
        self.emit(
            0x1e38_0000 | size.sf() | (double as u32) << 22 | (rn.0 as u32) << 5 | rd.0 as u32,
        );
    }

    /// FMOV Sd, Wn.
    pub fn fmov_sw(&mut self, rd: V, rn: Reg) {
        self.emit(0x1e27_0000 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// FMOV Wd, Sn.
    pub fn fmov_ws(&mut self, rd: Reg, rn: V) {
        self.emit(0x1e26_0000 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// The "three same" group: `opcode` and `u` pick ADD, UQADD, CMEQ, ...
    #[allow(clippy::too_many_arguments)]
    pub fn vec3(&mut self, q: bool, u: bool, size: u32, opcode: u32, rd: V, rn: V, rm: V) {
        self.emit(
            0x0e20_0400
                | (q as u32) << 30
                | (u as u32) << 29
                | size << 22
                | (rm.0 as u32) << 16
                | opcode << 11
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    /// TBL with one table register: each byte of `rd` is the byte of `rn`
    /// that the matching byte of `rm` indexes, or 0 when the index is 16 or more.
    pub fn tbl(&mut self, rd: V, rn: V, rm: V) {
        self.emit(0x4e00_0000 | (rm.0 as u32) << 16 | (rn.0 as u32) << 5 | rd.0 as u32)
    }

    /// DUP (element): every element of `rd` = element 0 of `rn`, elements of
    /// 1 << `size` bytes.
    pub fn dup_elem(&mut self, size: u32, rd: V, rn: V) {
        self.emit(0x4e00_0400 | (1 << size) << 16 | (rn.0 as u32) << 5 | rd.0 as u32)
    }

    /// DUP (general): every element of `rd` = the low 1 << `size` bytes of `rn`.
    pub fn dup_gen(&mut self, size: u32, rd: V, rn: Reg) {
        self.emit(0x4e00_0c00 | (1 << size) << 16 | (rn.0 as u32) << 5 | rd.0 as u32)
    }

    /// NEG on 16 bytes of elements of 1 << `size` bytes.
    pub fn neg_v(&mut self, size: u32, rd: V, rn: V) {
        self.emit(0x6e20_b800 | size << 22 | (rn.0 as u32) << 5 | rd.0 as u32)
    }

    /// SCVTF on four singles or two doubles.
    pub fn scvtf_v(&mut self, double: bool, rd: V, rn: V) {
        self.emit(0x4e21_d800 | (double as u32) << 22 | (rn.0 as u32) << 5 | rd.0 as u32)
    }

    /// FCVTZS on four singles or two doubles.
    pub fn fcvtzs_v(&mut self, double: bool, rd: V, rn: V) {
        self.emit(0x4ea1_b800 | (double as u32) << 22 | (rn.0 as u32) << 5 | rd.0 as u32)
    }

    /// FRINT32X (`exact`: the current rounding mode) or FRINT32Z (toward
    /// zero), on four singles or two doubles.
    pub fn frint32_v(&mut self, exact: bool, double: bool, rd: V, rn: V) {
        let base = if exact { 0x6e21_e800 } else { 0x4e21_e800 };
        self.emit(base | (double as u32) << 22 | (rn.0 as u32) << 5 | rd.0 as u32)
    }

    /// MOVI: every byte of `rd` = `imm`.
    pub fn movi_b(&mut self, rd: V, imm: u8) {
        let imm = imm as u32;
        self.emit(0x4f00_e400 | (imm >> 5) << 16 | (imm & 31) << 5 | rd.0 as u32)
    }

    /// ZIP1 (`second` false) or ZIP2.
    pub fn zip(&mut self, q: bool, size: u32, second: bool, rd: V, rn: V, rm: V) {
        self.emit(
            0x0e00_3800
                | (q as u32) << 30
                | size << 22
                | (rm.0 as u32) << 16
                | (second as u32) << 14
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    fn shift_imm_v(&mut self, q: bool, u: bool, opcode: u32, immhb: u32, rd: V, rn: V) {
        self.emit(
            0x0f00_0400
                | (q as u32) << 30
                | (u as u32) << 29
                | immhb << 16
                | opcode << 11
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    /// SHL by `shift` (0..esize) on elements of `esize` bits.
    pub fn shl_v(&mut self, q: bool, esize: u32, rd: V, rn: V, shift: u32) {
        assert!(shift < esize);
        self.shift_imm_v(q, false, 0b01010, esize + shift, rd, rn);
    }

    /// USHR (`signed` false) or SSHR by `shift` (1..=esize).
    pub fn shr_v(&mut self, q: bool, signed: bool, esize: u32, rd: V, rn: V, shift: u32) {
        assert!(shift >= 1 && shift <= esize);
        self.shift_imm_v(q, !signed, 0, 2 * esize - shift, rd, rn);
    }

    /// USRA: adds `rn` shifted right by `shift` (1..=esize) into `rd`.
    pub fn usra(&mut self, q: bool, esize: u32, rd: V, rn: V, shift: u32) {
        assert!(shift >= 1 && shift <= esize);
        self.shift_imm_v(q, true, 0b00010, 2 * esize - shift, rd, rn);
    }

    /// UMOV Wd, Vn.B[i].
    pub fn umov_b(&mut self, rd: Reg, rn: V, i: u32) {
        self.emit(0x0e00_3c00 | (i << 1 | 1) << 16 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// SHRN: narrows elements of 2*`esize` bits to `esize` after shifting right.
    pub fn shrn(&mut self, esize: u32, rd: V, rn: V, shift: u32) {
        assert!(shift >= 1 && shift <= esize);
        self.shift_imm_v(false, false, 0b10000, 2 * esize - shift, rd, rn);
    }

    /// INS Vd.D[1], Vn.D[0].
    pub fn ins_d1(&mut self, rd: V, rn: V) {
        self.emit(0x6e18_0400 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// SQXTUN, SQXTN or XTN of the lower half, from elements of `size` + 1.
    pub fn narrow(&mut self, kind: Narrow, size: u32, rd: V, rn: V) {
        let base = match kind {
            Narrow::Xtn => 0x0e21_2800,
            Narrow::Sqxtn => 0x0e21_4800,
            Narrow::Sqxtun => 0x2e21_2800,
        };
        self.emit(base | size << 22 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// SMULL or UMULL of the lower halves, widening from `size`.
    /// The "2" form of `narrow`: writes the upper half of `rd`, keeping the lower.
    pub fn narrow_hi(&mut self, kind: Narrow, size: u32, rd: V, rn: V) {
        self.narrow(kind, size, rd, rn);
        if let Some(w) = self.code.last_mut() {
            *w |= 1 << 30;
        }
    }

    /// SMULL2 or UMULL2: the upper halves of `rn` and `rm`.
    pub fn mull_hi(&mut self, signed: bool, size: u32, rd: V, rn: V, rm: V) {
        self.mull(signed, size, rd, rn, rm);
        if let Some(w) = self.code.last_mut() {
            *w |= 1 << 30;
        }
    }

    /// UZP1 (`second` false) or UZP2.
    pub fn uzp(&mut self, q: bool, size: u32, second: bool, rd: V, rn: V, rm: V) {
        self.emit(
            0x0e00_1800
                | (q as u32) << 30
                | size << 22
                | (rm.0 as u32) << 16
                | (second as u32) << 14
                | (rn.0 as u32) << 5
                | rd.0 as u32,
        );
    }

    /// EXT of 16 bytes: bytes `index`.. of `rn` followed by `rm`.
    pub fn ext(&mut self, rd: V, rn: V, rm: V, index: u32) {
        self.emit(
            0x6e00_0000 | (rm.0 as u32) << 16 | index << 11 | (rn.0 as u32) << 5 | rd.0 as u32,
        );
    }

    /// UADDLP: adds adjacent unsigned elements into elements twice as wide.
    pub fn uaddlp(&mut self, q: bool, size: u32, rd: V, rn: V) {
        self.emit(0x2e20_2800 | (q as u32) << 30 | size << 22 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// UMOV Wd, Vn.H[i].
    pub fn umov_h(&mut self, rd: Reg, rn: V, i: u32) {
        self.emit(0x0e00_3c00 | (i << 2 | 2) << 16 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    /// INS Vd.H[i], Wn.
    pub fn ins_h(&mut self, rd: V, i: u32, rn: Reg) {
        self.emit(0x4e00_1c00 | (i << 2 | 2) << 16 | (rn.0 as u32) << 5 | rd.0 as u32);
    }

    pub fn mull(&mut self, signed: bool, size: u32, rd: V, rn: V, rm: V) {
        let base = if signed { 0x0e20_c000 } else { 0x2e20_c000 };
        self.emit(base | size << 22 | (rm.0 as u32) << 16 | (rn.0 as u32) << 5 | rd.0 as u32);
    }
}

#[derive(Clone, Copy)]
pub enum Narrow {
    Xtn,
    Sqxtn,
    Sqxtun,
}

#[cfg(test)]
mod tests;
