//! Translates a basic block of x86-64 into AArch64. Register and flag
//! conventions are in `docs/jit.md`, "Translated code".

use a64::{Asm, Cond, Logic, Reg, Shift, Size, Width, r};
use memory::Memory;
use x86::decode::{Insn, Map, Mem, Rep, Seg, decode};

mod vector;
pub use vector::CTX;

/// Exits translated code takes, as `hvc` numbers; x16 holds the guest RIP.
pub const HVC_MISS: u16 = 1;
pub const HVC_INTERP: u16 = 2;

/// The guest RIP to continue at, for the dispatcher and exits.
pub const RIP: Reg = Reg(16);
const ADDR: Reg = Reg(19);
const A: Reg = Reg(20);
const B: Reg = Reg(21);
const T: Reg = Reg(22);
const U: Reg = Reg(23);
/// The last result, for PF, and the XOR of the operands, for AF.
pub const PF_RESULT: Reg = Reg(26);
pub const AF_AUX: Reg = Reg(27);
/// Counts guest instructions run in translated code.
pub const ICOUNT: Reg = Reg(24);
pub const LOOKUP: Reg = Reg(28);
pub const FS_BASE: Reg = Reg(29);
/// Scratch for flag work; the dispatcher's own scratch, free inside blocks.
const OLD_NZCV: Reg = Reg(17);
const OLD_AF: Reg = Reg(18);

const MAX_INSNS: usize = 64;
const C_BIT: u64 = 1 << 29;

/// A translated block: code, for each guest instruction the word offset
/// where its code starts, and its direct exits as (word offset of the
/// branch to the dispatcher, target RIP), which can be chained.
pub struct Block {
    pub code: Vec<u32>,
    pub insns: Vec<(u32, u64)>,
    pub chains: Vec<(u32, u64)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Alu {
    Add,
    Or,
    And,
    Sub,
    Xor,
}

/// What an instruction does to the arithmetic flags.
#[derive(Clone, Copy)]
struct FlagUse {
    reads: bool,
    writes: bool,
}

fn alu_of(n: u8) -> Option<(Alu, bool)> {
    Some(match n {
        0 => (Alu::Add, true),
        1 => (Alu::Or, true),
        4 => (Alu::And, true),
        5 => (Alu::Sub, true),
        6 => (Alu::Xor, true),
        7 => (Alu::Sub, false),
        _ => return None,
    })
}

/// The ARM condition for x86 condition code `cc`, with CF kept inverted in C.
fn cond(cc: u8) -> Option<Cond> {
    Some(match cc & 0xf {
        0x0 => Cond::Vs,
        0x1 => Cond::Vc,
        0x2 => Cond::Lo,
        0x3 => Cond::Hs,
        0x4 => Cond::Eq,
        0x5 => Cond::Ne,
        0x6 => Cond::Ls,
        0x7 => Cond::Hi,
        0x8 => Cond::Mi,
        0x9 => Cond::Pl,
        0xc => Cond::Lt,
        0xd => Cond::Ge,
        0xe => Cond::Le,
        0xf => Cond::Gt,
        _ => return None,
    })
}

/// Whether an instruction ends the block, and its flag use; `None` if it
/// cannot be translated.
fn classify(i: &Insn) -> Option<(bool, FlagUse)> {
    if let Some(c) = vector::classify(i) {
        return Some(c);
    }
    let none = FlagUse {
        reads: false,
        writes: false,
    };
    let writes = FlagUse {
        reads: false,
        writes: true,
    };
    let reads = FlagUse {
        reads: true,
        writes: false,
    };
    if i.vex.is_some() || i.lock || i.seg == Some(Seg::Gs) || i.rep != Rep::None {
        return None;
    }
    Some(match (i.map, i.op) {
        (Map::Primary, 0x00..=0x3f) if i.op & 7 <= 5 && alu_of(i.op >> 3).is_some() => {
            (false, writes)
        }
        (Map::Primary, 0x80 | 0x81 | 0x83) if alu_of(i.ext()).is_some() => (false, writes),
        (Map::Primary, 0x84 | 0x85) => (false, writes),
        (Map::Primary, 0x88..=0x8b) => (false, none),
        (Map::Primary, 0x8d) if i.mem.is_some() && i.osize() >= 4 => (false, none),
        (Map::Primary, 0x63) if i.rex_w() => (false, none),
        (Map::Primary, 0x50..=0x5f | 0x68 | 0x6a) if i.osize64() == 8 => (false, none),
        (Map::Primary, 0x70..=0x7f) if cond(i.op).is_some() => (true, reads),
        (Map::Primary, 0x90) if i.rex & 1 == 0 => (false, none),
        (Map::Primary, 0xb0..=0xbf) => (false, none),
        (Map::Primary, 0xc1 | 0xd1) if matches!(i.ext(), 4 | 5 | 7) && i.osize() >= 4 => (
            false,
            FlagUse {
                reads: true,
                writes: true,
            },
        ),
        (Map::Primary, 0xc3) => (true, none),
        (Map::Primary, 0xc6 | 0xc7) if i.ext() == 0 => (false, none),
        (Map::Primary, 0xe8 | 0xe9 | 0xeb) => (true, none),
        (Map::Primary, 0xf7) if i.ext() == 0 && i.osize() >= 4 => (false, writes),
        (Map::Primary, 0xf6 | 0xf7) if i.ext() == 2 => (false, none),
        (Map::Primary, 0xf6 | 0xf7) if i.ext() == 3 => (false, writes),
        (Map::Primary, 0x69 | 0x6b) if i.osize() >= 4 => (false, writes),
        (Map::Primary, 0x98) if i.osize() >= 4 => (false, none),
        (Map::Primary, 0xff) if matches!(i.ext(), 0 | 1) && i.osize() >= 4 => (
            false,
            FlagUse {
                reads: true,
                writes: true,
            },
        ),
        (Map::Primary, 0xff) if matches!(i.ext(), 2 | 4) => (true, none),
        (Map::M0F, 0x1f) => (false, none),
        (Map::M0F, 0xae)
            if !i.opsize_prefix
                && matches!((i.ext(), i.mem.is_some()), (5..=7, false) | (7, true)) =>
        {
            (false, none)
        }
        (Map::M0F, 0xa3) if i.mem.is_none() && i.osize() >= 4 => (
            false,
            FlagUse {
                reads: true,
                writes: true,
            },
        ),
        (Map::M0F, 0xba) if i.ext() == 4 && i.osize() >= 4 => (
            false,
            FlagUse {
                reads: true,
                writes: true,
            },
        ),
        (Map::M0F, 0x40..=0x4f) if cond(i.op).is_some() && i.osize() >= 4 => (false, reads),
        (Map::M0F, 0x80..=0x8f) if cond(i.op).is_some() => (true, reads),
        (Map::M0F, 0x90..=0x9f) if cond(i.op).is_some() => (false, reads),
        (Map::M0F, 0xaf) if i.osize() >= 4 => (false, writes),
        (Map::M0F, 0xb6 | 0xb7 | 0xbe | 0xbf) if i.osize() >= 4 => (false, none),
        _ => return None,
    })
}

/// Decodes the block at `rip`: instructions up to and including the first
/// that ends it, stopping before one that cannot be translated.
fn scan(mem: &Memory, rip: u64) -> Vec<(Insn, u64, FlagUse)> {
    let mut out = Vec::new();
    let mut at = rip;
    while out.len() < MAX_INSNS {
        let insn = decode(|i| {
            let mut b = [0u8];
            mem.fetch(at.wrapping_add(i as u64), &mut b).map(|_| b[0])
        });
        let Ok(insn) = insn else { break };
        let Some((ends, flags)) = classify(&insn) else {
            break;
        };
        let len = insn.len as u64;
        out.push((insn, at, flags));
        at += len;
        if ends {
            break;
        }
    }
    out
}

/// Translates the block at `rip` for runtime offset `place`, with the
/// dispatcher at `dispatch`; `None` if its first instruction cannot be
/// translated.
pub fn block(mem: &Memory, rip: u64, place: usize, dispatch: usize) -> Option<Block> {
    let insns = scan(mem, rip);
    if insns.is_empty() {
        return None;
    }
    let mut live = vec![true; insns.len()];
    let mut after = true;
    for (k, (_, _, f)) in insns.iter().enumerate().rev() {
        live[k] = after;
        after = f.reads || (after && !f.writes);
    }
    let mut t = Translator {
        a: Asm::new(),
        place,
        dispatch,
        insns: Vec::new(),
        chains: Vec::new(),
    };
    t.a.add_imm(Size::X, ICOUNT, ICOUNT, insns.len() as u32);
    let mut ended = false;
    for (k, (insn, at, _)) in insns.iter().enumerate() {
        t.insns.push((t.a.pos() as u32, *at));
        ended = t.insn(insn, *at, live[k]);
    }
    if !ended {
        let (insn, at, _) = insns.last().expect("not empty");
        let next = at + insn.len as u64;
        let hvc = if insns.len() == MAX_INSNS {
            None
        } else {
            Some(HVC_INTERP)
        };
        t.exit_to(next, hvc);
    }
    Some(Block {
        code: t.a.finish(),
        insns: t.insns,
        chains: t.chains,
    })
}

struct Translator {
    a: Asm,
    place: usize,
    dispatch: usize,
    insns: Vec<(u32, u64)>,
    chains: Vec<(u32, u64)>,
}

fn sz(size: u8) -> Size {
    if size == 8 { Size::X } else { Size::W }
}

fn width(size: u8) -> Width {
    match size {
        1 => Width::B,
        2 => Width::H,
        4 => Width::W,
        _ => Width::X,
    }
}

fn mask(size: u8) -> u64 {
    if size == 8 {
        u64::MAX
    } else {
        (1u64 << (size * 8)) - 1
    }
}

impl Translator {
    /// Continues at guest `rip`: through the dispatcher, or with an `hvc`.
    fn exit_to(&mut self, rip: u64, hvc: Option<u16>) {
        self.a.mov_imm(Size::X, RIP, rip);
        if hvc.is_none() {
            self.chains.push((self.a.pos() as u32, rip));
        }
        self.jump(hvc);
    }

    /// Continues at the guest RIP already in x16.
    fn jump(&mut self, hvc: Option<u16>) {
        match hvc {
            Some(n) => self.a.hvc(n),
            None => {
                let here = self.place + self.a.pos() * 4;
                self.a
                    .b_offset(((self.dispatch as i64 - here as i64) / 4) as i32);
            }
        }
    }

    /// x19 = the address of `m` (`next` is the following instruction's RIP).
    fn addr(&mut self, m: &Mem, next: u64) {
        let a = &mut self.a;
        if m.rip_relative {
            a.mov_imm(Size::X, ADDR, next.wrapping_add(m.disp as i64 as u64));
        } else {
            match m.base {
                Some(b) => a.mov(Size::X, ADDR, r(b)),
                None => a.mov_imm(Size::X, ADDR, 0),
            }
            if let Some(i) = m.index {
                a.add_shift(
                    Size::X,
                    ADDR,
                    ADDR,
                    r(i),
                    Shift::Lsl,
                    m.scale.trailing_zeros(),
                );
            }
            let d = m.disp as i64;
            if d > 0 && d < 4096 {
                a.add_imm(Size::X, ADDR, ADDR, d as u32);
            } else if d < 0 && d > -4096 {
                a.sub_imm(Size::X, ADDR, ADDR, (-d) as u32);
            } else if d != 0 {
                a.mov_imm(Size::X, T, d as u64);
                a.add(Size::X, ADDR, ADDR, T);
            }
        }
        if m.addr32 {
            a.mov(Size::W, ADDR, ADDR);
        }
        if m.seg == Some(Seg::Fs) {
            a.add(Size::X, ADDR, ADDR, FS_BASE);
        }
    }

    fn load(&mut self, dst: Reg, size: u8) {
        self.a.ldr(width(size), dst, ADDR, 0);
    }

    fn store(&mut self, src: Reg, size: u8) {
        self.a.str(width(size), src, ADDR, 0);
    }

    /// `dst` = guest register `n` at `size`, zero-extended; for 1-byte
    /// operands without REX, 4..7 are AH, CH, DH, BH.
    fn get(&mut self, dst: Reg, n: u8, size: u8, rex: bool) {
        match size {
            8 => self.a.mov(Size::X, dst, r(n)),
            4 => self.a.mov(Size::W, dst, r(n)),
            2 => self.a.uxth(dst, r(n)),
            _ if !rex && (4..8).contains(&n) => self.a.ubfx(Size::X, dst, r(n - 4), 8, 8),
            _ => self.a.uxtb(dst, r(n)),
        }
    }

    /// Guest register `n` at `size` = `src`, with x86's rules for the rest
    /// of the register.
    fn set(&mut self, n: u8, size: u8, rex: bool, src: Reg) {
        match size {
            8 => self.a.mov(Size::X, r(n), src),
            4 => self.a.mov(Size::W, r(n), src),
            2 => self.a.bfi(Size::X, r(n), src, 0, 16),
            _ if !rex && (4..8).contains(&n) => self.a.bfi(Size::X, r(n - 4), src, 8, 8),
            _ => self.a.bfi(Size::X, r(n), src, 0, 8),
        }
    }

    /// `A` = the r/m operand (loaded if in memory, its address left in x19).
    fn get_rm(&mut self, i: &Insn, size: u8, next: u64) {
        match i.mem {
            Some(m) => {
                self.addr(&m, next);
                self.load(A, size);
            }
            None => self.get(A, i.rm, size, i.rex != 0),
        }
    }

    fn set_rm(&mut self, i: &Insn, size: u8, src: Reg) {
        match i.mem {
            Some(_) => self.store(src, size),
            None => self.set(i.rm, size, i.rex != 0, src),
        }
    }

    /// `T` = `A op B` at `size`, setting the flags when `flags`. Narrow
    /// operands are shifted to the top of a W register so NZCV fits them.
    fn alu(&mut self, op: Alu, size: u8, flags: bool) {
        let a = &mut self.a;
        let s = sz(size);
        let shift = if size < 4 { 32 - 8 * size as u32 } else { 0 };
        let (x, y) = if shift > 0 {
            a.lsl_imm(Size::W, U, A, shift);
            a.lsl_imm(Size::W, T, B, shift);
            (U, T)
        } else {
            (A, B)
        };
        match (op, flags) {
            (Alu::Add, true) => a.adds(s, T, x, y),
            (Alu::Add, false) => a.add(s, T, x, y),
            (Alu::Sub, true) => a.subs(s, T, x, y),
            (Alu::Sub, false) => a.sub(s, T, x, y),
            (Alu::And, true) => a.ands(s, T, x, y),
            (Alu::And, false) => a.and(s, T, x, y),
            (Alu::Or, _) => a.orr(s, T, x, y),
            (Alu::Xor, _) => a.eor(s, T, x, y),
        }
        if flags {
            match op {
                Alu::Add | Alu::And => a.cfinv(),
                Alu::Sub => {}
                Alu::Or | Alu::Xor => {
                    a.tst(s, T, T);
                    a.cfinv();
                }
            }
        }
        if shift > 0 {
            a.lsr_imm(Size::W, T, T, shift);
        }
        if flags {
            a.mov(Size::X, PF_RESULT, T);
            if matches!(op, Alu::Add | Alu::Sub) {
                a.eor(Size::X, AF_AUX, A, B);
            } else {
                a.mov(Size::X, AF_AUX, T);
            }
        }
    }

    /// Translates one instruction; true if it ended the block.
    fn insn(&mut self, i: &Insn, rip: u64, flags: bool) -> bool {
        let next = rip + i.len as u64;
        if vector::classify(i).is_some() {
            self.vector(i, next, flags);
            return false;
        }
        let rex = i.rex != 0;
        match (i.map, i.op) {
            (Map::Primary, 0x00..=0x3f) => {
                let (op, keep) = alu_of(i.op >> 3).expect("classified");
                let size = if matches!(i.op & 7, 0 | 2 | 4) {
                    1
                } else {
                    i.osize()
                };
                match i.op & 7 {
                    0 | 1 => {
                        self.get_rm(i, size, next);
                        self.get(B, i.reg, size, rex);
                        self.alu(op, size, flags);
                        if keep {
                            self.set_rm(i, size, T);
                        }
                    }
                    2 | 3 => {
                        match i.mem {
                            Some(m) => {
                                self.addr(&m, next);
                                self.load(B, size);
                            }
                            None => self.get(B, i.rm, size, rex),
                        }
                        self.get(A, i.reg, size, rex);
                        self.alu(op, size, flags);
                        if keep {
                            self.set(i.reg, size, rex, T);
                        }
                    }
                    _ => {
                        self.get(A, 0, size, rex);
                        self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                        self.alu(op, size, flags);
                        if keep {
                            self.set(0, size, rex, T);
                        }
                    }
                }
            }
            (Map::Primary, 0x80 | 0x81 | 0x83) => {
                let (op, keep) = alu_of(i.ext()).expect("classified");
                let size = if i.op == 0x80 { 1 } else { i.osize() };
                self.get_rm(i, size, next);
                self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                self.alu(op, size, flags);
                if keep {
                    self.set_rm(i, size, T);
                }
            }
            (Map::Primary, 0x84 | 0x85) => {
                let size = if i.op == 0x84 { 1 } else { i.osize() };
                self.get_rm(i, size, next);
                self.get(B, i.reg, size, rex);
                self.alu(Alu::And, size, flags);
            }
            (Map::Primary, 0xf6 | 0xf7) if i.ext() == 2 => {
                let size = if i.op == 0xf6 { 1 } else { i.osize() };
                self.get_rm(i, size, next);
                self.a.mvn(Size::X, T, A);
                self.set_rm(i, size, T);
            }
            (Map::Primary, 0xf6 | 0xf7) if i.ext() == 3 => {
                let size = if i.op == 0xf6 { 1 } else { i.osize() };
                self.get_rm(i, size, next);
                self.a.mov(Size::X, B, A);
                self.a.mov_imm(Size::X, A, 0);
                self.alu(Alu::Sub, size, flags);
                self.set_rm(i, size, T);
            }
            (Map::Primary, 0xf7) => {
                let size = i.osize();
                self.get_rm(i, size, next);
                self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                self.alu(Alu::And, size, flags);
            }
            (Map::Primary, 0x98) => {
                if i.osize() == 8 {
                    self.a.sxtw(r(0), r(0));
                } else {
                    self.a.sxth(Size::W, r(0), r(0));
                }
            }
            (Map::Primary, 0x88..=0x8b) => {
                let size = if i.op & 1 == 0 { 1 } else { i.osize() };
                if i.op & 2 == 0 {
                    self.get(B, i.reg, size, rex);
                    match i.mem {
                        Some(m) => {
                            self.addr(&m, next);
                            self.store(B, size);
                        }
                        None => self.set(i.rm, size, rex, B),
                    }
                } else {
                    self.get_rm(i, size, next);
                    self.set(i.reg, size, rex, A);
                }
            }
            (Map::Primary, 0xc6 | 0xc7) => {
                let size = if i.op == 0xc6 { 1 } else { i.osize() };
                if let Some(m) = i.mem {
                    self.addr(&m, next);
                }
                self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                self.set_rm(i, size, B);
            }
            (Map::Primary, 0xb0..=0xbf) => {
                let size = if i.op < 0xb8 { 1 } else { i.osize() };
                self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                self.set(i.rm, size, rex, B);
            }
            (Map::Primary, 0x8d) => {
                let m = i.mem.expect("classified");
                self.addr(&Mem { seg: None, ..m }, next);
                self.set(i.reg, i.osize(), rex, ADDR);
            }
            (Map::Primary, 0x63) => {
                self.get_rm(i, 4, next);
                self.a.sxtw(A, A);
                self.set(i.reg, 8, rex, A);
            }
            (Map::M0F, 0xb6 | 0xb7 | 0xbe | 0xbf) => {
                let from = if i.op & 1 == 0 { 1 } else { 2 };
                self.get_rm(i, from, next);
                let size = i.osize();
                if i.op >= 0xbe {
                    match from {
                        1 => self.a.sxtb(sz(size), A, A),
                        _ => self.a.sxth(sz(size), A, A),
                    }
                }
                self.set(i.reg, size, rex, A);
            }
            (Map::Primary, 0x50..=0x57) => {
                self.a.sub_imm(Size::X, ADDR, r(4), 8);
                self.a.str(Width::X, r(i.rm), ADDR, 0);
                self.a.mov(Size::X, r(4), ADDR);
            }
            (Map::Primary, 0x58..=0x5f) => {
                self.a.ldr(Width::X, A, r(4), 0);
                self.a.add_imm(Size::X, r(4), r(4), 8);
                self.a.mov(Size::X, r(i.rm), A);
            }
            (Map::Primary, 0x68 | 0x6a) => self.push_const(i.imm as u64),
            (Map::Primary, 0x90) | (Map::M0F, 0x1f) => {}
            (Map::M0F, 0xae) => {
                if i.ext() == 6 {
                    self.a.dmb(0xb);
                }
            }
            (Map::M0F, 0xa3 | 0xba) => {
                let size = i.osize();
                let bits = size as u64 * 8;
                self.get_rm(i, size, next);
                if i.op == 0xa3 {
                    self.get(B, i.reg, size, rex);
                } else {
                    self.a.mov_imm(Size::X, B, i.imm as u64 & (bits - 1));
                }
                self.a.lsrv(sz(size), T, A, B);
                self.a.mrs(U, a64::sysreg::NZCV);
                assert!(self.a.logic_imm(Logic::Eor, Size::X, T, T, 1));
                self.a.bfi(Size::X, U, T, 29, 1);
                self.a.msr(a64::sysreg::NZCV, U);
            }
            (Map::Primary, 0xc1 | 0xd1) => {
                let size = i.osize();
                let bits = size as u32 * 8;
                let n = if i.op == 0xd1 {
                    1
                } else {
                    i.imm as u32 & (bits - 1)
                };
                if n == 0 {
                    return false;
                }
                self.get_rm(i, size, next);
                if flags {
                    self.shift_flags(i.ext(), size, n);
                } else {
                    let s = sz(size);
                    match i.ext() {
                        4 => self.a.lsl_imm(s, T, A, n),
                        5 => self.a.lsr_imm(s, T, A, n),
                        _ => self.a.asr_imm(s, T, A, n),
                    }
                }
                self.set_rm(i, size, T);
            }
            (Map::M0F, 0xaf) | (Map::Primary, 0x69 | 0x6b) => {
                let size = i.osize();
                self.get_rm(i, size, next);
                if i.map == Map::M0F {
                    self.get(B, i.reg, size, rex);
                } else {
                    self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                }
                if size == 8 {
                    self.a.mul(Size::X, T, A, B);
                    if flags {
                        self.a.smulh(U, A, B);
                        self.a.asr_imm(Size::X, B, T, 63);
                        self.a.cmp(Size::X, U, B);
                    }
                } else {
                    self.a.sxtw(A, A);
                    self.a.sxtw(B, B);
                    self.a.mul(Size::X, T, A, B);
                    if flags {
                        self.a.sxtw(B, T);
                        self.a.cmp(Size::X, T, B);
                    }
                }
                if flags {
                    self.imul_flags(size);
                }
                self.set(i.reg, size, rex, T);
            }
            (Map::Primary, 0xff) if matches!(i.ext(), 0 | 1) => {
                let size = i.osize();
                self.get_rm(i, size, next);
                self.a.mov_imm(Size::X, B, 1);
                if flags {
                    self.a.mrs(U, a64::sysreg::NZCV);
                }
                self.alu(if i.ext() == 0 { Alu::Add } else { Alu::Sub }, size, flags);
                if flags {
                    self.keep_carry();
                }
                self.set_rm(i, size, T);
            }
            (Map::M0F, 0x40..=0x4f) => {
                let size = i.osize();
                self.get_rm(i, size, next);
                self.get(B, i.reg, size, rex);
                let c = cond(i.op).expect("classified");
                self.a.csel(sz(size), T, A, B, c);
                self.set(i.reg, size, rex, T);
            }
            (Map::M0F, 0x90..=0x9f) => {
                if let Some(m) = i.mem {
                    self.addr(&m, next);
                }
                self.a.cset(Size::W, T, cond(i.op).expect("classified"));
                self.set_rm(i, 1, T);
            }
            (Map::Primary, 0x70..=0x7f) | (Map::M0F, 0x80..=0x8f) => {
                let target = next.wrapping_add(i.imm as u64);
                let taken = self.a.label();
                self.a.b_cond(cond(i.op).expect("classified"), taken);
                self.exit_to(next, None);
                self.a.bind(taken);
                self.exit_to(target, None);
                return true;
            }
            (Map::Primary, 0xe9 | 0xeb) => {
                self.exit_to(next.wrapping_add(i.imm as u64), None);
                return true;
            }
            (Map::Primary, 0xe8) => {
                self.push_const(next);
                self.exit_to(next.wrapping_add(i.imm as u64), None);
                return true;
            }
            (Map::Primary, 0xc3) => {
                self.a.ldr(Width::X, RIP, r(4), 0);
                self.a.add_imm(Size::X, r(4), r(4), 8);
                self.jump(None);
                return true;
            }
            (Map::Primary, 0xff) => {
                self.get_rm(i, 8, next);
                if i.ext() == 2 {
                    self.push_const(next);
                }
                self.a.mov(Size::X, RIP, A);
                self.jump(None);
                return true;
            }
            _ => unreachable!("classify accepted an instruction translate does not handle"),
        }
        false
    }

    fn push_const(&mut self, v: u64) {
        self.a.sub_imm(Size::X, ADDR, r(4), 8);
        self.a.mov_imm(Size::X, B, v);
        self.a.str(Width::X, B, ADDR, 0);
        self.a.mov(Size::X, r(4), ADDR);
    }

    /// Puts back the C flag saved in U, since INC and DEC leave CF alone.
    fn keep_carry(&mut self) {
        let a = &mut self.a;
        a.mrs(B, a64::sysreg::NZCV);
        assert!(a.logic_imm(Logic::And, Size::W, U, U, C_BIT));
        assert!(a.logic_imm(Logic::And, Size::W, B, B, !C_BIT & 0xffff_ffff));
        a.orr(Size::X, B, B, U);
        a.msr(a64::sysreg::NZCV, B);
    }

    /// `T` = SHL/SHR/SAR of `A` by `n`, with x86's flags: CF the last bit
    /// out, OF for a count of 1; AF (and OF otherwise) keep their old values.
    fn shift_flags(&mut self, ext: u8, size: u8, n: u32) {
        let s = sz(size);
        let bits = size as u32 * 8;
        let (old, old_af) = (OLD_NZCV, OLD_AF);
        let a = &mut self.a;
        a.mrs(old, a64::sysreg::NZCV);
        a.eor(Size::X, old_af, PF_RESULT, AF_AUX);
        assert!(a.logic_imm(Logic::And, Size::X, old_af, old_af, 0x10));
        let carry_bit = if ext == 4 { bits - n } else { n - 1 };
        match ext {
            4 => a.lsl_imm(s, T, A, n),
            5 => a.lsr_imm(s, T, A, n),
            _ => a.asr_imm(s, T, A, n),
        }
        a.ubfx(Size::X, U, A, carry_bit, 1);
        match (ext, n) {
            (4, 1) => {
                a.lsr_imm(s, B, T, bits - 1);
                a.eor(Size::X, B, B, U);
            }
            (5, 1) => a.lsr_imm(s, B, A, bits - 1),
            (_, 1) => a.mov_imm(Size::X, B, 0),
            _ => a.ubfx(Size::X, B, old, 28, 1),
        }
        a.tst(s, T, T);
        a.mrs(PF_RESULT, a64::sysreg::NZCV);
        assert!(a.logic_imm(Logic::Eor, Size::X, U, U, 1));
        a.bfi(Size::X, PF_RESULT, U, 29, 1);
        a.bfi(Size::X, PF_RESULT, B, 28, 1);
        a.msr(a64::sysreg::NZCV, PF_RESULT);
        a.mov(Size::X, PF_RESULT, T);
        a.eor(Size::X, AF_AUX, T, old_af);
    }

    /// IMUL's flags from the comparison just made (EQ when the product
    /// fits): CF = OF = it did not fit; SF, ZF from the result.
    fn imul_flags(&mut self, size: u8) {
        let a = &mut self.a;
        a.cset(Size::X, U, Cond::Ne);
        a.tst(sz(size), T, T);
        a.mrs(B, a64::sysreg::NZCV);
        a.bfi(Size::X, B, U, 28, 1);
        assert!(a.logic_imm(Logic::Eor, Size::X, U, U, 1));
        a.bfi(Size::X, B, U, 29, 1);
        a.msr(a64::sysreg::NZCV, B);
        a.mov(Size::X, PF_RESULT, T);
        a.mov(Size::X, AF_AUX, T);
    }
}

/// Whether a translated block could start at `rip`.
pub fn starts_block(mem: &Memory, rip: u64) -> bool {
    let insn = decode(|i| {
        let mut b = [0u8];
        mem.fetch(rip.wrapping_add(i as u64), &mut b).map(|_| b[0])
    });
    insn.is_ok_and(|i| classify(&i).is_some())
}
