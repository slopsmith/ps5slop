//! Translates a basic block of x86-64 into AArch64. Register and flag
//! conventions are in `docs/jit.md`, "Translated code".

use a64::{Asm, Cond, Label, Logic, Reg, Shift, Size, V, Width, r};
use memory::Memory;
use x86::decode::{Insn, Map, Mem, Rep, Seg, decode};

mod tier2;
mod vector;
mod x87;
pub use tier2::{Tier2, region};
pub use vector::CTX;

const DF_OFFSET: i32 = crate::layout::DF_OFFSET as i32;

/// Exits translated code takes, as `hvc` numbers; x16 holds the guest RIP.
pub const HVC_MISS: u16 = 1;
pub const HVC_INTERP: u16 = 2;
/// A native library function hands its call to the host function.
pub const HVC_HOST: u16 = 3;
/// A block's run count reached zero: compile its region (tier 2).
pub const HVC_TIERUP: u16 = 4;
/// Words of the counting prologue at the start of a block that counts.
pub const PROLOGUE_WORDS: usize = 5;

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
/// CNT V0.8B, V0.8B and ADDV B0, V0.8B.
const CNT_8B: u32 = 0x0e20_5800;
const ADDV_B: u32 = 0x0e31_b800;
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
    Adc,
    Sbb,
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
        2 => (Alu::Adc, true),
        3 => (Alu::Sbb, true),
        4 => (Alu::And, true),
        5 => (Alu::Sub, true),
        6 => (Alu::Xor, true),
        7 => (Alu::Sub, false),
        _ => return None,
    })
}

/// Whether a LOCK prefix on `i` is one the translator turns into an atomic.
fn lockable(i: &Insn) -> bool {
    let logic = |n: u8| matches!(n, 0 | 1 | 4 | 5 | 6);
    i.mem.is_some()
        && match (i.map, i.op) {
            (Map::Primary, 0x00..=0x3f) => i.op & 7 <= 1 && logic(i.op >> 3),
            (Map::Primary, 0x80 | 0x81 | 0x83) => logic(i.ext()),
            (Map::Primary, 0x86 | 0x87) => true,
            (Map::Primary, 0xfe | 0xff) => i.ext() <= 1,
            (Map::M0F, 0xb0 | 0xb1 | 0xc0 | 0xc1 | 0xab | 0xb3 | 0xbb) => true,
            (Map::M0F, 0xba) => i.ext() >= 5,
            _ => false,
        }
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
    let both = FlagUse {
        reads: true,
        writes: true,
    };
    let counts = i.rep == Rep::Rep && i.map == Map::M0F && matches!(i.op, 0xb8 | 0xbc | 0xbd);
    let crc32 = i.rep == Rep::Repne && i.map == Map::M0F38 && matches!(i.op, 0xf0 | 0xf1);
    let string = i.map == Map::Primary && matches!(i.op, 0xa4..=0xa7 | 0xaa..=0xaf);
    if i.vex.is_none()
        && !i.lock
        && i.seg != Some(Seg::Gs)
        && i.rep == Rep::None
        && let Some(c) = x87::classify(i)
    {
        return Some(c);
    }
    if i.vex.is_some()
        || (i.lock && !lockable(i))
        || i.seg == Some(Seg::Gs)
        || (i.rep != Rep::None && !counts && !crc32 && !string)
    {
        return None;
    }
    let carries = |n: u8| matches!(n, 2 | 3);
    Some(match (i.map, i.op) {
        (Map::Primary, 0x00..=0x3f) if i.op & 7 <= 5 && alu_of(i.op >> 3).is_some() => {
            if !carries(i.op >> 3) {
                (false, writes)
            } else if i.op & 1 == 1 && i.osize() >= 4 {
                (false, both)
            } else {
                return None;
            }
        }
        (Map::Primary, 0x80 | 0x81 | 0x83) if alu_of(i.ext()).is_some() => {
            if !carries(i.ext()) {
                (false, writes)
            } else if i.op != 0x80 && i.osize() >= 4 {
                (false, both)
            } else {
                return None;
            }
        }
        (Map::Primary, 0x86 | 0x87 | 0x91..=0x97) => (false, none),
        (Map::Primary, 0xa8 | 0xa9) => (false, writes),
        (Map::Primary, 0xc1 | 0xd1 | 0xd3) if i.ext() <= 1 && i.osize() >= 4 => (false, both),
        (Map::Primary, 0xc0 | 0xd0) if i.ext() != 2 && i.ext() != 3 => (false, both),
        (Map::Primary, 0xc1 | 0xd1) if i.ext() != 2 && i.ext() != 3 && i.osize() == 2 => {
            (false, both)
        }
        (Map::Primary, 0xf6 | 0xf7) if i.ext() == 0 => (false, writes),
        (Map::Primary, 0xf7) if matches!(i.ext(), 4 | 5) && i.osize() >= 4 => (false, writes),
        (Map::Primary, 0xf7) if matches!(i.ext(), 6 | 7) && i.osize() >= 4 => (false, none),
        (Map::Primary, 0xfe | 0xff) if i.ext() <= 1 => (false, both),
        (Map::M0F, 0xb0 | 0xb1) if i.mem.is_some() => (false, writes),
        (Map::M0F, 0xc0 | 0xc1) => (false, writes),
        (Map::M0F, 0xbc | 0xbd) if i.osize() >= 4 => (false, both),
        (Map::M0F, 0xc3) if i.mem.is_some() => (false, none),
        (Map::Primary, 0x84 | 0x85) => (false, writes),
        (Map::Primary, 0x88..=0x8b) => (false, none),
        (Map::Primary, 0x8d) if i.mem.is_some() && i.osize() >= 4 => (false, none),
        (Map::Primary, 0x63) if i.rex_w() => (false, none),
        (Map::Primary, 0x50..=0x5f | 0x68 | 0x6a) if i.osize64() == 8 => (false, none),
        (Map::Primary, 0x70..=0x7f) => (true, reads),
        (Map::Primary, 0xfc | 0xfd) => (false, none),
        (Map::Primary, 0xa4..=0xa7 | 0xaa..=0xaf) if !i.addr32 && i.seg.is_none() => {
            match (
                matches!(i.op, 0xa6 | 0xa7 | 0xae | 0xaf),
                i.rep != Rep::None,
            ) {
                (false, _) => (false, none),
                (true, false) => (false, writes),
                (true, true) => (false, both),
            }
        }
        (Map::M0F38, 0xf0 | 0xf1) if crc32 || (i.rep == Rep::None && i.mem.is_some()) => {
            (false, none)
        }
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
        (Map::Primary, 0xe2 | 0xe3) if !i.addr32 => (true, none),
        (Map::Primary, 0xf6 | 0xf7) if i.ext() == 2 => (false, none),
        (Map::Primary, 0xf6 | 0xf7) if i.ext() == 3 => (false, writes),
        (Map::Primary, 0x69 | 0x6b) => (false, writes),
        (Map::Primary, 0x98) if i.osize() >= 4 => (false, none),
        (Map::Primary, 0xff) if matches!(i.ext(), 2 | 4) => (true, none),
        (Map::M0F, 0x0d | 0x18..=0x1f) => (false, none),
        (Map::M0F, 0xc8..=0xcf) if i.osize() >= 4 => (false, none),
        (Map::M0F, 0xae)
            if !i.opsize_prefix
                && matches!((i.ext(), i.mem.is_some()), (5..=7, false) | (7, true)) =>
        {
            (false, none)
        }
        (Map::M0F, 0xa3 | 0xab | 0xb3 | 0xbb) if i.osize() >= 4 => (false, both),
        (Map::M0F, 0xba) if i.ext() >= 4 && i.osize() >= 4 => (false, both),
        (Map::Primary, 0x99) => (false, none),
        (Map::Primary, 0xd2 | 0xd3) if i.ext() >= 4 => (false, both),
        (Map::Primary, 0xc9) if i.osize64() == 8 => (false, none),
        (Map::Primary, 0xff) if i.ext() == 6 && i.osize64() == 8 => (false, none),
        (Map::M0F, 0x40..=0x4f) if i.osize() >= 4 => (false, reads),
        (Map::M0F, 0x80..=0x8f) => (true, reads),
        (Map::M0F, 0x90..=0x9f) => (false, reads),
        (Map::M0F, 0xaf) => (false, writes),
        (Map::M0F, 0xb8) if i.rep == Rep::Rep => (false, writes),
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
pub fn block(
    mem: &Memory,
    rip: u64,
    place: usize,
    dispatch: usize,
    count: Option<u32>,
) -> Option<Block> {
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
        bails: Vec::new(),
        cur: (rip, 0),
        x87_mode: false,
    };
    let counter = count.map(|n| {
        let (word, up) = (t.a.label(), t.a.label());
        let (at, left) = (Reg(17), Reg(18));
        t.a.adr(at, word);
        t.a.ldr(Width::W, left, at, 0);
        t.a.sub_imm(Size::W, left, left, 1);
        t.a.str(Width::W, left, at, 0);
        t.a.cbz(Size::W, left, up);
        (word, up, n)
    });
    t.a.add_imm(Size::X, ICOUNT, ICOUNT, insns.len() as u32);
    let mut ended = false;
    for (k, (insn, at, _)) in insns.iter().enumerate() {
        t.insns.push((t.a.pos() as u32, *at));
        t.cur = (*at, (insns.len() - k) as u32);
        let x87 = x87::classify(insn).is_some();
        if !x87 && (vector::classify(insn).is_some() || classify(insn).is_some_and(|c| c.0)) {
            t.leave_x87();
        }
        ended = t.insn(insn, *at, live[k]);
    }
    if !ended {
        t.leave_x87();
        let (insn, at, _) = insns.last().expect("not empty");
        let next = at + insn.len as u64;
        let hvc = if insns.len() == MAX_INSNS {
            None
        } else {
            Some(HVC_INTERP)
        };
        t.exit_to(next, hvc);
    }
    for (l, at, left, x87) in std::mem::take(&mut t.bails) {
        t.a.bind(l);
        if x87 {
            t.restore_fp();
        }
        t.a.sub_imm(Size::X, ICOUNT, ICOUNT, left);
        t.exit_to(at, Some(HVC_INTERP));
    }
    if let Some((word, up, n)) = counter {
        t.a.bind(up);
        t.a.mov_imm(Size::X, RIP, rip);
        t.a.hvc(HVC_TIERUP);
        t.a.bind(word);
        t.a.emit(n);
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
    /// Exits that leave the current instruction to the interpreter, as
    /// (label, its RIP, instructions of the block not run, x87 FP mode on).
    bails: Vec<(Label, u64, u32, bool)>,
    cur: (u64, u32),
    /// Whether the code emitted so far runs with x87's FPCR.
    x87_mode: bool,
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
    /// A label that hands the current instruction to the interpreter; code
    /// branching there must not have changed guest state or NZCV.
    fn bail(&mut self) -> Label {
        let l = self.a.label();
        self.bails.push((l, self.cur.0, self.cur.1, self.x87_mode));
        l
    }

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
            (Alu::Adc, f) => {
                a.cfinv();
                if f {
                    a.adcs(s, T, x, y);
                } else {
                    a.adc(s, T, x, y);
                }
            }
            (Alu::Sbb, true) => a.sbcs(s, T, x, y),
            (Alu::Sbb, false) => a.sbc(s, T, x, y),
        }
        if flags {
            match op {
                Alu::Add | Alu::And | Alu::Adc => a.cfinv(),
                Alu::Sub | Alu::Sbb => {}
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
            if matches!(op, Alu::Add | Alu::Sub | Alu::Adc | Alu::Sbb) {
                a.eor(Size::X, AF_AUX, A, B);
            } else {
                a.mov(Size::X, AF_AUX, T);
            }
        }
    }

    /// Translates one instruction; true if it ended the block.
    fn insn(&mut self, i: &Insn, rip: u64, flags: bool) -> bool {
        let next = rip + i.len as u64;
        if x87::classify(i).is_some() {
            self.x87(i, next);
            return false;
        }
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
                        self.get(B, i.reg, size, rex);
                        self.alu_rm(i, op, keep, size, next, flags);
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
                self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                self.alu_rm(i, op, keep, size, next, flags);
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
            (Map::Primary, 0xf6 | 0xf7) if i.ext() == 0 => {
                let size = if i.op == 0xf6 { 1 } else { i.osize() };
                self.get_rm(i, size, next);
                self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                self.alu(Alu::And, size, flags);
            }
            (Map::Primary, 0xa8 | 0xa9) => {
                let size = if i.op == 0xa8 { 1 } else { i.osize() };
                self.get(A, 0, size, rex);
                self.a.mov_imm(Size::X, B, i.imm as u64 & mask(size));
                self.alu(Alu::And, size, flags);
            }
            (Map::Primary, 0xf7) if matches!(i.ext(), 4 | 5) => self.mul_wide(i, next, flags),
            (Map::Primary, 0xf7) => self.div(i, next),
            (Map::Primary, 0xd2 | 0xd3) if i.ext() >= 4 => self.shift_cl(i, next, flags),
            (Map::Primary, 0xc0 | 0xd0) => self.narrow_shift(i, next, flags),
            (Map::Primary, 0xc1 | 0xd1) if i.osize() == 2 => self.narrow_shift(i, next, flags),
            (Map::Primary, 0xc1 | 0xd1 | 0xd3) if i.ext() <= 1 => self.rotate(i, next, flags),
            (Map::Primary, 0x86 | 0x87 | 0x91..=0x97) => self.xchg(i, next),
            (Map::M0F, 0xb0 | 0xb1 | 0xc0 | 0xc1) => self.atomic(i, next, flags),
            (Map::M0F, 0xbc | 0xbd) => self.bit_scan(i, next, flags),
            (Map::M0F, 0xc3) => {
                let size = i.osize();
                self.get(B, i.reg, size, rex);
                self.addr(&i.mem.expect("classified"), next);
                self.store(B, size);
            }
            (Map::Primary, 0x99) => match i.osize() {
                8 => self.a.asr_imm(Size::X, r(2), r(0), 63),
                4 => self.a.asr_imm(Size::W, r(2), r(0), 31),
                _ => {
                    self.a.sxth(Size::W, T, r(0));
                    self.a.asr_imm(Size::W, T, T, 15);
                    self.set(2, 2, rex, T);
                }
            },
            (Map::Primary, 0xc9) => {
                self.a.ldr(Width::X, A, r(5), 0);
                self.a.add_imm(Size::X, r(4), r(5), 8);
                self.a.mov(Size::X, r(5), A);
            }
            (Map::Primary, 0xff) if i.ext() == 6 => {
                self.get_rm(i, 8, next);
                self.a.sub_imm(Size::X, ADDR, r(4), 8);
                self.a.str(Width::X, A, ADDR, 0);
                self.a.mov(Size::X, r(4), ADDR);
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
            (Map::Primary, 0x90) | (Map::M0F, 0x0d | 0x18..=0x1f) => {}
            (Map::M0F, 0xc8..=0xcf) => {
                let s = sz(i.osize());
                self.a.rev(s, r(i.rm), r(i.rm));
            }
            (Map::Primary, 0xfc) => self.a.str(Width::X, Reg(31), CTX, DF_OFFSET),
            (Map::Primary, 0xfd) => {
                self.a.mov_imm(Size::X, B, u64::MAX);
                self.a.str(Width::X, B, CTX, DF_OFFSET);
            }
            (Map::Primary, 0xa4..=0xa7 | 0xaa..=0xaf) => self.string(i),
            (Map::M0F38, 0xf0 | 0xf1) => self.movbe_crc32(i, next),
            (Map::M0F, 0xae) => {
                if i.ext() == 6 {
                    self.a.dmb(0xb);
                }
            }
            (Map::M0F, 0xa3 | 0xab | 0xb3 | 0xbb | 0xba) => self.bit_test(i, next, flags),
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
            (Map::M0F, 0xb8) => self.popcnt(i, next, flags),
            (Map::M0F, 0xaf) | (Map::Primary, 0x69 | 0x6b) if i.osize() == 2 => {
                self.get_rm(i, 2, next);
                if i.map == Map::M0F {
                    self.get(B, i.reg, 2, rex);
                } else {
                    self.a.mov_imm(Size::X, B, i.imm as u64 & 0xffff);
                }
                let a = &mut self.a;
                a.sxth(Size::W, A, A);
                a.sxth(Size::W, B, B);
                a.mul(Size::W, T, A, B);
                if flags {
                    a.sxth(Size::W, U, T);
                    a.cmp(Size::W, T, U);
                    a.cset(Size::X, U, Cond::Ne);
                    a.lsl_imm(Size::W, B, T, 16);
                    a.tst(Size::W, B, B);
                    a.mrs(B, a64::sysreg::NZCV);
                    a.bfi(Size::X, B, U, 28, 1);
                    assert!(a.logic_imm(Logic::Eor, Size::X, U, U, 1));
                    a.bfi(Size::X, B, U, 29, 1);
                    a.msr(a64::sysreg::NZCV, B);
                    a.uxth(PF_RESULT, T);
                    a.mov(Size::X, AF_AUX, PF_RESULT);
                }
                self.set(i.reg, 2, rex, T);
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
            (Map::Primary, 0xfe | 0xff) if i.ext() <= 1 => {
                let size = if i.op == 0xfe { 1 } else { i.osize() };
                let op = if i.ext() == 0 { Alu::Add } else { Alu::Sub };
                if flags {
                    self.a.mrs(OLD_NZCV, a64::sysreg::NZCV);
                }
                if i.lock {
                    self.addr(&i.mem.expect("lockable"), next);
                    let delta = if op == Alu::Add { 1 } else { u64::MAX };
                    self.a.mov_imm(Size::X, B, delta);
                    self.a.ldaddal(width(size), B, A, ADDR);
                    if flags {
                        self.a.mov_imm(Size::X, B, 1);
                        self.alu(op, size, true);
                        self.keep_carry();
                    }
                } else {
                    self.get_rm(i, size, next);
                    self.a.mov_imm(Size::X, B, 1);
                    self.alu(op, size, flags);
                    if flags {
                        self.keep_carry();
                    }
                    self.set_rm(i, size, T);
                }
            }
            (Map::M0F, 0x40..=0x4f) => {
                let size = i.osize();
                self.get_rm(i, size, next);
                self.get(B, i.reg, size, rex);
                match cond(i.op) {
                    Some(c) => self.a.csel(sz(size), T, A, B, c),
                    None => {
                        let skip = self.a.label();
                        self.parity_branch(i.op, true, skip);
                        self.a.mov(Size::X, B, A);
                        self.a.bind(skip);
                        self.a.mov(Size::X, T, B);
                    }
                }
                self.set(i.reg, size, rex, T);
            }
            (Map::M0F, 0x90..=0x9f) => {
                if let Some(m) = i.mem {
                    self.addr(&m, next);
                }
                match cond(i.op) {
                    Some(c) => self.a.cset(Size::W, T, c),
                    None => {
                        self.parity(T);
                        if i.op & 1 == 0 {
                            assert!(self.a.logic_imm(Logic::Eor, Size::W, T, T, 1));
                        }
                    }
                }
                self.set_rm(i, 1, T);
            }
            (Map::Primary, 0x70..=0x7f) | (Map::M0F, 0x80..=0x8f) => {
                let target = next.wrapping_add(i.imm as u64);
                let taken = self.a.label();
                match cond(i.op) {
                    Some(c) => self.a.b_cond(c, taken),
                    None => self.parity_branch(i.op, false, taken),
                }
                self.exit_to(next, None);
                self.a.bind(taken);
                self.exit_to(target, None);
                return true;
            }
            (Map::Primary, 0xe2 | 0xe3) => {
                let target = next.wrapping_add(i.imm as u64);
                let taken = self.a.label();
                if i.op == 0xe2 {
                    self.a.sub_imm(Size::X, r(1), r(1), 1);
                    self.a.cbnz(Size::X, r(1), taken);
                } else {
                    self.a.cbz(Size::X, r(1), taken);
                }
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

    /// `dst` = 1 if the low byte of the last result has odd parity (PF clear).
    fn parity(&mut self, dst: Reg) {
        let a = &mut self.a;
        a.ubfx(Size::X, dst, PF_RESULT, 0, 8);
        for n in [4, 2, 1] {
            a.logic_shift(Logic::Eor, false, Size::X, dst, dst, dst, Shift::Lsr, n);
        }
        assert!(a.logic_imm(Logic::And, Size::X, dst, dst, 1));
    }

    /// Branches to `to` when parity condition `cc` (P or NP) holds, or when
    /// it fails if `negate`; NZCV is left alone.
    fn parity_branch(&mut self, cc: u8, negate: bool, to: Label) {
        self.parity(U);
        if (cc & 1 == 0) != negate {
            self.a.cbz(Size::X, U, to);
        } else {
            self.a.cbnz(Size::X, U, to);
        }
    }

    /// MOVS, CMPS, STOS, LODS and SCAS, with or without REP. Each step loads
    /// before it writes, so a fault leaves a state the interpreter can resume.
    fn string(&mut self, i: &Insn) {
        let size = if i.op & 1 == 0 { 1 } else { i.osize() };
        let w = width(size);
        let rep = i.rep != Rep::None;
        let step = OLD_AF;
        let (top, done) = (self.a.label(), self.a.label());
        self.a.ldr(Width::X, step, CTX, DF_OFFSET);
        self.a.mov_imm(Size::X, T, size as u64);
        self.a.eor(Size::X, T, T, step);
        self.a.sub(Size::X, step, T, step);
        self.a.bind(top);
        if rep {
            self.a.cbz(Size::X, r(1), done);
        }
        let (si, di) = (r(6), r(7));
        match i.op {
            0xa4 | 0xa5 => {
                self.a.ldr(w, A, si, 0);
                self.a.str(w, A, di, 0);
            }
            0xa6 | 0xa7 => {
                self.a.ldr(w, A, si, 0);
                self.a.ldr(w, B, di, 0);
                self.alu(Alu::Sub, size, true);
            }
            0xaa | 0xab => {
                self.get(B, 0, size, false);
                self.a.str(w, B, di, 0);
            }
            0xac | 0xad => {
                self.a.ldr(w, A, si, 0);
                self.set(0, size, false, A);
            }
            _ => {
                self.a.ldr(w, B, di, 0);
                self.get(A, 0, size, false);
                self.alu(Alu::Sub, size, true);
            }
        }
        if !matches!(i.op, 0xaa..=0xab | 0xae..=0xaf) {
            self.a.add(Size::X, si, si, step);
        }
        if !matches!(i.op, 0xac | 0xad) {
            self.a.add(Size::X, di, di, step);
        }
        if rep {
            self.a.sub_imm(Size::X, r(1), r(1), 1);
            match i.op {
                0xa6 | 0xa7 | 0xae | 0xaf => {
                    let stop = if i.rep == Rep::Rep {
                        Cond::Ne
                    } else {
                        Cond::Eq
                    };
                    self.a.b_cond(stop, done);
                }
                _ => {}
            }
            self.a.b(top);
        }
        self.a.bind(done);
    }

    /// CRC32 (`F2 0F38 F0/F1`, ARM's CRC32C) and MOVBE (`0F38 F0/F1`).
    fn movbe_crc32(&mut self, i: &Insn, next: u64) {
        let rex = i.rex != 0;
        if i.rep == Rep::Repne {
            let size = if i.op == 0xf0 { 1 } else { i.osize() };
            self.get_rm(i, size, next);
            let sf = (size == 8) as u32;
            let szf = size.trailing_zeros();
            self.a.emit(
                0x1ac0_5000
                    | sf << 31
                    | (A.0 as u32) << 16
                    | szf << 10
                    | (i.reg as u32 & 15) << 5
                    | T.0 as u32,
            );
            self.a.mov(Size::W, r(i.reg), T);
            return;
        }
        let size = i.osize();
        let s = sz(size);
        self.addr(&i.mem.expect("classified"), next);
        if i.op == 0xf0 {
            self.load(A, size);
        } else {
            self.get(A, i.reg, size, rex);
        }
        match size {
            2 => self.a.rev16(Size::W, A, A),
            _ => self.a.rev(s, A, A),
        }
        if i.op == 0xf0 {
            self.set(i.reg, size, rex, A);
        } else {
            self.store(A, size);
        }
    }

    fn push_const(&mut self, v: u64) {
        self.a.sub_imm(Size::X, ADDR, r(4), 8);
        self.a.mov_imm(Size::X, B, v);
        self.a.str(Width::X, B, ADDR, 0);
        self.a.mov(Size::X, r(4), ADDR);
    }

    /// Puts back the C flag saved in x17, since INC and DEC leave CF alone.
    fn keep_carry(&mut self) {
        let a = &mut self.a;
        a.mrs(B, a64::sysreg::NZCV);
        assert!(a.logic_imm(Logic::And, Size::W, OLD_NZCV, OLD_NZCV, C_BIT));
        assert!(a.logic_imm(Logic::And, Size::W, B, B, !C_BIT & 0xffff_ffff));
        a.orr(Size::X, B, B, OLD_NZCV);
        a.msr(a64::sysreg::NZCV, B);
    }

    /// `T` = the r/m operand `op` `B`, written back when `keep`; with LOCK,
    /// atomically, and `T` only when the flags are live.
    fn alu_rm(&mut self, i: &Insn, op: Alu, keep: bool, size: u8, next: u64, flags: bool) {
        if !i.lock {
            self.get_rm(i, size, next);
            self.alu(op, size, flags);
            if keep {
                self.set_rm(i, size, T);
            }
            return;
        }
        self.addr(&i.mem.expect("lockable"), next);
        let (a, w) = (&mut self.a, width(size));
        match op {
            Alu::Add => a.ldaddal(w, B, A, ADDR),
            Alu::Sub => {
                a.neg(Size::X, U, B);
                a.ldaddal(w, U, A, ADDR);
            }
            Alu::Or => a.ldsetal(w, B, A, ADDR),
            Alu::And => {
                a.mvn(Size::X, U, B);
                a.ldclral(w, U, A, ADDR);
            }
            Alu::Xor => a.ldeoral(w, B, A, ADDR),
            Alu::Adc | Alu::Sbb => unreachable!("not lockable"),
        }
        if flags {
            self.alu(op, size, true);
        }
    }

    /// XADD and CMPXCHG; with a memory operand they are always atomic.
    fn atomic(&mut self, i: &Insn, next: u64, flags: bool) {
        let rex = i.rex != 0;
        let size = if i.op & 1 == 0 { 1 } else { i.osize() };
        let w = width(size);
        if i.op >= 0xc0 {
            self.get(B, i.reg, size, rex);
            match i.mem {
                Some(m) => {
                    self.addr(&m, next);
                    self.a.ldaddal(w, B, A, ADDR);
                    if flags {
                        self.alu(Alu::Add, size, true);
                    }
                    self.set(i.reg, size, rex, A);
                }
                None => {
                    self.get(A, i.rm, size, rex);
                    self.alu(Alu::Add, size, flags);
                    self.set(i.reg, size, rex, A);
                    self.set(i.rm, size, rex, T);
                }
            }
            return;
        }
        self.addr(&i.mem.expect("classified"), next);
        self.get(A, 0, size, rex);
        self.get(B, i.reg, size, rex);
        self.a.mov(Size::X, U, A);
        self.a.casal(w, U, B, ADDR);
        self.a.mov(Size::X, B, U);
        self.alu(Alu::Sub, size, true);
        if size >= 4 {
            self.a.csel(Size::X, r(0), B, r(0), Cond::Ne);
        } else {
            self.a.mov(Size::X, OLD_NZCV, r(0));
            self.a.bfi(Size::X, OLD_NZCV, B, 0, size as u32 * 8);
            self.a.csel(Size::X, r(0), OLD_NZCV, r(0), Cond::Ne);
        }
    }

    /// XCHG; with memory it is atomic, as on x86.
    fn xchg(&mut self, i: &Insn, next: u64) {
        let rex = i.rex != 0;
        let size = if i.op == 0x86 { 1 } else { i.osize() };
        let (x, y) = if i.op >= 0x91 {
            (0, i.rm)
        } else {
            (i.reg, i.rm)
        };
        self.get(B, x, size, rex);
        match i.mem {
            Some(m) if i.op < 0x91 => {
                self.addr(&m, next);
                self.a.swpal(width(size), B, A, ADDR);
                self.set(x, size, rex, A);
            }
            _ => {
                self.get(A, y, size, rex);
                self.set(x, size, rex, A);
                self.set(y, size, rex, B);
            }
        }
    }

    /// One-operand MUL and IMUL: RDX:RAX = RAX * r/m.
    fn mul_wide(&mut self, i: &Insn, next: u64, flags: bool) {
        let size = i.osize();
        let signed = i.ext() == 5;
        self.get_rm(i, size, next);
        let (hi, a) = (OLD_NZCV, &mut self.a);
        if size == 8 {
            a.mul(Size::X, T, r(0), A);
            if signed {
                a.smulh(hi, r(0), A);
            } else {
                a.umulh(hi, r(0), A);
            }
            if flags && signed {
                a.asr_imm(Size::X, B, T, 63);
                a.cmp(Size::X, hi, B);
            }
        } else {
            if signed {
                a.smull(T, r(0), A);
                if flags {
                    a.sxtw(B, T);
                    a.cmp(Size::X, T, B);
                }
            } else {
                a.umull(T, r(0), A);
            }
            a.lsr_imm(Size::X, hi, T, 32);
        }
        if flags && !signed {
            self.a.cmp_imm(Size::X, hi, 0);
        }
        if flags {
            self.imul_flags(size);
        }
        let s = sz(size);
        self.a.mov(s, r(0), T);
        self.a.mov(s, r(2), hi);
    }

    /// DIV and IDIV of RDX:RAX by r/m. Division by zero, a quotient that does
    /// not fit and 64-bit dividends wider than RAX go to the interpreter.
    fn div(&mut self, i: &Insn, next: u64) {
        let size = i.osize();
        let signed = i.ext() == 7;
        let bail = self.bail();
        self.get_rm(i, size, next);
        let (x, a) = (OLD_NZCV, &mut self.a);
        if size == 8 {
            a.cbz(Size::X, A, bail);
            if signed {
                a.asr_imm(Size::X, x, r(0), 63);
                a.eor(Size::X, x, x, r(2));
                a.cbnz(Size::X, x, bail);
                a.add_imm(Size::X, x, A, 1);
                a.cbz(Size::X, x, bail);
                a.sdiv(Size::X, T, r(0), A);
            } else {
                let (fast, done) = (a.label(), a.label());
                a.cbz(Size::X, r(2), fast);
                self.wide_udiv(bail);
                self.a.b(done);
                self.a.bind(fast);
                self.a.udiv(Size::X, T, r(0), A);
                self.a.msub(Size::X, U, T, A, r(0));
                self.a.mov(Size::X, r(0), T);
                self.a.mov(Size::X, r(2), U);
                self.a.bind(done);
                return;
            }
            a.msub(Size::X, U, T, A, r(0));
            a.mov(Size::X, r(0), T);
            a.mov(Size::X, r(2), U);
            return;
        }
        a.cbz(Size::W, A, bail);
        a.mov(Size::W, U, r(0));
        a.bfi(Size::X, U, r(2), 32, 32);
        if signed {
            a.sxtw(A, A);
            a.sdiv(Size::X, T, U, A);
            a.sxtw(x, T);
        } else {
            a.udiv(Size::X, T, U, A);
            a.mov(Size::W, x, T);
        }
        a.eor(Size::X, x, x, T);
        a.cbnz(Size::X, x, bail);
        a.msub(Size::X, B, T, A, U);
        a.mov(Size::W, r(0), T);
        a.mov(Size::W, r(2), B);
    }

    /// SHL, SHR and SAR by CL; a count of 0 changes nothing.
    fn shift_cl(&mut self, i: &Insn, next: u64, flags: bool) {
        let size = if i.op == 0xd2 { 1 } else { i.osize() };
        let bits = size as u32 * 8;
        let ext = i.ext();
        let done = self.a.label();
        self.get_rm(i, size, next);
        let (carry, a) = (OLD_AF, &mut self.a);
        let count_mask = if size == 8 { 63 } else { 31 };
        assert!(a.logic_imm(Logic::And, Size::X, B, r(1), count_mask));
        a.cbz(Size::X, B, done);
        let m = mask(size);
        match ext {
            4 | 6 => {
                a.lslv(Size::X, T, A, B);
                if m != u64::MAX {
                    assert!(a.logic_imm(Logic::And, Size::X, T, T, m));
                }
                if size == 8 {
                    a.mov_imm(Size::X, U, 64);
                    a.sub(Size::X, U, U, B);
                    a.lsrv(Size::X, carry, A, U);
                } else {
                    a.lslv(Size::X, carry, A, B);
                    a.lsr_imm(Size::X, carry, carry, bits);
                }
            }
            5 => {
                a.lsrv(Size::X, T, A, B);
                a.sub_imm(Size::X, U, B, 1);
                a.lsrv(Size::X, carry, A, U);
            }
            _ => {
                match size {
                    1 => a.sxtb(Size::X, U, A),
                    2 => a.sxth(Size::X, U, A),
                    4 => a.sxtw(U, A),
                    _ => a.mov(Size::X, U, A),
                }
                a.asrv(Size::X, T, U, B);
                if m != u64::MAX {
                    assert!(a.logic_imm(Logic::And, Size::X, T, T, m));
                }
                a.sub_imm(Size::X, B, B, 1);
                a.asrv(Size::X, carry, U, B);
            }
        }
        self.set_rm(i, size, T);
        if flags {
            self.shift_cl_flags(ext, size);
        }
        self.a.bind(done);
    }

    /// The flags of [`Self::shift_cl`] from the old value in A, the result in
    /// T and the last bit shifted out in bit 0 of x18.
    fn shift_cl_flags(&mut self, ext: u8, size: u8) {
        let bits = size as u32 * 8;
        let (old, carry, a) = (OLD_NZCV, OLD_AF, &mut self.a);
        assert!(a.logic_imm(Logic::And, Size::X, carry, carry, 1));
        match ext {
            4 | 6 => {
                a.lsr_imm(Size::X, U, T, bits - 1);
                a.eor(Size::X, U, U, carry);
            }
            5 => a.lsr_imm(Size::X, U, A, bits - 1),
            _ => a.mov_imm(Size::X, U, 0),
        }
        assert!(a.logic_imm(Logic::And, Size::X, U, U, 1));
        a.eor(Size::X, old, PF_RESULT, AF_AUX);
        assert!(a.logic_imm(Logic::And, Size::X, old, old, 0x10));
        if bits < 32 {
            a.lsl_imm(Size::W, B, T, 32 - bits);
            a.tst(Size::W, B, B);
        } else {
            a.tst(sz(size), T, T);
        }
        a.mrs(B, a64::sysreg::NZCV);
        assert!(a.logic_imm(Logic::Eor, Size::X, carry, carry, 1));
        a.bfi(Size::X, B, carry, 29, 1);
        a.bfi(Size::X, B, U, 28, 1);
        a.msr(a64::sysreg::NZCV, B);
        a.mov(Size::X, PF_RESULT, T);
        a.eor(Size::X, AF_AUX, T, old);
    }

    /// RDX:RAX / A for RDX not zero, bit by bit; a quotient that does not fit
    /// in 64 bits goes to `bail`. NZCV is kept.
    fn wide_udiv(&mut self, bail: Label) {
        let (high, low, count, carry, saved) = (U, T, B, OLD_AF, OLD_NZCV);
        let a = &mut self.a;
        let (top, skip) = (a.label(), a.label());
        a.mrs(saved, a64::sysreg::NZCV);
        a.cmp(Size::X, r(2), A);
        a.cset(Size::X, ADDR, Cond::Hs);
        a.msr(a64::sysreg::NZCV, saved);
        a.cbnz(Size::X, ADDR, bail);
        a.mov(Size::X, high, r(2));
        a.mov(Size::X, low, r(0));
        a.mov_imm(Size::X, count, 64);
        a.bind(top);
        a.lsr_imm(Size::X, carry, high, 63);
        a.extr(Size::X, high, high, low, 63);
        a.lsl_imm(Size::X, low, low, 1);
        a.cmp(Size::X, high, A);
        a.cset(Size::X, ADDR, Cond::Hs);
        a.orr(Size::X, ADDR, ADDR, carry);
        a.cbz(Size::X, ADDR, skip);
        a.sub(Size::X, high, high, A);
        assert!(a.logic_imm(Logic::Orr, Size::X, low, low, 1));
        a.bind(skip);
        a.sub_imm(Size::X, count, count, 1);
        a.cbnz(Size::X, count, top);
        a.msr(a64::sysreg::NZCV, saved);
        a.mov(Size::X, r(0), low);
        a.mov(Size::X, r(2), high);
    }

    /// ROL and ROR by 1, a constant or CL: only CF and OF change, and
    /// nothing does for a count of 0.
    fn rotate(&mut self, i: &Insn, next: u64, flags: bool) {
        let size = i.osize();
        let s = sz(size);
        let bits = size as u32 * 8;
        let rol = i.ext() == 0;
        self.get_rm(i, size, next);
        let skip = self.a.label();
        let mut counted = true;
        if i.op == 0xd3 {
            assert!(
                self.a
                    .logic_imm(Logic::And, Size::X, B, r(1), bits as u64 - 1)
            );
            if rol {
                self.a.neg(s, U, B);
                self.a.rorv(s, T, A, U);
            } else {
                self.a.rorv(s, T, A, B);
            }
        } else {
            let n = if i.op == 0xd1 {
                1
            } else {
                i.imm as u32 & (bits - 1)
            };
            counted = n != 0;
            let k = if rol { (bits - n) % bits } else { n };
            self.a.ror_imm(s, T, A, k);
        }
        self.set_rm(i, size, T);
        if !flags || !counted {
            return;
        }
        let (old, cf, a) = (OLD_NZCV, OLD_AF, &mut self.a);
        if i.op == 0xd3 {
            a.cbz(Size::X, B, skip);
        }
        a.mrs(old, a64::sysreg::NZCV);
        if rol {
            a.ubfx(Size::X, cf, T, 0, 1);
            a.lsr_imm(s, U, T, bits - 1);
        } else {
            a.lsr_imm(s, cf, T, bits - 1);
            a.ubfx(Size::X, U, T, bits - 2, 1);
        }
        a.eor(Size::X, U, U, cf);
        assert!(a.logic_imm(Logic::Eor, Size::X, cf, cf, 1));
        a.bfi(Size::X, old, cf, 29, 1);
        a.bfi(Size::X, old, U, 28, 1);
        a.msr(a64::sysreg::NZCV, old);
        a.bind(skip);
    }

    /// BT, BTS, BTR and BTC: CF = the old bit, the other flags kept. A
    /// register offset into memory may reach outside the operand.
    fn bit_test(&mut self, i: &Insn, next: u64, flags: bool) {
        let size = i.osize();
        let s = sz(size);
        let bits = size as u32 * 8;
        let rex = i.rex != 0;
        let op = if i.op == 0xba {
            i.ext() - 4
        } else {
            (i.op >> 3) & 3
        };
        if i.op == 0xba {
            self.a.mov_imm(Size::X, B, i.imm as u64 & (bits as u64 - 1));
        } else {
            self.get(B, i.reg, size, rex);
        }
        match i.mem {
            Some(m) => {
                self.addr(&m, next);
                if i.op != 0xba {
                    if size == 4 {
                        self.a.sxtw(T, B);
                    } else {
                        self.a.mov(Size::X, T, B);
                    }
                    self.a.asr_imm(Size::X, T, T, bits.trailing_zeros());
                    self.a
                        .add_shift(Size::X, ADDR, ADDR, T, Shift::Lsl, size.trailing_zeros());
                }
            }
            None => self.get(A, i.rm, size, rex),
        }
        if op != 0 {
            self.a.mov_imm(Size::X, U, 1);
            self.a.lslv(s, U, U, B);
        }
        if i.lock {
            let w = width(size);
            match op {
                1 => self.a.ldsetal(w, U, A, ADDR),
                2 => self.a.ldclral(w, U, A, ADDR),
                _ => self.a.ldeoral(w, U, A, ADDR),
            }
        } else {
            if i.mem.is_some() {
                self.load(A, size);
            }
            if op != 0 {
                match op {
                    1 => self.a.orr(s, U, A, U),
                    2 => self.a.bic(s, U, A, U),
                    _ => self.a.eor(s, U, A, U),
                }
                self.set_rm(i, size, U);
            }
        }
        if flags {
            let a = &mut self.a;
            a.lsrv(s, T, A, B);
            a.mrs(U, a64::sysreg::NZCV);
            assert!(a.logic_imm(Logic::Eor, Size::X, T, T, 1));
            a.bfi(Size::X, U, T, 29, 1);
            a.msr(a64::sysreg::NZCV, U);
        }
    }

    /// POPCNT: ZF says the source was zero; the other arithmetic flags clear.
    fn popcnt(&mut self, i: &Insn, next: u64, flags: bool) {
        let size = i.osize();
        self.get_rm(i, size, next);
        let a = &mut self.a;
        a.fmov_dx(V(0), A);
        a.emit(CNT_8B);
        a.emit(ADDV_B);
        a.fmov_xd(T, V(0));
        if flags {
            a.cmp_imm(Size::X, A, 0);
            a.cset(Size::X, U, Cond::Eq);
            a.lsl_imm(Size::X, U, U, 30);
            assert!(a.logic_imm(Logic::Orr, Size::X, U, U, C_BIT));
            a.msr(a64::sysreg::NZCV, U);
            a.mov_imm(Size::X, PF_RESULT, 1);
            a.mov_imm(Size::X, AF_AUX, 1);
        }
        self.set(i.reg, size, i.rex != 0, T);
    }

    /// 8- and 16-bit shifts and rotates by 1 or an immediate, with the
    /// interpreter's flags (see `x86::alu::shift`).
    fn narrow_shift(&mut self, i: &Insn, next: u64, flags: bool) {
        let size = if i.op & 1 == 0 { 1 } else { 2 };
        let bits = size as u32 * 8;
        let n = if i.op >= 0xd0 { 1 } else { i.imm as u32 & 31 };
        if n == 0 {
            return;
        }
        let ext = i.ext();
        self.get_rm(i, size, next);
        let a = &mut self.a;
        let m = mask(size);
        let msb = bits - 1;
        let (cf, of): (Option<u32>, Option<u32>);
        match ext {
            0 | 1 => {
                let c = n % bits;
                a.mov(Size::X, T, A);
                for k in 1..(32 / bits) {
                    a.logic_shift(Logic::Orr, false, Size::W, T, T, A, Shift::Lsl, k * bits);
                }
                let r = if ext == 0 { (bits - c) % bits } else { c };
                a.ror_imm(Size::W, T, T, r);
                assert!(a.logic_imm(Logic::And, Size::X, T, T, m));
                (cf, of) = (None, None);
            }
            4 | 6 => {
                if n >= bits {
                    a.mov_imm(Size::X, T, 0);
                } else {
                    a.lsl_imm(Size::X, T, A, n);
                    assert!(a.logic_imm(Logic::And, Size::X, T, T, m));
                }
                (cf, of) = ((n <= bits).then(|| bits - n), None);
            }
            5 => {
                if n >= bits {
                    a.mov_imm(Size::X, T, 0);
                } else {
                    a.lsr_imm(Size::X, T, A, n);
                }
                (cf, of) = ((n <= bits).then(|| n - 1), Some(msb));
            }
            _ => {
                if size == 1 {
                    a.sxtb(Size::X, U, A);
                } else {
                    a.sxth(Size::X, U, A);
                }
                a.asr_imm(Size::X, T, U, n.min(63));
                assert!(a.logic_imm(Logic::And, Size::X, T, T, m));
                (cf, of) = (Some((n - 1).min(63)), None);
            }
        }
        self.set_rm(i, size, T);
        if !flags {
            return;
        }
        let (old, carry, a) = (OLD_NZCV, OLD_AF, &mut self.a);
        if ext <= 1 {
            a.mrs(old, a64::sysreg::NZCV);
            if ext == 0 {
                a.ubfx(Size::X, carry, T, 0, 1);
                a.lsr_imm(Size::X, U, T, msb);
            } else {
                a.lsr_imm(Size::X, carry, T, msb);
                a.ubfx(Size::X, U, T, msb - 1, 1);
            }
            a.eor(Size::X, U, U, carry);
            assert!(a.logic_imm(Logic::Eor, Size::X, carry, carry, 1));
            a.bfi(Size::X, old, carry, 29, 1);
            a.bfi(Size::X, old, U, 28, 1);
            a.msr(a64::sysreg::NZCV, old);
            return;
        }
        a.eor(Size::X, old, PF_RESULT, AF_AUX);
        assert!(a.logic_imm(Logic::And, Size::X, old, old, 0x10));
        match (cf, ext) {
            (Some(bit), 7) => a.ubfx(Size::X, carry, U, bit, 1),
            (Some(bit), _) => a.ubfx(Size::X, carry, A, bit, 1),
            (None, _) => a.mov_imm(Size::X, carry, 0),
        }
        match (of, ext) {
            (Some(bit), _) => a.ubfx(Size::X, U, A, bit, 1),
            (None, 7) => a.mov_imm(Size::X, U, 0),
            (None, _) => {
                a.lsr_imm(Size::X, U, T, msb);
                a.eor(Size::X, U, U, carry);
            }
        }
        a.lsl_imm(Size::W, B, T, 32 - bits);
        a.tst(Size::W, B, B);
        a.mrs(B, a64::sysreg::NZCV);
        assert!(a.logic_imm(Logic::Eor, Size::X, carry, carry, 1));
        a.bfi(Size::X, B, carry, 29, 1);
        a.bfi(Size::X, B, U, 28, 1);
        a.msr(a64::sysreg::NZCV, B);
        a.mov(Size::X, PF_RESULT, T);
        a.eor(Size::X, AF_AUX, T, old);
    }

    /// BSF and BSR (a zero source sets ZF and keeps the destination), TZCNT
    /// and LZCNT (CF for a zero source, ZF for a zero result).
    fn bit_scan(&mut self, i: &Insn, next: u64, flags: bool) {
        let size = i.osize();
        let s = sz(size);
        self.get_rm(i, size, next);
        if i.rep == Rep::Rep {
            if i.op == 0xbd {
                self.a.clz(s, T, A);
            } else {
                self.a.rbit(s, T, A);
                self.a.clz(s, T, T);
            }
            self.set(i.reg, size, i.rex != 0, T);
            if flags {
                let (old, z, a) = (OLD_NZCV, OLD_AF, &mut self.a);
                a.mrs(old, a64::sysreg::NZCV);
                a.cmp_imm(s, A, 0);
                a.cset(Size::X, U, Cond::Ne);
                a.cmp_imm(s, T, 0);
                a.cset(Size::X, z, Cond::Eq);
                a.bfi(Size::X, old, U, 29, 1);
                a.bfi(Size::X, old, z, 30, 1);
                a.msr(a64::sysreg::NZCV, old);
            }
            return;
        }
        let (zero, done) = (self.a.label(), self.a.label());
        if i.op == 0xbd {
            self.a.clz(s, T, A);
            self.a.mov_imm(Size::X, U, size as u64 * 8 - 1);
            self.a.sub(Size::X, T, U, T);
        } else {
            self.a.rbit(s, T, A);
            self.a.clz(s, T, T);
        }
        if flags {
            self.a.mrs(OLD_NZCV, a64::sysreg::NZCV);
        }
        self.a.cbz(Size::X, A, zero);
        self.set(i.reg, size, i.rex != 0, T);
        let z = 1u64 << 30;
        if flags {
            assert!(
                self.a
                    .logic_imm(Logic::And, Size::X, OLD_NZCV, OLD_NZCV, !z)
            );
        }
        self.a.b(done);
        self.a.bind(zero);
        if flags {
            assert!(self.a.logic_imm(Logic::Orr, Size::X, OLD_NZCV, OLD_NZCV, z));
        }
        self.a.bind(done);
        if flags {
            self.a.msr(a64::sysreg::NZCV, OLD_NZCV);
        }
    }

    /// `T` = SHL/SHR/SAR of `A` by `n`, with the interpreter's flags: CF the
    /// last bit out, OF as for a count of 1 whatever the count, AF kept.
    fn shift_flags(&mut self, ext: u8, size: u8, n: u32) {
        let s = sz(size);
        let bits = size as u32 * 8;
        let old_af = OLD_AF;
        let a = &mut self.a;
        a.eor(Size::X, old_af, PF_RESULT, AF_AUX);
        assert!(a.logic_imm(Logic::And, Size::X, old_af, old_af, 0x10));
        let carry_bit = if ext == 4 { bits - n } else { n - 1 };
        match ext {
            4 => a.lsl_imm(s, T, A, n),
            5 => a.lsr_imm(s, T, A, n),
            _ => a.asr_imm(s, T, A, n),
        }
        a.ubfx(Size::X, U, A, carry_bit, 1);
        match ext {
            4 => {
                a.lsr_imm(s, B, T, bits - 1);
                a.eor(Size::X, B, B, U);
            }
            5 => a.lsr_imm(s, B, A, bits - 1),
            _ => a.mov_imm(Size::X, B, 0),
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
