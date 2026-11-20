//! Tier 2: a hot block's region (the blocks its direct branches reach on its
//! 4 KiB page) translated as one piece, with each flag computed only where
//! the region may read it. See `docs/jit.md`, "Tier 2".

use std::collections::HashMap;

use a64::{Label, Reg, Size, Width, r};
use memory::Memory;
use x86::decode::{Insn, Map, Mem, Rep, Seg, decode};

use super::{
    A, ADDR, AF_AUX, Alu, B, Block, HVC_INTERP, ICOUNT, PF_RESULT, T, Translator, alu_of,
    classify, cond, mask, sz, width,
};

const MAX_BLOCKS: usize = 32;
const MAX_INSNS: usize = 512;
const MAX_BLOCK_INSNS: usize = 64;

const CF: u8 = 1;
const PF: u8 = 2;
const AF: u8 = 4;
const ZF: u8 = 8;
const SF: u8 = 16;
const OF: u8 = 32;
const ALL: u8 = 63;
const NZCV: u8 = CF | ZF | SF | OF;

enum End {
    /// Falls through or jumps to the region block with this index.
    Goto(usize),
    /// A Jcc whose targets are region blocks or leave it (`None`).
    Branch { cc: u8, taken: Option<usize>, taken_rip: u64, fall: Option<usize>, fall_rip: u64 },
    /// Leaves the region for the code at this RIP.
    Leave(u64),
    /// Leaves for the interpreter at this RIP.
    Interpret(u64),
    /// The last instruction ends the block itself (a call, return or
    /// indirect jump), as tier 1 translates it.
    Own,
}

struct XBlock {
    start: u64,
    insns: Vec<(Insn, u64)>,
    end: End,
}

fn fetch(mem: &Memory, at: u64) -> Option<Insn> {
    decode(|i| {
        let mut b = [0u8];
        mem.fetch(at.wrapping_add(i as u64), &mut b).map(|_| b[0])
    })
    .ok()
}

/// The blocks of the region at `entry`, entry first.
fn discover(mem: &Memory, entry: u64) -> Vec<XBlock> {
    let page = entry & !0xfff;
    let mut blocks: Vec<XBlock> = Vec::new();
    let mut index: HashMap<u64, usize> = HashMap::new();
    let mut work = vec![entry];
    let mut total = 0;
    index.insert(entry, 0);
    blocks.push(XBlock { start: entry, insns: Vec::new(), end: End::Leave(entry) });
    while let Some(start) = work.pop() {
        let b = index[&start];
        let mut insns = Vec::new();
        let mut at = start;
        let mut target = |t: u64, blocks: &mut Vec<XBlock>, work: &mut Vec<u64>| -> Option<usize> {
            if t & !0xfff != page {
                return None;
            }
            if let Some(&k) = index.get(&t) {
                return Some(k);
            }
            if blocks.len() >= MAX_BLOCKS {
                return None;
            }
            blocks.push(XBlock { start: t, insns: Vec::new(), end: End::Leave(t) });
            index.insert(t, blocks.len() - 1);
            work.push(t);
            Some(blocks.len() - 1)
        };
        let end = loop {
            if insns.len() >= MAX_BLOCK_INSNS || total >= MAX_INSNS || at & !0xfff != page {
                break End::Leave(at);
            }
            let Some(i) = fetch(mem, at) else { break End::Interpret(at) };
            let Some((ends, _)) = classify(&i) else { break End::Interpret(at) };
            let next = at + i.len as u64;
            total += 1;
            let jcc = matches!((i.map, i.op), (Map::Primary, 0x70..=0x7f) | (Map::M0F, 0x80..=0x8f));
            let jmp = i.map == Map::Primary && matches!(i.op, 0xe9 | 0xeb);
            if jcc {
                let t = next.wrapping_add(i.imm as u64);
                let taken = target(t, &mut blocks, &mut work);
                let fall = target(next, &mut blocks, &mut work);
                let cc = i.op & 0xf;
                insns.push((i, at));
                break End::Branch { cc, taken, taken_rip: t, fall, fall_rip: next };
            }
            if jmp {
                let t = next.wrapping_add(i.imm as u64);
                insns.push((i, at));
                break match target(t, &mut blocks, &mut work) {
                    Some(k) => End::Goto(k),
                    None => End::Leave(t),
                };
            }
            insns.push((i, at));
            if ends {
                break End::Own;
            }
            at = next;
        };
        blocks[b].insns = insns;
        blocks[b].end = end;
    }
    blocks
}

/// The flags `i` reads and the flags it surely writes. Instructions that may
/// hand themselves to the interpreter count as reading all flags.
fn flag_rw(i: &Insn) -> (u8, u8) {
    let reads_cc = |cc: u8| match cc >> 1 {
        0 => OF,
        1 => CF,
        2 => ZF,
        3 => CF | ZF,
        4 => SF,
        5 => PF,
        6 => SF | OF,
        _ => ZF | SF | OF,
    };
    match (i.map, i.op) {
        (Map::Primary, 0x00..=0x3f) if i.op & 7 <= 5 => match i.op >> 3 {
            2 | 3 => (CF, ALL),
            _ => (0, ALL),
        },
        (Map::Primary, 0x80 | 0x81 | 0x83) => match i.ext() {
            2 | 3 => (CF, ALL),
            _ => (0, ALL),
        },
        (Map::Primary, 0x84 | 0x85 | 0xa8 | 0xa9) => (0, ALL),
        (Map::Primary, 0xf6 | 0xf7) if matches!(i.ext(), 0 | 3) => (0, ALL),
        (Map::Primary, 0xf6 | 0xf7) if matches!(i.ext(), 4 | 5) => (0, ALL),
        (Map::Primary, 0xfe | 0xff) if i.ext() <= 1 => (0, ALL & !CF),
        (Map::Primary, 0x70..=0x7f) | (Map::M0F, 0x80..=0x8f | 0x90..=0x9f | 0x40..=0x4f) => {
            (reads_cc(i.op & 0xf), 0)
        }
        (Map::M0F, 0xaf) | (Map::Primary, 0x69 | 0x6b) => (0, ALL),
        (Map::Primary, 0xf7) if matches!(i.ext(), 6 | 7) => (ALL, 0),
        (Map::Primary, 0xdb) if i.mem.is_some() && matches!(i.ext(), 5 | 7) => (ALL, 0),
        _ => match classify(i) {
            Some((_, f)) if f.reads => (ALL, 0),
            _ => (0, 0),
        },
    }
}

/// For every instruction of every block, the flags code after it may read.
fn demand(blocks: &[XBlock]) -> Vec<Vec<u8>> {
    let succs = |b: &XBlock| -> (Vec<usize>, bool) {
        match b.end {
            End::Goto(k) => (vec![k], false),
            End::Branch { taken, fall, .. } => {
                let v: Vec<usize> = [taken, fall].into_iter().flatten().collect();
                let leaves = taken.is_none() || fall.is_none();
                (v, leaves)
            }
            _ => (vec![], true),
        }
    };
    let mut live_in = vec![0u8; blocks.len()];
    let mut after: Vec<Vec<u8>> = blocks.iter().map(|b| vec![0; b.insns.len()]).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (k, b) in blocks.iter().enumerate().rev() {
            let (next, leaves) = succs(b);
            let mut live = if leaves { ALL } else { 0 };
            for s in next {
                live |= live_in[s];
            }
            for (n, (i, _)) in b.insns.iter().enumerate().rev() {
                after[k][n] = live;
                let (reads, writes) = flag_rw(i);
                live = reads | (live & !writes);
            }
            if live != live_in[k] {
                live_in[k] = live;
                changed = true;
            }
        }
    }
    after
}

/// Translates the region at `rip` for runtime offset `place`; `None` if it
/// holds no more than what tier 1 already made of its first block.
pub fn region(mem: &Memory, rip: u64, place: usize, dispatch: usize) -> Option<(Block, Vec<u64>)> {
    let blocks = discover(mem, rip);
    if blocks.len() < 2 && blocks[0].insns.len() < 2 {
        return None;
    }
    let after = demand(&blocks);
    let mut t = Translator {
        a: a64::Asm::new(),
        place,
        dispatch,
        insns: Vec::new(),
        chains: Vec::new(),
        bails: Vec::new(),
        cur: (rip, 0),
        x87_mode: false,
    };
    let labels: Vec<Label> = blocks.iter().map(|_| t.a.label()).collect();
    for (k, b) in blocks.iter().enumerate() {
        t.a.bind(labels[k]);
        if b.insns.is_empty() {
            t.exit_to(b.start, Some(HVC_INTERP));
            continue;
        }
        t.a.add_imm(Size::X, ICOUNT, ICOUNT, b.insns.len() as u32);
        let last = b.insns.len() - 1;
        for (n, (i, at)) in b.insns.iter().enumerate() {
            t.insns.push((t.a.pos() as u32, *at));
            t.cur = (*at, (b.insns.len() - n) as u32);
            let ending = n == last && !matches!(b.end, End::Leave(_) | End::Interpret(_));
            if ending {
                if !matches!(b.end, End::Own) {
                    t.leave_x87();
                    t.region_end(b, &labels, k);
                    break;
                }
                t.leave_x87();
                t.insn(i, *at, after[k][n] != 0);
                break;
            }
            if super::x87::classify(i).is_none() && super::vector::classify(i).is_some() {
                t.leave_x87();
            }
            if !t.fast(i, *at, after[k][n]) {
                t.insn(i, *at, after[k][n] != 0);
            }
        }
        match b.end {
            End::Leave(at) => {
                t.leave_x87();
                t.exit_to(at, None);
            }
            End::Interpret(at) => {
                t.leave_x87();
                t.exit_to(at, Some(HVC_INTERP));
            }
            _ => {}
        }
    }
    for (l, at, left, x87) in std::mem::take(&mut t.bails) {
        t.a.bind(l);
        if x87 {
            t.restore_fp();
        }
        t.a.sub_imm(Size::X, ICOUNT, ICOUNT, left);
        t.exit_to(at, Some(HVC_INTERP));
    }
    let rips = blocks.iter().flat_map(|b| b.insns.iter().map(|&(_, at)| at)).collect();
    Some((
        Block {
            code: t.a.finish(),
            insns: t.insns,
            chains: t.chains,
        },
        rips,
    ))
}

/// Where an operand is.
enum Opd {
    Reg(Reg),
    Imm(u64),
}

impl Translator {
    /// The branch or jump ending region block `b`.
    fn region_end(&mut self, b: &XBlock, labels: &[Label], k: usize) {
        match b.end {
            End::Goto(t) => {
                if t != k + 1 {
                    self.a.b(labels[t]);
                }
            }
            End::Branch { cc, taken, taken_rip, fall, fall_rip } => {
                let to_taken = match taken {
                    Some(t) => labels[t],
                    None => self.a.label(),
                };
                match cond(cc) {
                    Some(c) => self.a.b_cond(c, to_taken),
                    None => self.parity_branch(cc, false, to_taken),
                }
                match fall {
                    Some(f) if f == k + 1 => {}
                    Some(f) => self.a.b(labels[f]),
                    None => self.exit_to(fall_rip, None),
                }
                if taken.is_none() {
                    if fall == Some(k + 1) {
                        let over = self.a.label();
                        self.a.b(over);
                        self.a.bind(to_taken);
                        self.exit_to(taken_rip, None);
                        self.a.bind(over);
                    } else {
                        self.a.bind(to_taken);
                        self.exit_to(taken_rip, None);
                    }
                }
            }
            _ => {}
        }
    }

    /// A register-and-offset form of memory operand `m`, folding a base
    /// register with a small displacement; anything else is computed into x19.
    fn mem_operand(&mut self, m: &Mem, next: u64, w: Width) -> (Reg, i32) {
        let scale = match w {
            Width::B => 1,
            Width::H => 2,
            Width::W => 4,
            Width::X => 8,
        };
        let d = m.disp as i64;
        let fits = (0..4096 * scale).contains(&d) && d % scale == 0 || (-256..256).contains(&d);
        match m.base {
            Some(base)
                if !m.rip_relative && m.index.is_none() && m.seg.is_none() && !m.addr32 && fits =>
            {
                (r(base), d as i32)
            }
            _ => {
                self.addr(m, next);
                (ADDR, 0)
            }
        }
    }

    /// MOV, LEA and two-operand ALU instructions on 32- and 64-bit operands,
    /// computing only the flags in `need`; false leaves `i` to tier 1.
    fn fast(&mut self, i: &Insn, rip: u64, need: u8) -> bool {
        if i.lock || i.rep != Rep::None || i.vex.is_some() || i.seg == Some(Seg::Gs) {
            return false;
        }
        let next = rip + i.len as u64;
        let size = i.osize();
        if size < 4 {
            return false;
        }
        let s = sz(size);
        let w = width(size);
        match (i.map, i.op) {
            (Map::Primary, 0x89) => {
                match i.mem {
                    Some(m) => {
                        let (base, off) = self.mem_operand(&m, next, w);
                        self.a.str(w, r(i.reg), base, off);
                    }
                    None => self.a.mov(s, r(i.rm), r(i.reg)),
                }
                true
            }
            (Map::Primary, 0x8b) => {
                match i.mem {
                    Some(m) => {
                        let (base, off) = self.mem_operand(&m, next, w);
                        self.a.ldr(w, r(i.reg), base, off);
                    }
                    None => self.a.mov(s, r(i.reg), r(i.rm)),
                }
                true
            }
            (Map::Primary, 0xc7) if i.ext() == 0 => {
                let v = i.imm as u64 & mask(size);
                match i.mem {
                    Some(m) => {
                        let (base, off) = self.mem_operand(&m, next, w);
                        let src = if v == 0 {
                            Reg(31)
                        } else {
                            self.a.mov_imm(Size::X, B, v);
                            B
                        };
                        self.a.str(w, src, base, off);
                    }
                    None => self.a.mov_imm(s, r(i.rm), v),
                }
                true
            }
            (Map::Primary, 0xb8..=0xbf) => {
                self.a.mov_imm(s, r(i.rm), i.imm as u64 & mask(size));
                true
            }
            (Map::Primary, 0x01 | 0x03 | 0x09 | 0x0b | 0x21 | 0x23 | 0x29 | 0x2b | 0x31 | 0x33 | 0x39 | 0x3b | 0x85) => {
                let (op, keep) = match i.op {
                    0x85 => (Alu::And, false),
                    _ => alu_of(i.op >> 3).expect("ALU opcode"),
                };
                let to_rm = i.op & 2 == 0;
                let (dst_reg, src) = if to_rm { (i.rm, Opd::Reg(r(i.reg))) } else { (i.reg, Opd::Reg(r(i.rm))) };
                if !to_rm && let Some(m) = i.mem {
                    let (base, off) = self.mem_operand(&m, next, w);
                    self.a.ldr(w, B, base, off);
                    return self.alu2(op, size, r(dst_reg), Opd::Reg(B), keep.then_some(Dst::Reg(dst_reg)), need);
                }
                match i.mem {
                    Some(m) if to_rm => {
                        let (base, off) = self.mem_operand(&m, next, w);
                        self.a.ldr(w, A, base, off);
                        let dst = keep.then_some(Dst::Mem(base, off));
                        self.alu2(op, size, A, src, dst, need)
                    }
                    _ => self.alu2(op, size, r(dst_reg), src, keep.then_some(Dst::Reg(dst_reg)), need),
                }
            }
            (Map::Primary, 0x81 | 0x83) if matches!(i.ext(), 0 | 1 | 4 | 5 | 6 | 7) => {
                let (op, keep) = alu_of(i.ext()).expect("group 1");
                let v = i.imm as u64 & mask(size);
                match i.mem {
                    Some(m) => {
                        let (base, off) = self.mem_operand(&m, next, w);
                        self.a.ldr(w, A, base, off);
                        let dst = keep.then_some(Dst::Mem(base, off));
                        self.alu2(op, size, A, Opd::Imm(v), dst, need)
                    }
                    None => self.alu2(op, size, r(i.rm), Opd::Imm(v), keep.then_some(Dst::Reg(i.rm)), need),
                }
            }
            (Map::Primary, 0xa9) => self.alu2(Alu::And, size, r(0), Opd::Imm(i.imm as u64 & mask(size)), None, need),
            (Map::Primary, 0x8d) if i.mem.is_some() => {
                let m = i.mem.expect("memory operand");
                if m.rip_relative || m.addr32 {
                    return false;
                }
                let d = m.disp as i64;
                match (m.base, m.index) {
                    (Some(b), None) if (0..4096).contains(&d) => {
                        self.a.add_imm(s, r(i.reg), r(b), d as u32);
                    }
                    (Some(b), None) if (-4095..0).contains(&d) => {
                        self.a.sub_imm(s, r(i.reg), r(b), (-d) as u32);
                    }
                    (Some(b), Some(x)) if d == 0 => {
                        self.a.add_shift(s, r(i.reg), r(b), r(x), a64::Shift::Lsl, m.scale.trailing_zeros());
                    }
                    _ => return false,
                }
                true
            }
            _ => false,
        }
    }

    /// `x op y` at `size` (4 or 8), the result written to `dst` when there is
    /// one, with the flags in `need` in tier 1's form.
    fn alu2(&mut self, op: Alu, size: u8, x: Reg, y: Opd, dst: Option<Dst>, need: u8) -> bool {
        if matches!(op, Alu::Adc | Alu::Sbb) {
            return false;
        }
        if dst.is_none() && need == 0 {
            return true;
        }
        let s = sz(size);
        let logic = matches!(op, Alu::And | Alu::Or | Alu::Xor);
        let nzcv = need & NZCV != 0;
        let parity = need & (PF | AF) != 0;
        let a = &mut self.a;
        let imm = match y {
            Opd::Imm(v) if !logic && v < 4096 && need & AF == 0 => Some(v as u32),
            _ => None,
        };
        let yr = match y {
            _ if imm.is_some() => B,
            Opd::Reg(reg) => reg,
            Opd::Imm(v) => {
                a.mov_imm(Size::X, B, v);
                B
            }
        };
        if need & AF != 0 && !logic {
            a.eor(Size::X, AF_AUX, x, yr);
        }
        let out = match dst {
            Some(Dst::Reg(d)) => r(d),
            _ if parity || dst.is_some() => T,
            _ => Reg(31),
        };
        match (op, imm, nzcv) {
            (Alu::Add, Some(v), true) => a.adds_imm(s, out, x, v),
            (Alu::Add, Some(v), false) => a.add_imm(s, out, x, v),
            (Alu::Sub, Some(v), true) => a.subs_imm(s, out, x, v),
            (Alu::Sub, Some(v), false) => a.sub_imm(s, out, x, v),
            (Alu::Add, None, true) => a.adds(s, out, x, yr),
            (Alu::Add, None, false) => a.add(s, out, x, yr),
            (Alu::Sub, None, true) => a.subs(s, out, x, yr),
            (Alu::Sub, None, false) => a.sub(s, out, x, yr),
            (Alu::And, _, true) => a.ands(s, out, x, yr),
            (Alu::And, _, false) => a.and(s, out, x, yr),
            (Alu::Or, _, _) => a.orr(s, out, x, yr),
            (Alu::Xor, _, _) => a.eor(s, out, x, yr),
            _ => unreachable!("ADC and SBB were refused"),
        }
        if nzcv && matches!(op, Alu::Or | Alu::Xor) {
            a.tst(s, out, out);
        }
        if need & CF != 0 && op != Alu::Sub {
            a.cfinv();
        }
        if parity {
            a.mov(Size::X, PF_RESULT, out);
        }
        if need & AF != 0 && logic {
            a.mov(Size::X, AF_AUX, out);
        }
        if let Some(Dst::Mem(base, off)) = dst {
            a.str(width(size), out, base, off);
        }
        true
    }
}

#[derive(Clone, Copy)]
enum Dst {
    Reg(u8),
    Mem(Reg, i32),
}

/// Tier-2 policy: off, or a block's region compiled after the block runs
/// this many times (1 is the eager test mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier2 {
    Off,
    Hot(u32),
}

impl Tier2 {
    /// From `PS5SLOP_JIT_TIER2`: unset or `off`, `eager`, or a count.
    pub fn from_env() -> Tier2 {
        match std::env::var("PS5SLOP_JIT_TIER2").as_deref() {
            Ok("eager") => Tier2::Hot(1),
            Ok("off") | Err(_) => Tier2::Off,
            Ok(n) => Tier2::Hot(n.parse().unwrap_or(1000)),
        }
    }

    pub fn count(self) -> Option<u32> {
        match self {
            Tier2::Off => None,
            Tier2::Hot(n) => Some(n.max(1)),
        }
    }
}
