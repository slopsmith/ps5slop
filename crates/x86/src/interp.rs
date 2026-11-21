//! The reference interpreter: decodes and runs one instruction at a time.
//! Semantics follow the Intel SDM, volume 2.

use memory::{Fault, FaultKind, Memory};

use crate::alu::{self, mask, sext};
use crate::cpu::{Cpu, flags::*, reg::*};
use crate::decode::{DecodeError, Insn, Map, Mem, Rep, Seg, decode};

/// Why [`run`] or [`step`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// The instruction ran; the next one is at `cpu.rip`.
    Continue,
    /// Execution reached a trap region at this address: a call into the host.
    Trap(u64),
    /// `syscall` ran; `cpu.rip` is already past it.
    Syscall,
    /// The step limit ran out.
    Limit,
    /// The instruction at `rip` faulted; the state is as before it.
    Fault { rip: u64, kind: CpuFault },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuFault {
    Memory(Fault),
    InvalidOpcode,
    /// A valid instruction the interpreter does not implement yet.
    Unimplemented {
        map: Map,
        op: u8,
    },
    /// A prefix family the decoder does not handle yet (VEX, EVEX, XOP).
    UnsupportedEncoding,
    DivideError,
    /// #GP: for example a misaligned operand of an aligned SSE move.
    GeneralProtection,
    Breakpoint,
    /// A privileged instruction (`hlt`, `in`, `out`, ...).
    Privileged,
}

impl From<Fault> for CpuFault {
    fn from(f: Fault) -> Self {
        CpuFault::Memory(f)
    }
}

type Res<T> = Result<T, CpuFault>;

/// Runs until something other than [`Exit::Continue`] happens, or for at most
/// `limit` instructions.
pub fn run(cpu: &mut Cpu, mem: &Memory, limit: u64) -> Exit {
    for _ in 0..limit {
        match step(cpu, mem) {
            Exit::Continue => {}
            other => return other,
        }
    }
    Exit::Limit
}

pub fn step(cpu: &mut Cpu, mem: &Memory) -> Exit {
    let rip = cpu.rip;
    if memory::watch::active() {
        memory::watch::set_site(rip);
    }
    let insn = decode(|i| {
        let mut b = [0u8];
        mem.fetch(rip.wrapping_add(i as u64), &mut b).map(|_| b[0])
    });
    let insn = match insn {
        Ok(i) => i,
        Err(DecodeError::Fetch(f)) if f.kind == FaultKind::Trap && f.addr == rip => {
            return Exit::Trap(rip);
        }
        Err(DecodeError::Fetch(f)) => {
            return Exit::Fault {
                rip,
                kind: f.into(),
            };
        }
        Err(DecodeError::Invalid | DecodeError::TooLong) => {
            return Exit::Fault {
                rip,
                kind: CpuFault::InvalidOpcode,
            };
        }
        Err(DecodeError::Unsupported) => {
            return Exit::Fault {
                rip,
                kind: CpuFault::UnsupportedEncoding,
            };
        }
    };
    let saved = cpu.clone();
    let mut ctx = Ctx {
        cpu,
        mem,
        insn,
        next: rip + insn.len as u64,
        half: 0,
        wide: false,
    };
    match ctx.exec() {
        Ok(exit) => exit,
        Err(kind) => {
            *ctx.cpu = saved;
            Exit::Fault { rip, kind }
        }
    }
}

#[derive(Clone, Copy)]
enum Loc {
    Reg(u8),
    Mem(u64),
}

struct Ctx<'a> {
    cpu: &'a mut Cpu,
    mem: &'a Memory,
    insn: Insn,
    /// Address of the next instruction.
    next: u64,
    /// The 128-bit half of a YMM operand an in-lane vector operation is on.
    half: u8,
    /// Set while a 256-bit in-lane operation runs, half by half.
    wide: bool,
}

impl Ctx<'_> {
    fn get(&self, r: u8, size: u8) -> u64 {
        if size == 1 && self.insn.rex == 0 && (4..8).contains(&r) {
            return (self.cpu.regs[r as usize - 4] >> 8) & 0xff;
        }
        self.cpu.regs[r as usize] & mask(size)
    }

    fn set(&mut self, r: u8, size: u8, v: u64) {
        if size == 1 && self.insn.rex == 0 && (4..8).contains(&r) {
            let reg = &mut self.cpu.regs[r as usize - 4];
            *reg = (*reg & !0xff00) | ((v & 0xff) << 8);
            return;
        }
        let reg = &mut self.cpu.regs[r as usize];
        *reg = match size {
            8 => v,
            4 => v & 0xffff_ffff,
            _ => (*reg & !mask(size)) | (v & mask(size)),
        };
    }

    fn ea(&self, m: &Mem) -> u64 {
        let mut a = if m.rip_relative {
            self.next
        } else {
            m.base.map_or(0, |b| self.cpu.regs[b as usize])
        };
        if let Some(i) = m.index {
            a = a.wrapping_add(self.cpu.regs[i as usize].wrapping_mul(m.scale as u64));
        }
        a = a.wrapping_add(m.disp as i64 as u64);
        if m.addr32 {
            a &= 0xffff_ffff;
        }
        match m.seg {
            Some(Seg::Fs) => a.wrapping_add(self.cpu.fs_base),
            Some(Seg::Gs) => a.wrapping_add(self.cpu.gs_base),
            None => a,
        }
    }

    fn rm(&self) -> Loc {
        match &self.insn.mem {
            Some(m) => Loc::Mem(self.ea(m)),
            None => Loc::Reg(self.insn.rm),
        }
    }

    fn load(&self, addr: u64, size: u8) -> Res<u64> {
        let mut b = [0u8; 8];
        self.mem.read(addr, &mut b[..size as usize])?;
        Ok(u64::from_le_bytes(b))
    }

    fn store(&mut self, addr: u64, size: u8, v: u64) -> Res<()> {
        self.mem.write(addr, &v.to_le_bytes()[..size as usize])?;
        Ok(())
    }

    fn read(&self, loc: Loc, size: u8) -> Res<u64> {
        match loc {
            Loc::Reg(r) => Ok(self.get(r, size)),
            Loc::Mem(a) => self.load(a, size),
        }
    }

    fn write(&mut self, loc: Loc, size: u8, v: u64) -> Res<()> {
        match loc {
            Loc::Reg(r) => {
                self.set(r, size, v);
                Ok(())
            }
            Loc::Mem(a) => self.store(a, size, v),
        }
    }

    fn push(&mut self, v: u64, size: u8) -> Res<()> {
        let sp = self.cpu.regs[RSP].wrapping_sub(size as u64);
        self.store(sp, size, v)?;
        self.cpu.regs[RSP] = sp;
        Ok(())
    }

    fn pop(&mut self, size: u8) -> Res<u64> {
        let sp = self.cpu.regs[RSP];
        let v = self.load(sp, size)?;
        self.cpu.regs[RSP] = sp.wrapping_add(size as u64);
        Ok(v)
    }

    /// A register used as an address or counter: 32 bits wide under `67`.
    fn areg(&self, r: usize) -> u64 {
        if self.insn.addr32 {
            self.cpu.regs[r] & 0xffff_ffff
        } else {
            self.cpu.regs[r]
        }
    }

    fn set_areg(&mut self, r: usize, v: u64) {
        self.cpu.regs[r] = if self.insn.addr32 { v & 0xffff_ffff } else { v };
    }

    fn imm(&self) -> u64 {
        self.insn.imm as u64
    }

    fn jump(&mut self, target: u64) {
        self.next = target;
    }

    fn rel(&self) -> u64 {
        self.next.wrapping_add(self.imm())
    }

    fn exec(&mut self) -> Res<Exit> {
        let i = self.insn;
        let exit = match i.map {
            Map::Primary => self.primary()?,
            Map::M0F => self.map0f()?,
            Map::M0F38 | Map::M0F3A => {
                if !self.ext()? {
                    return self.unimplemented();
                }
                Exit::Continue
            }
        };
        if exit == Exit::Continue || exit == Exit::Syscall {
            self.cpu.rip = self.next;
        }
        Ok(exit)
    }

    fn unimplemented<T>(&self) -> Res<T> {
        Err(CpuFault::Unimplemented {
            map: self.insn.map,
            op: self.insn.op,
        })
    }

    /// Stores `f(old)` at `loc` (unless `None`) and returns `old`; with `atomic`,
    /// a memory operand is updated atomically, as by a locked instruction.
    fn modify(
        &mut self,
        loc: Loc,
        size: u8,
        atomic: bool,
        f: impl Fn(u64) -> Option<u64>,
    ) -> Res<u64> {
        match loc {
            Loc::Mem(a) if atomic => Ok(self
                .mem
                .atomic_update(a, size, |old| f(old).map(|v| v & mask(size)))?),
            _ => {
                let old = self.read(loc, size)?;
                if let Some(new) = f(old) {
                    self.write(loc, size, new)?;
                }
                Ok(old)
            }
        }
    }

    /// `cmpxchg8b` and (`wide`) `cmpxchg16b`: compare EDX:EAX (RDX:RAX)
    /// with memory, store ECX:EBX (RCX:RBX) if equal, else load it.
    fn cmpxchg_wide(&mut self, a: u64, wide: bool) -> Res<()> {
        let r = &self.cpu.regs;
        let equal = if wide {
            if !a.is_multiple_of(16) {
                return Err(CpuFault::GeneralProtection);
            }
            let expected = (r[RDX] as u128) << 64 | r[RAX] as u128;
            let new = (r[RCX] as u128) << 64 | r[RBX] as u128;
            let old = self.mem.atomic_cas128(a, expected, new)?;
            if old != expected {
                self.cpu.regs[RAX] = old as u64;
                self.cpu.regs[RDX] = (old >> 64) as u64;
            }
            old == expected
        } else {
            let expected = (r[RDX] & 0xffff_ffff) << 32 | r[RAX] & 0xffff_ffff;
            let new = (r[RCX] & 0xffff_ffff) << 32 | r[RBX] & 0xffff_ffff;
            let old = self
                .mem
                .atomic_update(a, 8, |old| (old == expected).then_some(new))?;
            if old != expected {
                self.cpu.regs[RAX] = old & 0xffff_ffff;
                self.cpu.regs[RDX] = old >> 32;
            }
            old == expected
        };
        self.cpu.set_flag(ZF, equal);
        Ok(())
    }

    fn alu_rm(&mut self, op: u8, dst: Loc, src: u64, size: u8) -> Res<()> {
        let cf = self.cpu.flag(CF);
        let old = if op == 7 {
            self.read(dst, size)?
        } else {
            self.modify(dst, size, self.insn.lock, |a| {
                Some(alu::group1(op, a, src, cf, size).0)
            })?
        };
        self.cpu.set_arith(alu::group1(op, old, src, cf, size).1);
        Ok(())
    }

    fn primary(&mut self) -> Res<Exit> {
        let i = self.insn;
        let op = i.op;
        let osz = i.osize();
        match op {
            0x00..=0x3f => {
                let alu_op = op >> 3;
                let size = if op & 1 == 0 { 1 } else { osz };
                match op & 7 {
                    0 | 1 => {
                        let src = self.get(i.reg, size);
                        self.alu_rm(alu_op, self.rm(), src, size)?;
                    }
                    2 | 3 => {
                        let src = self.read(self.rm(), size)?;
                        self.alu_rm(alu_op, Loc::Reg(i.reg), src, size)?;
                    }
                    4 => self.alu_rm(alu_op, Loc::Reg(0), self.imm(), 1)?,
                    _ => self.alu_rm(alu_op, Loc::Reg(0), self.imm(), osz)?,
                }
            }
            0x50..=0x57 => {
                let size = i.osize64();
                self.push(self.cpu.regs[i.rm as usize], size)?;
            }
            0x58..=0x5f => {
                let size = i.osize64();
                let v = self.pop(size)?;
                self.set(i.rm, size, v);
            }
            0x63 => {
                let v = self.read(self.rm(), 4)?;
                let v = if i.rex_w() { sext(v, 4) } else { v };
                self.set(i.reg, osz, v);
            }
            0x68 | 0x6a => self.push(self.imm(), i.osize64())?,
            0x69 | 0x6b => {
                let a = self.read(self.rm(), osz)?;
                self.imul2(i.reg, a, self.imm(), osz);
            }
            0x70..=0x7f => {
                if self.cpu.condition(op & 0xf) {
                    self.jump(self.rel());
                }
            }
            0x80 | 0x81 | 0x83 => {
                let size = if op == 0x80 { 1 } else { osz };
                self.alu_rm(i.ext(), self.rm(), self.imm(), size)?;
            }
            0x84 | 0x85 => {
                let size = if op == 0x84 { 1 } else { osz };
                let a = self.read(self.rm(), size)?;
                let (_, f) = alu::logic(a & self.get(i.reg, size), size);
                self.cpu.set_arith(f);
            }
            0x86 | 0x87 => {
                let size = if op == 0x86 { 1 } else { osz };
                let b = self.get(i.reg, size);
                let a = self.modify(self.rm(), size, true, |_| Some(b))?;
                self.set(i.reg, size, a);
            }
            0x88 | 0x89 => {
                let size = if op == 0x88 { 1 } else { osz };
                let v = self.get(i.reg, size);
                self.write(self.rm(), size, v)?;
            }
            0x8a | 0x8b => {
                let size = if op == 0x8a { 1 } else { osz };
                let v = self.read(self.rm(), size)?;
                self.set(i.reg, size, v);
            }
            0x8d => {
                let m = i.mem.ok_or(CpuFault::InvalidOpcode)?;
                let m = Mem { seg: None, ..m };
                let a = self.ea(&m);
                self.set(i.reg, osz, a);
            }
            0x8f => {
                let size = i.osize64();
                let v = self.pop(size)?;
                self.write(self.rm(), size, v)?;
            }
            0x90 if i.rex & 1 == 0 => {}
            0x90..=0x97 => {
                let a = self.get(0, osz);
                let b = self.get(i.rm, osz);
                self.set(0, osz, b);
                self.set(i.rm, osz, a);
            }
            0x98 => {
                let half = osz / 2;
                let v = sext(self.get(0, half), half);
                self.set(0, osz, v);
            }
            0x99 => {
                let neg = self.get(0, osz) & alu::sign(osz) != 0;
                self.set(RDX as u8, osz, if neg { u64::MAX } else { 0 });
            }
            0x9c => self.push((self.cpu.rflags & !(TF | 1 << 16)) | FIXED, i.osize64())?,
            0x9d => {
                let v = self.pop(i.osize64())?;
                self.cpu.rflags = (self.cpu.rflags & !USER_WRITABLE) | (v & USER_WRITABLE) | FIXED;
            }
            0x9e => {
                let ah = (self.cpu.regs[RAX] >> 8) & (SF | ZF | AF | PF | CF);
                self.cpu.rflags = (self.cpu.rflags & !(SF | ZF | AF | PF | CF)) | ah;
            }
            0x9f => {
                let f = (self.cpu.rflags & (SF | ZF | AF | PF | CF)) | 2;
                self.cpu.regs[RAX] = (self.cpu.regs[RAX] & !0xff00) | f << 8;
            }
            0xa4..=0xa7 | 0xaa..=0xaf => self.string()?,
            0xa8 | 0xa9 => {
                let size = if op == 0xa8 { 1 } else { osz };
                let (_, f) = alu::logic(self.get(0, size) & self.imm(), size);
                self.cpu.set_arith(f);
            }
            0xb0..=0xb7 => self.set(i.rm, 1, self.imm()),
            0xb8..=0xbf => self.set(i.rm, osz, self.imm()),
            0xc0 | 0xc1 | 0xd0..=0xd3 => {
                let size = if op & 1 == 0 { 1 } else { osz };
                let count = match op {
                    0xc0 | 0xc1 => self.imm(),
                    0xd0 | 0xd1 => 1,
                    _ => self.cpu.regs[RCX],
                } as u32
                    & if size == 8 { 63 } else { 31 };
                let loc = self.rm();
                let a = self.read(loc, size)?;
                let (r, f) = alu::shift(i.ext(), a, count, self.cpu.rflags, size);
                self.cpu.set_arith(f);
                if count != 0 || matches!(loc, Loc::Reg(_)) {
                    self.write(loc, size, r)?;
                }
            }
            0xc2 | 0xc3 => {
                let target = self.pop(8)?;
                if op == 0xc2 {
                    self.cpu.regs[RSP] = self.cpu.regs[RSP].wrapping_add(self.imm());
                }
                self.jump(target);
            }
            0xc6 | 0xc7 if i.ext() == 0 => {
                let size = if op == 0xc6 { 1 } else { osz };
                self.write(self.rm(), size, self.imm())?;
            }
            0xc9 => {
                self.cpu.regs[RSP] = self.cpu.regs[RBP];
                let v = self.pop(i.osize64())?;
                self.set(RBP as u8, i.osize64(), v);
            }
            0xcc => return Err(CpuFault::Breakpoint),
            0x9b => {}
            0xd8..=0xdf => self.x87()?,
            0xe3 => {
                let rcx = if i.addr32 {
                    self.get(1, 4)
                } else {
                    self.cpu.regs[RCX]
                };
                if rcx == 0 {
                    self.jump(self.rel());
                }
            }
            0xe0..=0xe2 => {
                let rcx = self.areg(RCX).wrapping_sub(1);
                self.set_areg(RCX, rcx);
                let rcx = self.areg(RCX);
                let zf = self.cpu.flag(ZF);
                let taken = rcx != 0
                    && match op {
                        0xe0 => !zf,
                        0xe1 => zf,
                        _ => true,
                    };
                if taken {
                    self.jump(self.rel());
                }
            }
            0xe8 => {
                self.push(self.next, 8)?;
                self.jump(self.rel());
            }
            0xe9 | 0xeb => self.jump(self.rel()),
            0xe4..=0xe7 | 0xec..=0xef | 0xf4 | 0xfa | 0xfb => return Err(CpuFault::Privileged),
            0xf5 => self.cpu.rflags ^= CF,
            0xf8 => self.cpu.set_flag(CF, false),
            0xf9 => self.cpu.set_flag(CF, true),
            0xfc => self.cpu.set_flag(DF, false),
            0xfd => self.cpu.set_flag(DF, true),
            0xf6 | 0xf7 => self.group3(if op == 0xf6 { 1 } else { osz })?,
            0xfe | 0xff => {
                let size = if op == 0xfe { 1 } else { osz };
                match i.ext() {
                    0 | 1 => {
                        let (dec, flags) = (i.ext() == 1, self.cpu.rflags);
                        let a = self.modify(self.rm(), size, i.lock, |a| {
                            Some(alu::inc_dec(a, dec, flags, size).0)
                        })?;
                        self.cpu.set_arith(alu::inc_dec(a, dec, flags, size).1);
                    }
                    2 if op == 0xff => {
                        let target = self.read(self.rm(), 8)?;
                        self.push(self.next, 8)?;
                        self.jump(target);
                    }
                    4 if op == 0xff => {
                        let target = self.read(self.rm(), 8)?;
                        self.jump(target);
                    }
                    6 if op == 0xff => {
                        let size = i.osize64();
                        let v = self.read(self.rm(), size)?;
                        self.push(v, size)?;
                    }
                    _ => return self.unimplemented(),
                }
            }
            _ => return self.unimplemented(),
        }
        Ok(Exit::Continue)
    }

    fn imul2(&mut self, dst: u8, a: u64, b: u64, size: u8) {
        let full = sext(a, size) as i64 as i128 * sext(b, size) as i64 as i128;
        let r = full as u64 & mask(size);
        let fits = sext(r, size) as i64 as i128 == full;
        let mut f = alu::szp(r, size);
        if !fits {
            f |= CF | OF;
        }
        self.cpu.set_arith(f);
        self.set(dst, size, r);
    }

    fn group3(&mut self, size: u8) -> Res<()> {
        let i = self.insn;
        let loc = self.rm();
        if matches!(i.ext(), 2 | 3) {
            return self.not_neg(loc, size);
        }
        let a = self.read(loc, size)?;
        match i.ext() {
            0 | 1 => {
                let (_, f) = alu::logic(a & self.imm(), size);
                self.cpu.set_arith(f);
            }
            4 | 5 => {
                let acc = self.get(0, size);
                let (lo, hi, overflow) = if i.ext() == 4 {
                    let p = acc as u128 * a as u128;
                    (
                        p as u64 & mask(size),
                        (p >> (size * 8)) as u64 & mask(size),
                        (p >> (size * 8)) != 0,
                    )
                } else {
                    let p = sext(acc, size) as i64 as i128 * sext(a, size) as i64 as i128;
                    let lo = p as u64 & mask(size);
                    (
                        lo,
                        (p >> (size * 8)) as u64 & mask(size),
                        sext(lo, size) as i64 as i128 != p,
                    )
                };
                let mut f = alu::szp(lo, size);
                if overflow {
                    f |= CF | OF;
                }
                self.cpu.set_arith(f);
                if size == 1 {
                    self.set(0, 2, lo | hi << 8);
                } else {
                    self.set(0, size, lo);
                    self.set(RDX as u8, size, hi);
                }
            }
            _ => {
                let divisor = a & mask(size);
                if divisor == 0 {
                    return Err(CpuFault::DivideError);
                }
                let bits = size as u32 * 8;
                let (hi, lo) = if size == 1 {
                    let ax = self.get(0, 2);
                    (ax >> 8, ax & 0xff)
                } else {
                    (self.get(RDX as u8, size), self.get(0, size))
                };
                let dividend = (hi as u128) << bits | lo as u128;
                let (q, r) = if i.ext() == 6 {
                    let q = dividend / divisor as u128;
                    if q > mask(size) as u128 {
                        return Err(CpuFault::DivideError);
                    }
                    (q as u64, (dividend % divisor as u128) as u64)
                } else {
                    let shift = 128 - 2 * bits;
                    let n = ((dividend << shift) as i128) >> shift;
                    let d = sext(divisor, size) as i64 as i128;
                    let q = n.checked_div(d).ok_or(CpuFault::DivideError)?;
                    let limit = 1i128 << (bits - 1);
                    if q >= limit || q < -limit {
                        return Err(CpuFault::DivideError);
                    }
                    (q as u64 & mask(size), (n % d) as u64 & mask(size))
                };
                if size == 1 {
                    self.set(0, 2, q | r << 8);
                } else {
                    self.set(0, size, q);
                    self.set(RDX as u8, size, r);
                }
            }
        }
        Ok(())
    }

    fn not_neg(&mut self, loc: Loc, size: u8) -> Res<()> {
        let i = self.insn;
        match i.ext() {
            2 => {
                self.modify(loc, size, i.lock, |a| Some(!a))?;
            }
            3 => {
                let a = self.modify(loc, size, i.lock, |a| Some(alu::sub(0, a, false, size).0))?;
                let mut f = alu::sub(0, a, false, size).1;
                if a & mask(size) == 0 {
                    f &= !CF;
                }
                self.cpu.set_arith(f);
            }
            _ => unreachable!("only NOT and NEG"),
        }
        Ok(())
    }

    fn string(&mut self) -> Res<()> {
        let i = self.insn;
        let size = if i.op & 1 == 0 { 1 } else { i.osize() };
        let step = if self.cpu.flag(DF) {
            (size as u64).wrapping_neg()
        } else {
            size as u64
        };
        let rep = i.rep != Rep::None;
        let compare = matches!(i.op, 0xa6 | 0xa7 | 0xae | 0xaf);
        loop {
            if rep && self.areg(RCX) == 0 {
                break;
            }
            let si = self.areg(RSI);
            let di = self.areg(RDI);
            match i.op {
                0xa4 | 0xa5 => {
                    let v = self.load(si, size)?;
                    self.store(di, size, v)?;
                    self.set_areg(RSI, si.wrapping_add(step));
                    self.set_areg(RDI, di.wrapping_add(step));
                }
                0xaa | 0xab => {
                    self.store(di, size, self.get(0, size))?;
                    self.set_areg(RDI, di.wrapping_add(step));
                }
                0xac | 0xad => {
                    let v = self.load(si, size)?;
                    self.set(0, size, v);
                    self.set_areg(RSI, si.wrapping_add(step));
                }
                0xa6 | 0xa7 => {
                    let a = self.load(si, size)?;
                    let b = self.load(di, size)?;
                    self.cpu.set_arith(alu::sub(a, b, false, size).1);
                    self.set_areg(RSI, si.wrapping_add(step));
                    self.set_areg(RDI, di.wrapping_add(step));
                }
                _ => {
                    let b = self.load(di, size)?;
                    self.cpu
                        .set_arith(alu::sub(self.get(0, size), b, false, size).1);
                    self.set_areg(RDI, di.wrapping_add(step));
                }
            }
            if !rep {
                break;
            }
            self.set_areg(RCX, self.areg(RCX).wrapping_sub(1));
            if compare {
                let zf = self.cpu.flag(ZF);
                if (i.rep == Rep::Rep && !zf) || (i.rep == Rep::Repne && zf) {
                    break;
                }
            }
        }
        Ok(())
    }

    /// The time stamp counter counts nanoseconds since the Unix epoch (1 GHz).
    fn rdtsc(&mut self) {
        let t = savestate::clock::unix_nanos();
        self.cpu.regs[RAX] = t & 0xffff_ffff;
        self.cpu.regs[RDX] = t >> 32;
    }

    fn map0f(&mut self) -> Res<Exit> {
        if self.sse()? {
            return Ok(Exit::Continue);
        }
        let i = self.insn;
        let osz = i.osize();
        match i.op {
            0x05 => return Ok(Exit::Syscall),
            0x0b => return Err(CpuFault::InvalidOpcode),
            0x0d | 0x18..=0x1f => {}
            0x31 => self.rdtsc(),
            0x01 if i.modrm == Some(0xf9) => {
                self.rdtsc();
                self.cpu.regs[RCX] = 0;
            }
            0x40..=0x4f => {
                let v = self.read(self.rm(), osz)?;
                if self.cpu.condition(i.op & 0xf) {
                    self.set(i.reg, osz, v);
                } else if osz == 4 {
                    let cur = self.get(i.reg, 4);
                    self.set(i.reg, 4, cur);
                }
            }
            0x80..=0x8f => {
                if self.cpu.condition(i.op & 0xf) {
                    self.jump(self.rel());
                }
            }
            0x90..=0x9f => {
                let v = self.cpu.condition(i.op & 0xf) as u64;
                self.write(self.rm(), 1, v)?;
            }
            0xa2 => self.cpuid(),
            0x01 if i.modrm == Some(0xd0) => {
                if self.cpu.regs[RCX] as u32 != 0 {
                    return Err(CpuFault::GeneralProtection);
                }
                self.cpu.regs[RAX] = sse::XCR0 & 0xffff_ffff;
                self.cpu.regs[RDX] = sse::XCR0 >> 32;
            }
            0xc7 if i.ext() == 1 => {
                let Loc::Mem(a) = self.rm() else {
                    return Err(CpuFault::InvalidOpcode);
                };
                self.cmpxchg_wide(a, osz == 8)?;
            }
            0xc7 if i.ext() == 4 => {
                let Loc::Mem(a) = self.rm() else {
                    return Err(CpuFault::InvalidOpcode);
                };
                self.xsave(a, true)?;
            }
            0xa3 | 0xab | 0xb3 | 0xbb | 0xba => self.bit_test()?,
            0xa4 | 0xa5 | 0xac | 0xad => {
                let count = if i.op & 1 == 0 {
                    self.imm()
                } else {
                    self.cpu.regs[RCX]
                } as u32
                    & if osz == 8 { 63 } else { 31 };
                let loc = self.rm();
                let a = self.read(loc, osz)?;
                if count == 0 {
                    if let Loc::Reg(_) = loc {
                        self.write(loc, osz, a)?;
                    }
                } else {
                    let b = self.get(i.reg, osz);
                    let bits = osz as u32 * 8;
                    let (r, out) = if i.op < 0xa8 {
                        let wide = (a as u128) << bits | b as u128;
                        let r = (wide << count >> bits) as u64 & mask(osz);
                        (r, (a >> (bits - count)) & 1 != 0)
                    } else {
                        let wide = (b as u128) << bits | a as u128;
                        let r = (wide >> count) as u64 & mask(osz);
                        (r, (a >> (count - 1)) & 1 != 0)
                    };
                    let mut f = alu::szp(r, osz);
                    if out {
                        f |= CF;
                    }
                    if (r ^ a) & alu::sign(osz) != 0 {
                        f |= OF;
                    }
                    self.cpu.set_arith(f);
                    self.write(loc, osz, r)?;
                }
            }
            0xaf => {
                let a = self.get(i.reg, osz);
                let b = self.read(self.rm(), osz)?;
                self.imul2(i.reg, a, b, osz);
            }
            0xb0 | 0xb1 => {
                let size = if i.op == 0xb0 { 1 } else { osz };
                let acc = self.get(0, size);
                let src = self.get(i.reg, size);
                let cur =
                    self.modify(self.rm(), size, i.lock, |cur| (cur == acc).then_some(src))?;
                self.cpu.set_arith(alu::sub(acc, cur, false, size).1);
                if acc != cur {
                    self.set(0, size, cur);
                }
            }
            0xb6 | 0xb7 | 0xbe | 0xbf => {
                let src = if i.op & 1 == 0 { 1 } else { 2 };
                let v = self.read(self.rm(), src)?;
                let v = if i.op >= 0xbe { sext(v, src) } else { v };
                self.set(i.reg, osz, v);
            }
            0xb8 if i.rep == Rep::Rep => {
                let v = self.read(self.rm(), osz)?;
                let n = v.count_ones() as u64;
                self.cpu.set_arith(if v == 0 { ZF } else { 0 });
                self.set(i.reg, osz, n);
            }
            0xbc | 0xbd => {
                let v = self.read(self.rm(), osz)?;
                let bits = osz as u64 * 8;
                if i.rep == Rep::Rep {
                    let n = if i.op == 0xbc {
                        if v == 0 {
                            bits
                        } else {
                            v.trailing_zeros() as u64
                        }
                    } else if v == 0 {
                        bits
                    } else {
                        v.leading_zeros() as u64 - (64 - bits)
                    };
                    let mut f = self.cpu.rflags & !(CF | ZF);
                    if v == 0 {
                        f |= CF;
                    }
                    if n == 0 {
                        f |= ZF;
                    }
                    self.cpu.set_arith(f);
                    self.set(i.reg, osz, n);
                } else if v == 0 {
                    self.cpu.set_flag(ZF, true);
                } else {
                    self.cpu.set_flag(ZF, false);
                    let n = if i.op == 0xbc {
                        v.trailing_zeros() as u64
                    } else {
                        63 - v.leading_zeros() as u64
                    };
                    self.set(i.reg, osz, n);
                }
            }
            0xc0 | 0xc1 => {
                let size = if i.op == 0xc0 { 1 } else { osz };
                let b = self.get(i.reg, size);
                let loc = self.rm();
                let a = match loc {
                    Loc::Reg(r) => self.get(r, size),
                    Loc::Mem(_) => {
                        self.modify(loc, size, i.lock, |a| Some(alu::add(a, b, false, size).0))?
                    }
                };
                let (sum, f) = alu::add(a, b, false, size);
                self.cpu.set_arith(f);
                self.set(i.reg, size, a);
                if let Loc::Reg(r) = loc {
                    self.set(r, size, sum);
                }
            }
            0xc8..=0xcf => {
                let v = self.get(i.rm, osz);
                let r = if osz == 8 {
                    v.swap_bytes()
                } else {
                    (v as u32).swap_bytes() as u64
                };
                self.set(i.rm, osz, r);
            }
            _ => return self.unimplemented(),
        }
        Ok(Exit::Continue)
    }

    fn bit_test(&mut self) -> Res<()> {
        let i = self.insn;
        let osz = i.osize();
        let bits = osz as u64 * 8;
        let (op, offset) = if i.op == 0xba {
            if i.ext() < 4 {
                return Err(CpuFault::InvalidOpcode);
            }
            (i.ext() - 4, self.imm() & (bits - 1))
        } else {
            ((i.op >> 3) & 3, self.get(i.reg, osz))
        };
        let (loc, bit) = match (self.rm(), i.op) {
            (Loc::Mem(a), 0xa3 | 0xab | 0xb3 | 0xbb) => {
                let signed = sext(offset, osz) as i64;
                let word = signed.div_euclid(bits as i64);
                (
                    Loc::Mem(a.wrapping_add((word * osz as i64) as u64)),
                    signed.rem_euclid(bits as i64) as u64,
                )
            }
            (loc, _) => (loc, offset & (bits - 1)),
        };
        let v = if op == 0 {
            self.read(loc, osz)?
        } else {
            self.modify(loc, osz, i.lock, |v| {
                Some(match op {
                    1 => v | 1 << bit,
                    2 => v & !(1 << bit),
                    _ => v ^ 1 << bit,
                })
            })?
        };
        self.cpu.set_flag(CF, v >> bit & 1 != 0);
        Ok(())
    }

    fn cpuid(&mut self) {
        let (a, b, c, d) =
            crate::cpuid::cpuid(self.cpu.regs[RAX] as u32, self.cpu.regs[RCX] as u32);
        self.cpu.regs[RAX] = a as u64;
        self.cpu.regs[RBX] = b as u64;
        self.cpu.regs[RCX] = c as u64;
        self.cpu.regs[RDX] = d as u64;
    }
}

mod ext;
mod sse;
mod x87;

#[cfg(test)]
mod tests;
