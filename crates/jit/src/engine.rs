//! The run loop for one guest thread: enters the VM, handles its exits, and
//! falls back to the interpreter for what is not translated.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use memory::{FaultKind, Memory};
use x86::Cpu as X86;
use x86::cpu::flags;

use crate::translate::{AF_AUX, FS_BASE, HVC_INTERP, HVC_MISS, ICOUNT, PF_RESULT, RIP};
use crate::{CPSR_EL1H, Cpu, DISPATCH, ENTRY_FLUSH, Exit, RESUME_FLUSH, machine};

/// Instructions the interpreter may run in a row before trying translated
/// code again.
const INTERP_RUN: usize = 64;
/// What the interpreter runs per call when no vCPU is available.
const FALLBACK_SLICE: u64 = 1 << 16;

const N: u64 = 1 << 31;
const Z: u64 = 1 << 30;
const C: u64 = 1 << 29;
const V: u64 = 1 << 28;
const NZCV: u64 = N | Z | C | V;

struct Engine {
    cpu: Cpu,
    epoch: Option<u64>,
    /// The instruction counter's value at the last exit.
    icount: u64,
}

static TRANSLATED: AtomicU64 = AtomicU64::new(0);
static INTERPRETED: AtomicU64 = AtomicU64::new(0);

/// Guest instructions run so far by every thread: (interpreted, translated).
pub fn instructions() -> (u64, u64) {
    (
        INTERPRETED.load(Ordering::Relaxed),
        TRANSLATED.load(Ordering::Relaxed),
    )
}

fn count_interpreted(n: u64) {
    INTERPRETED.fetch_add(n, Ordering::Relaxed);
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Ok(mut m) = machine() {
            m.vcpus[self.cpu.slot] = None;
            m.running[self.cpu.slot] = false;
        }
    }
}

enum Slot {
    Untried,
    Unavailable,
    Ready(Box<Engine>),
}

thread_local! {
    static THREAD: RefCell<Slot> = const { RefCell::new(Slot::Untried) };
}

/// Runs `cpu` against `mem` until it calls into the host (a trap region or
/// `syscall`), faults, or `stop` says to stop (reported as `Exit::Limit`).
/// Threads that get no vCPU run in the interpreter.
pub fn run(cpu: &mut X86, mem: &Memory, stop: &dyn Fn() -> bool) -> x86::Exit {
    THREAD.with(|t| {
        let mut t = t.borrow_mut();
        if let Slot::Untried = *t {
            *t = match machine().and_then(|mut m| Cpu::new(&mut m)) {
                Ok(Some(cpu)) => Slot::Ready(Box::new(Engine {
                    cpu,
                    epoch: None,
                    icount: 0,
                })),
                _ => Slot::Unavailable,
            };
        }
        match &mut *t {
            Slot::Ready(e) => match e.run(cpu, mem, stop) {
                Ok(exit) => exit,
                Err(err) => panic!("the JIT's VM failed: {err:?}"),
            },
            _ => interpret(cpu, mem, FALLBACK_SLICE),
        }
    })
}

/// Forces every vCPU out of the VM, so their threads notice a stop.
pub fn kick_all() {
    if let Ok(m) = machine() {
        m.kick_running();
    }
}

fn parity_even(v: u64) -> bool {
    (v as u8).count_ones().is_multiple_of(2)
}

/// Where the next VM entry starts.
#[derive(Clone, Copy)]
enum Entry {
    /// The dispatcher, at the RIP in x16, with these NZCV bits.
    Dispatch(u64),
    /// Back at an interrupted `pc` with its `cpsr`.
    Resume { pc: u64, cpsr: u64 },
}

impl Engine {
    /// Puts `x` into the vCPU and returns its flags as NZCV bits.
    fn load(&self, x: &X86) -> Result<u64, crate::Error> {
        for (i, &v) in x.regs.iter().enumerate() {
            self.cpu.set_reg(i as u8, v)?;
        }
        self.cpu.set_reg(RIP.0, x.rip)?;
        self.cpu.set_reg(FS_BASE.0, x.fs_base)?;
        let f = x.rflags;
        let pf_result = if f & flags::PF != 0 { 0 } else { 1 };
        let af = if f & flags::AF != 0 { 0x10 } else { 0 };
        self.cpu.set_reg(PF_RESULT.0, pf_result)?;
        self.cpu.set_reg(AF_AUX.0, pf_result ^ af)?;
        let mut nzcv = 0;
        for (bit, flag) in [(N, flags::SF), (Z, flags::ZF), (V, flags::OF)] {
            if f & flag != 0 {
                nzcv |= bit;
            }
        }
        if f & flags::CF == 0 {
            nzcv |= C;
        }
        Ok(nzcv)
    }

    /// Copies the vCPU's registers and flags into `x` (not RIP).
    fn store(&self, x: &mut X86) -> Result<(), crate::Error> {
        let cpsr = self.cpsr()?;
        for i in 0..16 {
            x.regs[i] = self.cpu.reg(i as u8)?;
        }
        let pf_result = self.cpu.reg(PF_RESULT.0)?;
        let af_aux = self.cpu.reg(AF_AUX.0)?;
        let mut f = x.rflags & !flags::ARITH;
        for (bit, flag) in [(N, flags::SF), (Z, flags::ZF), (V, flags::OF)] {
            if cpsr & bit != 0 {
                f |= flag;
            }
        }
        if cpsr & C == 0 {
            f |= flags::CF;
        }
        if parity_even(pf_result) {
            f |= flags::PF;
        }
        if (pf_result ^ af_aux) & 0x10 != 0 {
            f |= flags::AF;
        }
        x.rflags = f;
        Ok(())
    }

    fn cpsr(&self) -> Result<u64, crate::Error> {
        self.cpu.vcpu.reg(hvf::reg::CPSR)
    }

    /// Enters the VM once, with the tables brought up to date first.
    fn enter(&mut self, mem: &Memory, entry: Entry) -> Result<Exit, crate::Error> {
        let (root, pool, flush) = loop {
            let mut m = machine()?;
            if m.sync(mem)? {
                let flush = self.epoch != Some(m.epoch);
                self.epoch = Some(m.epoch);
                let pool = m.active;
                m.users[pool] += 1;
                m.running[self.cpu.slot] = true;
                break (m.roots[pool], pool, flush);
            }
            drop(m);
            std::thread::yield_now();
        };
        let exit = match entry {
            Entry::Dispatch(nzcv) => {
                let at = if flush { ENTRY_FLUSH } else { DISPATCH };
                self.cpu.enter(root, at, None, nzcv)
            }
            Entry::Resume { pc, cpsr } => self.cpu.enter(root, RESUME_FLUSH, Some((pc, cpsr)), 0),
        };
        let mut m = machine()?;
        m.users[pool] -= 1;
        m.running[self.cpu.slot] = false;
        drop(m);
        let icount = self.cpu.reg(ICOUNT.0)?;
        TRANSLATED.fetch_add(icount.wrapping_sub(self.icount), Ordering::Relaxed);
        self.icount = icount;
        exit
    }

    fn run(
        &mut self,
        x: &mut X86,
        mem: &Memory,
        stop: &dyn Fn() -> bool,
    ) -> Result<x86::Exit, crate::Error> {
        let mut entry = Entry::Dispatch(self.load(x)?);
        loop {
            let exit = self.enter(mem, entry)?;
            let previous = entry;
            entry = Entry::Dispatch(self.cpsr()? & NZCV);
            let rip = match exit {
                Exit::Hvc(HVC_MISS) => {
                    let rip = self.cpu.reg(RIP.0)?;
                    if is_trap(mem, rip) {
                        self.store(x)?;
                        x.rip = rip;
                        return Ok(x86::Exit::Trap(rip));
                    }
                    let mut m = machine()?;
                    if m.translate(mem, rip) {
                        let verifying = m.verifying(rip);
                        drop(m);
                        match verifying {
                            Some((va, insns, true)) => {
                                entry = self.verify(x, mem, rip, va, insns)?
                            }
                            Some((va, _, false)) => {
                                let cpsr = CPSR_EL1H | (self.cpsr()? & NZCV);
                                entry = Entry::Resume { pc: va, cpsr };
                            }
                            None => {}
                        }
                        continue;
                    }
                    rip
                }
                Exit::Hvc(HVC_INTERP) => self.cpu.reg(RIP.0)?,
                Exit::Fault { elr: pc, .. } | Exit::Stage2 { pc } => {
                    let Some(rip) = machine()?.rip_at(pc) else {
                        panic!("the JIT faulted outside translated code at {pc:#x}");
                    };
                    rip
                }
                Exit::Canceled { pc, cpsr } => {
                    if stop() {
                        return Ok(x86::Exit::Limit);
                    }
                    entry = resume_after_cancel(previous, pc, cpsr);
                    continue;
                }
                Exit::Hvc(n) => panic!("unexpected hvc #{n:#x} from translated code"),
                Exit::Other => panic!("the JIT's VM stopped unexpectedly"),
            };
            self.store(x)?;
            x.rip = rip;
            let mut steps = 0;
            let mut stopped = None;
            while steps < INTERP_RUN {
                steps += 1;
                match x86::step(x, mem) {
                    x86::Exit::Continue => {}
                    other => {
                        stopped = Some(other);
                        break;
                    }
                }
                if crate::translate::starts_block(mem, x.rip) {
                    break;
                }
            }
            count_interpreted(steps as u64);
            if let Some(exit) = stopped {
                return Ok(exit);
            }
            if stop() {
                return Ok(x86::Exit::Limit);
            }
            entry = Entry::Dispatch(self.load(x)?);
        }
    }

    /// Verify mode: checks one run of the block at `rip` against the
    /// interpreter, and continues from the interpreter's state.
    fn verify(
        &mut self,
        x: &mut X86,
        mem: &Memory,
        rip: u64,
        va: u64,
        insns: usize,
    ) -> Result<Entry, crate::Error> {
        let nzcv = self.cpsr()? & NZCV;
        self.store(x)?;
        x.rip = rip;
        let mut want = x.clone();
        mem.start_journal();
        let comparable = (0..insns).all(|_| x86::step(&mut want, mem) == x86::Exit::Continue);
        let writes = mem.take_journal();
        for w in writes.iter().rev() {
            mem.write_host(w.addr, &w.old)
                .expect("the interpreter just wrote there");
        }
        let mut entry = Entry::Resume {
            pc: va,
            cpsr: CPSR_EL1H | nzcv,
        };
        let exit = loop {
            match self.enter(mem, entry)? {
                Exit::Canceled { pc, cpsr } => entry = resume_after_cancel(entry, pc, cpsr),
                other => break other,
            }
        };
        machine()?.verified(rip);
        if !comparable {
            return Ok(Entry::Dispatch(self.cpsr()? & NZCV));
        }
        let mut got = x.clone();
        self.store(&mut got)?;
        let mut diffs = Vec::new();
        match exit {
            Exit::Hvc(HVC_MISS | HVC_INTERP) => got.rip = self.cpu.reg(RIP.0)?,
            other => diffs.push(format!("the translated block stopped with {other:x?}")),
        }
        for i in 0..16 {
            if got.regs[i] != want.regs[i] {
                diffs.push(format!(
                    "r{i}: jit {:#x}, interpreter {:#x}",
                    got.regs[i], want.regs[i]
                ));
            }
        }
        if got.rip != want.rip {
            diffs.push(format!(
                "rip: jit {:#x}, interpreter {:#x}",
                got.rip, want.rip
            ));
        }
        let (gf, wf) = (got.rflags & flags::ARITH, want.rflags & flags::ARITH);
        if gf != wf {
            diffs.push(format!("flags: jit {gf:#x}, interpreter {wf:#x}"));
        }
        let mut last = BTreeMap::new();
        for w in &writes {
            for (k, &b) in w.new.iter().enumerate() {
                last.insert(w.addr + k as u64, b);
            }
        }
        for (&addr, &b) in &last {
            let mut real = [0u8];
            mem.read(addr, &mut real)
                .expect("the interpreter just wrote there");
            if real[0] != b {
                diffs.push(format!(
                    "mem[{addr:#x}]: jit {:#04x}, interpreter {b:#04x}",
                    real[0]
                ));
            }
        }
        if diffs.is_empty() {
            return Ok(Entry::Dispatch(self.cpsr()? & NZCV));
        }
        eprintln!("ps5slop: JIT verify: the block at {rip:#x} ({insns} instructions) differs:");
        for d in diffs {
            eprintln!("  {d}");
        }
        let mut code = [0u8; 64];
        if mem.fetch(rip, &mut code).is_ok() {
            let hex: Vec<String> = code.iter().map(|b| format!("{b:#04x}")).collect();
            eprintln!("  code: {}", hex.join(" "));
        }
        for (&addr, &b) in &last {
            mem.write_host(addr, &[b])
                .expect("the interpreter just wrote there");
        }
        *x = want;
        Ok(Entry::Dispatch(self.load(x)?))
    }
}

/// Where to go on after a cancel at `pc`. A cancel inside the resume stub
/// keeps the stub's own target, which ELR no longer holds.
fn resume_after_cancel(previous: Entry, pc: u64, cpsr: u64) -> Entry {
    let stub = crate::layout::va(RESUME_FLUSH)..crate::layout::va(crate::DISPATCH);
    match previous {
        Entry::Resume { .. } if stub.contains(&pc) => previous,
        _ => Entry::Resume { pc, cpsr },
    }
}

/// Runs the interpreter for up to `limit` instructions, counting them.
pub fn interpret(cpu: &mut X86, mem: &Memory, limit: u64) -> x86::Exit {
    let mut steps = 0;
    let exit = loop {
        if steps == limit {
            break x86::Exit::Limit;
        }
        steps += 1;
        match x86::step(cpu, mem) {
            x86::Exit::Continue => {}
            other => break other,
        }
    };
    count_interpreted(steps);
    exit
}

fn is_trap(mem: &Memory, rip: u64) -> bool {
    let mut b = [0u8];
    matches!(mem.fetch(rip, &mut b), Err(f) if f.kind == FaultKind::Trap)
}
