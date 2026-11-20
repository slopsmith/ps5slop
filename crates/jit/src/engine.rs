//! The run loop for one guest thread: enters the VM, handles its exits, and
//! falls back to the interpreter for what is not translated.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use memory::{FaultKind, Memory};
use x86::Cpu as X86;
use x86::cpu::flags;

use crate::translate::{AF_AUX, FS_BASE, HVC_HOST, HVC_INTERP, HVC_MISS, ICOUNT, PF_RESULT, RIP};
use crate::{CPSR_EL1H, Cpu, DISPATCH, ENTRY_FLUSH, Exit, RESUME_FLUSH, SHARED, machine};

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
    /// What the vCPU holds since the last `store`, so `load` sets only
    /// what changed: general registers, FS base, RFLAGS and MXCSR.
    shadow: Option<Shadow>,
    /// Whether FPSR may hold flags `store` already folded into MXCSR.
    fpsr_dirty: bool,
    /// Whether FPCR may not be what MXCSR gives (a fault in x87 code).
    fpcr_dirty: bool,
}

#[derive(Clone, Copy)]
struct Shadow {
    regs: [u64; 16],
    fs_base: u64,
    rflags: u64,
    mxcsr: u32,
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

/// Exits counted by statistics mode.
#[derive(Clone, Copy)]
pub(crate) enum Count {
    Miss,
    Interp,
    HostCall,
    Fault,
    Cancel,
    /// Lookup misses at a block already translated (a table collision).
    Reenter,
    /// Lookup misses at an address whose first instruction is not translated.
    NoBlock,
    /// TTBR0 table rebuilds after memory changes.
    Rebuild,
    /// VM entries that had to take the machine's lock.
    SlowEntry,
}

/// Statistics mode (`PS5SLOP_JIT_STATS`): what the interpreter runs, by
/// opcode and address, and why the VM is left.
#[derive(Default)]
struct Stats {
    opcodes: HashMap<(u8, u8, u8), u64>,
    rips: HashMap<u64, u64>,
    exits: [u64; 9],
}

fn stats() -> Option<&'static Mutex<Stats>> {
    static STATS: OnceLock<Option<Mutex<Stats>>> = OnceLock::new();
    STATS
        .get_or_init(|| std::env::var_os("PS5SLOP_JIT_STATS").map(|_| Mutex::default()))
        .as_ref()
}

pub(crate) fn count_exit(c: Count) {
    if let Some(s) = stats() {
        s.lock().unwrap_or_else(|e| e.into_inner()).exits[c as usize] += 1;
    }
}

/// In statistics mode, counts the instruction the interpreter is about to run.
fn count_insn(x: &X86, mem: &Memory) {
    let Some(s) = stats() else { return };
    let insn = x86::decode::decode(|i| {
        let mut b = [0u8];
        mem.fetch(x.rip.wrapping_add(i as u64), &mut b)
            .map(|_| b[0])
    });
    let mut s = s.lock().unwrap_or_else(|e| e.into_inner());
    if let Ok(i) = insn {
        let ext = i.modrm.map_or(0xff, |_| i.ext());
        *s.opcodes.entry((i.map as u8, i.op, ext)).or_default() += 1;
    }
    *s.rips.entry(x.rip).or_default() += 1;
}

/// Prints what statistics mode gathered, if it is on.
pub fn report(mem: &Memory) {
    let Some(s) = stats() else { return };
    let s = s.lock().unwrap_or_else(|e| e.into_inner());
    let (interp, jit) = instructions();
    let top = std::env::var("PS5SLOP_JIT_STATS_TOP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    eprintln!("ps5slop: JIT statistics: {jit} instructions translated, {interp} interpreted");
    let names = [
        "lookup misses",
        "untranslated",
        "host calls",
        "faults",
        "cancels",
        "misses at translated blocks",
        "misses at untranslatable code",
        "table rebuilds",
        "locked entries",
    ];
    let exits: Vec<String> = names
        .iter()
        .zip(s.exits)
        .map(|(n, c)| format!("{n} {c}"))
        .collect();
    eprintln!("  exits: {}", exits.join(", "));
    let mut ops: Vec<_> = s.opcodes.iter().collect();
    ops.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("  interpreted opcodes (map, opcode, /ext):");
    for ((map, op, ext), n) in ops.iter().take(top.max(30)) {
        let ext = if *ext == 0xff {
            String::new()
        } else {
            format!(" /{ext}")
        };
        eprintln!("    {n:>12}  map {map} {op:02x}{ext}");
    }
    let mut rips: Vec<_> = s.rips.iter().collect();
    rips.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("  interpreted addresses:");
    for (rip, n) in rips.iter().take(top) {
        let mut code = [0u8; 15];
        let bytes = match mem.fetch(**rip, &mut code) {
            Ok(()) => code
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" "),
            Err(_) => String::new(),
        };
        eprintln!("    {n:>12}  {rip:#x}  {bytes}");
    }
}

fn count_interpreted(n: u64) {
    INTERPRETED.fetch_add(n, Ordering::Relaxed);
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Ok(mut m) = machine() {
            m.vcpus[self.cpu.slot] = None;
        }
        SHARED.running[self.cpu.slot].store(false, Ordering::SeqCst);
    }
}

enum Slot {
    Untried,
    Unavailable,
    Ready(Box<Engine>),
}

thread_local! {
    static THREAD: RefCell<Slot> = const { RefCell::new(Slot::Untried) };
    static RECENT: RefCell<std::collections::VecDeque<u64>> = const { RefCell::new(std::collections::VecDeque::new()) };
}

/// Blocks this thread last ran, oldest first: in trace mode
/// (`PS5SLOP_JIT_TRACE`) every block start, otherwise only those the host saw.
pub fn recent_blocks() -> Vec<u64> {
    RECENT.with(|r| r.borrow().iter().copied().collect())
}

fn note_block(rip: u64) {
    RECENT.with(|r| {
        let mut r = r.borrow_mut();
        if r.len() == 64 {
            r.pop_front();
        }
        r.push_back(rip);
    });
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
                    shadow: None,
                    fpsr_dirty: true,
                    fpcr_dirty: true,
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

/// FPCR for MXCSR `mxcsr`: AH and NEP give x86's NaN, min/max and scalar
/// rules (86slopper's `docs/sse-neon.md`); DAZ, FTZ and RC map across.
fn fpcr(mxcsr: u32) -> u64 {
    const AH: u64 = 1 << 1;
    const NEP: u64 = 1 << 2;
    let daz = (mxcsr >> 6 & 1) as u64;
    let ftz = (mxcsr >> 15 & 1) as u64;
    let rc = (mxcsr >> 13 & 3) as u64;
    let rmode = (rc & 1) << 1 | rc >> 1;
    AH | NEP | daz | ftz << 24 | rmode << 22
}

/// MXCSR exception flags for FPSR's cumulative flags.
fn mxcsr_flags(fpsr: u64) -> u32 {
    let bit = |n: u32| (fpsr >> n & 1) as u32;
    bit(0) | bit(7) << 1 | bit(1) << 2 | bit(2) << 3 | bit(3) << 4 | bit(4) << 5
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
    fn load(&mut self, x: &X86) -> Result<u64, crate::Error> {
        let old = self.shadow.take();
        for (i, &v) in x.regs.iter().enumerate() {
            if old.is_none_or(|o| o.regs[i] != v) {
                self.cpu.set_reg(i as u8, v)?;
            }
        }
        self.cpu.set_reg(RIP.0, x.rip)?;
        if old.is_none_or(|o| o.fs_base != x.fs_base) {
            self.cpu.set_reg(FS_BASE.0, x.fs_base)?;
        }
        // SAFETY: the context block starts with the 16 YMM registers as
        // little-endian lanes, and only this thread's vCPU uses it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                x.xmm.as_ptr().cast::<u8>(),
                self.cpu.context,
                crate::layout::YMM_SIZE,
            )
        };
        let df = if x.rflags & flags::DF != 0 {
            u64::MAX
        } else {
            0
        };
        // SAFETY: the direction flag's slot is inside the same context block.
        unsafe { (self.cpu.context.add(crate::layout::DF_OFFSET) as *mut u64).write_unaligned(df) };
        if self.fpcr_dirty || old.is_none_or(|o| o.mxcsr != x.mxcsr) {
            let v = fpcr(x.mxcsr);
            self.cpu.vcpu.set_reg(hvf::reg::FPCR, v)?;
            self.put(crate::layout::FPCR_SSE, v);
            self.fpcr_dirty = false;
        }
        for (k, &v) in x.st.iter().enumerate() {
            self.put(crate::layout::X87_ST + 8 * k, v);
        }
        self.put(crate::layout::X87_FSW, x.fsw as u64);
        self.put(crate::layout::X87_FCW, x.fcw as u64);
        self.put(crate::layout::X87_FTAG, x.ftag as u64);
        if self.fpsr_dirty {
            self.cpu.vcpu.set_reg(hvf::reg::FPSR, 0)?;
            self.fpsr_dirty = false;
        }
        let f = x.rflags;
        if old.is_none_or(|o| (o.rflags ^ f) & (flags::PF | flags::AF) != 0) {
            let pf_result = if f & flags::PF != 0 { 0 } else { 1 };
            let af = if f & flags::AF != 0 { 0x10 } else { 0 };
            self.cpu.set_reg(PF_RESULT.0, pf_result)?;
            self.cpu.set_reg(AF_AUX.0, pf_result ^ af)?;
        }
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
    fn store(&mut self, x: &mut X86) -> Result<(), crate::Error> {
        let cpsr = self.cpsr()?;
        for i in 0..16 {
            x.regs[i] = self.cpu.reg(i as u8)?;
        }
        let fpsr = self.cpu.vcpu.reg(hvf::reg::FPSR)?;
        x.mxcsr |= mxcsr_flags(fpsr);
        self.fpsr_dirty = fpsr != 0;
        // SAFETY: as in `load`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.cpu.context,
                x.xmm.as_mut_ptr().cast::<u8>(),
                crate::layout::YMM_SIZE,
            )
        };
        // SAFETY: as in `load`.
        let df = unsafe {
            (self.cpu.context.add(crate::layout::DF_OFFSET) as *const u64).read_unaligned()
        };
        for k in 0..8 {
            x.st[k] = self.get(crate::layout::X87_ST + 8 * k);
        }
        x.fsw = self.get(crate::layout::X87_FSW) as u16;
        x.fcw = self.get(crate::layout::X87_FCW) as u16;
        x.ftag = self.get(crate::layout::X87_FTAG) as u8;
        let pf_result = self.cpu.reg(PF_RESULT.0)?;
        let af_aux = self.cpu.reg(AF_AUX.0)?;
        let mut f = x.rflags & !(flags::ARITH | flags::DF);
        if df != 0 {
            f |= flags::DF;
        }
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
        self.shadow = Some(Shadow {
            regs: x.regs,
            fs_base: x.fs_base,
            rflags: f,
            mxcsr: x.mxcsr,
        });
        Ok(())
    }

    /// Writes the 64-bit slot at `offset` in the vCPU's context block.
    fn put(&self, offset: usize, v: u64) {
        // SAFETY: callers pass offsets of slots inside the context block,
        // which only this thread's vCPU uses.
        unsafe { (self.cpu.context.add(offset) as *mut u64).write_unaligned(v) };
    }

    fn get(&self, offset: usize) -> u64 {
        // SAFETY: as in `put`.
        unsafe { (self.cpu.context.add(offset) as *const u64).read_unaligned() }
    }

    fn cpsr(&self) -> Result<u64, crate::Error> {
        self.cpu.vcpu.reg(hvf::reg::CPSR)
    }

    /// Enters the VM once, with the tables brought up to date first.
    fn enter(&mut self, mem: &Memory, entry: Entry) -> Result<Exit, crate::Error> {
        let sc = Ordering::SeqCst;
        self.shadow = None;
        SHARED.running[self.cpu.slot].store(true, sc);
        let (pool, flush) = loop {
            if let Some(pool) = SHARED.try_use(mem, self.epoch) {
                break (pool, false);
            }
            count_exit(Count::SlowEntry);
            let mut m = machine()?;
            if m.sync(mem)? {
                let epoch = SHARED.epoch.load(sc);
                let flush = self.epoch != Some(epoch);
                self.epoch = Some(epoch);
                let pool = SHARED.active.load(sc);
                SHARED.users[pool].fetch_add(1, sc);
                break (pool, flush);
            }
            drop(m);
            std::thread::yield_now();
        };
        let root = SHARED.roots[pool].load(sc);
        let exit = match entry {
            Entry::Dispatch(nzcv) => {
                let at = if flush { ENTRY_FLUSH } else { DISPATCH };
                self.cpu.enter(root, at, None, nzcv)
            }
            Entry::Resume { pc, cpsr } => self.cpu.enter(root, RESUME_FLUSH, Some((pc, cpsr)), 0),
        };
        SHARED.users[pool].fetch_sub(1, sc);
        SHARED.running[self.cpu.slot].store(false, sc);
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
                    note_block(rip);
                    count_exit(Count::Miss);
                    if is_trap(mem, rip) {
                        if machine()?.reenter_native(rip) {
                            continue;
                        }
                        count_exit(Count::HostCall);
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
                Exit::Hvc(HVC_HOST) => {
                    count_exit(Count::HostCall);
                    let rip = self.cpu.reg(RIP.0)?;
                    self.store(x)?;
                    x.rip = rip;
                    return Ok(x86::Exit::Trap(rip));
                }
                Exit::Hvc(HVC_INTERP) => {
                    count_exit(Count::Interp);
                    self.cpu.reg(RIP.0)?
                }
                Exit::Fault { elr: pc, .. } | Exit::Stage2 { pc } if crate::in_native(pc) => {
                    count_exit(Count::Fault);
                    let rip = self.cpu.reg(RIP.0)?;
                    self.store(x)?;
                    x.rip = rip;
                    return Ok(x86::Exit::Trap(rip));
                }
                Exit::Fault { elr: pc, .. } | Exit::Stage2 { pc } => {
                    count_exit(Count::Fault);
                    self.fpcr_dirty = true;
                    let Some(rip) = machine()?.rip_at(pc) else {
                        panic!("the JIT faulted outside translated code at {pc:#x}");
                    };
                    rip
                }
                Exit::Canceled { pc, cpsr } => {
                    count_exit(Count::Cancel);
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
                count_insn(x, mem);
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
        machine()?.verified(rip, mem);
        if !comparable {
            return Ok(Entry::Dispatch(self.cpsr()? & NZCV));
        }
        let mut got = x.clone();
        self.store(&mut got)?;
        let mut diffs = Vec::new();
        let mut finished_here = false;
        match exit {
            Exit::Hvc(HVC_MISS | HVC_INTERP) => got.rip = self.cpu.reg(RIP.0)?,
            other => diffs.push(format!("the translated block stopped with {other:x?}")),
        }
        if exit == Exit::Hvc(HVC_INTERP) && got.rip != want.rip {
            finished_here = true;
            for _ in 0..insns {
                if got.rip == want.rip || x86::step(&mut got, mem) != x86::Exit::Continue {
                    break;
                }
            }
        }
        for i in 0..16 {
            if got.regs[i] != want.regs[i] {
                diffs.push(format!(
                    "r{i}: jit {:#x}, interpreter {:#x}",
                    got.regs[i], want.regs[i]
                ));
            }
        }
        for n in 0..16 {
            if got.xmm[n] != want.xmm[n] {
                diffs.push(format!(
                    "ymm{n}: jit {:x?}, interpreter {:x?}",
                    got.xmm[n], want.xmm[n]
                ));
            }
        }
        if (got.fsw, got.fcw, got.ftag, got.st) != (want.fsw, want.fcw, want.ftag, want.st) {
            diffs.push(format!(
                "x87: jit fsw {:#x} fcw {:#x} tags {:#x} st {:x?}, interpreter fsw {:#x} fcw {:#x} tags {:#x} st {:x?}",
                got.fsw, got.fcw, got.ftag, got.st, want.fsw, want.fcw, want.ftag, want.st
            ));
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
            if finished_here {
                *x = got;
                return Ok(Entry::Dispatch(self.load(x)?));
            }
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
