//! Runs x86-64 user code as AArch64 at EL1 inside the process's
//! Hypervisor.framework VM, with guest memory mapped at its own addresses.
//! See `docs/jit.md`.

mod engine;
pub mod layout;
mod tables;
mod translate;

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use a64::{Asm, Reg, Shift, Size, Width};
use hvf::{GuestMemory, Vcpu, VcpuKicker, Vm, reg, sysreg};
use memory::Memory;

pub use engine::{instructions, interpret, kick_all, report, run};
pub use hvf::Error;

/// `hvc` numbers the runtime's exception vectors use.
pub const HVC_SYNC: u16 = 0xf000;
pub const HVC_OTHER: u16 = 0xf001;

/// MAIR attribute 0: normal memory, write-back.
const MAIR: u64 = 0xff;
/// T0SZ 24 (40-bit guest addresses), T1SZ 25, 4 KiB granules, inner
/// shareable write-back walks, 40-bit IPS.
const TCR: u64 =
    24 | 1 << 8 | 1 << 10 | 3 << 12 | 25 << 16 | 1 << 24 | 1 << 26 | 3 << 28 | 2 << 30 | 2 << 32;
/// MMU and caches on (86slopper's value, with RES1 bits).
const SCTLR: u64 = 0x30d0_0800 | 1 | 1 << 2 | 1 << 12;
/// EL1h with interrupts masked.
const CPSR_EL1H: u64 = 0x3c5;
/// `ACTLR_EL1.EnTSO`.
const TSO: u64 = 2;
/// 2^64 / golden ratio, for the lookup table's Fibonacci hash.
const HASH_MUL: u64 = 0x9e37_79b9_7f4a_7c15;
const EMPTY: u64 = u64::MAX;

/// Runtime code: flush then dispatch, flush then return to an interrupted
/// PC, the dispatcher, and free space for hand-written test code.
const ENTRY_FLUSH: usize = layout::CODE;
const RESUME_FLUSH: usize = layout::CODE + 0x40;
const DISPATCH: usize = layout::CODE + 0x80;
const NATIVE_MEMSET: usize = layout::CODE + 0x200;
const NATIVE_MEMMOVE: usize = layout::CODE + 0x400;
const NATIVE_END: usize = layout::CODE + 0x800;
const TEST_CODE: usize = layout::CODE + 0x1000;

/// Library functions with a native version inside the VM, which guest calls
/// reach without leaving it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Native {
    Memset,
    /// `memmove`, which also serves `memcpy`.
    Memmove,
}

/// Makes calls to the host function at trap address `addr` run `native`
/// inside the VM (not in verify mode, which checks the host versions).
pub fn set_native(addr: u64, native: Native) {
    if let Ok(mut m) = machine()
        && !m.verify
    {
        let place = match native {
            Native::Memset => NATIVE_MEMSET,
            Native::Memmove => NATIVE_MEMMOVE,
        };
        m.natives.retain(|&(a, _)| a != addr);
        m.natives.push((addr, place));
        m.enter(addr, place);
    }
}

/// Runs of each block verify mode checks against the interpreter. Verify
/// mode keeps the lookup table empty, so every block exits on its own.
const VERIFY_RUNS: u32 = 4;

struct BlockInfo {
    place: usize,
    insns: usize,
    verified: u32,
}

/// The process's VM and its runtime region. HVF allows one VM per process.
pub struct Machine {
    vm: Vm,
    runtime: GuestMemory,
    /// Guest ranges currently mapped at stage 2.
    mapped: Vec<(u64, u64)>,
    /// The address space (its id) and region generation last mirrored.
    generation: Option<(usize, u64)>,
    roots: [usize; 2],
    active: usize,
    /// Running vCPUs that entered with each table pool.
    users: [u32; 2],
    /// Bumped when the tables change, so vCPUs flush their TLBs.
    epoch: u64,
    vcpus: [Option<VcpuKicker>; layout::MAX_VCPUS],
    running: [bool; layout::MAX_VCPUS],
    code_next: usize,
    blocks: HashMap<u64, BlockInfo>,
    /// Check blocks against the interpreter before trusting them.
    verify: bool,
    /// Translated blocks by runtime offset: their guest instructions, as
    /// (word offset, RIP).
    by_place: BTreeMap<usize, Vec<(u32, u64)>>,
    cache_full: bool,
    /// Branches waiting for the block at a RIP to exist: runtime offsets.
    chain_wanted: HashMap<u64, Vec<usize>>,
    /// Trap addresses served by native library functions in the VM.
    natives: Vec<(u64, usize)>,
}

// SAFETY: the VM is process-wide and HVF's VM calls may come from any thread;
// the mutex around `Machine` serialises them.
unsafe impl Send for Machine {}

static MACHINE: OnceLock<Result<Mutex<Machine>, Error>> = OnceLock::new();

/// The process's machine, created on first use.
pub fn machine() -> Result<MutexGuard<'static, Machine>, Error> {
    let m = MACHINE.get_or_init(|| Machine::new().map(Mutex::new));
    match m {
        Ok(m) => Ok(m.lock().unwrap_or_else(|e| e.into_inner())),
        Err(e) => Err(*e),
    }
}

/// What made a vCPU leave the VM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// An `hvc` below [`HVC_SYNC`], with its number.
    Hvc(u16),
    /// A synchronous exception at EL1, such as a bad guest access.
    Fault {
        esr: u64,
        far: u64,
        elr: u64,
    },
    /// A stage-2 fault at `pc`: guest memory changed under the vCPU.
    Stage2 {
        pc: u64,
    },
    /// `hv_vcpus_exit`: resumable at `pc` with `cpsr`.
    Canceled {
        pc: u64,
        cpsr: u64,
    },
    Other,
}

unsafe extern "C" {
    fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
}

impl Machine {
    fn new() -> Result<Self, Error> {
        let vm = Vm::new(layout::IPA_BITS)?;
        let mut runtime = GuestMemory::new(layout::RUNTIME_SIZE).map_err(|_| Error(-1))?;
        let rt = runtime.as_mut_slice();
        tables::map_runtime(rt);
        write_runtime(rt);
        for e in 0..layout::LOOKUP_ENTRIES {
            let at = layout::LOOKUP + e * 16;
            rt[at..at + 8].copy_from_slice(&EMPTY.to_le_bytes());
        }
        // SAFETY: `runtime` lives as long as the machine, which is never dropped.
        unsafe {
            vm.map(
                runtime.as_ptr(),
                layout::RUNTIME_IPA,
                layout::RUNTIME_SIZE,
                hvf::mem::RWX,
            )?
        };
        let mut m = Self {
            vm,
            runtime,
            mapped: Vec::new(),
            generation: None,
            roots: [0; 2],
            active: 0,
            users: [0; 2],
            epoch: 0,
            vcpus: [None; layout::MAX_VCPUS],
            running: [false; layout::MAX_VCPUS],
            code_next: layout::CODE_CACHE,
            blocks: HashMap::new(),
            verify: std::env::var_os("PS5SLOP_JIT_VERIFY").is_some(),
            by_place: BTreeMap::new(),
            cache_full: false,
            chain_wanted: HashMap::new(),
            natives: Vec::new(),
        };
        for pool in 0..2 {
            m.roots[pool] = tables::Tables::new(m.runtime.as_mut_slice(), pool).root;
        }
        Ok(m)
    }

    /// Mirrors `mem`'s regions into the VM if they changed: stage-2
    /// mappings, and TTBR0 tables rebuilt in the pool no running vCPU uses.
    /// False if that pool is still in use, so the caller should retry.
    pub fn sync(&mut self, mem: &Memory) -> Result<bool, Error> {
        let generation = (mem.id() as usize, mem.generation());
        if self.generation == Some(generation) {
            return Ok(true);
        }
        if self
            .generation
            .is_some_and(|(base, _)| base != generation.0)
        {
            for (start, end) in self.mapped.drain(..) {
                self.vm.unmap(start, (end - start) as usize)?;
            }
            self.forget_code();
        }
        let pool = 1 - self.active;
        if self.users[pool] > 0 {
            return Ok(false);
        }
        let regions: Vec<_> = mem.regions().into_iter().filter(|r| !r.trap).collect();
        let wanted: Vec<(u64, u64)> = regions.iter().map(|r| (r.start, r.end)).collect();
        let fresh = mem.take_fresh();
        let stale = |&(s, e): &(u64, u64)| fresh.iter().any(|&(fs, fe)| fs < e && s < fe);
        let (old, kept): (Vec<_>, Vec<_>) = self.mapped.drain(..).partition(stale);
        for (start, end) in old {
            self.vm.unmap(start, (end - start) as usize)?;
        }
        self.mapped = kept;
        for &(start, end) in self.mapped.iter().filter(|m| !wanted.contains(m)) {
            self.vm.unmap(start, (end - start) as usize)?;
        }
        for &(start, end) in wanted.iter().filter(|w| !self.mapped.contains(w)) {
            // SAFETY: the region is mapped host memory inside `mem`'s
            // reservation, and is unmapped here before it goes away.
            unsafe {
                let host = mem.host_base().add(start as usize);
                self.vm
                    .map(host, start, (end - start) as usize, hvf::mem::RWX)?
            };
        }
        self.mapped = wanted;
        let mut t = tables::Tables::new(self.runtime.as_mut_slice(), pool);
        for r in &regions {
            let read = r.prot.allows(memory::Access::Read);
            let write = r.prot.allows(memory::Access::Write);
            if (read || write) && !t.map(r.start, r.end, write) {
                return Err(Error(-2));
            }
        }
        self.roots[pool] = t.root;
        self.active = pool;
        self.epoch += 1;
        self.generation = Some(generation);
        self.kick_running();
        Ok(true)
    }

    /// Drops every translation, for a new address space whose code may
    /// differ at the same addresses. Native functions stay.
    fn forget_code(&mut self) {
        self.blocks.clear();
        self.by_place.clear();
        self.chain_wanted.clear();
        self.code_next = layout::CODE_CACHE;
        self.cache_full = false;
        let rt = self.runtime.as_mut_slice();
        for e in 0..layout::LOOKUP_ENTRIES {
            let at = layout::LOOKUP + e * 16;
            rt[at..at + 8].copy_from_slice(&EMPTY.to_le_bytes());
        }
        if !self.verify {
            for (addr, place) in self.natives.clone() {
                self.enter(addr, place);
            }
        }
    }

    fn kick_running(&self) {
        for (slot, k) in self.vcpus.iter().enumerate() {
            if let Some(k) = k
                && self.running[slot]
            {
                k.kick();
            }
        }
    }

    /// Translates the block at `rip` and enters it in the lookup table;
    /// false if it cannot be translated or the cache is full.
    fn translate(&mut self, mem: &Memory, rip: u64) -> bool {
        if let Some(b) = self.blocks.get(&rip) {
            if !self.verify {
                let place = b.place;
                self.enter(rip, place);
            }
            return true;
        }
        if self.cache_full {
            return false;
        }
        let place = self.code_next;
        let Some(block) = translate::block(mem, rip, place, DISPATCH) else {
            return false;
        };
        let len = block.code.len() * 4;
        if place + len > layout::CODE_CACHE_END {
            self.cache_full = true;
            eprintln!("ps5slop: the JIT's code cache is full; interpreting new code from now on");
            return false;
        }
        self.runtime.write_words(place, &block.code);
        // SAFETY: the range was just written inside the runtime region.
        unsafe { sys_icache_invalidate(self.runtime.as_ptr().add(place).cast(), len) };
        self.code_next = (place + len).next_multiple_of(16);
        let insns = block.insns.len();
        self.blocks.insert(
            rip,
            BlockInfo {
                place,
                insns,
                verified: 0,
            },
        );
        self.by_place.insert(place, block.insns);
        if !self.verify {
            self.enter(rip, place);
            for (word, target) in block.chains {
                let site = place + word as usize * 4;
                match self.blocks.get(&target) {
                    Some(b) => {
                        let to = b.place;
                        self.patch(site, to);
                    }
                    None => self.chain_wanted.entry(target).or_default().push(site),
                }
            }
            for site in self.chain_wanted.remove(&rip).unwrap_or_default() {
                self.patch(site, place);
            }
        }
        true
    }

    /// Turns the branch to the dispatcher at runtime offset `site` into a
    /// branch straight to the block at `to`.
    fn patch(&mut self, site: usize, to: usize) {
        let delta = (to as i64 - site as i64) / 4;
        self.runtime
            .write_words(site, &[0x1400_0000 | (delta as u32 & 0x03ff_ffff)]);
        // SAFETY: one word inside the runtime region.
        unsafe { sys_icache_invalidate(self.runtime.as_ptr().add(site).cast(), 4) };
    }

    /// Points `rip`'s lookup slot at `place`. The tag is cleared first and
    /// written last, so no tag is ever paired with another block's code.
    fn enter(&mut self, rip: u64, place: usize) {
        let slot = (rip.wrapping_mul(HASH_MUL) >> (64 - layout::LOOKUP_BITS)) as usize;
        let at = layout::LOOKUP + slot * 16;
        // SAFETY: both words are inside the runtime region and 8-aligned.
        let (tag, code) = unsafe {
            let base = self.runtime.as_ptr();
            (
                &*(base.add(at) as *const AtomicU64),
                &*(base.add(at + 8) as *const AtomicU64),
            )
        };
        tag.store(EMPTY, Ordering::Release);
        code.store(layout::va(place), Ordering::Release);
        tag.store(rip, Ordering::Release);
    }

    /// In verify mode, the block at `rip`: its virtual address, how many guest
    /// instructions it holds, and whether it still has to be checked.
    fn verifying(&self, rip: u64) -> Option<(u64, usize, bool)> {
        let b = self.blocks.get(&rip).filter(|_| self.verify)?;
        Some((layout::va(b.place), b.insns, b.verified < VERIFY_RUNS))
    }

    /// Counts one checked run of the block at `rip`.
    fn verified(&mut self, rip: u64) {
        if let Some(b) = self.blocks.get_mut(&rip) {
            b.verified += 1;
        }
    }

    /// The guest RIP of the instruction whose translated code holds the
    /// virtual address `pc`.
    fn rip_at(&self, pc: u64) -> Option<u64> {
        let off = pc.checked_sub(layout::RUNTIME_VA)? as usize;
        let (&place, insns) = self.by_place.range(..=off).next_back()?;
        let word = ((off - place) / 4) as u32;
        insns
            .iter()
            .rev()
            .find(|(w, _)| *w <= word)
            .map(|&(_, rip)| rip)
    }

    /// Copies hand-written `code` into the runtime's free code space at
    /// `offset` and returns its virtual address.
    pub fn load_code(&mut self, offset: usize, code: &[u32]) -> u64 {
        let at = TEST_CODE + offset;
        assert!(at + code.len() * 4 <= layout::CODE_END);
        self.runtime.write_words(at, code);
        layout::va(at)
    }
}

fn write_runtime(rt: &mut [u8]) {
    let put = |rt: &mut [u8], at: usize, code: Vec<u32>| {
        for (i, w) in code.iter().enumerate() {
            rt[at + i * 4..at + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
    };
    for slot in 0..16 {
        let mut a = Asm::new();
        a.hvc(if slot == 4 { HVC_SYNC } else { HVC_OTHER });
        put(rt, layout::VECTORS + slot * 0x80, a.finish());
    }
    let flush = |a: &mut Asm| {
        a.tlbi_vmalle1();
        a.dsb(0xb);
        a.ic_iallu();
        a.dsb(0xb);
        a.isb();
    };
    let mut a = Asm::new();
    flush(&mut a);
    a.b_offset(((DISPATCH - (ENTRY_FLUSH + a.pos() * 4)) / 4) as i32);
    put(rt, ENTRY_FLUSH, a.finish());
    let mut a = Asm::new();
    flush(&mut a);
    a.eret();
    put(rt, RESUME_FLUSH, a.finish());

    let (rip, slot, tag) = (translate::RIP, Reg(17), Reg(18));
    let mut a = Asm::new();
    let miss = a.label();
    a.mov_imm(Size::X, slot, HASH_MUL);
    a.mul(Size::X, slot, rip, slot);
    a.lsr_imm(Size::X, slot, slot, 64 - layout::LOOKUP_BITS);
    a.add_shift(Size::X, slot, translate::LOOKUP, slot, Shift::Lsl, 4);
    a.ldr(Width::X, tag, slot, 0);
    a.ldr(Width::X, slot, slot, 8);
    a.eor(Size::X, tag, tag, rip);
    a.cbnz(Size::X, tag, miss);
    a.br(slot);
    a.bind(miss);
    a.hvc(translate::HVC_MISS);
    let code = a.finish();
    assert!(DISPATCH + code.len() * 4 <= NATIVE_MEMSET);
    put(rt, DISPATCH, code);
    let code = native_memset();
    assert!(NATIVE_MEMSET + code.len() * 4 <= NATIVE_MEMMOVE);
    put(rt, NATIVE_MEMSET, code);
    let code = native_memmove();
    assert!(NATIVE_MEMMOVE + code.len() * 4 <= NATIVE_END);
    put(rt, NATIVE_MEMMOVE, code);
}

/// Returns from a native library function as the guest's `ret` would, with
/// RAX = RDI (what `memset` and `memmove` return), and dispatches.
fn native_return(a: &mut Asm, at: usize) {
    a.mov(Size::X, Reg(0), Reg(7));
    a.ldr(Width::X, translate::RIP, Reg(4), 0);
    a.add_imm(Size::X, Reg(4), Reg(4), 8);
    a.b_offset(((DISPATCH as i64 - (at + a.pos() * 4) as i64) / 4) as i32);
}

/// `memset(rdi, sil, rdx)` on guest registers, 16 bytes at a time.
fn native_memset() -> Vec<u32> {
    let (p, n, v, k) = (Reg(19), Reg(20), Reg(21), Reg(22));
    let mut a = Asm::new();
    let (wide, tail, done) = (a.label(), a.label(), a.label());
    a.mov(Size::X, p, Reg(7));
    a.mov(Size::X, n, Reg(2));
    a.uxtb(v, Reg(6));
    a.mov_imm(Size::X, k, 0x0101_0101_0101_0101);
    a.mul(Size::X, v, v, k);
    a.bind(wide);
    a.cmp_imm(Size::X, n, 16);
    a.b_cond(a64::Cond::Lo, tail);
    a.stp(Size::X, v, v, p, 0);
    a.add_imm(Size::X, p, p, 16);
    a.sub_imm(Size::X, n, n, 16);
    a.b(wide);
    a.bind(tail);
    a.cbz(Size::X, n, done);
    a.str(Width::B, v, p, 0);
    a.add_imm(Size::X, p, p, 1);
    a.sub_imm(Size::X, n, n, 1);
    a.b(tail);
    a.bind(done);
    native_return(&mut a, NATIVE_MEMSET);
    a.finish()
}

/// `memmove(rdi, rsi, rdx)` on guest registers: forward unless the
/// destination starts inside the source, 16 bytes at a time.
fn native_memmove() -> Vec<u32> {
    let (d, s, n, x, y) = (Reg(19), Reg(18), Reg(20), Reg(21), Reg(22));
    let mut a = Asm::new();
    let (fwd, fwd1, back, back16, back1, done) = (
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
        a.label(),
    );
    a.mov(Size::X, d, Reg(7));
    a.mov(Size::X, s, Reg(6));
    a.mov(Size::X, n, Reg(2));
    a.sub(Size::X, x, d, s);
    a.cmp(Size::X, x, n);
    a.b_cond(a64::Cond::Lo, back);
    a.bind(fwd);
    a.cmp_imm(Size::X, n, 16);
    a.b_cond(a64::Cond::Lo, fwd1);
    a.ldp(Size::X, x, y, s, 0);
    a.stp(Size::X, x, y, d, 0);
    a.add_imm(Size::X, s, s, 16);
    a.add_imm(Size::X, d, d, 16);
    a.sub_imm(Size::X, n, n, 16);
    a.b(fwd);
    a.bind(fwd1);
    a.cbz(Size::X, n, done);
    a.ldr(Width::B, x, s, 0);
    a.str(Width::B, x, d, 0);
    a.add_imm(Size::X, s, s, 1);
    a.add_imm(Size::X, d, d, 1);
    a.sub_imm(Size::X, n, n, 1);
    a.b(fwd1);
    a.bind(back);
    a.add(Size::X, s, s, n);
    a.add(Size::X, d, d, n);
    a.bind(back16);
    a.cmp_imm(Size::X, n, 16);
    a.b_cond(a64::Cond::Lo, back1);
    a.sub_imm(Size::X, s, s, 16);
    a.sub_imm(Size::X, d, d, 16);
    a.ldp(Size::X, x, y, s, 0);
    a.stp(Size::X, x, y, d, 0);
    a.sub_imm(Size::X, n, n, 16);
    a.b(back16);
    a.bind(back1);
    a.cbz(Size::X, n, done);
    a.sub_imm(Size::X, s, s, 1);
    a.sub_imm(Size::X, d, d, 1);
    a.ldr(Width::B, x, s, 0);
    a.str(Width::B, x, d, 0);
    a.sub_imm(Size::X, n, n, 1);
    a.b(back1);
    a.bind(done);
    native_return(&mut a, NATIVE_MEMMOVE);
    a.finish()
}

/// Whether runtime virtual address `pc` is inside a native library function.
fn in_native(pc: u64) -> bool {
    (layout::va(NATIVE_MEMSET)..layout::va(NATIVE_END)).contains(&pc)
}

/// A vCPU, usable only on the thread that made it.
pub struct Cpu {
    vcpu: Vcpu,
    slot: usize,
    /// The host address of this vCPU's context block.
    context: *mut u8,
}

impl Cpu {
    /// Makes a vCPU with the runtime's system registers, or `None` when the
    /// VM has no vCPUs left.
    pub fn new(m: &mut Machine) -> Result<Option<Cpu>, Error> {
        let Some(slot) = m.vcpus.iter().position(|v| v.is_none()) else {
            return Ok(None);
        };
        let vcpu = Vcpu::new(&m.vm)?;
        vcpu.set_sys_reg(sysreg::MAIR_EL1, MAIR)?;
        vcpu.set_sys_reg(sysreg::TCR_EL1, TCR)?;
        vcpu.set_sys_reg(sysreg::TTBR1_EL1, layout::ipa(layout::TTBR1_L1))?;
        vcpu.set_sys_reg(sysreg::VBAR_EL1, layout::va(layout::VECTORS))?;
        let stack = layout::STACKS + (slot + 1) * layout::STACK_SIZE;
        vcpu.set_sys_reg(sysreg::SP_EL1, layout::va(stack))?;
        vcpu.set_sys_reg(sysreg::CPACR_EL1, 3 << 20)?;
        vcpu.set_sys_reg(sysreg::SCTLR_EL1, SCTLR)?;
        vcpu.set_sys_reg(sysreg::ACTLR_EL1, TSO)?;
        vcpu.set_reg(
            reg::X0 + translate::LOOKUP.0 as u32,
            layout::va(layout::LOOKUP),
        )?;
        let context = layout::CONTEXTS + slot * layout::CONTEXT_SIZE;
        vcpu.set_reg(reg::X0 + translate::CTX.0 as u32, layout::va(context))?;
        m.vcpus[slot] = Some(vcpu.kicker());
        // SAFETY: the context block is inside the runtime region, which lives as
        // long as the process.
        let context = unsafe { m.runtime.as_ptr().add(context) };
        Ok(Some(Cpu {
            vcpu,
            slot,
            context,
        }))
    }

    pub fn reg(&self, n: u8) -> Result<u64, Error> {
        self.vcpu.reg(reg::X0 + n as u32)
    }

    pub fn set_reg(&self, n: u8, v: u64) -> Result<(), Error> {
        self.vcpu.set_reg(reg::X0 + n as u32, v)
    }

    /// Whether the vCPU orders memory accesses as x86 does.
    pub fn tso(&self) -> Result<bool, Error> {
        Ok(self.vcpu.sys_reg(sysreg::ACTLR_EL1)? & TSO != 0)
    }

    /// Runs hand-written code at virtual address `pc`, after a TLB flush,
    /// until it leaves the VM.
    pub fn run(&mut self, m: &Machine, pc: u64) -> Result<Exit, Error> {
        self.enter(m.roots[m.active], RESUME_FLUSH, Some((pc, CPSR_EL1H)), 0)
    }

    /// Enters the VM at runtime offset `at` with TTBR0 at `root` and NZCV
    /// `nzcv`; `resume` sets ELR and SPSR for an `eret`.
    fn enter(
        &mut self,
        root: usize,
        at: usize,
        resume: Option<(u64, u64)>,
        nzcv: u64,
    ) -> Result<Exit, Error> {
        self.vcpu
            .set_sys_reg(sysreg::TTBR0_EL1, layout::ipa(root))?;
        if let Some((pc, cpsr)) = resume {
            self.vcpu.set_sys_reg(sysreg::ELR_EL1, pc)?;
            self.vcpu.set_sys_reg(sysreg::SPSR_EL1, cpsr)?;
        }
        self.vcpu.set_reg(reg::CPSR, CPSR_EL1H | nzcv)?;
        self.vcpu.set_reg(reg::PC, layout::va(at))?;
        let syndrome = match self.vcpu.run()? {
            hvf::Exit::Canceled => {
                let pc = self.vcpu.reg(reg::PC)?;
                let cpsr = self.vcpu.reg(reg::CPSR)?;
                return Ok(Exit::Canceled { pc, cpsr });
            }
            hvf::Exit::Exception { syndrome, .. } => syndrome,
            _ => return Ok(Exit::Other),
        };
        Ok(match syndrome >> 26 & 0x3f {
            0x16 => match syndrome as u16 {
                HVC_SYNC => Exit::Fault {
                    esr: self.vcpu.sys_reg(sysreg::ESR_EL1)?,
                    far: self.vcpu.sys_reg(sysreg::FAR_EL1)?,
                    elr: self.vcpu.sys_reg(sysreg::ELR_EL1)?,
                },
                HVC_OTHER => Exit::Other,
                n => Exit::Hvc(n),
            },
            0x20 | 0x24 => Exit::Stage2 {
                pc: self.vcpu.reg(reg::PC)?,
            },
            _ => Exit::Other,
        })
    }

    pub fn release(self, m: &mut Machine) {
        m.vcpus[self.slot] = None;
        m.running[self.slot] = false;
    }
}

#[cfg(test)]
mod tests;
