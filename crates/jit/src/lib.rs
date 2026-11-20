//! Runs x86-64 user code as AArch64 at EL1 inside the process's
//! Hypervisor.framework VM, with guest memory mapped at its own addresses.
//! See `docs/jit.md`.

mod engine;
pub mod layout;
mod tables;
mod translate;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use a64::{Asm, Reg, Shift, Size, Width};
use hvf::{GuestMemory, Vcpu, VcpuKicker, Vm, reg, sysreg};
use memory::Memory;

pub use engine::{instructions, interpret, kick_all, recent_blocks, report, run};
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
/// Leaves the VM as a lookup miss: where blocks still being checked in
/// verify mode exit, so each check runs exactly one block.
const VERIFY_EXIT: usize = layout::CODE + 0x1C0;
/// Hands the RIP in x16 to the interpreter: the lookup table's entry for
/// code whose first instruction is not translated.
const INTERP_EXIT: usize = layout::CODE + 0x1E0;
const NATIVE_MEMSET: usize = layout::CODE + 0x200;
const NATIVE_MEMMOVE: usize = layout::CODE + 0x400;
const NATIVE_SPECIFIC: usize = layout::CODE + 0x800;
const NATIVE_THREAD_ID: usize = layout::CODE + 0x880;
const NATIVE_SELF: usize = layout::CODE + 0x900;
const NATIVE_END: usize = layout::CODE + 0x980;
const TEST_CODE: usize = layout::CODE + 0x1000;

/// Library functions with a native version inside the VM, which guest calls
/// reach without leaving it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Native {
    Memset,
    /// `memmove`, which also serves `memcpy`.
    Memmove,
    /// `pthread_getspecific(key)` for keys whose values live at the thread
    /// pointer plus `specific` (the first `keys` of them); other keys go to
    /// the host function.
    Getspecific { specific: u32, keys: u32 },
    /// `scePthreadGetthreadid()`: the thread pointer shifted right 14 bits.
    ThreadId,
    /// `pthread_self()`: the thread pointer.
    SelfThread,
}

/// Trap addresses with a native function, readable without the machine lock.
static NATIVE_TRAPS: std::sync::RwLock<Vec<u64>> = std::sync::RwLock::new(Vec::new());

fn has_native(addr: u64) -> bool {
    NATIVE_TRAPS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains(&addr)
}

/// Makes calls to the host function at trap address `addr` run `native`
/// inside the VM (not in verify mode, which checks the host versions).
pub fn set_native(addr: u64, native: Native) {
    if let Ok(mut m) = machine()
        && !m.verify
        && !m.trace
    {
        let place = match native {
            Native::Memset => NATIVE_MEMSET,
            Native::Memmove => NATIVE_MEMMOVE,
            Native::Getspecific { specific, keys } => {
                let code = native_getspecific(specific, keys);
                assert!(NATIVE_SPECIFIC + code.len() * 4 <= NATIVE_THREAD_ID);
                m.runtime.write_words(NATIVE_SPECIFIC, &code);
                // SAFETY: the range was just written inside the runtime region.
                unsafe {
                    sys_icache_invalidate(m.runtime.as_ptr().add(NATIVE_SPECIFIC).cast(), code.len() * 4)
                };
                NATIVE_SPECIFIC
            }
            Native::ThreadId => NATIVE_THREAD_ID,
            Native::SelfThread => NATIVE_SELF,
        };
        m.natives.retain(|&(a, _)| a != addr);
        m.natives.push((addr, place));
        NATIVE_TRAPS
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(addr);
        m.enter(addr, place);
    }
}

/// Runs of each block verify mode checks against the interpreter before a
/// copy with ordinary exits replaces it.
const VERIFY_RUNS: u32 = 4;

/// How far verify mode runs the interpreter to check one run of translated
/// code: a block's instruction count, or while it stays in a region.
pub(crate) enum Steps {
    Insns(usize),
    Region(std::sync::Arc<HashSet<u64>>),
}

struct RegionInfo {
    place: usize,
    rips: std::sync::Arc<HashSet<u64>>,
    verified: u32,
}

struct BlockInfo {
    place: usize,
    insns: usize,
    verified: u32,
}

/// What a vCPU needs to enter and leave the VM without the machine's lock.
/// Only `Machine::sync`, under that lock, changes the tables and these.
struct Shared {
    /// The address space id and region generation the tables mirror; the
    /// generation is `u64::MAX` while the two change.
    id: AtomicU64,
    generation: AtomicU64,
    /// The table pool new entries use, and the root table of each pool.
    active: AtomicUsize,
    roots: [AtomicUsize; 2],
    /// vCPUs inside the VM with each pool.
    users: [AtomicU32; 2],
    /// Bumped when the tables change, so vCPUs flush their TLBs.
    epoch: AtomicU64,
    running: [AtomicBool; layout::MAX_VCPUS],
}

static SHARED: Shared = Shared {
    id: AtomicU64::new(0),
    generation: AtomicU64::new(u64::MAX),
    active: AtomicUsize::new(0),
    roots: [const { AtomicUsize::new(0) }; 2],
    users: [const { AtomicU32::new(0) }; 2],
    epoch: AtomicU64::new(0),
    running: [const { AtomicBool::new(false) }; layout::MAX_VCPUS],
};

impl Shared {
    /// Takes the active pool for an entry if the tables mirror `mem` and
    /// were last flushed at `epoch`; `None` means take the locked path.
    fn try_use(&self, mem: &Memory, epoch: Option<u64>) -> Option<usize> {
        let sc = Ordering::SeqCst;
        if self.generation.load(sc) != mem.generation()
            || self.id.load(sc) != mem.id()
            || Some(self.epoch.load(sc)) != epoch
        {
            return None;
        }
        let pool = self.active.load(sc);
        self.users[pool].fetch_add(1, sc);
        // `sync` bumps the epoch before it switches pools, so a pool that
        // is still active with the same epoch was not rebuilt meanwhile.
        if self.active.load(sc) == pool && Some(self.epoch.load(sc)) == epoch {
            return Some(pool);
        }
        self.users[pool].fetch_sub(1, sc);
        None
    }
}

/// The process's VM and its runtime region. HVF allows one VM per process.
pub struct Machine {
    vm: Vm,
    runtime: GuestMemory,
    /// Guest ranges currently mapped at stage 2.
    mapped: Vec<(u64, u64)>,
    /// The address space (its id) and region generation last mirrored.
    generation: Option<(usize, u64)>,
    /// The accessible ranges in the active tables, as (start, end, writable).
    mirrored: Vec<(u64, u64, bool)>,
    /// Where each table pool's free space starts.
    pool_next: [usize; 2],
    /// Ranges whose host pages were replaced since the last sync.
    fresh: Vec<(u64, u64)>,
    vcpus: [Option<VcpuKicker>; layout::MAX_VCPUS],
    code_next: usize,
    blocks: HashMap<u64, BlockInfo>,
    /// Check blocks against the interpreter before trusting them.
    verify: bool,
    /// Trace mode: every block exits to the host, which keeps the last ones
    /// run on each thread (see [`recent_blocks`]).
    trace: bool,
    /// Translated blocks by runtime offset: their guest instructions, as
    /// (word offset, RIP).
    by_place: BTreeMap<usize, Vec<(u32, u64)>>,
    cache_full: bool,
    /// Branches waiting for the block at a RIP to exist: runtime offsets.
    chain_wanted: HashMap<u64, Vec<usize>>,
    /// Trap addresses served by native library functions in the VM.
    natives: Vec<(u64, usize)>,
    /// Exits of the region `emit_region` made last, to chain.
    pending_chains: Vec<(u32, u64)>,
    tier2: translate::Tier2,
    /// Tier-2 regions by entry RIP.
    regions: HashMap<u64, RegionInfo>,
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
            mirrored: Vec::new(),
            pool_next: [0; 2],
            fresh: Vec::new(),
            vcpus: [None; layout::MAX_VCPUS],
            code_next: layout::CODE_CACHE,
            blocks: HashMap::new(),
            verify: std::env::var_os("PS5SLOP_JIT_VERIFY").is_some(),
            trace: std::env::var_os("PS5SLOP_JIT_TRACE").is_some(),
            by_place: BTreeMap::new(),
            cache_full: false,
            chain_wanted: HashMap::new(),
            natives: Vec::new(),
            pending_chains: Vec::new(),
            tier2: translate::Tier2::from_env(),
            regions: HashMap::new(),
        };
        for pool in 0..2 {
            let t = tables::Tables::new(m.runtime.as_mut_slice(), pool);
            m.pool_next[pool] = t.next();
            SHARED.roots[pool].store(t.root, Ordering::SeqCst);
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
        let same_space = self.generation.is_some_and(|(base, _)| base == generation.0);
        if self.generation.is_some() && !same_space {
            for (start, end) in self.mapped.drain(..) {
                self.vm.unmap(start, (end - start) as usize)?;
            }
            self.mirrored.clear();
            self.forget_code();
        }
        self.fresh.extend(mem.take_fresh());
        let regions: Vec<_> = mem
            .region_spans()
            .into_iter()
            .filter(|r| {
                !r.trap
                    && (r.prot.allows(memory::Access::Read)
                        || r.prot.allows(memory::Access::Write))
            })
            .collect();
        let access: Vec<(u64, u64, bool)> = regions
            .iter()
            .filter(|r| r.prot.allows(memory::Access::Read) || r.prot.allows(memory::Access::Write))
            .map(|r| (r.start, r.end, r.prot.allows(memory::Access::Write)))
            .collect();
        let sc = Ordering::SeqCst;
        let pool = 1 - SHARED.active.load(sc);
        let in_place = same_space && !self.reduces_access(&regions, &access);
        if !in_place && SHARED.users[pool].load(sc) > 0 {
            return Ok(false);
        }
        self.map_stage2(mem, &regions)?;
        if in_place && self.update_mappings(&access) {
            SHARED.generation.store(generation.1, sc);
            self.generation = Some(generation);
            return Ok(true);
        }
        if SHARED.users[pool].load(sc) > 0 {
            return Ok(false);
        }
        engine::count_exit(engine::Count::Rebuild);
        let mut t = tables::Tables::new(self.runtime.as_mut_slice(), pool);
        for &(start, end, write) in &access {
            if !t.map(start, end, write) {
                return Err(Error(-2));
            }
        }
        self.pool_next[pool] = t.next();
        self.mirrored = access;
        SHARED.roots[pool].store(t.root, sc);
        SHARED.epoch.fetch_add(1, sc);
        SHARED.active.store(pool, sc);
        SHARED.generation.store(u64::MAX, sc);
        SHARED.id.store(generation.0 as u64, sc);
        SHARED.generation.store(generation.1, sc);
        self.generation = Some(generation);
        self.kick_running();
        Ok(true)
    }

    /// Brings the stage-2 mappings in line with `regions`, mapping again the
    /// ranges whose host pages were replaced.
    fn map_stage2(&mut self, mem: &Memory, regions: &[memory::RegionSpan]) -> Result<(), Error> {
        let wanted: Vec<(u64, u64)> = regions.iter().map(|r| (r.start, r.end)).collect();
        let fresh = std::mem::take(&mut self.fresh);
        let stale = |&(s, e): &(u64, u64)| fresh.iter().any(|&(fs, fe)| fs < e && s < fe);
        let (old, kept): (Vec<_>, Vec<_>) = self.mapped.drain(..).partition(stale);
        for (start, end) in old {
            self.vm.unmap(start, (end - start) as usize)?;
        }
        self.mapped = kept;
        let wanted_set: HashSet<&(u64, u64)> = wanted.iter().collect();
        let mapped_set: HashSet<&(u64, u64)> = self.mapped.iter().collect();
        for &(start, end) in self.mapped.iter().filter(|m| !wanted_set.contains(m)) {
            self.vm.unmap(start, (end - start) as usize)?;
        }
        for &(start, end) in wanted.iter().filter(|w| !mapped_set.contains(w)) {
            // SAFETY: the region is mapped host memory inside `mem`'s
            // reservation, and is unmapped here before it goes away.
            unsafe {
                let host = mem.host_base().add(start as usize);
                self.vm
                    .map(host, start, (end - start) as usize, hvf::mem::RWX)?
            };
        }
        self.mapped = wanted;
        Ok(())
    }

    /// Whether a writable address in the active tables stays mapped read-only
    /// in `regions`; only then must vCPUs flush (see `docs/jit.md`).
    fn reduces_access(&self, regions: &[memory::RegionSpan], access: &[(u64, u64, bool)]) -> bool {
        let new: HashSet<&(u64, u64, bool)> = access.iter().collect();
        let removed: Vec<_> = self.mirrored.iter().filter(|m| !new.contains(m)).collect();
        if removed.is_empty() {
            return false;
        }
        regions.iter().any(|r| {
            let read = r.prot.allows(memory::Access::Read);
            let write = r.prot.allows(memory::Access::Write);
            removed
                .iter()
                .any(|&&(s, e, w)| s < r.end && r.start < e && read && w && !write)
        })
    }

    /// Changes the active tables in place to map `access`; false if the pool
    /// ran out of tables.
    fn update_mappings(&mut self, access: &[(u64, u64, bool)]) -> bool {
        let pool = SHARED.active.load(Ordering::SeqCst);
        let root = SHARED.roots[pool].load(Ordering::SeqCst);
        let new: HashSet<(u64, u64, bool)> = access.iter().copied().collect();
        let old: HashSet<(u64, u64, bool)> = self.mirrored.iter().copied().collect();
        let mut t = tables::Tables::resume(self.runtime.as_mut_slice(), pool, root, self.pool_next[pool]);
        for &(start, end, _) in old.difference(&new) {
            t.unmap(start, end);
        }
        for &(start, end, write) in new.difference(&old) {
            if !t.map(start, end, write) {
                return false;
            }
        }
        self.pool_next[pool] = t.next();
        self.mirrored = access.to_vec();
        true
    }

    /// Drops every translation, for a new address space whose code may
    /// differ at the same addresses. Native functions stay.
    fn forget_code(&mut self) {
        self.blocks.clear();
        self.regions.clear();
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
                && SHARED.running[slot].load(Ordering::SeqCst)
            {
                k.kick();
            }
        }
    }

    /// Translates the block at `rip` and enters it in the lookup table;
    /// false if it cannot be translated or the cache is full.
    fn translate(&mut self, mem: &Memory, rip: u64) -> bool {
        if let Some(place) = self.regions.get(&rip).map(|r| r.place) {
            self.enter(rip, place);
            return true;
        }
        if let Some(b) = self.blocks.get(&rip) {
            engine::count_exit(engine::Count::Reenter);
            if self.ready(b) {
                let place = b.place;
                self.enter(rip, place);
            }
            return true;
        }
        let exit = if self.verify || self.trace { VERIFY_EXIT } else { DISPATCH };
        let Some((place, block)) = self.emit(mem, rip, exit) else {
            engine::count_exit(engine::Count::NoBlock);
            if !self.trace {
                self.enter(rip, INTERP_EXIT);
            }
            return false;
        };
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
        if !self.verify && !self.trace {
            self.link(rip, place, block.chains);
        }
        true
    }

    /// Sets when blocks are compiled again as tier-2 regions (for tests;
    /// otherwise from `PS5SLOP_JIT_TIER2`). Applies to blocks translated after.
    pub fn set_tier2(&mut self, t: translate::Tier2) {
        self.tier2 = t;
    }

    /// Compiles the region at `rip`, whose tier-1 block counted down to zero,
    /// and makes the block branch to it; if there is no region worth having,
    /// the block stops counting.
    fn tier_up(&mut self, mem: &Memory, rip: u64) {
        let Some(block_place) = self.blocks.get(&rip).map(|b| b.place) else {
            return;
        };
        let skip = block_place + translate::PROLOGUE_WORDS * 4;
        if self.regions.contains_key(&rip) || self.cache_full || self.trace {
            self.patch(block_place, skip);
            return;
        }
        let exit = if self.verify { VERIFY_EXIT } else { DISPATCH };
        let Some((place, rips)) = self.emit_region(mem, rip, exit) else {
            self.patch(block_place, skip);
            return;
        };
        engine::count_exit(engine::Count::Region);
        let verified = if self.verify { 0 } else { VERIFY_RUNS };
        let rips = std::sync::Arc::new(rips);
        self.regions.insert(rip, RegionInfo { place, rips, verified });
        if self.verify {
            self.enter(rip, place);
        } else {
            self.install_region(rip, block_place, place);
        }
    }

    /// Translates the region at `rip` into the code cache, its exits going to
    /// `exit`; where it went and the RIPs of its instructions.
    fn emit_region(&mut self, mem: &Memory, rip: u64, exit: usize) -> Option<(usize, HashSet<u64>)> {
        let place = self.code_next;
        let (region, rips) = translate::region(mem, rip, place, exit)?;
        let len = region.code.len() * 4;
        if place + len > layout::CODE_CACHE_END {
            self.cache_full = true;
            return None;
        }
        self.runtime.write_words(place, &region.code);
        // SAFETY: the range was just written inside the runtime region.
        unsafe { sys_icache_invalidate(self.runtime.as_ptr().add(place).cast(), len) };
        self.code_next = (place + len).next_multiple_of(16);
        self.by_place.insert(place, region.insns);
        self.pending_chains = region.chains;
        Some((place, rips.into_iter().collect()))
    }

    /// Makes the region at `place` what the lookup table, chains and the
    /// tier-1 block at `block_place` lead to.
    fn install_region(&mut self, rip: u64, block_place: usize, place: usize) {
        self.patch(block_place, place);
        let chains = std::mem::take(&mut self.pending_chains);
        self.link(rip, place, chains);
    }

    /// Puts the native function for trap address `rip` back in the lookup
    /// table after a collision took its slot; false if it has none.
    fn reenter_native(&mut self, rip: u64) -> bool {
        match self.natives.iter().find(|&&(a, _)| a == rip) {
            Some(&(_, place)) => {
                self.enter(rip, place);
                true
            }
            None => false,
        }
    }

    /// Whether the block may be entered in the lookup table and chained to:
    /// always, except in verify mode before its checks are done.
    fn ready(&self, b: &BlockInfo) -> bool {
        !self.trace && (!self.verify || b.verified >= VERIFY_RUNS)
    }

    /// Translates the block at `rip` into the code cache, its exits going to
    /// `exit`; where it went and what was translated.
    fn emit(&mut self, mem: &Memory, rip: u64, exit: usize) -> Option<(usize, translate::Block)> {
        if self.cache_full {
            return None;
        }
        let place = self.code_next;
        let count = self.tier2.count().filter(|_| exit == DISPATCH && !self.trace);
        let block = translate::block(mem, rip, place, exit, count)?;
        let len = block.code.len() * 4;
        if place + len > layout::CODE_CACHE_END {
            self.cache_full = true;
            eprintln!("ps5slop: the JIT's code cache is full; interpreting new code from now on");
            return None;
        }
        self.runtime.write_words(place, &block.code);
        // SAFETY: the range was just written inside the runtime region.
        unsafe { sys_icache_invalidate(self.runtime.as_ptr().add(place).cast(), len) };
        self.code_next = (place + len).next_multiple_of(16);
        Some((place, block))
    }

    /// Enters `rip`'s block at `place` and chains it both ways to ready
    /// blocks (`chains`: exit word offset, target).
    fn link(&mut self, rip: u64, place: usize, chains: Vec<(u32, u64)>) {
        self.enter(rip, place);
        for (word, target) in chains {
            let site = place + word as usize * 4;
            match self.blocks.get(&target).filter(|b| self.ready(b)) {
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

    /// In verify mode, the block or region at `rip`: its virtual address, how
    /// far to check it, and whether it still has to be checked.
    fn verifying(&self, rip: u64) -> Option<(u64, Steps, bool)> {
        if !self.verify {
            return None;
        }
        if let Some(r) = self.regions.get(&rip) {
            let steps = Steps::Region(r.rips.clone());
            return Some((layout::va(r.place), steps, r.verified < VERIFY_RUNS));
        }
        let b = self.blocks.get(&rip)?;
        Some((layout::va(b.place), Steps::Insns(b.insns), b.verified < VERIFY_RUNS))
    }

    /// Counts one checked run of the block or region at `rip`. After the
    /// last, it is translated again with ordinary exits, entered and chained.
    fn verified(&mut self, rip: u64, mem: &Memory) {
        if let Some(r) = self.regions.get_mut(&rip) {
            r.verified += 1;
            if r.verified != VERIFY_RUNS {
                return;
            }
            let block_place = self.blocks.get(&rip).map(|b| b.place);
            if let (Some((place, _)), Some(block_place)) = (self.emit_region(mem, rip, DISPATCH), block_place) {
                if let Some(r) = self.regions.get_mut(&rip) {
                    r.place = place;
                }
                self.install_region(rip, block_place, place);
            }
            return;
        }
        let Some(b) = self.blocks.get_mut(&rip) else { return };
        b.verified += 1;
        if b.verified != VERIFY_RUNS {
            return;
        }
        let Some((place, block)) = self.emit(mem, rip, DISPATCH) else {
            return;
        };
        if let Some(b) = self.blocks.get_mut(&rip) {
            b.place = place;
        }
        self.by_place.insert(place, block.insns);
        self.link(rip, place, block.chains);
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
    // The tag is read again after the code: `enter` may replace the slot in
    // between, and loads stay in order (TSO), so two matching reads are safe.
    a.ldr(Width::X, tag, slot, 0);
    a.eor(Size::X, tag, tag, rip);
    a.cbnz(Size::X, tag, miss);
    a.ldr(Width::X, tag, slot, 8);
    a.ldr(Width::X, slot, slot, 0);
    a.eor(Size::X, slot, slot, rip);
    a.cbnz(Size::X, slot, miss);
    a.br(tag);
    a.bind(miss);
    a.hvc(translate::HVC_MISS);
    let code = a.finish();
    assert!(DISPATCH + code.len() * 4 <= VERIFY_EXIT);
    put(rt, DISPATCH, code);
    let mut a = Asm::new();
    a.hvc(translate::HVC_MISS);
    put(rt, VERIFY_EXIT, a.finish());
    let mut a = Asm::new();
    a.hvc(translate::HVC_INTERP);
    put(rt, INTERP_EXIT, a.finish());
    let code = native_memset();
    assert!(NATIVE_MEMSET + code.len() * 4 <= NATIVE_MEMMOVE);
    put(rt, NATIVE_MEMSET, code);
    let code = native_memmove();
    assert!(NATIVE_MEMMOVE + code.len() * 4 <= NATIVE_SPECIFIC);
    put(rt, NATIVE_MEMMOVE, code);
    let mut a = Asm::new();
    a.lsr_imm(Size::X, Reg(0), translate::FS_BASE, 14);
    a.mov(Size::W, Reg(0), Reg(0));
    guest_return(&mut a, NATIVE_THREAD_ID);
    put(rt, NATIVE_THREAD_ID, a.finish());
    let mut a = Asm::new();
    a.mov(Size::X, Reg(0), translate::FS_BASE);
    guest_return(&mut a, NATIVE_SELF);
    let code = a.finish();
    assert!(NATIVE_SELF + code.len() * 4 <= NATIVE_END);
    put(rt, NATIVE_SELF, code);
}

/// `pthread_getspecific(edi)` from the thread's control block; keys from
/// `keys` on leave the VM for the host function, whose address is in x16.
fn native_getspecific(specific: u32, keys: u32) -> Vec<u32> {
    let (key, at) = (Reg(19), Reg(20));
    let mut a = Asm::new();
    let host = a.label();
    a.mov(Size::W, key, Reg(7));
    a.mov_imm(Size::X, at, keys as u64);
    a.sub(Size::X, at, key, at);
    a.tbz(at, 63, host);
    a.add_imm(Size::X, at, translate::FS_BASE, specific);
    a.add_shift(Size::X, at, at, key, Shift::Lsl, 3);
    a.ldr(Width::X, Reg(0), at, 0);
    guest_return(&mut a, NATIVE_SPECIFIC);
    a.bind(host);
    a.hvc(translate::HVC_HOST);
    a.finish()
}

/// Returns from a native library function as the guest's `ret` would, with
/// RAX = RDI (what `memset` and `memmove` return), and dispatches.
fn native_return(a: &mut Asm, at: usize) {
    a.mov(Size::X, Reg(0), Reg(7));
    guest_return(a, at);
}

/// Returns from a native function as the guest's `ret` would, and dispatches.
fn guest_return(a: &mut Asm, at: usize) {
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
    pub fn run(&mut self, _m: &Machine, pc: u64) -> Result<Exit, Error> {
        let root = SHARED.roots[SHARED.active.load(Ordering::SeqCst)].load(Ordering::SeqCst);
        self.enter(root, RESUME_FLUSH, Some((pc, CPSR_EL1H)), 0)
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
        SHARED.running[self.slot].store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)] mod dump_tmp;
