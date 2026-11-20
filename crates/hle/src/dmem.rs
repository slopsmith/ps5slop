//! Direct memory: "physical" ranges a title allocates, then maps into its
//! address space. Prototypes and constants follow the OpenOrbis headers.
//!
//! Each map gets its own fresh memory: two maps of the same direct range do
//! not alias yet, and released ranges stay mapped.

use std::collections::BTreeMap;
use std::sync::Mutex;

use memory::{Access, PAGE_SIZE, Prot};

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome, errno};

/// How much direct memory a title gets: 12.5 GiB (our choice; the PS5's is
/// unknown). Astro Bot's engine allocates about 12.1 GiB of it.
pub const DIRECT_MEMORY_SIZE: u64 = 0x3_2000_0000;
/// Where maps without a fixed address go: the start of the user area.
const MAP_AREA: u64 = 0x10_0000_0000;
/// Where direct memory and `mmap` without a fixed address land (both search
/// upward from `MAP_AREA`), for the GPU's guest memory window.
pub(crate) const GPU_AREA: (u64, u64) = (MAP_AREA, MAP_AREA + (16 << 30));
const MAP_FIXED: u64 = 0x10;
const PROT_CPU_READ: u64 = 0x01;
const PROT_CPU_WRITE: u64 = 0x02;
const PROT_CPU_EXEC: u64 = 0x04;
const PROT_GPU_READ: u64 = 0x10;
const PROT_GPU_WRITE: u64 = 0x20;

/// libkernel's `sce*` error codes: `0x80020000` plus the errno value.
pub(crate) const fn sce_error(errno: i32) -> u64 {
    (0x8002_0000u32 + errno as u32) as u64
}

#[derive(Default)]
pub struct DirectMemory {
    /// Allocated ranges: start to length and memory type.
    allocated: Mutex<BTreeMap<u64, (u64, i32)>>,
    /// Failed calls by kind, for [`report`].
    failures: Mutex<BTreeMap<&'static str, u64>>,
    /// Direct memory mappings: virtual start to end and the direct offset
    /// minus the virtual address, so pieces of a split mapping keep it.
    mapped: Mutex<BTreeMap<u64, (u64, u64)>>,
}

/// What direct memory looks like: bytes allocated, allocations, bytes free
/// and the largest free block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectStats {
    pub allocated: u64,
    pub allocations: usize,
    pub free: u64,
    pub largest_free: u64,
}

impl DirectMemory {
    /// First fit for `len` bytes at `align` inside `start..end`.
    fn allocate(&self, start: u64, end: u64, len: u64, align: u64, kind: i32) -> Option<u64> {
        let mut allocated = lock(&self.allocated);
        let mut at = start.next_multiple_of(align);
        for (&a, &(l, _)) in allocated.range(..) {
            if at + len <= a {
                break;
            }
            at = at.max((a + l).next_multiple_of(align));
        }
        if at + len > end.min(DIRECT_MEMORY_SIZE) {
            return None;
        }
        allocated.insert(at, (len, kind));
        Some(at)
    }

    /// Frees `[start, start + len)`, keeping what allocations around it
    /// stick out on either side.
    fn release(&self, start: u64, len: u64) {
        let end = start + len;
        let mut allocated = lock(&self.allocated);
        let hit: Vec<(u64, u64, i32)> = allocated
            .range(..end)
            .map(|(&a, &(l, k))| (a, l, k))
            .filter(|&(a, l, _)| a + l > start)
            .collect();
        for (a, l, k) in hit {
            allocated.remove(&a);
            if a < start {
                allocated.insert(a, (start - a, k));
            }
            if a + l > end {
                allocated.insert(end, (a + l - end, k));
            }
        }
    }

    /// Whether every byte of `[start, start + len)` is allocated.
    fn covers(&self, start: u64, len: u64) -> bool {
        let allocated = lock(&self.allocated);
        let mut at = start;
        for (&a, &(l, _)) in allocated.range(..start + len) {
            if a + l <= at {
                continue;
            }
            if a > at {
                return false;
            }
            at = a + l;
        }
        at >= start + len
    }

    pub fn stats(&self) -> DirectStats {
        let allocated = lock(&self.allocated);
        let (mut used, mut at, mut largest) = (0, 0, 0);
        for (&a, &(l, _)) in allocated.iter() {
            largest = largest.max(a.saturating_sub(at));
            used += l;
            at = at.max(a + l);
        }
        largest = largest.max(DIRECT_MEMORY_SIZE.saturating_sub(at));
        DirectStats {
            allocated: used,
            allocations: allocated.len(),
            free: DIRECT_MEMORY_SIZE.saturating_sub(used),
            largest_free: largest,
        }
    }

    /// Records that `[addr, addr + len)` now maps direct memory from `offset`.
    fn note_map(&self, addr: u64, len: u64, offset: u64) {
        let mut mapped = lock(&self.mapped);
        cut(&mut mapped, addr, addr + len);
        mapped.insert(addr, (addr + len, offset.wrapping_sub(addr)));
    }

    /// Forgets the direct memory behind `[addr, addr + len)`.
    pub(crate) fn note_unmap(&self, addr: u64, len: u64) {
        cut(&mut lock(&self.mapped), addr, addr.saturating_add(len));
    }

    /// Forgets and returns the virtual ranges that map any of the direct
    /// memory `[start, start + len)`.
    fn take_mappings_of(&self, start: u64, len: u64) -> Vec<(u64, u64)> {
        let end = start.saturating_add(len);
        let mut mapped = lock(&self.mapped);
        let hits: Vec<(u64, u64)> = mapped
            .iter()
            .filter_map(|(&v, &(v_end, delta))| {
                let (p, p_end) = (v.wrapping_add(delta), v_end.wrapping_add(delta));
                let (from, to) = (p.max(start), p_end.min(end));
                (from < to).then(|| (from.wrapping_sub(delta), to.wrapping_sub(delta)))
            })
            .collect();
        for &(v, v_end) in &hits {
            cut(&mut mapped, v, v_end);
        }
        hits
    }

    /// The direct memory offset mapped at `addr`.
    fn offset_at(&self, addr: u64) -> Option<u64> {
        let mapped = lock(&self.mapped);
        let (_, &(end, delta)) = mapped.range(..=addr).next_back()?;
        (addr < end).then(|| addr.wrapping_add(delta))
    }

    /// Counts a failed call and returns whether it is the first of its kind.
    fn failed(&self, kind: &'static str) -> bool {
        let mut failures = lock(&self.failures);
        let n = failures.entry(kind).or_default();
        *n += 1;
        *n == 1
    }

    /// The allocation holding `offset`, or with `next` the first one after
    /// it: start, end and memory type.
    fn query(&self, offset: u64, next: bool) -> Option<(u64, u64, i32)> {
        let allocated = lock(&self.allocated);
        let holding = allocated
            .range(..=offset)
            .next_back()
            .filter(|&(&a, &(l, _))| a + l > offset);
        let found = match holding {
            Some(h) => Some(h),
            None if next => allocated.range(offset..).next(),
            None => None,
        };
        found.map(|(&a, &(l, k))| (a, a + l, k))
    }
}

/// Removes `[start, end)` from a map of ranges, trimming ranges that stick out.
fn cut(ranges: &mut BTreeMap<u64, (u64, u64)>, start: u64, end: u64) {
    let hit: Vec<(u64, (u64, u64))> = ranges
        .range(..end)
        .filter(|&(_, &(e, _))| e > start)
        .map(|(&s, &v)| (s, v))
        .collect();
    for (s, (e, delta)) in hit {
        ranges.remove(&s);
        if s < start {
            ranges.insert(s, (start, delta));
        }
        if e > end {
            ranges.insert(end, (e, delta));
        }
    }
}

/// Reports a failed memory call on stderr the first time its kind happens,
/// with the state of direct memory and the address space.
pub(crate) fn failure(env: &Env, kind: &'static str, detail: std::fmt::Arguments) {
    if env.kernel.dmem.failed(kind) {
        eprintln!("ps5slop: memory: {kind} ({detail}); later ones are only counted");
        eprintln!("{}", report(env.kernel));
    }
}

/// Memory use, for the control socket's `memory` command and failure
/// reports: direct memory, the guest's mappings by name, the host heap and
/// the host process.
pub fn report(kernel: &crate::Kernel) -> String {
    let mib = |b: u64| format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64);
    let d = kernel.dmem.stats();
    let mut out = vec![format!(
        "direct memory: {} allocated in {} allocations, {} free, largest free block {}",
        mib(d.allocated),
        d.allocations,
        mib(d.free),
        mib(d.largest_free)
    )];
    let mut names: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let (mut count, mut total) = (0, 0);
    let mut largest_gap = (0, 0);
    let mut at = GPU_AREA.0;
    for r in kernel.memory().regions().into_iter().filter(|r| !r.trap) {
        let e = names.entry(r.name.clone()).or_default();
        e.0 += 1;
        e.1 += r.end - r.start;
        count += 1;
        total += r.end - r.start;
        if r.end > GPU_AREA.0 && r.start < GPU_AREA.1 {
            let gap = r.start.saturating_sub(at);
            if gap > largest_gap.1 {
                largest_gap = (at, gap);
            }
            at = at.max(r.end.min(GPU_AREA.1));
        }
    }
    if GPU_AREA.1 - at > largest_gap.1 {
        largest_gap = (at, GPU_AREA.1 - at);
    }
    out.push(format!("mapped: {count} regions, {}", mib(total)));
    for (name, (n, bytes)) in &names {
        out.push(format!("  {name}: {n} regions, {}", mib(*bytes)));
    }
    out.push(format!(
        "GPU area {:#x}..{:#x}: largest free range {} at {:#x}",
        GPU_AREA.0,
        GPU_AREA.1,
        mib(largest_gap.1),
        largest_gap.0
    ));
    let (heap_size, heap_used, heap_blocks) = kernel.heap.stats();
    out.push(format!(
        "host heap: {} in {heap_blocks} blocks, {} mapped",
        mib(heap_used),
        mib(heap_size)
    ));
    if let Some((resident, footprint)) = host_memory() {
        out.push(format!("host process: {} resident, {} footprint", mib(resident), mib(footprint)));
    }
    let failures = lock(&kernel.dmem.failures);
    if !failures.is_empty() {
        let list: Vec<String> = failures.iter().map(|(k, n)| format!("{k} {n}")).collect();
        out.push(format!("failures: {}", list.join(", ")));
    }
    out.join("\n")
}

/// The host process's resident size and physical footprint.
fn host_memory() -> Option<(u64, u64)> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: RUSAGE_INFO_V4 fills a `rusage_info_v4`.
    let r = unsafe {
        libc::proc_pid_rusage(std::process::id() as i32, libc::RUSAGE_INFO_V4, info.as_mut_ptr().cast())
    };
    // SAFETY: zero-initialised and filled by the call on success.
    let info = unsafe { info.assume_init() };
    (r == 0).then_some((info.ri_resident_size, info.ri_phys_footprint))
}

pub static FUNCTIONS: &[Function] = &[
    f("sceKernelGetDirectMemorySize", size),
    f("sceKernelAllocateDirectMemory", allocate),
    f("sceKernelAllocateMainDirectMemory", allocate_main),
    f("sceKernelReleaseDirectMemory", release),
    f("sceKernelCheckedReleaseDirectMemory", checked_release),
    f("sceKernelMapDirectMemory", map),
    f("sceKernelMapNamedDirectMemory", map),
    f("sceKernelMapFlexibleMemory", map_flexible),
    f("sceKernelMapNamedFlexibleMemory", map_flexible),
    f("sceKernelMapNamedFlexibleMemoryInternal", map_flexible),
    f("sceKernelReserveVirtualRange", reserve),
    f("sceKernelVirtualQuery", virtual_query),
    f("sceKernelSetVirtualRangeName", set_range_name),
    f("sceKernelClearVirtualRangeName", set_range_name),
    f("sceKernelDirectMemoryQuery", query),
    f("sceKernelQueryMemoryProtection", query_protection),
    f("sceKernelBatchMap", batch_map),
    f("sceKernelBatchMap2", batch_map2),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

fn size(_: &mut Env) -> Result<Outcome, CallError> {
    ret(DIRECT_MEMORY_SIZE)
}

fn valid(len: u64, align: u64) -> bool {
    len > 0 && len.is_multiple_of(PAGE_SIZE) && (align == 0 || align.is_power_of_two())
}

fn allocate_in(
    env: &Env,
    start: u64,
    end: u64,
    len: u64,
    align: u64,
    kind: i32,
    out: u64,
) -> Result<Outcome, CallError> {
    if !valid(len, align) || start >= end {
        return ret(sce_error(errno::EINVAL));
    }
    match env
        .kernel
        .dmem
        .allocate(start, end, len, align.max(PAGE_SIZE), kind)
    {
        Some(at) => {
            env.mem.write_u64(out, at)?;
            ret(0)
        }
        None => {
            failure(
                env,
                "direct memory allocation failed",
                format_args!("{len:#x} bytes, alignment {align:#x}, in {start:#x}..{end:#x}"),
            );
            ret(sce_error(errno::EAGAIN))
        }
    }
}

/// `sceKernelAllocateDirectMemory(search_start, search_end, len, align, type, *start)`.
fn allocate(env: &mut Env) -> Result<Outcome, CallError> {
    let (start, end, len, align, kind, out) = (
        env.arg(0),
        env.arg(1),
        env.arg(2),
        env.arg(3),
        env.arg(4) as i32,
        env.arg(5),
    );
    allocate_in(env, start, end, len, align, kind, out)
}

/// `sceKernelAllocateMainDirectMemory(len, align, type, *start)`.
fn allocate_main(env: &mut Env) -> Result<Outcome, CallError> {
    let (len, align, kind, out) = (env.arg(0), env.arg(1), env.arg(2) as i32, env.arg(3));
    allocate_in(env, 0, DIRECT_MEMORY_SIZE, len, align, kind, out)
}

/// `sceKernelReleaseDirectMemory(start, len)`.
fn release(env: &mut Env) -> Result<Outcome, CallError> {
    let (start, len) = (env.arg(0), env.arg(1));
    if !env.kernel.dmem.covers(start, len) {
        failure(env, "release of unallocated direct memory", format_args!("{len:#x} bytes at {start:#x}"));
    }
    release_and_unmap(env, start, len);
    ret(0)
}

/// `sceKernelCheckedReleaseDirectMemory(start, len)`: fails with `EACCES`
/// (a guess) unless the whole range is allocated.
fn checked_release(env: &mut Env) -> Result<Outcome, CallError> {
    let (start, len) = (env.arg(0), env.arg(1));
    if !env.kernel.dmem.covers(start, len) {
        return ret(sce_error(errno::EACCES));
    }
    release_and_unmap(env, start, len);
    ret(0)
}

/// Frees direct memory and unmaps every mapping of it, as shadPS4 does on
/// the PS4 (unverified for the PS5): Minecraft releases some blocks without
/// unmapping them first.
fn release_and_unmap(env: &Env, start: u64, len: u64) {
    for (v, v_end) in env.kernel.dmem.take_mappings_of(start, len) {
        let _ = env.mem.unmap_range(v, v_end - v);
    }
    env.kernel.dmem.release(start, len);
}

/// `sceKernelMapDirectMemory(*addr, len, prot, flags, direct_start, align)`;
/// GPU access is granted as CPU access.
fn map(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr_at, len, prot, flags, offset, align) =
        (env.arg(0), env.arg(1), env.arg(2), env.arg(3), env.arg(4), env.arg(5));
    let wanted = env.mem.read_u64(addr_at)?;
    match map_direct(env, wanted, len, prot, flags, offset, align) {
        Ok(at) => {
            env.mem.write_u64(addr_at, at)?;
            ret(0)
        }
        Err(e) => ret(sce_error(e)),
    }
}

fn gpu_as_cpu(prot: u64) -> Prot {
    let read = prot & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_GPU_READ | PROT_GPU_WRITE) != 0;
    let write = prot & (PROT_CPU_WRITE | PROT_GPU_WRITE) != 0;
    Prot::new(read, write, prot & PROT_CPU_EXEC != 0)
}

fn map_direct(env: &Env, wanted: u64, len: u64, prot: u64, flags: u64, offset: u64, align: u64) -> Result<u64, i32> {
    if !valid(len, align) {
        return Err(errno::EINVAL);
    }
    let prot = gpu_as_cpu(prot);
    let mapped = if flags & MAP_FIXED != 0 {
        env.mem
            .map_over(wanted, len, prot, "direct memory")
            .map(|()| wanted)
    } else {
        env.mem
            .map_free(wanted.max(MAP_AREA), len, align, prot, "direct memory")
    };
    let at = match mapped {
        Ok(at) => at,
        Err(e) => {
            failure(env, "direct memory map failed", format_args!("{len:#x} bytes at {wanted:#x}, flags {flags:#x}, alignment {align:#x}: {e:?}"));
            return Err(errno::ENOMEM);
        }
    };
    if !env.kernel.dmem.covers(offset, len) {
        failure(env, "map of unallocated direct memory", format_args!("{len:#x} bytes at {offset:#x} to {at:#x}"));
    }
    env.kernel.dmem.note_map(at, len, offset);
    Ok(at)
}

/// `(addr, *start, *end, *prot)`; GPU bits mirror the CPU ones, since GPU
/// access is mapped as CPU access. `EACCES` when unmapped is a guess.
fn query_protection(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr, start, end, prot) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    let found = env
        .mem
        .regions()
        .into_iter()
        .find(|r| !r.trap && r.start <= addr && r.end > addr);
    let Some(r) = found else {
        return ret(sce_error(errno::EACCES));
    };
    let mut bits = 0u32;
    if r.prot.allows(Access::Read) {
        bits |= (PROT_CPU_READ | PROT_GPU_READ) as u32;
    }
    if r.prot.allows(Access::Write) {
        bits |= (PROT_CPU_WRITE | PROT_GPU_WRITE) as u32;
    }
    if r.prot.allows(Access::Execute) {
        bits |= PROT_CPU_EXEC as u32;
    }
    for (at, v) in [(start, r.start), (end, r.end)] {
        if at != 0 {
            env.mem.write_u64(at, v)?;
        }
    }
    if prot != 0 {
        env.mem.write(prot, &bits.to_le_bytes())?;
    }
    ret(0)
}

/// `sceKernelBatchMap(entries, count, *done)`: `sceKernelBatchMap2` with
/// `MAP_FIXED`.
fn batch_map(env: &mut Env) -> Result<Outcome, CallError> {
    let (entries, count, done) = (env.arg(0), env.arg(1), env.arg(2));
    batch(env, entries, count, done, MAP_FIXED)
}

/// `sceKernelBatchMap2(entries, count, *done, flags)`.
fn batch_map2(env: &mut Env) -> Result<Outcome, CallError> {
    let (entries, count, done, flags) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    batch(env, entries, count, done, flags)
}

/// Entries are shadPS4's PS4 `OrbisKernelBatchMapEntry`; operations 0 map
/// direct, 1 unmap, 2 protect, 3 map flexible, 4 protect with type.
fn batch(env: &Env, entries: u64, count: u64, done: u64, flags: u64) -> Result<Outcome, CallError> {
    let mut result = 0;
    let mut n = 0i32;
    for k in 0..count {
        let mut e = [0u8; 32];
        env.mem.read(entries + 32 * k, &mut e)?;
        let word = |at: usize| u64::from_le_bytes(e[at..at + 8].try_into().unwrap());
        let (start, offset, len, prot) = (word(0), word(8), word(16), e[24] as u64);
        let op = i32::from_le_bytes(e[28..32].try_into().unwrap());
        let r = match op {
            0 => map_direct(env, start, len, prot, flags | MAP_FIXED, offset, 0).map(drop),
            1 => {
                env.kernel.dmem.note_unmap(start, len);
                env.mem.unmap_range(start, len).map_err(|_| errno::EINVAL)
            }
            2 | 4 => env.mem.protect_range(start, len, gpu_as_cpu(prot)).map_err(|_| errno::EINVAL),
            3 => {
                env.kernel.dmem.note_unmap(start, len);
                env.mem
                    .map_over(start, len, gpu_as_cpu(prot), "mmap")
                    .map_err(|_| errno::ENOMEM)
            }
            _ => Err(errno::EINVAL),
        };
        if let Err(e) = r {
            eprintln!(
                "ps5slop: sceKernelBatchMap entry {k} (op {op}, {start:#x}+{len:#x}, prot {prot:#x}) failed with errno {e}"
            );
            result = sce_error(e);
            break;
        }
        n += 1;
    }
    if done != 0 {
        env.mem.write(done, &n.to_le_bytes())?;
    }
    ret(result)
}

/// `sceKernelMapFlexibleMemory(*addr, len, prot, flags)`, and the named
/// forms with a fifth argument, the name (GTA V's libc calls the `Internal`
/// one with flags 0x8000, whose meaning we do not know): zeroed memory with no direct range
/// behind it, kept as an `mmap` region so `sceKernelVirtualQuery` reports it
/// as flexible.
fn map_flexible(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr_at, len, prot, flags) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    if !valid(len, 0) {
        return ret(sce_error(errno::EINVAL));
    }
    let wanted = env.mem.read_u64(addr_at)?;
    let read = prot & (PROT_CPU_READ | PROT_CPU_WRITE) != 0;
    let prot = Prot::new(read, prot & PROT_CPU_WRITE != 0, prot & PROT_CPU_EXEC != 0);
    let mapped = if flags & MAP_FIXED != 0 {
        env.mem.map_over(wanted, len, prot, "mmap").map(|()| wanted)
    } else {
        env.mem
            .map_free(wanted.max(MAP_AREA), len, PAGE_SIZE, prot, "mmap")
    };
    let at = match mapped {
        Ok(at) => at,
        Err(e) => {
            failure(env, "flexible memory map failed", format_args!("{len:#x} bytes at {wanted:#x}, flags {flags:#x}: {e:?}"));
            return ret(sce_error(errno::ENOMEM));
        }
    };
    env.kernel.dmem.note_unmap(at, len);
    env.mem.write_u64(addr_at, at)?;
    ret(0)
}

/// `sceKernelReserveVirtualRange(*addr, len, flags, align)`: address space
/// no access is allowed to, for fixed maps to go into later.
fn reserve(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr_at, len, flags, align) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    if !valid(len, align) {
        return ret(sce_error(errno::EINVAL));
    }
    let wanted = env.mem.read_u64(addr_at)?;
    let reserved = if flags & MAP_FIXED != 0 {
        env.mem
            .map_over(wanted, len, Prot::NONE, "reserved")
            .map(|()| wanted)
    } else {
        env.mem
            .map_free(wanted.max(MAP_AREA), len, align, Prot::NONE, "reserved")
    };
    let at = match reserved {
        Ok(at) => at,
        Err(e) => {
            failure(env, "virtual range reservation failed", format_args!("{len:#x} bytes at {wanted:#x}, flags {flags:#x}, alignment {align:#x}: {e:?}"));
            return ret(sce_error(errno::ENOMEM));
        }
    };
    env.kernel.dmem.note_unmap(at, len);
    env.mem.write_u64(addr_at, at)?;
    ret(0)
}

/// `sceKernelVirtualQuery(addr, flags, info, size)`: the mapping holding
/// `addr` (flag 1: or the next one), as `OrbisVirtualQueryInfo` (shadPS4's
/// layout): start, end, direct memory offset, protection, memory type, kind
/// bits and name. Minecraft releases direct memory at the offset it reads here.
fn virtual_query(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr, flags, info, size) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3) as usize);
    let found = env
        .mem
        .regions()
        .into_iter()
        .filter(|r| !r.trap)
        .find(|r| r.end > addr && (r.start <= addr || flags & 1 != 0));
    let Some(r) = found else {
        return ret(sce_error(errno::EACCES));
    };
    let bit = |a: memory::Access, v: u32| if r.prot.allows(a) { v } else { 0 };
    let prot = bit(memory::Access::Read, 1)
        | bit(memory::Access::Write, 2)
        | bit(memory::Access::Execute, 4);
    let direct = r.name == "direct memory";
    let kinds = (r.name == "mmap") as u8
        | (direct as u8) << 1
        | (r.name.contains("stack") as u8) << 2
        | ((prot != 0) as u8) << 4;
    let dmem = &env.kernel.dmem;
    let offset = direct.then(|| dmem.offset_at(r.start)).flatten().unwrap_or(0);
    let kind = direct.then(|| dmem.query(offset, false)).flatten().map_or(0, |q| q.2);
    let mut b = [0u8; 72];
    b[0..8].copy_from_slice(&r.start.to_le_bytes());
    b[8..16].copy_from_slice(&r.end.to_le_bytes());
    b[16..24].copy_from_slice(&offset.to_le_bytes());
    b[24..28].copy_from_slice(&prot.to_le_bytes());
    b[28..32].copy_from_slice(&kind.to_le_bytes());
    b[32] = kinds;
    let name = r.name.as_bytes();
    let n = name.len().min(31);
    b[33..33 + n].copy_from_slice(&name[..n]);
    env.mem.write(info, &b[..size.min(72)])?;
    ret(0)
}

/// `sceKernelSetVirtualRangeName(addr, len, name)`: a debugging label;
/// accepted and dropped. `sceKernelClearVirtualRangeName(addr, len)` (name
/// confirmed by hash; Astro Bot asserts on its result when it frees a
/// thread stack) succeeds the same way.
fn set_range_name(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// `sceKernelDirectMemoryQuery(offset, flags, info, size)`: the allocation
/// holding `offset` (flag 1: or the next one) as `{ start, end, type }`
/// (shadPS4's PS4 layout, 24 bytes; unverified for the PS5).
fn query(env: &mut Env) -> Result<Outcome, CallError> {
    let (offset, flags, info, size) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3) as usize);
    let Some((start, end, kind)) = env.kernel.dmem.query(offset, flags & 1 != 0) else {
        return ret(sce_error(errno::EACCES));
    };
    let mut b = [0u8; 24];
    b[0..8].copy_from_slice(&start.to_le_bytes());
    b[8..16].copy_from_slice(&end.to_le_bytes());
    b[16..20].copy_from_slice(&kind.to_le_bytes());
    env.mem.write(info, &b[..size.min(24)])?;
    ret(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_respect_range_alignment_and_release() {
        let d = DirectMemory::default();
        let a = d.allocate(0, DIRECT_MEMORY_SIZE, 0x4000, 0x4000, 0).unwrap();
        let b = d.allocate(0, DIRECT_MEMORY_SIZE, 0x4000, 0x10000, 0).unwrap();
        assert_eq!((a, b), (0, 0x10000));
        assert_eq!(d.allocate(0, 0x8000, 0x8000, 0x4000, 0), None);
        d.release(a, 0x4000);
        assert_eq!(d.allocate(0, DIRECT_MEMORY_SIZE, 0x4000, 0x4000, 0), Some(0));
    }

    #[test]
    fn releasing_part_of_an_allocation_frees_only_that_part() {
        let d = DirectMemory::default();
        assert_eq!(d.allocate(0, 1 << 30, 0x200_0000, 0x1_0000, 0), Some(0));
        d.release(0x4d_0000, 0x3_0000);
        assert_eq!(d.allocate(0, 1 << 30, 0x1_0000, 0x1_0000, 0), Some(0x4d_0000));
        assert_eq!(d.allocate(0, 1 << 30, 0x3_0000, 0x1_0000, 0), Some(0x200_0000));
    }

    #[test]
    fn query_finds_the_holding_or_next_allocation_with_its_type() {
        let d = DirectMemory::default();
        d.allocate(0x1_0000, 1 << 30, 0x4000, 0x4000, 3).unwrap();
        assert_eq!(d.query(0x1_2000, false), Some((0x1_0000, 0x1_4000, 3)));
        assert_eq!(d.query(0, false), None);
        assert_eq!(d.query(0, true), Some((0x1_0000, 0x1_4000, 3)));
        assert_eq!(d.query(0x1_4000, true), None);
    }

    #[test]
    fn mappings_remember_their_offset_through_splits_and_remaps() {
        let d = DirectMemory::default();
        d.note_map(0x10_0000_0000, 0x4_0000, 0x200_0000);
        assert_eq!(d.offset_at(0x10_0000_0000), Some(0x200_0000));
        assert_eq!(d.offset_at(0x10_0001_0000), Some(0x201_0000));
        d.note_map(0x10_0001_0000, 0x1_0000, 0x50_0000);
        assert_eq!(d.offset_at(0x10_0001_0000), Some(0x50_0000));
        assert_eq!(d.offset_at(0x10_0002_0000), Some(0x202_0000));
        d.note_unmap(0x10_0000_0000, 0x1_0000);
        assert_eq!(d.offset_at(0x10_0000_0000), None);
        assert_eq!(d.offset_at(0x10_0004_0000), None);
    }

    #[test]
    fn releasing_direct_memory_finds_every_mapping_of_it() {
        let d = DirectMemory::default();
        d.note_map(0x10_0000_0000, 0x4_0000, 0x10_0000);
        d.note_map(0x10_1000_0000, 0x1_0000, 0x12_0000);
        d.note_map(0x10_2000_0000, 0x1_0000, 0x80_0000);
        let mut hits = d.take_mappings_of(0x12_0000, 0x4_0000);
        hits.sort();
        assert_eq!(hits, [(0x10_0002_0000, 0x10_0004_0000), (0x10_1000_0000, 0x10_1001_0000)]);
        assert_eq!(d.offset_at(0x10_0001_0000), Some(0x11_0000));
        assert_eq!(d.offset_at(0x10_0002_0000), None);
        assert_eq!(d.offset_at(0x10_2000_0000), Some(0x80_0000));
    }

    #[test]
    fn stats_count_allocated_bytes_and_the_largest_hole() {
        let d = DirectMemory::default();
        d.allocate(0, DIRECT_MEMORY_SIZE, 0x1_0000, 0x4000, 0).unwrap();
        d.allocate(0x10_0000, DIRECT_MEMORY_SIZE, 0x1_0000, 0x4000, 0).unwrap();
        let s = d.stats();
        assert_eq!((s.allocated, s.allocations), (0x2_0000, 2));
        assert_eq!(s.free, DIRECT_MEMORY_SIZE - 0x2_0000);
        assert_eq!(s.largest_free, DIRECT_MEMORY_SIZE - 0x11_0000);
    }

    #[test]
    fn covers_needs_every_byte_allocated_across_neighbours() {
        let d = DirectMemory::default();
        d.allocate(0, 1 << 30, 0x4000, 0x4000, 0).unwrap();
        d.allocate(0, 1 << 30, 0x4000, 0x4000, 0).unwrap();
        assert!(d.covers(0x2000, 0x4000));
        assert!(!d.covers(0x6000, 0x4000));
    }
}
