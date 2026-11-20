//! Direct memory: "physical" ranges a title allocates, then maps into its
//! address space. Prototypes and constants follow the OpenOrbis headers.
//!
//! Each map gets its own fresh memory: two maps of the same direct range do
//! not alias yet, and released ranges stay mapped.

use std::collections::BTreeMap;
use std::sync::Mutex;

use memory::{PAGE_SIZE, Prot};

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
    f("sceKernelDirectMemoryQuery", query),
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
        None => ret(sce_error(errno::EAGAIN)),
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
    env.kernel.dmem.release(env.arg(0), env.arg(1));
    ret(0)
}

/// `sceKernelCheckedReleaseDirectMemory(start, len)`: fails with `EACCES`
/// (a guess) unless the whole range is allocated.
fn checked_release(env: &mut Env) -> Result<Outcome, CallError> {
    let (start, len) = (env.arg(0), env.arg(1));
    if !env.kernel.dmem.covers(start, len) {
        return ret(sce_error(errno::EACCES));
    }
    env.kernel.dmem.release(start, len);
    ret(0)
}

/// `sceKernelMapDirectMemory(*addr, len, prot, flags, direct_start, align)`;
/// GPU access is granted as CPU access.
fn map(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr_at, len, prot, flags, align) =
        (env.arg(0), env.arg(1), env.arg(2), env.arg(3), env.arg(5));
    if !valid(len, align) {
        return ret(sce_error(errno::EINVAL));
    }
    let wanted = env.mem.read_u64(addr_at)?;
    let read = prot & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_GPU_READ | PROT_GPU_WRITE) != 0;
    let write = prot & (PROT_CPU_WRITE | PROT_GPU_WRITE) != 0;
    let exec = prot & PROT_CPU_EXEC != 0;
    let prot = Prot::new(read, write, exec);
    let mapped = if flags & MAP_FIXED != 0 {
        env.mem
            .map_over(wanted, len, prot, "direct memory")
            .map(|()| wanted)
    } else {
        env.mem
            .map_free(wanted.max(MAP_AREA), len, align, prot, "direct memory")
    };
    let Ok(at) = mapped else {
        return ret(sce_error(errno::ENOMEM));
    };
    env.mem.write_u64(addr_at, at)?;
    ret(0)
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
    let Ok(at) = mapped else {
        return ret(sce_error(errno::ENOMEM));
    };
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
    let Ok(at) = reserved else {
        return ret(sce_error(errno::ENOMEM));
    };
    env.mem.write_u64(addr_at, at)?;
    ret(0)
}

/// `sceKernelVirtualQuery(addr, flags, info, size)`: the mapping holding
/// `addr` (flag 1: or the next one), as `OrbisKernelVirtualQueryInfo`.
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
    let kinds = (r.name == "mmap") as u32
        | ((r.name == "direct memory") as u32) << 1
        | (r.name.contains("stack") as u32) << 2
        | ((prot != 0) as u32) << 4;
    let mut b = [0u8; 72];
    b[0..8].copy_from_slice(&r.start.to_le_bytes());
    b[8..16].copy_from_slice(&r.end.to_le_bytes());
    b[24..28].copy_from_slice(&prot.to_le_bytes());
    b[32..36].copy_from_slice(&kinds.to_le_bytes());
    let name = r.name.as_bytes();
    let n = name.len().min(31);
    b[36..36 + n].copy_from_slice(&name[..n]);
    env.mem.write(info, &b[..size.min(72)])?;
    ret(0)
}

/// `sceKernelSetVirtualRangeName(addr, len, name)`: a debugging label;
/// accepted and dropped.
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
    fn covers_needs_every_byte_allocated_across_neighbours() {
        let d = DirectMemory::default();
        d.allocate(0, 1 << 30, 0x4000, 0x4000, 0).unwrap();
        d.allocate(0, 1 << 30, 0x4000, 0x4000, 0).unwrap();
        assert!(d.covers(0x2000, 0x4000));
        assert!(!d.covers(0x6000, 0x4000));
    }
}
