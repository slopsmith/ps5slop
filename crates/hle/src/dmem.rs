//! Direct memory: "physical" ranges a title allocates, then maps into its
//! address space. Prototypes and constants follow the OpenOrbis headers.
//!
//! Each map gets its own fresh memory: two maps of the same direct range do
//! not alias yet, and released ranges stay mapped.

use std::collections::BTreeMap;
use std::sync::Mutex;

use memory::{Memory, PAGE_SIZE, Prot};

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome, errno};

/// How much direct memory a title gets (our choice; the PS5's is unknown).
pub const DIRECT_MEMORY_SIZE: u64 = 12 << 30;
/// Where maps without a fixed address go: the start of the user area.
const MAP_AREA: u64 = 0x10_0000_0000;
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
    /// Allocated ranges: start to length.
    allocated: Mutex<BTreeMap<u64, u64>>,
}

impl DirectMemory {
    /// First fit for `len` bytes at `align` inside `start..end`.
    fn allocate(&self, start: u64, end: u64, len: u64, align: u64) -> Option<u64> {
        let mut allocated = lock(&self.allocated);
        let mut at = start.next_multiple_of(align);
        for (&a, &l) in allocated.range(..) {
            if at + len <= a {
                break;
            }
            at = at.max((a + l).next_multiple_of(align));
        }
        if at + len > end.min(DIRECT_MEMORY_SIZE) {
            return None;
        }
        allocated.insert(at, len);
        Some(at)
    }

    fn release(&self, start: u64, len: u64) {
        lock(&self.allocated).retain(|&a, &mut l| a + l <= start || a >= start + len);
    }
}

pub static FUNCTIONS: &[Function] = &[
    f("sceKernelGetDirectMemorySize", size),
    f("sceKernelAllocateDirectMemory", allocate),
    f("sceKernelAllocateMainDirectMemory", allocate_main),
    f("sceKernelReleaseDirectMemory", release),
    f("sceKernelMapDirectMemory", map),
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
    out: u64,
) -> Result<Outcome, CallError> {
    if !valid(len, align) || start >= end {
        return ret(sce_error(errno::EINVAL));
    }
    match env
        .kernel
        .dmem
        .allocate(start, end, len, align.max(PAGE_SIZE))
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
    let (start, end, len, align, out) =
        (env.arg(0), env.arg(1), env.arg(2), env.arg(3), env.arg(5));
    allocate_in(env, start, end, len, align, out)
}

/// `sceKernelAllocateMainDirectMemory(len, align, type, *start)`.
fn allocate_main(env: &mut Env) -> Result<Outcome, CallError> {
    let (len, align, out) = (env.arg(0), env.arg(1), env.arg(3));
    allocate_in(env, 0, DIRECT_MEMORY_SIZE, len, align, out)
}

/// `sceKernelReleaseDirectMemory(start, len)`.
fn release(env: &mut Env) -> Result<Outcome, CallError> {
    env.kernel.dmem.release(env.arg(0), env.arg(1));
    ret(0)
}

/// A free, `align`-aligned range of `len` bytes at or above `min`.
fn find_aligned(mem: &Memory, min: u64, len: u64, align: u64) -> Option<u64> {
    let mut at = min.next_multiple_of(align);
    loop {
        let free = mem.find_free(at, len)?;
        if free.is_multiple_of(align) {
            return Some(free);
        }
        at = free.next_multiple_of(align);
    }
}

/// `sceKernelMapDirectMemory(*addr, len, prot, flags, direct_start, align)`.
/// GPU access is granted as CPU access, as the GPU reads through the same map.
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
    let at = if flags & MAP_FIXED != 0 {
        wanted
    } else {
        match find_aligned(env.mem, wanted.max(MAP_AREA), len, align.max(PAGE_SIZE)) {
            Some(a) => a,
            None => return ret(sce_error(errno::ENOMEM)),
        }
    };
    if env
        .mem
        .map(at, len, Prot::new(read, write, exec), "direct memory")
        .is_err()
    {
        return ret(sce_error(errno::ENOMEM));
    }
    env.mem.write_u64(addr_at, at)?;
    ret(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_respect_range_alignment_and_release() {
        let d = DirectMemory::default();
        let a = d.allocate(0, DIRECT_MEMORY_SIZE, 0x4000, 0x4000).unwrap();
        let b = d.allocate(0, DIRECT_MEMORY_SIZE, 0x4000, 0x10000).unwrap();
        assert_eq!((a, b), (0, 0x10000));
        assert_eq!(d.allocate(0, 0x8000, 0x8000, 0x4000), None);
        d.release(a, 0x4000);
        assert_eq!(d.allocate(0, DIRECT_MEMORY_SIZE, 0x4000, 0x4000), Some(0));
    }
}
