//! A heap in guest memory for host functions that hand out memory
//! (`strdup`, FILE objects, `getenv` strings). First fit over a free list;
//! the bookkeeping lives on the host.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use memory::{Memory, PAGE_SIZE, Prot};

use crate::pthread::lock;

/// Where the host's guest heap lives (our choice).
pub const HEAP_BASE: u64 = 0x20_0000_0000;
/// How much the heap grows by at a time.
const CHUNK: u64 = 1 << 20;
const ALIGN: u64 = 16;

#[derive(Default)]
struct State {
    end: u64,
    free: BTreeMap<u64, u64>,
    used: HashMap<u64, u64>,
}

#[derive(Default)]
pub struct Heap {
    state: Mutex<State>,
}

impl Heap {
    /// Allocates `size` bytes (16-byte aligned, zeroed on first use) or
    /// returns `None` when the address space is full.
    pub fn alloc(&self, mem: &Memory, size: u64) -> Option<u64> {
        self.alloc_aligned(mem, size, ALIGN)
    }

    /// Allocates `size` bytes at a multiple of `align`, a power of two.
    pub fn alloc_aligned(&self, mem: &Memory, size: u64, align: u64) -> Option<u64> {
        let size = size.max(1).checked_next_multiple_of(ALIGN)?;
        let align = align.max(ALIGN);
        let mut s = lock(&self.state);
        if s.end == 0 {
            s.end = HEAP_BASE;
        }
        let fits = |start: u64, len: u64| {
            let at = start.next_multiple_of(align);
            at + size <= start + len
        };
        let fit = s
            .free
            .iter()
            .find(|&(&a, &l)| fits(a, l))
            .map(|(&a, &l)| (a, l));
        let (start, len) = match fit {
            Some(f) => f,
            None => {
                let grow = size.checked_add(align)?.next_multiple_of(CHUNK);
                let at = s.end;
                mem.map(at, grow, Prot::RW, "host heap").ok()?;
                s.end += grow;
                let (start, len) = match s.free.range(..at).next_back() {
                    Some((&a, &l)) if a + l == at => (a, l + grow),
                    _ => (at, grow),
                };
                s.free.insert(start, len);
                (start, len)
            }
        };
        let addr = start.next_multiple_of(align);
        s.free.remove(&start);
        if addr > start {
            s.free.insert(start, addr - start);
        }
        if start + len > addr + size {
            s.free.insert(addr + size, start + len - addr - size);
        }
        s.used.insert(addr, size);
        Some(addr)
    }

    /// The usable size of the allocation at `addr`, if it is one.
    pub fn size_of(&self, addr: u64) -> Option<u64> {
        lock(&self.state).used.get(&addr).copied()
    }

    /// Frees an allocation; unknown addresses are ignored.
    pub fn free(&self, addr: u64) {
        let mut s = lock(&self.state);
        let Some(size) = s.used.remove(&addr) else {
            return;
        };
        let (mut start, mut len) = (addr, size);
        if let Some((&a, &l)) = s.free.range(..addr).next_back()
            && a + l == addr
        {
            s.free.remove(&a);
            start = a;
            len += l;
        }
        if let Some(&l) = s.free.get(&(addr + size)) {
            s.free.remove(&(addr + size));
            len += l;
        }
        s.free.insert(start, len);
    }
}

const _: () = assert!(HEAP_BASE.is_multiple_of(PAGE_SIZE));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_do_not_overlap_and_freed_memory_is_reused() {
        let mem = Memory::new(1 << 40).unwrap();
        let heap = Heap::default();
        let a = heap.alloc(&mem, 100).unwrap();
        let b = heap.alloc(&mem, 100).unwrap();
        assert!(b >= a + 100);
        mem.write(b, &[1; 100]).unwrap();
        heap.free(a);
        assert_eq!(heap.alloc(&mem, 50).unwrap(), a);
        let big = heap.alloc(&mem, 3 << 20).unwrap();
        mem.write(big + (3 << 20) - 1, &[7]).unwrap();
    }

    #[test]
    fn aligned_allocations_keep_the_gap_before_them_free() {
        let mem = Memory::new(1 << 40).unwrap();
        let heap = Heap::default();
        let small = heap.alloc(&mem, 16).unwrap();
        let aligned = heap.alloc_aligned(&mem, 100, 0x1000).unwrap();
        assert_eq!(aligned % 0x1000, 0);
        assert_eq!(heap.size_of(aligned), Some(112));
        assert_eq!(heap.alloc(&mem, 16).unwrap(), small + 16);
    }
}
