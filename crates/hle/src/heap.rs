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
        let size = size.max(1).next_multiple_of(ALIGN);
        let mut s = lock(&self.state);
        if s.end == 0 {
            s.end = HEAP_BASE;
        }
        let fit = s
            .free
            .iter()
            .find(|&(_, &len)| len >= size)
            .map(|(&a, &l)| (a, l));
        let (addr, len) = match fit {
            Some(f) => f,
            None => {
                let grow = size.next_multiple_of(CHUNK);
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
        s.free.remove(&addr);
        if len > size {
            s.free.insert(addr + size, len - size);
        }
        s.used.insert(addr, size);
        Some(addr)
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
}
