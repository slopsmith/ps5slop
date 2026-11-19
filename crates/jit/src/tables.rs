//! Stage-1 translation tables (4 KiB granule): TTBR0 maps guest addresses
//! onto the same physical addresses, TTBR1 maps the runtime.

use crate::layout::{self, ipa};

const TABLE: u64 = 0b11;
const BLOCK: u64 = 0b01;
const PAGE: u64 = 0b11;
const ACCESS_FLAG: u64 = 1 << 10;
const INNER_SHAREABLE: u64 = 3 << 8;
/// AP[2]: read-only at EL1.
const READ_ONLY: u64 = 1 << 7;
const PXN: u64 = 1 << 53;
const UXN: u64 = 1 << 54;
const NORMAL: u64 = ACCESS_FLAG | INNER_SHAREABLE;
const ADDRESS: u64 = 0x0000_ffff_ffff_f000;
const BLOCK_SIZE: u64 = 2 << 20;
const PAGE_BYTES: u64 = 4096;

/// Tables built in one of the runtime's pools, from scratch whenever guest
/// memory changes.
pub struct Tables<'a> {
    runtime: &'a mut [u8],
    next: usize,
    end: usize,
    pub root: usize,
}

impl<'a> Tables<'a> {
    /// Starts an empty set of tables in pool `pool` (see `layout::TABLES`).
    pub fn new(runtime: &'a mut [u8], pool: usize) -> Self {
        let (next, end) = layout::TABLES[pool];
        let mut t = Self {
            runtime,
            next,
            end,
            root: 0,
        };
        t.root = t.alloc().expect("the pool holds at least one table");
        t
    }

    fn alloc(&mut self) -> Option<usize> {
        if self.next + PAGE_BYTES as usize > self.end {
            return None;
        }
        let at = self.next;
        self.next += PAGE_BYTES as usize;
        self.runtime[at..at + PAGE_BYTES as usize].fill(0);
        Some(at)
    }

    fn entry(&self, table: usize, index: u64) -> u64 {
        let at = table + index as usize * 8;
        u64::from_le_bytes(self.runtime[at..at + 8].try_into().expect("8 bytes"))
    }

    fn set(&mut self, table: usize, index: u64, v: u64) {
        let at = table + index as usize * 8;
        self.runtime[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// The next-level table under `table[index]`, made if missing.
    fn child(&mut self, table: usize, index: u64) -> Option<usize> {
        let e = self.entry(table, index);
        if e & TABLE == TABLE {
            return Some(((e & ADDRESS) - layout::RUNTIME_IPA) as usize);
        }
        let t = self.alloc()?;
        self.set(table, index, ipa(t) | TABLE);
        Some(t)
    }

    /// Maps guest `start..end` (4 KiB aligned) onto the same physical
    /// addresses, never executable. False if the pool ran out.
    pub fn map(&mut self, start: u64, end: u64, writable: bool) -> bool {
        let attrs = NORMAL | PXN | UXN | if writable { 0 } else { READ_ONLY };
        let mut va = start;
        while va < end {
            let idx = |shift: u32| (va >> shift) & 0x1ff;
            let Some(l1) = self.child(self.root, idx(39)) else {
                return false;
            };
            let Some(l2) = self.child(l1, idx(30)) else {
                return false;
            };
            if va.is_multiple_of(BLOCK_SIZE) && end - va >= BLOCK_SIZE {
                self.set(l2, idx(21), va | attrs | BLOCK);
                va += BLOCK_SIZE;
                continue;
            }
            let Some(l3) = self.child(l2, idx(21)) else {
                return false;
            };
            self.set(l3, idx(12), va | attrs | PAGE);
            va += PAGE_BYTES;
        }
        true
    }
}

/// Writes the TTBR1 tables: the runtime region as 2 MiB blocks, executable
/// at EL1.
pub fn map_runtime(runtime: &mut [u8]) {
    let mut put = |at: usize, v: u64| runtime[at..at + 8].copy_from_slice(&v.to_le_bytes());
    put(layout::TTBR1_L1, ipa(layout::TTBR1_L2) | TABLE);
    for i in 0..layout::RUNTIME_SIZE as u64 / BLOCK_SIZE {
        put(
            layout::TTBR1_L2 + i as usize * 8,
            (ipa(0) + i * BLOCK_SIZE) | NORMAL | UXN | BLOCK,
        );
    }
}
