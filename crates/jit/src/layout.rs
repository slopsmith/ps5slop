//! Where things are in the VM. See `docs/jit.md`, "VM layout".

/// The VM's physical address space: guest memory at IPA = guest address.
pub const IPA_BITS: u32 = 40;

/// The runtime region: physical address (in the hole `docs/memory.md`
/// reserves), size and virtual address (TTBR1).
pub const RUNTIME_IPA: u64 = 0xfe_0000_0000;
pub const RUNTIME_SIZE: usize = 64 << 20;
pub const RUNTIME_VA: u64 = 0xffff_ff80_0000_0000;

pub const VECTORS: usize = 0;
pub const CODE: usize = 0x800;
pub const CODE_END: usize = 0x1_0000;
pub const TTBR1_L1: usize = 0x1_0000;
pub const TTBR1_L2: usize = 0x1_1000;
/// One EL1 stack per vCPU.
pub const STACKS: usize = 0x2_0000;
pub const STACK_SIZE: usize = 0x1000;
pub const MAX_VCPUS: usize = 64;
/// Two pools the TTBR0 tables are built in, alternately, so one can be
/// rebuilt while vCPUs still walk the other.
pub const TABLES: [(usize, usize); 2] = [(0x10_0000, 0x90_0000), (0x90_0000, 0x110_0000)];
/// Guest RIP to translated code: `LOOKUP_ENTRIES` of (RIP, address).
pub const LOOKUP: usize = 0x120_0000;
pub const LOOKUP_BITS: u32 = 16;
pub const LOOKUP_ENTRIES: usize = 1 << LOOKUP_BITS;
pub const CODE_CACHE: usize = 0x140_0000;
pub const CODE_CACHE_END: usize = RUNTIME_SIZE;

pub const fn va(offset: usize) -> u64 {
    RUNTIME_VA + offset as u64
}

pub const fn ipa(offset: usize) -> u64 {
    RUNTIME_IPA + offset as u64
}

const _: () = assert!(STACKS + MAX_VCPUS * STACK_SIZE <= TABLES[0].0);
const _: () = assert!(TABLES[1].1 <= LOOKUP && LOOKUP + LOOKUP_ENTRIES * 16 <= CODE_CACHE);
