//! Where things are in the VM. See `docs/jit.md`, "VM layout".

/// The VM's physical address space: guest memory at IPA = guest address.
pub const IPA_BITS: u32 = 40;

/// The runtime region: physical address (in the hole `docs/memory.md`
/// reserves), size and virtual address (TTBR1).
pub const RUNTIME_IPA: u64 = 0xfe_0000_0000;
pub const RUNTIME_SIZE: usize = 256 << 20;
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
/// One context block per vCPU: YMM0-15, then the direction flag (0 or all
/// ones, so a string step is `(size ^ df) - df`), then the x87 state (R0-R7
/// as doubles, and the status word, control word and tag bits as 64-bit
/// slots), then the FPCR that MXCSR gives and room to keep FPSR while x87
/// code runs with the host's default FPCR.
pub const CONTEXTS: usize = 0x6_0000;
pub const CONTEXT_SIZE: usize = 0x400;
pub const YMM_SIZE: usize = 0x200;
pub const DF_OFFSET: usize = 0x200;
pub const X87_ST: usize = 0x208;
pub const X87_FSW: usize = 0x248;
pub const X87_FCW: usize = 0x250;
pub const X87_FTAG: usize = 0x258;
pub const FPCR_SSE: usize = 0x260;
pub const FPSR_KEPT: usize = 0x268;
/// Two pools the TTBR0 tables are built in, alternately, so one can be
/// rebuilt while vCPUs still walk the other.
pub const TABLES: [(usize, usize); 2] = [(0x10_0000, 0x90_0000), (0x90_0000, 0x110_0000)];
/// Guest RIP to translated code: `LOOKUP_ENTRIES` of (RIP, address).
pub const LOOKUP: usize = RUNTIME_SIZE - LOOKUP_ENTRIES * 16;
pub const LOOKUP_BITS: u32 = 22;
pub const LOOKUP_ENTRIES: usize = 1 << LOOKUP_BITS;
pub const CODE_CACHE: usize = 0x120_0000;
/// Blocks branch to the dispatcher (at `CODE`) with B, which reaches 128 MiB.
pub const CODE_CACHE_END: usize = 0x800_0000;

pub const fn va(offset: usize) -> u64 {
    RUNTIME_VA + offset as u64
}

pub const fn ipa(offset: usize) -> u64 {
    RUNTIME_IPA + offset as u64
}

const _: () = assert!(STACKS + MAX_VCPUS * STACK_SIZE <= CONTEXTS);
const _: () = assert!(FPSR_KEPT + 8 <= CONTEXT_SIZE);
const _: () = assert!(CONTEXTS + MAX_VCPUS * CONTEXT_SIZE <= TABLES[0].0);
const _: () = assert!(TABLES[1].1 <= CODE_CACHE && CODE_CACHE_END <= LOOKUP);
const _: () = assert!(CODE_CACHE_END - CODE <= 1 << 27);
