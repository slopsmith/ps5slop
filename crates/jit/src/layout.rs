//! Where things are in the VM. See `docs/jit.md`.

/// The VM's physical address space: guest memory at IPA = guest address.
pub const IPA_BITS: u32 = 40;

/// The runtime region: physical address (in the hole `docs/memory.md`
/// reserves), size and virtual address (TTBR1).
pub const RUNTIME_IPA: u64 = 0xfe_0000_0000;
pub const RUNTIME_SIZE: usize = 1 << 30;
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
/// slots), then the FPCR that MXCSR gives, room to keep FPSR while x87
/// code runs with the host's default FPCR, MXCSR without the exception
/// flags FPSR has gathered since, and the guest clock's base
/// (`savestate::clock::base`, copied at every entry).
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
pub const MXCSR: usize = 0x270;
pub const CLOCK_BASE: usize = 0x278;
/// Two pools the TTBR0 tables are built in, alternately, so one can be
/// rebuilt while vCPUs still walk the other. Each holds 8,192 tables: a
/// fragmented map needs one per 2 MiB that is not wholly mapped, and GTA V's
/// needed 2,059 in its first Franklin mission, more than the 8 MiB pools
/// held before (see `docs/jit.md`).
pub const TABLES: [(usize, usize); 2] = [(0x10_0000, 0x210_0000), (0x210_0000, 0x410_0000)];
/// Guest RIP to translated code: `LOOKUP_ENTRIES` of (RIP, address).
pub const LOOKUP: usize = RUNTIME_SIZE - LOOKUP_ENTRIES * 16;
pub const LOOKUP_BITS: u32 = 22;
pub const LOOKUP_ENTRIES: usize = 1 << LOOKUP_BITS;
pub const CODE_CACHE: usize = 0x420_0000;
/// Blocks branch with B, which reaches 128 MiB, so each segment of the
/// cache starts with its own copy of the dispatcher (see `cache.rs`).
pub const CODE_CACHE_END: usize = LOOKUP;

pub const fn va(offset: usize) -> u64 {
    RUNTIME_VA + offset as u64
}

pub const fn ipa(offset: usize) -> u64 {
    RUNTIME_IPA + offset as u64
}

const _: () = assert!(STACKS + MAX_VCPUS * STACK_SIZE <= CONTEXTS);
const _: () = assert!(CLOCK_BASE + 8 <= CONTEXT_SIZE);
const _: () = assert!(CONTEXTS + MAX_VCPUS * CONTEXT_SIZE <= TABLES[0].0);
const _: () = assert!(TABLES[1].1 <= CODE_CACHE && CODE_CACHE_END <= LOOKUP);
