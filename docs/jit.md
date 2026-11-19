# JIT

x86-64 user code translated to AArch64 and run inside a Hypervisor.framework VM, as decided on 2026-11-19 (see [decisions.md](decisions.md) and [cpu.md](cpu.md#hypervisorframework)). The design follows 86slopper's (see its `docs/architecture.md`, "VM runtime layout"), simplified: there is no guest kernel, so no shadow paging of x86 page tables. Our own page tables mirror the regions in `crates/memory`.

Status: design, with stage A (below) in progress.

## Where the time goes now

`sample` on the interpreter while ScreenTester5 draws its first frame (2026-11-19, release build): decode and instruction fetch take about 40% of the busy time, memory checks (region lookup and permissions) about 25%, executing instructions about 20%, and copies inside HLE `memset`/`memcpy` about 10%. Translated code removes the first two.

## VM layout

**Guest physical (IPA) space**, 40 bits (the most this machine allows, 1 TiB):

- Each `Memory` region is mapped (`hv_vm_map`) at an IPA equal to its guest address, from the same host memory the interpreter uses. Unmapped guest addresses are unmapped at stage 2 too.
- `0xFE_0000_0000 .. 0xFF_0000_0000` is reserved for the runtime and never holds guest memory (see [memory.md](memory.md)).

**Stage 1**, set up by us:

- TTBR0: guest addresses, 40 bits (`T0SZ = 24`), 4 KiB granule, identity-mapped onto the IPA. A region's protection becomes the page's: no access is an invalid entry, read-only is AP read-only at EL1, read-write is read-write. Execute permission is the translator's business (it reads guest code through `Memory`).
- TTBR1: the runtime at `0xFFFF_FF80_0000_0000`: exception vectors, entry and exit code, per-vCPU state blocks and stacks, the block lookup table, the code cache, and the TTBR0 page tables themselves.

**Execution level:** translated code runs at EL1 with the runtime, so the MMU checks guest accesses (EL1 permissions), and a bad access is a synchronous exception our vectors turn into an exit. `ACTLR_EL1.EnTSO` is set on every vCPU.

**vCPUs:** one per guest thread, created on that guest thread's host thread (HVF binds a vCPU to its thread). The VM allows 64; further threads run in the interpreter.

**Memory changes** (`mmap`, `munmap`, protection changes) bump `Memory`'s generation. Before a vCPU enters the VM it brings the stage-2 mappings and TTBR0 tables up to date and flushes its TLB if anything changed. A change made while other vCPUs run kicks them out (`hv_vcpus_exit`) so they resync.

## Stages

A. **The VM itself:** mapping `Memory` at stage 2, TTBR0 tables from its regions, the runtime region, vectors, entering at EL1 and exiting with `hvc`. Tested with hand-written AArch64 that reads and writes guest memory, faults on read-only and unmapped pages, and exits.
B. **Blocks and the dispatcher:** translate a basic block of guest code at a time, look blocks up by guest RIP inside the VM, exit to the host on a miss. HLE calls are exits (the trap region stays unmapped, so a call into it faults). Instructions the translator does not handle yet end the block, and the host runs them in the interpreter.
C. **Integer instructions** with flags computed lazily, then **SSE/AVX** on NEON. The FEX ASM tests run against the JIT as they do against the interpreter.
D. **Speed:** block chaining, return prediction, in-VM paths for the hottest HLE functions, measured against the interpreter.

## Open questions

- Register assignment: 16 x86 GPRs, flags and the state pointer onto 31 AArch64 registers. Decided with stage B.
- How often HLE calls exit: ScreenTester5 calls `memset` tens of thousands of times during start-up; an exit costs about 700 ns (86slopper's measurement). In-VM `memset`/`memcpy` are the first candidates for stage D.
- Self-modifying and generated code: pages holding translated code become read-only in TTBR0 and a write faults to the host. No PS5 title known to need it yet.
