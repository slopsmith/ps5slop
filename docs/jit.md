# JIT

x86-64 user code translated to AArch64 and run on vCPUs of a Hypervisor.framework VM. Why this design is in [cpu.md](cpu.md). It is the default; `ps5slop run --interpreter` runs the interpreter only.

Code: `crates/jit` (`lib.rs` the machine and VM sync, `engine.rs` the per-thread run loop, `translate/` the translator with `vector.rs`, `x87.rs` and `tier2.rs`, `cache.rs` the code cache, `tables.rs` the stage-1 tables, `layout.rs` the runtime layout), `crates/a64` (an A64 assembler, encodings from the Arm ARM), `crates/hvf` (safe Hypervisor.framework wrappers). The design follows 86slopper's VM runtime, simplified: there is no guest kernel, so there is no shadow paging of x86 page tables.

## The VM

- **Physical (IPA) space**, 40 bits (the most this hardware allows). Each guest region the guest can read or write is mapped (`hv_vm_map`) at IPA = guest address, from the same host pages the interpreter uses. Unmapped and no-access guest ranges are unmapped at stage 2 too. `0xFE_0000_0000..0xFF_0000_0000` holds the runtime and is never guest memory.
- **Stage 1, TTBR0**: guest addresses, 4 KiB granule, identity-mapped onto the IPA. A region's protection becomes the page's (no access is an invalid entry, read-only is AP read-only). All guest pages are execute-never: translated code runs from the runtime, and the translator reads guest code through `Memory`. The tables live in two pools in the runtime, so one can be rebuilt while vCPUs still walk the other. A fragmented map needs one level-3 table per 2 MiB that is not wholly mapped, so the pools are sized for many thousands of small regions; a sync that runs out of tables is logged as an error, and memory mapped after it would never reach the VM (every access a stage-2 fault the host interprets).
- **Stage 1, TTBR1**: the 1 GiB runtime at `0xFFFF_FF80_0000_0000`: exception vectors, entry and exit stubs, native library functions, per-vCPU EL1 stacks and context blocks, the TTBR0 pools, the code cache and the lookup table.
- **EL1**: translated code runs at EL1, so the MMU checks guest accesses and a bad one is a synchronous exception that the vectors turn into an exit. `ACTLR_EL1.EnTSO` is set on every vCPU, so translated code needs no barriers for ordering.

### vCPUs and threads

HVF binds a vCPU to the thread that made it, so each guest thread creates its own vCPU on its own host thread. A VM allows 64 vCPUs (measured on an M4 Pro; `MAX_VCPUS` in `layout.rs`). A thread that finds none free runs in the interpreter and retries now and then. Mixing the two stays correct because the interpreter keeps TSO with barriers and uses real atomics.

A thread about to block in a host function calls `jit::blocking` (semaphores, event flags and queues, mutexes, condition variables, rwlocks, joins, sleeps, vblank and audio waits, the GPU submission throttle). While some running thread has no vCPU, the blocking thread gives its own back and takes a free one again at its next entry. Reason: titles start more than 64 threads, and without this a busy thread could be stuck in the interpreter for good while idle threads held vCPUs. The cost (a vCPU destroyed and made again) is paid only while more than 64 threads run.

### Memory changes and VM sync

Every change to guest regions (`mmap`, `munmap`, protection changes) bumps `Memory`'s generation. With the JIT on, the thread that makes the change syncs the VM right away (`jit::sync_eagerly`, hooked into `memory::on_change`), and vCPUs enter without comparing generations. Reason: titles change their memory map hundreds of times a second while loading, and making every running vCPU take the machine lock to sync on entry queued them behind translations. `PS5SLOP_JIT_LAZY_SYNC=1` (and the tests) sync on entry instead.

How a sync works (`Machine::sync`):

- Only the regions overlapping what changed are compared: `Memory` records the range of each change (`Memory::take_changes`), and each range grows to the whole regions and mirrored ranges it touches, old and new. Stage 2 and the tables are brought in line with those regions by sorted diffs and the slices spliced in. Reason: GTA VI maps direct memory in 64 KiB pieces, tens of thousands of times at start-up, and comparing the whole region list each time made every map cost a millisecond and stalled the title for over 30 s. The whole list is compared on the first sync of an address space, after more than 256 changes between syncs, and when a writable range becomes read-only.
- Ranges whose host pages were replaced (`Memory::take_fresh`, see [memory.md](memory.md)) are mapped again, because stage 2 still points at the old pages.
- Most changes edit the active TTBR0 tables in place while vCPUs run on them, with no TLB flush or kick. This is safe because new entries go from invalid to valid (invalid entries are never cached), and a vCPU that still holds a removed entry reaches the same IPA (stage 1 is the identity), where stage 2, which HVF updates for every vCPU, maps the current pages or nothing. A stage-2 fault is handled by the host, which interprets the instruction.
- Only a change that leaves a writable address mapped read-only takes the slow path: tables rebuilt in the other pool, TLB epoch bumped, running vCPUs kicked (`hv_vcpus_exit`) so they flush on entry.

A vCPU racing ahead of a sync sees what a vCPU already inside the VM always could: a fault on a range not mapped yet, or a range unmapped a moment ago.

## Translated code

- **Registers.** `RAX`..`R15` live in `x0`..`x15`, so exits read and write them directly. `x16` holds the guest RIP at block ends and exits; `x16`..`x18` are the dispatcher's scratch (blocks use `x17`/`x18` for flag work). `x19`..`x23` are temporaries, `x24` counts guest instructions, `x25` points at the vCPU's context block, `x26`/`x27` hold the last result and the operands' XOR (for PF and AF), `x28` the lookup table, `x29` the FS base. Constants in `translate/mod.rs`.
- **Context block** (`layout.rs`): YMM0-15, DF, the x87 state, the SSE FPCR, MXCSR, the guest clock's base. YMM registers are loaded per instruction; a 256-bit operation is two 128-bit halves; a VEX.128 write clears the upper half.
- **Flags.** NZCV holds SF, ZF, OF and CF inverted (ARM's borrow convention), so every x86 condition is one ARM condition. Narrow (8 and 16-bit) operations work on values shifted to the top of a W register so NZCV fits their width. PF and AF are computed from `x26`/`x27` only when read. Flags are computed only when a later instruction in the block, or the block's end, may read them.
- **Undefined flags match the interpreter.** Where the SDM leaves a flag undefined, the JIT produces what the interpreter does (AF kept after shifts, OF after multi-bit shifts as for a count of 1), so verify mode can compare exactly.
- **DF** is stored as 0 or all ones, so a string step is `(size ^ df) - df` without touching NZCV.
- **FPCR from MXCSR.** Every entry, and every translated `ldmxcsr`, sets FPCR: AH and NEP (FEAT_AFP) give x86's NaN, min/max and scalar-upper-lane rules (from 86slopper's `docs/sse-neon.md`), DAZ maps to FIZ, FTZ to FZ, RC to RMode. FPSR's flags fold back into MXCSR on exit (DE has no equivalent). x87 code switches to the host's default FPCR around its arithmetic, as the interpreter's own `f64` arithmetic runs, and restores the SSE FPCR before vector code, exits and bails.
- **x87** follows the interpreter's model: doubles, TOP in the status word, a tag bit per register, all kept in the context block. There is no register cache yet, so each instruction loads and stores the state it uses.
- **Time stamps.** `rdtsc` and `rdtscp` compute the interpreter's clock (`savestate::clock`) in the VM: CNTVCT_EL0, which reads `mach_absolute_time` because the vtimer offset is 0, scaled by the host timebase with the same integer arithmetic, plus the base the engine copies into the context block at every entry. Both sides read one counter with one formula, so mixing them never goes back in time, and a save-state load (which moves the base before threads run) reaches translated code.
- **Atomics.** Locked instructions and `xchg`, `xadd`, `cmpxchg` on memory are LSE atomics (`ldaddal`, `casal`, `swpal`, ...); `lock cmpxchg16b` is `caspal`. Flags are computed afterwards from the old value. `mfence` is `dmb ish`; other fences are nothing (hardware TSO covers them). `pause` and `mwaitx` are `isb`; `monitorx` arms nothing.

### Blocks, the dispatcher and chaining

A block is up to 64 guest instructions, ending at a branch, call, return or indirect jump. It ends with a jump to the dispatcher, which looks the next RIP up in a direct-mapped table of 2^22 entries (Fibonacci hash) and exits with `hvc #1` on a miss. Blocks branch to the copy of the dispatcher at the start of their own code cache segment, which a B instruction can reach.

**Lookup table protocol.** The host replaces a slot in three stores: tag cleared, code address, new tag. The dispatcher reads the tag, checks it, reads the code address, then reads the tag again and checks it before jumping. Loads stay in order under TSO, so two matching reads mean the address belongs to that tag. Reading the tag only once lets a thread jump into another address's block.

**Chaining.** A block end with a known target is patched to branch straight to the target once it is translated. Each patch records the word it replaced, so eviction can undo it. Returns and indirect jumps always go through the dispatcher.

**Host calls.** A call into the HLE trap region is a lookup miss the host recognises (the trap region is never mapped in the VM).

**Native library functions.** `memset`, `memmove` (also serving `memcpy`), `pthread_getspecific` for keys whose values HLE keeps in the thread control block, `scePthreadGetthreadid` and `pthread_self` have native versions in the runtime. The kernel registers their trap addresses in the lookup table, so guest calls reach them without leaving the VM. A fault inside one falls back to the host function, which reports it. If a block takes a native's lookup slot, the next miss there puts the native back. Verify and trace modes, and `PS5SLOP_JIT_NO_NATIVES`, use the host versions.

### Falling back to the interpreter

- **Untranslated instructions** end the block with `hvc #2`. The host runs up to 64 instructions in the interpreter, then tries translated code again. Addresses known to be untranslatable get a lookup entry pointing at a stub that exits straight to the interpreter, so they do not take the machine lock.
- **Bails.** An instruction can hand itself to the interpreter at run time: before changing any guest state or NZCV it branches to a stub that corrects the instruction count and exits with `hvc #2` at its own RIP. `div`/`idiv` do this for division by zero and quotients that do not fit; `ldmxcsr` for reserved bits (the interpreter raises #GP). Tier 2 must treat any instruction that can bail as reading all flags.
- **Faults.** A kick that lands after a vCPU took an exception but before the vector's `hvc` ran leaves the PC at the vector; `Cpu::enter` reports it as the exception it was taking, since resuming would overwrite ELR. A fault in translated code maps the host PC back to the guest instruction, which the interpreter runs again to produce the precise fault or, for a stale stage-2 or not-yet-synced mapping, the access. Translations therefore do their memory accesses before writing registers, so a faulting instruction has not changed guest state. Masked loads build the whole result before writing it.
- **Probes** (`PS5SLOP_PROBE_RIP`) keep listed addresses out of translated code, so each arrival is interpreted and logged while everything else runs at full speed.

Which instructions are translated is the code (`translate/`); there is no list here. New translations are picked by count: the per-run problem report, `profile` and `jit::stats` group interpreter fallbacks by readable opcode name (`x86::name::OpcodeKey`). See [debugging.md](debugging.md).

### Stopping a thread

A kicked vCPU's registers match no x86 instruction boundary. So when a thread must stop (a save state parks threads, an exception is pending, the process exits) and is kicked, it resumes translated code until the next exit that stores its registers (a host call, an untranslated instruction, a fault). Only after 5 s (`STOP_GRACE`) does it leave with the registers of its last entry, so process exit still ends a thread spinning in translated code. 5 s is longer than a save's stop limit (3 s), so a save never takes stale registers; it fails and retries instead. Leaving at the kick would make the thread later re-run code from stale registers while memory already holds that code's effects.

## Code cache

The code cache fills most of the runtime (about 890 MiB) in 8 MiB segments (`cache.rs`). Each segment starts with its own copy of the dispatcher and the verify-mode exit, because B reaches only 128 MiB. Code fills one segment at a time and never straddles two.

When no segment is free, the coldest full one is evicted: the lowest score, then the oldest. A sampler thread kicks every running vCPU every 10 ms; each cancelled exit adds one to the score of the segment its PC was in, and scores halve at every eviction round. Reason: per-block counters would cost a load, add and store per block and contend on blocks many threads run; sampling costs translated code nothing. Evicting a whole segment rather than flushing the cache keeps other threads running. One eviction is kept ahead so a drained segment is usually ready.

Evicting takes the segment's blocks and regions out of the lookup table and restores every patched branch into or inside it. The segment is reused only when every vCPU that was inside the VM at eviction has left once, and no vCPU still holds an address in it from its last exit (a cancelled run it will resume, or a fault RIP it will look up). A vCPU that cannot get a segment interprets until one is free. `PS5SLOP_JIT_SEGMENTS=N` uses only N segments, to test eviction.

## Tier 2

Off by default. `PS5SLOP_JIT_TIER2=N` compiles a block's region after the block ran N times; `eager` on its first run (the test mode). Phase 1 of 86slopper's tier-2 plan, synchronous:

- **Counting.** A tier-1 block starts with a five-instruction prologue counting down a word stored after the block (no flags touched); at zero it exits with `hvc #4`. The block's first word then branches to the region, and the lookup table points at it.
- **Regions** are the blocks the entry's direct branches reach on its 4 KiB page, up to 32 blocks and 512 instructions. Branches between them stay inside; other targets leave through chainable exits. Instructions tier 2 has no emitter for are emitted inline by tier 1.
- **Flag demand.** A backward pass finds the flags each instruction's successors may read (exits and possible bails count as reading all). The common integer ALU forms compute NZCV, PF and AF only when demanded, so `cmp; jcc` is one `subs` and a `b.cond`.
- **Not done:** an SSA IR with propagation and redundant load removal, register allocation beyond the pinned guest registers, loop-invariant motion, background compilation, x87 and SSE in tier 2, regions across pages.

It stays off until it shows a gain; while VM exits dominate, the quality of translated code is not the bottleneck.

## Checking the JIT

The interpreter is the oracle.

- **Tests** (`crates/jit/src/tests.rs`) run programs in the JIT and in the interpreter and compare registers, flags, YMM, MXCSR control bits, x87 state and memory. Every comparison test runs twice, with tier 2 off and eager. A new translation needs a test with its edge cases (zero and over-wide counts, NaNs, denormals, ties, signed zeros, faults at page ends), and breaking the translation on purpose should make the test fail.
- **Verify mode** (`PS5SLOP_JIT_VERIFY=1`) checks the first 4 runs of every block: the interpreter runs it with `Memory`'s write journal on, its writes are undone, the translated block runs alone, and registers, RIP, flags, x87 state and written bytes are compared. Differences are logged with the block's bytes, and execution continues from the interpreter's state. While a block is checked it exits through a stub that leaves the VM; once it passes, a copy with ordinary exits replaces it, so verify mode runs near full speed. Regions are compared step by step. Limits: writes only the JIT made are not caught; a block reading memory another thread changes between the two runs (lock words, counters) shows a false difference, and continuing from the interpreter's values there can break the program; time stamp reads differ between the two runs; verify mode does not evict, so its cache can fill on long runs. Given address ranges, only blocks starting in them are checked and the rest run as usual, which keeps a whole game fast and the log free of other code's false differences.
- **Trace mode** (`PS5SLOP_JIT_TRACE=1`) sends every block exit to the host, with no lookups or chaining in the VM, and keeps the last 64 blocks per thread; a guest fault prints them, which shows where a wild jump came from. Timing changes, so races may hide.
- **Statistics** (`RUST_LOG=jit::stats=info`) print at exit: translated and interpreted counts, exits by kind, interpreted instructions by opcode and address, host calls by function, lookup misses and table rebuilds. The `profile` control-socket command gives the same per thread while running ([debugging.md](debugging.md)).
- **Write watches** (`jit::watch`) remove write access at stage 2 for the watched pages, so translated stores fault and the interpreter runs and logs them.

## Known gaps

- **Self-modifying and generated code is not detected.** Nothing write-protects translated guest code; a block stays valid after its bytes change. No title is known to need it.
- With MXCSR.DAZ set, `minps`/`maxps` and their forms compare denormal inputs as they are (FIZ does not flush FMIN/FMAX inputs with AH set); the interpreter and hardware treat them as signed zeros.
- FRINT and FSQRT set MXCSR.IE for a signalling NaN, as x86 does; the interpreter does not. Tests save MXCSR before such inputs.
- Alignment faults of `movdqa`/`movaps` and friends are not raised.
- A fault inside x87 code may fold x87's FPSR flags into MXCSR (spurious exception flags only). Noted while reading the code, not re-checked.
- HLE host calls (mutexes and rwlocks lead) are now the largest source of VM exits. Only the natives above run inside the VM.

## Ideas, not planned

A persistent translation cache per module (like Ryujinx's PPTC): save translated blocks keyed by a hash of the module and module-relative RIPs, so later launches and evicted code skip translation, and saved counts let tier 2 compile hot regions ahead of time. It needs relocatable output (records for every branch into the runtime and every chain site) and only file-backed code. Whole-program AOT is not the aim: finding all code in an x86 binary is undecidable. The cache is derived from the game, so it lives in the user's data folder, never in the repo.
