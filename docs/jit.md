# JIT

x86-64 user code translated to AArch64 and run inside a Hypervisor.framework VM, as decided on 2026-11-19 (see [decisions.md](decisions.md) and [cpu.md](cpu.md#hypervisorframework)). The design follows 86slopper's (see its `docs/architecture.md`, "VM runtime layout"), simplified: there is no guest kernel, so no shadow paging of x86 page tables. Our own page tables mirror the regions in `crates/memory`.

Status (2026-11-19): stages A and B work, C and D are under way, and the JIT is what `ps5slop run` uses by default (`--interpreter` turns it off). ScreenTester5 and DriftCheck run correctly under it, with 0 differences from the interpreter in verify mode over 30 to 40 s of each. Measured with release builds:

- ScreenTester5 draws its menu 0.2 s after start (2.5 s with the interpreter), and its RGB flash mode runs at 60 frames a second, the vblank rate (no frame at all in 5 s with the interpreter).
- DriftCheck flips about 30 frames a second (none in 6 s with the interpreter). Its frames take a little over 1/60 s, so this is its own work, not exits.
- Both now interpret only a few thousand instructions per second.

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

**Memory changes** (`mmap`, `munmap`, protection changes) bump `Memory`'s generation. Mapping or unmapping replaces the host pages under a range (a fresh `MAP_FIXED` mapping, so freed memory reads as zeros), and the VM's stage 2 still points at the old pages; `Memory::take_fresh` lists those ranges, and the sync maps them again. Without this, Minecraft's start-up wrote to pages the interpreter and host functions never saw (found with verify mode). Before a vCPU enters the VM it brings the stage-2 mappings and TTBR0 tables up to date and flushes its TLB if anything changed. A change made while other vCPUs run kicks them out (`hv_vcpus_exit`) so they resync.

## Translated code

- x86 `RAX`..`R15` live in `x0`..`x15`, so exits read and write them directly.
- `x16` holds the guest RIP at block ends and exits; `x16`..`x18` are the dispatcher's scratch (blocks use `x17`/`x18` for flag work too).
- `x19`..`x23` are the translator's temporaries, `x24` counts translated guest instructions, `x26`/`x27` hold the last result and the operands' XOR (for PF and AF, computed when needed), `x28` points at the lookup table and `x29` holds the FS base.
- NZCV holds SF, ZF and OF, and CF inverted (ARM's borrow convention), so every x86 condition is one ARM condition (`jbe` is `LS`, `ja` is `HI`). Narrow (8- and 16-bit) operations work on values shifted to the top of a W register, so NZCV fits their width.
- Flags are only computed when a later instruction in the block (or the block's end) may read them. Where the interpreter keeps undefined flags (AF after shifts, OF after multi-bit shifts), the JIT keeps them too, so the two agree exactly.
- A block ends with a jump to the dispatcher, which looks the next RIP up in a direct-mapped table (2^16 entries, Fibonacci hash) and exits with `hvc #1` on a miss. Instructions not translated yet end the block with `hvc #2`; the host runs them in the interpreter (up to 64, until one could start a block) and re-enters. A call into the HLE trap region is a lookup miss the host recognises.
- A fault in translated code maps the host PC back to the guest instruction, which the interpreter runs again. Translations do their memory accesses before writing registers, so a faulting instruction has not changed guest state.

Translated so far:

- General: `mov` (all forms), `movzx`, `movsx`, `movsxd`, `lea`, `add`/`or`/`and`/`sub`/`xor`/`cmp`/`test` (all sizes), `inc`/`dec`, `shl`/`shr`/`sar` by a constant, two-operand `imul`, `cmovcc`, `setcc`, `push`/`pop`, `jmp`, `jcc`, `call`, `ret`, `nop`; BMI `andn`, `bextr`, `shlx`/`shrx`/`sarx`, `rorx`; `bt` (register form, and with an immediate); `mov r8, imm`; `lfence`/`sfence`/`clflush` as nothing and `mfence` as `dmb ish` (hardware TSO covers the rest).
- Vector (`translate/vector.rs`), as legacy SSE, VEX.128 and VEX.256: `movdqa`/`movdqu`/`movaps`/`movups`/`movapd`/`movupd`, `movd`/`movq`, the logic operations (integer and FP forms), packed add, subtract (also saturating), compare, min/max, average, `pmullw`, `pmulhw`/`pmulhuw`, `pmuludq`, `pmaddwd`, `psadbw`, unpacks and packs, shifts by a constant (including `pslldq`/`psrldq`), `pshufd`, `pshufb`, `pmovmskb` (128-bit), `vzeroupper`/`vzeroall`; AVX2 `vpbroadcast{b,w,d,q}` and the variable shifts `vpsllv`/`vpsrlv`/`vpsrav`. YMM registers live in the vCPU's context block in the runtime region (`x25` points at it) and are loaded per instruction; a 256-bit operation is two 128-bit halves; a VEX.128 write clears the upper half. Most NEON choices and the "three same" opcode table come from 86slopper's `translate/sse.rs`. Alignment faults of `movdqa`/`movaps` are not raised.

- Floating point (legacy SSE, VEX.128 and VEX.256): `add`, `sub`, `mul`, `div`, `min`, `max` in their `ps`, `pd`, `ss` and `sd` forms, `movss`/`movsd`, `ucomiss`/`comiss`/`ucomisd`/`comisd`, `cvtsi2ss`/`cvtsi2sd`, `cvtdq2ps`, `cvtps2dq`, `cvttps2dq`, `vbroadcastss`/`vbroadcastsd`, `vblendvps`/`vblendvpd`/`vpblendvb`. FPCR is set from MXCSR on every entry: AH and NEP (FEAT_AFP), which give x86's NaN, min/max and scalar-upper-lane rules (86slopper's `docs/sse-neon.md`), FIZ from DAZ, FZ from FTZ, and the rounding mode from RC. FPSR's flags are folded back into MXCSR on exit (DE has no equivalent).
- Also: `not`, `neg`, three-operand `imul`, `cdqe`/`cwde`.

Not yet: `adc`/`sbb`, parity conditions, string instructions, `lock`, `bts`/`btr`/`btc` and `bt` on memory with a register offset, `cmpps`/`cmpss`, `insertps`/`extractps`, `pinsr`/`pextr`, `popcnt`/`lzcnt`, cross-lane AVX2 (`vperm*`, `vinserti128`...), and x87.

**Library functions in the VM.** `memset`, `memcpy` and `memmove` have native versions in the runtime region. The kernel registers their trap addresses when the JIT is on, so the dispatcher sends guest calls straight to them, and they return as the guest's `ret` would. A fault inside one falls back to the host function at the same address, which reports it as before. Verify mode uses the host versions. ScreenTester5's flash mode called `memset` 3.2 M times in 5 s; each was a VM exit before.

Block ends with a known target are chained: the branch to the dispatcher is patched to go straight to the target block once it is translated (not in verify mode).

## Statistics mode

`PS5SLOP_JIT_STATS=1` prints, when the process ends, the translated and interpreted instruction counts, exits by kind, the interpreted instructions by opcode and by address (with their bytes, for `llvm-mc --disassemble`), and host calls by function. This is how the next instructions to translate are chosen.

## Verify mode

`PS5SLOP_JIT_VERIFY=1` checks the first 4 runs of every block against the interpreter: the interpreter runs the block with `Memory`'s write journal on, its writes are undone, the translated block runs alone, and registers, RIP, arithmetic flags and the bytes the interpreter wrote are compared. Differences are printed with the block's bytes, and execution carries on from the interpreter's state. The lookup table stays empty in this mode, so every block exits by itself. Writes only the JIT made are not caught.

## Stages

A. (done) **The VM itself:** mapping `Memory` at stage 2, TTBR0 tables from its regions, the runtime region, vectors, entering at EL1 and exiting with `hvc`. Tested with hand-written AArch64 that reads and writes guest memory, faults on read-only and unmapped pages, and exits.
B. (done) **Blocks and the dispatcher:** translate a basic block of guest code at a time, look blocks up by guest RIP inside the VM, exit to the host on a miss. HLE calls are exits (the trap region stays unmapped, so a call into it faults). Instructions the translator does not handle yet end the block, and the host runs them in the interpreter.
C. **Integer instructions** with flags computed lazily, then **SSE/AVX** on NEON. The FEX ASM tests run against the JIT as they do against the interpreter.
D. **Speed:** block chaining, return prediction, in-VM paths for the hottest HLE functions, measured against the interpreter.

## Open questions

- Self-modifying and generated code: pages holding translated code become read-only in TTBR0 and a write faults to the host. No PS5 title known to need it yet.
