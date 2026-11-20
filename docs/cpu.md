# CPU

How the game's x86-64 code runs on Apple Silicon. The strategy is decided (see [decisions.md](decisions.md)); the JIT's shape is not.

## What has to run

- **ISA:** user-mode x86-64 as a Zen 2 runs it: integer with REX, SSE to SSE4.2, AVX and AVX2 (VEX), and probably FMA3, BMI1/2, F16C, LZCNT/POPCNT, MOVBE. x87 only where compilers still emit it. No privileged instructions, no segmentation beyond the FS base (TLS).
- **How much of it games use:** unknown. Nobody has published an instruction-mix survey of PS5 binaries. A local scan of a dumped game (aggregate counts per instruction class, nothing committed) will answer it; [legal.md](legal.md) allows that.
- **Threads:** games run dozens of threads, so the memory model matters: x86 is TSO, ARM is weaker.

## The options

Facts below are from a research pass on 2026-11-19 plus a few measurements on vic's M4 Pro (macOS 27.3); "[M]" marks those.

### Rosetta 2 (build the emulator as an x86_64 process)

- Translates AVX, AVX2, FMA, F16C and BMI correctly, though CPUID hides them unless `ROSETTA_ADVERTISE_AVX=1` [M]. Apple's [Rosetta doc](https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment) says AVX/AVX2 are supported, AVX-512 is not.
- Gives 4 KiB pages [M] and hardware TSO for free, and FS can be pointed at TLS with `i386_set_ldt` (shadPS4 does this). This is the route KytyPS5, SharpEmu-mac and shadPS4 take on macOS.
- Known bugs that emulators patch around: `VCMPSS` with some predicates, `VMASKMOVPS` across a page fault (shadPS4's `cpu_patches.cpp`).
- **Ruled out as a foundation.** Apple's doc: Rosetta is available "through macOS 27" as a general-purpose tool, after which only "a subset ... aimed at supporting older unmaintained gaming titles" stays. This Mac already runs macOS 27. An arm64 process also cannot call Rosetta on a chunk of code.
- **Still useful as an oracle** while it exists: run the same x86-64 snippet under Rosetta and under our interpreter, and compare.

### FEX-Emu (FEXCore as a library)

- MIT, mature, monthly releases (FEX-2611.1). AVX/AVX2/FMA/F16C/BMI since FEX-2407. Hangover (Windows on ARM) and Prosperity (Linux/Android) already embed FEXCore; Prosperity even handles PS4/PS5 `syscall` and FS itself.
- On Apple CPUs (no SVE) a 256-bit operation is split into two 128-bit halves with the upper halves in memory: `vaddps ymm` is 5 A64 instructions, the median ymm operation 5, `vpgatherdd` 46 (FEX's InstructionCountCI).
- **No macOS support.** The maintainers have said they have no plans for it. KyTx claims an in-process Darwin port but has not published it.
- **TSO is in software** (acquire/release accesses, LRCPC where present, backpatching heuristics), because macOS gives a normal arm64 process no way to turn hardware TSO on. No published number for the cost.
- **Address layout.** FEX runs guest code at host addresses. In a native arm64 macOS process the low 4 GiB is `__PAGEZERO`, and shrinking it gets the binary killed at exec [M]; the dyld shared cache sits around `0x1_8000_0000..0x3_7000_0000` and a 385 GiB guard around `0xF_C000_0000..0x70_0000_0000` [M]. If games need the PS4-style layout (executable at `0x400000`, user area from `0x10_0000_0000`, see [memory.md](memory.md)), that collides. Unknown whether PS5 games need fixed addresses at all.
- Costs: a large C++ dependency, a macOS port we would have to write and maintain alone, and a TSO and layout story we do not control.

### Our own translator, interpreter first

- 86slopper already has an x86-32 decoder, interpreter, a tier-1 JIT with NZCV flag mapping and SSE-to-NEON, a tier-2 IR design, and the testing habit of an interpreter as the oracle. Much of the thinking carries over; the code needs 64-bit mode, REX, VEX, 16 GPRs and 256-bit vectors.
- 16 ymm registers fill all 32 NEON registers, so the upper halves will mostly live in memory, as in FEX. AVX2 on NEON is where most of the work is.
- It gives us control of the things FEX does not: where guest memory sits, how TSO is provided, and how HLE calls leave translated code.
- Cost: large. FEX took about five years to reach AVX2 on 128-bit hardware, with much wider goals than ours. An interpreter with enough coverage for small titles is months of evenings. The JIT is likely the biggest single piece of the project.

## Hypervisor.framework

There is no guest kernel, so we do not need HVF the way 86slopper does. It still offers three things nothing else on macOS gives a normal process:

1. **Hardware TSO.** `HV_SYS_REG_ACTLR_EL1` (macOS 15 SDK) exposes exactly the `EnTSO` bit; the header says setting it makes the vCPU use TSO. Public API. 86slopper measured an in-VM exception at ~17 ns with TSO off and ~31 ns with it on.
2. **Our own address space.** Our stage-1 page tables decide what guest addresses mean, so guest VA can be exactly what the game expects, with 4 KiB or 16 KiB granules, and nothing from macOS in the way.
3. **Cheap faults.** Write protection for self-modifying code and GPU dirty tracking is a fault handled inside the VM, not a signal.

Against it:

- **HLE calls cost a VM exit** (~700 ns round trip measured in 86slopper) unless the hot ones are handled inside the VM. Games call libc and libkernel a lot (`memcpy`, mutexes, time). How often is unknown until we can run a real title.
- **Each guest thread needs a vCPU** (or we multiplex them ourselves), and blocking calls must leave the VM. Measured 2026-11-19 on vic's M4 Pro, macOS 27 (a probe on 86slopper's `hvf` crate): at most **64 vCPUs** per VM, all 64 created concurrently without trouble; creating and destroying a vCPU costs ~15 µs, a host thread ~17 µs, so a new guest thread with its own vCPU costs ~37 µs. The largest IPA size is 40 bits (1 TiB).
- More moving parts before anything runs.

Without HVF, translated code runs in our process with guest addresses offset by a base register (one extra add on some accesses), TSO through acquire/release accesses, and faults through signals.

## Plan

1. **Interpreter first,** in Rust, user-mode x86-64 only, running in our process with a base-offset address space. It runs our own test programs from day one, it is the oracle for everything after it, and it lets the loader and HLE work start without waiting for a JIT. Guest memory accesses that may race use acquire/release ordering.
2. **Decide the JIT later, with data,** once a commercial title reaches its first GPU submission under the interpreter. (Decided earlier than planned, on 2026-11-19: see below.) By then we can measure: interpreter speed, the instruction mix, the rate of HLE calls (which decides whether HVF's exit cost hurts), and whether games need fixed addresses.
3. **Our own JIT.** A JIT is necessary; only its timing is deferred. Whether it runs inside an HVF VM (for TSO and the address layout, with the hottest HLE functions handled inside the VM) is decided at the decision point. Rosetta is an oracle only.

**Decision (2026-11-19): the JIT runs inside an HVF VM.** Translated code runs on vCPUs with `EnTSO` set, under our own stage-1 page tables that map guest addresses one to one. Each guest thread gets a vCPU on its own host thread; past the 64-vCPU limit, further threads run in the interpreter, which keeps TSO through barriers, so mixing the two stays correct. HLE calls leave the VM; hot ones may get in-VM paths once measured. The interpreter stays as the fallback for anything not yet translated, and as the oracle. See [decisions.md](decisions.md).

FEX is not a fallback: it was not designed for macOS, and forcing it in would mean porting and maintaining it alone. We may still read it for ideas (MIT) and use its tests as an oracle.

The CPU sits behind one interface (run a thread until it calls into HLE, faults or is interrupted), so the interpreter and the JIT can be swapped without touching the loader or HLE code.

## The interpreter today

`crates/x86`: a decoder for 64-bit mode and an interpreter that runs one instruction at a time. It assumes user mode and long mode throughout, so there are no segments beyond the FS and GS bases, no paging and no privilege changes; 86slopper's interpreter, which covers real, protected and kernel mode, shares only its general shape (decode, then execute with eagerly computed flags).

- **Decoder:** legacy prefixes, REX, the one-byte, `0F`, `0F38` and `0F3A` maps, ModRM and SIB, RIP-relative addressing, `67` for 32-bit addresses. VEX, EVEX and XOP are recognised and reported as unsupported.
- **SSE/SSE2:** XMM registers (stored 256 bits wide for AVX later) and MXCSR; moves, logic, integer SIMD, shuffles and unpacks, packed and scalar float arithmetic, conversions, compares, `ldmxcsr`/`stmxcsr`, fences. Float results follow x86's NaN rules (first-operand propagation, the negative "indefinite" NaN for invalid operations); MXCSR's rounding mode applies to conversions but not yet to arithmetic, and exception flags are not raised.
- **VEX and AVX/AVX2:** the decoder reads 2- and 3-byte VEX prefixes. VEX forms of SSE instructions take a separate first source (VEX.vvvv), clear the upper half of the destination at 128 bits, and run 256-bit in-lane operations as two 128-bit halves; legacy SSE keeps the upper half. Only the explicitly aligned moves check alignment under VEX. Cross-lane operations (permutes, broadcasts, insert/extract, sign and zero extension, F16C conversions, masked moves, gathers) work on the whole register.
- **SSSE3, SSE4.1, SSE4.2's CRC32, MOVBE, FMA3, BMI1/BMI2.** FMA uses Rust's fused `mul_add`; NaN results take the first NaN operand, which matches the tests but is not checked against the SDM's exact rule. MXCSR's DAZ and FTZ bits apply to SSE/AVX arithmetic (a FEX test checks DAZ).
- **x87:** the whole unit except the BCD loads and stores (`fbld`, `fbstp`), computing in double precision: 80-bit operands are rounded to `f64` when loaded (see `decisions.md`). Of the exception flags only IE (invalid operation) is tracked; exceptions are never raised.
- **Instructions:** the integer subset compiled C uses: moves and extensions, the ALU group, shifts and rotates (including `shld`/`shrd`), multiply and divide, `push`/`pop`/`call`/`ret`/`leave`, branches and loops, `setcc`/`cmovcc`, bit tests and scans (`bsf`, `bsr`, `tzcnt`, `lzcnt`, `popcnt`), `xchg`, `xadd`, `cmpxchg`, `bswap`, string instructions with `rep`, `cpuid` (see below), `rdtsc` (host time). Not yet: most of x87, MMX, SSE3's horizontal adds, SSE4.2's string compares, AES, `enter`, `cmpxchg16b`.
- **Threads:** each guest thread runs the interpreter on its own host thread. Loads and stores carry TSO barriers, and locked instructions (`lock`-prefixed ALU operations, `inc`/`dec`/`not`/`neg`, `xadd`, `cmpxchg`, `bts`/`btr`/`btc`, and `xchg` with memory) are atomic updates of host memory; a test runs `lock inc` from four threads and checks that no increment is lost (and fails when atomicity is removed). See [memory.md](memory.md#threads-and-memory-ordering).
- **Faults:** a faulting instruction leaves the state as it was before it, and the run loop gets the fault and its address. Unimplemented instructions are reported by map and opcode.
- **Memory:** goes through `crates/memory` with a guest-protection check on every access; nothing is optimised yet.

- **CPUID:** reports a Zen 2 core (family 17h) with the features Zen 2 has (SSE to SSE4.2, SSE4a, AVX, AVX2, FMA, F16C, BMI1/2, LZCNT, POPCNT, MOVBE, AES, PCLMULQDQ, SHA, ADX, RDRAND/RDSEED, CLZERO, ...), bit positions from AMD's public CPUID specification. The PS5's own CPUID values are unknown. Reporting features we do not implement yet is deliberate: programs then hit the missing instruction and we see it, instead of quietly taking a fallback path.

## Correctness oracle: FEX's ASM tests

There is no public x86-64 suite in the style of SingleStepTests (those stop at the 80386). The best public oracle we found is FEX-Emu's set of 2,291 handwritten x86-64 test programs (`unittests/ASM`, MIT). Each one runs a few instructions and states the register values it must end with. FEX runs the same tests natively on an x86-64 machine (its "host" runner on a Zen+ CI box), apart from a short list of tests it disables or knows to fail there, so the expected values have been checked on real hardware, not just against FEX itself.

`scripts/fetch-references` puts them in `references/src/FEX` (pinned commit, not committed); `ps5slop-asmtest` (in `nix develop`, which provides nasm) assembles each test as FEX does and runs it the way FEX's host runner does: code at `0x10000`, all registers zero, RFLAGS `0x202`, scratch memory at `0xE000_0000`, plus the memory each test asks for, until `hlt`. Then it compares the registers the test names. It skips FEX's disabled and known-failing lists, 32-bit tests, tests that need features Zen 2 lacks (3DNow!, AVX-VNNI, VAES, SHA512, ...) and tests that make Linux system calls. Results go to `target/asmtest/report.tsv`.

**Measured on 2026-11-19 (FEX `be66e03d`):** of 2,291 tests, 70 are skipped and 2,221 run.

| Interpreter | Pass | Wrong, fault or timeout | Unimplemented |
| --- | --- | --- | --- |
| Integer only | 568 (25.6%) | 0 | 1,653 |
| + SSE/SSE2 (XMM registers checked) | 777 (35.0%) | 0 | 1,444 |
| + x87 control words | 784 (35.3%) | 0 | 1,437 |
| + atomic locked instructions | 784 (35.3%) | 0 | 1,437 |
| + VEX forms of SSE/SSE2, 128 and 256 bits | 1,109 (49.9%) | 0 | 1,112 |
| + SSSE3, SSE4.1, AVX2-only, FMA, BMI | 1,530 (68.9%) | 0 | 691 |

Latest run, by area (passed / run): VEX 432/482, SSELanePreservation 204/231, `0F38` 57/86, `0F3A` 26/45, Primary 190/197, PrimaryGroup 151/156, OpSize 103/110, TwoByte 144/219, X87 3/213, X87_F64 3/150. Still missing: x87 (about 380 tests blocked on `D9`/`DB`/`DD`), MMX (`0F 6F` without a prefix, about 100), the SSE4.2 string compares, AES and `cmpxchg16b`. Every wrong result the suite found along the way was fixed: `vshufpd` taking the upper half's selector bits from the wrong place, DAZ being ignored, an operator-precedence slip in `pblendvb`/`blendvps`, `pextr`/`pinsr` reading a 128-bit memory operand they do not have, `dpps` summing in the wrong order (the hardware adds pairwise), and a gather with `xmm4` as index (index 4 means "none" in ordinary SIB addressing but not in VSIB).

The suite has since caught one more: rewriting `xadd` for atomicity wrote a same-register destination in the wrong order (`xadd rax, rax` must leave the sum). The first run found four bugs that unit tests had not: string instructions and `loop` ignored the `67` address-size prefix, a REX byte followed by another prefix was rejected instead of ignored, `shld`/`shrd` with a zero count did not zero-extend a 32-bit register, and CPUID reported no features. All four are fixed.

Unchecked so far: tests whose expected values include MMX registers.

**x87 (2026-11-19):** 1,780 pass, 0 wrong, 281 unimplemented, 226 skipped. Our x87 computes in double precision, as FEX's reduced-precision mode does, so the runner skips the `X87/` tests that have an `X87_F64` twin (FEX runs that one in this mode), plus `precision_test_*` and `FXAM_Simple`, which need 80-bit values. Adding x87 also let `vcmp*_full` run, which showed that VEX compare predicates 8-15 and 24-31 ignored their "unordered is true" bit; fixed.

**Other tests:** unit tests whose encodings come from llvm-mc, and our test programs (`scripts/run-test-programs`).

## Correctness oracle: Rosetta 2, by fuzzing

`ps5slop-fuzz [STATES [SEED]]` (in `crates/asmtest`) makes one register-form encoding (ModRM mod = 3, no operand naming RSP) for every opcode of the VEX maps (both L, both W, all four `pp`), the legacy `0F`, `0F 38` and `0F 3A` maps (with and without `66`/`F3`/`F2` and REX.W) and the safe primary opcodes, with random registers. Our decoder gives each one's length. `crates/asmtest/fuzz/harness.c`, built with `clang -arch x86_64`, runs them natively under Rosetta 2 with STATES random register states each (all GPRs, YMM0-15, arithmetic flags), and the tool runs the same states through our interpreter and compares GPRs, YMM registers and the arithmetic flags. Results go to `target/fuzz/report.txt`. Control flow, stack, string, I/O, x87, system and timing instructions are left out, as is `maskmov` (it stores to `[rdi]`). Only the interpreter is checked; the JIT has `PS5SLOP_JIT_VERIFY` for that.

Rosetta executes AVX2, FMA, F16C and BMI1/2 even though its CPUID hides them, so this covers the instructions FEX's suite reaches only partly. It is not hardware, though. Measured differences where the Intel SDM sides with us, not Rosetta (2026-11-20):

- `vpsllvq`/`vpsrlvq` keep the value when a count has bit 63 set (the SDM: any count above 63 gives 0).
- `cmpxchg` sets only OF, and zero-extends a 32-bit register destination when the compare fails (hardware leaves it untouched, as QEMU's 2022 fix describes).
- Legacy SSE `cmpps`/`cmpsd` use the whole immediate as the predicate (the SDM: only bits 2:0 count for the legacy forms).
- `vcvtps2ph` swaps round-down and round-up.
- `movsxd` without REX.W sign-extends into 64 bits.

The flags the SDM leaves undefined (after `imul`, shifts, `bsf`/`bsr`, `andn`, `ptest`'s AF/PF/SF/OF, ...) differ too and are not bugs. 16-bit `shld`/`shrd` with a count above 16 are undefined as well.

**First run (2026-11-20, seeds 0x5eed1234abcd0001 and 12345, 8 and 16 states):** 18,376 encodings, of which about 2,200 run in both. Real bugs it found, all fixed:

- VEX.256 shifts by an XMM count (`vpsrl`/`vpsra`/`vpsll` `w`/`d`/`q`) with the count register also the destination: the two 128-bit halves ran low first, so the upper half read an already shifted count. Halves now run upper first. This broke Minecraft's CPU texture tiling (see [debugging.md](debugging.md)).
- `lahf` and 8-bit `div`/`idiv` with a REX prefix used SPL instead of AH.
- Legacy `cmpps`/`cmppd`/`cmpss`/`cmpsd` used five predicate bits instead of three.

About 1,900 encodings that Rosetta rejects with `#UD` (wrong VEX.L or W for the instruction, mostly) run in our interpreter; games do not execute those, so this is left alone.

**Added for Minecraft's world creation (2026-11-20), each checked with the fuzzer:** `haddps`/`haddpd`/`hsubps`/`hsubpd` (all forms), the VEX.256 forms of `movmskps`/`movmskpd`, `pmovmskb`, `cvtdq2pd`, `cvttpd2dq` and `cvtpd2dq`, and `rcpps`/`rcpss`/`rsqrtps`/`rsqrtss`. We compute `rcp` and `rsqrt` exactly and round to f32; hardware returns an approximation good to about 12 bits (Rosetta's results have the low mantissa bits cleared), so results differ in the low bits; denormal inputs count as zero, as on hardware. Scalar FMA and `roundss`/`roundsd` now ignore VEX.L, as the SDM says.

**SSE4a (2026-11-20):** `extrq` and `insertq` (66/F2 0F 78 with two immediates, and 66/F2 0F 79), which Zen 2 has and Minecraft uses. Written from AMD's manual: a length of 0 means 64, and bits 127:64 of the destination, which AMD leaves undefined, are kept. Rosetta cannot check these: Intel CPUs do not have SSE4a.

