# CPU

How the game's x86-64 code runs on Apple Silicon: the decoder and interpreter in `crates/x86`, and how we check them. The translator and the VM it runs in are in [jit.md](jit.md); the address space is in [memory.md](memory.md).

## What has to run

User-mode x86-64 as a Zen 2 runs it: integer code with REX, SSE to SSE4.2, SSE4a, AVX, AVX2, FMA3, BMI1/2, ADX, F16C, LZCNT, POPCNT, MOVBE, AES-NI, PCLMULQDQ, SHA, CLZERO, MONITORX/MWAITX, and x87 where compilers still emit it (`long double` in C++ runtimes). No privileged instructions and no segmentation beyond the FS and GS bases (FS is the thread pointer). Games run dozens of threads, so the memory model matters: x86 is TSO, ARM is weaker.

The PS5's own CPUID values are unknown. That the PS5 CPU is a Zen 2 derivative is public knowledge; the exact feature set is assumed.

## Why our own translator, in an HVF VM

- **Rosetta 2** is general-purpose only through macOS 27 (Apple's [Rosetta doc](https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment)), and an arm64 process cannot hand it a piece of code. We use it only as a test oracle (below).
- **FEX-Emu** has no macOS support and its maintainers do not plan any. It assumes guest address = host address, which collides with macOS's layout (see [memory.md](memory.md)), and provides TSO in software. Porting and maintaining it alone would cost more than it saves. We read it for ideas (MIT) and use its tests.
- **Our own translator** gives us control of where guest memory sits, how TSO is provided and how host calls leave translated code. 86slopper (vic's x86-32 emulator) supplied the design habits: an interpreter as the oracle, a tier-1 JIT with NZCV flag mapping, SSE on NEON.
- **Hypervisor.framework** gives a normal macOS process three things nothing else does: hardware TSO (`ACTLR_EL1.EnTSO`, public API since the macOS 15 SDK), page tables of our own so guest addresses can be anything, and faults handled as VM exits instead of signals. The costs: a host call is a VM exit, and a VM has at most 64 vCPUs. See [jit.md](jit.md).

The CPU sits behind one interface (run a thread until it calls the host, faults or is stopped), so the interpreter and the JIT can be swapped without touching the loader or HLE.

## The interpreter

`crates/x86`: `decode.rs` (64-bit mode decoder), `interp.rs` with `interp/sse.rs`, `ext.rs` and `x87.rs`, `cpuid.rs`, `name.rs` (readable opcode names for reports). It runs one instruction at a time with eagerly computed flags. Semantics follow the Intel SDM, volume 2 (AMD's manual for SSE4a).

It is three things at once: the oracle for the JIT, the fallback for whatever the JIT does not translate, and the CPU for threads that have no vCPU. So it must be correct first and fast second.

Rules that matter when you change it:

- **Faults are precise.** A faulting instruction leaves the state as it was before it; the run loop gets the fault and its address. The JIT relies on this to re-run a faulting instruction in the interpreter.
- **TSO by barriers.** Every guest load is followed by a load barrier, every store preceded by a store barrier. That allows exactly the reordering x86-TSO allows (a store with a later load). Aligned 1, 2, 4 and 8-byte accesses are single host accesses, so they do not tear. Reason: the interpreter runs on host threads next to translated code, and a normal macOS process has no hardware TSO.
- **Locked instructions are real atomics** on host memory (`Memory::atomic_update`, a compare-and-swap loop), so they exclude translated code on other threads, which uses LSE atomics on the same words. `cmpxchg16b` uses ARM's `caspal` for the same reason; a lock in the interpreter would not exclude the JIT. A misaligned `cmpxchg16b` faults, as the SDM says. Misaligned locked accesses of other sizes fall back to a lock that excludes only other atomic updates.
- **x87 computes in double precision.** The eight registers are `f64`; 80-bit loads round to double, 80-bit stores widen. Reason: games reach x87 only through `long double` code in their runtimes, and exact 80-bit arithmetic would need soft-float for every operation. FEX's reduced-precision mode makes the same trade, and its `X87_F64` tests check it. The x87 rounding and precision controls do not affect arithmetic; of the exception flags only IE is tracked; exceptions are never raised.
- **MXCSR.** DAZ and FTZ apply to SSE/AVX arithmetic. RC applies to conversions and rounding instructions, not to arithmetic (which rounds to nearest). Exception flags are not raised by arithmetic. The JIT follows RC everywhere through FPCR, so tests that compare the two check rounding through conversions and `roundps`.
- **NaNs** follow x86: first-operand propagation, the negative "indefinite" NaN for invalid operations. FMA uses Rust's fused `mul_add`; which NaN an FMA returns matches the tests but is not checked against the SDM's exact rule.
- **`rcp*` and `rsqrt*`** are computed exactly and rounded to f32. Hardware gives a ~12-bit approximation, so low bits differ from hardware; programs that depend on the exact approximation would see a difference.
- **CPUID reports a Zen 2 core** (family 17h) with all of Zen 2's features, bit positions from AMD's public CPUID specification, implemented or not. Reason: a program then hits the missing instruction and we see it, instead of quietly taking a slower path.
- **`rdtsc`/`rdtscp`** return nanoseconds since the Unix epoch; `rdtscp` reports processor 0.
- **MONITORX/MWAITX**: the monitor arms nothing and MWAITX yields the host thread and returns, an early wake-up the hardware may also give. They run in the interpreter, so a title that spins on them pays a JIT exit per wait.
- **Undefined flags.** Where the SDM leaves a flag undefined, the interpreter picks a value and the JIT must pick the same one (see [jit.md](jit.md)).

Not implemented: MMX (`0F` vector opcodes without a prefix), x87 BCD (`fbld`, `fbstp`), and a few rare x87 forms. EVEX and XOP decode as unsupported (Zen 2 has neither). Unimplemented instructions are reported by readable name and encoding.

## Oracles

There is no PS5 to measure on, so correctness rests on oracles.

### FEX's ASM tests

There is no public hardware-generated x86-64 suite (SingleStepTests stops at the 80386). FEX-Emu's handwritten tests (`unittests/ASM`, MIT) each run a few instructions and state the registers they must end with; FEX runs them natively on x86-64 hardware in CI, so the expected values are hardware-checked.

`scripts/fetch-references` puts them in `references/src/FEX` (pinned, not committed). `ps5slop-asmtest` (`crates/asmtest`, needs nasm from `nix develop`) assembles each test as FEX does and runs it as FEX's host runner does: code at `0x10000`, registers zero, RFLAGS `0x202`, scratch memory at `0xE000_0000`, until `hlt`. It skips FEX's disabled and known-failing lists, 32-bit tests, features Zen 2 lacks, tests that make Linux system calls, and the `X87/` tests that have an `X87_F64` twin (plus the few that need 80-bit values). Results go to `target/asmtest/report.tsv`. Tests whose expected values include MMX registers are not checked.

### Rosetta 2, by fuzzing

`ps5slop-fuzz [STATES [SEED]]` makes one register-form encoding (ModRM mod = 3, no RSP) for every opcode of the VEX maps, the legacy `0F`, `0F 38` and `0F 3A` maps and the safe primary opcodes, with every prefix combination. `crates/asmtest/fuzz/harness.c`, built with `clang -arch x86_64`, runs each under Rosetta with random register states (GPRs, YMM0-15, arithmetic flags), and the tool compares our interpreter's result. A third of the vector registers hold text-like bytes, so string compares meet matches and terminators. Results go to `target/fuzz/report.txt`. Control flow, stack, string, x87, system and timing instructions are left out. Only the interpreter is checked; the JIT is checked against the interpreter (see [jit.md](jit.md)).

Rosetta runs AVX2, FMA, F16C and BMI even though its CPUID hides them, so it covers what FEX's suite reaches only partly. It is not hardware. Known places where the SDM sides with us, not Rosetta:

- `vpsllvq`/`vpsrlvq` keep the value for a count with bit 63 set (SDM: any count above 63 gives 0).
- `cmpxchg` sets only OF, and zero-extends a 32-bit destination when the compare fails.
- Legacy SSE `cmpps`/`cmpsd` use the whole immediate as the predicate (SDM: bits 2:0 only).
- `vcvtps2ph` swaps round-down and round-up.
- `movsxd` without REX.W sign-extends into 64 bits.
- `rcp*`/`rsqrt*` clear low mantissa bits (we compute exactly).

Flags the SDM leaves undefined differ too and are not bugs. Encodings Rosetta rejects with `#UD` (mostly a wrong VEX.L or W) run in our interpreter; games do not execute them. SSE4a cannot be checked this way (Intel CPUs lack it).

### Other checks

Unit tests with encodings from llvm-mc, and our own test programs (`scripts/run-test-programs`). The interpreter is the oracle for everything after it, so where it is wrong about hardware, the JIT is wrong the same way; the two external oracles above are what catch that.
