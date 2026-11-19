# CPU

How the game's x86-64 code runs on Apple Silicon. This is a proposal under discussion, not a decision (see [decisions.md](decisions.md)).

## What has to run

- **ISA:** user-mode x86-64 as a Zen 2 runs it: integer with REX, SSE to SSE4.2, AVX and AVX2 (VEX), and probably FMA3, BMI1/2, F16C, LZCNT/POPCNT, MOVBE. x87 only where compilers still emit it. No privileged instructions, no segmentation beyond the FS base (TLS).
- **How much of it games use:** unknown. Nobody has published an instruction-mix survey of PS5 binaries. A local scan of a dumped game (aggregate counts per instruction class, nothing committed) would answer it; whether that is fine under [legal.md](legal.md) is a question for vic.
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
- **Each guest thread needs a vCPU** (or we multiplex them ourselves), and blocking calls must leave the VM.
- More moving parts before anything runs.

Without HVF, translated code runs in our process with guest addresses offset by a base register (one extra add on some accesses), TSO through acquire/release accesses, and faults through signals.

## Proposal

1. **Interpreter first,** in Rust, user-mode x86-64 only, running in our process with a base-offset address space. It runs our own test programs from day one, it is the oracle for everything after it, and it lets the loader and HLE work start without waiting for a JIT. Guest memory accesses that may race use acquire/release ordering.
2. **Decide the JIT later, with data,** once a commercial title reaches its first GPU submission under the interpreter. By then we can measure: interpreter speed, the instruction mix, the rate of HLE calls (which decides whether HVF's exit cost hurts), and whether games need fixed addresses.
3. **Lean:** our own JIT, running inside an HVF VM for TSO and the address layout, with the hottest HLE functions handled inside the VM. FEX stays the fallback if our own JIT turns out too slow to build; a time-boxed experiment building FEXCore on macOS would make that fallback real. Rosetta is an oracle only.

The CPU sits behind one interface (run a thread until it calls into HLE, faults or is interrupted), so the interpreter, our JIT or FEX can be swapped without touching the loader or HLE code.
