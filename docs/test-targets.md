# Test targets and oracles

## Constraints

- **No PS5.** vic has no PS5, so we cannot measure PS5 behaviour. Anything about PS5 behaviour that rests only on documentation or other emulators is unverified, and the docs say so.
- **A PS4 on firmware 12.50.** Same Orbis lineage: x86-64 (Jaguar), a FreeBSD-derived kernel, the same library design. It can show us observed behaviour of the kernel and system-library side (syscall results, error codes, threading, memory and file semantics). It tells us nothing about RDNA2, AGC or any PS5-only library, and PS4 behaviour may differ from PS5 behaviour. vic handles the console side himself, outside the repo. We only ever run our own test programs on it and record what they print.
- "Measure first" applies to our own code on vic's M4 Pro: translator speed, renderer throughput, synthetic workloads.

## Oracles

| Area | Oracle | Status |
| --- | --- | --- |
| x86-64 instructions | Public test suites (FEX's assembly tests are MIT and carry expected results; QEMU's `tests/tcg`), checked against an interpreter of ours | To pick; check each suite's license and how its expected values were made |
| x86-64 instructions, differential | Our interpreter vs. Rosetta 2 on this Mac (a user-mode x86-64 implementation, AVX2 since macOS 15) for user-mode code | Idea; Rosetta's own bugs would show up as false alarms, so disagreements need a third opinion |
| JIT | Differential testing against our interpreter, as in 86slopper | Planned |
| Kernel and libraries | Our own test programs, run on vic's PS4 and on our HLE layer, outputs compared | Proposed, see below |
| GPU | AMD's ISA documents; open-source PM4 producers (Mesa) | Unverified by nature |

## Our own test programs come first

The first programs we load are ones we write: a small x86-64 ELF built with clang, with the Sony program headers and `NID#LIB#MOD` imports that we emit ourselves, computing NIDs only for names we chose (see [loader.md](loader.md)). This tests parsing, relocation, NID resolution, TLS and the calling convention with no third-party data, and no legal questions.

Binaries are built from source; they are never committed (`scripts/check-forbidden` refuses ELF files with Sony `e_type` values anyway).

## PS4 conformance programs

A test program that runs both on vic's PS4 and on our HLE layer, with the outputs compared, would give us a real behavioural oracle for the kernel and library side. It is practical, with limits:

- **Toolchain:** [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain), GPL-3.0, the only open-source toolchain that outputs the real format (fake-signed SELF, NID imports, `PT_SCE_DYNLIBDATA`). vic has built working PS4 homebrew with it, and has it at `~/OpenOrbis/PS4Toolchain` (the macOS `create-fself` and `readoelf` tools are there; the compiler is the host's clang). Our test sources stay ours (GPL-2.0-or-later); the binaries, linked with its GPL-3.0 runtime, are only built locally and never committed. The origin of its library stub lists is not documented and may be reverse engineering, so it is used for cross-compiling only, never as a reference for formats or behaviour (see [legal.md](legal.md)).
- **Our loader would need to accept PS4 binaries.** The format is shared, so this is mostly a question of which libraries we offer.
- **Scope:** libkernel, libc, pthreads, memory, files, time. Not graphics: the PS4 uses GNM on GCN, the PS5 AGC on RDNA.
- **Caveat:** PS4 firmware 12.50 is not PS5 firmware. Where the two might differ, a PS4 result is a hint, not proof.

## Commercial titles (user-supplied, later)

| Title | PS5 native? | Engine | Size | Notes |
| --- | --- | --- | --- | --- |
| Deltarune | Yes: PS5 product `PPSA27093`, separate PS5 and PS4 packages | GameMaker | ~452 MB (PS5) at launch | vic's suggestion. Sizes from PlatPrices; separate package inferred from sizes and PS5-only features. No public PS5 emulator report found. |
| Balatro | Yes (`PPSA21401`) | LÖVE (Lua) | | Likely very light. |
| Dead Cells | Yes (`PPSA15552`) | Heaps/HashLink | | Reaches menus or gameplay in several emulators. |
| Hollow Knight, GRIS, Cult of the Lamb, DREDGE | Yes | Unity | | Second wave: Unity brings a large runtime. |

Title IDs come from SharpEmu's compatibility list; `PPSA` is the PS5 title prefix. Engines other than Balatro's are from general knowledge (unverified).

Nothing public describes GameMaker's PS5 runtime or which graphics API it uses (very likely AGC, but unverified).

## Homebrew

- [ps5-payload-dev/sdk](https://github.com/ps5-payload-dev/sdk) (GPL-3.0) builds plain ELF "payloads" with their own run-time linker, which walks loaded system modules through kernel access. Running them unmodified means faking that environment; not a first target. Its `sce_stubs` are generated from decrypted Sony modules: do not use.
- A PS5 port of Mesa's radv on top of AGC exists in that organisation's package repository. It is the only known open-source producer of PS5 PM4 command buffers, and could become a GPU test once our AGC layer exists.
- No maintained open-source toolchain builds native-format PS5 apps (as of 2026-11-19).
