# Test targets and oracles

What we check behaviour against, given that there is no PS5.

## Constraints

- **No PS5.** We cannot measure PS5 behaviour. Anything about it that rests only on documentation or other emulators is unverified, and the docs say so.
- **A PS4 on firmware 12.50.** Same Orbis lineage: x86-64, a FreeBSD-derived kernel, the same library design. It can show kernel and system-library behaviour (results, error codes, threading, memory and file semantics). It tells us nothing about RDNA2, AGC or PS5-only libraries, and PS4 behaviour may differ from the PS5's: a PS4 result is a hint, not proof. vic runs things on it himself, outside the repo, and only our own test programs ever run there.
- "Measure first" applies to our own code on vic's M4 Pro.
- Tests are the regression bar. Agents do not run games to check a change; breakage the tests miss means the tests are missing something.

## Oracles in use

| Area | Oracle | Where |
| --- | --- | --- |
| x86-64 interpreter | FEX-Emu's ASM tests (MIT), whose expected values were checked on real x86-64 hardware | `ps5slop-asmtest` ([cpu.md](cpu.md)) |
| x86-64 interpreter, differential | Single instructions on random register states, our interpreter against Rosetta 2. Where Rosetta disagrees with the Intel SDM, we follow the SDM | `ps5slop-fuzz` ([cpu.md](cpu.md)) |
| JIT | Our interpreter: unit programs run both ways and compared, and `PS5SLOP_JIT_VERIFY` runs every translated block in the interpreter too during a real run | `crates/jit` ([jit.md](jit.md)) |
| GFX10 decoding and printing | `llvm-mc` for gfx1013: titles' programs, random encodings, every opcode | `crates/gfx10/tests/llvm.rs` ([shaders.md](shaders.md)) |
| Translated GPU programs | Hand-assembled programs translated and compiled with Metal; some run on the GPU with known results. Instruction semantics come from AMD's ISA documents; mostly no oracle checks values | `crates/gfx10/tests`, `crates/gpu` |
| Rendering | Captured frames replayed against golden pictures; the picture the game showed against the replay; reference screenshots and videos of the PS5 for visual bugs | `frame_replay`, `ps5slop-replay check` and `live` ([debugging.md](debugging.md)) |
| Kernel and libraries | Our own test programs, with expected output; on the PS4 where useful | `tests/programs` (below) |
| Audio | WAVE dumps cross-correlated with a title's own track files | [audio.md](audio.md) |

A visual bug needs a ground-truth picture (a PS5 screenshot or video frame) before it can be worked on: without one, "looks wrong" has no target.

## Our own test programs

Each program is a folder in `tests/programs/` with C sources, an `imports.txt` naming the library (and module) of every imported function, and `expected.txt` (plus `expected-status` when the exit status is not 0). `scripts/build-test-programs` (inside `nix develop`) compiles them with Apple's clang for `x86_64-unknown-freebsd`, links with `ld.lld` (16 KiB pages, page-separated segments), and `ps5slop-mkelf` turns the result into an Orbis executable with NID imports we compute only for names we chose ([loader.md](loader.md)). `scripts/run-test-programs` runs each under ps5slop and compares stdout and exit status.

They test loading, relocation, NID resolution, TLS, the calling convention and the HLE layer with no third-party data. Current programs cover writing and exiting, threads and synchronisation, and direct memory. Which library really exports each name on Orbis is not checked; the programs only have to agree with our HLE. Binaries are never committed (`scripts/check-forbidden` refuses Sony ELF types).

## PS4 conformance programs

A program that runs both on vic's PS4 and on our HLE, with outputs compared, is a real oracle for the kernel and library side:

- **Toolchain:** the [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain) (GPL-3.0), the only open-source toolchain that emits the real format (fake-signed SELF, NID imports). Our test sources stay ours (GPL-2.0-or-later); binaries linked with its GPL-3.0 runtime are built locally and never committed. Its library stub lists are of undocumented origin, so we keep its name lists out ([legal.md](legal.md)).
- Our loader runs PS4 binaries; the format is shared, so it comes down to which libraries we offer.
- **Scope:** libkernel, libc, pthreads, memory, files, time. Not graphics: the PS4 uses GNM on GCN.

## PS5 homebrew

Native PS5 homebrew ships as title folders (`eboot.bin`, `sce_sys/param.json`, `sce_module/libc.prx`), which ps5slop runs directly. Almost all of it comes from one template whose generated `libc.prx` (not Sony's) we load as guest code. The template's own reverse-engineered files are not read ([legal.md](legal.md)). Titles used:

- **ScreenTester5** (GPL-3.0): colour patterns drawn on the CPU through VideoOut, and the pad. The first homebrew target.
- **DriftCheck** (GPL-3.0): everything the controller reports, drawn on screen; writes a log to `/data`.
- Candidates not yet run: a raymarch demo with a `--smoke-test` mode, a DOOM port drawing through AGC, a BennuGD2 runtime on Mesa. Payload-style homebrew (ps5-payload-dev) needs a faked kernel environment and is not a target; its stubs are generated from decrypted Sony modules and are not used.

## Commercial titles

User-supplied, never in the repo. Each is a target for a different part of the system:

- **Balatro** (LÖVE): light; NGS2 audio and a small set of shaders.
- **The Binding of Isaac**: libc start-up, pad output, NGS2.
- **Minecraft** (Bedrock): FMOD through AudioOut2, save data, heavy threading and memory churn.
- **Superliminal** (Unity IL2CPP, FMOD): modules and plugins, exceptions, APR, Unity's PSN plugin.
- **Astro Bot** (Sony's own engine): fibers, fonts, AvPlayer, a large GPU workload with ray tracing.
- **Grand Theft Auto V** (RAGE): the heaviest target; multiple GPU queues, streaming through APR, Ajm audio, ambisonics.
- **Deltarune** (GameMaker, PS4 package): the PS4 library names and Fios2.

Compatibility status per title lives in the issues of [slopsmith/ps5slop-compat](https://github.com/slopsmith/ps5slop-compat), not here.
