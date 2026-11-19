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

How they are built (`scripts/build-test-programs`, inside `nix develop`): each program is a folder in `tests/programs/` with C sources and an `imports.txt` that names the library (and module) of every imported function. Apple's clang compiles for `x86_64-unknown-freebsd`, `ld.lld -shared -e _start` links with 16 KiB pages and page-separated segments, and `ps5slop-mkelf` turns the result into an Orbis executable in `target/test-programs/`.

- `hello`: calls `write` and `exit` from `libkernel`. It loads and relocates; the two imports are reported missing until the HLE layer provides them. Which library really exports those names on Orbis is not checked yet; the program only has to agree with our own HLE.

## PS4 conformance programs

A test program that runs both on vic's PS4 and on our HLE layer, with the outputs compared, would give us a real behavioural oracle for the kernel and library side. It is practical, with limits:

- **Toolchain:** [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain), GPL-3.0, the only open-source toolchain that outputs the real format (fake-signed SELF, NID imports, `PT_SCE_DYNLIBDATA`). vic has built working PS4 homebrew with it, and has it at `~/OpenOrbis/PS4Toolchain` (the macOS `create-fself` and `readoelf` tools are there; the compiler is the host's clang). Our test sources stay ours (GPL-2.0-or-later); the binaries, linked with its GPL-3.0 runtime, are only built locally and never committed. The origin of its library stub lists is not documented and may be reverse engineering, which is a reason to keep its name lists out, but not to avoid it as a reference otherwise (see [decisions.md](decisions.md)).
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

## PS5 homebrew

A survey on 2026-11-19 (by a research delegate; nothing downloaded, so archive contents come from READMEs and release notes and are unverified) found:

- **Native PS5 homebrew ships as title folders, not packages:** `<TITLEID>/eboot.bin` (a development fake-signed SELF), `sce_sys/param.json`, `sce_module/libc.prx` and assets, copied to the console's homebrew directory. That is the dumped-folder input ps5slop already takes. No clean homebrew `.pkg` turned up.
- **Almost all of it is built from one template**, ps5-native-app-boilerplate (GPL-3.0-or-later; the original repository is gone, copies survive), with clang 18's `x86_64-sie-ps5` target and the ps5-payload-sdk headers and stubs. Every title carries the template's own generated `libc.prx`: not Sony's, about 2,566 exports, mostly stubs, with `malloc`/`free` forwarded to `libSceLibcInternal`. So a PS5 homebrew title needs us to load its own `sce_module` libraries as guest code and link them to our HLE libraries.
- **Index:** the homebrew.page catalog (`https://homebrew.page/catalog/v1.json`, from `github.com/darkxex/ps5-homebrew-catalog`) lists about 209 apps with source, release URL, checksum and license. Many entries are recompiled commercial games that need the user's game data; we skip those.

Candidates, simplest first:

| Title | Source | What it exercises |
| --- | --- | --- |
| ScreenTester5 (`PPSA69420`, 1 MB) | github.com/Marice/ScreenTester5, GPL-3.0 | Colour patterns drawn on the CPU through VideoOut, the pad. **First target.** |
| DriftCheck (`PPSA98766`) | github.com/felipemelchior/driftcheck5, GPL-3.0 | Everything the DualSense reports |
| Raymarch tech demo | github.com/Alexius0815/ps5-raymarch-techdemo, GPL-3.0-or-later | Threads, VideoOut, AudioOut, pad; `--smoke-test` renders 180 frames and exits 0. Only ever run under another emulator (AnyPS5), not on a PS5 |
| DOOM port | github.com/lowbit/ps5-doom, GPL-2.0 | The first GPU target: draws through AGC with an OpenCL kernel compiled for gfx1010; ships the freely distributable shareware WAD |
| BennuGD2 PS5 runtime | github.com/Rufidj/BennuGD2 (`ps5` branch) | OpenGL through the PS5 Mesa port; its README records sandbox behaviour (`stat`, `getcwd`, `opendir` refused, `exit()` raising SIGSYS) |

Legal notes from the survey:

- The template's documentation says reverse engineering was used to learn interface facts for its `libc.prx`. Its builder sources and API-surface list are therefore not read here until vic decides (see [legal.md](legal.md)). Running a title's own `libc.prx` as guest code is a different matter: it is part of the homebrew, not a Sony module.
- LibProsperoPKG bundles key material (passcode and token files, a keys class). We only ever read its format write-up (see [references.md](references.md)); its code is not read or used.

## Homebrew (PS4 era and payloads)

- [ps5-payload-dev/sdk](https://github.com/ps5-payload-dev/sdk) (GPL-3.0) builds plain ELF "payloads" with their own run-time linker, which walks loaded system modules through kernel access. Running them unmodified means faking that environment; not a first target. Its `sce_stubs` are generated from decrypted Sony modules: do not use.
- A PS5 port of Mesa's radv on top of AGC exists in that organisation's package repository. It is the only known open-source producer of PS5 PM4 command buffers, and could become a GPU test once our AGC layer exists.
- No maintained open-source toolchain builds native-format PS5 apps (as of 2026-11-19).
