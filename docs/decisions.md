# Decisions

Newest last. Each entry: date, decision, reason. When a decision is reversed, add a new entry and mark the old one as superseded.

## 2026-11-19: HLE only, no LLE of Orbis

We run the game's own code and implement the Orbis kernel interface and system libraries in the host process. We do not emulate the PS5's boot chain, secure processor, hypervisor or SSD controller.

Reason: those parts are undocumented, and LLE of them would be roughly 100x the work. Running the Orbis shell under HLE is a late stretch goal at most.

## 2026-11-19: Rust, Cargo workspace, jj

Reason: vic's preference, and the same setup as 86slopper, so ideas and possibly code can move between the two.

## 2026-11-19: Correctness oracles without a PS5

vic has no PS5. Correctness rests on oracles that need no PS5: public instruction test suites, an interpreter as the reference for any JIT, public documentation, other emulators' notes, and our own test programs run on vic's PS4 for the kernel and library side (same Orbis lineage). Anything that rests only on documentation or other emulators is marked unverified.

## 2026-11-19: Own translator, interpreter first; no FEX

The CPU is our own x86-64 translator. An interpreter comes first: it runs our test programs, becomes the oracle for the JIT and lets loader and HLE work start. A JIT is necessary and will follow; its design (including whether it runs inside an HVF VM) is decided when a commercial title reaches its first GPU submission and we can measure. Rosetta is used only as a test oracle.

Reason: Rosetta is general-purpose only through macOS 27. FEX was not designed for macOS, and shoehorning it in (porting, maintaining a fork, software TSO, its guest-address-equals-host-address model against macOS's layout) is likely a mistake. Details in [cpu.md](cpu.md).

## 2026-11-19: Analysing user dumps

Games may be analysed locally (instruction mix, imports). System modules may be analysed to some extent: their metadata, such as symbol tables, is fine. Decompiling how modules work, and using the outputs of decompilation, is out of line. Details in [legal.md](legal.md).

Open: whether symbol tables taken from modules may be committed. (Settled in the entry below.)

## 2026-11-19: Module symbol tables may be committed in HLE code

Symbol metadata taken from user-dumped modules (exported NIDs, library and module names and versions) may be committed as part of the HLE OS code. This narrows the "no tables copied from a dump" rule for this one kind of table; code, data tables, constants and structure layouts from modules stay out.

Reason: it is interface metadata, which the HLE layer needs to know what it has to provide. Note that the export tables appear to hold only NIDs, not names (shadPS4's loader parses exports as `NID#LIB#MOD` and gets names from a separate database), so a firmware dump tells us which functions exist, not what they are called.

## 2026-11-19: OpenOrbis for cross-compiling test programs only (superseded below)

Our test programs may be built with OpenOrbis (or similar toolchains) to run on vic's PS4 and on our layer. The toolchain and its output are not a reference for formats or behaviour, and our own loader and test-ELF tool are written from public facts.

Reason: OpenOrbis is known to work on the PS4 (vic has built homebrew with it), but where its stub lists came from is undocumented, so it may be tainted.

## 2026-11-19: First test programs come from our own tool

A small Rust tool turns an ELF built with `clang -target x86_64-unknown-freebsd` into the Sony layout (`ET_SCE_DYNEXEC`, `PT_SCE_DYNLIBDATA`, `NID#LIB#MOD` imports for names we choose). We own every byte of it. Its accuracy rests on the public format facts in [loader.md](loader.md); the PS4 conformance programs will catch mistakes.

## 2026-11-19: Semi-LLE of user-supplied Sony modules

Sony modules from the game's own folder or from the user's firmware dump may be loaded and run as guest code, instead of an HLE version. (Narrowed to an allowlist in the entry below.) They are supplied at run time and never committed. HLE stays the default for modules that reach into the kernel or hardware.

Reason: an HLE layer runs Sony code in our process either way (the game's own code is linked against it), and loading libraries the user already has saves reimplementing them.

## 2026-11-19: Semi-LLE only through an allowlist

Only allowlisted modules are loaded as guest code. A module qualifies only if all its imports are provided by HLE modules (or by other allowlisted modules), so it runs on top of HLE. Details in [loader.md](loader.md).

Reason: not every module is a candidate. Ones that reach into the kernel or hardware would drag LLE in through the back door.

## 2026-11-19: Open fake-signed packages with their passcode

ps5slop may open fake-signed `.pkg` files as a convenience, deriving the keys from the package's passcode (supplied by the user; all zeros for OpenOrbis builds). No Sony key material is embedded or used. Fake packages without a known passcode, and retail packages, stay out. The loader still runs from an unpacked `app0` folder. Details in [packages.md](packages.md).

Reason: vic wants to run his own and other homebrew packages directly. The passcode route needs nothing but what is in the package plus the passcode, which the package's builder chose; verified against vic's own package.

## 2026-11-19: Supported inputs: fake-signed packages and dumped folders

ps5slop takes two inputs: fake-signed packages (FPKG), opened with the all-zeros passcode by default, and already extracted `app0` folders (dumps). Retail Sony-signed packages are not supported.

Reason: many dumping tools produce FPKGs, and folders are what other dumpers and our own builds give. The important line is that we never decrypt retail game data ourselves, and we cannot: retail packages need keys we do not have and will not embed.

## 2026-11-19: FEX's ASM tests as the interpreter's oracle

The interpreter is measured against FEX-Emu's handwritten x86-64 ASM tests (MIT), fetched into `references/` and run by `ps5slop-asmtest`. CPUID reports Zen 2's features, implemented or not.

Reason: no public hardware-generated x86-64 suite exists; FEX runs these tests natively on x86-64 hardware in CI, so their expected values are hardware-checked. Reporting the full Zen 2 feature set makes missing instructions show up as failures instead of hidden fallbacks.

## 2026-11-19: OpenOrbis may be used as a reference

OpenOrbis (its sources, its startup code and the binaries it builds) may be used as a reference for formats and interfaces. We still never import its NID-to-name lists or any other name database (see [loader.md](loader.md)).

Reason: nothing suggests OpenOrbis derives from Sony's SDK; the earlier caution was only that the origin of its library stub lists is undocumented. vic's rule is to stop using a source when there is reason to believe it comes from proprietary SDK material.

## 2026-11-19: One host thread per guest thread; TSO by barriers in the interpreter

Each guest thread runs on its own host thread with its own CPU state. The interpreter provides x86's memory ordering in software: every guest load is followed by a load barrier and every guest store is preceded by a store barrier, which allows exactly the reordering x86-TSO allows (a store with a later load). Aligned 1-, 2-, 4- and 8-byte accesses are single host accesses, so they do not tear. Locked instructions (`lock`-prefixed read-modify-writes, `xchg` with memory, `cmpxchg`) run as real atomic compare-and-swap loops on host memory, so a plain store from another thread is never lost between their read and write.

The guest address space becomes shared: one reservation, a region table behind a lock, and per-thread handles that cache the last region and notice changes through a generation counter. Unmapping never makes host pages inaccessible (they are replaced with fresh zero pages), so a thread racing with an unmap reads zeros instead of crashing the host.

Reason: vic did not want cooperative scheduling on one host thread. Hardware TSO is not available to a macOS user process (see [cpu.md](cpu.md)); barriers in an interpreter are correct and cheap next to its other costs. The JIT plan is unchanged: hardware TSO inside an HVF VM.

## 2026-11-19: The homebrew template's reverse-engineered files stay unread

The PS5 homebrew template (ps5-native-app-boilerplate) says it used reverse engineering to learn interface facts for its generated `libc.prx`. Its `libc_builder.cpp` and `api-surface.txt` are not read for this project. Running the `libc.prx` a homebrew title ships is fine: it is part of that title. The template's other source files (its startup code, build scripts) may be read for facts; vic agreed on 2026-11-19.

Reason: our rule is to build from public documentation, observed behaviour and clean-room notes, not from reverse-engineered material.

## 2026-11-19: Reuse 86slopper's window code

The VideoOut window reuses 86slopper's winit and `CAMetalLayer` code (vic's own, same license).

## 2026-11-19: The window runs on the main thread, the guest beside it

AppKit needs its event loop on the main thread, so `ps5slop run` starts the guest's main thread on a second host thread and runs the window on the first. VideoOut tells the window about flips through a callback; the window reads the front buffer from guest memory, untiles it on the host and uploads it to Metal. The window only opens once the title calls `sceVideoOutOpen`, so programs that never draw get none. `--headless` runs without a window, and `--dump-frame` writes the screen to a file, which gives tests a way to check what a title drew.

Reason: it keeps the guest unaware of the window, and untiling on the CPU is simple and fast enough until the GPU work brings Metal-side surfaces.

## 2026-11-19: Controllers through Apple's GameController framework

The window reads controllers through GameController (`objc2-game-controller`), polled every 8 ms on the main thread, and merges them with the keyboard mapping. It names the DualShock 4 and DualSense buttons directly and has the touchpad button; it can also drive the light bar and rumble later, which `gilrs` (the cross-platform alternative) cannot do well on macOS.

Reason: vic wants to use a PS4 controller, and GameController maps almost one-to-one onto `ScePadData`.

## 2026-11-19: The JIT runs translated code inside a Hypervisor.framework VM

Translated code runs on HVF vCPUs with hardware TSO (`ACTLR_EL1.EnTSO`) and our own page tables mapping guest addresses one to one, as in 86slopper. One vCPU per guest thread; threads beyond the VM's 64 vCPUs run in the interpreter. The interpreter's TSO-by-barriers decision stays for the interpreter. Details and measurements in [cpu.md](cpu.md#hypervisorframework).

Reason: hardware TSO and MMU-checked guest addresses remove the two largest costs of x86 on ARM (ordering every access, and checking or offsetting every address). 86slopper already runs this way on vic's machine, and its `hvf` and `jit` code can be reused (vic's own, same license). vCPUs are cheap to make (~15 µs), so threads that start and end often are not a problem; the 64-vCPU limit is handled by the interpreter fallback. Decided earlier than [cpu.md](cpu.md)'s plan (a commercial title reaching the GPU) because ScreenTester5 already shows the interpreter is too slow (a full 1080p redraw takes ~2 s), and vic asked to start the JIT.

## 2026-11-19: `/data` is a per-title folder on the host

Titles see a writable `/data`, backed by `~/Library/Application Support/ps5slop/<title folder>/data` unless `--data` names another folder. `/app0` stays read-only.

Reason: titles need somewhere to write logs and, later, saves (DriftCheck logs to `/data/driftcheck`). A folder per title keeps them apart; Application Support is where macOS apps keep such data.

## 2026-11-19: The JIT is the default

`ps5slop run` translates guest code and runs it in the VM; `--interpreter` runs the interpreter only. Threads beyond the VM's 64 vCPUs, and instructions the translator does not handle, still use the interpreter.

Reason: with SSE/AVX integer code and BMI translated, ScreenTester5 draws its menu in 0.2 s instead of 2.5 s, and verify mode shows no differences on ScreenTester5 or DriftCheck. vic asked for it.

## 2026-11-19: Load the modules a title imports from, and Sony's libc

`ps5slop run` loads the modules an executable imports from: the title's own modules always, and modules from `sce_module/` (Sony's) when `ALLOWED_SYSTEM_MODULES` names them, starting with `libc`. Details in [loader.md](loader.md#modules).

Reason: commercial titles (Balatro, Minecraft) import a C and C++ runtime from the `libc.prx` they ship: iostreams, locales, exception support. Reimplementing that in the HLE layer makes no sense. The module is the user's own copy, loaded at run time, never distributed, and it runs on top of our HLE `libkernel`, which is what the semi-LLE allowlist asks for. Minecraft also ships modules of its own that have to load anyway. vic may drop the allowlist later.

## 2026-11-19: The x87 computes in double precision

The interpreter's x87 keeps its eight registers as `f64`. 80-bit loads round to double, and 80-bit stores widen.

Reason: titles reach x87 only through `long double` code in their C++ runtime (Minecraft's start-up loads an 80-bit value). Exact 80-bit arithmetic would need a soft-float implementation of every operation, for code that is rare in games. FEX offers the same trade-off as its reduced-precision mode, and its `X87_F64` tests check it.

## 2026-11-19: Public documentation is usable even if others wrote it from decompilation

We never decompile or disassemble Sony's system code ourselves. Documentation that other people published (notes, wikis, the interface facts other projects describe) may be used as a specification, whatever its origin; we write our own code from it. Code is still only copied under compatible licenses, never decompiler output or ports of Sony code, and leaked SDK material stays out. Details in [legal.md](legal.md).

Reason: vic, while planning a fully HLE `libSceAgc`: there is no official AGC documentation, and the public knowledge about it comes from the scene's reverse engineering. The line that matters is that we do not do that analysis ourselves.

## 2026-11-19: Imports without a public name bind by NID

An HLE function named `#<NID>` binds to that NID. We use it only when no name matches and the title's own code shows what the import does.

Reason: some PS5 imports have no public name to hash. The NID is what the title imports, and the behaviour comes from observing the title's code, never from Sony's modules, so this stays on the interface side of the legal line. See [hle-kernel.md](hle-kernel.md).

## 2026-11-19: pthread handles are guest addresses

`pthread_mutex_t`, `pthread_cond_t`, `pthread_rwlock_t`, their attributes, thread attributes and `sem_t` hold the address of a small zeroed block of guest memory, and our objects are keyed by it.

Reason: Sony's libc reads through these as pointers (it faulted at the small counters we used before), as FreeBSD's pthread types are pointers.

## 2026-11-20: Thread-specific values live in the thread control block

Values of `pthread_key_t` keys 0 to 127 are kept at the thread pointer plus `0x400`, next to errno at `0x800`; the JIT runs `pthread_getspecific`, `scePthreadGetthreadid` and `pthread_self` as native code inside the VM.

Reason: Minecraft called `scePthreadGetspecific` and `scePthreadGetthreadid` about 470 thousand times a second, and each call was a VM exit. Inside the VM the thread pointer is the FS base register, so the values must be in guest memory at a fixed offset from it. The offset is our choice; whether Sony's libc uses those bytes of the control block is unknown (assumed not, as for errno).

## 2026-11-20: Guest mutexes hand off to waiters; conditions honour their clock

An unlocked guest mutex with waiters goes to one of them; a thread that unlocks and relocks it at once waits its turn. `pthread_condattr_setclock` is honoured (CLOCK_MONOTONIC deadlines are on the same clock as `clock_gettime`).

Reason: measured on Minecraft. Its main thread waited about 60 ms at a time for per-object mutexes that a worker released and took again before the woken waiter ran; and one thread ran `pthread_cond_timedwait` 37 thousand times a second because its monotonic deadlines were read as wall-clock times far in the past.

## 2026-11-20: A user logs in when the title first asks

`sceUserServiceGetEvent` reports user 1 logging in on its first call.

Reason: Minecraft's sign-in looked user 1 up among the users it had seen log in, found none and stopped with an assertion ("PlatformIdentity_sony: invalid user id"). A console reports users already logged in this way when a title starts (assumed from the event's existence; not verified on hardware).

## 2026-11-20: Shaders go through our own `gfx10` crate, straight to MSL

GFX10 machine code is decoded by a new crate, `gfx10`, checked against `llvm-mc --disassemble -mcpu=gfx1013`, and translated directly to MSL text with a straight register model: one Metal thread per lane, registers as locals, lane masks reduced to this lane's bit, branches as a `switch` in a loop, scalar and buffer loads from a binding of all guest memory. Details in [shaders.md](shaders.md).

Reason: no IR library in between (SPIR-V plus SPIRV-Cross would pull in Apache-2.0 code, see gpu.md), and a per-lane model is the shortest path to Balatro's shaders, which use no cross-lane operations. Guest-memory loads instead of per-buffer bindings because Balatro's vertex programs build their V#s at run time. Opcode data comes from Mesa ACO (MIT), the stage conventions from SharpEmu-mac (GPL-2.0-or-later); LLVM is used only as a tool.

## 2026-11-20: Run packages in place, through a read-only file system backend

`ps5slop run GAME.pkg` serves `/app0` (and the loader's reads of `eboot.bin` and modules) straight from the package's PFS image, through an `ImageFs` mount in the guest file system. We do not extract packages to a cache folder.

Reason: PS5 packages are large (Astro Bot is 38 GB), so extracting would double the disk use and delay the first run by minutes, while the game reads only part of it. `crates/pkg` already reads files at random offsets (block lists, PFSC, XTS), so a lazy backend is little extra code. The cost is that guest descriptors are no longer always host files (`GuestFile` in `vfs.rs`), and code that wants a host path (`Vfs::resolve`) gets none for package files; it should use `Vfs::read` instead.

## 2026-11-20: Per-title folders are named by the title id

`/data`, save data and `/temp0` live under `~/Library/Application Support/ps5slop/<title id>/` (`PPSA17221`), for packages and folders alike. Without a title id (a lone executable, a folder without `sce_sys/param.json` or `param.sfo`) the folder's or file's name is used, as before. A folder kept under the old name is renamed to the title id on the first run that finds the id.

Reason: the same title run from a package and from a folder (or from folders with different names) should see the same data and saves.

## 2026-11-20: Render targets live in Metal textures; flips present them directly

Render targets are Metal textures keyed by guest address. A flip of a buffer the GPU rendered presents that texture in the window, and guest memory is not updated with the rendered pixels until something on the CPU needs them (not implemented yet). Guest memory is wrapped in place as no-copy buffers for everything else.

Reason: vic wants all drawing on Metal, with no CPU rendering, and a read-back per frame would cost a full-frame copy for nothing. SharpEmu-mac (GPL-2.0-or-later) works this way on Apple Silicon and is the model we port from.

## 2026-11-20: Audio through NGS2 ported from SharpEmu-mac, out through an AudioQueue

libSceNgs2 is a port of SharpEmu-mac's (GPL-2.0-or-later) voice, mixer and waveform code, PCM only, with two rule changes Balatro needs (a voice ends only after playing its last block; a block's sample count is clamped to its data). AudioOut mixes its ports to stereo float and feeds the default output device through an AudioToolbox output queue, our own small FFI rather than a crate. The device opens only when the window is shown (`PS5SLOP_AUDIO` overrides); while it runs, its clock paces `sceAudioOutOutput`, otherwise the host clock does. `PS5SLOP_AUDIO_WAV` dumps what each port plays. Details in [audio.md](audio.md).

Reason: Balatro, the first title to play sound, uses NGS2 with 16-bit WAVE files only, so a decoder is not needed yet. SharpEmu-mac is the most complete license-compatible description of NGS2. An AudioQueue needs a few C declarations, where a crate such as cpal would pull in a dependency tree. Pacing by the device keeps the title's audio clock and the speakers in step without drift, and the WAVE dump lets us check output without listening.

## 2026-11-20: Compute programs branch per Metal SIMD group; descriptor sources by data flow

In compute programs, `s_cbranch_execz`/`execnz`/`vccz`/`vccnz` use `simd_any` over the Metal SIMD group (32 threads) instead of this thread's lane bit, while vertex and pixel programs keep branching per lane. Descriptors are no longer limited to unmodified user SGPRs: a data-flow pass over the program's control-flow graph follows user SGPRs through moves and scalar loads, and the renderer evaluates the resulting chain of loads per draw.

Reason: Minecraft's compute programs skip their work with `execz` before an `s_barrier`, and Metal needs barriers in control flow that is uniform across threads that reach them; per-lane branching would send threads of one SIMD group down different paths to the barrier. A SIMD group is the largest unit Metal lets a shader vote over, and matches a hardware wave branching as a whole within its 32-lane halves. Pixel programs are left alone because Balatro's pixel programs are verified with per-lane branching and fragments that discard make SIMD votes harder to reason about. Minecraft loads most texture and sampler descriptors from a table that a user SGPR pair points to, sometimes through two levels, so a fixed rule (user SGPRs only) translated none of them.

## 2026-11-20: GPL-3.0 decoders only as black-box oracles

To write our own Kraken decoder, we may run a GPL-3.0 decoder (ooz-wasm, pkg-to-anyps5 built from source without reading it) on a developer's machine as a black box and compare outputs. Its outputs (decompressed game files) stay outside the repo like any game data. We never read its source, never copy from it, never link or load it into ps5slop (statically or as a dynamic library), and never ask users to supply one. User-supplied Oodle libraries are out for the same reason.

Reason: vic does not want GPL-3.0 code in this GPL-2.0-or-later project, and loading a library at run time still counts as linking. Running a program to see what it outputs is not using its code; this is how vic built a softmodem before, by having it talk to another implementation.

## 2026-11-20: PS5 inner images are decoded block by block, scoped to what our packages use

`crates/pkg` reads a PS5 package's Kraken-compressed `pfs_image.dat` in place: a read decodes only the 256 KiB blocks it touches, through the NAPS layout, and keeps the last 16 decoded blocks. The decoder implements only what Superliminal and Astro Bot use (stored and three-stream Huffman arrays, code descriptions `00` and `01`, LZ chunks with raw or delta literals); anything else fails as unsupported.

Reason: vic asked for a decoder scoped to the packages we have, and running in place matches the earlier decision for PS4 packages (Astro Bot unpacks to 149 GB). The format and the source of each rule are in [packages.md](packages.md).

## 2026-11-20: integer textures by per-descriptor variants; GPU-written arrays and depth stay on the GPU

The translator does not guess whether a texture holds integers. The renderer reads the bound T#s per draw and translates the program again with masks of the integer slots (`uint_textures`, `sint_textures`), caching each variant. Exports to integer targets are made integer the same way, by changing the `SPI_SHADER_COL_FORMAT` the translator sees. Images programs write as 2D arrays or 3D, and depth targets, stay in Metal textures keyed by guest address and are sampled from there; nothing is written back to guest memory, and depth layouts are not detiled.

Reason: the format is only in the descriptor, which can differ between draws of one program, while the instruction alone cannot tell UINT from float data. A variant per descriptor combination is exact and costs one more translation per combination (Minecraft has a handful). Keeping GPU-written images on the GPU avoids read-backs and needs no equation for the depth swizzle modes, which no source we may use describes. The cost: a CPU read of such memory, or a T# that reads it through a layout we do not track, sees stale or zero data.

## 2026-11-20: A control socket with write watchpoints, and Rosetta as a fuzzing oracle

Every run serves a control socket (protocol taken from 86slopper) for reading and writing guest memory, listing threads and watching writes ([debugging.md](debugging.md)). Write watchpoints protect pages at stage 2 so translated stores fall back to the interpreter, which logs them with their RIP. `ps5slop-fuzz` compares single instructions between our interpreter and Rosetta 2 ([cpu.md](cpu.md)).

Reason: vic asked for a way to poke at the guest live, as 86slopper allows. Minecraft's striped textures came from a CPU store with a wrong address; a watchpoint found the stores and a fuzzer found the instruction, without reading the game's code. Rosetta is the only x86-64 implementation on this Mac, and it runs AVX2 and BMI2; where it disagrees with the Intel SDM we follow the SDM and list the case.

## 2026-11-20: GPU-written 2D arrays start from guest memory and reload after CPU writes

Changes the entry above for 2D arrays: an image programs write as a 2D array is loaded from guest memory when its Metal texture is made, and reloaded whenever a checksum of that memory changes (before every write to it, once a frame before reads, ranges up to 32 MiB). GPU writes still stay on the GPU. 3D images are still zero-filled.

Reason: Minecraft's UI glyph atlas starts as CPU-written contents and then gets glyphs from a copy program through a 2D-array descriptor. Zero-filling lost the CPU part (blank letters on the Play screen). A recreated atlas is uploaded again by the CPU, which a texture made once never saw. The cost: a CPU write that only touches part of the range throws away what programs wrote elsewhere in it, and every write to such an image hashes its whole range on the CPU (1 MiB for the atlas; not measured).

## 2026-11-20: The JIT evicts its coldest code cache segments instead of stopping

The code cache is split into 8 MiB segments; when it is full, the coldest segment (by sampled use, then age) is evicted and reused once no vCPU can be in it ([jit.md](jit.md)).

Reason: Minecraft filled the cache when creating a world, and from then on all new code ran in the interpreter. vic asked for eviction of the least used and oldest code, in chunks, rather than flushing the whole cache, which would stall every thread at once. Use is sampled by kicking vCPUs 100 times a second rather than counted per block, which would slow every block down.

## 2026-11-20: PS4 patches run over their base package

`ps5slop run` takes a PS4 patch package over its base (`--patch`), and given a patch alone finds the base beside it ([packages.md](packages.md)).

Reason: vic's Deltarune download is a base package and a patch, and the link points at the patch, which holds no executable. The console mounts both as one `/app0`; extracting and merging them by hand would copy 1.3 GB.

## 2026-11-20: PS4 GPU submissions run without drawing for now

libSceGnmDriver submissions go through the command processor with no renderer, and the packet builders write NOPs ([gpu.md](gpu.md), "PS4 titles").

Reason: drawing needs a GCN shader decoder, gfx7 state and GCN tiling, which is a large piece of work. Running the labels, interrupts and flips lets Deltarune's frame loop and game logic run now, so the CPU and system-library side can be worked on first.

## 2026-11-20: Small new mappings are zeroed in place

A new guest mapping under 64 MiB over host pages that already exist is zeroed in place instead of getting fresh pages, and unmaps under 64 MiB leave the host pages alone ([memory.md](memory.md)).

Reason: every page replacement invalidates whatever maps those pages elsewhere. Minecraft maps and unmaps small pieces of direct memory many times a second during world loading, and each time the GPU windows over that memory (up to 2 GB, for shaders that read through pointers) had to be made again. Metal ran out of memory ("Insufficient Memory") during world loading. Zeroing costs a write per byte, so large mappings keep the page replacement.

## 2026-11-20: The thread that changes guest memory syncs the VM

With the JIT on, every change to guest memory syncs the VM's tables on the changing thread at once, and vCPUs enter without comparing the region generation ([jit.md](jit.md)).

Reason: Minecraft changes its memory map about 450 times a second while loading, and every change made every running vCPU take the machine lock to sync, where they queued behind translations. World loading took 49 s with that and 30 s without it. A vCPU that races ahead of a sync sees what a vCPU already inside the VM always could: a fault on a range not mapped yet (interpreted by the host), or a range unmapped a moment ago.

## 2026-11-20: Command-stream memory writes wait like releases

`WRITE_DATA` to memory is queued with the deferred releases and applied, in order, after the GPU has finished the work before it ([gpu.md](gpu.md)).

Reason: Minecraft's UI uses these writes as labels telling its CPU that earlier GPU work is done. Written when parsed, they ran ahead of draws still queued in Metal, and the game reused vertex data those draws had not read yet. Waiting for the GPU before each write fixed it too but cost 37% of menu frames.

## 2026-11-20: Game modules anywhere in the title load at start; plugins start when asked

Every `.prx` outside `sce_module` and `sce_sys`, up to three folders deep, is loaded and linked before the entry point. Modules the executable needs start then; the rest start the first time the title calls `sceKernelLoadStartModule` for them, with the arguments it passes ([loader.md](loader.md#modules)).

Reason: Superliminal keeps its modules in `Media/Modules` and `Media/Plugins`, imports from two of them and loads Unity's plugins by path later. Linking everything at start keeps the loader in one place (no run-time ELF loading in the HLE crate); deferring the start keeps plugins' start code in the order and with the arguments the title expects. PSNCore refuses to start without them.

## 2026-11-20: Exceptions run on the target thread; blocked host calls restart

`sceKernelRaiseException` queues the exception and kicks the vCPUs; the target runs the handler between guest instructions, and a target blocked in a host function returns from it (condition waits and sleeps) or restarts the call afterwards (everything else) ([hle-kernel.md](hle-kernel.md#exceptions-libkernel_unity)).

Reason: Unity's garbage collector stops threads this way and waits for each to acknowledge, so a thread sitting in `pthread_cond_wait` that never ran the handler deadlocked the collector. Restarting a call by leaving the CPU at its stub needs no change to each wait's internals beyond giving up, and matches `SA_RESTART`. Running handlers inside each wait would have had to drop and retake every lock those waits hold.

## 2026-11-20: `lock cmpxchg16b` uses the host's 128-bit compare-and-swap

The interpreter runs `cmpxchg16b` with ARM's `caspal` (LSE, present on every Apple Silicon Mac), not with a lock ([cpu.md](cpu.md)).

Reason: translated code on other threads updates the same words with native atomics; a lock in the interpreter would not exclude them. Superliminal's runtime uses `cmpxchg16b` from its first instructions.

## 2026-11-20: Open-source stand-ins for system fonts

When a title asks under `/app0` for a font file it does not have and whose name is a known system font (SIE ShinGo, Futura, Heisei Maru Gothic, Mohr), the file system serves an OFL-licensed stand-in from a host fonts folder that `scripts/fetch-fonts` fills; system font sets get one too ([hle-kernel.md](hle-kernel.md#system-fonts)).

Reason: Astro Bot loads every font its settings name, and the system ones are not in its package (a console supplies them, vic). We may not ship Sony's fonts; the stand-ins are free to fetch but are not committed, like other reference material. Serving them under the title's own path keeps the engine's code paths as on a console.

## 2026-11-20: libSceFont on FreeType

libSceFont measures and renders glyphs with FreeType (`freetype-rs`, bundled build), and shaping for the text layout calls is to come from HarfBuzz ([hle-kernel.md](hle-kernel.md#libscefont)).

Reason: vic's call. Font loading, hinting and rasterising are large and well solved; both libraries are compatible with GPL-2.0-or-later (FreeType under its GPLv2 option, HarfBuzz under MIT). The bundled build needs no change to the Nix shell and no library at run time.

## 2026-11-20: Depth ranges outside 0..1 are mapped onto Metal's range

Draws whose depth can leave `[0, 1]` (a float depth buffer with `PA_SC_VPORT_ZMIN_0`/`ZMAX_0` wider than that) get their possible depth range mapped onto `[0, 1]` ([gpu.md](gpu.md)).

Reason: Metal clamps depth to `[0, 1]`, and Isaac draws its layers at -1 to 1 and depth-tests them against each other, so nothing after its first layer drew. Mapping keeps the order depth tests see. The alternative, writing depth from every pixel program, would cost early depth testing in every title.

## 2026-11-20: Unnamed imports a title only calls in passing fail as ENOSYS

Isaac's four libSceNpCppWebApi calls have no public names, so they are bound by NID as placeholders that fail with `ENOSYS`, as earlier unnamed ones were. Loading the title's own `libSceNpCppWebApi.prx` instead was not tried: it is a Sony system module outside the allowlist ([loader.md](loader.md#modules)), and the shipped file has a segment we cannot read.

## 2026-11-20: Controller output through GameController and Core Haptics

Vibration, the light bar and DualSense trigger effects go through Apple's GameController framework: Core Haptics engines from `GCController.haptics` for the motors, `GCDeviceLight` and `GCDualSenseAdaptiveTrigger` for the rest ([input.md](input.md)). The HLE only records what the guest asked; the window's poll applies it. Motors use plain Core Haptics players, one new player per intensity change, because the advanced player and the intensity control do not work with `gamecontrollerd` on macOS 27 ([input.md](input.md#core-haptics-on-macos-what-works)).

Reason: GameController already gives us input from the same controllers, needs no entitlement, and covers DualSense, DualShock 4 and Xbox pads over USB and Bluetooth. Writing HID output reports ourselves would mean per-controller report formats, CRCs for Bluetooth and fighting the system's own driver for the device. Keeping the HLE free of host calls lets headless runs and tests use the same code.

## 2026-11-20: Kraken features are added only when a package uses them

While GTA V failed, its blocks seemed to hold tANS and run-length arrays. Decoders for both were written (from kraken-decode's and lztans's MIT-licensed descriptions and code), then removed: the real fault was the stored offset of blocks past 64 GiB, and with that fixed GTA V uses only the array types the decoder already had ([packages.md](packages.md)).

Reason: code for a feature no package uses cannot be checked against real data or an oracle here (building lztans to run it was blocked), so it would be unverified code on the path every read takes. The guards that turn malformed Huffman descriptions into errors instead of panics stay.

## 2026-11-20: Direct memory still not shared between mappings, for GTA V too

GTA V's streamers map 64 KiB pages of direct memory with `sceKernelBatchMap`, read files into them and unmap them, which looks like it relies on two mappings of one direct range sharing bytes. Our maps still get fresh pages each ([hle-kernel.md](hle-kernel.md)).

Reason: a throwaway build that shared pages by direct offset (`Memory::map_physical`) stopped GTA V at the same assert, so it is not shown to be needed yet, and sharing broke Minecraft before ([jit.md](jit.md)). Whether and how to model physical memory is vic's call; change `uukyvrwo` also reworks `dmem.rs`.

## 2026-11-20: A GPU thread per queue, sharing one renderer

Graphics and each compute owner get their own command processor thread; all of them encode into the one Metal renderer behind a mutex ([gpu.md](gpu.md)).

Reason: Astro Bot's queues wait on labels each other sets, so running every submission in order on one thread let a wait block the work that would end it (24 timeouts of 5 s in 150 s, 3 flips). Separate renderers per queue would not share render targets, images or the GDS, which the queues pass between them. The mutex serialises encoding but not waiting.

## 2026-11-20: DMA_DATA and indirect arguments run on the CPU after a GPU wait

`DMA_DATA` and the reads of indirect draw and dispatch arguments wait for all GPU work encoded so far, then act on the CPU ([gpu.md](gpu.md)).

Reason: their sources (GDS counters, memory) are written by the compute work just before them, and a blit or indirect-buffer path in Metal would need the GDS and guest memory as Metal buffers in the same encoders. A wait costs about a millisecond and Astro Bot issues about ten such packets a frame; encoding them on the GPU is the thing to do if that shows up in a profile.

## 2026-11-20: Games' own code may be disassembled locally

Agents may disassemble or decompile a game's own code on vic's dumps to understand what it does and what it expects from the system, as long as nothing from it is committed ([legal.md](legal.md#analysing-user-supplied-binaries)). Sony's system code stays off limits.

Reason (vic): Sony's code would shape the emulator itself, modelling what the real system does under the hood; a game's code only tells us how it uses the interface and what answers it needs, and is never part of the emulator. Prompted by GTA V's and Astro Bot's asserts, which registers and memory alone could not explain.

## 2026-11-20: libSceAcm placeholders succeed

`sceAcmContextCreate` and the three other Acm functions Minecraft's FMOD imports return 0 and write nothing, where the context create used to fail with `ENOSYS` like other unknown services.

Reason: FMOD treats a failed context create as fatal for all output, so Minecraft was silent. With success it plays its music and sounds through AudioOut2 (verified against the game's own track, [audio.md](audio.md)). Nothing is known about what Acm computes; if FMOD starts batches in game and needs their results, that will show as missing or wrong effects rather than silence.

## 2026-11-20: The main thread gets an 8 MiB stack

The main thread's stack grew from 1 MiB to 8 MiB ([memory.md](memory.md)). Minecraft's first-run "Get started" flow wrote about 1.3 MiB below the top while creating a world and faulted; with 8 MiB it creates the world and spawns the player (checked headless on 2026-11-20).

Reason: the console's default main-thread stack size is unknown. The process parameters (`PT_SCE_PROCPARAM`) of Minecraft, Balatro and Isaac all leave the main-thread stack-size pointer null, so there is no per-title value to honour; 8 MiB is FreeBSD's default stack limit and costs only address space, since host pages are committed on first touch. Reading the field when a title does set it is left until one does.

## 2026-11-21: Unity's PSN plugin initialises (placeholders succeed)

The libSceNpCppWebApi initialisation (`#UYPxv8MIzGo`) and `sce::Json::Initializer::initialize(InitParameter2)` now succeed and do nothing, where they used to fail so that Superliminal's PSN plugin would run "offline". The PSN library placeholders it calls next (GameIntent, EntitlementAccess, GameUpdate, SessionSignaling, Trophy2, UDS) succeed too.

Reason: offline was not a state the title handles. Its PSN plugin's `PrxInitialize` stops at the first failure, the title's `PS5Manager` never leaves its initialising state, and the loading screen waits for that state before it activates the menu scene ([hle-kernel.md](hle-kernel.md)). With success it reaches its menu and first level. Nothing goes to the network: the calls do nothing. If a title later depends on results these calls should have produced (trophy data, web API responses), that will show as missing data rather than a stall.

## 2026-11-21: Middleware inside a game counts as game code

Unity's PSN and SaveData plugins (`PSNCore.prx` and similar), FMOD and other middleware a game ships may be disassembled locally like the game's own code ([legal.md](legal.md#analysing-user-supplied-binaries)), even when they are built on Sony's libraries. The reading is about what the middleware expects from the system; it is never used to learn how Sony's own libraries work, and nothing from it is committed.

Reason (vic): the middleware is not Sony's code and not part of the emulator, so it falls on the game side of the line drawn on 2026-11-20. Prompted by Superliminal, whose PSN plugin gave up silently on a failed `sce::Json` initialise.

## 2026-11-21: Per-pass attachment copies instead of Metal frame captures

GPU bugs are found with `gpusnap` (copies of every render pass's attachments during one frame), `gpuskip` and MSL overrides with `gpureload` ([debugging.md](debugging.md)), not with Metal's `.gputrace` captures.

Reason: a capture records every buffer the frame uses, and our no-copy windows over guest memory are several GiB; one capture of Minecraft wrote 25 GB and grew the process to 13 GB within seconds. The copies are bounded (1 GiB), need no Xcode, and can be read from the command line.

## 2026-11-21: Protection queries report the bits the guest gave

`sceKernelQueryMemoryProtection` reports the protection bits a page was mapped or protected with, including bits we do not map (such as `0x40` and `0x80`), instead of bits derived from our CPU mapping ([hle-kernel.md](hle-kernel.md)).

Reason: GTA V checks bit `0x80` before an APR read to choose between reading straight into its buffer and a bounce buffer, and its bounce path loses the first of two reads on one request; with derived bits every read bounced and its streamer asserted. Reporting what the guest gave is also what a title can expect from a console, whatever the bits mean. Found by disassembling the game's code locally, as vic allowed on 2026-11-20.

## 2026-11-21: Depth fast clears are read from HTILE

Before drawing to a depth target with HTILE enabled, the renderer reads the first HTILE dword of the slice from guest memory; a ZMask of 0 means a fast clear, so the slice is cleared to `DB_DEPTH_CLEAR`, and the dword is marked expanded (ZMask 0xF) so the clear happens once ([gpu.md](gpu.md)).

Reason: Superliminal clears every shadow map by filling its HTILE with a compute program and never touches the depth data; without this the shadow maps held whatever was drawn first, every shadow test failed and the menu's lights added nothing. Modelling HTILE compression fully is not needed for this. The check costs a wait for earlier GPU work whenever compute programs wrote memory, which did not show in Superliminal's or Minecraft's frame rates (not measured separately).

## 2026-11-21: Submissions still wait for Metal one by one

A change that ran queued submissions back to back and waited for Metal once the queue was empty was tried and dropped. Superliminal's menu ran at 22.9 flips a second with it and 25.2 without (single runs, within noise), and the long stall after New Game turned out to be shader compilation, not the per-submission waits ([hle-kernel.md](hle-kernel.md#what-superliminal-needs-2026-11-20)). Not worth an unproven change to GPU synchronisation.

## 2026-11-21: Programs see two guest memory windows

A program's guest memory was one no-copy window. Astro Bot's draws read constants in one region and vertex data in the GPU area, 47 GiB apart, more than one Metal buffer can span. The shader now gets a second window and maps each address into either; the second is reached from the first through the difference of their GPU addresses. Reason: no geometry drew without it, and binding a buffer per region would mean threading a second pointer through every memory helper for the same effect. Pointer arithmetic outside the bound buffer relies on Apple GPUs not bounds-checking device pointers; Metal's shader validation would flag it.

## 2026-11-21: Colour metadata passes are skipped only when their own register table set the mode

Astro Bot leaves `CB_COLOR_CONTROL` at mode 6 (DCC decompress in Mesa's enum) after its decompress passes and its own following draws (upscale, final output) must draw. Skipping every draw in modes 2, 5 and 6, as SharpEmu-mac does, removed those; skipping none let the passes overwrite the scene. Only a draw whose register table wrote `CB_COLOR_CONTROL` counts as a metadata pass. Reason: it is the one rule consistent with every trace taken; what the console does is unknown, so this is a guess to revisit when another title disagrees.

## 2026-11-21: Colour fast clears are applied from CMASK and DCC, like HTILE

As for depth, the first dword of a target's CMASK or DCC is checked before a draw to it and at its metadata passes; a clear code clears the Metal texture and marks the dword expanded. Reason: Astro Bot clears its targets only through metadata, and its transparency buffer must start at alpha 1. The same limits as HTILE apply: one dword stands for the whole surface, and the check waits for earlier GPU work that wrote memory.

## 2026-11-21: A title's own add-on entitlements are reported as installed

`sceNpEntitlementAccessGetAddcontEntitlementInfoList` reports, per title id, the add-on entitlements a title's own package carries, as installed (package type 3, download status 4). Only GTA V has a list so far (three labels that unlock its Story Mode). Every other title gets none.

Reason: GTA V's PS5 build unlocks Story Mode only when these entitlements are listed (seen in its code: it hashes each listed label and compares it with three constants); a full-game license includes them on a console (assumed, not checked). With an empty list its landing page offers "Buy Story Mode" and Story Mode cannot start. Handing every title these labels is wrong (SharpEmu-mac saw another title treat them as its own DLC), so the list is keyed by title id. Separately sold add-ons stay unowned.

## 2026-11-21: The viewport scissor needs `VPORT_SCISSOR_ENABLE`

`PA_SC_VPORT_SCISSOR_0` is intersected into a draw's scissor only when `PA_SC_MODE_CNTL_0` bit 1 is set, as on the hardware (the bit's name is from Mesa's register database). Reason: Astro Bot leaves a 1x1 viewport scissor behind with the bit clear, and its final pass and logo were clipped to one pixel. Balatro, Isaac, Minecraft and Superliminal draw the same as before.

## 2026-11-21: Vertex and pixel loops are cut off after 1024 backward branches

Translated vertex and pixel programs give up (discard, or return the vertex as it is) after 1024 backward branches. Reason: one GPU hang makes macOS ignore all of the process's later Metal work, so a loop that does not end, however it came about, costs the whole session; a few wrong pixels cost less. Astro Bot's logo program hung this way on a GPU-built linked list. Compute programs are left out, since threads waiting at a barrier for one that left could hang anyway. The limit hides the real bug behind the list, which stays open ([gpu.md](gpu.md#astro-bots-title-2026-11-21)).

## 2026-11-21: Depth textures are sampled only while depth used the address last

A sampled image at the address of a depth target reads the depth texture only if no colour target or storage image used the address since that depth target was. Reason: Astro Bot reuses the memory of a 960x540 depth buffer for a bloom level of the same size within the frame, and the bloom read depth values. Superliminal's and Minecraft's depth sampling (shadow maps, the water depth copy) use the depth target right before, so they keep reading depth.

## 2026-11-21: REWIND waits for its valid bit; WRITE_DATA stays queued

The command processor now stops at a `REWIND` until the CPU marks it valid and then reads the rest of the buffer again, instead of skipping the packet. Reason: GTA V chains each graphics buffer to the next by patching an INDIRECT_BUFFER behind a REWIND and only then setting the REWIND valid; skipping it lost that work ([gpu.md](gpu.md)). The wait has no time limit, as on the hardware; it ends with the process and prints a note after 5 s.

Landing `WRITE_DATA` on the GPU's timeline (a blit into the guest-memory buffer, so dispatches after it see the value) was tried for GTA V's culling queue and dropped: it did not change GTA V's stall and Minecraft's title fell from 56 to 43 flips a second, because every such write makes the renderer wait before it next reads guest memory. Writes stay queued until the queue drains, which is known to be wrong for a dispatch that reads what a write before it set.

## 2026-11-21: GTA V's add-on entitlements are reported as installed

`sceNpEntitlementAccess` reports GTA V's three Story Mode add-on labels as installed, so the game counts Story Mode as owned. Reason: the PS5 disc and store versions ship Story Mode as an add-on of the base package, and without it the game stops at a purchase screen. vic agreed (2026-11-21); the game's own files are all there, nothing is unlocked that the package lacks.

## 2026-11-21: Programs compile on worker threads; while a pipeline compiles, draws to redrawn targets are skipped and everything else waits

MSL and pipelines compile on a pool of compile threads instead of the GPU thread. A draw whose pipeline is not ready waits up to 4 ms (8 ms per frame), then is skipped for that frame only if all its targets were drawn to in an earlier frame and in the last two frames (or are display buffers) and its programs do not write memory; otherwise it waits. Dispatches always wait.

Reason: a cold Metal cache stopped the GPU thread for seconds (Superliminal: no frame for about 10.7 s before its menu, about 3,100 submitDone warnings). Waiting alone, with compiles in parallel, already cut most of that; skipping everything instead would lose draws whose output is made once, which no later frame repairs, and the first version of the skip lost a depth clear that way. Measured on Superliminal with `PS5SLOP_MSL_UNIQUE=1`, three runs each: time to menu 33.7 to 35.8 s before, 29.5 to 31.6 s after; longest flip gap after start-up 10.6 to 10.9 s before, 8.0 to 8.2 s after; submitDone warnings 2,956 to 3,438 before, 845 to 1,306 after. Only 11 draws were skipped there; the rest of the gain is compiling in parallel. Method and details in [gpu.md](gpu.md#compiling-off-the-gpu-thread-2026-11-21).

## 2026-11-21: Graphics submissions wait for the previous frame

`sceAgcDriverSubmitDcb` and the other graphics submits return only once every submission up to the last flip before it has run and its Metal work is done, unless the GPU thread is polling memory or 5 s pass. Reason: Isaac reuses a frame's command and vertex memory two frames later without waiting for the GPU, which is safe on a console that keeps up and not with our GPU thread, which can lag; its room backdrops were drawn from the next frame's data ([gpu.md](gpu.md#isaacs-black-rooms-2026-11-21)). Throttling every submission cost Superliminal a third of its frame rate, so frames are found from the flip packets our builders write. The polling exception keeps titles whose GPU work waits for the CPU from stalling. Titles that flip from the CPU are not throttled.

## 2026-11-21: Compute loops are cut off too, with a budget per dispatch

Compute programs with a backward branch give up after 16,384 backward branches per thread or 2^30 per dispatch; in programs with `s_barrier` the group leaves together at the next barrier ([shaders.md](shaders.md)). This replaces the rule that compute programs are left alone. Reason: GTA V's culling program looped on the GPU after its process was killed and the Mac needed a reboot; a thread that leaves early can leave others waiting at a barrier, but a loop that never ends is certain to hang, and leaving together at the barrier avoids the strand in the common case. The budget per dispatch is there because a per-thread limit alone lets a runaway in a large grid run for minutes. Checked on the GPU with loops that also end at a bound from user data, so the test cannot hang the GPU if the guard fails.

## 2026-11-21: Guest memory ordering across windows is opt-in

`PS5SLOP_GPU_FENCE=1` makes every encoder that reads guest memory wait (`MTLFence`) for the last one that wrote it; it is off by default. Reason: Metal orders work only within one buffer object and our windows are several buffers over the same memory, which may be why GTA V's culling program reads incomplete input when batched, but that is not shown, and the fence costs Minecraft's title about 18% of its frames.

## 2026-11-21: Direct memory mappings share their physical pages

Mappings of direct memory now use the physical offset: two mappings of one range see the same bytes, and an allocation zeroes its range ([hle-kernel.md](hle-kernel.md)). This reverses the decision of 2026-11-20 to keep fresh pages per mapping. Reason: GTA V maps the same physical pages at two addresses in its Story Mode load; the Minecraft breakage that stopped an earlier attempt did not show (title, and the other regression titles, draw as before), though Minecraft in a world was not tried. It did not cure GTA V's faults.

## 2026-11-21: Submissions that do not end a frame do not wait for the GPU

Their releases, memory writes and the queue's interrupt take effect when Metal reports their command buffers done, polled by the GPU thread, instead of after a commit and wait at the end of each submission; `DMA_DATA` fills of up to 64 bytes are queued with them. Frames still end with a wait. Reason: Superliminal's GPU thread spent about 90% of its time waiting for Metal, about 85 times a frame ([gpu.md](gpu.md#fewer-waits-for-the-gpu-2026-11-21)). Labels and interrupts now arrive when the GPU work is done rather than when it was encoded, which is closer to the console.

## 2026-11-21: The renderer waits only for work that may write what it reads, and checks HTILE on the GPU

The translator reports the V#s of every buffer store; CPU reads of HTILE, CMASK, DCC and unaligned indices wait only when encoded work may write there, and HTILE words a program may write are checked by a one-thread program and a conditional full-screen depth write instead. Reason: these waits were most of what was left of Superliminal's GPU waits. `DMA_DATA` as Metal blits was tried and dropped (Astro Bot's queues timed out on labels the blits wrote late).

## 2026-11-21: Threads about to block give their vCPU to threads that have none

HVF allows 64 vCPUs per VM and Astro Bot starts about 70 threads; its loading thread was one of those left in the interpreter. A thread about to block in a host function now gives its vCPU back while a running thread has none. Reason: Astro Bot's title came up about twice as fast ([jit.md](jit.md)). The cost, a vCPU destroyed and made again, is paid only while more than 64 threads run at once.

## 2026-11-21: Compute queues' WRITE_DATA into the GPU area lands in GPU order

A `WRITE_DATA` from the command stream (and its NOP form) is now a blit into the GPU area window when the queue never draws, the destination is in the GPU area and nothing unapplied is queued before it; all other writes stay queued until the queue drains, as decided earlier today. Reason: GTA V's culling buffer resets counters, bounds and table words with `WRITE_DATA` before each object's dispatches, in one submission of about 3,000 dispatches; with the writes queued, the dispatches read stale memory and one stored through a garbage descriptor across the game's own code, which caused the CPU faults of the Story Mode load ([gpu.md](gpu.md#the-games-code-overwritten-by-the-culling-queue-2026-11-21-later)). The limits come from Metal ordering work only within one buffer object: the area window is the one object dispatches share, while draws read through other windows, so a write landed for a graphics queue could reach memory before an earlier draw read it. A wider first version coincided with a broken Astro Bot title in runs that shared the GPU with other emulators; not pinned down, but the narrow rule is what GTA V needs. `PS5SLOP_GPU_QUEUED_WRITES=1` restores the old behaviour.

## 2026-11-21: Programs do not store into module images

Stores and atomics of translated programs skip the image of the loaded module or executable that their memory window covers. Reason: GTA V's programs live in its executable, so their window covers its code and globals, and a store through a wrong descriptor overwrote the game's code. Assumed, not verified: the console's GPU does not write module images (no title is known to). The guard costs one compare per store.

## 2026-11-21: The GPU-area window never shrinks

When the window over the GPU area is made again after a mapping change, it reaches at least as far as any window before it in the run, not only as far as the current program's user data points. Reason: programs load descriptors from memory and reach buffers through them, so user data does not bound what they read, and reads past the window return 0; Superliminal lost its geometry every third or sixth frame in about half its runs ([gpu.md](gpu.md#superliminals-flicker-and-dark-picture-2026-11-21)). A window over the whole 16 GiB area, or up to the highest mapping, fixed it too but cut Superliminal's and Minecraft's frame rates by half or more, and a reach that expired after 64 to 128 frames still flickered. A program whose pointers lie outside the area no longer asks for the whole area when no window exists. Memory that no program's user data ever pointed into is still out of reach.

## 2026-11-21: Float video-out formats are shown as linear light

Frames whose video-out format has top byte `0xC0` (Superliminal's `0xC001000600000000`, RGBA16Float) are encoded as sRGB in the window, `shot` and `frames`. Reason: Superliminal's last pass decodes sRGB before it writes the buffer, so the buffer holds linear light, and showing it as stored made every scene far too dark. No public source names the format; other float formats are assumed alike, and the PS4's float format is left as it was.

## 2026-11-21: A colour target drawn at a 2D array's address hides the array

Sampling an address reads the colour target drawn there when that draw came after the last 2D-array or 3D write to the address; 2D storage writes then go to the target too, and a later array write makes the array current again. Before, arrays were looked up first whenever one existed for the address. Reason: Superliminal reuses the memory of a full-screen surface that a program once wrote as a 2D array, and from then on every read of that surface returned the array's old contents, so the picture froze in level 1's rook corridor while the game ran on ([gpu.md](gpu.md#superliminals-frozen-picture-at-the-rook-2026-11-21)).

## 2026-11-21: Sampled textures get their whole mip chain

Textures read from guest memory in tiled layouts are made with every level the descriptor names, each detiled from its place in the chain; the mip tail follows SharpEmu-mac's tables. Reason: Superliminal's reflection probes need their levels, since Unity picks the probe's level by roughness, and with level 0 alone every surface looked like a mirror. Cost: about a third more texture memory and one detile dispatch per level each frame; Superliminal's first room ran at 56 flips a second. Linear textures and descriptors with a base level above 0 still get level 0.

## 2026-11-21: Compile threads per performance core; repeated requests first

The compile pool has one thread per performance core (`hw.perflevel0.physicalcpu`, 10 on vic's Mac; `PS5SLOP_COMPILE_THREADS` still overrides), render pipelines wait outside the pool for their functions, and a draw skipped again in a later frame moves its programs' compiles to the front of the queue. Reason: with a cold Metal cache, Superliminal's level start was black or missing geometry for seconds while its materials waited behind 54 s of compiling for shaders the title draws once at start-up. Considered and not done: hiding frames that had skipped draws (vic: skipping geometry while compiling is fine, the compile time is the problem), and an archive of compiled programs of our own (Metal's cache already persists across runs of the same build).

## 2026-11-21: Translated programs loop only where the program does

Programs with branches are guarded blocks in address order, with a loop only around each backward branch's range, instead of one `for (;;) switch (pc)` around the whole program; consecutive guarded vector instructions share one `if`. Reason: the whole-program loop made Metal's compiler spend over a third of its time in loop passes; Superliminal's largest programs compile in 780 ms instead of 1,070 ms. Overlapping loops keep the old form.

## 2026-11-21: Translated programs compile in safe math mode

`MTLCompileOptions.mathMode` is `Safe` for every translated program. Reason: with Metal's default fast math, one of Superliminal's material programs produced NaN pixels that bloom spread into a white glow; with safe (or relaxed) math it did not, same view and inputs (fast 6 of 14 angles, safe 0, relaxed 0). Safe rather than relaxed because the translated code mirrors the hardware's instructions one by one and the hardware does not reassociate them either. Measured cost: none in compile time; Superliminal's first room at 58 to 59 flips a second. Heavier titles were checked only by their regression screens.

## 2026-11-21: GTA V's intro: translate what Bink uses, leave `pcmpistri`

The JIT now translates `pmaddubsw`, `pshuflw`/`pshufhw` and `vtestps`/`vtestpd` (their SSE and VEX forms, not the MMX ones), the instructions GTA V's Bink movie decoder and RAGE ran in the interpreter during the intro ([jit.md](jit.md#gta-vs-intro-movie-2026-11-21)). `pcmpistri`, now the largest interpreted instruction there, stays in the interpreter. Reason: the Bink forms were 93% of all VM exits and the movie took about 15 times as long as it does now; `pcmpistri` sits in Sony's `libc`, whose immediates we do not read, so translating it means all of SSE4.2's string modes, a larger job for 0.25 s of a 10 s window. Minecraft runs it more (7.8 M per 40 s), so it stays the next candidate.

## 2026-11-21: Depth-layout arrays without a depth target read as zeros; linear 3D images read from memory

A 2D-array or cube T# in a depth swizzle mode that no depth target backs now binds a zero-filled texture, as 2D ones already did, instead of skipping the draw; linear 3D images are copied from guest memory; `SW_256B_S` textures are detiled. Reason: these were every image warning the survey found (Astro Bot's intro and loading, GTA V's Story Mode load). Zeros are a guess at what an untouched shadow map holds, not its real contents; drawing with them lets the rest of each draw through. Astro Bot's logo fly-in was black before and after (in the base build the letters appear black too, a little later), so the zeros are not what makes it black. GPU reports now print each distinct message once (64 kinds) instead of the first 24 of any kind, so a flood of one warning no longer hides the others.

## 2026-11-21: Maps without a fixed address start at `0x11_0000_0000`

Direct memory, flexible memory, reservations and `mmap` without `MAP_FIXED` now search from `0x11_0000_0000`; the first 4 GiB of the user area (`0x10_0000_0000..0x11_0000_0000`) is left to fixed maps, and the GPU area window grows from 16 to 20 GiB so it still covers 16 GiB of searched maps. Reason: GTA V's memory set-up (its own code, read locally) puts a 3.25 GiB streaming arena at `0x10_0000_0000` without reserving it and batch-maps streamed pages there with `MAP_FIXED`. Our maps had put libc's heap, its 4.5 GiB render pool and more in that range, so streamed pages replaced live render targets and GPU clears landed on CPU objects: the Story Mode faults ([gpu.md](gpu.md#what-stopped-gta-vs-story-mode-2026-11-21-night)). Assumed: the PS5 kernel does not place maps there; where it does place them is unknown. The 4 GiB figure covers GTA V's arena and is otherwise arbitrary.

## 2026-11-21: Run-time bits masked into a descriptor dword are dropped; `s_setreg` is ignored

When a program ORs or ANDs a register into a descriptor dword in place (`s_or_b32 s15, s15, vcc_lo`), the translator keeps the dword's origin and binds the descriptor as it is in memory, without those bits. GTA V's material programs set per-quad bits in a sampler's last dword this way (from `s_quadmask_b64`); their meaning is not known. `s_setreg_b32` (rounding and denormal modes) does nothing. Reason: both made whole programs fail to translate, so their draws were skipped; an approximate sample is closer than none.

## 2026-11-21: Astro Bot's lighting: ray-query emulation, a third memory window, lockstep wave64 groups

`image_bvh_intersect_ray` is emulated in MSL over guest memory as Mesa radv's software path does it; compute programs get a third guest-memory window; a compute group that is one wave64 talking through LDS without barriers runs its two SIMD groups in lockstep (barriers around LDS instructions, branches voted across the group, a loop guard that leaves as a group); a GDS `ds_append` uses M0's high half as its base; a storage write to a colour target applies a pending fast clear first. Reason: Astro Bot lights its G-buffer only in ray-tracing compute programs, and each of these stood between them and the screen ([gpu.md](gpu.md#astro-bots-logo-2026-11-21-later)). Emulating the node tests keeps the game's own BVHs (Metal's ray tracing would need them rebuilt); a third window was the smallest change that reaches the tables in 0x4_xxxxxxxx without guessing which two of three regions a program needs; lockstep is limited to one-wave groups because voting across waves that are independent on hardware would change what they compute. The lockstep barriers cost time in those 8 programs; measured only as the title showing at about the same time as before (70 to 90 s).

## 2026-11-21: Samplers get a slot per descriptor source

A sampler's Metal slot is chosen by where its descriptor comes from, not by the register it was loaded into. Reason: GTA V's final composite loads the same samplers into several registers, needed more than Metal's 16 sampler slots and never compiled, so the screen showed a stale target ("a melting pot"). Textures keep their per-register slots, since the pre-pass that finds image atomics numbers them that way and 31 slots have not run out.

## 2026-11-21: Rectangle lists take their index offset as the base instance

A rectangle-list draw with `GE_INDX_OFFSET` set draws with the offset as Metal's base instance; the rectangle vertex function adds it to the vertex ids and subtracts it from the instance id. Reason: GTA V's deferred lighting draws full-screen rectangles at offsets that are not multiples of 3, and they were skipped. A vertex start of twice the offset would have needed offsets divisible by 3; the base instance carries any offset and needs no extra binding.

## 2026-11-21: 64-bit bitwise and swap atomics as two 32-bit atomics

`buffer_atomic_and_x2`, `or_x2`, `xor_x2` and `swap_x2` run as two 32-bit atomics: the bitwise forms high dword first, `swap_x2` low dword first. Reason: Metal has no 64-bit forms of these and GTA V's compute programs use them; with this order a reader that polls the high dword of a record written with `swap_x2` sees the low dword too, if one thread's relaxed atomics reach memory in order (not shown). Bitwise results are exact apart from a torn return value. Other 64-bit atomics stay untranslated.

## 2026-11-21: Tiled 3D images follow AMD's addrlib

`SW_4KB_S` 3D images are read with addrlib's block patterns and mip placement (Mesa, MIT), every level before the mip tail, detiled on the CPU once a frame. Reason: GTA V's G-buffer materials sample a 64x64x32 RGBA8 volume and the draws were skipped without it. The CPU copy (512 KiB a frame for this one) was simpler than a 3D detile kernel; measure before moving it to the GPU.

## 2026-11-21: Descriptor dwords masked through another register keep their origin

An OR or AND of a descriptor dword with anything that is not itself a descriptor dword keeps the dword's origin in the result, in place or not; the changed bits are dropped as before. Reason: GTA V edits a T#'s type through `vcc_lo` and ORs per-quad sampler bits into a copy, and those programs (a 2D-array copy run about 1,500 times, ten material draws a frame) did not translate. Risk: a masked value used as an index rather than a descriptor would keep its unmasked origin; none seen.

