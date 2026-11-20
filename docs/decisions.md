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
