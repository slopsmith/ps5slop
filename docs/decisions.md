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
