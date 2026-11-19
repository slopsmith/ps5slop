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

## 2026-11-19: OpenOrbis for cross-compiling test programs only

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
