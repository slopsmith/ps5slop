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

Open: whether symbol tables taken from modules may be committed, or only generated on the user's machine. Until vic decides, nothing taken from a module is committed.
