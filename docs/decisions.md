# Decisions

Newest last. Each entry: date, decision, reason. When a decision is reversed, add a new entry and mark the old one as superseded.

## 2026-11-19: HLE only, no LLE of Orbis

We run the game's own code and implement the Orbis kernel interface and system libraries in the host process. We do not emulate the PS5's boot chain, secure processor, hypervisor or SSD controller.

Reason: those parts are undocumented, and LLE of them would be roughly 100x the work. Running the Orbis shell under HLE is a late stretch goal at most.

## 2026-11-19: Rust, Cargo workspace, jj

Reason: vic's preference, and the same setup as 86slopper, so ideas and possibly code can move between the two.

## 2026-11-19: Correctness oracles without a PS5

vic has no PS5. Correctness rests on oracles that need no PS5: public instruction test suites, an interpreter as the reference for any JIT, public documentation, other emulators' notes, and our own test programs run on vic's PS4 for the kernel and library side (same Orbis lineage). Anything that rests only on documentation or other emulators is marked unverified.
