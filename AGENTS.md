# ps5slop: rules for agents

ps5slop is a PS5 compatibility layer (HLE, Wine-style) for Apple Silicon Macs, written in Rust. Start with [README.md](README.md) and [docs/architecture.md](docs/architecture.md).

## Legal (hard rules, details in [docs/legal.md](docs/legal.md))

- Never commit, embed, ship or link to proprietary Sony material: firmware, system modules (`.sprx` and similar), keys, game binaries, game assets, shader or microcode blobs.
- Never commit reverse-engineered parts of proprietary modules: disassembly, decompilation, function-by-function ports, tables or constants copied from a dump. No leaked SDK material either.
- Never decompile or disassemble Sony system code yourself (not even a few bytes around a call site). Public documentation others wrote may be used as a spec, whatever its origin; write our own code from it.
- Document interface facts (NID names, syscall numbers, calling conventions) only where publicly known, with the source. Keep them apart from implementation.
- User-supplied games, firmware and keys never enter the repo. Run `scripts/check-forbidden` (or `jj check`) before pushing.
- Check a project's license before reading its code, and never copy across incompatible licenses. Record what was read and taken in [docs/references.md](docs/references.md).
- No code that circumvents console security (exploits, key extraction, decryption), and no exploit details in the repo.
- Unsure whether something is OK? Stop and ask vic.

## Process

- Version control is jj (colocated with git). Small, focused changes with Conventional Commits descriptions. Edit the earlier change instead of adding fix-up changes. Do not push or add remotes; vic does that.
- Local alias for the check: `jj config set --repo aliases.check '["util", "exec", "--", "scripts/check-forbidden"]'`.
- Docs live in `docs/`, one area per file. Update the doc when a decision changes, and log decisions in [docs/decisions.md](docs/decisions.md) with the date and the reason.
- Be honest in docs: say what is verified (and how), what is assumed and what is unknown.
- Measure before optimising. Test against an oracle wherever one exists (an interpreter, a public test suite, differential testing, our own test programs on vic's PS4).
- There is no PS5 to measure on. Anything about PS5 behaviour that rests only on documentation or other emulators is marked unverified.
- Build and lint inside `nix develop . -c ...` (`cargo test`, `cargo clippy --workspace --all-targets`). Nix only sees files git tracks, and jj does not stage files: a new file must be in a committed change (not only in `@`) before Nix sees it.
- Use `pnpm`, never `npm`, if JavaScript tooling ever shows up.
