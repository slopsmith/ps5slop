# ps5slop

A PS5 compatibility layer for Apple Silicon Macs. It runs a game's own x86-64 code, implements the Orbis kernel interface and system libraries in the host (HLE, the way Wine does it for Windows), and translates the GPU's command stream to Metal.

Nothing runs yet. The architecture is still being discussed.

You supply your own games, dumped from your own console. Nothing from Sony is part of this project. See [docs/legal.md](docs/legal.md).

## Setup

`nix develop` gives the toolchain: Rust with a matching clippy, and LLVM's `lld` for building test programs.

## Tools

```sh
cargo run --release -p pkg -- info GAME.pkg
cargo run --release -p pkg -- extract GAME.pkg OUT_DIR   # fake-signed packages; --passcode if not all zeros
cargo run --release -p loader -- info OUT_DIR/eboot.bin
cargo run --release -p loader -- imports OUT_DIR/eboot.bin --names names.txt   # label NIDs of names you know
cargo run --release -p loader -- load OUT_DIR/eboot.bin   # lay out, relocate, list missing imports
```

## Docs

- [Architecture](docs/architecture.md): overview, milestones and index of the other docs
- [CPU](docs/cpu.md), [memory](docs/memory.md), [loader](docs/loader.md), [packages](docs/packages.md), [GPU](docs/gpu.md)
- [Test targets and oracles](docs/test-targets.md)
- [References and prior art](docs/references.md)
- [Legal boundaries](docs/legal.md)
- [Decisions](docs/decisions.md)

Licensed under GPL-2.0-or-later. See [LICENSE](LICENSE) and the license notes in [docs/references.md](docs/references.md).
