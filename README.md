# ps5slop

A PS5 compatibility layer for Apple Silicon Macs. It runs a game's own x86-64 code, implements the Orbis kernel interface and system libraries in the host (HLE, the way Wine does it for Windows), and translates the GPU's command stream to Metal.

Some commercial games run, to different degrees: see [COMPATIBILITY.md](COMPATIBILITY.md).

You supply your own games, dumped from your own console. Nothing from Sony is part of this project. See [docs/legal.md](docs/legal.md).

## Setup

`nix develop` gives the toolchain: Rust with a matching clippy, and LLVM's `lld` for building test programs. `cargo run` and `cargo test` sign each binary ad hoc with the hypervisor entitlement (`scripts/hv-run.sh`), which the JIT's VM needs.

## Running games

See [docs/user/running.md](docs/user/running.md): starting a game, controls, save states, captures and reporting problems. Every environment variable is in [docs/user/environment.md](docs/user/environment.md).

## Developer tools

```sh
cargo run --release -p pkg -- info GAME.pkg
cargo run --release -p pkg -- extract GAME.pkg OUT_DIR   # fake-signed packages; --passcode if not all zeros
cargo run --release -p loader -- info OUT_DIR/eboot.bin
cargo run --release -p loader -- imports OUT_DIR/eboot.bin --names names.txt   # label NIDs of names you know
cargo run --release -p loader -- load OUT_DIR/eboot.bin   # lay out, relocate, list missing imports
scripts/build-test-programs   # our own test programs -> target/test-programs/
scripts/run-test-programs     # build, run under ps5slop, compare with expected output
scripts/fetch-references && cargo run --release -p asmtest   # FEX's x86-64 tests against the interpreter
cargo run --release -p ps5slop -- run target/test-programs/hello.elf
cargo run --release -p pkg -- ls GAME.pkg   # list a package's app files
cargo run --release -p pkg -- cat GAME.pkg data/x.xml   # one app file to stdout
PS5SLOP_JIT_VERIFY=1 RUST_LOG=jit::stats=info cargo run ...   # check the JIT against the interpreter / see what it leaves to the interpreter
```

## Docs

For users, in [docs/user/](docs/user/):

- [Running games](docs/user/running.md)
- [Environment variables](docs/user/environment.md)
- [Compatibility](COMPATIBILITY.md): what each title does so far

For developers and agents, in [docs/](docs/):

- [Architecture](docs/architecture.md): overview, milestones and index of the other docs
- [CPU](docs/cpu.md), [memory](docs/memory.md), [loader](docs/loader.md), [packages](docs/packages.md), [HLE](docs/hle-kernel.md), [input](docs/input.md), [audio](docs/audio.md), [JIT](docs/jit.md), [GPU](docs/gpu.md), [shaders](docs/shaders.md)
- [Debugging](docs/debugging.md), [save states](docs/savestates.md)
- [Test targets and oracles](docs/test-targets.md)
- [References and prior art](docs/references.md)
- [Legal boundaries](docs/legal.md)
- [Decisions](docs/decisions.md)

Licensed under GPL-2.0-or-later. See [LICENSE](LICENSE) and the license notes in [docs/references.md](docs/references.md).
