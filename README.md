# ps5slop

A PS5 compatibility layer for Apple Silicon Macs. It runs a game's own x86-64 code, implements the Orbis kernel interface and system libraries in the host (HLE, the way Wine does it for Windows), and translates the GPU's command stream to Metal.

Some commercial games run, to different degrees: see the [compatibility list](https://slopsmith.github.io/ps5slop/compat/), built from the reports in [slopsmith/ps5slop-compat](https://github.com/slopsmith/ps5slop-compat).

You supply your own games, dumped from your own console. Nothing from Sony is part of this project. See [docs/legal.md](docs/legal.md).

## Setup

`nix develop` gives the toolchain: Rust with a matching clippy, LLVM's `lld` for building test programs, and FFmpeg's libraries (Ajm's MP3 decoding). `cargo run` and `cargo test` sign each binary ad hoc with the hypervisor entitlement (`scripts/hv-run.sh`), which the JIT's VM needs.

## Running games

Players: download the app and follow [docs/user/getting-started.md](docs/user/getting-started.md).

From source: `cargo run --release -p ps5slop -- run GAME.pkg` (or an extracted folder with `eboot.bin`). `ps5slop run --help` lists the options. Some titles need `scripts/fetch-fonts` once. Every environment variable is in [docs/environment.md](docs/environment.md).

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

- [Getting started](docs/user/getting-started.md), [playing games](docs/user/playing.md), [reporting problems](docs/user/reporting.md), [troubleshooting](docs/user/troubleshooting.md) and the [command line](docs/user/command-line.md)
- [Compatibility](https://slopsmith.github.io/ps5slop/compat/): what each title does so far, and [how to report a game](https://github.com/slopsmith/ps5slop-compat)

For developers and agents, in [docs/](docs/):

- [Architecture](docs/architecture.md): overview, milestones and index of the other docs
- [CPU](docs/cpu.md), [memory](docs/memory.md), [loader](docs/loader.md), [packages](docs/packages.md), [HLE](docs/hle-kernel.md), [input](docs/input.md), [audio](docs/audio.md), [JIT](docs/jit.md), [GPU](docs/gpu.md), [shaders](docs/shaders.md)
- [Debugging](docs/debugging.md), [environment variables](docs/environment.md), [save states](docs/savestates.md)
- [Launcher app](docs/launcher.md)
- [Test targets and oracles](docs/test-targets.md)
- [References and prior art](docs/references.md)
- [Legal boundaries](docs/legal.md)
- [Decisions](docs/decisions.md)

Licensed under GPL-2.0-or-later. See [LICENSE](LICENSE) and the license notes in [docs/references.md](docs/references.md).
