# ps5slop

A PS5 compatibility layer for Apple Silicon Macs. It runs a game's own x86-64 code, implements the Orbis kernel interface and system libraries in the host (HLE, the way Wine does it for Windows), and translates the GPU's command stream to Metal.

Our own small test programs run under the interpreter. No game or homebrew app does yet.

You supply your own games, dumped from your own console. Nothing from Sony is part of this project. See [docs/legal.md](docs/legal.md).

## Setup

`nix develop` gives the toolchain: Rust with a matching clippy, and LLVM's `lld` for building test programs. `cargo run` and `cargo test` sign each binary ad hoc with the hypervisor entitlement (`scripts/hv-run.sh`), which the JIT's VM needs.

## Tools

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
cargo run --release -p ps5slop -- run games/TITLE   # an app folder; opens a window for VideoOut
cargo run --release -p ps5slop -- run --headless --dump-frame out.ppm --after 10 games/TITLE
cargo run --release -p ps5slop -- run --interpreter games/TITLE   # no JIT; by default guest code is translated and runs in a Hypervisor.framework VM
PS5SLOP_JIT_VERIFY=1 PS5SLOP_JIT_STATS=1 cargo run ...   # check the JIT against the interpreter / see what it leaves to the interpreter
```

A DualShock 4, DualSense or other controller macOS knows (through its GameController framework) works as the pad; plug it in or pair it before starting. The keyboard works too, and the two combine: arrow keys for the D-pad, X Cross, C Circle, Z Square, V Triangle, Q/E for L1/R1, 1/3 for L2/R2, Enter for Options, T for the touchpad button. Closing the window ends the guest.

## Docs

- [Architecture](docs/architecture.md): overview, milestones and index of the other docs
- [CPU](docs/cpu.md), [memory](docs/memory.md), [loader](docs/loader.md), [packages](docs/packages.md), [HLE](docs/hle-kernel.md), [JIT](docs/jit.md), [GPU](docs/gpu.md)
- [Test targets and oracles](docs/test-targets.md)
- [References and prior art](docs/references.md)
- [Legal boundaries](docs/legal.md)
- [Decisions](docs/decisions.md)

Licensed under GPL-2.0-or-later. See [LICENSE](LICENSE) and the license notes in [docs/references.md](docs/references.md).
