# ps5slop

A PS5 compatibility layer for Apple Silicon Macs. It runs a game's own x86-64 code, implements the Orbis kernel interface and system libraries in the host (HLE, the way Wine does it for Windows), and translates the GPU's command stream to Metal.

Nothing runs yet. The architecture is still being discussed.

You supply your own games, dumped from your own console. Nothing from Sony is part of this project. See [docs/legal.md](docs/legal.md).

## Docs

- [Architecture](docs/architecture.md): overview, milestones and index of the other docs
- [CPU](docs/cpu.md), [memory](docs/memory.md), [loader](docs/loader.md), [GPU](docs/gpu.md)
- [Test targets and oracles](docs/test-targets.md)
- [References and prior art](docs/references.md)
- [Legal boundaries](docs/legal.md)
- [Decisions](docs/decisions.md)

Licensed under GPL-2.0-or-later. See [LICENSE](LICENSE) and the license notes in [docs/references.md](docs/references.md).
