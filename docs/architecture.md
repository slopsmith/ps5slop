# Architecture

ps5slop runs PS5 games on Apple Silicon Macs the way Wine runs Windows programs. The game's own x86-64 code runs, and everything below it (the Orbis kernel interface and system libraries) is implemented in our process. The GPU command stream is translated to Metal.

```
 macOS process (Rust), one per game
+--------------------------------------------------------------+
| loader: SELF/ELF, NID imports -> HLE functions               |
| CPU: x86-64 interpreter + ARM64 JIT in a Hypervisor VM       |
| HLE: libkernel (threads, memory, files, time), libc, ...     |
|      VideoOut, AudioOut, NGS2, Pad, SaveData, ...            |
| GPU: libSceAgc -> PM4 processor -> Metal, GFX10 -> MSL       |
| guest address space: one host reservation we own             |
+--------------------------------------------------------------+
```

## Design choices

These hold across the project. Each area doc explains its own design in more detail.

- **HLE, not LLE.** We implement the system libraries and kernel interface in the host, and never emulate the PS5's boot chain, secure processor, hypervisor or SSD controller. Those parts are undocumented, and emulating them would be far more work for nothing a game needs.
- **No PS5 to measure on.** Correctness rests on oracles that need no PS5: public instruction test suites, the interpreter as the reference for the JIT, differential fuzzing against Rosetta, Metal tests of translated GPU programs, public documentation, and our own test programs on vic's PS4. PS5 behaviour that rests only on documentation or other emulators is marked unverified. See [test-targets.md](test-targets.md).
- **Our own CPU translator in a Hypervisor.framework VM.** Rosetta is general-purpose only through macOS 27, and FEX does not fit macOS. Translated code runs on HVF vCPUs with hardware TSO and our own page tables mapping guest addresses one to one, which removes the two largest costs of x86 on ARM. The interpreter came first and stays as the oracle and the fallback. See [cpu.md](cpu.md) and [jit.md](jit.md).
- **One host thread per guest thread.** No cooperative scheduling. Guest memory is one shared host reservation.
- **Metal only, straight from GFX10.** All drawing goes through Metal, with no CPU rendering and no Vulkan or SPIR-V layer in between. GFX10 programs are translated to MSL text by our own `gfx10` crate. See [gpu.md](gpu.md) and [shaders.md](shaders.md).
- **One process per game.** Hypervisor.framework allows one VM per process, HLE state is process-global, and a GPU hang poisons Metal for the whole process. The launcher therefore starts each game as a child `ps5slop run` process. See [launcher.md](launcher.md).
- **Clean room.** What we may read, run and commit is in [legal.md](legal.md). Every source we read is recorded in [references.md](references.md).
- **Diagnostics through `tracing` and one typed config.** Logging uses `tracing` targets. Every `PS5SLOP_*` switch is read once into `diag::Config` and listed in [environment.md](environment.md). Problems a run hits are counted into one end-of-run problem report. See [debugging.md](debugging.md).

## Crates

- `diag`: logging setup, rate-limited reporting, the problem report and the typed config.
- `pkg`: fake-signed PS4 and PS5 packages (PFS, Kraken), read in place or extracted. See [packages.md](packages.md).
- `loader`: ELF and fake-signed SELF files, dynamic sections, NID imports, relocation. See [loader.md](loader.md).
- `memory`: the guest address space, protections, trap regions and write watchpoints. See [memory.md](memory.md).
- `x86`: the x86-64 decoder and interpreter. See [cpu.md](cpu.md).
- `a64`, `hvf`, `jit`: the ARM64 encoder, the Hypervisor.framework wrapper and the translator with its code cache and VM engine. See [jit.md](jit.md).
- `gfx10`: the GFX10 decoder and the MSL translator. See [shaders.md](shaders.md).
- `gpu`: register decoding, tiling, the PM4 processor, the Metal renderer, the compile pool, frame capture and offline replay (`ps5slop-replay`). See [gpu.md](gpu.md).
- `hle`: the system libraries. See [hle-kernel.md](hle-kernel.md), [audio.md](audio.md) and [input.md](input.md).
- `savestate`: the save state format and the guest's wall clock. See [savestates.md](savestates.md).
- `ps5slop`: the `ps5slop` command, the game window, input polling and the control socket.
- `mkelf`: `ps5slop-mkelf`, which turns our own test programs into Orbis executables. See [test-targets.md](test-targets.md).
- `asmtest`: FEX's x86-64 tests against the interpreter, and a differential fuzzer against Rosetta.

The Mac launcher is in `apps/macos` ([launcher.md](launcher.md)). Docs for players are in [user/](user/getting-started.md) and are published on the website.

## Outside this repo

- Compatibility reports are issues in [slopsmith/ps5slop-compat](https://github.com/slopsmith/ps5slop-compat), accepted automatically. The website builds the compatibility pages and `compat.json` from them, and the launcher reads `compat.json`.
- Nightly builds come from a workflow shared through [slopsmith/ci](https://github.com/slopsmith/ci). The hosted runner has no Hypervisor.framework and only a paravirtual Metal device, so it does not start games and cannot run the JIT and HVF tests. Those run only on a real Mac.
