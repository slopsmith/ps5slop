# Architecture

ps5slop runs PS5 games on Apple Silicon Macs the way Wine runs Windows programs: the game's own x86-64 code runs, and everything below it (the Orbis kernel interface and system libraries) is implemented in our process. The GPU command stream is translated to Metal. We do not emulate Orbis itself (see [decisions.md](decisions.md)).

Status: commercial 2D and 3D PS5 titles boot and render through the HVF JIT, HLE OS and Metal backend (see [COMPATIBILITY.md](../COMPATIBILITY.md)).

## Big picture

```
 macOS process (Rust)
+--------------------------------------------------------------+
| loader: SELF/ELF, NID imports -> HLE functions               |
| CPU: x86-64 interpreter + ARM64 JIT in a Hypervisor VM       |
| HLE: libkernel (threads, memory, files, time), libc, ...     |
|      libSceVideoOut, libSceAudioOut, libScePad, ...          |
| GPU: libSceAgc -> PM4 processor -> Metal, GFX10 -> MSL       |
| guest address space: one host reservation we own             |
+--------------------------------------------------------------+
```

## Docs by area

- [cpu.md](cpu.md): running x86-64 on ARM64: Rosetta, FEX, our own translator, and whether Hypervisor.framework helps.
- [jit.md](jit.md): the translator and the VM it runs in.
- [memory.md](memory.md): the guest's address space, and what macOS lets us map.
- [loader.md](loader.md): executable format, NIDs and linking.
- [packages.md](packages.md): what a `.pkg` holds, and why we load unpacked folders.
- [hle-kernel.md](hle-kernel.md): how guest calls reach host functions, and which ones exist.
- [gpu.md](gpu.md): the PS5 GPU, AGC and the Metal backend.
- [shaders.md](shaders.md): the GFX10 decoder and MSL translator.
- [audio.md](audio.md): AudioOut, NGS2 and the host's output device.
- [test-targets.md](test-targets.md): test programs, commercial targets and the oracles we have without a PS5.
- [debugging.md](debugging.md): logging, debug switches, the control socket and write watchpoints.
- [input.md](input.md): controllers, keyboard, vibration, light bar and trigger effects.
- [savestates.md](savestates.md): save states: what a save keeps, how threads stop for it, the file format and its limits.
- [references.md](references.md): prior art, licenses, and what we read.
- [legal.md](legal.md): the hard rules.
- [decisions.md](decisions.md): what has been decided, when and why.

## Crates

- `diag`: `tracing` logging setup, rate-limited reporting (`FirstN`, `Distinct`), the per-run problem report (`problems`) and the typed `Config` read once from `PS5SLOP_*` switches. See [debugging.md](debugging.md).
- `pkg`: reads fake-signed PS4 and PS5 packages (including block-by-block Kraken decompression) and extracts them into `app0` folders, plus the `ps5slop-pkg` tool. See [packages.md](packages.md).
- `loader`: reads ELF and fake-signed SELF files, the dynamic section and NID imports, and lays out an image with relocations applied, plus the `ps5slop-elf` tool. See [loader.md](loader.md).
- `memory`: the guest address space, with protected regions, trap regions and write watchpoints. See [memory.md](memory.md).
- `x86`: the x86-64 decoder and interpreter. See [cpu.md](cpu.md).
- `a64`: ARM64 instruction encoder for the JIT. See [jit.md](jit.md).
- `hvf`: Apple Silicon Hypervisor.framework wrapper (VM, stage-2 page tables, vCPUs with hardware TSO). See [jit.md](jit.md).
- `jit`: x86-64 to ARM64 translator, segmented code cache with sampling eviction and the VM execution engine. See [jit.md](jit.md).
- `gfx10`: GFX10 shader decoder, AGS parser, shader capture format and direct-to-MSL translator (`translate/`: `flow`, `scalar`, `valu`, `half`, `mem`, `sources`, `image`, `emit`, `prelude`). See [shaders.md](shaders.md).
- `gpu`: GPU register decoders, tiling/detiling, background shader compile pool, PM4 packet processor (`pm4/`: `sync`, `copy`, `work`), Metal renderer (`metal/`: `commands`, `memory`, `targets`, `images`, `detile`, `htile`, `programs`, `bind`, `draw`, `compute`, `formats`, `readback`, `inspect`, `saved`), frame capture (`capture`) and deterministic offline replay (`replay`, `ps5slop-replay`). See [gpu.md](gpu.md) and [debugging.md](debugging.md).
- `hle`: host implementations of Orbis/Prospero libraries behind trap stubs (libkernel, VideoOut, AudioOut/AudioOut2, NGS2, Pad, Agc, Ampr, Font, SaveData, AvPlayer, ...). See [hle-kernel.md](hle-kernel.md).
- `savestate`: the save state file format (a small binary encoding in zstd sections) and the guest's wall clock. See [savestates.md](savestates.md).
- `ps5slop`: the `ps5slop run` command, window, input polling and control socket.
- `mkelf`: `ps5slop-mkelf`, which turns an ELF linked with `ld.lld -shared` into an Orbis executable, for our own test programs. See [test-targets.md](test-targets.md).
- `asmtest`: `ps5slop-asmtest` (FEX x86-64 tests against the interpreter) and `ps5slop-fuzz` (differential instruction fuzzer against Rosetta). See [cpu.md](cpu.md).
