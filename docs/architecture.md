# Architecture

ps5slop runs PS5 games on Apple Silicon Macs the way Wine runs Windows programs: the game's own x86-64 code runs, and everything below it (the Orbis kernel interface and system libraries) is implemented in our process. The GPU command stream is translated to Metal. We do not emulate Orbis itself (see [decisions.md](decisions.md)).

Status: milestones 1 and 2 are done; our own test programs run under the interpreter. Most of what follows milestone 2 is still a proposal.

## Big picture

```
 macOS process (Rust)
+--------------------------------------------------------------+
| loader: SELF/ELF, NID imports -> HLE functions               |
| CPU: interpreter first, a JIT later (maybe inside an HVF VM)  |
| HLE: libkernel (threads, memory, files, time), libc, ...     |
|      libSceVideoOut, libSceAudioOut, libScePad, ...          |
| GPU: libSceAgc -> PM4 interpreter -> Metal, shaders -> MSL   |
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
- [gpu.md](gpu.md): the PS5 GPU, AGC and the Metal plan.
- [test-targets.md](test-targets.md): test programs, commercial targets and the oracles we have without a PS5.
- [references.md](references.md): prior art, licenses, and what we read.
- [legal.md](legal.md): the hard rules.
- [decisions.md](decisions.md): what has been decided, when and why.

## Proposed milestones

Each step has a test program we can run and check. Steps 1 to 4 need only programs we write ourselves.

1. **Load.** Parse our own test ELF (Sony program headers, `NID#LIB#MOD` imports), map it, resolve imports to HLE stubs, list what is missing. No code runs. *Status: done. Parsing, layout and relocation work on vic's OpenOrbis build and on our own `hello` test program; imports bind to HLE trap stubs.*
2. **Hello.** The interpreter runs a test program that prints through an HLE `sceKernelWrite` (or libc `write`) and exits with a status we check. *Status: done; `hello` prints through HLE `write` and exits with status 0.*
3. **Kernel basics.** Our test programs exercise memory allocation (direct and flexible), TLS through FS, threads and mutexes, time and files under `/app0`. The same tests, built for the PS4, run on vic's console for comparison where that is practical.
4. **Pixels without a GPU.** A test program fills a buffer on the CPU and flips it through HLE `libSceVideoOut`; the Mac shows it in a Metal window.
5. **A commercial title starts.** Deltarune (or Balatro) reaches its first GPU submission, with command buffers parsed and dropped. **JIT decision point:** measure interpreter speed, instruction mix and the rate of HLE calls here.
6. **Clear screen.** PM4 clears and flips reach Metal.
7. **2D.** Deltarune's title screen: textured quads, blending, translated shaders.
8. **3D.**

## Crates

- `pkg`: reads fake-signed PS4 packages and extracts them into `app0` folders, plus the `ps5slop-pkg` tool. See [packages.md](packages.md).
- `loader`: reads ELF and fake-signed SELF files, the dynamic section and NID imports, and lays out an image with relocations applied, plus the `ps5slop-elf` tool. See [loader.md](loader.md).
- `memory`: the guest address space, with protected regions and trap regions. See [memory.md](memory.md).
- `x86`: the decoder and interpreter. See [cpu.md](cpu.md).
- `hle`: host functions behind trap stubs. See [hle-kernel.md](hle-kernel.md).
- `ps5slop`: the `ps5slop run` command, which puts the pieces together.
- `mkelf`: `ps5slop-mkelf`, which turns an ELF linked with `ld.lld -shared` into an Orbis executable, for our own test programs. See [test-targets.md](test-targets.md).

A `gpu` crate comes later.
