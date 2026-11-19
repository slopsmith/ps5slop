# GPU

How the PS5's GPU work reaches Metal. The command processor and the Metal plumbing exist; shader translation is in progress. Very little here is verified on a PS5. The GPU comes after the loader, the CPU and the kernel side (see [architecture.md](architecture.md)).

## What we know

| Fact | Source | Status |
| --- | --- | --- |
| The PS5 GPU (Oberon) is gfx1013 | Mesa and kernel patches from the PS5 Linux port; a PS5 radv patch set says "GFX1013" | Fairly solid |
| gfx1013 is the GFX10.1 (RDNA1) feature set plus BVH ray-tracing and MSAA-load instructions, without GFX10.3's encodings and dot instructions | LLVM `GCNFeatures.td` | Solid for LLVM's view of it |
| Games build PM4 command buffers in user mode and submit them through `libSceAgc` / `libSceAgcDriver` (for example `sceAgcDriverSubmitDcb`) | The PS5 radv patch set calls these | Observed in third-party code; the semantics are unknown |
| Games carry precompiled GFX10 shader machine code | Assumed by every PS5 emulator | Unverified |
| There is no public AGC documentation | Search | True as of 2026-11-19 |

So the useful public base is AMD's RDNA1 ISA guide plus the RDNA2 guide for the extras, LLVM's AMDGPU disassembler and Mesa (radv, ACO) for how PM4 and the ISA are used. That is more than we had for the R300 in 86slopper on the ISA side, and far less on the driver side: no open driver produces AGC calls, so the HLE layer for `libSceAgc` is built from observed behaviour.

## Plan

The same shape as 86slopper's `radeon` 3D engine (its `docs/gpu.md`): interpret the command stream in Rust, translate shaders to MSL text, draw with Metal directly. No Vulkan layer in between (MoltenVK and KosmicKrisp both add a translation step, and SharpEmu-mac shows a direct Metal backend works for PS5 workloads).

- **Command processor:** a PM4 type-3 interpreter on its own thread, fed by our `sceAgcDriverSubmitDcb`. Register writes build up state; draws and dispatches turn into Metal encoder calls.
- **Memory:** guest memory is one host allocation, so GPU-visible ranges can be wrapped as `MTLBuffer`s with `newBufferWithBytesNoCopy`, as 86slopper does for VRAM. Tiled surfaces and formats Metal cannot read in place need copies with dirty tracking. See [memory.md](memory.md).
- **Shaders:** decode GFX10 machine code into our own IR, then emit MSL. This gets its own doc when the design starts. SPIR-V plus SPIRV-Cross is the obvious alternative; SPIRV-Cross is Apache-2.0, which our GPL-2.0-or-later license cannot absorb without moving to GPLv3 (see [references.md](references.md)). `naga` (MIT OR Apache-2.0) is a Rust option for a SPIR-V route, if one is ever wanted.
- **Geometry:** RDNA's NGG primitive shaders have no direct Metal equivalent. SharpEmu-mac maps them to Metal mesh pipelines. Unknown how far that goes.

## Milestones that need almost no GPU

1. **Video out without AGC** (done for ScreenTester5, see below). `libSceVideoOut`: register buffers, flip, vblank events. A program that fills its buffer on the CPU and flips shows pixels with no GPU work at all. This is also how the open-source PS5 SDL2 port draws.
2. **Accept and drop command buffers.** `sceAgcDriverSubmitDcb` parses the PM4 stream, logs it and signals completion. Many programs get far with nothing drawn.
3. **Clears.** The first real Metal work: a clear or a fill to a render target that is then flipped.
4. **2D:** textured quads, blending. Then **3D**.

## Video out (implemented)

`libSceVideoOut` as ScreenTester5 uses it (its `src/display_native.cpp`, GPL-3.0, read for the call sequence; see [references.md](references.md)). None of it is verified on a PS5:

- `sceVideoOutOpen(0xff, 0, 0, NULL)` opens the one port; we return handle 1.
- `sceVideoOutSetBufferAttribute2(attr, format, tiling, width, height, option, dcc_control, dcc_clear)` fills an 80-byte block the guest treats as opaque, so ours holds our own layout. ScreenTester5 passes format `0x8000000022000000` (RGBA8, sRGB) and tiling mode 0.
- `sceVideoOutRegisterBuffers2(handle, set, first, buffers, count, attr, category, option)`, where `buffers` is an array of `{ data, metadata, reserved[2] }` (32 bytes each).
- `sceVideoOutSubmitFlip(handle, index, mode, arg)` puts the buffer on screen at once (index -1 blanks it) and fires flip events (`EVFILT_DISPLAY`, ident 0, `data` = `arg`). `sceVideoOutWaitVblank` sleeps to the next 1/60 s boundary.

**Frame layout for tiling mode 0**, as ScreenTester5 writes it: 32-bit RGBA pixels in 128x128-pixel tiles of 64 KiB, tiles row-major (`ceil(width / 128)` per row). Inside a tile, a pixel's byte offset is the XOR of these bits:

| Coordinate bit | Offset bits |
| --- | --- |
| x0, x1 | 2, 3 |
| x2, x3, x4 | 7, 8, 9 |
| x4 | 13 |
| x5 | 11 |
| x6 | 10, 15 |
| y0, y1, y2 | 4, 5, 6 |
| y3 | 8, 12 |
| y4, y5, y6 | 9, 10, 11 |
| y6 | 14 |

This is probably one of RDNA2's 64 KiB swizzle modes; which one is unconfirmed. We treat any other tiling mode as linear (unverified). `Kernel::frame` reads the front buffer from guest memory and untiles it for the host.

Status, 2026-11-19: ScreenTester5 shows its menu, and a pad press changes the pattern (checked with a frame dump).

## AGC (in progress)

`libSceAgc` and `libSceAgcDriver` are implemented in HLE (`crates/hle/src/agc/`), ported from SharpEmu (GPL-2.0-or-later) and checked against console measurements published by prosper (`oracle.rs`; the measurements stay in `references/`). There is no official AGC documentation; see [references.md](references.md) for every source and [legal.md](legal.md) for why documentation others wrote from decompilation is usable as a specification.

- Init, register defaults (versions 10 to 13), `CreateShader`, driver resource registration.
- Command-buffer builders append PM4 at the descriptor's upward cursor (`cmdbuf.rs`), calling the title's buffer-full callback when needed. One family per file.
- Submissions go to a GPU thread that runs them through `crates/gpu`'s command processor in order. It carries out what the CPU can observe (memory writes, end-of-pipe labels and their interrupts to the GPU event queues, waits, indirect buffers, flips), records register writes and counts draws and dispatches. Nothing is drawn yet; drawing will be Metal only, with no rendering on the CPU (vic, 2026-11-19).

Register state reaches the processor mostly through register tables (`SET_CONTEXT_REG_INDIRECT` 0x9F, `SET_SH_REG_INDIRECT` 0x63, `SET_UCONFIG_REG_INDIRECT` 0x64: a guest address and a count of (offset, value) pairs, as SharpEmu-mac reads them). Balatro sets no context register any other way.

Known difference from the console: `sceAgcDcbResetQueue` writes a 2-dword marker where the console writes a longer, mode-dependent sequence (tags in the internal-data register `0x342`).

## Metal renderer (in progress)

`crates/gpu/src/metal.rs`, structured after SharpEmu-mac's Metal backend (GPL-2.0-or-later; see its `docs/macos/metal-backend.md`):

- **Guest memory** is wrapped in place as no-copy `MTLBuffer` windows of up to 64 MiB (`newBufferWithBytesNoCopy` over our host reservation), so vertex, index and storage bindings copy nothing. Windows are dropped when the guest's mappings change, since a remapped range gets new host pages.
- **Render targets** are Metal textures keyed by guest address, filled from guest memory on first use. A flip of a buffer the GPU rendered shows that texture directly in the window (`gpu::metal::frame`); nothing is read back. Draws to the same targets share one render pass.
- **Textures** are detiled from guest memory on the GPU by generated compute kernels (`tiling.rs` gives the address equations as Rust and as MSL), refreshed once per frame. A texture at a render target's address is that render target's texture.
- **Sync** is simple for now: the processor finishes all recorded Metal work at end-of-pipe releases and flips, the points where the hardware waits for earlier work. `WRITE_DATA` and waits run at once, as the command processor does them on hardware.
- **Shaders** go through `shader::translate`, which the GFX10 translator (`crates/gfx10`, in progress) will provide. Until then draws are skipped and reported.

Swizzle modes: linear, `SW_64KB_S` (0x09) and the PS5's render-target mode 0x1B, whose equation SharpEmu-mac uses for textures and which equals the frame layout ScreenTester5 writes. A test checks one against the other, and a Metal test checks the GPU detiler against the CPU layout.

What Balatro draws (2026-11-20): an NGG vertex program in the ES slot (`VGT_SHADER_STAGES_EN` 0x2002000) for every draw; clears as 3-vertex rectangle lists (primitive type 7) with a pixel shader writing a colour from user SGPRs; then its scene into a 1920x1080 BGRA8 canvas in mode 0x1B, which a final textured draw (T# in s0..s7, S# in s8..s11) puts on the display buffer.

## Open questions

- Which AGC functions a GameMaker or LÖVE title calls, and how much of their meaning we can infer from the PM4 they leave behind.
- How games compile shaders: precompiled in the package (likely), or built at run time.
- How tiling (RDNA swizzle modes) maps onto what Metal can sample or render to without copies.
