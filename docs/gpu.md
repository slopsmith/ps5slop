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

- **Guest memory** is wrapped in place as no-copy `MTLBuffer` windows of up to 64 MiB (`newBufferWithBytesNoCopy` over our host reservation), so vertex, index and storage bindings copy nothing. Windows are dropped when the host pages under them are replaced (see [memory.md](memory.md)), and when a new window covers them. Windows use Metal's default hazard tracking, because compute programs now write guest memory that later work reads. Two windows over the same range are still separate buffers to Metal, so ordering across them is not guaranteed (not seen to matter yet).
- **Render targets** are Metal textures keyed by guest address, filled from guest memory on first use. A flip of a buffer the GPU rendered shows that texture directly in the window (`gpu::metal::frame`); nothing is read back. Draws to the same targets share one render pass.
- **Textures** are detiled from guest memory on the GPU by generated compute kernels (`tiling.rs` gives the address equations as Rust and as MSL), refreshed once per frame. A texture at a render target's address is that render target's texture. 2D arrays and cube maps are detiled slice by slice, one slice after another at the level size rounded up to 64 KiB (an assumption, fine for the mip-less arrays Minecraft uses); 3D textures are not supported yet.
- **Storage images**: an image a program writes (`image_store`) is kept in the render-target map under its address, filled from guest memory when first made, so later draws that sample that address see what the program wrote. 2D-array and 3D images, and 2D writes through a 2D-array or cube descriptor, are kept as Metal 2D-array or 3D textures by address. 2D arrays are loaded from guest memory when made (3D images start zero-filled), and a checksum of that memory is kept: when it changes, the CPU has written there and the array is loaded again (checked before every write to it and once a frame before reads; ranges over 32 MiB are not checked). Minecraft's UI needs both: its glyph atlas (a 1024x1024 R8 texture) is created with the glyphs rasterised so far as CPU-written contents, then a copy program writes each new glyph into it through a 2D-array descriptor; when the atlas is recreated, the CPU uploads it again. A write or read sees the slices from `BASE_ARRAY` (T# word 4, bits 16 to 28, as SharpEmu-mac's `TextureDescriptorWords` reads it) on, through a texture view; an array grows (copying its slices) when a write names a slice past its end. Minecraft builds its panorama cube map this way: one copy dispatch per face, each through a one-slice view, then samples the six slices as a cube.
- **Depth as a texture**: a T# at the address of a depth target the GPU rendered samples that Metal depth texture directly (bound as `texture2d<float>`; its channel selects are ignored, since depth textures cannot be viewed). Depth targets have array slices: `DB_DEPTH_VIEW`'s slice start picks the slice a pass draws to, and the texture becomes a Metal 2D array big enough for the highest slice start or slice max seen (at most 16; growing copies the old slices). A 2D-array T# at its address samples the array (from its `BASE_ARRAY`), a 2D one a single slice. This is for Minecraft's shadow cascades: a 2048x2048 depth target drawn about 100 times a frame, read by its lighting programs as a four-slice array (2026-11-20; whether each pass's slice is right is not checked yet). A T# in a depth swizzle mode (`_Z`, `_Z_T`, `_Z_X`) that no depth target backs reads as zeros; we have no equation for those layouts. A 2D image bound where a cube is read shows its one slice on every face, and one bound where a 3D texture is read becomes a one-slice 3D copy.
- **Integer formats**: textures whose format is UINT or SINT are declared `<uint>`/`<int>` in the MSL; the renderer reads the bound descriptors per draw and translates a program again for each combination it meets (`Variant`). An export to an integer target is made the matching 16-bit integer export format, so the CB gets the bits the program exported (the title's clear program exports FP16 to an RGBA16Uint target, which hardware stores as raw bits and Metal rejects).
- **Descriptors from memory**: the translator describes each descriptor as user SGPRs or a chain of loads from pointers in them (`Source::Memory`); `resolve()` follows the chain on the CPU per draw or dispatch.
- **Sync** is simple for now: the processor finishes all recorded Metal work at end-of-pipe releases and flips, the points where the hardware waits for earlier work. `WRITE_DATA` and waits run at once, as the command processor does them on hardware.
- **Shaders** are translated by `crates/gfx10` (see [shaders.md](shaders.md)) and compiled with `MTLDevice`. Programs that load from memory get one no-copy window over the mapped guest range their user data points into (as a 64-bit pointer, or as a V# base whose high word is masked to 16 bits), at `buffer(29)`, with its guest address at `buffer(28)`. Compute programs are dispatched as `COMPUTE_NUM_THREAD_X/Y/Z` threadgroups, refused when the pipeline cannot run that many threads in a group. Rectangle lists (primitive type 7) draw six vertices per rectangle through a wrapper around the translated vertex body; the fourth corner is v1 + v2 - v0.
- **Flips** wait for the next 60 Hz refresh, as the hardware's flip queue does, and copy the shown buffer into a presentation texture, so the window never shows a buffer the title is already drawing the next frame into.
- **Textures with mip chains**: in 64 KiB-block modes, smaller levels come first (a trailing mip-tail block, then levels from the smallest up), so level 0 starts after them (`Layout::mip0_offset`, from SharpEmu-mac's `TileGeometry`). Only level 0 is uploaded so far.
- **Debugging**: `PS5SLOP_GPU_STATS=1` prints counters each second, `PS5SLOP_GPU_DUMP=n` the register state of the first n draws, `PS5SLOP_GPU_TRACE=n` (or `first-last`, or the control socket's `gputrace N` for the next N frames) every draw and dispatch of those frames with its programs, vertex user data, viewport and scissor, targets, depth target, descriptors and result, `PS5SLOP_MSL_DUMP=1` every translated program, `--dump-frame` with `PS5SLOP_DUMP_SERIES=n` a series of frames half a second apart, `PS5SLOP_DUMP_TARGETS=1` every render target, depth target and sampled texture (formats other than RGBA8 converted on the GPU, clamped to [0, 1]), `PS5SLOP_MSL_ERRORS=1` the source of programs Metal rejects; failed command buffers are reported, `PS5SLOP_SHADER_DUMP=dir` every translated program with its register state, for `cargo run -p gfx10 --example captures -- dir` (see [shaders.md](shaders.md)).

Status, 2026-11-20: Balatro's intro and title screen render correctly through Metal (logo, cards, the swirl background and the CRT pass), checked against frame dumps. About 12 of 10,000 draws a second are skipped: a colour-masked stencil pass whose leftover texture descriptor is a buffer. Audio plays alongside.

Status, 2026-11-20, Minecraft (with the sign-in changes): all 49 programs it runs reach Metal, its 11 compute programs dispatch (about 4,000 dispatches in 45 s, none skipped), and its main menu draws its UI: the logo, the "Get started" and "More options" buttons, the button hints and the version (checked with frame dumps). Before, 7,500 dispatches and 1,900 draws were skipped in 45 s; now about 1,500 draws are. Wrong or missing:

- **Striped textures.** The logo and the font atlas (RGBA8, `SW_64KB_S`) show every other 16-texel column black, so glyphs miss strokes. The guest memory itself is zero in every 2 KiB chunk whose offset has bit 11 set (x bit 4 in this mode), measured by reading the textures back from guest memory, while Minecraft's RGBA16F textures in the same mode are complete and Balatro's RGBA8 ones in this mode are right. The game writes these textures on the CPU (no GPU copy fills them, and it imports no tiling library), and JIT verify mode reports no difference, so the cause is on the CPU side or in what the title is told about the GPU. Unknown.

Status, 2026-11-20, Minecraft's panorama: the title-screen background draws (a cherry-grove world under a blue sky, turning slowly; checked with frame series two seconds apart). Every draw and dispatch reaches Metal: 0 skipped in 85 s, against about 5,100 skipped in 75 s before. The panorama is a 1024x1024 RGBA16F cube map the title builds from six CPU-written images with a copy kernel, drawn as a sky box into the HDR target of its deferred pipeline, then run through its post-processing (a downsample chain and a final combine; what each pass computes is not checked). The pipeline's other inputs (a G-buffer of RGBA16Uint, RGBA8 and RGBA16F targets, the depth buffer sampled as R32Float, a 2D-array light grid and a 3D fog volume written by compute programs) all run; the menu has no world geometry, so they hold their clear values. The faces show the same 16-texel column stripes as the logo, since the title writes them on the CPU like the logo (see above).

Status, 2026-11-20, Minecraft's UI text and player model, checked with screenshots of the main menu, the Play screen, Create from template, Create New World and the "Welcome to Minecraft!" sign-in popup (fresh `--data` folders):

- **Missing glyphs** ('a', 'c' and 'k' blank on the Play screen; other letters after the sign-in popup): the glyph atlas was a zero-filled Metal array, so the glyphs the CPU had put in it ("B", "a", "c", "k" of a "Back" label, seen in a target dump) were lost, while glyphs the copy program added later showed. Fixed by loading 2D arrays from guest memory (see storage images above). The popup case was seen fixed once (full title and button text) after the checksum reload was added; the theory that the atlas is recreated there is inferred from an earlier run where the set of missing letters changed after the popup, not traced draw by draw.
- **Doll without legs**: the player model's vertex program reads a bone index per vertex with `buffer_load_format_x` in `BUF_FMT_16_UINT` (11, read from the title's vertex fetch table through the control socket), which the translator did not decode and returned as 0. Each draw holds several body parts; the legs share a draw with the torso, so with bone 0 they sat inside it. Fixed in the translator (see [shaders.md](shaders.md)). The legs draw on the main menu now.
- After choosing "Sign in now" in the popup, the whole UI (boxes and button hints too) disappeared and stayed gone; not investigated. Those are colour-target draws, which the atlas change does not touch.

An earlier note here said these passes read 780x438 regions of larger targets: that was the hexadecimal debug print of 1920x1080 (0x780 by 0x438). Every target in the pipeline is 1920x1080 or a power-of-two fraction of it, and the depth image's swizzle mode is 0x18 (`SW_64KB_Z_X`), not 0x12.

Swizzle modes: linear, `SW_64KB_S` (0x09) and the PS5's render-target mode 0x1B (the depth modes are not supported: depth is sampled from Metal depth textures instead, see above), whose equation SharpEmu-mac uses for textures and which equals the frame layout ScreenTester5 writes. A test checks one against the other, and a Metal test checks the GPU detiler against the CPU layout.

What Balatro draws (2026-11-20): an NGG vertex program in the ES slot (`VGT_SHADER_STAGES_EN` 0x2002000) for every draw; clears as 3-vertex rectangle lists (primitive type 7) with a pixel shader writing a colour from user SGPRs; then its scene into a 1920x1080 BGRA8 canvas in mode 0x1B, which a final textured draw (T# in s0..s7, S# in s8..s11) puts on the display buffer.

## Renderer performance (2026-11-20)

Measured on Minecraft's main menu with the control socket's `profile` (which now reports the GPU thread: encoding, waiting for Metal, GPU busy time from each command buffer's `GPUStartTime`/`GPUEndTime`, commit-to-scheduled and done-to-woken latencies), host `sample`, and a Metal System Trace with labelled encoders (`PS5SLOP_GPU_LABELS=1`; see [debugging.md](debugging.md)). The menu went from about 10 to about 16 flips a second. In order:

- **Deferred releases.** Every end-of-pipe release (a label write or an interrupt) committed and waited for the command buffer: about 450 round trips a second, 45 per frame, each about 1.4 ms. Releases now queue their writes and interrupts; the queue is drained, after one wait, before a `WAIT_REG_MEM`, a flip, and at the end of each submission. A `WAIT_REG_MEM` that a queued release will satisfy passes at once, since GPU work runs in order (about 57 per frame on the menu). `PS5SLOP_GPU_SYNC_RELEASES=1` restores the old behaviour.
- **Uploads in their own command buffer.** Each texture detile ended the render pass to start a compute encoder, so passes with three 1080p colour targets and a depth target were stored and reloaded several times a frame. Detiles now go into an upload command buffer committed just before the main one (`PS5SLOP_GPU_INLINE_UPLOADS=1` restores the old way).
- **Guest memory windows.** Any change to the guest's mappings (Minecraft maps and unmaps direct memory several times a second) flushed and waited for the GPU and dropped every no-copy window. Now only windows over ranges whose host pages were replaced are dropped (`Memory::replaced_since`), without waiting.

- **Memory writes from the command stream are queued too.** With releases deferred, Minecraft's menus broke (stray triangles, missing icons and letters; 3 of 3 runs broken, 3 of 3 correct with `PS5SLOP_GPU_SYNC_RELEASES=1`). Cause: `WRITE_DATA` to memory still ran when parsed, long before Metal ran the draws before it, and the UI uses such writes as labels (a NOP payload in the command buffer, and 64-bit slots) that tell its CPU earlier work is done, so it reused vertex data too early. `WRITE_DATA` now joins the release queue in order; a wait on a label a queued write sets passes at once. Waiting for the GPU before each `WRITE_DATA` also fixed it but cost 37% of menu frames.
- **CPU reads of what the GPU wrote.** Gameface copies data on the GPU with a small copy program (a buffer load and store per thread, about 13 dispatches a frame), including index data. Index buffers at addresses that are not multiples of 4 are copied on the CPU for Metal, and that copy read the indices before the copy program had run (stray triangles in 2 of 3 runs). The renderer now remembers that encoded work wrote guest memory (`writes_memory`) and waits for it before such a copy; 3 of 3 correct. Waiting before every descriptor or texture read as well cost 20% of frames and was not needed. The copies go into a shared scratch buffer instead of a new Metal buffer each.
- **One window for the GPU area.** Programs that read memory get one window from the start of the area where direct memory and `mmap` land (`Host::gpu_area`, 16 GiB from `0x10_0000_0000`) up to what their pointers reach, in 1 GiB steps (Metal allows 28 GiB buffers on this Mac). `Memory::back` gives every page in it host memory first, so holes read as zeros. Before, the window covered only the mapped extent behind the user data, and Minecraft's GPU memory is in about 135 extents, so a descriptor pointing into another extent read outside the buffer. This did not fix the menu by itself.
- **In the world (2026-11-20).** About 1,900 draws a frame once chunks load. The frame rate fell from 6 to 1 per second over 100 s. An Instruments Time Profiler trace (see [debugging.md](debugging.md)) showed 90% of the GPU thread in `Memory::mapped_extent`, which walked thousands of adjacent regions for each user-data word of each draw; merged extents are now cached per memory generation and found by binary search: 10 to 16 frames a second. Next in that trace: a text search through both programs' Metal source on every draw for pixel inputs the vertex program lacks (now cached per program pair), the 2D-array checksums (now a sum and an XOR that vectorise) and the per-draw index copy buffers.

What is left: the Metal System Trace shows the GPU executing encoders for only about 17% of the time (about 17 ms of work per frame), while each command buffer spans about 3 times that. The difference is spent before the GPU starts work, very likely making the many no-copy windows resident, which also have to be recreated whenever guest mappings change. A residency set that keeps the windows resident (`PS5SLOP_GPU_RESIDENCY_SET=1`) cut GPU time by two thirds but moved the cost into synchronous IOKit calls on the GPU thread, and was slower overall, so it is off. Images read from guest memory are still detiled again every frame.

## Open questions

- Which AGC functions a GameMaker or LÖVE title calls, and how much of their meaning we can infer from the PM4 they leave behind.
- How games compile shaders: precompiled in the package (likely), or built at run time.
- How tiling (RDNA swizzle modes) maps onto what Metal can sample or render to without copies.

Status, 2026-11-20, Minecraft in a world (after the SDWA and depth-slice fixes): every draw and dispatch of a traced frame reaches Metal. The presented frame shows the sky with clouds, the block-selection outline and the HUD (hotbar, hearts, hunger, button hints), but no terrain. Target dumps show the terrain and entities in the G-buffer, in the HDR scene target and in the tonemapped output, so they are lost in a later pass; the terrain also has wrong colours there (a blue, white and black checker), so its texture atlas is likely read wrongly. Two pixel programs are not translated yet (`v_writelane_b32`, and a descriptor whose source the translator cannot trace). After a `targets` dump the run presented only black frames; whether the dump causes this is not checked.

