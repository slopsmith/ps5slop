# GPU

How a PS5 title's GPU work reaches Metal: the AGC submission side, the PM4 command processor and the Metal renderer. Shader translation has its own doc, [shaders.md](shaders.md). Guest memory itself is in [memory.md](memory.md); the debugging tools (problem report, traces, captures, `ps5slop-replay`) are in [debugging.md](debugging.md); switches are in [environment.md](environment.md).

There is no PS5 to measure on. Everything below that describes PS5 behaviour rests on public documentation, other emulators or what titles visibly expect, and is unverified on hardware unless it says otherwise. The usual check is that a title's picture matches a capture of the console (screenshots, videos).

## Public facts

Interface facts only, kept apart from how we implement them.

| Fact | Source | Status |
| --- | --- | --- |
| The PS5 GPU is gfx1013 | Mesa and kernel patches from the PS5 Linux port; a PS5 radv patch set | Fairly solid |
| gfx1013 is GFX10.1 (RDNA1) plus BVH ray-tracing and MSAA-load instructions, without GFX10.3's encodings and dot instructions | LLVM `GCNFeatures.td` | Solid for LLVM's view |
| Titles build PM4 command buffers in user mode and submit them through `libSceAgc` / `libSceAgcDriver` | The PS5 radv patch set; SharpEmu | Observed in third-party code |
| Titles ship precompiled GFX10 machine code | Every PS5 emulator assumes it; true of every title seen | Unverified |
| PM4 packet layouts and register fields | AMD's PM4 description as Mesa uses it (`sid.h`, the register files `gfx10.json` and `gfx103.json`), SharpEmu-mac's register tables | Public; PS5 deviations unknown |
| Bytes the real `libSceAgc` builders write | Console measurements published by prosper (not in the repo; `scripts/fetch-references` puts them in `references/`) | Measured on a PS5 by others |
| Surface swizzle equations | SharpEmu-mac (`TileGeometry`); AMD addrlib in Mesa for 3D `SW_4KB_S` | Partly checked against title data |

There is no public AGC documentation, and no open driver produces AGC calls, so the HLE side is built from SharpEmu's port and from what titles do. Sources and licences are in [references.md](references.md); why documentation others wrote from decompilation is usable as a spec is in [legal.md](legal.md).

## Where the code lives

- `crates/hle/src/agc/`: `libSceAgc` and `libSceAgcDriver`. Command-buffer builders (ported from SharpEmu, checked against the console measurements by `oracle.rs`), `sceAgcCreateShader`, register defaults, and `submit.rs`, which owns the GPU threads.
- `crates/hle/src/videoout.rs`, `crates/hle/src/gnm.rs`: video out and the PS4 driver.
- `crates/gpu/src/pm4/`: the command processor (`Processor`), one per queue.
- `crates/gpu/src/metal/`: the renderer, one module per topic (memory windows, targets, images, detiling, HTILE and colour fast clears, programs, draws, dispatches, readback, inspection tools, save states).
- `crates/gpu/src/`: `regs.rs` (register offsets and decoded draw state), `tiling.rs` (swizzle equations as Rust and MSL), `compile.rs` (compile pool), `capture.rs` and `replay.rs` (frame capture and offline replay, CLI in `src/bin/ps5slop-replay.rs`).
- `crates/gfx10`: the shader decoder and translator ([shaders.md](shaders.md)).

## Data flow

1. A title builds PM4 in its own command buffers, with our `sceAgc*` builders or its own code, then submits.
2. `submit.rs` hands each submission to the GPU thread of its queue: one thread for the graphics queue (`sceAgcDriverSubmitDcb`, `SubmitMultiDcbs`) and one per compute owner (the first argument of `SubmitAcb`, `SubmitMultiAcbs`).
3. Each thread runs its buffers through its own `Processor`, in order. Packets the CPU can observe (memory writes, labels, interrupts, waits, flips) take effect; register writes go into register banks; draws and dispatches go to the backend.
4. All processors share one Metal renderer (`SharedBackend`, behind a mutex). Work is encoded in the order the calls arrive; a queue waiting on a label does not hold the mutex.
5. The renderer translates programs on the GPU thread, compiles them on the compile pool, binds guest memory through no-copy windows, and encodes Metal command buffers.
6. A flip shows the rendered texture in the window directly. Nothing is read back into guest memory.

**Why a thread per queue:** queues wait on labels each other set. With one thread in submission order, a graphics wait on a label that a compute buffer queued behind it would set could only time out (Astro Bot). Separate renderers per queue are no option either: queues pass render targets, images and the GDS between them.

**Submission throttle:** a graphics submission returns only once every submission up to the last frame before it has run and its Metal work is done, unless the queue's thread is polling memory the CPU may have to change (a wait, REWIND, semaphore or compare-swap) or 5 s pass. Frames are found from the flip packets our own builders wrote. Reason: titles reuse a frame's command and vertex memory two frames later without waiting for the GPU, which is safe on a console that keeps up but not with a GPU thread that lags (Isaac drew room backdrops from the next frame's data). Throttling every submission instead of every frame costs too much.

## Command processor

- **Register banks start at zero**, not at the hardware's reset values, which are not known reliably. Code that reads a register a title may never write must treat zero as "unset" (the scissor code does). Seeding the banks from AGC's defaults tables would be the general fix, but which values the hardware really starts with is unverified.
- **Register tables** (`SET_*_REG_INDIRECT`: a guest address and a count of offset/value pairs) carry most state. Some titles set no context register any other way.
- **Releases and memory writes are queued.** End-of-pipe releases (labels, interrupts), `WRITE_DATA` and `DMA_DATA` fills of up to 64 bytes join one queue in command order. They take effect once Metal reports the work before them done; the GPU thread polls for that while idle. Anything that reads what they write (a wait the queued values do not satisfy, a flip, `COND_EXEC`, `COPY_DATA`, ...) drains the queue first. Reason: committing and waiting per release cost most of the GPU thread's time, and writing at parse time let titles reuse memory that queued draws had not read yet (Minecraft's UI uses `WRITE_DATA` as "done" labels). Labels now arrive when the GPU work is done rather than when it was encoded, which is closer to the console.
- **A wait that a queued release will satisfy passes at once**, since GPU work runs in order. A wait on memory nothing queued writes polls memory directly, as the hardware's command processor does.
- **Waits give up after 5 s** (`WAIT_REG_MEM`, `ATOMIC_MEM` loops, `MEM_SEMAPHORE`). Time while any processor waits for a shader compile does not count: compiles are our cost, and the queue that would write the label may be the one compiling. A timeout is a bug signal, not normal operation.
- **Compute queues' `WRITE_DATA` into the GPU area lands in GPU order**, as a blit into the GPU-area window, when the queue never draws and nothing unapplied is queued before it. Reason: a culling queue resets counters and tables with `WRITE_DATA` between thousands of dispatches in one submission, and with queued writes the dispatches read stale memory (in GTA V one then stored across the game's code). The rule is narrow because Metal orders work only within one buffer object: dispatches share the area window, but draws read through other windows, so a blit for a graphics queue could land before an earlier draw read the old value.
- **`DMA_DATA`** larger than a small fill runs on the CPU after waiting for all encoded GPU work, which may write its source. Metal blits were tried and dropped: a label written by a blit appears only when the GPU reaches it, and queues timed out on each other.
- **Indirect draws** read their arguments on the CPU after a GPU wait. An instance count of 0 draws nothing. Multi-draw indirect packets are only counted.
- **Indirect dispatches** read their sizes on the GPU: a one-thread program reads them through the GPU-area window (so Metal orders it after the programs that wrote them), turns thread counts into groups, zeroes sizes over the group limit, and the dispatch runs with Metal's indirect dispatch. The CPU path remains for sizes outside the GPU area, some lockstep programs and a few debugging switches. Assumed: the programs that write the sizes do so through the area window.
- **`USE_THREAD_DIMENSIONS`** (dispatch initiator bit 5): dimensions count threads, not groups. A partial last group runs partly, as radv uses the bit; lockstep programs ([shaders.md](shaders.md)) run whole groups with grid bounds instead.
- **Dispatches over 2^24 groups are skipped.** Metal would run them for minutes, and only garbage sizes get there.
- **`REWIND`** waits, with no time limit, until the CPU sets its valid bit, then fetches the rest of the buffer again. GTA V chains each graphics buffer to the next by patching an `INDIRECT_BUFFER` behind a REWIND and then setting it valid, so the stream continues into work submitted later.
- **Chained `INDIRECT_BUFFER`** (bit 20 of the size dword) continues at its target at the same depth instead of nesting; packets after it are not run. Nesting stops at 8 levels.
- **`GE_INDX_OFFSET`** is added to every vertex index, for auto draws too.
- **GDS** is one 64 KiB buffer shared by all queues; the processor reaches it after a GPU wait. Ordered append (`ds_ordered_count`) and GWS are not modelled.
- `ATOMIC_MEM` and `MEM_SEMAPHORE` use host atomics on guest memory, with SharpEmu-mac's operation numbers.

Known difference from the console: `sceAgcDcbResetQueue` writes a short marker where the console writes a longer, mode-dependent sequence.

## Metal renderer

### Command buffers and flips

Work goes into one main command buffer. Texture uploads (detiles from guest memory) go into an upload command buffer committed just before it, so they do not split render passes; once the main buffer holds work that writes guest memory, uploads go inline after that work instead. Draws to the same targets share a render pass.

A submission that does not end a frame commits without waiting. A frame ends with a wait for Metal, which the throttle relies on. Flips wait for the next 60 Hz refresh and copy the shown texture into a presentation texture, so the window never shows a buffer the title is already drawing the next frame into.

Committing early, every N draws, lets the GPU start sooner and measured faster, but it is off by default because it changes some pixels: encoding reads guest memory (descriptors, tables) on the CPU while queued releases and writes land when Metal finishes, so commit timing changes what some draws read. Which picture is right is unknown without a PS5 reference. The timing dependence exists with or without early commits.

### Guest memory windows

Guest memory is one host allocation, so the renderer wraps guest ranges in place as no-copy `MTLBuffer`s (windows) and copies nothing for vertex, index, constant or storage access.

- **Ordinary windows** are about 64 MiB, inside one host-backed extent.
- **The GPU-area windows** cover the area where direct memory and `mmap` land (from `0x10_0000_0000`, see [memory.md](memory.md)). Every page in them gets host memory first, so holes read as zeros. The low window starts at the area's start and grows in 1 GiB steps to the furthest address a program reaches near it: every area mapping its user data points into, and those that V#s in tables behind the user data point into. It **never shrinks** when it is made again after a mapping change. Reason: programs reach memory through descriptors loaded from memory, so user data does not bound what they read, and a window rebuilt only as large as the current program's need lost whole frames.
- **Data 2 GiB or more past the low window gets a high window** (also in 1 GiB steps), so the unmapped space between is never given memory. Metal gives every page of a window memory: one window over a whole area with a large hole in it (GTA V maps at both ends of the area) failed command buffers with "Insufficient Memory".
- **Each program gets up to three windows**, chosen by `guest_window` from its user data: the region holding the program or the largest region the user data points into, the GPU area, and the largest other region. A program that needs the high area window gets it in the third slot instead of the other region. Candidates include the bases of V#-like quads in the first 64 dwords behind aligned pointers in the user data, since titles reach vertex pools outside the GPU area through such tables. Aligned pointers rank first, then V# table bases, then unaligned pointers, so a V#'s stride and record count do not pass for a pointer. Memory no window reaches reads as 0 and is not written.
- The translated program maps each address into one of its windows (`Win`); the second and third are reached through the difference of their GPU addresses from the first. This relies on Apple GPUs not bounds-checking device pointers (Metal's shader validation would flag it). Since programs reach those windows only by address, they are also declared with `useResource`: Metal does not make a bound but undeclared buffer resident.
- **Stores and atomics skip the image of a loaded module** that a window covers. Some titles keep GPU programs in their executable, so a window covers its code and globals, and a store through a wrong descriptor once overwrote game code. Assumed: the console's GPU never writes module images.
- **Windows are dropped** when the host pages under them are replaced. A command buffer keeps every window it binds alive, so once 1 GiB of dropped windows add up the renderer commits, and waits for the GPU if Metal's allocations would pass half its working set. Without this, titles that remap often (Minecraft) failed command buffers with "Insufficient Memory".
- **Ordering across windows is not guaranteed.** Metal orders work only within one buffer object, and several windows can cover the same memory. An opt-in fence orders every encoder that reads guest memory after the last one that wrote it; it is off by default because it costs a lot and no bug is proven to need it.
- A residency set for the windows was tried and was slower overall; it is off.

### What encoded work may write

The translator reports the V#s of every buffer store and atomic. The renderer turns them into ranges that encoded, not yet waited-for work may write (`Written`), or "anything" when a program's stores are not known. Before the CPU reads guest memory that GPU work may have written (HTILE, CMASK and DCC words, unaligned indices it copies for Metal, image digests), it waits only if the read overlaps those ranges. Waiting before every such read was a large share of frame time.

### Render targets and depth

- **Render targets, depth targets and program-written images are Metal textures keyed by guest address.** A target is filled from guest memory when first made. What the GPU renders stays in Metal and never reaches guest memory: a CPU read of target memory sees stale bytes, and a CPU write to it is not seen (only a buffer store that spans a whole cached 2D target drops it, so its next use reads memory). This is deliberate: all drawing is on Metal, and a read-back per frame would cost a full copy for nothing.
- Targets keep the mip chain their view names; a draw to level n renders into a one-level view. A draw to one slice of an array target keeps the array, and a target that must grow copies its slices (at most 16).
- **One address, several meanings.** Titles reuse memory for other surfaces within a frame, so the renderer remembers which kind of write used an address last:
  - a sampled image reads the depth texture only if no colour target or storage image wrote the address since the depth target did;
  - a colour target drawn at an address after the last 2D-array or 3D write hides the array (samples read the target, 2D stores go to it) until an array write uses it again;
  - a depth target takes over 32-bit and 16-bit images a program stored at its address, in every slice they share;
  - an 8-bit integer image at a depth target's `DB_STENCIL_WRITE_BASE` reads an `X32_Stencil8` view of that depth texture;
  - a texture in a depth swizzle mode that no depth target backs reads as zeros (2D, array or cube): we have no equation for depth layouts.
- **Depth sampled as a cube** grows the depth texture to six slices first; faces never drawn hold whatever the new texture held.
- **Depth range**: hardware keeps float depths outside `[0, 1]` when `PA_SC_VPORT_ZMIN/ZMAX` allow it, Metal clamps them. A draw whose possible depths leave `[0, 1]` has its range mapped onto it (`regs::depth_range`), which keeps the order depth tests see. Writing depth from every pixel program instead would cost early depth testing everywhere. Draws with different such ranges into one buffer, or such a buffer read as a texture, would see mapped values; no title seen does either.
- **The viewport scissor applies only with `VPORT_SCISSOR_ENABLE`** (`PA_SC_MODE_CNTL_0` bit 1), as on hardware. Empty scissor registers count as unset.
- **Component swap** (`CB_COLOR_INFO.COMP_SWAP`): four-channel ALT targets are made in Metal's swapped format (`BGRA8Unorm`, `BGR10A2Unorm`), and a descriptor that reads them in the unswapped format gets a view in that format, which gives back the hardware's byte order. Other swaps (STD_REV, ALT_REV, and every swap of one-, two- and three-channel formats, such as alpha-only targets) are made in the pixel program: each channel takes the export component the swap names, from Mesa's `ac_translate_colorswap` and the R6xx-era register documentation (four-channel ones checked on hardware). `CB_TARGET_MASK` is in export channel order, so the mask is permuted with the channels. A program-swizzled target that blends with alpha factors sees the wrong alpha channel (no title seen does that).
- **`SX_BLEND_OPT_CONTROL`** with both disable bits of an MRT set bypasses its blending (Mesa's `ac_formats.c`; measured: the hardware wrote the source unblended).
- **Blending is done in the pixel program**, which reads the target through `[[color(n)]]` (`Request::blends`): Metal's fixed-function blending rounds and treats NaNs differently, ignores the factors of min and max, and has no BOTH_SRC_ALPHA factors. The rules, measured on the Deck's 8-bit unorm and 16-bit float targets with FP16 exports: the CB blends in 32-bit float; a ZERO factor drops its term (NaN or infinity times it too); min and max apply both factors and ignore a NaN; BOTH_SRC_ALPHA and BOTH_INV_SRC_ALPHA (11, 12) are 0 wherever they appear; SRC_ALPHA_SATURATE ignores a NaN destination alpha; the blend constant keeps 12 significant bits, rounded toward zero (seen with 0.6 only, so whether it is a float or a fixed-point form is not known); a 16-bit float target gets the result rounded toward zero, every NaN as 0xFE00. An 8-bit unorm target is read as a float of 12 significant bits (rounded to nearest even) before the blend, and the result is rounded to nearest: the blend sweep (every destination value times 0.5 and times 1 - alpha over 16 alphas) matches this exactly, where plain floats round 64 of the 128 ties the other way. Whether other unorm widths and sRGB read the destination the same way is not measured; they blend it at full float precision. Integer targets and targets with `CB_COLOR_INFO.BLEND_BYPASS` do not blend; `BLEND_CLAMP` clamps the sources and the constant to the format's range first (measured on 8-bit unorm targets). Dual-source blending takes the MRT1 export as `SRC1`.
- **ROP3** (`CB_COLOR_CONTROL`) applies to unorm and integer targets of 8, 16 or 32 bits per channel that do not blend, only in `CB_NORMAL` mode: context registers start at 0 here (CLEAR_STATE's defaults are not modelled), and a ROP3 of 0 would clear every target.
- **Viewport sign**: Metal's viewport takes a negative height, so a positive `PA_CL_VPORT_YSCALE` draws upside down from a negative one, as on hardware. Face culling with a positive scale is not checked on hardware.
- **Points** are `PA_SU_POINT_SIZE` across (twice its 12.4 half height; a point size the vertex program exports is ignored). **Lines** are one pixel wide whatever `PA_SU_LINE_CNTL` says: Metal has no wide lines. **Rectangle lists** (primitive types 7 and 17) draw as parallelograms through their three vertices; the hardware draws three arbitrary vertices otherwise (screen-aligned), which titles do not depend on.
- **A target the program exports to keeps its attachment** even when its write mask is 0, so the following exports still reach the right MRT; but only before the last target the draw writes. A masked export after that writes nothing, and its target is often left bound from an earlier pass: depth-only shadow passes keep the previous pass's small target and pixel program, and attaching it clipped every cascade to its size.
- **Image stores through channel selects that swap red and blue** bind the 8-bit view of the other order: the hardware applies the selects to stores too, and Metal cannot write through a swizzled view. Other formats with such selects are not handled.
- **Draws without targets** whose pixel program stores to memory or images run with a stand-in R8 target, writes masked off, sized from the scissor. Other draws without targets are skipped.
- **Metadata passes.** Draws with `CB_COLOR_CONTROL.MODE` 2, 5 or 6 (fast-clear eliminate, FMASK and DCC decompress in Mesa's enum) only rewrite compression metadata on hardware and are skipped, but only when a register table wrote `CB_COLOR_CONTROL` since the previous draw. Reason: Astro Bot leaves mode 6 set for its following normal draws, so the console must reset it somewhere we do not see. A guess that fits every trace so far. Draws with `DB_RENDER_CONTROL` decompress flags and no colour target are skipped likewise.
- **Integer targets**: an export to a UINT or SINT target is made the matching 16-bit integer export format, so the target gets the bits the program exported (Metal rejects float outputs to integer attachments).
- **Sample mask**: a `[[sample_mask]]` output turned off Metal's stencil test on this Mac (measured, documented nowhere we know of). Targets are single-sampled, so an exported sample mask discards the fragment when bit 0 is clear instead.
- GFX10 stencil ops Metal lacks (set to ones, the bitwise family, adds other than 1) become keep; "replace with the op value" uses the reference value.

### Fast clears

Titles clear by writing compression metadata with a compute program and never touch the surface. The renderer reads the metadata and applies the clear itself. The encodings are radv's (MIT, from memory) and not checked on hardware:

- **Depth (HTILE**, enabled by `DB_Z_INFO` bit 29): a ZMask of 0 in the slice's first HTILE dword means cleared to `DB_DEPTH_CLEAR`. The renderer clears and writes ZMask 0xF back, as drawing would. HTILE slices are taken to be 4 bytes per 8x8 tile, rounded up to 64 KiB.
- **Colour**: with `FAST_CLEAR`, a CMASK dword whose low nibble is 0 means cleared to `CB_COLOR_CLEAR_WORD0/1`. With `DCC_ENABLE`, the DCC codes 0x00000000, 0x40404040, 0x80808080 and 0xC0C0C0C0 mean the four 0/1 colours, 0x20202020 the clear words, and 0x10101010 ("single") the first texel of each block as stored.
- Only the first dword is read, so a clear of part of a surface is not seen.
- A pending clear is applied before a draw to the target, before a program stores to it, and before it is sampled, since a title may clear the metadata and sample the target before it next draws to it.
- When encoded work may write the metadata word, the check runs on the GPU: a one-thread program reads the word after that work, marks it expanded and sets the vertex counts of indirect full-screen draws that write the clear value. This avoids a CPU wait per check, which kept the CPU and GPU from overlapping. DCC codes on integer targets still wait on the CPU.
- Metal renders a pass only over its smallest attachment, so the HTILE clear draw runs in a depth-only pass when a colour target is smaller than the depth target.

### Textures and tiling

- **Swizzle modes**: linear, `SW_256B_S` (1), `SW_4KB_S` (5), `SW_64KB_S` (9) and the PS5's render-target mode (0x1B, `SW_64KB_R_X` as the PS5 lays it out). Depth modes are not detiled. `SW_4KB_S` is taken as the first 4 KiB of a `SW_64KB_S` block and `SW_256B_S` as the first 256 bytes of a 4 KiB block, as the standard swizzle is built up; unit tests in `tiling.rs` check the modes against each other and the render-target mode against the video-out table below.
- 2D textures are detiled on the GPU by generated kernels. 3D `SW_4KB_S` images are detiled on the CPU with addrlib's patterns; linear 3D images are copied by a blit, each depth slice holding a whole mip chain laid out as a linear 2D one (measured on RDNA2).
- **Mip chains**: the smaller levels come first (in tiled modes the mip tail, then the levels from the smallest up), so level 0 starts after them. Linear surfaces have no tail, pad rows to 256 bytes and pack array slices with no further padding (measured on RDNA2). Sampled textures get every level; the tail follows SharpEmu-mac's tables. Descriptors with a base level above 0 get level 0 only.
- **A texture at a render target's address reads that target**, through a view with the descriptor's base and last level. A view in another format is made only where Metal allows it (same bytes per pixel); otherwise the draw goes without the texture and the problem report lists it.
- **Integer formats** are declared `<uint>`/`<int>` in the MSL. The renderer reads the bound descriptors per draw and translates the program again for each integer combination it meets, because the format lives only in the descriptor and can differ between draws of one program.
- **When guest memory changed**: an image read from guest memory keeps a 64-bit digest of its bytes, taken while no unwaited work may write them. In a later frame it is reused if the digest matches and no such work may write it; otherwise it is detiled again. Images unused for 300 frames are dropped. A CPU write that restores earlier bytes between two uses is missed only on a digest collision. Re-detiling every frame ended render passes constantly, because an upload after memory-writing work must sit in the main command buffer.
- **2D-array storage images** are loaded from guest memory when made and keep a copy of it; when the memory differs from the copy, only the changed texels are copied in (checked before writes and once a frame before reads, up to 32 MiB). Loading the whole array again would throw away texels programs wrote, which live only in Metal. Linear 3D storage images are loaded from guest memory when made; tiled ones start zero-filled.
- **Frame extraction writes program-written images back** to guest memory as their descriptors lay them out (`Metal::write_back_images`, linear and 2D tiled layouts), so a dispatch's image stores can be compared with hardware. Rendering never does.

### Programs and the compile pool

Translation runs on the GPU thread. Programs are cached by address together with their register state and a hash of the first 256 bytes of code, since titles load other programs where freed ones were. MSL and pipelines compile on a pool with one thread per performance core. A thread that waits for work nobody has started runs it itself, so waiting never queues behind unrelated compiles; a render pipeline is queued only once both its functions are done.

- Dispatches always wait for their pipeline: the guest may wait on their results, and we cannot tell which ones it reads. A dispatch with zero groups does not wait.
- A draw whose pipeline is not ready waits up to 4 ms (8 ms per frame for all such draws). Then it is skipped for this frame only if every target it draws to was first drawn in an earlier frame and also in each of the last two frames (or is a display buffer), and neither program writes memory. Otherwise it waits. A skipped draw still applies its depth clears. Reason: a draw into a new target may make output that is drawn once and never repaired (a texture rendered at load), and a skipped draw cannot be replayed later because the ring buffers it read are overwritten.
- A draw skipped again in a later frame moves its programs' compiles to the front, so warm-up shaders a title draws once sink behind what the scene keeps asking for.
- Metal's own shader cache persists across runs of the same build, so a second run compiles almost nothing; every translator change starts cold. An archive of our own would add nothing.

### GPU hangs

A command buffer that runs too long fails with a hang error, after which macOS ignores every later command buffer of the process; the picture freezes until restart. A runaway can also outlive its killed process and lock the Mac up until a reboot. So:

- Translated programs carry loop guards ([shaders.md](shaders.md)), and the problem report counts the threads that gave up. A guard trip means a real bug elsewhere (a list the GPU built wrongly, a translation fault), not a reason to raise the limit.
- Dispatches over 2^24 groups are skipped; indirect sizes over that are zeroed on the GPU.
- Lockstep wave meetings give up after a bounded number of polls.
- When working on the GPU path, stop running titles after one hang and find the cause offline, from captures.

## Frame capture and replay

A capture (`.ps5frame`) records one frame, from the flip that starts it: every queue's command buffers in execution order, each queue's register state and the GDS at the start, the GPU-only textures (as for save states), the non-zero guest pages the frame reads as they were before it, and the picture the game showed. `ps5slop-replay` runs it through a fresh Metal renderer with no game or hypervisor; its usage is in [debugging.md](debugging.md). Captures are game data and stay out of the repo.

How replay differs from live:

- Submissions run in the recorded order on one thread, one processor per queue, so cross-queue timing (and any race the live run had) is not reproduced.
- Replay waits for every pipeline instead of skipping draws while compiling.
- Times of single passes or draws do not add up to a live frame: timing them one by one removes overlap, and the first run of a frame pays one-time costs. Use medians over several warm frames.
- The live picture in the capture lets the replay be compared with what the game showed. A difference points at timing or state outside the frame, not at the renderer's logic. The replay maps each captured range as one segment, so the GPU area is a single extent there and its windows cover all of it: a window that misses data shows only live. Effects that read earlier frames (exposure, bloom) can differ a little when both are right.

## Save states

Render targets, depth targets and program-written volumes are read back into a save state and made again on load; other caches are rebuilt from guest memory. Writing them into guest memory in the console's tiled layouts would need tiling encoders we do not have. See [savestates.md](savestates.md).

## Video out

`libSceVideoOut` (`crates/hle/src/videoout.rs`) follows the call sequence ScreenTester5 uses (GPL-3.0, read for the sequence only). None of it is verified on a PS5:

- One port. `SetBufferAttribute2` fills a block the guest treats as opaque, so it holds our own layout. `RegisterBuffers2` takes `{ data, metadata, reserved[2] }` entries.
- A flip shows the buffer at once (index -1 blanks the screen) and fires a flip event; vblank is a 60 Hz clock.
- A buffer the GPU rendered is shown from its Metal texture; a buffer the CPU filled is read from guest memory and untiled.
- Formats whose top byte is `0xC0` (Superliminal's RGBA16Float `0xC001000600000000`) hold linear light, inferred from the title's own final pass, and are encoded as sRGB for display and screenshots. No public source names these formats.
- Tiling mode 0, as ScreenTester5 writes it, is the render-target mode (0x1B): 32-bit pixels in 128x128 tiles of 64 KiB, tiles row-major. Inside a tile a pixel's byte offset is the XOR of these bits (`tiling.rs` tests its equation against this table):

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

## PS4 titles (Gnm)

`libSceGnmDriver` submissions run on their own GPU thread through `gpu::Processor` with no renderer: labels, waits, `EVENT_WRITE_EOP` and its interrupt work as for AGC, register writes are recorded, draws are skipped, and a submit-and-flip flips after its buffers. Constant command buffers are not run. The packet builders a title calls write one NOP over the space it reserved, so the processor never reads stale data. Every end-of-pipe interrupt raises the event id the title registered; which id the console raises for which interrupt is unknown.

Drawing PS4 titles would need the driver's real packet builders, a gfx7 (GCN) decoder for the translator (other encodings and opcode numbers, no NGG), gfx7 render state and GCN tiling. There is no PS4 renderer unless a PS5 title needs one.

## Known gaps

- **NGG expansion programs** (vertex programs that use LDS, barriers, `exp prim` and `GS_ALLOC_REQ` to emit their own geometry) do not translate, and their draws are skipped. The fitting design is a Metal mesh shader per such program. Not started.
- **Ray tracing** is emulated in software inside the shader (`image_bvh_intersect_ray` over the game's own BVH nodes, see [shaders.md](shaders.md)). That stays only while no title needs more; ray tracing is to move to Metal's ray tracing (acceleration structures built from the game's BVHs or rebuilt from their geometry), not to a better software traversal. Bindless hit shading (textures picked by a run-time index) does not translate.
- Multi-draw indirect packets are counted, not drawn. GDS ordered append and GWS are not modelled. `RELEASE_MEM` data select 5 (GDS to memory) is ignored, and register sources and destinations of `COPY_DATA` and `WRITE_DATA` are dropped (and reported).
- Depth swizzle modes have no equation, so depth-layout images that nothing rendered read as zeros. Stencil-only targets are not made.
- Rendered pixels never reach guest memory, so a title that reads a render target on the CPU sees stale data.
- The hardware clips primitives at the guard band and interpolates over the clipped pieces; Metal does not, so values in clipped triangles can differ in the last bits (noperspective ones by more). `10_10_10_2` and `11_11_10` targets are made as `2_10_10_10` and `10_11_11`, whose bits lie elsewhere; sRGB encoding and 16-bit unorm/snorm exports to targets of fewer bits follow Metal rather than the CB (which keeps the low bits).
- Levels of 3D images that lie in a mip tail, and tiled 3D modes other than `SW_4KB_S`, are not read.
