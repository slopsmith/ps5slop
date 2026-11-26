# Shaders

How PS5 GPU programs become Metal Shading Language. The code is the `gfx10` crate (`crates/gfx10`): a decoder, a printer in LLVM's syntax, and a translator to MSL. The renderer that binds and runs the result is in [gpu.md](gpu.md).

PS5 titles ship AMD GFX10 machine code for gfx1013 (GFX10.1 plus BVH ray-tracing and MSAA-load instructions; see gpu.md for the source). We translate it straight to MSL text, with no IR library in between: SPIR-V plus SPIRV-Cross would pull in Apache-2.0 code that our GPL-2.0-or-later licence cannot absorb ([references.md](references.md)), and Metal's own compiler already builds SSA from the register locals and optimises well.

There is no PS5 to check instruction results on. Semantics come from the sources below; what titles draw (compared with console captures) and our own GPU tests are the checks.

## Public facts and their sources

| Fact | Source | Status |
| --- | --- | --- |
| Encodings and opcode numbers for gfx1013 | Mesa ACO's `aco_opcodes.py` (MIT), turned into `src/opcodes.rs` by `tools/gen_opcodes.py`, which drops what GFX10.1 lacks and renames to LLVM's mnemonics | Checked against llvm-mc |
| Instruction semantics | AMD's RDNA1 and RDNA2 ISA guides | Public; often recalled rather than re-read |
| Buffer range checks by `OOB_SELECT` | RDNA2 ISA guide, section 8.1.5, table 35 (as SharpEmu-mac states it) | Public |
| `v_cvt_pkrtz_f16_f32` rounds toward zero | AMD's ISA guide | Public |
| 16-bit VALU results keep the other half of the register on GFX10 | LLVM's AMDGPU backend, as remembered | Not checked on hardware |
| Register fields (`SPI_*`, `COMPUTE_PGM_RSRC2`, `USER_SGPR_MSB`, ...) | Mesa's register files (`gfx10.json`, `gfx103.json`, `gfx10-rsrc.json` for buffer formats) | Public |
| Stage inputs: NGG vertex programs take user SGPRs from s8, the vertex index in v5 and the instance index in v8; compute system SGPRs and thread ids by `COMPUTE_PGM_RSRC2` | SharpEmu-mac (`Gen5NggExports.cs`, `DecodeComputeSystemRegisters`) | Matches every program seen |
| MIMG opcodes `0xA0` to `0xBE` are PS5 "sampler adjust" forms of the sample instructions, each `0x80` above its base with the same operands | SharpEmu-mac; not in the public RDNA2 docs, llvm-mc rejects them | Unverified |
| BVH node layout and the ray-box and ray-triangle tests of `image_bvh_intersect_ray` | Mesa radv's software emulation (`radv_nir_rt_common.c`, MIT) | Games' descriptors match radv's layout; hardware details assumed |
| GDS `ds_append` base is `M0[31:16]` | Inferred: only the high half explains Astro Bot's `DMA_DATA` resets of its counters | Not documented anywhere we know of |

## The `.ags` container

Shader files as some titles ship them (seen in Balatro's `shaders/*.ags`). Everything here was observed by reading such files; nothing is documented anywhere we know of. The renderer does not need it (it finds code through `SPI_SHADER_PGM_LO_*`, patched by `sceAgcCreateShader`); `ags.rs` exists so tests and the `ags` example can translate files without the HLE layer.

- An ELF64 file with `e_machine` 0xE0 (`EM_AMDGPU`) and the sections `.shader_text`, `.shader_header`, `.metadata`, `.cfg`.
- `.shader_text` holds the machine code from offset 0, then padding and a trailer from the magic `sl00` to the string `barefoot` and a 32-bit hash (equal to the checksum SH register's value). No header field holds the code size; `Ags::code()` cuts at `sl00`.
- `.shader_header` starts with the AGC shader header `sceAgcCreateShader` takes (magic `0x34333231`). Its pointer fields are relative to the field itself, as `crates/hle/src/agc/shader.rs` relocates them, and point to `(register, value)` tables for SH and context registers.
- Every shader seen is wave64.

## Decoder and printer

`decode::decode` turns words into an `Inst` (opcode info and the format's fields) for every GFX10 format, with literals and inline constants. `print::print` writes LLVM's syntax; `print::print_checked` also says whether LLVM's disassembler would accept the encoding. Those acceptance rules were found by comparison with llvm-mc, not from documentation.

The oracle is `llvm-mc --disassemble -mcpu=gfx1013` from the dev shell; `tests/llvm.rs` runs it when it is on PATH: every Balatro shader decodes and prints as llvm-mc does, random encodings of every format agree (both reject, or both print the same, or LLVM's text does not assemble back to the same words), and an ignored test tries every opcode number of every format. llvm-mc knows only syntax, so this checks nothing about what instructions do.

## Translation model

`translate::translate(&Request) -> Result<Translated, TranslateError>`. The request carries the code, the stage, the register state the program starts with and per-draw choices (integer texture slots, grid bounds, a dropped depth export). The result is one MSL library plus what the renderer must bind: textures, samplers, constant buffers, guest memory, GDS, the loop guard. Binding indices are in `translate::binding`. Translation is two passes: a dry run finds descriptor sources, live SGPRs and image atomics, the second emits MSL.

### Lanes and registers

- **One Metal thread is one lane.** SGPRs, VGPRs, `vcc`, `exec`, `m0` and `scc` are local variables, declared only when used. It is a straight register model, not SSA.
- **A lane mask** (exec, vcc, a compare result, a carry) holds this lane in bit 0 of its low half, outside lockstep programs. Scalar arithmetic on masks is computed exactly on those values. This is exact as long as no lane looks at another lane's bit.
- **Outside lockstep programs a scalar write of EXEC keeps only bit 0** and clears `exec_hi`. Reason: `s_wqm_b64` or `s_mov_b64 exec, -1` would otherwise set bits for lanes that, in this model, are other threads, and waterfall loops (`v_readfirstlane`, `v_cmpx`, `s_andn2_b64 ... exec`) then never run out of lanes. Risk: code that compares EXEC with a full mask sees a partial wave; none found.
- Every vector instruction and export runs under the lane's EXEC bit; consecutive vector instructions share one `if`.
- **Wave32 programs** are detected by `*_saveexec_b32` or 32-bit scalar writes of `exec_lo`; there a VALU compare or carry-out writes one SGPR, not a pair.
- Cross-lane instructions map to Metal's quad and SIMD-group functions: DPP (quad permutations, row shifts and rotates, mirrors, `row_share`, `row_xmask`, DPP8), `v_permlane16`/`permlanex16`, `ds_bpermute_b32`, and `v_readlane` of a constant lane all become `quad_*` or `simd_shuffle`/`simd_broadcast` within the 32-lane Metal SIMD group. Rows of 16 line up with the hardware's. Outside lockstep programs nothing crosses lane 31.
- `v_writelane`/`v_readlane` with a constant lane that the program writes first are SGPR spills: one variable per (register, lane).

### Lockstep waves

On Apple GPUs a SIMD group is 32 threads, so a wave64 is two SIMD groups that know nothing of each other. That breaks programs whose lanes rely on running together. Compute programs where that matters run **in lockstep**:

- A group that is exactly one wave64, without `s_barrier`, that uses LDS or an instruction needing whole-wave masks (`crosses_lanes`: a non-spill `v_readlane` or `v_writelane`, `v_mbcnt`, 64-bit lane counts, `s_cmp_*_u64`, 64-bit scalar ops that pick lanes out of EXEC or VCC with a constant, 64-bit vector shifts of EXEC or VCC). Such programs pass values between lanes through LDS without barriers (tile depth reductions, look-back scans).
- A group of several whole wave64s that uses an instruction from `crosses_lanes` or an LDS atomic that returns a value (`ordered_lds`), whose result depends on lane order (radix sort ranks).

In lockstep programs:

- Lane masks are the whole wave's 64-bit value, the same in all 64 threads. VALU mask writes are a ballot of both SIMD groups through threadgroup memory; reads of this lane's bit use the lane's index in the wave. `v_readlane`, `v_readfirstlane` and `v_mbcnt` see the whole wave, and `v_writelane` writes only its lane. Scalar instructions on masks then need nothing special.
- Branches are voted across the wave (`wave_any`), so both halves take the same path. Scalar branches are voted too, which is exact only while scalar values agree across the wave.
- One-wave groups put a threadgroup barrier around every LDS instruction and every exchange.
- In groups of several waves, each wave's two halves meet by themselves instead of through a group barrier: each publishes how many meetings it reached in a threadgroup atomic and waits for its partner, so waves stay independent between `s_barrier`s as on hardware. A meeting gives up after a bounded number of polls (`MEET_SPINS`), so a lost partner slows a dispatch but cannot hang the GPU. A returning LDS atomic runs in the low half, then the high half; within one SIMD group Metal was seen to apply same-address atomics in lane order (measured, not documented). 512 bytes of threadgroup memory are taken from the LDS for the exchange slots.
- A lockstep group must run whole, so partial groups (thread-sized dispatches) run whole with grid bounds. Grid bounds on a lockstep program with a multi-dimensional group are refused.

Cost: every mask write is an exchange, which costs a barrier or a meeting. Narrowing that to the masks that really need the whole wave is planned (below).

### Float semantics

Float VALU instructions follow the float mode of the stage's `PGM_RSRC1` (FLOAT_MODE, DX10_CLAMP, IEEE_MODE), fixed at translation; `s_setreg`, `s_round_mode` and `s_denorm_mode` do not change it. The rules are in `float_prelude.metal`, whose helpers take and return bit patterns, because Metal flushes f32 denormals in all arithmetic and comparisons (also in its safe math mode) and returns its own NaN. Checked against RDNA2 hardware with the oracle below:

- A NaN result is the first NaN source as it is (quieted only in IEEE mode), after input modifiers; subtraction negates its second source first. An invalid operation gives `0xFFC00000`; in an FMA an invalid product wins over a NaN addend. Min and max follow the ISA guide's pseudo-code, and two NaNs give the first.
- Round modes other than nearest even come from the exact rounding error (TwoSum, `fma`), with tiny operands scaled up first so that Metal does not flush the error. f16 arithmetic runs in f32, where its sums, products and FMAs are exact or have an exact error, and is rounded to f16 once.
- In modes that keep f32 denormals, operations a denormal reaches run as exact integer arithmetic. Results below the smallest normal are flushed before rounding.
- Output modifiers are ignored in IEEE mode and when output denormals are kept; clamp turns NaN into 0 under DX10_CLAMP.
- `v_mad_f32` and its legacy, `mac`, `madmk` and `madak` forms round the product, as GFX10.1 does; GFX10.3 runs them fused, so the oracle's results for them differ in the last bit by design.
- `v_rcp`, `v_rsq`, `v_sqrt`, `v_exp`, `v_log`, `v_sin` and `v_cos` take Metal's functions with the hardware's special cases. AMD's approximations are not documented, and their f32 results differ from hardware's in the last bit or so for many inputs.

### Control flow


- Without branches the body is straight-line code.
- **Structured** (`Nest` in `flow.rs`): the translator reads branches as the guest compiler's structured code. A forward conditional branch over a region becomes `if`, with an `else` when the region ends in an unconditional branch further on; a backward branch target starts a loop up to the last branch back to it; branches back to the innermost loop's start are `continue` and branches just past it are `break`.
- **Fallback**: a program where a branch leaves its region or jumps back to an outer loop keeps a `pc` dispatcher: basic blocks in address order, each `if (pc == L)`, with a loop around each backward branch's range. Loops that overlap without nesting fall back further, to `for (;;) switch (pc)`. Reason for avoiding the switch: Metal's compiler treats it as one loop around the whole program and spends much of its time in loop passes, and loops only where the program has them compile and run faster.
- **Branch conditions**: vertex and pixel programs branch per lane. Compute programs use `simd_any` over the SIMD group, so a SIMD group branches as one, as a 32-lane slice of the wave would on hardware; that keeps `s_barrier` in control flow that is uniform per SIMD group (Metal requires it), and loops that exit through `execz` still end. Lockstep programs vote across the wave.
- Decoding ends at the first `s_endpgm`, `s_code_end` or `s_branch` that no branch jumps past.

### Loop guards

A program that never leaves a loop hangs the GPU, and after one hang macOS ignores all of the process's later Metal work ([gpu.md](gpu.md)). So every translated loop is guarded; the limits are guesses at "more than any honest loop":

- **Vertex and pixel** invocations give up after 1024 backward branches: a pixel is discarded, a vertex returns what it has.
- **Compute** threads give up after 16,384 backward branches of their own, or once all threads of the dispatch together took 2^30 (counted in units of 1024 in a buffer the renderer zeroes per dispatch), so a runaway in a huge grid also ends quickly. Threads that give up are counted for the problem report.
- In a program with `s_barrier`, a thread that gives up sets a threadgroup flag, and the group leaves together after the next barrier, so no thread waits at a barrier the others skipped. A thread that reaches no barrier soon leaves alone. Lockstep programs leave as a wave.
- A guard trip is a symptom: the cause is usually a GPU-built list or tree we got wrong, or a translation fault in the program that built it.

### Descriptors

Where each T#, S# and constant-buffer base comes from is found by a forward data-flow pass over the control-flow graph (`sources.rs`), and the renderer evaluates the result per draw or dispatch (`Source`):

- user SGPRs, followed through moves;
- memory reached by a chain of `s_load_dword*` with immediate offsets from a pointer in user SGPRs (`Source::Memory`);
- `s_buffer_load_dword*` through a V# found that way, at an immediate offset plus, optionally, a dword shifted left (`Source::Buffer`, `Source::Scaled`), which is how some titles pick a sampler by index;
- dwords built from constants (`s_mov`, `s_movk`, `s_bfm_b32/b64`; a 32-bit literal in `s_mov_b64` is taken as zero-extended).

VCC is tracked too, and forgotten only after instructions that can write it (VOPC without an explicit destination, carry ops, explicit VCC destinations, lane reads into VCC; not `v_cmpx`, which writes only EXEC on GFX10). A register whose origin differs between paths, or that anything else wrote, is unknown, and an image instruction using it fails translation.

- **Masked descriptor dwords keep their origin.** An OR or AND of a descriptor dword with anything that is not itself a descriptor dword (a constant, a run-time mask, through VCC or in place) keeps the dword's origin, and the changed bits are dropped: the descriptor is bound as it is in memory. Titles set per-quad sampler bits or rewrite a T#'s type field this way, and failing the whole program skipped its draws; an approximate sample is closer than none. Risk: a masked value used as an index would keep its unmasked origin.
- **Samplers get one slot per distinct source**, whatever register they were loaded into, since Metal has only 16 sampler slots and some programs load the same samplers into several registers. Textures keep per-register slots.
- **Integer textures**: slot numbers do not depend on the integer masks, so the renderer can translate once, read the descriptors, and translate again with `uint_textures`/`sint_textures`.
- Textures picked by a run-time index (bindless) are not supported.

### Constant buffers

A scalar load whose base the source pass traces (a pointer or a V# at a known place), with an immediate dword-aligned offset and no `glc`, in a program without `s_dcache_inv`/`s_gl1_inv`, reads `cbN[k]` with a constant index from a buffer the renderer binds per draw (up to 16). The renderer resolves the base like a descriptor and binds the no-copy window over it, the same memory programs use, or a copy made while encoding where the range is not all guest memory. Loads whose results only image and sampler instructions use are left out, since the renderer binds those descriptors itself.

Programs that write guest memory declare these buffers `device const`, not `constant`: Metal reads a `constant` buffer as it was when the dispatch began, and a program may read back with a scalar load what it stored before.

Other scalar loads read guest memory through the windows. A multi-dword scalar load looks up its window once for the whole range, with a fallback over at most two windows; per-dword 64-bit window lookups were a large share of some programs' time. Their `s_buffer_load_dword*` read 0 for each dword at or past the V#'s range: `num_records` bytes without a stride (matches RDNA2 in the hardware oracle), `num_records` strides with one (assumed, untested). Loads read through `cbN` are not range-checked.

### Memory access

- **Guest memory** is reached through up to three windows (see gpu.md), described to the program by `Win`. `goff` maps a guest address to an offset from the first window's buffer. Addresses outside every window read 0, and stores to them are dropped.
- **V# fields are decoded in the shader at run time** (base, stride, `num_records`, format, destination selects), because titles build V#s at run time: Balatro loads them from a table at a computed offset and rewrites the format bits before `buffer_load_format`. Static buffer bindings could come back as an optimisation where a V# provably sits in user SGPRs; passing formats known at translation is planned.
- **Range checks** follow `OOB_SELECT` (RDNA2 ISA 8.1.5); buffers without a stride check the byte offset against `num_records` in every mode. The SGPR offset only moves the address, and in mode 3 shrinks the buffer. Dword loads and stores check each dword on its own, typed accesses and atomics their whole element. Out-of-range loads return 0 (1 for a `SEL_1` select), out-of-range stores are dropped, out-of-range atomics return 0.
- **Swizzled buffers** interleave the dwords of `index_stride` records, as the ISA guide's formula says, but only for accesses with `idxen`: offset-only ones stay linear, and what one access moves lies contiguous from its swizzled address, so wide accesses of neighbouring lanes overlap (both measured on the reference GPU). `ADD_TID_ENABLE` is not handled.
- **Typed accesses** decode `BUF_FMT` at run time, with names listing components from the top bit down (`2_10_10_10` has x in its low bits, as Mesa maps it). `buffer_load_format` fills components a format lacks by repeating the ones it has (x y x y for two), then applies the destination selects; `tbuffer` loads ignore the selects and read 0 past the format. Typed stores write the whole element, missing data as 0, through the selects in `buffer_store_format` (a two-component format's second component takes `DST_SEL_W`'s data; `SEL_0` stores one and `SEL_1` zero, seen on one descriptor only) and in order in `tbuffer` stores. Stores saturate integers, store NaN as 0 in normalized and scaled kinds and round floats toward zero, keeping NaN payloads. D16 loads round normalized values to the nearest half, floats toward zero, and saturate integers. All measured on the reference GPU.
- **Stores and atomics skip the module image** a window covers (`Win.ro`), see gpu.md.
- **`glc` buffer, flat and global accesses are relaxed atomics** (`volatile` loads in programs that write no memory). Reason: other groups of the same dispatch may poll or write that memory (look-back scans poll the previous group's flag), and with plain accesses Metal's compiler hoists the load out of the polling loop, which then never sees the other group's store. Relaxed atomics order nothing else; whether data written before a flag is always seen with it is not checked.
- **LDS** is `threadgroup uint lds[]` in compute programs, sized from `COMPUTE_PGM_RSRC2.LDS_SIZE` (32 KiB, Metal's limit, when it is 0). Out-of-range accesses read 0 or are dropped. `s_barrier` is a threadgroup barrier on threadgroup and device memory. Vertex and pixel programs that use `ds_*_addtid` for register spills get a per-thread array.
- **GDS** is a 64 KiB device buffer. `ds_append`/`ds_consume` work at `M0[31:16]` plus the offset, in LDS as in GDS (measured for LDS). The hardware adds or subtracts a wave's active lane count once and returns the old value to every lane; lockstep programs do that, elsewhere each thread counts itself, so totals match but the values handed out differ.
- **LDS atomics in lockstep programs take effect in lane order**, as on hardware: native ones run the low half first, and those built from compare-and-swap loops or plain accesses (float ones, `rsub`, `inc`, `dec`, `wrap`, `mskor`, 64-bit ones) run one lane at a time. `wrxchg2` makes the whole wave's first exchange before its second, and `ds_permute_b32`, like `ds_bpermute_b32`, works within 32-lane halves of a wave64 (measured). LDS float min and max follow the memory atomics' rules (below); `ds_add_f32` returns a NaN operand quieted.
- **Atomics Metal lacks**: float min and max, wrapping increment and decrement and float compare-and-swap are compare-and-swap loops. Float min and max follow the hardware: denormals flush to signed zero, a signalling NaN comes back quieted, a quiet NaN loses to a number, -0 is below +0. Metal has no 64-bit atomics on every Apple GPU, so 64-bit and, or, xor and swap are two 32-bit atomics (bitwise ones high dword first, `swap_x2` low dword first, so a reader polling the high dword of a record also sees the low one, assuming one thread's relaxed atomics reach memory in order); add and sub carry into the high dword, which keeps concurrent sums exact; the rest read both dwords and write them back, which can tear when lanes update one place at once. Returned old values can tear. Image atomics use a 32-bit integer view made with atomic usage (Metal refuses that usage for `R32Float`).
- **Scratch**: each lane's scratch is a private array of `WAVESIZE` (from `COMPUTE_TMPRING_SIZE` or `SPI_TMPRING_SIZE`, in KiB per wave on GFX10) over the wave's lanes, zeroed at the start, read and written at any byte; accesses past its end read 0 and are dropped. Loads in a program that never stores to scratch read 0 without one (some ray-tracing programs read scratch only on an overflow path). A program that stores to scratch with no scratch size does not translate.

### Images

Image loads and stores follow the hardware's conversions, measured with the hardware oracle on RDNA2 (gfx1033; unverified on the PS5's GFX10.1). Where the ISA guide is silent, the rules below are what the reference GPU did.

- **Three views of a T#**, one texture slot each: sampled (Metal applies the channel selects through a swizzled view, decodes sRGB and filters), raw (`image_load`/`image_store`: bound as memory holds it; the program applies `DST_SEL` and converts), and bits (the `_pck` forms). T# dwords 1 and 3 reach the program in the user buffer (`binding::TEXTURE_WORDS`), because the translator drops descriptor loads.
- **Channel selects** that pick a channel the format lacks give channel `c mod n` of its `n` channels (an `X` format reads `xxxx`, `XY` reads `xyxy`), for loads, samples and gathers alike.
- **sRGB loads** decode to the exact value rounded to bfloat16. Stores to sRGB images do not encode: they write the value as UNORM.
- **Stores**: a two-channel format takes the first and the last component (x and w), others take theirs in order. SCALED formats are stored as NORM (clamped to [0, 1] or [-1, 1], then scaled). Stores to `4_4`, the depth and stencil formats (`8_24`, `24_8`, `X24_8_32`) and the 96-bit formats leave memory unchanged; the `_pck` forms do write them.
- **Formats Metal lacks** (SCALED, the integer and normalized `10_11_11`/`11_11_10`/`10_10_10_2` forms, `5_6_5`, `4_4`, shared exponent, depth and stencil) are bound as unsigned integers of the element's size for raw access, and the program converts each texel by the format in the T# (`image_decode`, `image_encode` in the prelude). The renderer asks for this per slot (`Request::coded_textures`). Sampling them is not supported.
- **`_pck`** moves the element's dwords unconverted, one VGPR per `DMASK` bit (zero past the element), 8- and 16-bit elements zero- or sign-extended. Other raw accesses to a T# a program also reads or writes as bits share that slot and convert in the program, so they see its stores within the thread.
- Out-of-range loads return 0 and out-of-range stores are dropped (ISA guide 8.2). `image_get_resinfo` returns 0 sizes past the last level and for dimensions an image lacks, and a 1D array's layer count in y.
- 1D and 1D-array images are 2D ones of height 1: Metal's 1D textures have no mip levels.
- **Filtering**: the hardware rounds the sample position (after the half-texel shift) to 8 fractional bits and blends with those weights. The exact blend of those weights matches about half of its bilinear results bit for bit and the rest within 4 / (255 * 65536); the hardware's blend arithmetic is not known. Metal's weights are coarser (errors around 4e-4 on RGBA8), so filtered samples differ from the hardware in the low bits.

### Stages

- **Vertex, NGG** (the program in the ES slot, run as the merged ES/GS program): user SGPRs from s8, the merged wave info in s3 set so the prologue's EXEC arithmetic leaves this lane active, vertex index in v5, instance index in v8. `GS_ALLOC_REQ` and the primitive export are dropped: one vertex per thread, primitives assembled by Metal. `pos0` becomes the position, `paramN` user outputs. Programs that emit their own geometry need mesh shaders (gaps, below).
- **Vertex, legacy VS**: user SGPRs from s0, vertex index in v0, instance index in v3. Untested.
- **Pixel**: user SGPRs from s0, VGPRs laid out by `SPI_PS_INPUT_ADDR`. Interpolated inputs follow `SPI_PS_INPUT_CNTL_n` and the i/j VGPRs an instruction reads (center, centroid or sample; perspective or linear). Inputs read with `v_interp_mov` are `uint4 [[flat]]`, and the paired vertex function exports them bit for bit, so integers pass unchanged. Inputs that share a location read the same export. `exp mrtN` becomes a colour output, unpacked by `SPI_SHADER_COL_FORMAT`; `exp mrtz` writes depth only when its depth component is exported. A lane whose EXEC bit is clear at the `done` export, or that ends without exporting, is discarded.
- Pixel inputs as RDNA2 gives them (measured with the hardware oracle): `POS_W` is the clip-space w (Metal's position holds 1/w), `POS_FIXED_PT` the pixel's x and y in the low and high halves, `SAMPLE_COVERAGE` 1 (targets are single-sampled). `FLAT_SHADE` in `SPI_PS_INPUT_CNTL` leaves `v_interp_p1/p2` interpolating. The ancillary word reads 0; the hardware sets bits 0 and 1 to 2 for triangles, for a reason we do not know. `v_interp_mov_f32` of `p10` or `p20` (vertex differences) reads `p0`: Metal gives a fragment no other vertex's value.
- The export does what the CB would do to the value where Metal's conversion differs (measured): F16 exports are decoded in integer steps (Metal's compiler may keep a value converted to half at float precision, so a plain `half` round trip did not round); 32-bit float exports to a 16-bit float target that does not blend are rounded toward zero, as the CB does; each target channel takes the export component `CB_COLOR_INFO.COMP_SWAP` names (`Request::swizzles`); when MRT0 blends with a `SRC1` factor, the MRT1 export is its second source (`[[color(0), index(1)]]`, `Request::dual_source`), as the hardware takes it for dual-source blending (Mesa's radeonsi, not yet checked on hardware); and a ROP3 other than copy reads the target through `[[color(n)]]` and combines the bits as stored (`Request::logic_targets`; integers saturate to the channel first).
- **Compute**: user SGPRs from s0, then the workgroup ids and the group size as `COMPUTE_PGM_RSRC2` enables them, thread ids in v0 to v2. The renderer dispatches `COMPUTE_NUM_THREAD_X/Y/Z` threads per group, one Metal thread per lane. With grid bounds, threads past the grid size start with EXEC clear, which is how hardware runs a partial wave.
- The vertex body is also a plain function, so the renderer can wrap it (rectangle lists, triangle fans).

### Math

- **Programs compile in safe math mode.** Metal's default fast math assumes no NaN or infinity and transformed a correct program into one that produced NaN pixels (bloom then spread them over the screen). Safe rather than relaxed because the translation mirrors the hardware's instructions one by one, and the hardware does not reassociate them either. It cost no measurable compile time.
- `v_cvt_pkrtz_f16_f32` rounds toward zero, so values past the largest half become 65504, not infinity. It converts in integer steps, since Metal may keep `half(x)` at float precision; a NaN keeps its sign and top payload bits and becomes quiet (measured). The flushing of half denormals under `FLOAT_MODE` is not modelled.
- 16-bit VALU ops write the low or (with `op_sel`) high half and keep the other half; float inline constants are halves. The older `v_cvt_f16_f32` still clears the high half. Packed half ops follow `op_sel`/`op_sel_hi`; a float inline constant is used in both halves (assumed).
- `s_setreg_b32` (rounding and denormal modes) is ignored; denormal and NaN handling follow Metal's. `v_mac_f32` and friends may be fused by Metal's compiler.

## Known approximations and gaps

- Ballots, `ds_swizzle`, and cross-lane operations that span all 64 lanes outside lockstep programs are not translated.
- Depth-compare samples compare one filtered sample instead of filtering four compared ones (no PCF); `image_gather4_c` compares each gathered texel. Integer textures are sampled through the sampler like float ones, which Metal allows only unfiltered.
- `image_sample_d` on cubes, `a16` sampling, gathers with a LOD, bias or clamp, `image_gather4h`, `image_get_lod`, float image atomics and the `a16` form of `image_bvh_intersect_ray` do not translate. Texel offsets on samples shift the coordinates by offset / size.
- `image_store` writes 0 to components `dmask` leaves out.
- Unaligned dword buffer accesses round down to a dword (Metal's behaviour), and typed accesses are read where they fall. The reference GPU honours unaligned dwords, and gave 0 to a whole wave of indexed typed loads through a stride-2 buffer with a wider format.
- GDS ordered counts do not translate. Scratch through buffer instructions (a V# with `ADD_TID_ENABLE`) is not handled.
- NGG programs that use LDS, barriers and `exp prim` to build their own geometry need a mesh-shader path (a threadgroup per wave, LDS as threadgroup memory, vertex and primitive exports as mesh outputs). Not started.
- `image_bvh_intersect_ray` is software traversal over the game's BVH nodes. It stays while no title needs more; ray tracing is to move to Metal's ray tracing ([gpu.md](gpu.md)). On a triangle the ray misses, the hardware still returns its numerators and denominator with an infinite t; we return (inf, 1, 0, 0) as radv's emulation does.

## Testing

- `tests/llvm.rs`: decoding and printing against llvm-mc.
- `tests/unit.rs`: hand-written programs, assembled with llvm-mc for gfx1013, translated, checked as MSL text and compiled with Metal's runtime compiler (the offline `xcrun metal` is not installed). This checks that the MSL compiles, not what it computes.
- GPU tests in `crates/gpu/src/metal/tests.rs` run our own programs on Metal and check what they compute. This is the only check of semantics short of a title's picture, so a translation fix should come with one. A test of a loop must also bound the loop from user data, so a broken guard cannot hang the GPU.
- `PS5SLOP_SHADER_DUMP` saves each program a title runs with its register state (`capture.rs`); the `captures` example translates and compiles a folder of them offline, or prints one program's disassembly and MSL, and `compile_times` times Metal's compiler on them. Captures are game data and stay out of the repo.
- Frame replay (gpu.md) is the oracle for changes to emitted code: old and new output of the same frame must render the same pixels.

### Hardware oracle

`crates/hwcheck` checks translated instructions against a real AMD GPU. Each test is one program around one instruction, over many lanes or waves of operands: edge cases of each operand's type paired with each other, then seeded random values. `ps5slop-hwcheck gen` builds the suites (`crates/hwcheck/src/suite/`):

- Vector ALU: every VOP1, VOP2, VOPC, VOP3 and VOP3P opcode with plain operands, in its own encoding, as VOP3, with neg and abs, clamp and omod, op_sel, DPP16, DPP8, SDWA, a literal and inline constants, and float instructions under each float mode (`FLOAT_MODE` rounding and denormals, IEEE mode, DX10 clamp). Each lane stores its result and its own bit of the mask written and of EXEC, since masks are per thread outside lockstep programs.
- Lane-crossing: `v_readlane`, `v_writelane`, `v_readfirstlane` under several EXEC masks, `v_permlane16`/`x16`, `v_swap_b32`, the `v_movrel*` family.
- Scalar ALU (SOP1, SOP2, SOPC, SOPK) and scalar loads, including buffer loads through descriptors of varied size and out-of-bounds mode; one record per wave, with SCC.
- LDS: every DS load, store, atomic, two-address form, swizzle and permute.
- Vector memory: MUBUF, MTBUF, GLOBAL and FLAT loads, stores and atomics, over buffer descriptors of varied format, stride, size, swizzle, destination select and out-of-bounds mode. Stores are compared as the memory they leave.
- Images: every MIMG load, store, atomic, sample, gather and query on every image dimension it allows, over linear textures in memory; one opcode of each family runs over every GFX10 image format (the list is generated from Mesa's register database by `tools/gen_formats.py`) and a set of samplers (filters, wrap modes, border colours, unnormalized coordinates, depth compare, mips, LOD bias). Stores are read back with `image_load`.
- Flow and machine state: every branch under conditions a wave agrees on and ones lanes disagree on, the `s_*_saveexec` family, `s_getpc_b64`, `s_setreg` on the float mode, and instructions that must leave results alone (waits, nops, priority, prefetch, cache invalidates, `s_endpgm`).
- Scratch: every SCRATCH load and store as a round trip through each lane's private memory, each wave with its own FLAT_SCRATCH.
- Ray tracing: `image_bvh*_intersect_ray` on random rays at box and triangle nodes laid out as radv lays them out.
- Draws: a pass-through vertex program, a pixel program built for the case and a linear render target, compared pixel by pixel: every colour format and number type through both export paths, component swaps, interpolation modes and pixel inputs, every blend factor and function (`src1` factors with a second export to MRT1, as dual-source blending needs), logic ops, write masks, culling, fill rule, viewport sign, scissors, points and lines. Draws use a negative viewport Y scale, as PS5 titles do, except where the sign is what is tested. Their triangles stay inside the clip volume, since no guard band is set and the hardware's clipping changes interpolation in the last bits; `draw clip` tests clipping on its own. Blending draws export FP16, as RB+ requires for 8-bit and half-float targets (Mesa's `ac_choose_spi_color_formats`): the Deck has RB+ and garbles blends of 32-bit exports into those targets, while gfx1013 has no RB+, so that pairing cannot be checked here. `SX_BLEND_OPT_CONTROL` stays 0: setting both of a target's disable bits makes the hardware skip blending. Blending draws set the `SX_MRT*_BLEND_OPT` hints, and dual-source ones `CB_BLEND1_CONTROL.ENABLE` and `DISABLE_DUAL_QUAD`, as radeonsi's `si_create_blend_state` does: with hints that did not fit the blend, the hardware wrote three of every four pixels as fragments of their neighbours.

Each test runs through our whole path (the PM4 command processor, the translator, Metal) on the Mac, and `ps5slop-hwcheck run` runs the same command buffer as raw PM4 through libdrm on a Linux machine with an amdgpu GPU, with the program and data at the same GPU addresses. `scripts/hwcheck HOST [--filter TEXT]...` does both over SSH, leaves the outcomes in `target/hwcheck/` and merges them into `crates/hwcheck/results/<gpu>.tsv`. `ps5slop-hwcheck coverage` lists which opcodes the translator refuses. On the status page an opcode whose every test the translator refused counts as not translated, and DPP, SDWA and VOP3 cells exist only for encodings LLVM accepts. A few opcodes cannot be tested in isolation: `v_illegal` and `s_trap` end the program, `s_code_end` is padding, `s_sendmsg` only signals fixed-function units. `image_msaa_load` waits for multisampled draws, since GFX10 has no linear multisampled surfaces.

A draw needs state the hardware would otherwise inherit from whoever used the GPU before: the hardware alone gets `CONTEXT_CONTROL`, `CLEAR_STATE` and compute-unit masks before the draw and a colour cache flush with an L2 write-back after it. `CLEAR_STATE` resets context registers but not user-config ones, so every draw also sets `GE_INDX_OFFSET` and the other index registers itself; left alone, they made the hardware read vertices off by one in some submissions.

Two kinds of behaviour are kept out of the tests on purpose:

- Stores from several lanes to one address. The hardware's order is defined (the highest lane's value stays) and Metal's is not, so every store test gives each lane its own place. Atomics still collide.
- Unaligned dword addresses. The reference machine runs in the unaligned memory mode Linux sets up and honours them, Metal rounds down, and which mode the PS5 sets is unknown.

What the hardware result is evidence of:

- The reference machine is a Steam Deck: RDNA2, gfx1033, GFX10.3. The PS5 is GFX10.1, so a match means "matches RDNA2", and PS5 behaviour stays unverified. llvm-mc disassembles every program for gfx1013 and for the reference GPU; a test whose program reads differently there is not run.
- GFX10.3 dropped the unfused multiply-adds (`v_mad_f32`, `v_mac_f32`, `v_madmk_f32`, `v_madak_f32`) and runs the legacy ones fused. The runner rewrites them to their fused forms and marks those results, which may differ from the PS5 in the last bit.
- Differences are sorted into NaN bits (both NaN), denormal or zero sign, and other values.

The latest results are kept in `crates/hwcheck/results/gfx1033.tsv`, one line per test, so the status page can be built from trunk without the GPU: `ps5slop-hwcheck chart crates/hwcheck/results/gfx1033.tsv OUT.html` colours every opcode form by whether the translator takes it and whether the hardware agrees. Results describe the translator as it was when they were made; run the oracle again after changing it.

## Toward a compiler

The translator emits each instruction as its own MSL snippet, with rare cases written out inline, and leaves the rest to Metal's compiler. On a GPU what that compiler cannot remove costs more than on a CPU: registers per thread set how many threads a core holds, spills go to memory, and the instruction cache is small. Heavy titles are bound by GPU time spent in code around the work, so the translator moves toward resolving what is known per draw, in steps. Each step keeps the current form as the fallback for what it cannot handle, so programs never stop translating, and each is measured by code size, spills and time per program, with the frame pixel-identical.

Done: constant buffers and structured control flow (above). Next:

1. Fewer whole-wave masks: in lockstep programs, ballot only the mask values that reach an instruction needing the whole wave.
2. Formats known at translation: buffer formats and strides from V#s the source pass traces go into the request, so `buffer_load_format` becomes the one conversion it needs.
3. The EXEC guard inside each structured `if`, which Metal's compiler may not see through.

An SSA form between decoding and emission is not planned until a step needs it.
