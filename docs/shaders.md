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

- A group that is exactly one wave64, without `s_barrier`, that uses LDS or an instruction needing whole-wave masks (`crosses_lanes`: a non-spill `v_readlane`, `v_mbcnt`, 64-bit lane counts, `s_cmp_*_u64`, 64-bit scalar ops that pick lanes out of EXEC or VCC with a constant, 64-bit vector shifts of EXEC or VCC). Such programs pass values between lanes through LDS without barriers (tile depth reductions, look-back scans).
- A group of several whole wave64s that uses an instruction from `crosses_lanes` or an LDS atomic that returns a value (`ordered_lds`), whose result depends on lane order (radix sort ranks).

In lockstep programs:

- Lane masks are the whole wave's 64-bit value, the same in all 64 threads. VALU mask writes are a ballot of both SIMD groups through threadgroup memory; reads of this lane's bit use the lane's index in the wave. `v_readlane`, `v_readfirstlane` and `v_mbcnt` see the whole wave. Scalar instructions on masks then need nothing special.
- Branches are voted across the wave (`wave_any`), so both halves take the same path. Scalar branches are voted too, which is exact only while scalar values agree across the wave.
- One-wave groups put a threadgroup barrier around every LDS instruction and every exchange.
- In groups of several waves, each wave's two halves meet by themselves instead of through a group barrier: each publishes how many meetings it reached in a threadgroup atomic and waits for its partner, so waves stay independent between `s_barrier`s as on hardware. A meeting gives up after a bounded number of polls (`MEET_SPINS`), so a lost partner slows a dispatch but cannot hang the GPU. A returning LDS atomic runs in the low half, then the high half; within one SIMD group Metal was seen to apply same-address atomics in lane order (measured, not documented). 512 bytes of threadgroup memory are taken from the LDS for the exchange slots.
- A lockstep group must run whole, so partial groups (thread-sized dispatches) run whole with grid bounds. Grid bounds on a lockstep program with a multi-dimensional group are refused.

Cost: every mask write is an exchange, which costs a barrier or a meeting. Narrowing that to the masks that really need the whole wave is planned (below).

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

Other scalar loads read guest memory through the windows. A multi-dword scalar load looks up its window once for the whole range, with a fallback over at most two windows; per-dword 64-bit window lookups were a large share of some programs' time.

### Memory access

- **Guest memory** is reached through up to three windows (see gpu.md), described to the program by `Win`. `goff` maps a guest address to an offset from the first window's buffer. Addresses outside every window read 0, and stores to them are dropped.
- **V# fields are decoded in the shader at run time** (base, stride, `num_records`, format, destination selects), because titles build V#s at run time: Balatro loads them from a table at a computed offset and rewrites the format bits before `buffer_load_format`. Static buffer bindings could come back as an optimisation where a V# provably sits in user SGPRs; passing formats known at translation is planned.
- **Range checks** follow `OOB_SELECT` for strided buffers; buffers without a stride check the byte offset against `num_records`. Out-of-range loads return 0, out-of-range stores are dropped, out-of-range atomics return 0.
- **Stores and atomics skip the module image** a window covers (`Win.ro`), see gpu.md.
- **`glc` buffer, flat and global accesses are relaxed atomics** (`volatile` loads in programs that write no memory). Reason: other groups of the same dispatch may poll or write that memory (look-back scans poll the previous group's flag), and with plain accesses Metal's compiler hoists the load out of the polling loop, which then never sees the other group's store. Relaxed atomics order nothing else; whether data written before a flag is always seen with it is not checked.
- **LDS** is `threadgroup uint lds[]` in compute programs, sized from `COMPUTE_PGM_RSRC2.LDS_SIZE` (32 KiB, Metal's limit, when it is 0). Out-of-range accesses read 0 or are dropped. `s_barrier` is a threadgroup barrier on threadgroup and device memory. Vertex and pixel programs that use `ds_*_addtid` for register spills get a per-thread array.
- **GDS** is a 64 KiB device buffer. `ds_append`/`ds_consume` add or subtract one per thread at `M0[31:16]` (GDS) or `M0[15:0]` (LDS) plus the offset. Hardware adds a wave's active lane count once and hands out ranks with `v_mbcnt`; here each thread counts itself, so totals and the indices handed out match but their order does not.
- **Atomics Metal lacks**: float min and max are compare-and-swap loops; 64-bit and, or, xor and swap are two 32-bit atomics (bitwise ones high dword first, `swap_x2` low dword first, so a reader polling the high dword of a record also sees the low one, assuming one thread's relaxed atomics reach memory in order). The returned old value can tear. Other 64-bit atomics do not translate. Image atomics use a 32-bit integer view made with atomic usage (Metal refuses that usage for `R32Float`).
- **Scratch**: loads in a program that never stores to scratch read 0 (some ray-tracing programs read scratch only on an overflow path). Programs that store to scratch do not translate.

### Stages

- **Vertex, NGG** (the program in the ES slot, run as the merged ES/GS program): user SGPRs from s8, the merged wave info in s3 set so the prologue's EXEC arithmetic leaves this lane active, vertex index in v5, instance index in v8. `GS_ALLOC_REQ` and the primitive export are dropped: one vertex per thread, primitives assembled by Metal. `pos0` becomes the position, `paramN` user outputs. Programs that emit their own geometry need mesh shaders (gaps, below).
- **Vertex, legacy VS**: user SGPRs from s0, vertex index in v0, instance index in v3. Untested.
- **Pixel**: user SGPRs from s0, VGPRs laid out by `SPI_PS_INPUT_ADDR`. Interpolated inputs follow `SPI_PS_INPUT_CNTL_n` and the i/j VGPRs an instruction reads (center, centroid or sample; perspective or linear). Inputs read with `v_interp_mov` or marked flat are `uint4 [[flat]]`, and the paired vertex function exports them bit for bit, so integers pass unchanged. Inputs that share a location read the same export. `exp mrtN` becomes a colour output, unpacked by `SPI_SHADER_COL_FORMAT`; `exp mrtz` writes depth only when its depth component is exported. A lane whose EXEC bit is clear at the `done` export, or that ends without exporting, is discarded.
- **Compute**: user SGPRs from s0, then the workgroup ids and the group size as `COMPUTE_PGM_RSRC2` enables them, thread ids in v0 to v2. The renderer dispatches `COMPUTE_NUM_THREAD_X/Y/Z` threads per group, one Metal thread per lane. With grid bounds, threads past the grid size start with EXEC clear, which is how hardware runs a partial wave.
- The vertex body is also a plain function, so the renderer can wrap it (rectangle lists, triangle fans).

### Math

- **Programs compile in safe math mode.** Metal's default fast math assumes no NaN or infinity and transformed a correct program into one that produced NaN pixels (bloom then spread them over the screen). Safe rather than relaxed because the translation mirrors the hardware's instructions one by one, and the hardware does not reassociate them either. It cost no measurable compile time.
- `v_cvt_pkrtz_f16_f32` rounds toward zero, so values past the largest half become 65504, not infinity.
- 16-bit VALU ops write the low or (with `op_sel`) high half and keep the other half; float inline constants are halves. The older `v_cvt_f16_f32` still clears the high half. Packed half ops follow `op_sel`/`op_sel_hi`; a float inline constant is used in both halves (assumed).
- `s_setreg_b32` (rounding and denormal modes) is ignored; denormal and NaN handling follow Metal's. `v_mac_f32` and friends may be fused by Metal's compiler.

## Known approximations and gaps

- Ballots, `ds_swizzle`, `ds_permute_b32`, and cross-lane operations that span all 64 lanes outside lockstep programs are not translated (`ds_bpermute` works within 32-lane halves).
- Depth-compare samples compare one filtered sample instead of filtering four compared ones (no PCF). Integer textures are sampled through the sampler like float ones, which Metal allows only unfiltered.
- `image_sample_d` on cubes, `a16` sampling and the `a16` form of `image_bvh_intersect_ray` do not translate. Texel offsets on samples shift the coordinates by offset / size, 2D and 2D arrays only.
- `image_store` writes 0 to components `dmask` leaves out. Destination selects are not applied to `buffer_store_format`.
- Buffer formats cover every `BUF_FMT` with 8-, 16- and 32-bit components and `2_10_10_10`; `10_11_11`, `11_11_10` and `10_10_10_2` read as 0.
- FLAT and GLOBAL atomics, GDS ordered counts and scratch stores do not translate.
- NGG programs that use LDS, barriers and `exp prim` to build their own geometry need a mesh-shader path (a threadgroup per wave, LDS as threadgroup memory, vertex and primitive exports as mesh outputs). Not started.
- `image_bvh_intersect_ray` is software traversal over the game's BVH nodes. It stays while no title needs more; ray tracing is to move to Metal's ray tracing ([gpu.md](gpu.md)).

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
- Draws: a pass-through vertex program, a pixel program built for the case and a linear render target, compared pixel by pixel: every colour format and number type through both export paths, component swaps, interpolation modes and pixel inputs, every blend factor and function, logic ops, write masks, culling, fill rule, viewport sign, scissors, points and lines. Draws use a negative viewport Y scale, as PS5 titles do, except where the sign is what is tested.

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
