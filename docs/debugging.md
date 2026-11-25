# Debugging a running guest

## Logging

Diagnostics go through the `tracing` crate to stderr as `LEVEL target: message`. By default a run shows errors (work that was dropped: skipped draws, failed pipelines, guest faults), warnings, and the main crate's milestones (the title, the control socket, loaded states). `RUST_LOG` takes `tracing-subscriber`'s `EnvFilter` syntax and is applied on top of that default, so `RUST_LOG=gpu=debug` adds the GPU crate's debug output and `RUST_LOG=hle::stub=off` silences placeholders. A target is the module path of the code that logs (`gpu::metal`, `hle::dmem`, `jit::engine`, ...) unless the table names it; a filter on a prefix covers every module under it.

| Target | Level | What |
| --- | --- | --- |
| `hle::stub` | warn | The first calls of each placeholder, and imports passed over by `PS5SLOP_UNIMPLEMENTED_ZERO` |
| `gpu::trace`, `gpu::snap`, `gpu::probe`, `gpu::watch`, `jit::probe` | info | Output asked for by `gputrace`, `gpusnap`, `PS5SLOP_GPU_TRACE`, `PS5SLOP_GPU_PROBE`, `PS5SLOP_GPU_WATCH` and `PS5SLOP_PROBE_RIP`; on by default |
| `gpu::stats` | info, debug | Counters per queue and for Metal each second; at debug also packets by opcode (NOP codes as `0x100 \| code`, `EVENT_WRITE` as `0x46TT`) |
| `jit::stats` | info | Statistics mode: what the interpreter ran and why the VM was left, and host calls by function, at exit. Turning it on makes the JIT count, which costs time |
| `gpu::sync` | debug | Every submission, wait, release, command-processor write, REWIND wait and dispatch, per queue thread, with seconds since start |
| `gpu::compile` | debug | Each translated program, compile job and draw skipped for want of a pipeline |
| `gpu::msl` | debug, trace | The MSL of programs that fail to compile; at trace every program's MSL |
| `gpu::targets` | debug | Each render target the renderer creates and each image first read from guest memory, with the frame number |
| `gpu::dma` | debug | Every `DMA_DATA` packet |
| `hle::vfs` | debug | Every guest path through the mount table, found or missing |
| `hle::ampr` | debug | Every APR read |
| `hle::dmem` | debug | Direct memory allocations, frees, maps and aliases |
| `hle::pad` | debug | Changes to the pad's output state (vibration, light bar, trigger effects) |
| `hle::ngs2` | debug | NGS2 voice controls, commands and waveform blocks |
| `hle::videoout` | debug | Which texture a `shot` or frame dump reads |
| `ps5slop` | debug | The imports a title uses that are not implemented, by name |
| `ps5slop::report` | info | Per-run problem report printed at exit when any problems were recorded; on by default |

Reports that would repeat every frame are rate-limited (`diag::FirstN`, `diag::Distinct`): each kind of problem is logged the first few times.

## Debug switches

`PS5SLOP_*` environment variables, read once at start into `diag::Config` (`crates/diag/src/config.rs`). All of them are listed in [user/environment.md](user/environment.md). Traces that need nothing but output are log targets instead (see above).

## The control socket

Every `ps5slop run` listens on a Unix socket: `ps5slop-PID.sock` in the temporary folder, or the path given with `--control PATH`. The path is printed on stderr at start. macOS limits socket paths to 104 bytes, so a deep scratch folder does not work; the default does.

The protocol is the one 86slopper uses: one command per line, and each answer ends with a line holding a single `.`. `scripts/ps5slop-ctl SOCKET COMMAND...` sends commands and prints the answers; `socat - UNIX-CONNECT:SOCKET` works too. `help` lists the commands. Numbers are hex, with or without `0x`.

| Command | What it does |
| --- | --- |
| `regions` | Mapped regions with protection and name |
| `memory` | Memory use: direct memory allocated, free and its largest free block; mapped regions by name with their totals; the largest free range in the GPU area; our host heap; the host process's resident size and physical footprint; how often each kind of memory failure happened |
| `mem ADDR [LEN]` | Hex dump |
| `dump ADDR LEN PATH` | Writes guest memory to a host file |
| `poke ADDR HEXBYTES` | Writes bytes, ignoring guest protection |
| `find HEXBYTES [START END]` | Searches readable memory, first 32 hits |
| `sym ADDR` | The region (module) and offset of an address, or the HLE stub |
| `threads` | Guest threads: name, number of host calls, and the host function each called last with its caller |
| `watch ADDR LEN` | Logs writes to a range; adds to earlier watches |
| `hits [N [ADDR LEN]]` | The writers of watched memory (or of one range) with counts and address spans, and the last N writes |
| `trace PATH [SITE...]` / `trace off` | Appends every watched write (from the given sites only) to a text file |
| `unwatch` | Ends all watches and clears the log |
| `calls PATH [FILTER]` / `calls off` | Appends every host call whose description contains FILTER (or, for `@LO-HI`, made from that address range, such as one module's code) to a file: thread, name, first six arguments, result, caller |
| `gputrace N` | Traces every draw and dispatch of the next N frames on stderr, as `PS5SLOP_GPU_TRACE` does (N decimal) |
| `targets DIR` | Writes every render target, depth target and sampled texture as `ADDR.ppm` (the low hex digit says which: 1 texture, 2 colour target, 4 depth). The first call starts recording and writes nothing (a target is recorded when a draw next uses it); with `PS5SLOP_DUMP_TARGETS=1` recording runs from the start |
| `press BUTTONS [MS]` | Holds pad buttons (`cross`, `up+cross`, ...) for MS milliseconds (default 400: games read the pad once per frame, and our frames can be slow). `l2` and `r2` pull the triggers fully; `lup`, `ldown`, `lleft`, `lright` and `rup`, ... push a stick to its edge (`lup+rright` walks forward while turning); `ls-up`, `rs-left`, ... are the same; `l3`, `r3` and `touchpad` press those. `lx=N`, `ly=N`, `rx=N`, `ry=N` (0 to 255, decimal, 128 centred, y grows downwards) push one stick axis part way, and `l2=N`, `r2=N` pull a trigger part way (`press lx=170+ly=40 2000` walks forward and a little right) |
| `tap BUTTONS [COUNT [MS]]` | Presses and lets go COUNT times (default 1), MS milliseconds each way (default 600), for mashing a button (GTA V's sprint is tapped `cross`) |
| `hold BUTTONS` / `release [BUTTONS]` | Keeps buttons, sticks and triggers pressed until released (all, or only those named), so a held stick or aim can be combined with `press` and `tap`, which add to what is held. Each answers with what is held now |
| `profile on` / `profile off` / `profile [N]` | Starts a fresh profile window, stops profiling, or reports the window so far (top N rows): per guest thread the time in translated code, the interpreter, translating and host functions, the time in none of these (exits, locks, the engine), the CPU time used and the guest instructions per second of translated-code time; memory map changes; the hottest guest code by sample (module and offset); host functions by time and calls; the instructions and addresses the interpreter ran most; VM exits by cause (host call, interpreter, new block, lookup table collision, fault, kick), each with its hottest guest RIPs or host functions; translations and evicted cache segments. Samples at a VM entry point are counted as "kicked before entering": the sampler's kick was pending while the thread was outside the VM |
| `problems [PATH]` | Prints the per-run problem report (or writes it to `PATH`): skipped draws and dispatches (including compiling skips), untranslated programs with the failing instruction, failed Metal compilations/pipelines with Metal's reason, unsupported formats/primitives/tile modes, memory window failures, loop-guard trips, PM4 wait timeouts, unimplemented and placeholder HLE calls, and JIT interpreter fallbacks by opcode, grouped by reason and sorted by impact (count and per-frame rate) |
| `gpusnap DIR [FIRST]` | During the next frame, copies every attachment of each render pass when the pass ends (from pass FIRST on, decimal), and writes them as `NNN-Dd-PROGRAMS-mrtN-ADDR-sS.ppm` and `...-depthS-ADDR.ppm`: the frame's composition pass by pass. Depth is stretched from its nearest to its farthest value below 1; colours are clamped to [0, 1]. At most 1 GiB of copies |
| `gpucapture [PATH]` | Captures the next frame (PM4 command streams across all queues, initial queue register/GDS state, GPU-only render/depth/volume textures, and the non-zero 16 KiB guest memory pages referenced during the frame) to `PATH` (or a timestamped `.ps5frame` in `PS5SLOP_CAPTURE_DIR`) for deterministic offline replay with `ps5slop-replay`. Requests made while another capture is armed, recording or writing are queued and run in order |
| `metalcapture [PATH]` | Captures the next frame as an Apple Metal `.gputrace` bundle at `PATH` (or a timestamped `.gputrace` in `PS5SLOP_CAPTURE_DIR`) via `MTLCaptureManager`, for inspection in Xcode's GPU debugger. Requires `METAL_CAPTURE_ENABLED=1` in the process environment (also available offline via `ps5slop-replay dump --gputrace`) |
| `gpuskip [ADDR...]` | Skips every draw and dispatch that uses one of these program addresses (vertex, pixel or compute); with none, skips nothing. Comparing screenshots with and without a pass shows which pass causes a bug. Careful with compute programs: a skipped program's output may hold the arguments of a later indirect dispatch or draw, which then reads garbage (see GTA V, [gpu.md](gpu.md#gta-vs-prologue-played-on-2026-11-22)) |
| `gpureload` | Translates every program again at the end of the frame, reading `PS5SLOP_MSL_OVERRIDE` files again |
| `shot PATH` | Writes the screen as PPM; with `press`, this drives menus step by step |
| `frames DIR N` | Writes the next N frames the guest flips as `DIR/FLIP.ppm` (N decimal), in the background; a flip number missing from the names was not caught. Shows flicker and one-frame glitches that single shots miss |
| `save PATH` | Writes a save state and lets the guest go on; `ps5slop run --load-state PATH GAME` goes on from it in a fresh process (same title and `--data` folder). Answers with the size and the times. See [savestates.md](savestates.md) |
| `rates` | The guest's instructions per second, translated and interpreted, since the last `rates`, as the window title shows them (the first call starts the window) |
| `quit` | Ends the guest |

Thread names come from `scePthreadCreate`, `scePthreadRename` and `pthread_set_name_np`. Host threads are named `guest 0xTCB`.

A guest fault also prints the return addresses along the frame-pointer chain (module and offset), so the code that called into a failing routine shows without reading any code. The `calls` log is flushed when the guest exits, so its last lines are the calls just before the end.

## Profiling

`profile on` starts a window; `profile` reports it (see the table above). When the run ends with profiling on, the window so far is printed to stdout (top 20 rows), also when the app is quit with ⌘Q, which ends the process inside the window's event loop; the problem report is written then too. Times come from the engine: around each VM entry, each interpreter run and each translation, and around each host call in the process. Samples come from the code cache sampler, which kicks running vCPUs 100 times a second (see [jit.md](jit.md)); while profiling, each sample's PC is mapped back to a guest RIP. Off, it costs one relaxed load per VM exit and per host call. It reports addresses and module offsets only, never code bytes: the hot code may be Sony's own modules, which we do not disassemble ([legal.md](legal.md)).

The report also has a GPU thread line: time encoding draws and dispatches, waiting for Metal, the GPU's own busy time per command buffer, flip pacing, memory waits and idle time, and the scheduling and wake-up latencies around each command buffer. For the GPU itself, record a Metal System Trace: copy the binary, sign the copy with `scripts/hypervisor.entitlements` plus `com.apple.security.get-task-allow`, run it with `PS5SLOP_GPU_LABELS=1` (encoders get labels naming their targets, programs and detiled images), then `xcrun xctrace record --template 'Metal System Trace' --attach PID --time-limit 3s --output X.trace` and export the `metal-gpu-intervals` table (its values are deduplicated through `id`/`ref` attributes). macOS's `sample PID SECONDS` gives host stacks per thread.

`PS5SLOP_TRACE_USER_MEMORY=1` adds, to a `gputrace`, the first 256 bytes behind every pointer-like pair of user data words of each draw's vertex and pixel programs and each dispatch (attribute tables, vertex buffer descriptors, constant buffers; for pixel programs and dispatches V# bases too), read at encoding time; ring-buffer contents are gone by the time the control socket could read them. Traced draws also list the Metal texture each slot is bound to (`bound tN`, object and size), the raw scissor registers and `PA_SC_MODE_CNTL_0`, and each frame which texture was copied for the screen. `RUST_LOG=hle::videoout=debug` prints which texture a `shot` or frame dump reads.

`gpusnap` options: `PS5SLOP_SNAP_INPUTS=1` also copies every 2D texture a pixel program samples just before its draw (`NNN-in-PROGRAM-tN.ppm`; this ends the render pass before each draw, so it splits passes and uses up the 1 GiB faster: start late with `FIRST`), and `PS5SLOP_SNAP_RAW=1` writes each copy's texels as stored next to the PPM (`...-WxH-FORMAT.raw`, FORMAT the `MTLPixelFormat` number), for reading exact float values. These showed that the inputs of Astro Bot's final pass were right, which moved the search from the program to the scissor. A frame that depends on the previous one (exposure, bloom, history) can stay broken after its cause is gone, and a save state keeps the broken targets: `gpuskip` on the draw that reads last frame's picture for a few seconds, then without it, tells a loop that recovers from one that diverges again (GTA V's white picture, [gpu.md](gpu.md#gta-vs-graphics-modes-2026-11-22-late-night)).

**Shader costs.** Which shaders take the GPU's time, how big their compiled code is and whether they spill registers: record the replay with GPU counters, `PS5SLOP_GPU_LABELS=1 xcrun xctrace record --no-prompt --template 'Metal System Trace' --instrument 'Metal GPU Counters' --output X.trace --launch -- target/release/ps5slop-replay time --warmup --runs 3 CAPTURE`, then `scripts/shader-profile X.trace`. It runs headless and needs no signing. The labels name pipelines and functions after their guest program; Instruments keeps them for most compute shaders but drops many fragment ones, which then show as `gen5_ps (N)`. Spills come per encoder, so a render pass names its targets, not the program that spilled. Shader-profiler time overlaps with other work and is not GPU time a shader adds to the frame: it gave GTA V's HTILE depth clear (`htile_fs`) about 3 ms a frame, and drawing that clear without a fragment shader changed the frame by less than the noise. Confirm with `time` before acting on it. Apple's compiler output itself cannot be read: the public AGX disassembler (dougallj/applegpu) covers the M1's G13 only.

**GPU hangs.** A command buffer that runs too long fails with `kIOGPUCommandBufferCallbackErrorHang`; after that macOS ignores every command buffer of the process, everything shown stays frozen or black, and `gpusnap` copies come back as garbage. The `gpu::stats` line counts failed command buffers and the first eight errors are printed. `PS5SLOP_GPU_ENCODER_STATUS=1` (with `PS5SLOP_GPU_LABELS=1` for names) asks Metal for each encoder's state and prints the faulted and pending ones with their label and last draws (a signpost per draw names its programs). Then `gpuskip` on candidate programs (those with backward branches) narrows it down.

**A hang can outlive the process.** GTA V's culling program once ran on the GPU after its process was killed: the process stayed stuck exiting (state `E`, `kill -9` has no effect), the GPU stayed at 100%, WindowServer restarted and only a reboot helped (2026-11-21). Translated loops now give up on their own ([shaders.md](shaders.md)), but before any GPU work (a game run or the GPU tests) check `ps -axo pid,stat,etime,command | awk '$2 ~ /E/'` for a stuck ps5slop process and do not start anything while one exists; give every run a hard time limit and kill it by PID; stop all GPU work after a single hang. Other agents and vic share the Mac's GPU, and runs that overlap can fail with `kIOGPUCommandBufferCallbackErrorOutOfMemory` or as the victim of another process's GPU recovery.

Switches for chasing a program that does not end or writes where it should not:

- `PS5SLOP_GPU_COMMIT_EVERY=N` commits the command buffer every N dispatches (1: each dispatch alone), and a command buffer still running after 10 s is reported with its label (`PS5SLOP_GPU_LABELS=1` names the last dispatch in it), so the first unfinished one names the program. With more than 32 command buffers in flight, the oldest still running after 2 s is reported too.
- Compute loops that give up are reported as `compute program ADDR: N threads gave up their loops (U of B branch units), user data [...]`, and counted in the `gpu::stats` line.
- `PS5SLOP_GPU_PROBE=ADDR` prints each dispatch of the compute program at ADDR with its groups, queue and user data, when the command processor reaches it.
- `PS5SLOP_GPU_WATCH=ADDR[+LEN][@SECONDS]` waits for the GPU after every draw and dispatch (from SECONDS after start) and reports the ones that change the 64-bit word at ADDR (or the LEN bytes there), with the queue and the compute user data. Slow; start it late. The word is read with guest permissions, so execute-only code always reads as 0: watch a readable word on the same write pattern instead. Each queue remembers the value it saw last, so a queue can report a change another queue made in between; trust the first report after a change. With `PS5SLOP_GPU_WATCH_DUMP=DIR` every new content of the range is written to a numbered file there, which is how a chain of compute passes can be checked step by step against a CPU oracle.
- `PS5SLOP_CHECK_CODE=MS` keeps a copy of every non-writable region (code and read-only data) as loaded, compares it every MS milliseconds and prints each 16 KiB page that changed: the time, the changed range, how many bytes differ and the old and new bytes. Nothing in the guest writes there, so any change is a host or GPU write that went astray. It showed GTA V's culling program writing 64-byte records every 0x3F80 bytes across the game's own code (see [gpu.md](gpu.md#gta-vs-culling-queue-and-the-compute-loop-guard-2026-11-21)).
- `WRITE_DATA` and `COPY_DATA` forms the command processor drops (register destinations, unknown selects) are reported once each.
- `PS5SLOP_GPU_FENCE=1` orders encoders that read guest memory after those that wrote it, across our window buffers ([gpu.md](gpu.md)).
- `cargo run -p gfx10 --example raw -- FILE [OFFSET]` disassembles raw GFX10 code with byte offsets and branch targets, for a program found in a title's own executable (`ps5slop-elf segment` writes a segment's bytes; the guest address minus the load base and the segment's address is the offset). Local analysis only, see [legal.md](legal.md).
- `PS5SLOP_CALLS=PATH[:FILTER]` writes the `calls` log from the start of the run instead of from a control-socket command, and `PS5SLOP_JIT_NO_NATIVES=1` keeps `memset`, `memcpy` and `memmove` as HLE calls so that log and write watches see them (the JIT normally runs them natively).

**A screen that never changes** while traces look right: compare `bound` textures in the trace with what a probe reads (`get_width()` in an `PS5SLOP_MSL_OVERRIDE`), and check whether the traced frame is the one shown. In Astro Bot only frames that drew UI reached the screen; the others' final pass was clipped to one pixel.

**Flicker**: `frames DIR 60`, then the mean brightness of each PPM, shows which flips are bad and whether they come with a period (Superliminal's came every third or sixth flip). In a run that flickers, keep the process alive, `gputrace` a few frames together with `frames` (a frame's trace ends with `frame N done`, and flip numbers run one ahead of N), and compare the normalised traces of good and bad frames; if they match, the difference is in data, so take several `gpusnap`s, find a bad one by its last pass, and compare pass by pass for the first that differs. Whether a run flickers can depend on where its allocations land, so judge a fix over several runs. Traced dispatches list their guest-memory windows (`windows FIRST SECOND THIRD [ranges]`), and name their program and user data even when it does not translate.

`PS5SLOP_MSL_OVERRIDE=DIR` replaces the translated MSL of a program by `DIR/STAGE-ADDR.metal` (for example `Pixel-1020040300.metal`) when that file exists. With `gpureload`, an edited file takes effect in the running game. Start from the program's translation (`cargo run -p gfx10 --example captures -- DIR NAME` on a `PS5SLOP_SHADER_DUMP` capture), and insert an early `{ PsOut p = {}; p.c0 = float4(...); return p; }` to show an intermediate value on screen. Probes written into an HDR target go through tonemapping, so scale values down (a twentieth worked for Minecraft). Moving the probe through the program finds the first value that is wrong.

A Metal frame capture (`MTLCaptureManager` to a `.gputrace`, opened in Xcode) is available on demand via `metalcapture PATH` over the control socket (launch with `METAL_CAPTURE_ENABLED=1`), or offline during replay via `ps5slop-replay dump CAPTURE OUT --gputrace OUT.gputrace`. Live captures can still be large because Metal records bound guest memory windows; replaying a `.ps5frame` capture keeps windows bounded to the pages the frame touched, making Xcode captures smaller and repeatable.

## GPU frame capture and replay

In windowed mode, press **F9** to capture the next frame as a zstd-compressed `.ps5frame` file, or **F10** to capture an Apple Metal `.gputrace` bundle (launch with `METAL_CAPTURE_ENABLED=1`). Captures are written off the render thread to `~/Library/Application Support/ps5slop/captures/<title>-YYYYMMDD-HHMMSS.ps5frame` (override the folder with `PS5SLOP_CAPTURE_DIR`), and the exact output path is logged to stderr (`gpu::snap`). Pressing F9 (or running `gpucapture` over the control socket) while a capture is already armed, recording or writing queues the new request so multiple captures in one run all complete in order. Captures hold game data and are blocked from commits by `.gitignore` and `scripts/check-forbidden`. A capture records the guest memory the renderer reads through the host during the frame. Images and memory windows the renderer made earlier are checked again for the captured frame (`Backend::capture_starting`), since they would otherwise never be read: before 2026-11-24, GTA V captures missed about 50 draws whose streamed textures had been detiled before the capture, and replayed with walls and ceilings missing.

`ps5slop-replay` replays a `.ps5frame` capture directly through the PM4 processor and Metal renderer without the game or hypervisor, waiting synchronously for every pipeline so results are deterministic (`ps5slop-replay --help` lists all subcommands and options):

```sh
# 1. Capture a frame in windowed mode: press F9 (or F10 for a Metal .gputrace)
#    and note the printed path, e.g. ~/Library/Application Support/ps5slop/captures/PPSA03420-20261122-143015.ps5frame

# 2. Turn the capture into a PNG of the final frame
cargo run --release -p gpu --bin ps5slop-replay -- render CAPTURE.ps5frame final.png

# 3. List every draw (programs, primitive type, counts, targets, scissor) and compute dispatch
cargo run --release -p gpu --bin ps5slop-replay -- list CAPTURE.ps5frame

# 4. Render the target after draw N (0-based) to a PNG (add --target HEX to pick a specific target)
cargo run --release -p gpu --bin ps5slop-replay -- render CAPTURE.ps5frame after-draw-42.png --draw 42

# 5. Dump every color and depth target (plus shown.png) to a folder, optionally recording a .gputrace
cargo run --release -p gpu --bin ps5slop-replay -- dump CAPTURE.ps5frame /tmp/targets --draw 42

# 6. Time every render pass and compute dispatch (sorted by GPU ms)
cargo run --release -p gpu --bin ps5slop-replay -- time CAPTURE.ps5frame --warmup

# 6b. Time every draw and dispatch on its own; GPU time by program and the slowest draws
cargo run --release -p gpu --bin ps5slop-replay -- time CAPTURE.ps5frame --warmup --draws --runs 5 --top 40

# 6c. Which draws write one texel of a target: its value after every draw that binds the target
cargo run --release -p gpu --bin ps5slop-replay -- history CAPTURE.ps5frame --target 0x117f860000 1024 1178

# 7. Compare against a golden PNG (or write/update the golden with --update)
cargo run --release -p gpu --bin ps5slop-replay -- check CAPTURE.ps5frame golden.png

# 8. CPU cost of processing and encoding the frame, warm, as the GPU thread spends it
cargo run --release -p gpu --bin ps5slop-replay -- bench CAPTURE.ps5frame --runs 5 --warmup 2
```

`time --draws` commits and waits after every draw and dispatch, so each has its own GPU time (Apple GPUs cannot sample their counters at draw boundaries). Each draw then runs in a render pass of its own and pays for loading and storing its targets, so the times add up to more than the frame's (GTA V in Fidelity mode: 232 ms against 207 ms by passes), and a cheap draw into many 4K targets looks dearer than it is. Small draws look far dearer alone than they are: GTA V's 326 shadow draws in one pass take 1.2 ms together and 24 ms timed one by one, so the per-program table is only trustworthy for draws of a millisecond or more. `--runs N` runs the frame N times in one renderer and reports each pass's or draw's median: the first run costs more (one-time costs a running game has already paid; a GTA V pass took 53 ms in a first frame and 7.6 ms warm) and single runs vary by a few milliseconds a draw, so compare variants with `--runs 5`, and by pass rather than by draw where a pass holds the draws in question. Even then the same pass varied by up to a fifth between runs (GTA V's 20 ms pass from 19 to 25 ms), so a change smaller than that needs several runs, interleaved with the unchanged program. To try a change to one program, put its edited MSL in a folder and set `PS5SLOP_MSL_OVERRIDE` (the `compile_times` example's `--dump` writes the MSL of programs `PS5SLOP_SHADER_DUMP` saved, and the replay honours both); a program that returns at once gives the draw's cost without its program. It answers which programs and draws to look at, not what they would cost in a normal pass ([gpu.md](gpu.md#gta-vs-fidelity-mode-on-the-gpu-2026-11-23)).

Targets start with what the game drew before the capture was taken, so `render --draw N` and `dump --draw N` often show the finished frame even for small N. `history` answers which draws write a pixel instead: it prints the texel's stored bytes after the first draw that binds the target and after each draw that changes it (`--all` lists the draws that left it alone too). The value after the first draw may already be last frame's. It syncs after every draw that binds the target, so it takes about as long as `time`.

`bench` runs the frame several times in one renderer, putting guest memory and the queues' registers back as captured before each run, so its caches are as warm as in a running game (a fresh renderer per run costs about 10 times as much, mostly texture uploads). Per run it prints the wall time of running the command buffers, the `profile` parts (draw and dispatch encoding, syncs, waits for command buffers anywhere, compile waits), the encoding time without the waits (`encode`), per draw, and the GPU's busy time; then the median. Loading the capture and resetting memory are not counted. The replay also records each draw for `list`, which costs about 1 us a draw that the game does not spend. Profile it with Time Profiler (`xcrun xctrace record --template 'Time Profiler' --launch -- ps5slop-replay bench ... --runs 80`); Time Profiler counts on-CPU samples only, so time blocked on the GPU does not show there, which is what the `waiting for command buffers` part is for.

`cargo test -p gpu --test frame_replay` runs a synthetic capture-and-replay regression test always, and replays any local `*.ps5frame` + `*.golden.png` (or `.golden.ppm`) pairs in `PS5SLOP_FRAME_CAPTURES` (default `captures/` at repo root, skipped when absent).

With `RUST_LOG=gpu::stats=info`, the Metal line ends its skip reasons with `images not supported [((KIND, ACCESS, TYPE, SWIZZLE, FORMAT), COUNT), ...]` once any image could not be bound: the numbers are hex, as in the `image Image { ... } not supported` reports (so `raw_tile: 18` is mode 24, `SW_64KB_Z_X`). It tallies what the first-seen reports cannot: Astro Bot's 6,237 "resources" skips in 300 s were three kinds of image, all in its first minute.

`RUST_LOG=gpu::stats=debug` (with `RUST_LOG=gpu::stats=info`) prints each queue's packet counts by opcode once a second (NOP codes as `0x100 | code`), which shows the packets a title uses that the command processor ignores. A `WAIT_REG_MEM` that gives up after 5 s (not counting time a command processor waits for a compile) prints its address, current value, function and reference (the first 16 per queue); a label nobody sets points at a missing packet, a skipped program or a queue blocked behind another. The message names the queue's thread (`gpu graphics`, `gpu compute 0x20`, ...).

`RUST_LOG=gpu::targets=debug` prints each render target the renderer creates and each image it first reads from guest memory, with the frame number: light enough to leave on for a whole run, it shows whether a target a later draw samples was ever drawn (Isaac's backdrop targets were not, in bad boots). `PS5SLOP_GPU_NO_THROTTLE=1` lets submitters run ahead of the GPU thread as before the submission throttle ([gpu.md](gpu.md)), for comparing.

With `RUST_LOG=gpu::stats=info`, each queue also prints its waits for the GPU by the packet that needed them (`DMA_DATA`, indirect arguments, `COND_EXEC`, releases drained before a poll or flip, ...), and the Metal line its waits to read memory programs wrote, by what it read (HTILE, CMASK, DCC, indices), and the HTILE checks it left to the GPU ([gpu.md](gpu.md#fewer-waits-for-the-gpu-2026-11-21)). Indirect dispatches whose sizes the GPU read are counted at the end of the waits line (`indirect dispatches sized on the GPU: N`); `PS5SLOP_GPU_CPU_INDIRECT=1` reads every indirect dispatch's sizes on the CPU after a wait, as before 2026-11-22 ([gpu.md](gpu.md#indirect-dispatch-sizes-on-the-gpu-2026-11-22-late-night)). `PS5SLOP_GPU_WAIT_SUBMISSIONS=1` makes every submission wait for its Metal work before its releases take effect, and `PS5SLOP_GPU_HTILE_CPU=1` waits and reads HTILE words on the CPU, both as before 2026-11-21, for comparing. The profile's first line counts the vCPUs threads gave back and the threads running without one ([jit.md](jit.md)).

`RUST_LOG=gpu::sync=debug` logs, per queue thread, every submission (address and size), wait, release, command-processor write, REWIND wait and dispatch (groups and program address). Grepping it for a label address shows who sets it and who waits for it; GTA V's Story Mode queues were followed this way. Each line carries the seconds since start, and the CPU's REWIND validations (`sceAgcRewindPatchSetRewindState`) appear in the same stream as `CPU sets REWIND`, so with `PS5SLOP_GPU_WATCH` (also timestamped) it shows when work became visible to the GPU, when it ran and when the CPU wrote the same memory. `PS5SLOP_GPU_SYNC_DISPATCHES=1` waits for Metal after every dispatch (slow), so when a program never finishes, the last `dispatch` line of its queue names it.

`RUST_LOG=gpu::dma=debug` prints every `DMA_DATA` packet as the command processor runs it: source and destination selects and addresses, size and the first word moved. It ruled out a DMA fill as what clears Astro Bot's transparency buffer (it was a DCC fast clear).

`PS5SLOP_TRACE_PIXELS=1` counts, with Metal's visibility results, the samples each traced draw passes (depth and stencil tests, discards), and prints them after the frame as `pixels slot N: COUNT`; each traced draw names its slot (`pixels slot N`). A draw that runs but covers nothing shows as 0. Traced draws and dispatches also name each program's guest-memory windows (`program ADDR windows [...]`, the `Win` values), and a skipped draw its unsupported primitive type. With `RUST_LOG=gpu::stats=debug`, `EVENT_WRITE` packets are counted by event type as `0x46TT` (bit 7 set when the packet carries an address), which told GTA V's per-frame cache flushes from its one occlusion query.

**Draws that cover nothing** (GTA V's characters, 2026-11-22): with the counts, override the vertex program (`PS5SLOP_MSL_OVERRIDE`) to change one thing and count again: the depth forced to the near plane (`o.pos.z = 0.999 * o.pos.w`) tells occlusion from geometry; a small jitter by vertex index (`o.pos.xy += float2(vid % 7, vid % 5) * 0.05 * o.pos.w`) tells collapsed triangles from culled ones; and moving every vertex to one of 16 screen cells by a 4-bit code (zero weights, zero position, off screen, `w <= 0`, ...) and dumping the G-buffer with `targets` shows which conditions hold. Values the program fetched can be classified the same way by copying them into locals right after their load. That found GTA V's vertex buffers reading as zeros (see [gpu.md](gpu.md#gta-vs-missing-objects-and-characters-2026-11-22)). If neither helps, override the pixel program with an early constant colour (as above): GTA V's glass covered nothing because its pixel program discarded every fragment, and the constant colour showed the panes in place (2026-11-22). An override that turns each `discard_fragment()` into a different colour then names the one taken.

`PS5SLOP_TRACE_REGS=1` adds to every traced draw the context (0xA000 + n) and uconfig (0xC000 + n) registers that changed since the previous traced draw. It showed that Superliminal's draws without a colour target were AGC's decompress draws rather than lights with lost state.

The profile's GPU thread line includes "compiling (compile threads)", the time the compile threads spent compiling MSL and Metal pipelines (summed over the threads), and "waiting for compiles (in encoding)", the time the GPU thread waited for them (see [gpu.md](gpu.md#compiling-off-the-gpu-thread-2026-11-21)). Metal keeps compiled programs in a cache shared by all processes, so a run of the same build usually compiles almost nothing; `PS5SLOP_MSL_UNIQUE=1` makes every program's source unique to measure a first run (Superliminal's New Game, when compiling still ran on the GPU thread: 33 s of compiling, a 31 s stall). `PS5SLOP_COMPILE_WAIT=1` makes draws wait for their pipelines instead of skipping, to rule out skipped draws when something is missing on screen; `PS5SLOP_COMPILE_THREADS=n` sets the number of compile threads (4). With `RUST_LOG=gpu::stats=info`, each gap between flips over 1 s is printed.

`RUST_LOG=gpu::compile=debug` prints a line per translated program (time on the GPU thread, MSL size, compiles queued), per compile job (library, vertex variant, render or compute pipeline: time in the queue, time waiting for its functions, its own time and the compile thread) and per draw skipped because its pipeline was not ready (frame, programs, targets, draw). Timestamps are seconds since the first such line. With `PS5SLOP_SHADER_DUMP`, `cargo run --release -p gfx10 --example compile_times -- DIR --threads N` compiles the captured programs again and times Metal's compiler (`--warm` lets Metal's cache answer, `--fast` uses Metal's fast math instead of the renderer's safe math, `--dump OUT` writes the MSL instead, `.metal` files in DIR are compiled as they are), for trying translator changes and compile options without running the game. To see where Metal's compiler spends its time, `sample` the busiest `MTLCompilerService` process while `compile_times` runs one large program repeatedly.

An MSL probe needs to compile: check it first with a few lines of Swift around `MTLDevice.makeLibrary(source:options:)`, since compile errors past the first few reports are not printed. The `captures` example's output ends with a summary line that is not MSL.

`PS5SLOP_PROBE_FLAT=1` makes every pixel program with a flat input write that input's raw value into colour output 1 (low byte red, next byte green, third byte blue) instead of what it computes, so a target dump shows per pixel which integer arrived (used to check Minecraft's terrain texture-set index).

For where host time goes at a given moment, record a Time Profiler trace of the running process with the same signed copy: `xcrun xctrace record --template 'Time Profiler' --attach PID --time-limit 15s --output X.trace`. The trace opens in Instruments with a timeline per thread, and selecting a time range shows the call tree for it. On the command line, export the `time-profile` table and summarise it with `scripts/time-profile` (busiest threads, then one thread's functions by inclusive and self samples; its docstring has the export command). This found the GPU thread spending 90% of its time in one memory lookup in the world (see [gpu.md](gpu.md)). samply (`, samply record --save-only -- target/release/ps5slop run ...`) also works when it launches the process, but attaching needs `samply setup`.

First use (2026-11-20, Minecraft loading a world, 60 s): each of the 7 world-streaming threads spent about 12 s in the interpreter and 4.5 s in translated code. The interpreter ran mostly `roundss`/`roundsd` (13M), `blendps`, `cmpps`/`cmpss`, `btr`, `pmulld` and `movlps`/`movhlps`, none of which the JIT translated then. After the fourth pass of [jit.md](jit.md) those threads spend under 0.1 s each in the interpreter in the same kind of window.

## Memory failures

The first time a memory call of one kind fails, an error is logged (target `hle::dmem`) with the request (size, address, flags) followed by the `memory` report. Kinds: direct memory allocation, direct memory map, flexible memory map, virtual range reservation and `mmap` failing, plus two that do not fail the call but point at a wrong offset somewhere: a release or a map of direct memory that is not allocated. Later ones of the same kind are only counted, in the report's `failures` line.

## Example: Minecraft running out of direct memory (2026-11-20)

vic's Minecraft stopped after some time in a world with a store to address 0 (`0xDEADC0DE`, the game's own fatal stop) on two threads at once. Headless, walking around with `press lup+rright` and friends, it reproduced in 13 minutes. The `memory` command showed direct memory filling at 1 to 2 GiB a minute until an allocation failed, just before the fault. A `calls PATH` log of the memory functions then showed the game's allocator freeing a block with `sceKernelVirtualQuery` on its mapping, a reservation over the mapping, and `sceKernelReleaseDirectMemory` at the offset the query reported, which we always reported as 0. So every free released whatever lay at offset 0 and the real block leaked. With the offset reported, direct memory grew from 1.6 GiB at spawn to 2.3 GiB after 10 minutes in the same kind of run. With releases also unmapping, a run in which a subagent walked across new terrain held direct memory at 2.5 to 2.6 GiB from minute 7 to minute 26 (mapped memory flat at 8.27 GiB, the host process's footprint 5 to 7 GB) and did not fault. See [hle-kernel.md](hle-kernel.md).

## Write watchpoints

A watch logs each write into the watched ranges with the guest instruction that made it (its "site"):

- **Translated code:** the watched pages are mapped read-only at stage 2 (`hv_vm_protect`), so a store faults out of the VM. The fault path already maps the ARM PC back to the guest RIP and runs from there in the interpreter, which makes the write through `Memory`.
- **Interpreter:** `x86::step` names the current RIP as the site while a watch is on, and `Memory::write` checks the ranges.
- **Host functions:** writes made by HLE functions (memcpy, file reads, ...) are logged with the caller's return address, marked "host call from".

Not seen: writes by the GPU (Metal writes guest memory through no-copy buffers), atomic read-modify-write instructions, and anything before the watch was set. Watching costs one relaxed atomic load per interpreted instruction and per write while off. While on, every store to a watched 16 KiB page leaves the VM, so a large watch slows the guest a lot. Watching all of direct memory (`watch 1000000000 100000000`) from start-up still reaches Minecraft's menu in about a minute.

A watch may name memory that is not mapped yet: the range is kept and protected when it is mapped.

## Example: Minecraft's striped logo (2026-11-20)

The logo and font textures (RGBA8, Standard64K tiling) showed 16-pixel stripes: every 2 KiB chunk of the texture with address bit 11 set was zero in guest memory. The texture address changes from run to run, so the hunt went:

1. `watch 1000000000 100000000` at start-up, then `hits 20 ADDR 300000` once the texture appeared. Eight 16-byte store sites in the game's own code wrote the whole texture.
2. `trace PATH SITES...` showed that four of the eight sites always stored at the 64 KiB block base instead of block base + 0x800 and so on. Their addresses come from the upper half of a 256-bit vector.
3. The instruction-level fuzzer (see [cpu.md](cpu.md)) then found the cause without looking at the game's code: VEX.256 shifts by an XMM count read the count from the low half of the count register after the low half of the destination had been written, so `vpsrlq ymm3, ymm3, xmm3` shifted the upper half by the wrong amount.

We observed only addresses and values here; we did not disassemble the game.

## Looking at a title's own code

A title's own code may be disassembled locally to learn what it expects of us; Sony's code, including Sony libraries shipped with a title, may not, and nothing from either is committed ([legal.md](legal.md)). `ps5slop-elf segment FILE INDEX OUT` writes one segment of an executable (plain ELF or fake-signed SELF) to a file; `llvm-objcopy -I binary -O elf64-x86-64 --rename-section .data=.text,alloc,load,code,readonly OUT X.o` wraps it so `llvm-objdump -d --x86-asm-syntax=intel --start-address=A --stop-address=B X.o` disassembles it (addresses are segment offsets: subtract the load base, 0x400000 for an executable, from guest addresses). The control socket's `calls PATH FILTER` gives each host call's return address, the way in. Code segments are execute-only in the guest, so `dump` cannot read them. Astro Bot's movie player layouts were found this way (see [hle-kernel.md](hle-kernel.md#libsceavplayer)).

## Walking past unimplemented imports

`PS5SLOP_UNIMPLEMENTED_ZERO=1` makes every unimplemented import return 0 and log `UNIMPLEMENTED (returning 0) LIBRARY NID NAME(args) from CALLER` instead of stopping. Names come from `PS5SLOP_NAMES` when given. A title then runs on until something actually depends on a missing result, so one run shows the next dozen imports a title needs and in what order, where stopping shows one. It is for exploring only: a 0 that should have been an output, a handle or an error can send the guest down paths it never takes on a console, so each import it passes over still needs a real (or a declared placeholder) implementation. Astro Bot's first run got from libc's start-up to its GPU set-up this way. With `PS5SLOP_UNIMPLEMENTED_ZERO=dump`, each such line is followed by the first 32 bytes behind every argument register (of six) that points at readable memory, which shows strings, output slots preset by the caller and structures without a second run.

`RUST_LOG=hle::ampr=debug` prints every APR read as it runs: the file, offset, length, bytes read and destination. With the package's own bytes (`ps5slop-pkg cat`) it shows whether a read was whole and where it went; that is how GTA V's reads were ruled out as the cause of its streamer assert.

`PS5SLOP_PROBE_RIP=ADDR[,ADDR...]` (hex guest addresses) logs every arrival of any thread at those addresses (target `jit::probe`): `ADDR` with rax, rbx, rcx, rdx, rsi, rdi, r8, r9, rsp, the low float of xmm0 and the 32 bytes rax points at. The JIT never translates code at a probe (blocks end before it, regions that contain it are not built), so each arrival runs in the interpreter; away from the probes everything runs as usual. It answers "is this function called, with what" for a title's own code without a debugger, for example which script natives GTA V's prologue calls with which arguments ([gpu.md](gpu.md#gta-vs-blast-wall-2026-11-22)). Only for a title's own code ([legal.md](legal.md)).

`PS5SLOP_HOLD_ON_FAULT=1` keeps a thread that faulted from ending the process: it prints the fault and its registers and waits, so the control socket can still read memory (`mem`, `regions`, `threads`) at the moment of the fault. Other threads keep running. `quit` ends it. Hold only as long as the inspection takes: a held guest keeps all its memory.

## Files a title asks for

`RUST_LOG=hle::vfs=debug` prints every guest path that goes through the mount table (opens, `stat`, APR resolves, ...) (target `hle::vfs`) as `PATH found`, `missing` or `outside every mount`, which shows how far a title's loading got and which files it looks for and does not find. `PS5SLOP_DEVLOG=DIR` mounts `DIR` writable as `/devlog`, where Unity players open `/devlog/app/debug.log`; Superliminal's non-development player creates the file but writes nothing to it.

## Unity (IL2CPP) titles

The C# code of a Unity title is native code in `Il2CppUserAssemblies.prx` (game code, so it may be disassembled locally), and the names are in `Media/Metadata/global-metadata.dat`. The metadata's method definitions give each method's name, declaring type and token; the running game holds, per assembly, an `Il2CppCodeGenModule` (its first field points at the assembly's file name, then the method count and the method pointer table, indexed by token minus one), which `find` in the module's data segments locates. The two together map addresses to `Type::Method`, which makes the disassembly readable. Field offsets of a class's static fields are read through the class pointer the code loads (its static field block is at +0xb8 in the build looked at). Superliminal's loading stall was found this way ([hle-kernel.md](hle-kernel.md)). The scripts stay local: they read the title's files, and nothing they print is committed.

Live objects can be read too (local scripts over the control socket, nothing committed). A class pointer (from a static field slot the code loads) has the name at +0x10, the namespace at +0x18, the field table at +0x80 (0x20-byte entries: name, type, parent class, offset) and the static block at +0xb8; `find` of the class pointer in the heap lists its instances (header word 1 zero). A `UnityEngine.Camera` wrapper holds its native camera at +0x10, where the build looked at keeps the position at +0x42c and a view matrix at +0x5c (column-major; the forward vector is minus its third row). Superliminal's player is `CharacterMotor`, `FPSInputController` and two `MouseLook` components (body yaw, camera pitch), gated by `PlayerManager.canControl` (the `GameManager` singleton's field at +0x68); reading them showed that its "frozen input" was a frozen picture ([gpu.md](gpu.md#superliminals-frozen-picture-at-the-rook-2026-11-21)).

## Superliminal to the rook corridor

A route that worked headless on 2026-11-21, with `press` through the control socket. Stick turns accelerate the longer they are held, so the same presses do not give the same angle twice: check with `shot` and correct, or steer by the camera position and heading read as above (positions are `x z` in metres).

1. Menu at about 20 to 30 s: `press cross` on NEW GAME. The intro and loading take about 50 s, until the blurred wall with two stick icons shows; any right-stick press opens the eyes.
2. First room: turn right until the small table with the Terms of Service is ahead (about `rright 1600`, `rleft 900`, `rright 550`), then `lup 1800` and `rdown 700`: the crosshair turns into a hand on the paper. Interact is R2: `press r2`.
3. `rup 700`, then turn right until the doorway that opened in the wall is ahead (about `rright 1600`, `rleft 950`) and walk through: a long corridor. The pawn room's door is in its wall at x -13.4 (a wall runs along z 26.5); walk in heading -z.
4. In the pawn room walk heading about 165 (toward the table) to the glass at z 19.1, then along it to x -13.4, where a gap leads to -13.3 17.2. The rook corridor runs from there toward +x, with the rook a few metres in.
5. The puzzle: back off a few metres, look up at the rook (crosshair on it, it is outlined), `press r2`, look down at the floor close by (`rdown 900`), `press r2` again: a small rook drops, and the corridor leads on to the white box.

Notes from a later run (2026-11-21): the eyes may need `press r3`, then `press rright 400` and `press cross 300`; close to the table, turns of 150 to 250 ms and steps of 300 to 900 ms put the crosshair on the paper; from the pawn room's door no glass wall was in sight and the gap toward the rook was not found by screenshots alone (reading the camera position, as above, is the better guide).

## Naming unknown NIDs

A NID is a hash of the symbol name, so a name can be confirmed by hashing guesses. `scripts/nid NAME...` hashes names; `scripts/nid --find NID... < names` checks a list. `scripts/nid-guess` brute-forces: it joins a prefix (`-p sceKernel`, repeatable) with up to `-n` words in CamelCase and reports candidates whose hash matches. Words come from files (`-w` one per line, `-s` splits every identifier in a file into words); with neither given, or with `--own`, it also uses the words of the sce* names `crates/hle` already implements, and with no `-p` it tries every prefix those names use. It prints the candidate count and a rough run time first: candidates grow as words to the power `-n`, so check that before waiting, and run anything past a minute in the background. It is a Rust example (`crates/loader/examples/nid-guess.rs`) run on every core: about 34 million candidates a second on 3 cores of an M-series Mac, so three words with one prefix take seconds and four words need a short custom list. Nothing it reads is stored; per [legal.md](legal.md) no NID-to-name list enters the repo. A name found this way is noted in the docs as hashed from a guess.

## Pitfalls

- Debug binaries need the hypervisor entitlement. `cargo run` signs through `scripts/hv-run.sh`; a binary started directly after `cargo build` must be signed first (`codesign -s - -f --entitlements scripts/hypervisor.entitlements target/release/ps5slop`), or the JIT is unavailable and everything runs in the interpreter.
- The guest polls the pad once a frame and frames can be slow, so a short `press` can be missed; 250 to 400 ms works for Minecraft, and much longer holds repeat.
- Runs that share the default data folder change each other's remembered menu state and saves. Give each run its own `--data DIR`.
- Kill earlier runs before reusing a log file: an older process still writes into a truncated file.
- Another process's GPU fault resets the GPU for everyone: Metal then reports our command buffers as `Discarded (victim of GPU error/recovery)` and soon `Ignored (for causing prior/excessive GPU errors)`, and the screen freezes on the last frame while the guest runs on. Grep a log for `command buffer failed` before trusting a run, especially while other agents run titles.

## GTA V to the prologue

A route that worked headless on 2026-11-21 with a fresh `--data` folder, times from start, `press` through the control socket (`press cross` holds 400 ms):

1. About 36 s: the title screen ("Continue"): `press cross`.
2. About 46 s: the language picker: `press cross`. About 55 s: Display Calibration: `press cross`.
3. About 64 s: the landing page: `press r1`, and again at about 72 s, to the STORY tab ("Resume Story"). About 78 s: `press cross`.
4. The "Entering Story Mode" loading screen runs to about 130 s, then the card "Ludendorff, North Yankton, nine years ago." and, at about 175 s, gameplay: HUD, minimap, "Go to the guard.". The camera still moves for a few seconds after control is given. A PS5 capture of the same moment (the depot lobby, from a 4K playthrough video) was the reference for the picture work of 2026-11-21.

`press lup 2000` and `press rright 600` move and turn the player (watch the minimap arrow). The times are loose: wait for each screen with `shot` rather than trusting them.

From "Go to the guard." (the gta-perf state, 2026-11-22): `press lup 3000` reaches the guard and starts a cutscene of about 40 s; then `hold l2` for 5 s and `release` (aim at the hostages), `press up 300` (phone), `press cross 300` (contacts, "Detonate" is selected), `press cross 300` (call). The blast cutscene follows; then `press lup 3000` through the double doors, `press lup 1200`, `press lleft+rleft 900` and `press lup 2200` lead down the red corridor to the closed double doors with the ATTENTION sign (the gta-blast state starts there, at about (5308.9, -5203.3)). The way on turns left before those doors: west along the side corridor (y about -5196) to the CAUTION double doors at x about 5293, north to y about -5189 through the blast area, then east into the vault room at about (5300.5, -5190.7) ([gpu.md](gpu.md#gta-vs-blast-wall-2026-11-22)). Wait for each step with `shot`; presses sent while a cutscene ends can be lost.

Once there, `save PATH` keeps the moment: `--load-state PATH` starts at "Go to the guard." about 5 s after launch instead of 3 to 5 minutes ([savestates.md](savestates.md)). On 2026-11-22 gameplay with the objective text came at about 320 s, and the location card stayed up from about 150 s to 175 s.
