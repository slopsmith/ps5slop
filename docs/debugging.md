# Debugging a running guest

The tools, and when to reach for which. Environment variables are listed in [environment.md](environment.md); this doc says what they are for.

| Question | Tool |
| --- | --- |
| What went wrong in this run? | The problem report, then the log |
| Which import is missing, and what does the title call next? | The stop message; `PS5SLOP_UNIMPLEMENTED_ZERO` to see several at once |
| What is the guest doing right now? | Control socket: `threads`, `mem`, `regions`, `calls` |
| Who wrote this memory? | Write watchpoints (CPU), `PS5SLOP_GPU_WATCH` (GPU), `PS5SLOP_CHECK_CODE` (code pages) |
| Is this function of the title called, with what? | `PS5SLOP_PROBE_RIP` |
| Which files does the title look for? | `RUST_LOG=hle::vfs=debug` |
| Why does this frame look wrong? | Frame capture and `ps5slop-replay`, `gpusnap`, MSL overrides |
| Where does the time go? | `profile`, Instruments traces, `ps5slop-replay time` and `bench` |
| How do I get back to this moment? | Save states ([savestates.md](savestates.md)) |

## Logging

Diagnostics go through `tracing` to stderr as `LEVEL target: message`. The default shows errors (work that was dropped: skipped draws, failed pipelines, guest faults), warnings, the main crate's milestones, and output a switch or socket command asked for (traces, probes, watches: asking is the filter). `RUST_LOG` takes `EnvFilter` syntax and adds to the default: `RUST_LOG=gpu=debug`, `RUST_LOG=hle::stub=off`.

A target is the module path of the code that logs, so a prefix covers a whole area: `hle::dmem` (direct memory), `hle::vfs` (every guest path: found, missing or outside every mount), `hle::ampr` (every APR read), `hle::pad`, `hle::ngs2`, `hle::videoout`, `gpu::metal`, `jit::engine`. A few targets are named by purpose instead:

- `hle::stub`: placeholder calls, and imports passed over by `PS5SLOP_UNIMPLEMENTED_ZERO`.
- `ps5slop` at debug: the imports a title uses that we do not implement, by name.
- `ps5slop::report`: the problem report at exit.
- `gpu::stats`: per-queue and Metal counters each second (at debug, packets by opcode); waits by cause; gaps between flips over 1 s.
- `gpu::sync`: every submission, wait, release, command-processor write, REWIND and dispatch, per queue, timestamped. Grep it for a label address to see who sets it and who waits.
- `gpu::compile`, `gpu::msl`: translations, compile jobs and draws skipped for want of a pipeline; the MSL of failed compiles (at trace, all MSL).
- `gpu::targets`, `gpu::dma`: render targets created and images first read; every `DMA_DATA`.
- `gpu::trace`, `gpu::snap`, `gpu::probe`, `gpu::watch`, `jit::probe`: output of the matching command or switch.
- `jit::stats`: interpreter use, VM exits and host calls, at exit. Turning it on makes the JIT count, which costs time.

Reports that would repeat every frame are rate-limited (`diag::FirstN`, `diag::Distinct`).

## Problem report

Every run counts its problems (`diag::problems`) and prints a report at exit when there are any: skipped draws and dispatches, untranslated programs with the failing instruction, failed Metal compiles with Metal's reason, unsupported formats and tile modes, memory window failures, loop-guard trips, wait timeouts, unimplemented and placeholder HLE calls, JIT interpreter fallbacks by opcode, late audio buffers. Grouped by reason, sorted by count and per-frame rate. `PS5SLOP_REPORT=PATH` also writes it to a file; the socket's `problems` prints it live. Start here: structured counts find gaps faster than screenshots.

## The control socket

Every `ps5slop run` listens on a Unix socket, `ps5slop-PID.sock` in the temporary folder or `--control PATH`, printed at start. macOS limits socket paths to 104 bytes, so a deep folder does not work. One command per line; each answer ends with a line holding a single `.` (86slopper's protocol). `scripts/ps5slop-ctl SOCKET COMMAND...` sends commands; `socat - UNIX-CONNECT:SOCKET` works too. `help` lists every command with its arguments; numbers are hex.

- **Memory:** `regions`, `memory` (direct memory, regions by name, host footprint, failure counts), `mem`, `dump`, `poke`, `find`, `sym` (module and offset, or the HLE stub).
- **Threads and calls:** `threads` (name, host calls, last host function and caller), `calls PATH [FILTER]` (every host call matching FILTER, or made from an address range `@LO-HI`, with arguments, result and caller; flushed at exit).
- **Watches:** `watch`, `hits`, `trace`, `unwatch` (below).
- **Input:** `press BUTTONS [MS]`, `tap`, `hold`, `release`. Buttons join with `+`; `lup`, `rright` and so on push sticks to the edge, `lx=N` and `l2=N` set an axis or trigger part way. The guest reads the pad once a frame and frames can be slow, so presses default to 400 ms.
- **Screen:** `shot PATH`, `frames DIR N` (the next N flips, for flicker), `targets DIR` (every target and sampled texture).
- **GPU:** `gputrace N`, `gpusnap DIR [FIRST]`, `gpucapture [PATH]`, `metalcapture [PATH]`, `gpuskip [ADDR...]`, `gpureload`.
- **Run:** `profile`, `rates`, `problems`, `save PATH`, `quit`.

A guest fault prints the return addresses along the frame-pointer chain (module and offset). `PS5SLOP_HOLD_ON_FAULT` keeps a faulting thread waiting instead of ending the process, so the socket can still read memory at the moment of the fault; a held guest keeps all its memory, so do not leave it.

## Write watchpoints

`watch ADDR LEN` logs each write into the range with the guest instruction that made it (its "site"):

- Translated code: watched pages are read-only at stage 2, so a store faults out of the VM and runs in the interpreter, which logs it.
- Interpreter: `Memory::write` checks the ranges with the current RIP as the site.
- Host functions: writes are logged with the caller's return address ("host call from"). The JIT runs `memset`, `memcpy` and `memmove` natively; `PS5SLOP_JIT_NO_NATIVES` keeps them as host calls so watches and `calls` see them.

A watch may name memory not mapped yet. Not seen: GPU writes, atomic read-modify-writes, anything before the watch. Every store to a watched 16 KiB page leaves the VM, so large watches are slow, though watching all of direct memory from start-up stays usable.

The technique for a wrong value whose address changes between runs: watch a large range from start-up, `hits N ADDR LEN` once the data appears to find the store sites, then `trace PATH SITES...` to see each site's addresses and values. If a site computes a wrong address, the instruction fuzzer ([cpu.md](cpu.md)) can find a CPU bug without reading the title's code.

For writes nobody should make, `PS5SLOP_CHECK_CODE=MS` compares code and read-only data with their loaded contents and prints each changed page: any change is a host or GPU write gone astray.

## Missing imports and unknown names

- `PS5SLOP_UNIMPLEMENTED_ZERO=1` makes unimplemented imports return 0 and log `UNIMPLEMENTED (returning 0) LIBRARY NID NAME(args) from CALLER`, so one run shows the next dozen imports a title needs instead of one. `=dump` adds the 32 bytes behind each pointer argument (strings, preset output slots). For exploring only: a 0 that should have been a handle or an error sends the guest down paths it never takes on a console, so every import it passes over still needs an implementation or a declared placeholder.
- **Naming NIDs.** `scripts/nid NAME...` hashes names; `scripts/nid --find NID... < names` checks a list. `scripts/nid-guess` brute-forces: a prefix (`-p sceKernel`) joined with up to `-n` CamelCase words from files (`-w`, `-s`) or from the names `crates/hle` already has. It prints the candidate count and a rough time first; candidates grow as words to the power `-n`, so check before waiting. `scripts/names` writes a local name list from SDK headers for `PS5SLOP_NAMES`. Nothing these read is stored, and no NID-to-name list enters the repo ([legal.md](legal.md)). Note a name found this way as hashed from a guess.

## Looking at a title's own code

A title's own code may be disassembled locally to learn what it expects of us. Sony's code, including Sony libraries shipped with a title, may not, and nothing from either is committed ([legal.md](legal.md)).

- `ps5slop-elf segment FILE INDEX OUT` writes one segment of an executable; `llvm-objcopy -I binary -O elf64-x86-64 --rename-section .data=.text,alloc,load,code,readonly OUT X.o` wraps it so `llvm-objdump -d --x86-asm-syntax=intel` can read it (subtract the load base from guest addresses). Code segments are execute-only in the guest, so `dump` cannot read them.
- The way in is a host call's return address, from `calls` or `threads`. A `calls` log filtered to one module's address range (`@LO-HI`) shows which host call a module stopped after, without reading the module.
- `PS5SLOP_PROBE_RIP=ADDR,...` logs every arrival at those addresses with the main registers and the bytes `rax` points at. The JIT never translates code at a probe, so each arrival runs in the interpreter. It answers "is this called, with what" without a debugger.
- **Unity (IL2CPP).** C# code is native code in `Il2CppUserAssemblies.prx`; names are in `Media/Metadata/global-metadata.dat`. The metadata's method definitions give name, type and token; the running game holds per assembly an `Il2CppCodeGenModule` (first field: the assembly's file name; then the method pointer table, indexed by token minus one), which `find` locates. Together they map addresses to `Type::Method`. Live objects: a class pointer has its name at +0x10, namespace at +0x18, field table at +0x80 and static block at +0xb8 (in the build looked at). Scripts for this stay local.
- When a title waits forever on its own logic, check what loaded (`hle::vfs`), whether loading threads are idle (`profile`), then find the condition in the title's code: usually a placeholder whose failure the title treats as fatal.
- With a package's own bytes (`ps5slop-pkg cat`), `hle::ampr` shows whether a read was whole and where it went.

## GPU frame capture and replay

A capture holds one frame: the PM4 streams of every queue, the queues' initial registers, GPU-only textures, the guest memory pages the frame read, and the picture the game showed. `ps5slop-replay` runs it through the command processor and Metal renderer without the game, waiting for every pipeline so results are deterministic.

- Capture: **F9** in the window, or `gpucapture [PATH]`. Files go to `~/Library/Application Support/ps5slop/captures/` (or `PS5SLOP_CAPTURE_DIR`), and the path is logged. Requests made during a capture queue up.
- **F10** or `metalcapture` writes an Xcode `.gputrace` (needs `METAL_CAPTURE_ENABLED=1`). `ps5slop-replay dump ... --gputrace` makes a smaller, repeatable one from a capture.
- Captures hold game data; `.gitignore` and `scripts/check-forbidden` keep them out of commits.

```sh
R="cargo run --release -p gpu --bin ps5slop-replay --"
$R render CAPTURE.ps5frame final.png                  # the final picture
$R list CAPTURE.ps5frame                              # every draw and dispatch
$R render CAPTURE.ps5frame after-42.png --draw 42     # a target after draw 42 (--target HEX picks one)
$R dump CAPTURE.ps5frame OUT --draw 42                # every target after draw 42, plus shown.png
$R stats CAPTURE.ps5frame --target HEX                # min, max, mean per channel
$R history CAPTURE.ps5frame --target HEX X Y          # one texel after every draw that changes it
$R live CAPTURE.ps5frame live.png --diff diff.png     # what the game showed, against the replay
$R mem CAPTURE.ps5frame ADDR 128                      # captured memory, as hex and floats
$R check CAPTURE.ps5frame golden.png                  # compare with a golden (--update writes it)
$R time CAPTURE.ps5frame --warmup --runs 5            # GPU time per pass (--draws: per draw)
$R bench CAPTURE.ps5frame --runs 5 --warmup 2         # CPU cost of processing and encoding
```

`--skip HEX` (any subcommand) drops draws and dispatches of one program. `ps5slop-replay --help` has the rest.

Reading the results:

- Targets start with what the game drew before the capture, so `render --draw N` often shows the finished frame even for small N. `history` answers which draws write a pixel.
- `live` differences mean state only the running game had (caches, windows carried over the whole run). Effects that read earlier frames can differ a little even when both are right.
- `mem` shows only captured pages; memory the frame did not read is zeros.
- `time --draws` gives each draw its own render pass, so small draws look far dearer than they are; trust it for draws of a millisecond or more. Single runs vary by up to a fifth: compare variants with `--runs 5`, interleaved, by pass where possible. The first run pays one-time costs.
- `bench` resets memory and registers before each run so caches are warm, as in a running game.

`cargo test -p gpu --test frame_replay` replays a synthetic capture, plus any local `*.ps5frame` and golden pairs in `PS5SLOP_FRAME_CAPTURES`.

## Looking inside a frame

- `gputrace N` (or `PS5SLOP_GPU_TRACE`) traces every draw and dispatch: programs, user data, guest-memory windows, bound textures, scissors. Add `PS5SLOP_TRACE_REGS` for changed registers, `PS5SLOP_TRACE_PIXELS` for samples each draw passed (0 means it covered nothing), `PS5SLOP_TRACE_USER_MEMORY` for the bytes behind pointer-like user data (ring buffers are gone by the time the socket could read them).
- `gpusnap DIR [FIRST]` copies every attachment at the end of each render pass of the next frame: the frame's composition pass by pass. `PS5SLOP_SNAP_INPUTS` adds the textures each pixel program samples, `PS5SLOP_SNAP_RAW` the exact texels.
- `gpuskip ADDR...` drops draws and dispatches of a program; comparing screenshots with and without finds the pass behind a bug. A skipped compute program may leave garbage arguments for a later indirect draw.
- **MSL overrides.** `PS5SLOP_MSL_OVERRIDE=DIR` replaces a program's MSL with `DIR/STAGE-ADDR.metal` when it exists; `gpureload` picks up edits in a running game. Start from the translation (`cargo run -p gfx10 --example captures` on a `PS5SLOP_SHADER_DUMP` capture), insert an early return of an intermediate value as a colour, and move it through the program to find the first wrong value. Values in HDR targets go through tonemapping, so scale probes down. Check that a probe compiles first: errors past the first few are not printed.
- **Draws that cover nothing.** With pixel counts on, override the vertex program to change one thing and count again: depth forced to the near plane tells occlusion from geometry; a small jitter by vertex index tells collapsed triangles from culled ones; moving each vertex to a screen cell by a condition code (zero weights, `w <= 0`, ...) shows which conditions hold. An early constant colour in the pixel program shows whether fragments are discarded; giving each `discard_fragment()` its own colour names the one taken.
- `PS5SLOP_PROBE_FLAT` writes each pixel program's flat input as raw bytes into colour output 1, to see which integer arrived.
- **Flicker.** `frames DIR 60` and the mean brightness per picture show which flips are bad and whether they have a period. Trace good and bad frames together and compare; if traces match, the difference is in data, so compare `gpusnap`s pass by pass. Whether a run flickers can depend on where allocations land: judge a fix over several runs.
- **Feedback loops.** A frame that reads the previous one (exposure, bloom, history) can stay broken after its cause is gone, and a save state keeps the broken targets. `gpuskip` on the draw that reads last frame's picture for a few seconds, then without it, tells a loop that recovers from one that diverges again.
- **A screen that never changes** while traces look right: check whether the traced frame is the one shown, and compare `bound` textures with what a probe reads.
- **GPU-side writes.** `PS5SLOP_GPU_WATCH=ADDR[+LEN][@SECONDS]` waits after every draw and dispatch and reports those that change the range (slow; start late). With `PS5SLOP_GPU_WATCH_DUMP` each new content goes to a file, to check a chain of compute passes against a CPU oracle. With `gpu::sync` (also timestamped) it shows when work became visible to the GPU, when it ran and when the CPU wrote the same memory. `PS5SLOP_GPU_PROBE=ADDR` logs each dispatch of one compute program.
- `cargo run -p gfx10 --example raw -- FILE [OFFSET]` disassembles raw GFX10 code from a title's own executable (local analysis only).

## GPU hangs

A command buffer that runs too long fails with `kIOGPUCommandBufferCallbackErrorHang`; after that macOS ignores every command buffer of the process and the screen freezes. A hang can outlive the process: the process sticks in state `E`, `kill -9` does nothing, and only a reboot helps. Translated loops give up on their own ([shaders.md](shaders.md)), but:

- Before any GPU work, check `ps -axo pid,stat,etime,command | awk '$2 ~ /E/'` for a stuck ps5slop and start nothing while one exists. Give every run a hard time limit and kill it by PID. Stop all GPU work after a single hang.
- To find a program that does not end: `PS5SLOP_GPU_COMMIT_EVERY=1` with `PS5SLOP_GPU_LABELS` names the first unfinished command buffer; `PS5SLOP_GPU_SYNC_DISPATCHES` makes the last `dispatch` line in `gpu::sync` name it; `PS5SLOP_GPU_ENCODER_STATUS` prints faulted encoders. Loop-guard trips are reported with the program and user data.
- Another process's GPU fault resets the GPU for everyone: our command buffers then fail as "victim of GPU error/recovery". Grep a log for `command buffer failed` before trusting a run.

Most other `PS5SLOP_GPU_*` switches bring back an older way of scheduling work, for comparing ([environment.md](environment.md)).

## Profiling

- `profile on` / `profile [N]` / `profile off` over the socket: per guest thread the time in translated code, the interpreter, translating and host functions; hottest guest code by sample (module and offset); host functions by time; interpreted instructions; VM exits by cause; and a GPU thread line (encoding, waiting for Metal, GPU busy time, compile threads and waits for compiles). If profiling is on at exit (also on ⌘Q), the profile is printed. It reports addresses only, never code bytes: hot code may be Sony's.
- `rates` gives guest instructions per second since the last call. Total MIPS mean little when workers spin; look at the threads that matter.
- `RUST_LOG=jit::stats=info` at exit lists what the interpreter ran, which is how the next instructions to translate are chosen ([jit.md](jit.md)).
- **Host CPU:** `xcrun xctrace record --template 'Time Profiler' --attach PID --time-limit 15s --output X.trace`, then `scripts/time-profile` on the exported table. Attaching needs a copy of the binary signed with `scripts/hypervisor.entitlements` plus `com.apple.security.get-task-allow`. `sample PID SECONDS` gives quick stacks; samply works when it launches the process.
- **GPU:** a Metal System Trace (`--template 'Metal System Trace'`) of the same signed copy with `PS5SLOP_GPU_LABELS=1`, which names encoders after their targets and programs.
- **Shader costs:** record `ps5slop-replay time` under the Metal System Trace with `--instrument 'Metal GPU Counters'` (`--launch`, no signing needed), then `scripts/shader-profile X.trace`. Shader-profiler time overlaps other work and is not what a shader adds to the frame; confirm with `time` before acting. Apple's compiled shader code cannot be read.
- **Compiles:** Metal caches compiled programs across processes, so a second run of a build compiles almost nothing. `PS5SLOP_MSL_UNIQUE` measures a cold first run. `PS5SLOP_COMPILE_WAIT` makes draws wait for pipelines, to rule out skipped draws. With a `PS5SLOP_SHADER_DUMP`, `cargo run --release -p gfx10 --example compile_times -- DIR` times Metal's compiler offline.

## Pitfalls

- A binary started directly after `cargo build` must be signed with the hypervisor entitlement (`codesign -s - -f --entitlements scripts/hypervisor.entitlements target/release/ps5slop`), or there is no JIT and everything runs in the interpreter. `cargo run` signs through `scripts/hv-run.sh`.
- Give each run its own `--data DIR`: runs sharing a data folder change each other's menu state and saves.
- Kill earlier runs before reusing a log file: an older process still writes into a truncated file.
- Stick turns in games accelerate the longer they are held, so the same presses do not give the same angle twice. Check with `shot`, or save a state at the moment you need.
- Agents run games `--headless`, and do not play their way to a bug: use a save state vic made, counters and captures.
