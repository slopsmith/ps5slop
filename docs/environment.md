# Environment variables

Every environment variable ps5slop reads; this is the one list, and other docs link here. The `PS5SLOP_*` ones are read once at start into `diag::Config` (`crates/diag/src/config.rs`). A variable shown without a value is on when set to anything, including `0`.

## Everyday

| Variable | What it does |
| --- | --- |
| `RUST_LOG=FILTER` | Adds log output on top of the default (errors, warnings, milestones), in `tracing-subscriber`'s `EnvFilter` syntax: `RUST_LOG=gpu=debug`, `RUST_LOG=hle::stub=off`. Targets are described in [debugging.md](debugging.md) |
| `PS5SLOP_REPORT=PATH` | Also writes the end-of-run problem report (skipped draws, placeholder calls, JIT fallbacks) to `PATH` |
| `PS5SLOP_CAPTURE_DIR=DIR` | Where F9/F10 captures and unnamed `gpucapture`/`metalcapture` files go (default `~/Library/Application Support/ps5slop/captures`) |
| `METAL_CAPTURE_ENABLED=1` | Needed for F10 (Metal `.gputrace` captures). Read by Metal, not by ps5slop |
| `PS5SLOP_AUDIO=0\|1` | `1` opens the Mac's audio device even with `--headless`, `0` never opens it |
| `PS5SLOP_AUDIO_WAV=PATH` | Records everything the game plays as WAVE files ([audio.md](audio.md)) |
| `PS5SLOP_COMPILE_THREADS=N` | Shader compile threads (default one per performance core) |
| `PS5SLOP_INPUT=SECONDS:BUTTONS,...` | Scripted pad input: each step holds the buttons (joined by `+`, names as for the control socket's `press`) from that many seconds after start; an empty list lets go |

## Running and HLE

| Variable | What it does |
| --- | --- |
| `PS5SLOP_CHECK_CODE=MS` | Compares code and read-only data with their contents at start every MS milliseconds (100 if not a number) and logs each changed 16 KiB page |
| `PS5SLOP_CALLS=PATH[:FILTER]` | Writes the `calls` log from the start of the run |
| `PS5SLOP_DEVLOG=DIR` | Mounts `DIR` writable as `/devlog` |
| `PS5SLOP_DUMP_SERIES=N`, `PS5SLOP_DUMP_EVERY_MS=MS` | `--dump-frame` writes N frames (`OUT.0.ppm`, ...) MS milliseconds apart (default 1 frame, 500 ms) |
| `PS5SLOP_DUMP_TARGETS` | Records render targets and textures for `targets` from the start; with `--dump-frame`, also writes every frame texture as `OUT.ADDR.ppm` |
| `PS5SLOP_HOLD_ON_FAULT` | A thread that faults waits instead of ending the process, so the control socket can still read memory |
| `PS5SLOP_NAMES=FILE` | Names (one per line) hashed to label NIDs in reports |
| `PS5SLOP_UNIMPLEMENTED_ZERO[=dump]` | Unimplemented imports return 0 and log instead of stopping; `dump` adds the bytes behind each argument |

## JIT

| Variable | What it does |
| --- | --- |
| `PS5SLOP_JIT_VERIFY` | Runs every translated block in the interpreter too and logs differences |
| `PS5SLOP_JIT_TRACE` | Sends every block exit to the host and keeps the last 64 blocks per thread for fault reports ([jit.md](jit.md)) |
| `PS5SLOP_JIT_LAZY_SYNC` | Memory map changes reach running vCPUs at their next entry instead of at once |
| `PS5SLOP_JIT_NO_NATIVES` | Keeps `memset`, `memcpy` and `memmove` as host calls, so `calls` and watches see them |
| `PS5SLOP_JIT_SEGMENTS=N` | Uses N code cache segments (at least 2), to test eviction |
| `PS5SLOP_JIT_TIER2=N\|eager\|off` | Compiles a block's tier-2 region after N runs, at its first run, or never (default) |
| `PS5SLOP_JIT_STATS_TOP=N` | Rows in each table of the `jit::stats` report (20) |
| `PS5SLOP_PROBE_RIP=ADDR[,ADDR...]` | Interprets and logs every arrival at these guest addresses (target `jit::probe`) |

## GPU tracing and inspection

| Variable | What it does |
| --- | --- |
| `PS5SLOP_GPU_RENDER=0` | Parses command buffers without rendering them |
| `PS5SLOP_GPU_TRACE=N\|FIRST-LAST` | Traces every draw and dispatch of those frames (target `gpu::trace`), as `gputrace` does |
| `PS5SLOP_TRACE_REGS` | Traced draws also list the context and uconfig registers changed since the previous traced draw |
| `PS5SLOP_TRACE_PIXELS` | Traced draws count the samples they pass, with Metal's visibility results |
| `PS5SLOP_TRACE_USER_MEMORY` | Traced draws and dispatches dump 256 bytes behind each pointer-like pair of user data words |
| `PS5SLOP_GPU_PROBE=ADDR` | Logs each dispatch of the compute program at ADDR (target `gpu::probe`) |
| `PS5SLOP_GPU_WATCH=ADDR[+LEN][@SECONDS]` | Waits for the GPU after every draw and dispatch (from SECONDS after the first) and logs those that change the LEN bytes at ADDR (default 8; hex with `0x`) (target `gpu::watch`) |
| `PS5SLOP_GPU_WATCH_DUMP=DIR` | With `PS5SLOP_GPU_WATCH`, writes each new content of the watched range to `DIR/NNNNN.bin` |
| `PS5SLOP_SNAP_INPUTS`, `PS5SLOP_SNAP_RAW` | `gpusnap` also copies every texture a pixel program samples, and writes raw texels next to each PPM |
| `PS5SLOP_SHADER_DUMP=DIR` | Writes every translation request as a capture for the `captures` and `compile_times` examples |
| `PS5SLOP_MSL_OVERRIDE=DIR` | Replaces a program's MSL by `DIR/STAGE-ADDR.metal` when that file exists |
| `PS5SLOP_MSL_UNIQUE` | Makes every program's source unique, so Metal's shader cache cannot answer (measures a first run) |
| `PS5SLOP_FLOAT_EXACT=PARTS` | Translates parts of the float semantics to follow RDNA2 exactly, as the hardware oracle checks them, instead of as plain Metal arithmetic (the default: on a GTA V frame exact semantics drew the same pixels at about 4x the GPU time): a comma-separated list of `nan` (NaN results, quieting, min/max NaN rules), `denorm` (denormal flushing and keeping, compares), `round` (round modes; f16 arithmetic computed in f32 and rounded once), `mods` (output modifier and clamp rules), or `all` |
| `PS5SLOP_PROBE_FLAT` | Pixel programs with a flat input write its raw value to colour output 1 |
| `PS5SLOP_GPU_LABELS` | Labels Metal encoders and command buffers for GPU traces and reports |
| `PS5SLOP_GPU_ENCODER_STATUS` | Failed command buffers report which encoders faulted |
| `PS5SLOP_COMPILE_WAIT` | Draws wait for their pipelines instead of being skipped |

## GPU scheduling

These change when work reaches Metal. Most bring back an older way of doing something, for comparing when a bug may come from scheduling ([gpu.md](gpu.md)).

| Variable | What it does |
| --- | --- |
| `PS5SLOP_GPU_COMMIT_DRAWS=N` | Commits the command buffer when a render pass ends after N draws, so the GPU starts while encoding goes on (default 0: only at sync points). Committing early can change a few pixels ([gpu.md](gpu.md)) |
| `PS5SLOP_GPU_COMMIT_EVERY=N` | Commits the command buffer every N dispatches |
| `PS5SLOP_GPU_SYNC_DISPATCHES` | Waits for Metal after every dispatch |
| `PS5SLOP_GPU_FENCE` | Orders encoders that read guest memory after those that wrote it |
| `PS5SLOP_GPU_RESIDENCY_SET` | Keeps guest memory windows resident through a Metal residency set |
| `PS5SLOP_GPU_INLINE_UPLOADS` | Texture uploads go into the main command buffer instead of their own |
| `PS5SLOP_GPU_SYNC_RELEASES` | Releases and memory writes wait for the GPU instead of being queued |
| `PS5SLOP_GPU_QUEUED_WRITES` | `WRITE_DATA` of compute queues stays queued until the queue drains instead of landing in GPU order |
| `PS5SLOP_GPU_WAIT_SUBMISSIONS` | Each submission waits for its Metal work before its releases take effect |
| `PS5SLOP_GPU_HTILE_CPU` | HTILE words programs may write are read on the CPU after a wait |
| `PS5SLOP_GPU_FAST_CLEAR_CPU` | CMASK and DCC fast-clear words programs may write are read on the CPU after a wait |
| `PS5SLOP_GPU_CPU_INDIRECT` | Indirect dispatches read their sizes on the CPU after a wait |
| `PS5SLOP_GPU_NO_THROTTLE` | Submitters may run ahead of the GPU thread |

## Tests

| Variable | What it does |
| --- | --- |
| `PS5SLOP_FRAME_CAPTURES=DIR` | Folder of local `*.ps5frame` + `*.golden.png`/`.ppm` pairs checked by `cargo test -p gpu --test frame_replay` (default `captures/`) |
| `GFX10_CASES=N` | Random encodings per instruction format in `cargo test -p gfx10 --test llvm` (default 300) |
| `GFX10_SHOW=N`, `GFX10_SHOW_FALLBACK` | How many differing cases that test prints per format (default 6), and whether it also prints LLVM's fallback texts |
| `PS5SLOP_TEST_PKG=PKG`, `PS5SLOP_TEST_DIR=DIR` | The ignored test `package_files_match_their_extraction` (`crates/ps5slop`) reads every file of `PKG` through the guest file system and compares it with the extraction in `DIR` |

## Launcher app

Read by the Swift launcher ([launcher.md](launcher.md)), not by `ps5slop`.

| Variable | What it does |
| --- | --- |
| `PS5SLOP_BIN=PATH` | The `ps5slop` binary to run when the launcher is not inside `ps5slop.app` (default: `target/release/ps5slop` in the current folder or a parent) |

## Building

Read when ps5slop is compiled, not when it runs.

| Variable | What it does |
| --- | --- |
| `SLOP_COMMIT=SHA` | The commit `ps5slop --version` and the logs name (default: `git rev-parse HEAD` plus `-dev`). Set by `scripts/bundle-app` and the nightly workflow ([launcher.md](launcher.md)) |

## Adding a variable

New switches go into `diag::Config`, not into a `std::env::var` call at the use site, and get a row here.
