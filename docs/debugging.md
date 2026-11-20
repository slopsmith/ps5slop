# Debugging a running guest

## The control socket

Every `ps5slop run` listens on a Unix socket: `ps5slop-PID.sock` in the temporary folder, or the path given with `--control PATH`. The path is printed on stderr at start. macOS limits socket paths to 104 bytes, so a deep scratch folder does not work; the default does.

The protocol is the one 86slopper uses: one command per line, and each answer ends with a line holding a single `.`. `scripts/ps5slop-ctl SOCKET COMMAND...` sends commands and prints the answers; `socat - UNIX-CONNECT:SOCKET` works too. `help` lists the commands. Numbers are hex, with or without `0x`.

| Command | What it does |
| --- | --- |
| `regions` | Mapped regions with protection and name |
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
| `calls PATH [FILTER]` / `calls off` | Appends every host call whose description contains FILTER to a file: thread, name, first six arguments, result, caller |
| `gputrace N` | Traces every draw and dispatch of the next N frames on stderr, as `PS5SLOP_GPU_TRACE` does (N decimal) |
| `targets DIR` | Writes every render target, depth target and sampled texture as `ADDR.ppm` (the low hex digit says which: 1 texture, 2 colour target, 4 depth); needs `PS5SLOP_DUMP_TARGETS=1` |
| `press BUTTONS [MS]` | Holds pad buttons (`cross`, `up+cross`, ...) for MS milliseconds (default 400: games read the pad once per frame, and our frames can be slow) |
| `profile on` / `profile off` / `profile [N]` | Starts a fresh profile window, stops profiling, or reports the window so far (top N rows): per guest thread the time in translated code, the interpreter, translating and host functions, the time in none of these (exits, locks, the engine), the CPU time used and the guest instructions per second of translated-code time; memory map changes; the hottest guest code by sample (module and offset); host functions by time and calls; the instructions and addresses the interpreter ran most; VM exits by cause (host call, interpreter, new block, lookup table collision, fault, kick), each with its hottest guest RIPs or host functions; translations and evicted cache segments. Samples at a VM entry point are counted as "kicked before entering": the sampler's kick was pending while the thread was outside the VM |
| `shot PATH` | Writes the screen as PPM; with `press`, this drives menus step by step |
| `quit` | Ends the guest |

Thread names come from `scePthreadCreate`, `scePthreadRename` and `pthread_set_name_np`. Host threads are named `guest 0xTCB`.

A guest fault also prints the return addresses along the frame-pointer chain (module and offset), so the code that called into a failing routine shows without reading any code. The `calls` log is flushed when the guest exits, so its last lines are the calls just before the end.

## Profiling

`profile on` starts a window; `profile` reports it (see the table above). Times come from the engine: around each VM entry, each interpreter run and each translation, and around each host call in the process. Samples come from the code cache sampler, which kicks running vCPUs 100 times a second (see [jit.md](jit.md)); while profiling, each sample's PC is mapped back to a guest RIP. Off, it costs one relaxed load per VM exit and per host call. It reports addresses and module offsets only, never code bytes: the hot code may be Sony's own modules, which we do not disassemble ([legal.md](legal.md)).

The report also has a GPU thread line: time encoding draws and dispatches, waiting for Metal, the GPU's own busy time per command buffer, flip pacing, memory waits and idle time, and the scheduling and wake-up latencies around each command buffer. For the GPU itself, record a Metal System Trace: copy the binary, sign the copy with `scripts/hypervisor.entitlements` plus `com.apple.security.get-task-allow`, run it with `PS5SLOP_GPU_LABELS=1` (encoders get labels naming their targets, programs and detiled images), then `xcrun xctrace record --template 'Metal System Trace' --attach PID --time-limit 3s --output X.trace` and export the `metal-gpu-intervals` table (its values are deduplicated through `id`/`ref` attributes). macOS's `sample PID SECONDS` gives host stacks per thread.

`PS5SLOP_TRACE_USER_MEMORY=1` adds, to a `gputrace`, the first 64 bytes behind every pointer-like pair of user data words of each draw's vertex program and each dispatch (attribute tables, vertex buffer descriptors, constant buffers), read at encoding time; ring-buffer contents are gone by the time the control socket could read them.

`PS5SLOP_PROBE_FLAT=1` makes every pixel program with a flat input write that input's raw value into colour output 1 (low byte red, next byte green, third byte blue) instead of what it computes, so a target dump shows per pixel which integer arrived (used to check Minecraft's terrain texture-set index).

For where host time goes at a given moment, record a Time Profiler trace of the running process with the same signed copy: `xcrun xctrace record --template 'Time Profiler' --attach PID --time-limit 15s --output X.trace`. The trace opens in Instruments with a timeline per thread, and selecting a time range shows the call tree for it. On the command line, export the `time-profile` table and summarise it with `scripts/time-profile` (busiest threads, then one thread's functions by inclusive and self samples; its docstring has the export command). This found the GPU thread spending 90% of its time in one memory lookup in the world (see [gpu.md](gpu.md)). samply (`, samply record --save-only -- target/release/ps5slop run ...`) also works when it launches the process, but attaching needs `samply setup`.

First use (2026-11-20, Minecraft loading a world, 60 s): each of the 7 world-streaming threads spent about 12 s in the interpreter and 4.5 s in translated code. The interpreter ran mostly `roundss`/`roundsd` (13M), `blendps`, `cmpps`/`cmpss`, `btr`, `pmulld` and `movlps`/`movhlps`, none of which the JIT translated then. After the fourth pass of [jit.md](jit.md) those threads spend under 0.1 s each in the interpreter in the same kind of window.

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

## Walking past unimplemented imports

`PS5SLOP_UNIMPLEMENTED_ZERO=1` makes every unimplemented import return 0 and log `UNIMPLEMENTED (returning 0) LIBRARY NID NAME(args) from CALLER` instead of stopping. Names come from `PS5SLOP_NAMES` when given. A title then runs on until something actually depends on a missing result, so one run shows the next dozen imports a title needs and in what order, where stopping shows one. It is for exploring only: a 0 that should have been an output, a handle or an error can send the guest down paths it never takes on a console, so each import it passes over still needs a real (or a declared placeholder) implementation. Astro Bot's first run got from libc's start-up to its GPU set-up this way.

## Naming unknown NIDs

A NID is a hash of the symbol name, so a name can be confirmed by hashing guesses. `scripts/nid NAME...` hashes names; `scripts/nid --find NID... < names` checks a list. `scripts/nid-guess` brute-forces: it joins a prefix (`-p sceKernel`, repeatable) with up to `-n` words in CamelCase and reports candidates whose hash matches. Words come from files (`-w` one per line, `-s` splits every identifier in a file into words) and, unless `--no-own` is given, from the sce* names `crates/hle` already implements; with no `-p` it tries every prefix those names use. It is a Rust example (`crates/loader/examples/nid-guess.rs`) run on every core: about 34 million candidates a second on 3 cores of an M-series Mac, so three words with one prefix take seconds and four words need a short custom list. Nothing it reads is stored; per [legal.md](legal.md) no NID-to-name list enters the repo. A name found this way is noted in the docs as hashed from a guess.

## Pitfalls

- Debug binaries need the hypervisor entitlement. `cargo run` signs through `scripts/hv-run.sh`; a binary started directly after `cargo build` must be signed first (`codesign -s - -f --entitlements scripts/hypervisor.entitlements target/release/ps5slop`), or the JIT is unavailable and everything runs in the interpreter.
- The guest polls the pad once a frame and frames can be slow, so a short `press` can be missed; 250 to 400 ms works for Minecraft, and much longer holds repeat.
- Runs that share the default data folder change each other's remembered menu state and saves. Give each run its own `--data DIR`.
- Kill earlier runs before reusing a log file: an older process still writes into a truncated file.
