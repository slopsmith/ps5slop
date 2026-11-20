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
| `press BUTTONS [MS]` | Holds pad buttons (`cross`, `up+cross`, ...) for MS milliseconds (default 150) |
| `shot PATH` | Writes the screen as PPM; with `press`, this drives menus step by step |
| `quit` | Ends the guest |

Thread names come from `scePthreadCreate`, `scePthreadRename` and `pthread_set_name_np`. Host threads are named `guest 0xTCB`.

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

## Pitfalls

- Debug binaries need the hypervisor entitlement. `cargo run` signs through `scripts/hv-run.sh`; a binary started directly after `cargo build` must be signed first (`codesign -s - -f --entitlements scripts/hypervisor.entitlements target/release/ps5slop`), or the JIT is unavailable and everything runs in the interpreter.
- The guest polls the pad once a frame and frames can be slow, so a short `press` can be missed; 250 to 400 ms works for Minecraft, and much longer holds repeat.
- Runs that share the default data folder change each other's remembered menu state and saves. Give each run its own `--data DIR`.
- Kill earlier runs before reusing a log file: an older process still writes into a truncated file.
