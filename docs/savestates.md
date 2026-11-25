# Save states

A save state is the whole guest written to a file while every guest thread is stopped, from which a fresh process goes on. They are a debugging tool for getting back to a hard-to-reach moment in seconds, not save games for players. Code: `crates/savestate` (encoding, wall clock), `crates/hle/src/snapshot.rs` (what is kept), the GPU side in `crates/gpu`.

## Using them

- Save: the control socket's `save PATH` ([debugging.md](debugging.md)). The answer names the threads, the file size, the memory written and how long stopping and writing took. The game goes on afterwards.
- Load: `ps5slop run --load-state PATH [options] GAME`. `GAME` must be the same title and package, and the binary must give guest memory the same host functions (see "Compatibility"). Pass the same `--data` folder: open files, save data mounts and APR files are opened again by path there.
- Files are written to `PATH.partial` and renamed when complete, so a failed save never leaves a half file.
- States hold game memory. They are local user data like a game dump: they never enter the repo, and nothing from one may be committed.

## How threads stop

Guest threads are host threads, and a thread blocked in a host function has state on the host stack that cannot be written down. So a save parks every thread at a point where its CPU registers are its whole state:

1. A flag asks every thread to stop, and running ones are kicked out of translated code. A thread parks at the top of its run loop, between two host calls.
2. A thread blocked in a host function is woken the way exceptions wake it ([hle-kernel.md](hle-kernel.md)): the call returns with the CPU untouched and restarts after the load, or returns early where its contract allows (a sleep).
3. A condition wait cannot restart, because it has released its mutex, and retaking the mutex before parking can deadlock against a parked owner. It parks at the internal stub `cond_relock`, which takes the mutex back after the load and returns 0, a spurious wake-up POSIX allows.
4. A thread inside a guest callback run by a host function (`Env::call_guest`) parks only once back at depth 0.
5. Once all threads are parked, each GPU queue hands over its state when idle, after waiting for all Metal work. If a queue still runs, or waits on memory a parked thread would write, the guest goes on for 300 ms and the save tries again, up to 8 times, then fails naming the queue or threads. Each failed try is printed (`save state: not stopped yet: ...`).

## Format

Our own small binary encoding (`crates/savestate`: little-endian fields in order, a `record!` macro for structs), so no dependency beyond zstd and every field of the private HLE structures stays visible in its own module. The file is a magic number, the format version (`savestate::FORMAT`) and a header (writing build, title id, host-function fingerprint, the guest's wall clock and uptime), then zstd-compressed sections:

- `REGN`: every mapped region with protection, name and, for direct memory, its physical offset, so aliases map the same pages again. Trap regions are checked, not made.
- `PAGE`: one record per run of non-zero pages, compressed in parallel. Pages the host never backed and all-zero pages are skipped. Physical memory is read through its shared memory object, so allocated but unmapped direct memory is kept too.
- `THRD`: each parked thread's CPU state, thread pointers and names.
- `KERN`: the HLE objects (synchronisation objects, thread records, heaps, files, mounts, audio ports and NGS2 voices, pad and VideoOut state, modules, libc state, handle counters). `snapshot.rs` is the list.
- `AGC.`: AGC's register defaults, shader and resource registries, each GPU queue's registers and the global data share, pending flips, and the textures only Metal holds (render targets, depth targets, program-written volumes) with their texels. GPU-written textures are needed because a frame can sample something the GPU wrote once at start-up, which guest memory does not hold in a form we can read; writing them back to guest memory would need tiling encoders we do not have.

Not kept, rebuilt after a load: JIT translations, Metal pipelines and compiled programs, textures read from guest memory, GPU memory windows. Audio already queued for the device is dropped, and WAVE recordings do not go on.

**Clocks.** Every wall-clock read the guest can make (`clock_gettime` realtime, `gettimeofday`, `time`, `rdtsc`, `sceKernelReadTsc`, RTC ticks, GPU timestamps) goes through `savestate::clock`, which a load sets to go on from the saved time, so a state loaded hours later does not see hours pass in one frame. Uptime clocks (kernel, VideoOut, pad) are set back the same way.

## Refused

A save fails, saying why, when something in use is not kept: fonts (libSceFont's FreeType faces live on the host) and the PS4 command path (libSceGnmDriver). **Any subsystem that grows host-side state must add it to `snapshot.rs` or to this refusal list.**

## Compatibility

Guest memory holds host-function addresses (import tables, return addresses, `atexit` entries), so a state loads only in a process whose stubs sit where the saving one's did. The header's fingerprint hashes every stub (NID and library, in order), every HLE variable address and the TLS layout; a mismatch is refused before anything is touched. Internal stubs are made in sorted name order for this reason.

- A rebuild that changes nothing guest-visible loads older states (a note is printed). This is the point of a fingerprint rather than a build id: states stay usable across debugging rebuilds.
- A change to anything encoded bumps `FORMAT`; older states are refused with both versions named.
- Adding an HLE function a title imports does not move its stubs (they follow the title's imports); adding an internal function does.

## Limits and unknowns

- Host files the guest wrote after the save keep their newer contents; a load sees the folder as it is.
- A thread that blocks forever inside a guest callback stops saves.
- GPU textures in compressed formats cannot be read back and are skipped with a message. The renderer's fast-clear metadata and some bookkeeping around depth and volume textures start empty after a load.
- Not kept: Ajm decoder instances and AudioOut2 ambisonics channels (set again by titles each grain).
- Most of a large save's stop and load time is reading back and uploading GPU textures one at a time.
- Whether a loaded run behaves exactly like an unbroken one is not checked frame by frame. A test that saves, loads and compares frames and memory hashes with a run that never saved is still to do.
