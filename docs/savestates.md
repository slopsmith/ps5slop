# Save states

A save state is the whole guest written to a file while every guest thread is stopped, from which a fresh process goes on. They are a debugging tool: GTA V's prologue takes 3 to 5 minutes of loading and scripted input to reach, and starts in 5 seconds from a state. They are not save games for players. vic asked for them on 2026-11-20; they exist since 2026-11-22.

## Using them

- Save: the control socket's `save PATH` ([debugging.md](debugging.md)). The answer names the threads, the file size, the guest memory written before compression, how long stopping took and how long writing took. The game goes on afterwards.
- Load: `ps5slop run --load-state PATH [other options] GAME`. `GAME` must be the same title (and package), and the binary must give guest memory the same host functions (see "Compatibility" below). Pass the same `--data` folder the saving run used: open files, save data mounts and APR files are opened again by path there.
- Files are written to `PATH.partial` and renamed when complete, so a failed save never leaves a half file under `PATH`.

States hold game memory: they are local user data like a game dump, never enter the repo, and nothing from one may be committed.

## When a save can happen

A save stops the world first:

1. A flag asks every guest thread to stop, and running ones are kicked out of translated code. A thread stops ("parks") at the top of its run loop, between two host calls, where its whole state is its CPU registers. A thread inside a blocking host function (mutex, condition variable, semaphore, event flag, event queue, join, sleep, `pthread_once`, ...) is woken by the mechanism exceptions use (`exception::interrupted`): the call either restarts after the load (it returns `Outcome::Resume` with the CPU untouched, so the guest calls it again) or returns early where its contract allows (a sleep ends early). A condition wait cannot restart, since it gave up its mutex: it leaves its CPU at the internal stub `cond_relock`, which takes the mutex back after the load and returns 0 to the caller, a spurious wake-up POSIX allows.
2. A thread inside a guest callback run by a host function (`Env::call_guest`) does not park until it is back at depth 0, because part of its state is on the host stack.
3. Once all threads are parked, each GPU queue is asked for its state. Its thread answers when it is idle, waits for all Metal work (`Processor::finish`) and hands over its register state. If a queue still runs, or waits on memory a parked thread would have written, the save lets the guest go on for 300 ms and tries again, up to 8 times, then fails with a message naming the queue or the threads that did not stop.

## What is kept

The file starts with a magic number, the format version (`savestate::FORMAT`) and a header: the build that wrote it, the title id, a fingerprint of the host-function layout, the guest's wall clock and uptime. Then sections, each zstd-compressed:

- `REGN`: every mapped region: address, size, protection, name, and for regions on physical (direct) memory the physical offset, so aliases of the same direct memory map the same pages again. Trap regions are checked, not made.
- `PAGE`: one per run of non-zero pages, compressed by one worker per core: the address (or physical offset) and the bytes. Pages the host never backed (`mincore`) and all-zero pages are skipped. Physical memory is read through a temporary mapping of its shared memory object, so allocated but unmapped direct memory is kept too.
- `THRD`: each parked thread's CPU state, every thread pointer handed out, and thread names.
- `KERN`: the HLE objects: pthread mutexes (owner and count), condition variables, read-write locks, mutex attributes, keys and thread-specific values; thread records (join results, detach state), attributes and stacks; the host heap; POSIX and Sce semaphores; event flags; fibers; APR resolved files (by path) and command buffers; exception handlers and pending exceptions; direct memory allocations, mappings and protections; event queues and their registrations; pad state, AudioOut ports, AudioOut2 contexts and ports, AudioIn ports, NGS2 systems, racks and voices; VideoOut buffers and flip count; loaded modules and the libc hooks rtld keeps; libc's atexit list, environment, `strerror` strings and `rand` state; open files, stdio streams and directory descriptors (guest path, mode, position); save data mount points; and the handle counters of a few libraries (AJM batches, net pools, dialogs, the ids placeholder functions hand out, the user service's login event). Waiter counts are not kept: no thread waits at a save.
- `AGC.`: AGC's register-default blocks, shader headers, owner and resource registry and AMM block; each GPU queue's processor registers (context, SH, uconfig, index and indirect bases, instance count) and the global data share; flip packets not yet submitted; and the textures only Metal holds: render targets, depth targets and program-written volumes, with their descriptors and texels (`Backend::save_textures`). Without these GTA V came back black: its frame samples a 3D colour-grading volume the GPU wrote once at start-up, which guest memory does not hold in a form we can read.

Not kept, rebuilt after the load: JIT translations and code cache, Metal pipelines and compiled programs, textures read from guest memory, GPU memory windows. Audio already queued for the device is dropped, so sound restarts cleanly, and `PS5SLOP_AUDIO_WAV` recordings do not go on. The guest's clocks go on from the saved readings: the kernel's, VideoOut's and the pad's uptime are set back, and the guest's wall clock (`savestate::clock`, also behind `rdtsc` and GPU timestamps) runs behind the host's by the time between save and load.

## Refused

A save fails, saying why, when something in use is not kept: fonts (libSceFont, whose FreeType faces live on the host) and the PS4 command path (libSceGnmDriver). Any subsystem that grows host-side state must add it to `crates/hle/src/snapshot.rs` or to this refusal list.

## Compatibility

A state loads only in a process whose host functions sit where the saving one's did, because guest memory holds their addresses (import tables, return addresses, `atexit` entries). The header's fingerprint hashes every stub (NID and library, in order), every HLE variable address and the TLS layout; a mismatch is refused before anything is touched. Internal stubs are made in sorted name order so this holds across processes (before, their order followed a `HashMap` and changed from run to run). A rebuild that changes nothing guest-visible loads older states (only a note is printed); one that changes anything encoded bumps `FORMAT`, and older states are refused with both version numbers. Adding an HLE function that a title imports does not move its stubs (they follow the title's imports), but adding an internal function does.

## Measured (2026-11-22, M4 Pro, headless)

| Title | Where | File | Memory written (raw) | Save: stop + write | Load |
| --- | --- | --- | --- | --- | --- |
| Balatro | main menu | 57 MiB | 247 MiB | 0.05 + 0.18 s | 0.03 s |
| GTA V | title screen | 256 MiB | 957 MiB | 2.35 + 0.50 s | 2.6 s |
| GTA V | prologue, "Go to the guard." | 853 MiB | 1,866 MiB | 3.89 + 2.88 s | 4.5 to 4.7 s |

Frames come right after the load; most of GTA V's stop and load time is reading back and uploading GPU textures one at a time.

The GTA V prologue state was loaded three times into fresh processes and played for 130 s each: frames flipped throughout (about 2 a second; the rate just before the save was not measured), the depot and HUD rendered as before the save, `press lup` and `press rright` moved and turned the player, no faults, no failed command buffers, footprint about 3.4 GB. Balatro's menu went on animating and took pad input.

## Limits and unknowns

- Host files the guest wrote after the save keep their newer contents; a load sees the folder as it is.
- A thread that blocks forever inside a guest callback (depth above 0) stops saves; none seen so far.
- GPU textures in compressed formats cannot be read back and are skipped with a message. The renderer's fast-clear metadata per target (`color_meta`) and its bookkeeping sets around depth and volume textures start empty after a load; that changed nothing visible so far.
- Whether a loaded run behaves exactly like an unbroken one is not checked frame by frame. A test that saves, loads and compares frames and memory hashes with a run that never saved is still to do.
