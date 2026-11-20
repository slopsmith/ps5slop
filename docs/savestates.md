# Save states (plan, not started)

vic asked for save states like 86slopper's (2026-11-20). This is the plan; nothing is implemented yet.

## Why it is harder than in 86slopper

86slopper emulates a whole PC: its state is the CPU, RAM and devices, saved between runs of the JIT on one thread. Here much of the guest's state lives in our own host code:

- Guest threads run on host threads, and at any moment many are blocked inside host functions (condition variables, event queues, sleeps). A save needs every thread at a known point, and a load must put each back, including the ones that were waiting.
- Kernel objects (mutexes, condition variables, semaphores, event queues and flags, files and directories, save data mounts, TLS blocks, thread-specific values) are Rust objects with handles the guest holds.
- The GPU state is in Metal: render targets, depth targets, images written by programs, pipelines. Guest memory does not hold rendered pixels. Command buffers may be in flight.
- Audio (NGS2 voices, AudioOut ports), pad and video out state are smaller but host-side too.
- Guest memory is a sparse 1 TiB reservation with gigabytes mapped.

## Plan

1. **Stop the world.** A way to park every guest thread at a host-call boundary or a VM exit, with its CPU state stored, and to resume them. Threads blocked in a host wait count as parked once their wait is recorded as data (what they wait on, with what timeout), so a load can restart the wait. The control socket's `pause`/`resume` and the debugger would use this too.
2. **Memory.** Write every mapped region (address, size, protection, name, contents), compressed. Restore by mapping and filling.
3. **Kernel objects.** One serialiser per HLE module, behind a common trait, with stable handles. Files are saved as paths and offsets and reopened.
4. **GPU.** Drain the command queue, read back every Metal texture that has no copy in guest memory (targets, depth targets, written images), and upload them on load. Caches (translated programs, pipelines) are rebuilt, not saved.
5. **Audio, pad, video out.** Small serialisers; audio restarts cleanly rather than resuming mid-buffer.
6. **Check.** A test that saves, loads and runs on, and compares frames (and guest memory hashes) with a run that never saved. Balatro first, since it plays end to end.

The JIT's code cache is not saved: translations are rebuilt on load. Save files hold user game data, so they live in the data folder and never in the repo.
