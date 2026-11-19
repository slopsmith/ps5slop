# HLE: the kernel interface and system libraries

How guest code reaches host code, and what we implement. `crates/hle`.

## How a call reaches the host

- Every import binds, at load time, to its own 16-byte slot in a **trap region** at `0xFF_0000_0000` (our choice, see [memory.md](memory.md)). The region cannot be read, written or executed.
- When guest code jumps there, the interpreter stops with `Exit::Trap(address)` before decoding anything. The slot names a `(library, NID)` pair; if a host function implements it, it runs with the SysV AMD64 arguments (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`), its result goes to `rax`, and the return address is popped from the stack, as a `ret` would.
- Imports with no host function still get a slot. Calling one stops the process with "unimplemented *library* function with NID *x*" and the caller's address, instead of a jump to address 0. `ps5slop run` reports how many imports are unimplemented before it starts.
- A raw `syscall` instruction stops the process with an error. We serve the kernel through libkernel's functions, not its system call numbers.

A JIT can keep the same scheme: a trap region is just memory it never maps executable.

## Threads

Each guest thread runs the interpreter on its own host thread (see [decisions.md](decisions.md)). `hle::Kernel` holds what they share: the stub table, the kernel objects, the thread records, the guest's stdout and stderr, and the process's exit state. A thread checks for process exit every 65,536 instructions, and a host function that blocks (a mutex, a join, a condition variable) wakes every 20 ms to check too. The process ends at `exit`, at the first guest error in any thread, or when its last thread ends.

A host function can also redirect the CPU instead of returning (`Outcome::Resume`): `pthread_once` uses that to call the guest's init routine with a return address in our trap region, so the host takes over again when the routine returns.

## errno

Each thread's errno lives in its thread control block, at the thread pointer plus `0x800` (our choice); `__error()` returns its address. POSIX-style functions that fail set it and return -1. The values are FreeBSD's (`hle::errno`), assumed rather than verified for Orbis. The pthread functions return error numbers directly, as POSIX says.

## Implemented

| Library | Functions | Behaviour | Status |
| --- | --- | --- | --- |
| libkernel | `write`, `_write` | fd 1 and 2 go to the host's stdout and stderr; other fds return -1; writes over 1 MiB are cut short | vic's homebrew logs through `_write`. Error convention unverified |
| libkernel | `exit` | Ends the process | |
| libkernel | `__error` | The calling thread's errno address | |
| libkernel | `ioctl` | Always fails with `ENOTTY` | vic's homebrew asks for a terminal size and carries on |
| libkernel | `mmap`, `munmap` | Anonymous mappings only, FreeBSD flag values (`MAP_FIXED` 0x10, `MAP_ANON` 0x1000); without `MAP_FIXED` from `0x10_0000_0000` (shadPS4's PS4 user area); failures return `MAP_FAILED` without errno; `munmap` only whole mappings | `mmap` called by vic's homebrew |
| libkernel | `pthread_mutex_*`, `pthread_mutexattr_{init,destroy,settype}` | Error-checking by default (FreeBSD's default), recursive and normal types | Unit tests and the `threads` test program |
| libkernel | `pthread_cond_{init,destroy,wait,timedwait,signal,broadcast}` | `signal` wakes all waiters (allowed: POSIX permits spurious wake-ups); `timedwait` takes a CLOCK_REALTIME deadline | `threads` test program |
| libkernel | `pthread_rwlock_{init,destroy,rdlock,wrlock,unlock}` | | Untested by a program |
| libkernel | `pthread_key_{create,delete}`, `pthread_{get,set}specific` | Destructors are not run | Unit test |
| libkernel | `pthread_once` | The routine runs as a guest call; other callers wait | `threads` test program |
| libkernel | `pthread_create`, `join`, `detach`, `exit`, `self`, `equal` | `pthread_t` is the thread pointer; attributes are ignored and every thread gets a 1 MiB stack (the Orbis default is unknown) | `threads` test program, 30 runs out of 30 |
| libkernel | `__stack_chk_fail`, `__stack_chk_guard` (variable) | Stops the process / a fixed non-zero canary in the HLE data region (`0xFF_8000_0000`) | `__stack_chk_guard` used by vic's homebrew |
| libSceSysmodule | `sceSysmoduleLoadModule`, `sceSysmoduleLoadModuleInternal` | Succeed without doing anything: every library is provided by the HLE layer | Called by vic's homebrew |

**Variables.** Some imports are data, not functions: libkernel's `__stack_chk_guard` is read through its GOT entry, although OpenOrbis's stub declares it a function. So the symbol type cannot be trusted; the HLE layer knows its variables by name and binds them to readable memory in a data region next to the trap region.

**Finding names for NIDs.** When a program stops at an unimplemented NID, we hash candidate names ourselves (function names from public API headers, such as the OpenOrbis headers) until one matches. That table stays local; we do not commit or import NID-to-name databases.

## How a process starts

The entry point gets what shadPS4's `RunMainEntry` passes, which OpenOrbis's startup code (`_start` in its `crt1.o`) relies on:

- `rdi`: the entry parameter block `{ i32 argc; u32 pad; char *argv[]; }`, with `argv` ending in a null pointer. We pass one argument, `eboot.bin`.
- `rsi`: an exit function, `exit(status)`; ours is the internal `ps5slop::program_exit` slot.
- `rsp`: 8 bytes off a 16-byte boundary, with the parameter block's first two words (argc and `argv[0]`) copied to the top of the stack.
- FS base: the main thread's thread pointer. Static TLS uses the x86-64 "variant II" layout: FS points at a word that holds its own address, and the executable's `PT_TLS` image sits just below it.

Status: matches both sources and gets vic's OpenOrbis `eboot.bin` through its startup code into libc, where it stops at the first SSE instruction (`movd`). What the real Orbis kernel puts on the stack beyond this (environment, auxiliary vector) is unknown; shadPS4 notes that it pushes more.

## How far vic's OpenOrbis build gets

As of 2026-11-19, `ps5slop run` on `ps4build/pkg_scratch` (vic's PS4 package store, built with OpenOrbis on his own `orbitengine`) runs through `_start`, libc start-up, its first `mmap` and a failed `ioctl(1, TIOCGWINSZ)`, into `main`, and prints its own log:

```
INFO: ../subprojects/orbitengine/src/setup.cpp:15: Setting exception handler
INFO: ../subprojects/orbitengine/src/setup.cpp:31: Running platform setup
INFO: ../subprojects/orbitengine/platforms/orbis/src/setup.cpp:14: Loading all needed modules
```

It then loads system modules (which succeeds) and stops at `sceCommonDialogInitialize` in libSceCommonDialog. From here it needs system services (common dialogs, user service, pads) and PS4 graphics through `libScePigletv2VSH` (the PS4's OpenGL ES), which is outside our PS5 GPU plan.

## What a real executable needs next

Libraries beyond libkernel (system services, pads, video out), the process parameters in `PT_SCE_PROCPARAM`, module start functions and init arrays, thread attributes, key destructors, and file and socket I/O.
