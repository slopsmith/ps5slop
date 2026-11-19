# HLE: the kernel interface and system libraries

How guest code reaches host code, and what we implement. `crates/hle`.

## How a call reaches the host

- Every import binds, at load time, to its own 16-byte slot in a **trap region** at `0xFF_0000_0000` (our choice, see [memory.md](memory.md)). The region cannot be read, written or executed.
- When guest code jumps there, the interpreter stops with `Exit::Trap(address)` before decoding anything. The slot names a `(library, NID)` pair; if a host function implements it, it runs with the SysV AMD64 arguments (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`), its result goes to `rax`, and the return address is popped from the stack, as a `ret` would.
- Imports with no host function still get a slot. Calling one stops the process with "unimplemented *library* function with NID *x*" and the caller's address, instead of a jump to address 0. `ps5slop run` reports how many imports are unimplemented before it starts.
- A raw `syscall` instruction stops the process with an error. We serve the kernel through libkernel's functions, not its system call numbers.

A JIT can keep the same scheme: a trap region is just memory it never maps executable.

## Implemented

| Library | Function | Behaviour | Status |
| --- | --- | --- | --- |
| libkernel | `write(fd, buf, len)` | fd 1 and 2 go to the host's stdout and stderr; other fds return -1; writes over 1 MiB are cut short | Our own test programs only. Whether libkernel exports `write` under this name, and its error convention (-1 and errno, or an `SCE_KERNEL_ERROR` code), is unverified |
| libkernel | `exit(status)` | Ends the process | Same |
| libkernel | `mmap(addr, len, prot, flags, fd, off)` | Anonymous mappings only, FreeBSD flag values (`MAP_FIXED` 0x10, `MAP_ANON` 0x1000); without `MAP_FIXED` they go to the lowest free range from `0x10_0000_0000` (shadPS4's PS4 user area); failures return `MAP_FAILED` without errno | Called by vic's OpenOrbis build with these flags |
| libkernel | `munmap(addr, len)` | Whole mappings only | Untested by a real program yet |
| libkernel | `__stack_chk_fail()` | Stops the process: "stack smashing detected" | |
| libkernel | `__stack_chk_guard` (variable) | A fixed non-zero canary in the HLE data region (`0xFF_8000_0000`) | Used by vic's OpenOrbis build |

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

As of 2026-11-19, `ps5slop run` on `ps4build/pkg_scratch` (vic's PS4 package store, built with OpenOrbis) runs through `_start`, the libc's start-up code, the stack canary, the x87 and SSE set-up, and its first `mmap`, and stops at its first `ioctl(1, 0x5413, ...)`. `0x5413` is Linux's `TIOCGWINSZ`, so OpenOrbis's libc (musl) apparently asks a terminal-size question with Linux's numbering; what libkernel answers on a console is unknown (presumably an error, since there is no terminal).

## What a real executable needs next

Through libc start-up the program imports about 60 libkernel functions: pthreads (mutexes, condition variables, keys, `pthread_once`, `pthread_create`), files (`open`, `read`, `fstat`, `lseek`, `mkdir`), sockets, signals, time (`nanosleep`, `sceKernelGetProcessTimeCounter`), `sysconf`, `ioctl`. Two pieces of design come before most of them:

- **errno:** libkernel's `__error()` returns a per-thread pointer; failing functions must set it.
- **Calls back into guest code:** `pthread_once` and `pthread_create` run guest functions. The HLE layer needs a way to call a guest function and continue when it returns (a return trap that resumes the host function), and threads need more than one CPU state.

Then the process parameters in `PT_SCE_PROCPARAM`, module start functions and init arrays.
