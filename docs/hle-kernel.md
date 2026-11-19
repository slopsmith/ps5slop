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

## How a process starts

The entry point gets what shadPS4's `RunMainEntry` passes, which OpenOrbis's startup code (`_start` in its `crt1.o`) relies on:

- `rdi`: the entry parameter block `{ i32 argc; u32 pad; char *argv[]; }`, with `argv` ending in a null pointer. We pass one argument, `eboot.bin`.
- `rsi`: an exit function, `exit(status)`; ours is the internal `ps5slop::program_exit` slot.
- `rsp`: 8 bytes off a 16-byte boundary, with the parameter block's first two words (argc and `argv[0]`) copied to the top of the stack.
- FS base: the main thread's thread pointer. Static TLS uses the x86-64 "variant II" layout: FS points at a word that holds its own address, and the executable's `PT_TLS` image sits just below it.

Status: matches both sources and gets vic's OpenOrbis `eboot.bin` through its startup code into libc, where it stops at the first SSE instruction (`movd`). What the real Orbis kernel puts on the stack beyond this (environment, auxiliary vector) is unknown; shadPS4 notes that it pushes more.

## What a real executable needs next

SSE, then the process parameters in `PT_SCE_PROCPARAM`, module start functions and init arrays, and the hundred-odd libkernel functions a homebrew program imports.
