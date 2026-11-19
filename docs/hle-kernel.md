# HLE: the kernel interface and system libraries

How guest code reaches host code, and what we implement. `crates/hle`.

## How a call reaches the host

- Every import binds, at load time, to its own 16-byte slot in a **trap region** at `0xFF_0000_0000` (our choice, see [memory.md](memory.md)). The region cannot be read, written or executed.
- When guest code jumps there, the interpreter stops with `Exit::Trap(address)` before decoding anything. The slot names a `(library, NID)` pair; if a host function implements it, it runs with the SysV AMD64 arguments (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`), its result goes to `rax`, and the return address is popped from the stack, as a `ret` would.
- Imports with no host function still get a slot. Calling one stops the process with "unimplemented *library* function with NID *x*" and the caller's address, instead of a jump to address 0. `ps5slop run` reports how many imports are unimplemented before it starts.
- The entry point gets a return address too: the internal `ps5slop::return_from_entry` slot, which exits with `rax` as the status.
- A raw `syscall` instruction stops the process with an error. We serve the kernel through libkernel's functions, not its system call numbers.

A JIT can keep the same scheme: a trap region is just memory it never maps executable.

## Implemented

| Library | Function | Behaviour | Status |
| --- | --- | --- | --- |
| libkernel | `write(fd, buf, len)` | fd 1 and 2 go to the host's stdout and stderr; other fds return -1; writes over 1 MiB are cut short | Our own test programs only. Whether libkernel exports `write` under this name, and its error convention (-1 and errno, or an `SCE_KERNEL_ERROR` code), is unverified |
| libkernel | `exit(status)` | Ends the process | Same |

## What a real executable needs next

vic's OpenOrbis-built `eboot.bin` (PS4 homebrew) runs its first instructions and then faults at `0x400028`, reading through `rdi`, which we leave at 0. So at least OpenOrbis's startup code expects a pointer in `rdi` at the entry point, presumably to a block of process parameters (argc, argv, ...). What that block holds on Orbis is not known to us yet. Candidates for finding out without reading Sony code: public documentation, the startup code of open-source toolchains (as a description of the interface, not as code to copy), and a test program on vic's PS4 that prints what it receives.

Other pieces still missing before a real program can start: thread-local storage (the FS base and the `PT_TLS` image), the process parameters in `PT_SCE_PROCPARAM`, module start functions and init arrays, and the hundred-odd libkernel functions a homebrew program imports.
