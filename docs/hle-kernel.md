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
| libkernel | `sem_init`, `sem_destroy`, `sem_wait`, `sem_trywait`, `sem_timedwait`, `sem_post`, `sem_getvalue` | Our handle in the first word of `sem_t`; errors set errno and return -1 | Unit test |
| libkernel | `sceKernelGetDirectMemorySize`, `sceKernelAllocateDirectMemory`, `sceKernelAllocateMainDirectMemory`, `sceKernelReleaseDirectMemory`, `sceKernelMapDirectMemory` | 12 GiB of direct memory (our choice), first-fit allocation. Each map gets fresh memory: two maps of one range do not alias, and released ranges stay mapped. GPU protection bits grant CPU access. Errors are `0x80020000` + errno | Allocator unit-tested; reached by ScreenTester5 |
| libkernel | `sceKernelCreateEqueue`, `sceKernelDeleteEqueue`, `sceKernelWaitEqueue` | Events are `struct kevent`s, reported once per trigger (as with `EV_CLEAR`). Other libraries register and trigger them through `Kernel::add_event` and `Kernel::trigger_event`. A timeout returns `0x8002003C` (`ETIMEDOUT`) | Unit test of the queue logic |
| libkernel | `pthread_attr_{init,destroy,setdetachstate,setstacksize}`, `pthread_set_name_np` | `pthread_create` uses the stack size and detach state; names are dropped | Reached by ScreenTester5 |
| libkernel | `stat`, `fstat`, `mkdir` | `stat` sees only `/app0`; `fstat` works on descriptors of stdio streams. `struct stat` uses the pre-FreeBSD-12 layout shadPS4 uses for the PS4 (unverified for the PS5). `mkdir` fails (`EROFS` in `/app0`, `ENOENT` elsewhere) | |
| libkernel | `gettimeofday`, `nanosleep`, `sysconf`, `sysctl`, `sysctlbyname` | `sysconf` knows the page size (16 KiB), CPU count (8, our choice), clock ticks and open files. `sysctl` knows no names yet and fails with `ENOENT` | |
| libkernel | `sigaction`, `signal`, `sigemptyset`, `sigaddset`, `pthread_sigmask`, `pthread_setcanceltype` | Accepted, never acted on: signals are not delivered and threads are not cancelled | |
| libkernel | `pthread_{get,set}schedparam`, `sched_get_priority_{min,max}` | Priorities 256 (most urgent) to 767, default 700, from shadPS4 (PS4). Host threads keep their own priority | |
| libkernel | `dlopen`, `dlsym`, `dlclose`, `dlerror` | Load nothing: `dlopen` returns null | |
| libkernel | `sceKernelSendNotificationRequest` | Prints the message on stderr | |
| libSceLibcInternal | `mem*`, `str*`, `ctype` functions, `wcs*` (16-bit `wchar_t`), `mbrtowc`, `wcrtomb`, `wcstombs` (UTF-8) | C standard semantics; FreeBSD's choices where it leaves room | Unit tests for the parsers. Orbis's locale behaviour is unknown |
| libSceLibcInternal | `sprintf`, `snprintf`, `vsnprintf`, `fprintf`, `vsscanf` | Our own printf and scanf engines. Variadic arguments come from registers and the stack, or a SysV `va_list`. `%ls` and `%lc` take 16-bit characters | Unit tests. Float output uses Rust's formatting, so the last digit of long expansions may differ from Orbis |
| libSceLibcInternal | `sin`, `cos`, `pow`, `atan2`, ... (the `<math.h>` functions ScreenTester5 imports, double and float) | The host's libm | May differ from Orbis in the last bit |
| libSceLibcInternal | `fopen`, `fclose`, `fread`, `fwrite`, `fgets`, `fseeko`, `ftello`, `fflush`, `ferror`, `fileno`, `__stdinp`, `__stdoutp`, `__stderrp` | `/app0/...` opens files in the title folder, read-only (writes fail with `EROFS`); other paths fail with `ENOENT`. stdout and stderr go to the host's; stdin is always at end of file. A FILE is an opaque handle, with the stream on the host | Path mapping unit-tested |
| libSceLibcInternal | `_init_env`, `atexit`, `exit`, `_Exit`, `abort` | `_init_env` ignores the process parameters (layout unknown). `exit` runs the `atexit` handlers latest first, as guest calls | Reached by ScreenTester5 |
| libSceLibcInternal | `qsort`, `bsearch`, `strto{l,ll,ul,ull,d}`, `atoi`, `atof`, `getenv`, `setenv`, `getcwd`, `strerror`, `time`, `rand`, `srand` | `qsort` is a heapsort calling the guest comparator (not stable). `getcwd` is `/app0`. The environment starts empty. `rand` is our own generator, so sequences differ from a console's | Number parsers unit-tested |
| libSceUserService | `sceUserServiceInitialize`, `GetForegroundUser`, `GetLoginUserIdList`, `GetUserName` | One signed-in user, id 1, named `ps5slop` (our choices) | Reached by ScreenTester5 |
| libSceSystemService | `sceSystemServiceHideSplashScreen`, `sceSystemServiceLaunchWebBrowser` | Succeed; the browser call prints the address on stderr | Reached by ScreenTester5 |
| libSceAudioOut | `sceAudioOutInit`, `Open`, `Output`, `Close` | Samples are dropped. `Output` waits until the previous buffer would have finished playing, so audio threads keep a console's pace | |
| libScePad | `scePadInit`, `Open`, `GetHandle`, `Close`, `ReadState`, `SetVibration`, `SetVibrationMode`, `SetLightBar` | One controller for user 1. `ReadState` reports what the host set with `Kernel::set_pad_input` (the window feeds it from a controller and the keyboard), in OpenOrbis's `OrbisPadData` layout (the PS4's; unverified for the PS5). Vibration and the light bar are ignored | Layout unit-tested |
| libSceVideoOut | `sceVideoOutOpen`, `Close`, `SetFlipRate`, `SetBufferAttribute2`, `RegisterBuffers2`, `SubmitFlip`, `WaitVblank`, `AddFlipEvent`, `DeleteFlipEvent` | One port; flips take effect at once; a 60 Hz vblank clock; flip rate ignored. Details in [gpu.md](gpu.md) | ScreenTester5 draws its menu |
| libSceSysmodule | `sceSysmoduleLoadModule`, `sceSysmoduleLoadModuleInternal` | Succeed without doing anything: every library is provided by the HLE layer | Called by vic's homebrew |

**Variables.** Some imports are data, not functions: libkernel's `__stack_chk_guard` is read through its GOT entry, although OpenOrbis's stub declares it a function. So the symbol type cannot be trusted; the HLE layer knows its variables by name and binds them to readable memory in a data region next to the trap region.

**Finding names for NIDs.** When a program stops at an unimplemented NID, we hash candidate names ourselves (function names from public API headers, such as the OpenOrbis headers) until one matches. That table stays local; we do not commit or import NID-to-name databases.

## How a process starts

The entry point gets what shadPS4's `RunMainEntry` passes, which OpenOrbis's startup code (`_start` in its `crt1.o`) relies on:

- `rdi`: the entry parameter block `{ i32 argc; u32 pad; char *argv[]; }`, with `argv` ending in a null pointer. We pass one argument, `eboot.bin`.
- `rsi`: a loader teardown callback. The PS5 homebrew template's `_start(process_parameters, loader_teardown)` registers it with `atexit`, so it is called during `exit` and must return. Ours is the internal `ps5slop::loader_teardown`, which does nothing. What the real loader does in it is unknown.
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

## What the first PS5 homebrew needs

ScreenTester5 (see [test-targets.md](test-targets.md)) loads and relocates (2,097 relocations), with 208 imports. Names found by hashing function names from public headers (199 of the 208):

- **libSceLibcInternal (105):** the C library: memory and string functions, the `printf` and `scanf` families, math (`sin`, `pow`, `atan2`, ...), stdio files (`fopen`, `fread`, `fseeko`, ...), `qsort`, `bsearch`, `strtod`, `getenv`, `atexit`, `exit`.
- **libkernel (67):** pthreads (mostly done), semaphores, direct memory (`sceKernelAllocateDirectMemory`, `sceKernelMapDirectMemory`, ...), event queues (`sceKernelCreateEqueue`, `sceKernelWaitEqueue`), signals, `stat`/`fstat`/`mkdir`, `dlopen`, `sysconf`, notifications.
- **libSceVideoOut (9):** open, flips and flip events, vblank wait, flip rate.
- **libScePad (8):** open, read state, light bar, vibration.
- **libSceUserService, libSceSystemService, libSceAudioOut, libSceKeyboard, libSceImeDialog, libScePosixForWebKit (17).**

It does not import from the `libc.prx` it ships.

Status on 2026-11-19: the C library is in place (one libc import, NID `Fm-dmyywH9Q`, still has no name and no host function). ScreenTester5 runs through `_init_env` and its start-up code. With the libkernel functions above it gets through its set-up, including starting threads, through the system services and VideoOut, and draws its menu (see [gpu.md](gpu.md)). It runs slowly: a full 1080p redraw takes about 2 s in the interpreter. One libkernel import (NID `FJmglmTMdr4`) still has no name.

## What a real executable needs next

Libraries beyond libkernel (system services, pads, video out), the process parameters in `PT_SCE_PROCPARAM`, module start functions and init arrays, thread attributes, key destructors, and file and socket I/O.
