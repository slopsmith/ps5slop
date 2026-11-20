# HLE: the kernel interface and system libraries

How guest code reaches host code, and what we implement. `crates/hle`.

## How a call reaches the host

- Every import binds, at load time, to its own 16-byte slot in a **trap region** at `0xFF_0000_0000` (our choice, see [memory.md](memory.md)). The region cannot be read, written or executed.
- When guest code jumps there, the interpreter stops with `Exit::Trap(address)` before decoding anything. The slot names a `(library, NID)` pair; if a host function implements it, it runs with the SysV AMD64 arguments (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`), its result goes to `rax`, and the return address is popped from the stack, as a `ret` would.
- Imports with no host function still get a slot. Calling one stops the process with "unimplemented *library* function with NID *x*" and the caller's address, instead of a jump to address 0. `ps5slop run` reports how many imports are unimplemented before it starts, and lists them (library and NID) when the environment variable `PS5SLOP_LIST_UNIMPLEMENTED` is set.
- A raw `syscall` instruction stops the process with an error. We serve the kernel through libkernel's functions, not its system call numbers.

A JIT can keep the same scheme: a trap region is just memory it never maps executable.

## Threads

Each guest thread runs the interpreter on its own host thread (see [decisions.md](decisions.md)). `hle::Kernel` holds what they share: the stub table, the kernel objects, the thread records, the guest's stdout and stderr, and the process's exit state. A thread checks for process exit every 65,536 instructions, and a host function that blocks (a mutex, a join, a condition variable) wakes every 20 ms to check too. The process ends at `exit`, at the first guest error in any thread, or when its last thread ends.

A host function can also redirect the CPU instead of returning (`Outcome::Resume`): `pthread_once` uses that to call the guest's init routine with a return address in our trap region, so the host takes over again when the routine returns.

## errno

Each thread's errno lives in its thread control block, at the thread pointer plus `0x800` (our choice); `__error()` returns its address. The values of thread-specific keys 0 to 127 live at the thread pointer plus `0x400` (our choice, so the JIT's native `pthread_getspecific` reads them without leaving the VM); higher keys stay in a host table. POSIX-style functions that fail set it and return -1. The values are FreeBSD's (`hle::errno`), assumed rather than verified for Orbis. The pthread functions return error numbers directly, as POSIX says.

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
| libkernel | `pthread_rwlock_{init,destroy,rdlock,wrlock,tryrdlock,trywrlock,unlock}`, `pthread_condattr_{init,destroy}` | Condition variable attributes are ignored | Untested by a program |
| libkernel | `pthread_key_{create,delete}`, `pthread_{get,set}specific` | Destructors are not run. Keys below 128 keep their values in the thread control block (see errno above) | Unit test; JIT test of the native version |
| libkernel | `pthread_once` | The routine runs as a guest call; other callers wait | `threads` test program |
| libkernel | `pthread_create`, `join`, `detach`, `exit`, `self`, `equal` | `pthread_t` is the thread pointer; attributes are ignored and every thread gets a 1 MiB stack (the Orbis default is unknown) | `threads` test program, 30 runs out of 30 |
| libkernel | `__stack_chk_fail`, `__stack_chk_guard` (variable) | Stops the process / a fixed non-zero canary in the HLE data region (`0xFF_8000_0000`) | `__stack_chk_guard` used by vic's homebrew |
| libkernel | `sem_init`, `sem_destroy`, `sem_wait`, `sem_trywait`, `sem_timedwait`, `sem_post`, `sem_getvalue` | Our handle in the first word of `sem_t`; errors set errno and return -1 | Unit test |
| libkernel | `sceKernelGetDirectMemorySize`, `sceKernelAllocateDirectMemory`, `sceKernelAllocateMainDirectMemory`, `sceKernelReleaseDirectMemory`, `sceKernelMapDirectMemory` | 12 GiB of direct memory (our choice), first-fit allocation. Each map gets fresh memory: two maps of one range do not alias, and released ranges stay mapped. GPU protection bits grant CPU access. Errors are `0x80020000` + errno | Allocator unit-tested; reached by ScreenTester5 |
| libkernel | `sceKernelCreateEqueue`, `sceKernelDeleteEqueue`, `sceKernelWaitEqueue` | Events are `struct kevent`s, reported once per trigger (as with `EV_CLEAR`). Other libraries register and trigger them through `Kernel::add_event` and `Kernel::trigger_event`. A timeout returns `0x8002003C` (`ETIMEDOUT`) | Unit test of the queue logic |
| libkernel | `pthread_attr_{init,destroy,setdetachstate,setstacksize}`, `pthread_set_name_np` | `pthread_create` uses the stack size and detach state; names are dropped | Reached by ScreenTester5 |
| libkernel | `stat`, `fstat`, `mkdir`, `sceKernelStat`, `sceKernelFstat` | Paths go through the guest file system (below); `fstat` works on descriptors of stdio streams and `open`. `struct stat` uses the pre-FreeBSD-12 layout shadPS4 uses for the PS4 (unverified for the PS5). `mkdir` works in `/data` and fails with `EROFS` in `/app0` | Unit test of the file system |
| libkernel | `open`, `read`, `write`, `close` (and `_`-prefixed), `sceKernelOpen`, `sceKernelRead`, `sceKernelWrite`, `sceKernelClose`, `sceKernelMkdir` | Through the guest file system: `/app0` read-only (writing fails with `EROFS`), `/data` read-write with `O_CREAT`, `O_TRUNC`, `O_APPEND`, `O_EXCL` (FreeBSD values). Descriptors share one number space with stdio's FILEs. fds 1 and 2 write to the host's stdout and stderr. The `sceKernel*` forms return `0x80020000` + errno | DriftCheck reads `/app0/build-label.txt` and writes its log to `/data` |
| libkernel | `lseek`, `unlink`, `rmdir`, `rename`, `fsync`, `ftruncate`, `truncate` and their `sceKernel*` forms; `sceKernelChmod`, `sceKernelUtimes`, `sceKernelSync` | Through the guest file system, writable mounts only. Permissions and file times are accepted and not kept | Minecraft |
| libkernel | `sceKernelGetProcessTimeCounter`, `...Frequency`, `sceKernelGettimeofday`, `sceKernelNanosleep`, `_nanosleep`, `usleep`, `sched_yield`, `getpid`, `getrusage`, `_exit`, `sceKernelConvertUtcToLocaltime` | The counter ticks in nanoseconds (our choice). `getpid` is 100, `getrusage` all zeros, the time zone UTC | Minecraft |
| libkernel | `sceKernelAddUserEvent`, `AddUserEventEdge`, `DeleteUserEvent`, `TriggerUserEvent`, `sceKernelGetEventId`, `Filter`, `Data`, `UserData` | `EVFILT_USER` events on the event queues; both kinds are reported once per trigger | |
| libkernel | `sceKernelIsTrinityMode`, `sceKernelMlock`, `__pthread_cxa_finalize`, `pthread_condattr_setclock`, `_sigprocmask`, `scePthreadAttrGetstacksize` | `IsTrinityMode` (is this a PS5 Pro?) answers no; the rest are accepted | Minecraft (it picks `"playstation_5"` over `"playstation_5_pro"` from the answer) |
| libkernel | `sceKernelDebugRaiseException` | Stops the process | |
| libkernel | `clock_gettime`, `clock_getres`, `sceKernelClockGettime` | FreeBSD clock ids: the realtime ones give wall-clock time, the monotonic, uptime and CPU-time ones count from process start. Resolution reported as 1 µs | Minecraft's libc |
| libkernel | `sceKernelCreateSema`, `DeleteSema`, `WaitSema`, `PollSema`, `SignalSema`, `CancelSema` | Counting semaphores with a maximum; `WaitSema` takes a pointer to microseconds and writes back what is left. Errors are `0x80020000` + errno (`ETIMEDOUT`, `ECANCELED`, `EBUSY` from `Poll`) | Untested by a program |
| libkernel | `sceKernelUuidCreate` | A random version-4 UUID | |
| libkernel | `gettimeofday`, `nanosleep`, `sceKernelUsleep`, `sceKernelGetProcessTime`, `sysconf`, `sysctl`, `sysctlbyname` | `sceKernelGetProcessTime` counts microseconds from when the process started (the unit is assumed from how DriftCheck uses it). `sysconf` knows the page size (16 KiB), CPU count (8, our choice), clock ticks and open files. `sysctl` knows no names yet and fails with `ENOENT` | |
| libkernel | `sigaction`, `signal`, `sigemptyset`, `sigaddset`, `pthread_sigmask`, `pthread_setcanceltype` | Accepted, never acted on: signals are not delivered and threads are not cancelled | |
| libkernel | `pthread_{get,set}schedparam`, `sched_get_priority_{min,max}` | Priorities 256 (most urgent) to 767, default 700, from shadPS4 (PS4). Host threads keep their own priority | |
| libkernel | `dlopen`, `dlsym`, `dlclose`, `dlerror` | Load nothing: `dlopen` returns null | |
| libkernel | `sceKernelSendNotificationRequest` | Prints the message on stderr | |
| libkernel | `_sceKernelSetThreadDtors`, `_sceKernelSetThreadAtexitCount`, `_sceKernelSetThreadAtexitReport`, `_sceKernelRtldSetApplicationHeapAPI`, `sceKernelGetProcParam`, the four `sceKernelGetSanitizer*Replace*` and `sceKernelIsAddressSanitizerEnabled` (in `rtld.rs`) | What Sony's libc asks the loader while it starts: callbacks are stored (not called yet), the process parameters are the executable's `PT_SCE_PROCPARAM`, no sanitizer is present | Balatro's libc starts |
| libSceLibcInternalExt | `sceLibcHeapGetTraceInfo` | Reports success without writing (layout unknown) | Balatro's libc starts |
| libkernel | `scePthread*`: mutexes, mutex attributes, condition variables (`Timedwait` takes relative microseconds), rwlocks, keys, `Once`, `Self`, `Equal`, `Create`, `Join`, `Detach`, `Exit`, `Yield`, attributes; affinity, priority, scheduling and names accepted and ignored | Wrappers of the POSIX functions with Sony's error convention, `0x80020000` + errno | Balatro's libc starts |
| libkernel | `__tls_get_addr` | Module TLS, see [loader.md](loader.md#modules) | Balatro's libc |
| libScePosix | everything libkernel implements | The same functions under this library name too | |
| libSceLibcInternal | `mem*`, `str*`, `ctype` functions, `wcs*` (16-bit `wchar_t`), `mbrtowc`, `wcrtomb`, `wcstombs` (UTF-8) | C standard semantics; FreeBSD's choices where it leaves room | Unit tests for the parsers. Orbis's locale behaviour is unknown |
| libSceLibcInternal | `sprintf`, `snprintf`, `vsnprintf`, `fprintf`, `vsscanf` | Our own printf and scanf engines. Variadic arguments come from registers and the stack, or a SysV `va_list`. `%ls` and `%lc` take 16-bit characters | Unit tests. Float output uses Rust's formatting, so the last digit of long expansions may differ from Orbis |
| libSceLibcInternal | `sin`, `cos`, `pow`, `atan2`, ... (the `<math.h>` functions ScreenTester5 imports, double and float) | The host's libm | May differ from Orbis in the last bit |
| libSceLibcInternal | `fopen`, `fclose`, `fread`, `fwrite`, `fgets`, `fseeko`, `ftello`, `fflush`, `ferror`, `fileno`, `__stdinp`, `__stdoutp`, `__stderrp` | Through the guest file system, with the `r`, `w`, `a` and `+` modes; writing in `/app0` fails with `EROFS`. stdout and stderr go to the host's; stdin is always at end of file. A FILE is an opaque handle, with the stream on the host | |
| libSceLibcInternal | `_init_env`, `atexit`, `exit`, `_Exit`, `abort` | `_init_env` ignores the process parameters (layout unknown). `exit` runs the `atexit` handlers latest first, as guest calls | Reached by ScreenTester5 |
| libSceLibcInternal | `qsort`, `bsearch`, `strto{l,ll,ul,ull,d}`, `atoi`, `atof`, `getenv`, `setenv`, `getcwd`, `strerror`, `time`, `rand`, `srand` | `qsort` is a heapsort calling the guest comparator (not stable). `getcwd` is `/app0`. The environment starts empty. `rand` is our own generator, so sequences differ from a console's | Number parsers unit-tested |
| libSceLibcInternal | `malloc`, `calloc`, `realloc`, `free`, `posix_memalign`, `memalign`, `aligned_alloc` | Our first-fit heap in guest memory at `0x20_0000_0000` (our choice), 16-byte aligned by default; the bookkeeping lives on the host, so a guest heap overflow cannot corrupt it. `realloc` of a pointer we did not hand out stops the process | Heap unit-tested; reached by DriftCheck |
| libSceUserService | `sceUserServiceInitialize`, `GetForegroundUser`, `GetInitialUser`, `GetLoginUserIdList`, `GetUserName` | One signed-in user, id 1, named `ps5slop` (our choices) | Reached by ScreenTester5 |
| libSceSystemService | `sceSystemServiceHideSplashScreen`, `sceSystemServiceLaunchWebBrowser` | Succeed; the browser call prints the address on stderr | Reached by ScreenTester5 |
| libSceAudioOut | `sceAudioOutInit`, `Open`, `Output`, `Outputs`, `Close`, `SetVolume` | Ports are mixed to stereo and played on the Mac (when the window is shown) or dumped to a WAVE file; `Output` blocks for about one buffer period. Details in [audio.md](audio.md) | Balatro plays its music |
| libScePad | `scePadInit`, `Open`, `GetHandle`, `Close`, `ReadState`, `Read`, `SetVibration`, `SetVibrationMode`, `SetLightBar` | One controller for user 1. `ReadState` reports what the host set with `Kernel::set_pad_input` (the window feeds it from a controller and the keyboard), in OpenOrbis's `OrbisPadData` layout (the PS4's; unverified for the PS5). `Read` keeps no history and always returns one sample of the current state. DriftCheck's own `scePadRead` sample struct (which it takes from the PS5 SDL port) puts every field we fill at the same offset; still unverified on a PS5. Vibration and the light bar are ignored | Layout unit-tested |
| libSceVideoOut | `sceVideoOutOpen`, `Close`, `SetFlipRate`, `SetBufferAttribute2`, `RegisterBuffers2`, `SubmitFlip`, `WaitVblank`, `AddFlipEvent`, `DeleteFlipEvent` | One port; flips take effect at once; a 60 Hz vblank clock; flip rate ignored. Details in [gpu.md](gpu.md) | ScreenTester5 draws its menu |
| libSceSystemService | `sceSystemServiceParamGetInt` | English (1), ISO date format, 24-hour time, UTC, no daylight saving, no parental limit, cross confirms (1). Our choices; the value encodings follow public headers and are unverified | Minecraft |
| libSceNet | `sceNetInit`, `Term`, `PoolCreate`, `PoolDestroy`, `GetMacAddress`, `EtherNtostr`, `in6addr_any`, `in6addr_loopback` | Set-up succeeds; the MAC address is a fixed locally administered one | Minecraft |
| libSceSaveData | `sceSaveDataMount3`, `Umount`, `Umount2`, `GetMountInfo`, `DirNameSearch` (and an unnamed search variant) | Each save is a folder in `savedata/` next to the title's `/data` folder, mounted at `/savedataN`. Mount modes create (fails if it exists) and create-if-missing; 32 KiB blocks; search patterns take `%` and `_`. Layouts come from what Minecraft passes and shadPS4's PS4 layouts, error codes from shadPS4: unverified for the PS5 | Minecraft creates and searches its settings save |
| libSceVideoOut | `sceVideoOutGetOutputStatus` | Writes 1 to the first field, which Minecraft reads as a 1080p output; the rest of the layout is unknown | Minecraft |
| libScePad | `scePadGetControllerInformation` | OpenOrbis's `OrbisPadInformation`: one connected controller with PS4 touch pad values | |
| libSceUserService, libSceSystemService | `sceUserServiceGetEvent`, `sceSystemServiceGetStatus`, `sceSystemServiceReceiveEvent` | No events: `GetEvent` returns OpenOrbis's `NO_EVENT` code; the status reports zero events and foreground | Minecraft |
| libSceAppContent | `sceAppContentTemporaryDataGetAvailableSpaceKb`, `sceAppContentAppParamGetInt` | 1 GiB free; the full game (SKU flag 3, unverified) | Minecraft |
| libSceAppContent | `sceAppContentTemporaryDataMount2` | `/temp0`, a `temp` folder next to the title's `/data` folder | Minecraft |
| libSceRandom | `sceRandomGetRandomNumber` | Host random bytes, at most 64 per call (shadPS4's limit; unverified) | Minecraft |
| many (`stubs.rs`) | Set-up and tear-down of dialogs (common, message, save data, sign-in, web browser, commerce, IME, authorized app), save data, SSL, HTTP, HTTP2, NP (web API 2, universal data system, trophies, game intent, entitlements, auth and manager requests), NetCtl callbacks, Ajm, AudioOut2, mouse, text to speech and core dumps | Placeholders: initialise and terminate calls succeed, create calls hand back fresh ids. `sceNpGetState` reports signed out and `sceNetCtlGetState` disconnected (encodings unverified) | Minecraft gets past them |
| libkernel | `getdents`, `sceKernelGetdents` | FreeBSD's pre-12 `struct dirent` records for descriptors opened on a directory, `.` and `..` first, then names in sorted order | Balatro lists its save folder |
| libSceRtc | `sceRtcGetCurrentTick` | Microseconds since 0001-01-01 UTC | |
| libSceNgs2 | systems, racks, voices (`VoiceControl`, `VoiceRunCommands`, `VoiceGetState`, `VoiceGetStateFlags`), `SystemRender`, `ParseWaveformData`, `ParseWaveformFile` | Sampler voices play PCM blocks through mastering voices into the title's buffers; no effects, no ATRAC9 or VAG. Ported from SharpEmu-mac, see [audio.md](audio.md) | Balatro |
| libSceSysmodule | `sceSysmoduleLoadModule`, `sceSysmoduleLoadModuleInternal` | Succeed without doing anything: every library is provided by the HLE layer | Called by vic's homebrew |

**Variables.** Some imports are data, not functions: libkernel's `__stack_chk_guard` is read through its GOT entry, although OpenOrbis's stub declares it a function. So the symbol type cannot be trusted; the HLE layer knows its variables by name and binds them to readable memory in a data region next to the trap region.

**System dialogs and PSN.** We are always signed out of PSN (`sceNpGetState` reports state 1). libSceMsgDialog and libSceSigninDialog (`dialog.rs`) show nothing: an opened dialog is finished at once, a sign-in dialog as cancelled by the user, a message dialog as if its first button was pressed, and a user message's text goes to stderr. Status values (0 none, 1 initialized, 2 running, 3 finished), results (0 OK, 1 cancelled), the message dialog's result struct and the PS4 parameter offsets (mode at 56, user message parameters at 64, their text at 8) come from the OpenOrbis headers; the sign-in result layout (the result first) is a guess, and none of it is checked on a PS5. `sceNpGetAccountCountryA` answers "us"; account id, authorization codes, Plus checks and reachability fail with ENOSYS. With these, Minecraft's Play button reaches the world list and the world creation screens (2026-11-20).

**Save data, as Minecraft uses it (2026-11-20).** Save folder names may hold any printable ASCII but `/`, `\` and a leading dot: Minecraft names world saves like `BedrockWorld2QAAC4ePE14@P1` (the console's own rules are unknown). Each save folder records the size it was created with (`sce_sys/blocks`, our own file); `sceSaveDataDirNameSearch` fills the optional per-save `infos` (blocks and free blocks, PS4 layout `OrbisSaveDataSearchInfo` from the OpenOrbis headers, offset 32 of the result) and `params` (offset 24), and Minecraft refuses to create a world ("Storage full") when the free blocks are missing. Saves made before sizes were recorded count as 1 GiB. `sceSaveDataSaveIcon`/`LoadIcon` keep `sce_sys/icon0.png`; `sceSaveDataDelete` removes an unmounted folder. libSceSaveDataDialog finishes at once and leaves its result as the caller set it (no layout known).

**Placeholders say so.** Functions that answer without doing the real work live in each module's `STUBS` table. The first five calls of each print `STUB! <library>: <name>(<first four argument registers>, ...) -> <result>` on stderr, so a log shows which placeholders a title leans on.

**Imports without a public name.** A few imports have no name we can find. When a title's own code shows what one does (its arguments and how it uses the result; we never look inside Sony's modules), we bind it by NID: a function named `#<NID>` in the HLE tables binds to that NID. So far: libSceLibcInternalExt `qBS714-Jr3g` (registers a `thread_local` destructor, as `__cxa_thread_atexit` does; accepted), libkernel_write_throttling `YFC3dBBipj8` (fills a write budget; fails, so Minecraft uses its default), and a save data search.

**Finding names for NIDs.** When a program stops at an unimplemented NID, we hash candidate names ourselves (function names from public API headers, such as the OpenOrbis headers) until one matches. That table stays local; we do not commit or import NID-to-name databases. Errors and `PS5SLOP_LIST_UNIMPLEMENTED=1` name an import when we know the name: any function we implement, in any library, and the names in the file `PS5SLOP_NAMES` points to. `scripts/names > references/names.txt` writes such a list from the macOS SDK and OpenOrbis headers (gitignored). For Balatro it names 287 of 566 missing imports.

## The guest file system

Guest paths resolve through a small mount table (`crates/hle/src/vfs.rs`): `/app0` is the title, read-only: a host folder, or a package's image served through the `ImageFs` trait (see [packages.md](packages.md)); `/data` is a writable per-title folder on the host, by default `~/Library/Application Support/ps5slop/<title id>/data`, or the folder's name when there is no title id (`--data DIR` picks another). Descriptors and FILEs hold a `GuestFile`, a host file or a file in an image, so reads, seeks, `fstat` and `getdents` work on both. `sceNgs2ParseWaveformFile` still uses host paths (`Vfs::resolve`), so it cannot read files from a package yet. Anything else does not exist, and paths that climb out of a mount with `..` are refused. What a real PS5 title sees under `/data`, and which other mounts it expects (`/savedata0`, `/temp0`, `/hostapp`), is unverified.

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

## What DriftCheck needs

DriftCheck (see [test-targets.md](test-targets.md)) is built with the same homebrew template as ScreenTester5. On top of what ScreenTester5 uses it imports 13 functions, all named by hashing names from the OpenOrbis and macOS SDK headers: `malloc`, `free`, `posix_memalign` (libSceLibcInternal); `open`, `read`, `close`, `sceKernelOpen`, `sceKernelWrite`, `sceKernelMkdir`, `sceKernelUsleep`, `sceKernelGetProcessTime` (libkernel); `sceUserServiceGetInitialUser`; `scePadRead`.

Status on 2026-11-19: with these implemented it has no unimplemented imports left, runs its start-up, sends its "DriftCheck ready" notification and draws its whole screen (both sticks, triggers, the button grid, touchpad and motion panels) with "CONTROLLER CONNECTED", checked with a frame dump after 15 s. Its log file under `/data/driftcheck` is not written, because `/data` does not exist yet: `sceKernelMkdir` and `sceKernelOpen` fail and DriftCheck carries on without a log. We report no touch points and no motion (all zero), so its touchpad and motion panels stay empty.

## What a real executable needs next

Libraries beyond libkernel (system services, pads, video out), the process parameters in `PT_SCE_PROCPARAM`, module start functions and init arrays, thread attributes, key destructors, and file and socket I/O.
