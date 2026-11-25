# HLE: the kernel interface and system libraries

How guest calls reach host code, and the rules the host side follows. Code: `crates/hle`. The functions we implement are the `FUNCTIONS` and `STUBS` tables in each module; this doc does not list them.

## How a call reaches the host

- Every import binds at load time to its own 16-byte slot in a **trap region** at `0xFF_0000_0000` (our choice, see [memory.md](memory.md)). The region is never readable, writable or executable, so a call there leaves the guest at once: the interpreter stops before decoding, and the JIT never maps it.
- A slot names a `(library, NID)` pair. If a host function implements it, it runs with the SysV AMD64 arguments, its result goes to `rax`, and the return address is popped as a `ret` would.
- A host function can also return `Outcome::Resume`: it has set up the CPU itself, for example to call a guest function with a return address in the trap region (`pthread_once`, `atexit` handlers). `Env::call_guest` runs a guest callback from inside a host function and returns when the callback does.
- Imports with no host function still get a slot. Calling one stops the process with "unimplemented *library* function *name* (*NID*)" and the caller, instead of jumping to 0. `PS5SLOP_UNIMPLEMENTED_ZERO` makes them return 0 instead (for exploring only, see [debugging.md](debugging.md)).
- **Variables.** Some imports are data (libkernel's `__stack_chk_guard`, libc's stdio FILEs). The symbol type cannot be trusted, so the HLE layer knows its variables by name and binds them to a readable data region at `0xFF_8000_0000`.
- Every libkernel function is also bound under `libScePosix`. A few PS4 library names (`libSceSaveData`, `libSceSaveDataDialog`, `libSceMsgDialog`) bind to the PS5 libraries' functions (`PS4_TWINS` in `lib.rs`): the NIDs are the same, and our layouts there are the PS4 ones anyway.
- A raw `syscall` instruction stops the process. We serve the kernel through libkernel's functions, not its system call numbers.
- Internal stubs (library `ps5slop`: the loader teardown callback, callback returns, `cond_relock`) are made at start in sorted name order, so every process gives them the same addresses. Save states depend on that ([savestates.md](savestates.md)).

## Names, NIDs and unnamed imports

- A NID is a hash of the symbol name. We find names by hashing guesses built from public headers (`scripts/nid`, `scripts/nid-guess`, see [debugging.md](debugging.md)). A guessed name confirmed by a hash match is kept. Name lists stay local; no NID-to-name database enters the repo ([legal.md](legal.md)).
- An import with no name we can find binds by NID: a host function named `#<NID>` binds to that NID. We do this only when the title's own code shows what the import does. We never look inside Sony's modules.
- An unnamed import that a title only calls in passing, and whose result it does not depend on, fails with `ENOSYS`. One whose failure stops the title succeeds instead.

## Placeholders

A placeholder answers without doing the real work. Placeholders live in each module's `STUBS` table. The first five calls of each are logged as warnings on the `hle::stub` target (`library: name(first four args) -> result`), so a log shows which placeholders a title leans on. Unimplemented imports and placeholder calls are also counted in the problem report.

Rules:

- **Set-up and tear-down succeed**, and create calls hand back fresh ids. Failing them usually sends a title down an error path it never takes on a console.
- **Succeed when failure stalls the title.** FMOD gives up all audio when `sceAcmContextCreate` fails. Unity's PSN plugin never finishes initialising if `sce::Json::Initializer::initialize` or the NpCppWebApi set-up fail, and the title waits for it forever. So these succeed and do nothing. If a title later needs data they should have produced (trophies, web API replies), that shows as missing data rather than a stall.
- **Offline.** We are always signed out of PSN (`sceNpGetState`), NetCtl reports disconnected, sockets fail with `ENETDOWN`, account and authorisation queries fail. Nothing is ever sent to a network.
- **System dialogs show nothing.** An opened message or sign-in dialog finishes at once (a message dialog as if its first button was pressed, sign-in as cancelled); a message's text goes to stderr. The IME panel never opens.
- **Movies.** libSceAvPlayer has no decoder: it reports one 3-second 1080p stream and plays it as black NV12 frames, then stops, so titles move on.

## Error conventions

- POSIX-style functions set errno and return -1. Each thread's errno lives in its thread control block at the thread pointer plus `0x800` (our choice); `__error()` returns its address.
- Error values are FreeBSD's (`hle::errno`), assumed for Orbis, not verified.
- `sceKernel*` and `scePthread*` functions return `0x80020000 + errno`. The POSIX pthread functions return the error number, as POSIX says.

## Threads and synchronisation

- **One host thread per guest thread**, each with its own CPU state; no cooperative scheduling. `hle::Kernel` holds what threads share: stubs, kernel objects, thread records, stdio and the exit state.
- A thread checks for process exit every 65,536 instructions, and a host function that blocks wakes every 20 ms to check too. The process ends at `exit`, at the first guest error in any thread, or when its last thread ends.
- **Thread pointer and TLS.** FS base is the thread pointer. Static TLS uses the x86-64 "variant II" layout: FS points at a word holding its own address, and the executable's `PT_TLS` image sits just below it. Module TLS goes through `__tls_get_addr` ([loader.md](loader.md)). `pthread_t` is the thread pointer.
- **Thread-specific values** for keys 0 to 127 live at the thread pointer plus `0x400` (our choice). The JIT runs `pthread_getspecific`, `scePthreadGetthreadid` and `pthread_self` natively inside the VM, where FS is the only way to the thread; titles call these hundreds of thousands of times a second, too often for a VM exit each. Higher keys stay in a host table. Key destructors are not run. Whether Sony's libc uses these control block bytes itself is unknown (assumed not).
- **Stacks.** The main thread gets 8 MiB: the console default is unknown, and titles leave the process parameters' stack size null. Other threads get the stack set with `pthread_attr_setstack`, or 1 MiB. Stacks are recorded, so attribute queries on a thread report its real stack.
- **pthread handles are guest addresses.** `pthread_mutex_t`, `pthread_cond_t`, `pthread_rwlock_t`, their attributes and thread attributes hold the address of a zeroed 64-byte block from our guest heap, and our objects are keyed by it. Sony's libc reads through these words as pointers, as FreeBSD's types are. Destroying an object frees its block, since titles create and destroy mutexes by the thousand a second. A zero word means statically initialised: the object is made on first use. `sem_t` holds our handle in its first word.
- **Mutexes hand off.** An unlocked mutex with waiters goes to one of them, and a thread that unlocks and relocks at once waits its turn. Without this, a worker that releases and retakes a lock in a loop starves the waiter. The default type is error-checking (FreeBSD's default).
- **Condition variables** honour `pthread_condattr_setclock`: monotonic deadlines use the same clock as `clock_gettime`. `signal` wakes all waiters (POSIX permits spurious wake-ups).
- **Timed waits step toward their deadline.** Every guest wait with a deadline (conditions, mutexes, semaphores, event flags, event queues, sleeps, vblank and audio pacing) blocks for two thirds of the time left and loops (`hle::timing`). macOS timer coalescing lets a plain 5 ms wait take 7.5 ms, which made audio threads that refill on short timeouts fall behind. Real-time thread policies were rejected: a spinning guest thread could starve the Mac.
- Priorities, affinity and scheduling parameters are accepted; host threads keep their own priority. Signals are accepted and never delivered; threads are never cancelled.
- **Fibers** (libSceFiber) are saved guest CPU states on the host, keyed by the guest fiber object, whose bytes we leave alone. A fiber's context memory is its stack, and it keeps the thread pointer of whichever thread runs it.

## Exceptions

Unity's garbage collector stops threads with `sceKernelRaiseException(thread, signal)` after installing a handler, and waits for each thread to acknowledge. So the handler must run on the target thread, also while that thread is blocked in a host function.

- A raised exception is queued for its thread and every vCPU is kicked. The thread's run loop (and `Env::call_guest`) delivers it before running guest code again.
- Delivery writes the interrupted registers below the stack pointer (past the red zone) and calls `handler(signal, context, context)` on the thread's own stack. The context is FreeBSD's amd64 `mcontext_t` starting at byte 64. Observed: Superliminal's handler reads `rsp` at byte 248 (`mc_rsp`). The rest of the layout is assumed.
- A thread blocked in a host function gives up the wait. Condition waits and sleeps return early. Every other blocking call returns with the CPU untouched, so the guest calls it again after the handler, as `SA_RESTART` would. This needs no handler support inside each wait, only the ability to give up. Save states reuse the same mechanism.
- Unverified: what Orbis passes besides the context, and whether a console restarts these calls.

## Memory

The address-space layout is in [memory.md](memory.md). The HLE side:

- **Direct memory** is 12.5 GiB (our choice; one title's engine asks for about 12.1 GiB), allocated first fit. It is one host object: a mapping puts its physical pages at the mapping's offset, so two mappings of one range share bytes, and an allocation zeroes its range. Releasing a range frees exactly that range and unmaps every mapping of it, as shadPS4 does on the PS4 (unverified for the PS5).
- **Flexible memory** is fresh zeroed memory; its budget is not tracked.
- **Maps without a fixed address** search from `0x11_0000_0000`. The first 4 GiB of the user area is left to `MAP_FIXED` maps, because a title can place a large arena there without reserving it. Where the PS5 kernel places maps is unknown.
- **Queries report what the guest gave.** `sceKernelVirtualQuery` reports a direct mapping's offset and memory type. `sceKernelQueryMemoryProtection` reports the protection bits the guest last gave the page, including bits we do not map (`0x40`, `0x80`), since a title may pick a code path by them. What those bits mean on a console is unknown.
- **Our guest heap** at `0x20_0000_0000` backs `malloc` and friends and every block host functions hand out (pthread handles, FILEs, font objects). It is first fit and 16-byte aligned; its bookkeeping lives on the host, so a guest overflow cannot corrupt it.
- The first failure of each kind of memory call logs the request and a `memory` report (target `hle::dmem`); later ones are only counted.

## Files

Guest paths resolve through a mount table (`vfs.rs`). Anything outside a mount does not exist, and paths that climb out of one with `..` are refused.

| Mount | Backing |
| --- | --- |
| `/app0` | The title, read-only (writes fail with `EROFS`): a host folder, or the package's PFS image read in place |
| `/data` | The title's writable folder |
| `/savedataN` | A save data folder, while mounted |
| `/temp0`, `/download0` | Writable folders beside the data folder |
| `/devlog` | `PS5SLOP_DEVLOG`, when set |

- **Per-title folders.** `/data` is `~/Library/Application Support/ps5slop/<title id>/data` unless `--data DIR` names another. Without a title id the folder's or file's name is used, and such a folder is renamed to the title id once one is found, so a title run from a package and from a folder sees the same data. Saves, temp and download folders sit beside the data folder: siblings named `savedata`, `temp` and `download0` when it is called `data`, otherwise `DIR.savedata` and so on, so data folders side by side never share saves.
- **Packages run in place.** `/app0` and the loader's reads come straight from the package through the `ImageFs` trait ([packages.md](packages.md)); we never extract. Packages are tens of gigabytes and titles read only part of them. Descriptors hold a `GuestFile` (host file or image file), so reads, seeks, `fstat` and `getdents` work on both. Code that wants a host path (`Vfs::resolve`) gets none for package files and should use `Vfs::read`.
- Descriptors share one number space with stdio's FILEs. fds 1 and 2 go to the host's stdout and stderr; stdin is always at end of file. `struct stat` and `struct dirent` use FreeBSD's pre-12 layouts, as shadPS4 does for the PS4 (unverified for the PS5). Permissions and file times are accepted and not kept.
- **APR** (the PS5's command-buffer file reads): paths resolve to ids and sizes, `ReadFile` commands are kept on the host keyed by the guest command buffer's address (whose bytes we never touch), and the reads run synchronously when the buffer is submitted. A "write a value once the reads before it are done" command is inferred from one title's use. Argument positions are read off titles' calls and unverified beyond them.
- **Save data**: each save is a folder in the saves folder, with the size it was created with recorded in our own `sce_sys/blocks`. Folder names may hold any printable ASCII but `/`, `\` and a leading dot. Search results report the save's newest file time as `mtime`, which titles check before loading. Save data memory is one file per user and slot in the saves folder's `.memory`. Layouts are the PS4's from shadPS4 and the OpenOrbis headers, checked only against what titles pass.
- PS4 titles' own `libc.prx` does its file I/O through libSceFios2, which we provide over our descriptors. Its error codes are a guess.
- **System fonts.** A title may name system fonts as its own files under `/app0`. When the title lacks a file whose name is a known system font, the VFS serves an open-source stand-in from the fonts folder (`--fonts DIR`, default `~/Library/Application Support/ps5slop/fonts`, filled by `scripts/fetch-fonts`). Sony's fonts may not be shipped; the stand-ins are OFL-licensed, fetched, and never committed. A file the title has always wins. Weights are not matched.
- Unverified: what a PS5 title sees under `/data`, and which other mounts it expects.

## Process and module start

- The entry point gets what OpenOrbis's startup code and shadPS4's `RunMainEntry` agree on: `rdi` points at `{ i32 argc; u32 pad; char *argv[]; }` (one argument, `eboot.bin`), `rsi` is a loader teardown callback (start-up registers it with `atexit`, so it must return; ours does nothing), and `rsp` is 8 bytes off 16-byte alignment with argc and `argv[0]` on top. What the real kernel adds (environment, auxiliary vector) is unknown.
- Sony's libc asks the loader for its callbacks and the process parameters at start (`rtld.rs`): callbacks are stored, the parameters are the executable's `PT_SCE_PROCPARAM`, and no sanitizer is present.
- Modules the executable needs start before the entry point. `sceKernelLoadStartModule` starts other modules on first use. Sony-built modules start through `DT_INIT`, called as `DT_INIT(args, argp, 0)` with the title's arguments (observed: a plugin reads its arguments through the second). See [loader.md](loader.md).
- `sceSysmoduleLoadModule` succeeds and does nothing: every library is ours. `dlopen` loads nothing.

## System services

- **User.** One user, id 1, named `ps5slop`, logged in. `sceUserServiceGetEvent` reports that user logging in on its first call: titles look the user up among the logins they have seen, and stop when there is none. That a console reports already logged-in users this way is assumed.
- **System parameters** are our choices: English, ISO dates, 24-hour time, UTC, no daylight saving, cross confirms. The machine is a base PS5, not a Pro.
- **Entitlements.** A title's own add-on entitlements, which its package needs to unlock content it already ships, are reported as installed, keyed by title id (only GTA V's Story Mode labels so far). Every other title gets none, and separately sold add-ons stay unowned. Handing every title the same labels would be wrong: another title treats them as its own DLC.
- **Fonts.** libSceFont loads, measures and rasterises with FreeType (`freetype-rs`, bundled build); every font set is the Noto Sans JP stand-in. Libraries, renderers and fonts are blocks from our guest heap, so titles get real pointers. Prototypes and the public structure layouts come from shadPS4's `font.h` ([references.md](references.md)), unverified for the PS5. Not done: text layout and shaping (planned with HarfBuzz), slant and weight effects, vertical rendering, outlines. Fonts in use block save states.
- Pads, audio and VideoOut are in [input.md](input.md), [audio.md](audio.md) and [gpu.md](gpu.md).

## Where layouts come from

Structure layouts, constants and argument orders come from, in order of preference: public headers (OpenOrbis), other emulators' notes and code under compatible licenses (shadPS4, SharpEmu-mac; see [references.md](references.md)), and what a title's own code passes and reads. Most are PS4 layouts or another emulator's reading; none is checked on a PS5. The code comments name the source of each.

## Known gaps

- No network, no PSN, no trophies, no voice chat.
- No video decoding (AvPlayer plays black).
- Signals and thread cancellation are not delivered; key destructors are not run.
- `sysctl` knows no names. The environment starts empty.
- Our `printf` formats floats with Rust, so the last digit of long expansions may differ from Orbis; `rand` sequences differ from a console's; math functions use the host's libm.
