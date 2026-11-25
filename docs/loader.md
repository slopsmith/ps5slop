# Loader and linking

How executables and modules are read, laid out and linked. `crates/loader` reads the format (SELF wrapper, ELF, dynamic section, NIDs) and lays out an image with relocations applied; `crates/ps5slop/src/modules.rs` finds, chooses and links modules; `crates/mkelf` writes the same format for our own test programs from the same facts. What the HLE kernel offers loaded code (module start, `sceKernelLoadStartModule`, proc param) is in [hle-kernel.md](hle-kernel.md).

## Inputs

A title is a dumped folder (`PPSAxxxxx-app0` for PS5: `eboot.bin`, the game's modules, `sce_sys/param.json`) or a fake-signed package read in place ([packages.md](packages.md)). Executables must be plain ELF, or a SELF whose segments are plaintext; dumping tools decrypt while they dump. Encrypted SELFs are out of scope: we never decrypt anything ([legal.md](legal.md)).

## Format facts

Interface facts only, each with its source. The sources are open-source emulators (shadPS4, KytyPS5, Prosperity), used for facts, not code ([references.md](references.md)). How their authors learned them is not always known.

### SELF container

From shadPS4's `elf.h`/`elf.cpp` and Prosperity:

- Magic (u32 LE): PS4 `0x1D3D154F`, PS5 `0xEEF51454`. The PS5 container is said to wrap the same header layout (unverified beyond the files we load).
- A 0x20-byte header (segment count u16 at `0x18`), then one 0x20-byte entry per SELF segment: flags (u64), file offset, file size, memory size. Flags: `0x1` ordered, `0x2` encrypted, `0x4` signed, `0x8` compressed, `0x800` blocked; bits 20-31 name the program header the segment holds.
- Then the ELF header; program header offsets count from the ELF header. A program header's bytes live in the blocked SELF segment covering its file range; some headers (for example `PT_DYNAMIC` inside `PT_SCE_DYNLIBDATA`) have no segment of their own.
- We read only segments with neither the encrypted nor the compressed bit.

### ELF

x86-64 ELF with Sony types in the OS range (shadPS4; KytyPS5, which targets PS5, uses the same values):

- `e_type`: `ET_SCE_EXEC 0xFE00`, `ET_SCE_STUBLIB 0xFE0C`, `ET_SCE_DYNEXEC 0xFE10` (the usual game executable), `ET_SCE_DYNAMIC 0xFE18` (modules).
- Program headers: `PT_SCE_RELA 0x60000000`, `PT_SCE_DYNLIBDATA 0x61000000`, `PT_SCE_PROCPARAM 0x61000001`, `PT_SCE_MODULE_PARAM 0x61000002`, `PT_SCE_RELRO 0x61000010`, `PT_SCE_COMMENT 0x6FFFFF00`, `PT_SCE_LIBVERSION 0x6FFFFF01`.
- **PS4-style dynamic section** (shadPS4 `module.cpp`, `linker.cpp`): `PT_DYNAMIC` holds (tag, value) pairs. The `DT_SCE_*` table tags are offsets into `PT_SCE_DYNLIBDATA`'s file bytes: `STRTAB`/`STRSZ` `0x61000035`/`37`, `SYMTAB`/`SYMTABSZ` `39`/`3F`, `RELA`/`RELASZ` `2F`/`31`, `JMPREL`/`PLTRELSZ` `29`/`2D`. `DT_SCE_NEEDED_MODULE` (`0F`) and `DT_SCE_MODULE_INFO` (`0D`) pack a module: string offset bits 0-31, minor version 32-39, major 40-47, id 48-63. `DT_SCE_IMPORT_LIB` (`15`) and `DT_SCE_EXPORT_LIB` (`13`) pack a library: string offset, version bits 32-47, id 48-63. Standard tags carry image addresses as usual.
- **PS5-style** (seen in homebrew from clang's `x86_64-sie-ps5` target; tag numbers match KytyPS5's `elf.h`): no `PT_SCE_DYNLIBDATA`. The tables sit in a `PT_LOAD` segment with no permissions and are found through the standard tags (`DT_STRTAB`, `DT_SYMTAB`, `DT_RELA`, `DT_JMPREL`, `DT_HASH`, ...) as image addresses; `DT_SCE_SYMTABSZ` still gives the symbol table size. Module and library tags move: original filename `0x61000041`, module info `43`, needed module `45`, export library `47`, import library `49`. Code segments are execute-only.
- Symbols and relocations use the standard ELF64 `Sym` and `Rela` layouts. Relocation types handled: `R_X86_64_64`, `GLOB_DAT`, `JUMP_SLOT`, `RELATIVE`, `DTPMOD64`.

Unknown: the PS5 `PROCPARAM` layout beyond what the runtime asks for.

### NIDs

Imports do not name functions. Each imported symbol is `NID#LIB#MOD`:

- **NID**: SHA-1 of the name followed by a fixed 16-byte suffix; the first 8 bytes of the digest as a little-endian u64, written as 11 characters in the alphabet `A-Z a-z 0-9 + -` (ten 6-bit groups from the top, then the low 4 bits times 4). The suffix appears in many public repositories. Algorithm from shadPS4's `module.cpp`; checked by hashing names from vic's own homebrew source and finding them among its imports. Code: `crates/loader/src/nid.rs`.
- **LIB**, **MOD**: small ids in the same alphabet, indexing the library and module tables.
- Exports carry NIDs only, so a module's table says which functions a library exports, not their names.

Which names we may hash, and what symbol data may be committed, is in [legal.md](legal.md). An import we have no name for stays an opaque NID; HLE can still implement it as `#<NID>` when the title's own code shows what it does ([hle-kernel.md](hle-kernel.md)).

## Loading a title

`Process::load` in `crates/ps5slop` with `modules.rs`:

- **Layout.** The executable loads at `0x40_0000`; modules follow one another from `0x8_0000_0000` (see [memory.md](memory.md)). Metadata segments that share a page with data keep their bytes but get no region of their own.
- **Finding modules.** Candidates are the `.prx`/`.sprx` files in the title up to three folders deep, shallowest first, skipping `sce_sys`, keyed by the module name their own tables give. Files in a folder named `sce_module` are Sony's. Unity titles keep theirs in `Media/Modules` and `Media/Plugins`.
- **Choosing.** Every game module (outside `sce_module`) loads. A Sony module loads only if the executable or a chosen module imports from it and it is on the allowlist (below). Everything else is HLE.
- **Binding.** All exports (global or weak symbols with a value, as `(library, NID)`) are collected before anything is relocated. An import binds to a loaded module's export first, then to an HLE stub in the trap region. Every import gets a stub; calling one HLE does not implement stops the process by default ([hle-kernel.md](hle-kernel.md)).
- **TLS.** One static variant II layout holds the executable's block (module id 1) and each module's below it. `DTPMOD64` gets the module's id, and `__tls_get_addr` maps (module, offset) to the calling thread's block.
- **Initialisers.** Before the entry point, on the main thread, dependencies first, for the modules the executable needs directly or indirectly: the module's entry point (`module_start`) if it has one, else `DT_INIT` and the init array, with zero arguments. Other modules (plugins) are linked at start but started only when the title calls `sceKernelLoadStartModule`, with the arguments it passes. Reason: linking everything at start keeps ELF loading out of the HLE crate, and deferring the start keeps plugins' start code in the order and with the arguments the title expects. Finalisers are not run.

## Sony modules as guest code (semi-LLE)

Games ship some Sony libraries in `sce_module/`. We may load such a module as ordinary guest code instead of writing an HLE version. Modules come from the user at run time and are never committed ([legal.md](legal.md)).

Only modules on an allowlist load (`ALLOWED_SYSTEM_MODULES` in `modules.rs`; today `libc`). A module qualifies only if everything it imports is provided by HLE or by other allowlisted modules, so it runs on top of HLE and never reaches the kernel or hardware itself. Reason: loading a module that reaches into the kernel or GPU would bring LLE in through the back door. `libc` qualifies (it needs only `libkernel`) and is worth it: titles import a whole C and C++ runtime from it (iostreams, locales, exceptions). Whether to keep the allowlist at all is vic's open question.

What Sony's libc expects from the loader and kernel is learned from the calls it makes and their arguments, never by disassembling it ([hle-kernel.md](hle-kernel.md)).
