# Loader and linking

What the loader has to read, and what is publicly known about it. Nothing here is implemented yet.

Sources are open-source emulators (shadPS4, KytyPS5, Prosperity), read through a research summary on 2026-11-19. They are a "public" source only in the sense that they are published; how their authors learned these facts is not always known. We use them for format facts, not code (see [references.md](references.md)). psdevwiki blocks automated fetches and has not been checked yet.

## What the user supplies

Dumping tools decrypt while they dump. The result is a folder (`PPSAxxxxx-app0` for a PS5 title) with `eboot.bin`, the game's own `.prx`/`.sprx` modules and `sce_sys/param.json`. We require that decrypted form: a plain ELF, or an fself whose segments are plaintext. Encrypted SELFs are out of scope; we never decrypt anything (see [legal.md](legal.md)).

Status: the folder layout comes from a public backup guide; unverified by us.

## Container

| Fact | Value | Status |
| --- | --- | --- |
| PS4 SELF magic (u32, little-endian) | `0x1D3D154F` | From shadPS4 and Prosperity source |
| PS5 SELF magic | `0xEEF51454` | Same; the PS5 container is said to wrap the same header layout (unverified) |

The loader strips a (fake-signed) SELF wrapper to reach the ELF inside, or takes a plain ELF.

## ELF

x86-64 ELF with Sony-specific types in the OS range:

- `e_type`: `ET_SCE_EXEC 0xFE00`, `ET_SCE_STUBLIB 0xFE0C`, `ET_SCE_DYNEXEC 0xFE10` (the usual game executable), `ET_SCE_DYNAMIC 0xFE18` (modules).
- Program headers beyond the usual ones: `PT_SCE_RELA 0x60000000`, `PT_SCE_DYNLIBDATA 0x61000000`, `PT_SCE_PROCPARAM 0x61000001`, `PT_SCE_MODULE_PARAM 0x61000002`, `PT_SCE_RELRO 0x61000010`, `PT_SCE_COMMENT 0x6FFFFF00`, `PT_SCE_LIBVERSION 0x6FFFFF01`.
- Dynamic tags `DT_SCE_*` in `0x61000007..0x6100003F`. Their payload (symbol table, string table, relocations, needed modules, imported libraries) lives in the `PT_SCE_DYNLIBDATA` segment instead of the loaded image.

KytyPS5, which targets PS5, uses the same values as shadPS4 (PS4). Finer PS4/PS5 differences (contents of `PROCPARAM`, any new tags) are unknown.

## Symbols and NIDs

Imports do not name functions. Each imported symbol is a string `NID#LIB#MOD`:

- **NID**: a hash of the function's name. Take SHA-1 of the name followed by a fixed 16-byte suffix, read the first 8 bytes of the digest as a little-endian u64, and write it as 11 characters in the alphabet `A-Z a-z 0-9 + -`: ten 6-bit groups from the top, then the low 4 bits times 4. The suffix is in many public repositories and is a fact we may use.
- **LIB** and **MOD**: small ids in the same alphabet, which index the library and module tables in the dynamic section.

Status: algorithm read from shadPS4's `module.cpp`; we will check it with our own test binaries, whose names we choose.

**Rules:**

- We compute NIDs only from names we know from public documentation or from our own test programs. We never import a name-to-NID database (aerolib, ps4libdoc, generated stub lists): their names come from leaked or extracted material.
- Symbol tables read from user-dumped modules (exported NIDs, library and module names and versions) may be committed as part of the HLE code (see [legal.md](legal.md)).

Exports seem to carry NIDs only: shadPS4's loader (`Module::LoadSymbols`) parses exported symbols as `NID#LIB#MOD`, the same as imports, and looks names up in its separate aerolib database. So a module's table tells us which functions a library exports, not their names. PS4 only, from an emulator; PS5 unverified.

An import we have no name for stays an opaque NID. The HLE layer can still log it and fail loudly.

## Sony modules as guest code (semi-LLE)

Games ship some Sony libraries in their own folder (`sce_module/`), and users may supply firmware modules from their own console. We may load either kind as ordinary guest code instead of writing an HLE version, as shadPS4 does for a whitelist. The modules come from the user at run time and are never committed (see [legal.md](legal.md)). HLE stays the default for anything that talks to the kernel, the GPU or other hardware.

## Open questions

- PS5 `PROCPARAM` layout and what the runtime reads from it (SDK version, heap sizes, and so on).
- How module start and stop functions, TLS images and init arrays are laid out for `ET_SCE_DYNAMIC` modules.
- Which Sony modules are worth loading as guest code (see below) and which we must HLE because they reach into the kernel or hardware.
