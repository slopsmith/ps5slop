# Legal boundaries

These are hard rules. If something is not clearly covered, stop and ask vic before doing it.

## What never enters the repo

- Proprietary Sony material of any kind: firmware, system modules (`.sprx`, `.prx` and similar), keys, game binaries (encrypted or decrypted), game assets, fonts, shader binaries and GPU microcode.
- Reverse-engineered parts of proprietary modules: disassembly or decompiler output, function-by-function ports, and tables, constants or structures copied out of a dump. The one exception is symbol metadata, below.
- Anything a user supplies: their dumped games, firmware files and keys. Loading a user-supplied Sony module as guest code at run time is allowed (see [loader.md](loader.md)); committing it is not. These live outside the repo or under gitignored paths (`games/`, `firmware/`, `keys/`, `dumps/`, `references/`).
- Anything derived from a game: frame captures, save states, shader dumps, decrypted files.
- Leaked SDK material: headers, samples, documentation or tools from Sony's official SDK. Not even "just to check a struct layout".
- Links to places that host any of the above.

## What we may write down

Facts about interfaces and behaviour, from public sources or from our own observation:

- Library and function names, NIDs and how they are derived, syscall numbers and calling conventions, where these are publicly documented.
- The layout of file formats and structures as far as public documentation (psdevwiki, AMD's ISA guides, open-source emulators with compatible licenses) describes them.
- Behaviour we observe by running **our own** test programs, for example on vic's PS4 (see [test-targets.md](test-targets.md)).

Each such fact in the docs names its source. Facts with no public source are marked unverified.

Some imports have no public name. An HLE function may bind to such an import by its NID alone, when the title's own code shows what the import does. The behaviour then comes from the title, never from Sony's modules.

## How we implement system libraries

Clean room. We implement the Orbis and Prospero library and kernel interfaces from public documentation, observed behaviour and notes written by others. We do not translate proprietary code into ours, even loosely.

We never decompile or disassemble Sony's system code ourselves. Public documentation that others wrote, even if they learned it by decompiling, may be used as a specification: names, signatures, structure layouts, the command packets a function emits, behaviour. We write our own code from it. Not usable: decompiler output itself, code that is a port of Sony's code (whatever its license), and leaked SDK material. This matters most for AGC, which has no official documentation: what is public about it comes from the scene's reverse engineering.

Code or raw output that says it was made by reverse engineering is not read, even when the documentation next to it may be. Example: the PS5 homebrew template (ps5-native-app-boilerplate) generates its `libc.prx` from `libc_builder.cpp` and `api-surface.txt`; those two files are never read. Its other sources may be read for facts, and running the `libc.prx` a homebrew title ships is fine, since it is part of that title.

## Analysing user-supplied binaries

Locally, on the user's own dumps, and with nothing but our conclusions entering the repo.

**Game code may be read.** Agents may count instruction classes, log imports, and disassemble or decompile a game's code locally to learn what the game does and what it expects of us (for example which fields of a structure it checks before an assert). Nothing from it is committed: no code, listings or constants, only our conclusions in prose. Game code here includes:

- the game's own executables and modules;
- middleware the game ships (Unity's plugins, FMOD and similar), even when built on Sony's libraries, as long as the reading is about what the middleware expects of us;
- Sony libraries a game includes, GPU programs a game ships or builds (also when they come from a Sony library, such as GTA V's ray-tracing and BVH programs), and code in a game whose origin is unclear.

The reason: game code tells us how the game uses the interface and what answers it needs. It never becomes part of the emulator.

**Not game code:** libraries in a game that may be system code, such as the AGC-like library linked into Astro Bot's executable. These are not read.

**A game's data files** may be decrypted locally with a key taken from that game's own executable, to see what the game loads. Neither the key, the tool nor the output enters the repo, and the rule against decryption code below still holds.

**System modules, to some extent:** reading their metadata (which libraries and NIDs a module exports, library and module names and versions, imports) is fine, and such symbol tables may be committed as part of the HLE code. Nothing else from a module is committed: no code, data tables, constants or structure layouts. Decompiling or disassembling how a module works is out of line.

## Toolchains for test programs

Open-source toolchains such as OpenOrbis may be used to build test programs and, unless there is reason to believe they derive from Sony's SDK, as a reference for formats and interfaces. Name databases (NID-to-name lists) from any source stay out. Our own test programs are built with our own `ps5slop-mkelf`.

## Other projects

Before reading another emulator's code, check its license. Reading for ideas is fine for any open-source license. Copying code needs a license compatible with GPL-2.0-or-later. Record what we read and what we took in [references.md](references.md).

GPL-3.0 code is never copied, linked or loaded into ps5slop, statically or as a dynamic library, and users are never asked to supply such a library (the same goes for Oodle). A GPL-3.0 program may be run on a developer's machine as a black-box oracle, comparing its outputs with ours, as long as its source is not read and its outputs (game data) stay outside the repo. Our Kraken decoder was checked this way.

Projects that are built on leaked SDK material are not read at all.

## Console security

No code that circumvents console security: no exploits, no key extraction, no decryption of protected content. One exception: opening a fake-signed package with its own passcode, which uses no Sony key material (see [packages.md](packages.md)). Opening packages with Sony keys (the fake-package RSA key, retail or licence keys) stays out, so retail packages are not supported. Dumping is the user's own business and happens outside this repo, with their own hardware. The docs do not describe how to do it.

Entitlements: HLE may report an add-on as owned only when its content is already in the user's own package (GTA V ships Story Mode as an add-on of the base package). Separately sold add-ons stay unowned.

System fonts are not shipped. Open-source stand-ins are fetched by the user (`scripts/fetch-fonts` or the launcher) and served in their place.

## The check

`scripts/check-forbidden [REVSET]` (also `jj check` once the repo alias is set, see [AGENTS.md](../AGENTS.md)) fails when a revision tracks files that look like console material: known file names, files under the reserved directories, SELF (PS4 and PS5) or package (PS4 and PS5) headers, ELF files with an OS-specific `e_type`, and anything over 4 MiB. Run it before every push. It is a safety net, not a proof: a file it accepts can still be forbidden.
