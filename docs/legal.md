# Legal boundaries

These are hard rules. If something is not clearly covered, stop and ask vic before doing it.

## What never enters the repo

- Proprietary Sony material of any kind: firmware, system modules (`.sprx`, `.prx` and similar), keys, game binaries (encrypted or decrypted), game assets, shader binaries and GPU microcode.
- Reverse-engineered parts of proprietary modules: disassembly or decompiler output, function-by-function ports, and tables, constants or structures copied out of a dump.
- Anything a user supplies: their dumped games, firmware files and keys. These live outside the repo or under gitignored paths (`games/`, `firmware/`, `keys/`, `dumps/`, `references/`).
- Leaked SDK material: headers, samples, documentation or tools from Sony's official SDK. Not even "just to check a struct layout".
- Links to places that host any of the above.

## What we may write down

Facts about interfaces and behaviour, from public sources or from our own observation:

- Library and function names, NIDs and how they are derived, syscall numbers and calling conventions, where these are publicly documented.
- The layout of file formats and structures as far as public documentation (psdevwiki, AMD's ISA guides, open-source emulators with compatible licenses) describes them.
- Behaviour we observe by running **our own** test programs, for example on vic's PS4 (see [test-targets.md](test-targets.md)).

Each such fact in the docs names its source. Facts with no public source are marked unverified.

## How we implement system libraries

Clean room. We implement the Orbis library and kernel interfaces from public documentation, observed behaviour and notes written by someone who did not read the proprietary code. We do not translate proprietary code into ours, even loosely.

## Analysing user-supplied binaries

Locally, on the user's own dumps, and with nothing but our conclusions entering the repo:

- **Games:** fine to analyse, for example counting instruction classes to see what the CPU must support, or logging which imports a game uses.
- **System modules, to some extent:** reading their metadata (which libraries and NIDs a module exports, its imports, its sections) is fine. Whether such a table may be committed, or only generated on the user's machine from their own files, is still open (see [decisions.md](decisions.md)).
- **Out of line:** decompiling or disassembling how a module works, and using any output of that, even indirectly. Our implementation must never be informed by it.

## Other projects

Before reading another emulator's code, check its license (see [references.md](references.md)). Reading for ideas is fine for any open-source license. Copying code needs a license compatible with GPL-2.0-or-later (or a deliberate decision to move the project to GPLv3, recorded in [decisions.md](decisions.md)). Record what we read and what we took in [references.md](references.md).

Projects that are built on leaked SDK material are not read at all.

## Console security

No code that circumvents console security: no exploits, no key extraction, no decryption of protected content. Dumping is the user's own business and happens outside this repo, with their own hardware. The docs do not describe how to do it.

## The check

`scripts/check-forbidden [REVSET]` (also `jj check` once the repo alias is set, see [AGENTS.md](../AGENTS.md)) fails when a revision tracks files that look like console material: known file names, files under the reserved directories, SELF (PS4 and PS5) or package headers, ELF files with an OS-specific `e_type`, and anything over 4 MiB. Run it before every push. It is a safety net, not a proof: a file it accepts can still be forbidden.
