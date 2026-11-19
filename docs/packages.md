# Packages

What a `.pkg` holds, and what running one directly would take. Nothing is implemented.

## PS4 PKG

Sources: vic's own `ps4-pkg-db` (`pkg.py`, `sfo.py`), LibOrbisPkg's `PfsStructs.cs` (LGPL-3.0: read for facts only), and the header of vic's own homebrew package `ps4-pkg-store/ps4build/pkgstore.pkg`, built with OpenOrbis, read on 2026-11-19.

| Field | Where | Notes |
| --- | --- | --- |
| Magic | `0x00`, `7F 43 4E 54` (`\x7FCNT`) | Fields are big-endian |
| Entry count, entry table offset | `0x10` (u32), `0x18` (u32) | 32-byte entries: id at `+0x00`, data offset `+0x10`, size `+0x14` |
| Content ID | `0x40`, 36 bytes | e.g. `IV0000-VICS00001_00-PKGSTORE00000000` |
| Content type, flags | `0x74`, `0x78` | `0x1A` is a game or app |
| PFS image offset, size | `0x410` (u64), `0x418` (u64) | |
| Package size | `0x430` (u64) | |

Entries with ids from `0x1000` up are metadata files stored in the clear: `0x1000` is `param.sfo`, `0x1200` `icon0.png`, `0x1006` `pic1.png`, `0x1220` `pic0.png`. That is how `ps4-pkg-db` reads titles and icons without any keys.

The game's files (`eboot.bin`, `sce_module/`, assets) are not entries. They live in the **PFS image**, a file system with its own header: magic `20130315` (decimal, u64 little-endian at `+0x08`) and a mode field (u16 at `+0x1C`) with the flags `0x1` signed, `0x2` 64-bit, `0x4` encrypted, `0x8` always set.

vic's homebrew package has mode `0xD`: signed and **encrypted**. So even a homebrew "fake" package keeps its files encrypted, and reading them needs key material. For fake packages that is a well-known key set, which LibOrbisPkg and shadPS4 embed; for retail packages it is console and licence keys. All of it is Sony key material.

## What this means for us

- Reading the PKG header and its clear metadata (title, content ID, icons) needs no keys and is fine.
- Reading the files inside needs decryption with Sony keys. [legal.md](legal.md) forbids committing keys and writing code that decrypts protected content. So running a `.pkg` directly is out, unless vic decides otherwise.
- The unpacked form is what the loader takes: an `app0` folder with `eboot.bin`, `sce_module/` and `sce_sys/param.sfo`. OpenOrbis builds exactly that folder before it packs (`pkg_scratch/` in vic's build), and external tools can extract a package into it.

## PS5

Not researched yet. PS5 dumps usually arrive as `PPSAxxxxx-app0` folders (see [loader.md](loader.md)), and PS5 packages use a different format from PS4's. Whether open homebrew PS5 packages exist, and whether any are unencrypted, is unknown.
