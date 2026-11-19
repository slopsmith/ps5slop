# Packages

What a `.pkg` holds, and how we open fake-signed ones. `crates/pkg` implements this for PS4 packages. For PS5 packages it reads the container, its metadata and an outer PFS image stored in the clear, but not yet the compressed image that holds the game's files (see "PS5 packages we have looked at").

## PS4 PKG

Sources: vic's own `ps4-pkg-db` (`pkg.py`, `sfo.py`), the psdevwiki [PKG files](https://www.psdevwiki.com/ps4/PKG_files) page (a copy vic saved; the site blocks automated fetches), LibOrbisPkg's `PfsStructs.cs` and `Crypto.cs` (LGPL-3.0: read for facts only), and vic's own homebrew package `ps4-pkg-store/ps4build/pkgstore.pkg`, built with OpenOrbis, all read on 2026-11-19.

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

vic's homebrew package has mode `0xD`: signed and **encrypted**. So even a homebrew "fake" package keeps its files encrypted.

### Opening a fake-signed package with its passcode

Every package has a 32-character passcode chosen by whoever built it. OpenOrbis uses all zeros by default (vic's `project.gp4` has that). From the passcode we can derive the keys ourselves, with no Sony key material:

1. **Derived keys:** `DK(i) = SHA-256( SHA-256(be32(i)) || SHA-256(content ID padded to 0x30 bytes) || passcode )`. `DK(1)` is the **EKPFS**, the root of the PFS keys.
2. **Check:** the entry-keys entry (id `0x10`, 0x800 bytes) starts with SHA-256 of the padded content ID, then for each index i a 32-byte check value `SHA-256(DK(i)) XOR DK(i)` at `0x20 + 0x20*i`. A wrong passcode fails this check before we try to decrypt anything.
3. **PFS keys:** `HMAC-SHA256(key = EKPFS, message = le32(1) || seed)`, where the 16-byte seed sits at `+0x370` in the PFS header. The first 16 bytes of the result are the XTS tweak key, the last 16 the data key.
4. **Decryption:** AES-128-XTS over 0x1000-byte sectors, with the sector's index from the start of the PFS image as the tweak. The first 16 sectors (0x10000 bytes, the header block) are not encrypted.
5. Inside the outer PFS is one file, `pfs_image.dat`, a PFSC-compressed inner PFS that holds the game's files.

### PFS and PFSC

From LibOrbisPkg's `PfsStructs.cs`, `PfsReader.cs` and `PFSCReader.cs` (facts only), checked against vic's package:

- **Superblock** (little-endian): version 1, magic, mode at `0x1C`, block size at `0x20` (0x10000 in practice), inode count at `0x30`, inode block count at `0x40`, seed at `0x370`. Inodes fill blocks from block 1 on: 0xA8 bytes each in unsigned images, 0x2C8 in signed ones (each block pointer carries a 32-byte signature). Images with the 64-bit flag use other inode layouts; we refuse them for now.
- **Inode:** flags at `0x04` (bit 0: compressed), two sizes at `0x08` and `0x10`, block count at `0x60`, then 12 direct and 5 indirect block pointers. A file whose second direct pointer is `0xFFFFFFFF` (or that has one block) is contiguous. Signed images can list blocks one by one, with indirect blocks holding 36-byte (signature, block) pairs.
- **Sizes:** the first size is what the blocks store. For the PFSC file in vic's package the second size (LibOrbisPkg calls it `SizeCompressed`) is the *decompressed* length, which there is smaller than the stored one.
- **Directories:** entries of inode (u32), type (u32: 2 file, 3 directory, 4 `.`, 5 `..`), name length, entry size, then the name. Inode 0 is the superroot; the files live under its `uroot` directory.
- **PFSC:** a header (magic `PFSC`, block size, offset of a table of u64 block offsets, decompressed length), then blocks stored raw (exactly one block long), as zeros (longer), or as deflate data after a 2-byte zlib header (shorter).

The outer image in vic's package is signed and encrypted with an all-zero seed; the inner one is what holds `eboot.bin`.

### Extraction

`ps5slop-pkg extract PKG DIR [--passcode P]` writes the inner image's files, then the clear `sce_sys` entries (`param.sfo`, icons, PlayGo files) that the image does not already hold. It refuses to overwrite files.

**Checked on 2026-11-19** against vic's `pkgstore.pkg`: every file in the image matches `ps4build/pkg_scratch/` (the folder it was packed from) byte for byte. The only differences are ones the packer makes: it adds `keystone` and the PlayGo files, adds `PUBTOOLINFO` and `PUBTOOLVER` to `param.sfo`, and does not pack the `project.gp4` build file. The package also carries `sce_sys/about/right.sprx` from OpenOrbis, whose origin is unknown.

**Verified on 2026-11-19** against vic's `pkgstore.pkg` with a throwaway script (Python `hashlib` and LibreSSL's `aes-128-xts`): `DK(1)` to `DK(3)` match the package's check values, and sectors from 16 on decrypt to structured data and zero runs with these keys, while other key orders give noise.

Sources disagree on step 3: the psdevwiki PKG page (vic's saved copy) says the seed is the HMAC key, while LibOrbisPkg's `PfsGenCryptoKey` uses the EKPFS as the key and `le32(1) || seed` as the message. LibOrbisPkg's order is the one that decrypts. LibOrbisPkg also has a "new crypt" variant (EKPFS first replaced by `HMAC-SHA256(EKPFS, seed)`), which this package does not use.

### What stays out

- **Fake packages without their passcode.** Tools open these by decrypting the image-key entry (`0x20`) with an RSA private key for derived key 3. That key is Sony key material (psdevwiki lists it as "private key known"). We do not embed or use it.
- **Retail packages.** Their keys come from console and licence (`rif`) material. Out of scope.

## What this means for us

- Reading the PKG header and its clear metadata (title, content ID, icons) needs no keys.
- A fake-signed package opens with its passcode (the user supplies it, all zeros by default), with no Sony keys. This is allowed as a convenience feature (see [decisions.md](decisions.md)).
- Supported inputs are fake-signed packages (passcode all zeros unless the user gives another) and dumped `app0` folders (`eboot.bin`, `sce_module/`, `sce_sys/`).

## Running from a package

`ps5slop run GAME.pkg` reads the package in place; nothing is extracted (see [decisions.md](decisions.md)).

- `/app0` is a read-only mount backed by the package's app image (`crates/ps5slop/src/title.rs`, `PackageFs`, behind the `hle::ImageFs` trait in `crates/hle/src/vfs.rs`). Files are read on demand through the PFS block lists (and PFSC, XTS where the image needs them); directory listings are cached once read.
- `sce_sys/` also shows the clear metadata entries the image does not hold itself (`param.sfo`, `param.json`, icons, ...), the same set `extract` writes.
- The loader reads `eboot.bin` and the title's modules (`*.prx` at the root and in `sce_module/`) from the package the same way.
- Packages that need keys we do not have (retail images, encrypted PS5 images) or a format we do not read fail at start with a message that says which.
- Per-title data folders (`/data`, save data, `/temp0`) are named by the title id: `param.json`'s `titleId` (PS5), `param.sfo`'s `TITLE_ID` (PS4), else the one in the content id (`UP0000-PPSA01234_00-...`). Folder runs use the same id from `sce_sys/`.

**Checked on 2026-11-20** with vic's two OpenOrbis packages (`pkgstore.pkg`, and `example.pkg` from `funny-ps4-engine`): every file read through `/app0` matches an extraction of the same package byte for byte (the ignored test `title::tests::package_files_match_their_extraction`), and `ps5slop run --headless` on each package gets as far as on its extracted folder (both stop at the unimplemented `sceMsgDialogInitialize`). No PS5 title has run from a package yet: see below.

## PS5

Source: LibProsperoPKG's [ps5-pkg-format.md](https://github.com/SvenGDK/LibProsperoPKG/blob/main/docs/ps5-pkg-format.md) (GPL-3.0: read for facts only; unverified by us, no PS5 package tested yet).

- A finished PS5 package is a **finalized image** (magic `\x7FFIH`, little-endian header) with four segments: FIH (header and digest table), PFS (the encrypted outer image), SC (an embedded `\x7FCNT` container like the PS4's), and SI (install metadata).
- The PFS uses the same scheme: AES-XTS over 0x1000-byte sectors, a plain 0x10000-byte header block, EKPFS plus the superblock seed giving the keys, but with a SHA3-256 based key schedule. Digests are SHA3-256.
- Debug or passcode images: the EKPFS comes from the content ID and passcode (the writeup says SHA-256 first, SHA3-256 as a fallback). Retail outer images use an image key delivered through the console; not derivable, out of scope.
- The writeup is partly inconsistent about which image uses which key (section 3.3 against section 11). Treat it as a lead to test, not a fact.

### PS5 packages we have looked at

Two PS5 fake packages from vic's downloads, Superliminal (`PPSA06084`, 4.5 GB) and Astro Bot (`PPSA21564`, 38 GB), looked at on 2026-11-20 with a hex viewer and our own reader. What they show:

- **FIH:** `0x10` and `0x18` are the PFS segment's offset (`0x10000`) and size, and `0x58` the embedded container's offset, as LibProsperoPKG says. `0xA0` is not the container's size in either file: it equals the stored size of `pfs_image.dat`. The u64 at `0x20` is the offset of the PFS superblock in the file. We read the container from `0x58` to the end of the file.
- **Embedded container:** a `\x7FCNT` like the PS4's. Entry offsets count from the container's start. `param.json` (entry `0x2000`) is stored in the clear; we read the title id from it.
- **Outer PFS:** superblock version 2, mode `0xD` (signed and the encrypted flag), yet stored in the clear: its seed reads `PPRPLAIN-NOAUTH!`, and we take that seed to mean "not encrypted". The superblock does not start the image: it sits near the end, after the file data (block `0x10707` of `0x10733` in Superliminal), with the inode block right after it. Block numbers still count from the image's start. Inodes use the signed PS4 layout (0x2C8 bytes).
- **Contents:** the superroot holds `inode_flat_path_table` and `uroot`; `uroot` holds `pfs_image.dat` (compressed flag set; Superliminal: 4.4 GB stored, 14.9 GB as its second size) and `naps_pkg_layout.dat` (1.2 MB, unknown format).
- **`pfs_image.dat`** has no PFSC header. It starts with a PS5 SELF in the clear (`54 14 F5 EE`), and elsewhere holds what looks like LZ-compressed PFS metadata (`inode_flat_path_table`, `uroot`, `sce_module`, `eboot.bin` appear split up by back-references). We do not know this compression format or where its block table lives, so `ps5slop` stops there with "compressed with a scheme ps5slop does not know yet". Unknown: the codec (Kraken-like? LZ4-like?), the role of `naps_pkg_layout.dat`, and whether other PS5 packages share the plain seed.

The PS4 cannot install PS5 packages, so any PS5 package work happens on the Mac. PS5 game dumps usually arrive as `PPSAxxxxx-app0` folders anyway (see [loader.md](loader.md)).
