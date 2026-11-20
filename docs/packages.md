# Packages

What a `.pkg` holds, and how we open fake-signed ones. `crates/pkg` implements this for PS4 packages. For PS5 packages it reads the container, its metadata, an outer PFS image stored in the clear, and the Kraken-compressed inner image that holds the game's files (see "The inner image" under PS5).

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

- `/app0` is a read-only mount backed by the package's app image (`crates/ps5slop/src/title.rs`, `PackageFs`, behind the `hle::ImageFs` trait in `crates/hle/src/vfs.rs`). Files are read on demand through the PFS block lists (and PFSC, XTS where the image needs them; on PS5, the NAPS block map and Kraken, decoding only the 256 KiB blocks a read touches, with a 16-block cache); directory listings are cached once read.
- `sce_sys/` also shows the clear metadata entries the image does not hold itself (`param.sfo`, `param.json`, icons, ...), the same set `extract` writes.
- The loader reads `eboot.bin` and the title's modules (`*.prx` at the root and in `sce_module/`) from the package the same way.
- Packages that need keys we do not have (retail images, encrypted PS5 images) or a format we do not read fail at start with a message that says which.
- Per-title data folders (`/data`, save data, `/temp0`) are named by the title id: `param.json`'s `titleId` (PS5), `param.sfo`'s `TITLE_ID` (PS4), else the one in the content id (`UP0000-PPSA01234_00-...`). Folder runs use the same id from `sce_sys/`.

**Checked on 2026-11-20** with vic's two OpenOrbis packages (`pkgstore.pkg`, and `example.pkg` from `funny-ps4-engine`): every file read through `/app0` matches an extraction of the same package byte for byte (the ignored test `title::tests::package_files_match_their_extraction`), and `ps5slop run --headless` on each package gets as far as on its extracted folder (both stop at the unimplemented `sceMsgDialogInitialize`). **Checked on 2026-11-20** with the two PS5 fake packages: `ps5slop run --headless` loads both straight from the `.pkg`. Superliminal stops at the unimplemented libkernel import with NID `DGMG3JshrZU`, exactly as its extracted folder does; Astro Bot stops at the unimplemented libkernel import `4oXYe9Xmk0Q` (1,292 imports not implemented yet). Format details: "The inner image" below.

### Patches

A PS4 update ships as its own package, a **patch**: `param.sfo`'s `CATEGORY` is `gp` (psdevwiki, "PARAM.SFO"; a base game is `gd`), and its image holds only the files the update changes. On the console the system mounts `/app0` with the patch's files over the base game's. We do the same: `ps5slop run BASE.pkg --patch PATCH.pkg` lays the patch's image over the base's (`Layered` in `crates/ps5slop/src/title.rs`): a path the patch holds comes from the patch, anything else from the base, and directory listings merge. Given a patch alone, `ps5slop run` looks for its base in the same folder, and beside the link's target when the path is a link: the first package by name with the same title id that is not a patch. Both open with the same passcode.

**Checked on 2026-11-20** with Deltarune (`CUSA52463`): the v1.12 patch (`A0112`, 111 files, content flags `0x62300000`) holds new `game.win` files and a fifth chapter, but no `eboot.bin` and no `sce_module/`; those come from the base package (`A0100`, 402 files, flags `0x0A000000`) next to it. Both open with the all-zero passcode. `ps5slop run` on the patch finds the base, loads the base's `eboot.bin` and reads the patch's `game.win` files. Whether the console also takes the patch's `sce_sys` files (its `param.sfo` says 01.12) we assume, unverified.

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
- **Contents:** the superroot holds `inode_flat_path_table` and `uroot`; `uroot` holds `pfs_image.dat` (compressed flag set; Superliminal: 4.4 GB stored, 14.9 GB as its second size) and `naps_pkg_layout.dat` (Superliminal 1.2 MB, Astro Bot 13.9 MB).

### The inner image (`pfs_image.dat` and `naps_pkg_layout.dat`)

`crates/pkg` reads it: `naps.rs` (the layout), `kraken.rs` (the codec), `inner.rs` (the decompressed view), and flat mode in `pfs.rs`. `ps5slop-pkg naps PKG` prints the layout.

**How this was established.** Prose first: pkg-to-anyps5's wiki page [Package-Format](https://github.com/awake-devel/pkg-to-anyps5/wiki/Package-Format) and README, LibProsperoPKG's `docs/`, kraken-decode's `docs/FORMAT.md` (MIT), Fabian Giesen's posts on Oodle's entropy coding. The rest was worked out on the two packages with two programs run as black-box oracles and never read (vic's decision, see [decisions.md](decisions.md)): ooz-wasm (to decode single arrays) and pkg-to-anyps5 built from source (its extracted files as known plaintext). Rules are marked by where they come from: **[prose]**, **[oracle]** (an array or file the oracle decoded), **[bytes]** (fitted on the packages' data, checked by decoding).

**Verified on 2026-11-20:** every file of Superliminal extracted with `ps5slop-pkg extract` is byte-identical to pkg-to-anyps5's output (145 of 145; executables, which pkg-to-anyps5 unwraps to ELF, were checked by running the game instead), Astro Bot's listing matches name for name and size for size (166,703 files), all of its files read without an error (159.7 GB in 7 minutes on an M-series Mac), and 706 of them (random samples plus the three largest, 1.9 GB) are byte-identical. Our decoder, run on 1,800 random Superliminal blocks before the Rust port, matched the plaintext in all of them, without any oracle in the loop.

#### Layout

- The inner image is "data-first" **[prose]**: a logical image in which the files are laid out back to back, with the inner PFS metadata in the last region. Compression works on 256 KiB blocks counted from each file's start, each block made of one or two 128 KiB chunks **[bytes]**.
- `naps_pkg_layout.dat` **[prose for the section order, bytes for the fields]**: a 16-byte header (file count u24 at 0, codec byte 3 (= 2), logical 256 KiB block count u32 at 4, outer block count u24 at 8), 8-byte outer-block digests (all zero here), a file-offset table (file count + 1 rows of 5-byte logical offset + kind byte; kind `0x40` marks zero-filled regions and the last row, the mount size), u2c groups (one 10-byte entry per 8 logical blocks: u24 index of the block's first record, then seven u8 offsets for the other blocks), then the records.
- **Records** are 72-bit little-endian values starting **4 bytes into** the record section (the first four bytes are not part of one), one every 9 bytes **[bytes]**:

| Bits | Normal record | Anchor (bit 18 set) |
| --- | --- | --- |
| 0-15 | low 16 bits of the stored offset | stored offset bits 0-15 |
| 16-17 | stored offset bits 16-17 | — |
| 18 | 0 | 1 |
| 20-37 | where the piece starts within its logical block | stored offset bits 16-33 |
| 38-54 | first chunk's stored size - 1 | bits 38-39: stored offset bits 34-35 (needed in Astro Bot, 37.6 GB) |
| 55-71 | chunk flags | — |

- **Stored offsets [bytes]:** an anchor gives one whole; a normal record gives the low 18 bits, completed from a reference (the last anchor, or the previous record's offset + its first chunk's size): the first value with those low bits at or above reference - 64 KiB. A piece's stored size is where the next record starts; when an anchor follows, the piece ends at the anchor's low 16 bits, above the piece's first chunk and at or below the next piece stored after it (deduplicated pieces in between point backwards; the next one can be 100+ records on). Identical blocks are stored once and shared.
- **Chunk flags [bytes]:** bit 2: chunk 1 is LZ, bit 1: its literals are deltas; bit 5: chunk 2 is LZ, bit 4: deltas; otherwise a chunk is stored as is (stored size = decoded size) or as one entropy-coded array (all-zero chunks are 8 bytes).
- **Lookup:** the piece for a file block at logical offset `L` is the non-anchor record among block `L / 256 KiB`'s records whose start equals `L % 256 KiB` **[bytes]**.

#### Kraken as these packages use it

- **Chunk [bytes, oracle]:** the first chunk of a block starts with 8 bytes copied as is; then (both chunks) a flag byte with the top bit set, whose low 5 bits count trailing bytes (with bit 5 set, a second byte follows and the count is `(b0 & 0x1f) + 32 * (b1 + 1)`); then four arrays: literals, commands, offset codes, long lengths; then the offset bit streams; then the trailing bytes. The 3-byte Oodle chunk header is not stored (NAPS replaces it).
- **Arrays [prose: kraken-decode's header and three-stream layout; oracle: everything checked against ooz-wasm on 1,500 arrays]:** only two kinds occur: stored (type 0) and three-stream Huffman (type 2). Huffman codes are described in one of two forms (kraken-decode documents a third, `10`, which never occurs here and is not implemented):
  - `00` **[oracle]**: an 8-bit symbol count n; if n = 1, the 8-bit symbol (the array is that byte repeated); else a 3-bit width w and, per symbol, 8 bits of symbol and w bits of length - 1.
  - `01` **[oracle, worked out by crafting descriptions and streams]**: k (2 bits), a flag (the first span is a run of used symbols), then alternating runs of used and unused symbols, each run coded `(1 << (u+1) | u+1 bits) - 1` with u in unary (kraken-decode's skip code), until symbol 256. Each used symbol is followed by a Rice(k) value v (unary quotient, k low bits), a zigzag step (even +v/2, odd -(v+1)/2) against a running estimate e: length = step + (e >> 2) + 1, then e = e + length - ((e + 3) >> 2), with e starting at 30.
- **Commands [bytes]:** per command byte: bits 0-1 literal run (3 = long), bits 2-5 match length - 2 (15 = long), bits 6-7: 3 = new offset, 0-2 = reuse the 1st/2nd/3rd most recent offset (move to front). Long lengths come from the length array in order (literal run + 3, match + 17); a length byte of 255 adds an escape value. The three recent offsets start at 8. Literals in delta mode add the byte at the most recent offset.
- **Offsets [bytes]:** new offset = 8 + (high << 4 | code & 15), with k = code >> 4 and high = (next 4 + k bits) + ((2^k - 1) << 4). The high bits alternate between two bit streams in the tail, read most significant first: one forward from its start, one backward from its end.
- **Escapes [bytes]:** in the trailing bytes, alternating forward and backward as above; each is an Exp-Golomb code of order 6 (count zeros z, skip the 1, read z + 6 bits, value = that number with its leading 1, minus 64).
- **Inner PFS [prose: flat mode, bytes: the rest]:** superblock at the start of the last region (version 2, mode `0x18`, block size 0x10000, block count × block size = the mount size), unsigned 0xA8-byte inodes from the next block, each holding its data's byte address in the logical image at `+0x60`; directory entries as on PS4. The superroot holds `inode_flat_path_table`, `apr_flat_path_table`, `afid_to_ino_table` and `uroot`. Executables are stored raw.

#### What is not handled

Code-description form `10`, array types other than 0 and 2, inner images in other modes, and any feature neither package uses. They fail with "unsupported" naming the feature.

The PS4 cannot install PS5 packages, so any PS5 package work happens on the Mac. PS5 game dumps usually arrive as `PPSAxxxxx-app0` folders anyway (see [loader.md](loader.md)).
