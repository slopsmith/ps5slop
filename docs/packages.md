# Packages

What a `.pkg` holds and how we read it. `crates/pkg` reads fake-signed PS4 packages (opened with their passcode) and PS5 packages whose outer image is stored in the clear, including the Kraken-compressed inner image. `ps5slop-pkg` (`ls`, `cat`, `extract`, `naps`) is its command line.

## What is supported, and why

- **Inputs:** fake-signed packages (passcode all zeros unless the user gives another) and dumped `app0` folders. Retail packages are not supported: their keys come from console and licence material we do not have and will not embed.
- **Passcode only.** A fake-signed PS4 package's keys derive from its content ID and the passcode its builder chose, with no Sony key material. Fake packages without a known passcode are opened by other tools with an RSA private key that is Sony key material; we do not embed or use it.
- **PS5 packages** are read when their outer image is stored in the clear (every PS5 fake package we have seen). Encrypted PS5 images fail at start with a message.
- **Run in place.** `ps5slop run GAME.pkg` serves `/app0` straight from the package; nothing is extracted. Reason: PS5 packages are tens of GB (one unpacks to 149 GB), a game reads only part of its files, and the reader already reads at random offsets. The cost: package files have no host path (`Vfs::resolve` gives none; use `Vfs::read`).
- Legal limits on what we may read or run while working these formats out are in [legal.md](legal.md).

## Running from a package

- `/app0` is a read-only `ImageFs` mount (`crates/hle/src/vfs.rs`) backed by `PackageFs` in `crates/ps5slop/src/title.rs`. Files are read on demand through the PFS block lists, with PFSC and XTS where the image needs them, and on PS5 the NAPS layout and Kraken, decoding only the 256 KiB blocks a read touches (with a 16-block cache). Directory listings are cached.
- `sce_sys/` also shows the clear metadata entries the image does not hold (`param.sfo`, `param.json`, icons), the same set `extract` writes.
- The loader reads `eboot.bin` and modules from the package the same way.
- Per-title data folders are named by the title id: `param.json`'s `titleId` (PS5), `param.sfo`'s `TITLE_ID` (PS4), else the one in the content id.
- **PS4 patches.** A patch package (`param.sfo` `CATEGORY` `gp`, psdevwiki "PARAM.SFO"; a base game is `gd`) holds only changed files. `ps5slop run BASE.pkg --patch PATCH.pkg` lays the patch over the base (`Layered` in `title.rs`): paths the patch holds come from it, the rest from the base, and listings merge. Given a patch alone, the base is looked for beside it (and beside a link's target): the first package by name with the same title id that is not a patch. Both open with the same passcode. That the console also takes the patch's `sce_sys` files is assumed.

Verified: every file read through `/app0` matches an extraction of the same package (ignored test `title::tests::package_files_match_their_extraction`, needs vic's packages).

## PS4 PKG

Sources: vic's own `ps4-pkg-db` (`pkg.py`, `sfo.py`), psdevwiki's [PKG files](https://www.psdevwiki.com/ps4/PKG_files) page (a saved copy; the site blocks automated fetches), LibOrbisPkg's `PfsStructs.cs`, `PfsReader.cs`, `PFSCReader.cs` and `Crypto.cs` (LGPL-3.0, read for facts only). Checked against vic's OpenOrbis-built packages.

| Field | Where | Notes |
| --- | --- | --- |
| Magic | `0x00`, `7F 43 4E 54` (`\x7FCNT`) | Fields are big-endian |
| Entry count, entry table offset | `0x10` (u32), `0x18` (u32) | 32-byte entries: id `+0x00`, data offset `+0x10`, size `+0x14` |
| Content ID | `0x40`, 36 bytes | |
| Content type, flags | `0x74`, `0x78` | `0x1A` is a game or app |
| PFS image offset, size | `0x410`, `0x418` (u64) | |
| Package size | `0x430` (u64) | |

Entries from `0x1000` up are metadata stored in the clear (`0x1000` `param.sfo`, `0x1200` `icon0.png`, ...), so titles and icons read without keys. The game's files live in the **PFS image**: magic `20130315` (u64 LE at `+0x08`), mode u16 at `+0x1C` (`0x1` signed, `0x2` 64-bit, `0x4` encrypted, `0x8` always set). Even homebrew packages are signed and encrypted (mode `0xD`).

### Keys from the passcode

1. `DK(i) = SHA-256( SHA-256(be32(i)) || SHA-256(content ID padded to 0x30) || passcode )`. `DK(1)` is the **EKPFS**.
2. **Check:** the entry-keys entry (id `0x10`) starts with SHA-256 of the padded content ID, then for each i a check value `SHA-256(DK(i)) XOR DK(i)` at `0x20 + 0x20*i`. A wrong passcode fails here, before any decryption.
3. **PFS keys:** `HMAC-SHA256(key = EKPFS, message = le32(1) || seed)`, seed = 16 bytes at `+0x370` of the PFS header. First 16 bytes: XTS tweak key; last 16: data key. psdevwiki says the seed is the HMAC key; LibOrbisPkg's order is the one that decrypts. LibOrbisPkg's "new crypt" variant is not implemented (no package of ours uses it).
4. **Decryption:** AES-128-XTS over 0x1000-byte sectors, tweak = sector index from the image start. The first 16 sectors (the header block) are plain.
5. The outer PFS holds one file, `pfs_image.dat`, a PFSC-compressed inner PFS with the game's files.

Code: `keys.rs`, `xts.rs`.

### PFS and PFSC

- **Superblock** (LE): version, magic, mode `0x1C`, block size `0x20` (0x10000 in practice), inode count `0x30`, inode block count `0x40`, seed `0x370`. Inodes fill blocks from block 1: 0xA8 bytes unsigned, 0x2C8 signed (each block pointer carries a 32-byte signature). Images with 64-bit inodes are refused.
- **Inode:** flags `0x04` (bit 0 compressed), two sizes `0x08` and `0x10`, block count `0x60`, 12 direct and 5 indirect block pointers. A second direct pointer of `0xFFFFFFFF` (or one block) means contiguous. Signed images can list blocks one by one; indirect blocks then hold 36-byte (signature, block) pairs. For a PFSC file the second size is the decompressed length.
- **Directories:** entries of inode (u32), type (u32: 2 file, 3 directory, 4 `.`, 5 `..`), name length, entry size, name. Inode 0 is the superroot; files live under `uroot`.
- **PFSC:** header (magic `PFSC`, block size, offset of a u64 block-offset table, decompressed length); blocks are stored raw (exactly one block long), as zeros (longer), or as deflate after a 2-byte zlib header (shorter).

Code: `pfs.rs`, `pfsc.rs`.

## PS5 packages

Sources: LibProsperoPKG's [ps5-pkg-format.md](https://github.com/SvenGDK/LibProsperoPKG/blob/main/docs/ps5-pkg-format.md) (GPL-3.0, read for facts only; partly inconsistent about keys), pkg-to-anyps5's wiki page [Package-Format](https://github.com/awake-devel/pkg-to-anyps5/wiki/Package-Format), kraken-decode's `docs/FORMAT.md` (MIT), Fabian Giesen's posts on Oodle's entropy coding. Everything below was checked on the PS5 fake packages we have.

### Container and outer image

- A finished package is a **finalized image** (`\x7FFIH`, LE header). `0x10`/`0x18` are the PFS segment's offset and size, `0x20` the offset of the outer PFS superblock, `0x58` the embedded `\x7FCNT` container (read from there to the end of the file; entry offsets count from its start). `0xA0` is not the container's size. `param.json` (entry `0x2000`) is in the clear.
- **Outer PFS:** superblock version 2, mode `0xD`, but stored in the clear: its seed reads `PPRPLAIN-NOAUTH!`, which we take to mean "not encrypted" (`PLAIN_SEED` in `pfs.rs`). The superblock sits near the end of the image, after the file data; block numbers still count from the image start. Inodes use the signed 0x2C8-byte layout.
- `uroot` holds `pfs_image.dat` (compressed; second size = decompressed) and `naps_pkg_layout.dat`.
- Per LibProsperoPKG, encrypted PS5 images use the same XTS scheme with a SHA3-256 key schedule, and passcode images derive the EKPFS from content ID and passcode. Not implemented or tested: none of our packages need it.

### The inner image

Code: `naps.rs` (layout), `kraken.rs` (codec), `inner.rs` (the decompressed view), flat mode in `pfs.rs`. Rules are tagged by origin: **[prose]** from the sources above, **[oracle]** checked against a black-box decoder's output, **[bytes]** fitted on the packages' data and checked by decoding.

Verified: `ps5slop-pkg naps --verify` decodes every block of each of our PS5 packages without an error, and extracted files match a black-box extractor's output byte for byte (all files of the smallest package, samples and the largest files of the others).

**Layout:**

- The inner image is "data-first" **[prose]**: files back to back in a logical image, inner PFS metadata in the last region. Compression works on 256 KiB blocks counted from each file's start, each one or two 128 KiB chunks **[bytes]**.
- `naps_pkg_layout.dat` **[prose for section order, bytes for fields]**: a 16-byte header (file count u24 at 0, codec byte 3 = 2, logical block count u32 at 4, outer block count u24 at 8), 8-byte outer-block digests, a file-offset table (file count + 1 rows of 5-byte logical offset + kind byte; kind `0x40` marks zero-filled regions and the last row), u2c groups (one 10-byte entry per 8 logical blocks: u24 index of the first block's first record, then seven u8 offsets), then the records.
- **Mount size** is the file-offset table's last row. It must fall inside the last logical block but need not fill it.
- **Records** are 72-bit LE values starting 4 bytes into the record section, one every 9 bytes **[bytes]**:

| Bits | Normal record | Anchor (bit 18 set) |
| --- | --- | --- |
| 0-15 | stored offset bits 0-15 | stored offset bits 0-15 |
| 16-17 | stored offset bits 16-17 | |
| 18 | 0 | 1 |
| 20-37 | start within its logical block | stored offset bits 16-33 |
| 38-54 | first chunk's stored size - 1 | bits 38-45: stored offset bits 34-41 |
| 55-71 | chunk flags | |

- **Stored offsets [bytes]:** an anchor gives a whole offset; a normal record gives the low 18 bits, completed from a reference (the last anchor, or the previous record's offset + its first chunk's size) as the first value with those low bits at or above reference - 64 KiB. The anchor's high field holds at least 3 used bits (packages over 64 GiB need bit 36); we read 8 (offsets up to 4 TiB). A piece's stored size runs to where the next record starts; when an anchor follows, the piece ends at the anchor's low 16 bits, above its first chunk and at or below the next piece stored after it. Identical blocks are stored once and shared (deduplicated records point backwards).
- **Chunk flags [bytes]:** bit 2: chunk 1 is LZ, bit 1: its literals are deltas; bit 5: chunk 2 is LZ, bit 4: deltas. Otherwise a chunk is stored as is (stored size = decoded size) or as one entropy-coded array (an all-zero chunk is 8 bytes).
- **Lookup [bytes]:** the piece for logical offset `L` is the non-anchor record among block `L / 256 KiB`'s records whose start equals `L % 256 KiB`.

**Kraken as these packages use it:**

- **Chunk [bytes, oracle]:** the first chunk of a block starts with 8 raw bytes; then (both chunks) a flag byte with the top bit set whose low 5 bits count trailing bytes (with bit 5 set a second byte follows: count = `(b0 & 0x1f) + 32 * (b1 + 1)`); then four arrays (literals, commands, offset codes, long lengths), the offset bit streams, and the trailing bytes. The 3-byte Oodle chunk header is not stored (NAPS replaces it).
- **Arrays [prose for header and three-stream layout, oracle for the rest]:** only stored (type 0) and three-stream Huffman (type 2) occur. Huffman code descriptions:
  - `00` **[oracle]**: 8-bit symbol count n; n = 1 gives an 8-bit symbol repeated; else a 3-bit width w and per symbol 8 bits of symbol and w bits of length - 1.
  - `01` **[oracle, worked out by crafting inputs]**: k (2 bits), a flag (the first span is used symbols), then alternating runs of used and unused symbols, each coded `(1 << (u+1) | u+1 bits) - 1` with u in unary, until symbol 256. Each used symbol is followed by a Rice(k) value v and a zigzag step (even +v/2, odd -(v+1)/2) against a running estimate e: length = step + (e >> 2) + 1, then e = e + length - ((e + 3) >> 2), e starting at 30.
- **Commands [bytes]:** bits 0-1 literal run (3 = long), bits 2-5 match length - 2 (15 = long), bits 6-7: 3 = new offset, 0-2 = reuse the 1st/2nd/3rd recent offset (move to front). Long lengths come from the length array in order (literal run + 3, match + 17); a length byte of 255 adds an escape value. Recent offsets start at 8. Delta literals add the byte at the most recent offset.
- **Offsets [bytes]:** new offset = 8 + (high << 4 | code & 15), with k = code >> 4 and high = (next 4 + k bits) + ((2^k - 1) << 4). High bits alternate between two bit streams in the tail, most significant first: one read forward from its start, one backward from its end.
- **Escapes [bytes]:** in the trailing bytes, alternating forward and backward; each is Exp-Golomb of order 6 (count zeros z, skip the 1, read z + 6 bits, value = that number with its leading 1, minus 64).
- **Inner PFS [prose for flat mode, bytes for the rest]:** superblock at the start of the last region (version 2, mode `0x18`, block size 0x10000), unsigned 0xA8-byte inodes from the next block, each holding its data's byte address in the logical image at `+0x60`; directory entries as on PS4. The superroot holds `inode_flat_path_table`, `apr_flat_path_table`, `afid_to_ino_table` and `uroot`. Executables are stored raw.

**Scope rule.** Kraken features are added only when a package we have uses them: code for a feature no package uses cannot be checked against real data, and would be unverified code on every read's path. Not handled: code description `10`, array types other than 0 and 2, inner images in other modes. They fail with "unsupported" naming the feature. Malformed Huffman descriptions are errors, not panics. Garbage array types (1, 3) in a block have so far meant a wrong stored offset, not a new feature.

The PS4 cannot install PS5 packages, so PS5 package work happens on the Mac only.
