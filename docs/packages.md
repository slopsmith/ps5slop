# Packages

What a `.pkg` holds, and how we can open fake-signed ones. Nothing is implemented.

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
5. Inside the outer PFS is one file, `pfs_image.dat`, a PFSC-compressed inner PFS that holds the game's files. Not looked at yet.

**Verified on 2026-11-19** against vic's `pkgstore.pkg` with a throwaway script (Python `hashlib` and LibreSSL's `aes-128-xts`): `DK(1)` to `DK(3)` match the package's check values, and sectors from 16 on decrypt to structured data and zero runs with these keys, while other key orders give noise.

Sources disagree on step 3: the psdevwiki PKG page (vic's saved copy) says the seed is the HMAC key, while LibOrbisPkg's `PfsGenCryptoKey` uses the EKPFS as the key and `le32(1) || seed` as the message. LibOrbisPkg's order is the one that decrypts. LibOrbisPkg also has a "new crypt" variant (EKPFS first replaced by `HMAC-SHA256(EKPFS, seed)`), which this package does not use.

### What stays out

- **Fake packages without their passcode.** Tools open these by decrypting the image-key entry (`0x20`) with an RSA private key for derived key 3. That key is Sony key material (psdevwiki lists it as "private key known"). We do not embed or use it.
- **Retail packages.** Their keys come from console and licence (`rif`) material. Out of scope.

## What this means for us

- Reading the PKG header and its clear metadata (title, content ID, icons) needs no keys.
- A fake-signed package opens with its passcode (the user supplies it, all zeros by default), with no Sony keys. This is allowed as a convenience feature (see [decisions.md](decisions.md)).
- The loader itself still takes an unpacked `app0` folder: `eboot.bin`, `sce_module/`, `sce_sys/`. Opening a package means extracting (or mounting) it into that form.

## PS5

Source: LibProsperoPKG's [ps5-pkg-format.md](https://github.com/SvenGDK/LibProsperoPKG/blob/main/docs/ps5-pkg-format.md) (GPL-3.0: read for facts only; unverified by us, no PS5 package tested yet).

- A finished PS5 package is a **finalized image** (magic `\x7FFIH`, little-endian header) with four segments: FIH (header and digest table), PFS (the encrypted outer image), SC (an embedded `\x7FCNT` container like the PS4's), and SI (install metadata).
- The PFS uses the same scheme: AES-XTS over 0x1000-byte sectors, a plain 0x10000-byte header block, EKPFS plus the superblock seed giving the keys, but with a SHA3-256 based key schedule. Digests are SHA3-256.
- Debug or passcode images: the EKPFS comes from the content ID and passcode (the writeup says SHA-256 first, SHA3-256 as a fallback). Retail outer images use an image key delivered through the console; not derivable, out of scope.
- The writeup is partly inconsistent about which image uses which key (section 3.3 against section 11). Treat it as a lead to test, not a fact.

The PS4 cannot install PS5 packages, so any PS5 package work happens on the Mac. PS5 game dumps usually arrive as `PPSAxxxxx-app0` folders anyway (see [loader.md](loader.md)).
