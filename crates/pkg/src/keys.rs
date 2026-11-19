//! Keys derived from a package's passcode. No console key material is involved;
//! see `docs/packages.md` for the derivation and its sources.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::error::invalid;
use crate::{Error, Result};

/// The passcode homebrew toolchains (OpenOrbis) use unless told otherwise.
pub const DEFAULT_PASSCODE: Passcode = Passcode([b'0'; 32]);

const CONTENT_ID_PADDED: usize = 0x30;
const CHECK_TABLE: usize = 0x20;

/// The derived key that roots the PFS image keys.
pub const EKPFS_INDEX: u32 = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Passcode([u8; 32]);

impl Passcode {
    pub fn new(s: &str) -> Result<Self> {
        let bytes: [u8; 32] = s
            .as_bytes()
            .try_into()
            .map_err(|_| invalid("a passcode is exactly 32 ASCII characters"))?;
        if !bytes.is_ascii() {
            return Err(invalid("a passcode is exactly 32 ASCII characters"));
        }
        Ok(Self(bytes))
    }
}

impl std::fmt::Debug for Passcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Passcode(..)")
    }
}

fn padded_content_id(content_id: &str) -> [u8; CONTENT_ID_PADDED] {
    let mut out = [0u8; CONTENT_ID_PADDED];
    let n = content_id.len().min(CONTENT_ID_PADDED);
    out[..n].copy_from_slice(&content_id.as_bytes()[..n]);
    out
}

/// `SHA-256(SHA-256(be32(index)) || SHA-256(padded content ID) || passcode)`.
pub fn derive_key(content_id: &str, passcode: &Passcode, index: u32) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(Sha256::digest(index.to_be_bytes()));
    h.update(Sha256::digest(padded_content_id(content_id)));
    h.update(passcode.0);
    h.finalize().into()
}

/// Checks a derived key against the entry-keys entry (id `0x10`), which holds
/// `SHA-256(key) XOR key` for each index.
pub fn check_derived_key(
    entry_keys: &[u8],
    content_id: &str,
    key: &[u8; 32],
    index: u32,
) -> Result<()> {
    let at = CHECK_TABLE + 0x20 * index as usize;
    let stored = entry_keys
        .get(at..at + 0x20)
        .ok_or_else(|| invalid("entry-keys entry is too short"))?;
    if entry_keys[..0x20] != Sha256::digest(padded_content_id(content_id))[..] {
        return Err(invalid("entry-keys entry belongs to another content ID"));
    }
    let digest = Sha256::digest(key);
    let check = digest.iter().zip(key).map(|(d, k)| d ^ k);
    if !check.eq(stored.iter().copied()) {
        return Err(Error::WrongPasscode);
    }
    Ok(())
}

/// The AES-XTS key pair of a PFS image.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PfsKeys {
    pub tweak: [u8; 16],
    pub data: [u8; 16],
}

impl std::fmt::Debug for PfsKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PfsKeys(..)")
    }
}

impl PfsKeys {
    /// `HMAC-SHA256(key = ekpfs, le32(1) || seed)`: tweak key first, data key second.
    /// With `new_crypt`, the HMAC key is `HMAC-SHA256(ekpfs, seed)` instead (LibOrbisPkg's
    /// "new crypt"; not yet seen in a real package).
    pub fn derive(ekpfs: &[u8; 32], seed: &[u8; 16], new_crypt: bool) -> Self {
        let key: [u8; 32] = if new_crypt {
            hmac_sha256(ekpfs, &[seed])
        } else {
            *ekpfs
        };
        let out = hmac_sha256(&key, &[&1u32.to_le_bytes(), seed]);
        Self {
            tweak: out[..16].try_into().unwrap(),
            data: out[16..].try_into().unwrap(),
        }
    }
}

fn hmac_sha256(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC takes any key length");
    for p in parts {
        mac.update(p);
    }
    mac.finalize().into_bytes().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CID: &str = "UP0000-TEST00000_00-TESTTESTTESTTEST";

    // Expected values computed independently with Python's hashlib and hmac.
    fn hex<const N: usize>(s: &str) -> [u8; N] {
        let v: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect();
        v.try_into().unwrap()
    }

    #[test]
    fn derives_keys_from_content_id_and_passcode() {
        assert_eq!(
            derive_key(CID, &DEFAULT_PASSCODE, 1),
            hex("9bba34e2e3adada693e4a1eea77c05fbf41b8099aeb5fb342308b310125ea125")
        );
        assert_eq!(
            derive_key(CID, &DEFAULT_PASSCODE, 3),
            hex("7ad8952cacbe62fee09d3dc46cdc113453c95fdbc2023b7dd1f17abf2d5c05f1")
        );
    }

    #[test]
    fn derives_pfs_keys_from_ekpfs_and_seed() {
        let ekpfs = derive_key(CID, &DEFAULT_PASSCODE, EKPFS_INDEX);
        let seed: [u8; 16] = core::array::from_fn(|i| i as u8);
        let keys = PfsKeys::derive(&ekpfs, &seed, false);
        assert_eq!(keys.tweak, hex("8fae73c4f6d0e2f65d978d75fa971c88"));
        assert_eq!(keys.data, hex("ce9945551b7ccfb868183462b446885f"));
        let keys = PfsKeys::derive(&ekpfs, &seed, true);
        assert_eq!(keys.tweak, hex("03768683611d097bd8696880690128a8"));
        assert_eq!(keys.data, hex("b0f61696375321f203eedf11896cf09d"));
    }

    fn entry_keys_for(key: &[u8; 32], index: u32) -> Vec<u8> {
        let mut e = vec![0u8; 0x800];
        e[..0x20].copy_from_slice(&Sha256::digest(padded_content_id(CID)));
        let d = Sha256::digest(key);
        let at = CHECK_TABLE + 0x20 * index as usize;
        for i in 0..32 {
            e[at + i] = d[i] ^ key[i];
        }
        e
    }

    #[test]
    fn accepts_the_right_passcode_and_rejects_others() {
        let ekpfs = derive_key(CID, &DEFAULT_PASSCODE, 1);
        let entry = entry_keys_for(&ekpfs, 1);
        assert!(check_derived_key(&entry, CID, &ekpfs, 1).is_ok());

        let other = derive_key(
            CID,
            &Passcode::new("0123456789abcdef0123456789abcdef").unwrap(),
            1,
        );
        assert!(matches!(
            check_derived_key(&entry, CID, &other, 1),
            Err(Error::WrongPasscode)
        ));
    }

    #[test]
    fn rejects_entry_keys_of_another_content_id() {
        let ekpfs = derive_key(CID, &DEFAULT_PASSCODE, 1);
        let entry = entry_keys_for(&ekpfs, 1);
        assert!(matches!(
            check_derived_key(&entry, "UP0000-OTHER0000_00-TESTTESTTESTTEST", &ekpfs, 1),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn passcodes_must_be_32_ascii_characters() {
        assert!(Passcode::new(&"0".repeat(31)).is_err());
        assert!(Passcode::new(&"0".repeat(33)).is_err());
        assert!(Passcode::new(&"é".repeat(16)).is_err());
    }
}
