//! NIDs and the short ids that name libraries and modules in symbol strings.
//! The algorithm is described in `docs/loader.md`.

use sha1::{Digest, Sha1};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

/// Appended to a name before hashing it.
const SUFFIX: [u8; 16] = [
    0x51, 0x8d, 0x64, 0xa6, 0x35, 0xde, 0xd8, 0xc1, 0xe6, 0xb0, 0x39, 0xb1, 0xc3, 0xe5, 0x52, 0x30,
];

/// The 11-character NID of a symbol name.
pub fn nid(name: &str) -> String {
    let digest = Sha1::new()
        .chain_update(name)
        .chain_update(SUFFIX)
        .finalize();
    let v = u64::from_le_bytes(digest[..8].try_into().unwrap());
    let mut out = String::with_capacity(11);
    for i in 0..10 {
        out.push(ALPHABET[((v >> (58 - 6 * i)) & 0x3f) as usize] as char);
    }
    out.push(ALPHABET[((v & 0xf) * 4) as usize] as char);
    out
}

/// Writes a library or module id as 1 to 3 characters, most significant first.
pub fn encode_id(id: u16) -> String {
    let digits = match id {
        0..0x40 => 1,
        0x40..0x1000 => 2,
        _ => 3,
    };
    (0..digits)
        .rev()
        .map(|d| ALPHABET[((id >> (6 * d)) & 0x3f) as usize] as char)
        .collect()
}

pub fn decode_id(s: &str) -> Option<u16> {
    if s.is_empty() || s.len() > 3 {
        return None;
    }
    s.bytes().try_fold(0u16, |acc, c| {
        let digit = ALPHABET.iter().position(|&a| a == c)? as u16;
        Some((acc << 6) | digit)
    })
}

/// A symbol string `NID#LIB#MOD`, split up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolRef<'a> {
    pub nid: &'a str,
    pub library: u16,
    pub module: u16,
}

impl<'a> SymbolRef<'a> {
    pub fn parse(s: &'a str) -> Option<Self> {
        let mut parts = s.split('#');
        let nid = parts.next()?;
        let library = decode_id(parts.next()?)?;
        let module = decode_id(parts.next()?)?;
        if parts.next().is_some() || nid.len() != 11 {
            return None;
        }
        Some(Self {
            nid,
            library,
            module,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values computed independently with Python's hashlib.
    #[test]
    fn hashes_names_into_nids() {
        assert_eq!(nid("ps5slop_hello"), "x-5h8llFYgg");
        assert_eq!(nid("main"), "qrSLrVnRmSs");
        assert_eq!(nid("sceKernelWrite"), "4wSze92BhLI");
        assert_eq!(nid(""), "B4c6vzgC+Xc");
    }

    #[test]
    fn ids_round_trip_at_every_width() {
        for id in [0, 1, 0x3f, 0x40, 0xfff, 0x1000, 0xffff] {
            assert_eq!(decode_id(&encode_id(id)), Some(id), "id {id:#x}");
        }
        assert_eq!(encode_id(0), "A");
        assert_eq!(encode_id(0x41), "BB");
        assert_eq!(encode_id(0x1000), "BAA");
    }

    #[test]
    fn rejects_malformed_ids() {
        assert_eq!(decode_id(""), None);
        assert_eq!(decode_id("ABCD"), None);
        assert_eq!(decode_id("A/"), None);
    }

    #[test]
    fn splits_symbol_strings() {
        let s = SymbolRef::parse("4wSze92BhLI#B#C").unwrap();
        assert_eq!((s.nid, s.library, s.module), ("4wSze92BhLI", 1, 2));
        assert!(SymbolRef::parse("main").is_none());
        assert!(SymbolRef::parse("short#A#A").is_none());
        assert!(SymbolRef::parse("4wSze92BhLI#A#A#A").is_none());
    }
}
