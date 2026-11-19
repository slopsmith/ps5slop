//! Title ids (`PPSA01234`, `CUSA01234`) from a title's metadata files.

use crate::read::{u16_le, u32_le};
use crate::{Package, entry_id};

/// The `titleId` string of a PS5 `param.json`.
pub fn from_param_json(json: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(json).ok()?;
    let after_key = &text[text.find("\"titleId\"")? + "\"titleId\"".len()..];
    let value = after_key.trim_start().strip_prefix(':')?.trim_start();
    let value = value.strip_prefix('"')?;
    let id = &value[..value.find('"')?];
    is_title_id(id).then(|| id.to_owned())
}

/// The `TITLE_ID` string of a PS4 `param.sfo`: a header (magic `\0PSF`, key
/// table offset at `0x08`, data table offset at `0x0C`, entry count at
/// `0x10`) followed by 16-byte entries (key offset u16, format u16, length
/// u32, maximum length u32, data offset u32), per psdevwiki's "PARAM.SFO".
pub fn from_param_sfo(sfo: &[u8]) -> Option<String> {
    if sfo.get(..4)? != b"\0PSF" || sfo.len() < 0x14 {
        return None;
    }
    let keys = u32_le(sfo, 0x08) as usize;
    let data = u32_le(sfo, 0x0c) as usize;
    let count = u32_le(sfo, 0x10) as usize;
    for i in 0..count {
        let at = 0x14 + i * 16;
        let entry = sfo.get(at..at + 16)?;
        let key_at = keys + u16_le(entry, 0) as usize;
        let key = sfo.get(key_at..)?;
        let key = &key[..key.iter().position(|&c| c == 0)?];
        if key != b"TITLE_ID" {
            continue;
        }
        let len = u32_le(entry, 4) as usize;
        let value_at = data + u32_le(entry, 12) as usize;
        let value = sfo.get(value_at..value_at + len)?;
        let value = &value[..value.iter().position(|&c| c == 0).unwrap_or(len)];
        let id = std::str::from_utf8(value).ok()?;
        return is_title_id(id).then(|| id.to_owned());
    }
    None
}

/// The title id inside a content id (`UP0000-PPSA01234_00-...`).
pub fn from_content_id(content_id: &str) -> Option<String> {
    let id = content_id.get(7..16)?;
    is_title_id(id).then(|| id.to_owned())
}

/// Four capital letters and five digits. Anything else is not used as a
/// folder name.
pub fn is_title_id(id: &str) -> bool {
    let b = id.as_bytes();
    b.len() == 9
        && b[..4].iter().all(u8::is_ascii_uppercase)
        && b[4..].iter().all(u8::is_ascii_digit)
}

impl Package {
    /// The title id from `param.json` (PS5) or `param.sfo` (PS4) when they
    /// are stored in the clear, else from the content id.
    pub fn title_id(&self) -> Option<String> {
        let clear = |id| {
            self.entry(id)
                .filter(|e| !e.is_encrypted())
                .and_then(|e| self.read_entry(e).ok())
        };
        clear(entry_id::PARAM_JSON)
            .and_then(|j| from_param_json(&j))
            .or_else(|| clear(entry_id::PARAM_SFO).and_then(|s| from_param_sfo(&s)))
            .or_else(|| from_content_id(&self.header().content_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_title_id_from_param_json() {
        let json = br#"{ "ageLevel": {}, "titleId" : "PPSA17221", "x": 1 }"#;
        assert_eq!(from_param_json(json).as_deref(), Some("PPSA17221"));
        assert_eq!(from_param_json(br#"{"titleId": "../etc"}"#), None);
        assert_eq!(from_param_json(b"{}"), None);
    }

    fn sfo(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut keys = Vec::new();
        let mut data = Vec::new();
        let mut table = Vec::new();
        for (k, v) in entries {
            let mut e = [0u8; 16];
            e[0..2].copy_from_slice(&(keys.len() as u16).to_le_bytes());
            e[2..4].copy_from_slice(&0x0204u16.to_le_bytes());
            e[4..8].copy_from_slice(&(v.len() as u32 + 1).to_le_bytes());
            e[8..12].copy_from_slice(&16u32.to_le_bytes());
            e[12..16].copy_from_slice(&(data.len() as u32).to_le_bytes());
            table.extend(e);
            keys.extend(k.bytes().chain([0]));
            data.extend(v.bytes().chain([0]));
            data.resize(data.len().next_multiple_of(16), 0);
        }
        let key_at = 0x14 + table.len();
        let data_at = key_at + keys.len();
        let mut out = b"\0PSF\x01\x01\0\0".to_vec();
        out.extend((key_at as u32).to_le_bytes());
        out.extend((data_at as u32).to_le_bytes());
        out.extend((entries.len() as u32).to_le_bytes());
        out.extend(table);
        out.extend(keys);
        out.extend(data);
        out
    }

    #[test]
    fn reads_title_id_from_param_sfo() {
        let s = sfo(&[
            ("CATEGORY", "gd"),
            ("TITLE_ID", "CUSA00001"),
            ("TITLE", "x"),
        ]);
        assert_eq!(from_param_sfo(&s).as_deref(), Some("CUSA00001"));
        assert_eq!(from_param_sfo(&sfo(&[("TITLE", "x")])), None);
        assert_eq!(from_param_sfo(b"junk"), None);
    }

    #[test]
    fn reads_title_id_from_content_id() {
        assert_eq!(
            from_content_id("UP3126-PPSA06084_00-2911532214834517").as_deref(),
            Some("PPSA06084")
        );
        assert_eq!(
            from_content_id("IV0000-VICS00001_00-PKGSTORE00000000").as_deref(),
            Some("VICS00001")
        );
        assert_eq!(from_content_id("short"), None);
    }
}
