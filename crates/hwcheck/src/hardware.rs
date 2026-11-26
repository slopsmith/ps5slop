//! A hardware run's result kept for later: what a test's program left on
//! the GPU, tied to the exact test by a fingerprint of its inputs. Lets the
//! Mac compare a changed translator with hardware without the GPU.

use std::path::Path;

use crate::Case;

const HEADER: [u8; 8] = *b"PS5IHWRS";
const FORMAT: u32 = 1;
const TAG_META: [u8; 4] = *b"META";
const TAG_WORDS: [u8; 4] = *b"WRDS";

/// The words a test's run left (records, then checked memory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hardware {
    /// [`fingerprint`] of the test the words belong to.
    pub fingerprint: u64,
    /// The GPU that ran it, e.g. "gfx1033".
    pub gpu: String,
    /// Whether the GPU ran multiply-adds fused that gfx1013 runs unfused.
    pub fused: bool,
    pub words: Vec<u32>,
}

struct Meta {
    fingerprint: u64,
    gpu: String,
    fused: bool,
}

savestate::record!(Meta {
    fingerprint,
    gpu,
    fused
});

/// A hash of everything a test's hardware result depends on: its memory,
/// registers and command buffers, not Metal's result.
pub fn fingerprint(case: &Case) -> u64 {
    let mut inputs = case.clone();
    inputs.output.clear();
    inputs.untranslated = None;
    let mut bytes = Vec::new();
    inputs
        .write_to(&mut bytes)
        .expect("writing into memory does not fail");
    savestate::fnv(&bytes)
}

impl Hardware {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        let mut header = HEADER.to_vec();
        header.extend_from_slice(&FORMAT.to_le_bytes());
        let mut w = savestate::Writer::new(std::io::BufWriter::new(file), &header)
            .map_err(|e| e.to_string())?;
        let meta = Meta {
            fingerprint: self.fingerprint,
            gpu: self.gpu.clone(),
            fused: self.fused,
        };
        w.section(TAG_META, &savestate::to_bytes(&meta))
            .map_err(|e| e.to_string())?;
        let raw: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        w.section(TAG_WORDS, &raw).map_err(|e| e.to_string())?;
        w.finish().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mut r =
            savestate::Reader::new(std::io::BufReader::new(file)).map_err(|e| e.to_string())?;
        if r.header.len() != 12 || r.header[..8] != HEADER {
            return Err(format!("{} is not a hardware result", path.display()));
        }
        let meta: Meta = savestate::from_bytes(&r.expect(TAG_META).map_err(|e| e.0)?)
            .map_err(|e| e.to_string())?;
        let raw = r.expect(TAG_WORDS).map_err(|e| e.0)?;
        Ok(Hardware {
            fingerprint: meta.fingerprint,
            gpu: meta.gpu,
            fused: meta.fused,
            words: crate::words(&raw),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_results_survive_a_round_trip_and_follow_their_test() {
        let dir = std::env::temp_dir().join(format!("hwcheck-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.hw");
        let h = Hardware {
            fingerprint: 7,
            gpu: "gfx1033".into(),
            fused: true,
            words: (0..1000).collect(),
        };
        h.save(&path).unwrap();
        assert_eq!(Hardware::load(&path).unwrap(), h);
        std::fs::remove_dir_all(&dir).unwrap();

        let mut case = Case::default();
        let before = fingerprint(&case);
        case.output.insert(0, vec![1; 4]);
        assert_eq!(fingerprint(&case), before);
        case.groups = 2;
        assert_ne!(fingerprint(&case), before);
    }
}
