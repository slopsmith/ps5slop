//! A 16-bit PCM WAVE file written as audio arrives. The header's sizes
//! are rewritten after every append, so the file stays valid however the
//! process ends.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

pub struct Wav {
    file: File,
    data: u32,
}

impl Wav {
    pub fn create(path: &Path, channels: u16, rate: u32) -> std::io::Result<Self> {
        let mut file = File::create(path)?;
        let align = channels * 2;
        let mut h = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        h.extend(16u32.to_le_bytes());
        h.extend(1u16.to_le_bytes());
        h.extend(channels.to_le_bytes());
        h.extend(rate.to_le_bytes());
        h.extend((rate * align as u32).to_le_bytes());
        h.extend(align.to_le_bytes());
        h.extend(16u16.to_le_bytes());
        h.extend(b"data\0\0\0\0");
        file.write_all(&h)?;
        Ok(Self { file, data: 0 })
    }

    /// Appends interleaved samples in -1..1.
    pub fn append(&mut self, samples: &[f32]) -> std::io::Result<()> {
        let pcm: Vec<u8> = samples
            .iter()
            .flat_map(|&x| ((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
            .collect();
        self.file.write_all(&pcm)?;
        self.data = self.data.saturating_add(pcm.len() as u32);
        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&(36 + self.data).to_le_bytes())?;
        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&self.data.to_le_bytes())?;
        self.file.seek(SeekFrom::End(0))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_in_the_header_follow_the_data() {
        let path = std::env::temp_dir().join(format!("ps5slop-wav-{}.wav", std::process::id()));
        let mut w = Wav::create(&path, 2, 48_000).unwrap();
        w.append(&[0.0, 1.0, -1.0, 0.5]).unwrap();
        let b = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(b.len(), 44 + 8);
        assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 44);
        assert_eq!(i16::from_le_bytes([b[46], b[47]]), 32767);
    }
}
