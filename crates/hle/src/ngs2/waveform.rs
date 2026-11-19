//! `SceNgs2WaveformInfo` from a RIFF WAVE image, as
//! `sceNgs2ParseWaveformData` and `sceNgs2ParseWaveformFile` report it.
//! Ported from SharpEmu-mac's `Ngs2WaveformParser.cs` (GPL-2.0-or-later),
//! PCM and float formats only; VAG and ATRAC9 images are refused.

pub const TYPE_PCM_U8: u32 = 0x11;
pub const TYPE_PCM_I16L: u32 = 0x12;
pub const TYPE_PCM_I24L: u32 = 0x14;
pub const TYPE_PCM_I32L: u32 = 0x16;
pub const TYPE_PCM_F32L: u32 = 0x18;
pub const TYPE_PCM_F64L: u32 = 0x1a;

/// `SceNgs2WaveformBlock`: data offset, size, repeats, skipped and played
/// samples, user data.
pub const BLOCK_SIZE: usize = 0x28;
const BLOCKS_AT: usize = 0x48;
const MAX_BLOCKS: usize = 4;
pub const INFO_SIZE: usize = BLOCKS_AT + MAX_BLOCKS * BLOCK_SIZE;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Info {
    pub kind: u32,
    pub channels: u32,
    pub rate: u32,
    pub data_offset: u32,
    pub data_size: u32,
    pub loop_begin: u32,
    pub loop_end: u32,
    pub samples: u32,
    pub unit_size: u32,
}

/// Bytes per sample of a PCM waveform type, 0 for anything else.
pub fn sample_bytes(kind: u32) -> usize {
    match kind {
        0x10 | 0x11 => 1,
        0x12 | 0x13 => 2,
        0x14 | 0x15 => 3,
        0x16..=0x19 => 4,
        0x1a | 0x1b => 8,
        _ => 0,
    }
}

/// One sample of a PCM waveform type, scaled to -1..1.
pub fn read_sample(d: &[u8], kind: u32) -> f32 {
    let v = match kind {
        0x10 => d[0] as i8 as f32 / 128.0,
        0x11 => (d[0] as f32 - 128.0) / 128.0,
        0x12 => i16::from_le_bytes([d[0], d[1]]) as f32 / 32768.0,
        0x13 => i16::from_be_bytes([d[0], d[1]]) as f32 / 32768.0,
        0x14 => (i32::from_le_bytes([0, d[0], d[1], d[2]]) >> 8) as f32 / 8_388_608.0,
        0x15 => (i32::from_le_bytes([0, d[2], d[1], d[0]]) >> 8) as f32 / 8_388_608.0,
        0x16 => (i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as f64 / 2_147_483_648.0) as f32,
        0x17 => (i32::from_be_bytes([d[0], d[1], d[2], d[3]]) as f64 / 2_147_483_648.0) as f32,
        0x18 => f32::from_le_bytes([d[0], d[1], d[2], d[3]]),
        0x19 => f32::from_be_bytes([d[0], d[1], d[2], d[3]]),
        0x1a => f64::from_le_bytes(d[..8].try_into().unwrap_or_default()) as f32,
        0x1b => f64::from_be_bytes(d[..8].try_into().unwrap_or_default()) as f32,
        _ => 0.0,
    };
    if v.is_finite() { v } else { 0.0 }
}

/// Whether `data` starts like a RIFF WAVE image.
pub fn is_riff(data: &[u8]) -> bool {
    data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE"
}

fn u16_at(d: &[u8], at: usize) -> Option<u32> {
    Some(u16::from_le_bytes(d.get(at..at + 2)?.try_into().ok()?) as u32)
}

fn u32_at(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(at..at + 4)?.try_into().ok()?))
}

/// Parses the header of a WAVE image. `total` is the image's full length,
/// which may run past the part in `data`.
pub fn parse(data: &[u8], total: u64) -> Option<Info> {
    if !is_riff(data) {
        return None;
    }
    let mut info = Info::default();
    let (mut at, mut have_fmt, mut have_data) = (12usize, false, false);
    while at + 8 <= data.len() {
        let size = u32_at(data, at + 4)?;
        let body = at + 8;
        let len = (size as usize).min(data.len() - body);
        let chunk = &data[body..body + len];
        match &data[at..at + 4] {
            b"fmt " => {
                format_chunk(chunk, &mut info)?;
                have_fmt = true;
            }
            b"data" => {
                info.data_offset = body as u32;
                info.data_size = size;
                have_data = true;
            }
            b"smpl" if len >= 0x3c && u32_at(chunk, 0x1c)? != 0 => {
                info.loop_begin = u32_at(chunk, 0x2c)?;
                info.loop_end = u32_at(chunk, 0x30)?;
            }
            _ => {}
        }
        if have_data {
            break;
        }
        at = body + size as usize + (size as usize & 1);
    }
    if !have_fmt || !have_data {
        return None;
    }
    let available = total
        .saturating_sub(info.data_offset as u64)
        .min(u32::MAX as u64) as u32;
    info.data_size = info.data_size.min(available);
    info.samples = info.data_size / info.unit_size;
    if info.loop_begin == 0 && info.loop_end == 0 && info.samples != 0 {
        info.loop_end = info.samples - 1;
    }
    Some(info)
}

fn format_chunk(c: &[u8], info: &mut Info) -> Option<()> {
    let mut tag = u16_at(c, 0)?;
    let channels = u16_at(c, 2)?;
    let rate = u32_at(c, 4)?;
    let align = u16_at(c, 12)?;
    let bits = u16_at(c, 14)?;
    if tag == 0xfffe {
        let sub = c.get(0x18..0x28)?;
        const TAIL: [u8; 14] = [0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71];
        if sub[2..] != TAIL || sub[1] != 0 {
            return None;
        }
        tag = sub[0] as u32;
    }
    info.kind = match (tag, bits) {
        (1, 8) => TYPE_PCM_U8,
        (1, 16) => TYPE_PCM_I16L,
        (1, 24) => TYPE_PCM_I24L,
        (1, 32) => TYPE_PCM_I32L,
        (3, 32) => TYPE_PCM_F32L,
        (3, 64) => TYPE_PCM_F64L,
        _ => return None,
    };
    info.channels = channels.max(1);
    info.rate = if rate == 0 { 48_000 } else { rate };
    info.unit_size = if align != 0 {
        align
    } else {
        bits / 8 * info.channels
    };
    (info.unit_size != 0).then_some(())
}

/// The guest `SceNgs2WaveformInfo`: the format, then the data and loop
/// fields, the audio unit table and one block covering the whole data.
pub fn encode(info: &Info) -> [u8; INFO_SIZE] {
    let mut out = [0u8; INFO_SIZE];
    let words = [
        info.kind,
        info.channels,
        info.rate,
        0,
        0,
        0,
        info.data_offset,
        info.data_size,
        info.loop_begin,
        info.loop_end,
        info.samples,
        info.unit_size,
        1,
        1,
        info.unit_size,
        1,
        0,
        1,
    ];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    let b = &mut out[BLOCKS_AT..];
    b[0..8].copy_from_slice(&(info.data_offset as u64).to_le_bytes());
    b[8..16].copy_from_slice(&(info.data_size as u64).to_le_bytes());
    b[0x18..0x1c].copy_from_slice(&info.samples.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(channels: u16, bits: u16, frames: u32) -> Vec<u8> {
        let align = channels * bits / 8;
        let data = frames * align as u32;
        let mut w = b"RIFF".to_vec();
        w.extend((36 + data).to_le_bytes());
        w.extend(b"WAVEfmt ");
        w.extend(16u32.to_le_bytes());
        w.extend(1u16.to_le_bytes());
        w.extend(channels.to_le_bytes());
        w.extend(48_000u32.to_le_bytes());
        w.extend((48_000 * align as u32).to_le_bytes());
        w.extend(align.to_le_bytes());
        w.extend(bits.to_le_bytes());
        w.extend(b"data");
        w.extend(data.to_le_bytes());
        w.resize(w.len() + data as usize, 0);
        w
    }

    #[test]
    fn stereo_pcm16_wave_is_described_with_one_block() {
        let w = wave(2, 16, 1000);
        let info = parse(&w, w.len() as u64).unwrap();
        assert_eq!(
            info,
            Info {
                kind: TYPE_PCM_I16L,
                channels: 2,
                rate: 48_000,
                data_offset: 44,
                data_size: 4000,
                loop_begin: 0,
                loop_end: 999,
                samples: 1000,
                unit_size: 4,
            }
        );
        let e = encode(&info);
        assert_eq!(u32::from_le_bytes(e[0x28..0x2c].try_into().unwrap()), 1000);
        assert_eq!(u64::from_le_bytes(e[0x50..0x58].try_into().unwrap()), 4000);
    }

    #[test]
    fn a_header_alone_takes_its_length_from_the_total() {
        let w = wave(1, 16, 10);
        let info = parse(&w[..44], 44 + 20).unwrap();
        assert_eq!(info.samples, 10);
        assert!(parse(b"OggS....", 8).is_none());
    }

    #[test]
    fn samples_are_scaled_to_unit_range() {
        assert_eq!(read_sample(&(-32768i16).to_le_bytes(), TYPE_PCM_I16L), -1.0);
        assert_eq!(read_sample(&[0x80], TYPE_PCM_U8), 0.0);
        assert_eq!(read_sample(&0.5f32.to_le_bytes(), TYPE_PCM_F32L), 0.5);
    }
}
