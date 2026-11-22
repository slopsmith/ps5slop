//! MPEG audio frame headers (ISO/IEC 11172-3 and 13818-3), enough to split
//! a stream into frames and size each frame's decoded output.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Bytes of the whole frame, header included.
    pub len: usize,
    pub samples: usize,
    pub channels: usize,
    pub rate: u32,
}

const BITRATES_V1: [[u32; 15]; 3] = [
    [
        0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
    ],
    [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    ],
    [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ],
];
const BITRATES_V2: [[u32; 15]; 2] = [
    [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
    ],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];
const RATES: [u32; 3] = [44100, 48000, 32000];

/// Parses the four header bytes at the start of `b`. Free-format frames
/// (bitrate index 0) are refused: their length is not in the header.
pub fn parse(b: &[u8]) -> Option<Header> {
    if b.len() < 4 || b[0] != 0xFF || b[1] & 0xE0 != 0xE0 {
        return None;
    }
    let version = (b[1] >> 3) & 3;
    let layer = match (b[1] >> 1) & 3 {
        3 => 1,
        2 => 2,
        1 => 3,
        _ => return None,
    };
    let bitrate_index = (b[2] >> 4) as usize;
    let rate_index = ((b[2] >> 2) & 3) as usize;
    if version == 1 || bitrate_index == 0 || bitrate_index == 15 || rate_index == 3 {
        return None;
    }
    let mpeg1 = version == 3;
    let kbps = if mpeg1 {
        BITRATES_V1[layer - 1][bitrate_index]
    } else {
        BITRATES_V2[usize::from(layer != 1)][bitrate_index]
    };
    let rate = RATES[rate_index]
        >> match version {
            3 => 0,
            2 => 1,
            _ => 2,
        };
    let padding = usize::from((b[2] >> 1) & 1);
    let bps = kbps as usize * 1000;
    let (len, samples) = match layer {
        1 => ((12 * bps / rate as usize + padding) * 4, 384),
        2 => (144 * bps / rate as usize + padding, 1152),
        _ if mpeg1 => (144 * bps / rate as usize + padding, 1152),
        _ => (72 * bps / rate as usize + padding, 576),
    };
    let channels = if b[3] >> 6 == 3 { 1 } else { 2 };
    Some(Header {
        len,
        samples,
        channels,
        rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpeg1_layer3_mono() {
        let h = parse(&[0xFF, 0xFA, 0x64, 0xC0]).unwrap();
        assert_eq!(
            h,
            Header {
                len: 240,
                samples: 1152,
                channels: 1,
                rate: 48000
            }
        );
    }

    #[test]
    fn padding_and_stereo() {
        let h = parse(&[0xFF, 0xFB, 0x92, 0x44]).unwrap();
        assert_eq!((h.len, h.channels, h.rate), (418, 2, 44100));
    }

    #[test]
    fn mpeg2_layer3() {
        let h = parse(&[0xFF, 0xF3, 0x84, 0xC4]).unwrap();
        assert_eq!((h.len, h.samples, h.rate), (192, 576, 24000));
    }

    #[test]
    fn refuses_non_headers() {
        assert!(parse(&[0xFF, 0x00, 0, 0]).is_none());
        assert!(parse(&[0xFF, 0xFA, 0x04, 0xC0]).is_none());
        assert!(parse(&[0xFF, 0xFA]).is_none());
    }
}
