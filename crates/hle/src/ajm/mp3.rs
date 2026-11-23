//! MP3 decoding through FFmpeg's libavcodec, one MPEG audio frame at a time.

use std::sync::Once;

use ffmpeg_next as ff;

static INIT: Once = Once::new();
static FAILURES: diag::FirstN = diag::FirstN::new(8);

/// The sample format an instance writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    S16,
    S32,
    F32,
}

impl Format {
    pub fn bytes(self) -> usize {
        match self {
            Format::S16 => 2,
            Format::S32 | Format::F32 => 4,
        }
    }

    fn put(self, v: f32, out: &mut Vec<u8>) {
        let v = v.clamp(-1.0, 1.0);
        match self {
            Format::S16 => out.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes()),
            Format::S32 => out.extend_from_slice(&((v as f64 * 2147483647.0) as i32).to_le_bytes()),
            Format::F32 => out.extend_from_slice(&v.to_le_bytes()),
        }
    }
}

/// One stream's decoder. The decoder keeps the bit reservoir between frames,
/// so frames of a stream must go through the same `Decoder` in order.
pub struct Decoder {
    inner: Option<ff::decoder::Audio>,
    frame: ff::frame::Audio,
}

// SAFETY: the codec context and frame are only used by whoever holds the
// decoder, never from two threads at once.
unsafe impl Send for Decoder {}

impl Default for Decoder {
    fn default() -> Self {
        INIT.call_once(|| {
            ff::util::log::set_level(ff::util::log::Level::Quiet);
        });
        let inner = ff::decoder::find(ff::codec::Id::MP3).and_then(|codec| {
            ff::codec::Context::new_with_codec(codec)
                .decoder()
                .audio()
                .ok()
        });
        if inner.is_none() && FAILURES.allow() {
            tracing::error!("FFmpeg has no MP3 decoder; Ajm writes silence");
        }
        Self {
            inner,
            frame: ff::frame::Audio::empty(),
        }
    }
}

impl Decoder {
    /// Decodes one whole frame and appends exactly `samples * channels`
    /// interleaved samples to `out`, padding with silence when FFmpeg gives
    /// fewer (a frame whose reservoir is missing, a decode error).
    pub fn frame(
        &mut self,
        bytes: &[u8],
        samples: usize,
        channels: usize,
        format: Format,
        out: &mut Vec<u8>,
    ) {
        let start = out.len();
        let want = start + samples * channels * format.bytes();
        if let Some(dec) = &mut self.inner {
            let sent = dec.send_packet(&ff::Packet::copy(bytes));
            if let Err(e) = sent
                && FAILURES.allow()
            {
                tracing::warn!("Ajm MP3 frame not decoded: {e}");
            }
            while out.len() < want && dec.receive_frame(&mut self.frame).is_ok() {
                interleave(&self.frame, channels, format, want, out);
            }
        }
        out.resize(want, 0);
    }
}

/// Appends a decoded frame's samples as `channels` interleaved samples in
/// `format`, up to `limit` bytes of `out`.
fn interleave(
    frame: &ff::frame::Audio,
    channels: usize,
    format: Format,
    limit: usize,
    out: &mut Vec<u8>,
) {
    use ff::format::Sample;
    use ff::format::sample::Type;
    let have = usize::from(frame.channels().max(1));
    let n = frame.samples();
    let f32_at = |plane: usize, i: usize| {
        let b = &frame.data(plane)[i * 4..i * 4 + 4];
        f32::from_ne_bytes(b.try_into().unwrap())
    };
    let i16_at = |plane: usize, i: usize| {
        let b = &frame.data(plane)[i * 2..i * 2 + 2];
        i16::from_ne_bytes(b.try_into().unwrap()) as f32 / 32768.0
    };
    let read = |c: usize, i: usize| -> f32 {
        let c = c.min(have - 1);
        match frame.format() {
            Sample::F32(Type::Planar) => f32_at(c, i),
            Sample::F32(Type::Packed) => f32_at(0, i * have + c),
            Sample::I16(Type::Planar) => i16_at(c, i),
            Sample::I16(Type::Packed) => i16_at(0, i * have + c),
            _ => 0.0,
        }
    };
    for i in 0..n {
        for c in 0..channels {
            if out.len() >= limit {
                return;
            }
            format.put(read(c, i), out);
        }
    }
}
