//! Rendering one grain: each playing sampler voice is resampled linearly
//! to the system rate, sent through its port patches (submixers pass
//! their input on, effects are ignored) to a mastering voice, which adds
//! it to its render buffer. Ported from SharpEmu-mac's
//! `Ngs2Exports.Mixer.cs` (GPL-2.0-or-later).

use memory::Memory;

use super::voice::{Block, Voice};
use super::waveform::{read_sample, sample_bytes};
use super::{
    Callback, ERROR_INVALID_CONTROL_ADDRESS, ERROR_INVALID_CONTROL_SIZE,
    ERROR_INVALID_SYSTEM_HANDLE, ERROR_INVALID_WAVEFORM_FORMAT, RACK_CUSTOM_MASTERING,
    RACK_MASTERING, State,
};
use crate::{CallError, Env};

/// `SceNgs2RenderBufferInfo`: address, size, waveform type, channels.
const BUFFER_INFO_SIZE: u64 = 24;
const MAX_BUFFER: u64 = 16 << 20;
/// Patches followed from one voice before we assume a loop.
const MAX_ROUTE_DEPTH: usize = 8;

struct Target {
    address: u64,
    kind: u32,
    channels: usize,
    samples: Vec<f32>,
}

/// Voice callback flags: a block ended, a block looped, and "the voice
/// ends with its queue even though it has a callback".
const CALLBACK_BLOCK_END: u32 = 1;
const CALLBACK_BLOCK_LOOP: u32 = 2;
const CALLBACK_END_WHEN_DRY: u32 = 0x1_0000;

/// Mixes one grain of `system` into the `count` buffers described at
/// `list`. Returns the callbacks owed, or an NGS2 error code.
pub(super) fn render(
    env: &Env,
    s: &mut State,
    system: u64,
    list: u64,
    count: u32,
) -> Result<Result<Vec<Callback>, u64>, CallError> {
    if count > 32 || (count != 0 && list == 0) {
        return Ok(Err(ERROR_INVALID_CONTROL_ADDRESS));
    }
    let Some(sys) = s.systems.get(&system) else {
        return Ok(Err(ERROR_INVALID_SYSTEM_HANDLE));
    };
    let (grain, rate) = (sys.grain, sys.rate);
    let mut targets = Vec::with_capacity(count as usize);
    for i in 0..count as u64 {
        let mut d = [0u8; BUFFER_INFO_SIZE as usize];
        env.mem.read(list + i * BUFFER_INFO_SIZE, &mut d)?;
        let address = u64::from_le_bytes(d[0..8].try_into().unwrap_or_default());
        let size = u64::from_le_bytes(d[8..16].try_into().unwrap_or_default());
        let kind = u32::from_le_bytes(d[16..20].try_into().unwrap_or_default());
        let channels = u32::from_le_bytes(d[20..24].try_into().unwrap_or_default()) as usize;
        if !matches!(kind, 0x12 | 0x13 | 0x18 | 0x19) || !(1..=8).contains(&channels) {
            return Ok(Err(ERROR_INVALID_WAVEFORM_FORMAT));
        }
        let need = (grain * channels * sample_bytes(kind)) as u64;
        if size < need || size > MAX_BUFFER || address == 0 {
            return Ok(Err(ERROR_INVALID_CONTROL_SIZE));
        }
        targets.push(Target {
            address,
            kind,
            channels,
            samples: vec![0.0; grain * channels],
        });
    }

    let mut callbacks = Vec::new();
    let mut sources: Vec<u64> = s
        .voices
        .iter()
        .filter(|(_, v)| {
            v.playing
                && !v.paused
                && sample_bytes(v.kind) != 0
                && s.racks.get(&v.rack).is_some_and(|r| r.system == system)
        })
        .map(|(&h, _)| h)
        .collect();
    sources.sort_unstable();
    for handle in sources {
        let Some(v) = s.voices.get_mut(&handle) else {
            continue;
        };
        let mut samples = vec![0.0; grain * v.channels];
        if let Err(code) = mix_voice(env.mem, handle, v, rate, &mut samples, &mut callbacks) {
            v.error = code as i32;
            v.playing = false;
            v.stopped = true;
            continue;
        }
        let channels = v.channels;
        route(
            s,
            system,
            handle,
            &samples,
            channels,
            grain,
            &mut targets,
            0,
        );
    }

    for t in &targets {
        let bytes = sample_bytes(t.kind);
        let mut out = vec![0u8; t.samples.len() * bytes];
        for (o, &x) in out.chunks_exact_mut(bytes).zip(&t.samples) {
            let x = x.clamp(-1.0, 1.0);
            match t.kind {
                0x18 => o.copy_from_slice(&x.to_le_bytes()),
                0x19 => o.copy_from_slice(&x.to_be_bytes()),
                0x12 => o.copy_from_slice(&to_i16(x).to_le_bytes()),
                _ => o.copy_from_slice(&to_i16(x).to_be_bytes()),
            }
        }
        env.mem.write(t.address, &out)?;
    }
    Ok(Ok(callbacks))
}

fn to_i16(x: f32) -> i16 {
    (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// Frames of one block, read from guest memory and scaled to -1..1.
struct Cache {
    key: (u64, u32, u32),
    first: i64,
    frames: Vec<f32>,
}

fn read_frames(mem: &Memory, v: &Voice, block: &Block, first: i64, n: usize) -> Option<Vec<f32>> {
    let bytes = sample_bytes(v.kind);
    let frame = bytes * v.channels;
    let mut raw = vec![0u8; n * frame];
    let at = block.address + (block.skip as u64 + first as u64) * frame as u64;
    mem.read(at, &mut raw).ok()?;
    Some(
        raw.chunks_exact(bytes)
            .map(|s| read_sample(s, v.kind))
            .collect(),
    )
}

/// Resamples one grain of `v` into `out` (interleaved, in the voice's
/// channel count), consuming its blocks.
fn mix_voice(
    mem: &Memory,
    handle: u64,
    v: &mut Voice,
    rate: u32,
    out: &mut [f32],
    callbacks: &mut Vec<Callback>,
) -> Result<(), u64> {
    let ch = v.channels;
    let grain = out.len() / ch;
    let frame_bytes = (sample_bytes(v.kind) * ch) as u64;
    let step = v.rate as f64 / rate as f64 * v.pitch as f64;
    let mut cache: Option<Cache> = None;
    for f in 0..grain {
        advance(handle, v, callbacks);
        let Some(block) = v.blocks.front() else {
            break;
        };
        let src = v.position as i64;
        let last = block.frames as i64 - 1;
        let next = (src + 1).min(last);
        let key = (block.address, block.skip, block.frames);
        let valid = cache.as_ref().is_some_and(|c| {
            c.key == key && src >= c.first && next < c.first + (c.frames.len() / ch) as i64
        });
        if !valid {
            let n = ((grain - f) as f64 * step) as usize + 2;
            let n = n.min((block.frames as i64 - src) as usize).max(1);
            let Some(frames) = read_frames(mem, v, block, src, n) else {
                return Err(ERROR_INVALID_WAVEFORM_FORMAT);
            };
            cache = Some(Cache {
                key,
                first: src,
                frames,
            });
        }
        let c = cache.as_ref().unwrap_or_else(|| unreachable!());
        let frac = (v.position - src as f64) as f32;
        let rel = (src - c.first) as usize;
        let rel_next = (next - c.first) as usize;
        let mut join = None;
        if frac > 0.0 && src == last {
            let other = if block.repeats != 0 {
                Some(block)
            } else {
                v.blocks.get(1).filter(|b| b.joins && b.frames != 0)
            };
            join = other.and_then(|b| read_frames(mem, v, b, 0, 1));
        }
        for k in 0..ch {
            let a = c.frames[rel * ch + k];
            let b = join.as_ref().map_or(c.frames[rel_next * ch + k], |j| j[k]);
            out[f * ch + k] = a + (b - a) * frac;
        }
        let advanced = ((v.position + step) as i64 - src) as u64;
        v.decoded_samples += advanced;
        v.decoded_bytes += advanced * frame_bytes;
        v.position += step;
    }
    advance(handle, v, callbacks);
    Ok(())
}

/// Drops or loops the blocks the play position has passed, queueing the
/// callbacks the voice asked for.
fn advance(handle: u64, v: &mut Voice, callbacks: &mut Vec<Callback>) {
    let mut ran_dry = false;
    while let Some(b) = v.blocks.front_mut() {
        if b.frames != 0 && v.position < b.frames as f64 {
            break;
        }
        let flag = if b.frames != 0 && b.repeats != 0 {
            v.position -= b.frames as f64;
            if b.repeats != u32::MAX {
                b.repeats -= 1;
            }
            b.repeated += 1;
            CALLBACK_BLOCK_LOOP
        } else {
            v.position = (v.position - b.frames as f64).max(0.0);
            CALLBACK_BLOCK_END
        };
        if v.callback != 0 && v.callback_flags & flag != 0 {
            let mut info = [0u8; 64];
            info[0..8].copy_from_slice(&v.callback_data.to_le_bytes());
            info[8..16].copy_from_slice(&handle.to_le_bytes());
            info[16..20].copy_from_slice(&flag.to_le_bytes());
            info[24..32].copy_from_slice(&b.user_data.to_le_bytes());
            info[32..40].copy_from_slice(&b.address.to_le_bytes());
            info[40..48].copy_from_slice(&b.length.to_le_bytes());
            info[48..52].copy_from_slice(&b.repeated.to_le_bytes());
            callbacks.push(Callback {
                handler: v.callback,
                info,
            });
        }
        if flag == CALLBACK_BLOCK_END {
            let streamed = b.joins;
            v.blocks.pop_front();
            ran_dry = v.blocks.is_empty() && !streamed;
        }
    }
    // A voice started before its first block, or fed a stream, waits for more.
    if ran_dry && (v.callback == 0 || v.callback_flags & CALLBACK_END_WHEN_DRY != 0) {
        v.playing = false;
        v.in_use = false;
        v.paused = false;
        v.stopped = false;
    }
}

/// Sends `samples` from voice `handle` along its patches until they reach
/// a mastering voice or a render buffer.
#[allow(clippy::too_many_arguments)]
fn route(
    s: &State,
    system: u64,
    handle: u64,
    samples: &[f32],
    channels: usize,
    grain: usize,
    targets: &mut [Target],
    depth: usize,
) {
    let Some(v) = s.voices.get(&handle) else {
        return;
    };
    if depth > MAX_ROUTE_DEPTH {
        return;
    }
    if matches!(s.rack_id(v), RACK_MASTERING | RACK_CUSTOM_MASTERING) {
        if v.playing
            && !v.paused
            && let Some(t) = targets.get_mut(v.output as usize)
        {
            add_channels(samples, channels, &mut t.samples, t.channels, v.gain, None);
        }
        return;
    }
    let mut ports: Vec<_> = v.ports.iter().collect();
    ports.sort_unstable_by_key(|&(&i, _)| i);
    for (_, port) in ports {
        if port.volume == 0.0 || port.patch == 0 {
            continue;
        }
        let levels = port
            .matrix
            .and_then(|m| v.matrices.get(&m))
            .map(Vec::as_slice);
        if port.patch == 2 {
            if let Some(t) = targets.get_mut(port.buffer as usize) {
                add_channels(
                    samples,
                    channels,
                    &mut t.samples,
                    t.channels,
                    port.volume,
                    levels,
                );
            }
            continue;
        }
        if port.patch != 1 {
            continue;
        }
        let dest = if s.voices.contains_key(&port.destination) {
            Some(port.destination)
        } else if s.racks.contains_key(&port.destination) {
            s.voices
                .iter()
                .find(|(_, d)| d.rack == port.destination && d.index == 0)
                .map(|(&h, _)| h)
        } else {
            None
        };
        let Some((dh, d)) = dest.and_then(|h| s.voices.get(&h).map(|d| (h, d))) else {
            continue;
        };
        if !d.playing || d.paused || !s.racks.get(&d.rack).is_some_and(|r| r.system == system) {
            continue;
        }
        let mut routed = vec![0.0; grain * d.channels];
        add_channels(
            samples,
            channels,
            &mut routed,
            d.channels,
            port.volume,
            levels,
        );
        route(
            s,
            system,
            dh,
            &routed,
            d.channels,
            grain,
            targets,
            depth + 1,
        );
    }
}

/// Level matrices are input-major; without one, mono feeds the front pair.
fn add_channels(
    src: &[f32],
    src_ch: usize,
    dst: &mut [f32],
    dst_ch: usize,
    gain: f32,
    levels: Option<&[f32]>,
) {
    let levels = levels.filter(|l| l.len() >= src_ch * dst_ch);
    for (s, d) in src.chunks_exact(src_ch).zip(dst.chunks_exact_mut(dst_ch)) {
        for (o, out) in d.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (i, &x) in s.iter().enumerate() {
                let level = match levels {
                    Some(l) => l[i * dst_ch + o],
                    None if dst_ch == 1 => 1.0 / src_ch as f32,
                    None if (src_ch == 1 && o < 2) || i == o => 1.0,
                    None => 0.0,
                };
                sum += x * level;
            }
            *out += sum * gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_goes_to_both_front_channels_and_stereo_downmixes_to_mono() {
        let mut stereo = vec![0.0; 4];
        add_channels(&[0.5, 0.25], 1, &mut stereo, 2, 1.0, None);
        assert_eq!(stereo, [0.5, 0.5, 0.25, 0.25]);
        let mut mono = vec![0.0; 1];
        add_channels(&[1.0, 0.5], 2, &mut mono, 1, 2.0, None);
        assert_eq!(mono, [1.5]);
    }

    #[test]
    fn a_finished_block_ends_the_voice_without_a_callback() {
        let mut v = Voice::new(1, 0);
        v.playing = true;
        v.in_use = true;
        v.blocks.push_back(Block {
            address: 0,
            length: 8,
            skip: 0,
            frames: 2,
            repeats: 1,
            repeated: 0,
            joins: false,
            user_data: 7,
        });
        v.position = 2.5;
        let mut calls = Vec::new();
        advance(9, &mut v, &mut calls);
        assert_eq!(v.position, 0.5);
        assert_eq!(v.blocks.front().map(|b| b.repeats), Some(0));
        v.position = 2.0;
        advance(9, &mut v, &mut calls);
        assert!(v.blocks.is_empty() && !v.playing && calls.is_empty());
    }
}
