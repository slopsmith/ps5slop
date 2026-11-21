//! Voices and what titles tell them: parameter lists through
//! `sceNgs2VoiceControl` and 16-byte command records through
//! `sceNgs2VoiceRunCommands`. Ported from SharpEmu-mac's
//! `Ngs2Exports.Voices.cs` (GPL-2.0-or-later).

use std::collections::{HashMap, VecDeque};

use super::waveform::{self, BLOCK_SIZE};
use super::{
    ERROR_INVALID_CONTROL_ADDRESS, ERROR_INVALID_CONTROL_ID, ERROR_INVALID_CONTROL_SIZE,
    ERROR_INVALID_VOICE_HANDLE, ERROR_INVALID_WAVEFORM_FORMAT, RACK_CUSTOM_SAMPLER, RACK_SAMPLER,
    ret, tracing,
};
use crate::pthread::lock;
use crate::{CallError, Env, Outcome};

/// A voice output port: where its signal goes and how loud.
#[derive(Clone)]
pub struct Port {
    pub volume: f32,
    pub matrix: Option<u32>,
    /// 1 patches to another voice (or a rack's voice 0), 2 to a render buffer.
    pub patch: u32,
    pub destination: u64,
    pub buffer: u32,
}

impl Default for Port {
    fn default() -> Self {
        Self {
            volume: 1.0,
            matrix: None,
            patch: 0,
            destination: 0,
            buffer: 0,
        }
    }
}

/// A queued piece of waveform data, in frames of the voice's format.
pub struct Block {
    pub address: u64,
    pub length: u64,
    pub skip: u32,
    pub frames: u32,
    /// Times left to repeat; `u32::MAX` loops forever.
    pub repeats: u32,
    pub repeated: u32,
    /// Interpolation may read the next block's first frame.
    pub joins: bool,
    pub user_data: u64,
}

pub struct Voice {
    pub rack: u64,
    pub index: u32,
    pub kind: u32,
    pub channels: usize,
    pub rate: u32,
    pub source: u64,
    pub blocks: VecDeque<Block>,
    pub position: f64,
    pub playing: bool,
    pub paused: bool,
    pub in_use: bool,
    pub stopped: bool,
    pub error: i32,
    pub pitch: f32,
    pub gain: f32,
    /// Render buffer a mastering voice writes to.
    pub output: u32,
    pub decoded_samples: u64,
    pub decoded_bytes: u64,
    pub callback: u64,
    pub callback_data: u64,
    pub callback_flags: u32,
    pub ports: HashMap<u32, Port>,
    pub matrices: HashMap<u32, Vec<f32>>,
}

/// `SceNgs2VoiceEvent` bits.
const EVENT_PLAY: u32 = 1;
const EVENT_STOP: u32 = 2;
const EVENT_STOP_IMMEDIATE: u32 = 4;
const EVENT_KILL: u32 = 8;
const EVENT_PAUSE: u32 = 16;
const EVENT_RESUME: u32 = 32;

impl Voice {
    pub fn new(rack: u64, index: u32) -> Self {
        Self {
            rack,
            index,
            kind: 0,
            channels: 2,
            rate: super::DEFAULT_RATE,
            source: 0,
            blocks: VecDeque::new(),
            position: 0.0,
            playing: false,
            paused: false,
            in_use: false,
            stopped: false,
            error: 0,
            pitch: 1.0,
            gain: 1.0,
            output: 0,
            decoded_samples: 0,
            decoded_bytes: 0,
            callback: 0,
            callback_data: 0,
            callback_flags: 0,
            ports: HashMap::new(),
            matrices: HashMap::new(),
        }
    }

    /// `SceNgs2VoiceStateFlags`: in use, playing, paused, stopped, error,
    /// and out of blocks.
    pub fn flags(&self) -> u32 {
        if !self.in_use {
            return 0;
        }
        1 | (self.playing as u32) << 1
            | (self.paused as u32) << 2
            | (self.stopped as u32) << 3
            | ((self.error != 0) as u32) << 4
            | (self.blocks.is_empty() as u32) << 5
    }

    fn port(&mut self, index: u32) -> &mut Port {
        self.ports.entry(index).or_default()
    }

    fn kick(&mut self, events: u32) {
        if events & EVENT_KILL != 0 {
            self.blocks.clear();
            self.position = 0.0;
            self.error = 0;
            self.in_use = false;
            self.playing = false;
            self.paused = false;
            self.stopped = false;
        }
        if events & (EVENT_STOP | EVENT_STOP_IMMEDIATE) != 0 {
            self.playing = false;
            self.paused = false;
            self.stopped = true;
        }
        if events & EVENT_PLAY != 0 {
            self.playing = true;
            self.in_use = true;
            self.paused = false;
            self.stopped = false;
        }
        if events & EVENT_PAUSE != 0 {
            self.paused = true;
        }
        if events & EVENT_RESUME != 0 {
            self.paused = false;
        }
    }

    /// Sets the waveform format: type, channels, rate (and codec data we
    /// do not use). The queue empties.
    fn set_format(&mut self, d: &[u8]) -> u64 {
        let (kind, channels, rate) = (u32_at(d, 0), u32_at(d, 4), u32_at(d, 8));
        if kind != 0
            && (waveform::sample_bytes(kind) == 0
                || !(1..=8).contains(&channels)
                || !(1..=192_000).contains(&rate))
        {
            return ERROR_INVALID_WAVEFORM_FORMAT;
        }
        self.kind = kind;
        self.channels = if channels == 0 { 2 } else { channels as usize };
        self.rate = if rate == 0 { super::DEFAULT_RATE } else { rate };
        self.blocks.clear();
        self.position = 0.0;
        self.decoded_samples = 0;
        self.decoded_bytes = 0;
        self.error = 0;
        self.in_use = true;
        self.playing = false;
        self.paused = false;
        self.stopped = false;
        0
    }

    /// Queues `count` `SceNgs2WaveformBlock`s at `at`. Flags 4 and 8
    /// replace the queue; flag 1 joins the blocks to what came before.
    fn add_blocks(&mut self, env: &Env, at: u64, count: u32, flags: u32) -> Result<u64, CallError> {
        let replace = flags & 12 != 0;
        if count > 256
            || (count != 0 && at == 0)
            || (!replace && self.blocks.len() + count as usize > 1024)
        {
            return Ok(ERROR_INVALID_CONTROL_SIZE);
        }
        let frame = self.channels * waveform::sample_bytes(self.kind);
        if frame == 0 {
            return Ok(ERROR_INVALID_WAVEFORM_FORMAT);
        }
        let mut blocks = Vec::with_capacity(count as usize);
        for i in 0..count as u64 {
            let mut d = [0u8; BLOCK_SIZE];
            env.mem.read(at + i * BLOCK_SIZE as u64, &mut d)?;
            let offset = u64_at(&d, 0);
            let length = u64_at(&d, 8);
            let skip = u32_at(&d, 20);
            let samples = u32_at(&d, 24);
            let available = length / frame as u64;
            if tracing() {
                eprintln!(
                    "ngs2: block +{offset:#x} len {length:#x} repeats {} skip {skip} samples {samples} flags {flags:#x}",
                    u32_at(&d, 16)
                );
            }
            if available < skip as u64 {
                return Ok(ERROR_INVALID_WAVEFORM_FORMAT);
            }
            // Balatro's streamed blocks give the whole track's length here.
            let frames = match samples as u64 {
                0 => available - skip as u64,
                n => n.min(available - skip as u64),
            };
            blocks.push(Block {
                address: self.source.wrapping_add(offset),
                length,
                skip,
                frames: frames.min(u32::MAX as u64) as u32,
                repeats: u32_at(&d, 16),
                repeated: 0,
                joins: flags & 1 != 0 && flags & 16 == 0,
                user_data: u64_at(&d, 32),
            });
        }
        if replace {
            self.blocks.clear();
            self.position = 0.0;
        }
        self.blocks.extend(blocks);
        Ok(0)
    }

    fn set_matrix(&mut self, env: &Env, id: u32, count: u32, at: u64) -> Result<u64, CallError> {
        if id >= 20 || !(1..=64).contains(&count) {
            return Ok(ERROR_INVALID_CONTROL_SIZE);
        }
        let mut d = vec![0u8; count as usize * 4];
        env.mem.read(at, &mut d)?;
        let levels: Vec<f32> = d
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        if levels.iter().any(|l| !l.is_finite() || l.abs() > 4.0) {
            return Ok(ERROR_INVALID_CONTROL_SIZE);
        }
        self.matrices.insert(id, levels);
        Ok(0)
    }

    /// One entry of a `sceNgs2VoiceControl` parameter list: header
    /// (size, next, id), then the parameter.
    fn apply(&mut self, env: &Env, id: u32, d: &[u8]) -> Result<u64, CallError> {
        let short = |n: usize| d.len() < n;
        if matches!(id >> 16, RACK_SAMPLER | RACK_CUSTOM_SAMPLER) {
            return Ok(match id & 0xffff {
                0 if short(32) => ERROR_INVALID_CONTROL_SIZE,
                0 => self.set_format(&d[8..]),
                1 if short(32) => ERROR_INVALID_CONTROL_SIZE,
                1 => {
                    self.source = u64_at(d, 8);
                    return self.add_blocks(env, u64_at(d, 24), u32_at(d, 20), u32_at(d, 16));
                }
                4 => {
                    self.blocks.iter_mut().for_each(|b| b.repeats = 0);
                    0
                }
                5 if short(12) => ERROR_INVALID_CONTROL_SIZE,
                5 => self.set_pitch(f32_at(d, 8)),
                _ => 0,
            });
        }
        Ok(match id {
            1 if short(24) => ERROR_INVALID_CONTROL_SIZE,
            1 => return self.set_matrix(env, u32_at(d, 8), u32_at(d, 12), u64_at(d, 16)),
            2 | 3 | 5 if short(16) || u32_at(d, 8) >= 32 => ERROR_INVALID_CONTROL_SIZE,
            2 => {
                let level = f32_at(d, 12);
                if !level.is_finite() || level.abs() > 4.0 {
                    return Ok(ERROR_INVALID_CONTROL_SIZE);
                }
                self.port(u32_at(d, 8)).volume = level;
                0
            }
            3 => {
                self.port(u32_at(d, 8)).matrix = Some(u32_at(d, 12));
                0
            }
            5 if short(24) => ERROR_INVALID_CONTROL_SIZE,
            5 => {
                let destination = u64_at(d, 16);
                let port = self.port(u32_at(d, 8));
                port.destination = destination;
                port.patch = (destination != 0) as u32;
                0
            }
            6 if short(12) => ERROR_INVALID_CONTROL_SIZE,
            6 => {
                self.kick(u32_at(d, 8));
                0
            }
            7 if short(28) => ERROR_INVALID_CONTROL_SIZE,
            7 => {
                self.callback = u64_at(d, 8);
                self.callback_data = u64_at(d, 16);
                self.callback_flags = u32_at(d, 24);
                0
            }
            0x2000_0000 | 0x3000_0000 if short(12) || !(1..=8).contains(&u32_at(d, 8)) => {
                ERROR_INVALID_CONTROL_SIZE
            }
            0x2000_0000 | 0x3000_0000 => {
                self.channels = u32_at(d, 8) as usize;
                0
            }
            0x3000_0004 if short(12) => ERROR_INVALID_CONTROL_SIZE,
            0x3000_0004 => {
                let gain = f32_at(d, 8);
                if !gain.is_finite() || gain.abs() > 32.0 {
                    return Ok(ERROR_INVALID_CONTROL_SIZE);
                }
                self.gain = gain;
                0
            }
            0x3000_0005 if short(12) || u32_at(d, 8) >= 32 => ERROR_INVALID_CONTROL_SIZE,
            0x3000_0005 => {
                self.output = u32_at(d, 8);
                0
            }
            _ => 0,
        })
    }

    fn set_pitch(&mut self, pitch: f32) -> u64 {
        if !pitch.is_finite() || !(0.0..=16.0).contains(&pitch) {
            return ERROR_INVALID_WAVEFORM_FORMAT;
        }
        self.pitch = pitch;
        0
    }

    /// One `SceNgs2VoiceCommand`-style record: id (component in bits 16
    /// to 23, slot in bits 24 to 31), flags, count, value.
    fn command(&mut self, env: &Env, r: &[u8; 16]) -> Result<u64, CallError> {
        let id = u32_at(r, 0);
        let value = u64_at(r, 8);
        let count = u16::from_le_bytes([r[6], r[7]]) as u32;
        let (component, command, slot) = ((id >> 16) & 0xff, id & 0x3fff, id >> 24);
        if id & 0xc000 != 0 {
            return Ok(ERROR_INVALID_CONTROL_ID);
        }
        match (component, command) {
            (0, 1) => {
                let channels = (env.mem.read_u64(value)? & 0xff) as usize;
                if (1..=8).contains(&channels) {
                    self.channels = channels;
                }
                self.in_use = true;
            }
            (0, 2) => self.kick(value as u32),
            (0, 5) => return self.set_matrix(env, slot, count, value),
            (0, 6) => {
                let level = f32::from_bits(value as u32);
                if slot >= 32 || !level.is_finite() || level.abs() > 4.0 {
                    return Ok(ERROR_INVALID_CONTROL_SIZE);
                }
                self.port(slot).volume = level;
            }
            (0, 7) if slot >= 32 => return Ok(ERROR_INVALID_CONTROL_SIZE),
            (0, 7) => self.port(slot).matrix = Some(value as u32),
            (0, 9) if slot >= 32 => return Ok(ERROR_INVALID_CONTROL_SIZE),
            (0, 9) => {
                let mut d = [0u8; 16];
                env.mem.read(value, &mut d)?;
                let port = self.port(slot);
                port.patch = u32_at(&d, 0);
                port.destination = u64_at(&d, 8);
                port.buffer = u32_at(&d, 8);
            }
            (1, 0) => {
                let mut d = [0u8; 24];
                env.mem.read(value, &mut d)?;
                return Ok(self.set_format(&d));
            }
            (1, 1) => return self.add_blocks(env, value, count, r[4] as u32),
            (1, 2) => self.source = value,
            (1, 3) => return Ok(self.set_pitch(f32::from_bits(value as u32))),
            (1, 4) => self.blocks.iter_mut().for_each(|b| b.repeats = 0),
            _ => {}
        }
        Ok(0)
    }
}

fn u32_at(d: &[u8], at: usize) -> u32 {
    d.get(at..at + 4)
        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap_or_default()))
}

fn u64_at(d: &[u8], at: usize) -> u64 {
    d.get(at..at + 8)
        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap_or_default()))
}

fn f32_at(d: &[u8], at: usize) -> f32 {
    f32::from_bits(u32_at(d, at))
}

/// `sceNgs2VoiceControl(voice, params)`: a list of parameters, each with
/// a signed offset to the next (0 ends the list).
pub(super) fn control(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, mut at) = (env.arg(0), env.arg(1));
    if at == 0 {
        return ret(ERROR_INVALID_CONTROL_ADDRESS);
    }
    let mut s = lock(&env.kernel.ngs2.state);
    let Some(voice) = s.voices.get_mut(&handle) else {
        return ret(ERROR_INVALID_VOICE_HANDLE);
    };
    for _ in 0..256 {
        let mut d = [0u8; 128];
        env.mem.read(at, &mut d[..8])?;
        let size = u16::from_le_bytes([d[0], d[1]]) as usize;
        let next = i16::from_le_bytes([d[2], d[3]]);
        let id = u32_at(&d, 4);
        if !(8..=4096).contains(&size) {
            return ret(ERROR_INVALID_CONTROL_SIZE);
        }
        let len = size.min(d.len());
        env.mem.read(at, &mut d[..len])?;
        if tracing() {
            eprintln!(
                "ngs2: voice {handle:#x} control {id:#x} {:02x?}",
                &d[8..len]
            );
        }
        let r = voice.apply(env, id, &d[..len])?;
        if tracing() && r != 0 {
            eprintln!("ngs2: voice {handle:#x} control {id:#x} failed with {r:#x}");
        }
        if r != 0 || next == 0 {
            return ret(r);
        }
        at = at.wrapping_add_signed(next as i64);
    }
    ret(ERROR_INVALID_CONTROL_SIZE)
}

/// `sceNgs2VoiceRunCommands(voice, commands, count)`.
pub(super) fn run_commands(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, at, count) = (env.arg(0), env.arg(1), env.arg(2));
    if count > 4096 || (count != 0 && at == 0) {
        return ret(ERROR_INVALID_CONTROL_ADDRESS);
    }
    let mut s = lock(&env.kernel.ngs2.state);
    let Some(voice) = s.voices.get_mut(&handle) else {
        return ret(ERROR_INVALID_VOICE_HANDLE);
    };
    for i in 0..count {
        let mut r = [0u8; 16];
        env.mem.read(at + 16 * i, &mut r)?;
        if tracing() {
            eprintln!("ngs2: voice {handle:#x} command {r:02x?}");
        }
        let result = voice.command(env, &r)?;
        if result != 0 {
            return ret(result);
        }
    }
    ret(0)
}

savestate::record!(Port {
    volume,
    matrix,
    patch,
    destination,
    buffer
});
savestate::record!(Block {
    address,
    length,
    skip,
    frames,
    repeats,
    repeated,
    joins,
    user_data
});
savestate::record!(Voice {
    rack,
    index,
    kind,
    channels,
    rate,
    source,
    blocks,
    position,
    playing,
    paused,
    in_use,
    stopped,
    error,
    pitch,
    gain,
    output,
    decoded_samples,
    decoded_bytes,
    callback,
    callback_data,
    callback_flags,
    ports,
    matrices
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_move_a_voice_through_its_states() {
        let mut v = Voice::new(1, 0);
        assert_eq!(v.flags(), 0);
        v.kick(EVENT_PLAY);
        assert_eq!(v.flags(), 1 | 2 | 32);
        v.kick(EVENT_PAUSE);
        assert_eq!(v.flags() & 4, 4);
        v.kick(EVENT_STOP);
        assert_eq!(v.flags(), 1 | 8 | 32);
        v.kick(EVENT_KILL);
        assert_eq!(v.flags(), 0);
    }
}
