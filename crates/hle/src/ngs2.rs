//! libSceNgs2, the sample-mixing audio library, without sound: systems,
//! racks and voices are handles, voices never play, and rendering fills
//! the title's buffers with silence. Prototypes follow the PS4 ones as
//! shadPS4 knows them, checked against Balatro's calls; unverified.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{CallError, Env, Function, Outcome};

pub static FUNCTIONS: &[Function] = &[];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f("sceNgs2ParseWaveformData", parse_waveform_data),
    f("sceNgs2ParseWaveformFile", parse_waveform_file),
    f("sceNgs2SystemCreateWithAllocator", system_create),
    f("sceNgs2SystemDestroy", ok),
    f("sceNgs2SystemGetInfo", ok),
    f("sceNgs2SystemRender", system_render),
    f("sceNgs2RackCreateWithAllocator", rack_create),
    f("sceNgs2RackDestroy", ok),
    f("sceNgs2RackGetVoiceHandle", rack_voice),
    f("sceNgs2VoiceControl", ok),
    f("sceNgs2VoiceRunCommands", ok),
    f("sceNgs2VoiceGetState", voice_state),
    f("sceNgs2VoiceGetStateFlags", voice_state_flags),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceNgs2",
        name,
        func,
    }
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

static NEXT: AtomicU64 = AtomicU64::new(0x1000);

fn new_handle(env: &Env, out: u64) -> Result<Outcome, CallError> {
    env.mem
        .write_u64(out, NEXT.fetch_add(1, Ordering::Relaxed))?;
    Ok(Outcome::Return(0))
}

/// `sceNgs2SystemCreateWithAllocator(option, allocator, &handle)`.
fn system_create(env: &mut Env) -> Result<Outcome, CallError> {
    new_handle(env, env.arg(2))
}

/// `sceNgs2RackCreateWithAllocator(system, rack_id, option, allocator, &handle)`.
fn rack_create(env: &mut Env) -> Result<Outcome, CallError> {
    new_handle(env, env.arg(4))
}

/// `sceNgs2RackGetVoiceHandle(rack, index, &handle)`.
fn rack_voice(env: &mut Env) -> Result<Outcome, CallError> {
    new_handle(env, env.arg(2))
}

/// `sceNgs2VoiceGetState(voice, state, size)`: all zeros.
fn voice_state(env: &mut Env) -> Result<Outcome, CallError> {
    let size = env.arg(2).min(0x1000) as usize;
    env.mem.write(env.arg(1), &vec![0u8; size])?;
    Ok(Outcome::Return(0))
}

/// `sceNgs2VoiceGetStateFlags(voice, &flags)`: not playing.
fn voice_state_flags(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), &0u32.to_le_bytes())?;
    Ok(Outcome::Return(0))
}

/// `sceNgs2SystemRender(system, buffers, count)`: every buffer gets silence.
fn system_render(env: &mut Env) -> Result<Outcome, CallError> {
    let (list, count) = (env.arg(1), env.arg(2) as u32 as u64);
    for k in 0..count.min(64) {
        let at = list + 24 * k;
        let (buf, size) = (env.mem.read_u64(at)?, env.mem.read_u64(at + 8)?);
        if buf != 0 {
            env.mem.write(buf, &vec![0u8; size.min(1 << 24) as usize])?;
        }
    }
    Ok(Outcome::Return(0))
}

/// `sceNgs2WaveformType` of 16-bit little-endian PCM (unverified).
const WAVEFORM_PCM_I16L: u32 = 0x11;

/// The `fmt ` fields (channels, rate, bits) and `data` chunk (offset,
/// size) of a RIFF WAVE file.
fn riff(data: &[u8]) -> Option<(u32, u32, u32, u32, u32)> {
    if data.get(0..4)? != b"RIFF" || data.get(8..12)? != b"WAVE" {
        return None;
    }
    let u16_at = |at: usize| Some(u16::from_le_bytes(data.get(at..at + 2)?.try_into().ok()?) as u32);
    let u32_at = |at: usize| Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?));
    let (mut at, mut fmt, mut body) = (12, None, None);
    while at + 8 <= data.len() {
        let size = u32_at(at + 4)?;
        match data.get(at..at + 4)? {
            b"fmt " => fmt = Some((u16_at(at + 10)?, u32_at(at + 12)?, u16_at(at + 22)?)),
            b"data" => body = Some((at as u32 + 8, size)),
            _ => {}
        }
        at += 8 + size as usize + (size as usize & 1);
    }
    let ((channels, rate, bits), (offset, size)) = (fmt?, body?);
    Some((channels, rate, bits, offset, size))
}

/// `sceNgs2ParseWaveformData(data, size, &info)` for WAVE files, in the
/// waveform-info layout shadPS4 uses (from memory; unverified).
fn parse_waveform_data(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, size, out) = (env.arg(0), env.arg(1).min(64 << 20) as usize, env.arg(2));
    let mut data = vec![0u8; size.min(4096)];
    env.mem.read(at, &mut data)?;
    write_waveform_info(env, &data, out)
}

/// `sceNgs2ParseWaveformFile(path, offset, &info)`.
fn parse_waveform_file(env: &mut Env) -> Result<Outcome, CallError> {
    let (path, offset, out) = (crate::libc::cstr(env, env.arg(0))?, env.arg(1) as usize, env.arg(2));
    let data = env
        .kernel
        .vfs
        .resolve(&path)
        .and_then(|(host, _)| std::fs::read(host).ok())
        .map(|d| d.get(offset..).unwrap_or_default().iter().take(4096).copied().collect::<Vec<u8>>())
        .unwrap_or_default();
    write_waveform_info(env, &data, out)
}

fn write_waveform_info(env: &Env, data: &[u8], out: u64) -> Result<Outcome, CallError> {
    let Some((channels, rate, bits, offset, bytes)) = riff(data) else {
        return Ok(Outcome::Return(ERROR_INVALID_WAVEFORM));
    };
    let frame = (channels * bits / 8).max(1);
    let samples = bytes / frame;
    let mut info = [0u32; 18 + 4 * 8];
    info[0] = WAVEFORM_PCM_I16L;
    info[1] = channels;
    info[2] = rate;
    info[6] = offset;
    info[7] = bytes;
    info[9] = samples;
    info[10] = samples;
    info[11] = frame;
    info[12] = 1;
    info[13] = 1;
    info[14] = frame;
    info[15] = 1;
    info[17] = 1;
    info[18] = offset;
    info[19] = bytes;
    info[22] = samples;
    let b: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    env.mem.write(out, &b)?;
    Ok(Outcome::Return(0))
}

/// Returned for data that is not a WAVE file (our guess).
const ERROR_INVALID_WAVEFORM: u64 = 0x804A_0000;
