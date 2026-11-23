//! libSceAudioOut2, the PS5's audio output: a context owns ports, a title
//! points each port at one grain of samples with `PortSetAttributes`, and
//! `ContextAdvance`/`ContextPush` play the grain. Grains are mixed to
//! stereo and go where AudioOut's ports go (the host device, a WAVE dump).
//! Structure layouts, attribute ids and error codes follow SharpEmu-mac's
//! `AudioOut2Exports.cs` (GPL-2.0-or-later); the pacing is ours.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::{RATE, to_stereo, wait, wav};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

const ERROR_INVALID_PARAM: u64 = 0x8026_8001;
const ERROR_INVALID_POINTER: u64 = 0x8026_800C;
const ERROR_INVALID_USER: u64 = 0x8026_8010;

const CONTEXT_PARAM_SIZE: usize = 0x40;
const PORT_PARAM_SIZE: usize = 0x40;
const ATTRIBUTE_SIZE: u64 = 0x18;
const ATTRIBUTE_PCM: u32 = 0;
const ATTRIBUTE_GAIN: u32 = 1;
const DEFAULT_GRAIN: u32 = 256;
const DEFAULT_QUEUE_DEPTH: u32 = 4;

struct Context {
    grain: u32,
    depth: u32,
    period: Duration,
    /// When the grains queued so far have played out; `None` before the first.
    next: Option<Instant>,
    /// An `Advance` played a grain that its `Push` has not waited for.
    advanced: bool,
    /// The context's queue in the host mixer.
    queue: u64,
    wav: Option<wav::Wav>,
}

struct Port {
    context: u64,
    channels: usize,
    float: bool,
    gain: f32,
    /// The grain to play next, 0 for none.
    pcm: u64,
}

#[derive(Default)]
pub struct Out2 {
    contexts: Mutex<HashMap<u64, Context>>,
    ports: Mutex<HashMap<u64, Port>>,
    next_id: AtomicU64,
}

pub static FUNCTIONS: &[Function] = &[
    f("sceAudioOut2Initialize", ok),
    f("sceAudioOut2ContextResetParam", reset_param),
    f("sceAudioOut2ContextQueryMemory", query_memory),
    f("sceAudioOut2ContextCreate", context_create),
    f("sceAudioOut2ContextDestroy", context_destroy),
    f("sceAudioOut2UserCreate", user_create),
    f("sceAudioOut2UserDestroy", ok),
    f("sceAudioOut2PortCreate", port_create),
    f("sceAudioOut2PortDestroy", port_destroy),
    f("sceAudioOut2PortSetAttributes", port_set_attributes),
    f("sceAudioOut2ContextAdvance", advance),
    f("sceAudioOut2ContextPush", push),
    f("sceAudioOut2ContextGetQueueLevel", queue_level),
    f("sceAudioOut2PortGetState", port_get_state),
    f("sceAudioOut2GetSpeakerInfo", speaker_info),
    f("sceAudioOut2GetSpeakerArrayMemorySize", speaker_array_size),
    f("sceAudioOut2SpeakerArrayCreate", speaker_array_create),
    f("sceAudioOut2SpeakerArrayDestroy", ok),
    f(
        "sceAudioOut2GetSpeakerArrayAmbisonicsCoefficients",
        ambisonics_coefficients,
    ),
];

const PORT_STATE_SIZE: usize = 0x40;
/// `SceAudioOut2PortState`'s output: connected to the primary output.
const PORT_STATE_PRIMARY: u16 = 0x01;
const SPEAKER_INFO_SIZE: usize = 0x50;
const SPEAKER_ANGLES: usize = 0x10;
/// Front left and front right.
const STEREO_SPEAKERS: u32 = 0b11;

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAudioOut2",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn read_u32(env: &Env, at: u64) -> Result<u32, CallError> {
    let mut b = [0u8; 4];
    env.mem.read(at, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// `sceAudioOut2ContextResetParam(param)`: SharpEmu-mac's defaults.
fn reset_param(env: &mut Env) -> Result<Outcome, CallError> {
    let at = env.arg(0);
    if at == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let mut b = [0u8; CONTEXT_PARAM_SIZE];
    for (k, v) in [256u32, 256, 0, DEFAULT_QUEUE_DEPTH, DEFAULT_GRAIN, 1]
        .into_iter()
        .enumerate()
    {
        b[4 * k..4 * k + 4].copy_from_slice(&v.to_le_bytes());
    }
    env.mem.write(at, &b)?;
    ret(0)
}

/// The queue depth (offset 0x0C) and grain (0x10) a context parameter
/// block asks for, with defaults for values out of range.
fn depth_and_grain(env: &Env, at: u64) -> Result<(u32, u32), CallError> {
    let depth = read_u32(env, at + 0x0C)?;
    let grain = read_u32(env, at + 0x10)?;
    Ok((
        if (1..=32).contains(&depth) {
            depth
        } else {
            DEFAULT_QUEUE_DEPTH
        },
        if (64..=0x4000).contains(&grain) {
            grain
        } else {
            DEFAULT_GRAIN
        },
    ))
}

/// `sceAudioOut2ContextQueryMemory(param, *size)`: the context's memory
/// lives on the host; the size asked for is SharpEmu-mac's.
fn query_memory(env: &mut Env) -> Result<Outcome, CallError> {
    let (param, out) = (env.arg(0), env.arg(1));
    if param == 0 || out == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let (depth, _) = depth_and_grain(env, param)?;
    env.mem.write_u64(out, 0x1_0000 + depth as u64 * 0x590)?;
    ret(0)
}

/// `sceAudioOut2ContextCreate(param, memory, size, *context)`.
fn context_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (param, memory, size, out) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    if param == 0 || memory == 0 || out == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    if size == 0 {
        return ret(ERROR_INVALID_PARAM);
    }
    let (depth, grain) = depth_and_grain(env, param)?;
    let audio = &env.kernel.audio;
    audio.start_device();
    let queue = audio.new_queue();
    let wav = audio.open_wav(queue);
    let handle = env
        .kernel
        .audio
        .out2
        .next_id
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    lock(&audio.out2.contexts).insert(
        handle,
        Context {
            grain,
            depth,
            period: Duration::from_secs_f64(grain as f64 / RATE as f64),
            next: None,
            advanced: false,
            queue,
            wav,
        },
    );
    env.mem.write_u64(out, handle)?;
    ret(0)
}

fn context_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let audio = &env.kernel.audio;
    if let Some(c) = lock(&audio.out2.contexts).remove(&env.arg(0)) {
        lock(&audio.mixer.queues).remove(&c.queue);
    }
    lock(&audio.out2.ports).retain(|_, p| p.context != env.arg(0));
    ret(0)
}

/// `sceAudioOut2UserCreate(user_id, *handle)`.
fn user_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (user, out) = (env.arg(0) as i32, env.arg(1));
    if out == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    if user == -1 {
        return ret(ERROR_INVALID_USER);
    }
    let handle = env
        .kernel
        .audio
        .out2
        .next_id
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    env.mem.write_u64(out, handle)?;
    ret(0)
}

/// `(context, param, *port)`; format word at 4: type in the low bits
/// (0 float, 1 S16), channels in bits 8 to 15.
fn port_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (context, param, out) = (env.arg(0), env.arg(1), env.arg(2));
    if param == 0 || out == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let mut b = [0u8; PORT_PARAM_SIZE];
    env.mem.read(param, &mut b)?;
    let format = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let channels = match (format >> 8) & 0xff {
        0 => 2,
        n @ 1..=16 => n as usize,
        _ => return ret(ERROR_INVALID_PARAM),
    };
    let float = match format & 0x7f {
        0 => true,
        1 => false,
        _ => return ret(ERROR_INVALID_PARAM),
    };
    let out2 = &env.kernel.audio.out2;
    let handle = out2.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    lock(&out2.ports).insert(
        handle,
        Port {
            context,
            channels,
            float,
            gain: 1.0,
            pcm: 0,
        },
    );
    env.mem.write_u64(out, handle)?;
    ret(0)
}

/// `sceAudioOut2PortGetState(port, *state)`: output, channels, volume.
fn port_get_state(env: &mut Env) -> Result<Outcome, CallError> {
    let out = env.arg(1);
    if out == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let channels = lock(&env.kernel.audio.out2.ports)
        .get(&env.arg(0))
        .map_or(2, |p| p.channels);
    let mut state = [0u8; PORT_STATE_SIZE];
    state[..2].copy_from_slice(&PORT_STATE_PRIMARY.to_le_bytes());
    state[2] = channels as u8;
    state[4..6].copy_from_slice(&127i16.to_le_bytes());
    env.mem.write(out, &state)?;
    ret(0)
}

/// `sceAudioOut2GetSpeakerInfo(*info, flags)`: a stereo TV, speakers at
/// -30 and 30 degrees.
fn speaker_info(env: &mut Env) -> Result<Outcome, CallError> {
    let out = env.arg(0);
    if out == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let mut info = [0u8; SPEAKER_INFO_SIZE];
    info[4..8].copy_from_slice(&STEREO_SPEAKERS.to_le_bytes());
    info[SPEAKER_ANGLES..SPEAKER_ANGLES + 2].copy_from_slice(&(-30i16).to_le_bytes());
    info[SPEAKER_ANGLES + 4..SPEAKER_ANGLES + 6].copy_from_slice(&30i16.to_le_bytes());
    env.mem.write(out, &info)?;
    ret(0)
}

const SPEAKER_ARRAY_HEADER: u64 = 0x40;
const SPEAKER_ARRAY_ENTRY: u64 = 0x100;
const SPEAKER_ARRAY_SCRATCH: u64 = 0x400;
const SPEAKER_ARRAY_MAX: u32 = 32;
const SPEAKER_ARRAY_DEFAULT: u32 = 8;

fn speaker_count(n: u32) -> u32 {
    if (1..=SPEAKER_ARRAY_MAX).contains(&n) {
        n
    } else {
        SPEAKER_ARRAY_DEFAULT
    }
}

fn speaker_array_bytes(speakers: u32) -> u64 {
    SPEAKER_ARRAY_HEADER + speakers as u64 * SPEAKER_ARRAY_ENTRY + SPEAKER_ARRAY_SCRATCH
}

/// `(speakers, is_3d, is_ambisonics)`, returning the size. The layout is
/// not public; the size is SharpEmu-mac's choice.
fn speaker_array_size(env: &mut Env) -> Result<Outcome, CallError> {
    ret(speaker_array_bytes(speaker_count(env.arg(0) as u32)))
}

/// `(*handle, vbap_param, ambisonics_param)`: the handle is the caller's
/// buffer, with SharpEmu-mac's header (GTA V reads 0x34).
fn speaker_array_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, param) = (env.arg(0), env.arg(1));
    if out == 0 || param == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let mut p = [0u8; 0x20];
    env.mem.read(param, &mut p)?;
    let speakers = speaker_count(u32::from_le_bytes(p[8..12].try_into().unwrap()));
    let buffer = u64::from_le_bytes(p[16..24].try_into().unwrap());
    let size = u64::from_le_bytes(p[24..32].try_into().unwrap());
    if buffer == 0 || size < speaker_array_bytes(speakers) {
        return ret(ERROR_INVALID_PARAM);
    }
    let mut header = [0u8; SPEAKER_ARRAY_HEADER as usize];
    header[0..4].copy_from_slice(&(SPEAKER_ARRAY_HEADER as u32).to_le_bytes());
    header[4..8].copy_from_slice(&speakers.to_le_bytes());
    header[0x34..0x38].copy_from_slice(&1u32.to_le_bytes());
    env.mem.write(buffer, &header)?;
    env.mem.write_u64(out, buffer)?;
    ret(0)
}

/// `(handle, channel, *coefficients, count)`: all zero, nothing is decoded.
fn ambisonics_coefficients(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, count) = (env.arg(2), env.arg(3).min(1024));
    if out == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    env.mem.write(out, &vec![0u8; 4 * count as usize])?;
    ret(0)
}

fn port_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.audio.out2.ports).remove(&env.arg(0));
    ret(0)
}

/// `(port, list, count)`, entries `{ u32 id, u64 value, u64 size }`; PCM's
/// value points at the grain's address, gain's at a float.
fn port_set_attributes(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, list, count) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if list == 0 {
        return ret(if count == 0 { 0 } else { ERROR_INVALID_POINTER });
    }
    for k in 0..count.min(32) as u64 {
        let at = list + ATTRIBUTE_SIZE * k;
        let id = read_u32(env, at)?;
        let value = env.mem.read_u64(at + 8)?;
        if value == 0 {
            continue;
        }
        let mut ports = lock(&env.kernel.audio.out2.ports);
        let Some(p) = ports.get_mut(&handle) else {
            return ret(0);
        };
        match id {
            ATTRIBUTE_PCM => p.pcm = env.mem.read_u64(value)?,
            ATTRIBUTE_GAIN => {
                let g = f32::from_bits(read_u32(env, value)?);
                if g.is_finite() {
                    p.gain = g;
                }
            }
            _ => {}
        }
    }
    ret(0)
}

/// One grain of context `handle`: every port's samples to stereo, summed.
/// Each port's grain is used once.
fn mix(env: &Env, handle: u64, grain: usize) -> Result<Vec<f32>, CallError> {
    let mut sum = vec![0.0f32; grain * 2];
    let mut ports = lock(&env.kernel.audio.out2.ports);
    for p in ports
        .values_mut()
        .filter(|p| p.context == handle && p.pcm != 0)
    {
        let bytes = if p.float { 4 } else { 2 };
        let mut raw = vec![0u8; grain * p.channels * bytes];
        env.mem.read(p.pcm, &mut raw)?;
        p.pcm = 0;
        let samples: Vec<f32> = raw
            .chunks_exact(bytes)
            .map(|b| {
                let x = if p.float {
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                } else {
                    i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0
                };
                if x.is_finite() { x * p.gain } else { 0.0 }
            })
            .collect();
        for (s, x) in sum.iter_mut().zip(to_stereo(&samples, p.channels)) {
            *s += x;
        }
    }
    Ok(sum)
}

/// Plays one grain of context `handle` and returns when its queue has a
/// free slot again, with the context's mixer queue; `None` for no context.
fn play_grain(env: &Env, handle: u64) -> Result<Option<(Instant, u64)>, CallError> {
    let Some(grain) = lock(&env.kernel.audio.out2.contexts)
        .get(&handle)
        .map(|c| c.grain as usize)
    else {
        return Ok(None);
    };
    let stereo = mix(env, handle, grain)?;
    let audio = &env.kernel.audio;
    let mut contexts = lock(&audio.out2.contexts);
    let Some(c) = contexts.get_mut(&handle) else {
        return Ok(None);
    };
    let now = Instant::now();
    let next = c.next.unwrap_or(now);
    if now > next + c.period {
        diag::problems::record(
            diag::problems::Kind::AudioLate,
            "more than one grain late (a dropout on a device)",
            format!("AudioOut2 context {handle} ({} frames)", c.grain),
        );
    }
    let next = next.max(now) + c.period;
    c.next = Some(next);
    let free = next - c.period * (c.depth - 1);
    if let Some(w) = &mut c.wav
        && let Err(e) = w.append(&stereo)
    {
        tracing::warn!("audio dump stopped: {e}");
        c.wav = None;
    }
    let queue = c.queue;
    drop(contexts);
    if audio.device_running()
        && let Some(q) = lock(&audio.mixer.queues).get_mut(&queue)
    {
        q.extend(stereo);
    }
    Ok(Some((free, queue)))
}

/// `sceAudioOut2ContextAdvance(context)`: plays a grain without waiting.
fn advance(env: &mut Env) -> Result<Outcome, CallError> {
    let handle = env.arg(0);
    if play_grain(env, handle)?.is_some()
        && let Some(c) = lock(&env.kernel.audio.out2.contexts).get_mut(&handle)
    {
        c.advanced = true;
    }
    ret(0)
}

/// `sceAudioOut2ContextPush(context, blocking)`: plays a grain unless an
/// `Advance` already did, then, if `blocking`, waits for a free slot.
fn push(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, blocking) = (env.arg(0), env.arg(1) as u32 != 0);
    let advanced = lock(&env.kernel.audio.out2.contexts)
        .get_mut(&handle)
        .map(|c| std::mem::take(&mut c.advanced));
    let slot = match advanced {
        None => return ret(0),
        Some(true) => lock(&env.kernel.audio.out2.contexts).get(&handle).map(|c| {
            let next = c.next.unwrap_or_else(Instant::now);
            (next - c.period * (c.depth - 1), c.queue)
        }),
        Some(false) => play_grain(env, handle)?,
    };
    if blocking && let Some((free, queue)) = slot {
        wait(env, queue, free);
    }
    ret(0)
}

/// `sceAudioOut2ContextGetQueueLevel(context, *queued, *free)`: grains not
/// yet played by the context's clock, and slots left (32 bits each).
fn queue_level(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, queued_at, free_at) = (env.arg(0), env.arg(1), env.arg(2));
    let (queued, depth) = match lock(&env.kernel.audio.out2.contexts).get(&handle) {
        Some(c) => {
            let ahead = c.next.map_or(Duration::ZERO, |n| {
                n.saturating_duration_since(Instant::now())
            });
            let n = ahead.as_nanos().div_ceil(c.period.as_nanos().max(1)) as u32;
            (n.min(c.depth), c.depth)
        }
        None => (0, DEFAULT_QUEUE_DEPTH),
    };
    if queued_at != 0 {
        env.mem.write(queued_at, &queued.to_le_bytes())?;
    }
    if free_at != 0 && free_at != queued_at {
        env.mem.write(free_at, &(depth - queued).to_le_bytes())?;
    }
    ret(0)
}

impl Out2 {
    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        let contexts: HashMap<u64, (u32, u32, Duration, bool, u64)> = lock(&self.contexts)
            .iter()
            .map(|(&id, c)| (id, (c.grain, c.depth, c.period, c.advanced, c.queue)))
            .collect();
        let ports: HashMap<u64, (u64, usize, bool, f32, u64)> = lock(&self.ports)
            .iter()
            .map(|(&id, p)| (id, (p.context, p.channels, p.float, p.gain, p.pcm)))
            .collect();
        e.put(&contexts)
            .put(&ports)
            .put(&self.next_id.load(Ordering::Relaxed));
    }

    /// Loads what [`Out2::save`] wrote and returns the mixer queues it uses.
    pub(crate) fn load(&self, d: &mut savestate::Decoder) -> Result<Vec<u64>, savestate::Error> {
        let contexts: HashMap<u64, (u32, u32, Duration, bool, u64)> = d.get()?;
        let queues = contexts.values().map(|c| c.4).collect();
        *lock(&self.contexts) = contexts
            .into_iter()
            .map(|(id, (grain, depth, period, advanced, queue))| {
                let c = Context {
                    grain,
                    depth,
                    period,
                    next: None,
                    advanced,
                    queue,
                    wav: None,
                };
                (id, c)
            })
            .collect();
        let ports: HashMap<u64, (u64, usize, bool, f32, u64)> = d.get()?;
        *lock(&self.ports) = ports
            .into_iter()
            .map(|(id, (context, channels, float, gain, pcm))| {
                let p = Port {
                    context,
                    channels,
                    float,
                    gain,
                    pcm,
                };
                (id, p)
            })
            .collect();
        self.next_id.store(d.get()?, Ordering::Relaxed);
        Ok(queues)
    }
}
