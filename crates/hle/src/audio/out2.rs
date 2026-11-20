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
    /// When the grains queued so far have played out.
    next: Instant,
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
];

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
        if (1..=32).contains(&depth) { depth } else { DEFAULT_QUEUE_DEPTH },
        if (64..=0x4000).contains(&grain) { grain } else { DEFAULT_GRAIN },
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
    let handle = env.kernel.audio.out2.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    lock(&audio.out2.contexts).insert(
        handle,
        Context {
            grain,
            depth,
            period: Duration::from_secs_f64(grain as f64 / RATE as f64),
            next: Instant::now(),
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
    let handle = env.kernel.audio.out2.next_id.fetch_add(1, Ordering::Relaxed) + 1;
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
    for p in ports.values_mut().filter(|p| p.context == handle && p.pcm != 0) {
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
    c.next = c.next.max(Instant::now()) + c.period;
    let free = c.next - c.period * (c.depth - 1);
    if let Some(w) = &mut c.wav
        && let Err(e) = w.append(&stereo)
    {
        eprintln!("ps5slop: audio dump stopped: {e}");
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
        Some(true) => lock(&env.kernel.audio.out2.contexts)
            .get(&handle)
            .map(|c| (c.next - c.period * (c.depth - 1), c.queue)),
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
            let ahead = c.next.saturating_duration_since(Instant::now());
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
