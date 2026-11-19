//! libSceNgs2, the sample-mixing audio library: systems, racks and voices,
//! sampler voices playing PCM waveform blocks, and a render that mixes
//! them into the title's buffers through the mastering rack. Ported from
//! SharpEmu-mac's `Ngs2Exports*.cs` (GPL-2.0-or-later); see `docs/audio.md`
//! for what is verified.

mod mix;
mod voice;
pub mod waveform;

use std::collections::HashMap;
use std::sync::Mutex;

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

use voice::Voice;

const ERROR_INVALID_OUT_ADDRESS: u64 = 0x804A_0053;
const ERROR_INVALID_SYSTEM_HANDLE: u64 = 0x804A_0230;
const ERROR_INVALID_RACK_HANDLE: u64 = 0x804A_0261;
const ERROR_INVALID_VOICE_HANDLE: u64 = 0x804A_0300;
const ERROR_INVALID_PARSE_OUT_ADDRESS: u64 = 0x804A_8010;
const ERROR_INVALID_WAVEFORM_ADDRESS: u64 = 0x804A_8055;
const ERROR_INVALID_WAVEFORM_DATA: u64 = 0x804A_8430;
const ERROR_INVALID_WAVEFORM_FORMAT: u64 = 0x804A_8431;
const ERROR_INVALID_CONTROL_ADDRESS: u64 = 0x804A_8360;
const ERROR_INVALID_CONTROL_SIZE: u64 = 0x804A_8361;
const ERROR_INVALID_CONTROL_ID: u64 = 0x804A_8362;

const DEFAULT_GRAIN: usize = 256;
const DEFAULT_RATE: u32 = 48_000;

pub const RACK_SAMPLER: u32 = 0x1000;
pub const RACK_MASTERING: u32 = 0x3000;
const RACK_CUSTOM_SAMPLER: u32 = 0x4001;
const RACK_CUSTOM_MASTERING: u32 = 0x4003;

pub static FUNCTIONS: &[Function] = &[
    f("sceNgs2ParseWaveformData", parse_waveform_data),
    f("sceNgs2ParseWaveformFile", parse_waveform_file),
    f("sceNgs2SystemCreateWithAllocator", system_create),
    f("sceNgs2SystemCreate", system_create),
    f("sceNgs2SystemDestroy", system_destroy),
    f("sceNgs2SystemRender", system_render),
    f("sceNgs2SystemSetGrainSamples", system_set_grain),
    f("sceNgs2SystemSetSampleRate", system_set_rate),
    f("sceNgs2SystemLock", system_check),
    f("sceNgs2SystemUnlock", system_check),
    f("sceNgs2RackCreateWithAllocator", rack_create),
    f("sceNgs2RackCreate", rack_create),
    f("sceNgs2RackDestroy", rack_destroy),
    f("sceNgs2RackGetVoiceHandle", rack_voice),
    f("sceNgs2VoiceControl", voice::control),
    f("sceNgs2VoiceRunCommands", voice::run_commands),
    f("sceNgs2VoiceGetState", voice_state),
    f("sceNgs2VoiceGetStateFlags", voice_state_flags),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[f("sceNgs2SystemGetInfo", ok)];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceNgs2",
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

/// Whether `PS5SLOP_NGS2_TRACE` asks for a log of voice controls.
fn tracing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PS5SLOP_NGS2_TRACE").is_some())
}

struct System {
    grain: usize,
    rate: u32,
    /// Guest memory for the info block passed to voice callbacks.
    callback_info: u64,
}

struct Rack {
    system: u64,
    id: u32,
}

/// A guest callback owed after a render, with its info block.
struct Callback {
    handler: u64,
    info: [u8; 64],
}

/// All NGS2 objects of the process. Handles are small numbers from one
/// counter, so a handle names exactly one system, rack or voice.
#[derive(Default)]
pub struct Ngs2 {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    next: u64,
    systems: HashMap<u64, System>,
    racks: HashMap<u64, Rack>,
    voices: HashMap<u64, Voice>,
}

impl State {
    fn handle(&mut self) -> u64 {
        self.next = self.next.max(0x1000) + 1;
        self.next
    }

    fn rack_id(&self, voice: &Voice) -> u32 {
        self.racks.get(&voice.rack).map_or(0, |r| r.id)
    }
}

/// `sceNgs2SystemCreate[WithAllocator](option, buffer_or_allocator, &handle)`.
/// The option struct may carry the grain size and sample rate.
fn system_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (option, out) = (env.arg(0), env.arg(2));
    if out == 0 {
        return ret(ERROR_INVALID_OUT_ADDRESS);
    }
    let mut system = System {
        grain: DEFAULT_GRAIN,
        rate: DEFAULT_RATE,
        callback_info: 0,
    };
    if option != 0 {
        let size = env.mem.read_u64(option)?;
        let at = option + if size >= 144 { 112 } else { 32 };
        let mut b = [0u8; 8];
        env.mem.read(at, &mut b)?;
        let grain = u32::from_le_bytes(b[0..4].try_into().unwrap_or_default());
        let rate = u32::from_le_bytes(b[4..8].try_into().unwrap_or_default());
        if (1..=8192).contains(&grain) {
            system.grain = grain as usize;
        }
        if (8000..=192_000).contains(&rate) {
            system.rate = rate;
        }
    }
    let mut s = lock(&env.kernel.ngs2.state);
    let handle = s.handle();
    s.systems.insert(handle, system);
    env.mem.write_u64(out, handle)?;
    ret(0)
}

fn system_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let handle = env.arg(0);
    let mut s = lock(&env.kernel.ngs2.state);
    if s.systems.remove(&handle).is_none() {
        return ret(ERROR_INVALID_SYSTEM_HANDLE);
    }
    let racks: Vec<u64> = s
        .racks
        .iter()
        .filter(|(_, r)| r.system == handle)
        .map(|(&h, _)| h)
        .collect();
    for r in racks {
        remove_rack(&mut s, r);
    }
    ret(0)
}

fn remove_rack(s: &mut State, rack: u64) {
    s.racks.remove(&rack);
    s.voices.retain(|_, v| v.rack != rack);
}

fn system_check(env: &mut Env) -> Result<Outcome, CallError> {
    let known = lock(&env.kernel.ngs2.state)
        .systems
        .contains_key(&env.arg(0));
    ret(if known {
        0
    } else {
        ERROR_INVALID_SYSTEM_HANDLE
    })
}

fn system_set_grain(env: &mut Env) -> Result<Outcome, CallError> {
    let grain = env.arg(1) as u32;
    let mut s = lock(&env.kernel.ngs2.state);
    let Some(system) = s.systems.get_mut(&env.arg(0)) else {
        return ret(ERROR_INVALID_SYSTEM_HANDLE);
    };
    if (1..=8192).contains(&grain) {
        system.grain = grain as usize;
    }
    ret(0)
}

fn system_set_rate(env: &mut Env) -> Result<Outcome, CallError> {
    let rate = env.arg(1) as u32;
    let mut s = lock(&env.kernel.ngs2.state);
    let Some(system) = s.systems.get_mut(&env.arg(0)) else {
        return ret(ERROR_INVALID_SYSTEM_HANDLE);
    };
    if !(8000..=192_000).contains(&rate) {
        return ret(0x8002_0000 + crate::errno::EINVAL as u64);
    }
    system.rate = rate;
    ret(0)
}

/// `sceNgs2RackCreate[WithAllocator](system, rack_id, option, buffer_or_allocator, &handle)`.
fn rack_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (system, id, out) = (env.arg(0), env.arg(1) as u32, env.arg(4));
    let mut s = lock(&env.kernel.ngs2.state);
    if !s.systems.contains_key(&system) {
        return ret(ERROR_INVALID_SYSTEM_HANDLE);
    }
    if out == 0 {
        return ret(ERROR_INVALID_OUT_ADDRESS);
    }
    let handle = s.handle();
    s.racks.insert(handle, Rack { system, id });
    env.mem.write_u64(out, handle)?;
    ret(0)
}

fn rack_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let mut s = lock(&env.kernel.ngs2.state);
    if !s.racks.contains_key(&env.arg(0)) {
        return ret(ERROR_INVALID_RACK_HANDLE);
    }
    remove_rack(&mut s, env.arg(0));
    ret(0)
}

/// `sceNgs2RackGetVoiceHandle(rack, index, &handle)`: the same index gives
/// the same voice every time.
fn rack_voice(env: &mut Env) -> Result<Outcome, CallError> {
    let (rack, index, out) = (env.arg(0), env.arg(1) as u32, env.arg(2));
    let mut s = lock(&env.kernel.ngs2.state);
    if !s.racks.contains_key(&rack) {
        return ret(ERROR_INVALID_RACK_HANDLE);
    }
    if out == 0 {
        return ret(ERROR_INVALID_OUT_ADDRESS);
    }
    let found = s
        .voices
        .iter()
        .find(|(_, v)| v.rack == rack && v.index == index)
        .map(|(&h, _)| h);
    let handle = match found {
        Some(h) => h,
        None => {
            let h = s.handle();
            s.voices.insert(h, Voice::new(rack, index));
            h
        }
    };
    if tracing() {
        eprintln!("ngs2: rack {rack:#x} voice {index} is {handle:#x}");
    }
    env.mem.write_u64(out, handle)?;
    ret(0)
}

/// `sceNgs2VoiceGetState(voice, &state, size)`: flags, error, then the
/// decoded sample and byte counts and the current block.
fn voice_state(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, size) = (env.arg(1), env.arg(2).min(0x400) as usize);
    if out == 0 || size < 4 {
        return ret(ERROR_INVALID_OUT_ADDRESS);
    }
    let mut b = vec![0u8; size];
    {
        let s = lock(&env.kernel.ngs2.state);
        let Some(v) = s.voices.get(&env.arg(0)) else {
            return ret(ERROR_INVALID_VOICE_HANDLE);
        };
        b[0..4].copy_from_slice(&v.flags().to_le_bytes());
        if size >= 8 {
            b[4..8].copy_from_slice(&(v.error as u32).to_le_bytes());
        }
        if size >= 56 {
            let playing: f32 = if v.playing { 1.0 } else { 0.0 };
            b[8..12].copy_from_slice(&playing.to_le_bytes());
            b[24..32].copy_from_slice(&v.decoded_samples.to_le_bytes());
            b[32..40].copy_from_slice(&v.decoded_bytes.to_le_bytes());
            if let Some(block) = v.blocks.front() {
                b[40..48].copy_from_slice(&block.user_data.to_le_bytes());
                b[48..56].copy_from_slice(&block.address.to_le_bytes());
            }
        }
    }
    env.mem.write(out, &b)?;
    ret(0)
}

fn voice_state_flags(env: &mut Env) -> Result<Outcome, CallError> {
    let flags = {
        let s = lock(&env.kernel.ngs2.state);
        let Some(v) = s.voices.get(&env.arg(0)) else {
            return ret(ERROR_INVALID_VOICE_HANDLE);
        };
        v.flags()
    };
    if env.arg(1) == 0 {
        return ret(ERROR_INVALID_OUT_ADDRESS);
    }
    env.mem.write(env.arg(1), &flags.to_le_bytes())?;
    ret(0)
}

/// `sceNgs2SystemRender(system, buffers, count)`: mixes one grain into
/// each buffer, then calls the voice callbacks it owes.
fn system_render(env: &mut Env) -> Result<Outcome, CallError> {
    let (system, list, count) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    let callbacks = {
        let mut s = lock(&env.kernel.ngs2.state);
        match mix::render(env, &mut s, system, list, count)? {
            Ok(c) => c,
            Err(code) => return ret(code),
        }
    };
    if callbacks.is_empty() {
        return ret(0);
    }
    let info = {
        let mut s = lock(&env.kernel.ngs2.state);
        let Some(sys) = s.systems.get_mut(&system) else {
            return ret(0);
        };
        if sys.callback_info == 0 {
            sys.callback_info = env.kernel.heap.alloc(env.mem, 64).unwrap_or(0);
        }
        sys.callback_info
    };
    if info == 0 {
        return ret(0);
    }
    for c in callbacks {
        env.mem.write(info, &c.info)?;
        env.call_guest(c.handler, &[info])?;
    }
    ret(0)
}

/// Most of a waveform's header we read.
const HEADER_MAX: usize = 0x1000;

/// `sceNgs2ParseWaveformData(data, size, &info)`.
fn parse_waveform_data(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, size, out) = (env.arg(0), env.arg(1), env.arg(2));
    if out == 0 {
        return ret(ERROR_INVALID_PARSE_OUT_ADDRESS);
    }
    if at == 0 || size == 0 {
        return ret(ERROR_INVALID_WAVEFORM_ADDRESS);
    }
    let mut data = vec![0u8; (size as usize).min(HEADER_MAX)];
    env.mem.read(at, &mut data)?;
    write_waveform_info(env, &data, size, out)
}

/// `sceNgs2ParseWaveformFile(path, offset, &info)`.
fn parse_waveform_file(env: &mut Env) -> Result<Outcome, CallError> {
    let (offset, out) = (env.arg(1), env.arg(2));
    if out == 0 {
        return ret(ERROR_INVALID_PARSE_OUT_ADDRESS);
    }
    let path = crate::libc::cstr(env, env.arg(0))?;
    let Some(file) = env.kernel.vfs.read(&path) else {
        return ret(ERROR_INVALID_WAVEFORM_DATA);
    };
    let total = file.len() as u64;
    if offset >= total {
        return ret(ERROR_INVALID_WAVEFORM_DATA);
    }
    let start = offset as usize;
    let data = &file[start..(start + HEADER_MAX).min(file.len())];
    write_waveform_info(env, data, total - offset, out)
}

fn write_waveform_info(env: &Env, data: &[u8], total: u64, out: u64) -> Result<Outcome, CallError> {
    match waveform::parse(data, total) {
        Some(info) => {
            env.mem.write(out, &waveform::encode(&info))?;
            ret(0)
        }
        None if waveform::is_riff(data) => ret(ERROR_INVALID_WAVEFORM_FORMAT),
        None => ret(ERROR_INVALID_WAVEFORM_DATA),
    }
}
