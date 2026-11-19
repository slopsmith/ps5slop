//! libSceNgs2, the sample-mixing audio library, without sound: systems,
//! racks and voices are handles, voices never play, and rendering fills
//! the title's buffers with silence. Prototypes follow the PS4 ones as
//! shadPS4 knows them, checked against Balatro's calls; unverified.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{CallError, Env, Function, Outcome};

pub static FUNCTIONS: &[Function] = &[
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
    env.mem.write_u64(out, NEXT.fetch_add(1, Ordering::Relaxed))?;
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
