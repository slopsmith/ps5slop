//! libSceAvPlayer placeholders: no video is decoded. A player opens, takes
//! its sources and reports through the title's event callback that the
//! source is ready, at `Start` that it plays, and at the first frame request
//! after that that it has stopped; it is never active and has no streams or
//! frames, so a title moves past its movies.
//! Names hashed from guesses after the PS4 AvPlayer API; prototypes read off
//! the calls. The init data's event callback sits at +0x58 with its object
//! at +0x50 (observed in Astro Bot's init data); the event numbers (stop 1,
//! ready 2, play 3) are shadPS4's, from memory, unverified.

use crate::{CallError, Env, Function, Outcome};

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAvPlayer",
        name,
        func,
    }
}

pub static STUBS: &[Function] = &[
    f("sceAvPlayerInit", init),
    f("sceAvPlayerPostInit", ok),
    f("sceAvPlayerAddSource", add_source),
    f("sceAvPlayerStart", start),
    f("sceAvPlayerStop", ok),
    f("sceAvPlayerPause", ok),
    f("sceAvPlayerResume", ok),
    f("sceAvPlayerClose", ok),
    f("sceAvPlayerSetLooping", ok),
    f("sceAvPlayerSetTrickSpeed", ok),
    f("sceAvPlayerSetAvSyncMode", ok),
    f("sceAvPlayerJumpToTime", ok),
    f("sceAvPlayerEnableStream", ok),
    f("sceAvPlayerGetStreamInfo", ok),
    f("sceAvPlayerIsActive", ok),
    f("sceAvPlayerGetVideoDataEx", video_data),
    f("sceAvPlayerStreamCount", ok),
];

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

const STATE_STOP: u64 = 1;
const STATE_READY: u64 = 2;
const STATE_PLAY: u64 = 3;

/// `sceAvPlayerInit(init_data)`: the handle, a small guest block holding the
/// event callback's object and function (titles only pass it back).
fn init(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(handle) = env.kernel.heap.alloc(env.mem, 64) else {
        return Ok(Outcome::Return(0));
    };
    let mut event = [0u8; 16];
    if env.arg(0) != 0 {
        env.mem.read(env.arg(0) + 0x50, &mut event)?;
    }
    env.mem.write(handle, &event)?;
    env.mem.write(handle + STOP_PENDING, &[0])?;
    Ok(Outcome::Return(handle))
}

/// Calls the title's event callback `(object, event, source, data)`, if any.
fn event(env: &mut Env, handle: u64, id: u64) -> Result<(), CallError> {
    if handle == 0 {
        return Ok(());
    }
    let object = env.mem.read_u64(handle)?;
    let callback = env.mem.read_u64(handle + 8)?;
    if callback != 0 {
        env.call_guest(callback, &[object, id, 0, 0])?;
    }
    Ok(())
}

/// `sceAvPlayerAddSource(handle, path)`: the source is ready at once.
fn add_source(env: &mut Env) -> Result<Outcome, CallError> {
    event(env, env.arg(0), STATE_READY)?;
    Ok(Outcome::Return(0))
}

/// Byte of the handle block set while a stop is still to be reported.
const STOP_PENDING: u64 = 16;

/// `sceAvPlayerStart(handle)`: plays; the stop follows at the next frame request.
fn start(env: &mut Env) -> Result<Outcome, CallError> {
    let handle = env.arg(0);
    event(env, handle, STATE_PLAY)?;
    if handle != 0 {
        env.mem.write(handle + STOP_PENDING, &[1])?;
    }
    Ok(Outcome::Return(0))
}

/// `sceAvPlayerGetVideoDataEx(handle, frame)`: never a frame (false).
fn video_data(env: &mut Env) -> Result<Outcome, CallError> {
    let handle = env.arg(0);
    if handle != 0 {
        let mut pending = [0u8];
        env.mem.read(handle + STOP_PENDING, &mut pending)?;
        if pending[0] != 0 {
            env.mem.write(handle + STOP_PENDING, &[0])?;
            event(env, handle, STATE_STOP)?;
        }
    }
    Ok(Outcome::Return(0))
}
