//! libSceAvPlayer without a decoder: a player opens, reports one 1920x1080
//! video stream, tells the title's event callback that its source is ready
//! and, at `Start`, that it plays; it then hands out black NV12 frames for
//! 3 s and reports that it has stopped. That is enough for Astro Bot to play
//! its logo movie through. Names hashed from guesses after the PS4 AvPlayer
//! API; prototypes read off the calls. The init data, stream info and frame
//! info layouts are what Astro Bot reads (its own code, looked at locally;
//! see docs/hle-kernel.md); the event numbers (stop 1, ready 2, play 3) are
//! shadPS4's, from memory, and agree with what the title checks for "ready".

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
    f("sceAvPlayerGetStreamInfo", stream_info),
    f("sceAvPlayerIsActive", ok),
    f("sceAvPlayerGetVideoDataEx", video_data),
    f("sceAvPlayerStreamCount", one),
];

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

fn one(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(1))
}

/// `sceAvPlayerGetStreamInfo(handle, index, info)`: one video stream (type 1
/// at 0, width and height at 8, duration in ms at 0x18, as Astro Bot reads it).
fn stream_info(env: &mut Env) -> Result<Outcome, CallError> {
    let info = env.arg(2);
    if info == 0 {
        return Ok(Outcome::Return(ERROR_INVALID_PARAMS));
    }
    let mut d = [0u8; 0x20];
    d[0..4].copy_from_slice(&1u32.to_le_bytes());
    d[8..12].copy_from_slice(&WIDTH.to_le_bytes());
    d[12..16].copy_from_slice(&HEIGHT.to_le_bytes());
    d[0x18..0x20].copy_from_slice(&(FRAME_COUNT as u64 * 1000 / 30).to_le_bytes());
    env.mem.write(info, &d)?;
    Ok(Outcome::Return(0))
}

/// Returned for a null output pointer (our choice of code, unverified).
const ERROR_INVALID_PARAMS: u64 = 0x806A_0001;
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
    let mut allocator = [0u8; 16];
    if env.arg(0) != 0 {
        let object = env.mem.read_u64(env.arg(0))?;
        let allocate_texture = env.mem.read_u64(env.arg(0) + 0x18)?;
        allocator[..8].copy_from_slice(&object.to_le_bytes());
        allocator[8..].copy_from_slice(&allocate_texture.to_le_bytes());
    }
    env.mem.write(handle + PLAYING, &[0u8; 24])?;
    env.mem.write(handle + ALLOCATOR, &allocator)?;
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

/// The handle block: event object and callback (0, 8), playing flag (16),
/// frames handed out (24), frame buffer (32), texture allocator object and
/// function from the init data's memory replacement (40, 48).
const PLAYING: u64 = 16;
const FRAMES: u64 = 24;
const BUFFER: u64 = 32;
const ALLOCATOR: u64 = 40;

/// Black frames handed out after `Start` before the stop: 3 s at 30 a second.
const FRAME_COUNT: u32 = 90;
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

/// `sceAvPlayerStart(handle)`: plays black frames, then stops.
fn start(env: &mut Env) -> Result<Outcome, CallError> {
    let handle = env.arg(0);
    event(env, handle, STATE_PLAY)?;
    if handle != 0 {
        env.mem.write(handle + PLAYING, &[1])?;
        env.mem.write(handle + FRAMES, &0u32.to_le_bytes())?;
    }
    Ok(Outcome::Return(0))
}

/// An NV12 frame buffer from the title's texture allocator
/// `(object, alignment, size)`, filled with black.
fn frame_buffer(env: &mut Env, handle: u64) -> Result<u64, CallError> {
    let have = env.mem.read_u64(handle + BUFFER)?;
    if have != 0 {
        return Ok(have);
    }
    let object = env.mem.read_u64(handle + ALLOCATOR)?;
    let allocate = env.mem.read_u64(handle + ALLOCATOR + 8)?;
    let size = (WIDTH * HEIGHT * 3 / 2) as usize;
    let buffer = if allocate != 0 {
        env.call_guest(allocate, &[object, 0x10000, size as u64])?
    } else {
        env.kernel.heap.alloc(env.mem, size as u64).unwrap_or(0)
    };
    if buffer != 0 {
        let mut black = vec![16u8; size];
        black[(WIDTH * HEIGHT) as usize..].fill(128);
        env.mem.write(buffer, &black)?;
        env.mem.write(handle + BUFFER, &buffer.to_le_bytes())?;
    }
    Ok(buffer)
}

/// `sceAvPlayerGetVideoDataEx(handle, frame)`: while playing, a black frame
/// (true); after the last one the stop event, and no frame (false). Frame
/// info as Astro Bot reads it: data at 0, time stamp in ms at 0x10, width,
/// height and aspect ratio from 0x18, crop offsets from 0x2c, pitch at 0x3c.
fn video_data(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, info) = (env.arg(0), env.arg(1));
    if handle == 0 {
        return Ok(Outcome::Return(0));
    }
    let mut playing = [0u8];
    env.mem.read(handle + PLAYING, &mut playing)?;
    if playing[0] == 0 {
        return Ok(Outcome::Return(0));
    }
    let mut n = [0u8; 4];
    env.mem.read(handle + FRAMES, &mut n)?;
    let n = u32::from_le_bytes(n);
    if n >= FRAME_COUNT {
        env.mem.write(handle + PLAYING, &[0])?;
        event(env, handle, STATE_STOP)?;
        return Ok(Outcome::Return(0));
    }
    let buffer = frame_buffer(env, handle)?;
    if buffer == 0 || info == 0 {
        return Ok(Outcome::Return(0));
    }
    let mut frame = [0u8; 0x48];
    frame[0..8].copy_from_slice(&buffer.to_le_bytes());
    frame[0x10..0x18].copy_from_slice(&(n as u64 * 1000 / 30).to_le_bytes());
    frame[0x18..0x1c].copy_from_slice(&WIDTH.to_le_bytes());
    frame[0x1c..0x20].copy_from_slice(&HEIGHT.to_le_bytes());
    frame[0x20..0x24].copy_from_slice(&(WIDTH as f32 / HEIGHT as f32).to_le_bytes());
    frame[0x3c..0x40].copy_from_slice(&WIDTH.to_le_bytes());
    frame[0x40] = 8;
    frame[0x41] = 8;
    env.mem.write(info, &frame)?;
    env.mem.write(handle + FRAMES, &(n + 1).to_le_bytes())?;
    Ok(Outcome::Return(1))
}
