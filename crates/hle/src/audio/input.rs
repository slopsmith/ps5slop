//! libSceAudioIn: a microphone that is always silent. `sceAudioInInput`
//! blocks for one grain's time, as a capture call does, and hands back
//! zeros. Argument positions are what Deltarune passes (`docs/audio.md`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

/// Our guess at "no such port"; unverified.
const ERROR_INVALID_HANDLE: u64 = 0x8026_0105;

pub static FUNCTIONS: &[Function] = &[
    f("sceAudioInOpen", open),
    f("sceAudioInInput", input),
    f("sceAudioInClose", close),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAudioIn",
        name,
        func,
    }
}

/// Grain (samples) and rate of each open port.
static PORTS: Mutex<Option<HashMap<u64, (u64, u64)>>> = Mutex::new(None);
static NEXT: AtomicU64 = AtomicU64::new(1);

/// `sceAudioInOpen(user, type, index, grain, rate, param)`: a handle.
fn open(env: &mut Env) -> Result<Outcome, CallError> {
    let (grain, rate) = (
        env.arg(3).clamp(1, 1 << 16),
        env.arg(4).clamp(1000, 192_000),
    );
    let handle = NEXT.fetch_add(1, Ordering::Relaxed);
    lock(&PORTS)
        .get_or_insert_with(HashMap::new)
        .insert(handle, (grain, rate));
    Ok(Outcome::Return(handle))
}

/// `sceAudioInInput(handle, dest)`: one grain of 16-bit mono silence.
fn input(env: &mut Env) -> Result<Outcome, CallError> {
    let port = lock(&PORTS)
        .as_ref()
        .and_then(|p| p.get(&env.arg(0)).copied());
    let Some((grain, rate)) = port else {
        return Ok(Outcome::Return(ERROR_INVALID_HANDLE));
    };
    std::thread::sleep(Duration::from_micros(grain * 1_000_000 / rate));
    env.mem.write(env.arg(1), &vec![0u8; grain as usize * 2])?;
    Ok(Outcome::Return(grain))
}

fn close(env: &mut Env) -> Result<Outcome, CallError> {
    let removed = lock(&PORTS)
        .as_mut()
        .and_then(|p| p.remove(&env.arg(0)))
        .is_some();
    Ok(Outcome::Return(if removed {
        0
    } else {
        ERROR_INVALID_HANDLE
    }))
}

pub(crate) fn save(e: &mut savestate::Encoder) {
    e.put(&*lock(&PORTS)).put(&NEXT.load(Ordering::Relaxed));
}

pub(crate) fn load(d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
    *lock(&PORTS) = d.get()?;
    NEXT.store(d.get()?, Ordering::Relaxed);
    Ok(())
}
