//! System services a title calls at start-up: the user service (one
//! signed-in user), the system service, audio out, the keyboard and the
//! IME dialog. Constants follow the OpenOrbis headers.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

/// The one user (our choice of id).
pub const USER_ID: i32 = 1;
const USER_NAME: &[u8] = b"ps5slop";
const USER_ID_INVALID: i32 = -1;
/// `SceUserServiceLoginUserIdList` holds four ids.
const LOGIN_LIST_LEN: usize = 4;

/// Our guess; unverified.
const SYSTEM_SERVICE_ERROR_PARAMETER: u64 = 0x80A1_0003;

const AUDIO_ERROR_INVALID_PORT: u64 = 0x8026_0003;
const AUDIO_ERROR_INVALID_FORMAT: u64 = 0x8026_0007;
const AUDIO_ERROR_INVALID_SAMPLE_FREQ: u64 = 0x8026_0008;

pub static FUNCTIONS: &[Function] = &[
    f("libSceUserService", "sceUserServiceInitialize", ok),
    f(
        "libSceUserService",
        "sceUserServiceGetForegroundUser",
        foreground_user,
    ),
    f(
        "libSceUserService",
        "sceUserServiceGetLoginUserIdList",
        login_users,
    ),
    f("libSceUserService", "sceUserServiceGetUserName", user_name),
    f(
        "libSceUserService",
        "sceUserServiceGetInitialUser",
        foreground_user,
    ),
    f(
        "libSceSystemService",
        "sceSystemServiceHideSplashScreen",
        ok,
    ),
    f(
        "libSceSystemService",
        "sceSystemServiceLaunchWebBrowser",
        launch_web_browser,
    ),
    f(
        "libSceSystemService",
        "sceSystemServiceParamGetInt",
        param_get_int,
    ),
    f("libSceAudioOut", "sceAudioOutInit", ok),
    f("libSceAudioOut", "sceAudioOutOpen", audio_open),
    f("libSceAudioOut", "sceAudioOutOutput", audio_output),
    f("libSceAudioOut", "sceAudioOutClose", audio_close),
];

const fn f(library: &'static str, name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library,
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

fn foreground_user(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(0), &USER_ID.to_le_bytes())?;
    ret(0)
}

fn login_users(env: &mut Env) -> Result<Outcome, CallError> {
    let mut list = [USER_ID_INVALID; LOGIN_LIST_LEN];
    list[0] = USER_ID;
    let bytes: Vec<u8> = list.iter().flat_map(|id| id.to_le_bytes()).collect();
    env.mem.write(env.arg(0), &bytes)?;
    ret(0)
}

/// `sceUserServiceGetUserName(user, name, size)`; a short buffer gets a
/// truncated name.
fn user_name(env: &mut Env) -> Result<Outcome, CallError> {
    let (buf, size) = (env.arg(1), env.arg(2) as usize);
    if size == 0 {
        return ret(0);
    }
    let n = USER_NAME.len().min(size - 1);
    env.mem.write(buf, &USER_NAME[..n])?;
    env.mem.write(buf + n as u64, &[0])?;
    ret(0)
}

/// `sceSystemServiceParamGetInt(id, &value)`: English, ISO dates, 24-hour
/// time, UTC, no daylight saving, no parental limit, cross confirms.
fn param_get_int(env: &mut Env) -> Result<Outcome, CallError> {
    let value: i32 = match env.arg(0) as u32 {
        1 => 1,
        2 => 0,
        3 => 1,
        4 => 0,
        5 => 0,
        7 => 0,
        1000 => 1,
        _ => return ret(SYSTEM_SERVICE_ERROR_PARAMETER),
    };
    env.mem.write(env.arg(1), &value.to_le_bytes())?;
    ret(0)
}

/// `sceSystemServiceLaunchWebBrowser(uri, param)`: prints the address.
fn launch_web_browser(env: &mut Env) -> Result<Outcome, CallError> {
    let mut line = b"[web browser] ".to_vec();
    line.extend(env.mem.read_cstr(env.arg(0), 4096)?);
    line.push(b'\n');
    let _ = env.kernel.write_output(2, &line);
    ret(0)
}

struct Port {
    /// How long one `sceAudioOutOutput` buffer plays for.
    period: Duration,
    next: Instant,
}

/// Audio ports. Samples are dropped; `sceAudioOutOutput` only keeps time,
/// so audio threads run at the pace they would on a console.
#[derive(Default)]
pub struct Audio {
    next: Mutex<u64>,
    ports: Mutex<HashMap<u64, Port>>,
}

/// `sceAudioOutOpen(user, type, index, samples, freq, format)`; formats 0
/// to 7 are the mono, stereo and 8-channel S16 and float formats.
fn audio_open(env: &mut Env) -> Result<Outcome, CallError> {
    let (samples, freq, format) = (env.arg(3) as u32, env.arg(4) as u32, env.arg(5) as u32);
    if freq == 0 {
        return ret(AUDIO_ERROR_INVALID_SAMPLE_FREQ);
    }
    if format & 0xff > 7 || samples == 0 {
        return ret(AUDIO_ERROR_INVALID_FORMAT);
    }
    let audio = &env.kernel.audio;
    let id = {
        let mut next = lock(&audio.next);
        *next += 1;
        *next
    };
    let period = Duration::from_secs_f64(samples as f64 / freq as f64);
    lock(&audio.ports).insert(
        id,
        Port {
            period,
            next: Instant::now(),
        },
    );
    ret(id)
}

/// `sceAudioOutOutput(port, data)`: waits until the previous buffer would
/// have finished playing.
fn audio_output(env: &mut Env) -> Result<Outcome, CallError> {
    let wake = {
        let mut ports = lock(&env.kernel.audio.ports);
        let Some(p) = ports.get_mut(&env.arg(0)) else {
            return ret(AUDIO_ERROR_INVALID_PORT);
        };
        let wake = p.next;
        p.next = wake.max(Instant::now()) + p.period;
        wake
    };
    if let Some(left) = wake.checked_duration_since(Instant::now()) {
        std::thread::sleep(left);
    }
    ret(0)
}

fn audio_close(env: &mut Env) -> Result<Outcome, CallError> {
    match lock(&env.kernel.audio.ports).remove(&env.arg(0)) {
        Some(_) => ret(0),
        None => ret(AUDIO_ERROR_INVALID_PORT),
    }
}
