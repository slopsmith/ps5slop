//! System services a title calls at start-up: the user service (one
//! signed-in user), the system service, the keyboard and the IME
//! dialog. Constants follow the OpenOrbis headers.

use crate::{CallError, Env, Function, Outcome};

/// The one user (our choice of id).
pub const USER_ID: i32 = 1;
const USER_NAME: &[u8] = b"ps5slop";
const USER_ID_INVALID: i32 = -1;
/// `SceUserServiceLoginUserIdList` holds four ids.
const LOGIN_LIST_LEN: usize = 4;

/// Our guess; unverified.
const SYSTEM_SERVICE_ERROR_PARAMETER: u64 = 0x80A1_0003;

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
    f("libSceUserService", "sceUserServiceGetEvent", user_event),
    f("libSceUserService", "sceUserServiceTerminate", ok),
    f("libSceRtc", "sceRtcGetCurrentTick", rtc_current_tick),
    f(
        "libSceSystemService",
        "sceSystemServiceGetDisplaySafeAreaInfo",
        safe_area,
    ),
    f("libSceRtc", "sceRtcGetCurrentNetworkTick", rtc_current_tick),
    f("libSceRtc", "sceRtcSetTick", rtc_set_tick),
];

/// `sceSystemServiceGetDisplaySafeAreaInfo(info)`: the whole screen is safe.
/// The first field is taken to be the ratio as an `f32` (unverified).
fn safe_area(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(0), &1.0f32.to_le_bytes())?;
    Ok(Outcome::Return(0))
}

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f(
        "libSceSystemService",
        "sceSystemServiceGetStatus",
        system_status,
    ),
    f(
        "libSceSystemService",
        "sceSystemServiceReceiveEvent",
        no_system_event,
    ),
    f("libSceUserService", "sceUserServiceGetAgeLevel", setting_zero),
    f(
        "libSceUserService",
        "sceUserServiceGetAccessibilityVibration",
        setting_zero,
    ),
    f(
        "libSceUserService",
        "sceUserServiceGetAccessibilityChatTranscription",
        setting_zero,
    ),
    f(
        "libSceUserService",
        "sceUserServiceGetAccessibilityZoomEnabled",
        setting_zero,
    ),
    f(
        "libSceUserService",
        "sceUserServiceGetAccessibilityZoomFollowFocus",
        setting_zero,
    ),
    f(
        "libSceUserService",
        "sceUserServiceGetAccessibilityTriggerEffect",
        setting_zero,
    ),
    f(
        "libSceUserService",
        "sceUserServiceGetAccessibilityPressAndHoldDelay",
        setting_zero,
    ),
    f("libSceUserService", "sceUserServiceGetGamePresets", ok),
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

/// A per-user setting getter `(user_id, *value)`: writes 0 as a 32-bit
/// value (the types and the meaning of 0 are guesses).
fn setting_zero(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(1) != 0 {
        env.mem.write(env.arg(1), &0i32.to_le_bytes())?;
    }
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

/// From OpenOrbis.
const USER_SERVICE_ERROR_NO_EVENT: u64 = 0x8096_0007;

/// `sceUserServiceGetEvent(event)`: [`USER_ID`] logging in on the first call
/// (`{ i32 type = 0, i32 user }`, from OpenOrbis), then no events.
fn user_event(env: &mut Env) -> Result<Outcome, CallError> {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if REPORTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return ret(USER_SERVICE_ERROR_NO_EVENT);
    }
    env.mem.write(env.arg(0), &0i32.to_le_bytes())?;
    env.mem.write(env.arg(0) + 4, &USER_ID.to_le_bytes())?;
    ret(0)
}

/// `sceSystemServiceGetStatus(status)`: no events waiting, in the
/// foreground; only the event count and five flags are written.
fn system_status(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(0), &[0u8; 9])?;
    ret(0)
}

/// `sceSystemServiceReceiveEvent(event)`: there are none (the status
/// reports zero); the error code is our guess.
fn no_system_event(_: &mut Env) -> Result<Outcome, CallError> {
    ret(SYSTEM_SERVICE_ERROR_PARAMETER)
}

/// Seconds from 0001-01-01 to the Unix epoch.
const RTC_EPOCH_OFFSET: u64 = 62_135_596_800;

/// `sceRtcSetTick(&date_time, &tick)`: the tick as a date in shadPS4's
/// layout: year, month, day, hour, minute and second as `u16`, then the
/// microseconds as `u32`.
fn rtc_set_tick(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, tick_at) = (env.arg(0), env.arg(1));
    if out == 0 || tick_at == 0 {
        return ret(crate::dmem::sce_error(crate::errno::EINVAL));
    }
    let tick = env.mem.read_u64(tick_at)?;
    let (secs, micros) = (tick / 1_000_000, (tick % 1_000_000) as u32);
    let (year, month, day) = civil_from_days((secs / 86_400) as i64 - 719_162);
    let in_day = secs % 86_400;
    let mut b = [0u8; 16];
    for (k, v) in [year as u64, month, day, in_day / 3600, in_day / 60 % 60, in_day % 60]
        .into_iter()
        .enumerate()
    {
        b[2 * k..2 * k + 2].copy_from_slice(&(v as u16).to_le_bytes());
    }
    b[12..16].copy_from_slice(&micros.to_le_bytes());
    env.mem.write(out, &b)?;
    ret(0)
}

/// Year, month and day of the day `days` after 1970-01-01 (proleptic
/// Gregorian calendar).
fn civil_from_days(days: i64) -> (i64, u64, u64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u64;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u64;
    let year = yoe + era * 400 + (month <= 2) as i64;
    (year, month, day)
}

/// `sceRtcGetCurrentTick(&tick)`: microseconds since 0001-01-01 UTC. The
/// network tick (`sceRtcGetCurrentNetworkTick`) is the same clock here.
fn rtc_current_tick(env: &mut Env) -> Result<Outcome, CallError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let tick = (now.as_secs() + RTC_EPOCH_OFFSET) * 1_000_000 + now.subsec_micros() as u64;
    env.mem.write_u64(env.arg(0), tick)?;
    ret(0)
}

#[cfg(test)]
mod tests {
    use super::civil_from_days;

    #[test]
    fn days_map_to_gregorian_dates_across_leap_days_and_the_epoch() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(-719_162), (1, 1, 1));
    }
}
