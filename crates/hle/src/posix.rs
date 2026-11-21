//! libkernel's smaller POSIX functions: files by path, time, signals (which
//! we never deliver), scheduling, `sysconf`, and the dynamic linker
//! interface (which loads nothing yet). Constants are FreeBSD's.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::dmem::sce_error;
use crate::libc::FdInfo;
use crate::pthread::{POLL, ret as ret_errno};
use crate::vfs::Meta;
use crate::{CallError, Env, Function, Outcome, errno};

pub static FUNCTIONS: &[Function] = &[
    f("stat", stat),
    f("fstat", fstat),
    f("sceKernelStat", sce_stat),
    f("sceKernelFstat", sce_fstat),
    f("sceKernelCheckReachability", check_reachability),
    f("mkdir", mkdir),
    f("gettimeofday", gettimeofday),
    f("clock_gettime", clock_gettime),
    f("clock_getres", clock_getres),
    f("sceKernelClockGettime", sce_clock_gettime),
    f("nanosleep", nanosleep),
    f("sceKernelUsleep", usleep),
    f("sceKernelGetProcessTime", process_time),
    f("sceKernelGetProcessTimeCounter", process_time_counter),
    f(
        "sceKernelGetProcessTimeCounterFrequency",
        process_time_frequency,
    ),
    f("sceKernelReadTsc", read_tsc),
    f("sceKernelGetTscFrequency", process_time_frequency),
    f("sceKernelGettimeofday", gettimeofday),
    f("sceKernelNanosleep", sce_nanosleep),
    f("_nanosleep", nanosleep),
    f("usleep", posix_usleep),
    f("sched_yield", sched_yield),
    f("getpid", getpid),
    f("getrusage", getrusage),
    f("_exit", crate::kernel::exit),
    f("_sigprocmask", pthread_sigmask),
    f("sceKernelConvertUtcToLocaltime", utc_to_localtime),
    f("sceKernelConvertLocaltimeToUtc", localtime_to_utc),
    f("sceKernelDebugRaiseException", raise_exception),
    f("sceKernelDebugRaiseExceptionOnReleaseMode", raise_exception),
    f("sysconf", sysconf),
    f("sysctl", sysctl),
    f("sysctlbyname", sysctl),
    f("sigaction", sigaction),
    f("signal", signal),
    f("sigemptyset", sigemptyset),
    f("sigaddset", sigaddset),
    f("pthread_sigmask", pthread_sigmask),
    f("pthread_setcanceltype", pthread_setcanceltype),
    f("pthread_getschedparam", pthread_getschedparam),
    f("pthread_setschedparam", pthread_setschedparam),
    f("scePthreadGetschedparam", pthread_getschedparam),
    f("scePthreadSetschedparam", pthread_setschedparam),
    f("sched_get_priority_min", sched_get_priority_min),
    f("sched_get_priority_max", sched_get_priority_max),
    f("dlopen", dlopen),
    f("dlsym", dlsym),
    f("dlclose", dlclose),
    f("dlerror", dlerror),
    f("sceKernelSendNotificationRequest", send_notification),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f("socket", socket),
    f("__pthread_cxa_finalize", accept),
    f("sceKernelMlock", accept),
    f("sceKernelIsTrinityMode", accept),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

const S_IFCHR: u16 = 0o020000;
const S_IFDIR: u16 = 0o040000;
const S_IFREG: u16 = 0o100000;
/// The size of `struct stat` in the pre-FreeBSD-12 layout shadPS4 uses for
/// the PS4. Unverified for the PS5.
const STAT_SIZE: usize = 120;

/// Writes a `struct stat` for a file of `mode` and `size`.
fn write_stat(
    env: &Env,
    at: u64,
    mode: u16,
    size: u64,
    mtime: Option<SystemTime>,
) -> Result<(), CallError> {
    let mut sb = [0u8; STAT_SIZE];
    sb[8..10].copy_from_slice(&mode.to_le_bytes());
    sb[10..12].copy_from_slice(&1u16.to_le_bytes());
    let t = mtime
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    for off in [24, 40, 56, 104] {
        sb[off..off + 8].copy_from_slice(&t.as_secs().to_le_bytes());
        sb[off + 8..off + 16].copy_from_slice(&(t.subsec_nanos() as u64).to_le_bytes());
    }
    sb[72..80].copy_from_slice(&size.to_le_bytes());
    sb[80..88].copy_from_slice(&size.div_ceil(512).to_le_bytes());
    sb[88..92].copy_from_slice(&0x4000u32.to_le_bytes());
    env.mem.write(at, &sb)?;
    Ok(())
}

fn write_metadata(env: &Env, at: u64, m: &Meta) -> Result<(), CallError> {
    let mode = if m.dir {
        S_IFDIR | 0o555
    } else {
        S_IFREG | 0o444
    };
    write_stat(env, at, mode, m.len, m.modified)
}

fn stat_path(env: &Env) -> Result<Result<(), i32>, CallError> {
    let path = super::libc::cstr(env, env.arg(0))?;
    match env.kernel.vfs.metadata(&path) {
        Some(m) => write_metadata(env, env.arg(1), &m).map(Ok),
        None => Ok(Err(errno::ENOENT)),
    }
}

fn stat_fd(env: &Env) -> Result<Result<(), i32>, CallError> {
    let (fd, at) = (env.arg(0) as i32, env.arg(1));
    match env.kernel.libc.files.fd_info(fd) {
        Some(FdInfo::Console) => write_stat(env, at, S_IFCHR | 0o666, 0, None)?,
        Some(FdInfo::File(m)) => write_metadata(env, at, &m)?,
        None => return Ok(Err(errno::EBADF)),
    }
    Ok(Ok(()))
}

/// `stat(path, sb)`.
fn stat(env: &mut Env) -> Result<Outcome, CallError> {
    match stat_path(env)? {
        Ok(()) => ret(0),
        Err(e) => env.fail(e),
    }
}

fn fstat(env: &mut Env) -> Result<Outcome, CallError> {
    match stat_fd(env)? {
        Ok(()) => ret(0),
        Err(e) => env.fail(e),
    }
}

fn sce_stat(env: &mut Env) -> Result<Outcome, CallError> {
    ret(stat_path(env)?.map_or_else(sce_error, |()| 0))
}

fn sce_fstat(env: &mut Env) -> Result<Outcome, CallError> {
    ret(stat_fd(env)?.map_or_else(sce_error, |()| 0))
}

/// `sceKernelCheckReachability(path)`: whether the path lies in a mount
/// (name hashed from a guess; GTA V asks about `/download0/saves`). What a
/// console checks beyond that is unknown.
fn check_reachability(env: &mut Env) -> Result<Outcome, CallError> {
    let path = super::libc::cstr(env, env.arg(0))?;
    ret(match env.kernel.vfs.lookup(&path) {
        Some(_) => 0,
        None => sce_error(errno::ENOENT),
    })
}

/// `mkdir(path, mode)`.
fn mkdir(env: &mut Env) -> Result<Outcome, CallError> {
    let path = super::libc::cstr(env, env.arg(0))?;
    match env.kernel.vfs.mkdir(&path) {
        Ok(()) => ret(0),
        Err(e) => env.fail(e),
    }
}

/// `gettimeofday(tv, tz)`, with `struct timeval { i64 sec; i64 usec; }`.
fn gettimeofday(env: &mut Env) -> Result<Outcome, CallError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    if env.arg(0) != 0 {
        env.mem.write_u64(env.arg(0), now.as_secs())?;
        env.mem
            .write_u64(env.arg(0) + 8, now.subsec_micros() as u64)?;
    }
    ret(0)
}

/// `nanosleep(req, rem)`, waking early only when the process exits.
fn nanosleep(env: &mut Env) -> Result<Outcome, CallError> {
    let req = env.arg(0);
    let (sec, nsec) = (env.mem.read_u64(req)?, env.mem.read_u64(req + 8)?);
    if nsec >= 1_000_000_000 {
        return env.fail(errno::EINVAL);
    }
    sleep(env, Duration::new(sec, nsec as u32));
    ret(0)
}

/// Sleeps for `time`, waking early when the process exits or an exception
/// is pending.
fn sleep(env: &Env, time: Duration) {
    let end = Instant::now() + time;
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        if env.kernel.has_exited() || crate::exception::interrupted(env) {
            break;
        }
        std::thread::sleep(left.min(POLL));
    }
}

/// `sceKernelUsleep(microseconds)`.
fn usleep(env: &mut Env) -> Result<Outcome, CallError> {
    sleep(env, Duration::from_micros(env.arg(0) as u32 as u64));
    ret(0)
}

/// The time on FreeBSD clock `id`; the non-realtime ones count from start.
fn clock(env: &Env, id: u64) -> Option<Duration> {
    match id {
        0 | 9 | 10 | 13 => Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default(),
        ),
        1 | 2 | 4 | 5 | 7 | 8 | 11 | 12 | 14 | 15 => Some(env.kernel.started.elapsed()),
        _ => None,
    }
}

/// Writes a `struct timespec { i64 sec; i64 nsec; }`.
fn write_timespec(env: &Env, at: u64, t: Duration) -> Result<(), CallError> {
    env.mem.write_u64(at, t.as_secs())?;
    env.mem.write_u64(at + 8, t.subsec_nanos() as u64)?;
    Ok(())
}

/// `clock_gettime(id, tp)`.
fn clock_gettime(env: &mut Env) -> Result<Outcome, CallError> {
    match clock(env, env.arg(0)) {
        Some(t) => {
            write_timespec(env, env.arg(1), t)?;
            ret(0)
        }
        None => env.fail(errno::EINVAL),
    }
}

fn sce_clock_gettime(env: &mut Env) -> Result<Outcome, CallError> {
    match clock(env, env.arg(0)) {
        Some(t) => {
            write_timespec(env, env.arg(1), t)?;
            ret(0)
        }
        None => ret(sce_error(errno::EINVAL)),
    }
}

/// `clock_getres(id, tp)`: one microsecond (our choice).
fn clock_getres(env: &mut Env) -> Result<Outcome, CallError> {
    if clock(env, env.arg(0)).is_none() {
        return env.fail(errno::EINVAL);
    }
    if env.arg(1) != 0 {
        write_timespec(env, env.arg(1), Duration::from_micros(1))?;
    }
    ret(0)
}

/// `sceKernelGetProcessTime()`: microseconds since the process started.
fn process_time(env: &mut Env) -> Result<Outcome, CallError> {
    ret(env.kernel.started.elapsed().as_micros() as u64)
}

const SC_CLK_TCK: u64 = 3;
const SC_OPEN_MAX: u64 = 5;
const SC_PAGESIZE: u64 = 47;
const SC_NPROCESSORS_CONF: u64 = 57;
const SC_NPROCESSORS_ONLN: u64 = 58;
/// CPU cores we report (our choice; the PS5 has 8, of which games get fewer).
const CPUS: u64 = 8;

fn sysconf(env: &mut Env) -> Result<Outcome, CallError> {
    match env.arg(0) {
        SC_CLK_TCK => ret(128),
        SC_OPEN_MAX => ret(1024),
        SC_PAGESIZE => ret(memory::PAGE_SIZE),
        SC_NPROCESSORS_CONF | SC_NPROCESSORS_ONLN => ret(CPUS),
        _ => env.fail(errno::EINVAL),
    }
}

/// `sysctl` and `sysctlbyname`: no names are known yet.
fn sysctl(env: &mut Env) -> Result<Outcome, CallError> {
    env.fail(errno::ENOENT)
}

/// `struct sigaction`: handler, flags and a 128-bit mask, padded.
const SIGACTION_SIZE: usize = 32;
const SIGSET_SIZE: usize = 16;
const MAX_SIGNAL: u64 = 128;

/// `sigaction(sig, act, oact)`: accepted and never delivered; the old
/// action reads as the default.
fn sigaction(env: &mut Env) -> Result<Outcome, CallError> {
    let (sig, oact) = (env.arg(0) as i32, env.arg(2));
    if sig <= 0 || sig as u64 > MAX_SIGNAL {
        return env.fail(errno::EINVAL);
    }
    if oact != 0 {
        env.mem.write(oact, &[0; SIGACTION_SIZE])?;
    }
    ret(0)
}

/// `signal(sig, handler)`: returns `SIG_DFL`.
fn signal(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn sigemptyset(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(0), &[0; SIGSET_SIZE])?;
    ret(0)
}

fn sigaddset(env: &mut Env) -> Result<Outcome, CallError> {
    let (set, sig) = (env.arg(0), env.arg(1) as i32 as i64);
    if sig <= 0 || sig as u64 > MAX_SIGNAL {
        return env.fail(errno::EINVAL);
    }
    let bit = sig as u64 - 1;
    let word = set + bit / 32 * 4;
    let mut b = [0u8; 4];
    env.mem.read(word, &mut b)?;
    let v = u32::from_le_bytes(b) | 1 << (bit % 32);
    env.mem.write(word, &v.to_le_bytes())?;
    ret(0)
}

/// `pthread_sigmask(how, set, oset)`: the mask is always empty.
fn pthread_sigmask(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(2) != 0 {
        env.mem.write(env.arg(2), &[0; SIGSET_SIZE])?;
    }
    ret_errno(0)
}

/// `pthread_setcanceltype(type, oldtype)`: threads are never cancelled.
fn pthread_setcanceltype(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(1) != 0 {
        env.mem.write(env.arg(1), &0u32.to_le_bytes())?;
    }
    ret_errno(0)
}

const SCHED_OTHER: u32 = 2;
/// Orbis thread priorities: lower is more urgent. Range and default from
/// shadPS4 (PS4); unverified for the PS5.
const PRIORITY_HIGHEST: u32 = 256;
const PRIORITY_LOWEST: u32 = 767;
const PRIORITY_DEFAULT: u32 = 700;

/// `pthread_getschedparam(thread, policy, param)`.
fn pthread_getschedparam(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), &SCHED_OTHER.to_le_bytes())?;
    env.mem.write(env.arg(2), &PRIORITY_DEFAULT.to_le_bytes())?;
    ret_errno(0)
}

/// `pthread_setschedparam`: accepted; host threads keep their priority.
fn pthread_setschedparam(_: &mut Env) -> Result<Outcome, CallError> {
    ret_errno(0)
}

fn sched_get_priority_min(_: &mut Env) -> Result<Outcome, CallError> {
    ret(PRIORITY_HIGHEST as u64)
}

fn sched_get_priority_max(_: &mut Env) -> Result<Outcome, CallError> {
    ret(PRIORITY_LOWEST as u64)
}

/// `dlopen(path, mode)`: loading modules is not supported yet.
fn dlopen(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn dlsym(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn dlclose(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn dlerror(env: &mut Env) -> Result<Outcome, CallError> {
    ret(super::libc::heap_cstr(
        env,
        b"ps5slop cannot load modules yet",
    )?)
}

/// Where the text sits in a notification request (OpenOrbis's
/// `OrbisNotificationRequest`), and its size.
const NOTIFICATION_MESSAGE: u64 = 45;
const NOTIFICATION_MESSAGE_LEN: usize = 1024;

/// `sceKernelSendNotificationRequest(device, request, size, blocking)`:
/// prints the message on stderr.
fn send_notification(env: &mut Env) -> Result<Outcome, CallError> {
    let mut text = env
        .mem
        .read_cstr(env.arg(1) + NOTIFICATION_MESSAGE, NOTIFICATION_MESSAGE_LEN)?;
    text.splice(0..0, b"[notification] ".iter().copied());
    text.push(b'\n');
    if env.kernel.write_output(2, &text).is_err() {
        return ret(sce_error(errno::EINVAL));
    }
    ret(0)
}

/// Ticks per second of `sceKernelGetProcessTimeCounter` (our choice).
const COUNTER_FREQUENCY: u64 = 1_000_000_000;

fn process_time_counter(env: &mut Env) -> Result<Outcome, CallError> {
    ret(env.kernel.started.elapsed().as_nanos() as u64)
}

/// `sceKernelReadTsc()`: what `rdtsc` reads, nanoseconds since the Unix
/// epoch, so its frequency is [`COUNTER_FREQUENCY`] too.
fn read_tsc(_: &mut Env) -> Result<Outcome, CallError> {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    ret(t)
}

fn process_time_frequency(_: &mut Env) -> Result<Outcome, CallError> {
    ret(COUNTER_FREQUENCY)
}

fn sce_nanosleep(env: &mut Env) -> Result<Outcome, CallError> {
    let req = env.arg(0);
    let (sec, nsec) = (env.mem.read_u64(req)?, env.mem.read_u64(req + 8)?);
    if nsec >= 1_000_000_000 {
        return ret(sce_error(errno::EINVAL));
    }
    sleep(env, Duration::new(sec, nsec as u32));
    ret(0)
}

fn posix_usleep(env: &mut Env) -> Result<Outcome, CallError> {
    sleep(env, Duration::from_micros(env.arg(0) as u32 as u64));
    ret(0)
}

fn sched_yield(_: &mut Env) -> Result<Outcome, CallError> {
    std::thread::yield_now();
    ret(0)
}

/// The process id we report (our choice).
const PID: u64 = 100;

fn getpid(_: &mut Env) -> Result<Outcome, CallError> {
    ret(PID)
}

/// `getrusage(who, usage)`: all zeros (`struct rusage` is 144 bytes on amd64).
fn getrusage(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), &[0u8; 144])?;
    ret(0)
}

/// `socket(domain, type, protocol)`: there is no network, so no address
/// family is supported.
fn socket(env: &mut Env) -> Result<Outcome, CallError> {
    env.fail(errno::EAFNOSUPPORT)
}

/// Calls with nothing to do. `sceKernelIsTrinityMode` (is this a PS5
/// Pro?) answers no.
fn accept(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// `sceKernelConvertUtcToLocaltime(utc, &local, &timesec, &dst)`: the
/// console's time zone is UTC without daylight saving.
fn utc_to_localtime(env: &mut Env) -> Result<Outcome, CallError> {
    let utc = env.arg(0);
    if env.arg(1) != 0 {
        env.mem.write_u64(env.arg(1), utc)?;
    }
    if env.arg(2) != 0 {
        env.mem.write_u64(env.arg(2), utc)?;
        env.mem.write_u64(env.arg(2) + 8, 0)?;
    }
    if env.arg(3) != 0 {
        env.mem.write_u64(env.arg(3), 0)?;
    }
    ret(0)
}

/// `sceKernelConvertLocaltimeToUtc(local, dst, &utc, &timezone, &dst_sec)`
/// (shadPS4's PS4 prototype): in UTC the time stays as it is, with a zero
/// `struct timezone` and no daylight saving.
fn localtime_to_utc(env: &mut Env) -> Result<Outcome, CallError> {
    let (local, utc, tz, dst) = (env.arg(0), env.arg(2), env.arg(3), env.arg(4));
    if utc != 0 {
        env.mem.write_u64(utc, local)?;
    }
    if tz != 0 {
        env.mem.write(tz, &[0; 16])?;
    }
    if dst != 0 {
        env.mem.write(dst, &0i32.to_le_bytes())?;
    }
    ret(0)
}

fn raise_exception(env: &mut Env) -> Result<Outcome, CallError> {
    let _ = env.kernel.write_output(
        2,
        format!("[sceKernelDebugRaiseException {:#x}]\n", env.arg(0)).as_bytes(),
    );
    Err(CallError::Abort("the title raised a debug exception"))
}
