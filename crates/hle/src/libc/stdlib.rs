//! `<stdlib.h>`, `<time.h>` and the rest of the C library that keeps state
//! or calls back into the guest.

use super::format::{float_prefix, int_prefix};
use super::{cstr, f, heap_cstr, ret, ret_f64};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome, errno};

pub static FUNCTIONS: &[Function] = &[
    f("_init_env", init_env),
    f("atexit", atexit),
    f("exit", exit),
    f("_Exit", exit_now),
    f("abort", abort),
    f("time", time),
    f("srand", srand),
    f("rand", rand),
    f("strerror", strerror),
    f("getenv", getenv),
    f("setenv", setenv),
    f("getcwd", getcwd),
    f("qsort", qsort),
    f("bsearch", bsearch),
    f("atoi", atoi),
    f("atof", atof),
    f("strtol", strtol),
    f("strtoll", strtol),
    f("strtoul", strtoul),
    f("strtoull", strtoul),
    f("strtod", strtod),
];

/// The current directory a title sees (our choice: its own folder).
const CWD: &[u8] = b"/app0";

/// `_init_env(process_parameters)`, which start-up code calls before
/// anything else. We keep no environment from the parameters (unknown layout).
fn init_env(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn atexit(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.libc.atexit).push(env.arg(0));
    ret(0)
}

/// Runs the `atexit` handlers, latest first, then ends the process.
fn exit(env: &mut Env) -> Result<Outcome, CallError> {
    let status = env.arg(0) as i32;
    loop {
        let Some(handler) = lock(&env.kernel.libc.atexit).pop() else {
            break;
        };
        env.call_guest(handler, &[])?;
    }
    Ok(Outcome::Exit(status))
}

fn exit_now(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Exit(env.arg(0) as i32))
}

fn abort(_: &mut Env) -> Result<Outcome, CallError> {
    Err(CallError::Abort("abort() called"))
}

fn time(env: &mut Env) -> Result<Outcome, CallError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    if env.arg(0) != 0 {
        env.mem.write_u64(env.arg(0), now)?;
    }
    ret(now)
}

fn srand(env: &mut Env) -> Result<Outcome, CallError> {
    *lock(&env.kernel.libc.rand) = env.arg(0) as u32 as u64;
    ret(0)
}

/// A 64-bit LCG (Knuth's MMIX constants), top 31 bits out. Orbis's
/// sequence is unknown, so the numbers differ from a real console's.
fn rand(env: &mut Env) -> Result<Outcome, CallError> {
    let mut s = lock(&env.kernel.libc.rand);
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ret(*s >> 33)
}

/// Messages from the host's C library: macOS and FreeBSD share the BSD
/// errno numbers and wording for the common errors.
fn strerror(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(0) as i32;
    if let Some(&at) = lock(&env.kernel.libc.strerror).get(&n) {
        return ret(at);
    }
    let message = std::io::Error::from_raw_os_error(n).to_string();
    let message = message.split(" (os error").next().unwrap_or_default();
    let at = heap_cstr(env, message.as_bytes())?;
    lock(&env.kernel.libc.strerror).insert(n, at);
    ret(at)
}

fn getenv(env: &mut Env) -> Result<Outcome, CallError> {
    let name = cstr(env, env.arg(0))?;
    ret(lock(&env.kernel.libc.environment)
        .get(&name)
        .copied()
        .unwrap_or(0))
}

/// `setenv(name, value, overwrite)`. Replaced strings are not freed: a
/// pointer from an earlier `getenv` stays valid.
fn setenv(env: &mut Env) -> Result<Outcome, CallError> {
    let name = cstr(env, env.arg(0))?;
    if name.is_empty() || name.contains(&b'=') {
        return env.fail(errno::EINVAL);
    }
    let overwrite = env.arg(2) as i32 != 0;
    if !overwrite && lock(&env.kernel.libc.environment).contains_key(&name) {
        return ret(0);
    }
    let value = cstr(env, env.arg(1))?;
    let at = heap_cstr(env, &value)?;
    if at == 0 {
        return env.fail(errno::ENOMEM);
    }
    lock(&env.kernel.libc.environment).insert(name, at);
    ret(0)
}

/// `getcwd(buf, size)`; a null `buf` gets a heap string, as FreeBSD does.
fn getcwd(env: &mut Env) -> Result<Outcome, CallError> {
    let (buf, size) = (env.arg(0), env.arg(1));
    if buf == 0 {
        return ret(heap_cstr(env, CWD)?);
    }
    if size < CWD.len() as u64 + 1 {
        env.set_errno(if size == 0 {
            errno::EINVAL
        } else {
            errno::ERANGE
        })?;
        return ret(0);
    }
    env.mem.write(buf, CWD)?;
    env.mem.write(buf + CWD.len() as u64, &[0])?;
    ret(buf)
}

/// Calls the guest comparator and reads its `int` result.
fn compare(env: &mut Env, cmp: u64, a: u64, b: u64) -> Result<i32, CallError> {
    Ok(env.call_guest(cmp, &[a, b])? as i32)
}

fn swap(env: &Env, a: u64, b: u64, size: usize) -> Result<(), CallError> {
    let (mut x, mut y) = (vec![0u8; size], vec![0u8; size]);
    env.mem.read(a, &mut x)?;
    env.mem.read(b, &mut y)?;
    env.mem.write(a, &y)?;
    env.mem.write(b, &x)?;
    Ok(())
}

/// `qsort(base, n, size, cmp)` as an in-place heapsort. Not stable, so equal
/// elements may land in a different order than on a console.
fn qsort(env: &mut Env) -> Result<Outcome, CallError> {
    let (base, n, size, cmp) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    if n < 2 || size == 0 {
        return ret(0);
    }
    let at = |i: u64| base + i * size;
    let sift = |env: &mut Env, mut root: u64, end: u64| -> Result<(), CallError> {
        loop {
            let mut child = 2 * root + 1;
            if child >= end {
                return Ok(());
            }
            if child + 1 < end && compare(env, cmp, at(child), at(child + 1))? < 0 {
                child += 1;
            }
            if compare(env, cmp, at(root), at(child))? >= 0 {
                return Ok(());
            }
            swap(env, at(root), at(child), size as usize)?;
            root = child;
        }
    };
    for root in (0..n / 2).rev() {
        sift(env, root, n)?;
    }
    for end in (1..n).rev() {
        swap(env, at(0), at(end), size as usize)?;
        sift(env, 0, end)?;
    }
    ret(0)
}

fn bsearch(env: &mut Env) -> Result<Outcome, CallError> {
    let (key, base, n, size, cmp) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3), env.arg(4));
    let (mut lo, mut hi) = (0, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let elem = base + mid * size;
        match compare(env, cmp, key, elem)? {
            0 => return ret(elem),
            c if c < 0 => hi = mid,
            _ => lo = mid + 1,
        }
    }
    ret(0)
}

fn skip_space(s: &[u8]) -> usize {
    s.iter()
        .take_while(|c| matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r'))
        .count()
}

/// Shared by the `strto*` integer functions: parses, stores `*endptr`, and
/// sets ERANGE on overflow.
fn strto_int(env: &mut Env, signed: bool) -> Result<Outcome, CallError> {
    let (at, end, base) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if base == 1 || base > 36 {
        return env.fail(errno::EINVAL);
    }
    let s = cstr(env, at)?;
    let lead = skip_space(&s);
    let (v, overflow, n) = int_prefix(&s[lead..], base, signed);
    if end != 0 {
        let used = if n == 0 { 0 } else { lead + n };
        env.mem.write_u64(end, at + used as u64)?;
    }
    if overflow {
        env.set_errno(errno::ERANGE)?;
    }
    ret(v)
}

fn strtol(env: &mut Env) -> Result<Outcome, CallError> {
    strto_int(env, true)
}

fn strtoul(env: &mut Env) -> Result<Outcome, CallError> {
    strto_int(env, false)
}

fn atoi(env: &mut Env) -> Result<Outcome, CallError> {
    let s = cstr(env, env.arg(0))?;
    let lead = skip_space(&s);
    ret(int_prefix(&s[lead..], 10, true).0 as i32 as u32 as u64)
}

/// The value of the floating-point prefix of `s` after white space, and the
/// bytes it used (0 if there is none).
fn parse_float(s: &[u8]) -> (f64, usize) {
    let lead = skip_space(s);
    let n = float_prefix(&s[lead..]);
    if n == 0 {
        return (0.0, 0);
    }
    let text = std::str::from_utf8(&s[lead..lead + n]).unwrap_or("0");
    (text.parse().unwrap_or(0.0), lead + n)
}

fn atof(env: &mut Env) -> Result<Outcome, CallError> {
    let s = cstr(env, env.arg(0))?;
    ret_f64(env, parse_float(&s).0)
}

fn strtod(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, end) = (env.arg(0), env.arg(1));
    let s = cstr(env, at)?;
    let (v, n) = parse_float(&s);
    if end != 0 {
        env.mem.write_u64(end, at + n as u64)?;
    }
    let text = &s[..n];
    let numeric = !text
        .iter()
        .any(|c| c.is_ascii_alphabetic() && !matches!(c, b'e' | b'E'));
    let mantissa = text
        .split(|c| matches!(c, b'e' | b'E'))
        .next()
        .unwrap_or_default();
    let underflow = v == 0.0 && mantissa.iter().any(|c| matches!(c, b'1'..=b'9'));
    if numeric && (v.is_infinite() || underflow) {
        env.set_errno(errno::ERANGE)?;
    }
    ret_f64(env, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_parsing_skips_white_space_and_reports_what_it_used() {
        assert_eq!(parse_float(b"  2.5abc"), (2.5, 5));
        assert_eq!(parse_float(b"x"), (0.0, 0));
        assert!(parse_float(b"-inf").0.is_infinite());
    }
}
