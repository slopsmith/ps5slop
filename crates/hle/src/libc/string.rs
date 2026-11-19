//! `<string.h>`, `<ctype.h>`, `<wchar.h>` and the string functions that
//! need no state.

use super::{cstr, f, heap_cstr, ret, wstr};
use crate::{CallError, Env, Function, Outcome};

pub static FUNCTIONS: &[Function] = &[
    f("memcpy", memmove),
    f("memmove", memmove),
    f("memset", memset),
    f("memcmp", memcmp),
    f("strlen", strlen),
    f("strcmp", strcmp),
    f("strncmp", strncmp),
    f("strcasecmp", strcasecmp),
    f("strncasecmp", strncasecmp),
    f("strcpy", strcpy),
    f("strcat", strcat),
    f("strlcpy", strlcpy),
    f("strlcat", strlcat),
    f("strchr", strchr),
    f("strrchr", strrchr),
    f("strstr", strstr),
    f("strcasestr", strcasestr),
    f("strdup", strdup),
    f("strtok_r", strtok_r),
    f("isalnum", isalnum),
    f("isalpha", isalpha),
    f("isblank", isblank),
    f("iscntrl", iscntrl),
    f("isdigit", isdigit),
    f("isgraph", isgraph),
    f("islower", islower),
    f("isprint", isprint),
    f("ispunct", ispunct),
    f("isspace", isspace),
    f("isupper", isupper),
    f("isxdigit", isxdigit),
    f("tolower", tolower),
    f("toupper", toupper),
    f("wcslen", wcslen),
    f("wcscmp", wcscmp),
    f("wcsncmp", wcsncmp),
    f("wcsstr", wcsstr),
    f("mbrtowc", mbrtowc),
    f("wcrtomb", wcrtomb),
    f("wcstombs", wcstombs),
];

/// Copies through a host buffer, so overlapping ranges work as for memmove.
fn memmove(env: &mut Env) -> Result<Outcome, CallError> {
    let (dst, src, n) = (env.arg(0), env.arg(1), env.arg(2) as usize);
    let mut buf = vec![0u8; n];
    env.mem.read(src, &mut buf)?;
    env.mem.write(dst, &buf)?;
    ret(dst)
}

fn memset(env: &mut Env) -> Result<Outcome, CallError> {
    let (dst, c, n) = (env.arg(0), env.arg(1) as u8, env.arg(2) as usize);
    env.mem.write(dst, &vec![c; n])?;
    ret(dst)
}

fn order(o: std::cmp::Ordering) -> u64 {
    o as i32 as i64 as u64
}

fn memcmp(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(2) as usize;
    let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
    env.mem.read(env.arg(0), &mut a)?;
    env.mem.read(env.arg(1), &mut b)?;
    ret(order(a.cmp(&b)))
}

fn strlen(env: &mut Env) -> Result<Outcome, CallError> {
    ret(cstr(env, env.arg(0))?.len() as u64)
}

fn strcmp(env: &mut Env) -> Result<Outcome, CallError> {
    ret(order(cstr(env, env.arg(0))?.cmp(&cstr(env, env.arg(1))?)))
}

fn truncated(mut s: Vec<u8>, n: usize) -> Vec<u8> {
    s.truncate(n);
    s
}

fn strncmp(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(2) as usize;
    let (a, b) = (
        truncated(cstr(env, env.arg(0))?, n),
        truncated(cstr(env, env.arg(1))?, n),
    );
    ret(order(a.cmp(&b)))
}

fn lower(s: Vec<u8>) -> Vec<u8> {
    s.to_ascii_lowercase()
}

fn strcasecmp(env: &mut Env) -> Result<Outcome, CallError> {
    ret(order(
        lower(cstr(env, env.arg(0))?).cmp(&lower(cstr(env, env.arg(1))?)),
    ))
}

fn strncasecmp(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(2) as usize;
    let a = lower(truncated(cstr(env, env.arg(0))?, n));
    let b = lower(truncated(cstr(env, env.arg(1))?, n));
    ret(order(a.cmp(&b)))
}

fn put(env: &Env, at: u64, s: &[u8]) -> Result<(), CallError> {
    env.mem.write(at, s)?;
    env.mem.write(at + s.len() as u64, &[0])?;
    Ok(())
}

fn strcpy(env: &mut Env) -> Result<Outcome, CallError> {
    let s = cstr(env, env.arg(1))?;
    put(env, env.arg(0), &s)?;
    ret(env.arg(0))
}

fn strcat(env: &mut Env) -> Result<Outcome, CallError> {
    let d = cstr(env, env.arg(0))?;
    let s = cstr(env, env.arg(1))?;
    put(env, env.arg(0) + d.len() as u64, &s)?;
    ret(env.arg(0))
}

/// `strlcpy(dst, src, size)`: returns `strlen(src)`.
fn strlcpy(env: &mut Env) -> Result<Outcome, CallError> {
    let (dst, size) = (env.arg(0), env.arg(2) as usize);
    let s = cstr(env, env.arg(1))?;
    if size > 0 {
        put(env, dst, &s[..s.len().min(size - 1)])?;
    }
    ret(s.len() as u64)
}

/// `strlcat(dst, src, size)`: returns the length it tried to make.
fn strlcat(env: &mut Env) -> Result<Outcome, CallError> {
    let (dst, size) = (env.arg(0), env.arg(2) as usize);
    let s = cstr(env, env.arg(1))?;
    let d = truncated(cstr(env, dst)?, size);
    if d.len() < size {
        let room = size - d.len() - 1;
        put(env, dst + d.len() as u64, &s[..s.len().min(room)])?;
    }
    ret((d.len() + s.len()) as u64)
}

fn strchr(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, c) = (env.arg(0), env.arg(1) as u8);
    let s = cstr(env, at)?;
    if c == 0 {
        return ret(at + s.len() as u64);
    }
    ret(s.iter().position(|&x| x == c).map_or(0, |i| at + i as u64))
}

fn strrchr(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, c) = (env.arg(0), env.arg(1) as u8);
    let s = cstr(env, at)?;
    if c == 0 {
        return ret(at + s.len() as u64);
    }
    ret(s.iter().rposition(|&x| x == c).map_or(0, |i| at + i as u64))
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn strstr(env: &mut Env) -> Result<Outcome, CallError> {
    let at = env.arg(0);
    let (h, n) = (cstr(env, at)?, cstr(env, env.arg(1))?);
    ret(find(&h, &n).map_or(0, |i| at + i as u64))
}

fn strcasestr(env: &mut Env) -> Result<Outcome, CallError> {
    let at = env.arg(0);
    let (h, n) = (lower(cstr(env, at)?), lower(cstr(env, env.arg(1))?));
    ret(find(&h, &n).map_or(0, |i| at + i as u64))
}

fn strdup(env: &mut Env) -> Result<Outcome, CallError> {
    let s = cstr(env, env.arg(0))?;
    ret(heap_cstr(env, &s)?)
}

/// `strtok_r(str, sep, last)`.
fn strtok_r(env: &mut Env) -> Result<Outcome, CallError> {
    let (start, sep, last) = (env.arg(0), env.arg(1), env.arg(2));
    let mut at = if start != 0 {
        start
    } else {
        env.mem.read_u64(last)?
    };
    if at == 0 {
        return ret(0);
    }
    let sep = cstr(env, sep)?;
    let s = cstr(env, at)?;
    let skip = s.iter().take_while(|c| sep.contains(c)).count();
    if skip == s.len() {
        env.mem.write_u64(last, 0)?;
        return ret(0);
    }
    at += skip as u64;
    let token = &s[skip..];
    match token.iter().position(|c| sep.contains(c)) {
        Some(end) => {
            env.mem.write(at + end as u64, &[0])?;
            env.mem.write_u64(last, at + end as u64 + 1)?;
        }
        None => env.mem.write_u64(last, 0)?,
    }
    ret(at)
}

fn ctype(env: &Env, test: fn(&u8) -> bool) -> Result<Outcome, CallError> {
    let c = env.arg(0) as i32;
    ret(((0..=255).contains(&c) && test(&(c as u8))) as u64)
}

fn isalnum(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_alphanumeric)
}
fn isalpha(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_alphabetic)
}
fn isblank(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, |c| *c == b' ' || *c == b'\t')
}
fn iscntrl(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_control)
}
fn isdigit(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_digit)
}
fn isgraph(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_graphic)
}
fn islower(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_lowercase)
}
fn isprint(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, |c| c.is_ascii_graphic() || *c == b' ')
}
fn ispunct(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_punctuation)
}
fn isspace(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, |c| {
        matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
    })
}
fn isupper(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_uppercase)
}
fn isxdigit(env: &mut Env) -> Result<Outcome, CallError> {
    ctype(env, u8::is_ascii_hexdigit)
}

fn tolower(env: &mut Env) -> Result<Outcome, CallError> {
    let c = env.arg(0) as i32;
    ret(if (b'A' as i32..=b'Z' as i32).contains(&c) {
        c + 32
    } else {
        c
    } as u32 as u64)
}

fn toupper(env: &mut Env) -> Result<Outcome, CallError> {
    let c = env.arg(0) as i32;
    ret(if (b'a' as i32..=b'z' as i32).contains(&c) {
        c - 32
    } else {
        c
    } as u32 as u64)
}

fn wcslen(env: &mut Env) -> Result<Outcome, CallError> {
    ret(wstr(env, env.arg(0))?.len() as u64)
}

fn wcscmp(env: &mut Env) -> Result<Outcome, CallError> {
    ret(order(wstr(env, env.arg(0))?.cmp(&wstr(env, env.arg(1))?)))
}

fn wcsncmp(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(2) as usize;
    let (mut a, mut b) = (wstr(env, env.arg(0))?, wstr(env, env.arg(1))?);
    a.truncate(n);
    b.truncate(n);
    ret(order(a.cmp(&b)))
}

fn wcsstr(env: &mut Env) -> Result<Outcome, CallError> {
    let at = env.arg(0);
    let (h, n) = (wstr(env, at)?, wstr(env, env.arg(1))?);
    if n.is_empty() {
        return ret(at);
    }
    ret(h
        .windows(n.len())
        .position(|w| w == n.as_slice())
        .map_or(0, |i| at + 2 * i as u64))
}

/// `mbrtowc(pwc, s, n, ps)` for UTF-8 (shift state is not kept between calls).
fn mbrtowc(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, s, n) = (env.arg(0), env.arg(1), env.arg(2) as usize);
    if s == 0 {
        return ret(0);
    }
    let mut buf = vec![0u8; n.min(4)];
    for (i, b) in buf.iter_mut().enumerate() {
        let mut x = [0u8];
        env.mem.read(s + i as u64, &mut x)?;
        *b = x[0];
    }
    let (c, used) = match buf.first() {
        None => return ret(-2i64 as u64),
        Some(0) => (0u32, 0usize),
        Some(&b) if b < 0x80 => (b as u32, 1),
        Some(_) => match std::str::from_utf8(&buf)
            .or_else(|e| std::str::from_utf8(&buf[..e.valid_up_to()]))
        {
            Ok(st) if !st.is_empty() => {
                let ch = st.chars().next().unwrap();
                (ch as u32, ch.len_utf8())
            }
            _ => {
                env.set_errno(crate::errno::EILSEQ)?;
                return ret(-1i64 as u64);
            }
        },
    };
    if out != 0 {
        env.mem.write(out, &(c.min(0xffff) as u16).to_le_bytes())?;
    }
    ret(used as u64)
}

/// `wcrtomb(s, wc, ps)` to UTF-8.
fn wcrtomb(env: &mut Env) -> Result<Outcome, CallError> {
    let (s, wc) = (env.arg(0), env.arg(1) as u16 as u32);
    let Some(ch) = char::from_u32(wc) else {
        env.set_errno(crate::errno::EILSEQ)?;
        return ret(-1i64 as u64);
    };
    let mut buf = [0u8; 4];
    let bytes = ch.encode_utf8(&mut buf).as_bytes();
    if s != 0 {
        env.mem.write(s, bytes)?;
    }
    ret(bytes.len() as u64)
}

/// `wcstombs(dst, src, n)` to UTF-8.
fn wcstombs(env: &mut Env) -> Result<Outcome, CallError> {
    let (dst, src, n) = (env.arg(0), env.arg(1), env.arg(2) as usize);
    let s = String::from_utf16_lossy(&wstr(env, src)?);
    let bytes = s.as_bytes();
    if dst == 0 {
        return ret(bytes.len() as u64);
    }
    let take = bytes.len().min(n);
    env.mem.write(dst, &bytes[..take])?;
    if take < n {
        env.mem.write(dst + take as u64, &[0])?;
    }
    ret(take as u64)
}
