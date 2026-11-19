//! The `printf` and `scanf` families. Variadic arguments come from the
//! registers and stack as the SysV AMD64 convention passes them, or from a
//! `va_list` (`{ u32 gp_offset; u32 fp_offset; void *overflow_arg_area;
//! void *reg_save_area; }`).

use x86::cpu::reg::*;

use super::{cstr, f, ret, wstr};
use crate::{CallError, Env, Function, Outcome};

pub static FUNCTIONS: &[Function] = &[
    f("sprintf", sprintf),
    f("snprintf", snprintf),
    f("vsnprintf", vsnprintf),
    f("vsscanf", vsscanf),
];

const GP_ORDER: [usize; 6] = [RDI, RSI, RDX, RCX, R8, R9];

/// Where the variadic arguments of a call come from.
pub(crate) enum Args {
    /// A variadic call: `gp` integer and `fp` float registers are already
    /// taken by fixed arguments; the rest come after the return address.
    Call {
        gp: usize,
        fp: usize,
        stack: u64,
    },
    VaList(u64),
}

impl Args {
    /// The variadic arguments of the current call, after `fixed` integer arguments.
    pub(crate) fn call(env: &Env, fixed: usize) -> Self {
        Args::Call {
            gp: fixed,
            fp: 0,
            stack: env.cpu.regs[RSP] + 8,
        }
    }

    fn int(&mut self, env: &Env) -> Result<u64, CallError> {
        match self {
            Args::Call { gp, stack, .. } => {
                if *gp < 6 {
                    *gp += 1;
                    Ok(env.cpu.regs[GP_ORDER[*gp - 1]])
                } else {
                    let v = env.mem.read_u64(*stack)?;
                    *stack += 8;
                    Ok(v)
                }
            }
            Args::VaList(va) => {
                let mut b = [0u8; 4];
                env.mem.read(*va, &mut b)?;
                let gp_offset = u32::from_le_bytes(b) as u64;
                if gp_offset < 48 {
                    let v = env.mem.read_u64(env.mem.read_u64(*va + 16)? + gp_offset)?;
                    env.mem
                        .write(*va, &((gp_offset + 8) as u32).to_le_bytes())?;
                    Ok(v)
                } else {
                    overflow(env, *va)
                }
            }
        }
    }

    fn float(&mut self, env: &Env) -> Result<f64, CallError> {
        match self {
            Args::Call { fp, stack, .. } => {
                if *fp < 8 {
                    *fp += 1;
                    Ok(f64::from_bits(env.cpu.xmm[*fp - 1][0]))
                } else {
                    let v = env.mem.read_u64(*stack)?;
                    *stack += 8;
                    Ok(f64::from_bits(v))
                }
            }
            Args::VaList(va) => {
                let mut b = [0u8; 4];
                env.mem.read(*va + 4, &mut b)?;
                let fp_offset = u32::from_le_bytes(b) as u64;
                if fp_offset < 176 {
                    let v = env.mem.read_u64(env.mem.read_u64(*va + 16)? + fp_offset)?;
                    env.mem
                        .write(*va + 4, &((fp_offset + 16) as u32).to_le_bytes())?;
                    Ok(f64::from_bits(v))
                } else {
                    overflow(env, *va).map(f64::from_bits)
                }
            }
        }
    }
}

fn overflow(env: &Env, va: u64) -> Result<u64, CallError> {
    let area = env.mem.read_u64(va + 8)?;
    let v = env.mem.read_u64(area)?;
    env.mem.write_u64(va + 8, area + 8)?;
    Ok(v)
}

#[derive(Default, Clone, Copy)]
struct Spec {
    left: bool,
    plus: bool,
    space: bool,
    alt: bool,
    zero: bool,
    width: usize,
    precision: Option<usize>,
}

fn pad(spec: Spec, sign: &str, prefix: &str, body: String) -> Vec<u8> {
    let len = sign.len() + prefix.len() + body.len();
    let fill = spec.width.saturating_sub(len);
    let mut out = String::new();
    if spec.left {
        out += sign;
        out += prefix;
        out += &body;
        out += &" ".repeat(fill);
    } else if spec.zero {
        out += sign;
        out += prefix;
        out += &"0".repeat(fill);
        out += &body;
    } else {
        out += &" ".repeat(fill);
        out += sign;
        out += prefix;
        out += &body;
    }
    out.into_bytes()
}

fn sign_of(negative: bool, spec: Spec) -> &'static str {
    if negative {
        "-"
    } else if spec.plus {
        "+"
    } else if spec.space {
        " "
    } else {
        ""
    }
}

/// `%e` body for a finite non-negative value: `d.ddde+XX`.
fn exp_form(v: f64, prec: usize, upper: bool) -> String {
    let s = format!("{v:.prec$e}");
    let (mant, exp) = s.split_once('e').expect("Rust's {:e} has an exponent");
    let exp: i32 = exp.parse().expect("exponent is a number");
    let e = if upper { 'E' } else { 'e' };
    format!(
        "{mant}{e}{}{:02}",
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    )
}

fn strip_zeros(s: String) -> String {
    let (num, exp) = match s.find(['e', 'E']) {
        Some(i) => (s[..i].to_owned(), s[i..].to_owned()),
        None => (s, String::new()),
    };
    let num = if num.contains('.') {
        num.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        num
    };
    num + &exp
}

fn float(v: f64, conv: u8, spec: Spec) -> Vec<u8> {
    let upper = conv.is_ascii_uppercase();
    let negative = v.is_sign_negative();
    let a = v.abs();
    if !a.is_finite() {
        let s = if a.is_nan() { "nan" } else { "inf" };
        let s = if upper {
            s.to_uppercase()
        } else {
            s.to_owned()
        };
        return pad(
            Spec {
                zero: false,
                ..spec
            },
            sign_of(negative, spec),
            "",
            s,
        );
    }
    let prec = spec.precision.unwrap_or(6);
    let body = match conv.to_ascii_lowercase() {
        b'f' => {
            let s = format!("{a:.prec$}");
            if spec.alt && prec == 0 { s + "." } else { s }
        }
        b'e' => exp_form(a, prec, upper),
        b'g' => {
            let p = if prec == 0 { 1 } else { prec };
            let x: i32 = exp_form(a, p - 1, false)
                .split_once('e')
                .unwrap()
                .1
                .parse()
                .unwrap();
            let s = if (p as i32) > x && x >= -4 {
                format!("{a:.*}", (p as i32 - 1 - x) as usize)
            } else {
                exp_form(a, p - 1, upper)
            };
            if spec.alt { s } else { strip_zeros(s) }
        }
        _ => {
            let bits = a.to_bits();
            let exp = ((bits >> 52) & 0x7ff) as i32;
            let mant = bits & ((1 << 52) - 1);
            let (lead, e) = match (exp, mant) {
                (0, 0) => (0, 0),
                (0, _) => (0, -1022),
                _ => (1, exp - 1023),
            };
            let hex = format!("{mant:013x}");
            let hex = match spec.precision {
                Some(p) => hex[..p.min(13)].to_owned(),
                None => hex.trim_end_matches('0').to_owned(),
            };
            let dot = if hex.is_empty() && !spec.alt { "" } else { "." };
            let s = format!(
                "0x{lead}{dot}{hex}p{}{}",
                if e < 0 { '-' } else { '+' },
                e.abs()
            );
            if upper { s.to_uppercase() } else { s }
        }
    };
    pad(spec, sign_of(negative, spec), "", body)
}

fn integer(v: u64, conv: u8, signed: bool, spec: Spec) -> Vec<u8> {
    let negative = signed && (v as i64) < 0;
    let mag = if negative {
        (v as i64).unsigned_abs()
    } else {
        v
    };
    let mut body = match conv {
        b'o' => format!("{mag:o}"),
        b'x' => format!("{mag:x}"),
        b'X' => format!("{mag:X}"),
        _ => mag.to_string(),
    };
    let mut spec = spec;
    if let Some(p) = spec.precision {
        spec.zero = false;
        if p == 0 && mag == 0 {
            body.clear();
        } else if body.len() < p {
            body = "0".repeat(p - body.len()) + &body;
        }
    }
    let prefix = match conv {
        b'x' if spec.alt && mag != 0 => "0x",
        b'X' if spec.alt && mag != 0 => "0X",
        b'o' if spec.alt && !body.starts_with('0') => "0",
        _ => "",
    };
    let sign = if signed { sign_of(negative, spec) } else { "" };
    pad(spec, sign, prefix, body)
}

/// Formats `fmt` with arguments from `args`, as `printf` does.
pub(crate) fn printf(env: &Env, fmt: &[u8], args: &mut Args) -> Result<Vec<u8>, CallError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        i += 1;
        let mut spec = Spec::default();
        while let Some(&c) = fmt.get(i) {
            match c {
                b'-' => spec.left = true,
                b'+' => spec.plus = true,
                b' ' => spec.space = true,
                b'#' => spec.alt = true,
                b'0' => spec.zero = true,
                _ => break,
            }
            i += 1;
        }
        if fmt.get(i) == Some(&b'*') {
            let w = args.int(env)? as i32;
            if w < 0 {
                spec.left = true;
            }
            spec.width = w.unsigned_abs() as usize;
            i += 1;
        } else {
            while let Some(d) = fmt.get(i).filter(|c| c.is_ascii_digit()) {
                spec.width = spec.width * 10 + (d - b'0') as usize;
                i += 1;
            }
        }
        if fmt.get(i) == Some(&b'.') {
            i += 1;
            if fmt.get(i) == Some(&b'*') {
                let p = args.int(env)? as i32;
                spec.precision = (p >= 0).then_some(p as usize);
                i += 1;
            } else {
                let mut p = 0;
                while let Some(d) = fmt.get(i).filter(|c| c.is_ascii_digit()) {
                    p = p * 10 + (d - b'0') as usize;
                    i += 1;
                }
                spec.precision = Some(p);
            }
        }
        let mut bits = 32;
        while let Some(&c) = fmt.get(i) {
            match c {
                b'h' if bits == 16 => bits = 8,
                b'h' => bits = 16,
                b'l' | b'j' | b'z' | b't' | b'q' | b'L' => bits = 64,
                _ => break,
            }
            i += 1;
        }
        let Some(&conv) = fmt.get(i) else { break };
        i += 1;
        if spec.left {
            spec.zero = false;
        }
        match conv {
            b'%' => out.push(b'%'),
            b'd' | b'i' => {
                let v = args.int(env)?;
                let v = match bits {
                    8 => v as i8 as i64 as u64,
                    16 => v as i16 as i64 as u64,
                    32 => v as i32 as i64 as u64,
                    _ => v,
                };
                out.extend(integer(v, conv, true, spec));
            }
            b'u' | b'o' | b'x' | b'X' => {
                let v = args.int(env)?;
                let v = if bits == 64 {
                    v
                } else {
                    v & ((1u64 << bits) - 1)
                };
                out.extend(integer(v, conv, false, spec));
            }
            b'c' => {
                let c = args.int(env)?;
                let body = if bits == 64 {
                    char::from_u32(c as u16 as u32).unwrap_or('?').to_string()
                } else {
                    (c as u8 as char).to_string()
                };
                out.extend(pad(
                    Spec {
                        zero: false,
                        ..spec
                    },
                    "",
                    "",
                    body,
                ));
            }
            b's' => {
                let at = args.int(env)?;
                let mut s = if at == 0 {
                    b"(null)".to_vec()
                } else if bits == 64 {
                    String::from_utf16_lossy(&wstr(env, at)?).into_bytes()
                } else {
                    cstr(env, at)?
                };
                if let Some(p) = spec.precision {
                    s.truncate(p);
                }
                let fill = spec.width.saturating_sub(s.len());
                if spec.left {
                    out.extend(&s);
                    out.extend(std::iter::repeat_n(b' ', fill));
                } else {
                    out.extend(std::iter::repeat_n(b' ', fill));
                    out.extend(&s);
                }
            }
            b'p' => {
                let v = args.int(env)?;
                out.extend(pad(spec, "", "0x", format!("{v:x}")));
            }
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' | b'a' | b'A' => {
                let v = args.float(env)?;
                out.extend(float(v, conv, spec));
            }
            b'n' => {
                let at = args.int(env)?;
                let n = out.len() as u64;
                env.mem
                    .write(at, &n.to_le_bytes()[..(bits / 8).max(4) as usize])?;
            }
            other => {
                out.push(b'%');
                out.push(other);
            }
        }
    }
    Ok(out)
}

/// Writes at most `size - 1` bytes of `s` and a NUL to `dst`; returns the
/// full length, as `snprintf` does.
fn bounded(env: &Env, dst: u64, size: u64, s: &[u8]) -> Result<Outcome, CallError> {
    if size > 0 {
        let n = s.len().min(size as usize - 1);
        env.mem.write(dst, &s[..n])?;
        env.mem.write(dst + n as u64, &[0])?;
    }
    ret(s.len() as u64)
}

fn sprintf(env: &mut Env) -> Result<Outcome, CallError> {
    let fmt = cstr(env, env.arg(1))?;
    let s = printf(env, &fmt, &mut Args::call(env, 2))?;
    bounded(env, env.arg(0), s.len() as u64 + 1, &s)
}

fn snprintf(env: &mut Env) -> Result<Outcome, CallError> {
    let fmt = cstr(env, env.arg(2))?;
    let s = printf(env, &fmt, &mut Args::call(env, 3))?;
    bounded(env, env.arg(0), env.arg(1), &s)
}

fn vsnprintf(env: &mut Env) -> Result<Outcome, CallError> {
    let fmt = cstr(env, env.arg(2))?;
    let s = printf(env, &fmt, &mut Args::VaList(env.arg(3)))?;
    bounded(env, env.arg(0), env.arg(1), &s)
}

fn vsscanf(env: &mut Env) -> Result<Outcome, CallError> {
    let input = cstr(env, env.arg(0))?;
    let fmt = cstr(env, env.arg(1))?;
    let n = scanf(env, &input, &fmt, &mut Args::VaList(env.arg(2)))?;
    ret(n as u64)
}

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// The length of the longest prefix of `s` that reads as a floating-point number.
pub(crate) fn float_prefix(s: &[u8]) -> usize {
    let mut i = 0;
    if matches!(s.first(), Some(b'+' | b'-')) {
        i = 1;
    }
    let rest = &s[i..];
    let lower: Vec<u8> = rest
        .iter()
        .take(8)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if lower.starts_with(b"infinity") {
        return i + 8;
    }
    if lower.starts_with(b"inf") || lower.starts_with(b"nan") {
        return i + 3;
    }
    let start = i;
    while s.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    if s.get(i) == Some(&b'.') {
        i += 1;
        while s.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if i == start || (i == start + 1 && s[start] == b'.') {
        return 0;
    }
    if matches!(s.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(s.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        if s.get(j).is_some_and(u8::is_ascii_digit) {
            while s.get(j).is_some_and(u8::is_ascii_digit) {
                j += 1;
            }
            i = j;
        }
    }
    i
}

/// Parses an integer prefix in `base` (0 = auto); returns the value, whether
/// it overflowed, and how many bytes it took.
pub(crate) fn int_prefix(s: &[u8], base: u32, signed: bool) -> (u64, bool, usize) {
    let mut i = 0;
    let negative = match s.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let mut base = base;
    let hex_prefix = s.get(i) == Some(&b'0') && matches!(s.get(i + 1), Some(b'x' | b'X'));
    if (base == 16 || base == 0) && hex_prefix && s.get(i + 2).is_some_and(u8::is_ascii_hexdigit) {
        i += 2;
        base = 16;
    } else if base == 0 {
        base = if s.get(i) == Some(&b'0') { 8 } else { 10 };
    }
    let start = i;
    let mut v: u64 = 0;
    let mut overflow = false;
    while let Some(d) = s
        .get(i)
        .and_then(|c| (*c as char).to_digit(36))
        .filter(|&d| d < base)
    {
        match v
            .checked_mul(base as u64)
            .and_then(|x| x.checked_add(d as u64))
        {
            Some(x) => v = x,
            None => overflow = true,
        }
        i += 1;
    }
    if i == start {
        return (0, false, 0);
    }
    if signed {
        let limit = if negative {
            1u64 << 63
        } else {
            (1u64 << 63) - 1
        };
        if v > limit {
            overflow = true;
        }
        if overflow {
            v = limit;
        }
    } else if overflow {
        v = u64::MAX;
    }
    let v = if negative && !(overflow && !signed) {
        v.wrapping_neg()
    } else {
        v
    };
    (v, overflow, i)
}

/// Scans `input` by `fmt`, as `sscanf` does; returns the number of
/// assignments, or -1 if the input ended before the first conversion.
pub(crate) fn scanf(
    env: &Env,
    input: &[u8],
    fmt: &[u8],
    args: &mut Args,
) -> Result<i64, CallError> {
    let mut pos = 0;
    let mut assigned = 0i64;
    let mut converted = false;
    let mut i = 0;
    let store = |args: &mut Args, v: u64, bytes: usize| -> Result<(), CallError> {
        let at = args.int(env)?;
        env.mem.write(at, &v.to_le_bytes()[..bytes])?;
        Ok(())
    };
    while i < fmt.len() {
        let c = fmt[i];
        if is_space(c) {
            while input.get(pos).is_some_and(|&x| is_space(x)) {
                pos += 1;
            }
            i += 1;
            continue;
        }
        if c != b'%' || fmt.get(i + 1) == Some(&b'%') {
            let want = if c == b'%' {
                i += 1;
                b'%'
            } else {
                c
            };
            if input.get(pos) != Some(&want) {
                break;
            }
            pos += 1;
            i += 1;
            continue;
        }
        i += 1;
        let suppress = fmt.get(i) == Some(&b'*');
        if suppress {
            i += 1;
        }
        let mut width = 0usize;
        while let Some(d) = fmt.get(i).filter(|c| c.is_ascii_digit()) {
            width = width * 10 + (d - b'0') as usize;
            i += 1;
        }
        let mut bytes = 4;
        let mut double = false;
        while let Some(&l) = fmt.get(i) {
            match l {
                b'h' if bytes == 2 => bytes = 1,
                b'h' => bytes = 2,
                b'l' | b'L' | b'j' | b'z' | b't' | b'q' => {
                    bytes = 8;
                    double = true;
                }
                _ => break,
            }
            i += 1;
        }
        let Some(&conv) = fmt.get(i) else { break };
        i += 1;
        if !matches!(conv, b'c' | b'[' | b'n') {
            while input.get(pos).is_some_and(|&x| is_space(x)) {
                pos += 1;
            }
        }
        let limit = if width == 0 { usize::MAX } else { width };
        let avail = &input[pos..input.len().min(pos.saturating_add(limit))];
        if avail.is_empty() && conv != b'n' {
            return Ok(if converted || assigned > 0 {
                assigned
            } else {
                -1
            });
        }
        match conv {
            b'd' | b'i' | b'u' | b'o' | b'x' | b'X' | b'p' => {
                let base = match conv {
                    b'd' | b'u' => 10,
                    b'o' => 8,
                    b'x' | b'X' | b'p' => 16,
                    _ => 0,
                };
                let (v, _, n) = int_prefix(avail, base, matches!(conv, b'd' | b'i'));
                if n == 0 {
                    break;
                }
                pos += n;
                if !suppress {
                    store(args, v, if conv == b'p' { 8 } else { bytes })?;
                    assigned += 1;
                }
            }
            b'f' | b'e' | b'g' | b'E' | b'G' | b'a' => {
                let n = float_prefix(avail);
                if n == 0 {
                    break;
                }
                let v: f64 = std::str::from_utf8(&avail[..n])
                    .ok()
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(0.0);
                pos += n;
                if !suppress {
                    if double {
                        store(args, v.to_bits(), 8)?;
                    } else {
                        store(args, (v as f32).to_bits() as u64, 4)?;
                    }
                    assigned += 1;
                }
            }
            b's' => {
                let n = avail.iter().take_while(|&&x| !is_space(x)).count();
                if !suppress {
                    let at = args.int(env)?;
                    env.mem.write(at, &avail[..n])?;
                    env.mem.write(at + n as u64, &[0])?;
                    assigned += 1;
                }
                pos += n;
            }
            b'c' => {
                let n = if width == 0 { 1 } else { width }.min(avail.len());
                if !suppress {
                    let at = args.int(env)?;
                    env.mem.write(at, &avail[..n])?;
                    assigned += 1;
                }
                pos += n;
            }
            b'[' => {
                let negate = fmt.get(i) == Some(&b'^');
                if negate {
                    i += 1;
                }
                let mut set = Vec::new();
                if fmt.get(i) == Some(&b']') {
                    set.push(b']');
                    i += 1;
                }
                while let Some(&s) = fmt.get(i) {
                    i += 1;
                    if s == b']' {
                        break;
                    }
                    if fmt.get(i) == Some(&b'-') && fmt.get(i + 1).is_some_and(|&e| e != b']') {
                        let end = fmt[i + 1];
                        set.extend(s..=end);
                        i += 2;
                    } else {
                        set.push(s);
                    }
                }
                let n = avail
                    .iter()
                    .take_while(|x| set.contains(x) != negate)
                    .count();
                if n == 0 {
                    break;
                }
                if !suppress {
                    let at = args.int(env)?;
                    env.mem.write(at, &avail[..n])?;
                    env.mem.write(at + n as u64, &[0])?;
                    assigned += 1;
                }
                pos += n;
            }
            b'n' => {
                if !suppress {
                    store(args, pos as u64, bytes)?;
                }
                continue;
            }
            _ => break,
        }
        converted = true;
    }
    Ok(assigned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(width: usize, precision: Option<usize>) -> Spec {
        Spec {
            width,
            precision,
            ..Spec::default()
        }
    }

    #[test]
    fn integers_follow_printf_padding_rules() {
        assert_eq!(integer(42, b'd', true, spec(5, None)), b"   42");
        assert_eq!(
            integer(
                -42i64 as u64,
                b'd',
                true,
                Spec {
                    zero: true,
                    ..spec(6, None)
                }
            ),
            b"-00042"
        );
        assert_eq!(
            integer(
                255,
                b'x',
                false,
                Spec {
                    alt: true,
                    ..spec(0, None)
                }
            ),
            b"0xff"
        );
        assert_eq!(integer(7, b'd', true, spec(0, Some(3))), b"007");
        assert_eq!(integer(0, b'd', true, spec(0, Some(0))), b"");
    }

    #[test]
    fn floats_follow_printf_conversions() {
        assert_eq!(float(1.23456, b'f', spec(0, Some(2))), b"1.23");
        assert_eq!(float(1500.0, b'e', spec(0, None)), b"1.500000e+03");
        assert_eq!(float(0.0001, b'g', spec(0, None)), b"0.0001");
        assert_eq!(float(123456789.0, b'g', spec(0, None)), b"1.23457e+08");
        assert_eq!(float(100.0, b'g', spec(0, None)), b"100");
        assert_eq!(float(f64::NEG_INFINITY, b'F', spec(0, None)), b"-INF");
        assert_eq!(float(1.0, b'a', spec(0, None)), b"0x1p+0");
    }

    #[test]
    fn integer_prefixes_detect_base_and_overflow() {
        assert_eq!(int_prefix(b"0x1f!", 0, true), (31, false, 4));
        assert_eq!(int_prefix(b"017", 0, true), (15, false, 3));
        assert_eq!(int_prefix(b"-12abc", 10, true), (-12i64 as u64, false, 3));
        assert_eq!(
            int_prefix(b"99999999999999999999", 10, true).0,
            i64::MAX as u64
        );
        assert_eq!(int_prefix(b"zz", 10, true), (0, false, 0));
    }

    #[test]
    fn float_prefixes_stop_at_the_first_bad_byte() {
        assert_eq!(float_prefix(b"1.5e3x"), 5);
        assert_eq!(float_prefix(b"-inf"), 4);
        assert_eq!(float_prefix(b"1e+"), 1);
        assert_eq!(float_prefix(b"."), 0);
    }
}
