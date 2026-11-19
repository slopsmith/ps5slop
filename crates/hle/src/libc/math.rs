//! `<math.h>` on the host's floating point. Results may differ from Orbis's
//! libm in the last bit; which algorithms it uses is unknown.

use super::{f, farg, farg32, ret, ret_f32, ret_f64};
use crate::{CallError, Env, Function, Outcome};

macro_rules! unary {
    ($($name:ident, $name_f:ident, $op:ident;)*) => {
        $(
            fn $name(env: &mut Env) -> Result<Outcome, CallError> {
                let v = farg(env, 0).$op();
                ret_f64(env, v)
            }
            fn $name_f(env: &mut Env) -> Result<Outcome, CallError> {
                let v = farg32(env, 0).$op();
                ret_f32(env, v)
            }
        )*
    };
}

unary! {
    acos, acosf, acos;
    asin, asinf, asin;
    atan, atanf, atan;
    cos, cosf, cos;
    sin, sinf, sin;
    tan, tanf, tan;
    exp, expf, exp;
    log, logf, ln;
    log10, log10f, log10;
}

pub static FUNCTIONS: &[Function] = &[
    f("acos", acos),
    f("acosf", acosf),
    f("asin", asin),
    f("asinf", asinf),
    f("atan", atan),
    f("atanf", atanf),
    f("cos", cos),
    f("cosf", cosf),
    f("sin", sin),
    f("sinf", sinf),
    f("tan", tan),
    f("tanf", tanf),
    f("exp", exp),
    f("expf", expf),
    f("log", log),
    f("logf", logf),
    f("log10", log10),
    f("log10f", log10f),
    f("atan2", atan2),
    f("atan2f", atan2f),
    f("pow", pow),
    f("powf", powf),
    f("fmod", fmod),
    f("fmodf", fmodf),
    f("scalbn", scalbn),
    f("scalbnf", scalbnf),
    f("lround", lround),
    f("lroundf", lroundf),
];

fn atan2(env: &mut Env) -> Result<Outcome, CallError> {
    let v = farg(env, 0).atan2(farg(env, 1));
    ret_f64(env, v)
}

fn atan2f(env: &mut Env) -> Result<Outcome, CallError> {
    let v = farg32(env, 0).atan2(farg32(env, 1));
    ret_f32(env, v)
}

fn pow(env: &mut Env) -> Result<Outcome, CallError> {
    let v = farg(env, 0).powf(farg(env, 1));
    ret_f64(env, v)
}

fn powf(env: &mut Env) -> Result<Outcome, CallError> {
    let v = farg32(env, 0).powf(farg32(env, 1));
    ret_f32(env, v)
}

/// C's `fmod` truncates, which is Rust's `%` on floats.
fn fmod(env: &mut Env) -> Result<Outcome, CallError> {
    let v = farg(env, 0) % farg(env, 1);
    ret_f64(env, v)
}

fn fmodf(env: &mut Env) -> Result<Outcome, CallError> {
    let v = farg32(env, 0) % farg32(env, 1);
    ret_f32(env, v)
}

/// `scalbn(x, n)`: `x * 2^n`, with `n` an `int` in EDI.
fn scalbn(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(0) as i32;
    let v = farg(env, 0) * 2f64.powi(n);
    ret_f64(env, v)
}

fn scalbnf(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(0) as i32;
    let v = (farg32(env, 0) as f64 * 2f64.powi(n)) as f32;
    ret_f32(env, v)
}

/// `lround(x)`: rounds half away from zero, as Rust's `round` does.
fn lround(env: &mut Env) -> Result<Outcome, CallError> {
    ret(farg(env, 0).round() as i64 as u64)
}

fn lroundf(env: &mut Env) -> Result<Outcome, CallError> {
    ret(farg32(env, 0).round() as i64 as u64)
}
