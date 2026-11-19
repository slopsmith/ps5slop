//! libSceLibcInternal: the C library. Behaviour follows the C standard and
//! POSIX, with FreeBSD's choices where they leave room (Orbis's libc is
//! FreeBSD-derived; details unverified). `wchar_t` is 16 bits, as clang's
//! PS4/PS5 targets define it.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::{CallError, DATA_BASE, Env, Function, Outcome, Variable};

mod format;
mod malloc;
mod math;
mod stdio;
mod stdlib;
mod string;

pub(crate) use stdio::{FdInfo, Files};

pub const LIBRARY: &str = "libSceLibcInternal";

/// stdio's standard streams, at fixed addresses in the HLE data region.
pub const STDIN_FILE: u64 = DATA_BASE;
pub const STDOUT_FILE: u64 = DATA_BASE + 0x100;
pub const STDERR_FILE: u64 = DATA_BASE + 0x200;

const STDINP: [u8; 8] = STDIN_FILE.to_le_bytes();
const STDOUTP: [u8; 8] = STDOUT_FILE.to_le_bytes();
const STDERRP: [u8; 8] = STDERR_FILE.to_le_bytes();

pub static VARIABLES: &[Variable] = &[
    Variable {
        library: LIBRARY,
        name: "__stdinp",
        init: &STDINP,
    },
    Variable {
        library: LIBRARY,
        name: "__stdoutp",
        init: &STDOUTP,
    },
    Variable {
        library: LIBRARY,
        name: "__stderrp",
        init: &STDERRP,
    },
    Variable {
        library: LIBRARY,
        name: "__isthreaded",
        init: &[1, 0, 0, 0],
    },
];

pub fn functions() -> impl Iterator<Item = &'static Function> {
    string::FUNCTIONS
        .iter()
        .chain(format::FUNCTIONS)
        .chain(malloc::FUNCTIONS)
        .chain(math::FUNCTIONS)
        .chain(stdio::FUNCTIONS)
        .chain(stdlib::FUNCTIONS)
}

/// The C library's process-wide state.
#[derive(Default)]
pub struct State {
    atexit: Mutex<Vec<u64>>,
    environment: Mutex<HashMap<Vec<u8>, u64>>,
    strerror: Mutex<HashMap<i32, u64>>,
    rand: Mutex<u64>,
    pub(crate) files: Files,
}

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: LIBRARY,
        name,
        func,
    }
}

/// The longest string a host function reads from guest memory.
const MAX_STRING: usize = 1 << 20;

/// The NUL-terminated string at `addr`.
pub(crate) fn cstr(env: &Env, addr: u64) -> Result<Vec<u8>, CallError> {
    Ok(env.mem.read_cstr(addr, MAX_STRING)?)
}

/// The NUL-terminated 16-bit string at `addr`.
fn wstr(env: &Env, addr: u64) -> Result<Vec<u16>, CallError> {
    let mut out = Vec::new();
    for i in 0..MAX_STRING as u64 {
        let mut b = [0u8; 2];
        env.mem.read(addr + 2 * i, &mut b)?;
        let c = u16::from_le_bytes(b);
        if c == 0 {
            break;
        }
        out.push(c);
    }
    Ok(out)
}

/// Copies `s` and a NUL into the heap and returns its address (0 when the
/// heap is full).
pub(crate) fn heap_cstr(env: &Env, s: &[u8]) -> Result<u64, CallError> {
    let Some(at) = env.kernel.heap.alloc(env.mem, s.len() as u64 + 1) else {
        return Ok(0);
    };
    env.mem.write(at, s)?;
    env.mem.write(at + s.len() as u64, &[0])?;
    Ok(at)
}

/// Floating-point argument `n` (XMMn) as a double.
fn farg(env: &Env, n: usize) -> f64 {
    f64::from_bits(env.cpu.xmm[n][0])
}

fn farg32(env: &Env, n: usize) -> f32 {
    f32::from_bits(env.cpu.xmm[n][0] as u32)
}

/// Returns a double in XMM0.
fn ret_f64(env: &mut Env, v: f64) -> Result<Outcome, CallError> {
    env.cpu.xmm[0] = [v.to_bits(), 0, 0, 0];
    Ok(Outcome::Return(0))
}

fn ret_f32(env: &mut Env, v: f32) -> Result<Outcome, CallError> {
    env.cpu.xmm[0] = [v.to_bits() as u64, 0, 0, 0];
    Ok(Outcome::Return(0))
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}
