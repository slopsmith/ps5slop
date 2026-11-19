//! `malloc` and friends, on the host's guest heap ([`crate::heap`]).
//! Failures return null with `ENOMEM`, as POSIX says.

use super::{f, ret};
use crate::{CallError, Env, Function, Outcome, errno};

pub static FUNCTIONS: &[Function] = &[
    f("malloc", malloc),
    f("calloc", calloc),
    f("realloc", realloc),
    f("free", free),
    f("posix_memalign", posix_memalign),
    f("memalign", memalign),
    f("aligned_alloc", aligned_alloc),
];

/// Allocates `size` bytes at a multiple of `align`, or sets `ENOMEM` and
/// returns 0.
fn alloc(env: &Env, size: u64, align: u64) -> Result<u64, CallError> {
    match env.kernel.heap.alloc_aligned(env.mem, size, align) {
        Some(at) => Ok(at),
        None => {
            env.set_errno(errno::ENOMEM)?;
            Ok(0)
        }
    }
}

fn malloc(env: &mut Env) -> Result<Outcome, CallError> {
    ret(alloc(env, env.arg(0), 16)?)
}

fn calloc(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(size) = env.arg(0).checked_mul(env.arg(1)) else {
        env.set_errno(errno::ENOMEM)?;
        return ret(0);
    };
    let at = alloc(env, size, 16)?;
    if at != 0 {
        env.mem.write(at, &vec![0; size as usize])?;
    }
    ret(at)
}

/// `realloc(ptr, size)`: shrinking keeps the block, growing moves it.
fn realloc(env: &mut Env) -> Result<Outcome, CallError> {
    let (old, size) = (env.arg(0), env.arg(1));
    if old == 0 {
        return ret(alloc(env, size, 16)?);
    }
    let Some(old_size) = env.kernel.heap.size_of(old) else {
        return Err(CallError::Abort(
            "realloc of a pointer malloc did not return",
        ));
    };
    if size <= old_size {
        return ret(old);
    }
    let new = alloc(env, size, 16)?;
    if new == 0 {
        return ret(0);
    }
    let mut data = vec![0u8; old_size as usize];
    env.mem.read(old, &mut data)?;
    env.mem.write(new, &data)?;
    env.kernel.heap.free(old);
    ret(new)
}

fn free(env: &mut Env) -> Result<Outcome, CallError> {
    env.kernel.heap.free(env.arg(0));
    ret(0)
}

fn valid_alignment(align: u64) -> bool {
    align.is_power_of_two() && align.is_multiple_of(8)
}

/// `posix_memalign(out, align, size)` returns an error number instead of
/// setting errno.
fn posix_memalign(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, align, size) = (env.arg(0), env.arg(1), env.arg(2));
    if !valid_alignment(align) {
        return ret(errno::EINVAL as u64);
    }
    match env.kernel.heap.alloc_aligned(env.mem, size, align) {
        Some(at) => {
            env.mem.write_u64(out, at)?;
            ret(0)
        }
        None => ret(errno::ENOMEM as u64),
    }
}

fn memalign(env: &mut Env) -> Result<Outcome, CallError> {
    let (align, size) = (env.arg(0), env.arg(1));
    if !align.is_power_of_two() {
        env.set_errno(errno::EINVAL)?;
        return ret(0);
    }
    ret(alloc(env, size, align)?)
}

fn aligned_alloc(env: &mut Env) -> Result<Outcome, CallError> {
    memalign(env)
}
