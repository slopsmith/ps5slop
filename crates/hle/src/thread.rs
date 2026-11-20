//! Creating, joining and ending guest threads. Each guest thread runs on its
//! own host thread (see `docs/decisions.md`); `pthread_t` is the thread
//! pointer. Of the thread attributes, only the stack size and detach state
//! are used.

use std::collections::HashMap;

use std::sync::{Arc, Condvar, Mutex};

use x86::Cpu;
use x86::cpu::reg::*;

use crate::errno::{EDEADLK, EINVAL, ESRCH};
use crate::pthread::{POLL, lock, ret};
use crate::{CallError, Env, Function, INTERNAL, Outcome};

/// Stack size of threads made by `pthread_create`. The Orbis default is unknown.
const THREAD_STACK: u64 = 1 << 20;

/// The internal function a thread's start routine returns to.
pub const THREAD_RETURN: &str = "thread_return";

#[derive(Default)]
struct Record {
    result: Mutex<Option<u64>>,
    finished: Condvar,
    detached: Mutex<bool>,
}

/// FreeBSD's `PTHREAD_CREATE_DETACHED`.
const CREATE_DETACHED: u64 = 1;
/// FreeBSD's `PTHREAD_STACK_MIN`; unverified for Orbis.
const STACK_MIN: u64 = 2048;

#[derive(Clone, Copy)]
struct Attr {
    stack: u64,
    detached: bool,
}

impl Default for Attr {
    fn default() -> Self {
        Self {
            stack: THREAD_STACK,
            detached: false,
        }
    }
}

/// Threads made by `pthread_create`, by thread pointer, and thread
/// attribute objects, by the handle in the guest's `pthread_attr_t` word.
#[derive(Default)]
pub struct Threads {
    records: Mutex<HashMap<u64, Arc<Record>>>,
    attrs: Mutex<HashMap<u64, Attr>>,
}

pub static FUNCTIONS: &[Function] = &[
    f("pthread_create", create),
    f("pthread_join", join),
    f("pthread_detach", detach),
    f("pthread_exit", exit),
    f("pthread_set_name_np", set_name),
    f("pthread_attr_init", attr_init),
    f("pthread_attr_destroy", attr_destroy),
    f("pthread_attr_setdetachstate", attr_setdetachstate),
    f("pthread_attr_setstacksize", attr_setstacksize),
    Function {
        library: INTERNAL,
        name: THREAD_RETURN,
        func: thread_return,
    },
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

/// `pthread_set_name_np(thread, name)`, also `scePthreadRename`: kept for
/// the control socket only.
fn set_name(env: &mut Env) -> Result<Outcome, CallError> {
    remember_name(env, env.arg(0), env.arg(1));
    ret(0)
}

fn remember_name(env: &Env, tcb: u64, name: u64) {
    if let Ok(n) = env.mem.read_cstr(name, 64) {
        lock(&env.kernel.names).insert(tcb, String::from_utf8_lossy(&n).into_owned());
    }
}

/// `scePthreadCreate(thread, attr, start, arg, name)`.
fn sce_create_named(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, name) = (env.arg(0), env.arg(4));
    let result = sce_create(env)?;
    if name != 0
        && matches!(result, Outcome::Return(0))
        && let Ok(tcb) = env.mem.read_u64(out)
    {
        remember_name(env, tcb, name);
    }
    Ok(result)
}

/// The attributes behind the `pthread_attr_t` at `at` (0 for the defaults).
fn attr(env: &Env, at: u64) -> Result<Option<Attr>, CallError> {
    if at == 0 {
        return Ok(Some(Attr::default()));
    }
    let id = env.mem.read_u64(at)?;
    Ok(lock(&env.kernel.threads.attrs).get(&id).copied())
}

fn attr_init(env: &mut Env) -> Result<Outcome, CallError> {
    let threads = &env.kernel.threads;
    let id = crate::pthread::new_handle(env)?;
    lock(&threads.attrs).insert(id, Attr::default());
    env.mem.write_u64(env.arg(0), id)?;
    ret(0)
}

fn attr_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    match lock(&env.kernel.threads.attrs).remove(&id) {
        Some(_) => ret(0),
        None => ret(EINVAL),
    }
}

/// Changes the attributes at `arg(0)` with `change`, which may refuse.
fn update_attr(env: &Env, change: impl FnOnce(&mut Attr) -> bool) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    let mut attrs = lock(&env.kernel.threads.attrs);
    let accepted = attrs.get_mut(&id).is_some_and(change);
    ret(if accepted { 0 } else { EINVAL })
}

fn attr_setdetachstate(env: &mut Env) -> Result<Outcome, CallError> {
    let state = env.arg(1) as u32 as u64;
    update_attr(env, |a| {
        a.detached = state == CREATE_DETACHED;
        state <= CREATE_DETACHED
    })
}

fn attr_setstacksize(env: &mut Env) -> Result<Outcome, CallError> {
    let size = env.arg(1);
    update_attr(env, |a| {
        a.stack = size.next_multiple_of(memory::PAGE_SIZE);
        size >= STACK_MIN
    })
}

/// `pthread_create(thread, attr, start, arg)`.
fn create(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, start, arg) = (env.arg(0), env.arg(2), env.arg(3));
    let Some(attr) = attr(env, env.arg(1))? else {
        return ret(EINVAL);
    };
    let kernel = env.kernel;
    let (Ok(tcb), Ok(top)) = (
        kernel.new_tls(env.mem),
        kernel.new_stack(env.mem, attr.stack),
    ) else {
        return ret(crate::errno::EAGAIN);
    };
    let sp = top - 8;
    env.mem.write_u64(sp, kernel.hle.internal(THREAD_RETURN))?;
    let mut cpu = Cpu {
        rip: start,
        fs_base: tcb,
        mxcsr: env.cpu.mxcsr,
        fcw: env.cpu.fcw,
        ..Cpu::default()
    };
    cpu.regs[RDI] = arg;
    cpu.regs[RSP] = sp;
    let record = Record {
        detached: Mutex::new(attr.detached),
        ..Record::default()
    };
    lock(&kernel.threads.records).insert(tcb, Arc::new(record));
    env.mem.write_u64(out, tcb)?;
    kernel.spawn_thread(cpu);
    ret(0)
}

/// Ends the calling thread with `result` for `pthread_join`.
fn finish(env: &mut Env, result: u64) -> Result<Outcome, CallError> {
    let tcb = env.thread.tcb;
    let record = lock(&env.kernel.threads.records).get(&tcb).cloned();
    if let Some(r) = record {
        *lock(&r.result) = Some(result);
        r.finished.notify_all();
        if *lock(&r.detached) {
            lock(&env.kernel.threads.records).remove(&tcb);
        }
    }
    Ok(Outcome::ThreadExit)
}

/// Where a start routine returns: its result is in RAX.
fn thread_return(env: &mut Env) -> Result<Outcome, CallError> {
    let result = env.cpu.regs[RAX];
    finish(env, result)
}

/// `pthread_exit(result)`.
fn exit(env: &mut Env) -> Result<Outcome, CallError> {
    let result = env.arg(0);
    finish(env, result)
}

/// `pthread_join(thread, result)`: waits for the thread and stores its result.
fn join(env: &mut Env) -> Result<Outcome, CallError> {
    let (tcb, out) = (env.arg(0), env.arg(1));
    if tcb == env.thread.tcb {
        return ret(EDEADLK);
    }
    let Some(r) = lock(&env.kernel.threads.records).get(&tcb).cloned() else {
        return ret(ESRCH);
    };
    if *lock(&r.detached) {
        return ret(EINVAL);
    }
    let mut result = lock(&r.result);
    while result.is_none() {
        if env.kernel.has_exited() {
            return ret(0);
        }
        result = r
            .finished
            .wait_timeout(result, POLL)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    let value = result.expect("checked above");
    drop(result);
    lock(&env.kernel.threads.records).remove(&tcb);
    if out != 0 {
        env.mem.write_u64(out, value)?;
    }
    ret(0)
}

/// `pthread_detach(thread)`: its record goes away when it ends.
fn detach(env: &mut Env) -> Result<Outcome, CallError> {
    let tcb = env.arg(0);
    let Some(r) = lock(&env.kernel.threads.records).get(&tcb).cloned() else {
        return ret(ESRCH);
    };
    *lock(&r.detached) = true;
    if lock(&r.result).is_some() {
        lock(&env.kernel.threads.records).remove(&tcb);
    }
    ret(0)
}

macro_rules! sce_wrappers {
    ($($sce:ident => $posix:ident),* $(,)?) => {
        $(fn $sce(env: &mut Env) -> Result<Outcome, CallError> {
            crate::pthread::sce($posix(env))
        })*
    };
}

sce_wrappers! {
    sce_create => create,
    sce_join => join,
    sce_detach => detach,
    sce_attr_init => attr_init,
    sce_attr_destroy => attr_destroy,
    sce_attr_setdetachstate => attr_setdetachstate,
    sce_attr_setstacksize => attr_setstacksize,
}

/// Accepted and ignored: affinity, priorities, scheduling policy and
/// inheritance, and names (host threads keep their own).
fn sce_ok(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// `scePthreadAttrGetdetachstate(attr, state)`.
fn sce_attr_getdetachstate(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(a) = attr(env, env.arg(0))? else {
        return crate::pthread::sce(ret(EINVAL));
    };
    env.mem
        .write(env.arg(1), &(a.detached as u32).to_le_bytes())?;
    ret(0)
}

/// `scePthreadAttrGetstacksize(attr, size)`.
fn sce_attr_getstacksize(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(a) = attr(env, env.arg(0))? else {
        return crate::pthread::sce(ret(EINVAL));
    };
    env.mem.write_u64(env.arg(1), a.stack)?;
    ret(0)
}

/// `scePthreadAttrGetschedparam(attr, param)`: the default priority.
fn sce_attr_getschedparam(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), &PRIORITY_DEFAULT.to_le_bytes())?;
    ret(0)
}

/// `scePthreadAttrGetaffinity(attr, mask)`: every CPU.
fn sce_attr_getaffinity(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write_u64(env.arg(1), ALL_CPUS)?;
    ret(0)
}

/// `scePthreadAttrGet(thread, attr)`: fresh default attributes.
fn sce_attr_get(env: &mut Env) -> Result<Outcome, CallError> {
    let threads = &env.kernel.threads;
    let id = crate::pthread::new_handle(env)?;
    lock(&threads.attrs).insert(id, Attr::default());
    env.mem.write_u64(env.arg(1), id)?;
    ret(0)
}

/// `scePthreadGetname(thread, name)`: names are not kept.
fn sce_getname(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), b"thread\0")?;
    ret(0)
}

/// `scePthreadGetthreadid()`: a number unique to the thread.
fn sce_getthreadid(env: &mut Env) -> Result<Outcome, CallError> {
    ret((env.thread.tcb >> 14) as i32)
}

fn sce_yield(_: &mut Env) -> Result<Outcome, CallError> {
    std::thread::yield_now();
    ret(0)
}

/// Orbis's default thread priority (shadPS4, PS4; unverified for the PS5).
const PRIORITY_DEFAULT: u32 = 700;
const ALL_CPUS: u64 = 0xff;

pub static SCE_FUNCTIONS: &[Function] = &[
    f("scePthreadCreate", sce_create_named),
    f("scePthreadJoin", sce_join),
    f("scePthreadDetach", sce_detach),
    f("scePthreadExit", exit),
    f("scePthreadYield", sce_yield),
    f("scePthreadGetthreadid", sce_getthreadid),
    f("scePthreadGetname", sce_getname),
    f("scePthreadRename", set_name),
    f("scePthreadSetaffinity", sce_ok),
    f("scePthreadSetprio", sce_ok),
    f("scePthreadAttrInit", sce_attr_init),
    f("scePthreadAttrDestroy", sce_attr_destroy),
    f("scePthreadAttrSetdetachstate", sce_attr_setdetachstate),
    f("scePthreadAttrGetdetachstate", sce_attr_getdetachstate),
    f("scePthreadAttrGetstacksize", sce_attr_getstacksize),
    f("scePthreadAttrSetstacksize", sce_attr_setstacksize),
    f("scePthreadAttrSetaffinity", sce_ok),
    f("scePthreadAttrGetaffinity", sce_attr_getaffinity),
    f("scePthreadAttrSetschedpolicy", sce_ok),
    f("scePthreadAttrSetschedparam", sce_ok),
    f("scePthreadAttrGetschedparam", sce_attr_getschedparam),
    f("scePthreadAttrSetinheritsched", sce_ok),
    f("scePthreadAttrGet", sce_attr_get),
];
