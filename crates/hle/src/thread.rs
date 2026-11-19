//! Creating, joining and ending guest threads. Each guest thread runs on its
//! own host thread (see `docs/decisions.md`); `pthread_t` is the thread
//! pointer. Of the thread attributes, only the stack size and detach state
//! are used.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
    next_attr: AtomicU64,
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

/// `pthread_set_name_np(thread, name)`: names are not kept.
fn set_name(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
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
    let id = threads.next_attr.fetch_add(1, Ordering::Relaxed) + 1;
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
