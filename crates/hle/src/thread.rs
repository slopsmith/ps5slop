//! Creating, joining and ending guest threads. Each guest thread runs on its
//! own host thread (see `docs/decisions.md`); `pthread_t` is the thread
//! pointer. Thread attributes are not read yet: every thread gets the
//! default stack.

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

/// Threads made by `pthread_create`, by thread pointer.
#[derive(Default)]
pub struct Threads {
    records: Mutex<HashMap<u64, Arc<Record>>>,
}

pub static FUNCTIONS: &[Function] = &[
    f("pthread_create", create),
    f("pthread_join", join),
    f("pthread_detach", detach),
    f("pthread_exit", exit),
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

/// `pthread_create(thread, attr, start, arg)`.
fn create(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, start, arg) = (env.arg(0), env.arg(2), env.arg(3));
    let kernel = env.kernel;
    let (Ok(tcb), Ok(top)) = (
        kernel.new_tls(env.mem),
        kernel.new_stack(env.mem, THREAD_STACK),
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
    lock(&kernel.threads.records).insert(tcb, Arc::default());
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
