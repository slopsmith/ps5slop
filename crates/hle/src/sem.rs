//! POSIX semaphores (`sem_*` in libkernel). As with pthread objects, we keep
//! our own handle in the first word of the guest's `sem_t`. These functions
//! set errno and return -1 on failure.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::errno::{EAGAIN, EINVAL, ETIMEDOUT};
use crate::pthread::{POLL, lock, realtime_deadline};
use crate::{CallError, Env, Function, Outcome};

/// FreeBSD's `SEM_VALUE_MAX` (`INT_MAX`); unverified for Orbis.
const SEM_VALUE_MAX: u64 = i32::MAX as u64;

#[derive(Default)]
struct Sem {
    count: Mutex<u64>,
    posted: Condvar,
}

#[derive(Default)]
pub struct Semaphores {
    next: AtomicU64,
    sems: Mutex<HashMap<u64, Arc<Sem>>>,
}

pub static FUNCTIONS: &[Function] = &[
    f("sem_init", init),
    f("sem_destroy", destroy),
    f("sem_wait", wait),
    f("sem_trywait", trywait),
    f("sem_timedwait", timedwait),
    f("sem_post", post),
    f("sem_getvalue", getvalue),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

fn ok() -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

fn sem(env: &Env, at: u64) -> Result<Option<Arc<Sem>>, CallError> {
    let id = env.mem.read_u64(at)?;
    Ok(lock(&env.kernel.sems.sems).get(&id).cloned())
}

/// `sem_init(sem, pshared, value)`; `pshared` is ignored.
fn init(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, value) = (env.arg(0), env.arg(2) as u32 as u64);
    if value > SEM_VALUE_MAX {
        return env.fail(EINVAL);
    }
    let sems = &env.kernel.sems;
    let id = sems.next.fetch_add(1, Ordering::Relaxed) + 1;
    let s = Sem {
        count: Mutex::new(value),
        posted: Condvar::new(),
    };
    lock(&sems.sems).insert(id, Arc::new(s));
    env.mem.write_u64(at, id)?;
    ok()
}

fn destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    match lock(&env.kernel.sems.sems).remove(&id) {
        Some(_) => ok(),
        None => env.fail(EINVAL),
    }
}

/// Takes one count, waiting until `deadline` (forever if `None`) for a post.
fn take(env: &mut Env, deadline: Option<Instant>) -> Result<Outcome, CallError> {
    let Some(s) = sem(env, env.arg(0))? else {
        return env.fail(EINVAL);
    };
    let mut count = lock(&s.count);
    while *count == 0 {
        if env.kernel.has_exited() {
            return ok();
        }
        let wait = match deadline {
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) => left.min(POLL),
                None => {
                    drop(count);
                    return env.fail(ETIMEDOUT);
                }
            },
            None => POLL,
        };
        count = s
            .posted
            .wait_timeout(count, wait)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    *count -= 1;
    ok()
}

fn wait(env: &mut Env) -> Result<Outcome, CallError> {
    take(env, None)
}

/// `sem_timedwait(sem, abstime)`, with a CLOCK_REALTIME deadline.
fn timedwait(env: &mut Env) -> Result<Outcome, CallError> {
    let deadline = realtime_deadline(env, env.arg(1))?;
    take(env, Some(deadline))
}

fn trywait(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(s) = sem(env, env.arg(0))? else {
        return env.fail(EINVAL);
    };
    let mut count = lock(&s.count);
    if *count == 0 {
        drop(count);
        return env.fail(EAGAIN);
    }
    *count -= 1;
    ok()
}

fn post(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(s) = sem(env, env.arg(0))? else {
        return env.fail(EINVAL);
    };
    let mut count = lock(&s.count);
    if *count >= SEM_VALUE_MAX {
        drop(count);
        return env.fail(crate::errno::EOVERFLOW);
    }
    *count += 1;
    s.posted.notify_one();
    ok()
}

/// `sem_getvalue(sem, sval)`.
fn getvalue(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(s) = sem(env, env.arg(0))? else {
        return env.fail(EINVAL);
    };
    let value = *lock(&s.count) as u32;
    env.mem.write(env.arg(1), &value.to_le_bytes())?;
    ok()
}
