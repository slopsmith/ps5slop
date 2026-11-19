//! POSIX semaphores (`sem_*` in libkernel). As with pthread objects, we keep
//! our own handle in the first word of the guest's `sem_t`. These functions
//! set errno and return -1 on failure.

use std::collections::HashMap;

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

/// A libkernel semaphore (`sceKernel*Sema`): waits take several counts at
/// once, and the count has a maximum.
struct SceSem {
    state: Mutex<SceState>,
    changed: Condvar,
    max: i64,
}

#[derive(Default)]
struct SceState {
    count: i64,
    /// Bumped by `sceKernelCancelSema` so that waiters give up.
    cancels: u64,
    waiters: u32,
}

#[derive(Default)]
pub struct Semaphores {
    sems: Mutex<HashMap<u64, Arc<Sem>>>,
    sce: Mutex<HashMap<u64, Arc<SceSem>>>,
}

pub static FUNCTIONS: &[Function] = &[
    f("sem_init", init),
    f("sem_destroy", destroy),
    f("sem_wait", wait),
    f("sem_trywait", trywait),
    f("sem_timedwait", timedwait),
    f("sem_post", post),
    f("sem_getvalue", getvalue),
    f("sceKernelCreateSema", sce_create),
    f("sceKernelDeleteSema", sce_delete),
    f("sceKernelWaitSema", sce_wait),
    f("sceKernelPollSema", sce_poll),
    f("sceKernelSignalSema", sce_signal),
    f("sceKernelCancelSema", sce_cancel),
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
    let id = crate::pthread::new_handle(env)?;
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

fn sce_ret(errno: i32) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(crate::dmem::sce_error(errno)))
}

fn sce_sem(env: &Env) -> Option<Arc<SceSem>> {
    lock(&env.kernel.sems.sce).get(&env.arg(0)).cloned()
}

/// `sceKernelCreateSema(&sem, name, attr, init, max, opt)`; the handle is
/// written to `sem`.
fn sce_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (init, max) = (env.arg(3) as i32 as i64, env.arg(4) as i32 as i64);
    if env.arg(0) == 0 || init < 0 || max <= 0 || init > max {
        return sce_ret(EINVAL);
    }
    let id = crate::pthread::new_handle(env)?;
    let s = SceSem {
        state: Mutex::new(SceState {
            count: init,
            ..Default::default()
        }),
        changed: Condvar::new(),
        max,
    };
    lock(&env.kernel.sems.sce).insert(id, Arc::new(s));
    env.mem.write_u64(env.arg(0), id)?;
    ok()
}

fn sce_delete(env: &mut Env) -> Result<Outcome, CallError> {
    match lock(&env.kernel.sems.sce).remove(&env.arg(0)) {
        Some(s) => {
            lock(&s.state).cancels += 1;
            s.changed.notify_all();
            ok()
        }
        None => sce_ret(crate::errno::ESRCH),
    }
}

/// `sceKernelWaitSema(sem, need, timeout)`: `timeout` points to
/// microseconds (and gets what is left), or is null to wait forever.
fn sce_wait(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(s) = sce_sem(env) else {
        return sce_ret(crate::errno::ESRCH);
    };
    let (need, timeout) = (env.arg(1) as i32 as i64, env.arg(2));
    if need <= 0 || need > s.max {
        return sce_ret(EINVAL);
    }
    let start = Instant::now();
    let deadline = match timeout {
        0 => None,
        at => Some(start + std::time::Duration::from_micros(read_u32(env, at)? as u64)),
    };
    let mut st = lock(&s.state);
    let cancels = st.cancels;
    st.waiters += 1;
    let result = loop {
        if st.cancels != cancels {
            break Err(crate::errno::ECANCELED);
        }
        if st.count >= need {
            st.count -= need;
            break Ok(());
        }
        if env.kernel.has_exited() {
            break Ok(());
        }
        let wait = match deadline {
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) => left.min(POLL),
                None => break Err(ETIMEDOUT),
            },
            None => POLL,
        };
        st = s.changed.wait_timeout(st, wait).unwrap_or_else(|e| e.into_inner()).0;
    };
    st.waiters -= 1;
    drop(st);
    if let Some(d) = deadline {
        let left = d.saturating_duration_since(Instant::now()).as_micros() as u32;
        env.mem.write(timeout, &left.to_le_bytes())?;
    }
    match result {
        Ok(()) => ok(),
        Err(e) => sce_ret(e),
    }
}

/// `sceKernelPollSema(sem, need)`: takes `need` counts without waiting.
fn sce_poll(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(s) = sce_sem(env) else {
        return sce_ret(crate::errno::ESRCH);
    };
    let need = env.arg(1) as i32 as i64;
    if need <= 0 || need > s.max {
        return sce_ret(EINVAL);
    }
    let mut st = lock(&s.state);
    if st.count < need {
        return sce_ret(crate::errno::EBUSY);
    }
    st.count -= need;
    ok()
}

/// `sceKernelSignalSema(sem, count)`.
fn sce_signal(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(s) = sce_sem(env) else {
        return sce_ret(crate::errno::ESRCH);
    };
    let add = env.arg(1) as i32 as i64;
    let mut st = lock(&s.state);
    if add < 0 || st.count + add > s.max {
        return sce_ret(EINVAL);
    }
    st.count += add;
    s.changed.notify_all();
    ok()
}

/// `sceKernelCancelSema(sem, count, &waiters)`: wakes every waiter with an
/// error and sets the count (`count` < 0 keeps the current one).
fn sce_cancel(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(s) = sce_sem(env) else {
        return sce_ret(crate::errno::ESRCH);
    };
    let count = env.arg(1) as i32 as i64;
    let mut st = lock(&s.state);
    if count > s.max {
        return sce_ret(EINVAL);
    }
    let waiters = st.waiters;
    st.cancels += 1;
    if count >= 0 {
        st.count = count;
    }
    s.changed.notify_all();
    drop(st);
    if env.arg(2) != 0 {
        env.mem.write(env.arg(2), &waiters.to_le_bytes())?;
    }
    ok()
}

fn read_u32(env: &Env, at: u64) -> Result<u32, CallError> {
    let mut b = [0u8; 4];
    env.mem.read(at, &mut b)?;
    Ok(u32::from_le_bytes(b))
}
