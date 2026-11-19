//! libkernel's pthread synchronisation: mutexes, condition variables,
//! read-write locks, thread-specific keys and `pthread_once`.
//!
//! As on FreeBSD, `pthread_mutex_t`, `pthread_cond_t`, `pthread_rwlock_t`
//! and `pthread_mutexattr_t` are pointer-sized words, 0 meaning "statically
//! initialised". We keep our own handle in that word, created on first use.
//! Constants and the default mutex type follow FreeBSD's `pthread.h`;
//! whether Orbis keeps them is unverified. These functions return an error
//! number instead of setting errno.

use std::collections::HashMap;

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use x86::cpu::reg::RSP;

use crate::errno::{EBUSY, EDEADLK, EINVAL, EPERM, ETIMEDOUT};
use crate::{CallError, Env, Function, INTERNAL, Outcome};

/// How often a blocked call checks whether the process has exited.
pub(crate) const POLL: Duration = Duration::from_millis(20);

const PTHREAD_MUTEX_ERRORCHECK: u64 = 1;
const PTHREAD_MUTEX_RECURSIVE: u64 = 2;

/// The internal function `pthread_once` returns through.
pub const ONCE_DONE: &str = "once_done";

const ONCE_NEEDS_INIT: u64 = 0;
const ONCE_DONE_STATE: u64 = 1;
const ONCE_IN_PROGRESS: u64 = 2;

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Default)]
struct MutexState {
    owner: Option<u64>,
    count: u32,
}

struct HostMutex {
    kind: u64,
    state: Mutex<MutexState>,
    released: Condvar,
}

#[derive(Default)]
struct HostCond {
    generation: Mutex<u64>,
    signalled: Condvar,
}

#[derive(Default)]
struct RwState {
    readers: u32,
    writer: Option<u64>,
}

#[derive(Default)]
struct HostRwLock {
    state: Mutex<RwState>,
    changed: Condvar,
}

/// The kernel objects behind guest handles.
#[derive(Default)]
pub struct Objects {
    mutexes: Mutex<HashMap<u64, Arc<HostMutex>>>,
    conds: Mutex<HashMap<u64, Arc<HostCond>>>,
    rwlocks: Mutex<HashMap<u64, Arc<HostRwLock>>>,
    mutex_attrs: Mutex<HashMap<u64, u64>>,
    keys: Mutex<Vec<bool>>,
    specific: Mutex<HashMap<(u64, u32), u64>>,
    once: Mutex<()>,
    once_done: Condvar,
}

/// A new handle for a pthread object: the address of a small zeroed block
/// of guest memory, since guest code (Sony's libc) reads through
/// `pthread_mutex_t` and its siblings as pointers.
pub(crate) fn new_handle(env: &Env) -> Result<u64, CallError> {
    let at = env
        .kernel
        .heap
        .alloc(env.mem, HANDLE_SIZE)
        .ok_or(CallError::Abort("out of memory for pthread objects"))?;
    env.mem.write(at, &[0; HANDLE_SIZE as usize])?;
    Ok(at)
}

/// The guest memory behind each handle.
const HANDLE_SIZE: u64 = 64;

pub static FUNCTIONS: &[Function] = &[
    f("pthread_mutexattr_init", mutexattr_init),
    f("pthread_mutexattr_destroy", mutexattr_destroy),
    f("pthread_mutexattr_settype", mutexattr_settype),
    f("pthread_mutex_init", mutex_init),
    f("pthread_mutex_destroy", mutex_destroy),
    f("pthread_mutex_lock", mutex_lock),
    f("pthread_mutex_trylock", mutex_trylock),
    f("pthread_mutex_unlock", mutex_unlock),
    f("pthread_condattr_init", condattr),
    f("pthread_condattr_destroy", condattr),
    f("pthread_cond_init", cond_init),
    f("pthread_cond_destroy", cond_destroy),
    f("pthread_cond_wait", cond_wait),
    f("pthread_cond_timedwait", cond_timedwait),
    f("pthread_cond_signal", cond_signal),
    f("pthread_cond_broadcast", cond_signal),
    f("pthread_rwlock_init", rwlock_init),
    f("pthread_rwlock_destroy", rwlock_destroy),
    f("pthread_rwlock_rdlock", rwlock_rdlock),
    f("pthread_rwlock_wrlock", rwlock_wrlock),
    f("pthread_rwlock_tryrdlock", rwlock_tryrdlock),
    f("pthread_rwlock_trywrlock", rwlock_trywrlock),
    f("pthread_rwlock_unlock", rwlock_unlock),
    f("pthread_key_create", key_create),
    f("pthread_key_delete", key_delete),
    f("pthread_getspecific", getspecific),
    f("pthread_setspecific", setspecific),
    f("pthread_once", once),
    f("pthread_self", self_),
    f("pthread_equal", equal),
    Function {
        library: INTERNAL,
        name: ONCE_DONE,
        func: once_done,
    },
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

pub(crate) fn ret(v: i32) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v as u32 as u64))
}

/// The handle stored at `at`, made when the word is 0. Another thread may
/// see the handle before `make` runs, so `make` must not replace its object.
fn handle(env: &Env, at: u64, make: impl FnOnce(u64)) -> Result<u64, CallError> {
    let current = env.mem.read_u64(at)?;
    if current != 0 {
        return Ok(current);
    }
    let id = new_handle(env)?;
    let old = env.mem.atomic_update(at, 8, |v| (v == 0).then_some(id))?;
    if old != 0 {
        return Ok(old);
    }
    make(id);
    Ok(id)
}

fn mutex(env: &Env, at: u64) -> Result<Arc<HostMutex>, CallError> {
    let objects = &env.kernel.objects;
    let id = handle(env, at, |id| {
        lock(&objects.mutexes)
            .entry(id)
            .or_insert_with(|| Arc::new(new_mutex(PTHREAD_MUTEX_ERRORCHECK)));
    })?;
    let mut map = lock(&objects.mutexes);
    Ok(map
        .entry(id)
        .or_insert_with(|| Arc::new(new_mutex(PTHREAD_MUTEX_ERRORCHECK)))
        .clone())
}

fn new_mutex(kind: u64) -> HostMutex {
    HostMutex {
        kind,
        state: Mutex::new(MutexState::default()),
        released: Condvar::new(),
    }
}

fn mutexattr_init(env: &mut Env) -> Result<Outcome, CallError> {
    let id = new_handle(env)?;
    lock(&env.kernel.objects.mutex_attrs).insert(id, PTHREAD_MUTEX_ERRORCHECK);
    env.mem.write_u64(env.arg(0), id)?;
    ret(0)
}

/// Condition variable attributes carry nothing we use (clock and
/// process-shared are ignored); the attribute word is zeroed.
fn condattr(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write_u64(env.arg(0), 0)?;
    ret(0)
}

fn mutexattr_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    lock(&env.kernel.objects.mutex_attrs).remove(&id);
    env.mem.write_u64(env.arg(0), 0)?;
    ret(0)
}

fn mutexattr_settype(env: &mut Env) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    let kind = env.arg(1) & 0xffff_ffff;
    match lock(&env.kernel.objects.mutex_attrs).get_mut(&id) {
        Some(k) if (1..=4).contains(&kind) => {
            *k = kind;
            ret(0)
        }
        _ => ret(EINVAL),
    }
}

fn mutex_init(env: &mut Env) -> Result<Outcome, CallError> {
    let attr = env.arg(1);
    let kind = if attr == 0 {
        PTHREAD_MUTEX_ERRORCHECK
    } else {
        let id = env.mem.read_u64(attr)?;
        match lock(&env.kernel.objects.mutex_attrs).get(&id) {
            Some(&k) => k,
            None => return ret(EINVAL),
        }
    };
    let id = new_handle(env)?;
    lock(&env.kernel.objects.mutexes).insert(id, Arc::new(new_mutex(kind)));
    env.mem.write_u64(env.arg(0), id)?;
    ret(0)
}

fn mutex_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    lock(&env.kernel.objects.mutexes).remove(&id);
    env.mem.write_u64(env.arg(0), 0)?;
    ret(0)
}

/// Locks `m` for `me`, blocking unless `try_only`; returns the error number.
fn acquire(env: &Env, m: &HostMutex, me: u64, try_only: bool) -> i32 {
    let mut s = lock(&m.state);
    loop {
        match s.owner {
            None => {
                s.owner = Some(me);
                s.count = 1;
                return 0;
            }
            Some(o) if o == me => {
                return if m.kind == PTHREAD_MUTEX_RECURSIVE {
                    s.count += 1;
                    0
                } else if try_only {
                    EBUSY
                } else {
                    EDEADLK
                };
            }
            Some(_) if try_only => return EBUSY,
            Some(_) => {
                if env.kernel.has_exited() {
                    return 0;
                }
                s = m
                    .released
                    .wait_timeout(s, POLL)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        }
    }
}

/// Unlocks `m` held by `me`; returns the error number.
fn release(m: &HostMutex, me: u64) -> i32 {
    let mut s = lock(&m.state);
    if s.owner != Some(me) {
        return EPERM;
    }
    s.count -= 1;
    if s.count == 0 {
        s.owner = None;
        m.released.notify_one();
    }
    0
}

fn mutex_lock(env: &mut Env) -> Result<Outcome, CallError> {
    let m = mutex(env, env.arg(0))?;
    ret(acquire(env, &m, env.thread.tcb, false))
}

fn mutex_trylock(env: &mut Env) -> Result<Outcome, CallError> {
    let m = mutex(env, env.arg(0))?;
    ret(acquire(env, &m, env.thread.tcb, true))
}

fn mutex_unlock(env: &mut Env) -> Result<Outcome, CallError> {
    let m = mutex(env, env.arg(0))?;
    ret(release(&m, env.thread.tcb))
}

fn cond(env: &Env, at: u64) -> Result<Arc<HostCond>, CallError> {
    let objects = &env.kernel.objects;
    let id = handle(env, at, |id| {
        lock(&objects.conds).entry(id).or_default();
    })?;
    Ok(lock(&objects.conds).entry(id).or_default().clone())
}

fn cond_init(env: &mut Env) -> Result<Outcome, CallError> {
    let id = new_handle(env)?;
    lock(&env.kernel.objects.conds).insert(id, Arc::default());
    env.mem.write_u64(env.arg(0), id)?;
    ret(0)
}

fn cond_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    lock(&env.kernel.objects.conds).remove(&id);
    env.mem.write_u64(env.arg(0), 0)?;
    ret(0)
}

/// Waits on the condition at `arg(0)`, holding the mutex at `arg(1)`, until
/// signalled or `deadline`; spurious wake-ups are allowed by POSIX.
fn wait(env: &mut Env, deadline: Option<Instant>) -> Result<Outcome, CallError> {
    let c = cond(env, env.arg(0))?;
    let m = mutex(env, env.arg(1))?;
    let me = env.thread.tcb;
    let mut generation = lock(&c.generation);
    let start = *generation;
    let held = {
        let mut s = lock(&m.state);
        if s.owner != Some(me) {
            return ret(EPERM);
        }
        let held = s.count;
        s.owner = None;
        s.count = 0;
        m.released.notify_one();
        held
    };
    let mut timed_out = false;
    while *generation == start && !env.kernel.has_exited() {
        let wait = match deadline {
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) => left.min(POLL),
                None => {
                    timed_out = true;
                    break;
                }
            },
            None => POLL,
        };
        generation = c
            .signalled
            .wait_timeout(generation, wait)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    drop(generation);
    acquire(env, &m, me, false);
    lock(&m.state).count = held;
    ret(if timed_out { ETIMEDOUT } else { 0 })
}

fn cond_wait(env: &mut Env) -> Result<Outcome, CallError> {
    wait(env, None)
}

/// `pthread_cond_timedwait(cond, mutex, abstime)`: `abstime` is a
/// CLOCK_REALTIME `struct timespec { i64 sec; i64 nsec; }`.
fn cond_timedwait(env: &mut Env) -> Result<Outcome, CallError> {
    let deadline = realtime_deadline(env, env.arg(2))?;
    wait(env, Some(deadline))
}

/// The instant a CLOCK_REALTIME `struct timespec { i64 sec; i64 nsec; }`
/// at `ts` names.
pub(crate) fn realtime_deadline(env: &Env, ts: u64) -> Result<Instant, CallError> {
    let (sec, nsec) = (env.mem.read_u64(ts)?, env.mem.read_u64(ts + 8)?);
    let target = Duration::new(sec, nsec.min(999_999_999) as u32);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(Instant::now() + target.saturating_sub(now))
}

fn cond_signal(env: &mut Env) -> Result<Outcome, CallError> {
    let c = cond(env, env.arg(0))?;
    *lock(&c.generation) += 1;
    c.signalled.notify_all();
    ret(0)
}

fn rwlock(env: &Env, at: u64) -> Result<Arc<HostRwLock>, CallError> {
    let objects = &env.kernel.objects;
    let id = handle(env, at, |id| {
        lock(&objects.rwlocks).entry(id).or_default();
    })?;
    Ok(lock(&objects.rwlocks).entry(id).or_default().clone())
}

fn rwlock_init(env: &mut Env) -> Result<Outcome, CallError> {
    let id = new_handle(env)?;
    lock(&env.kernel.objects.rwlocks).insert(id, Arc::default());
    env.mem.write_u64(env.arg(0), id)?;
    ret(0)
}

fn rwlock_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    let id = env.mem.read_u64(env.arg(0))?;
    lock(&env.kernel.objects.rwlocks).remove(&id);
    env.mem.write_u64(env.arg(0), 0)?;
    ret(0)
}

/// Takes the lock for reading or writing; without `wait`, fails with
/// EBUSY instead of waiting.
fn rwlock_take(env: &mut Env, write: bool, wait: bool) -> Result<Outcome, CallError> {
    let l = rwlock(env, env.arg(0))?;
    let me = env.thread.tcb;
    let mut s = lock(&l.state);
    loop {
        if s.writer == Some(me) {
            return ret(if wait { EDEADLK } else { EBUSY });
        }
        if s.writer.is_none() && (!write || s.readers == 0) {
            if write {
                s.writer = Some(me);
            } else {
                s.readers += 1;
            }
            return ret(0);
        }
        if !wait {
            return ret(EBUSY);
        }
        if env.kernel.has_exited() {
            return ret(0);
        }
        s = l
            .changed
            .wait_timeout(s, POLL)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}

fn rwlock_rdlock(env: &mut Env) -> Result<Outcome, CallError> {
    rwlock_take(env, false, true)
}

fn rwlock_wrlock(env: &mut Env) -> Result<Outcome, CallError> {
    rwlock_take(env, true, true)
}

fn rwlock_tryrdlock(env: &mut Env) -> Result<Outcome, CallError> {
    rwlock_take(env, false, false)
}

fn rwlock_trywrlock(env: &mut Env) -> Result<Outcome, CallError> {
    rwlock_take(env, true, false)
}

fn rwlock_unlock(env: &mut Env) -> Result<Outcome, CallError> {
    let l = rwlock(env, env.arg(0))?;
    let mut s = lock(&l.state);
    if s.writer == Some(env.thread.tcb) {
        s.writer = None;
    } else if s.readers > 0 {
        s.readers -= 1;
    } else {
        return ret(EPERM);
    }
    l.changed.notify_all();
    ret(0)
}

/// `pthread_key_create(key, destructor)`: `pthread_key_t` is an `int`.
/// Destructors are not run yet.
fn key_create(env: &mut Env) -> Result<Outcome, CallError> {
    let mut keys = lock(&env.kernel.objects.keys);
    let index = match keys.iter().position(|used| !used) {
        Some(i) => i,
        None => {
            keys.push(false);
            keys.len() - 1
        }
    };
    keys[index] = true;
    env.mem.write(env.arg(0), &(index as u32).to_le_bytes())?;
    ret(0)
}

fn key_delete(env: &mut Env) -> Result<Outcome, CallError> {
    let key = env.arg(0) as u32;
    match lock(&env.kernel.objects.keys).get_mut(key as usize) {
        Some(used) if *used => {
            *used = false;
            lock(&env.kernel.objects.specific).retain(|k, _| k.1 != key);
            ret(0)
        }
        _ => ret(EINVAL),
    }
}

fn getspecific(env: &mut Env) -> Result<Outcome, CallError> {
    let key = (env.thread.tcb, env.arg(0) as u32);
    let v = lock(&env.kernel.objects.specific)
        .get(&key)
        .copied()
        .unwrap_or(0);
    Ok(Outcome::Return(v))
}

fn setspecific(env: &mut Env) -> Result<Outcome, CallError> {
    let key = env.arg(0) as u32;
    let known = lock(&env.kernel.objects.keys)
        .get(key as usize)
        .copied()
        .unwrap_or(false);
    if !known {
        return ret(EINVAL);
    }
    lock(&env.kernel.objects.specific).insert((env.thread.tcb, key), env.arg(1));
    ret(0)
}

/// `pthread_once(once, routine)`: the first caller runs `routine` as a guest call
/// returning through [`ONCE_DONE`]; the state is the first `int` of `pthread_once_t`.
fn once(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, routine) = (env.arg(0), env.arg(1));
    loop {
        let old = env.mem.atomic_update(at, 4, |s| {
            (s == ONCE_NEEDS_INIT).then_some(ONCE_IN_PROGRESS)
        })?;
        match old {
            ONCE_NEEDS_INIT => break,
            ONCE_DONE_STATE => return ret(0),
            _ => {
                let objects = &env.kernel.objects;
                let guard = lock(&objects.once);
                let _ = objects.once_done.wait_timeout(guard, POLL);
                if env.kernel.has_exited() {
                    return ret(0);
                }
            }
        }
    }
    let sp = env.cpu.regs[RSP] - 16;
    env.mem.write_u64(sp, env.kernel.hle.internal(ONCE_DONE))?;
    env.mem.write_u64(sp + 8, at)?;
    env.cpu.regs[RSP] = sp;
    env.cpu.rip = routine;
    Ok(Outcome::Resume)
}

/// Where a `pthread_once` routine returns: the once word sits on the stack
/// above the original return address.
fn once_done(env: &mut Env) -> Result<Outcome, CallError> {
    let sp = env.cpu.regs[RSP];
    let at = env.mem.read_u64(sp)?;
    env.cpu.regs[RSP] = sp + 8;
    env.mem.write(at, &(ONCE_DONE_STATE as u32).to_le_bytes())?;
    let objects = &env.kernel.objects;
    drop(lock(&objects.once));
    objects.once_done.notify_all();
    ret(0)
}

/// `pthread_self()`: a thread is identified by its thread pointer.
fn self_(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(env.thread.tcb))
}

fn equal(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return((env.arg(0) == env.arg(1)) as u64))
}

/// Sony's error convention for `scePthread*`: 0, or `0x80020000` + errno.
pub(crate) fn sce(out: Result<Outcome, CallError>) -> Result<Outcome, CallError> {
    match out? {
        Outcome::Return(v) if v as u32 != 0 => {
            Ok(Outcome::Return(crate::dmem::sce_error(v as i32)))
        }
        other => Ok(other),
    }
}

macro_rules! sce_wrappers {
    ($($sce:ident => $posix:ident),* $(,)?) => {
        $(fn $sce(env: &mut Env) -> Result<Outcome, CallError> {
            sce($posix(env))
        })*
    };
}

sce_wrappers! {
    sce_mutexattr_init => mutexattr_init,
    sce_mutexattr_destroy => mutexattr_destroy,
    sce_mutexattr_settype => mutexattr_settype,
    sce_mutex_init => mutex_init,
    sce_mutex_destroy => mutex_destroy,
    sce_mutex_lock => mutex_lock,
    sce_mutex_trylock => mutex_trylock,
    sce_mutex_unlock => mutex_unlock,
    sce_cond_init => cond_init,
    sce_cond_destroy => cond_destroy,
    sce_cond_wait => cond_wait,
    sce_cond_signal => cond_signal,
    sce_rwlock_init => rwlock_init,
    sce_rwlock_destroy => rwlock_destroy,
    sce_rwlock_rdlock => rwlock_rdlock,
    sce_rwlock_wrlock => rwlock_wrlock,
    sce_rwlock_tryrdlock => rwlock_tryrdlock,
    sce_rwlock_trywrlock => rwlock_trywrlock,
    sce_rwlock_unlock => rwlock_unlock,
    sce_key_create => key_create,
    sce_key_delete => key_delete,
    sce_setspecific => setspecific,
    sce_once => once,
}

/// `scePthreadCondTimedwait(cond, mutex, usec)`: a relative timeout.
fn sce_cond_timedwait(env: &mut Env) -> Result<Outcome, CallError> {
    let usec = env.arg(2) as u32 as u64;
    sce(wait(
        env,
        Some(Instant::now() + Duration::from_micros(usec)),
    ))
}

/// `scePthreadMutexTimedlock(mutex, usec)`.
fn sce_mutex_timedlock(env: &mut Env) -> Result<Outcome, CallError> {
    let m = mutex(env, env.arg(0))?;
    let deadline = Instant::now() + Duration::from_micros(env.arg(1) as u32 as u64);
    loop {
        let r = acquire(env, &m, env.thread.tcb, true);
        if r != EBUSY || Instant::now() >= deadline || env.kernel.has_exited() {
            return sce(ret(if r == EBUSY { ETIMEDOUT } else { r }));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// `scePthreadCondattrInit`/`Destroy`: condition attributes are not kept.
fn sce_ok(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

pub static SCE_FUNCTIONS: &[Function] = &[
    f("scePthreadMutexattrInit", sce_mutexattr_init),
    f("scePthreadMutexattrDestroy", sce_mutexattr_destroy),
    f("scePthreadMutexattrSettype", sce_mutexattr_settype),
    f("scePthreadMutexInit", sce_mutex_init),
    f("scePthreadMutexDestroy", sce_mutex_destroy),
    f("scePthreadMutexLock", sce_mutex_lock),
    f("scePthreadMutexTrylock", sce_mutex_trylock),
    f("scePthreadMutexTimedlock", sce_mutex_timedlock),
    f("scePthreadMutexUnlock", sce_mutex_unlock),
    f("scePthreadCondattrInit", sce_ok),
    f("scePthreadCondattrDestroy", sce_ok),
    f("scePthreadCondInit", sce_cond_init),
    f("scePthreadCondDestroy", sce_cond_destroy),
    f("scePthreadCondWait", sce_cond_wait),
    f("scePthreadCondTimedwait", sce_cond_timedwait),
    f("scePthreadCondSignal", sce_cond_signal),
    f("scePthreadCondSignalto", sce_cond_signal),
    f("scePthreadCondBroadcast", sce_cond_signal),
    f("scePthreadRwlockInit", sce_rwlock_init),
    f("scePthreadRwlockDestroy", sce_rwlock_destroy),
    f("scePthreadRwlockRdlock", sce_rwlock_rdlock),
    f("scePthreadRwlockWrlock", sce_rwlock_wrlock),
    f("scePthreadRwlockTryrdlock", sce_rwlock_tryrdlock),
    f("scePthreadRwlockTrywrlock", sce_rwlock_trywrlock),
    f("scePthreadRwlockUnlock", sce_rwlock_unlock),
    f("scePthreadKeyCreate", sce_key_create),
    f("scePthreadKeyDelete", sce_key_delete),
    f("scePthreadSetspecific", sce_setspecific),
    f("scePthreadGetspecific", getspecific),
    f("scePthreadOnce", sce_once),
    f("scePthreadSelf", self_),
    f("scePthreadEqual", equal),
];
