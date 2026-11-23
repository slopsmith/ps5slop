//! libkernel event flags (`sceKernel*EventFlag`): a 64-bit pattern that
//! threads set, clear and wait on. Prototypes and constants follow the
//! OpenOrbis headers; the handle is ours. Single-waiter flags are not
//! enforced.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::dmem::sce_error;
use crate::errno::{EBUSY, ECANCELED, EINVAL, ESRCH, ETIMEDOUT};
use crate::pthread::{POLL, lock};
use crate::{CallError, Env, Function, Outcome};

const WAITMODE_AND: u64 = 0x01;
const WAITMODE_OR: u64 = 0x02;
const WAITMODE_CLEAR_ALL: u64 = 0x10;
const WAITMODE_CLEAR_PAT: u64 = 0x20;

struct Flag {
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Default)]
struct State {
    pattern: u64,
    /// Bumped by cancel and delete so that waiters give up.
    cancels: u64,
    waiters: u32,
}

#[derive(Default)]
pub struct EventFlags {
    flags: Mutex<HashMap<u64, Arc<Flag>>>,
}

pub static FUNCTIONS: &[Function] = &[
    f("sceKernelCreateEventFlag", create),
    f("sceKernelDeleteEventFlag", delete),
    f("sceKernelSetEventFlag", set),
    f("sceKernelClearEventFlag", clear),
    f("sceKernelWaitEventFlag", wait),
    f("sceKernelPollEventFlag", poll),
    f("sceKernelCancelEventFlag", cancel),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

fn ret(errno: i32) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(if errno == 0 {
        0
    } else {
        sce_error(errno)
    }))
}

fn flag(env: &Env) -> Option<Arc<Flag>> {
    lock(&env.kernel.evfs.flags).get(&env.arg(0)).cloned()
}

/// Whether `pattern` satisfies a wait for `bits` in `mode`; if so, applies
/// the mode's clearing and returns the pattern as it was.
fn try_take(pattern: &mut u64, bits: u64, mode: u64) -> Option<u64> {
    let hit = if mode & WAITMODE_AND != 0 {
        *pattern & bits == bits
    } else {
        *pattern & bits != 0
    };
    if !hit {
        return None;
    }
    let seen = *pattern;
    if mode & WAITMODE_CLEAR_ALL != 0 {
        *pattern = 0;
    } else if mode & WAITMODE_CLEAR_PAT != 0 {
        *pattern &= !bits;
    }
    Some(seen)
}

fn valid_mode(bits: u64, mode: u64) -> bool {
    let wait = mode & (WAITMODE_AND | WAITMODE_OR);
    let clear = mode & (WAITMODE_CLEAR_ALL | WAITMODE_CLEAR_PAT);
    bits != 0
        && (wait == WAITMODE_AND || wait == WAITMODE_OR)
        && clear != (WAITMODE_CLEAR_ALL | WAITMODE_CLEAR_PAT)
}

/// `sceKernelCreateEventFlag(&flag, name, attr, init, opt)`.
fn create(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(0) == 0 {
        return ret(EINVAL);
    }
    let id = crate::pthread::new_handle(env)?;
    let flag = Flag {
        state: Mutex::new(State {
            pattern: env.arg(3),
            ..State::default()
        }),
        changed: Condvar::new(),
    };
    lock(&env.kernel.evfs.flags).insert(id, Arc::new(flag));
    env.mem.write_u64(env.arg(0), id)?;
    ret(0)
}

fn delete(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(f) = lock(&env.kernel.evfs.flags).remove(&env.arg(0)) else {
        return ret(ESRCH);
    };
    lock(&f.state).cancels += 1;
    f.changed.notify_all();
    ret(0)
}

/// `sceKernelSetEventFlag(flag, bits)`: ORs `bits` into the pattern.
fn set(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(f) = flag(env) else {
        return ret(ESRCH);
    };
    lock(&f.state).pattern |= env.arg(1);
    f.changed.notify_all();
    ret(0)
}

/// `sceKernelClearEventFlag(flag, bits)`: ANDs the pattern with `bits`.
fn clear(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(f) = flag(env) else {
        return ret(ESRCH);
    };
    lock(&f.state).pattern &= env.arg(1);
    ret(0)
}

/// `sceKernelWaitEventFlag(flag, bits, mode, *result, *timeout)`: `timeout`
/// points to microseconds (and gets what is left), or is null.
fn wait(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(f) = flag(env) else {
        return ret(ESRCH);
    };
    let (bits, mode, result, timeout) = (env.arg(1), env.arg(2), env.arg(3), env.arg(4));
    if !valid_mode(bits, mode) {
        return ret(EINVAL);
    }
    let deadline = match timeout {
        0 => None,
        at => {
            let mut b = [0u8; 4];
            env.mem.read(at, &mut b)?;
            Some(Instant::now() + Duration::from_micros(u32::from_le_bytes(b) as u64))
        }
    };
    let mut st = lock(&f.state);
    let cancels = st.cancels;
    st.waiters += 1;
    let outcome = loop {
        if st.cancels != cancels {
            break Err(ECANCELED);
        }
        if let Some(seen) = try_take(&mut st.pattern, bits, mode) {
            break Ok(seen);
        }
        if env.kernel.has_exited() {
            break Err(ECANCELED);
        }
        if crate::exception::interrupted(env) {
            st.waiters -= 1;
            drop(st);
            if let Some(d) = deadline {
                let left = d.saturating_duration_since(Instant::now()).as_micros() as u32;
                env.mem.write(timeout, &left.to_le_bytes())?;
            }
            return Ok(Outcome::Resume);
        }
        let step = match deadline {
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) => crate::timing::step(left).min(POLL),
                None => break Err(ETIMEDOUT),
            },
            None => POLL,
        };
        jit::blocking();
        st = f
            .changed
            .wait_timeout(st, step)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    };
    st.waiters -= 1;
    let pattern = st.pattern;
    drop(st);
    if result != 0 {
        env.mem
            .write_u64(result, *outcome.as_ref().unwrap_or(&pattern))?;
    }
    if let Some(d) = deadline {
        let left = d.saturating_duration_since(Instant::now()).as_micros() as u32;
        env.mem.write(timeout, &left.to_le_bytes())?;
    }
    ret(outcome.err().unwrap_or(0))
}

/// `sceKernelPollEventFlag(flag, bits, mode, *result)`.
fn poll(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(f) = flag(env) else {
        return ret(ESRCH);
    };
    let (bits, mode, result) = (env.arg(1), env.arg(2), env.arg(3));
    if !valid_mode(bits, mode) {
        return ret(EINVAL);
    }
    let mut st = lock(&f.state);
    let taken = try_take(&mut st.pattern, bits, mode);
    let pattern = st.pattern;
    drop(st);
    if result != 0 {
        env.mem.write_u64(result, taken.unwrap_or(pattern))?;
    }
    ret(if taken.is_some() { 0 } else { EBUSY })
}

/// `sceKernelCancelEventFlag(flag, pattern, *waiters)`: sets the pattern
/// and makes every waiter give up with `ECANCELED`.
fn cancel(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(f) = flag(env) else {
        return ret(ESRCH);
    };
    let mut st = lock(&f.state);
    st.pattern = env.arg(1);
    st.cancels += 1;
    let waiters = st.waiters;
    drop(st);
    f.changed.notify_all();
    if env.arg(2) != 0 {
        env.mem.write(env.arg(2), &waiters.to_le_bytes())?;
    }
    ret(0)
}

impl EventFlags {
    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        let flags: HashMap<u64, (u64, u64)> = lock(&self.flags)
            .iter()
            .map(|(&id, f)| {
                let st = lock(&f.state);
                (id, (st.pattern, st.cancels))
            })
            .collect();
        e.put(&flags);
    }

    pub(crate) fn load(&self, d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
        let flags: HashMap<u64, (u64, u64)> = d.get()?;
        *lock(&self.flags) = flags
            .into_iter()
            .map(|(id, (pattern, cancels))| {
                let f = Flag {
                    state: Mutex::new(State {
                        pattern,
                        cancels,
                        waiters: 0,
                    }),
                    changed: Condvar::new(),
                };
                (id, Arc::new(f))
            })
            .collect();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn and_waits_need_every_bit_and_or_waits_need_one() {
        let mut p = 0b0110;
        assert_eq!(try_take(&mut p, 0b0111, WAITMODE_AND), None);
        assert_eq!(try_take(&mut p, 0b0011, WAITMODE_OR), Some(0b0110));
        assert_eq!(p, 0b0110);
    }

    #[test]
    fn clear_modes_clear_everything_or_only_the_waited_bits() {
        let mut p = 0b1110;
        assert_eq!(
            try_take(&mut p, 0b0110, WAITMODE_AND | WAITMODE_CLEAR_PAT),
            Some(0b1110)
        );
        assert_eq!(p, 0b1000);
        assert_eq!(
            try_take(&mut p, 0b1000, WAITMODE_OR | WAITMODE_CLEAR_ALL),
            Some(0b1000)
        );
        assert_eq!(p, 0);
    }

    #[test]
    fn modes_need_exactly_one_wait_kind_and_some_bits() {
        assert!(valid_mode(1, WAITMODE_AND));
        assert!(!valid_mode(1, WAITMODE_AND | WAITMODE_OR));
        assert!(!valid_mode(0, WAITMODE_OR));
        assert!(!valid_mode(
            1,
            WAITMODE_OR | WAITMODE_CLEAR_ALL | WAITMODE_CLEAR_PAT
        ));
    }
}
