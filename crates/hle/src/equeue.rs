//! Event queues (`sceKernel*Equeue`), modelled on FreeBSD's kqueue, which
//! they wrap. Other libraries (VideoOut flips) add and trigger events. An
//! event is reported once per trigger, as with `EV_CLEAR`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::dmem::sce_error;
use crate::pthread::{POLL, lock};
use crate::{CallError, Env, Function, Kernel, Outcome, errno};

/// `struct kevent` on amd64: ident, filter, flags, fflags, data, udata.
const EVENT_SIZE: u64 = 32;

struct Registration {
    ident: u64,
    filter: i16,
    udata: u64,
    /// The `data` of a trigger not reported yet.
    pending: Option<u64>,
}

#[derive(Default)]
struct Queue {
    registrations: Mutex<Vec<Registration>>,
    triggered: Condvar,
}

#[derive(Default)]
pub struct Queues {
    next: AtomicU64,
    queues: Mutex<HashMap<u64, Arc<Queue>>>,
}

impl Queues {
    fn get(&self, id: u64) -> Option<Arc<Queue>> {
        lock(&self.queues).get(&id).cloned()
    }

    /// Registers event `(ident, filter)` on queue `id`, replacing an earlier
    /// registration of the same event. False if there is no such queue.
    fn add(&self, id: u64, ident: u64, filter: i16, udata: u64) -> bool {
        let Some(q) = self.get(id) else { return false };
        let mut regs = lock(&q.registrations);
        regs.retain(|r| (r.ident, r.filter) != (ident, filter));
        regs.push(Registration {
            ident,
            filter,
            udata,
            pending: None,
        });
        true
    }

    fn remove(&self, id: u64, ident: u64, filter: i16) -> bool {
        let Some(q) = self.get(id) else { return false };
        let mut regs = lock(&q.registrations);
        let before = regs.len();
        regs.retain(|r| (r.ident, r.filter) != (ident, filter));
        regs.len() != before
    }

    /// Triggers every registration of `(ident, filter)` on every queue.
    fn trigger(&self, ident: u64, filter: i16, data: u64) {
        let queues: Vec<_> = lock(&self.queues).values().cloned().collect();
        for q in queues {
            let mut hit = false;
            for r in lock(&q.registrations).iter_mut() {
                if (r.ident, r.filter) == (ident, filter) {
                    r.pending = Some(data);
                    hit = true;
                }
            }
            if hit {
                q.triggered.notify_all();
            }
        }
    }
}

pub static FUNCTIONS: &[Function] = &[
    f("sceKernelCreateEqueue", create),
    f("sceKernelDeleteEqueue", delete),
    f("sceKernelWaitEqueue", wait),
    f("sceKernelAddUserEvent", add_user_event),
    f("sceKernelAddReadEvent", add_read_event),
    f("sceKernelDeleteReadEvent", delete_read_event),
    f("sceKernelAddWriteEvent", add_write_event),
    f("sceKernelDeleteWriteEvent", delete_write_event),
    f("sceKernelAddUserEventEdge", add_user_event),
    f("sceKernelDeleteUserEvent", delete_user_event),
    f("sceKernelTriggerUserEvent", trigger_user_event),
    f("sceKernelGetEventId", event_id),
    f("sceKernelGetEventFilter", event_filter),
    f("sceKernelGetEventData", event_data),
    f("sceKernelGetEventUserData", event_user_data),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

/// `sceKernelCreateEqueue(*eq, name)`.
fn create(env: &mut Env) -> Result<Outcome, CallError> {
    let queues = &env.kernel.equeues;
    let id = queues.next.fetch_add(1, Ordering::Relaxed) + 1;
    lock(&queues.queues).insert(id, Arc::default());
    env.mem.write_u64(env.arg(0), id)?;
    ret(0)
}

fn delete(env: &mut Env) -> Result<Outcome, CallError> {
    match lock(&env.kernel.equeues.queues).remove(&env.arg(0)) {
        Some(_) => ret(0),
        None => ret(sce_error(errno::EBADF)),
    }
}

/// Takes up to `max` pending events from `q`, as `(ident, filter, data, udata)`.
fn take(regs: &mut [Registration], max: usize) -> Vec<(u64, i16, u64, u64)> {
    regs.iter_mut()
        .filter_map(|r| r.pending.take().map(|d| (r.ident, r.filter, d, r.udata)))
        .take(max)
        .collect()
}

/// `sceKernelWaitEqueue(eq, events, max, *count, *timeout_usec)`; a null
/// timeout waits forever.
fn wait(env: &mut Env) -> Result<Outcome, CallError> {
    let (id, out, max, count_at, timeout_at) = (
        env.arg(0),
        env.arg(1),
        env.arg(2) as i32,
        env.arg(3),
        env.arg(4),
    );
    if max < 1 {
        return ret(sce_error(errno::EINVAL));
    }
    let Some(q) = env.kernel.equeues.get(id) else {
        return ret(sce_error(errno::EBADF));
    };
    let deadline = if timeout_at == 0 {
        None
    } else {
        let mut b = [0u8; 4];
        env.mem.read(timeout_at, &mut b)?;
        Some(Instant::now() + Duration::from_micros(u32::from_le_bytes(b) as u64))
    };
    let mut regs = lock(&q.registrations);
    let events = loop {
        let events = take(&mut regs, max as usize);
        if !events.is_empty() || env.kernel.has_exited() {
            break events;
        }
        if crate::exception::interrupted(env) {
            return Ok(Outcome::Resume);
        }
        let wait = match deadline {
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) => crate::timing::step(left).min(POLL),
                None => break events,
            },
            None => POLL,
        };
        jit::blocking();
        regs = q
            .triggered
            .wait_timeout(regs, wait)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    };
    drop(regs);
    write_events(env, out, &events)?;
    if count_at != 0 {
        env.mem
            .write(count_at, &(events.len() as u32).to_le_bytes())?;
    }
    ret(if events.is_empty() {
        sce_error(errno::ETIMEDOUT)
    } else {
        0
    })
}

fn write_events(env: &Env, out: u64, events: &[(u64, i16, u64, u64)]) -> Result<(), CallError> {
    for (i, &(ident, filter, data, udata)) in events.iter().enumerate() {
        let mut e = [0u8; EVENT_SIZE as usize];
        e[0..8].copy_from_slice(&ident.to_le_bytes());
        e[8..10].copy_from_slice(&filter.to_le_bytes());
        e[16..24].copy_from_slice(&data.to_le_bytes());
        e[24..32].copy_from_slice(&udata.to_le_bytes());
        env.mem.write(out + i as u64 * EVENT_SIZE, &e)?;
    }
    Ok(())
}

/// FreeBSD's `EVFILT_READ` and `EVFILT_WRITE`.
const EVFILT_READ: i16 = -1;
const EVFILT_WRITE: i16 = -2;

/// `sceKernelAdd{Read,Write}Event(queue, fd, size, udata)`: registered, but
/// never triggered (no descriptor we provide becomes ready this way).
fn add_fd_event(env: &Env, filter: i16) -> Result<Outcome, CallError> {
    let ok = env
        .kernel
        .equeues
        .add(env.arg(0), env.arg(1) as u32 as u64, filter, env.arg(3));
    Ok(Outcome::Return(if ok {
        0
    } else {
        sce_error(errno::EBADF)
    }))
}

fn delete_fd_event(env: &Env, filter: i16) -> Result<Outcome, CallError> {
    let ok = env
        .kernel
        .equeues
        .remove(env.arg(0), env.arg(1) as u32 as u64, filter);
    Ok(Outcome::Return(if ok {
        0
    } else {
        sce_error(errno::ENOENT)
    }))
}

fn add_read_event(env: &mut Env) -> Result<Outcome, CallError> {
    add_fd_event(env, EVFILT_READ)
}

fn delete_read_event(env: &mut Env) -> Result<Outcome, CallError> {
    delete_fd_event(env, EVFILT_READ)
}

fn add_write_event(env: &mut Env) -> Result<Outcome, CallError> {
    add_fd_event(env, EVFILT_WRITE)
}

fn delete_write_event(env: &mut Env) -> Result<Outcome, CallError> {
    delete_fd_event(env, EVFILT_WRITE)
}

/// FreeBSD's `EVFILT_USER`.
const EVFILT_USER: i16 = -11;

/// `sceKernelAddUserEvent(queue, id)`.
fn add_user_event(env: &mut Env) -> Result<Outcome, CallError> {
    let ok = env
        .kernel
        .equeues
        .add(env.arg(0), env.arg(1) as u32 as u64, EVFILT_USER, 0);
    Ok(Outcome::Return(if ok {
        0
    } else {
        sce_error(errno::EBADF)
    }))
}

fn delete_user_event(env: &mut Env) -> Result<Outcome, CallError> {
    let ok = env
        .kernel
        .equeues
        .remove(env.arg(0), env.arg(1) as u32 as u64, EVFILT_USER);
    Ok(Outcome::Return(if ok {
        0
    } else {
        sce_error(errno::ENOENT)
    }))
}

/// `sceKernelTriggerUserEvent(queue, id, udata)`: `udata` replaces the
/// registration's.
fn trigger_user_event(env: &mut Env) -> Result<Outcome, CallError> {
    let (id, ident, udata) = (env.arg(0), env.arg(1) as u32 as u64, env.arg(2));
    let Some(q) = env.kernel.equeues.get(id) else {
        return Ok(Outcome::Return(sce_error(errno::EBADF)));
    };
    let mut hit = false;
    for r in lock(&q.registrations).iter_mut() {
        if (r.ident, r.filter) == (ident, EVFILT_USER) {
            r.udata = udata;
            r.pending = Some(0);
            hit = true;
        }
    }
    q.triggered.notify_all();
    Ok(Outcome::Return(if hit {
        0
    } else {
        sce_error(errno::ENOENT)
    }))
}

fn event_id(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(env.mem.read_u64(env.arg(0))?))
}

fn event_filter(env: &mut Env) -> Result<Outcome, CallError> {
    let mut b = [0u8; 2];
    env.mem.read(env.arg(0) + 8, &mut b)?;
    Ok(Outcome::Return(i16::from_le_bytes(b) as i64 as u64))
}

fn event_data(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(env.mem.read_u64(env.arg(0) + 16)?))
}

fn event_user_data(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(env.mem.read_u64(env.arg(0) + 24)?))
}

impl Kernel {
    /// Registers event `(ident, filter)` on queue `queue` for libraries
    /// outside libkernel; false if there is no such queue.
    pub fn add_event(&self, queue: u64, ident: u64, filter: i16, udata: u64) -> bool {
        self.equeues.add(queue, ident, filter, udata)
    }

    pub fn remove_event(&self, queue: u64, ident: u64, filter: i16) -> bool {
        self.equeues.remove(queue, ident, filter)
    }

    /// Triggers an event on every queue it is registered with (for
    /// libraries outside libkernel, such as VideoOut).
    pub fn trigger_event(&self, ident: u64, filter: i16, data: u64) {
        self.equeues.trigger(ident, filter, data);
    }
}

/// A registration as saved: ident, filter, user data and pending data.
type SavedRegistration = (u64, i16, u64, Option<u64>);

impl Queues {
    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        let queues: HashMap<u64, Vec<SavedRegistration>> = lock(&self.queues)
            .iter()
            .map(|(&id, q)| {
                let regs = lock(&q.registrations)
                    .iter()
                    .map(|r| (r.ident, r.filter, r.udata, r.pending))
                    .collect();
                (id, regs)
            })
            .collect();
        e.put(&self.next.load(Ordering::Relaxed)).put(&queues);
    }

    pub(crate) fn load(&self, d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
        self.next.store(d.get()?, Ordering::Relaxed);
        let queues: HashMap<u64, Vec<SavedRegistration>> = d.get()?;
        *lock(&self.queues) = queues
            .into_iter()
            .map(|(id, regs)| {
                let q = Queue::default();
                *lock(&q.registrations) = regs
                    .into_iter()
                    .map(|(ident, filter, udata, pending)| Registration {
                        ident,
                        filter,
                        udata,
                        pending,
                    })
                    .collect();
                (id, Arc::new(q))
            })
            .collect();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triggers_are_reported_once_to_queues_that_registered() {
        let qs = Queues::default();
        lock(&qs.queues).insert(1, Arc::default());
        assert!(qs.add(1, 7, -13, 0xabc));
        qs.trigger(7, -13, 5);
        qs.trigger(8, -13, 6);
        let q = qs.get(1).unwrap();
        assert_eq!(take(&mut lock(&q.registrations), 4), [(7, -13, 5, 0xabc)]);
        assert!(take(&mut lock(&q.registrations), 4).is_empty());
        assert!(qs.remove(1, 7, -13));
        assert!(!qs.add(2, 7, -13, 0));
    }
}
