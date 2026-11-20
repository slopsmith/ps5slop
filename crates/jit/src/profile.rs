//! The run-time profiler (the control socket's `profile`): where each guest
//! thread's time goes, the hottest guest code by sample, and what falls back
//! to the interpreter. Off, it costs one relaxed load per VM exit.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use crate::layout;

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);

/// What a thread was doing, for [`add`].
#[derive(Clone, Copy)]
pub enum Kind {
    /// Running translated code.
    Vm,
    Interp,
    Translate,
    /// In a host function (HLE), counted by the caller of [`add`].
    Host,
}

struct State {
    started: Instant,
    /// By thread name: nanoseconds per [`Kind`].
    threads: HashMap<String, [u64; 4]>,
    /// Samples by guest RIP, and samples in runtime code.
    samples: HashMap<u64, u64>,
    runtime_samples: u64,
    /// Samples in runtime code by what is there.
    runtime: HashMap<&'static str, u64>,
    /// Host functions by thread name and trap address: calls and nanoseconds.
    host: HashMap<(String, u64), (u64, u64)>,
    translations_at_start: u64,
    evictions_at_start: u64,
}

/// What a profile window gathered.
pub struct Report {
    pub seconds: f64,
    /// Thread name and nanoseconds in the VM, interpreter, translating and
    /// host functions, busiest first.
    pub threads: Vec<(String, [u64; 4])>,
    /// Guest RIPs by samples, most first, with the total.
    pub samples: Vec<(u64, u64)>,
    pub total_samples: u64,
    pub runtime_samples: u64,
    /// Runtime samples by where they fell, most first.
    pub runtime: Vec<(&'static str, u64)>,
    /// Host functions by trap address, by time: calls and nanoseconds.
    pub host: Vec<(u64, u64, u64)>,
    /// The same per thread: thread name, trap address, calls, nanoseconds.
    pub host_by_thread: Vec<(String, u64, u64, u64)>,
    pub translations: u64,
    pub evictions: u64,
}

pub static TRANSLATIONS: AtomicU64 = AtomicU64::new(0);

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Starts a fresh profile window, or stops profiling.
pub fn set(enable: bool) {
    crate::engine::reset_stats();
    let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
    *s = enable.then(|| State {
        started: Instant::now(),
        threads: HashMap::new(),
        samples: HashMap::new(),
        runtime_samples: 0,
        runtime: HashMap::new(),
        host: HashMap::new(),
        translations_at_start: TRANSLATIONS.load(Ordering::Relaxed),
        evictions_at_start: crate::cache::evictions(),
    });
    ON.store(enable, Ordering::SeqCst);
}

/// Now, if profiling.
pub fn start() -> Option<Instant> {
    on().then(Instant::now)
}

/// Adds the time since `since` to this thread's `kind`.
pub fn add(kind: Kind, since: Option<Instant>) {
    let Some(t) = since else { return };
    let ns = t.elapsed().as_nanos() as u64;
    let name = std::thread::current().name().unwrap_or("?").to_owned();
    if let Some(s) = STATE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        s.threads.entry(name).or_default()[kind as usize] += ns;
    }
}

/// Counts one call of the host function at `addr` that began at `since`.
pub fn add_host(addr: u64, since: Option<Instant>) {
    let Some(t) = since else { return };
    let ns = t.elapsed().as_nanos() as u64;
    let name = std::thread::current().name().unwrap_or("?").to_owned();
    if let Some(s) = STATE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        s.threads.entry(name.clone()).or_default()[Kind::Host as usize] += ns;
        let e = s.host.entry((name, addr)).or_default();
        e.0 += 1;
        e.1 += ns;
    }
}

/// Records a sample of a vCPU stopped at runtime address `pc`.
pub(crate) fn sample(pc: u64) {
    if !on() {
        return;
    }
    let rip = crate::machine().ok().and_then(|m| m.rip_at(pc));
    if let Some(s) = STATE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        match rip {
            Some(r) => *s.samples.entry(r).or_default() += 1,
            None if pc >= layout::RUNTIME_VA => {
                s.runtime_samples += 1;
                *s.runtime.entry(crate::runtime_part((pc - layout::RUNTIME_VA) as usize)).or_default() += 1;
            }
            None => {}
        }
    }
}

/// What the current window gathered, or `None` when profiling is off.
pub fn report() -> Option<Report> {
    let s = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let s = s.as_ref()?;
    let mut threads: Vec<_> = s.threads.iter().map(|(n, t)| (n.clone(), *t)).collect();
    threads.sort_by_key(|(_, t)| std::cmp::Reverse(t[0] + t[1] + t[2]));
    let mut samples: Vec<_> = s.samples.iter().map(|(&r, &n)| (r, n)).collect();
    samples.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let total_samples = samples.iter().map(|s| s.1).sum::<u64>() + s.runtime_samples;
    let mut runtime: Vec<_> = s.runtime.iter().map(|(&k, &n)| (k, n)).collect();
    runtime.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let mut totals: HashMap<u64, (u64, u64)> = HashMap::new();
    for (&(_, a), &(c, ns)) in &s.host {
        let t = totals.entry(a).or_default();
        t.0 += c;
        t.1 += ns;
    }
    let mut host: Vec<_> = totals.iter().map(|(&a, &(c, ns))| (a, c, ns)).collect();
    host.sort_by_key(|&(_, _, ns)| std::cmp::Reverse(ns));
    let mut host_by_thread: Vec<_> = s.host.iter().map(|((t, a), &(c, ns))| (t.clone(), *a, c, ns)).collect();
    host_by_thread.sort_by_key(|h| std::cmp::Reverse(h.3));
    Some(Report {
        seconds: s.started.elapsed().as_secs_f64(),
        threads,
        samples,
        total_samples,
        runtime_samples: s.runtime_samples,
        runtime,
        host,
        host_by_thread,
        translations: TRANSLATIONS.load(Ordering::Relaxed) - s.translations_at_start,
        evictions: crate::cache::evictions() - s.evictions_at_start,
    })
}
