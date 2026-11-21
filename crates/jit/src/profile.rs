//! The run-time profiler (the control socket's `profile`): where each guest
//! thread's time goes, the hottest guest code by sample, and what falls back
//! to the interpreter. Off, it costs one relaxed load per VM exit.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::layout;

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Bumped for every new window, so threads start their counts afresh.
static WINDOW: AtomicU64 = AtomicU64::new(0);
static THREADS: Mutex<Vec<Arc<Thread>>> = Mutex::new(Vec::new());

thread_local! {
    static MINE: OnceCell<Arc<Thread>> = const { OnceCell::new() };
}

/// One thread's counts, locked only by that thread and by reports.
struct Thread {
    name: String,
    port: u32,
    counts: Mutex<Counts>,
}

struct Counts {
    window: u64,
    kinds: [u64; 5],
    instructions: u64,
    cpu_at_start: u64,
    first: Instant,
    /// Host functions by trap address: calls and nanoseconds.
    host: HashMap<u64, (u64, u64)>,
}

impl Counts {
    fn new(window: u64, port: u32) -> Self {
        Counts {
            window,
            kinds: [0; 5],
            instructions: 0,
            cpu_at_start: cpu_ns(port).unwrap_or(0),
            first: Instant::now(),
            host: HashMap::new(),
        }
    }
}

#[cfg(target_os = "macos")]
fn this_port() -> u32 {
    // SAFETY: plain queries about the calling thread.
    unsafe { libc::pthread_mach_thread_np(libc::pthread_self()) }
}

#[cfg(not(target_os = "macos"))]
fn this_port() -> u32 {
    0
}

/// The CPU time the thread with Mach port `port` has used.
#[cfg(target_os = "macos")]
fn cpu_ns(port: u32) -> Option<u64> {
    // SAFETY: `info` is large enough for THREAD_BASIC_INFO, as `count` says.
    unsafe {
        let mut info: libc::thread_basic_info = std::mem::zeroed();
        let mut count = libc::THREAD_BASIC_INFO_COUNT;
        let r = libc::thread_info(
            port,
            libc::THREAD_BASIC_INFO as u32,
            (&raw mut info).cast(),
            &mut count,
        );
        (r == 0).then(|| {
            let t = |v: libc::time_value_t| {
                v.seconds as u64 * 1_000_000_000 + v.microseconds as u64 * 1000
            };
            t(info.user_time) + t(info.system_time)
        })
    }
}

#[cfg(not(target_os = "macos"))]
fn cpu_ns(_port: u32) -> Option<u64> {
    None
}

fn with_counts(f: impl FnOnce(&mut Counts)) {
    MINE.with(|m| {
        let t = m.get_or_init(|| {
            let port = this_port();
            let t = Arc::new(Thread {
                name: std::thread::current().name().unwrap_or("?").to_owned(),
                port,
                counts: Mutex::new(Counts::new(u64::MAX, port)),
            });
            THREADS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(t.clone());
            t
        });
        let mut c = t.counts.lock().unwrap_or_else(|e| e.into_inner());
        let window = WINDOW.load(Ordering::Relaxed);
        if c.window != window {
            *c = Counts::new(window, t.port);
        }
        f(&mut c)
    })
}

/// What a thread was doing, for [`add`].
#[derive(Clone, Copy)]
pub enum Kind {
    /// Running translated code.
    Vm,
    Interp,
    Translate,
    /// In a host function (HLE), counted by the caller of [`add`].
    Host,
    /// Waiting for the JIT's lock to translate.
    Lock,
}

struct State {
    started: Instant,
    /// Samples by guest RIP, and samples in runtime code.
    samples: HashMap<u64, u64>,
    runtime_samples: u64,
    /// Samples in runtime code by what is there.
    runtime: HashMap<&'static str, u64>,
    translations_at_start: u64,
    retranslations_at_start: u64,
    emitted_at_start: u64,
    evictions_at_start: u64,
}

/// What a profile window gathered.
pub struct Report {
    pub seconds: f64,
    /// Thread name and nanoseconds in the VM, interpreter, translating,
    /// host functions and waiting for the JIT's lock, then CPU time used,
    /// wall time seen in the window and guest instructions run in the VM,
    /// busiest first.
    pub threads: Vec<(String, [u64; 8])>,
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
    /// Of those, translations of block starts translated before.
    pub retranslations: u64,
    /// Bytes of code they emitted.
    pub emitted: u64,
    /// Blocks in the code cache at the report.
    pub live_blocks: u64,
    /// The JIT's lock by where it was taken (`file:line`): times, nanoseconds
    /// waited and nanoseconds held, longest held first.
    pub lock_sites: Vec<(String, u64, u64, u64)>,
    pub evictions: u64,
}

pub static TRANSLATIONS: AtomicU64 = AtomicU64::new(0);
/// Translations of a block start that was translated before (and evicted).
pub static RETRANSLATIONS: AtomicU64 = AtomicU64::new(0);
/// Bytes of code emitted by translations.
pub static EMITTED: AtomicU64 = AtomicU64::new(0);
/// Blocks in the code cache now.
pub static LIVE_BLOCKS: AtomicU64 = AtomicU64::new(0);

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Starts a fresh profile window, or stops profiling.
pub fn set(enable: bool) {
    crate::engine::reset_stats();
    LOCK_SITES.lock().unwrap_or_else(|e| e.into_inner()).clear();
    WINDOW.fetch_add(1, Ordering::SeqCst);
    let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
    *s = enable.then(|| State {
        started: Instant::now(),
        samples: HashMap::new(),
        runtime_samples: 0,
        runtime: HashMap::new(),
        translations_at_start: TRANSLATIONS.load(Ordering::Relaxed),
        retranslations_at_start: RETRANSLATIONS.load(Ordering::Relaxed),
        emitted_at_start: EMITTED.load(Ordering::Relaxed),
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
    with_counts(|c| c.kinds[kind as usize] += ns);
}

/// Counts `n` guest instructions run in the VM by this thread.
pub fn add_instructions(n: u64) {
    if on() {
        with_counts(|c| c.instructions += n);
    }
}

/// Counts one call of the host function at `addr` that began at `since`.
pub fn add_host(addr: u64, since: Option<Instant>) {
    let Some(t) = since else { return };
    let ns = t.elapsed().as_nanos() as u64;
    with_counts(|c| {
        c.kinds[Kind::Host as usize] += ns;
        let e = c.host.entry(addr).or_default();
        e.0 += 1;
        e.1 += ns;
    });
}

type Site = &'static std::panic::Location<'static>;

/// The machine's lock by where it was taken: times taken, nanoseconds
/// waited for it and nanoseconds held.
static LOCK_SITES: Mutex<Vec<(Site, u64, u64, u64)>> = Mutex::new(Vec::new());

fn lock_site(site: Site, f: impl FnOnce(&mut (Site, u64, u64, u64))) {
    let mut sites = LOCK_SITES.lock().unwrap_or_else(|e| e.into_inner());
    let i = match sites.iter().position(|s| std::ptr::eq(s.0, site)) {
        Some(i) => i,
        None => {
            sites.push((site, 0, 0, 0));
            sites.len() - 1
        }
    };
    f(&mut sites[i]);
}

pub(crate) fn lock_waited(site: Site, since: Option<Instant>) {
    if let Some(t) = since {
        let ns = t.elapsed().as_nanos() as u64;
        lock_site(site, |s| s.2 += ns);
    }
}

pub(crate) fn lock_held(site: Site, since: Option<Instant>) {
    if let Some(t) = since {
        let ns = t.elapsed().as_nanos() as u64;
        lock_site(site, |s| {
            s.1 += 1;
            s.3 += ns;
        });
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
                *s.runtime
                    .entry(crate::runtime_part((pc - layout::RUNTIME_VA) as usize))
                    .or_default() += 1;
            }
            None => {}
        }
    }
}

/// What the current window gathered, or `None` when profiling is off.
pub fn report() -> Option<Report> {
    let s = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let s = s.as_ref()?;
    let window = WINDOW.load(Ordering::SeqCst);
    let mut threads = Vec::new();
    let mut by_thread: Vec<(String, u64, u64, u64)> = Vec::new();
    for t in THREADS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let c = t.counts.lock().unwrap_or_else(|e| e.into_inner());
        if c.window != window {
            continue;
        }
        let cpu = cpu_ns(t.port).map_or(0, |n| n.saturating_sub(c.cpu_at_start));
        let k = c.kinds;
        threads.push((
            t.name.clone(),
            [
                k[0],
                k[1],
                k[2],
                k[3],
                k[4],
                cpu,
                c.first.elapsed().as_nanos() as u64,
                c.instructions,
            ],
        ));
        by_thread.extend(
            c.host
                .iter()
                .map(|(&a, &(n, ns))| (t.name.clone(), a, n, ns)),
        );
    }
    threads.sort_by_key(|(_, t)| std::cmp::Reverse(t[5]));
    let mut samples: Vec<_> = s.samples.iter().map(|(&r, &n)| (r, n)).collect();
    samples.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let total_samples = samples.iter().map(|s| s.1).sum::<u64>() + s.runtime_samples;
    let mut runtime: Vec<_> = s.runtime.iter().map(|(&k, &n)| (k, n)).collect();
    runtime.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let mut totals: HashMap<u64, (u64, u64)> = HashMap::new();
    for &(_, a, c, ns) in &by_thread {
        let t = totals.entry(a).or_default();
        t.0 += c;
        t.1 += ns;
    }
    let mut host: Vec<_> = totals.iter().map(|(&a, &(c, ns))| (a, c, ns)).collect();
    host.sort_by_key(|&(_, _, ns)| std::cmp::Reverse(ns));
    let mut host_by_thread = by_thread;
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
        retranslations: RETRANSLATIONS.load(Ordering::Relaxed) - s.retranslations_at_start,
        emitted: EMITTED.load(Ordering::Relaxed) - s.emitted_at_start,
        live_blocks: LIVE_BLOCKS.load(Ordering::Relaxed),
        lock_sites: {
            let mut v: Vec<_> = LOCK_SITES
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|&(s, n, w, h)| (format!("{}:{}", s.file(), s.line()), n, w, h))
                .collect();
            v.sort_by_key(|s| std::cmp::Reverse(s.3));
            v
        },
        evictions: crate::cache::evictions() - s.evictions_at_start,
    })
}
