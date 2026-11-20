//! Write watchpoints for debugging: writes made through [`crate::Memory`]
//! into watched ranges are logged with the guest code that made them.
//! Executors name that code with [`set_site`] while watching is on.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Set on a site that is the return address of a host function call, not
/// the writing instruction itself.
pub const HOST_CALL: u64 = 1 << 63;

const RECENT: usize = 4096;

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<State> = Mutex::new(State {
    ranges: Vec::new(),
    sites: None,
    recent: VecDeque::new(),
    total: 0,
    trace: None,
});

thread_local! {
    static SITE: Cell<u64> = const { Cell::new(0) };
}

struct State {
    ranges: Vec<(u64, u64)>,
    /// By writing site and 64 KiB page.
    sites: Option<HashMap<(u64, u64), Site>>,
    recent: VecDeque<Hit>,
    total: u64,
    /// Where every hit from the listed sites (all if empty) is written.
    trace: Option<(std::io::BufWriter<std::fs::File>, Vec<u64>)>,
}

/// What the log keeps per writing site.
#[derive(Debug, Clone, Copy, Default)]
pub struct Site {
    pub count: u64,
    pub bytes: u64,
    pub lowest: u64,
    pub highest: u64,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub site: u64,
    pub addr: u64,
    /// Up to the first 8 bytes written.
    pub data: Vec<u8>,
    pub len: usize,
    pub thread: String,
}

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn active() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Names the guest code whose writes follow on this thread.
#[inline]
pub fn set_site(site: u64) {
    SITE.with(|s| s.set(site));
}

/// Adds `[start, end)` to the watched ranges.
pub fn add(start: u64, end: u64) {
    let mut s = state();
    s.ranges.push((start, end));
    s.sites.get_or_insert_with(HashMap::new);
    ON.store(true, Ordering::SeqCst);
}

/// Writes every later hit from `sites` (all if empty) to `file` as a text
/// line, or stops when `file` is `None`.
pub fn trace(file: Option<std::fs::File>, sites: Vec<u64>) {
    use std::io::Write;
    let mut s = state();
    if let Some((mut w, _)) = s.trace.take() {
        let _ = w.flush();
    }
    s.trace = file.map(|f| (std::io::BufWriter::new(f), sites));
}

/// Stops watching and forgets the log.
pub fn clear() {
    let mut s = state();
    ON.store(false, Ordering::SeqCst);
    s.ranges.clear();
    s.sites = None;
    s.recent.clear();
    s.total = 0;
    drop(s);
    trace(None, Vec::new());
}

pub fn ranges() -> Vec<(u64, u64)> {
    state().ranges.clone()
}

/// The sites that wrote inside `[lo, hi)`, most writes first, the latest
/// hits there, and the number of writes seen in all.
pub fn log(lo: u64, hi: u64) -> (Vec<(u64, Site)>, Vec<Hit>, u64) {
    let s = state();
    let mut sites: HashMap<u64, Site> = HashMap::new();
    for (&(site, _), v) in s.sites.iter().flatten() {
        if v.lowest < hi && lo < v.highest {
            let e = sites.entry(site).or_insert(Site {
                lowest: u64::MAX,
                ..Site::default()
            });
            e.count += v.count;
            e.bytes += v.bytes;
            e.lowest = e.lowest.min(v.lowest);
            e.highest = e.highest.max(v.highest);
        }
    }
    let mut sites: Vec<_> = sites.into_iter().collect();
    sites.sort_by_key(|&(_, v)| std::cmp::Reverse(v.count));
    let recent = s.recent.iter().filter(|h| h.addr < hi && lo < h.addr + h.len as u64).cloned().collect();
    (sites, recent, s.total)
}

pub(crate) fn note(addr: u64, data: &[u8]) {
    let end = addr + data.len() as u64;
    let mut s = state();
    if !s.ranges.iter().any(|&(lo, hi)| addr < hi && lo < end) {
        return;
    }
    let site = SITE.with(Cell::get);
    s.total += 1;
    let e = s.sites.get_or_insert_with(HashMap::new).entry((site, addr >> 16)).or_insert(Site {
        lowest: u64::MAX,
        ..Site::default()
    });
    e.count += 1;
    e.bytes += data.len() as u64;
    e.lowest = e.lowest.min(addr);
    e.highest = e.highest.max(end);
    if s.recent.len() == RECENT {
        s.recent.pop_front();
    }
    if let Some((w, sites)) = s.trace.as_mut()
        && (sites.is_empty() || sites.contains(&site))
    {
        use std::io::Write;
        let hex: String = data.iter().take(16).map(|b| format!("{b:02x}")).collect();
        let _ = writeln!(w, "{site:#x} {addr:#x} {:#x} {hex}", data.len());
    }
    let thread = std::thread::current().name().unwrap_or("?").to_owned();
    s.recent.push_back(Hit {
        site,
        addr,
        data: data[..data.len().min(8)].to_vec(),
        len: data.len(),
        thread,
    });
}
