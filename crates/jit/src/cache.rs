//! The code cache in segments. Code fills one segment at a time; when none
//! is free, the coldest full segment is evicted: least sampled first, then
//! oldest. A sampler kicks running vCPUs about 100 times a second and
//! credits the segment each one was executing in. An evicted segment is
//! reused once no vCPU can still be executing in it or hold an address in
//! it (see `docs/jit.md`).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{EMPTY, HASH_MUL, Machine, SHARED, layout, sys_icache_invalidate};

pub(crate) const SEGMENT: usize = 8 << 20;
/// Each segment starts with a copy of the dispatcher, and at `LOCAL_VERIFY`
/// a copy of the verify-mode exit, for its blocks to branch to; code follows.
const LOCAL_VERIFY: usize = 0x60;
const STUBS: usize = 0x80;
const _: () = assert!(SEGMENT <= 1 << 27);
const SEGMENTS: usize = (layout::CODE_CACHE_END - layout::CODE_CACHE).div_ceil(SEGMENT);
const SAMPLE_PERIOD: std::time::Duration = std::time::Duration::from_millis(10);

static SAMPLES: [AtomicU64; SEGMENTS] = [const { AtomicU64::new(0) }; SEGMENTS];
static EVICTIONS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Free,
    Filling,
    Full,
    Draining,
}

pub(crate) struct Segments {
    state: [State; SEGMENTS],
    score: [u64; SEGMENTS],
    filled: [u64; SEGMENTS],
    /// For a draining segment: the vCPUs inside the VM when it was evicted,
    /// and their exit counts then.
    wait: Vec<Vec<(usize, u64)>>,
    current: usize,
    seq: u64,
    /// Segments in use; fewer than all with `PS5SLOP_JIT_SEGMENTS`, to test
    /// eviction.
    limit: usize,
}

impl Segments {
    pub(crate) fn new() -> Self {
        let mut state = [State::Free; SEGMENTS];
        state[0] = State::Filling;
        Self {
            state,
            score: [0; SEGMENTS],
            filled: [0; SEGMENTS],
            wait: vec![Vec::new(); SEGMENTS],
            current: 0,
            seq: 0,
            limit: std::env::var("PS5SLOP_JIT_SEGMENTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .map_or(SEGMENTS, |n: usize| n.clamp(2, SEGMENTS)),
        }
    }
}

fn start(s: usize) -> usize {
    layout::CODE_CACHE + s * SEGMENT
}

/// Where code in the segment holding `place` exits to instead of `exit`
/// (`DISPATCH` or `VERIFY_EXIT`).
pub(crate) fn local_exit(place: usize, exit: usize) -> usize {
    let s = start((place - layout::CODE_CACHE) / SEGMENT);
    if exit == crate::DISPATCH {
        s
    } else {
        s + LOCAL_VERIFY
    }
}

/// Writes every segment's stubs: `dispatcher` and `verify_exit` copies.
pub(crate) fn write_stubs(rt: &mut [u8], dispatcher: &[u32], verify_exit: &[u32]) {
    assert!(dispatcher.len() * 4 <= LOCAL_VERIFY && LOCAL_VERIFY + verify_exit.len() * 4 <= STUBS);
    for s in 0..SEGMENTS {
        for (at, code) in [
            (start(s), dispatcher),
            (start(s) + LOCAL_VERIFY, verify_exit),
        ] {
            for (i, w) in code.iter().enumerate() {
                rt[at + i * 4..at + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
    }
}

/// Where code starts in segment `s`.
pub(crate) fn code_start(s: usize) -> usize {
    start(s) + STUBS
}

fn end(s: usize) -> usize {
    (start(s) + SEGMENT).min(layout::CODE_CACHE_END)
}

/// Credits the segment holding `pc` (a virtual address) with one sample.
pub(crate) fn sample(pc: u64) {
    let Some(off) = pc.checked_sub(layout::va(layout::CODE_CACHE)) else {
        return;
    };
    if let Some(s) = SAMPLES.get(off as usize / SEGMENT) {
        s.fetch_add(1, Ordering::Relaxed);
    }
}

/// Segments evicted so far.
pub fn evictions() -> u64 {
    EVICTIONS.load(Ordering::Relaxed)
}

/// Starts the thread that kicks running vCPUs so their position is sampled.
pub fn start_sampler() {
    let spawned = std::thread::Builder::new()
        .name("jit sampler".into())
        .spawn(|| {
            loop {
                std::thread::sleep(SAMPLE_PERIOD);
                if let Ok(m) = crate::machine() {
                    m.kick_running();
                }
            }
        });
    if let Err(e) = spawned {
        tracing::error!("cannot start the JIT sampler: {e}");
    }
}

impl Machine {
    /// Whether `len` bytes at `place` stay inside the segment being filled.
    pub(crate) fn fits(&self, place: usize, len: usize) -> bool {
        place + len <= end(self.segments.current)
    }

    /// Moves to a free segment; false if none is free yet. Keeps one
    /// eviction ahead, so a segment is usually drained before it is needed.
    pub(crate) fn next_segment(&mut self) -> bool {
        self.reclaim();
        let seg = &mut self.segments;
        if seg.state[seg.current] == State::Filling {
            seg.state[seg.current] = State::Full;
            seg.filled[seg.current] = seg.seq;
            seg.seq += 1;
        }
        let Some(next) = (0..seg.limit).find(|&s| seg.state[s] == State::Free) else {
            if !seg.state.contains(&State::Draining) {
                self.evict_coldest();
            }
            return false;
        };
        seg.state[next] = State::Filling;
        seg.current = next;
        self.code_next = code_start(next);
        let seg = &self.segments;
        if !seg.state[..seg.limit].contains(&State::Free) && !seg.state.contains(&State::Draining) {
            self.evict_coldest();
        }
        true
    }

    fn evict_coldest(&mut self) {
        let seg = &mut self.segments;
        for (score, samples) in seg.score.iter_mut().zip(&SAMPLES) {
            *score = *score / 2 + samples.swap(0, Ordering::Relaxed);
        }
        let Some(s) = (0..SEGMENTS)
            .filter(|&s| seg.state[s] == State::Full)
            .min_by_key(|&s| (seg.score[s], seg.filled[s]))
        else {
            return;
        };
        if EVICTIONS.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::warn!(
                "the JIT's code cache is full; evicting its coldest segments from now on"
            );
        }
        self.evict(s);
    }

    /// Takes everything in segment `s` out of use and starts draining it.
    fn evict(&mut self, s: usize) {
        let range = start(s)..end(s);
        let blocks: Vec<(u64, usize)> = self
            .blocks
            .iter()
            .filter(|(_, b)| range.contains(&b.place))
            .map(|(&r, b)| (r, b.place))
            .collect();
        for (rip, place) in blocks {
            self.blocks.remove(&rip);
            self.forget_lookup(rip, place);
        }
        crate::profile::LIVE_BLOCKS.store(
            self.blocks.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let regions: Vec<(u64, usize)> = self
            .regions
            .iter()
            .filter(|(_, r)| range.contains(&r.place))
            .map(|(&r, i)| (r, i.place))
            .collect();
        for (rip, place) in regions {
            self.regions.remove(&rip);
            self.forget_lookup(rip, place);
        }
        let places: Vec<usize> = self
            .by_place
            .range(range.clone())
            .map(|(&p, _)| p)
            .collect();
        for p in places {
            self.by_place.remove(&p);
        }
        let mut sites: Vec<usize> = self
            .chained_to
            .range(range.clone())
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        sites.extend(self.patched.keys().copied().filter(|p| range.contains(p)));
        let targets: Vec<usize> = self
            .chained_to
            .range(range.clone())
            .map(|(&t, _)| t)
            .collect();
        for t in targets {
            self.chained_to.remove(&t);
        }
        for site in sites {
            if let Some(word) = self.patched.remove(&site) {
                self.runtime.write_words(site, &[word]);
                // SAFETY: one word inside the runtime region.
                unsafe { sys_icache_invalidate(self.runtime.as_ptr().add(site).cast(), 4) };
            }
        }
        for v in self.chained_to.values_mut() {
            v.retain(|p| !range.contains(p));
        }
        for v in self.chain_wanted.values_mut() {
            v.retain(|p| !range.contains(p));
        }
        self.chain_wanted.retain(|_, v| !v.is_empty());
        let sc = Ordering::SeqCst;
        self.segments.wait[s] = (0..layout::MAX_VCPUS)
            .filter(|&v| SHARED.slots[v].inside.load(sc))
            .map(|v| (v, SHARED.slots[v].exits.load(sc)))
            .collect();
        self.segments.state[s] = State::Draining;
        SHARED.reclaim_wanted.store(true, sc);
    }

    /// Frees draining segments that no vCPU can be executing in or hold an
    /// address in.
    pub(crate) fn reclaim(&mut self) {
        let sc = Ordering::SeqCst;
        let held: Vec<u64> = SHARED
            .slots
            .iter()
            .map(|s| s.old_pc.load(sc))
            .filter(|&p| p != 0)
            .collect();
        for (s, samples) in SAMPLES.iter().enumerate() {
            if self.segments.state[s] != State::Draining {
                continue;
            }
            let (lo, hi) = (layout::va(start(s)), layout::va(end(s)));
            let left = self.segments.wait[s]
                .iter()
                .all(|&(v, n)| SHARED.slots[v].exits.load(sc) > n);
            if left && !held.iter().any(|&p| (lo..hi).contains(&p)) {
                self.segments.state[s] = State::Free;
                self.segments.score[s] = 0;
                self.segments.wait[s].clear();
                samples.store(0, Ordering::Relaxed);
            }
        }
        let draining = self.segments.state.contains(&State::Draining);
        SHARED.reclaim_wanted.store(draining, sc);
    }

    /// Empties `rip`'s lookup slot if it still leads to `place`.
    fn forget_lookup(&mut self, rip: u64, place: usize) {
        let slot = (rip.wrapping_mul(HASH_MUL) >> (64 - layout::LOOKUP_BITS)) as usize;
        let at = layout::LOOKUP + slot * 16;
        // SAFETY: both words are inside the runtime region and 8-aligned.
        let (tag, code) = unsafe {
            let base = self.runtime.as_ptr();
            (
                &*(base.add(at) as *const AtomicU64),
                &*(base.add(at + 8) as *const AtomicU64),
            )
        };
        if tag.load(Ordering::Acquire) == rip && code.load(Ordering::Acquire) == layout::va(place) {
            tag.store(EMPTY, Ordering::Release);
        }
    }

    /// Starts over with every segment free.
    pub(crate) fn reset_segments(&mut self) {
        self.segments = Segments::new();
        self.code_next = code_start(0);
        self.patched.clear();
        self.chained_to.clear();
        for s in &SAMPLES {
            s.store(0, Ordering::Relaxed);
        }
    }
}
