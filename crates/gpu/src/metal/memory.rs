//! Guest memory as Metal sees it: no-copy windows, scratch memory, and what encoded work may write.

use super::*;

/// Guest memory is wrapped in windows of about this size.
const WINDOW: u64 = 64 << 20;

const PAGE: u64 = 0x4000;

pub(super) struct Window {
    pub(super) start: u64,
    pub(super) end: u64,
    pub(super) buffer: Retained<Buffer>,
}

/// Binding of the second guest memory window (reached through `Win.delta`).
pub(super) const SECOND_WINDOW: usize = 26;

/// Binding of the third window (`Win.delta3`).
pub(super) const THIRD_WINDOW: usize = 30;

/// The guest memory bound for a program and its `Win`: both windows' ranges,
/// the second's GPU address minus the first's, and the range stores skip.
pub(super) struct Memory {
    pub(super) buffer: Retained<Buffer>,
    pub(super) second: Option<Retained<Buffer>>,
    pub(super) third: Option<Retained<Buffer>>,
    pub(super) win: [u64; 10],
}

/// Guest memory `base..base + len` in place, if it is host-backed and at most 32 MiB.
pub(super) fn guest_bytes(host: &dyn Host, base: u64, len: u64) -> Option<&[u8]> {
    let (start, end) = host.extent(base)?;
    if base < start || base + len > end || len > 32 << 20 {
        return None;
    }
    let p = host.host_pointer(base)?;
    // SAFETY: the range is mapped guest memory; the guest may write it concurrently.
    Some(unsafe { std::slice::from_raw_parts(p, len as usize) })
}

/// A 64-bit digest of `bytes`, to tell whether guest memory changed.
pub(super) fn digest(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    // Sixteen independent lanes keep the multipliers busy: about memory speed.
    let (chunks, rest) = bytes.as_chunks::<128>();
    let mut lanes: [u64; 16] = std::array::from_fn(|i| i as u64 + 1);
    for c in chunks {
        for (lane, w) in lanes.iter_mut().zip(c.as_chunks::<8>().0) {
            *lane = (lane.rotate_left(23) ^ u64::from_le_bytes(*w)).wrapping_mul(K);
        }
    }
    let mut h = bytes.len() as u64;
    for (i, b) in rest.iter().enumerate() {
        h ^= u64::from(*b) << (8 * (i % 8));
        h = h.rotate_left(7).wrapping_mul(K);
    }
    lanes
        .iter()
        .fold(h, |h, l| (h.rotate_left(29) ^ l).wrapping_mul(K))
}

impl Metal {
    /// The no-copy buffer holding `[addr, addr + len)` and the offset of `addr` in it.
    pub(super) fn window(
        &mut self,
        host: &dyn Host,
        addr: u64,
        len: u64,
    ) -> Option<(Retained<Buffer>, usize)> {
        let generation = host.generation();
        if generation != self.generation {
            self.generation = generation;
            let (next, replaced) = host.replaced_since(self.replaced_mark);
            self.replaced_mark = next;
            let dropped: Vec<Window> = match replaced {
                Some(ranges) => {
                    let (gone, kept) = std::mem::take(&mut self.windows)
                        .into_iter()
                        .partition(|w| ranges.iter().any(|&(s, e)| s < w.end && w.start < e));
                    self.windows = kept;
                    if self
                        .area
                        .as_ref()
                        .is_some_and(|w| ranges.iter().any(|&(s, e)| s < w.end && w.start < e))
                    {
                        self.stale += self.area.take().map_or(0, |w| w.end - w.start);
                    }
                    gone
                }
                None => {
                    self.flush_and_wait();
                    self.area = None;
                    std::mem::take(&mut self.windows)
                }
            };
            self.stale += dropped.iter().map(|w| w.end - w.start).sum::<u64>();
            if let Some(r) = &self.resident
                && !dropped.is_empty()
            {
                for w in &dropped {
                    r.removeAllocation(ProtocolObject::from_ref(&*w.buffer));
                }
                r.commit();
            }
        }
        let end = addr.checked_add(len.max(1))?;
        if let Some(w) = self
            .windows
            .iter()
            .find(|w| w.start <= addr && end <= w.end)
        {
            if self.capture_reads {
                host.extent(addr);
            }
            return Some((w.buffer.clone(), (addr - w.start) as usize));
        }
        let (lo, hi) = host.extent(addr)?;
        if end > hi {
            return None;
        }
        let start = (addr & !(WINDOW - 1)).max(lo) & !(PAGE - 1);
        let stop = end
            .next_multiple_of(WINDOW)
            .min(hi)
            .next_multiple_of(PAGE)
            .min(hi);
        if stop - start > 512 << 20 {
            self.report(
                tracing::Level::INFO,
                format!(
                    "guest memory window of {} MiB at {start:#x} for {len:#x} bytes at {addr:#x}",
                    (stop - start) >> 20
                ),
            );
        }
        self.make_room(stop - start);
        let ptr = NonNull::new(host.host_pointer(start)?)?;
        // SAFETY: the range is host memory that stays mapped while the address
        // space lives; windows are dropped when its mappings change.
        let Some(buffer) = (unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr.cast(),
                    (stop - start) as usize,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }) else {
            diag::problems::record(
                diag::problems::Kind::MemoryWindow,
                "Metal refused guest window",
                format!("{start:#x}..{stop:#x}"),
            );
            return None;
        };
        let (inside, kept) = std::mem::take(&mut self.windows)
            .into_iter()
            .partition::<Vec<_>, _>(|w| start <= w.start && w.end <= stop);
        self.windows = kept;
        if let Some(r) = &self.resident {
            for w in &inside {
                r.removeAllocation(ProtocolObject::from_ref(&*w.buffer));
            }
            r.addAllocation(ProtocolObject::from_ref(&*buffer));
            r.commit();
        }
        self.windows.push(Window {
            start,
            end: stop,
            buffer: buffer.clone(),
        });
        Some((buffer, (addr - start) as usize))
    }

    /// The buffer and offset for `len` bytes of guest memory at `at`: the one
    /// programs use there, so Metal orders other work on it with theirs.
    pub(super) fn memory_buffer(
        &mut self,
        host: &dyn Host,
        at: u64,
        len: u64,
    ) -> Option<(Retained<Buffer>, usize)> {
        match host.gpu_area() {
            Some((start, end)) if (start..end).contains(&at) && at + len <= end => Some((
                self.area_window(host, start, end, at + len)?,
                (at - start) as usize,
            )),
            _ => self.window(host, at, len),
        }
    }

    /// `len` bytes of shared memory no GPU work uses yet, 4-byte aligned:
    /// a buffer and an offset into it.
    pub(super) fn scratch(&mut self, len: usize) -> Option<(Retained<Buffer>, usize)> {
        const SCRATCH: usize = 4 << 20;
        let need = len.next_multiple_of(4);
        if self
            .scratch
            .as_ref()
            .is_none_or(|(b, used)| used + need > b.length())
        {
            let buffer = self.device.newBufferWithLength_options(
                SCRATCH.max(need),
                MTLResourceOptions::StorageModeShared,
            )?;
            self.scratch = Some((buffer, 0));
        }
        let (buffer, used) = self.scratch.as_mut()?;
        let at = *used;
        *used += need;
        Some((buffer.clone(), at))
    }

    pub(super) fn gds_buffer(&mut self) -> Option<Retained<Buffer>> {
        if self.gds.is_none() {
            self.gds =
                Some(self.device.newBufferWithLength_options(
                    GDS_BYTES,
                    MTLResourceOptions::StorageModeShared,
                )?);
        }
        self.gds.clone()
    }

    /// Before the CPU reads `len` bytes of guest memory at `at`: waits for
    /// encoded work that may write there.
    pub(super) fn settle_for(&mut self, at: u64, len: u64, why: &'static str) {
        if self.written.covers(at, len) {
            *self.settles.entry(why).or_default() += 1;
            self.flush_and_wait();
        }
    }

    /// Adds what a program encoded with user data `user` may write to `written`.
    pub(super) fn note_writes(
        &mut self,
        host: &dyn Host,
        program: &gfx10::Translated,
        user: &[u32],
    ) {
        if !program.writes_memory {
            return;
        }
        let ranges: Option<Vec<(u64, u64)>> = program.store_buffers.as_ref().and_then(|list| {
            list.iter()
                .map(|(source, indexed)| {
                    let v = self.descriptor(host, user, source, 4)?;
                    Some(buffer_range(&v, *indexed))
                })
                .collect()
        });
        if let Some(r) = &ranges {
            self.drop_overwritten(r);
        }
        self.note_ranges(ranges);
    }

    /// Forgets 2D textures whose memory an encoded buffer store spans, so
    /// their next use reads guest memory.
    fn drop_overwritten(&mut self, ranges: &[(u64, u64)]) {
        for &(s, e) in ranges {
            let Some(t) = self.targets.get(&s) else {
                continue;
            };
            let tex = &t.texture;
            if tex.textureType() != MTLTextureType::Type2D || tex.mipmapLevelCount() > 1 {
                continue;
            }
            let Some(bpe) = texel_bytes(t.pixel_format) else {
                continue;
            };
            let len = (tex.width() * tex.height() * bpe) as u64;
            if len <= e - s && e - s <= 2 * len + 16 {
                self.targets.remove(&s);
            }
        }
    }

    /// Adds `ranges` (`None`: anywhere) to what encoded work may write.
    fn note_ranges(&mut self, ranges: Option<Vec<(u64, u64)>>) {
        match &ranges {
            Some(r) => self
                .gpu_checked
                .retain(|&a| !r.iter().any(|&(s, e)| s < a + 4 && a < e)),
            None => self.gpu_checked.clear(),
        }
        self.written = match (
            std::mem::replace(&mut self.written, Written::Anything),
            ranges,
        ) {
            (Written::Nothing, Some(r)) => Written::Ranges(r),
            (Written::Ranges(mut old), Some(r)) if old.len() + r.len() <= MAX_WRITTEN_RANGES => {
                old.extend(r);
                Written::Ranges(old)
            }
            _ => Written::Anything,
        };
    }

    /// Windows over the largest region a program's user data points into and
    /// over the GPU area (or the other way round), as programs read both, and
    /// over the largest other region the user data points into. Buffer
    /// descriptors in tables the user data points to count after aligned
    /// pointers and before unaligned ones.
    pub(super) fn guest_window(
        &mut self,
        host: &dyn Host,
        user: &[u32],
        code: u64,
    ) -> Option<Memory> {
        let max = self.device.maxBufferLength() as u64 & !(PAGE - 1);
        let area = host.gpu_area();
        let in_area = |a: u64| area.is_some_and(|(s, e)| s <= a && a < e);
        let mut best = host.extent(code);
        // (start, end, above 4 GiB, rank): 2 for aligned pointers in the user
        // data, 1 for V# bases in tables it points to, 0 for unaligned ones.
        let mut pointed: Vec<(u64, u64, bool, u8)> = Vec::new();
        if let Some((lo, hi)) = best {
            pointed.push((lo, hi, lo >> 32 != 0, 2));
        }
        let mut tables = Vec::new();
        for pair in user.windows(2) {
            for ptr in [
                pair[0] as u64 | (pair[1] as u64) << 32,
                pair[0] as u64 | ((pair[1] & 0xFFFF) as u64) << 32,
            ] {
                if let Some((lo, hi)) = host.extent(ptr) {
                    pointed.push((lo, hi, ptr >> 32 != 0, if ptr % 4 == 0 { 2 } else { 0 }));
                    if best.is_none_or(|(l, h)| hi - lo > h - l) {
                        best = Some((lo, hi));
                    }
                    if pair[1] >> 16 == 0 && ptr >> 32 != 0 && ptr % 16 == 0 {
                        tables.push((ptr, hi));
                    }
                }
            }
        }
        // Follow V# tables in user data to discover buffer pools outside the GPU area.
        for (ptr, hi) in tables {
            let mut words = [0u8; 4 * TABLE_DWORDS];
            let n = ((hi - ptr) as usize).min(words.len()) & !15;
            if !host.read(ptr, &mut words[..n]) {
                continue;
            }
            for q in words[..n].as_chunks::<16>().0 {
                let d =
                    |i: usize| u32::from_le_bytes(q[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
                let base = d(0) as u64 | ((d(1) & 0xFFFF) as u64) << 32;
                if base >> 32 == 0 || in_area(base) || d(2) == 0 {
                    continue;
                }
                if let Some((lo, hi)) = host.extent(base) {
                    pointed.push((lo, hi, true, 1));
                }
            }
        }
        let outside = pointed
            .iter()
            .filter(|&&(l, _, _, _)| !in_area(l))
            .max_by_key(|&&(l, h, _, rank)| (rank, h - l))
            .map(|&(l, h, _, _)| (l, h));
        let (lo, hi) = best?;
        let region = |this: &mut Self, (lo, hi): (u64, u64)| -> Option<(Retained<Buffer>, u64)> {
            let (buf, offset) = this.window(host, lo, (hi - lo).min(max))?;
            Some((buf, lo - offset as u64))
        };
        let (first, second) = match area {
            Some((start, end)) if in_area(lo) => {
                let w = self.area_window(host, start, end, hi.min(end))?;
                (Some((w, start)), outside.and_then(|r| region(self, r)))
            }
            Some((start, end)) => {
                let need = self
                    .area
                    .as_ref()
                    .map_or(self.area_reach.max(start + 1), |w| w.end);
                let a = self.area_window(host, start, end, need).map(|w| (w, start));
                (region(self, (lo, hi)), a)
            }
            None => (region(self, (lo, hi)), None),
        };
        // A word followed by 0 often only looks like a pointer: prefer ones above 4 GiB.
        let third = pointed
            .into_iter()
            .filter(|&(l, _, _, _)| !in_area(l) && Some(l) != outside.map(|o| o.0) && l != lo)
            .max_by_key(|&(l, h, wide, rank)| (rank, wide, h - l))
            .and_then(|(l, h, _, _)| region(self, (l, h)));
        let (buffer, base) = first?;
        let (ro, ro_end) = outside.and_then(|(lo, _)| host.image(lo)).unwrap_or((0, 0));
        let mut win = [
            base,
            base + buffer.length() as u64,
            0,
            0,
            0,
            ro,
            ro_end,
            0,
            0,
            0,
        ];
        let second = second.map(|(b, base2)| {
            win[2] = base2;
            win[3] = base2 + b.length() as u64;
            win[4] = b.gpuAddress().wrapping_sub(buffer.gpuAddress());
            b
        });
        let third = third.map(|(b, base3)| {
            win[7] = base3;
            win[8] = base3 + b.length() as u64;
            win[9] = b.gpuAddress().wrapping_sub(buffer.gpuAddress());
            b
        });
        Some(Memory {
            buffer,
            second,
            third,
            win,
        })
    }

    /// Before a new window of `len` bytes: commits the open command buffer once
    /// it may hold 1 GiB of dropped windows, and waits for the GPU when Metal's
    /// allocations would pass half its working set. A command buffer keeps every
    /// window it binds, and Metal fails it ("Insufficient Memory") past that.
    fn make_room(&mut self, len: u64) {
        if self.stale < 1 << 30 {
            return;
        }
        self.commit();
        let budget = self.device.recommendedMaxWorkingSetSize() / 2;
        if self.device.currentAllocatedSize() as u64 + len > budget {
            self.flush_and_wait();
        }
    }

    /// The window over the GPU area `[start, end)` reaching at least `need`
    /// and as far as any window before it, made again (in 1 GiB steps) when
    /// it does not reach that far.
    pub(super) fn area_window(
        &mut self,
        host: &dyn Host,
        start: u64,
        end: u64,
        need: u64,
    ) -> Option<Retained<Buffer>> {
        if let Some(w) = &self.area
            && need <= w.end
        {
            return Some(w.buffer.clone());
        }
        let max = self.device.maxBufferLength() as u64 & !(PAGE - 1);
        if need > end.min(start + max) {
            let stop = end.min(start + max);
            let msg = format!(
                "{need:#x} past {stop:#x} (area {start:#x}..{end:#x}, Metal limit {max:#x})"
            );
            diag::problems::record_with_detail(
                diag::problems::Kind::MemoryWindow,
                "GPU area window past Metal limit",
                format!("{start:#x}..{end:#x}"),
                &msg,
            );
            self.report(tracing::Level::ERROR, format!("GPU area window: {msg}"));
            return None;
        }
        self.area_reach = self.area_reach.max(need);
        let stop = self
            .area_reach
            .next_multiple_of(1 << 30)
            .min(end)
            .min(start + max);
        if !host.back(start, stop) {
            diag::problems::record(
                diag::problems::Kind::MemoryWindow,
                "GPU area window backing failed",
                format!("{start:#x}..{stop:#x}"),
            );
            self.report(tracing::Level::ERROR, format!(
                "GPU area window: guest memory {start:#x}..{stop:#x} could not be given host pages"
            ));
            return None;
        }
        self.stale += self.area.take().map_or(0, |w| w.end - w.start);
        self.make_room(stop - start);
        let ptr = NonNull::new(host.host_pointer(start)?)?;
        // SAFETY: `back` gave every page host memory, which stays mapped; the
        // window is dropped when pages under it are replaced.
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr.cast(),
                    (stop - start) as usize,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        };
        let Some(buffer) = buffer else {
            diag::problems::record_with_detail(
                diag::problems::Kind::MemoryWindow,
                "Metal refused GPU area buffer",
                format!("{start:#x}..{stop:#x}"),
                format!(
                    "{} MiB ({} MiB allocated)",
                    (stop - start) >> 20,
                    self.device.currentAllocatedSize() >> 20
                ),
            );
            self.report(tracing::Level::ERROR, format!(
                "GPU area window: Metal refused a {} MiB buffer over guest memory ({} MiB allocated)",
                (stop - start) >> 20,
                self.device.currentAllocatedSize() >> 20
            ));
            return None;
        };
        // Made again after mapping changes: report only growth, so the
        // window does not use up the reports.
        static LARGEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if LARGEST.fetch_max(stop - start, std::sync::atomic::Ordering::Relaxed) < stop - start {
            self.report(
                tracing::Level::INFO,
                format!(
                    "GPU area window of {} MiB at {start:#x}",
                    (stop - start) >> 20
                ),
            );
        }
        self.note(|| format!("      new GPU area window of {} MiB", (stop - start) >> 20));
        self.area = Some(Window {
            start,
            end: stop,
            buffer: buffer.clone(),
        });
        Some(buffer)
    }
}

/// Guest memory that encoded work may write.
pub(super) enum Written {
    Nothing,
    /// Start and end addresses.
    Ranges(Vec<(u64, u64)>),
    Anything,
}

/// Ranges past this many are tracked as [`Written::Anything`].
const MAX_WRITTEN_RANGES: usize = 256;

impl Written {
    pub(super) fn covers(&self, at: u64, len: u64) -> bool {
        match self {
            Written::Nothing => false,
            Written::Ranges(r) => r.iter().any(|&(s, e)| s < at + len && at < e),
            Written::Anything => true,
        }
    }
}

/// The bytes a buffer store through V# `v` can reach: `buffer_offset` checks
/// the index against the record count when `indexed` (the offset is taken to
/// stay within one record), else the offset; plus the widest store.
pub(super) fn buffer_range(v: &[u32], indexed: bool) -> (u64, u64) {
    let base = v[0] as u64 | (v[1] as u64 & 0xFFFF) << 32;
    let stride = ((v[1] >> 16) & 0x3FFF) as u64;
    let records = v[2] as u64;
    let len = if indexed && stride != 0 {
        (records + 1) * stride
    } else {
        records
    };
    (base, base + len + 16)
}

/// The global data share's size on GFX10.
pub(super) const GDS_BYTES: usize = 64 << 10;
