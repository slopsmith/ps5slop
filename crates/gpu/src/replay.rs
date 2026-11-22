//! Deterministic offline replay of a [`FrameCapture`] through the Metal
//! renderer without the game or the hypervisor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use crate::capture::{FrameCapture, PAGE_SIZE};
use crate::metal::{Metal, SharedTexture};
use crate::regs::{ColorTarget, DepthTarget, Viewport, context, sh, uconfig};
use crate::{Backend, Dispatch, Draw, Flip, Host, Processor, State};

/// Metadata recorded for one draw call during replay.
#[derive(Clone, Debug)]
pub struct DrawRecord {
    pub index: usize,
    pub queue: u64,
    pub vs: u64,
    pub ps: u64,
    pub primitive: u32,
    pub draw: Draw,
    pub color_targets: Vec<(u64, u32, u32)>,
    pub depth_target: Option<(u64, u32, u32)>,
    pub viewport: Viewport,
    pub scissor: (u32, u32, u32, u32),
}

/// Metadata recorded for one compute dispatch during replay.
#[derive(Clone, Debug)]
pub struct DispatchRecord {
    pub index: usize,
    pub queue: u64,
    pub cs: u64,
    pub groups: Option<[u32; 3]>,
    pub indirect_at: Option<u64>,
}

/// Timing for one render pass or compute dispatch during `time_passes` replay.
#[derive(Clone, Debug)]
pub struct PassTiming {
    pub index: usize,
    pub kind: String,
    pub draws: usize,
    pub targets: Vec<u64>,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
}

/// Options controlling how [`replay`] runs a capture.
#[derive(Clone, Debug, Default)]
pub struct ReplayOptions {
    /// Stop right after this 0-based draw index, keeping targets at that point.
    pub stop_after_draw: Option<usize>,
    /// Commit and time each render pass and compute dispatch individually.
    pub time_passes: bool,
    /// Record the replay into a Metal `.gputrace` bundle (needs `METAL_CAPTURE_ENABLED=1`).
    pub metal_capture: Option<PathBuf>,
}

/// Output of replaying a [`FrameCapture`].
pub struct ReplayResult {
    pub draws: Vec<DrawRecord>,
    pub dispatches: Vec<DispatchRecord>,
    pub timings: Vec<PassTiming>,
    pub shown: Option<u64>,
    pub targets: Vec<(u64, SharedTexture)>,
    pub depths: Vec<(u64, SharedTexture)>,
    pub summary: String,
}

struct Segment {
    start: u64,
    end: u64,
    base: *mut u8,
}

impl Drop for Segment {
    fn drop(&mut self) {
        let len = (self.end - self.start) as usize;
        if len > 0 && !self.base.is_null() {
            // SAFETY: `base` was allocated with `mmap` of length `len`.
            unsafe {
                libc::munmap(self.base.cast(), len);
            }
        }
    }
}

/// Standalone guest memory host reconstructed from a [`FrameCapture`].
pub struct ReplayHost {
    segments: Vec<Segment>,
    gpu_area: Option<(u64, u64)>,
    images: Vec<(u64, u64)>,
    display_buffers: HashMap<i32, u64>,
    flips: Mutex<Vec<Flip>>,
}

impl ReplayHost {
    pub fn from_capture(cap: &FrameCapture) -> Result<Self, String> {
        // Merge overlapping/adjacent extents, plus any page or command buffer
        // ranges not already covered by a recorded extent.
        let mut ranges: Vec<(u64, u64)> = cap.meta.extents.clone();
        if let Some((a, b)) = cap.meta.gpu_area {
            // Cover only up to the highest page or extent inside the GPU area,
            // rounded up to 1 GiB so Metal's area window stays bounded.
            let mut max_in_area = a + PAGE_SIZE;
            for &p in cap.pages.keys() {
                if p >= a && p < b {
                    max_in_area = max_in_area.max(p + PAGE_SIZE);
                }
            }
            for &(lo, hi) in &cap.meta.extents {
                if lo >= a && lo < b {
                    max_in_area = max_in_area.max(hi.min(b));
                }
            }
            for sub in &cap.submissions {
                for cb in &sub.buffers {
                    let end = cb.address + cb.dwords.len() as u64 * 4;
                    if cb.address >= a && cb.address < b {
                        max_in_area = max_in_area.max(end.next_multiple_of(PAGE_SIZE).min(b));
                    }
                }
            }
            let area_end = max_in_area.next_multiple_of(1 << 20).min(b);
            ranges.push((a, area_end));
        }
        for &p in cap.pages.keys() {
            ranges.push((p, p + PAGE_SIZE));
        }
        for sub in &cap.submissions {
            for cb in &sub.buffers {
                let lo = cb.address & !(PAGE_SIZE - 1);
                let hi = (cb.address + cb.dwords.len() as u64 * 4)
                    .next_multiple_of(PAGE_SIZE)
                    .max(lo + PAGE_SIZE);
                ranges.push((lo, hi));
            }
        }
        ranges.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in ranges {
            let lo = lo & !(PAGE_SIZE - 1);
            let hi = hi.next_multiple_of(PAGE_SIZE);
            if hi <= lo {
                continue;
            }
            if let Some(last) = merged.last_mut()
                && lo <= last.1
            {
                last.1 = last.1.max(hi);
            } else {
                merged.push((lo, hi));
            }
        }

        let mut segments = Vec::with_capacity(merged.len());
        for (start, end) in merged {
            let len = (end - start) as usize;
            // SAFETY: fresh anonymous private mapping, zero-initialized on demand.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(format!(
                    "mmap failed for {start:#x}..{end:#x} ({len} bytes)"
                ));
            }
            segments.push(Segment {
                start,
                end,
                base: ptr.cast(),
            });
        }

        let gpu_area = cap.meta.gpu_area.map(|(a, b)| {
            let actual_end = segments
                .iter()
                .find(|s| s.start <= a && a < s.end)
                .map_or(b, |s| s.end.min(b));
            (a, actual_end)
        });

        let host = Self {
            segments,
            gpu_area,
            images: cap.meta.images.clone(),
            display_buffers: cap.meta.display_buffers.iter().copied().collect(),
            flips: Mutex::new(Vec::new()),
        };

        for (&addr, bytes) in &cap.pages {
            host.write(addr, bytes);
        }
        Ok(host)
    }

    fn find(&self, addr: u64) -> Option<&Segment> {
        let i = self.segments.partition_point(|s| s.end <= addr);
        self.segments
            .get(i)
            .filter(|s| s.start <= addr && addr < s.end)
    }

    pub fn last_flip(&self) -> Option<Flip> {
        self.flips
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .copied()
    }
}

impl Host for ReplayHost {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        if buf.is_empty() {
            return true;
        }
        let Some(end) = addr.checked_add(buf.len() as u64) else {
            return false;
        };
        let Some(seg) = self.find(addr) else {
            return false;
        };
        if end > seg.end {
            return false;
        }
        // SAFETY: `[addr, end)` is within `seg`'s mmap region.
        unsafe {
            std::ptr::copy_nonoverlapping(
                seg.base.add((addr - seg.start) as usize),
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        true
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        let Some(end) = addr.checked_add(data.len() as u64) else {
            return false;
        };
        let Some(seg) = self.find(addr) else {
            return false;
        };
        if end > seg.end {
            return false;
        }
        // SAFETY: `[addr, end)` is within `seg`'s writable mmap region.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                seg.base.add((addr - seg.start) as usize),
                data.len(),
            );
        }
        true
    }

    fn interrupt(&self, _context: u32) {}

    fn flip(&self, flip: Flip) {
        self.flips
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(flip);
    }

    fn exiting(&self) -> bool {
        false
    }

    fn extent(&self, addr: u64) -> Option<(u64, u64)> {
        self.find(addr).map(|s| (s.start, s.end))
    }

    fn image(&self, addr: u64) -> Option<(u64, u64)> {
        self.images
            .iter()
            .copied()
            .find(|&(lo, hi)| lo <= addr && addr < hi)
    }

    fn host_pointer(&self, addr: u64) -> Option<*mut u8> {
        self.find(addr)
            .map(|s| unsafe { s.base.add((addr - s.start) as usize) })
    }

    fn display_buffer(&self, index: i32) -> Option<u64> {
        self.display_buffers.get(&index).copied()
    }

    fn gpu_area(&self) -> Option<(u64, u64)> {
        self.gpu_area
    }

    fn back(&self, start: u64, end: u64) -> bool {
        self.find(start).is_some_and(|s| end <= s.end)
    }
}

struct ReplayState {
    metal: Metal,
    current_queue: u64,
    options: ReplayOptions,
    stopped: bool,
    draws: Vec<DrawRecord>,
    dispatches: Vec<DispatchRecord>,
    timings: Vec<PassTiming>,
    open_pass: Option<(Vec<u64>, usize, f64)>,
    shown: Option<u64>,
}

impl ReplayState {
    fn flush_pass_timing(&mut self) {
        if let Some((targets, draws, cpu_ms)) = self.open_pass.take() {
            self.metal.sync();
            let gpu_ms = self.metal.take_gpu_time_secs() * 1e3;
            let index = self.timings.len();
            self.timings.push(PassTiming {
                index,
                kind: "render".into(),
                draws,
                targets,
                cpu_ms,
                gpu_ms,
            });
        }
    }
}

/// Shared wrapper around [`ReplayState`] so multiple replayed queues share one renderer.
#[derive(Clone)]
struct ReplayBackend(std::sync::Arc<Mutex<ReplayState>>);

impl ReplayBackend {
    fn lock(&self) -> std::sync::MutexGuard<'_, ReplayState> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Backend for ReplayBackend {
    fn draw(&mut self, host: &dyn Host, state: &State, draw: Draw) {
        let mut s = self.lock();
        if s.stopped {
            return;
        }
        let c = state.context;
        let sh = state.sh;
        let ngg = c[context::VGT_SHADER_STAGES_EN] & (1 << 13) != 0;
        let vs_lo = if ngg {
            sh::SPI_SHADER_PGM_LO_ES
        } else {
            sh::SPI_SHADER_PGM_LO_VS
        };
        let addr = |lo: usize| ((sh[lo] as u64) | (sh[lo + 1] as u64) << 32) << 8;
        let vs = addr(vs_lo);
        let ps = addr(sh::SPI_SHADER_PGM_LO_PS);
        let mut color_targets = Vec::new();
        let mut target_addrs = Vec::new();
        for n in 0..8 {
            if c[context::CB_TARGET_MASK] >> (4 * n) & 0xF != 0
                && let Some(t) = ColorTarget::read(c, n)
            {
                color_targets.push((t.address, t.width, t.height));
                target_addrs.push(t.address);
            }
        }
        let depth_target = DepthTarget::read(c).map(|d| (d.address, d.width, d.height));
        if let Some((a, _, _)) = depth_target {
            target_addrs.push(a);
        }

        let draw_index = s.draws.len();
        let queue = s.current_queue;
        s.draws.push(DrawRecord {
            index: draw_index,
            queue,
            vs,
            ps,
            primitive: state.uconfig[uconfig::VGT_PRIMITIVE_TYPE],
            draw,
            color_targets,
            depth_target,
            viewport: Viewport::read(c),
            scissor: crate::regs::scissor(c),
        });

        if s.options.time_passes {
            let same = s
                .open_pass
                .as_ref()
                .is_some_and(|(t, _, _)| *t == target_addrs);
            if !same {
                s.flush_pass_timing();
                s.metal.sync();
                let _ = s.metal.take_gpu_time_secs();
                s.open_pass = Some((target_addrs, 0, 0.0));
            }
        }

        let t0 = Instant::now();
        s.metal.draw(host, state, draw);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        if let Some((_, count, cpu)) = s.open_pass.as_mut() {
            *count += 1;
            *cpu += cpu_ms;
        }

        if s.options.stop_after_draw == Some(draw_index) {
            s.flush_pass_timing();
            s.metal.sync();
            s.stopped = true;
        }
    }

    fn dispatch(&mut self, host: &dyn Host, state: &State, dispatch: Dispatch) {
        let mut s = self.lock();
        if s.stopped {
            return;
        }
        let cs = ((state.sh[sh::COMPUTE_PGM_LO] as u64)
            | (state.sh[sh::COMPUTE_PGM_LO + 1] as u64) << 32)
            << 8;
        let idx = s.dispatches.len();
        let queue = s.current_queue;
        s.dispatches.push(DispatchRecord {
            index: idx,
            queue,
            cs,
            groups: Some(dispatch.groups),
            indirect_at: None,
        });
        if s.options.time_passes {
            s.flush_pass_timing();
            s.metal.sync();
            let _ = s.metal.take_gpu_time_secs();
        }
        let t0 = Instant::now();
        s.metal.dispatch(host, state, dispatch);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        if s.options.time_passes {
            s.metal.sync();
            let gpu_ms = s.metal.take_gpu_time_secs() * 1e3;
            let index = s.timings.len();
            s.timings.push(PassTiming {
                index,
                kind: format!("compute {cs:#x}"),
                draws: 1,
                targets: Vec::new(),
                cpu_ms,
                gpu_ms,
            });
        }
    }

    fn dispatch_indirect(
        &mut self,
        host: &dyn Host,
        state: &State,
        at: u64,
        threads: bool,
    ) -> bool {
        let mut s = self.lock();
        if s.stopped {
            return true;
        }
        let cs = ((state.sh[sh::COMPUTE_PGM_LO] as u64)
            | (state.sh[sh::COMPUTE_PGM_LO + 1] as u64) << 32)
            << 8;
        if s.options.time_passes {
            s.flush_pass_timing();
            s.metal.sync();
            let _ = s.metal.take_gpu_time_secs();
        }
        let t0 = Instant::now();
        let handled = s.metal.dispatch_indirect(host, state, at, threads);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        if handled {
            let idx = s.dispatches.len();
            let queue = s.current_queue;
            s.dispatches.push(DispatchRecord {
                index: idx,
                queue,
                cs,
                groups: None,
                indirect_at: Some(at),
            });
            if s.options.time_passes {
                s.metal.sync();
                let gpu_ms = s.metal.take_gpu_time_secs() * 1e3;
                let index = s.timings.len();
                s.timings.push(PassTiming {
                    index,
                    kind: format!("compute-indirect {cs:#x}"),
                    draws: 1,
                    targets: Vec::new(),
                    cpu_ms,
                    gpu_ms,
                });
            }
        }
        handled
    }

    fn write_memory(&mut self, host: &dyn Host, write: &crate::MemoryWrite) -> bool {
        let mut s = self.lock();
        if s.stopped {
            return false;
        }
        s.metal.write_memory(host, write)
    }

    fn sync(&mut self) {
        self.lock().metal.sync();
    }

    fn flush(&mut self) {
        self.lock().metal.flush();
    }

    fn commit_point(&mut self) -> u64 {
        self.lock().metal.commit_point()
    }

    fn done(&mut self, point: u64) -> bool {
        self.lock().metal.done(point)
    }

    fn frame_done(&mut self, shown: Option<u64>) {
        let mut s = self.lock();
        if s.stopped {
            return;
        }
        s.flush_pass_timing();
        s.shown = shown;
        s.metal.frame_done(shown);
    }

    fn with_gds(&mut self, f: &mut dyn FnMut(&mut [u8])) -> bool {
        self.lock().metal.with_gds(f)
    }

    fn summary(&self) -> String {
        self.lock().metal.summary()
    }
}

/// Replays `capture` deterministically through a fresh Metal renderer.
pub fn replay(capture: &FrameCapture, options: ReplayOptions) -> Result<ReplayResult, String> {
    crate::with_pool(|| {
        let host = ReplayHost::from_capture(capture)?;
        let mut metal = Metal::new().ok_or("no Metal device")?;
        metal.set_wait_for_compiles(true);
        if !capture.textures.is_empty() {
            metal.restore_textures(&capture.textures)?;
        }
        if let Some(path) = &options.metal_capture {
            metal.begin_metal_capture(path)?;
        }
        let shared = ReplayBackend(std::sync::Arc::new(Mutex::new(ReplayState {
            metal,
            current_queue: 0,
            options: options.clone(),
            stopped: false,
            draws: Vec::new(),
            dispatches: Vec::new(),
            timings: Vec::new(),
            open_pass: None,
            shown: None,
        })));

        let mut processors: HashMap<u64, Processor> = HashMap::new();
        let mut gds_restored = false;
        for (&queue, saved) in &capture.queues {
            let mut p = Processor::default();
            p.backend = Some(Box::new(shared.clone()));
            p.restore(saved.clone(), !gds_restored);
            gds_restored = true;
            processors.insert(queue, p);
        }

        for sub in &capture.submissions {
            if shared.lock().stopped {
                break;
            }
            shared.lock().current_queue = sub.queue;
            let p = processors.entry(sub.queue).or_insert_with(|| {
                let mut p = Processor::default();
                p.backend = Some(Box::new(shared.clone()));
                p
            });
            for cb in &sub.buffers {
                // Write the submission's exact captured command buffer words before running.
                let bytes: Vec<u8> = cb.dwords.iter().flat_map(|w| w.to_le_bytes()).collect();
                host.write(cb.address, &bytes);
                p.run(&host, cb.address, cb.dwords.len() as u32)
                    .map_err(|e| format!("PM4 error during replay: {e}"))?;
                if shared.lock().stopped {
                    break;
                }
            }
            if !shared.lock().stopped {
                p.end_submission(&host, sub.ends_frame);
            }
        }

        let mut s = shared.lock();
        s.flush_pass_timing();
        s.metal.sync();
        if options.metal_capture.is_some() {
            s.metal.end_metal_capture();
        }
        let targets = s.metal.targets_snapshot();
        let depths = s.metal.depths_snapshot();
        let shown = s
            .shown
            .or_else(|| host.last_flip().and_then(|f| host.display_buffer(f.index)));
        let summary = s.metal.summary();
        Ok(ReplayResult {
            draws: std::mem::take(&mut s.draws),
            dispatches: std::mem::take(&mut s.dispatches),
            timings: std::mem::take(&mut s.timings),
            shown,
            targets,
            depths,
            summary,
        })
    })
}

/// Writes `width`x`height` RGBA8 pixels as a binary PPM (P6) file.
pub fn write_ppm(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    if rgba.len() != width as usize * height as usize * 4 {
        return Err(format!(
            "RGBA length {} does not match {width}x{height}",
            rgba.len()
        ));
    }
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.reserve(width as usize * height as usize * 3);
    for px in rgba.as_chunks::<4>().0 {
        ppm.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, ppm).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Reads a binary PPM (P6, maxval 255) file as `(width, height, rgb_bytes)`.
pub fn read_ppm(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut pos = 0usize;
    let next_token = |pos: &mut usize| -> Option<String> {
        while *pos < data.len() {
            if data[*pos] == b'#' {
                while *pos < data.len() && data[*pos] != b'\n' {
                    *pos += 1;
                }
            } else if data[*pos].is_ascii_whitespace() {
                *pos += 1;
            } else {
                break;
            }
        }
        let start = *pos;
        while *pos < data.len() && !data[*pos].is_ascii_whitespace() && data[*pos] != b'#' {
            *pos += 1;
        }
        (start < *pos).then(|| String::from_utf8_lossy(&data[start..*pos]).into_owned())
    };
    let magic = next_token(&mut pos).ok_or("empty PPM")?;
    if magic != "P6" {
        return Err(format!("expected P6 PPM, got {magic}"));
    }
    let width: u32 = next_token(&mut pos)
        .and_then(|t| t.parse().ok())
        .ok_or("bad PPM width")?;
    let height: u32 = next_token(&mut pos)
        .and_then(|t| t.parse().ok())
        .ok_or("bad PPM height")?;
    let maxval: u32 = next_token(&mut pos)
        .and_then(|t| t.parse().ok())
        .ok_or("bad PPM maxval")?;
    if maxval != 255 {
        return Err(format!("unsupported PPM maxval {maxval}"));
    }
    if pos >= data.len() || !data[pos].is_ascii_whitespace() {
        return Err("missing whitespace after PPM header".into());
    }
    pos += 1;
    let expected = width as usize * height as usize * 3;
    let rgb = data
        .get(pos..pos + expected)
        .ok_or("truncated PPM pixel data")?;
    Ok((width, height, rgb.to_vec()))
}

/// Difference summary between a rendered image and a golden PPM.
#[derive(Clone, Copy, Debug)]
pub struct ImageDiff {
    pub max_channel_diff: u8,
    pub differing_pixels: usize,
    pub total_pixels: usize,
    pub mean_abs_error: f64,
}

/// Compares RGBA8 pixels against RGB8 golden pixels.
pub fn compare_rgba_to_ppm(
    width: u32,
    height: u32,
    rgba: &[u8],
    golden_w: u32,
    golden_h: u32,
    golden_rgb: &[u8],
) -> Result<ImageDiff, String> {
    if (width, height) != (golden_w, golden_h) {
        return Err(format!(
            "size mismatch: got {width}x{height}, golden is {golden_w}x{golden_h}"
        ));
    }
    let total_pixels = width as usize * height as usize;
    if rgba.len() != total_pixels * 4 || golden_rgb.len() != total_pixels * 3 {
        return Err("buffer length mismatch".into());
    }
    let mut max_diff = 0u8;
    let mut differing = 0usize;
    let mut sum_err = 0u64;
    for (a, g) in rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden_rgb.as_chunks::<3>().0)
    {
        let mut pix_diff = 0u8;
        for c in 0..3 {
            let d = a[c].abs_diff(g[c]);
            pix_diff = pix_diff.max(d);
            sum_err += d as u64;
        }
        max_diff = max_diff.max(pix_diff);
        if pix_diff > 0 {
            differing += 1;
        }
    }
    Ok(ImageDiff {
        max_channel_diff: max_diff,
        differing_pixels: differing,
        total_pixels,
        mean_abs_error: if total_pixels == 0 {
            0.0
        } else {
            sum_err as f64 / (total_pixels as f64 * 3.0)
        },
    })
}
