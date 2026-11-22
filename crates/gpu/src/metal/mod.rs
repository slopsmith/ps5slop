//! The Metal renderer: draws and dispatches from the command processor
//! become Metal work on guest memory, which Metal sees in place through
//! no-copy buffers. Render targets live in Metal textures keyed by guest
//! address, so a flip shows them without a CPU copy. Structure after
//! SharpEmu-mac's Metal backend (GPL-2.0-or-later, `docs/macos/metal-backend.md`).

use std::collections::HashMap;

use std::ptr::NonNull;

use std::sync::{Arc, Mutex, OnceLock};

use objc2::rc::Retained;

use objc2::runtime::ProtocolObject;

use objc2_foundation::NSString;

use objc2_metal::*;

use crate::compile::Task;

use crate::regs::{
    self, Blend, ColorTarget, DepthState, DepthTarget, Image, Sampler, Viewport, context, sh,
    uconfig,
};

use crate::tiling::{Layout, Mip, TileMode};

use crate::{Backend, Draw, Host, State};

use gfx10::translate::{
    self as shader, Access, Resource, Source, Stage, TextureKind, Translated, binding,
};

mod bind;
mod commands;
mod compute;
mod detile;
mod draw;
mod formats;
mod htile;
mod images;
mod inspect;
mod memory;
mod programs;
mod readback;
mod saved;
mod targets;
#[cfg(test)]
mod tests;

use bind::*;
use compute::*;
use detile::*;
use draw::*;
use formats::*;
use htile::*;
use images::*;
pub use inspect::{DUMP_TARGETS, RELOAD, SKIP, SNAP, TRACE_NEXT, all_frames, frame};
use inspect::{Snap, frames, record, skipped};
use memory::*;
use programs::*;
use readback::{read_depth, read_raw};
pub use readback::{read_frame, read_rgba};
use saved::*;
use targets::*;

type Device = ProtocolObject<dyn MTLDevice>;

type Buffer = ProtocolObject<dyn MTLBuffer>;

type Texture = ProtocolObject<dyn MTLTexture>;

type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;

type RenderEncoder = ProtocolObject<dyn MTLRenderCommandEncoder>;

type RenderPipeline = ProtocolObject<dyn MTLRenderPipelineState>;

type ComputePipeline = ProtocolObject<dyn MTLComputePipelineState>;

type Function = ProtocolObject<dyn MTLFunction>;

/// A Metal texture other threads may hold.
#[derive(Clone)]
pub struct SharedTexture(pub Retained<Texture>);

// SAFETY: Metal textures may be used from any thread.
unsafe impl Send for SharedTexture {}

// SAFETY: as above.
unsafe impl Sync for SharedTexture {}

/// How many different problems are reported; repeats are not logged.
const REPORTS: usize = 64;

/// The Metal side of the GPU.
pub struct Metal {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Keeps the guest memory windows resident between command buffers.
    resident: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    command: Option<Retained<CommandBuffer>>,
    /// Committed command buffers not waited for yet, with their commit time
    /// and the number of the commit that sent them.
    inflight: Vec<(Retained<CommandBuffer>, f64, u64)>,
    /// Commits so far, for [`Backend::commit_point`].
    commits: u64,
    /// Texture uploads for the work in `command`, committed just before it
    /// so they do not split its render passes.
    upload: Option<Retained<CommandBuffer>>,
    /// Where this backend is in the log of replaced guest pages.
    replaced_mark: u64,
    pass: Option<OpenPass>,
    snap: Option<Snap>,
    depths: HashMap<u64, Retained<Texture>>,
    /// Stencil plane addresses and the depth target they belong to.
    stencils: HashMap<u64, u64>,
    /// Depth-layout images a program wrote since their depth texture was
    /// last used: the depth texture takes their contents first.
    depth_stores: std::collections::HashSet<u64>,
    /// Addresses a colour target or storage image wrote after any depth target
    /// there, so sampling them must not read the depth texture.
    color_newer: std::collections::HashSet<u64>,
    shown: HashMap<u64, Retained<Texture>>,
    swizzles: HashMap<(usize, [u8; 4]), Retained<Texture>>,
    depth_states: HashMap<(DepthState, bool), Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    windows: Vec<Window>,
    /// One window from the start of the host's GPU area (see
    /// [`Host::gpu_area`]) up to what programs have pointed into so far.
    area: Option<Window>,
    /// The furthest programs have pointed into the GPU area, so a window
    /// made again after a mapping change reaches as far as earlier ones.
    area_reach: u64,
    /// Bytes of windows dropped since the last commit, which the open
    /// command buffer may still hold.
    stale: u64,
    /// The command buffer last reported as stuck, so each is reported once.
    reported_stuck: Option<usize>,
    /// Guest memory that work encoded since the last wait may write, so the
    /// CPU must wait for it before reading there (see `settle_for`).
    written: Written,
    /// Waits [`Metal::settle_for`] made, by what read the memory.
    settles: std::collections::BTreeMap<&'static str, u64>,
    /// HTILE words checked on the GPU since work that may write them was
    /// last encoded or waited for: later draws find them expanded.
    gpu_htiles: std::collections::HashSet<u64>,
    /// HTILE checks made on the GPU.
    htile_checks: u64,
    htile_helpers: Option<HtileHelpers>,
    indirect_sizes: Option<Retained<ComputePipeline>>,
    /// By colour formats, depth format and stencil.
    htile_pipelines: HashMap<([usize; 8], usize, bool), Retained<RenderPipeline>>,
    /// The uncommitted main command buffer holds work that writes guest
    /// memory, so detiles from guest memory must run after it, not before.
    pending_writes: bool,
    /// Command buffers Metal reported as failed.
    failed_commands: u64,
    /// Loop guard words of compute dispatches (see [`GUARD_SLOTS`]), the
    /// next slot, the slots of dispatches not yet checked with their program,
    /// and how many dispatches gave up.
    guard: Option<Retained<Buffer>>,
    /// With `PS5SLOP_GPU_FENCE`, orders encoders that read guest memory after
    /// those that wrote it (see [`Metal::wait_memory`]); whether one updated
    /// it yet, and whether the open pass writes guest memory.
    fence: Option<Retained<ProtocolObject<dyn MTLFence>>>,
    fence_updated: bool,
    pass_memory: bool,
    guard_next: usize,
    guarded: Vec<(usize, u64, [u32; 16])>,
    guard_trips: u64,
    /// Values of `WRITE_DATA` blits: a shared buffer filled from the front and
    /// never rewritten, replaced by a fresh one when full.
    write_staging: Option<(Retained<Buffer>, usize)>,
    /// `WRITE_DATA`s written in GPU order.
    ordered_writes: u64,
    /// Shared memory handed out front to back for data copied for the GPU,
    /// and how much of it is used; replaced when full.
    scratch: Option<(Retained<Buffer>, usize)>,
    /// The global data share, made on first use.
    gds: Option<Retained<Buffer>>,
    /// Pixel inputs each (vertex, pixel) program pair lacks, by program ids.
    missing_inputs: HashMap<(u64, u64), Vec<u32>>,
    generation: u64,
    targets: HashMap<u64, Target>,
    /// Stand-in colour targets, by size, for draws that only store to memory.
    blank_targets: HashMap<(u32, u32), Retained<Texture>>,
    images: HashMap<(Image, TextureKind), (u64, Retained<Texture>)>,
    /// A digest of each image's guest bytes as last detiled, while no encoded work could write them then.
    image_digests: HashMap<(Image, TextureKind), u64>,
    /// Images found unchanged in a later frame, found changed, and detiled without a digest.
    image_checks: [u64; 3],
    /// 2D-array and 3D images programs write, by guest address.
    volumes: HashMap<u64, Retained<Texture>>,
    volume_sources: HashMap<u64, VolumeSource>,
    /// Addresses of volumes a colour target was drawn to since a program last
    /// wrote the volume: sampling them reads the target.
    volume_older: std::collections::HashSet<u64>,
    /// The last colour target with CMASK or DCC at each address, so a program
    /// storing to it first applies a fast clear still pending there.
    color_meta: HashMap<u64, ColorTarget>,
    /// Zero-filled stand-ins for depth-layout images no depth target backs.
    blank: HashMap<(Image, TextureKind), Retained<Texture>>,
    epoch: u64,
    detilers: HashMap<(TileMode, u32, Detile), Retained<ComputePipeline>>,
    programs: HashMap<(u64, u64), Option<Arc<Program>>>,
    variants: HashMap<VariantKey, Arc<Task<Retained<Function>>>>,
    pipelines: HashMap<PipelineKey, Arc<Task<Retained<RenderPipeline>>>>,
    /// The first and last frame each colour or depth target (address,
    /// pixel format, width) was drawn to.
    drawn: HashMap<(u64, usize, usize), (u64, u64)>,
    /// What this frame may still spend waiting for pipelines before skipping draws.
    compile_budget: std::time::Duration,
    /// Draws and dispatches that waited for a pipeline: how many, and how long.
    compile_waits: [(u64, std::time::Duration); 2],
    samplers: HashMap<Sampler, Retained<ProtocolObject<dyn MTLSamplerState>>>,
    reports: diag::Distinct,
    /// Images no texture could be made for: (read as, access, T# type,
    /// swizzle mode, format) and how often.
    image_misses: HashMap<(TextureKind, &'static str, u32, u32, u32), u64>,
    null: HashMap<TextureKind, Retained<Texture>>,
    pub draws: u64,
    pub dispatches: u64,
    pub skipped: u64,
    reasons: HashMap<&'static str, u64>,
    pass_ends: HashMap<u32, u64>,
    /// The frames `PS5SLOP_GPU_TRACE` names (`n` or `first-last`), and the
    /// trace lines of the current draw.
    trace_frames: Option<(u64, u64)>,
    /// The context and uconfig registers at the last traced draw, for
    /// `PS5SLOP_TRACE_REGS`.
    traced_regs: Vec<u32>,
    trace: Option<Vec<String>>,
    /// `PS5SLOP_TRACE_PIXELS`: a visibility buffer counting the samples
    /// each traced draw passes, and the next free slot.
    pixels: Option<(Retained<Buffer>, usize)>,
    fan: Option<Retained<Buffer>>,
}

// SAFETY: a `Metal` is used by one thread at a time; Metal objects may move
// between threads, and an open encoder is never used concurrently.
unsafe impl Send for Metal {}

impl Metal {
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;
        let config = &diag::config().gpu;
        let fence = config.fence.then(|| device.newFence()).flatten();
        let resident = config
            .residency_set
            .then(|| {
                device
                    .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
                    .ok()
            })
            .flatten();
        if let Some(r) = &resident {
            queue.addResidencySet(r);
        }
        Some(Self {
            device,
            queue,
            resident,
            command: None,
            inflight: Vec::new(),
            commits: 0,
            upload: None,
            replaced_mark: 0,
            pass: None,
            snap: None,
            depths: HashMap::new(),
            stencils: HashMap::new(),
            depth_stores: Default::default(),
            color_newer: Default::default(),
            shown: HashMap::new(),
            swizzles: HashMap::new(),
            depth_states: HashMap::new(),
            windows: Vec::new(),
            area: None,
            area_reach: 0,
            stale: 0,
            reported_stuck: None,
            written: Written::Nothing,
            settles: Default::default(),
            gpu_htiles: Default::default(),
            htile_checks: 0,
            htile_helpers: None,
            indirect_sizes: None,
            htile_pipelines: HashMap::new(),
            pending_writes: false,
            failed_commands: 0,
            guard: None,
            fence,
            fence_updated: false,
            pass_memory: false,
            guard_next: 0,
            guarded: Vec::new(),
            guard_trips: 0,
            write_staging: None,
            ordered_writes: 0,
            missing_inputs: HashMap::new(),
            scratch: None,
            gds: None,
            generation: 0,
            targets: HashMap::new(),
            blank_targets: HashMap::new(),
            images: HashMap::new(),
            image_digests: HashMap::new(),
            image_checks: [0; 3],
            volumes: HashMap::new(),
            volume_sources: HashMap::new(),
            volume_older: Default::default(),
            color_meta: HashMap::new(),
            blank: HashMap::new(),
            epoch: 0,
            detilers: HashMap::new(),
            programs: HashMap::new(),
            variants: HashMap::new(),
            pipelines: HashMap::new(),
            drawn: HashMap::new(),
            compile_budget: COMPILE_BUDGET,
            compile_waits: Default::default(),
            samplers: HashMap::new(),
            reports: diag::Distinct::new(REPORTS),
            image_misses: HashMap::new(),
            null: HashMap::new(),
            draws: 0,
            dispatches: 0,
            skipped: 0,
            reasons: HashMap::new(),
            pass_ends: HashMap::new(),
            traced_regs: Vec::new(),
            trace_frames: diag::config().gpu.trace_frames,
            trace: None,
            pixels: None,
            fan: None,
        })
    }

    /// Counts a wait for a pipeline that began at `since`, by a draw (0) or a dispatch (1).
    fn note_wait(&mut self, kind: usize, since: std::time::Instant) {
        let waited = since.elapsed();
        crate::timing::add_raw(crate::timing::Part::CompileWait, waited.as_nanos() as u64);
        self.compile_waits[kind].0 += 1;
        self.compile_waits[kind].1 += waited;
    }

    fn note(&mut self, what: impl FnOnce() -> String) {
        if let Some(t) = &mut self.trace {
            t.push(what());
        }
    }

    /// Logs `what` the first time it happens, until [`REPORTS`] different reports were made.
    fn report(&mut self, level: tracing::Level, what: String) {
        if !self.reports.first(&what) {
            return;
        }
        match level {
            tracing::Level::ERROR => tracing::error!("{what}"),
            tracing::Level::WARN => tracing::warn!("{what}"),
            _ => tracing::info!("{what}"),
        }
    }
}

impl Backend for Metal {
    fn save_textures(&mut self) -> Result<Vec<u8>, String> {
        self.flush_and_wait();
        let mut out = Vec::new();
        let mut skipped = 0;
        let mut keep = |r: Result<SavedTexture, String>| match r {
            Ok(t) => out.push(t),
            Err(e) => {
                skipped += 1;
                tracing::warn!("save state: {e}");
            }
        };
        for (&a, t) in &self.targets {
            keep(self.saved_texture(CACHE_TARGET, a, &t.texture, t.pixel_format));
        }
        for (&a, t) in &self.depths {
            keep(self.saved_texture(CACHE_DEPTH, a, t, t.pixelFormat()));
        }
        for (&a, t) in &self.volumes {
            keep(self.saved_texture(CACHE_VOLUME, a, t, t.pixelFormat()));
        }
        if skipped > 0 {
            tracing::warn!("save state: {skipped} GPU textures not saved");
        }
        out.sort_by_key(|t| (t.cache, t.address));
        Ok(savestate::to_bytes(&out))
    }

    fn restore_textures(&mut self, saved: &[u8]) -> Result<(), String> {
        let saved: Vec<SavedTexture> = savestate::from_bytes(saved).map_err(|e| e.0)?;
        for s in &saved {
            let t = self.make_saved(s)?;
            match s.cache {
                CACHE_TARGET => {
                    let target = Target {
                        texture: t,
                        pixel_format: MTLPixelFormat(s.target_format as _),
                    };
                    self.targets.insert(s.address, target);
                }
                CACHE_DEPTH => {
                    self.depths.insert(s.address, t);
                }
                _ => {
                    self.volumes.insert(s.address, t);
                }
            }
        }
        Ok(())
    }

    fn with_gds(&mut self, f: &mut dyn FnMut(&mut [u8])) -> bool {
        let Some(buffer) = self.gds_buffer() else {
            return false;
        };
        // SAFETY: a shared buffer's contents stay mapped while `buffer` lives;
        // the processor touches it only after the GPU work before is done.
        f(unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast(), GDS_BYTES) });
        true
    }

    fn draw(&mut self, host: &dyn Host, state: &State, draw: Draw) {
        self.trace = self.tracing().then(Vec::new);
        if self.trace.is_some() && diag::config().gpu.trace_regs {
            self.note_changed_registers(state);
        }
        let result = self.try_draw(host, state, draw);
        self.end_trace(|| format!("draw {draw:?}"), result);
        match result {
            Ok(()) => self.draws += 1,
            Err(why) => {
                self.skipped += 1;
                *self.reasons.entry(why).or_default() += 1;
            }
        }
    }

    fn dispatch(&mut self, host: &dyn Host, state: &State, dispatch: crate::Dispatch) {
        self.trace = self.tracing().then(Vec::new);
        let groups = dispatch.groups;
        let result = self.try_dispatch(host, state, Grid::Direct(dispatch));
        self.end_trace(|| format!("dispatch {groups:?}"), result);
        match result {
            Ok(()) => self.dispatches += 1,
            Err(why) => {
                self.skipped += 1;
                *self.reasons.entry(why).or_default() += 1;
            }
        }
    }

    fn dispatch_indirect(
        &mut self,
        host: &dyn Host,
        state: &State,
        at: u64,
        threads: bool,
    ) -> bool {
        self.trace = self.tracing().then(Vec::new);
        let result = self.try_dispatch(host, state, Grid::Indirect { at, threads });
        if result == Err(INDIRECT_ON_CPU) {
            self.trace = None;
            return false;
        }
        self.end_trace(|| format!("dispatch with dimensions at {at:#x}"), result);
        match result {
            Ok(()) => self.dispatches += 1,
            Err(why) => {
                self.skipped += 1;
                *self.reasons.entry(why).or_default() += 1;
            }
        }
        true
    }

    fn write_memory(&mut self, host: &dyn Host, write: &crate::MemoryWrite) -> bool {
        let dest = write.dest;
        let words = match (write.fixed, write.values.last()) {
            (_, None) => return false,
            (true, Some(&v)) => vec![v],
            (false, _) => write.values.to_vec(),
        };
        let len = 4 * words.len();
        if !dest.is_multiple_of(4) || diag::config().gpu.queued_writes {
            return false;
        }
        // Only the GPU area has one buffer object that programs share, and
        // Metal orders work only within one object.
        let target = match host.gpu_area() {
            Some((start, end)) if start <= dest && dest + len as u64 <= end => self
                .area_window(host, start, end, dest + len as u64)
                .map(|b| (b, (dest - start) as usize)),
            _ => None,
        };
        let Some((buffer, offset)) = target else {
            return false;
        };
        const STAGING: usize = 1 << 20;
        if self
            .write_staging
            .as_ref()
            .is_none_or(|(_, used)| used + len > STAGING)
        {
            let Some(b) = self
                .device
                .newBufferWithLength_options(STAGING, MTLResourceOptions::StorageModeShared)
            else {
                return false;
            };
            self.write_staging = Some((b, 0));
        }
        let Some((source, used)) = self.write_staging.as_mut() else {
            return false;
        };
        let (source, from) = (source.clone(), *used);
        *used += len;
        // SAFETY: `[from, from + len)` lies inside the shared buffer and no
        // earlier write used it; the GPU reads it only once the blit runs.
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                source.contents().as_ptr().cast::<u8>().add(from).cast(),
                words.len(),
            )
        };
        self.end_pass();
        let command = self.command();
        let Some(blit) = command.blitCommandEncoder() else {
            return false;
        };
        if let Some(f) = self.wait_memory() {
            blit.waitForFence(f);
        }
        // SAFETY: both ranges lie inside their buffers; offsets and size are multiples of 4.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &source, from, &buffer, offset, len,
            )
        };
        if let Some(f) = self.fence.as_deref() {
            blit.updateFence(f);
            self.fence_updated = true;
        }
        blit.endEncoding();
        self.pending_writes = true;
        self.ordered_writes += 1;
        if self.ordered_writes.is_power_of_two() {
            tracing::debug!("{} writes in GPU order", self.ordered_writes);
        }
        true
    }

    fn sync(&mut self) {
        self.flush_and_wait();
    }

    fn flush(&mut self) {
        self.commit();
    }

    fn commit_point(&mut self) -> u64 {
        self.commit();
        self.commits
    }

    fn done(&mut self, point: u64) -> bool {
        self.retire_completed();
        self.inflight.first().is_none_or(|(_, _, n)| *n > point)
    }

    fn summary(&self) -> String {
        let mut reasons: Vec<_> = self.reasons.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1));
        let mib = |t: &Retained<Texture>| MTLAllocation::allocatedSize(&**t) as u64;
        let image_bytes: u64 = self.images.values().map(|(_, t)| mib(t)).sum();
        let target_bytes: u64 = self.targets.values().map(|t| mib(&t.texture)).sum();
        let mut misses: Vec<_> = self.image_misses.iter().collect();
        misses.sort_by(|a, b| b.1.cmp(a.1));
        let misses = if misses.is_empty() {
            String::new()
        } else {
            format!(", images not supported {misses:x?}")
        };
        let mut ends: Vec<_> = self.pass_ends.iter().collect();
        ends.sort_by(|a, b| b.1.cmp(a.1));
        let misses = format!(
            "{misses}, render passes ended at lines {ends:?}, images kept {} changed {} without digest {}",
            self.image_checks[0], self.image_checks[1], self.image_checks[2]
        );
        format!(
            "{} drawn, {} dispatched, {} skipped {reasons:?}{misses}, {} writes in GPU order, {} command buffers failed, {} dispatches gave up loops; caches: {} images ({} MiB), {} targets ({} MiB), {} depths, {} volumes, {} swizzle views, {} windows, {} pipelines, {} compiles queued; compile waits: {} draws ({} ms), {} dispatches ({} ms); waits to read memory programs wrote {:?}, {} HTILE checks on the GPU",
            self.draws,
            self.dispatches,
            self.skipped,
            self.ordered_writes,
            self.failed_commands,
            self.guard_trips,
            self.images.len(),
            image_bytes >> 20,
            self.targets.len(),
            target_bytes >> 20,
            self.depths.len(),
            self.volumes.len(),
            self.swizzles.len(),
            self.windows.len(),
            self.pipelines.len(),
            crate::compile::queued(),
            self.compile_waits[0].0,
            self.compile_waits[0].1.as_millis(),
            self.compile_waits[1].0,
            self.compile_waits[1].1.as_millis(),
            self.settles,
            self.htile_checks
        )
    }

    fn frame_done(&mut self, shown: Option<u64>) {
        if let Some(addr) = shown
            && let Some(t) = self.targets.get(&addr)
        {
            let src = t.texture.clone();
            let copy = match self.shown.get(&addr) {
                Some(c)
                    if c.width() == src.width()
                        && c.height() == src.height()
                        && c.pixelFormat() == src.pixelFormat() =>
                {
                    c.clone()
                }
                _ => {
                    let c = self.new_texture(
                        src.pixelFormat(),
                        src.width() as u32,
                        src.height() as u32,
                        false,
                    );
                    let Some(c) = c else { return };
                    self.shown.insert(addr, c.clone());
                    c
                }
            };
            self.end_pass();
            let cb = self.command();
            if let Some(blit) = cb.blitCommandEncoder() {
                unsafe { blit.copyFromTexture_toTexture(&src, &copy) };
                blit.endEncoding();
            }
            self.flush_and_wait();
            if self.tracing() {
                let (a, b) = (
                    Retained::as_ptr(&src) as usize,
                    Retained::as_ptr(&copy) as usize,
                );
                tracing::info!(target: "gpu::trace", "shown {addr:#x} copied from {a:#x} into {b:#x}");
            }
            if let Ok(mut f) = frames().lock() {
                f.insert(addr, SharedTexture(copy));
            }
        } else {
            self.flush_and_wait();
        }
        if let Some(snap) = self.snap.take() {
            Self::write_snap(snap);
        }
        self.frame_tools();
        if self.tracing() {
            if let Some((b, n)) = self.pixels.take() {
                let counts =
                    unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const u64, n) };
                for (slot, c) in counts.iter().enumerate() {
                    tracing::info!(target: "gpu::trace", "pixels slot {slot}: {c}");
                }
            }
            tracing::info!(
                target: "gpu::trace",
                "frame {} done, showing {shown:x?}",
                self.epoch
            );
        }
        self.epoch += 1;
        self.compile_budget = COMPILE_BUDGET;
        self.swizzles.clear();
        let epoch = self.epoch;
        self.images
            .retain(|_, (used, _)| epoch - *used < IMAGE_LIFETIME);
        let n = TRACE_NEXT.swap(0, std::sync::atomic::Ordering::Relaxed);
        if n > 0 {
            self.trace_frames = Some((self.epoch, self.epoch + n - 1));
        }
    }
}
