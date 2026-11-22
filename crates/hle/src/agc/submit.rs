//! Submitting command buffers: `sceAgcDriverSubmitDcb` and friends hand
//! them to a GPU thread per queue (graphics, and each compute owner), which
//! runs them through the `gpu` crate's command processor in submission
//! order. The queues share one Metal renderer, and a queue waiting on a
//! label does not hold up the others. Ported in shape from SharpEmu's
//! `AgcExports.Submissions.cs`.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::{ERROR_INVALID_ARGUMENT, ret};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Kernel, Outcome};

/// The event filter of GPU interrupts (Orbis's `EVFILT_GRAPHICS_CORE`, as
/// public emulators number it; unverified).
pub(crate) const GRAPHICS_FILTER: i16 = -14;
/// The event ident the graphics queue's end-of-pipe interrupts use.
const GRAPHICS_EOP_EVENT: u64 = 0;
/// Submissions waiting for the GPU before a submitter blocks, as a full ring
/// would block it (our choice of depth, unverified).
const MAX_QUEUED: usize = 1;
/// SharpEmu's limit on one submission's size.
const MAX_DWORDS: u32 = 0xF_FFFF;

/// How long a submitter waits for the submission before its own at most.
const THROTTLE_LIMIT: Duration = Duration::from_secs(5);
/// How often an idle GPU thread checks whether ended submissions' GPU work
/// is done, so their releases take effect.
const RETIRE_POLL: Duration = Duration::from_micros(100);

/// Command buffers to run, in order, on one queue.
struct Job {
    buffers: Vec<(u64, u32)>,
    /// One of the buffers holds a flip packet, so this ends a frame.
    ends_frame: bool,
}

/// Where a queue's jobs go, and how far its GPU thread has got.
type Queue = (SyncSender<Job>, Arc<Progress>);

/// How far a queue's GPU thread has got, for throttling submitters.
#[derive(Default)]
struct Progress {
    counts: Mutex<Counts>,
    changed: Condvar,
    /// Held from numbering a submission until it is queued, so both keep one order.
    order: Mutex<()>,
    /// A save state wants the processor's state once the queue is idle.
    wants_state: std::sync::atomic::AtomicBool,
    /// That state, with the number of submissions finished before it.
    state: Mutex<Option<(u64, gpu::Saved)>>,
}

/// How often an idle GPU thread looks whether a save state wants its state.
const IDLE_POLL: Duration = Duration::from_millis(20);

#[derive(Default)]
struct Counts {
    submitted: u64,
    finished: u64,
    /// The last submission that ended a frame.
    frame_end: u64,
    /// The GPU thread polls memory that the CPU may have to change.
    polling: bool,
}

impl Progress {
    fn update(&self, f: impl FnOnce(&mut Counts)) {
        f(&mut lock(&self.counts));
        self.changed.notify_all();
    }

    /// Counts a submission, which ends a frame if `ends_frame`, and returns
    /// the last submission before it that ended one.
    fn add(&self, ends_frame: bool) -> u64 {
        let mut c = lock(&self.counts);
        c.submitted += 1;
        let previous = c.frame_end;
        if ends_frame {
            c.frame_end = c.submitted;
        }
        previous
    }

    /// Waits until submission `n` has finished, the GPU thread waits on
    /// memory, or the limit passes, unless the throttle is off.
    fn wait_for(&self, n: u64, exiting: impl Fn() -> bool) {
        if diag::config().gpu.no_throttle {
            return;
        }
        let start = Instant::now();
        let mut c = lock(&self.counts);
        while c.finished < n && !c.polling && !exiting() && start.elapsed() < THROTTLE_LIMIT {
            jit::blocking();
            c = match self.changed.wait_timeout(c, Duration::from_millis(50)) {
                Ok((c, _)) => c,
                Err(e) => e.into_inner().0,
            };
        }
    }
}

/// The graphics queue's key; compute queues are keyed by their owner.
const GRAPHICS: u64 = u64::MAX;

/// The GPU threads' queues, each started by its first submission.
#[derive(Default)]
pub(super) struct Gpu {
    jobs: Mutex<HashMap<u64, Queue>>,
    backend: Arc<OnceLock<Option<gpu::SharedBackend>>>,
    /// Flip packets built and not yet seen in a submitted buffer, by address.
    flips: Mutex<Vec<u64>>,
    /// Processor states from a loaded save state, by queue, for queues
    /// not started yet.
    restored: Mutex<HashMap<u64, gpu::Saved>>,
    gds_restored: std::sync::atomic::AtomicBool,
    /// GPU-only textures from a loaded save state, for the backend once made.
    textures: Mutex<Option<Vec<u8>>>,
    /// One-frame GPU capture queue shared across all GPU threads and background writers.
    capture: gpu::capture::CaptureManager,
}

impl Gpu {
    /// Schedules a one-frame GPU capture to `path` starting at the next flip,
    /// queueing it if another capture is already armed, active or writing.
    pub(super) fn request_capture(&self, path: std::path::PathBuf) -> String {
        self.capture.request(path).describe()
    }

    /// Waits for any background capture write to finish on process exit.
    pub(super) fn finish_captures(&self) {
        self.capture.finish_on_exit();
    }

    /// `sceAgcDcbSetFlip` and friends built a flip packet at `at`.
    pub(super) fn flip_built(&self, at: u64) {
        let mut flips = lock(&self.flips);
        if flips.len() >= 16 {
            flips.remove(0);
        }
        flips.push(at);
    }

    /// Whether one of `buffers` holds a flip packet built for it.
    fn takes_flip(&self, buffers: &[(u64, u32)]) -> bool {
        let mut flips = lock(&self.flips);
        let before = flips.len();
        flips.retain(|&at| {
            !buffers
                .iter()
                .any(|&(addr, dwords)| (addr..addr + 4 * dwords as u64).contains(&at))
        });
        flips.len() != before
    }
}

pub(super) static FUNCTIONS: &[Function] = &[
    d("sceAgcDriverSubmitDcb", submit_dcb),
    d("sceAgcDriverAgrSubmitDcb", submit_dcb),
    d("sceAgcDriverSubmitAcb", submit_acb),
    d("sceAgcDriverSubmitMultiDcbs", submit_multi_dcbs),
    d("sceAgcDriverAgrSubmitMultiDcbs", submit_multi_dcbs),
    d("sceAgcDriverSubmitMultiAcbs", submit_multi_acbs),
    Function {
        library: "libSceAgc",
        name: "sceAgcSuspendPoint",
        func: suspend_point,
    },
];

const fn d(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgcDriver",
        name,
        func,
    }
}

/// The guest memory, events and display, as the command processor sees them.
struct Host {
    kernel: Arc<Kernel>,
    mem: memory::Memory,
    progress: Arc<Progress>,
}

impl gpu::Host for Host {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.mem.read(addr, buf).is_ok()
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        self.mem.write(addr, data).is_ok()
    }

    fn interrupt(&self, context: u32) {
        self.kernel
            .trigger_event(GRAPHICS_EOP_EVENT, GRAPHICS_FILTER, context as u64);
    }

    fn flip(&self, flip: gpu::Flip) {
        self.kernel.flip(flip.index, flip.arg as u64);
    }

    fn exiting(&self) -> bool {
        self.kernel.has_exited()
    }

    fn extent(&self, addr: u64) -> Option<(u64, u64)> {
        self.mem.mapped_extent(addr)
    }

    fn image(&self, addr: u64) -> Option<(u64, u64)> {
        self.kernel.module_image(addr)
    }

    fn host_pointer(&self, addr: u64) -> Option<*mut u8> {
        (addr < self.mem.size()).then(|| self.mem.host_base().wrapping_add(addr as usize))
    }

    fn display_buffer(&self, index: i32) -> Option<u64> {
        self.kernel.display_buffer(index)
    }

    fn generation(&self) -> u64 {
        self.mem.generation()
    }

    fn replaced_since(&self, since: u64) -> (u64, Option<Vec<(u64, u64)>>) {
        self.mem.replaced_since(since)
    }

    fn gpu_area(&self) -> Option<(u64, u64)> {
        Some(crate::dmem::GPU_AREA)
    }

    fn back(&self, start: u64, end: u64) -> bool {
        self.mem.back(start, end).is_ok()
    }

    fn waiting(&self, polling: bool) {
        self.progress.update(|c| c.polling = polling);
    }
}

fn new_backend() -> Option<gpu::SharedBackend> {
    #[cfg(target_os = "macos")]
    if diag::config().gpu.render {
        let metal = gpu::metal::Metal::new()?;
        return Some(gpu::SharedBackend(Arc::new(Mutex::new(Box::new(metal)))));
    }
    None
}

fn gpu_thread(
    host: Host,
    jobs: Receiver<Job>,
    backend: Arc<OnceLock<Option<gpu::SharedBackend>>>,
    capture: gpu::capture::CaptureManager,
    queue_id: u64,
    name: String,
    restored: Option<(gpu::Saved, bool)>,
) {
    let mut processor = gpu::Processor::default();
    if let Some(b) = backend.get_or_init(new_backend) {
        processor.backend = Some(Box::new(b.clone()));
    }
    if let Some((saved, gds)) = restored {
        processor.restore(saved, gds);
    }
    let report = tracing::enabled!(target: "gpu::stats", tracing::Level::INFO);
    let mut last_report = std::time::Instant::now();
    loop {
        let idle = gpu::timing::start();
        let job = loop {
            let retiring = processor.retire(&host);
            let p = &host.progress;
            if !retiring
                && p.wants_state
                    .swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                processor.finish(&host);
                let finished = lock(&p.counts).finished;
                *lock(&p.state) = Some((finished, processor.saved()));
                p.changed.notify_all();
            }
            let poll = if retiring { RETIRE_POLL } else { IDLE_POLL };
            match jobs.recv_timeout(poll) {
                Err(RecvTimeoutError::Timeout) => continue,
                job => break job.ok(),
            }
        };
        let Some(job) = job else { break };
        gpu::timing::add(gpu::timing::Part::Idle, idle);
        gpu::with_pool(|| {
            let flips_before = processor.stats.flips;
            let active_recorder = capture.active_recorder();
            if let Some(recorder) = &active_recorder {
                recorder.note_queue(queue_id, || processor.saved());
                // Snapshot mapped GPU-area spans before the frame's first GPU work runs
                // so CPU writes made after the previous flip are captured.
                recorder.snapshot_mapped_spans_once(&host, || {
                    let (area_lo, area_hi) = crate::dmem::GPU_AREA;
                    host.mem
                        .region_spans()
                        .into_iter()
                        .filter(|r| !r.trap && r.end > area_lo && r.start < area_hi)
                        .map(|r| (r.start.max(area_lo), r.end.min(area_hi)))
                        .collect()
                });
                let rec_host = recorder.recording_host(&host);
                for (addr, dwords) in &job.buffers {
                    if let Err(e) = processor.run(&rec_host, *addr, *dwords) {
                        tracing::error!("{e}");
                    }
                }
                // Read command buffers after `run` (and before `end_submission` releases
                // fences to the CPU) so REWIND-patched packets are captured.
                let cmd_bufs: Vec<_> = job
                    .buffers
                    .iter()
                    .map(|&(addr, dwords)| gpu::capture::read_command_buffer(&host, addr, dwords))
                    .collect();
                let flipped = job.ends_frame || processor.stats.flips > flips_before;
                // Unverified: the driver signals each submission's completion.
                processor.end_submission(&rec_host, flipped);
                recorder.record_submission(queue_id, cmd_bufs, flipped);
            } else {
                for (addr, dwords) in &job.buffers {
                    if let Err(e) = processor.run(&host, *addr, *dwords) {
                        tracing::error!("{e}");
                    }
                }
                let flipped = job.ends_frame || processor.stats.flips > flips_before;
                // Unverified: the driver signals each submission's completion.
                processor.end_submission(&host, flipped);
            }
            let flipped = job.ends_frame || processor.stats.flips > flips_before;
            if queue_id == GRAPHICS && flipped {
                match capture.on_frame_end() {
                    gpu::capture::FrameBoundary::None => {}
                    gpu::capture::FrameBoundary::Start(path) => {
                        let textures = processor
                            .backend
                            .as_mut()
                            .and_then(|b| b.save_textures().ok())
                            .unwrap_or_default();
                        let recorder =
                            gpu::capture::FrameRecorder::new(gpu::Host::gpu_area(&host), textures);
                        capture.activate(path, recorder);
                    }
                    gpu::capture::FrameBoundary::Finish { path, recorder } => {
                        let cap = recorder.finish();
                        capture.spawn_writer(path, cap);
                    }
                }
            }
        });
        host.progress.update(|c| c.finished += 1);
        if report && last_report.elapsed().as_secs() >= 1 {
            last_report = std::time::Instant::now();
            let s = &processor.stats;
            tracing::info!(
                target: "gpu::stats",
                "{name}: {} submissions, {} packets, {} draws, {} dispatches, {} flips, {} interrupts, {} wait timeouts, {} waits met by pending releases, {} memory writes, longest flip gap {} ms",
                s.submissions,
                s.packets,
                s.draws,
                s.dispatches,
                s.flips,
                s.interrupts,
                s.timeouts,
                s.deferred_waits,
                s.opcodes.get(&0x37).unwrap_or(&0) + s.opcodes.get(&0x115).unwrap_or(&0),
                s.longest_flip_gap.as_millis()
            );
            if !processor.stats.long_flip_gaps.is_empty() {
                let gaps: Vec<u128> = processor
                    .stats
                    .long_flip_gaps
                    .drain(..)
                    .map(|g| g.as_millis())
                    .collect();
                tracing::info!(
                    target: "gpu::stats",
                    "{name}: flip gaps over 1 s since the last report (ms): {gaps:?}"
                );
            }
            if let Some(b) = processor.backend.as_ref().filter(|_| name == "graphics") {
                tracing::info!(target: "gpu::stats", "Metal: {}", b.summary());
            }
            if !processor.stats.gpu_waits.is_empty() || processor.stats.gpu_indirect > 0 {
                tracing::info!(
                    target: "gpu::stats",
                    "{name}: waits for the GPU by packet: {:?}, indirect dispatches sized on the GPU: {}",
                    processor.stats.gpu_waits, processor.stats.gpu_indirect
                );
            }
            tracing::debug!(
                target: "gpu::stats",
                "{name}: packets by opcode: {:x?}",
                processor.stats.opcodes
            );
        }
        if host.kernel.has_exited() {
            break;
        }
    }
}

fn enqueue(env: &Env, queue: u64, buffers: Vec<(u64, u32)>) {
    let gpu = &env.kernel.agc.gpu;
    let mut jobs = lock(&gpu.jobs);
    let (sender, progress) = jobs.entry(queue).or_insert_with(|| {
        let (tx, rx) = sync_channel(MAX_QUEUED);
        let progress = Arc::new(Progress::default());
        let host = Host {
            kernel: env.kernel.this(),
            mem: env.kernel.memory(),
            progress: progress.clone(),
        };
        let backend = gpu.backend.clone();
        if let Some(textures) = lock(&gpu.textures).take()
            && let Some(b) = backend.get_or_init(new_backend)
            && let Err(e) = gpu::Backend::restore_textures(&mut b.clone(), &textures)
        {
            tracing::warn!("save state: GPU textures not restored: {e}");
        }
        let name = if queue == GRAPHICS {
            "graphics".to_string()
        } else {
            format!("compute {queue:#x}")
        };
        let restored = lock(&gpu.restored).remove(&queue).map(|saved| {
            let gds = !gpu
                .gds_restored
                .swap(true, std::sync::atomic::Ordering::AcqRel);
            (saved, gds)
        });
        let capture = gpu.capture.clone();
        std::thread::Builder::new()
            .name(format!("gpu {name}"))
            .spawn(move || gpu_thread(host, rx, backend, capture, queue, name, restored))
            .expect("the GPU thread starts");
        (tx, progress)
    });
    let (sender, progress) = (sender.clone(), progress.clone());
    drop(jobs);
    let order = lock(&progress.order);
    let ends_frame = gpu.takes_flip(&buffers);
    let previous = progress.add(ends_frame);
    let _ = sender.send(Job {
        buffers,
        ends_frame,
    });
    drop(order);
    // Titles reuse the previous frame's memory once the next frame is submitted.
    progress.wait_for(previous, || {
        env.kernel.has_exited() || env.kernel.park_requested()
    });
}

/// A submit descriptor: the command address (u64 at 0) and dword count (u32 at 8).
fn descriptor(env: &Env, at: u64) -> Result<Option<(u64, u32)>, CallError> {
    if at == 0 {
        return Ok(None);
    }
    let addr = env.mem.read_u64(at)?;
    let mut b = [0u8; 4];
    env.mem.read(at + 8, &mut b)?;
    let dwords = u32::from_le_bytes(b);
    Ok((dwords <= MAX_DWORDS).then_some((addr, dwords)))
}

/// `sceAgcDriverSubmitDcb(descriptor)`.
fn submit_dcb(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffer) = descriptor(env, env.arg(0))? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, GRAPHICS, vec![buffer]);
    ret(0)
}

/// `sceAgcDriverSubmitAcb(owner, descriptor)`.
fn submit_acb(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffer) = descriptor(env, env.arg(1))? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, env.arg(0), vec![buffer]);
    ret(0)
}

/// The `count` buffers whose addresses (u64s) and sizes (u32s) are listed
/// at `addrs` and `sizes`.
fn buffer_list(
    env: &Env,
    addrs: u64,
    sizes: u64,
    count: u64,
) -> Result<Option<Vec<(u64, u32)>>, CallError> {
    if addrs == 0 || sizes == 0 || count > 1024 {
        return Ok(None);
    }
    let mut out = Vec::new();
    for i in 0..count {
        let addr = env.mem.read_u64(addrs + 8 * i)?;
        let mut b = [0u8; 4];
        env.mem.read(sizes + 4 * i, &mut b)?;
        out.push((addr, u32::from_le_bytes(b)));
    }
    Ok(Some(out))
}

/// `sceAgcDriverSubmitMultiDcbs(addresses, sizes, count)`.
fn submit_multi_dcbs(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffers) = buffer_list(env, env.arg(0), env.arg(1), env.arg(2) as u32 as u64)? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, GRAPHICS, buffers);
    ret(0)
}

/// `sceAgcDriverSubmitMultiAcbs(owner, addresses, sizes, count)`.
fn submit_multi_acbs(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffers) = buffer_list(env, env.arg(1), env.arg(2), env.arg(3) as u32 as u64)? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, env.arg(0), buffers);
    ret(0)
}

/// `sceAgcSuspendPoint()`: where a title may be suspended; nothing to do.
fn suspend_point(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// How long a save state waits for the GPU queues to finish their work.
const DRAIN_LIMIT: Duration = Duration::from_secs(3);

impl Gpu {
    /// Waits until every queue has finished what was submitted and the GPU
    /// is done with it, then writes each queue's processor state. Guest
    /// threads must be stopped, so nothing new is submitted.
    pub(super) fn save(&self, e: &mut savestate::Encoder) -> Result<(), String> {
        let queues: Vec<(u64, Arc<Progress>)> = lock(&self.jobs)
            .iter()
            .map(|(&k, (_, p))| (k, p.clone()))
            .collect();
        for (_, p) in &queues {
            *lock(&p.state) = None;
            p.wants_state
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let start = Instant::now();
        let mut saved = HashMap::new();
        for (key, p) in &queues {
            let mut c = lock(&p.counts);
            loop {
                let ready = lock(&p.state)
                    .as_ref()
                    .is_some_and(|(finished, _)| *finished == c.submitted);
                if ready {
                    break;
                }
                if lock(&p.state).is_some() {
                    *lock(&p.state) = None;
                    p.wants_state
                        .store(true, std::sync::atomic::Ordering::Release);
                }
                if start.elapsed() > DRAIN_LIMIT {
                    p.wants_state
                        .store(false, std::sync::atomic::Ordering::Release);
                    let what = if c.polling {
                        "waits on memory"
                    } else {
                        "is still running"
                    };
                    return Err(format!(
                        "GPU queue {key:#x} {what}: {} of {} submissions finished",
                        c.finished, c.submitted
                    ));
                }
                c = match p.changed.wait_timeout(c, Duration::from_millis(10)) {
                    Ok((c, _)) => c,
                    Err(e) => e.into_inner().0,
                };
            }
            drop(c);
            let (_, state) = lock(&p.state).take().expect("checked above");
            saved.insert(*key, state);
        }
        let textures = match self.backend.get() {
            Some(Some(b)) => gpu::Backend::save_textures(&mut b.clone())?,
            _ => Vec::new(),
        };
        e.put(&saved).put(&*lock(&self.flips)).put(&textures);
        Ok(())
    }

    pub(super) fn load(&self, d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
        *lock(&self.restored) = d.get()?;
        *lock(&self.flips) = d.get()?;
        let textures: Vec<u8> = d.get()?;
        *lock(&self.textures) = (!textures.is_empty()).then_some(textures);
        Ok(())
    }
}
