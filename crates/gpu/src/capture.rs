//! GPU frame capture: records one frame's command streams, initial queue and
//! GPU-only texture state, and the guest memory pages referenced during the
//! frame into a zstd-compressed file (`.ps5frame`) for deterministic offline replay.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::{Flip, Host, Saved};

/// Header bytes inside the `savestate` container identifying a GPU frame capture.
const HEADER_MAGIC: [u8; 8] = *b"PS5FRAME";
const FORMAT: u32 = 1;
/// Page granularity for snapshotted guest memory (16 KiB, matching Apple Silicon pages).
pub const PAGE_SIZE: u64 = 16 << 10;

const TAG_META: [u8; 4] = *b"META";
const TAG_QUEU: [u8; 4] = *b"QUEU";
const TAG_TEXS: [u8; 4] = *b"TEXS";
const TAG_SUBS: [u8; 4] = *b"SUBS";
const TAG_MEMP: [u8; 4] = *b"MEMP";

/// One submitted command buffer captured at submission time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandBuffer {
    pub address: u64,
    pub dwords: Vec<u32>,
}

savestate::record!(CommandBuffer { address, dwords });

/// One queue submission during the captured frame, in execution order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Submission {
    pub queue: u64,
    pub buffers: Vec<CommandBuffer>,
    pub ends_frame: bool,
}

savestate::record!(Submission {
    queue,
    buffers,
    ends_frame
});

/// Address-space metadata needed to reconstruct guest memory during replay.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureMeta {
    pub gpu_area: Option<(u64, u64)>,
    pub extents: Vec<(u64, u64)>,
    pub images: Vec<(u64, u64)>,
    pub display_buffers: Vec<(i32, u64)>,
}

savestate::record!(CaptureMeta {
    gpu_area,
    extents,
    images,
    display_buffers
});

/// A self-contained GPU frame capture ready for offline replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameCapture {
    pub meta: CaptureMeta,
    /// Initial PM4 register banks and GDS for each queue active in the frame.
    pub queues: HashMap<u64, Saved>,
    /// Encoded GPU-only textures from `Backend::save_textures` at frame start.
    pub textures: Vec<u8>,
    pub submissions: Vec<Submission>,
    /// Non-zero 16 KiB guest memory pages as seen before the frame modified them.
    pub pages: BTreeMap<u64, Vec<u8>>,
}

impl FrameCapture {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let file = std::fs::File::create(path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        self.write_to(std::io::BufWriter::new(file))
    }

    pub fn write_to<W: Write>(&self, output: W) -> Result<(), String> {
        let mut header = Vec::with_capacity(12);
        header.extend_from_slice(&HEADER_MAGIC);
        header.extend_from_slice(&FORMAT.to_le_bytes());
        let mut w = savestate::Writer::new(output, &header).map_err(|e| e.to_string())?;
        w.section(TAG_META, &savestate::to_bytes(&self.meta))
            .map_err(|e| e.to_string())?;
        w.section(TAG_QUEU, &savestate::to_bytes(&self.queues))
            .map_err(|e| e.to_string())?;
        let packed_texs = savestate::compress(&self.textures, 1);
        w.packed(TAG_TEXS, self.textures.len() as u64, &packed_texs)
            .map_err(|e| e.to_string())?;
        w.section(TAG_SUBS, &savestate::to_bytes(&self.submissions))
            .map_err(|e| e.to_string())?;

        // Compress 16 MiB page chunks in parallel at zstd level 1 to keep capture
        // writes fast without holding the whole compressed stream in memory.
        const CHUNK_PAGES: usize = 1024;
        const PARALLEL_CHUNKS: usize = 4;
        let pages: Vec<(u64, &[u8])> = self
            .pages
            .iter()
            .map(|(&addr, data)| (addr, data.as_slice()))
            .collect();
        let all_chunks: Vec<&[(u64, &[u8])]> = pages.chunks(CHUNK_PAGES).collect();
        for batch in all_chunks.chunks(PARALLEL_CHUNKS) {
            let compressed: Vec<(u64, Vec<u8>)> = std::thread::scope(|scope| {
                let handles: Vec<_> = batch
                    .iter()
                    .map(|chunk| {
                        scope.spawn(move || {
                            let mut enc = savestate::Encoder::new();
                            enc.put(&chunk.len());
                            for (addr, data) in *chunk {
                                enc.put(addr).put(*data);
                            }
                            let raw = enc.finish();
                            let raw_len = raw.len() as u64;
                            (raw_len, savestate::compress(&raw, 1))
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("page compression thread"))
                    .collect()
            });
            for (raw_len, packed) in compressed {
                w.packed(TAG_MEMP, raw_len, &packed)
                    .map_err(|e| e.to_string())?;
            }
        }
        w.finish().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        Self::read_from(std::io::BufReader::new(file))
    }

    pub fn read_from<R: Read>(input: R) -> Result<Self, String> {
        let mut r = savestate::Reader::new(input).map_err(|e| e.to_string())?;
        if r.header.len() != 12 || r.header[..8] != HEADER_MAGIC {
            return Err("not a ps5slop GPU frame capture".into());
        }
        let format = u32::from_le_bytes(r.header[8..12].try_into().expect("4 bytes"));
        if format != FORMAT {
            return Err(format!(
                "frame capture format {format}, but this build reads format {FORMAT}"
            ));
        }
        let meta: CaptureMeta = savestate::from_bytes(&r.expect(TAG_META).map_err(|e| e.0)?)
            .map_err(|e| e.to_string())?;
        let queues: HashMap<u64, Saved> =
            savestate::from_bytes(&r.expect(TAG_QUEU).map_err(|e| e.0)?)
                .map_err(|e| e.to_string())?;
        let textures = r.expect(TAG_TEXS).map_err(|e| e.0)?;
        let submissions: Vec<Submission> =
            savestate::from_bytes(&r.expect(TAG_SUBS).map_err(|e| e.0)?)
                .map_err(|e| e.to_string())?;
        let mut pages = BTreeMap::new();
        while let Some(section) = r.next_section().map_err(|e| e.to_string())? {
            if section.tag != TAG_MEMP {
                return Err(format!("unexpected capture section {}", section.name()));
            }
            let raw = section.unpack().map_err(|e| e.to_string())?;
            let chunk: Vec<(u64, Vec<u8>)> =
                savestate::from_bytes(&raw).map_err(|e| e.to_string())?;
            for (addr, data) in chunk {
                pages.insert(addr, data);
            }
        }
        Ok(Self {
            meta,
            queues,
            textures,
            submissions,
            pages,
        })
    }
}

struct RecorderState {
    gpu_area: Option<(u64, u64)>,
    textures: Vec<u8>,
    queues: HashMap<u64, Saved>,
    submissions: Vec<Submission>,
    extents: BTreeSet<(u64, u64)>,
    snapshotted_extents: BTreeSet<(u64, u64)>,
    snapshotted_area: bool,
    images: BTreeSet<(u64, u64)>,
    display_buffers: BTreeMap<i32, u64>,
    /// Pages already locked by an explicit read or write during the captured frame.
    seen_pages: BTreeSet<u64>,
    /// Pages speculatively snapshotted from the GPU area before the first submission;
    /// an explicit first read refreshes them in case the CPU wrote them later in the frame.
    speculative_pages: BTreeSet<u64>,
    pages: BTreeMap<u64, Vec<u8>>,
}

impl RecorderState {
    fn snapshot_range(&mut self, host: &dyn Host, start: u64, end: u64, speculative: bool) {
        if end <= start {
            return;
        }
        let first = start & !(PAGE_SIZE - 1);
        let last = end.next_multiple_of(PAGE_SIZE);
        let mut page = vec![0u8; PAGE_SIZE as usize];
        let mut addr = first;
        while addr < last {
            let take = if speculative {
                !self.seen_pages.contains(&addr) && self.speculative_pages.insert(addr)
            } else {
                self.seen_pages.insert(addr)
            };
            if take && host.read(addr, &mut page) {
                if page.iter().any(|&b| b != 0) {
                    self.pages.insert(addr, page.clone());
                } else {
                    self.pages.remove(&addr);
                }
            }
            addr += PAGE_SIZE;
        }
    }

    /// Makes the captured pages hold `words` at `addr`, the command dwords the
    /// processor ran, whatever the CPU wrote there since the page was read.
    fn overlay_commands(&mut self, addr: u64, words: &[u32]) {
        let mut at = addr;
        for word in words {
            let page = self
                .pages
                .entry(at & !(PAGE_SIZE - 1))
                .or_insert_with(|| vec![0u8; PAGE_SIZE as usize]);
            let offset = (at & (PAGE_SIZE - 1)) as usize;
            page[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            at += 4;
        }
    }

    fn snapshot_extent(&mut self, host: &dyn Host, lo: u64, hi: u64, speculative: bool) {
        self.extents.insert((lo, hi));
        if !speculative && !self.snapshotted_extents.insert((lo, hi)) {
            return;
        }
        self.snapshot_range(host, lo, hi, speculative);
    }
}

/// Thread-safe recorder active for the duration of one captured frame.
#[derive(Clone)]
pub struct FrameRecorder {
    state: Arc<Mutex<RecorderState>>,
}

impl FrameRecorder {
    pub fn new(gpu_area: Option<(u64, u64)>, textures: Vec<u8>) -> Self {
        Self {
            state: Arc::new(Mutex::new(RecorderState {
                gpu_area,
                textures,
                queues: HashMap::new(),
                submissions: Vec::new(),
                extents: BTreeSet::new(),
                snapshotted_extents: BTreeSet::new(),
                snapshotted_area: false,
                images: BTreeSet::new(),
                display_buffers: BTreeMap::new(),
                seen_pages: BTreeSet::new(),
                speculative_pages: BTreeSet::new(),
                pages: BTreeMap::new(),
            })),
        }
    }

    /// Saves `queue`'s initial register/GDS state if not already captured.
    pub fn note_queue(&self, queue: u64, saved: impl FnOnce() -> Saved) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.queues.entry(queue).or_insert_with(saved);
    }

    /// Records a submission's command buffers after running it.
    pub fn record_submission(&self, queue: u64, buffers: Vec<CommandBuffer>, ends_frame: bool) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.submissions.push(Submission {
            queue,
            buffers,
            ends_frame,
        });
    }

    /// The `dwords` command dwords at `address` as the processor read them while
    /// running the submission. Reading guest memory afterwards would see what the
    /// title wrote once the run released it, which may be the next frame's buffer.
    pub fn command_buffer(&self, address: u64, dwords: u32) -> CommandBuffer {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let words = (0..dwords as u64)
            .map(|i| {
                let at = address + i * 4;
                s.pages.get(&(at & !(PAGE_SIZE - 1))).map_or(0, |page| {
                    let o = (at & (PAGE_SIZE - 1)) as usize;
                    u32::from_le_bytes([page[o], page[o + 1], page[o + 2], page[o + 3]])
                })
            })
            .collect();
        CommandBuffer {
            address,
            dwords: words,
        }
    }

    /// Snapshots guest memory spans mapped inside the GPU area so shaders
    /// reading through the area window see their initial contents on replay.
    pub fn snapshot_mapped_spans(&self, host: &dyn Host, spans: &[(u64, u64)]) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.snapshotted_area = true;
        for &(lo, hi) in spans {
            s.snapshot_extent(host, lo, hi, true);
        }
    }

    /// Snapshots mapped GPU-area spans once on the first recorded submission,
    /// capturing CPU writes made after the previous flip before GPU work runs.
    pub fn snapshot_mapped_spans_once(
        &self,
        host: &dyn Host,
        spans: impl FnOnce() -> Vec<(u64, u64)>,
    ) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.snapshotted_area {
            return;
        }
        s.snapshotted_area = true;
        for (lo, hi) in spans() {
            s.snapshot_extent(host, lo, hi, true);
        }
    }

    /// Wraps `inner` so every memory read, write, extent, image and display
    /// buffer query made during execution is recorded into this capture.
    pub fn recording_host<'a>(&'a self, inner: &'a dyn Host) -> RecordingHost<'a> {
        RecordingHost {
            inner,
            state: &self.state,
        }
    }

    /// Drains the recorded data and returns the finished [`FrameCapture`]
    /// without cloning its page and texture buffers.
    pub fn finish(&self) -> FrameCapture {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        FrameCapture {
            meta: CaptureMeta {
                gpu_area: s.gpu_area,
                extents: std::mem::take(&mut s.extents).into_iter().collect(),
                images: std::mem::take(&mut s.images).into_iter().collect(),
                display_buffers: std::mem::take(&mut s.display_buffers).into_iter().collect(),
            },
            queues: std::mem::take(&mut s.queues),
            textures: std::mem::take(&mut s.textures),
            submissions: std::mem::take(&mut s.submissions),
            pages: std::mem::take(&mut s.pages),
        }
    }
}

/// Phase of a [`CaptureManager`]: at most one capture is armed, snapshotting,
/// recording or writing at a time so host RAM never holds two captures at once.
#[derive(Default)]
enum CapturePhase {
    #[default]
    Idle,
    Armed(PathBuf),
    Snapshotting(PathBuf),
    Active {
        path: PathBuf,
        recorder: FrameRecorder,
    },
    Writing(PathBuf),
}

impl CapturePhase {
    fn describe(&self) -> Option<(&'static str, &Path)> {
        match self {
            CapturePhase::Idle => None,
            CapturePhase::Armed(p) => Some(("armed for", p)),
            CapturePhase::Snapshotting(p) => Some(("snapshotting", p)),
            CapturePhase::Active { path, .. } => Some(("capturing", path)),
            CapturePhase::Writing(p) => Some(("writing", p)),
        }
    }
}

/// What the graphics thread must do at a frame boundary.
pub enum FrameBoundary {
    None,
    /// Snapshot initial GPU textures and guest memory, then call [`CaptureManager::activate`].
    Start(PathBuf),
    /// Drain the recorder and hand it to [`CaptureManager::spawn_writer`].
    Finish {
        path: PathBuf,
        recorder: FrameRecorder,
    },
}

/// Result of scheduling a frame capture via [`CaptureManager::request`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureQueued {
    Armed(PathBuf),
    Queued {
        path: PathBuf,
        ahead: usize,
        current_state: &'static str,
        current_path: PathBuf,
    },
}

impl CaptureQueued {
    /// Human-readable status for the control socket or log output.
    pub fn describe(&self) -> String {
        match self {
            CaptureQueued::Armed(path) => format!("capturing the next frame to {}", path.display()),
            CaptureQueued::Queued {
                path,
                ahead,
                current_state,
                current_path,
            } => format!(
                "queued frame capture to {} ({ahead} ahead; currently {current_state} {})",
                path.display(),
                current_path.display()
            ),
        }
    }
}

#[derive(Default)]
struct CaptureQueue {
    phase: CapturePhase,
    pending: VecDeque<PathBuf>,
}

/// Coordinates one-frame GPU captures across GPU queue threads, control socket
/// commands, window hotkeys and background capture writers.
#[derive(Clone, Default)]
pub struct CaptureManager {
    inner: Arc<(Mutex<CaptureQueue>, Condvar)>,
}

impl CaptureManager {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CaptureQueue> {
        self.inner.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Arms a capture immediately if idle, or queues it behind active/pending
    /// captures so overlapping requests are never silently dropped.
    pub fn request(&self, path: PathBuf) -> CaptureQueued {
        let mut q = self.lock();
        if let Some((current_state, cur)) = q.phase.describe() {
            let current_path = cur.to_path_buf();
            let ahead = 1 + q.pending.len();
            q.pending.push_back(path.clone());
            CaptureQueued::Queued {
                path,
                ahead,
                current_state,
                current_path,
            }
        } else {
            q.phase = CapturePhase::Armed(path.clone());
            CaptureQueued::Armed(path)
        }
    }

    /// Returns the active [`FrameRecorder`] while a frame is being recorded.
    pub fn active_recorder(&self) -> Option<FrameRecorder> {
        match &self.lock().phase {
            CapturePhase::Active { recorder, .. } => Some(recorder.clone()),
            _ => None,
        }
    }

    /// Advances the state machine at the end of a graphics frame.
    pub fn on_frame_end(&self) -> FrameBoundary {
        let mut q = self.lock();
        match std::mem::take(&mut q.phase) {
            CapturePhase::Armed(path) => {
                q.phase = CapturePhase::Snapshotting(path.clone());
                FrameBoundary::Start(path)
            }
            CapturePhase::Active { path, recorder } => {
                q.phase = CapturePhase::Writing(path.clone());
                FrameBoundary::Finish { path, recorder }
            }
            other => {
                q.phase = other;
                FrameBoundary::None
            }
        }
    }

    /// Installs the recorder created after [`FrameBoundary::Start`] finishes
    /// snapshotting initial textures and guest memory.
    pub fn activate(&self, path: PathBuf, recorder: FrameRecorder) {
        let mut q = self.lock();
        q.phase = CapturePhase::Active { path, recorder };
    }

    /// Writes `capture` to `path` on a background thread, then arms the next
    /// queued capture (if any).
    pub fn spawn_writer(&self, path: PathBuf, capture: FrameCapture) {
        let manager = self.clone();
        let fallback_path = path.clone();
        let spawn = std::thread::Builder::new()
            .name("gpu-capture-writer".into())
            .spawn(move || {
                Self::write_and_advance(&manager, &path, &capture);
            });
        if let Err(e) = spawn {
            tracing::error!(
                "cannot start capture writer thread for {}: {e}",
                fallback_path.display()
            );
            self.finish_writing();
        }
    }

    fn write_and_advance(&self, path: &Path, capture: &FrameCapture) {
        let started = Instant::now();
        let subs = capture.submissions.len();
        let pages = capture.pages.len();
        match capture.save(path) {
            Ok(()) => {
                let mib = std::fs::metadata(path)
                    .map(|m| m.len() as f64 / (1u64 << 20) as f64)
                    .unwrap_or(0.0);
                tracing::info!(
                    target: "gpu::snap",
                    "wrote frame capture to {} ({subs} submissions, {pages} non-zero pages, {mib:.1} MiB in {:.2} s)",
                    path.display(),
                    started.elapsed().as_secs_f64()
                );
            }
            Err(e) => tracing::error!("frame capture to {} failed: {e}", path.display()),
        }
        self.finish_writing();
    }

    fn finish_writing(&self) {
        let mut q = self.lock();
        if let Some(next) = q.pending.pop_front() {
            tracing::info!(
                target: "gpu::snap",
                "starting queued frame capture to {}",
                next.display()
            );
            q.phase = CapturePhase::Armed(next);
        } else {
            q.phase = CapturePhase::Idle;
        }
        self.inner.1.notify_all();
    }

    /// Waits for any background capture write to finish and warns about any
    /// armed, active or queued captures that did not complete before exit.
    pub fn finish_on_exit(&self) {
        let mut q = self.lock();
        while matches!(q.phase, CapturePhase::Writing(_)) {
            q = self.inner.1.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        let mut incomplete = Vec::new();
        if let Some((_, p)) = q.phase.describe() {
            incomplete.push(p.to_path_buf());
        }
        incomplete.extend(q.pending.drain(..));
        q.phase = CapturePhase::Idle;
        drop(q);
        for p in incomplete {
            tracing::warn!(
                "frame capture to {} did not complete before exit",
                p.display()
            );
        }
    }
}

/// A [`Host`] wrapper that forwards every operation to `inner` while capturing
/// initial guest memory pages and metadata for replay.
pub struct RecordingHost<'a> {
    inner: &'a dyn Host,
    state: &'a Mutex<RecorderState>,
}

impl Host for RecordingHost<'_> {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let ok = self.inner.read(addr, buf);
        if ok && !buf.is_empty() {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            s.snapshot_range(
                self.inner,
                addr,
                addr.saturating_add(buf.len() as u64),
                false,
            );
        }
        ok
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        if !data.is_empty() {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            // Capture pre-write contents so replay sees the initial frame memory.
            s.snapshot_range(
                self.inner,
                addr,
                addr.saturating_add(data.len() as u64),
                false,
            );
        }
        self.inner.write(addr, data)
    }

    fn commands_read(&self, addr: u64, words: &[u32]) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.overlay_commands(addr, words);
    }

    fn interrupt(&self, context: u32) {
        self.inner.interrupt(context);
    }

    fn flip(&self, flip: Flip) {
        self.inner.flip(flip);
    }

    fn exiting(&self) -> bool {
        self.inner.exiting()
    }

    fn extent(&self, addr: u64) -> Option<(u64, u64)> {
        let ext = self.inner.extent(addr)?;
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.snapshot_extent(self.inner, ext.0, ext.1, false);
        Some(ext)
    }

    fn image(&self, addr: u64) -> Option<(u64, u64)> {
        let img = self.inner.image(addr)?;
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.images.insert(img);
        Some(img)
    }

    fn host_pointer(&self, addr: u64) -> Option<*mut u8> {
        self.inner.host_pointer(addr)
    }

    fn display_buffer(&self, index: i32) -> Option<u64> {
        let addr = self.inner.display_buffer(index)?;
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.display_buffers.insert(index, addr);
        Some(addr)
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    fn replaced_since(&self, since: u64) -> (u64, Option<Vec<(u64, u64)>>) {
        self.inner.replaced_since(since)
    }

    fn gpu_area(&self) -> Option<(u64, u64)> {
        self.inner.gpu_area()
    }

    fn back(&self, start: u64, end: u64) -> bool {
        self.inner.back(start, end)
    }

    fn waiting(&self, polling: bool) {
        self.inner.waiting(polling);
    }
}

/// Reads `dwords` 32-bit words from `host` at `address`.
pub fn read_command_buffer(host: &dyn Host, address: u64, dwords: u32) -> CommandBuffer {
    let mut bytes = vec![0u8; dwords as usize * 4];
    host.read(address, &mut bytes);
    let words = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    CommandBuffer {
        address,
        dwords: words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MemHost(Mutex<Vec<u8>>);
    impl Host for MemHost {
        fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
            let m = self.0.lock().unwrap();
            let start = addr as usize;
            let Some(slice) = m.get(start..start + buf.len()) else {
                return false;
            };
            buf.copy_from_slice(slice);
            true
        }
        fn write(&self, addr: u64, data: &[u8]) -> bool {
            let mut m = self.0.lock().unwrap();
            let start = addr as usize;
            let Some(slice) = m.get_mut(start..start + data.len()) else {
                return false;
            };
            slice.copy_from_slice(data);
            true
        }
        fn interrupt(&self, _: u32) {}
        fn flip(&self, _: Flip) {}
        fn exiting(&self) -> bool {
            false
        }
    }

    #[test]
    fn frame_capture_round_trips_through_bytes() {
        let mut pages = BTreeMap::new();
        let mut page = vec![0u8; PAGE_SIZE as usize];
        page[0] = 0xAB;
        page[123] = 0xCD;
        pages.insert(0x4000, page);

        let cap = FrameCapture {
            meta: CaptureMeta {
                gpu_area: Some((0x10_0000_0000, 0x11_0000_0000)),
                extents: vec![(0x4000, 0x8000)],
                images: vec![(0x40_0000, 0x50_0000)],
                display_buffers: vec![(0, 0x10_0010_0000)],
            },
            queues: [(u64::MAX, crate::Processor::default().saved())]
                .into_iter()
                .collect(),
            textures: vec![1, 2, 3, 4],
            submissions: vec![Submission {
                queue: u64::MAX,
                buffers: vec![CommandBuffer {
                    address: 0x4000,
                    dwords: vec![0xC000_1000, 0],
                }],
                ends_frame: true,
            }],
            pages,
        };

        let mut buf = Vec::new();
        cap.write_to(&mut buf).unwrap();
        let back = FrameCapture::read_from(&buf[..]).unwrap();
        assert_eq!(back, cap);
    }

    #[test]
    fn capture_manager_queues_overlapping_requests_and_writes_all_off_thread() {
        let dir =
            std::env::temp_dir().join(format!("ps5slop-capture-queue-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mgr = CaptureManager::new();
        let p1 = dir.join("first.ps5frame");
        let p2 = dir.join("second.ps5frame");
        let p3 = dir.join("third.ps5frame");

        // Request 1 arms immediately; requests 2 and 3 queue without overwriting.
        assert_eq!(mgr.request(p1.clone()), CaptureQueued::Armed(p1.clone()));
        assert!(matches!(
            mgr.request(p2.clone()),
            CaptureQueued::Queued { ahead: 1, .. }
        ));

        // Frame 1 ends -> start capturing p1.
        let FrameBoundary::Start(start1) = mgr.on_frame_end() else {
            panic!("expected Start for first capture");
        };
        assert_eq!(start1, p1);
        mgr.activate(start1, FrameRecorder::new(None, Vec::new()));
        assert!(mgr.active_recorder().is_some());

        // Request 3 arrives while p1 is actively recording -> queued behind p2.
        assert!(matches!(
            mgr.request(p3.clone()),
            CaptureQueued::Queued { ahead: 2, .. }
        ));

        // Frame 2 ends -> finish p1 and write off-thread.
        let FrameBoundary::Finish { path, recorder } = mgr.on_frame_end() else {
            panic!("expected Finish for first capture");
        };
        assert_eq!(path, p1);
        mgr.spawn_writer(path, recorder.finish());

        // Wait for writer to advance p2 to Armed, then capture p2 and p3 in sequence.
        let drive_next = |expected: &Path| {
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            let start = loop {
                if let FrameBoundary::Start(p) = mgr.on_frame_end() {
                    break p;
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {expected:?}"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            };
            assert_eq!(start, expected);
            mgr.activate(start, FrameRecorder::new(None, Vec::new()));
            let FrameBoundary::Finish { path, recorder } = mgr.on_frame_end() else {
                panic!("expected Finish for {expected:?}");
            };
            assert_eq!(path, expected);
            mgr.spawn_writer(path, recorder.finish());
        };

        drive_next(&p2);
        drive_next(&p3);
        mgr.finish_on_exit();

        for p in [&p1, &p2, &p3] {
            assert!(p.is_file(), "missing capture {}", p.display());
            FrameCapture::load(p).expect("written capture loads cleanly");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_capture_write_still_advances_queue() {
        let dir =
            std::env::temp_dir().join(format!("ps5slop-capture-fail-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Using a directory path as a file path fails to open for writing.
        let bad = dir.clone();
        let good = dir.join("after-error.ps5frame");
        let mgr = CaptureManager::new();
        mgr.request(bad.clone());
        mgr.request(good.clone());

        let FrameBoundary::Start(p) = mgr.on_frame_end() else {
            panic!("expected Start");
        };
        mgr.activate(p, FrameRecorder::new(None, Vec::new()));
        let FrameBoundary::Finish { path, recorder } = mgr.on_frame_end() else {
            panic!("expected Finish");
        };
        mgr.spawn_writer(path, recorder.finish());

        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        let next = loop {
            if let FrameBoundary::Start(p) = mgr.on_frame_end() {
                break p;
            }
            assert!(
                Instant::now() < deadline,
                "queue did not advance after error"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert_eq!(next, good);
        mgr.activate(next, FrameRecorder::new(None, Vec::new()));
        let FrameBoundary::Finish { path, recorder } = mgr.on_frame_end() else {
            panic!("expected Finish");
        };
        mgr.spawn_writer(path, recorder.finish());
        mgr.finish_on_exit();

        assert!(good.is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explicit_read_refreshes_speculative_area_snapshot_without_clobbering_gpu_writes() {
        let host = MemHost(Mutex::new(vec![0u8; PAGE_SIZE as usize * 2]));
        host.write(0, &[0x11, 0x22, 0x33, 0x44]);
        host.write(PAGE_SIZE, &[0xAA, 0xBB]);

        let rec = FrameRecorder::new(Some((0, PAGE_SIZE * 2)), Vec::new());
        // Speculative snapshot at submission 0 sees initial contents of both pages.
        rec.snapshot_mapped_spans_once(&host, || vec![(0, PAGE_SIZE * 2)]);
        // CPU later writes a fresh indirect buffer into page 0 before submission 1 reads it.
        host.write(0, &[0x55, 0x66, 0x77, 0x88]);
        let rhost = rec.recording_host(&host);
        let mut buf = [0u8; 4];
        assert!(rhost.read(0, &mut buf));
        assert_eq!(buf, [0x55, 0x66, 0x77, 0x88]);
        // GPU write to page 1 must preserve pre-write bytes (0xAA, 0xBB), not the new bytes.
        assert!(rhost.write(PAGE_SIZE, &[0xCC, 0xDD]));

        let cap = rec.finish();
        assert_eq!(&cap.pages[&0][..4], &[0x55, 0x66, 0x77, 0x88]);
        assert_eq!(&cap.pages[&PAGE_SIZE][..2], &[0xAA, 0xBB]);
    }

    #[test]
    fn command_buffer_is_what_the_processor_read_not_what_the_title_wrote_after() {
        const NOP: u32 = 0xC000_1000;
        let host = MemHost(Mutex::new(vec![0u8; PAGE_SIZE as usize]));
        let ran = [NOP, 1, NOP, 2, NOP, 3];
        for (i, w) in ran.iter().enumerate() {
            host.write(0x100 + i as u64 * 4, &w.to_le_bytes());
        }

        let rec = FrameRecorder::new(None, Vec::new());
        let rhost = rec.recording_host(&host);
        crate::Processor::default()
            .run(&rhost, 0x100, ran.len() as u32)
            .unwrap();

        // The title reuses the buffer as soon as the run releases it.
        for i in 0..ran.len() as u64 {
            host.write(0x100 + i * 4, &0xDEAD_BEEFu32.to_le_bytes());
        }

        assert_eq!(rec.command_buffer(0x100, ran.len() as u32).dwords, ran);
        assert_eq!(read_command_buffer(&host, 0x100, 1).dwords, [0xDEAD_BEEF]);
    }
}
