//! GPU frame capture: records one frame's command streams, initial queue and
//! GPU-only texture state, and the guest memory pages referenced during the
//! frame into a zstd-compressed file (`.ps5frame`) for deterministic offline replay.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

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
        w.section(TAG_TEXS, &self.textures)
            .map_err(|e| e.to_string())?;
        w.section(TAG_SUBS, &savestate::to_bytes(&self.submissions))
            .map_err(|e| e.to_string())?;

        // Write memory pages in 16 MiB chunks so zstd compresses each batch quickly.
        const CHUNK_PAGES: usize = 1024;
        let pages: Vec<(u64, Vec<u8>)> = self
            .pages
            .iter()
            .map(|(&addr, data)| (addr, data.clone()))
            .collect();
        for chunk in pages.chunks(CHUNK_PAGES) {
            let chunk_vec: Vec<(u64, Vec<u8>)> = chunk.to_vec();
            w.section(TAG_MEMP, &savestate::to_bytes(&chunk_vec))
                .map_err(|e| e.to_string())?;
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
    images: BTreeSet<(u64, u64)>,
    display_buffers: BTreeMap<i32, u64>,
    /// Pages already snapshotted (stored in `pages` if non-zero, omitted if all zero).
    seen_pages: BTreeSet<u64>,
    pages: BTreeMap<u64, Vec<u8>>,
}

impl RecorderState {
    fn snapshot_range(&mut self, host: &dyn Host, start: u64, end: u64) {
        if end <= start {
            return;
        }
        let first = start & !(PAGE_SIZE - 1);
        let last = end.next_multiple_of(PAGE_SIZE);
        let mut page = vec![0u8; PAGE_SIZE as usize];
        let mut addr = first;
        while addr < last {
            if self.seen_pages.insert(addr)
                && host.read(addr, &mut page)
                && page.iter().any(|&b| b != 0)
            {
                self.pages.insert(addr, page.clone());
            }
            addr += PAGE_SIZE;
        }
    }

    fn snapshot_extent(&mut self, host: &dyn Host, lo: u64, hi: u64) {
        self.extents.insert((lo, hi));
        if !self.snapshotted_extents.insert((lo, hi)) {
            return;
        }
        self.snapshot_range(host, lo, hi);
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
                images: BTreeSet::new(),
                display_buffers: BTreeMap::new(),
                seen_pages: BTreeSet::new(),
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

    /// Snapshots guest memory spans mapped inside the GPU area so shaders
    /// reading through the area window see their initial contents on replay.
    pub fn snapshot_mapped_spans(&self, host: &dyn Host, spans: &[(u64, u64)]) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for &(lo, hi) in spans {
            s.snapshot_extent(host, lo, hi);
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

    /// Consumes the recorded data and returns the finished [`FrameCapture`].
    pub fn finish(&self) -> FrameCapture {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        FrameCapture {
            meta: CaptureMeta {
                gpu_area: s.gpu_area,
                extents: s.extents.iter().copied().collect(),
                images: s.images.iter().copied().collect(),
                display_buffers: s.display_buffers.iter().map(|(&k, &v)| (k, v)).collect(),
            },
            queues: s.queues.clone(),
            textures: s.textures.clone(),
            submissions: s.submissions.clone(),
            pages: s.pages.clone(),
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
            s.snapshot_range(self.inner, addr, addr.saturating_add(buf.len() as u64));
        }
        ok
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        if !data.is_empty() {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            // Capture pre-write contents so replay sees the initial frame memory.
            s.snapshot_range(self.inner, addr, addr.saturating_add(data.len() as u64));
        }
        self.inner.write(addr, data)
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
        s.snapshot_extent(self.inner, ext.0, ext.1);
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
}
