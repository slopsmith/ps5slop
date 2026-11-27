//! Save states: the whole guest written to a file at a moment when every
//! guest thread is stopped at a host call boundary and the GPU queues are
//! idle, and a fresh process made to look the same from it. The format is
//! in the `savestate` crate; what is kept is in `docs/savestates.md`.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use memory::{Memory, PAGE_SIZE, SavedRegion};
use savestate::{Decoder, Encoder, Error, Reader, Writer};
use x86::Cpu;

use crate::Kernel;

const REGIONS: [u8; 4] = *b"REGN";
const PAGES: [u8; 4] = *b"PAGE";
const THREADS: [u8; 4] = *b"THRD";
const KERNEL: [u8; 4] = *b"KERN";
const GPU: [u8; 4] = *b"AGC.";

/// Memory is read, checked for zero pages and compressed in pieces of
/// this size, one piece per worker at a time.
const PIECE: u64 = 8 << 20;
/// Tries at stopping the world with the GPU idle before a save gives up.
const STOP_TRIES: u32 = 8;

/// What a save wrote and how long it took.
#[derive(Debug, Clone)]
pub struct SaveReport {
    pub path: std::path::PathBuf,
    pub bytes: u64,
    /// Guest memory written, before compression.
    pub memory: u64,
    pub threads: usize,
    /// Time to stop every thread and drain the GPU.
    pub stop: Duration,
    /// Time to write the file.
    pub write: Duration,
}

/// The header: who wrote the state, for what, and the clocks at the time.
struct Header {
    build: String,
    title: String,
    layout: u64,
    guest_unix: Duration,
    uptime: Duration,
}

savestate::record!(Header {
    build,
    title,
    layout,
    guest_unix,
    uptime
});

type RegionRecord = (u64, u64, u8, bool, String, Option<u64>);

fn region_record(r: &SavedRegion) -> RegionRecord {
    let n = r.name.clone();
    (r.start, r.end, r.prot.bits(), r.trap, n, r.physical)
}

fn saved_region((start, end, prot, trap, name, physical): RegionRecord) -> SavedRegion {
    let prot = memory::Prot::from_bits(prot);
    SavedRegion {
        start,
        end,
        prot,
        trap,
        name,
        physical,
    }
}

fn build_id() -> String {
    format!("ps5slop {}", env!("CARGO_PKG_VERSION"))
}

static SAVING: Mutex<()> = Mutex::new(());

/// Lets stopped threads go on when dropped.
struct Stopped<'a>(&'a Kernel);

impl Drop for Stopped<'_> {
    fn drop(&mut self) {
        self.0.resume_world();
    }
}

impl Kernel {
    fn layout(&self) -> u64 {
        let mut b = self.hle.layout_fingerprint().to_le_bytes().to_vec();
        b.extend_from_slice(&self.tls_span().to_le_bytes());
        savestate::fnv(&b)
    }

    /// Why a save state cannot be made now, if something in use is not kept.
    fn unsaved(&self) -> Option<&'static str> {
        if self.fonts.in_use() {
            return Some("fonts (libSceFont) are in use, and their FreeType faces are not saved");
        }
        if self.gnm.in_use() {
            return Some("the PS4 command path (libSceGnmDriver) has run, which is not saved");
        }
        None
    }

    /// Stops the guest, writes everything it can observe to `path` and lets
    /// it go on. The file is written next to `path` first and renamed.
    pub fn save_state(&self, path: &Path) -> Result<SaveReport, String> {
        let Ok(_one) = SAVING.try_lock() else {
            return Err("a save is already running".into());
        };
        if let Some(why) = self.unsaved() {
            return Err(format!("cannot save: {why}"));
        }
        let start = Instant::now();
        let mut last = String::new();
        let mut stopped = None;
        for _ in 0..STOP_TRIES {
            match self.stop_world() {
                Err(e) => last = e,
                Ok(cpus) => {
                    let guard = Stopped(self);
                    let mut gpu = Encoder::new();
                    match self.agc.save(&mut gpu) {
                        Ok(()) => {
                            stopped = Some((guard, cpus, gpu.finish()));
                            break;
                        }
                        Err(e) => last = e,
                    }
                }
            }
            tracing::warn!("save state: not stopped yet: {last}");
            std::thread::sleep(Duration::from_millis(300));
        }
        let Some((_guard, cpus, gpu)) = stopped else {
            return Err(format!(
                "cannot save: the guest did not reach a point to save at ({last})"
            ));
        };
        let stop = start.elapsed();
        let tmp = path.with_extension("partial");
        let written = self.write_state(&tmp, &cpus, &gpu);
        let (bytes, memory) = match written {
            Ok(v) => v,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(format!("cannot write {}: {e}", tmp.display()));
            }
        };
        std::fs::rename(&tmp, path)
            .map_err(|e| format!("cannot rename {} to {}: {e}", tmp.display(), path.display()))?;
        Ok(SaveReport {
            path: path.to_owned(),
            bytes,
            memory,
            threads: cpus.len(),
            stop,
            write: start.elapsed() - stop,
        })
    }

    fn write_state(
        &self,
        path: &Path,
        cpus: &std::collections::BTreeMap<u64, Cpu>,
        gpu: &[u8],
    ) -> Result<(u64, u64), String> {
        let header = Header {
            build: build_id(),
            title: self.title_id.get().cloned().unwrap_or_default(),
            layout: self.layout(),
            guest_unix: savestate::clock::unix(),
            uptime: self.started.elapsed(),
        };
        let mut kernel = Encoder::new();
        self.save_objects(&mut kernel)?;
        let mut threads = Encoder::new();
        let cpus: Vec<(u64, Cpu)> = cpus.iter().map(|(&t, c)| (t, c.clone())).collect();
        threads
            .put(&cpus)
            .put(&*crate::pthread::lock(&self.tcbs))
            .put(&*crate::pthread::lock(&self.names));

        let file = File::create(path).map_err(|e| e.to_string())?;
        let out = BufWriter::with_capacity(8 << 20, file);
        let io = |e: std::io::Error| e.to_string();
        let mut w = Writer::new(out, &savestate::to_bytes(&header)).map_err(io)?;
        let mem = self.memory();
        let mut regions = Encoder::new();
        let saved: Vec<RegionRecord> = mem.saved_regions().iter().map(region_record).collect();
        regions.put(&saved).put(&mem.physical_size());
        w.section(REGIONS, &regions.finish()).map_err(io)?;
        let raw = write_pages(&mem, &mut w).map_err(io)?;
        w.section(THREADS, &threads.finish()).map_err(io)?;
        w.section(KERNEL, &kernel.finish()).map_err(io)?;
        w.section(GPU, gpu).map_err(io)?;
        let bytes = w.written;
        let out = w.finish().map_err(io)?;
        out.into_inner()
            .map_err(|e| e.to_string())?
            .sync_all()
            .map_err(io)?;
        Ok((bytes, raw))
    }

    pub(crate) fn save_objects(&self, e: &mut Encoder) -> Result<(), String> {
        self.objects.save(e);
        self.threads.save(e);
        self.heap.save(e);
        self.sems.save(e);
        self.evfs.save(e);
        self.fibers.save(e);
        self.apr.save(e);
        self.exceptions.save(e);
        self.dmem.save(e);
        self.equeues.save(e);
        self.pad.save(e);
        self.audio.save(e);
        crate::audio::save_input(e);
        self.ngs2.save(e);
        self.videoout.save(e);
        self.rtld.save(e);
        self.libc.save(e)?;
        self.saves.save(e);
        crate::ajm::save(e);
        crate::net::save(e);
        crate::dialog::save(e);
        crate::stubs::save(e);
        crate::services::save(e);
        self.apr.save_amm(e);
        Ok(())
    }

    pub(crate) fn load_objects(&self, d: &mut Decoder) -> Result<(), Error> {
        self.objects.load(d)?;
        self.threads.load(d)?;
        self.heap.load(d)?;
        self.sems.load(d)?;
        self.evfs.load(d)?;
        self.fibers.load(d)?;
        self.apr.load(d, &self.vfs)?;
        self.exceptions.load(d)?;
        self.dmem.load(d)?;
        self.equeues.load(d)?;
        self.pad.load(d)?;
        self.audio.load(d)?;
        crate::audio::load_input(d)?;
        self.ngs2.load(d)?;
        self.videoout.load(d)?;
        self.rtld.load(d)?;
        self.libc.load(d, &self.vfs)?;
        self.saves.load(d, &self.vfs)?;
        crate::ajm::load(d)?;
        crate::net::load(d)?;
        crate::dialog::load(d)?;
        crate::stubs::load(d)?;
        crate::services::load(d)?;
        // Optional: states from before these fields existed end here.
        if !d.is_empty() {
            self.apr.load_amm(d)?;
        }
        Ok(())
    }

    /// Makes this freshly booted process the one saved in `path`: its memory,
    /// kernel objects, GPU state and clocks. Returns the CPU state of each
    /// guest thread, for [`Kernel::run_restored`].
    pub fn load_state(&self, path: &Path) -> Result<Vec<Cpu>, String> {
        let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mut r = Reader::new(BufReader::with_capacity(8 << 20, file)).map_err(|e| e.0)?;
        let header: Header = savestate::from_bytes(&r.header).map_err(|e| e.0)?;
        let title = self.title_id.get().cloned().unwrap_or_default();
        if !header.title.is_empty() && !title.is_empty() && header.title != title {
            return Err(format!(
                "this save state is of {}, not {title}",
                header.title
            ));
        }
        if header.layout != self.layout() {
            return Err(format!(
                "this save state does not fit this build and title: the host functions guest memory refers to sit elsewhere (saved by {})",
                header.build
            ));
        }
        if header.build != build_id() {
            tracing::warn!(
                "save state written by {}, loading in {}",
                header.build,
                build_id()
            );
        }
        let mem = self.memory();
        let regions = r.expect(REGIONS).map_err(|e| e.0)?;
        restore_regions(&mem, &mut Decoder::new(&regions)).map_err(|e| e.0)?;
        let next = read_pages(&mem, &mut r).map_err(|e| e.0)?;
        if next.as_ref().map(|s| s.tag) != Some(THREADS) {
            return Err("save state has no thread section after its memory".into());
        }
        let threads = next.expect("checked above").unpack().map_err(|e| e.0)?;
        let mut d = Decoder::new(&threads);
        let cpus: Vec<(u64, Cpu)> = d.get().map_err(|e| e.0)?;
        *crate::pthread::lock(&self.tcbs) = d.get().map_err(|e| e.0)?;
        *crate::pthread::lock(&self.names) = d.get().map_err(|e| e.0)?;
        let kernel = r.expect(KERNEL).map_err(|e| e.0)?;
        let mut d = Decoder::new(&kernel);
        self.load_objects(&mut d).map_err(|e| e.0)?;
        if !d.is_empty() {
            return Err("save state's kernel section has bytes left over".into());
        }
        let gpu = r.expect(GPU).map_err(|e| e.0)?;
        self.agc.load(&mut Decoder::new(&gpu)).map_err(|e| e.0)?;
        if r.next_section().map_err(|e| e.0)?.is_some() {
            return Err("save state has sections this build does not know".into());
        }
        self.started.set(header.uptime);
        savestate::clock::set_unix(header.guest_unix);
        Ok(cpus.into_iter().map(|(_, c)| c).collect())
    }
}

/// Splits `[start, end)` into [`PIECE`]-sized pieces of kind `kind`.
fn pieces(out: &mut Vec<(u8, u64, u64)>, kind: u8, start: u64, end: u64) {
    let mut at = start;
    while at < end {
        let next = (at + PIECE).min(end);
        out.push((kind, at, next));
        at = next;
    }
}

// Pages the host never backed read as zero, so they need not be read.
fn touched(mem: &Memory, start: u64, end: u64) -> Option<Vec<bool>> {
    let host_page = 1usize << 14;
    let len = (end - start) as usize;
    let mut v = vec![0 as libc::c_char; len.div_ceil(host_page)];
    // SAFETY: the range lies inside the reservation, which mincore may query.
    let r = unsafe {
        libc::mincore(
            mem.host_base().add(start as usize).cast(),
            len,
            v.as_mut_ptr(),
        )
    };
    (r == 0).then(|| v.iter().map(|&b| b != 0).collect())
}

/// Writes the non-zero pages as [`PAGES`] sections; returns their raw size.
fn write_pages<W: std::io::Write>(mem: &Memory, w: &mut Writer<W>) -> std::io::Result<u64> {
    let mut todo = Vec::new();
    for r in mem.saved_regions() {
        if !r.trap && r.physical.is_none() {
            pieces(&mut todo, 0, r.start, r.end);
        }
    }
    for (s, e) in mem.physical_ranges() {
        pieces(&mut todo, 1, s, e);
    }
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (tx, rx) = std::sync::mpsc::sync_channel::<(u64, Vec<u8>)>(workers * 4);
    let mut raw = 0;
    std::thread::scope(|scope| -> std::io::Result<()> {
        for _ in 0..workers {
            let (tx, todo, next) = (tx.clone(), &todo, &next);
            let mem = mem.clone();
            scope.spawn(move || {
                let mut buf = vec![0u8; PIECE as usize];
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(kind, start, end)) = todo.get(i) else {
                        break;
                    };
                    let len = (end - start) as usize;
                    let buf = &mut buf[..len];
                    let touched = if kind == 0 {
                        touched(&mem, start, end)
                    } else {
                        None
                    };
                    if touched.as_ref().is_some_and(|t| !t.contains(&true)) {
                        continue;
                    }
                    let read = if kind == 0 {
                        mem.read_host(start, buf).is_ok()
                    } else {
                        mem.with_physical(start, len as u64, |p| buf.copy_from_slice(p))
                            .is_some()
                    };
                    if !read {
                        continue;
                    }
                    let page = PAGE_SIZE as usize;
                    let live = |k: usize| {
                        touched
                            .as_ref()
                            .is_none_or(|t| t.get(k).copied().unwrap_or(true))
                            && buf[k * page..(k + 1) * page].iter().any(|&b| b != 0)
                    };
                    let mut k = 0;
                    let pages = len / page;
                    while k < pages {
                        if !live(k) {
                            k += 1;
                            continue;
                        }
                        let first = k;
                        while k < pages && live(k) {
                            k += 1;
                        }
                        let mut chunk = Vec::with_capacity(9 + (k - first) * page);
                        chunk.push(kind);
                        chunk.extend_from_slice(&(start + (first * page) as u64).to_le_bytes());
                        chunk.extend_from_slice(&buf[first * page..k * page]);
                        let packed = savestate::compress(&chunk, 1);
                        if tx.send((chunk.len() as u64, packed)).is_err() {
                            return;
                        }
                    }
                }
            });
        }
        drop(tx);
        for (len, packed) in rx {
            w.packed(PAGES, len, &packed)?;
            raw += len - 9;
        }
        Ok(())
    })?;
    Ok(raw)
}

/// Replaces every region of the freshly booted address space with the
/// saved ones (trap regions must match).
fn restore_regions(mem: &Memory, d: &mut Decoder) -> Result<(), Error> {
    let regions: Vec<RegionRecord> = d.get()?;
    let regions: Vec<SavedRegion> = regions.into_iter().map(saved_region).collect();
    let physical: u64 = d.get()?;
    for r in mem.saved_regions().into_iter().filter(|r| !r.trap) {
        mem.unmap_range(r.start, r.end - r.start)
            .map_err(|e| savestate::error(format!("cannot clear {:#x}: {e}", r.start)))?;
    }
    if physical != 0 && !mem.init_physical(physical) {
        return Err(savestate::error("cannot make physical memory"));
    }
    for r in &regions {
        mem.restore_region(r, physical).map_err(|e| {
            savestate::error(format!(
                "cannot map {:#x}..{:#x} ({}): {e}",
                r.start, r.end, r.name
            ))
        })?;
    }
    Ok(())
}

/// Reads [`PAGES`] sections into memory with worker threads, and returns
/// the section after them.
fn read_pages<R: std::io::Read>(
    mem: &Memory,
    r: &mut Reader<R>,
) -> Result<Option<savestate::Section>, Error> {
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (tx, rx) = std::sync::mpsc::sync_channel::<savestate::Section>(workers * 2);
    let rx = Mutex::new(rx);
    let failed: Mutex<Option<Error>> = Mutex::new(None);
    let after = std::thread::scope(|scope| {
        for _ in 0..workers {
            let (rx, failed) = (&rx, &failed);
            let mem = mem.clone();
            scope.spawn(move || {
                loop {
                    let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    let Ok(section) = next else { break };
                    if let Err(e) = write_chunk(&mem, &section) {
                        *failed.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                    }
                }
            });
        }
        let after = loop {
            match r.next_section() {
                Ok(Some(s)) if s.tag == PAGES => {
                    if tx.send(s).is_err() {
                        break Ok(None);
                    }
                }
                other => break other,
            }
        };
        drop(tx);
        after
    })?;
    if let Some(e) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(e);
    }
    Ok(after)
}

fn write_chunk(mem: &Memory, section: &savestate::Section) -> Result<(), Error> {
    let chunk = section.unpack()?;
    if chunk.len() < 9 {
        return Err(savestate::error("memory chunk too short"));
    }
    let addr = u64::from_le_bytes(chunk[1..9].try_into().expect("8 bytes"));
    let data = &chunk[9..];
    let ok = match chunk[0] {
        0 => mem.write_host(addr, data).is_ok(),
        1 => mem.write_physical(addr, data),
        k => return Err(savestate::error(format!("bad memory chunk kind {k}"))),
    };
    if !ok {
        return Err(savestate::error(format!(
            "cannot restore memory at {addr:#x} ({} bytes)",
            data.len()
        )));
    }
    Ok(())
}

impl Kernel {
    /// Saves on a new thread, for callers that must not block (the window).
    pub fn save_state_in_background(self: &Arc<Self>, path: std::path::PathBuf) {
        let kernel = self.clone();
        std::thread::spawn(move || match kernel.save_state(&path) {
            Ok(r) => tracing::info!("{}", r.describe()),
            Err(e) => tracing::error!("{e}"),
        });
    }
}

impl SaveReport {
    pub fn describe(&self) -> String {
        format!(
            "saved {} threads to {}: {:.1} MiB ({:.1} MiB of memory before compression), stopped in {:.2} s, written in {:.2} s",
            self.threads,
            self.path.display(),
            self.bytes as f64 / (1 << 20) as f64,
            self.memory as f64 / (1 << 20) as f64,
            self.stop.as_secs_f64(),
            self.write.as_secs_f64()
        )
    }
}
