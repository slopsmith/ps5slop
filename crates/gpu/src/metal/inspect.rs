//! The control socket's GPU tools: `gputrace`, `gpusnap`, `targets`, `gpuskip`, `gpureload`.

use super::*;

/// Render targets by guest address, for presenting flipped buffers.
static FRAMES: OnceLock<Mutex<HashMap<u64, SharedTexture>>> = OnceLock::new();

pub(super) fn frames() -> &'static Mutex<HashMap<u64, SharedTexture>> {
    FRAMES.get_or_init(Default::default)
}

/// Frames still to trace, from the next one on (the control socket's `gputrace`).
pub static TRACE_NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records targets and textures for the control socket's `targets`, set
/// at run time; `PS5SLOP_DUMP_TARGETS` sets it from the start.
pub static DUMP_TARGETS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn dump_targets() -> bool {
    DUMP_TARGETS.load(std::sync::atomic::Ordering::Relaxed) || diag::config().gpu.dump_targets
}

/// Keeps `texture` under `key` for `targets`, replacing what is there when `replace`.
pub(super) fn record(key: u64, texture: &Retained<Texture>, replace: bool) {
    if dump_targets()
        && let Ok(mut f) = frames().lock()
        && (replace || !f.contains_key(&key))
    {
        f.insert(key, SharedTexture(texture.clone()));
    }
}

/// Where the next frame's render passes are written as images, one file per
/// attachment as each pass ends (the control socket's `gpusnap`).
/// Passes before the given index are counted but not copied.
pub static SNAP: Mutex<Option<(std::path::PathBuf, u32)>> = Mutex::new(None);

/// Bytes of attachment copies one `gpusnap` keeps at most.
const SNAP_BYTES: u64 = 1 << 30;

/// Copies of render pass attachments taken during one frame.
pub(super) struct Snap {
    dir: std::path::PathBuf,
    from: u32,
    passes: u32,
    pub(super) bytes: u64,
    /// File name and copy; depth copies are `R32Float`.
    copies: Vec<(String, Retained<Texture>)>,
    /// The programs of the open pass's draws, in order, without repeats.
    pub(super) programs: Vec<(u64, u64)>,
    pub(super) draws: u32,
}

/// Set by the control socket's `gpureload`: drop translated programs and
/// pipelines at the end of the frame, so `PS5SLOP_MSL_OVERRIDE` files are read again.
pub static RELOAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Programs whose draws and dispatches are skipped, for finding the pass
/// that causes a rendering bug (the control socket's `gpuskip`).
pub static SKIP: Mutex<Vec<u64>> = Mutex::new(Vec::new());

pub(super) fn skipped(address: u64) -> bool {
    SKIP.lock().is_ok_and(|s| s.contains(&address))
}

/// The texture the GPU rendered at guest address `address`, if any.
pub fn frame(address: u64) -> Option<SharedTexture> {
    frames().lock().ok()?.get(&address).cloned()
}

/// Every render target, by guest address.
pub fn all_frames() -> Vec<(u64, SharedTexture)> {
    frames()
        .lock()
        .map(|f| f.iter().map(|(a, t)| (*a, t.clone())).collect())
        .unwrap_or_default()
}

impl Metal {
    /// For `gpusnap`: copies the 2D textures a pixel program samples, as they are just before its draw.
    pub(super) fn snap_inputs(&mut self, program: u64, bound: &Bound) {
        self.end_pass();
        let command = self.command();
        let Some(snap) = self.snap.as_mut() else {
            return;
        };
        if snap.passes < snap.from {
            return;
        }
        let Some(blit) = command.blitCommandEncoder() else {
            return;
        };
        for (slot, t) in &bound.textures {
            let size = (t.width() * t.height() * texel_bytes(t.pixelFormat()).unwrap_or(16)) as u64;
            if t.textureType() != MTLTextureType::Type2D
                || snap.bytes + size > SNAP_BYTES
                || is_compressed(t.pixelFormat())
            {
                continue;
            }
            let d = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    t.pixelFormat(),
                    t.width(),
                    t.height(),
                    false,
                )
            };
            d.setUsage(MTLTextureUsage::ShaderRead);
            d.setStorageMode(MTLStorageMode::Private);
            let Some(copy) = self.device.newTextureWithDescriptor(&d) else {
                continue;
            };
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    t, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: t.width(), height: t.height(), depth: 1 }, &copy, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            snap.bytes += size;
            snap.copies.push((
                format!("{:03}-in-{program:x}-t{slot}", snap.passes + 1),
                copy,
            ));
        }
        blit.endEncoding();
    }

    /// Copies the attachments of the pass that just ended for `gpusnap`.
    pub(super) fn snap_pass(
        &mut self,
        addresses: &[u64],
        colors: &[Retained<Texture>],
        depth: Option<(Retained<Texture>, usize)>,
    ) {
        let Some(command) = self.command.clone() else {
            return;
        };
        let Some(snap) = self.snap.as_mut() else {
            return;
        };
        snap.passes += 1;
        let programs: Vec<String> = snap
            .programs
            .iter()
            .map(|(v, p)| format!("{v:x}-{p:x}"))
            .collect();
        let label = format!("{:03}-{}d-{}", snap.passes, snap.draws, programs.join("+"));
        snap.programs.clear();
        snap.draws = 0;
        if snap.passes < snap.from {
            return;
        }
        let Some(blit) = command.blitCommandEncoder() else {
            return;
        };
        let copy_desc = |t: &Texture, format: MTLPixelFormat| {
            let d = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    format,
                    t.width(),
                    t.height(),
                    false,
                )
            };
            d.setUsage(MTLTextureUsage::ShaderRead);
            d.setStorageMode(MTLStorageMode::Private);
            d
        };
        for (i, t) in colors.iter().enumerate() {
            let size = (t.width() * t.height() * texel_bytes(t.pixelFormat()).unwrap_or(16)) as u64;
            if snap.bytes + size > SNAP_BYTES || is_compressed(t.pixelFormat()) {
                continue;
            }
            let Some(copy) = self
                .device
                .newTextureWithDescriptor(&copy_desc(t, t.pixelFormat()))
            else {
                continue;
            };
            let slice = (addresses[i] >> 48) as usize;
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    t, slice, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: t.width(), height: t.height(), depth: 1 }, &copy, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            snap.bytes += size;
            snap.copies.push((
                format!(
                    "{label}-mrt{i}-{:x}-s{slice}",
                    addresses[i] & 0xFFFF_FFFF_FFFF
                ),
                copy,
            ));
        }
        if let Some((t, slice)) = depth {
            let (w, h) = (t.width(), t.height());
            let size = (w * h * 4) as u64;
            let staging = self
                .device
                .newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModePrivate);
            if let (true, Some(staging), Some(copy)) = (
                snap.bytes + 2 * size <= SNAP_BYTES,
                staging,
                self.device
                    .newTextureWithDescriptor(&copy_desc(&t, MTLPixelFormat::R32Float)),
            ) {
                let options = if t.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8 {
                    MTLBlitOption::DepthFromDepthStencil
                } else {
                    MTLBlitOption::empty()
                };
                if t.pixelFormat() != MTLPixelFormat::Depth16Unorm {
                    unsafe {
                        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                            &t, slice, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &staging, 0, w * 4, w * h * 4, options,
                        );
                        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            &staging, 0, w * 4, w * h * 4, MTLSize { width: w, height: h, depth: 1 }, &copy, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                        );
                    }
                    snap.bytes += 2 * size;
                    let depth_address = addresses.last().copied().unwrap_or(0) & 0xFFFF_FFFF_FFFF;
                    snap.copies
                        .push((format!("{label}-depth{slice}-{depth_address:x}"), copy));
                }
            }
        }
        blit.endEncoding();
    }

    /// Writes what `gpusnap` copied during the frame that just finished.
    pub(super) fn write_snap(snap: Snap) {
        let n = snap.copies.len();
        for (name, t) in snap.copies {
            let shared = SharedTexture(t);
            let Some((w, h, px)) = (if shared.0.pixelFormat() == MTLPixelFormat::R32Float {
                read_depth(&shared.0)
            } else {
                read_rgba(&shared)
            }) else {
                continue;
            };
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            ppm.extend(
                px.as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2]]),
            );
            let _ = std::fs::write(snap.dir.join(format!("{name}.ppm")), ppm);
            if diag::config().gpu.snap_raw
                && let Some(raw) = read_raw(&shared.0)
            {
                let _ = std::fs::write(
                    snap.dir.join(format!(
                        "{name}-{}x{}-{}.raw",
                        w,
                        h,
                        shared.0.pixelFormat().0
                    )),
                    raw,
                );
            }
        }
        tracing::info!(
            target: "gpu::snap",
            "gpusnap wrote {n} attachments of {} passes to {}",
            snap.passes,
            snap.dir.display()
        );
    }

    /// `PS5SLOP_TRACE_USER_MEMORY`: what pointer-like user data pairs point at (256 bytes; V# bases too unless `strict`).
    pub(super) fn trace_user_memory(&mut self, host: &dyn Host, user: &[u32], strict: bool) {
        if self.trace.is_none() || !diag::config().gpu.trace_user_memory {
            return;
        }
        for pair in user.windows(2) {
            let ptr = pair[0] as u64 | ((pair[1] & 0xFFFF) as u64) << 32;
            let mut b = [0u8; 256];
            if (!strict || pair[1] >> 16 == 0) && ptr >> 32 != 0 && host.read(ptr, &mut b) {
                let words: Vec<u32> = b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                self.note(|| format!("      at {ptr:#x}: {words:08x?}"));
            }
        }
    }

    /// Starts what `gpusnap` asked for the next frame.
    pub(super) fn frame_tools(&mut self) {
        if RELOAD.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.programs.clear();
            self.variants.clear();
            self.pipelines.clear();
            self.missing_inputs.clear();
        }
        if let Some((dir, from)) = SNAP.lock().ok().and_then(|mut s| s.take()) {
            let _ = std::fs::create_dir_all(&dir);
            self.snap = Some(Snap {
                dir,
                from,
                passes: 0,
                bytes: 0,
                copies: Vec::new(),
                programs: Vec::new(),
                draws: 0,
            });
        }
    }

    /// Notes every context (0xA000 + n) and uconfig (0xC000 + n) register
    /// that changed since the last traced draw.
    pub(super) fn note_changed_registers(&mut self, state: &State) {
        let now: Vec<u32> = state.context.iter().chain(state.uconfig).copied().collect();
        let old = std::mem::replace(&mut self.traced_regs, now);
        let split = state.context.len();
        let changed: Vec<String> = self
            .traced_regs
            .iter()
            .enumerate()
            .filter(|&(k, v)| old.get(k) != Some(v))
            .map(|(k, v)| {
                let reg = if k < split {
                    0xA000 + k
                } else {
                    0xC000 + k - split
                };
                format!("{reg:x}={v:x}")
            })
            .collect();
        self.note(|| format!("changed {}", changed.join(" ")));
    }

    pub(super) fn tracing(&self) -> bool {
        self.trace_frames
            .is_some_and(|(a, b)| (a..=b).contains(&self.epoch))
    }

    pub(super) fn end_trace(
        &mut self,
        what: impl FnOnce() -> String,
        result: Result<(), &'static str>,
    ) {
        if let Some(lines) = self.trace.take() {
            let mut out = format!("{} -> {result:?}", what());
            for l in lines {
                out += &format!("\n    {l}");
            }
            tracing::info!(target: "gpu::trace", "{out}");
        }
    }
}
