//! Colour and depth targets, their fast clears and depth states.

use super::*;

/// Depth target array slices we keep at most (cascaded shadow maps use 4).
const MAX_DEPTH_SLICES: usize = 16;

/// Color target array slices we keep at most (cube maps use 6).
pub(super) const MAX_TARGET_SLICES: u32 = 16;

/// The pass address of the stand-in target of draws without targets.
pub(super) const BLANK_TARGET: u64 = 0xFFFF_FFFF_F000;

pub(super) struct Target {
    pub(super) texture: Retained<Texture>,
    pub(super) pixel_format: MTLPixelFormat,
}

impl Metal {
    pub(super) fn blank_target(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<Retained<Texture>, &'static str> {
        if let Some(t) = self.blank_targets.get(&(width, height)) {
            return Ok(t.clone());
        }
        let t = self
            .new_texture(MTLPixelFormat::R8Unorm, width, height, true)
            .ok_or("blank target")?;
        self.blank_targets.insert((width, height), t.clone());
        Ok(t)
    }

    /// The texture backing color target `t`, made (and filled from guest memory) on first use.
    pub(super) fn target(
        &mut self,
        host: &dyn Host,
        t: &ColorTarget,
    ) -> Option<(Retained<Texture>, MTLPixelFormat)> {
        let Some(format) = color_format(t.format) else {
            diag::problems::record(
                diag::problems::Kind::UnsupportedFormat,
                format!("color format {:?}", t.format),
                format!("target {:#x}", t.address),
            );
            return None;
        };
        self.color_newer.insert(t.address);
        if t.fast_clear.is_some() || t.dcc.is_some() {
            self.color_meta.insert(t.address, *t);
        } else {
            self.color_meta.remove(&t.address);
        }
        let volume_newer =
            self.volumes.contains_key(&t.address) && !self.volume_older.contains(&t.address);
        if self.volumes.contains_key(&t.address) {
            self.volume_older.insert(t.address);
        }
        let fits = |tex: &Texture| {
            tex.pixelFormat() == format
                && tex.width() == t.width as usize
                && tex.height() == t.height as usize
        };
        let old = self
            .targets
            .get(&t.address)
            .map(|e| e.texture.clone())
            .filter(|e| fits(e));
        // What a program stored at this address since the target was last drawn.
        let volume = self
            .volumes
            .get(&t.address)
            .filter(|v| volume_newer && fits(v) && v.textureType() == MTLTextureType::Type2DArray)
            .cloned();
        let mut slices = t.slices.max(t.slice + 1);
        // A face of a cube drawn on its own (one slice in the view) keeps the array.
        if let Some(tex) = &old
            && tex.textureType() == MTLTextureType::Type2DArray
        {
            slices = slices.max(tex.arrayLength() as u32);
        }
        let slices = slices.min(MAX_TARGET_SLICES);
        let layered = slices > 1;
        let levels = if layered { 1 } else { t.max_mip + 1 };
        if let Some(existing) = &old
            && (existing.textureType() == MTLTextureType::Type2DArray) == layered
            && existing.arrayLength() >= slices as usize
            && (levels == 1 || existing.mipmapLevelCount() > t.mip_level as usize)
        {
            if let Some(v) = &volume {
                self.copy_slices(v, existing);
            }
            record(t.address | 2, existing, false);
            return Some((existing.clone(), format));
        }
        if tracing::enabled!(target: "gpu::targets", tracing::Level::DEBUG) {
            tracing::debug!(
                target: "gpu::targets",
                "frame {} new target {:#x} {}x{} {format:?}{}",
                self.epoch,
                t.address,
                t.width,
                t.height,
                if self.targets.contains_key(&t.address) {
                    " (replacing)"
                } else {
                    ""
                }
            );
        }
        let texture = if layered {
            self.new_target_array(format, t.width, t.height, slices)?
        } else {
            self.new_texture_levels(format, t.width, t.height, true, levels)?
        };
        if let Some(old) = &old {
            self.copy_slices(old, &texture);
        } else if let (Some(mode), Some(bpe)) = (t.tile, t.format.bytes())
            && let Some(layout) = Layout::new(mode, bpe, t.width, t.height)
        {
            let at = t.address + layout.mip0_offset(t.max_mip + 1);
            let stride = layout.size().next_multiple_of(64 << 10);
            for slice in 0..slices as usize {
                self.detile_into(host, &texture, slice, at + slice as u64 * stride, layout);
            }
        }
        if let Some(v) = &volume {
            self.copy_slices(v, &texture);
        }
        self.targets.insert(
            t.address,
            Target {
                texture: texture.clone(),
                pixel_format: format,
            },
        );
        record(t.address | 2, &texture, true);
        Some((texture, format))
    }

    /// Copies level 0 of every slice the two same-sized textures share.
    fn copy_slices(&mut self, src: &Texture, dst: &Texture) {
        self.end_pass();
        let command = self.command();
        let Some(blit) = command.blitCommandEncoder() else {
            return;
        };
        for slice in 0..src.arrayLength().min(dst.arrayLength()) {
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount(
                    src, slice, 0, dst, slice, 0, 1, 1,
                );
            }
        }
        blit.endEncoding();
    }

    /// The depth texture at `d`'s address, adopting any recent image store
    /// there; new ones start cleared.
    pub(super) fn depth(
        &mut self,
        d: &DepthTarget,
        width: u32,
        height: u32,
    ) -> Option<(Retained<Texture>, MTLPixelFormat, bool)> {
        let (t, format, fresh) = self.depth_texture(d, width, height)?;
        self.color_newer.remove(&d.address);
        if !self.depth_stores.remove(&d.address) {
            return Some((t, format, fresh));
        }
        let stored = self
            .volumes
            .get(&d.address)
            .cloned()
            .or_else(|| self.targets.get(&d.address).map(|x| x.texture.clone()))
            .filter(|s| {
                s.pixelFormat() == MTLPixelFormat::R32Float
                    && s.width() == t.width()
                    && s.height() == t.height()
            });
        let Some(src) = stored.filter(|_| format != MTLPixelFormat::Depth16Unorm) else {
            return Some((t, format, fresh));
        };
        let (w, h) = (t.width(), t.height());
        let staging = self
            .device
            .newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModePrivate)?;
        self.end_pass();
        let command = self.command();
        let blit = command.blitCommandEncoder()?;
        let options = if format == MTLPixelFormat::Depth32Float_Stencil8 {
            MTLBlitOption::DepthFromDepthStencil
        } else {
            MTLBlitOption::empty()
        };
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &src, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &staging, 0, w * 4, w * h * 4,
            );
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                &staging, 0, w * 4, w * h * 4, MTLSize { width: w, height: h, depth: 1 }, &t, d.slice as usize, 0, MTLOrigin { x: 0, y: 0, z: 0 }, options,
            );
        }
        blit.endEncoding();
        self.note(|| {
            format!(
                "      depth {:#x} filled from the image a program stored there",
                d.address
            )
        });
        Some((t, format, false))
    }

    /// Applies a colour fast clear made through `t`'s CMASK or DCC, judged by
    /// its first dword, which is then marked expanded (see gpu.md).
    pub(super) fn fast_clear(&mut self, host: &dyn Host, t: &ColorTarget, tex: &Texture) {
        let mut word = [0u8; 4];
        if let Some((cmask, words)) = t.fast_clear {
            self.settle_for(cmask, 4, "CMASK");
            if host.read(cmask, &mut word) && u32::from_le_bytes(word) & 0xF == 0 {
                host.write(cmask, &(u32::from_le_bytes(word) | 0xF).to_le_bytes());
                self.fill_raw(t, tex, words);
                self.note(|| {
                    format!(
                        "      color {:#x} fast-cleared through CMASK at {cmask:#x} to {words:x?}",
                        t.address
                    )
                });
            }
        }
        if let Some(dcc) = t.dcc {
            self.settle_for(dcc, 4, "DCC");
            if !host.read(dcc, &mut word) {
                return;
            }
            let code = u32::from_le_bytes(word);
            let colour = match code {
                0x0000_0000 => [0.0, 0.0, 0.0, 0.0],
                0x4040_4040 => [0.0, 0.0, 0.0, 1.0],
                0x8080_8080 => [1.0, 1.0, 1.0, 0.0],
                0xC0C0_C0C0 => [1.0, 1.0, 1.0, 1.0],
                0x2020_2020 | 0x1010_1010 => [f64::NAN; 4],
                _ => return,
            };
            // 0x10 ("single", Mesa's GFX9_DCC_CLEAR_SINGLE): each 256-byte block
            // holds its colour in its first element, which the clearing program stored.
            let mut single = [0u8; 8];
            let bpp = texel_bytes(tex.pixelFormat()).unwrap_or(0).min(8);
            if code == 0x1010_1010 {
                self.settle_for(t.address, bpp as u64, "DCC");
                if bpp == 0 || !host.read(t.address, &mut single[..bpp]) {
                    return;
                }
            }
            host.write(dcc, &u32::MAX.to_le_bytes());
            if code == 0x1010_1010 {
                let words = [
                    u32::from_le_bytes(single[..4].try_into().unwrap()),
                    u32::from_le_bytes(single[4..].try_into().unwrap()),
                ];
                self.fill_raw(t, tex, words);
            } else if colour[0].is_nan() {
                let words = t.fast_clear.map_or([0; 2], |f| f.1);
                self.fill_raw(t, tex, words);
            } else {
                self.fill_colour(t, tex, colour);
            }
            self.note(|| {
                format!(
                    "      color {:#x} fast-cleared through DCC at {dcc:#x} with code {code:#x}",
                    t.address
                )
            });
        }
    }

    fn target_slice(t: &ColorTarget, tex: &Texture) -> usize {
        if tex.textureType() == MTLTextureType::Type2DArray {
            t.slice as usize
        } else {
            0
        }
    }

    /// Fills `tex` with the raw bits of `words` per texel (up to 8 bytes).
    fn fill_raw(&mut self, t: &ColorTarget, tex: &Texture, words: [u32; 2]) {
        let Some(bpp) = texel_bytes(tex.pixelFormat()).filter(|&b| b <= 8) else {
            return;
        };
        let (w, h) = (tex.width(), tex.height());
        let pixel: Vec<u8> = words
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .take(bpp)
            .collect();
        let Some((buf, offset)) = self.scratch(w * h * bpp) else {
            return;
        };
        // SAFETY: the scratch range is fresh shared memory of w * h * bpp bytes.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                buf.contents().as_ptr().cast::<u8>().add(offset),
                w * h * bpp,
            )
        };
        for p in dst.chunks_exact_mut(bpp) {
            p.copy_from_slice(&pixel);
        }
        self.end_pass();
        let command = self.command();
        let Some(blit) = command.blitCommandEncoder() else {
            return;
        };
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &buf, offset, w * bpp, w * h * bpp, MTLSize { width: w, height: h, depth: 1 }, tex, Self::target_slice(t, tex), 0, MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
    }

    /// Clears `tex` to `colour` with a render pass of its own.
    fn fill_colour(&mut self, t: &ColorTarget, tex: &Texture, colour: [f64; 4]) {
        self.end_pass();
        let command = self.command();
        let pass = MTLRenderPassDescriptor::new();
        let att = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        att.setTexture(Some(tex));
        att.setSlice(Self::target_slice(t, tex));
        att.setLoadAction(MTLLoadAction::Clear);
        att.setStoreAction(MTLStoreAction::Store);
        att.setClearColor(MTLClearColor {
            red: colour[0],
            green: colour[1],
            blue: colour[2],
            alpha: colour[3],
        });
        if let Some(enc) = command.renderCommandEncoderWithDescriptor(&pass) {
            enc.endEncoding();
        }
    }

    /// The stencil plane of a depth target, read as 8-bit integers.
    pub(super) fn stencil_view(
        &mut self,
        image: Image,
        kind: TextureKind,
        format: MTLPixelFormat,
    ) -> Option<Retained<Texture>> {
        if format != MTLPixelFormat::R8Uint
            || !matches!(kind, TextureKind::Tex2d | TextureKind::Tex2dArray)
        {
            return None;
        }
        let depth = *self.stencils.get(&image.address)?;
        let d = self.depths.get(&depth)?.clone();
        if d.pixelFormat() != MTLPixelFormat::Depth32Float_Stencil8
            || d.width() as u32 != image.width
            || d.height() as u32 != image.height
        {
            return None;
        }
        let base = image.base_array as usize;
        let count = if kind == TextureKind::Tex2dArray {
            (image.depth as usize).min(d.arrayLength().saturating_sub(base))
        } else {
            1
        };
        if count == 0 {
            return None;
        }
        let ty = if kind == TextureKind::Tex2dArray {
            MTLTextureType::Type2DArray
        } else {
            MTLTextureType::Type2D
        };
        self.note(|| {
            format!(
                "      {:#x} the stencil plane of depth target {depth:#x}",
                image.address
            )
        });
        unsafe {
            d.newTextureViewWithPixelFormat_textureType_levels_slices(
                MTLPixelFormat::X32_Stencil8,
                ty,
                objc2_foundation::NSRange::new(0, 1),
                objc2_foundation::NSRange::new(base, count),
            )
        }
    }

    /// The depth texture at `d`'s address; new ones start cleared.
    pub(super) fn depth_texture(
        &mut self,
        d: &DepthTarget,
        width: u32,
        height: u32,
    ) -> Option<(Retained<Texture>, MTLPixelFormat, bool)> {
        let format = match (d.format, d.stencil) {
            (_, true) => MTLPixelFormat::Depth32Float_Stencil8,
            (1, false) => MTLPixelFormat::Depth16Unorm,
            _ => MTLPixelFormat::Depth32Float,
        };
        let (w, h) = if d.width != 0 {
            (d.width, d.height)
        } else {
            (width, height)
        };
        if d.slice >= MAX_DEPTH_SLICES as u32 {
            diag::problems::record(
                diag::problems::Kind::UnsupportedFormat,
                format!("depth slice {}", d.slice),
                format!("depth {:#x}", d.address),
            );
            self.report(
                tracing::Level::ERROR,
                format!(
                    "depth target slice {} at {:#x} not supported",
                    d.slice, d.address
                ),
            );
            return None;
        }
        let slices = (d.slice.max(d.slice_max) as usize + 1).min(MAX_DEPTH_SLICES);
        let old = self.depths.get(&d.address).cloned();
        if let Some(t) = &old
            && t.pixelFormat() == format
            && t.width() == w as usize
            && t.height() == h as usize
            && t.arrayLength() >= slices
        {
            record(d.address | 4, t, false);
            return Some((t.clone(), format, false));
        }
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format, w as usize, h as usize, false,
            )
        };
        if slices > 1 {
            desc.setTextureType(MTLTextureType::Type2DArray);
            // SAFETY: a small positive count, below Metal's array limit.
            unsafe { desc.setArrayLength(slices) };
        }
        desc.setUsage(
            MTLTextureUsage::RenderTarget
                | MTLTextureUsage::ShaderRead
                | MTLTextureUsage::PixelFormatView,
        );
        desc.setStorageMode(MTLStorageMode::Private);
        let t = self.device.newTextureWithDescriptor(&desc)?;
        let kept = old.filter(|o| {
            o.pixelFormat() == format && o.width() == w as usize && o.height() == h as usize
        });
        if let Some(o) = &kept {
            self.end_pass();
            let command = self.command();
            let blit = command.blitCommandEncoder()?;
            for slice in 0..o.arrayLength() {
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount(
                        o, slice, 0, &t, slice, 0, 1, 1,
                    );
                }
            }
            blit.endEncoding();
        }
        self.depths.insert(d.address, t.clone());
        self.blank.retain(|(i, _), _| i.address != d.address);
        record(d.address | 4, &t, true);
        Some((t, format, kept.is_none()))
    }

    pub(super) fn depth_state(
        &mut self,
        s: DepthState,
        bound: bool,
    ) -> Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
        if let Some(x) = self.depth_states.get(&(s, bound)) {
            return Some(x.clone());
        }
        let d = MTLDepthStencilDescriptor::new();
        if bound && s.depth_test() {
            d.setDepthCompareFunction(compare(s.depth_func()));
            d.setDepthWriteEnabled(s.depth_write());
        }
        if bound && s.stencil() {
            let face = |func: u32, fail: u32, zpass: u32, zfail: u32, mask: u32| {
                let f = MTLStencilDescriptor::new();
                f.setStencilCompareFunction(compare(func));
                f.setStencilFailureOperation(stencil_op(fail));
                f.setDepthStencilPassOperation(stencil_op(zpass));
                f.setDepthFailureOperation(stencil_op(zfail));
                f.setReadMask((mask >> 8) & 0xFF);
                f.setWriteMask((mask >> 16) & 0xFF);
                f
            };
            let sc = s.stencil_control;
            let front = face(
                (s.control >> 8) & 7,
                sc & 0xF,
                (sc >> 4) & 0xF,
                (sc >> 8) & 0xF,
                s.ref_mask,
            );
            let back_on = s.control & (1 << 7) != 0;
            let back = if back_on {
                face(
                    (s.control >> 20) & 7,
                    (sc >> 12) & 0xF,
                    (sc >> 16) & 0xF,
                    (sc >> 20) & 0xF,
                    s.ref_mask_back,
                )
            } else {
                face(
                    (s.control >> 8) & 7,
                    sc & 0xF,
                    (sc >> 4) & 0xF,
                    (sc >> 8) & 0xF,
                    s.ref_mask,
                )
            };
            d.setFrontFaceStencil(Some(&front));
            d.setBackFaceStencil(Some(&back));
        }
        let state = self.device.newDepthStencilStateWithDescriptor(&d)?;
        self.depth_states.insert((s, bound), state.clone());
        Some(state)
    }
}
