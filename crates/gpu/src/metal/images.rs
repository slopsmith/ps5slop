//! Textures for image descriptors: cached copies of guest memory, storage images and samplers.

use super::*;

/// Frames an image read from guest memory stays cached without being used.
pub(super) const IMAGE_LIFETIME: u64 = 300;

/// The guest memory a 2D-array storage image was last loaded from.
#[derive(Clone)]
pub(super) struct VolumeSource {
    pub(super) image: Image,
    pub(super) bpe: u32,
    pub(super) slices: u32,
    pub(super) base: u64,
    pub(super) len: u64,
    shadow: Option<Arc<Vec<u8>>>,
    pub(super) epoch: u64,
}

impl Metal {
    /// The texture for image descriptor `image` read as `kind`: a render
    /// target at that address, or a copy of guest memory refreshed once per frame.
    pub(super) fn image(
        &mut self,
        host: &dyn Host,
        image: Image,
        kind: TextureKind,
    ) -> Option<Retained<Texture>> {
        let (format, bpe) = raw_format(image.format)?;
        let depth_current = !self.color_newer.contains(&image.address);
        if kind == TextureKind::Cube
            && depth_current
            && let Some(mut d) = self.depths.get(&image.address).cloned()
            && d.width() as u32 == image.width
            && d.height() as u32 == image.height
        {
            let base = image.base_array as usize;
            if d.textureType() != MTLTextureType::Type2DArray || d.arrayLength() < base + 6 {
                // Grow partial shadow cubes to 6 slices so cube views are valid.
                let last = (base + 5) as u32;
                let target = DepthTarget {
                    address: image.address,
                    format: if d.pixelFormat() == MTLPixelFormat::Depth16Unorm {
                        1
                    } else {
                        3
                    },
                    stencil: d.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8,
                    stencil_address: 0,
                    width: image.width,
                    height: image.height,
                    slice: last,
                    slice_max: last,
                };
                d = self.depth_texture(&target, image.width, image.height)?.0;
            }
            self.note(|| {
                format!(
                    "      {:#x} a cube of a layered depth target",
                    image.address
                )
            });
            return unsafe {
                d.newTextureViewWithPixelFormat_textureType_levels_slices(
                    d.pixelFormat(),
                    MTLTextureType::TypeCube,
                    objc2_foundation::NSRange::new(0, 1),
                    objc2_foundation::NSRange::new(base, 6),
                )
            };
        }
        if matches!(kind, TextureKind::Tex2d | TextureKind::Tex2dArray)
            && depth_current
            && let Some(d) = self.depths.get(&image.address)
            && d.width() as u32 == image.width
            && d.height() as u32 == image.height
        {
            let array = kind == TextureKind::Tex2dArray;
            let available = d.arrayLength().saturating_sub(image.base_array as usize);
            let count = if array {
                (image.depth as usize).min(available)
            } else {
                1
            };
            if count == 0 {
                return None;
            }
            if d.textureType()
                == if array {
                    MTLTextureType::Type2DArray
                } else {
                    MTLTextureType::Type2D
                }
                && image.base_array == 0
                && count == d.arrayLength()
            {
                return Some(d.clone());
            }
            let ty = if array {
                MTLTextureType::Type2DArray
            } else {
                MTLTextureType::Type2D
            };
            return unsafe {
                d.newTextureViewWithPixelFormat_textureType_levels_slices(
                    d.pixelFormat(),
                    ty,
                    objc2_foundation::NSRange::new(0, 1),
                    objc2_foundation::NSRange::new(image.base_array as usize, count),
                )
            };
        }
        if let Some(view) = self.stencil_view(image, kind, format) {
            return Some(view);
        }
        if !self.volume_older.contains(&image.address) {
            self.refresh_volume(host, image.address, false);
            if let Some(v) = self.volume(image, kind, format) {
                self.note(|| format!("      {:#x} from a 2D array or 3D image", image.address));
                return Some(v);
            }
        }
        // A depth layout no draw rendered reads as zeros, unless a program stored it.
        let stored = self.targets.get(&image.address).is_some_and(|t| {
            t.texture.width() as u32 == image.width && t.texture.height() as u32 == image.height
        });
        if image.tile.is_none()
            && is_depth_layout(image.raw_tile)
            && kind != TextureKind::Tex3d
            && !stored
        {
            return self.blank_texture(image, kind, format);
        }
        if kind == TextureKind::Tex3d
            && let Some(t) = self.standard_volume(host, image, format, bpe)
        {
            return Some(t);
        }
        if kind == TextureKind::Tex3d
            && image.kind == 10
            && image.tile == Some(TileMode::Linear)
            && !is_compressed(format)
        {
            return self.linear_volume(host, image, format, bpe);
        }
        let slices = match kind {
            TextureKind::Tex2d => 1,
            TextureKind::Cube => 6,
            TextureKind::Tex2dArray => image.depth as usize,
            TextureKind::Tex3d if image.kind != 10 || one_texel_volume(image) => {
                return self.flat_volume(host, image, format);
            }
            TextureKind::Tex3d => return None,
        };
        if let Some(t) = self.targets.get(&image.address)
            && t.texture.textureType() == MTLTextureType::Type2DArray
            && t.texture.width() as u32 == image.width
            && t.texture.height() as u32 == image.height
        {
            let (tex, have) = (t.texture.clone(), t.pixel_format);
            let (base, len) = (image.base_array as usize, tex.arrayLength());
            let (ty, count) = match kind {
                TextureKind::Tex2d if base < len => (MTLTextureType::Type2D, 1),
                TextureKind::Tex2dArray if base < len => (
                    MTLTextureType::Type2DArray,
                    (image.depth as usize).clamp(1, len - base),
                ),
                TextureKind::Cube if base + 6 <= len => (MTLTextureType::TypeCube, 6),
                _ => return None,
            };
            if have != format && !view_compatible(have, format) {
                diag::problems::record(
                    diag::problems::Kind::UnsupportedFormat,
                    format!("incompatible view {have:?} as {format:?}"),
                    format!("target {:#x}", image.address),
                );
                self.report(
                    tracing::Level::ERROR,
                    format!(
                        "render target at {:#x} ({have:?}) cannot be viewed as {format:?}",
                        image.address
                    ),
                );
                return None;
            }
            self.note(|| format!("      {:#x} from a layered render target", image.address));
            return unsafe {
                tex.newTextureViewWithPixelFormat_textureType_levels_slices(
                    format,
                    ty,
                    objc2_foundation::NSRange::new(0, 1),
                    objc2_foundation::NSRange::new(base, count),
                )
            };
        }
        if kind == TextureKind::Tex2d
            && let Some(t) = self.targets.get(&image.address)
            && t.texture.textureType() == MTLTextureType::Type2D
            && t.texture.width() as u32 == image.width
            && t.texture.height() as u32 == image.height
            && (t.pixel_format == format || view_compatible(t.pixel_format, format))
            && let Some(meta) = self.color_meta.get(&image.address).copied()
        {
            // A clear made through the metadata since the last draw is what sampling sees.
            let texture = t.texture.clone();
            self.fast_clear(host, &meta, &texture);
        }
        if kind == TextureKind::Tex2d
            && let Some(t) = self.targets.get(&image.address)
            && t.texture.textureType() == MTLTextureType::Type2D
        {
            let tex = &t.texture;
            if tex.width() as u32 == image.width && tex.height() as u32 == image.height {
                // Memory reused for another surface: read guest memory below.
                if t.pixel_format == format || view_compatible(t.pixel_format, format) {
                    let count = tex.mipmapLevelCount();
                    let base = (image.base_level as usize).min(count - 1);
                    let last = (image.last_level as usize).clamp(base, count - 1);
                    let found = if base > 0 || last + 1 < count {
                        unsafe {
                            tex.newTextureViewWithPixelFormat_textureType_levels_slices(
                                format,
                                MTLTextureType::Type2D,
                                objc2_foundation::NSRange::new(base, last + 1 - base),
                                objc2_foundation::NSRange::new(0, 1),
                            )
                        }
                    } else if t.pixel_format == format {
                        Some(tex.clone())
                    } else {
                        tex.newTextureViewWithPixelFormat(format)
                    };
                    let ptr = Retained::as_ptr(tex) as usize;
                    self.note(|| {
                        format!("      {:#x} from a render target {ptr:#x}", image.address)
                    });
                    return found;
                }
                let have = t.pixel_format;
                self.note(|| format!("      {:#x}: the render target there ({have:?}) cannot be viewed as {format:?}", image.address));
            }
        }
        if let Some((epoch, tex)) = self.images.get(&(image, kind))
            && *epoch == self.epoch
        {
            record(image.address | 1, tex, false);
            return Some(tex.clone());
        }
        let compressed = is_compressed(format);
        let (ew, eh) = if compressed {
            (image.width.div_ceil(4), image.height.div_ceil(4))
        } else {
            (image.width, image.height)
        };
        let layout = Layout::new(image.tile?, bpe, ew, eh)?;
        let chain = image.last_level + 1;
        // A 2D image bound where a cube is read shows its one slice on every face.
        let stride = if kind == TextureKind::Cube && image.kind != 11 {
            0
        } else {
            layout
                .chain_size(chain)
                .next_multiple_of(layout.block_bytes())
        };
        let extent = (slices as u64 - 1) * stride + layout.chain_size(chain);
        let digest = if self.written.covers(image.address, extent) {
            None
        } else {
            guest_bytes(host, image.address, extent).map(digest)
        };
        if let Some((epoch, tex)) = self.images.get_mut(&(image, kind)) {
            let old = self.image_digests.get(&(image, kind));
            if digest.is_some() && old == digest.as_ref() {
                *epoch = self.epoch;
                self.image_checks[0] += 1;
                record(image.address | 1, tex, false);
                return Some(tex.clone());
            }
            self.image_checks[if old.is_some() { 1 } else { 2 }] += 1;
        }
        self.note(|| format!("      {:#x} detiled from guest memory", image.address));
        if tracing::enabled!(target: "gpu::targets", tracing::Level::DEBUG)
            && !self.images.contains_key(&(image, kind))
        {
            tracing::debug!(
                target: "gpu::targets",
                "frame {} image {:#x} {}x{} from guest memory",
                self.epoch, image.address, image.width, image.height
            );
        }
        let levels = sampled_levels(image);
        let tex = match self.images.get(&(image, kind)) {
            Some((_, t)) => t.clone(),
            None => {
                self.new_layered_texture(format, (image.width, image.height), kind, slices, levels)?
            }
        };
        let mips = layout.mips(
            chain,
            (image.width, image.height),
            if compressed { 4 } else { 1 },
        );
        for slice in 0..slices {
            let at = image.address + slice as u64 * stride;
            for (level, mip) in mips.iter().enumerate().take(levels as usize) {
                let filled = if compressed {
                    self.detile_blocks_into(host, &tex, (slice, level), at, *mip)
                } else {
                    self.detile_level_into(host, &tex, slice, level, at, *mip)
                };
                if !filled {
                    return None;
                }
            }
        }
        self.images.insert((image, kind), (self.epoch, tex.clone()));
        match digest {
            Some(d) => self.image_digests.insert((image, kind), d),
            None => self.image_digests.remove(&(image, kind)),
        };
        record(image.address | 1, &tex, false);
        Some(tex)
    }

    /// A 2D image bound where a 3D texture is read, as a one-slice 3D texture.
    fn flat_volume(
        &mut self,
        host: &dyn Host,
        image: Image,
        format: MTLPixelFormat,
    ) -> Option<Retained<Texture>> {
        if let Some((epoch, tex)) = self.images.get(&(image, TextureKind::Tex3d))
            && *epoch == self.epoch
        {
            return Some(tex.clone());
        }
        let flat = self.image(host, image, TextureKind::Tex2d)?;
        let volume = match self.images.get(&(image, TextureKind::Tex3d)) {
            Some((_, t)) => t.clone(),
            None => {
                let desc = MTLTextureDescriptor::new();
                desc.setPixelFormat(format);
                desc.setTextureType(MTLTextureType::Type3D);
                unsafe {
                    desc.setWidth(image.width as usize);
                    desc.setHeight(image.height as usize);
                    desc.setDepth(1);
                }
                desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
                desc.setStorageMode(MTLStorageMode::Private);
                self.device.newTextureWithDescriptor(&desc)?
            }
        };
        self.end_pass();
        let command = self.command();
        let blit = command.blitCommandEncoder()?;
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &flat,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: image.width as usize, height: image.height as usize, depth: 1 },
                &volume,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
        self.images
            .insert((image, TextureKind::Tex3d), (self.epoch, volume.clone()));
        Some(volume)
    }

    /// A linear 3D image with its mip levels, copied from guest memory once a frame.
    fn linear_volume(
        &mut self,
        host: &dyn Host,
        image: Image,
        format: MTLPixelFormat,
        bpe: u32,
    ) -> Option<Retained<Texture>> {
        if let Some((epoch, tex)) = self.images.get(&(image, TextureKind::Tex3d))
            && *epoch == self.epoch
        {
            return Some(tex.clone());
        }
        let tex = match self.images.get(&(image, TextureKind::Tex3d)) {
            Some((_, t)) => t.clone(),
            None => {
                tracing::debug!(
                    target: "gpu::targets",
                    "frame {} linear volume {:#x} {}x{}x{} from guest memory",
                    self.epoch,
                    image.address,
                    image.width,
                    image.height,
                    image.depth
                );
                let desc = MTLTextureDescriptor::new();
                desc.setPixelFormat(format);
                desc.setTextureType(MTLTextureType::Type3D);
                unsafe {
                    desc.setWidth(image.width as usize);
                    desc.setHeight(image.height as usize);
                    desc.setDepth(image.depth as usize);
                    desc.setMipmapLevelCount(volume_levels(image) as usize);
                }
                desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
                desc.setStorageMode(MTLStorageMode::Private);
                self.device.newTextureWithDescriptor(&desc)?
            }
        };
        self.fill_linear_volume(host, &tex, image, bpe)?;
        self.note(|| format!("      {:#x} a linear 3D image", image.address));
        self.images
            .insert((image, TextureKind::Tex3d), (self.epoch, tex.clone()));
        record(image.address | 1, &tex, false);
        Some(tex)
    }

    /// Copies a linear 3D image's levels from guest memory into `tex`. Each
    /// depth slice holds a whole mip chain laid out as a linear 2D one, as
    /// for a 2D array (measured on RDNA2).
    fn fill_linear_volume(
        &mut self,
        host: &dyn Host,
        tex: &Texture,
        image: Image,
        bpe: u32,
    ) -> Option<()> {
        let layout = Layout::new(TileMode::Linear, bpe, image.width, image.height)?;
        let chain = image.last_level + 1;
        let stride = layout.chain_size(chain);
        let len = stride * image.depth as u64;
        let (buffer, offset) = self.window(host, image.address, len)?;
        let mips = layout.mips(chain, (image.width, image.height), 1);
        self.end_pass();
        let command = self.command();
        let blit = command.blitCommandEncoder()?;
        for (level, mip) in mips.iter().enumerate().take(tex.mipmapLevelCount()) {
            let l = mip.layout;
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &buffer,
                    offset + mip.offset as usize,
                    (l.pitch * bpe) as usize,
                    stride as usize,
                    MTLSize {
                        width: l.width as usize,
                        height: l.height as usize,
                        depth: (image.depth >> level).max(1) as usize,
                    },
                    tex,
                    0,
                    level,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
        }
        blit.endEncoding();
        Some(())
    }

    /// A 3D image in `SW_4KB_S` with the levels before its mip tail, or of
    /// one texel of any layout, read from guest memory once a frame.
    fn standard_volume(
        &mut self,
        host: &dyn Host,
        image: Image,
        format: MTLPixelFormat,
        bpe: u32,
    ) -> Option<Retained<Texture>> {
        let (w, h, d) = (image.width, image.height, image.depth);
        let one = w == 1 && h == 1 && d == 1;
        if image.kind != 10 || image.base_level != 0 || is_compressed(format) {
            return None;
        }
        let layout = if one {
            None
        } else if image.raw_tile == 5 {
            Some(crate::tiling::Volume::new(
                bpe,
                w,
                h,
                d,
                image.last_level + 1,
            )?)
        } else {
            return None;
        };
        if let Some((epoch, tex)) = self.images.get(&(image, TextureKind::Tex3d))
            && *epoch == self.epoch
        {
            return Some(tex.clone());
        }
        let bpe = bpe as usize;
        let levels: Vec<((u32, u32, u32), Vec<u8>)> = match &layout {
            None => {
                let mut texel = vec![0u8; bpe];
                if !host.read(image.address, &mut texel) {
                    return None;
                }
                vec![((1, 1, 1), texel)]
            }
            Some(v) => {
                let mut raw = vec![0u8; v.bytes() as usize];
                if !host.read(image.address, &mut raw) {
                    return None;
                }
                (0..v.levels.len())
                    .map(|l| {
                        let (lw, lh, ld) = v.levels[l].size;
                        let mut linear = vec![0u8; lw as usize * lh as usize * ld as usize * bpe];
                        let mut at = 0;
                        for z in 0..ld {
                            for y in 0..lh {
                                for x in 0..lw {
                                    let o = v.offset(l, x, y, z) as usize;
                                    if let Some(src) = raw.get(o..o + bpe) {
                                        linear[at..at + bpe].copy_from_slice(src);
                                    }
                                    at += bpe;
                                }
                            }
                        }
                        ((lw, lh, ld), linear)
                    })
                    .collect()
            }
        };
        let tex = match self.images.get(&(image, TextureKind::Tex3d)) {
            Some((_, t)) => t.clone(),
            None => {
                let desc = MTLTextureDescriptor::new();
                desc.setPixelFormat(format);
                desc.setTextureType(MTLTextureType::Type3D);
                let (w, h, d) = levels[0].0;
                unsafe {
                    desc.setWidth(w as usize);
                    desc.setHeight(h as usize);
                    desc.setDepth(d as usize);
                    desc.setMipmapLevelCount(levels.len());
                }
                desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
                desc.setStorageMode(MTLStorageMode::Shared);
                self.device.newTextureWithDescriptor(&desc)?
            }
        };
        for (level, ((w, h, d), mut linear)) in levels.into_iter().enumerate() {
            let (w, h, d) = (w as usize, h as usize, d as usize);
            unsafe {
                tex.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize {
                            width: w,
                            height: h,
                            depth: d,
                        },
                    },
                    level,
                    0,
                    NonNull::new(linear.as_mut_ptr())?.cast(),
                    w * bpe,
                    w * h * bpe,
                );
            }
        }
        self.note(|| format!("      {:#x} a standard 3D image", image.address));
        self.images
            .insert((image, TextureKind::Tex3d), (self.epoch, tex.clone()));
        Some(tex)
    }

    /// `tex` with an image descriptor's channel selects (`DST_SEL_X..W`) applied.
    pub(super) fn swizzled(
        &mut self,
        tex: Retained<Texture>,
        sel: [u8; 4],
    ) -> Option<Retained<Texture>> {
        // Depth textures cannot be viewed; their descriptors select only x.
        if sel == [4, 5, 6, 7] || is_depth_format(tex.pixelFormat()) {
            return Some(tex);
        }
        let key = (Retained::as_ptr(&tex) as usize, sel);
        if let Some(v) = self.swizzles.get(&key) {
            return Some(v.clone());
        }
        let channel = |s: u8| match s {
            0 => MTLTextureSwizzle::Zero,
            1 => MTLTextureSwizzle::One,
            4 => MTLTextureSwizzle::Red,
            5 => MTLTextureSwizzle::Green,
            6 => MTLTextureSwizzle::Blue,
            _ => MTLTextureSwizzle::Alpha,
        };
        let swizzle = MTLTextureSwizzleChannels {
            red: channel(sel[0]),
            green: channel(sel[1]),
            blue: channel(sel[2]),
            alpha: channel(sel[3]),
        };
        let view = unsafe {
            tex.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                tex.pixelFormat(),
                tex.textureType(),
                objc2_foundation::NSRange::new(0, tex.mipmapLevelCount()),
                objc2_foundation::NSRange::new(
                    0,
                    if tex.textureType() == MTLTextureType::TypeCube {
                        6
                    } else {
                        tex.arrayLength()
                    },
                ),
                swizzle,
            )
        }?;
        if self.swizzles.len() > 4096 {
            self.swizzles.clear();
        }
        self.swizzles.insert(key, view.clone());
        Some(view)
    }

    /// A 1x1 transparent black texture of `kind`, for unbound image descriptors.
    pub(super) fn null_texture(&mut self, kind: TextureKind) -> Option<Retained<Texture>> {
        if let Some(t) = self.null.get(&kind) {
            return Some(t.clone());
        }
        let d = MTLTextureDescriptor::new();
        d.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        let slices = match kind {
            TextureKind::Tex2d => 1,
            TextureKind::Cube => {
                d.setTextureType(MTLTextureType::TypeCube);
                6
            }
            TextureKind::Tex2dArray => {
                d.setTextureType(MTLTextureType::Type2DArray);
                1
            }
            TextureKind::Tex3d => {
                d.setTextureType(MTLTextureType::Type3D);
                1
            }
        };
        unsafe {
            d.setWidth(1);
            d.setHeight(1);
        }
        d.setStorageMode(MTLStorageMode::Shared);
        let t = self.device.newTextureWithDescriptor(&d)?;
        let zero = [0u8; 4];
        for slice in 0..slices {
            unsafe {
                t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                    },
                    0,
                    slice,
                    NonNull::from(&zero).cast(),
                    4,
                    4,
                );
            }
        }
        self.null.insert(kind, t.clone());
        Some(t)
    }

    pub(super) fn sampler(
        &mut self,
        s: Sampler,
    ) -> Option<Retained<ProtocolObject<dyn MTLSamplerState>>> {
        if let Some(x) = self.samplers.get(&s) {
            return Some(x.clone());
        }
        let d = MTLSamplerDescriptor::new();
        let filter = |linear| {
            if linear {
                MTLSamplerMinMagFilter::Linear
            } else {
                MTLSamplerMinMagFilter::Nearest
            }
        };
        d.setMinFilter(filter(s.min_linear));
        d.setMagFilter(filter(s.mag_linear));
        d.setMipFilter(if s.mip_linear {
            MTLSamplerMipFilter::Linear
        } else {
            MTLSamplerMipFilter::Nearest
        });
        let mode = |c: u8| match c {
            0 => MTLSamplerAddressMode::Repeat,
            1 => MTLSamplerAddressMode::MirrorRepeat,
            3 | 5 | 7 => MTLSamplerAddressMode::MirrorClampToEdge,
            6 => MTLSamplerAddressMode::ClampToBorderColor,
            _ => MTLSamplerAddressMode::ClampToEdge,
        };
        d.setSAddressMode(mode(s.clamp[0]));
        d.setTAddressMode(mode(s.clamp[1]));
        d.setRAddressMode(mode(s.clamp[2]));
        let state = self.device.newSamplerStateWithDescriptor(&d)?;
        self.samplers.insert(s, state.clone());
        Some(state)
    }

    /// The texture a program writes for image descriptor `image`: kept like a
    /// render target, so later reads of that address see what was written.
    pub(super) fn storage_image(
        &mut self,
        host: &dyn Host,
        image: Image,
        kind: TextureKind,
    ) -> Option<Retained<Texture>> {
        let (format, bpe) = raw_format(image.format)?;
        self.color_newer.insert(image.address);
        if is_depth_layout(image.raw_tile) {
            self.depth_stores.insert(image.address);
        }
        if is_compressed(format) {
            return None;
        }
        let layered = matches!(image.kind, 11 | 13)
            || (self.volumes.contains_key(&image.address)
                && !self.volume_older.contains(&image.address));
        match kind {
            TextureKind::Tex2d if !layered => {}
            TextureKind::Cube => return None,
            _ => {
                self.volume_older.remove(&image.address);
                self.refresh_volume(host, image.address, true);
                if let Some(v) = self.volume(image, kind, format) {
                    return Some(v);
                }
                let slices = image.depth.max(image.base_array + 1);
                let old = self.volumes.get(&image.address).cloned();
                let texture = self.new_volume(
                    format,
                    image,
                    if kind == TextureKind::Tex3d {
                        kind
                    } else {
                        TextureKind::Tex2dArray
                    },
                    slices,
                )?;
                let source = if kind != TextureKind::Tex3d {
                    self.fill_volume(host, &texture, image, bpe, slices)
                } else {
                    if image.tile == Some(TileMode::Linear) {
                        self.fill_linear_volume(host, &texture, image, bpe);
                    }
                    None
                };
                if let Some(old) = old
                    && old.textureType() == MTLTextureType::Type2DArray
                    && texture.textureType() == MTLTextureType::Type2DArray
                    && (old.width(), old.height(), old.pixelFormat())
                        == (texture.width(), texture.height(), texture.pixelFormat())
                {
                    self.end_pass();
                    let command = self.command();
                    let blit = command.blitCommandEncoder()?;
                    for slice in 0..old.arrayLength().min(texture.arrayLength()) {
                        unsafe {
                            blit.copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount(
                                &old, slice, 0, &texture, slice, 0, 1, 1,
                            );
                        }
                    }
                    blit.endEncoding();
                }
                self.volumes.insert(image.address, texture);
                match source {
                    Some(src) => self.volume_sources.insert(image.address, src),
                    None => self.volume_sources.remove(&image.address),
                };
                return self.volume(image, kind, format);
            }
        }
        if let Some(t) = self.targets.get(&image.address)
            && t.texture.textureType() == MTLTextureType::Type2D
            && t.texture.width() as u32 == image.width
            && t.texture.height() as u32 == image.height
            && (t.pixel_format == format || view_compatible(t.pixel_format, format))
            && let Some(meta) = self.color_meta.get(&image.address).copied()
        {
            // Stores leave the blocks they touch uncompressed; the rest keep the clear.
            let texture = t.texture.clone();
            self.fast_clear(host, &meta, &texture);
        }
        if let Some(t) = self.targets.get(&image.address)
            && t.texture.textureType() == MTLTextureType::Type2D
            && t.texture.width() as u32 == image.width
            && t.texture.height() as u32 == image.height
        {
            if t.pixel_format == format {
                return Some(t.texture.clone());
            }
            if view_compatible(t.pixel_format, format) {
                return t.texture.newTextureViewWithPixelFormat(format);
            }
            // Memory reused for another surface: the old contents are dead.
            let have = t.pixel_format;
            self.note(|| {
                format!(
                    "      image at {:#x} ({have:?}) replaced by a {format:?} one",
                    image.address
                )
            });
            self.targets.remove(&image.address);
        }
        let chain = image.last_level + 1;
        let texture = self.new_texture_levels(format, image.width, image.height, true, chain)?;
        if let Some(mode) = image.tile
            && let Some(layout) = Layout::new(mode, bpe, image.width, image.height)
        {
            let mips = layout.mips(chain, (image.width, image.height), 1);
            for (level, mip) in mips
                .into_iter()
                .enumerate()
                .take(texture.mipmapLevelCount())
            {
                self.detile_level_into(host, &texture, 0, level, image.address, mip);
            }
        }
        self.targets.insert(
            image.address,
            Target {
                texture: texture.clone(),
                pixel_format: format,
                swizzle: 0,
            },
        );
        record(image.address | 2, &texture, true);
        Some(texture)
    }

    /// Writes what programs stored to images since the last call into
    /// guest memory, laid out as each descriptor says. Rendering never
    /// writes images back (guest memory keeps what the CPU wrote); frame
    /// extraction does, so a dispatch's stores can be compared with hardware.
    pub fn write_back_images(&mut self, host: &dyn Host) {
        self.sync();
        for (image, tex) in std::mem::take(&mut self.stored_images) {
            let Some((format, bpe)) = raw_format(image.format) else {
                continue;
            };
            let Some(layout) = image
                .tile
                .and_then(|t| Layout::new(t, bpe, image.width, image.height))
            else {
                continue;
            };
            let volume = tex.textureType() == MTLTextureType::Type3D;
            if is_compressed(format) || volume && image.tile != Some(TileMode::Linear) {
                continue;
            }
            let chain = image.last_level + 1;
            let mips = layout.mips(chain, (image.width, image.height), 1);
            let stride = layout
                .chain_size(chain)
                .next_multiple_of(layout.block_bytes());
            // A linear 3D image's depth slices are laid out as array slices.
            let (first, slices) = if volume {
                (0, image.depth as usize)
            } else {
                (image.base_array as u64, tex.arrayLength())
            };
            for slice in 0..slices {
                let at = image.address + (first + slice as u64) * stride;
                for (level, mip) in mips.iter().enumerate().take(tex.mipmapLevelCount()) {
                    if volume && slice >= (image.depth >> level).max(1) as usize {
                        continue;
                    }
                    let Some(bytes) = read_level(&tex, slice, level) else {
                        continue;
                    };
                    let l = mip.layout;
                    let row = l.width as usize * bpe as usize;
                    for y in 0..l.height {
                        let src = &bytes[y as usize * row..][..row];
                        for x in 0..l.width {
                            let o = at + mip.offset + l.offset(x + mip.origin.0, y + mip.origin.1);
                            let b = bpe as usize;
                            host.write(o, &src[x as usize * b..][..b]);
                        }
                    }
                }
            }
        }
    }

    /// Fills the slices of a 2D-array `texture` from the guest memory behind
    /// `image`, returning what [`Metal::refresh_volume`] needs to notice later CPU writes there.
    fn fill_volume(
        &mut self,
        host: &dyn Host,
        texture: &Texture,
        image: Image,
        bpe: u32,
        slices: u32,
    ) -> Option<VolumeSource> {
        let layout = Layout::new(image.tile?, bpe, image.width, image.height)?;
        let base = image.address + layout.mip0_offset(image.last_level + 1);
        let stride = layout.size().next_multiple_of(layout.block_bytes());
        for slice in 0..slices {
            self.detile_into(
                host,
                texture,
                slice as usize,
                base + slice as u64 * stride,
                layout,
            );
        }
        let len = stride * slices as u64;
        let shadow = guest_bytes(host, base, len).map(|b| Arc::new(b.to_vec()));
        Some(VolumeSource {
            image,
            bpe,
            slices,
            base,
            len,
            shadow,
            epoch: self.epoch,
        })
    }

    /// Copies into the 2D-array image at `address` the texels the CPU changed
    /// in guest memory since the last look (on every write, once a frame for
    /// reads). Texels the GPU wrote stay: those writes never reach guest memory.
    fn refresh_volume(&mut self, host: &dyn Host, address: u64, write: bool) {
        let Some(src) = self.volume_sources.get_mut(&address) else {
            return;
        };
        if !write && src.epoch == self.epoch {
            return;
        }
        src.epoch = self.epoch;
        let src = src.clone();
        let Some(old) = src.shadow.as_deref() else {
            return;
        };
        let Some(new) = guest_bytes(host, src.base, src.len) else {
            return;
        };
        if new == old.as_slice() {
            return;
        }
        let new = new.to_vec();
        let Some(texture) = self.volumes.get(&address).cloned() else {
            return;
        };
        let Some(layout) = src
            .image
            .tile
            .and_then(|t| Layout::new(t, src.bpe, src.image.width, src.image.height))
        else {
            return;
        };
        if texture.arrayLength() < src.slices as usize {
            return;
        }
        let changed = new.iter().zip(old).filter(|(a, b)| a != b).count();
        self.report(
            tracing::Level::INFO,
            format!(
                "2D array at {address:#x}: {changed} bytes changed in guest memory, copying them in"
            ),
        );
        let stride = (src.len / src.slices as u64) as usize;
        let size = layout.size() as usize;
        for slice in 0..src.slices as usize {
            let range = slice * stride..slice * stride + size;
            if new[range.clone()] != old[range.clone()] {
                self.detile_changes(
                    &texture,
                    slice,
                    layout,
                    &new[range.clone()],
                    &old[range],
                    src.base + (slice * stride) as u64,
                );
            }
        }
        if let Some(s) = self.volume_sources.get_mut(&address) {
            s.shadow = Some(Arc::new(new));
        }
    }

    /// The 2D-array or 3D image programs wrote at `image`'s address, viewed as
    /// `kind` from slice `BASE_ARRAY` on, if it has that shape.
    fn volume(
        &self,
        image: Image,
        kind: TextureKind,
        format: MTLPixelFormat,
    ) -> Option<Retained<Texture>> {
        let v = self.volumes.get(&image.address)?;
        if v.width() as u32 != image.width || v.height() as u32 != image.height {
            return None;
        }
        let (base, have) = (image.base_array as usize, v.arrayLength());
        let (ty, first, count) = match (kind, v.textureType()) {
            (TextureKind::Tex3d, MTLTextureType::Type3D) if v.depth() as u32 == image.depth => {
                (MTLTextureType::Type3D, 0, 1)
            }
            (TextureKind::Tex2d, MTLTextureType::Type2DArray) if base < have => {
                (MTLTextureType::Type2D, base, 1)
            }
            (TextureKind::Tex2dArray, MTLTextureType::Type2DArray)
                if (image.depth as usize) <= have && base < image.depth as usize =>
            {
                (
                    MTLTextureType::Type2DArray,
                    base,
                    image.depth as usize - base,
                )
            }
            (TextureKind::Cube, MTLTextureType::Type2DArray) if base + 6 <= have => {
                (MTLTextureType::TypeCube, base, 6)
            }
            _ => return None,
        };
        unsafe {
            v.newTextureViewWithPixelFormat_textureType_levels_slices(
                format,
                ty,
                objc2_foundation::NSRange::new(0, v.mipmapLevelCount()),
                objc2_foundation::NSRange::new(first, count),
            )
        }
    }

    /// A zero-filled 2D-array (`slices` slices) or 3D image for a program to write.
    fn new_volume(
        &self,
        format: MTLPixelFormat,
        image: Image,
        kind: TextureKind,
        slices: u32,
    ) -> Option<Retained<Texture>> {
        let desc = MTLTextureDescriptor::new();
        desc.setPixelFormat(format);
        let (w, h, d) = (
            image.width as usize,
            image.height as usize,
            slices.max(1) as usize,
        );
        unsafe {
            desc.setWidth(w);
            desc.setHeight(h);
            if kind == TextureKind::Tex3d {
                desc.setTextureType(MTLTextureType::Type3D);
                desc.setDepth(d);
            } else {
                desc.setTextureType(MTLTextureType::Type2DArray);
                desc.setArrayLength(d);
            }
        }
        let mut usage = MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView;
        if atomic_capable(format) {
            usage |= MTLTextureUsage::ShaderAtomic;
        }
        desc.setUsage(usage);
        desc.setStorageMode(MTLStorageMode::Shared);
        if kind == TextureKind::Tex3d {
            unsafe { desc.setMipmapLevelCount(volume_levels(image) as usize) };
        }
        let t = self.device.newTextureWithDescriptor(&desc)?;
        let row = w * texel_bytes(format)?;
        let zero = vec![0u8; row * h * if kind == TextureKind::Tex3d { d } else { 1 }];
        let (depth, slices) = if kind == TextureKind::Tex3d {
            (d, 1)
        } else {
            (1, d)
        };
        for slice in 0..slices {
            unsafe {
                t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize {
                            width: w,
                            height: h,
                            depth,
                        },
                    },
                    0,
                    slice,
                    NonNull::new(zero.as_ptr() as *mut u8)?.cast(),
                    row,
                    row * h,
                );
            }
        }
        Some(t)
    }

    /// A zero-filled 2D, 2D-array or cube texture shaped like `image`,
    /// cached until a depth target is made at its address.
    fn blank_texture(
        &mut self,
        image: Image,
        kind: TextureKind,
        format: MTLPixelFormat,
    ) -> Option<Retained<Texture>> {
        if let Some(t) = self.blank.get(&(image, kind)) {
            return Some(t.clone());
        }
        let slices = match kind {
            TextureKind::Tex2d => 1,
            TextureKind::Tex2dArray => image.depth.max(1) as usize,
            TextureKind::Cube => 6,
            TextureKind::Tex3d => return None,
        };
        let desc = MTLTextureDescriptor::new();
        desc.setPixelFormat(format);
        unsafe {
            desc.setWidth(image.width as usize);
            desc.setHeight(image.height as usize);
            match kind {
                TextureKind::Cube => desc.setTextureType(MTLTextureType::TypeCube),
                TextureKind::Tex2dArray => {
                    desc.setTextureType(MTLTextureType::Type2DArray);
                    desc.setArrayLength(slices);
                }
                _ => desc.setTextureType(MTLTextureType::Type2D),
            }
        }
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(MTLStorageMode::Shared);
        let t = self.device.newTextureWithDescriptor(&desc)?;
        let row = image.width as usize * texel_bytes(format)?;
        let zero = vec![0u8; row * image.height as usize];
        for slice in 0..slices {
            unsafe {
                t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize {
                            width: image.width as usize,
                            height: image.height as usize,
                            depth: 1,
                        },
                    },
                    0,
                    slice,
                    NonNull::new(zero.as_ptr() as *mut u8)?.cast(),
                    row,
                    zero.len(),
                );
            }
        }
        self.blank.insert((image, kind), t.clone());
        Some(t)
    }
}
