//! Textures in save states.

use super::*;

/// A texture as a save state keeps it: which cache it sits in and at what
/// guest address, how it was made, and its texels slice by slice and level
/// by level, rows packed.
pub(super) struct SavedTexture {
    pub(super) cache: u8,
    pub(super) address: u64,
    pub(super) format: usize,
    pub(super) kind: usize,
    pub(super) size: (usize, usize, usize),
    pub(super) slices: usize,
    pub(super) levels: usize,
    usage: usize,
    storage: usize,
    pub(super) target_format: usize,
    texels: Vec<u8>,
}

savestate::record!(SavedTexture {
    cache,
    address,
    format,
    kind,
    size,
    slices,
    levels,
    usage,
    storage,
    target_format,
    texels
});

pub(super) const CACHE_TARGET: u8 = 0;

pub(super) const CACHE_DEPTH: u8 = 1;

pub(super) const CACHE_VOLUME: u8 = 2;

/// How a texture's texels are copied to and from a buffer: per aspect, the
/// bytes per texel and the blit option.
fn texel_aspects(f: MTLPixelFormat) -> Option<Vec<(usize, MTLBlitOption)>> {
    Some(match f {
        MTLPixelFormat::Depth32Float => vec![(4, MTLBlitOption::empty())],
        MTLPixelFormat::Depth16Unorm => vec![(2, MTLBlitOption::empty())],
        MTLPixelFormat::Depth32Float_Stencil8 => vec![
            (4, MTLBlitOption::DepthFromDepthStencil),
            (1, MTLBlitOption::StencilFromDepthStencil),
        ],
        MTLPixelFormat::Stencil8 => vec![(1, MTLBlitOption::empty())],
        f if is_compressed(f) => return None,
        f => vec![(texel_bytes(f)?, MTLBlitOption::empty())],
    })
}

/// Each copy one texture needs: slice, level, size, bytes per texel and
/// blit option, and where its texels start in the packed form.
#[allow(clippy::type_complexity)]
fn texture_copies(
    format: MTLPixelFormat,
    kind: MTLTextureType,
    (w, h, d): (usize, usize, usize),
    slices: usize,
    levels: usize,
) -> Option<(
    Vec<(usize, usize, MTLSize, usize, MTLBlitOption, usize)>,
    usize,
)> {
    let mut out = Vec::new();
    let mut at = 0;
    for (bpp, option) in texel_aspects(format)? {
        for slice in 0..slices {
            for level in 0..levels {
                let size = MTLSize {
                    width: (w >> level).max(1),
                    height: (h >> level).max(1),
                    depth: if kind == MTLTextureType::Type3D {
                        (d >> level).max(1)
                    } else {
                        1
                    },
                };
                out.push((slice, level, size, bpp, option, at));
                at += size.width * size.height * size.depth * bpp;
            }
        }
    }
    Some((out, at))
}

fn texture_slices(t: &Texture) -> usize {
    match t.textureType() {
        MTLTextureType::TypeCube => 6,
        MTLTextureType::TypeCubeArray => 6 * t.arrayLength(),
        MTLTextureType::Type2DArray => t.arrayLength(),
        _ => 1,
    }
}

impl Metal {
    /// Copies `t`'s texels into the packed form of [`SavedTexture`].
    fn read_texels(&self, t: &Texture) -> Result<Vec<u8>, String> {
        let (copies, len) = texture_copies(
            t.pixelFormat(),
            t.textureType(),
            (t.width(), t.height(), t.depth()),
            texture_slices(t),
            t.mipmapLevelCount(),
        )
        .ok_or_else(|| format!("texture format {:?} cannot be saved", t.pixelFormat()))?;
        let buffer = self
            .device
            .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or("no memory to read a texture back")?;
        let cb = self.queue.commandBuffer().ok_or("no command buffer")?;
        let blit = cb.blitCommandEncoder().ok_or("no blit encoder")?;
        for (slice, level, size, bpp, option, at) in copies {
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                    t, slice, level, MTLOrigin { x: 0, y: 0, z: 0 }, size, &buffer, at,
                    size.width * bpp, size.width * size.height * bpp, option,
                );
            }
        }
        blit.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(
            unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8, len) }
                .to_vec(),
        )
    }

    pub(super) fn make_saved(&self, s: &SavedTexture) -> Result<Retained<Texture>, String> {
        let format = MTLPixelFormat(s.format as _);
        let kind = MTLTextureType(s.kind as _);
        let desc = MTLTextureDescriptor::new();
        desc.setPixelFormat(format);
        unsafe {
            desc.setTextureType(kind);
            desc.setWidth(s.size.0);
            desc.setHeight(s.size.1);
            desc.setDepth(s.size.2);
            let arrays = match kind {
                MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray => s.slices / 6,
                _ => s.slices,
            };
            desc.setArrayLength(arrays.max(1));
            desc.setMipmapLevelCount(s.levels);
        }
        desc.setUsage(MTLTextureUsage(s.usage as _));
        desc.setStorageMode(MTLStorageMode(s.storage as _));
        let t = self
            .device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| format!("cannot make a {format:?} texture again"))?;
        let (copies, len) = texture_copies(format, kind, s.size, s.slices, s.levels)
            .ok_or("saved texture format has no copy layout")?;
        if len != s.texels.len() {
            return Err(format!(
                "saved texture at {:#x} has the wrong size",
                s.address
            ));
        }
        if len == 0 {
            return Ok(t);
        }
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(s.texels.as_ptr() as *mut _).ok_or("empty texels")?,
                len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or("no memory to upload a texture")?;
        let cb = self.queue.commandBuffer().ok_or("no command buffer")?;
        let blit = cb.blitCommandEncoder().ok_or("no blit encoder")?;
        for (slice, level, size, bpp, option, at) in copies {
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                    &buffer, at, size.width * bpp, size.width * size.height * bpp, size, &t,
                    slice, level, MTLOrigin { x: 0, y: 0, z: 0 }, option,
                );
            }
        }
        blit.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(t)
    }

    pub(super) fn saved_texture(
        &self,
        cache: u8,
        address: u64,
        t: &Texture,
        target_format: MTLPixelFormat,
    ) -> Result<SavedTexture, String> {
        Ok(SavedTexture {
            cache,
            address,
            format: t.pixelFormat().0,
            kind: t.textureType().0,
            size: (t.width(), t.height(), t.depth()),
            slices: texture_slices(t),
            levels: t.mipmapLevelCount(),
            usage: t.usage().0,
            storage: t.storageMode().0,
            target_format: target_format.0,
            texels: self.read_texels(t)?,
        })
    }
}
