//! Copying tiled guest surfaces into textures on the GPU.

use super::*;

/// What a detile kernel writes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Detile {
    Texture,
    Buffer,
    /// Only the texels whose bytes differ from an older copy of the surface.
    Changed,
}

impl Metal {
    fn detiler(
        &mut self,
        mode: TileMode,
        bpe: u32,
        output: Detile,
    ) -> Option<Retained<ComputePipeline>> {
        if let Some(p) = self.detilers.get(&(mode, bpe, output)) {
            return Some(p.clone());
        }
        let layout = Layout::new(mode, bpe, 1, 1)?;
        let (ty, load) = match bpe {
            1 => ("uint", "uint(src[o])"),
            2 => ("uint", "uint(*(device const ushort*)(src + o))"),
            4 => ("uint", "*(device const uint*)(src + o)"),
            8 => ("uint2", "*(device const uint2*)(src + o)"),
            _ => ("uint4", "*(device const uint4*)(src + o)"),
        };
        let width = match ty {
            "uint" => "uint4(v, 0, 0, 0)",
            "uint2" => "uint4(v, 0, 0)",
            _ => "v",
        };
        let same = if ty == "uint" {
            "v == w"
        } else {
            "all(v == w)"
        };
        let (args, store) = match output {
            Detile::Buffer => (format!("device {ty}* dst [[buffer(2)]]"), "dst[id.y * p.x + id.x] = v;".to_string()),
            Detile::Texture => ("texture2d<uint, access::write> dst [[texture(0)]]".to_string(), format!("dst.write({width}, id);")),
            Detile::Changed => (
                "device const uchar* old [[buffer(2)]], texture2d<uint, access::write> dst [[texture(0)]]".to_string(),
                format!("{ty} w = {};\n    if ({same}) return;\n    dst.write({width}, id);", load.replace("src", "old")),
            ),
        };
        let msl = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void detile(device const uchar* src [[buffer(0)]], constant uint4& p [[buffer(1)]], {args}, uint2 id [[thread_position_in_grid]]) {{\n    if (id.x >= p.x || id.y >= p.y) return;\n    ulong o = surface_offset(id.x + (p.w & 0xffffu), id.y + (p.w >> 16), p.z);\n    {ty} v = {load};\n    {store}\n}}\n",
            layout.msl_offset()
        );
        let function = self.compile(&msl, "detile")?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .ok()?;
        self.detilers.insert((mode, bpe, output), pipeline.clone());
        Some(pipeline)
    }

    /// Copies a tiled guest surface into slice `slice` of `texture` on the GPU.
    pub(super) fn detile_into(
        &mut self,
        host: &dyn Host,
        texture: &Texture,
        slice: usize,
        address: u64,
        layout: Layout,
    ) -> bool {
        let mip = Mip {
            offset: 0,
            layout,
            origin: (0, 0),
        };
        self.detile_level_into(host, texture, slice, 0, address, mip)
    }

    /// As [`Self::detile_into`], for mip level `level` placed at `mip` from `address`.
    pub(super) fn detile_level_into(
        &mut self,
        host: &dyn Host,
        texture: &Texture,
        slice: usize,
        level: usize,
        address: u64,
        mip: Mip,
    ) -> bool {
        let at = address + mip.offset;
        let Some((buffer, offset)) = self.window(host, at, mip.layout.size()) else {
            return false;
        };
        self.encode_detile(texture, (slice, level), mip, (&buffer, offset), None, at)
    }

    /// Copies into slice `slice` of `texture` only the texels whose bytes in
    /// the tiled surface `new` differ from those in `old`.
    pub(super) fn detile_changes(
        &mut self,
        texture: &Texture,
        slice: usize,
        layout: Layout,
        new: &[u8],
        old: &[u8],
        address: u64,
    ) -> bool {
        let bytes = |b: &[u8]| {
            let ptr = NonNull::new(b.as_ptr() as *mut u8)?.cast();
            unsafe {
                self.device.newBufferWithBytes_length_options(
                    ptr,
                    b.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
        };
        let (Some(new), Some(old)) = (bytes(new), bytes(old)) else {
            return false;
        };
        let mip = Mip {
            offset: 0,
            layout,
            origin: (0, 0),
        };
        self.encode_detile(texture, (slice, 0), mip, (&new, 0), Some(&old), address)
    }

    fn encode_detile(
        &mut self,
        texture: &Texture,
        (slice, level): (usize, usize),
        mip: Mip,
        src: (&Buffer, usize),
        old: Option<&Buffer>,
        address: u64,
    ) -> bool {
        let layout = mip.layout;
        let output = if old.is_some() {
            Detile::Changed
        } else {
            Detile::Texture
        };
        let Some(pipeline) = self.detiler(layout.mode, layout.bpe, output) else {
            return false;
        };
        let raw_format = match layout.bpe {
            1 => MTLPixelFormat::R8Uint,
            2 => MTLPixelFormat::R16Uint,
            4 => MTLPixelFormat::R32Uint,
            8 => MTLPixelFormat::RG32Uint,
            _ => MTLPixelFormat::RGBA32Uint,
        };
        let raw = unsafe {
            texture.newTextureViewWithPixelFormat_textureType_levels_slices(
                raw_format,
                MTLTextureType::Type2D,
                objc2_foundation::NSRange::new(level, 1),
                objc2_foundation::NSRange::new(slice, 1),
            )
        };
        let Some(raw) = raw else { return false };
        let command = self.upload_command();
        let Some(enc) = command.computeCommandEncoder() else {
            return false;
        };
        if crate::timing::labels() {
            enc.setLabel(Some(&NSString::from_str(&format!(
                "detile {address:#x} {}x{}",
                layout.width, layout.height
            ))));
        }
        let params = [
            layout.width,
            layout.height,
            layout.pitch,
            mip.origin.0 | mip.origin.1 << 16,
        ];
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src.0), src.1, 0);
            enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 16, 1);
            if let Some(old) = old {
                enc.setBuffer_offset_atIndex(Some(old), 0, 2);
            }
            enc.setTexture_atIndex(Some(&raw), 0);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: layout.width as usize,
                height: layout.height as usize,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            },
        );
        enc.endEncoding();
        true
    }

    /// A 2D, 2D-array or cube texture with `levels` mip levels, to fill from guest memory.
    pub(super) fn new_layered_texture(
        &self,
        format: MTLPixelFormat,
        (width, height): (u32, u32),
        kind: TextureKind,
        slices: usize,
        levels: u32,
    ) -> Option<Retained<Texture>> {
        let desc = MTLTextureDescriptor::new();
        desc.setPixelFormat(format);
        unsafe {
            desc.setWidth(width as usize);
            desc.setHeight(height as usize);
            desc.setMipmapLevelCount(levels as usize);
            match kind {
                TextureKind::Cube => desc.setTextureType(MTLTextureType::TypeCube),
                TextureKind::Tex2d => desc.setTextureType(MTLTextureType::Type2D),
                _ => {
                    desc.setTextureType(MTLTextureType::Type2DArray);
                    desc.setArrayLength(slices.max(1));
                }
            }
        }
        let usage = if is_compressed(format) {
            MTLTextureUsage::ShaderRead
        } else {
            MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView
        };
        desc.setUsage(usage);
        desc.setStorageMode(MTLStorageMode::Private);
        self.device.newTextureWithDescriptor(&desc)
    }

    pub(super) fn new_texture(
        &self,
        format: MTLPixelFormat,
        width: u32,
        height: u32,
        target: bool,
    ) -> Option<Retained<Texture>> {
        self.new_texture_levels(format, width, height, target, 1)
    }

    pub(super) fn new_texture_levels(
        &self,
        format: MTLPixelFormat,
        width: u32,
        height: u32,
        target: bool,
        levels: u32,
    ) -> Option<Retained<Texture>> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                width as usize,
                height as usize,
                false,
            )
        };
        let most = 32 - width.max(height).max(1).leading_zeros();
        // SAFETY: at most the full chain for this size.
        unsafe { desc.setMipmapLevelCount(levels.clamp(1, most) as usize) };
        let mut usage = MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView;
        if target {
            usage |= MTLTextureUsage::RenderTarget;
        }
        if atomic_capable(format) {
            usage |= MTLTextureUsage::ShaderAtomic;
        }
        if is_compressed(format) {
            usage = MTLTextureUsage::ShaderRead;
        }
        desc.setUsage(usage);
        desc.setStorageMode(MTLStorageMode::Private);
        self.device.newTextureWithDescriptor(&desc)
    }

    /// A 2D-array render target of `slices` slices (cube maps the GPU renders face by face).
    pub(super) fn new_target_array(
        &self,
        format: MTLPixelFormat,
        width: u32,
        height: u32,
        slices: u32,
    ) -> Option<Retained<Texture>> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                width as usize,
                height as usize,
                false,
            )
        };
        desc.setTextureType(MTLTextureType::Type2DArray);
        // SAFETY: at most MAX_TARGET_SLICES, below Metal's array limit.
        unsafe { desc.setArrayLength(slices as usize) };
        desc.setUsage(
            MTLTextureUsage::ShaderRead
                | MTLTextureUsage::ShaderWrite
                | MTLTextureUsage::PixelFormatView
                | MTLTextureUsage::RenderTarget,
        );
        desc.setStorageMode(MTLStorageMode::Private);
        self.device.newTextureWithDescriptor(&desc)
    }

    /// Fills block-compressed `texture` from a tiled guest surface of 4x4
    /// blocks: detiled into a linear buffer, then blitted.
    pub(super) fn detile_blocks_into(
        &mut self,
        host: &dyn Host,
        texture: &Texture,
        (slice, level): (usize, usize),
        address: u64,
        mip: Mip,
    ) -> bool {
        let layout = mip.layout;
        let Some(pipeline) = self.detiler(layout.mode, layout.bpe, Detile::Buffer) else {
            return false;
        };
        let at = address + mip.offset;
        let Some((buffer, offset)) = self.window(host, at, mip.layout.size()) else {
            return false;
        };
        let row = (layout.width * layout.bpe) as usize;
        let len = row * layout.height as usize;
        let Some(staging) = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate)
        else {
            return false;
        };
        let command = self.upload_command();
        let Some(enc) = command.computeCommandEncoder() else {
            return false;
        };
        let params = [
            layout.width,
            layout.height,
            layout.pitch,
            mip.origin.0 | mip.origin.1 << 16,
        ];
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buffer), offset, 0);
            enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 16, 1);
            enc.setBuffer_offset_atIndex(Some(&staging), 0, 2);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: layout.width as usize,
                height: layout.height as usize,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            },
        );
        enc.endEncoding();
        let Some(blit) = command.blitCommandEncoder() else {
            return false;
        };
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &staging,
                0,
                row,
                len,
                MTLSize {
                    width: (texture.width() >> level).max(1),
                    height: (texture.height() >> level).max(1),
                    depth: 1,
                },
                texture,
                slice,
                level,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
        true
    }
}
