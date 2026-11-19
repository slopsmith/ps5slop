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

use crate::regs::{self, Blend, ColorTarget, DepthState, DepthTarget, Image, Sampler, Viewport, context, sh, uconfig};
use gfx10::translate::{self as shader, Resource, Source, Stage, TextureKind, Translated, binding};
use crate::tiling::{Layout, TileMode};
use crate::{Backend, Draw, Host, State};

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

/// Render targets by guest address, for presenting flipped buffers.
static FRAMES: OnceLock<Mutex<HashMap<u64, SharedTexture>>> = OnceLock::new();

fn frames() -> &'static Mutex<HashMap<u64, SharedTexture>> {
    FRAMES.get_or_init(Default::default)
}

/// The texture the GPU rendered at guest address `address`, if any.
pub fn frame(address: u64) -> Option<SharedTexture> {
    frames().lock().ok()?.get(&address).cloned()
}

/// Every render target, by guest address.
pub fn all_frames() -> Vec<(u64, SharedTexture)> {
    frames().lock().map(|f| f.iter().map(|(a, t)| (*a, t.clone())).collect()).unwrap_or_default()
}

/// `texture`'s pixels as RGBA8, read back for screenshots.
pub fn read_rgba(texture: &SharedTexture) -> Option<(u32, u32, Vec<u8>)> {
    let t = &texture.0;
    let (w, h) = (t.width(), t.height());
    let format = t.pixelFormat();
    let bgra = matches!(format, MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::BGRA8Unorm_sRGB);
    if !bgra && !matches!(format, MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::RGBA8Unorm_sRGB) {
        return None;
    }
    let device = t.device();
    let queue = device.newCommandQueue()?;
    let buffer = device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let blit = cb.blitCommandEncoder()?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            t,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: w, height: h, depth: 1 },
            &buffer,
            0,
            w * 4,
            w * h * 4,
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let mut px = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8, w * h * 4) }.to_vec();
    if bgra {
        for p in px.as_chunks_mut::<4>().0 {
            p.swap(0, 2);
        }
    }
    Some((w as u32, h as u32, px))
}

/// Guest memory is wrapped in windows of about this size.
const WINDOW: u64 = 64 << 20;
const PAGE: u64 = 0x4000;
/// Draws whose shaders failed are reported this many times.
const REPORTS: u32 = 24;

struct Window {
    start: u64,
    end: u64,
    buffer: Retained<Buffer>,
}

struct Target {
    texture: Retained<Texture>,
    pixel_format: MTLPixelFormat,
}

struct Bound {
    memory: Option<(Retained<Buffer>, u64)>,
    textures: Vec<(usize, Retained<Texture>)>,
    samplers: Vec<(usize, Retained<ProtocolObject<dyn MTLSamplerState>>)>,
}

struct Program {
    address: u64,
    function: Retained<Function>,
    translated: Translated,
    /// The rectangle-list form of a vertex program, made on first use.
    rect: Mutex<Option<Option<Retained<Function>>>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    vertex: u64,
    pixel: u64,
    formats: [usize; 8],
    blends: [(Blend, u8); 8],
    depth: usize,
    stencil: bool,
    rect: bool,
}

/// The Metal side of the GPU.
pub struct Metal {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    command: Option<Retained<CommandBuffer>>,
    pass: Option<(Retained<RenderEncoder>, Vec<u64>)>,
    depths: HashMap<u64, Retained<Texture>>,
    shown: HashMap<u64, Retained<Texture>>,
    depth_states: HashMap<(DepthState, bool), Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    windows: Vec<Window>,
    generation: u64,
    targets: HashMap<u64, Target>,
    images: HashMap<Image, (u64, Retained<Texture>)>,
    epoch: u64,
    detilers: HashMap<(TileMode, u32, bool), Retained<ComputePipeline>>,
    programs: HashMap<(u64, u64), Option<Arc<Program>>>,
    computes: HashMap<u64, Option<Retained<ComputePipeline>>>,
    pipelines: HashMap<PipelineKey, Option<Retained<RenderPipeline>>>,
    samplers: HashMap<Sampler, Retained<ProtocolObject<dyn MTLSamplerState>>>,
    reports: u32,
    null: Option<Retained<Texture>>,
    pub draws: u64,
    pub dispatches: u64,
    pub skipped: u64,
}

/// Where a program's resources are bound: one stage of an encoder.
#[derive(Clone, Copy)]
enum Slot<'a> {
    Vertex(&'a RenderEncoder),
    Fragment(&'a RenderEncoder),
    Compute(&'a ProtocolObject<dyn MTLComputeCommandEncoder>),
}

impl Slot<'_> {
    fn bytes(self, data: *const std::ffi::c_void, len: usize, index: usize) {
        let data = NonNull::new(data as *mut std::ffi::c_void).expect("bound data is not null");
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexBytes_length_atIndex(data, len, index),
                Slot::Fragment(e) => e.setFragmentBytes_length_atIndex(data, len, index),
                Slot::Compute(e) => e.setBytes_length_atIndex(data, len, index),
            }
        }
    }

    fn texture(self, t: &Texture, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexTexture_atIndex(Some(t), index),
                Slot::Fragment(e) => e.setFragmentTexture_atIndex(Some(t), index),
                Slot::Compute(e) => e.setTexture_atIndex(Some(t), index),
            }
        }
    }

    fn sampler(self, s: &ProtocolObject<dyn MTLSamplerState>, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexSamplerState_atIndex(Some(s), index),
                Slot::Fragment(e) => e.setFragmentSamplerState_atIndex(Some(s), index),
                Slot::Compute(e) => e.setSamplerState_atIndex(Some(s), index),
            }
        }
    }

    fn buffer(self, b: &Buffer, offset: usize, index: usize) {
        unsafe {
            match self {
                Slot::Vertex(e) => e.setVertexBuffer_offset_atIndex(Some(b), offset, index),
                Slot::Fragment(e) => e.setFragmentBuffer_offset_atIndex(Some(b), offset, index),
                Slot::Compute(e) => e.setBuffer_offset_atIndex(Some(b), offset, index),
            }
        }
    }
}

// SAFETY: a `Metal` is used by one thread at a time; Metal objects may move
// between threads, and an open encoder is never used concurrently.
unsafe impl Send for Metal {}

impl Metal {
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;
        Some(Self {
            device,
            queue,
            command: None,
            pass: None,
            depths: HashMap::new(),
            shown: HashMap::new(),
            depth_states: HashMap::new(),
            windows: Vec::new(),
            generation: 0,
            targets: HashMap::new(),
            images: HashMap::new(),
            epoch: 0,
            detilers: HashMap::new(),
            programs: HashMap::new(),
            computes: HashMap::new(),
            pipelines: HashMap::new(),
            samplers: HashMap::new(),
            reports: 0,
            null: None,
            draws: 0,
            dispatches: 0,
            skipped: 0,
        })
    }

    fn report(&mut self, what: String) {
        if self.reports < REPORTS {
            self.reports += 1;
            eprintln!("ps5slop: GPU: {what}");
        }
    }

    fn command(&mut self) -> Retained<CommandBuffer> {
        if self.command.is_none() {
            self.command = self.queue.commandBuffer();
        }
        self.command.clone().expect("Metal hands out command buffers")
    }

    fn end_pass(&mut self) {
        if let Some((enc, _)) = self.pass.take() {
            enc.endEncoding();
        }
    }

    /// The no-copy buffer holding `[addr, addr + len)` and the offset of `addr` in it.
    fn window(&mut self, host: &dyn Host, addr: u64, len: u64) -> Option<(Retained<Buffer>, usize)> {
        let generation = host.generation();
        if generation != self.generation {
            self.flush_and_wait();
            self.windows.clear();
            self.generation = generation;
        }
        let end = addr.checked_add(len.max(1))?;
        if let Some(w) = self.windows.iter().find(|w| w.start <= addr && end <= w.end) {
            return Some((w.buffer.clone(), (addr - w.start) as usize));
        }
        let (lo, hi) = host.extent(addr)?;
        if end > hi {
            return None;
        }
        let start = (addr & !(WINDOW - 1)).max(lo) & !(PAGE - 1);
        let stop = end.next_multiple_of(WINDOW).min(hi).next_multiple_of(PAGE).min(hi);
        let ptr = NonNull::new(host.host_pointer(start)?)?;
        // SAFETY: the range is host memory that stays mapped while the address
        // space lives; windows are dropped when its mappings change.
        let buffer = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr.cast(),
                (stop - start) as usize,
                MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
                None,
            )
        }?;
        self.windows.push(Window {
            start,
            end: stop,
            buffer: buffer.clone(),
        });
        Some((buffer, (addr - start) as usize))
    }

    fn detiler(&mut self, mode: TileMode, bpe: u32, to_buffer: bool) -> Option<Retained<ComputePipeline>> {
        if let Some(p) = self.detilers.get(&(mode, bpe, to_buffer)) {
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
        let (output, store) = if to_buffer {
            (
                format!("device {ty}* dst [[buffer(2)]]"),
                "dst[id.y * p.x + id.x] = v;".to_string(),
            )
        } else {
            (
                "texture2d<uint, access::write> dst [[texture(0)]]".to_string(),
                format!("dst.write({width}, id);"),
            )
        };
        let msl = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void detile(device const uchar* src [[buffer(0)]], constant uint4& p [[buffer(1)]], {output}, uint2 id [[thread_position_in_grid]]) {{\n    if (id.x >= p.x || id.y >= p.y) return;\n    ulong o = surface_offset(id.x, id.y, p.z);\n    {ty} v = {load};\n    {store}\n}}\n",
            layout.msl_offset()
        );
        let function = self.compile(&msl, "detile")?;
        let pipeline = self.device.newComputePipelineStateWithFunction_error(&function).ok()?;
        self.detilers.insert((mode, bpe, to_buffer), pipeline.clone());
        Some(pipeline)
    }

    fn compile(&mut self, msl: &str, entry: &str) -> Option<Retained<Function>> {
        match self.device.newLibraryWithSource_options_error(&NSString::from_str(msl), None) {
            Ok(lib) => lib.newFunctionWithName(&NSString::from_str(entry)),
            Err(e) => {
                self.report(format!("MSL did not compile: {}", e.localizedDescription()));
                None
            }
        }
    }

    /// Copies a tiled guest surface into `texture` on the GPU.
    fn detile_into(&mut self, host: &dyn Host, texture: &Texture, address: u64, layout: Layout) -> bool {
        let Some(pipeline) = self.detiler(layout.mode, layout.bpe, false) else { return false };
        let Some((buffer, offset)) = self.window(host, address, layout.size()) else { return false };
        let raw_format = match layout.bpe {
            1 => MTLPixelFormat::R8Uint,
            2 => MTLPixelFormat::R16Uint,
            4 => MTLPixelFormat::R32Uint,
            8 => MTLPixelFormat::RG32Uint,
            _ => MTLPixelFormat::RGBA32Uint,
        };
        let Some(raw) = texture.newTextureViewWithPixelFormat(raw_format) else { return false };
        self.end_pass();
        let command = self.command();
        let Some(enc) = command.computeCommandEncoder() else { return false };
        let params = [layout.width, layout.height, layout.pitch, 0];
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buffer), offset, 0);
            enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 16, 1);
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

    fn new_texture(&self, format: MTLPixelFormat, width: u32, height: u32, target: bool) -> Option<Retained<Texture>> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                width as usize,
                height as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite | MTLTextureUsage::PixelFormatView;
        if target {
            usage |= MTLTextureUsage::RenderTarget;
        }
        if is_compressed(format) {
            usage = MTLTextureUsage::ShaderRead;
        }
        desc.setUsage(usage);
        desc.setStorageMode(MTLStorageMode::Private);
        self.device.newTextureWithDescriptor(&desc)
    }

    /// Fills block-compressed `texture` from a tiled guest surface of 4x4
    /// blocks: detiled into a linear buffer, then blitted.
    fn detile_blocks_into(&mut self, host: &dyn Host, texture: &Texture, address: u64, layout: Layout) -> bool {
        let Some(pipeline) = self.detiler(layout.mode, layout.bpe, true) else { return false };
        let Some((buffer, offset)) = self.window(host, address, layout.size()) else { return false };
        let row = (layout.width * layout.bpe) as usize;
        let len = row * layout.height as usize;
        let Some(staging) = self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate) else {
            return false;
        };
        self.end_pass();
        let command = self.command();
        let Some(enc) = command.computeCommandEncoder() else { return false };
        let params = [layout.width, layout.height, layout.pitch, 0];
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buffer), offset, 0);
            enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 16, 1);
            enc.setBuffer_offset_atIndex(Some(&staging), 0, 2);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: layout.width as usize, height: layout.height as usize, depth: 1 },
            MTLSize { width: 16, height: 16, depth: 1 },
        );
        enc.endEncoding();
        let Some(blit) = command.blitCommandEncoder() else { return false };
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &staging,
                0,
                row,
                len,
                MTLSize { width: texture.width(), height: texture.height(), depth: 1 },
                texture,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
        true
    }

    /// The texture backing color target `t`, made (and filled from guest memory) on first use.
    fn target(&mut self, host: &dyn Host, t: &ColorTarget) -> Option<(Retained<Texture>, MTLPixelFormat)> {
        let format = color_format(t.format)?;
        if let Some(existing) = self.targets.get(&t.address)
            && existing.pixel_format == format
            && existing.texture.width() == t.width as usize
            && existing.texture.height() == t.height as usize
        {
            return Some((existing.texture.clone(), format));
        }
        let texture = self.new_texture(format, t.width, t.height, true)?;
        if let (Some(mode), Some(bpe)) = (t.tile, t.format.bytes())
            && let Some(layout) = Layout::new(mode, bpe, t.width, t.height)
        {
            self.detile_into(host, &texture, t.address, layout);
        }
        self.targets.insert(
            t.address,
            Target {
                texture: texture.clone(),
                pixel_format: format,
            },
        );
        if std::env::var_os("PS5SLOP_DUMP_TARGETS").is_some()
            && let Ok(mut f) = frames().lock()
        {
            f.insert(t.address | 2, SharedTexture(texture.clone()));
        }
        Some((texture, format))
    }

    /// The texture for image descriptor `image`: a render target at that
    /// address, or a copy of guest memory refreshed once per frame.
    fn image(&mut self, host: &dyn Host, image: Image) -> Option<Retained<Texture>> {
        let (format, bpe) = image_format(image.format)?;
        if let Some(t) = self.targets.get(&image.address) {
            let tex = &t.texture;
            if tex.width() as u32 == image.width && tex.height() as u32 == image.height {
                return if t.pixel_format == format {
                    Some(tex.clone())
                } else {
                    tex.newTextureViewWithPixelFormat(format)
                };
            }
        }
        if let Some((epoch, tex)) = self.images.get(&image)
            && *epoch == self.epoch
        {
            return Some(tex.clone());
        }
        let compressed = is_compressed(format);
        let (ew, eh) = if compressed {
            (image.width.div_ceil(4), image.height.div_ceil(4))
        } else {
            (image.width, image.height)
        };
        let layout = Layout::new(image.tile?, bpe, ew, eh)?;
        let tex = match self.images.get(&image) {
            Some((_, t)) => t.clone(),
            None => self.new_texture(format, image.width, image.height, false)?,
        };
        let base = image.address + layout.mip0_offset(image.last_level + 1);
        let filled = if compressed {
            self.detile_blocks_into(host, &tex, base, layout)
        } else {
            self.detile_into(host, &tex, base, layout)
        };
        if !filled {
            return None;
        }
        self.images.insert(image, (self.epoch, tex.clone()));
        if std::env::var_os("PS5SLOP_DUMP_TARGETS").is_some()
            && let Ok(mut f) = frames().lock()
        {
            f.entry(image.address | 1).or_insert_with(|| SharedTexture(tex.clone()));
        }
        Some(tex)
    }

    /// A 1x1 transparent black texture, for unbound image descriptors.
    fn null_texture(&mut self) -> Option<Retained<Texture>> {
        if self.null.is_none() {
            let d = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA8Unorm, 1, 1, false)
            };
            d.setStorageMode(MTLStorageMode::Shared);
            let t = self.device.newTextureWithDescriptor(&d)?;
            let zero = [0u8; 4];
            unsafe {
                t.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: 1, height: 1, depth: 1 } },
                    0,
                    NonNull::from(&zero).cast(),
                    4,
                );
            }
            self.null = Some(t);
        }
        self.null.clone()
    }

    fn sampler(&mut self, s: Sampler) -> Option<Retained<ProtocolObject<dyn MTLSamplerState>>> {
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

    /// The translated program at `address`, translated on first use.
    fn program(&mut self, host: &dyn Host, state: &State, stage: Stage, address: u64, user_sgprs: u32, ngg: bool) -> Option<Arc<Program>> {
        let c = state.context;
        let cntl = &c[context::SPI_PS_INPUT_CNTL_0..context::SPI_PS_INPUT_CNTL_0 + 32];
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (stage as u8, user_sgprs, ngg).hash(&mut h);
            if stage == Stage::Pixel {
                (cntl, c[context::SPI_PS_INPUT_ENA], c[context::SPI_PS_INPUT_ADDR], c[context::SPI_SHADER_COL_FORMAT]).hash(&mut h);
            }
            h.finish()
        };
        if let Some(p) = self.programs.get(&(address, key)) {
            return p.clone();
        }
        let (_, hi) = host.extent(address)?;
        let len = (hi - address).min(64 << 10) as usize & !3;
        let mut bytes = vec![0u8; len];
        let program = if host.read(address, &mut bytes) {
            let code: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            let request = shader::Request {
                stage,
                code: &code,
                user_sgprs,
                ngg,
                ps_input_ena: c[context::SPI_PS_INPUT_ENA],
                ps_input_addr: c[context::SPI_PS_INPUT_ADDR],
                ps_input_cntl: cntl,
                col_format: c[context::SPI_SHADER_COL_FORMAT],
            };
            match shader::translate(&request) {
                Ok(translated) => {
                    if std::env::var_os("PS5SLOP_MSL_DUMP").is_some() {
                        eprintln!("=== {stage:?} {address:#x}\n{}", translated.msl);
                    }
                    self.compile(&translated.msl, &translated.entry).map(|function| {
                        Arc::new(Program {
                            address,
                            function,
                            translated,
                            rect: Mutex::new(None),
                        })
                    })
                }
                Err(e) => {
                    self.report(format!("{stage:?} program at {address:#x} not translated: {e}"));
                    None
                }
            }
        } else {
            None
        };
        self.programs.insert((address, key), program.clone());
        program
    }

    /// The vertex function drawing `program`'s rectangles: six vertices per
    /// three guest ones, the fourth corner being v1 + v2 - v0.
    fn rect_function(&mut self, program: &Program) -> Option<Retained<Function>> {
        if let Some(f) = program.rect.lock().ok()?.as_ref() {
            return f.clone();
        }
        let f = rect_source(&program.translated.msl).and_then(|msl| self.compile(&msl, "gen5_vs_rect"));
        if let Ok(mut r) = program.rect.lock() {
            *r = Some(f.clone());
        }
        f
    }

    fn pipeline(&mut self, key: &PipelineKey, vertex: &Function, pixel: &Program) -> Option<Retained<RenderPipeline>> {
        if let Some(p) = self.pipelines.get(key) {
            return p.clone();
        }
        let desc = MTLRenderPipelineDescriptor::new();
        desc.setVertexFunction(Some(vertex));
        desc.setFragmentFunction(Some(&pixel.function));
        for (i, &raw) in key.formats.iter().enumerate() {
            if raw == 0 {
                continue;
            }
            let att = unsafe { desc.colorAttachments().objectAtIndexedSubscript(i) };
            att.setPixelFormat(MTLPixelFormat(raw));
            let (b, mask) = key.blends[i];
            att.setWriteMask(MTLColorWriteMask(mask as usize));
            if b.enable {
                att.setBlendingEnabled(true);
                att.setSourceRGBBlendFactor(blend_factor(b.color_src));
                att.setDestinationRGBBlendFactor(blend_factor(b.color_dst));
                att.setRgbBlendOperation(blend_op(b.color_op));
                att.setSourceAlphaBlendFactor(blend_factor(b.alpha_src));
                att.setDestinationAlphaBlendFactor(blend_factor(b.alpha_dst));
                att.setAlphaBlendOperation(blend_op(b.alpha_op));
            }
        }
        if key.depth != 0 {
            desc.setDepthAttachmentPixelFormat(MTLPixelFormat(key.depth));
            if key.stencil {
                desc.setStencilAttachmentPixelFormat(MTLPixelFormat(key.depth));
            }
        }
        let pipeline = match self.device.newRenderPipelineStateWithDescriptor_error(&desc) {
            Ok(p) => Some(p),
            Err(e) => {
                self.report(format!("render pipeline failed: {}", e.localizedDescription()));
                None
            }
        };
        self.pipelines.insert(key.clone(), pipeline.clone());
        pipeline
    }

    /// The render encoder for color targets `targets`, reusing the open pass when it matches.
    fn pass(
        &mut self,
        targets: &[(u64, Retained<Texture>)],
        depth: Option<(u64, Retained<Texture>, bool)>,
        clear: Option<(f32, u32)>,
    ) -> Option<Retained<RenderEncoder>> {
        let mut addresses: Vec<u64> = targets.iter().map(|t| t.0).collect();
        addresses.push(depth.as_ref().map_or(0, |d| d.0));
        if let Some((enc, open)) = &self.pass
            && *open == addresses
            && clear.is_none()
        {
            return Some(enc.clone());
        }
        self.end_pass();
        let command = self.command();
        let pass = MTLRenderPassDescriptor::new();
        for (i, (_, tex)) in targets.iter().enumerate() {
            let att = unsafe { pass.colorAttachments().objectAtIndexedSubscript(i) };
            att.setTexture(Some(tex));
            att.setLoadAction(MTLLoadAction::Load);
            att.setStoreAction(MTLStoreAction::Store);
        }
        if let Some((_, tex, stencil)) = &depth {
            let d = pass.depthAttachment();
            d.setTexture(Some(tex));
            d.setStoreAction(MTLStoreAction::Store);
            let st = pass.stencilAttachment();
            if *stencil {
                st.setTexture(Some(tex));
                st.setStoreAction(MTLStoreAction::Store);
            }
            match clear {
                Some((z, s)) => {
                    d.setLoadAction(MTLLoadAction::Clear);
                    d.setClearDepth(z as f64);
                    if *stencil {
                        st.setLoadAction(MTLLoadAction::Clear);
                        st.setClearStencil(s);
                    }
                }
                None => {
                    d.setLoadAction(MTLLoadAction::Load);
                    st.setLoadAction(MTLLoadAction::Load);
                }
            }
        }
        let enc = command.renderCommandEncoderWithDescriptor(&pass)?;
        self.pass = Some((enc.clone(), addresses));
        Some(enc)
    }

    /// The depth texture at `d`'s address; new ones start cleared.
    fn depth(&mut self, d: &DepthTarget, width: u32, height: u32) -> Option<(Retained<Texture>, MTLPixelFormat, bool)> {
        let format = match (d.format, d.stencil) {
            (_, true) => MTLPixelFormat::Depth32Float_Stencil8,
            (1, false) => MTLPixelFormat::Depth16Unorm,
            _ => MTLPixelFormat::Depth32Float,
        };
        let (w, h) = if d.width != 0 { (d.width, d.height) } else { (width, height) };
        if let Some(t) = self.depths.get(&d.address)
            && t.pixelFormat() == format
            && t.width() == w as usize
            && t.height() == h as usize
        {
            return Some((t.clone(), format, false));
        }
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format, w as usize, h as usize, false)
        };
        desc.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        desc.setStorageMode(MTLStorageMode::Private);
        let t = self.device.newTextureWithDescriptor(&desc)?;
        self.depths.insert(d.address, t.clone());
        Some((t, format, true))
    }

    fn depth_state(&mut self, s: DepthState, bound: bool) -> Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
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
            let front = face((s.control >> 8) & 7, sc & 0xF, (sc >> 4) & 0xF, (sc >> 8) & 0xF, s.ref_mask);
            let back_on = s.control & (1 << 7) != 0;
            let back = if back_on {
                face((s.control >> 20) & 7, (sc >> 12) & 0xF, (sc >> 16) & 0xF, (sc >> 20) & 0xF, s.ref_mask_back)
            } else {
                face((s.control >> 8) & 7, sc & 0xF, (sc >> 4) & 0xF, (sc >> 8) & 0xF, s.ref_mask)
            };
            d.setFrontFaceStencil(Some(&front));
            d.setBackFaceStencil(Some(&back));
        }
        let state = self.device.newDepthStencilStateWithDescriptor(&d)?;
        self.depth_states.insert((s, bound), state.clone());
        Some(state)
    }

    /// Descriptor dwords from user data or memory.
    fn descriptor(&self, host: &dyn Host, user: &[u32], source: Source, n: usize) -> Option<Vec<u32>> {
        match source {
            Source::UserSgpr(s) => user.get(s as usize..s as usize + n).map(<[u32]>::to_vec),
            Source::Memory { sgpr, offset } => {
                let s = sgpr as usize;
                let ptr = *user.get(s)? as u64 | (*user.get(s + 1)? as u64) << 32;
                let mut b = vec![0u8; n * 4];
                host.read(ptr + offset as u64, &mut b)
                    .then(|| b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
            }
        }
    }

    /// The resources `program` binds, made ready before any encoder opens
    /// (uploads end the open pass).
    fn resolve(&mut self, host: &dyn Host, program: &Program, user: &[u32]) -> Option<Bound> {
        let mut bound = Bound { memory: None, textures: Vec::new(), samplers: Vec::new() };
        if program.translated.guest_memory {
            let Some(m) = self.guest_window(host, user, program.address) else {
                self.report(format!("no guest memory window for the program at {:#x}, user data {user:x?}", program.address));
                return None;
            };
            bound.memory = Some(m);
        }
        for r in &program.translated.resources {
            match *r {
                Resource::Texture { slot, source, kind } => {
                    if kind != TextureKind::Tex2d {
                        self.report(format!("{kind:?} textures not supported"));
                        return None;
                    }
                    let d = self.descriptor(host, user, source, 8)?;
                    let image = Image::read(&d);
                    let tex = if image.address == 0 { self.null_texture() } else { self.image(host, image) };
                    let Some(tex) = tex else {
                        self.report(format!("image {image:x?} for the program at {:#x} not supported", program.address));
                        return None;
                    };
                    bound.textures.push((slot as usize, tex));
                }
                Resource::Sampler { slot, source } => {
                    let d = self.descriptor(host, user, source, 4)?;
                    bound.samplers.push((slot as usize, self.sampler(Sampler::read(&d))?));
                }
            }
        }
        Some(bound)
    }

    fn apply(enc: Slot, bound: &Bound, user: &[u32]) {
        enc.bytes(user.as_ptr().cast(), user.len() * 4, binding::USER as usize);
        if let Some((buf, base)) = &bound.memory {
            enc.buffer(buf, 0, binding::MEM as usize);
            enc.bytes((base as *const u64).cast(), 8, binding::MEM_BASE as usize);
        }
        for (slot, t) in &bound.textures {
            enc.texture(t, *slot);
        }
        for (slot, s) in &bound.samplers {
            enc.sampler(s, *slot);
        }
    }

    /// A window over the guest memory a program's user data points into,
    /// and the guest address of its first byte.
    fn guest_window(&mut self, host: &dyn Host, user: &[u32], code: u64) -> Option<(Retained<Buffer>, u64)> {
        let max = self.device.maxBufferLength() as u64;
        let mut best = host.extent(code);
        for pair in user.windows(2) {
            let ptr = pair[0] as u64 | (pair[1] as u64) << 32;
            if let Some((lo, hi)) = host.extent(ptr)
                && best.is_none_or(|(l, h)| hi - lo > h - l)
            {
                best = Some((lo, hi));
            }
        }
        let (lo, hi) = best?;
        let len = (hi - lo).min(max & !(PAGE - 1));
        let (buf, offset) = self.window(host, lo, len)?;
        Some((buf, lo - offset as u64))
    }

    fn try_dispatch(&mut self, host: &dyn Host, state: &State, groups: [u32; 3]) -> Result<(), &'static str> {
        let s = state.sh;
        let address = ((s[sh::COMPUTE_PGM_LO] as u64) | (s[sh::COMPUTE_PGM_LO + 1] as u64) << 32) << 8;
        if address == 0 {
            return Err("no compute program");
        }
        let users = (s[sh::COMPUTE_PGM_RSRC2] >> 1) & 0x1F;
        let program = self.program(host, state, Stage::Compute, address, users, false).ok_or("compute program")?;
        let pipeline = match self.computes.get(&address) {
            Some(p) => p.clone(),
            None => {
                let p = self.device.newComputePipelineStateWithFunction_error(&program.function).ok();
                self.computes.insert(address, p.clone());
                p
            }
        }
        .ok_or("compute pipeline")?;
        let threads = [0, 1, 2].map(|i| (s[sh::COMPUTE_NUM_THREAD_X + i] & 0xFFFF).max(1) as usize);
        let user = &s[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + sh::COMPUTE_USER_DATA_COUNT];
        let bound = self.resolve(host, &program, user);
        self.end_pass();
        let command = self.command();
        let enc = command.computeCommandEncoder().ok_or("compute encoder")?;
        enc.setComputePipelineState(&pipeline);
        if let Some(b) = &bound {
            Self::apply(Slot::Compute(&enc), b, user);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups[0] as usize,
                    height: groups[1] as usize,
                    depth: groups[2] as usize,
                },
                MTLSize {
                    width: threads[0],
                    height: threads[1],
                    depth: threads[2],
                },
            );
        }
        enc.endEncoding();
        if bound.is_some() { Ok(()) } else { Err("resources") }
    }

    fn try_draw(&mut self, host: &dyn Host, state: &State, draw: Draw) -> Result<(), &'static str> {
        let (c, s, u) = (state.context, state.sh, state.uconfig);
        let ngg = c[context::VGT_SHADER_STAGES_EN] & (1 << 13) != 0;
        let (vs_lo, vs_rsrc2, vs_user) = if ngg {
            (sh::SPI_SHADER_PGM_LO_ES, sh::SPI_SHADER_PGM_RSRC2_GS, sh::SPI_SHADER_USER_DATA_GS_0)
        } else {
            (sh::SPI_SHADER_PGM_LO_VS, sh::SPI_SHADER_PGM_RSRC2_VS, sh::SPI_SHADER_USER_DATA_VS_0)
        };
        let program_address = |lo: usize| ((s[lo] as u64) | (s[lo + 1] as u64) << 32) << 8;
        let user_count = |rsrc2: u32| (rsrc2 >> 1) & 0x1F;
        let vs_addr = program_address(vs_lo);
        let ps_addr = program_address(sh::SPI_SHADER_PGM_LO_PS);
        if vs_addr == 0 || ps_addr == 0 {
            return Err("no vertex or pixel program");
        }
        let vs_users = user_count(s[vs_rsrc2]);
        let ps_users = user_count(s[sh::SPI_SHADER_PGM_RSRC2_PS]);
        let vertex = self.program(host, state, Stage::Vertex, vs_addr, vs_users, ngg).ok_or("vertex program")?;
        let pixel = self.program(host, state, Stage::Pixel, ps_addr, ps_users, false).ok_or("pixel program")?;
        let rect = u[uconfig::VGT_PRIMITIVE_TYPE] == 7;
        let vertex_function = if rect {
            self.rect_function(&vertex).ok_or("rectangle-list vertex program")?
        } else {
            vertex.function.clone()
        };

        let mask = c[context::CB_TARGET_MASK];
        let mut targets = Vec::new();
        let mut key = PipelineKey {
            vertex: vs_addr,
            pixel: ps_addr,
            formats: [0; 8],
            blends: [(Blend::read(c, 0), 0); 8],
            depth: 0,
            stencil: false,
            rect,
        };
        for n in 0..8 {
            let write = (mask >> (4 * n)) & 0xF;
            if write == 0 {
                continue;
            }
            let Some(t) = ColorTarget::read(c, n) else { continue };
            let (tex, format) = self.target(host, &t).ok_or("color target format")?;
            key.formats[targets.len()] = format.0;
            key.blends[targets.len()] = (Blend::read(c, n), write_mask(write, t.format.swap) as u8);
            targets.push((t.address, tex));
        }
        let depth_state = DepthState::read(c);
        let wants_depth = depth_state.depth_test() || depth_state.stencil();
        let size = targets.first().map(|t| (t.1.width() as u32, t.1.height() as u32));
        let mut depth = None;
        let mut clear = None;
        if let Some(d) = DepthTarget::read(c)
            && (wants_depth || c[context::DB_RENDER_CONTROL] & 3 != 0)
        {
            let (dw, dh) = size.unwrap_or((d.width, d.height));
            if let Some((tex, format, fresh)) = self.depth(&d, dw, dh) {
                key.depth = format.0;
                key.stencil = d.stencil;
                let clearing = c[context::DB_RENDER_CONTROL] & 3 != 0;
                if clearing || fresh {
                    let z = if clearing { f32::from_bits(c[context::DB_DEPTH_CLEAR]) } else { 1.0 };
                    clear = Some((z, c[context::DB_STENCIL_CLEAR] & 0xFF));
                }
                depth = Some((d.address, tex, d.stencil));
            }
        }
        if targets.is_empty() && depth.is_none() {
            return Err("no color target");
        }
        let pipeline = self.pipeline(&key, &vertex_function, &pixel).ok_or("pipeline")?;
        let depth_bound = depth.is_some();
        let (w, h) = match (size, &depth) {
            (Some((w, h)), _) => (w as f64, h as f64),
            (None, Some((_, t, _))) => (t.width() as f64, t.height() as f64),
            _ => return Err("no target size"),
        };
        let vs_data = &s[vs_user..vs_user + sh::USER_DATA_COUNT];
        let ps_data = &s[sh::SPI_SHADER_USER_DATA_PS_0..sh::SPI_SHADER_USER_DATA_PS_0 + sh::USER_DATA_COUNT];
        let vs_bound = self.resolve(host, &vertex, vs_data).ok_or("resources")?;
        let ps_bound = self.resolve(host, &pixel, ps_data).ok_or("resources")?;
        let index = match draw {
            Draw::Indexed { count, address, .. } if !rect => {
                let wide = u[uconfig::VGT_INDEX_TYPE] & 3 == 1;
                let size = if wide { 4 } else { 2 };
                Some((self.window(host, address, count as u64 * size).ok_or("index buffer")?, wide))
            }
            _ => None,
        };
        let enc = self.pass(&targets, depth, clear).ok_or("render pass")?;
        enc.setRenderPipelineState(&pipeline);
        if depth_bound {
            let ds = self.depth_state(depth_state, true).ok_or("depth state")?;
            enc.setDepthStencilState(Some(&ds));
            enc.setStencilReferenceValue(depth_state.ref_mask & 0xFF);
        }
        let vp = Viewport::read(c);
        enc.setViewport(MTLViewport {
            originX: (vp.x_offset - vp.x_scale) as f64,
            originY: (vp.y_offset - vp.y_scale.abs()) as f64,
            width: (2.0 * vp.x_scale) as f64,
            height: (2.0 * vp.y_scale.abs()) as f64,
            znear: (vp.z_offset) as f64,
            zfar: (vp.z_offset + vp.z_scale) as f64,
        });
        let (x0, y0, x1, y1) = regs::scissor(c);
        let (x1, y1) = ((x1 as f64).min(w) as usize, (y1 as f64).min(h) as usize);
        let (x0, y0) = ((x0 as usize).min(x1), (y0 as usize).min(y1));
        enc.setScissorRect(MTLScissorRect {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        });
        let mode = c[context::PA_SU_SC_MODE_CNTL];
        enc.setCullMode(match mode & 3 {
            1 => MTLCullMode::Front,
            2 => MTLCullMode::Back,
            _ => MTLCullMode::None,
        });
        enc.setFrontFacingWinding(if mode & 4 != 0 {
            MTLWinding::Clockwise
        } else {
            MTLWinding::CounterClockwise
        });
        Self::apply(Slot::Vertex(&enc), &vs_bound, vs_data);
        Self::apply(Slot::Fragment(&enc), &ps_bound, ps_data);
        let primitive = match u[uconfig::VGT_PRIMITIVE_TYPE] {
            1 => MTLPrimitiveType::Point,
            2 => MTLPrimitiveType::Line,
            3 => MTLPrimitiveType::LineStrip,
            4 | 7 => MTLPrimitiveType::Triangle,
            6 => MTLPrimitiveType::TriangleStrip,
            _ => return Err("primitive type"),
        };
        match draw {
            Draw::Auto { count, instances } if rect => unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(primitive, 0, (count / 3 * 6) as usize, instances.max(1) as usize);
            },
            Draw::Indexed { .. } if rect => return Err("indexed rectangle list"),
            Draw::Auto { count, instances } => unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(primitive, 0, count as usize, instances.max(1) as usize);
            },
            Draw::Indexed { count, instances, .. } => {
                let ((buf, offset), wide) = index.ok_or("index buffer")?;
                unsafe {
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                        primitive,
                        count as usize,
                        if wide { MTLIndexType::UInt32 } else { MTLIndexType::UInt16 },
                        &buf,
                        offset,
                        instances.max(1) as usize,
                    );
                }
            }
        }
        Ok(())
    }

    fn flush_and_wait(&mut self) {
        self.end_pass();
        if let Some(cb) = self.command.take() {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }
}

impl Backend for Metal {
    fn draw(&mut self, host: &dyn Host, state: &State, draw: Draw) {
        match self.try_draw(host, state, draw) {
            Ok(()) => self.draws += 1,
            Err(why) => {
                self.skipped += 1;
                if why != "vertex program" && why != "pixel program" {
                    self.report(format!("draw skipped: {why}"));
                }
            }
        }
    }

    fn dispatch(&mut self, host: &dyn Host, state: &State, groups: [u32; 3]) {
        match self.try_dispatch(host, state, groups) {
            Ok(()) => self.dispatches += 1,
            Err(why) => {
                self.skipped += 1;
                if why != "compute program" {
                    self.report(format!("dispatch skipped: {why}"));
                }
            }
        }
    }

    fn sync(&mut self) {
        self.flush_and_wait();
    }

    fn summary(&self) -> String {
        format!("{} drawn, {} dispatched, {} skipped", self.draws, self.dispatches, self.skipped)
    }

    fn frame_done(&mut self, shown: Option<u64>) {
        if let Some(addr) = shown
            && let Some(t) = self.targets.get(&addr)
        {
            let src = t.texture.clone();
            let copy = match self.shown.get(&addr) {
                Some(c) if c.width() == src.width() && c.height() == src.height() && c.pixelFormat() == src.pixelFormat() => c.clone(),
                _ => {
                    let c = self.new_texture(src.pixelFormat(), src.width() as u32, src.height() as u32, false);
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
            if let Ok(mut f) = frames().lock() {
                f.insert(addr, SharedTexture(copy));
            }
        } else {
            self.flush_and_wait();
        }
        self.epoch += 1;
    }
}

// Each rectangle draws guest vertices 0, 1, 2, 1, 3, 2; corner 3 is v1 + v2 - v0.
fn rect_source(msl: &str) -> Option<String> {
    let start = msl.find("vertex VsOut gen5_vs(")?;
    let open = start + "vertex VsOut gen5_vs(".len();
    let close = open + msl[open..].find(") {")?;
    let params = msl[open..close].split_once("[[instance_id]], ")?.1;
    let call = msl[close..].find("return gen5_vs_body(vid, iid")? + close + "return gen5_vs_body(vid, iid".len();
    let call_end = call + msl[call..].find(");")?;
    let args = &msl[call..call_end];
    let fields: Vec<&str> = msl[msl.find("struct VsOut {")?..]
        .lines()
        .take_while(|l| !l.starts_with("};"))
        .filter_map(|l| l.trim().strip_prefix("float4 ")?.split_whitespace().next())
        .collect();
    let combine: String = fields.iter().map(|f| format!("    o.{f} = b.{f} + c.{f} - a.{f};\n")).collect();
    Some(format!(
        "{msl}\nvertex VsOut gen5_vs_rect(uint rv [[vertex_id]], uint iid [[instance_id]], {params}) {{\n    uint base = rv / 6 * 3;\n    uint k = (0x231210u >> ((rv % 6) * 4)) & 0xFu;\n    if (k < 3) return gen5_vs_body(base + k, iid{args});\n    VsOut a = gen5_vs_body(base, iid{args});\n    VsOut b = gen5_vs_body(base + 1, iid{args});\n    VsOut c = gen5_vs_body(base + 2, iid{args});\n    VsOut o = a;\n{combine}    return o;\n}}\n"
    ))
}

fn write_mask(bits: u32, swap: u32) -> usize {
    let (r, g, b, a) = (bits & 1, (bits >> 1) & 1, (bits >> 2) & 1, (bits >> 3) & 1);
    let (r, b) = if swap & 1 == 1 { (b, r) } else { (r, b) };
    let m = |on: u32, bit: MTLColorWriteMask| if on != 0 { bit.0 } else { 0 };
    m(r, MTLColorWriteMask::Red) | m(g, MTLColorWriteMask::Green) | m(b, MTLColorWriteMask::Blue) | m(a, MTLColorWriteMask::Alpha)
}

/// The Metal format of a color target, for the layouts titles render to.
fn color_format(f: regs::ColorFormat) -> Option<MTLPixelFormat> {
    use MTLPixelFormat as P;
    let srgb = f.number == 6;
    Some(match (f.layout, f.number, f.swap & 1) {
        (10, _, 0) if srgb => P::RGBA8Unorm_sRGB,
        (10, _, 1) if srgb => P::BGRA8Unorm_sRGB,
        (10, 0, 0) => P::RGBA8Unorm,
        (10, 0, 1) => P::BGRA8Unorm,
        (10, 4, _) => P::RGBA8Uint,
        (12, 0, _) => P::RGB10A2Unorm,
        (6, 7, _) => P::RG11B10Float,
        (12, 7, _) | (13, 7, _) => P::RGB10A2Unorm,
        (5, 7, _) => P::RG16Float,
        (5, 0, _) => P::RG16Unorm,
        (2, 7, _) => P::R16Float,
        (2, 0, _) => P::R16Unorm,
        (4, 7, _) => P::R32Float,
        (4, 4, _) => P::R32Uint,
        (1, 0, _) => P::R8Unorm,
        (3, 0, _) => P::RG8Unorm,
        (14, 7, _) => P::RGBA16Float,
        (14, 0, _) => P::RGBA16Unorm,
        (11, 7, _) => P::RG32Float,
        (17, 7, _) => P::RGBA32Float,
        _ => return None,
    })
}

/// The Metal format and bytes per element of a GFX10 image format.
fn image_format(format: u32) -> Option<(MTLPixelFormat, u32)> {
    use MTLPixelFormat as P;
    Some(match format {
        1 => (P::R8Unorm, 1),
        5 => (P::R8Uint, 1),
        7 => (P::R16Unorm, 2),
        13 => (P::R16Float, 2),
        14 => (P::RG8Unorm, 2),
        20 => (P::R32Uint, 4),
        22 => (P::R32Float, 4),
        23 => (P::RG16Unorm, 4),
        29 => (P::RG16Float, 4),
        36 | 43 => (P::RG11B10Float, 4),
        44 | 50 => (P::RGB10A2Unorm, 4),
        56 => (P::RGBA8Unorm, 4),
        57 => (P::RGBA8Snorm, 4),
        60 => (P::RGBA8Uint, 4),
        64 => (P::RG32Float, 8),
        65 => (P::RGBA16Unorm, 8),
        71 => (P::RGBA16Float, 8),
        77 => (P::RGBA32Float, 16),
        128 => (P::R8Unorm_sRGB, 1),
        130 => (P::RGBA8Unorm_sRGB, 4),
        169 => (P::BC1_RGBA, 8),
        170 => (P::BC1_RGBA_sRGB, 8),
        171 => (P::BC2_RGBA, 16),
        172 => (P::BC2_RGBA_sRGB, 16),
        173 => (P::BC3_RGBA, 16),
        174 => (P::BC3_RGBA_sRGB, 16),
        175 => (P::BC4_RUnorm, 8),
        176 => (P::BC4_RSnorm, 8),
        177 => (P::BC5_RGUnorm, 16),
        178 => (P::BC5_RGSnorm, 16),
        179 => (P::BC6H_RGBUfloat, 16),
        180 => (P::BC6H_RGBFloat, 16),
        181 => (P::BC7_RGBAUnorm, 16),
        182 => (P::BC7_RGBAUnorm_sRGB, 16),
        _ => return None,
    })
}

fn is_compressed(f: MTLPixelFormat) -> bool {
    (MTLPixelFormat::BC1_RGBA.0..=MTLPixelFormat::BC7_RGBAUnorm_sRGB.0).contains(&f.0)
}

fn blend_factor(f: u32) -> MTLBlendFactor {
    use MTLBlendFactor as B;
    match f {
        0 => B::Zero,
        1 => B::One,
        2 => B::SourceColor,
        3 => B::OneMinusSourceColor,
        4 => B::SourceAlpha,
        5 => B::OneMinusSourceAlpha,
        6 => B::DestinationAlpha,
        7 => B::OneMinusDestinationAlpha,
        8 => B::DestinationColor,
        9 => B::OneMinusDestinationColor,
        10 => B::SourceAlphaSaturated,
        13 => B::BlendColor,
        14 => B::OneMinusBlendColor,
        15 => B::Source1Color,
        16 => B::OneMinusSource1Color,
        17 => B::Source1Alpha,
        18 => B::OneMinusSource1Alpha,
        19 => B::BlendAlpha,
        20 => B::OneMinusBlendAlpha,
        _ => B::One,
    }
}

fn compare(f: u32) -> MTLCompareFunction {
    use MTLCompareFunction as C;
    match f {
        0 => C::Never,
        1 => C::Less,
        2 => C::Equal,
        3 => C::LessEqual,
        4 => C::Greater,
        5 => C::NotEqual,
        6 => C::GreaterEqual,
        _ => C::Always,
    }
}

fn stencil_op(op: u32) -> MTLStencilOperation {
    use MTLStencilOperation as S;
    match op {
        1 => S::Zero,
        3 | 4 => S::Replace,
        5 => S::IncrementClamp,
        6 => S::DecrementClamp,
        7 => S::Invert,
        8 => S::IncrementWrap,
        9 => S::DecrementWrap,
        _ => S::Keep,
    }
}

fn blend_op(op: u32) -> MTLBlendOperation {
    match op {
        1 => MTLBlendOperation::Subtract,
        2 => MTLBlendOperation::Min,
        3 => MTLBlendOperation::Max,
        4 => MTLBlendOperation::ReverseSubtract,
        _ => MTLBlendOperation::Add,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Guest memory for tests: one page-aligned host allocation at guest address 0.
    struct Mem {
        bytes: RefCell<Vec<u8>>,
        base: *mut u8,
        len: u64,
    }

    impl Mem {
        fn new(len: u64) -> Self {
            // SAFETY: a fresh anonymous mapping.
            let base = unsafe {
                libc::mmap(std::ptr::null_mut(), len as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0)
            } as *mut u8;
            Self {
                bytes: RefCell::new(Vec::new()),
                base,
                len,
            }
        }
    }

    impl Host for Mem {
        fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
            let _ = &self.bytes;
            // SAFETY: tests stay inside the mapping.
            unsafe { std::ptr::copy_nonoverlapping(self.base.add(addr as usize), buf.as_mut_ptr(), buf.len()) };
            true
        }
        fn write(&self, addr: u64, data: &[u8]) -> bool {
            // SAFETY: tests stay inside the mapping.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(addr as usize), data.len()) };
            true
        }
        fn interrupt(&self, _: u32) {}
        fn flip(&self, _: crate::Flip) {}
        fn exiting(&self) -> bool {
            false
        }
        fn extent(&self, addr: u64) -> Option<(u64, u64)> {
            (addr < self.len).then_some((0, self.len))
        }
        fn host_pointer(&self, addr: u64) -> Option<*mut u8> {
            // SAFETY: inside the mapping.
            Some(unsafe { self.base.add(addr as usize) })
        }
    }

    #[test]
    fn detiling_a_render_target_on_the_gpu_matches_the_cpu_layout() {
        let Some(mut metal) = Metal::new() else { return };
        let mem = Mem::new(4 << 20);
        let (w, h) = (300u32, 200u32);
        let layout = Layout::new(TileMode::RenderTarget, 4, w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                mem.write(layout.offset(x, y), &(x | y << 16).to_le_bytes());
            }
        }
        let tex = metal.new_texture(MTLPixelFormat::R32Uint, w, h, false).unwrap();
        assert!(metal.detile_into(&mem, &tex, 0, layout));
        let shared = {
            let d = unsafe { MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::R32Uint, w as usize, h as usize, false) };
            d.setStorageMode(MTLStorageMode::Shared);
            metal.device.newTextureWithDescriptor(&d).unwrap()
        };
        let cb = metal.command();
        let blit = cb.blitCommandEncoder().unwrap();
        unsafe { blit.copyFromTexture_toTexture(&tex, &shared) };
        blit.endEncoding();
        metal.flush_and_wait();
        let mut out = vec![0u32; (w * h) as usize];
        unsafe {
            shared.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new_unchecked(out.as_mut_ptr().cast()),
                w as usize * 4,
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize { width: w as usize, height: h as usize, depth: 1 },
                },
                0,
            );
        }
        for y in 0..h {
            for x in 0..w {
                assert_eq!(out[(y * w + x) as usize], x | y << 16);
            }
        }
    }
}
