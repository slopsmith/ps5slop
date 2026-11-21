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

use crate::compile::Task;
use crate::regs::{self, Blend, ColorTarget, DepthState, DepthTarget, Image, Sampler, Viewport, context, sh, uconfig};
use gfx10::translate::{self as shader, Access, Resource, Source, Stage, TextureKind, Translated, binding};
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

/// A vertex program, whether it draws rectangles or uses OpenGL clip space,
/// and the attributes it must add.
type VariantKey = (u64, bool, bool, Vec<u32>, Vec<u32>);

/// A Metal texture other threads may hold.
#[derive(Clone)]
pub struct SharedTexture(pub Retained<Texture>);

// SAFETY: Metal textures may be used from any thread.
unsafe impl Send for SharedTexture {}
// SAFETY: as above.
unsafe impl Sync for SharedTexture {}

/// Render targets by guest address, for presenting flipped buffers.
static FRAMES: OnceLock<Mutex<HashMap<u64, SharedTexture>>> = OnceLock::new();

/// Depth target array slices we keep at most (cascaded shadow maps use 4).
const MAX_DEPTH_SLICES: usize = 16;
/// Color target array slices we keep at most (cube maps use 6).
const MAX_TARGET_SLICES: u32 = 16;

fn frames() -> &'static Mutex<HashMap<u64, SharedTexture>> {
    FRAMES.get_or_init(Default::default)
}

/// Frames still to trace, from the next one on (the control socket's `gputrace`).
pub static TRACE_NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records targets and textures for the control socket's `targets`, set
/// at run time; `PS5SLOP_DUMP_TARGETS` sets it from the start.
pub static DUMP_TARGETS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn dump_targets() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    DUMP_TARGETS.load(std::sync::atomic::Ordering::Relaxed) || *ENV.get_or_init(|| std::env::var_os("PS5SLOP_DUMP_TARGETS").is_some())
}

/// Keeps `texture` under `key` for `targets`, replacing what is there when `replace`.
fn record(key: u64, texture: &Retained<Texture>, replace: bool) {
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
struct Snap {
    dir: std::path::PathBuf,
    from: u32,
    passes: u32,
    bytes: u64,
    /// File name and copy; depth copies are `R32Float`.
    copies: Vec<(String, Retained<Texture>)>,
    /// The programs of the open pass's draws, in order, without repeats.
    programs: Vec<(u64, u64)>,
    draws: u32,
}

/// Set by the control socket's `gpureload`: drop translated programs and
/// pipelines at the end of the frame, so `PS5SLOP_MSL_OVERRIDE` files are read again.
pub static RELOAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Programs whose draws and dispatches are skipped, for finding the pass
/// that causes a rendering bug (the control socket's `gpuskip`).
pub static SKIP: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// `PS5SLOP_TRACE_TARGETS`: logs each new render target and each image first read from guest memory.
fn trace_targets() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("PS5SLOP_TRACE_TARGETS").is_some())
}

fn skipped(address: u64) -> bool {
    SKIP.lock().is_ok_and(|s| s.contains(&address))
}

/// The texture the GPU rendered at guest address `address`, if any.
pub fn frame(address: u64) -> Option<SharedTexture> {
    frames().lock().ok()?.get(&address).cloned()
}

/// Every render target, by guest address.
pub fn all_frames() -> Vec<(u64, SharedTexture)> {
    frames().lock().map(|f| f.iter().map(|(a, t)| (*a, t.clone())).collect()).unwrap_or_default()
}

/// `texture`'s pixels as RGBA8, read back for screenshots. Other 2D formats
/// are converted on the GPU, clamped to [0, 1] (integers scaled from 16 bits).
pub fn read_rgba(texture: &SharedTexture) -> Option<(u32, u32, Vec<u8>)> {
    let t = &texture.0;
    let (w, h) = (t.width(), t.height());
    let format = t.pixelFormat();
    let bgra = matches!(format, MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::BGRA8Unorm_sRGB);
    if !bgra && !matches!(format, MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::RGBA8Unorm_sRGB) {
        return read_converted(t);
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

/// The texels of a 2D texture as stored, row by row with no padding.
fn read_raw(t: &Texture) -> Option<Vec<u8>> {
    let bpp = texel_bytes(t.pixelFormat())?;
    let (w, h) = (t.width(), t.height());
    let device = t.device();
    let queue = device.newCommandQueue()?;
    let buffer = device.newBufferWithLength_options(w * h * bpp, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let blit = cb.blitCommandEncoder()?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            t, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &buffer, 0, w * bpp, w * h * bpp,
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    Some(unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8, w * h * bpp) }.to_vec())
}

fn read_converted(t: &Texture) -> Option<(u32, u32, Vec<u8>)> {
    if t.textureType() != MTLTextureType::Type2D || is_compressed(t.pixelFormat()) {
        return None;
    }
    let (w, h) = (t.width(), t.height());
    let (elem, scale) = match number(t.pixelFormat()) {
        Number::Float => ("float", "1.0"),
        Number::Uint => ("uint", "1.0 / 65535.0"),
        Number::Sint => ("int", "1.0 / 32767.0"),
    };
    let msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void convert(texture2d<{elem}, access::read> src [[texture(0)]], device uchar4* dst [[buffer(0)]], uint2 id [[thread_position_in_grid]]) {{\n    if (id.x >= src.get_width() || id.y >= src.get_height()) return;\n    float4 v = clamp(float4(src.read(id)) * {scale}, 0.0, 1.0);\n    dst[id.y * src.get_width() + id.x] = uchar4(v * 255.0 + 0.5);\n}}\n"
    );
    let device = t.device();
    let lib = device.newLibraryWithSource_options_error(&NSString::from_str(&msl), None).ok()?;
    let f = lib.newFunctionWithName(&NSString::from_str("convert"))?;
    let pipeline = device.newComputePipelineStateWithFunction_error(&f).ok()?;
    let queue = device.newCommandQueue()?;
    let buffer = device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let enc = cb.computeCommandEncoder()?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setTexture_atIndex(Some(t), 0);
        enc.setBuffer_offset_atIndex(Some(&buffer), 0, 0);
    }
    enc.dispatchThreads_threadsPerThreadgroup(MTLSize { width: w, height: h, depth: 1 }, MTLSize { width: 16, height: 16, depth: 1 });
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let px = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8, w * h * 4) }.to_vec();
    Some((w as u32, h as u32, px))
}

/// An `R32Float` depth copy as grey RGBA8, stretched from its nearest to
/// its farthest depth below 1 (the cleared value) so near differences show.
fn read_depth(t: &Texture) -> Option<(u32, u32, Vec<u8>)> {
    let (w, h) = (t.width(), t.height());
    let device = t.device();
    let queue = device.newCommandQueue()?;
    let buffer = device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModeShared)?;
    let cb = queue.commandBuffer()?;
    let blit = cb.blitCommandEncoder()?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            t, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &buffer, 0, w * 4, w * h * 4,
        );
    }
    blit.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    let z = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, w * h) };
    let (lo, hi) = z.iter().filter(|v| **v < 1.0 && **v > 0.0).fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let scale = if hi > lo { 255.0 / (hi - lo) } else { 0.0 };
    let px = z
        .iter()
        .flat_map(|v| {
            let g = if *v >= 1.0 { 255 } else if *v <= 0.0 { 0 } else { ((v - lo) * scale).clamp(0.0, 254.0) as u8 };
            [g, g, g, 255]
        })
        .collect();
    Some((w as u32, h as u32, px))
}

/// Frames an image read from guest memory stays cached without being used.
const IMAGE_LIFETIME: u64 = 300;

/// Guest memory is wrapped in windows of about this size.
const WINDOW: u64 = 64 << 20;
const PAGE: u64 = 0x4000;
/// Draws whose shaders failed are reported this many times.
const REPORTS: u32 = 24;
/// Time per frame draws to redrawn targets may wait for their pipelines
/// before they are skipped.
const COMPILE_BUDGET: std::time::Duration = std::time::Duration::from_millis(8);

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
    memory: Option<Memory>,
    gds: Option<Retained<Buffer>>,
    /// Dword 0 of each sampler slot's S#, by slot.
    sampler_words: Vec<(usize, u32)>,
    textures: Vec<(usize, Retained<Texture>)>,
    samplers: Vec<(usize, Retained<ProtocolObject<dyn MTLSamplerState>>)>,
}

/// Binding of the second guest memory window (reached through `Win.delta`).
const SECOND_WINDOW: usize = 26;

/// The guest memory bound for a program and its `Win`: both windows' ranges
/// and the second's GPU address minus the first's.
struct Memory {
    buffer: Retained<Buffer>,
    second: Option<Retained<Buffer>>,
    win: [u64; 5],
}

struct Program {
    address: u64,
    /// Unique per translation, so caches tell apart programs loaded at one address.
    id: u64,
    function: Arc<Task<Retained<Function>>>,
    /// The pipeline of a compute program, compiled as soon as it is translated.
    compute: Option<Arc<Task<Retained<ComputePipeline>>>>,
    translated: Translated,
    /// The `user(locnN)` locations a pixel program reads flat.
    flat: Vec<u32>,
    /// Texture slots declared `<int>`.
    sint_textures: u32,
}

/// What a program is translated for besides its code and registers.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Variant {
    /// `SPI_SHADER_COL_FORMAT`, with integer targets' exports made integer.
    col_format: u32,
    uint_textures: u32,
    sint_textures: u32,
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

/// An open render pass: its encoder, attachment addresses, and color
/// textures and depth texture and slice (for `gpusnap`).
type OpenPass = (Retained<RenderEncoder>, Vec<u64>, Vec<Retained<Texture>>, Option<(Retained<Texture>, usize)>);

/// The Metal side of the GPU.
pub struct Metal {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Keeps the guest memory windows resident between command buffers.
    resident: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    command: Option<Retained<CommandBuffer>>,
    /// Committed command buffers not waited for yet, with their commit time.
    inflight: Vec<(Retained<CommandBuffer>, f64)>,
    /// Texture uploads for the work in `command`, committed just before it
    /// so they do not split its render passes.
    upload: Option<Retained<CommandBuffer>>,
    /// Where this backend is in the log of replaced guest pages.
    replaced_mark: u64,
    pass: Option<OpenPass>,
    snap: Option<Snap>,
    depths: HashMap<u64, Retained<Texture>>,
    /// Depth-layout images a program wrote since their depth texture was
    /// last used: the depth texture takes their contents first.
    depth_stores: std::collections::HashSet<u64>,
    /// Addresses a colour target or storage image wrote after any depth target
    /// there, so sampling them must not read the depth texture.
    color_newer: std::collections::HashSet<u64>,
    shown: HashMap<u64, Retained<Texture>>,
    swizzles: HashMap<(usize, [u8; 4]), Retained<Texture>>,
    depth_states: HashMap<(DepthState, bool), Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    windows: Vec<Window>,
    /// One window from the start of the host's GPU area (see
    /// [`Host::gpu_area`]) up to what programs have pointed into so far.
    area: Option<Window>,
    /// The command buffer last reported as stuck, so each is reported once.
    reported_stuck: Option<usize>,
    /// Work encoded since the last wait writes guest memory, so the CPU must
    /// wait for it before reading guest memory for later work (see `settle`).
    wrote_memory: bool,
    /// The uncommitted main command buffer holds work that writes guest
    /// memory, so detiles from guest memory must run after it, not before.
    pending_writes: bool,
    /// Command buffers Metal reported as failed.
    failed_commands: u64,
    /// Loop guard words of compute dispatches (see [`GUARD_SLOTS`]), the
    /// next slot, the slots of dispatches not yet checked with their program,
    /// and how many dispatches gave up.
    guard: Option<Retained<Buffer>>,
    /// With `PS5SLOP_GPU_FENCE`, orders encoders that read guest memory after
    /// those that wrote it (see [`Metal::wait_memory`]); whether one updated
    /// it yet, and whether the open pass writes guest memory.
    fence: Option<Retained<ProtocolObject<dyn MTLFence>>>,
    fence_updated: bool,
    pass_memory: bool,
    guard_next: usize,
    guarded: Vec<(usize, u64, [u32; 16])>,
    guard_trips: u64,
    /// Shared memory handed out front to back for data copied for the GPU,
    /// and how much of it is used; replaced when full.
    scratch: Option<(Retained<Buffer>, usize)>,
    /// The global data share, made on first use.
    gds: Option<Retained<Buffer>>,
    /// Pixel inputs each (vertex, pixel) program pair lacks, by program ids.
    missing_inputs: HashMap<(u64, u64), Vec<u32>>,
    generation: u64,
    targets: HashMap<u64, Target>,
    images: HashMap<(Image, TextureKind), (u64, Retained<Texture>)>,
    /// 2D-array and 3D images programs write, by guest address.
    volumes: HashMap<u64, Retained<Texture>>,
    volume_sources: HashMap<u64, VolumeSource>,
    /// Zero-filled stand-ins for depth-layout images no depth target backs.
    blank: HashMap<(Image, TextureKind), Retained<Texture>>,
    epoch: u64,
    detilers: HashMap<(TileMode, u32, Detile), Retained<ComputePipeline>>,
    programs: HashMap<(u64, u64), Option<Arc<Program>>>,
    variants: HashMap<VariantKey, Arc<Task<Retained<Function>>>>,
    pipelines: HashMap<PipelineKey, Arc<Task<Retained<RenderPipeline>>>>,
    /// The first and last frame each colour or depth target (address,
    /// pixel format, width) was drawn to.
    drawn: HashMap<(u64, usize, usize), (u64, u64)>,
    /// What this frame may still spend waiting for pipelines before skipping draws.
    compile_budget: std::time::Duration,
    /// Draws and dispatches that waited for a pipeline: how many, and how long.
    compile_waits: [(u64, std::time::Duration); 2],
    samplers: HashMap<Sampler, Retained<ProtocolObject<dyn MTLSamplerState>>>,
    reports: u32,
    null: HashMap<TextureKind, Retained<Texture>>,
    pub draws: u64,
    pub dispatches: u64,
    pub skipped: u64,
    reasons: HashMap<&'static str, u64>,
    /// The frames `PS5SLOP_GPU_TRACE` names (`n` or `first-last`), and the
    /// trace lines of the current draw.
    trace_frames: Option<(u64, u64)>,
    /// The context and uconfig registers at the last traced draw, for
    /// `PS5SLOP_TRACE_REGS`.
    traced_regs: Vec<u32>,
    trace: Option<Vec<String>>,
}

/// What a detile kernel writes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Detile {
    Texture,
    Buffer,
    /// Only the texels whose bytes differ from an older copy of the surface.
    Changed,
}

/// The guest memory a 2D-array storage image was last loaded from.
#[derive(Clone)]
struct VolumeSource {
    image: Image,
    bpe: u32,
    slices: u32,
    base: u64,
    len: u64,
    shadow: Option<Arc<Vec<u8>>>,
    epoch: u64,
}

/// Guest memory `base..base + len` in place, if it is host-backed and at most 32 MiB.
fn guest_bytes(host: &dyn Host, base: u64, len: u64) -> Option<&[u8]> {
    let (start, end) = host.extent(base)?;
    if base < start || base + len > end || len > 32 << 20 {
        return None;
    }
    let p = host.host_pointer(base)?;
    // SAFETY: the range is mapped guest memory; the guest may write it concurrently.
    Some(unsafe { std::slice::from_raw_parts(p, len as usize) })
}

/// The `user(locnN)` locations a pixel program reads flat (declared `uint4`).
fn flat_locations(msl: &str) -> Vec<u32> {
    msl.lines()
        .filter(|l| l.trim_start().starts_with("uint4 a") && l.ends_with("flat]];"))
        .filter_map(|l| l.split("[[user(locn").nth(1)?.split(')').next()?.parse().ok())
        .collect()
}

/// `msl` with vertex output `p{loc}` declared `uint4` and stored bit for bit.
fn uint_output(msl: &str, loc: u32) -> String {
    let decl = format!("    float4 p{loc} [[user(locn{loc})]];");
    let store = format!("o.p{loc} = ");
    let mut out = String::with_capacity(msl.len() + 64);
    let mut rest = msl.replacen(&decl, &format!("    uint4 p{loc} [[user(locn{loc})]];"), 1);
    while let Some(i) = rest.find(&store) {
        let value = i + store.len();
        let mut depth = 0i32;
        let Some(end) = rest[value..].char_indices().find_map(|(k, c)| {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ';' if depth == 0 => return Some(value + k),
                _ => {}
            }
            None
        }) else {
            break;
        };
        out.push_str(&rest[..value]);
        out.push_str(&format!("as_type<uint4>({})", &rest[value..end]));
        rest = rest[end..].to_string();
    }
    out.push_str(&rest);
    out
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
        let fence = std::env::var_os("PS5SLOP_GPU_FENCE").and_then(|_| device.newFence());
        let resident = std::env::var_os("PS5SLOP_GPU_RESIDENCY_SET").is_some()
            .then(|| device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).ok())
            .flatten();
        if let Some(r) = &resident {
            queue.addResidencySet(r);
        }
        Some(Self {
            device,
            queue,
            resident,
            command: None,
            inflight: Vec::new(),
            upload: None,
            replaced_mark: 0,
            pass: None,
            snap: None,
            depths: HashMap::new(),
            depth_stores: Default::default(),
            color_newer: Default::default(),
            shown: HashMap::new(),
            swizzles: HashMap::new(),
            depth_states: HashMap::new(),
            windows: Vec::new(),
            area: None,
            reported_stuck: None,
            wrote_memory: false,
            pending_writes: false,
            failed_commands: 0,
            guard: None,
            fence,
            fence_updated: false,
            pass_memory: false,
            guard_next: 0,
            guarded: Vec::new(),
            guard_trips: 0,
            missing_inputs: HashMap::new(),
            scratch: None,
            gds: None,
            generation: 0,
            targets: HashMap::new(),
            images: HashMap::new(),
            volumes: HashMap::new(),
            volume_sources: HashMap::new(),
            blank: HashMap::new(),
            epoch: 0,
            detilers: HashMap::new(),
            programs: HashMap::new(),
            variants: HashMap::new(),
            pipelines: HashMap::new(),
            drawn: HashMap::new(),
            compile_budget: COMPILE_BUDGET,
            compile_waits: Default::default(),
            samplers: HashMap::new(),
            reports: 0,
            null: HashMap::new(),
            draws: 0,
            dispatches: 0,
            skipped: 0,
            reasons: HashMap::new(),
            traced_regs: Vec::new(),
            trace_frames: std::env::var("PS5SLOP_GPU_TRACE").ok().and_then(|v| match v.split_once('-') {
                Some((a, b)) => Some((a.parse().ok()?, b.parse().ok()?)),
                None => v.parse().ok().map(|n| (n, n)),
            }),
            trace: None,
        })
    }

    /// Counts a wait for a pipeline that began at `since`, by a draw (0) or a dispatch (1).
    fn note_wait(&mut self, kind: usize, since: std::time::Instant) {
        let waited = since.elapsed();
        crate::timing::add_raw(crate::timing::Part::CompileWait, waited.as_nanos() as u64);
        self.compile_waits[kind].0 += 1;
        self.compile_waits[kind].1 += waited;
    }

    fn note(&mut self, what: impl FnOnce() -> String) {
        if let Some(t) = &mut self.trace {
            t.push(what());
        }
    }

    fn report(&mut self, what: String) {
        if self.reports < REPORTS {
            self.reports += 1;
            eprintln!("ps5slop: GPU: {what}");
        }
    }

    fn command(&mut self) -> Retained<CommandBuffer> {
        if self.command.is_none() {
            self.command = self.new_command_buffer();
        }
        self.command.clone().expect("Metal hands out command buffers")
    }

    /// A command buffer; with `PS5SLOP_GPU_ENCODER_STATUS`, one whose errors say which encoder failed.
    fn new_command_buffer(&self) -> Option<Retained<CommandBuffer>> {
        static STATUS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*STATUS.get_or_init(|| std::env::var_os("PS5SLOP_GPU_ENCODER_STATUS").is_some()) {
            return self.queue.commandBuffer();
        }
        let d = objc2_metal::MTLCommandBufferDescriptor::new();
        d.setErrorOptions(objc2_metal::MTLCommandBufferErrorOption::EncoderExecutionStatus);
        self.queue.commandBufferWithDescriptor(&d)
    }

    /// The command buffer for a texture upload the next draw needs.
    fn upload_command(&mut self) -> Retained<CommandBuffer> {
        static INLINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if self.pending_writes || *INLINE.get_or_init(|| std::env::var_os("PS5SLOP_GPU_INLINE_UPLOADS").is_some()) {
            self.end_pass();
            return self.command();
        }
        if self.upload.is_none() {
            self.upload = self.new_command_buffer();
        }
        self.upload.clone().expect("Metal hands out command buffers")
    }

    /// The fence encoders that read guest memory wait for, once a writer updated it:
    /// Metal orders work only within one buffer, and our windows are several.
    fn wait_memory(&self) -> Option<&ProtocolObject<dyn MTLFence>> {
        self.fence.as_deref().filter(|_| self.fence_updated)
    }

    fn end_pass(&mut self) {
        if let Some((enc, addresses, colors, depth)) = self.pass.take() {
            if let Some(f) = self.fence.as_deref()
                && std::mem::take(&mut self.pass_memory)
            {
                enc.updateFence_afterStages(f, MTLRenderStages::Fragment);
                self.fence_updated = true;
            }
            enc.endEncoding();
            if self.snap.is_some() {
                self.snap_pass(&addresses, &colors, depth);
            }
        }
    }

    /// For `gpusnap`: copies the 2D textures a pixel program samples, as they are just before its draw.
    fn snap_inputs(&mut self, program: u64, bound: &Bound) {
        self.end_pass();
        let command = self.command();
        let Some(snap) = self.snap.as_mut() else { return };
        if snap.passes < snap.from {
            return;
        }
        let Some(blit) = command.blitCommandEncoder() else { return };
        for (slot, t) in &bound.textures {
            let size = (t.width() * t.height() * texel_bytes(t.pixelFormat()).unwrap_or(16)) as u64;
            if t.textureType() != MTLTextureType::Type2D || snap.bytes + size > SNAP_BYTES || is_compressed(t.pixelFormat()) {
                continue;
            }
            let d = unsafe { MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(t.pixelFormat(), t.width(), t.height(), false) };
            d.setUsage(MTLTextureUsage::ShaderRead);
            d.setStorageMode(MTLStorageMode::Private);
            let Some(copy) = self.device.newTextureWithDescriptor(&d) else { continue };
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    t, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: t.width(), height: t.height(), depth: 1 }, &copy, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            snap.bytes += size;
            snap.copies.push((format!("{:03}-in-{program:x}-t{slot}", snap.passes + 1), copy));
        }
        blit.endEncoding();
    }

    /// Copies the attachments of the pass that just ended for `gpusnap`.
    fn snap_pass(&mut self, addresses: &[u64], colors: &[Retained<Texture>], depth: Option<(Retained<Texture>, usize)>) {
        let Some(command) = self.command.clone() else { return };
        let Some(snap) = self.snap.as_mut() else { return };
        snap.passes += 1;
        let programs: Vec<String> = snap.programs.iter().map(|(v, p)| format!("{v:x}-{p:x}")).collect();
        let label = format!("{:03}-{}d-{}", snap.passes, snap.draws, programs.join("+"));
        snap.programs.clear();
        snap.draws = 0;
        if snap.passes < snap.from {
            return;
        }
        let Some(blit) = command.blitCommandEncoder() else { return };
        let copy_desc = |t: &Texture, format: MTLPixelFormat| {
            let d = unsafe { MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format, t.width(), t.height(), false) };
            d.setUsage(MTLTextureUsage::ShaderRead);
            d.setStorageMode(MTLStorageMode::Private);
            d
        };
        for (i, t) in colors.iter().enumerate() {
            let size = (t.width() * t.height() * texel_bytes(t.pixelFormat()).unwrap_or(16)) as u64;
            if snap.bytes + size > SNAP_BYTES || is_compressed(t.pixelFormat()) {
                continue;
            }
            let Some(copy) = self.device.newTextureWithDescriptor(&copy_desc(t, t.pixelFormat())) else { continue };
            let slice = (addresses[i] >> 48) as usize;
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    t, slice, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: t.width(), height: t.height(), depth: 1 }, &copy, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            snap.bytes += size;
            snap.copies.push((format!("{label}-mrt{i}-{:x}-s{slice}", addresses[i] & 0xFFFF_FFFF_FFFF), copy));
        }
        if let Some((t, slice)) = depth {
            let (w, h) = (t.width(), t.height());
            let size = (w * h * 4) as u64;
            let staging = self.device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModePrivate);
            if let (true, Some(staging), Some(copy)) =
                (snap.bytes + 2 * size <= SNAP_BYTES, staging, self.device.newTextureWithDescriptor(&copy_desc(&t, MTLPixelFormat::R32Float)))
            {
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
                    snap.copies.push((format!("{label}-depth{slice}-{depth_address:x}"), copy));
                }
            }
        }
        blit.endEncoding();
    }

    /// Writes what `gpusnap` copied during the frame that just finished.
    fn write_snap(snap: Snap) {
        let n = snap.copies.len();
        for (name, t) in snap.copies {
            let shared = SharedTexture(t);
            let Some((w, h, px)) = (if shared.0.pixelFormat() == MTLPixelFormat::R32Float { read_depth(&shared.0) } else { read_rgba(&shared) }) else {
                continue;
            };
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            ppm.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
            let _ = std::fs::write(snap.dir.join(format!("{name}.ppm")), ppm);
            if std::env::var_os("PS5SLOP_SNAP_RAW").is_some()
                && let Some(raw) = read_raw(&shared.0)
            {
                let _ = std::fs::write(snap.dir.join(format!("{name}-{}x{}-{}.raw", w, h, shared.0.pixelFormat().0)), raw);
            }
        }
        eprintln!("ps5slop: GPU: gpusnap wrote {n} attachments of {} passes to {}", snap.passes, snap.dir.display());
    }

    /// The no-copy buffer holding `[addr, addr + len)` and the offset of `addr` in it.
    fn window(&mut self, host: &dyn Host, addr: u64, len: u64) -> Option<(Retained<Buffer>, usize)> {
        let generation = host.generation();
        if generation != self.generation {
            self.generation = generation;
            let (next, replaced) = host.replaced_since(self.replaced_mark);
            self.replaced_mark = next;
            let dropped: Vec<Window> = match replaced {
                Some(ranges) => {
                    let (gone, kept) = std::mem::take(&mut self.windows)
                        .into_iter()
                        .partition(|w| ranges.iter().any(|&(s, e)| s < w.end && w.start < e));
                    self.windows = kept;
                    if self.area.as_ref().is_some_and(|w| ranges.iter().any(|&(s, e)| s < w.end && w.start < e)) {
                        self.area = None;
                    }
                    gone
                }
                None => {
                    self.flush_and_wait();
                    self.area = None;
                    std::mem::take(&mut self.windows)
                }
            };
            if let Some(r) = &self.resident
                && !dropped.is_empty()
            {
                for w in &dropped {
                    r.removeAllocation(ProtocolObject::from_ref(&*w.buffer));
                }
                r.commit();
            }
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
        if stop - start > 512 << 20 {
            self.report(format!("guest memory window of {} MiB at {start:#x} for {len:#x} bytes at {addr:#x}", (stop - start) >> 20));
        }
        let ptr = NonNull::new(host.host_pointer(start)?)?;
        // SAFETY: the range is host memory that stays mapped while the address
        // space lives; windows are dropped when its mappings change.
        let buffer = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr.cast(),
                (stop - start) as usize,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }?;
        let (inside, kept) = std::mem::take(&mut self.windows)
            .into_iter()
            .partition::<Vec<_>, _>(|w| start <= w.start && w.end <= stop);
        self.windows = kept;
        if let Some(r) = &self.resident {
            for w in &inside {
                r.removeAllocation(ProtocolObject::from_ref(&*w.buffer));
            }
            r.addAllocation(ProtocolObject::from_ref(&*buffer));
            r.commit();
        }
        self.windows.push(Window {
            start,
            end: stop,
            buffer: buffer.clone(),
        });
        Some((buffer, (addr - start) as usize))
    }

    fn detiler(&mut self, mode: TileMode, bpe: u32, output: Detile) -> Option<Retained<ComputePipeline>> {
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
        let same = if ty == "uint" { "v == w" } else { "all(v == w)" };
        let (args, store) = match output {
            Detile::Buffer => (format!("device {ty}* dst [[buffer(2)]]"), "dst[id.y * p.x + id.x] = v;".to_string()),
            Detile::Texture => ("texture2d<uint, access::write> dst [[texture(0)]]".to_string(), format!("dst.write({width}, id);")),
            Detile::Changed => (
                "device const uchar* old [[buffer(2)]], texture2d<uint, access::write> dst [[texture(0)]]".to_string(),
                format!("{ty} w = {};\n    if ({same}) return;\n    dst.write({width}, id);", load.replace("src", "old")),
            ),
        };
        let msl = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void detile(device const uchar* src [[buffer(0)]], constant uint4& p [[buffer(1)]], {args}, uint2 id [[thread_position_in_grid]]) {{\n    if (id.x >= p.x || id.y >= p.y) return;\n    ulong o = surface_offset(id.x, id.y, p.z);\n    {ty} v = {load};\n    {store}\n}}\n",
            layout.msl_offset()
        );
        let function = self.compile(&msl, "detile")?;
        let pipeline = self.device.newComputePipelineStateWithFunction_error(&function).ok()?;
        self.detilers.insert((mode, bpe, output), pipeline.clone());
        Some(pipeline)
    }

    fn compile(&mut self, msl: &str, entry: &str) -> Option<Retained<Function>> {
        compile(&self.device, msl, entry)
    }

    /// Compiles `msl` on the compile threads.
    fn compile_later(&self, msl: String, entry: String) -> Arc<Task<Retained<Function>>> {
        let device = self.device.clone();
        Task::spawn(move || compile(&device, &msl, &entry))
    }

    /// Copies a tiled guest surface into slice `slice` of `texture` on the GPU.
    fn detile_into(&mut self, host: &dyn Host, texture: &Texture, slice: usize, address: u64, layout: Layout) -> bool {
        let Some((buffer, offset)) = self.window(host, address, layout.size()) else { return false };
        self.encode_detile(texture, slice, layout, (&buffer, offset), None, address)
    }

    /// Copies into slice `slice` of `texture` only the texels whose bytes in
    /// the tiled surface `new` differ from those in `old`.
    fn detile_changes(&mut self, texture: &Texture, slice: usize, layout: Layout, new: &[u8], old: &[u8], address: u64) -> bool {
        let bytes = |b: &[u8]| {
            let ptr = NonNull::new(b.as_ptr() as *mut u8)?.cast();
            unsafe { self.device.newBufferWithBytes_length_options(ptr, b.len(), MTLResourceOptions::StorageModeShared) }
        };
        let (Some(new), Some(old)) = (bytes(new), bytes(old)) else { return false };
        self.encode_detile(texture, slice, layout, (&new, 0), Some(&old), address)
    }

    fn encode_detile(&mut self, texture: &Texture, slice: usize, layout: Layout, src: (&Buffer, usize), old: Option<&Buffer>, address: u64) -> bool {
        let output = if old.is_some() { Detile::Changed } else { Detile::Texture };
        let Some(pipeline) = self.detiler(layout.mode, layout.bpe, output) else { return false };
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
                objc2_foundation::NSRange::new(0, 1),
                objc2_foundation::NSRange::new(slice, 1),
            )
        };
        let Some(raw) = raw else { return false };
        let command = self.upload_command();
        let Some(enc) = command.computeCommandEncoder() else { return false };
        if crate::timing::labels() {
            enc.setLabel(Some(&NSString::from_str(&format!("detile {address:#x} {}x{}", layout.width, layout.height))));
        }
        let params = [layout.width, layout.height, layout.pitch, 0];
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

    /// A 2D array or cube texture to fill from guest memory.
    fn new_layered_texture(&self, format: MTLPixelFormat, width: u32, height: u32, kind: TextureKind, slices: usize) -> Option<Retained<Texture>> {
        let desc = MTLTextureDescriptor::new();
        desc.setPixelFormat(format);
        unsafe {
            desc.setWidth(width as usize);
            desc.setHeight(height as usize);
            if kind == TextureKind::Cube {
                desc.setTextureType(MTLTextureType::TypeCube);
            } else {
                desc.setTextureType(MTLTextureType::Type2DArray);
                desc.setArrayLength(slices.max(1));
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
    fn new_target_array(&self, format: MTLPixelFormat, width: u32, height: u32, slices: u32) -> Option<Retained<Texture>> {
        let desc = unsafe { MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format, width as usize, height as usize, false) };
        desc.setTextureType(MTLTextureType::Type2DArray);
        // SAFETY: at most MAX_TARGET_SLICES, below Metal's array limit.
        unsafe { desc.setArrayLength(slices as usize) };
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite | MTLTextureUsage::PixelFormatView | MTLTextureUsage::RenderTarget);
        desc.setStorageMode(MTLStorageMode::Private);
        self.device.newTextureWithDescriptor(&desc)
    }

    /// Fills block-compressed `texture` from a tiled guest surface of 4x4
    /// blocks: detiled into a linear buffer, then blitted.
    fn detile_blocks_into(&mut self, host: &dyn Host, texture: &Texture, slice: usize, address: u64, layout: Layout) -> bool {
        let Some(pipeline) = self.detiler(layout.mode, layout.bpe, Detile::Buffer) else { return false };
        let Some((buffer, offset)) = self.window(host, address, layout.size()) else { return false };
        let row = (layout.width * layout.bpe) as usize;
        let len = row * layout.height as usize;
        let Some(staging) = self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate) else {
            return false;
        };
        let command = self.upload_command();
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
                slice,
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
        self.color_newer.insert(t.address);
        let slices = t.slices.max(t.slice + 1).min(MAX_TARGET_SLICES);
        let layered = slices > 1;
        if let Some(existing) = self.targets.get(&t.address)
            && existing.pixel_format == format
            && existing.texture.width() == t.width as usize
            && existing.texture.height() == t.height as usize
            && (existing.texture.textureType() == MTLTextureType::Type2DArray) == layered
            && existing.texture.arrayLength() >= slices as usize
        {
            record(t.address | 2, &existing.texture, false);
            return Some((existing.texture.clone(), format));
        }
        if trace_targets() {
            eprintln!(
                "ps5slop: targets: frame {} new target {:#x} {}x{} {format:?}{}",
                self.epoch,
                t.address,
                t.width,
                t.height,
                if self.targets.contains_key(&t.address) { " (replacing)" } else { "" }
            );
        }
        let texture = if layered {
            self.new_target_array(format, t.width, t.height, slices)?
        } else {
            self.new_texture(format, t.width, t.height, true)?
        };
        if let (Some(mode), Some(bpe)) = (t.tile, t.format.bytes())
            && let Some(layout) = Layout::new(mode, bpe, t.width, t.height)
        {
            let at = t.address + layout.mip0_offset(t.max_mip + 1);
            let stride = layout.size().next_multiple_of(64 << 10);
            for slice in 0..slices as usize {
                self.detile_into(host, &texture, slice, at + slice as u64 * stride, layout);
            }
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

    /// The texture for image descriptor `image` read as `kind`: a render
    /// target at that address, or a copy of guest memory refreshed once per frame.
    fn image(&mut self, host: &dyn Host, image: Image, kind: TextureKind) -> Option<Retained<Texture>> {
        let (format, bpe) = image_format(image.format)?;
        let depth_current = !self.color_newer.contains(&image.address);
        if kind == TextureKind::Cube
            && depth_current
            && let Some(d) = self.depths.get(&image.address).cloned()
            && d.width() as u32 == image.width
            && d.height() as u32 == image.height
        {
            let base = image.base_array as usize;
            if d.textureType() != MTLTextureType::Type2DArray || d.arrayLength() < base + 6 {
                return None;
            }
            self.note(|| format!("      {:#x} a cube of a layered depth target", image.address));
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
            let count = if array { (image.depth as usize).min(available) } else { 1 };
            if count == 0 {
                return None;
            }
            if d.textureType() == if array { MTLTextureType::Type2DArray } else { MTLTextureType::Type2D }
                && image.base_array == 0
                && count == d.arrayLength()
            {
                return Some(d.clone());
            }
            let ty = if array { MTLTextureType::Type2DArray } else { MTLTextureType::Type2D };
            return unsafe {
                d.newTextureViewWithPixelFormat_textureType_levels_slices(
                    d.pixelFormat(),
                    ty,
                    objc2_foundation::NSRange::new(0, 1),
                    objc2_foundation::NSRange::new(image.base_array as usize, count),
                )
            };
        }
        self.refresh_volume(host, image.address, false);
        if let Some(v) = self.volume(image, kind, format) {
            self.note(|| format!("      {:#x} from a 2D array or 3D image", image.address));
            return Some(v);
        }
        if image.tile.is_none() && is_depth_layout(image.raw_tile) && kind == TextureKind::Tex2d {
            return self.blank_texture(image, kind, format);
        }
        if kind == TextureKind::Tex3d
            && let Some(t) = self.standard_volume(host, image, format, bpe)
        {
            return Some(t);
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
                TextureKind::Tex2dArray if base < len => (MTLTextureType::Type2DArray, (image.depth as usize).clamp(1, len - base)),
                TextureKind::Cube if base + 6 <= len => (MTLTextureType::TypeCube, 6),
                _ => return None,
            };
            if have != format && !view_compatible(have, format) {
                self.report(format!("render target at {:#x} ({have:?}) cannot be viewed as {format:?}", image.address));
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
        {
            let tex = &t.texture;
            if tex.width() as u32 == image.width && tex.height() as u32 == image.height {
                // Memory reused for another surface: read guest memory below.
                if t.pixel_format == format || view_compatible(t.pixel_format, format) {
                    let found = if t.pixel_format == format { Some(tex.clone()) } else { tex.newTextureViewWithPixelFormat(format) };
                    let ptr = Retained::as_ptr(tex) as usize;
                    self.note(|| format!("      {:#x} from a render target {ptr:#x}", image.address));
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
        self.note(|| format!("      {:#x} detiled from guest memory", image.address));
        let compressed = is_compressed(format);
        let (ew, eh) = if compressed {
            (image.width.div_ceil(4), image.height.div_ceil(4))
        } else {
            (image.width, image.height)
        };
        let layout = Layout::new(image.tile?, bpe, ew, eh)?;
        if trace_targets() && !self.images.contains_key(&(image, kind)) {
            eprintln!("ps5slop: targets: frame {} image {:#x} {}x{} from guest memory", self.epoch, image.address, image.width, image.height);
        }
        let tex = match self.images.get(&(image, kind)) {
            Some((_, t)) => t.clone(),
            None if kind == TextureKind::Tex2d => self.new_texture(format, image.width, image.height, false)?,
            None => self.new_layered_texture(format, image.width, image.height, kind, slices)?,
        };
        let base = image.address + layout.mip0_offset(image.last_level + 1);
        // A 2D image bound where a cube is read shows its one slice on every face.
        let stride = if kind == TextureKind::Cube && image.kind != 11 { 0 } else { layout.size().next_multiple_of(layout.block_bytes()) };
        for slice in 0..slices {
            let at = base + slice as u64 * stride;
            let filled = if compressed {
                self.detile_blocks_into(host, &tex, slice, at, layout)
            } else {
                self.detile_into(host, &tex, slice, at, layout)
            };
            if !filled {
                return None;
            }
        }
        self.images.insert((image, kind), (self.epoch, tex.clone()));
        record(image.address | 1, &tex, false);
        Some(tex)
    }

    /// A 2D image bound where a 3D texture is read, as a one-slice 3D texture.
    fn flat_volume(&mut self, host: &dyn Host, image: Image, format: MTLPixelFormat) -> Option<Retained<Texture>> {
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
        self.images.insert((image, TextureKind::Tex3d), (self.epoch, volume.clone()));
        Some(volume)
    }

    /// A 3D image of one-byte texels in `SW_4KB_S` that fits one 4 KiB
    /// block (16x16x16), or of one texel of any size, read from guest
    /// memory once a frame. Address bits
    /// are x0 x1 z0 y0 z1 y1 x2 z2 y2 x3 z3 y3: the z bits were confirmed on
    /// Superliminal's 4x4x16 dither mask (its slices hold 0 to 16 set
    /// texels in order), the x and y order is assumed.
    fn standard_volume(&mut self, host: &dyn Host, image: Image, format: MTLPixelFormat, bpe: u32) -> Option<Retained<Texture>> {
        const BITS: [(usize, u32); 12] = [(0, 0), (0, 1), (2, 0), (1, 0), (2, 1), (1, 1), (0, 2), (2, 2), (1, 2), (0, 3), (2, 3), (1, 3)];
        let (w, h, d) = (image.width as usize, image.height as usize, image.depth as usize);
        let one = w == 1 && h == 1 && d == 1;
        if image.kind != 10 || image.last_level != 0 || !one && (image.raw_tile != 5 || bpe != 1 || w > 16 || h > 16 || d > 16) {
            return None;
        }
        if let Some((epoch, tex)) = self.images.get(&(image, TextureKind::Tex3d))
            && *epoch == self.epoch
        {
            return Some(tex.clone());
        }
        let mut block = vec![0u8; if one { bpe as usize } else { 4096 }];
        if !host.read(image.address, &mut block) {
            return None;
        }
        let mut linear = if one { block.clone() } else { vec![0u8; w * h * d] };
        for z in (0..d).filter(|_| !one) {
            for y in 0..h {
                for x in 0..w {
                    let at = BITS.iter().enumerate().fold(0, |a, (bit, &(axis, n))| a | ([x, y, z][axis] >> n & 1) << bit);
                    linear[(z * h + y) * w + x] = block[at];
                }
            }
        }
        let tex = match self.images.get(&(image, TextureKind::Tex3d)) {
            Some((_, t)) => t.clone(),
            None => {
                let desc = MTLTextureDescriptor::new();
                desc.setPixelFormat(format);
                desc.setTextureType(MTLTextureType::Type3D);
                unsafe {
                    desc.setWidth(w);
                    desc.setHeight(h);
                    desc.setDepth(d);
                }
                desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
                desc.setStorageMode(MTLStorageMode::Shared);
                self.device.newTextureWithDescriptor(&desc)?
            }
        };
        unsafe {
            tex.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: w, height: h, depth: d } },
                0,
                0,
                NonNull::new(linear.as_mut_ptr())?.cast(),
                w * bpe as usize,
                w * h * bpe as usize,
            );
        }
        self.note(|| format!("      {:#x} a one-block standard 3D image", image.address));
        self.images.insert((image, TextureKind::Tex3d), (self.epoch, tex.clone()));
        Some(tex)
    }

    /// `tex` with an image descriptor's channel selects (`DST_SEL_X..W`) applied.
    fn swizzled(&mut self, tex: Retained<Texture>, sel: [u8; 4]) -> Option<Retained<Texture>> {
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
                objc2_foundation::NSRange::new(0, if tex.textureType() == MTLTextureType::TypeCube { 6 } else { tex.arrayLength() }),
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
    fn null_texture(&mut self, kind: TextureKind) -> Option<Retained<Texture>> {
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
                    MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: 1, height: 1, depth: 1 } },
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

    /// The program at `address` as the bound descriptors need it: translated
    /// again with integer texture types when a T# has an integer format.
    #[allow(clippy::too_many_arguments)]
    fn typed_program(
        &mut self,
        host: &dyn Host,
        state: &State,
        stage: Stage,
        address: u64,
        user_sgprs: u32,
        ngg: bool,
        col_format: u32,
        user: &[u32],
    ) -> Option<Arc<Program>> {
        let base = Variant { col_format, uint_textures: 0, sint_textures: 0 };
        let program = self.program(host, state, stage, address, user_sgprs, ngg, base)?;
        let mut variant = base;
        for r in &program.translated.resources {
            if let Resource::Texture { slot, ref source, .. } = *r
                && let Some(d) = self.descriptor(host, user, source, 8)
            {
                let image = Image::read(&d);
                if image.address == 0 || image.kind < 8 || self.depths.contains_key(&image.address) {
                    continue;
                }
                match image_format(image.format).map(|f| number(f.0)) {
                    Some(Number::Uint) => variant.uint_textures |= 1 << slot,
                    Some(Number::Sint) => variant.sint_textures |= 1 << slot,
                    _ => {}
                }
            }
        }
        if variant == base { Some(program) } else { self.program(host, state, stage, address, user_sgprs, ngg, variant) }
    }

    /// The translated program at `address`, translated on first use.
    #[allow(clippy::too_many_arguments)]
    fn program(&mut self, host: &dyn Host, state: &State, stage: Stage, address: u64, user_sgprs: u32, ngg: bool, variant: Variant) -> Option<Arc<Program>> {
        let c = state.context;
        let cntl = &c[context::SPI_PS_INPUT_CNTL_0..context::SPI_PS_INPUT_CNTL_0 + 32];
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (stage as u8, user_sgprs, ngg, variant).hash(&mut h);
            if stage == Stage::Pixel {
                (cntl, c[context::SPI_PS_INPUT_ENA], c[context::SPI_PS_INPUT_ADDR]).hash(&mut h);
            }
            if stage == Stage::Compute {
                let s = &state.sh;
                (s[sh::COMPUTE_PGM_RSRC2], &s[sh::COMPUTE_NUM_THREAD_X..sh::COMPUTE_NUM_THREAD_X + 3]).hash(&mut h);
            }
            // Titles load other programs where freed ones were.
            let mut head = [0u8; 256];
            host.read(address, &mut head);
            head.hash(&mut h);
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
                address,
                user_sgprs,
                ngg,
                ps_input_ena: c[context::SPI_PS_INPUT_ENA],
                ps_input_addr: c[context::SPI_PS_INPUT_ADDR],
                ps_input_cntl: cntl,
                col_format: variant.col_format,
                compute_rsrc2: state.sh[sh::COMPUTE_PGM_RSRC2],
                threads: [0, 1, 2].map(|i| (state.sh[sh::COMPUTE_NUM_THREAD_X + i] & 0xFFFF).max(1)),
                uint_textures: variant.uint_textures,
                sint_textures: variant.sint_textures,
            };
            if let Some(dir) = std::env::var_os("PS5SLOP_SHADER_DUMP") {
                let path = std::path::Path::new(&dir).join(format!("{stage:?}-{address:x}-{key:016x}.bin"));
                let _ = std::fs::write(path, gfx10::capture::Capture::new(&request).to_bytes());
            }
            match shader::translate(&request) {
                Ok(mut translated) => {
                    if let Some(dir) = std::env::var_os("PS5SLOP_MSL_OVERRIDE")
                        && let Ok(msl) = std::fs::read_to_string(std::path::Path::new(&dir).join(format!("{stage:?}-{address:x}.metal")))
                    {
                        eprintln!("ps5slop: GPU: {stage:?} program at {address:#x} replaced from {}", dir.to_string_lossy());
                        translated.msl = msl;
                    }
                    if std::env::var_os("PS5SLOP_MSL_DUMP").is_some() {
                        eprintln!("=== {stage:?} {address:#x}\n{}", translated.msl);
                    }
                    let function = self.compile_later(translated.msl.clone(), translated.entry.clone());
                    let compute = (stage == Stage::Compute).then(|| {
                        let (device, function) = (self.device.clone(), function.clone());
                        Task::spawn(move || {
                            let function = function.wait()?;
                            let t = crate::timing::start();
                            let p = device.newComputePipelineStateWithFunction_error(&function);
                            crate::timing::add(crate::timing::Part::Compile, t);
                            p.map_err(|e| late_report(format!("compute pipeline failed: {}", e.localizedDescription()))).ok()
                        })
                    });
                    Some(Arc::new(Program {
                        address,
                        id: {
                            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        },
                        function,
                        compute,
                        flat: flat_locations(&translated.msl),
                        sint_textures: variant.sint_textures,
                        translated,
                    }))
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

    // Metal rejects pixel inputs the vertex function lacks; hardware reads
    // zeros, so variants declare them.
    fn vertex_function(&mut self, vertex: &Program, pixel: &Program, rect: bool, gl_clip: bool) -> Arc<Task<Retained<Function>>> {
        let locations = |msl: &str| -> Vec<u32> {
            let mut v: Vec<u32> = msl
                .match_indices("[[user(locn")
                .filter_map(|(i, m)| msl[i + m.len()..].split(|c: char| !c.is_ascii_digit()).next()?.parse().ok())
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let missing = match self.missing_inputs.get(&(vertex.id, pixel.id)) {
            Some(m) => m.clone(),
            None => {
                let have = locations(&vertex.translated.msl);
                let m: Vec<u32> = locations(&pixel.translated.msl).into_iter().filter(|l| !have.contains(l)).collect();
                self.missing_inputs.insert((vertex.id, pixel.id), m.clone());
                m
            }
        };
        let flat: Vec<u32> = if rect { Vec::new() } else { pixel.flat.clone() };
        if missing.is_empty() && flat.is_empty() && !rect && !gl_clip {
            return vertex.function.clone();
        }
        let key = (vertex.id, rect, gl_clip, missing.clone(), flat.clone());
        if let Some(f) = self.variants.get(&key) {
            return f.clone();
        }
        let (device, source, entry) = (self.device.clone(), vertex.translated.msl.clone(), vertex.translated.entry.clone());
        let f = Task::spawn(move || {
            let extra: String = missing
                .iter()
                .map(|l| format!("    {} x{l} [[user(locn{l})]];\n", if flat.contains(l) { "uint4" } else { "float4" }))
                .collect();
            let mut msl = source.replacen("    float4 pos [[position]];\n", &format!("    float4 pos [[position]];\n{extra}"), 1);
            for l in &flat {
                msl = uint_output(&msl, *l);
            }
            if gl_clip {
                msl = gl_clip_space(&msl);
            }
            if rect {
                rect_source(&msl).and_then(|m| compile(&device, &m, "gen5_vs_rect"))
            } else {
                compile(&device, &msl, &entry)
            }
        });
        self.variants.insert(key, f.clone());
        f
    }

    /// The render pipeline for `key`, compiled on the compile threads once both functions are.
    fn pipeline(&mut self, key: &PipelineKey, vertex: Arc<Task<Retained<Function>>>, pixel: &Program) -> Arc<Task<Retained<RenderPipeline>>> {
        if let Some(p) = self.pipelines.get(key) {
            return p.clone();
        }
        let (device, k, pixel) = (self.device.clone(), key.clone(), pixel.function.clone());
        let task = Task::spawn(move || {
            let key = &k;
            let (vertex, pixel) = (vertex.wait()?, pixel.wait()?);
            let desc = MTLRenderPipelineDescriptor::new();
            desc.setVertexFunction(Some(&vertex));
            desc.setFragmentFunction(Some(&pixel));
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
            let t = crate::timing::start();
            let made = device.newRenderPipelineStateWithDescriptor_error(&desc);
            crate::timing::add(crate::timing::Part::Compile, t);
            made.map_err(|e| late_report(format!("render pipeline failed: {}", e.localizedDescription()))).ok()
        });
        self.pipelines.insert(key.clone(), task.clone());
        task
    }

    /// The render encoder for color targets `targets`, reusing the open pass when it matches.
    fn pass(
        &mut self,
        targets: &[(u64, Retained<Texture>, u32)],
        depth: Option<(u64, Retained<Texture>, bool, u32)>,
        clear: Option<(f32, u32)>,
    ) -> Option<Retained<RenderEncoder>> {
        let mut addresses: Vec<u64> = targets.iter().map(|t| t.0 | (t.2 as u64) << 48).collect();
        addresses.push(depth.as_ref().map_or(0, |d| d.0 | (d.3 as u64) << 48));
        if let Some((enc, open, _, _)) = &self.pass
            && *open == addresses
            && clear.is_none()
        {
            return Some(enc.clone());
        }
        self.end_pass();
        let command = self.command();
        let pass = MTLRenderPassDescriptor::new();
        for (i, (_, tex, slice)) in targets.iter().enumerate() {
            let att = unsafe { pass.colorAttachments().objectAtIndexedSubscript(i) };
            att.setTexture(Some(tex));
            att.setSlice(*slice as usize);
            att.setLoadAction(MTLLoadAction::Load);
            att.setStoreAction(MTLStoreAction::Store);
        }
        if let Some((_, tex, stencil, slice)) = &depth {
            let d = pass.depthAttachment();
            d.setTexture(Some(tex));
            d.setSlice(*slice as usize);
            d.setStoreAction(MTLStoreAction::Store);
            let st = pass.stencilAttachment();
            if *stencil {
                st.setTexture(Some(tex));
                st.setSlice(*slice as usize);
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
        if let Some(f) = self.wait_memory() {
            enc.waitForFence_beforeStages(f, MTLRenderStages::Vertex);
        }
        if crate::timing::labels() {
            let what: Vec<String> = addresses.iter().map(|a| format!("{a:#x}")).collect();
            enc.setLabel(Some(&NSString::from_str(&format!("pass {}", what.join(" ")))));
        }
        let colors = targets.iter().map(|t| t.1.clone()).collect();
        let depth_texture = depth.as_ref().map(|(_, t, _, slice)| (t.clone(), *slice as usize));
        self.pass = Some((enc.clone(), addresses, colors, depth_texture));
        Some(enc)
    }

    /// The depth texture at `d`'s address, holding what a program last
    /// stored to that address as an image (Minecraft copies its scene depth
    /// that way before drawing water); new ones start cleared.
    fn depth(&mut self, d: &DepthTarget, width: u32, height: u32) -> Option<(Retained<Texture>, MTLPixelFormat, bool)> {
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
            .filter(|s| s.pixelFormat() == MTLPixelFormat::R32Float && s.width() == t.width() && s.height() == t.height());
        let Some(src) = stored.filter(|_| format != MTLPixelFormat::Depth16Unorm) else {
            return Some((t, format, fresh));
        };
        let (w, h) = (t.width(), t.height());
        let staging = self.device.newBufferWithLength_options(w * h * 4, MTLResourceOptions::StorageModePrivate)?;
        self.end_pass();
        let command = self.command();
        let blit = command.blitCommandEncoder()?;
        let options = if format == MTLPixelFormat::Depth32Float_Stencil8 { MTLBlitOption::DepthFromDepthStencil } else { MTLBlitOption::empty() };
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &src, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &staging, 0, w * 4, w * h * 4,
            );
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                &staging, 0, w * 4, w * h * 4, MTLSize { width: w, height: h, depth: 1 }, &t, d.slice as usize, 0, MTLOrigin { x: 0, y: 0, z: 0 }, options,
            );
        }
        blit.endEncoding();
        self.note(|| format!("      depth {:#x} filled from the image a program stored there", d.address));
        Some((t, format, false))
    }

    /// Whether `d`'s slice was fast-cleared through its HTILE since we last
    /// looked: titles clear depth by filling HTILE with a compute program,
    /// and a tile whose ZMask (bits 0 to 3) is 0 holds the clear value
    /// (Mesa's radv writes HTILE this way). Only the slice's first HTILE
    /// dword is checked, and it is then marked expanded (ZMask 0xF), as
    /// drawing would leave it. HTILE slices are taken to be 4 bytes per
    /// 8x8 tile rounded up to 64 KiB (seen for 1024x1024 cube faces).
    fn htile_cleared(&mut self, host: &dyn Host, c: &[u32], d: &DepthTarget) -> bool {
        const TILE_SURFACE_ENABLE: u32 = 1 << 29;
        if c[context::DB_Z_INFO] & TILE_SURFACE_ENABLE == 0 {
            return false;
        }
        let base = ((c[context::DB_HTILE_DATA_BASE] as u64) | ((c[context::DB_HTILE_DATA_BASE_HI] as u64 & 0xFF) << 32)) << 8;
        if base == 0 {
            return false;
        }
        let (w, h) = if d.width != 0 { (d.width, d.height) } else { (1, 1) };
        let slice_bytes = (w.div_ceil(8) as u64 * h.div_ceil(8) as u64 * 4).next_multiple_of(64 << 10);
        let at = base + d.slice as u64 * slice_bytes;
        self.settle();
        let mut word = [0u8; 4];
        if !host.read(at, &mut word) || u32::from_le_bytes(word) & 0xF != 0 {
            return false;
        }
        host.write(at, &(u32::from_le_bytes(word) | 0xF).to_le_bytes());
        self.note(|| format!("      depth {:#x} slice {} fast-cleared through HTILE at {at:#x}", d.address, d.slice));
        true
    }

    /// Applies a colour fast clear made through `t`'s CMASK or DCC, judged by
    /// its first dword, which is then marked expanded (see gpu.md).
    fn fast_clear(&mut self, host: &dyn Host, t: &ColorTarget, tex: &Texture) {
        let mut word = [0u8; 4];
        if let Some((cmask, words)) = t.fast_clear {
            self.settle();
            if host.read(cmask, &mut word) && u32::from_le_bytes(word) & 0xF == 0 {
                host.write(cmask, &(u32::from_le_bytes(word) | 0xF).to_le_bytes());
                self.fill_raw(t, tex, words);
                self.note(|| format!("      color {:#x} fast-cleared through CMASK at {cmask:#x} to {words:x?}", t.address));
            }
        }
        if let Some(dcc) = t.dcc {
            self.settle();
            if !host.read(dcc, &mut word) {
                return;
            }
            let code = u32::from_le_bytes(word);
            let colour = match code {
                0x0000_0000 => [0.0, 0.0, 0.0, 0.0],
                0x4040_4040 => [0.0, 0.0, 0.0, 1.0],
                0x8080_8080 => [1.0, 1.0, 1.0, 0.0],
                0xC0C0_C0C0 => [1.0, 1.0, 1.0, 1.0],
                0x2020_2020 => [f64::NAN; 4],
                _ => return,
            };
            host.write(dcc, &u32::MAX.to_le_bytes());
            if colour[0].is_nan() {
                let words = t.fast_clear.map_or([0; 2], |f| f.1);
                self.fill_raw(t, tex, words);
            } else {
                self.fill_colour(t, tex, colour);
            }
            self.note(|| format!("      color {:#x} fast-cleared through DCC at {dcc:#x} with code {code:#x}", t.address));
        }
    }

    fn target_slice(t: &ColorTarget, tex: &Texture) -> usize {
        if tex.textureType() == MTLTextureType::Type2DArray { t.slice as usize } else { 0 }
    }

    /// Fills `tex` with the raw bits of `words` per texel (up to 8 bytes).
    fn fill_raw(&mut self, t: &ColorTarget, tex: &Texture, words: [u32; 2]) {
        let Some(bpp) = texel_bytes(tex.pixelFormat()).filter(|&b| b <= 8) else { return };
        let (w, h) = (tex.width(), tex.height());
        let pixel: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).take(bpp).collect();
        let Some((buf, offset)) = self.scratch(w * h * bpp) else { return };
        // SAFETY: the scratch range is fresh shared memory of w * h * bpp bytes.
        let dst = unsafe { std::slice::from_raw_parts_mut(buf.contents().as_ptr().cast::<u8>().add(offset), w * h * bpp) };
        for p in dst.chunks_exact_mut(bpp) {
            p.copy_from_slice(&pixel);
        }
        self.end_pass();
        let command = self.command();
        let Some(blit) = command.blitCommandEncoder() else { return };
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
        att.setClearColor(MTLClearColor { red: colour[0], green: colour[1], blue: colour[2], alpha: colour[3] });
        if let Some(enc) = command.renderCommandEncoderWithDescriptor(&pass) {
            enc.endEncoding();
        }
    }

    /// The depth texture at `d`'s address; new ones start cleared.
    fn depth_texture(&mut self, d: &DepthTarget, width: u32, height: u32) -> Option<(Retained<Texture>, MTLPixelFormat, bool)> {
        let format = match (d.format, d.stencil) {
            (_, true) => MTLPixelFormat::Depth32Float_Stencil8,
            (1, false) => MTLPixelFormat::Depth16Unorm,
            _ => MTLPixelFormat::Depth32Float,
        };
        let (w, h) = if d.width != 0 { (d.width, d.height) } else { (width, height) };
        if d.slice >= MAX_DEPTH_SLICES as u32 {
            self.report(format!("depth target slice {} at {:#x} not supported", d.slice, d.address));
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
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format, w as usize, h as usize, false)
        };
        if slices > 1 {
            desc.setTextureType(MTLTextureType::Type2DArray);
            // SAFETY: a small positive count, below Metal's array limit.
            unsafe { desc.setArrayLength(slices) };
        }
        desc.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
        desc.setStorageMode(MTLStorageMode::Private);
        let t = self.device.newTextureWithDescriptor(&desc)?;
        let kept = old.filter(|o| o.pixelFormat() == format && o.width() == w as usize && o.height() == h as usize);
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
        record(d.address | 4, &t, true);
        Some((t, format, kept.is_none()))
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
    fn descriptor(&self, host: &dyn Host, user: &[u32], source: &Source, n: usize) -> Option<Vec<u32>> {
        match source {
            Source::UserSgpr(s) => user.get(*s as usize..*s as usize + n).map(<[u32]>::to_vec),
            Source::Memory { pointer, offset } => {
                let p = self.descriptor(host, user, pointer, 2)?;
                let ptr = p[0] as u64 | (p[1] as u64) << 32;
                let mut b = vec![0u8; n * 4];
                host.read(ptr.wrapping_add(*offset as u64), &mut b)
                    .then(|| b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
            }
            Source::Buffer { buffer, offset, index } => {
                let v = self.descriptor(host, user, buffer, 4)?;
                let base = v[0] as u64 | (v[1] as u64 & 0xFFFF) << 32;
                let indexed = match *index {
                    Some((ref of, shift)) => self.descriptor(host, user, of, 1)?[0].checked_shl(shift as u32)?,
                    None => 0,
                };
                let mut b = vec![0u8; n * 4];
                host.read(base.wrapping_add(*offset as u64).wrapping_add(indexed as u64), &mut b)
                    .then(|| b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
            }
            Source::Scaled { .. } => None,
            Source::Constant(v) => v.get(..n).map(<[u32]>::to_vec),
        }
    }

    /// The texture a program writes for image descriptor `image`: kept like a
    /// render target, so later reads of that address see what was written.
    fn storage_image(&mut self, host: &dyn Host, image: Image, kind: TextureKind) -> Option<Retained<Texture>> {
        let (format, bpe) = image_format(image.format)?;
        self.color_newer.insert(image.address);
        if is_depth_layout(image.raw_tile) {
            self.depth_stores.insert(image.address);
        }
        if is_compressed(format) {
            return None;
        }
        let layered = matches!(image.kind, 11 | 13) || self.volumes.contains_key(&image.address);
        match kind {
            TextureKind::Tex2d if !layered => {}
            TextureKind::Cube => return None,
            _ => {
                self.refresh_volume(host, image.address, true);
                if let Some(v) = self.volume(image, kind, format) {
                    return Some(v);
                }
                let slices = image.depth.max(image.base_array + 1);
                let old = self.volumes.get(&image.address).cloned();
                let texture = self.new_volume(format, image, if kind == TextureKind::Tex3d { kind } else { TextureKind::Tex2dArray }, slices)?;
                let source = if kind == TextureKind::Tex3d { None } else { self.fill_volume(host, &texture, image, bpe, slices) };
                if let Some(old) = old
                    && old.textureType() == MTLTextureType::Type2DArray
                    && texture.textureType() == MTLTextureType::Type2DArray
                    && (old.width(), old.height(), old.pixelFormat()) == (texture.width(), texture.height(), texture.pixelFormat())
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
        {
            if t.pixel_format == format {
                return Some(t.texture.clone());
            }
            if view_compatible(t.pixel_format, format) {
                return t.texture.newTextureViewWithPixelFormat(format);
            }
            // Memory reused for another surface: the old contents are dead.
            let have = t.pixel_format;
            self.note(|| format!("      image at {:#x} ({have:?}) replaced by a {format:?} one", image.address));
            self.targets.remove(&image.address);
        }
        let texture = self.new_texture(format, image.width, image.height, true)?;
        if let Some(mode) = image.tile
            && let Some(layout) = Layout::new(mode, bpe, image.width, image.height)
        {
            let at = image.address + layout.mip0_offset(image.last_level + 1);
            self.detile_into(host, &texture, 0, at, layout);
        }
        self.targets.insert(image.address, Target { texture: texture.clone(), pixel_format: format });
        record(image.address | 2, &texture, true);
        Some(texture)
    }

    /// Fills the slices of a 2D-array `texture` from the guest memory behind
    /// `image`, returning what [`Metal::refresh_volume`] needs to notice later CPU writes there.
    fn fill_volume(&mut self, host: &dyn Host, texture: &Texture, image: Image, bpe: u32, slices: u32) -> Option<VolumeSource> {
        let layout = Layout::new(image.tile?, bpe, image.width, image.height)?;
        let base = image.address + layout.mip0_offset(image.last_level + 1);
        let stride = layout.size().next_multiple_of(layout.block_bytes());
        for slice in 0..slices {
            self.detile_into(host, texture, slice as usize, base + slice as u64 * stride, layout);
        }
        let len = stride * slices as u64;
        let shadow = guest_bytes(host, base, len).map(|b| Arc::new(b.to_vec()));
        Some(VolumeSource { image, bpe, slices, base, len, shadow, epoch: self.epoch })
    }

    /// Copies into the 2D-array image at `address` the texels the CPU changed
    /// in guest memory since the last look (on every write, once a frame for
    /// reads). Texels the GPU wrote stay: those writes never reach guest memory.
    fn refresh_volume(&mut self, host: &dyn Host, address: u64, write: bool) {
        let Some(src) = self.volume_sources.get_mut(&address) else { return };
        if !write && src.epoch == self.epoch {
            return;
        }
        src.epoch = self.epoch;
        let src = src.clone();
        let Some(old) = src.shadow.as_deref() else { return };
        let Some(new) = guest_bytes(host, src.base, src.len) else { return };
        if new == old.as_slice() {
            return;
        }
        let new = new.to_vec();
        let Some(texture) = self.volumes.get(&address).cloned() else { return };
        let Some(layout) = src.image.tile.and_then(|t| Layout::new(t, src.bpe, src.image.width, src.image.height)) else { return };
        if texture.arrayLength() < src.slices as usize {
            return;
        }
        let changed = new.iter().zip(old).filter(|(a, b)| a != b).count();
        self.report(format!("2D array at {address:#x}: {changed} bytes changed in guest memory, copying them in"));
        let stride = (src.len / src.slices as u64) as usize;
        let size = layout.size() as usize;
        for slice in 0..src.slices as usize {
            let range = slice * stride..slice * stride + size;
            if new[range.clone()] != old[range.clone()] {
                self.detile_changes(&texture, slice, layout, &new[range.clone()], &old[range], src.base + (slice * stride) as u64);
            }
        }
        if let Some(s) = self.volume_sources.get_mut(&address) {
            s.shadow = Some(Arc::new(new));
        }
    }

    /// The 2D-array or 3D image programs wrote at `image`'s address, viewed as
    /// `kind` from slice `BASE_ARRAY` on, if it has that shape.
    fn volume(&self, image: Image, kind: TextureKind, format: MTLPixelFormat) -> Option<Retained<Texture>> {
        let v = self.volumes.get(&image.address)?;
        if v.width() as u32 != image.width || v.height() as u32 != image.height {
            return None;
        }
        let (base, have) = (image.base_array as usize, v.arrayLength());
        let (ty, first, count) = match (kind, v.textureType()) {
            (TextureKind::Tex3d, MTLTextureType::Type3D) if v.depth() as u32 == image.depth => (MTLTextureType::Type3D, 0, 1),
            (TextureKind::Tex2d, MTLTextureType::Type2DArray) if base < have => (MTLTextureType::Type2D, base, 1),
            (TextureKind::Tex2dArray, MTLTextureType::Type2DArray) if (image.depth as usize) <= have && base < image.depth as usize => {
                (MTLTextureType::Type2DArray, base, image.depth as usize - base)
            }
            (TextureKind::Cube, MTLTextureType::Type2DArray) if base + 6 <= have => (MTLTextureType::TypeCube, base, 6),
            _ => return None,
        };
        unsafe {
            v.newTextureViewWithPixelFormat_textureType_levels_slices(
                format,
                ty,
                objc2_foundation::NSRange::new(0, 1),
                objc2_foundation::NSRange::new(first, count),
            )
        }
    }

    /// A zero-filled 2D-array (`slices` slices) or 3D image for a program to write.
    fn new_volume(&self, format: MTLPixelFormat, image: Image, kind: TextureKind, slices: u32) -> Option<Retained<Texture>> {
        let desc = MTLTextureDescriptor::new();
        desc.setPixelFormat(format);
        let (w, h, d) = (image.width as usize, image.height as usize, slices.max(1) as usize);
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
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite | MTLTextureUsage::PixelFormatView;
        if atomic_capable(format) {
            usage |= MTLTextureUsage::ShaderAtomic;
        }
        desc.setUsage(usage);
        desc.setStorageMode(MTLStorageMode::Shared);
        let t = self.device.newTextureWithDescriptor(&desc)?;
        let row = w * texel_bytes(format)?;
        let zero = vec![0u8; row * h * if kind == TextureKind::Tex3d { d } else { 1 }];
        let (depth, slices) = if kind == TextureKind::Tex3d { (d, 1) } else { (1, d) };
        for slice in 0..slices {
            unsafe {
                t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: w, height: h, depth } },
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

    /// A zero-filled texture shaped like `image`, cached.
    fn blank_texture(&mut self, image: Image, kind: TextureKind, format: MTLPixelFormat) -> Option<Retained<Texture>> {
        if let Some(t) = self.blank.get(&(image, kind)) {
            return Some(t.clone());
        }
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format, image.width as usize, image.height as usize, false)
        };
        desc.setUsage(MTLTextureUsage::ShaderRead);
        desc.setStorageMode(MTLStorageMode::Shared);
        let t = self.device.newTextureWithDescriptor(&desc)?;
        let row = image.width as usize * texel_bytes(format)?;
        let zero = vec![0u8; row * image.height as usize];
        unsafe {
            t.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize { width: image.width as usize, height: image.height as usize, depth: 1 },
                },
                0,
                NonNull::new(zero.as_ptr() as *mut u8)?.cast(),
                row,
            );
        }
        self.blank.insert((image, kind), t.clone());
        Some(t)
    }

    /// `len` bytes of shared memory no GPU work uses yet, 4-byte aligned:
    /// a buffer and an offset into it.
    fn scratch(&mut self, len: usize) -> Option<(Retained<Buffer>, usize)> {
        const SCRATCH: usize = 4 << 20;
        let need = len.next_multiple_of(4);
        if self.scratch.as_ref().is_none_or(|(b, used)| used + need > b.length()) {
            let buffer = self.device.newBufferWithLength_options(SCRATCH.max(need), MTLResourceOptions::StorageModeShared)?;
            self.scratch = Some((buffer, 0));
        }
        let (buffer, used) = self.scratch.as_mut()?;
        let at = *used;
        *used += need;
        Some((buffer.clone(), at))
    }

    fn gds_buffer(&mut self) -> Option<Retained<Buffer>> {
        if self.gds.is_none() {
            self.gds = Some(self.device.newBufferWithLength_options(GDS_BYTES, MTLResourceOptions::StorageModeShared)?);
        }
        self.gds.clone()
    }

    /// A zeroed slot of loop guard words for the next dispatch, as a byte offset.
    fn guard_slot(&mut self, program: u64, user: &[u32]) -> Option<(Retained<Buffer>, usize)> {
        if self.guard.is_none() {
            self.guard = Some(self.device.newBufferWithLength_options(GUARD_SLOTS * 8, MTLResourceOptions::StorageModeShared)?);
        }
        let buffer = self.guard.clone()?;
        if self.guarded.len() >= GUARD_SLOTS / 2 {
            self.check_guards();
        }
        let slot = self.guard_next;
        self.guard_next = (slot + 1) % GUARD_SLOTS;
        // SAFETY: a shared buffer stays mapped; the dispatch that used this
        // slot before is long done, and a late one could only gain budget.
        unsafe { buffer.contents().as_ptr().cast::<u64>().add(slot).write_volatile(0) };
        let mut words = [0; 16];
        words.copy_from_slice(&user[..16]);
        self.guarded.push((slot, program, words));
        Some((buffer, slot * 8))
    }

    /// Reports dispatches whose loops gave up since the last check.
    fn check_guards(&mut self) {
        let Some(buffer) = &self.guard else { return };
        let words = buffer.contents().as_ptr().cast::<u32>();
        for (slot, program, user) in std::mem::take(&mut self.guarded) {
            // SAFETY: the slot is inside the shared buffer, which stays mapped.
            let (units, gave_up) = unsafe { (words.add(2 * slot).read_volatile(), words.add(2 * slot + 1).read_volatile()) };
            if gave_up != 0 {
                self.guard_trips += 1;
                if self.guard_trips <= 16 {
                    eprintln!("ps5slop: GPU: compute program {program:#x}: {gave_up} threads gave up their loops ({units} of {} branch units), user data {user:x?}", gfx10::translate::GUARD_BUDGET);
                }
            }
        }
    }

    // Before the CPU reads guest memory: waits for encoded work that wrote it.
    fn settle(&mut self) {
        if self.wrote_memory {
            self.wrote_memory = false;
            self.flush_and_wait();
        }
    }

    /// The resources `program` binds, made ready before any encoder opens
    /// (uploads end the open pass).
    fn resolve(&mut self, host: &dyn Host, program: &Program, user: &[u32]) -> Option<Bound> {
        let mut bound = Bound { memory: None, gds: None, sampler_words: Vec::new(), textures: Vec::new(), samplers: Vec::new() };
        if program.translated.gds {
            bound.gds = Some(self.gds_buffer()?);
        }
        if program.translated.guest_memory {
            let Some(m) = self.guest_window(host, user, program.address) else {
                self.report(format!("no guest memory window for the program at {:#x}, user data {user:x?}", program.address));
                return None;
            };
            bound.memory = Some(m);
        }
        for r in &program.translated.resources {
            match *r {
                Resource::Texture { slot, ref source, kind, access } => {
                    let d = self.descriptor(host, user, source, 8)?;
                    let image = Image::read(&d);
                    self.note(|| format!("program {:#x} t{slot} {kind:?} {access:?} {image:x?}", program.address));
                    if kind == TextureKind::Cube && access != Access::Sample {
                        self.report(format!("{kind:?} textures with {access:?} access not supported"));
                        return None;
                    }
                    let tex = if access == Access::Atomic {
                        let sint = program.sint_textures >> slot & 1 != 0;
                        let want = if sint { MTLPixelFormat::R32Sint } else { MTLPixelFormat::R32Uint };
                        if image.address == 0 || image.kind < 8 {
                            None
                        } else {
                            self.storage_image(host, image, kind).and_then(|t| {
                                if t.pixelFormat() == want {
                                    Some(t)
                                } else if atomic_capable(t.pixelFormat()) {
                                    t.newTextureViewWithPixelFormat(want)
                                } else {
                                    None
                                }
                            })
                        }
                    } else if access != Access::Sample {
                        if image.address == 0 || image.kind < 8 { None } else { self.storage_image(host, image, kind) }
                    } else if image.address == 0 || image.kind < 8 {
                        self.null_texture(kind)
                    } else {
                        self.image(host, image, kind).and_then(|t| self.swizzled(t, image.swizzle))
                    };
                    let Some(tex) = tex else {
                        self.report(format!("image {image:x?} for the program at {:#x} not supported", program.address));
                        return None;
                    };
                    bound.textures.push((slot as usize, tex));
                }
                Resource::Sampler { slot, ref source } => {
                    let d = self.descriptor(host, user, source, 4)?;
                    bound.sampler_words.push((slot as usize, d[0]));
                    bound.samplers.push((slot as usize, self.sampler(Sampler::read(&d))?));
                }
            }
        }
        Some(bound)
    }

    fn apply(enc: Slot, bound: &Bound, user: &[u32]) {
        let at = binding::SAMPLER_WORDS as usize;
        let mut data = [0u32; 2 * binding::SAMPLER_WORDS as usize];
        let n = user.len().min(at);
        data[..n].copy_from_slice(&user[..n]);
        for &(slot, word) in &bound.sampler_words {
            if let Some(d) = data.get_mut(at + slot) {
                *d = word;
            }
        }
        enc.bytes(data.as_ptr().cast(), data.len() * 4, binding::USER as usize);
        if let Some(m) = &bound.memory {
            enc.buffer(&m.buffer, 0, binding::MEM as usize);
            if let Some(b) = &m.second {
                enc.buffer(b, 0, SECOND_WINDOW);
            }
            enc.bytes(m.win.as_ptr().cast(), std::mem::size_of_val(&m.win), binding::MEM_BASE as usize);
        }
        if let Some(gds) = &bound.gds {
            enc.buffer(gds, 0, binding::GDS as usize);
        }
        for (slot, t) in &bound.textures {
            enc.texture(t, *slot);
        }
        for (slot, s) in &bound.samplers {
            enc.sampler(s, *slot);
        }
    }

    /// Windows over the largest region a program's user data points into and
    /// over the GPU area (or the other way round), as programs read both.
    fn guest_window(&mut self, host: &dyn Host, user: &[u32], code: u64) -> Option<Memory> {
        let max = self.device.maxBufferLength() as u64 & !(PAGE - 1);
        let area = host.gpu_area();
        let in_area = |a: u64| area.is_some_and(|(s, e)| s <= a && a < e);
        let mut best = host.extent(code);
        let mut outside = best.filter(|&(lo, _)| !in_area(lo));
        for pair in user.windows(2) {
            for ptr in [pair[0] as u64 | (pair[1] as u64) << 32, pair[0] as u64 | ((pair[1] & 0xFFFF) as u64) << 32] {
                if let Some((lo, hi)) = host.extent(ptr) {
                    if best.is_none_or(|(l, h)| hi - lo > h - l) {
                        best = Some((lo, hi));
                    }
                    if !in_area(lo) && outside.is_none_or(|(l, h)| hi - lo > h - l) {
                        outside = Some((lo, hi));
                    }
                }
            }
        }
        let (lo, hi) = best?;
        let region = |this: &mut Self, (lo, hi): (u64, u64)| -> Option<(Retained<Buffer>, u64)> {
            let (buf, offset) = this.window(host, lo, (hi - lo).min(max))?;
            Some((buf, lo - offset as u64))
        };
        let (first, second) = match area {
            Some((start, end)) if in_area(lo) => {
                let w = self.area_window(host, start, end, hi)?;
                (Some((w, start)), outside.and_then(|r| region(self, r)))
            }
            Some((start, end)) => {
                let need = self.area.as_ref().map_or(end, |w| w.end);
                let a = self.area_window(host, start, end, need).map(|w| (w, start));
                (region(self, (lo, hi)), a)
            }
            None => (region(self, (lo, hi)), None),
        };
        let (buffer, base) = first?;
        let mut win = [base, base + buffer.length() as u64, 0, 0, 0];
        let second = second.map(|(b, base2)| {
            win[2] = base2;
            win[3] = base2 + b.length() as u64;
            win[4] = b.gpuAddress().wrapping_sub(buffer.gpuAddress());
            b
        });
        Some(Memory { buffer, second, win })
    }

    /// The window over the GPU area `[start, end)` reaching at least `need`,
    /// made again (in 1 GiB steps) when it does not reach that far.
    fn area_window(&mut self, host: &dyn Host, start: u64, end: u64, need: u64) -> Option<Retained<Buffer>> {
        if let Some(w) = &self.area
            && need <= w.end
        {
            return Some(w.buffer.clone());
        }
        let max = self.device.maxBufferLength() as u64 & !(PAGE - 1);
        let stop = need.next_multiple_of(1 << 30).min(end).min(start + max);
        if need > stop || !host.back(start, stop) {
            return None;
        }
        let ptr = NonNull::new(host.host_pointer(start)?)?;
        // SAFETY: `back` gave every page host memory, which stays mapped; the
        // window is dropped when pages under it are replaced.
        let buffer = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr.cast(),
                (stop - start) as usize,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }?;
        self.report(format!("GPU area window of {} MiB at {start:#x}", (stop - start) >> 20));
        self.area = Some(Window { start, end: stop, buffer: buffer.clone() });
        Some(buffer)
    }

    /// `PS5SLOP_TRACE_USER_MEMORY`: what pointer-like user data pairs point at (256 bytes; V# bases too unless `strict`).
    fn trace_user_memory(&mut self, host: &dyn Host, user: &[u32], strict: bool) {
        if self.trace.is_none() || std::env::var_os("PS5SLOP_TRACE_USER_MEMORY").is_none() {
            return;
        }
        for pair in user.windows(2) {
            let ptr = pair[0] as u64 | ((pair[1] & 0xFFFF) as u64) << 32;
            let mut b = [0u8; 256];
            if (!strict || pair[1] >> 16 == 0) && ptr >> 32 != 0 && host.read(ptr, &mut b) {
                let words: Vec<u32> = b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
                self.note(|| format!("      at {ptr:#x}: {words:08x?}"));
            }
        }
    }

    fn try_dispatch(&mut self, host: &dyn Host, state: &State, groups: [u32; 3]) -> Result<(), &'static str> {
        let s = state.sh;
        let address = ((s[sh::COMPUTE_PGM_LO] as u64) | (s[sh::COMPUTE_PGM_LO + 1] as u64) << 32) << 8;
        if address == 0 {
            return Err("no compute program");
        }
        if skipped(address) {
            return Err("skipped by gpuskip");
        }
        let users = (s[sh::COMPUTE_PGM_RSRC2] >> 1) & 0x1F;
        let user = &s[sh::COMPUTE_USER_DATA_0..sh::COMPUTE_USER_DATA_0 + sh::COMPUTE_USER_DATA_COUNT];
        let program = self.typed_program(host, state, Stage::Compute, address, users, false, 0, user).ok_or("compute program")?;
        self.note(|| format!("compute {address:#x} groups {groups:?} user {:x?}", &user[..users as usize]));
        self.trace_user_memory(host, &user[..users as usize], false);
        let task = program.compute.as_ref().ok_or("compute pipeline")?;
        let pipeline = match task.poll() {
            Some(p) => p,
            None => {
                let t = std::time::Instant::now();
                let p = task.wait();
                self.note_wait(1, t);
                p
            }
        }
        .ok_or("compute pipeline")?;
        let threads = [0, 1, 2].map(|i| (s[sh::COMPUTE_NUM_THREAD_X + i] & 0xFFFF).max(1) as usize);
        if threads.iter().product::<usize>() > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err("threadgroup too large");
        }
        let bound = self.resolve(host, &program, user);
        self.end_pass();
        let command = self.command();
        let enc = command.computeCommandEncoder().ok_or("compute encoder")?;
        if crate::timing::labels() {
            enc.setLabel(Some(&NSString::from_str(&format!("dispatch {:#x} {groups:?}", program.address))));
        }
        let memory = program.translated.guest_memory && bound.is_some();
        if memory && let Some(f) = self.wait_memory() {
            enc.waitForFence(f);
        }
        enc.setComputePipelineState(&pipeline);
        let guard = if program.translated.loop_guard { self.guard_slot(program.address, user) } else { None };
        if program.translated.loop_guard && guard.is_none() {
            enc.endEncoding();
            return Err("loop guard buffer");
        }
        if let Some(b) = &bound {
            Self::apply(Slot::Compute(&enc), b, user);
            if let Some((g, offset)) = &guard {
                Slot::Compute(&enc).buffer(g, *offset, binding::GUARD as usize);
            }
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
            self.wrote_memory |= program.translated.writes_memory;
            self.pending_writes |= program.translated.writes_memory;
        }
        if memory && program.translated.writes_memory && let Some(f) = self.fence.as_deref() {
            enc.updateFence(f);
            self.fence_updated = true;
        }
        enc.endEncoding();
        if crate::timing::labels() {
            command.setLabel(Some(&NSString::from_str(&format!("last dispatch #{} {:#x} {groups:?}", self.dispatches, program.address))));
        }
        static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let every = *EVERY.get_or_init(|| std::env::var("PS5SLOP_GPU_COMMIT_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
        if every > 0 && (self.dispatches + 1).is_multiple_of(every) {
            self.commit();
        }
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
        // USER_SGPR is 5 bits, with USER_SGPR_MSB at bit 27 for counts of 32
        // (Mesa's gfx10.json); a count of 32 reads as 0 without it.
        let user_count = |rsrc2: u32| ((rsrc2 >> 1) & 0x1F) | ((rsrc2 >> 27) & 1) << 5;
        let vs_addr = program_address(vs_lo);
        let ps_addr = program_address(sh::SPI_SHADER_PGM_LO_PS);
        if vs_addr == 0 || ps_addr == 0 {
            return Err("no vertex or pixel program");
        }
        if skipped(vs_addr) || skipped(ps_addr) {
            return Err("skipped by gpuskip");
        }
        // Modes 2, 5 and 6 only rewrite compression metadata; later draws
        // inheriting the mode are normal ones (see gpu.md).
        if state.color_control_written && matches!(c[context::CB_COLOR_CONTROL] >> 4 & 7, 2 | 5 | 6) {
            for n in 0..8 {
                if c[context::CB_TARGET_MASK] >> (4 * n) & 0xF != 0
                    && let Some(t) = ColorTarget::read(c, n)
                    && t.mip_level == 0
                    && let Some((tex, _)) = self.target(host, &t)
                {
                    self.fast_clear(host, &t, &tex);
                }
            }
            return Err("metadata pass");
        }
        let vs_users = user_count(s[vs_rsrc2]);
        let ps_users = user_count(s[sh::SPI_SHADER_PGM_RSRC2_PS]);
        let vs_data = &s[vs_user..vs_user + sh::USER_DATA_COUNT];
        let ps_data = &s[sh::SPI_SHADER_USER_DATA_PS_0..sh::SPI_SHADER_USER_DATA_PS_0 + sh::USER_DATA_COUNT];
        let col_format = export_formats(c);
        self.note(|| format!("vertex {vs_addr:#x} pixel {ps_addr:#x} col_format {:#x}", c[context::SPI_SHADER_COL_FORMAT]));
        let vertex = self.typed_program(host, state, Stage::Vertex, vs_addr, vs_users, ngg, 0, vs_data).ok_or("vertex program")?;
        let pixel = self.typed_program(host, state, Stage::Pixel, ps_addr, ps_users, false, col_format, ps_data).ok_or("pixel program")?;
        self.note(|| format!("vertex user {:x?}", &vs_data[..vs_users as usize]));
        self.note(|| format!("pixel user {:x?}", &ps_data[..ps_users as usize]));
        self.trace_user_memory(host, &vs_data[..vs_users as usize], true);
        self.trace_user_memory(host, &ps_data[..ps_users as usize], false);
        self.note(|| {
            format!(
                "{:?} scissor {:?} from {:x?} mode_cntl_0 {:#x} clip_cntl {:#x} vte_cntl {:#x}",
                Viewport::read(c),
                regs::scissor(c),
                [0x0C, 0x0D, 0x81, 0x82, 0x90, 0x91, 0x94, 0x95].map(|r| c[r]),
                c[context::PA_SC_MODE_CNTL_0],
                c[context::PA_CL_CLIP_CNTL],
                c[context::PA_CL_VTE_CNTL]
            )
        });
        self.note(|| {
            format!(
                "CB_COLOR_CONTROL {:#x} CB_TARGET_MASK {:#x} CB_SHADER_MASK {:#x} CB_BLEND0..3 {:x?} DB_DEPTH_CONTROL {:#x} DB_STENCIL_CONTROL {:#x} DB_STENCIL_REF_MASK {:#x} DB_RENDER_CONTROL {:#x}",
                c[context::CB_COLOR_CONTROL],
                c[context::CB_TARGET_MASK],
                c[context::CB_TARGET_MASK + 1],
                &c[context::CB_BLEND0_CONTROL..context::CB_BLEND0_CONTROL + 4],
                c[context::DB_DEPTH_CONTROL],
                c[context::DB_STENCIL_CONTROL],
                c[context::DB_STENCIL_REF_MASK],
                c[context::DB_RENDER_CONTROL]
            )
        });
        let rect = u[uconfig::VGT_PRIMITIVE_TYPE] == 7;
        let gl_clip = c[context::PA_CL_CLIP_CNTL] & (1 << 19) == 0;
        let vertex_function = self.vertex_function(&vertex, &pixel, rect, gl_clip);

        let mask = c[context::CB_TARGET_MASK];
        let mut targets = Vec::new();
        let mut key = PipelineKey {
            vertex: vertex.id,
            pixel: pixel.id,
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
            self.note(|| format!("mrt{n} {t:x?}"));
            if t.mip_level != 0 {
                return Err("draw to a mip level above 0");
            }
            let (tex, format) = self.target(host, &t).ok_or("color target format")?;
            self.note(|| format!("      texture {:p}", Retained::as_ptr(&tex)));
            self.fast_clear(host, &t, &tex);
            key.formats[targets.len()] = format.0;
            key.blends[targets.len()] = (Blend::read(c, n), write_mask(write, t.format.swap) as u8);
            let slice = if tex.textureType() == MTLTextureType::Type2DArray { t.slice } else { 0 };
            targets.push((t.address, tex, slice));
        }
        let depth_state = DepthState::read(c);
        let wants_depth = depth_state.depth_test() || depth_state.stencil();
        let size = targets.first().map(|t| (t.1.width() as u32, t.1.height() as u32));
        let mut depth = None;
        let mut clear = None;
        if let Some(d) = DepthTarget::read(c)
            && (wants_depth || c[context::DB_RENDER_CONTROL] & 3 != 0)
        {
            self.note(|| format!("depth {d:x?} test {wants_depth}"));
            let (dw, dh) = size.unwrap_or((d.width, d.height));
            if let Some((tex, format, fresh)) = self.depth(&d, dw, dh) {
                key.depth = format.0;
                key.stencil = d.stencil;
                let clearing = c[context::DB_RENDER_CONTROL] & 3 != 0 || self.htile_cleared(host, c, &d);
                if clearing || fresh {
                    let z = if clearing {
                        let (zs, zo) = regs::depth_range(c, gl_clip);
                        (f32::from_bits(c[context::DB_DEPTH_CLEAR]) * zs + zo).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    clear = Some((z, c[context::DB_STENCIL_CLEAR] & 0xFF));
                }
                depth = Some((d.address, tex, d.stencil, d.slice));
            }
        }
        if targets.is_empty() && depth.is_none() {
            return Err("no color target");
        }
        let pipeline = self.pipeline(&key, vertex_function, &pixel);
        let mut redrawn = !vertex.translated.writes_memory && !pixel.translated.writes_memory && !compile_wait();
        let surfaces: Vec<_> = targets
            .iter()
            .map(|t| (t.0, &t.1))
            .chain(depth.as_ref().map(|d| (d.0, &d.1)))
            .map(|(address, t)| (address, t.pixelFormat().0, t.width()))
            .collect();
        for surface in surfaces {
            let (first, last) = self.drawn.entry(surface).or_insert((self.epoch, self.epoch));
            redrawn &= self.shown.contains_key(&surface.0) || (*first < self.epoch && *last + 2 >= self.epoch);
            *last = self.epoch;
        }
        let mut ready = pipeline.poll();
        if ready.is_none() && redrawn && !self.compile_budget.is_zero() {
            let t = std::time::Instant::now();
            ready = pipeline.wait_for(self.compile_budget.min(std::time::Duration::from_millis(4)));
            self.compile_budget = self.compile_budget.saturating_sub(t.elapsed());
            crate::timing::add_raw(crate::timing::Part::CompileWait, t.elapsed().as_nanos() as u64);
        }
        let pipeline = match ready {
            Some(p) => p,
            // A target drawn every frame shows this draw from a later frame on.
            None if redrawn => {
                // The clear rides on the pass and would be lost with the draw.
                if clear.is_some() {
                    self.pass(&targets, depth, clear).ok_or("render pass")?;
                }
                return Err("compiling");
            }
            None => {
                let t = std::time::Instant::now();
                let p = pipeline.wait();
                self.note_wait(0, t);
                p
            }
        }
        .ok_or("pipeline")?;
        let depth_bound = depth.is_some();
        let (w, h) = match (size, &depth) {
            (Some((w, h)), _) => (w as f64, h as f64),
            (None, Some((_, t, _, _))) => (t.width() as f64, t.height() as f64),
            _ => return Err("no target size"),
        };
        let vs_bound = self.resolve(host, &vertex, vs_data).ok_or("resources")?;
        let ps_bound = self.resolve(host, &pixel, ps_data).ok_or("resources")?;
        if self.snap.is_some() && std::env::var_os("PS5SLOP_SNAP_INPUTS").is_some() {
            self.snap_inputs(ps_addr, &ps_bound);
        }
        let index = match draw {
            Draw::Indexed { count, address, .. } if !rect => {
                let wide = u[uconfig::VGT_INDEX_TYPE] & 3 == 1;
                let size = if wide { 4 } else { 2 };
                let len = count as u64 * size;
                let bound = if address % 4 == 0 {
                    self.window(host, address, len).ok_or("index buffer")?
                } else {
                    // Metal needs index buffer offsets to be multiples of 4.
                    self.settle();
                    let (_, hi) = host.extent(address).ok_or("index buffer")?;
                    let src = host.host_pointer(address).ok_or("index buffer")?;
                    let n = (len as usize).min((hi - address) as usize);
                    let (buf, offset) = self.scratch(len.max(4) as usize).ok_or("index buffer")?;
                    // SAFETY: `n` bytes from `address` lie in host-backed guest memory, and
                    // the scratch range is fresh shared memory no GPU work uses yet.
                    unsafe { std::ptr::copy_nonoverlapping(src, buf.contents().as_ptr().cast::<u8>().add(offset), n) };
                    (buf, offset)
                };
                Some((bound, wide))
            }
            _ => None,
        };
        let enc = self.pass(&targets, depth, clear).ok_or("render pass")?;
        if let Some(snap) = self.snap.as_mut() {
            snap.draws += 1;
            if snap.programs.last() != Some(&(vs_addr, ps_addr)) {
                snap.programs.push((vs_addr, ps_addr));
                if snap.programs.len() > 4 {
                    snap.programs.remove(1);
                }
            }
        }
        enc.setRenderPipelineState(&pipeline);
        if crate::timing::labels() {
            enc.insertDebugSignpost(&NSString::from_str(&format!("draw {vs_addr:#x} {ps_addr:#x}")));
        }
        if depth_bound {
            let ds = self.depth_state(depth_state, true).ok_or("depth state")?;
            enc.setDepthStencilState(Some(&ds));
            enc.setStencilReferenceValue(depth_state.ref_mask & 0xFF);
        }
        let vp = Viewport::read(c);
        let (znear, zfar) = if gl_clip {
            (vp.z_offset - vp.z_scale, vp.z_offset + vp.z_scale)
        } else {
            (vp.z_offset, vp.z_offset + vp.z_scale)
        };
        let (zs, zo) = regs::depth_range(c, gl_clip);
        let (znear, zfar) = (znear * zs + zo, zfar * zs + zo);
        enc.setViewport(MTLViewport {
            originX: (vp.x_offset - vp.x_scale) as f64,
            originY: (vp.y_offset - vp.y_scale.abs()) as f64,
            width: (2.0 * vp.x_scale) as f64,
            height: (2.0 * vp.y_scale.abs()) as f64,
            znear: znear as f64,
            zfar: zfar as f64,
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
        for (slot, t) in &ps_bound.textures {
            let (ptr, w, h) = (Retained::as_ptr(t) as usize, t.width(), t.height());
            self.note(|| format!("bound t{slot} {ptr:#x} {w}x{h}"));
        }
        let primitive = match u[uconfig::VGT_PRIMITIVE_TYPE] {
            1 => MTLPrimitiveType::Point,
            2 => MTLPrimitiveType::Line,
            3 => MTLPrimitiveType::LineStrip,
            4 | 7 => MTLPrimitiveType::Triangle,
            6 => MTLPrimitiveType::TriangleStrip,
            _ => return Err("primitive type"),
        };
        let start = u[uconfig::GE_INDX_OFFSET];
        match draw {
            Draw::Auto { .. } if rect && start != 0 => return Err("rectangle list with an index offset"),
            Draw::Auto { count, instances } if rect => unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(primitive, 0, (count / 3 * 6) as usize, instances.max(1) as usize);
            },
            Draw::Indexed { .. } if rect => return Err("indexed rectangle list"),
            Draw::Auto { count, instances } => unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(primitive, start as usize, count as usize, instances.max(1) as usize);
            },
            Draw::Indexed { count, instances, .. } => {
                let ((buf, offset), wide) = index.ok_or("index buffer")?;
                unsafe {
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive,
                        count as usize,
                        if wide { MTLIndexType::UInt32 } else { MTLIndexType::UInt16 },
                        &buf,
                        offset,
                        instances.max(1) as usize,
                        start as i32 as isize,
                        0,
                    );
                }
            }
        }
        self.pass_memory |= vertex.translated.writes_memory || pixel.translated.writes_memory;
        self.wrote_memory |= vertex.translated.writes_memory || pixel.translated.writes_memory;
        self.pending_writes |= vertex.translated.writes_memory || pixel.translated.writes_memory;
        Ok(())
    }

    /// Starts what `gpusnap` asked for the next frame.
    fn frame_tools(&mut self) {
        if RELOAD.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.programs.clear();
            self.variants.clear();
            self.pipelines.clear();
            self.missing_inputs.clear();
        }
        if let Some((dir, from)) = SNAP.lock().ok().and_then(|mut s| s.take()) {
            let _ = std::fs::create_dir_all(&dir);
            self.snap = Some(Snap { dir, from, passes: 0, bytes: 0, copies: Vec::new(), programs: Vec::new(), draws: 0 });
        }
    }

    /// Notes every context (0xA000 + n) and uconfig (0xC000 + n) register
    /// that changed since the last traced draw.
    fn note_changed_registers(&mut self, state: &State) {
        let now: Vec<u32> = state.context.iter().chain(state.uconfig).copied().collect();
        let old = std::mem::replace(&mut self.traced_regs, now);
        let split = state.context.len();
        let changed: Vec<String> = self
            .traced_regs
            .iter()
            .enumerate()
            .filter(|&(k, v)| old.get(k) != Some(v))
            .map(|(k, v)| {
                let reg = if k < split { 0xA000 + k } else { 0xC000 + k - split };
                format!("{reg:x}={v:x}")
            })
            .collect();
        self.note(|| format!("changed {}", changed.join(" ")));
    }

    fn tracing(&self) -> bool {
        self.trace_frames.is_some_and(|(a, b)| (a..=b).contains(&self.epoch))
    }

    fn end_trace(&mut self, what: impl FnOnce() -> String, result: Result<(), &'static str>) {
        if let Some(lines) = self.trace.take() {
            eprintln!("ps5slop: trace: {} -> {result:?}", what());
            for l in lines {
                eprintln!("ps5slop: trace:     {l}");
            }
        }
    }

    fn commit(&mut self) {
        self.end_pass();
        let now = crate::timing::host_seconds();
        if let Some(cb) = self.upload.take() {
            cb.commit();
            self.inflight.push((cb, now));
        }
        self.pending_writes = false;
        if let Some(cb) = self.command.take() {
            cb.commit();
            self.inflight.push((cb, now));
        }
        if self.inflight.len() > 32 {
            self.inflight.retain(|(cb, _)| cb.status() != MTLCommandBufferStatus::Completed);
            if self.inflight.len() > 48 {
                wait_reporting_hangs(&self.inflight[0].0);
            }
            if let Some((cb, at)) = self.inflight.first()
                && now - at > 2.0
                && self.reported_stuck != Some(Retained::as_ptr(cb) as usize)
            {
                self.reported_stuck = Some(Retained::as_ptr(cb) as usize);
                eprintln!("ps5slop: GPU: command buffer {:?} still running after 2 s (status {:?})", cb.label().map(|l| l.to_string()), cb.status());
            }
        }
    }

    fn flush_and_wait(&mut self) {
        self.commit();
        for (cb, committed) in std::mem::take(&mut self.inflight) {
            wait_reporting_hangs(&cb);
            let woken = crate::timing::host_seconds();
            let ns = |s: f64| (s.max(0.0) * 1e9) as u64;
            crate::timing::add_raw(crate::timing::Part::Submission, ns(cb.kernelStartTime() - committed));
            crate::timing::add_raw(crate::timing::Part::WakeUp, ns(woken - cb.GPUEndTime()));
            if let Some(e) = cb.error() {
                self.failed_commands += 1;
                if self.failed_commands <= 8 {
                    eprintln!("ps5slop: GPU: command buffer failed: {}", e.localizedDescription());
                    report_encoders(&e);
                }
            }
            let busy = cb.GPUEndTime() - cb.GPUStartTime();
            let scheduled = cb.GPUStartTime() - cb.kernelStartTime();
            crate::timing::add_raw(crate::timing::Part::Scheduling, (scheduled.max(0.0) * 1e9) as u64);
            crate::timing::add_raw(crate::timing::Part::GpuBusy, (busy * 1e9) as u64);
            crate::timing::add_raw(crate::timing::Part::Syncs, 1_000_000);
        }
        self.check_guards();
    }
}

/// Waits for `cb`; one still running after 10 s is reported with its label
/// (`PS5SLOP_GPU_LABELS` names the dispatches in it), then waited for.
fn wait_reporting_hangs(cb: &CommandBuffer) {
    let start = std::time::Instant::now();
    while !matches!(cb.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error) {
        if start.elapsed() > std::time::Duration::from_secs(10) {
            eprintln!("ps5slop: GPU: command buffer {:?} still running after 10 s (status {:?})", cb.label().map(|l| l.to_string()), cb.status());
            cb.waitUntilCompleted();
            return;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
}

/// Prints the encoders a failed command buffer reports as faulted or pending
/// (`PS5SLOP_GPU_ENCODER_STATUS`), with their labels and signposts.
fn report_encoders(e: &objc2_foundation::NSError) {
    use objc2_metal::{MTLCommandBufferEncoderInfo, MTLCommandBufferEncoderInfoErrorKey, MTLCommandEncoderErrorState as State};
    let info = e.userInfo();
    // SAFETY: Metal documents this key's value as an array of MTLCommandBufferEncoderInfo.
    let Some(list) = (unsafe { info.objectForKey(MTLCommandBufferEncoderInfoErrorKey) }) else { return };
    let list: Retained<objc2_foundation::NSArray<ProtocolObject<dyn MTLCommandBufferEncoderInfo>>> = unsafe { Retained::cast_unchecked(list) };
    for enc in list.iter() {
        let state = enc.errorState();
        if state == State::Faulted || state == State::Pending {
            // SAFETY: `debugSignposts` returns an array of strings, or nil when there are none.
            let posts: Option<Retained<objc2_foundation::NSArray<NSString>>> = unsafe { objc2::msg_send![&*enc, debugSignposts] };
            let posts: Vec<String> = posts.map(|p| p.iter().map(|s| s.to_string()).collect()).unwrap_or_default();
            let n = posts.len();
            eprintln!("ps5slop: GPU:   encoder {:?} {} {:?}", state.0, enc.label(), &posts[n.saturating_sub(4)..]);
        }
    }
}

/// The global data share's size on GFX10.
const GDS_BYTES: usize = 64 << 10;
/// Dispatches that can hold a slot of loop guard words at once; a slot is
/// zeroed when handed out again.
const GUARD_SLOTS: usize = 1 << 16;

impl Backend for Metal {
    fn with_gds(&mut self, f: &mut dyn FnMut(&mut [u8])) -> bool {
        let Some(buffer) = self.gds_buffer() else { return false };
        // SAFETY: a shared buffer's contents stay mapped while `buffer` lives;
        // the processor touches it only after the GPU work before is done.
        f(unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast(), GDS_BYTES) });
        true
    }

    fn draw(&mut self, host: &dyn Host, state: &State, draw: Draw) {
        self.trace = self.tracing().then(Vec::new);
        if self.trace.is_some() && std::env::var_os("PS5SLOP_TRACE_REGS").is_some() {
            self.note_changed_registers(state);
        }
        let result = self.try_draw(host, state, draw);
        self.end_trace(|| format!("draw {draw:?}"), result);
        match result {
            Ok(()) => self.draws += 1,
            Err(why) => {
                self.skipped += 1;
                *self.reasons.entry(why).or_default() += 1;
            }
        }
    }

    fn dispatch(&mut self, host: &dyn Host, state: &State, groups: [u32; 3]) {
        self.trace = self.tracing().then(Vec::new);
        let result = self.try_dispatch(host, state, groups);
        self.end_trace(|| format!("dispatch {groups:?}"), result);
        match result {
            Ok(()) => self.dispatches += 1,
            Err(why) => {
                self.skipped += 1;
                *self.reasons.entry(why).or_default() += 1;
            }
        }
    }

    fn sync(&mut self) {
        self.flush_and_wait();
    }

    fn flush(&mut self) {
        self.commit();
    }

    fn summary(&self) -> String {
        let mut reasons: Vec<_> = self.reasons.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1));
        let mib = |t: &Retained<Texture>| MTLAllocation::allocatedSize(&**t) as u64;
        let image_bytes: u64 = self.images.values().map(|(_, t)| mib(t)).sum();
        let target_bytes: u64 = self.targets.values().map(|t| mib(&t.texture)).sum();
        format!(
            "{} drawn, {} dispatched, {} skipped {reasons:?}, {} command buffers failed, {} dispatches gave up loops; caches: {} images ({} MiB), {} targets ({} MiB), {} depths, {} volumes, {} swizzle views, {} windows, {} pipelines, {} compiles queued; compile waits: {} draws ({} ms), {} dispatches ({} ms)",
            self.draws,
            self.dispatches,
            self.skipped,
            self.failed_commands,
            self.guard_trips,
            self.images.len(),
            image_bytes >> 20,
            self.targets.len(),
            target_bytes >> 20,
            self.depths.len(),
            self.volumes.len(),
            self.swizzles.len(),
            self.windows.len(),
            self.pipelines.len(),
            crate::compile::queued(),
            self.compile_waits[0].0,
            self.compile_waits[0].1.as_millis(),
            self.compile_waits[1].0,
            self.compile_waits[1].1.as_millis()
        )
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
            if self.tracing() {
                let (a, b) = (Retained::as_ptr(&src) as usize, Retained::as_ptr(&copy) as usize);
                eprintln!("ps5slop: trace: shown {addr:#x} copied from {a:#x} into {b:#x}");
            }
            if let Ok(mut f) = frames().lock() {
                f.insert(addr, SharedTexture(copy));
            }
        } else {
            self.flush_and_wait();
        }
        if let Some(snap) = self.snap.take() {
            Self::write_snap(snap);
        }
        self.frame_tools();
        if self.tracing() {
            eprintln!("ps5slop: trace: frame {} done, showing {shown:x?}", self.epoch);
        }
        self.epoch += 1;
        self.compile_budget = COMPILE_BUDGET;
        self.swizzles.clear();
        let epoch = self.epoch;
        self.images.retain(|_, (used, _)| epoch - *used < IMAGE_LIFETIME);
        let n = TRACE_NEXT.swap(0, std::sync::atomic::Ordering::Relaxed);
        if n > 0 {
            self.trace_frames = Some((self.epoch, self.epoch + n - 1));
        }
    }
}

// Each rectangle draws guest vertices 0, 1, 2, 1, 3, 2; corner 3 is v1 + v2 - v0.
/// Compiles `msl` and returns its function `entry`; any thread.
fn compile(device: &Device, msl: &str, entry: &str) -> Option<Retained<Function>> {
    let t = crate::timing::start();
    let unique;
    let msl = if std::env::var_os("PS5SLOP_MSL_UNIQUE").is_some() {
        unique = format!("{msl}\n// {:?}\n", std::time::SystemTime::now());
        unique.as_str()
    } else {
        msl
    };
    let lib = device.newLibraryWithSource_options_error(&NSString::from_str(msl), None);
    crate::timing::add(crate::timing::Part::Compile, t);
    match lib {
        Ok(lib) => lib.newFunctionWithName(&NSString::from_str(entry)),
        Err(e) => {
            late_report(format!("MSL did not compile: {}", e.localizedDescription()));
            if std::env::var_os("PS5SLOP_MSL_ERRORS").is_some() {
                eprintln!("{msl}");
            }
            None
        }
    }
}

/// Reports a failure from the compile threads, the first [`REPORTS`] times.
fn late_report(what: String) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < REPORTS {
        eprintln!("ps5slop: GPU: {what}");
    }
}

/// Whether draws wait for their pipeline instead of skipping (`PS5SLOP_COMPILE_WAIT`).
fn compile_wait() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PS5SLOP_COMPILE_WAIT").is_some())
}

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

// OpenGL-style clip space puts z in [-w, w]; Metal clips z below 0.
fn gl_clip_space(msl: &str) -> String {
    let mut out = String::with_capacity(msl.len() + 128);
    let mut rest = msl;
    while let Some(i) = rest.find("o.pos = float4(") {
        let Some(end) = rest[i..].find(");").map(|e| i + e) else { break };
        out.push_str(&rest[..i]);
        out.push_str("o.pos = gl_z(");
        out.push_str(&rest[i + "o.pos = ".len()..end + 1]);
        out.push_str(");");
        rest = &rest[end + 2..];
    }
    out.push_str(rest);
    out.replacen("struct VsOut {", "static inline float4 gl_z(float4 p) { p.z = (p.z + p.w) * 0.5; return p; }\n\nstruct VsOut {", 1)
}

/// `SPI_SHADER_COL_FORMAT` with each integer target's export made the
/// matching 16-bit integer format, so the bits the CB receives reach it as
/// they are (Metal rejects float outputs to integer attachments).
fn export_formats(c: &[u32]) -> u32 {
    let mut formats = c[context::SPI_SHADER_COL_FORMAT];
    for n in 0..8 {
        let Some(t) = ColorTarget::read(c, n) else { continue };
        let export = formats >> (4 * n) & 0xF;
        let want = match t.format.number {
            4 => 7,
            5 => 8,
            _ => continue,
        };
        if export != 0 && export != want {
            formats = formats & !(0xF << (4 * n)) | want << (4 * n);
        }
    }
    formats
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Number {
    Float,
    Uint,
    Sint,
}

/// Whether Metal format `f` holds floats (including normalised) or integers.
fn number(f: MTLPixelFormat) -> Number {
    use MTLPixelFormat as P;
    if [P::R8Uint, P::R16Uint, P::RG8Uint, P::R32Uint, P::RG16Uint, P::RGBA8Uint, P::RGB10A2Uint, P::RG32Uint, P::RGBA16Uint, P::RGBA32Uint].contains(&f) {
        Number::Uint
    } else if [P::R8Sint, P::R16Sint, P::RG8Sint, P::R32Sint, P::RG16Sint, P::RGBA8Sint, P::RG32Sint, P::RGBA16Sint, P::RGBA32Sint].contains(&f) {
        Number::Sint
    } else {
        Number::Float
    }
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
    let bgr = f.swap & 1 == 1;
    Some(match (f.layout, f.number) {
        (10, 6) if bgr => P::BGRA8Unorm_sRGB,
        (10, 6) => P::RGBA8Unorm_sRGB,
        (10, 0) if bgr => P::BGRA8Unorm,
        (10, 0) => P::RGBA8Unorm,
        (10, 1) => P::RGBA8Snorm,
        (10, 4) => P::RGBA8Uint,
        (10, 5) => P::RGBA8Sint,
        (1, 0) => P::R8Unorm,
        (1, 4) => P::R8Uint,
        (2, 0) => P::R16Unorm,
        (2, 4) => P::R16Uint,
        (2, 7) => P::R16Float,
        (3, 0) => P::RG8Unorm,
        (3, 4) => P::RG8Uint,
        (4, 4) => P::R32Uint,
        (4, 5) => P::R32Sint,
        (4, 7) => P::R32Float,
        (5, 0) => P::RG16Unorm,
        (5, 4) => P::RG16Uint,
        (5, 7) => P::RG16Float,
        (6 | 7, _) => P::RG11B10Float,
        (8 | 9, 4) => P::RGB10A2Uint,
        (8 | 9, _) => P::RGB10A2Unorm,
        (11, 4) => P::RG32Uint,
        (11, 7) => P::RG32Float,
        (12, 0) => P::RGBA16Unorm,
        (12, 4) => P::RGBA16Uint,
        (12, 7) => P::RGBA16Float,
        (14, 4) => P::RGBA32Uint,
        (14, 7) => P::RGBA32Float,
        (16, 0) => P::B5G6R5Unorm,
        _ => return None,
    })
}

/// A 3D image of one texel, whose layout cannot differ from a 2D one.
fn one_texel_volume(image: Image) -> bool {
    image.width == 1 && image.height == 1 && image.depth == 1
}

// Metal allows views of the same bytes per pixel; packed float formats
// only as 32-bit integers.
fn view_compatible(from: MTLPixelFormat, to: MTLPixelFormat) -> bool {
    use MTLPixelFormat as P;
    let bytes = |f: MTLPixelFormat| -> Option<u32> {
        Some(match f {
            P::R8Unorm | P::R8Snorm | P::R8Uint | P::R8Sint => 1,
            P::R16Unorm | P::R16Snorm | P::R16Uint | P::R16Sint | P::R16Float | P::RG8Unorm | P::RG8Snorm | P::RG8Uint | P::RG8Sint => 2,
            P::R32Uint | P::R32Sint | P::R32Float | P::RG16Unorm | P::RG16Snorm | P::RG16Uint | P::RG16Sint | P::RG16Float => 4,
            P::RGBA8Unorm | P::RGBA8Unorm_sRGB | P::RGBA8Snorm | P::RGBA8Uint | P::RGBA8Sint | P::BGRA8Unorm | P::BGRA8Unorm_sRGB => 4,
            P::RGB10A2Unorm | P::RG11B10Float | P::RGB9E5Float => 4,
            P::RG32Uint | P::RG32Sint | P::RG32Float | P::RGBA16Unorm | P::RGBA16Snorm | P::RGBA16Uint | P::RGBA16Sint | P::RGBA16Float => 8,
            P::RGBA32Uint | P::RGBA32Sint | P::RGBA32Float => 16,
            _ => return None,
        })
    };
    let packed = |f: MTLPixelFormat| matches!(f, P::RG11B10Float | P::RGB9E5Float);
    if from == to {
        return true;
    }
    match (bytes(from), bytes(to)) {
        (Some(a), Some(b)) if a == b => {
            if packed(from) || packed(to) {
                let other = if packed(from) { to } else { from };
                matches!(other, P::R32Uint | P::R32Sint)
            } else {
                true
            }
        }
        _ => false,
    }
}

/// The Metal format and bytes per element of a GFX10 image format.
fn image_format(format: u32) -> Option<(MTLPixelFormat, u32)> {
    use MTLPixelFormat as P;
    Some(match format {
        1 => (P::R8Unorm, 1),
        2 => (P::R8Snorm, 1),
        5 => (P::R8Uint, 1),
        6 => (P::R8Sint, 1),
        7 => (P::R16Unorm, 2),
        8 => (P::R16Snorm, 2),
        11 => (P::R16Uint, 2),
        12 => (P::R16Sint, 2),
        13 => (P::R16Float, 2),
        14 => (P::RG8Unorm, 2),
        15 => (P::RG8Snorm, 2),
        18 => (P::RG8Uint, 2),
        19 => (P::RG8Sint, 2),
        20 => (P::R32Uint, 4),
        21 => (P::R32Sint, 4),
        22 => (P::R32Float, 4),
        23 => (P::RG16Unorm, 4),
        24 => (P::RG16Snorm, 4),
        27 => (P::RG16Uint, 4),
        28 => (P::RG16Sint, 4),
        29 => (P::RG16Float, 4),
        36 | 43 => (P::RG11B10Float, 4),
        44 | 50 => (P::RGB10A2Unorm, 4),
        56 => (P::RGBA8Unorm, 4),
        57 => (P::RGBA8Snorm, 4),
        60 => (P::RGBA8Uint, 4),
        61 => (P::RGBA8Sint, 4),
        62 => (P::RG32Uint, 8),
        63 => (P::RG32Sint, 8),
        64 => (P::RG32Float, 8),
        65 => (P::RGBA16Unorm, 8),
        66 => (P::RGBA16Snorm, 8),
        69 => (P::RGBA16Uint, 8),
        70 => (P::RGBA16Sint, 8),
        71 => (P::RGBA16Float, 8),
        75 => (P::RGBA32Uint, 16),
        76 => (P::RGBA32Sint, 16),
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

fn is_depth_format(f: MTLPixelFormat) -> bool {
    matches!(f, MTLPixelFormat::Depth32Float | MTLPixelFormat::Depth16Unorm | MTLPixelFormat::Depth32Float_Stencil8)
}

/// Whether swizzle mode `raw` is one of the `_Z` (depth) modes: 4 KiB and
/// 64 KiB, plain, `_T` and `_X`.
fn is_depth_layout(raw: u32) -> bool {
    matches!(raw, 0x04 | 0x08 | 0x10 | 0x14 | 0x18)
}

/// Bytes per texel of an uncompressed format, from the image format table.
fn texel_bytes(f: MTLPixelFormat) -> Option<usize> {
    (0..256).find_map(|n| image_format(n).filter(|(m, _)| *m == f).map(|(_, b)| b as usize))
}

/// Formats made with atomic usage, for image atomics.
fn atomic_capable(format: MTLPixelFormat) -> bool {
    matches!(format, MTLPixelFormat::R32Uint | MTLPixelFormat::R32Sint)
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
    fn integer_targets_take_integer_exports() {
        let mut c = vec![0u32; 0x400];
        for (n, number) in [(0, 4), (1, 7)] {
            c[context::CB_COLOR0_BASE + n * context::CB_COLOR_STRIDE] = 0x1000 * (n as u32 + 1);
            c[context::CB_COLOR0_INFO + n * context::CB_COLOR_STRIDE] = 12 << 2 | number << 8;
        }
        c[context::SPI_SHADER_COL_FORMAT] = 0x944;
        assert_eq!(export_formats(&c), 0x947);
    }

    /// Runs a compute program whose loop counts s2 up to the bound in s0 and
    /// then appends to GDS; returns how many threads got there, how many
    /// dispatches gave up, and the GPU time.
    fn run_bounded_loop(metal: &mut Metal, mem: &Mem, code: &[u32], at: u64, groups: u32, bound: u32) -> (u32, u64, f64) {
        for (i, w) in code.iter().enumerate() {
            mem.write(at + 4 * i as u64, &w.to_le_bytes());
        }
        metal.with_gds(&mut |g| g.fill(0));
        let (c, u) = (vec![0u32; 0x400], vec![0u32; 0x400]);
        let mut sh = vec![0u32; 0x400];
        sh[sh::COMPUTE_PGM_LO] = (at >> 8) as u32;
        sh[sh::COMPUTE_PGM_RSRC2] = 1 << 1;
        sh[sh::COMPUTE_NUM_THREAD_X] = 64;
        sh[sh::COMPUTE_USER_DATA_0] = bound;
        let state = State { context: &c, sh: &sh, uconfig: &u, color_control_written: false };
        let trips = metal.guard_trips;
        metal.dispatch(mem, &state, [groups, 1, 1]);
        assert_eq!(metal.skipped, 0, "{:?}", metal.reasons);
        metal.commit();
        let start = std::time::Instant::now();
        let mut gpu = 0.0;
        for (cb, _) in std::mem::take(&mut metal.inflight) {
            while !matches!(cb.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error) {
                assert!(start.elapsed() < std::time::Duration::from_secs(5), "the loop did not end within 5 s");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(cb.error().is_none());
            gpu += cb.GPUEndTime() - cb.GPUStartTime();
        }
        metal.check_guards();
        let mut done = 0;
        metal.with_gds(&mut |g| done = u32::from_le_bytes(g[..4].try_into().unwrap()));
        (done, metal.guard_trips - trips, gpu)
    }

    // Each loop is also bounded by its user SGPR, so a guard that fails cannot
    // hang the GPU: without the guard every case ends in well under a second.
    #[test]
    fn runaway_compute_loops_end_on_their_own() {
        let Some(mut metal) = Metal::new() else { return };
        let mem = Mem::new(1 << 20);
        const HEAD: [u32; 3] = [
            0xBEFC_0380, // s_mov_b32 m0, 0
            0xBE82_0380, // s_mov_b32 s2, 0
            0x8002_8102, // s_add_u32 s2, s2, 1
        ];
        const TAIL: [u32; 5] = [
            0xBF0A_0002, // s_cmp_lt_u32 s2, s0
            0xBF85_FFFD, // s_cbranch_scc1 (to s_add_u32)
            0xD8FA_0000, 0x0100_0000, // ds_append v1 gds
            0xBF81_0000, // s_endpgm
        ];
        let plain: Vec<u32> = HEAD.iter().chain(&TAIL).copied().collect();
        let mut barrier = HEAD.to_vec();
        barrier.push(0xBF8A_0000); // s_barrier
        barrier.extend(TAIL);
        barrier[5] = 0xBF85_FFFC;

        let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &plain, 0x1000, 4, 1 << 12);
        assert_eq!((done, trips), (256, 0), "an honest loop runs to its end");
        eprintln!("honest: {gpu:.4} s");
        let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &plain, 0x1000, 1, 1 << 25);
        assert_eq!((done, trips), (0, 1), "a long loop in one group gives up");
        assert!(gpu < 0.1, "{gpu} s");
        eprintln!("one group: {gpu:.4} s");
        let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &plain, 0x1000, 1 << 14, 1 << 13);
        assert_eq!(trips, 1);
        assert!(done < (1 << 20) / 2, "the dispatch's budget stops most of {} threads, {done} finished", 1 << 20);
        assert!(gpu < 1.0, "{gpu} s");
        eprintln!("large grid: {gpu:.4} s, {done} finished");
        let (done, trips, gpu) = run_bounded_loop(&mut metal, &mem, &barrier, 0x2000, 2, 1 << 25);
        assert_eq!((done, trips), (0, 1), "groups with a barrier leave together");
        assert!(gpu < 0.1, "{gpu} s");
        eprintln!("barrier: {gpu:.4} s");
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
        assert!(metal.detile_into(&mem, &tex, 0, 0, layout));
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
