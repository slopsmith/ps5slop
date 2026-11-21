//! Draws guest frames into a CAMetalLayer on the window's view. From
//! 86slopper, taking RGBA frames.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{Message, msg_send};
use objc2_core_foundation::CGSize;
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceSRGB};
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

use super::layout::draw_rect;

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct V { float4 pos [[position]]; float2 uv; };

vertex V vs(uint id [[vertex_id]]) {
    float2 uv = float2((id << 1) & 2, id & 2);
    V o;
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uv;
    return o;
}

fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                   constant uint& linear [[buffer(0)]]) {
    float3 c = t.sample(s, in.uv).rgb;
    if (linear != 0) {
        c = clamp(c, 0.0, 1.0);
        c = select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
    }
    return float4(c, 1);
}
"#;

type Device = ProtocolObject<dyn MTLDevice>;
type Texture = ProtocolObject<dyn MTLTexture>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;

pub struct Presenter {
    layer: Retained<CAMetalLayer>,
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    linear: Retained<ProtocolObject<dyn MTLSamplerState>>,
    nearest: Retained<ProtocolObject<dyn MTLSamplerState>>,
    texture: Option<Retained<Texture>>,
    /// Whether `texture` is ours to upload into, rather than the GPU's.
    uploadable: bool,
    /// Whether `texture` holds linear light, to be encoded as sRGB.
    linear_light: bool,
    picture: (u32, u32),
    last: Option<Retained<CommandBuffer>>,
}

impl Presenter {
    /// Puts a Metal layer on `window`'s view.
    pub fn new(window: &Window) -> Result<Self, String> {
        let view = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
            RawWindowHandle::AppKit(h) => h.ns_view,
            _ => return Err("not an AppKit window".into()),
        };
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(SHADER), None)
            .map_err(|e| e.localizedDescription().to_string())?;
        let function = |name: &str| {
            lib.newFunctionWithName(&NSString::from_str(name))
                .ok_or(format!("no {name} in the shader"))
        };
        let desc = MTLRenderPipelineDescriptor::new();
        let (vs, fs) = (function("vs")?, function("fs")?);
        desc.setVertexFunction(Some(&vs));
        desc.setFragmentFunction(Some(&fs));
        unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .map_err(|e| e.localizedDescription().to_string())?;
        let sampler = |filter| {
            let d = MTLSamplerDescriptor::new();
            d.setMinFilter(filter);
            d.setMagFilter(filter);
            d.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
            d.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
            device
                .newSamplerStateWithDescriptor(&d)
                .ok_or("no Metal sampler")
        };
        let linear = sampler(MTLSamplerMinMagFilter::Linear)?;
        let nearest = sampler(MTLSamplerMinMagFilter::Nearest)?;

        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setFramebufferOnly(true);
        layer.setDisplaySyncEnabled(true);
        layer.setOpaque(true);
        let srgb = unsafe { CGColorSpace::with_name(Some(kCGColorSpaceSRGB)) };
        layer.setColorspace(srgb.as_deref());
        attach(view, &layer);
        let presenter = Self {
            layer,
            device,
            queue,
            pipeline,
            linear,
            nearest,
            texture: None,
            uploadable: false,
            linear_light: false,
            picture: (0, 0),
            last: None,
        };
        presenter.resize(window.inner_size().into(), window.scale_factor());
        Ok(presenter)
    }

    /// Matches the drawable to the view's size in pixels and its backing scale.
    pub fn resize(&self, pixels: (u32, u32), scale: f64) {
        self.layer.setContentsScale(scale);
        self.layer
            .setDrawableSize(CGSize::new(pixels.0.max(1) as f64, pixels.1.max(1) as f64));
    }

    fn drawable_size(&self) -> (u32, u32) {
        let s = self.layer.drawableSize();
        (s.width as u32, s.height as u32)
    }

    /// Shows a texture the GPU rendered, from the next draw on; `linear`
    /// when it holds linear light.
    pub fn show(&mut self, texture: &Texture, linear: bool) {
        self.picture = (texture.width() as u32, texture.height() as u32);
        self.texture = Some(texture.retain());
        self.linear_light = linear;
        self.uploadable = false;
    }

    /// Copies a frame of RGBA8 pixels into the texture the next draw shows.
    pub fn upload(&mut self, pixels: &[u8], w: u32, h: u32) {
        if w == 0 || h == 0 || pixels.len() < (w * h * 4) as usize {
            return;
        }
        self.linear_light = false;
        if self.picture != (w, h) || self.texture.is_none() || !self.uploadable {
            let desc = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    MTLPixelFormat::RGBA8Unorm,
                    w as usize,
                    h as usize,
                    false,
                )
            };
            desc.setUsage(MTLTextureUsage::ShaderRead);
            desc.setStorageMode(MTLStorageMode::Shared);
            self.texture = self.device.newTextureWithDescriptor(&desc);
            self.uploadable = true;
            self.picture = (w, h);
        }
        let Some(tex) = &self.texture else { return };
        if let Some(cb) = self.last.take() {
            cb.waitUntilCompleted();
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w as usize,
                height: h as usize,
                depth: 1,
            },
        };
        unsafe {
            tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new_unchecked(pixels.as_ptr() as *mut _),
                w as usize * 4,
            );
        }
    }

    /// Draws the current picture letterboxed and presents it at the next refresh.
    pub fn draw(&mut self) {
        let size = self.drawable_size();
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        let Some(drawable) = self.layer.nextDrawable() else {
            return;
        };
        let Some(cb) = self.queue.commandBuffer() else {
            return;
        };
        let pass = MTLRenderPassDescriptor::new();
        let att = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        att.setTexture(Some(&drawable.texture()));
        att.setLoadAction(MTLLoadAction::Clear);
        att.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        });
        att.setStoreAction(MTLStoreAction::Store);
        let Some(enc) = cb.renderCommandEncoderWithDescriptor(&pass) else {
            return;
        };
        if let Some(tex) = &self.texture {
            let (r, nearest) = draw_rect(size, self.picture);
            enc.setRenderPipelineState(&self.pipeline);
            enc.setViewport(MTLViewport {
                originX: r.x,
                originY: r.y,
                width: r.w,
                height: r.h,
                znear: 0.0,
                zfar: 1.0,
            });
            let linear = u32::from(self.linear_light);
            unsafe {
                enc.setFragmentTexture_atIndex(Some(tex), 0);
                enc.setFragmentBytes_length_atIndex(
                    NonNull::from(&linear).cast(),
                    std::mem::size_of::<u32>(),
                    0,
                );
                enc.setFragmentSamplerState_atIndex(
                    Some(if nearest { &self.nearest } else { &self.linear }),
                    0,
                );
                enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
        }
        enc.endEncoding();
        cb.presentDrawable(ProtocolObject::from_ref(&*drawable));
        cb.commit();
        self.last = Some(cb);
    }
}

/// Makes `layer` the backing layer of the NSView `view`.
fn attach(view: NonNull<std::ffi::c_void>, layer: &CAMetalLayer) {
    let view: &AnyObject = unsafe { view.cast().as_ref() };
    unsafe {
        let _: () = msg_send![view, setLayer: layer];
        let _: () = msg_send![view, setWantsLayer: true];
    }
}
