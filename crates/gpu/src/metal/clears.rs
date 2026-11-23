//! Colour fast clears (CMASK and DCC) checked on the GPU, after the work
//! that may write the metadata word, instead of waiting for it on the CPU.

use super::*;

/// What a GPU-side check reads.
#[derive(Clone, Copy)]
pub(super) enum ColourCheck {
    /// A CMASK word; a cleared one fills the target with the raw `words`.
    Cmask { at: u64, words: [u32; 2] },
    /// A DCC word; codes as in [`Metal::fast_clear`]. `first` is the target's
    /// first texel, which holds the colour for the "single" code.
    Dcc {
        at: u64,
        words: [u32; 2],
        first: u64,
    },
}

/// Integer formats as wide as a texel of `bpp` bytes, to write raw bits through.
fn raw_view_format(bpp: usize) -> Option<MTLPixelFormat> {
    match bpp {
        1 => Some(MTLPixelFormat::R8Uint),
        2 => Some(MTLPixelFormat::R16Uint),
        4 => Some(MTLPixelFormat::R32Uint),
        8 => Some(MTLPixelFormat::RG32Uint),
        _ => None,
    }
}

impl Metal {
    /// Encodes `check` and the fills it may ask for, into `slice` of `tex`.
    /// Returns `None`, having encoded nothing, when this cannot be done on the GPU.
    pub(super) fn fast_clear_on_gpu(
        &mut self,
        host: &dyn Host,
        tex: &Texture,
        slice: usize,
        check: ColourCheck,
    ) -> Option<()> {
        let bpp = texel_bytes(tex.pixelFormat()).filter(|&b| b <= 8)?;
        let float = number(tex.pixelFormat()) == Number::Float;
        if matches!(check, ColourCheck::Dcc { .. }) && !float {
            return None;
        }
        let view = tex.newTextureViewWithPixelFormat(raw_view_format(bpp)?)?;
        let helpers = self.clear_helpers()?;
        let raw_pipeline = self.clear_pipeline(&helpers, view.pixelFormat(), true)?;
        let colour_pipeline = if float {
            Some(self.clear_pipeline(&helpers, tex.pixelFormat(), false)?)
        } else {
            None
        };
        let (at, words, first) = match check {
            ColourCheck::Cmask { at, words } => (at, words, None),
            ColourCheck::Dcc { at, words, first } => (at, words, Some(first)),
        };
        let (word, word_at) = self.memory_buffer(host, at, 4)?;
        let first = match first {
            Some(f) => Some(self.memory_buffer(host, f, bpp as u64)?),
            None => None,
        };
        let (state, state_at) = self.scratch(64)?;

        self.end_pass();
        let command = self.command();
        let enc = command.computeCommandEncoder()?;
        let params = [words[0], words[1], bpp as u32, 0];
        if let Some((first, first_at)) = &first {
            enc.setComputePipelineState(&helpers.dcc);
            unsafe { enc.setBuffer_offset_atIndex(Some(first), *first_at, 3) };
        } else {
            enc.setComputePipelineState(&helpers.cmask);
        }
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&word), word_at, 0);
            enc.setBuffer_offset_atIndex(Some(&state), state_at, 1);
            enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 16, 2);
        }
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        enc.endEncoding();

        let (w, h) = (tex.width(), tex.height());
        self.clear_draw(
            &command,
            &view,
            slice,
            &raw_pipeline,
            &state,
            state_at,
            0,
            (w, h),
        );
        if let Some(p) = colour_pipeline {
            self.clear_draw(&command, tex, slice, &p, &state, state_at, 16, (w, h));
        }
        self.gpu_checked.insert(at);
        self.colour_checks += 1;
        self.note(|| format!("      colour fast clear at {at:#x} checked on the GPU"));
        Some(())
    }

    /// A full-screen triangle into `slice` of `target`, drawn as many times as
    /// the indirect arguments at `state_at + args` say (0 or 1).
    #[allow(clippy::too_many_arguments)]
    fn clear_draw(
        &self,
        command: &CommandBuffer,
        target: &Texture,
        slice: usize,
        pipeline: &RenderPipeline,
        state: &Buffer,
        state_at: usize,
        args: usize,
        (w, h): (usize, usize),
    ) {
        let pass = MTLRenderPassDescriptor::new();
        let att = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        att.setTexture(Some(target));
        att.setSlice(slice);
        att.setLoadAction(MTLLoadAction::Load);
        att.setStoreAction(MTLStoreAction::Store);
        let Some(enc) = command.renderCommandEncoderWithDescriptor(&pass) else {
            return;
        };
        enc.setRenderPipelineState(pipeline);
        enc.setCullMode(MTLCullMode::None);
        enc.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: w as f64,
            height: h as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        unsafe {
            enc.setFragmentBuffer_offset_atIndex(Some(state), state_at, 0);
            enc.drawPrimitives_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::Triangle,
                state,
                state_at + args,
            );
        }
        enc.endEncoding();
    }

    fn clear_pipeline(
        &mut self,
        helpers: &ClearHelpers,
        format: MTLPixelFormat,
        raw: bool,
    ) -> Option<Retained<RenderPipeline>> {
        if let Some(p) = self.clear_pipelines.get(&(format.0, raw)) {
            return Some(p.clone());
        }
        let desc = MTLRenderPipelineDescriptor::new();
        desc.setVertexFunction(Some(&helpers.vertex));
        desc.setFragmentFunction(Some(if raw { &helpers.raw } else { &helpers.colour }));
        unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }.setPixelFormat(format);
        let p = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .ok()?;
        self.clear_pipelines.insert((format.0, raw), p.clone());
        Some(p)
    }

    fn clear_helpers(&mut self) -> Option<ClearHelpers> {
        if self.clear_helpers.is_none() {
            let lib = self
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(CLEAR_MSL), None)
                .ok()?;
            let f = |name: &str| lib.newFunctionWithName(&NSString::from_str(name));
            let kernel = |name: &str| {
                self.device
                    .newComputePipelineStateWithFunction_error(&*f(name)?)
                    .ok()
            };
            self.clear_helpers = Some(ClearHelpers {
                cmask: kernel("cmask_check")?,
                dcc: kernel("dcc_check")?,
                vertex: f("clear_vs")?,
                raw: f("clear_raw_fs")?,
                colour: f("clear_colour_fs")?,
            });
        }
        self.clear_helpers.clone()
    }
}

/// The programs of the GPU-side colour fast-clear check.
#[derive(Clone)]
pub(super) struct ClearHelpers {
    cmask: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    dcc: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    vertex: Retained<Function>,
    raw: Retained<Function>,
    colour: Retained<Function>,
}

const CLEAR_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;
struct Clear { uint raw[4]; uint colour[4]; uint words[2]; uint pad[2]; float4 rgba; };
struct Params { uint words[2]; uint bpp; uint pad; };
static void begin(device Clear* c) {
    for (uint i = 0; i < 4; i++) { c->raw[i] = i == 1 ? 1u : 0u; c->colour[i] = i == 1 ? 1u : 0u; }
}
static void set_words(device Clear* c, uint a, uint b, uint bpp) {
    uint mask = bpp >= 4 ? 0xffffffffu : (1u << (8u * bpp)) - 1u;
    c->words[0] = a & mask;
    c->words[1] = bpp == 8 ? b : 0u;
    c->raw[0] = 3u;
}
kernel void cmask_check(device uint* word [[buffer(0)]], device Clear* c [[buffer(1)]],
                        constant Params& p [[buffer(2)]]) {
    begin(c);
    uint w = *word;
    if ((w & 0xfu) == 0u) {
        set_words(c, p.words[0], p.words[1], p.bpp);
        *word = w | 0xfu;
    }
}
kernel void dcc_check(device uint* word [[buffer(0)]], device Clear* c [[buffer(1)]],
                      constant Params& p [[buffer(2)]], device const uint* first [[buffer(3)]]) {
    begin(c);
    uint code = *word;
    float4 rgba;
    switch (code) {
        case 0x00000000u: rgba = float4(0, 0, 0, 0); break;
        case 0x40404040u: rgba = float4(0, 0, 0, 1); break;
        case 0x80808080u: rgba = float4(1, 1, 1, 0); break;
        case 0xC0C0C0C0u: rgba = float4(1, 1, 1, 1); break;
        case 0x20202020u: set_words(c, p.words[0], p.words[1], p.bpp); *word = 0xffffffffu; return;
        case 0x10101010u: set_words(c, first[0], p.bpp == 8 ? first[1] : 0u, p.bpp); *word = 0xffffffffu; return;
        default: return;
    }
    c->rgba = rgba;
    c->colour[0] = 3u;
    *word = 0xffffffffu;
}
vertex float4 clear_vs(uint id [[vertex_id]]) {
    float2 q = float2(float((id << 1) & 2u), float(id & 2u));
    return float4(q * 2.0 - 1.0, 0.0, 1.0);
}
fragment uint4 clear_raw_fs(device const Clear& c [[buffer(0)]]) {
    return uint4(c.words[0], c.words[1], 0u, 0u);
}
fragment float4 clear_colour_fs(device const Clear& c [[buffer(0)]]) {
    return c.rgba;
}
"#;
