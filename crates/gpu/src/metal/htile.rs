//! Depth fast clears read from HTILE, on the CPU or on the GPU.

use super::*;

impl Metal {
    /// Checks `d`'s first HTILE dword for a ZMask-0 fast clear (radv layout)
    /// and marks it expanded (0xF), or defers to [`Htile::OnGpu`] when pending GPU work may write it.
    pub(super) fn htile(&mut self, host: &dyn Host, c: &[u32], d: &DepthTarget) -> Htile {
        const TILE_SURFACE_ENABLE: u32 = 1 << 29;
        if c[context::DB_Z_INFO] & TILE_SURFACE_ENABLE == 0 {
            return Htile::Kept;
        }
        let base = ((c[context::DB_HTILE_DATA_BASE] as u64)
            | ((c[context::DB_HTILE_DATA_BASE_HI] as u64 & 0xFF) << 32))
            << 8;
        if base == 0 {
            return Htile::Kept;
        }
        let (w, h) = if d.width != 0 {
            (d.width, d.height)
        } else {
            (1, 1)
        };
        let slice_bytes =
            (w.div_ceil(8) as u64 * h.div_ceil(8) as u64 * 4).next_multiple_of(64 << 10);
        let at = base + d.slice as u64 * slice_bytes;
        if self.gpu_checked.contains(&at) {
            return Htile::Kept;
        }
        if self.written.covers(at, 4) && !diag::config().gpu.htile_cpu {
            return Htile::OnGpu(at);
        }
        self.settle_for(at, 4, "HTILE");
        let mut word = [0u8; 4];
        if !host.read(at, &mut word) || u32::from_le_bytes(word) & 0xF != 0 {
            return Htile::Kept;
        }
        host.write(at, &(u32::from_le_bytes(word) | 0xF).to_le_bytes());
        self.note(|| {
            format!(
                "      depth {:#x} slice {} fast-cleared through HTILE at {at:#x}",
                d.address, d.slice
            )
        });
        Htile::Cleared
    }

    /// Checks the HTILE word at `at` on the GPU, after the work that may write
    /// it: a one-thread program reads it, marks it expanded and sets the vertex
    /// count of an indirect full-screen draw opening the pass that writes the clear values.
    pub(super) fn htile_on_gpu(
        &mut self,
        host: &dyn Host,
        at: u64,
        targets: &[(u64, Retained<Texture>, u32)],
        depth: &(u64, Retained<Texture>, bool, u32),
        clear: Option<(f32, u32)>,
        values: (f32, u32),
    ) -> Option<()> {
        let (word, word_at) = self.memory_buffer(host, at, 4)?;
        let (flag, flag_at) = self.scratch(16)?;
        let helpers = self.htile_helpers()?;
        self.end_pass();
        let command = self.command();
        let enc = command.computeCommandEncoder()?;
        enc.setComputePipelineState(&helpers.check);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&word), word_at, 0);
            enc.setBuffer_offset_atIndex(Some(&flag), flag_at, 1);
        }
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        enc.endEncoding();
        let (dw, dh) = (depth.1.width(), depth.1.height());
        let targets = if targets
            .iter()
            .any(|t| t.1.width() < dw || t.1.height() < dh)
        {
            &[]
        } else {
            targets
        };
        // The clear draw covers the whole depth target only without smaller colour targets.
        let enc = self.pass(targets, Some(depth.clone()), clear)?;
        let mut formats = [0usize; 8];
        for (f, t) in formats.iter_mut().zip(targets) {
            *f = t.1.pixelFormat().0;
        }
        let stencil = depth.2;
        let key = (formats, depth.1.pixelFormat().0, stencil);
        let pipeline = match self.htile_pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let desc = MTLRenderPipelineDescriptor::new();
                desc.setVertexFunction(Some(&helpers.vertex));
                desc.setFragmentFunction(Some(&helpers.fragment));
                for (i, &raw) in formats.iter().enumerate().filter(|f| *f.1 != 0) {
                    let att = unsafe { desc.colorAttachments().objectAtIndexedSubscript(i) };
                    att.setPixelFormat(MTLPixelFormat(raw));
                    att.setWriteMask(MTLColorWriteMask(0));
                }
                desc.setDepthAttachmentPixelFormat(MTLPixelFormat(key.1));
                if stencil {
                    desc.setStencilAttachmentPixelFormat(MTLPixelFormat(key.1));
                }
                let p = self
                    .device
                    .newRenderPipelineStateWithDescriptor_error(&desc)
                    .ok()?;
                self.htile_pipelines.insert(key, p.clone());
                p
            }
        };
        let (w, h) = targets
            .iter()
            .map(|t| &t.1)
            .chain([&depth.1])
            .fold((usize::MAX, usize::MAX), |(w, h), t| {
                (w.min(t.width()), h.min(t.height()))
            });
        enc.setRenderPipelineState(&pipeline);
        enc.setDepthStencilState(Some(&helpers.states[stencil as usize]));
        enc.setStencilReferenceValue(values.1);
        enc.setCullMode(MTLCullMode::None);
        enc.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: w as f64,
            height: h as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        enc.setScissorRect(MTLScissorRect {
            x: 0,
            y: 0,
            width: w,
            height: h,
        });
        unsafe {
            enc.setFragmentBytes_length_atIndex(
                NonNull::from(&values.0).cast(),
                std::mem::size_of::<f32>(),
                0,
            );
            enc.drawPrimitives_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::Triangle,
                &flag,
                flag_at,
            );
        }
        enc.setDepthStencilState(Some(&helpers.states[2]));
        self.gpu_checked.insert(at);
        self.htile_checks += 1;
        self.note(|| {
            format!(
                "      depth {:#x} HTILE at {at:#x} checked on the GPU",
                depth.0
            )
        });
        Some(())
    }

    pub(super) fn htile_helpers(&mut self) -> Option<HtileHelpers> {
        if self.htile_helpers.is_none() {
            let lib = self
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(HTILE_MSL), None)
                .ok()?;
            let f = |name: &str| lib.newFunctionWithName(&NSString::from_str(name));
            let kernel = f("htile_check")?;
            let check = self
                .device
                .newComputePipelineStateWithFunction_error(&kernel)
                .ok()?;
            let state = |write: bool, stencil: bool| {
                let d = MTLDepthStencilDescriptor::new();
                if write {
                    d.setDepthCompareFunction(MTLCompareFunction::Always);
                    d.setDepthWriteEnabled(true);
                }
                if stencil {
                    let f = MTLStencilDescriptor::new();
                    f.setStencilCompareFunction(MTLCompareFunction::Always);
                    f.setDepthStencilPassOperation(MTLStencilOperation::Replace);
                    d.setFrontFaceStencil(Some(&f));
                    d.setBackFaceStencil(Some(&f));
                }
                self.device.newDepthStencilStateWithDescriptor(&d)
            };
            self.htile_helpers = Some(HtileHelpers {
                check,
                vertex: f("htile_vs")?,
                fragment: f("htile_fs")?,
                states: [
                    state(true, false)?,
                    state(true, true)?,
                    state(false, false)?,
                ],
            });
        }
        self.htile_helpers.clone()
    }
}

/// What a draw's HTILE word says (see [`Metal::htile`]).
pub(super) enum Htile {
    /// Fast-cleared: the pass clears depth and stencil.
    Cleared,
    Kept,
    /// Encoded work may write the word at this address, so the GPU checks it.
    OnGpu(u64),
}

/// The programs and depth states of the GPU-side HTILE check.
#[derive(Clone)]
pub(super) struct HtileHelpers {
    pub(super) check: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) vertex: Retained<Function>,
    fragment: Retained<Function>,
    /// Writing depth, writing depth and stencil, and leaving both alone.
    states: [Retained<ProtocolObject<dyn MTLDepthStencilState>>; 3],
}

const HTILE_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;
kernel void htile_check(device uint* word [[buffer(0)]], device uint* draw [[buffer(1)]]) {
    uint w = *word;
    bool cleared = (w & 0xfu) == 0u;
    draw[0] = cleared ? 3u : 0u;
    draw[1] = 1u;
    draw[2] = 0u;
    draw[3] = 0u;
    if (cleared) *word = w | 0xfu;
}
vertex float4 htile_vs(uint id [[vertex_id]]) {
    float2 p = float2(float((id << 1) & 2u), float(id & 2u));
    return float4(p * 2.0 - 1.0, 0.0, 1.0);
}
struct HtileOut { float depth [[depth(any)]]; };
fragment HtileOut htile_fs(constant float& z [[buffer(0)]]) {
    return { z };
}
"#;
