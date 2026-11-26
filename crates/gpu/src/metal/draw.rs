//! Draws: render passes and the draw path.

use super::*;

const PIXEL_SLOTS: usize = 1 << 14;

/// The longest triangle fan drawn.
const FAN_MAX: usize = 1 << 16;

/// An open render pass: its encoder, attachment addresses, and color
/// textures and depth texture and slice (for `gpusnap`).
pub(super) type OpenPass = (
    Retained<RenderEncoder>,
    Vec<u64>,
    Vec<Retained<Texture>>,
    Option<(Retained<Texture>, usize)>,
);

/// Converts a 16-bit or 32-bit indexed triangle fan into a 32-bit triangle list,
/// honoring primitive restart when `restart` is `Some`.
fn expand_indexed_triangle_fan(bytes: &[u8], wide: bool, restart: Option<u32>) -> Vec<u32> {
    let mut out = Vec::new();
    let mut first: Option<u32> = None;
    let mut prev: Option<u32> = None;
    let mut push = |idx: u32| {
        if restart == Some(idx) {
            first = None;
            prev = None;
            return;
        }
        match (first, prev) {
            (None, _) => first = Some(idx),
            (Some(_), None) => prev = Some(idx),
            (Some(a), Some(b)) => {
                out.push(a);
                out.push(b);
                out.push(idx);
                prev = Some(idx);
            }
        }
    };
    if wide {
        for chunk in bytes.as_chunks::<4>().0 {
            push(u32::from_le_bytes(*chunk));
        }
    } else {
        for chunk in bytes.as_chunks::<2>().0 {
            push(u16::from_le_bytes(*chunk) as u32);
        }
    }
    out
}

/// The colour targets a draw attaches, in attachment order: those it writes and
/// those its pixel program exports to before the last one it writes, so export
/// `k` lands on attachment `k`. A masked export after that writes nothing, and
/// its target (often left bound from an earlier pass) would clip the pass to
/// its size.
pub(super) fn attached_targets(mask: u32, exports: u32) -> impl Iterator<Item = usize> {
    let set = |bits: u32, n: usize| bits >> (4 * n) & 0xF != 0;
    let end = (0..8).rev().find(|&n| set(mask, n)).map_or(0, |n| n + 1);
    (0..8).filter(move |&n| set(mask, n) || (set(exports, n) && n < end))
}

/// Derives fallback target dimensions from the scissor and viewport registers.
fn viewport_target_size(c: &[u32]) -> (u32, u32) {
    let (_, _, x1, y1) = regs::scissor(c);
    let vp = Viewport::read(c);
    let w = x1.min((vp.x_offset + vp.x_scale.abs()).ceil().max(1.0) as u32);
    let h = y1.min((vp.y_offset + vp.y_scale.abs()).ceil().max(1.0) as u32);
    (w.clamp(1, 16384), h.clamp(1, 16384))
}

impl Metal {
    /// The render encoder for color targets `targets`, reusing the open pass when it matches.
    pub(super) fn pass(
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
        let every = diag::config().gpu.commit_draws;
        if every > 0 && self.uncommitted_draws >= every {
            self.commit();
        }
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
        if self.tracing() && diag::config().gpu.trace_pixels && self.pixels.is_none() {
            self.pixels = self
                .device
                .newBufferWithLength_options(8 * PIXEL_SLOTS, MTLResourceOptions::StorageModeShared)
                .map(|b| (b, 0));
        }
        if let Some((b, _)) = &self.pixels {
            pass.setVisibilityResultBuffer(Some(b));
        }
        let enc = command.renderCommandEncoderWithDescriptor(&pass)?;
        if let Some(f) = self.wait_memory() {
            enc.waitForFence_beforeStages(f, MTLRenderStages::Vertex);
        }
        if crate::timing::labels() {
            let what: Vec<String> = addresses.iter().map(|a| format!("{a:#x}")).collect();
            enc.setLabel(Some(&NSString::from_str(&format!(
                "pass {}",
                what.join(" ")
            ))));
        }
        let colors = targets.iter().map(|t| t.1.clone()).collect();
        let depth_texture = depth
            .as_ref()
            .map(|(_, t, _, slice)| (t.clone(), *slice as usize));
        self.pass = Some((enc.clone(), addresses, colors, depth_texture));
        Some(enc)
    }

    /// An index buffer that draws a triangle fan of `count` vertices as a
    /// triangle list (0, i, i + 1); Metal has no fans.
    fn fan_indices(&mut self, count: u32) -> Result<Retained<Buffer>, &'static str> {
        if count as usize > FAN_MAX {
            return Err("triangle fan too long");
        }
        if self.fan.is_none() {
            let indices: Vec<u32> = (1..FAN_MAX as u32 - 1)
                .flat_map(|i| [0, i, i + 1])
                .collect();
            let ptr = NonNull::new(indices.as_ptr() as *mut std::ffi::c_void).ok_or("fan")?;
            self.fan = unsafe {
                self.device.newBufferWithBytes_length_options(
                    ptr,
                    4 * indices.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            };
        }
        self.fan.clone().ok_or("fan index buffer")
    }

    pub(super) fn try_draw(
        &mut self,
        host: &dyn Host,
        state: &State,
        draw: Draw,
    ) -> Result<(), &'static str> {
        let (c, s, u) = (state.context, state.sh, state.uconfig);
        let ngg = c[context::VGT_SHADER_STAGES_EN] & (1 << 13) != 0;
        let (vs_lo, vs_rsrc2, vs_user) = if ngg {
            (
                sh::SPI_SHADER_PGM_LO_ES,
                sh::SPI_SHADER_PGM_RSRC2_GS,
                sh::SPI_SHADER_USER_DATA_GS_0,
            )
        } else {
            (
                sh::SPI_SHADER_PGM_LO_VS,
                sh::SPI_SHADER_PGM_RSRC2_VS,
                sh::SPI_SHADER_USER_DATA_VS_0,
            )
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
        if state.color_control_written && matches!(c[context::CB_COLOR_CONTROL] >> 4 & 7, 2 | 5 | 6)
        {
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
        let ps_data =
            &s[sh::SPI_SHADER_USER_DATA_PS_0..sh::SPI_SHADER_USER_DATA_PS_0 + sh::USER_DATA_COUNT];
        let col_format = export_formats(c);
        let (logic_op, logic_targets) = logic_targets(c);
        let swizzles = std::array::from_fn(|n| {
            ColorTarget::read(c, n).map_or(0, |t| export_swizzle(t.format))
        });
        self.note(|| {
            format!(
                "vertex {vs_addr:#x} pixel {ps_addr:#x} col_format {:#x}",
                c[context::SPI_SHADER_COL_FORMAT]
            )
        });
        let vertex = self
            .typed_program(
                host,
                state,
                Stage::Vertex,
                vs_addr,
                vs_users,
                ngg,
                vs_data,
                Variant::default(),
            )
            .ok_or("vertex program")?;
        let depth_bound = DepthTarget::read(c).is_some() && {
            let d = DepthState::read(c);
            d.depth_test() || d.stencil() || c[context::DB_RENDER_CONTROL] & 3 != 0
        };
        let pixel = self
            .typed_program(
                host,
                state,
                Stage::Pixel,
                ps_addr,
                ps_users,
                false,
                ps_data,
                Variant {
                    col_format,
                    drop_depth_export: !depth_bound,
                    half_targets: half_targets(c),
                    dual_source: dual_source(c),
                    blends: blends(c),
                    logic_op,
                    logic_targets,
                    swizzles,
                    ..Variant::default()
                },
            )
            .ok_or("pixel program")?;
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
        let prim = u[uconfig::VGT_PRIMITIVE_TYPE];
        // 7 (`DI_PT_2D_RECTANGLE`) as titles use it, 17 `DI_PT_RECTLIST`.
        let rect = matches!(prim, 7 | 17);
        // PA_SU_POINT_SIZE: half the height in 12.4 fixed point.
        let point_size = (prim == 1)
            .then(|| (f32::from((c[context::PA_SU_POINT_SIZE] & 0xFFFF) as u16) / 8.0).to_bits());
        let gl_clip = c[context::PA_CL_CLIP_CNTL] & (1 << 19) == 0;
        let form = VertexForm {
            rect,
            gl_clip,
            point_size,
        };
        let vertex_function = self.vertex_function(&vertex, &pixel, form);

        let mask = c[context::CB_TARGET_MASK];
        let mut targets = Vec::new();
        let mut key = PipelineKey {
            vertex: vertex.id,
            pixel: pixel.id,
            formats: [0; 8],
            write_masks: [0; 8],
            depth: 0,
            stencil: false,
            form,
        };
        for n in attached_targets(mask, col_format) {
            let write = (mask >> (4 * n)) & 0xF;
            let Some(t) = ColorTarget::read(c, n) else {
                continue;
            };
            self.note(|| format!("mrt{n} {t:x?}"));
            let (tex, format) = self.target(host, &t).ok_or("color target format")?;
            self.note(|| format!("      texture {:p}", Retained::as_ptr(&tex)));
            self.fast_clear(host, &t, &tex);
            let (tex, address) = if t.mip_level == 0 {
                (tex, t.address)
            } else {
                let level = t.mip_level as usize;
                if level >= tex.mipmapLevelCount() {
                    return Err("draw to a mip level above 0");
                }
                let view = unsafe {
                    tex.newTextureViewWithPixelFormat_textureType_levels_slices(
                        tex.pixelFormat(),
                        tex.textureType(),
                        objc2_foundation::NSRange::new(level, 1),
                        objc2_foundation::NSRange::new(0, tex.arrayLength()),
                    )
                }
                .ok_or("mip level view")?;
                // Bits 40 and up keep the level's passes apart from level 0's.
                (view, t.address | (t.mip_level as u64) << 40)
            };
            key.formats[targets.len()] = format.0;
            // The write mask is in export order; Metal's is in channel order.
            let swizzle = export_swizzle(t.format);
            let write = if swizzle == 0 {
                write
            } else {
                (0..4).fold(0, |m, i| m | (write >> (swizzle >> (2 * i) & 3) & 1) << i)
            };
            key.write_masks[targets.len()] = write_mask(write) as u8;
            let slice = if tex.textureType() == MTLTextureType::Type2DArray {
                t.slice
            } else {
                0
            };
            targets.push((address, tex, slice));
        }
        let depth_state = DepthState::read(c);
        let wants_depth = depth_state.depth_test() || depth_state.stencil();
        let mut size = targets
            .first()
            .map(|t| (t.1.width() as u32, t.1.height() as u32));
        let mut depth = None;
        let mut clear = None;
        let mut gpu_htile = None;
        if let Some(d) = DepthTarget::read(c)
            && (wants_depth || c[context::DB_RENDER_CONTROL] & 3 != 0)
        {
            self.note(|| format!("depth {d:x?} test {wants_depth}"));
            let (dw, dh) = size.unwrap_or_else(|| {
                if d.width != 0 && d.height != 0 {
                    (d.width, d.height)
                } else if let Some(existing) = self.depths.get(&d.address) {
                    (existing.width() as u32, existing.height() as u32)
                } else {
                    viewport_target_size(c)
                }
            });
            if let Some((tex, format, fresh)) = self.depth(&d, dw, dh) {
                if d.stencil && d.stencil_address != 0 {
                    self.stencils.insert(d.stencil_address, d.address);
                }
                key.depth = format.0;
                key.stencil = d.stencil;
                let (zs, zo) = regs::depth_range(c, gl_clip);
                let values = (
                    (f32::from_bits(c[context::DB_DEPTH_CLEAR]) * zs + zo).clamp(0.0, 1.0),
                    c[context::DB_STENCIL_CLEAR] & 0xFF,
                );
                let clearing = c[context::DB_RENDER_CONTROL] & 3 != 0
                    || match self.htile(host, c, &d) {
                        Htile::Cleared => true,
                        Htile::Kept => false,
                        Htile::OnGpu(at) => {
                            gpu_htile = Some((at, values));
                            false
                        }
                    };
                if clearing || fresh {
                    let z = if clearing { values.0 } else { 1.0 };
                    clear = Some((z, values.1));
                }
                depth = Some((d.address, tex, d.stencil, d.slice));
            }
        }
        if targets.is_empty() && depth.is_none() {
            // DB_RENDER_CONTROL bits 2..=4 are depth/stencil copy and HTILE resummarize
            // metadata passes (AMD GFX10); Metal keeps depth uncompressed, so only any
            // pending HTILE fast clear needs to land.
            if c[context::DB_RENDER_CONTROL] & 0x1C != 0 {
                if let Some(d) = DepthTarget::read(c) {
                    let (dw, dh) = if d.width != 0 && d.height != 0 {
                        (d.width, d.height)
                    } else if let Some(existing) = self.depths.get(&d.address) {
                        (existing.width() as u32, existing.height() as u32)
                    } else {
                        viewport_target_size(c)
                    };
                    if let Some((tex, _, _)) = self.depth(&d, dw, dh) {
                        let (zs, zo) = regs::depth_range(c, gl_clip);
                        let values = (
                            (f32::from_bits(c[context::DB_DEPTH_CLEAR]) * zs + zo).clamp(0.0, 1.0),
                            c[context::DB_STENCIL_CLEAR] & 0xFF,
                        );
                        match self.htile(host, c, &d) {
                            Htile::Cleared => {
                                let _ = self.pass(
                                    &[],
                                    Some((d.address, tex, d.stencil, d.slice)),
                                    Some(values),
                                );
                            }
                            Htile::OnGpu(at) => {
                                let _ = self.htile_on_gpu(
                                    host,
                                    at,
                                    &[],
                                    &(d.address, tex, d.stencil, d.slice),
                                    None,
                                    values,
                                );
                            }
                            Htile::Kept => {}
                        }
                    }
                }
                return Err("metadata pass");
            }
            let stores = pixel.translated.writes_memory
                || pixel.translated.resources.iter().any(
                    |r| matches!(r, Resource::Texture { access, .. } if *access != Access::Sample),
                );
            if !stores {
                return Err("no color target");
            }
            let (w, h) = viewport_target_size(c);
            let tex = self.blank_target(w, h)?;
            key.formats[0] = tex.pixelFormat().0;
            size = Some((tex.width() as u32, tex.height() as u32));
            targets.push((BLANK_TARGET, tex, 0));
        }
        if let (Some((at, values)), Some(d)) = (gpu_htile, &depth) {
            self.htile_on_gpu(host, at, &targets, d, clear, values)
                .ok_or("HTILE check")?;
            clear = None;
        }
        let pipeline = self.pipeline(&key, vertex_function.clone(), &pixel);
        let mut redrawn = !vertex.translated.writes_memory
            && !pixel.translated.writes_memory
            && !self.wait_for_compiles
            && !diag::config().gpu.compile_wait;
        let surfaces: Vec<_> = targets
            .iter()
            .map(|t| (t.0, &t.1))
            .chain(depth.as_ref().map(|d| (d.0, &d.1)))
            .map(|(address, t)| (address, t.pixelFormat().0, t.width()))
            .collect();
        for surface in surfaces {
            let (first, last) = self
                .drawn
                .entry(surface)
                .or_insert((self.epoch, self.epoch));
            redrawn &= self.shown.contains_key(&surface.0)
                || (*first < self.epoch && *last + 2 >= self.epoch);
            *last = self.epoch;
        }
        let mut ready = pipeline.poll();
        if ready.is_none() && redrawn && !self.compile_budget.is_zero() {
            let t = std::time::Instant::now();
            ready = pipeline.wait_for(self.compile_budget.min(std::time::Duration::from_millis(4)));
            self.compile_budget = self.compile_budget.saturating_sub(t.elapsed());
            crate::timing::add_raw(
                crate::timing::Part::CompileWait,
                t.elapsed().as_nanos() as u64,
            );
        }
        let pipeline = match ready {
            Some(p) => p,
            // A target drawn every frame shows this draw from a later frame on.
            None if redrawn => {
                if vertex.frame < self.epoch || pixel.frame < self.epoch {
                    vertex_function.hurry();
                    pixel.function.hurry();
                }
                // The clear rides on the pass and would be lost with the draw.
                if clear.is_some() {
                    self.pass(&targets, depth, clear).ok_or("render pass")?;
                }
                if tracing::enabled!(target: "gpu::compile", tracing::Level::DEBUG) {
                    tracing::debug!(
                        target: "gpu::compile",
                        "{:.3} s skipped draw in frame {} vertex {:#x} pixel {:#x} targets {:x?} count {:?}",
                        log_clock(),
                        self.epoch,
                        vertex.address,
                        pixel.address,
                        targets.iter().map(|t| (t.0, t.1.width(), t.1.height())).collect::<Vec<_>>(),
                        draw
                    );
                }
                return Err("compiling");
            }
            None => {
                let t = std::time::Instant::now();
                let p = pipeline.wait_blocking();
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
        if self.snap.is_some() && diag::config().gpu.snap_inputs {
            self.snap_inputs(ps_addr, &ps_bound);
        }
        let fan = u[uconfig::VGT_PRIMITIVE_TYPE] == 5;
        let index = match draw {
            Draw::Indexed { count, address, .. } if !rect && fan => {
                if count < 3 {
                    return Ok(());
                }
                let wide = u[uconfig::VGT_INDEX_TYPE] & 3 == 1;
                let elem = if wide { 4u64 } else { 2u64 };
                let len = count as u64 * elem;
                self.settle_for(address, len, "indices");
                let mut raw = vec![0u8; len as usize];
                if !host.read(address, &mut raw) {
                    return Err("index buffer");
                }
                let restart = (c[context::VGT_MULTI_PRIM_IB_RESET_EN] & 1 != 0).then(|| {
                    let r = c[context::VGT_MULTI_PRIM_IB_RESET_INDX];
                    if wide { r } else { r & 0xFFFF }
                });
                let expanded = expand_indexed_triangle_fan(&raw, wide, restart);
                if expanded.is_empty() {
                    return Ok(());
                }
                let out_bytes = expanded.len() * 4;
                let (buf, offset) = self.scratch(out_bytes).ok_or("fan index buffer")?;
                // SAFETY: `scratch` returns fresh shared memory of at least `out_bytes` bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        expanded.as_ptr().cast::<u8>(),
                        buf.contents().as_ptr().cast::<u8>().add(offset),
                        out_bytes,
                    );
                }
                Some(((buf, offset), true, expanded.len()))
            }
            Draw::Indexed { count, address, .. } if !rect => {
                let wide = u[uconfig::VGT_INDEX_TYPE] & 3 == 1;
                let size = if wide { 4 } else { 2 };
                let len = count as u64 * size;
                let bound = if address % 4 == 0 {
                    self.window(host, address, len).ok_or("index buffer")?
                } else {
                    // Metal needs index buffer offsets to be multiples of 4.
                    self.settle_for(address, len, "indices");
                    let (_, hi) = host.extent(address).ok_or("index buffer")?;
                    let src = host.host_pointer(address).ok_or("index buffer")?;
                    let n = (len as usize).min((hi - address) as usize);
                    let (buf, offset) = self.scratch(len.max(4) as usize).ok_or("index buffer")?;
                    // SAFETY: `n` bytes from `address` lie in host-backed guest memory, and
                    // the scratch range is fresh shared memory no GPU work uses yet.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src,
                            buf.contents().as_ptr().cast::<u8>().add(offset),
                            n,
                        )
                    };
                    (buf, offset)
                };
                Some((bound, wide, count as usize))
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
            enc.insertDebugSignpost(&NSString::from_str(&format!(
                "draw {vs_addr:#x} {ps_addr:#x}"
            )));
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
            originY: (vp.y_offset + vp.y_scale) as f64,
            width: (2.0 * vp.x_scale) as f64,
            height: (-2.0 * vp.y_scale) as f64,
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
        if pixel.translated.reads_blend_color {
            let color = blend_color(c);
            Slot::Fragment(&enc).bytes(
                color.as_ptr().cast(),
                std::mem::size_of_val(&color),
                binding::BLEND_COLOR as usize,
            );
        }
        for (slot, t) in &ps_bound.textures {
            let (ptr, w, h) = (Retained::as_ptr(t) as usize, t.width(), t.height());
            self.note(|| format!("bound t{slot} {ptr:#x} {w}x{h}"));
        }
        let primitive = match u[uconfig::VGT_PRIMITIVE_TYPE] {
            1 => MTLPrimitiveType::Point,
            2 => MTLPrimitiveType::Line,
            3 => MTLPrimitiveType::LineStrip,
            4 | 5 | 7 | 17 => MTLPrimitiveType::Triangle,
            6 => MTLPrimitiveType::TriangleStrip,
            p => {
                self.note(|| format!("primitive type {p:#x}"));
                diag::problems::record(
                    diag::problems::Kind::UnsupportedFormat,
                    format!("primitive type {p:#x}"),
                    format!("vs {vs_addr:#x} ps {ps_addr:#x}"),
                );
                return Err("primitive type");
            }
        };
        let start = u[uconfig::GE_INDX_OFFSET];
        let counted = match &mut self.pixels {
            Some((_, next)) if self.trace.is_some() && *next < PIXEL_SLOTS => {
                enc.setVisibilityResultMode_offset(MTLVisibilityResultMode::Counting, 8 * *next);
                *next += 1;
                Some(*next - 1)
            }
            _ => None,
        };
        if let Some(slot) = counted {
            self.note(|| format!("pixels slot {slot}"));
        }
        match draw {
            // gen5_vs_rect takes the index offset as its base instance.
            Draw::Auto { count, instances } if rect => unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    primitive,
                    0,
                    (count / 3 * 6) as usize,
                    instances.max(1) as usize,
                    start as usize,
                );
            },
            Draw::Indexed { .. } if rect => return Err("indexed rectangle list"),
            Draw::Auto { count, instances } if fan => {
                if count < 3 {
                    return Ok(());
                }
                let indices = self.fan_indices(count)?;
                unsafe {
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive,
                        3 * (count as usize - 2),
                        MTLIndexType::UInt32,
                        &indices,
                        0,
                        instances.max(1) as usize,
                        start as isize,
                        0,
                    );
                }
            }
            Draw::Auto { count, instances } => unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(
                    primitive,
                    start as usize,
                    count as usize,
                    instances.max(1) as usize,
                );
            },
            Draw::Indexed { instances, .. } => {
                let ((buf, offset), wide, index_count) = index.ok_or("index buffer")?;
                unsafe {
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive,
                        index_count,
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
        if counted.is_some() {
            enc.setVisibilityResultMode_offset(MTLVisibilityResultMode::Disabled, 0);
        }
        self.pass_memory |= vertex.translated.writes_memory || pixel.translated.writes_memory;
        self.note_writes(host, &vertex.translated, vs_data);
        self.note_writes(host, &pixel.translated, ps_data);
        self.pending_writes |= vertex.translated.writes_memory || pixel.translated.writes_memory;
        Ok(())
    }
}

#[cfg(test)]
mod fan_tests {
    use super::expand_indexed_triangle_fan;

    #[test]
    fn expands_16bit_and_32bit_indexed_triangle_fans_with_restart() {
        let idx16: Vec<u8> = [10u16, 11, 12, 13, 0xFFFF, 20, 21, 0xFFFF, 30, 31, 32]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(
            expand_indexed_triangle_fan(&idx16, false, Some(0xFFFF)),
            vec![10, 11, 12, 10, 12, 13, 30, 31, 32]
        );

        let idx32: Vec<u8> = [0u32, 1, 2, 3, 4]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(
            expand_indexed_triangle_fan(&idx32, true, None),
            vec![0, 1, 2, 0, 2, 3, 0, 3, 4]
        );

        let short: Vec<u8> = [0u16, 1].into_iter().flat_map(u16::to_le_bytes).collect();
        assert!(expand_indexed_triangle_fan(&short, false, None).is_empty());
    }
}
