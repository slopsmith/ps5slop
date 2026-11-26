//! Translated programs, their Metal functions and pipelines, compiled on the compile threads.

use super::*;

/// A vertex program, whether it draws rectangles or uses OpenGL clip space,
/// and the attributes it must add.
pub(super) type VariantKey = (u64, VertexForm, Vec<u32>, Vec<u32>);

/// How a draw's vertex function differs from its program's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct VertexForm {
    /// Rectangle lists: three vertices per rectangle, the fourth corner made.
    pub(super) rect: bool,
    /// OpenGL clip space (z from -w to w).
    pub(super) gl_clip: bool,
    /// Point lists: the points' diameter in pixels, as f32 bits.
    pub(super) point_size: Option<u32>,
}

/// Time per frame draws to redrawn targets may wait for their pipelines
/// before they are skipped.
pub(super) const COMPILE_BUDGET: std::time::Duration = std::time::Duration::from_millis(8);

pub(super) struct Program {
    pub(super) address: u64,
    /// Unique per translation, so caches tell apart programs loaded at one address.
    pub(super) id: u64,
    pub(super) function: Arc<Task<Retained<Function>>>,
    /// The pipeline of a compute program, compiled as soon as it is translated.
    pub(super) compute: Option<Arc<Task<Retained<ComputePipeline>>>>,
    pub(super) translated: Translated,
    /// The `user(locnN)` locations a pixel program reads flat.
    flat: Vec<u32>,
    /// Texture slots declared `<int>`.
    pub(super) sint_textures: u32,
    /// The frame it was translated in.
    pub(super) frame: u64,
}

/// What a program is translated for besides its code and registers.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(super) struct Variant {
    /// `SPI_SHADER_COL_FORMAT`, with integer targets' exports made integer.
    pub(super) col_format: u32,
    pub(super) uint_textures: u32,
    pub(super) sint_textures: u32,
    /// Raw texture slots whose format Metal lacks (see `Request::coded_textures`).
    pub(super) coded_textures: u32,
    /// Compute: threads past a grid size read from a buffer do nothing.
    pub(super) grid_bounds: bool,
    /// Pixel: no depth target is bound, so a depth export is dropped.
    pub(super) drop_depth_export: bool,
    /// Pixel: `Request::half_targets`.
    pub(super) half_targets: u8,
    /// Pixel: `Request::dual_source`.
    pub(super) dual_source: bool,
    /// Pixel: `Request::blends`.
    pub(super) blends: [u32; 8],
    /// Pixel: `Request::logic_op` and `Request::logic_targets`.
    pub(super) logic_op: u8,
    pub(super) logic_targets: [u8; 8],
    /// Pixel: `Request::swizzles`.
    pub(super) swizzles: [u8; 8],
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct PipelineKey {
    pub(super) vertex: u64,
    pub(super) pixel: u64,
    pub(super) formats: [usize; 8],
    /// Each attachment's Metal write mask.
    pub(super) write_masks: [u8; 8],
    pub(super) depth: usize,
    pub(super) stencil: bool,
    pub(super) form: VertexForm,
}

/// The `user(locnN)` locations a pixel program reads flat (declared `uint4`).
fn flat_locations(msl: &str) -> Vec<u32> {
    msl.lines()
        .filter(|l| l.trim_start().starts_with("uint4 a") && l.ends_with("flat]];"))
        .filter_map(|l| {
            l.split("[[user(locn")
                .nth(1)?
                .split(')')
                .next()?
                .parse()
                .ok()
        })
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

impl Metal {
    pub(super) fn compile(&mut self, msl: &str, entry: &str) -> Option<Retained<Function>> {
        compile(&self.device, msl, entry, entry)
    }

    /// Compiles `msl` on the compile threads.
    fn compile_later(
        &self,
        msl: String,
        entry: String,
        label: String,
    ) -> Arc<Task<Retained<Function>>> {
        let device = self.device.clone();
        let queued = std::time::Instant::now();
        Task::spawn(move || {
            let start = std::time::Instant::now();
            let f = compile(&device, &msl, &entry, &label);
            log_compile("library", &label, msl.len(), queued, start, None);
            f
        })
    }

    /// The program at `address` as the bound descriptors need it: translated
    /// again with integer texture types when a T# has an integer format.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn typed_program(
        &mut self,
        host: &dyn Host,
        state: &State,
        stage: Stage,
        address: u64,
        user_sgprs: u32,
        ngg: bool,
        user: &[u32],
        base: Variant,
    ) -> Option<Arc<Program>> {
        let program = self.program(host, state, stage, address, user_sgprs, ngg, base)?;
        let mut variant = base;
        for r in &program.translated.resources {
            if let Resource::Texture {
                slot,
                ref source,
                view,
                ..
            } = *r
                && let Some(d) = self.descriptor(host, user, source, 8)
            {
                let image = Image::read(&d);
                if image.address == 0 || image.kind < 8 || self.depths.contains_key(&image.address)
                {
                    continue;
                }
                match image_format(image.format).map(|f| number(f.0)) {
                    Some(Number::Uint) => variant.uint_textures |= 1 << slot,
                    Some(Number::Sint) => variant.sint_textures |= 1 << slot,
                    Some(Number::Float) => {}
                    None if view == View::Raw && coded_format(image.format).is_some() => {
                        variant.coded_textures |= 1 << slot
                    }
                    None => {}
                }
            }
        }
        if variant == base {
            Some(program)
        } else {
            self.program(host, state, stage, address, user_sgprs, ngg, variant)
        }
    }

    /// The translated program at `address`, translated on first use.
    #[allow(clippy::too_many_arguments)]
    fn program(
        &mut self,
        host: &dyn Host,
        state: &State,
        stage: Stage,
        address: u64,
        user_sgprs: u32,
        ngg: bool,
        variant: Variant,
    ) -> Option<Arc<Program>> {
        let c = state.context;
        let cntl = &c[context::SPI_PS_INPUT_CNTL_0..context::SPI_PS_INPUT_CNTL_0 + 32];
        let rsrc1 = state.sh[match (stage, ngg) {
            (Stage::Pixel, _) => sh::SPI_SHADER_PGM_RSRC1_PS,
            (Stage::Compute, _) => sh::COMPUTE_PGM_RSRC1,
            (Stage::Vertex, true) => sh::SPI_SHADER_PGM_RSRC1_GS,
            (Stage::Vertex, false) => sh::SPI_SHADER_PGM_RSRC1_VS,
        }];
        let tmpring = if stage == Stage::Compute {
            state.sh[sh::COMPUTE_TMPRING_SIZE]
        } else {
            c[context::SPI_TMPRING_SIZE]
        };
        let scratch_wave_bytes = (tmpring >> 12 & 0x1FFF) << 10;
        let plain_float = {
            let e = diag::config().gpu.exact_float;
            [
                (e.nan, shader::PLAIN_NAN),
                (e.denorm, shader::PLAIN_DENORM),
                (e.round, shader::PLAIN_ROUND),
                (e.mods, shader::PLAIN_MODS),
            ]
            .iter()
            .filter(|(exact, _)| !*exact)
            .fold(0, |bits, (_, bit)| bits | bit)
        };
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (
                stage as u8,
                user_sgprs,
                ngg,
                variant,
                scratch_wave_bytes,
                rsrc1,
                plain_float,
            )
                .hash(&mut h);
            if stage == Stage::Pixel {
                (
                    cntl,
                    c[context::SPI_PS_INPUT_ENA],
                    c[context::SPI_PS_INPUT_ADDR],
                )
                    .hash(&mut h);
            }
            if stage == Stage::Compute {
                let s = &state.sh;
                (
                    s[sh::COMPUTE_PGM_RSRC2],
                    &s[sh::COMPUTE_NUM_THREAD_X..sh::COMPUTE_NUM_THREAD_X + 3],
                )
                    .hash(&mut h);
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
            let code: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
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
                threads: [0, 1, 2]
                    .map(|i| (state.sh[sh::COMPUTE_NUM_THREAD_X + i] & 0xFFFF).max(1)),
                uint_textures: variant.uint_textures,
                sint_textures: variant.sint_textures,
                coded_textures: variant.coded_textures,
                grid_bounds: variant.grid_bounds,
                drop_depth_export: variant.drop_depth_export,
                half_targets: variant.half_targets,
                dual_source: variant.dual_source,
                blends: variant.blends,
                logic_op: variant.logic_op,
                logic_targets: variant.logic_targets,
                swizzles: variant.swizzles,
                probe_flat: diag::config().gpu.probe_flat,
                filter_bilinear: diag::config().gpu.filter_bilinear,
                plain_float,
                scratch_wave_bytes,
                rsrc1,
            };
            if let Some(dir) = &diag::config().gpu.shader_dump {
                let path = dir.join(format!("{stage:?}-{address:x}-{key:016x}.bin"));
                let _ = std::fs::write(path, gfx10::capture::Capture::new(&request).to_bytes());
            }
            let translate_start = std::time::Instant::now();
            match shader::translate(&request) {
                Ok(mut translated) => {
                    if tracing::enabled!(target: "gpu::compile", tracing::Level::DEBUG) {
                        tracing::debug!(
                            target: "gpu::compile",
                            "{:.3} s translated {stage:?} {address:#x} key {key:016x} in {:.2} ms, {} bytes of MSL, {} queued",
                            log_clock(),
                            translate_start.elapsed().as_secs_f64() * 1e3,
                            translated.msl.len(),
                            crate::compile::queued()
                        );
                    }
                    if let Some(dir) = &diag::config().gpu.msl_override
                        && let Ok(msl) = std::fs::read_to_string(
                            dir.join(format!("{stage:?}-{address:x}.metal")),
                        )
                    {
                        tracing::warn!(
                            "{stage:?} program at {address:#x} replaced from {}",
                            dir.display()
                        );
                        translated.msl = msl;
                    }
                    tracing::trace!(
                        target: "gpu::msl",
                        "{stage:?} {address:#x}\n{}",
                        translated.msl
                    );
                    let label = format!("{stage:?} {address:#x} key {key:016x}");
                    let function = self.compile_later(
                        translated.msl.clone(),
                        translated.entry.clone(),
                        label.clone(),
                    );
                    let compute = (stage == Stage::Compute).then(|| {
                        let (device, f) = (self.device.clone(), function.clone());
                        let queued = std::time::Instant::now();
                        Task::spawn_after(&[&*function], move || {
                            let function = f;
                            let start = std::time::Instant::now();
                            let function = function.wait()?;
                            let made = std::time::Instant::now();
                            let t = crate::timing::start();
                            let p = if crate::timing::labels() {
                                let desc = objc2_metal::MTLComputePipelineDescriptor::new();
                                desc.setComputeFunction(Some(&function));
                                desc.setLabel(Some(&NSString::from_str(&label)));
                                device
                                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                                        &desc,
                                        objc2_metal::MTLPipelineOption::None,
                                        None,
                                    )
                            } else {
                                device.newComputePipelineStateWithFunction_error(&function)
                            };
                            crate::timing::add(crate::timing::Part::Compile, t);
                            log_compile("compute pipeline", &label, 0, queued, start, Some(made));
                            p.map_err(|e| {
                                let msg = e.localizedDescription().to_string();
                                diag::problems::record_with_detail(
                                    diag::problems::Kind::PipelineFailed,
                                    "compute pipeline",
                                    &label,
                                    &msg,
                                );
                                late_report(format!("compute pipeline failed: {msg}"))
                            })
                            .ok()
                        })
                    });
                    Some(Arc::new(Program {
                        address,
                        id: {
                            static NEXT: std::sync::atomic::AtomicU64 =
                                std::sync::atomic::AtomicU64::new(1);
                            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        },
                        function,
                        compute,
                        flat: flat_locations(&translated.msl),
                        sint_textures: variant.sint_textures,
                        frame: self.epoch,
                        translated,
                    }))
                }
                Err(e) => {
                    diag::problems::record_with_detail(
                        diag::problems::Kind::ShaderTranslate,
                        e.message.clone(),
                        format!("{stage:?} {address:#x}"),
                        e.to_string(),
                    );
                    self.report(
                        tracing::Level::ERROR,
                        format!("{stage:?} program at {address:#x} not translated: {e}"),
                    );
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
    pub(super) fn vertex_function(
        &mut self,
        vertex: &Program,
        pixel: &Program,
        form: VertexForm,
    ) -> Arc<Task<Retained<Function>>> {
        let VertexForm {
            rect,
            gl_clip,
            point_size,
        } = form;
        let locations = |msl: &str| -> Vec<u32> {
            let mut v: Vec<u32> = msl
                .match_indices("[[user(locn")
                .filter_map(|(i, m)| {
                    msl[i + m.len()..]
                        .split(|c: char| !c.is_ascii_digit())
                        .next()?
                        .parse()
                        .ok()
                })
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let missing = match self.missing_inputs.get(&(vertex.id, pixel.id)) {
            Some(m) => m.clone(),
            None => {
                let have = locations(&vertex.translated.msl);
                let m: Vec<u32> = locations(&pixel.translated.msl)
                    .into_iter()
                    .filter(|l| !have.contains(l))
                    .collect();
                self.missing_inputs.insert((vertex.id, pixel.id), m.clone());
                m
            }
        };
        // Rectangles keep flat outputs from the first vertex: `rect_source` combines floats only.
        let flat = pixel.flat.clone();
        if missing.is_empty() && flat.is_empty() && form == VertexForm::default() {
            return vertex.function.clone();
        }
        let key = (vertex.id, form, missing.clone(), flat.clone());
        if let Some(f) = self.variants.get(&key) {
            return f.clone();
        }
        let (device, source, entry) = (
            self.device.clone(),
            vertex.translated.msl.clone(),
            vertex.translated.entry.clone(),
        );
        let label = format!(
            "vertex {:#x} variant {form:?} missing {missing:?} flat {flat:?}",
            vertex.address
        );
        let queued = std::time::Instant::now();
        let f = Task::spawn(move || {
            let start = std::time::Instant::now();
            let extra: String = missing
                .iter()
                .map(|l| {
                    format!(
                        "    {} x{l} [[user(locn{l})]];\n",
                        if flat.contains(l) { "uint4" } else { "float4" }
                    )
                })
                .collect();
            let mut msl = source.replacen(
                "    float4 pos [[position]];\n",
                &format!("    float4 pos [[position]];\n{extra}"),
                1,
            );
            for l in &flat {
                msl = uint_output(&msl, *l);
            }
            if gl_clip {
                msl = gl_clip_space(&msl);
            }
            if let Some(size) = point_size {
                msl = with_point_size(&msl, f32::from_bits(size));
            }
            let f = if rect {
                rect_source(&msl).and_then(|m| compile(&device, &m, "gen5_vs_rect", &label))
            } else {
                compile(&device, &msl, &entry, &label)
            };
            log_compile("library", &label, msl.len(), queued, start, None);
            f
        });
        self.variants.insert(key, f.clone());
        f
    }

    /// The render pipeline for `key`, compiled on the compile threads once both functions are.
    pub(super) fn pipeline(
        &mut self,
        key: &PipelineKey,
        vertex: Arc<Task<Retained<Function>>>,
        pixel: &Program,
    ) -> Arc<Task<Retained<RenderPipeline>>> {
        if let Some(p) = self.pipelines.get(key) {
            return p.clone();
        }
        let (device, k, pixel) = (self.device.clone(), key.clone(), pixel.function.clone());
        let queued = std::time::Instant::now();
        let (dv, dp) = (vertex.clone(), pixel.clone());
        let deps: [&dyn crate::compile::Dependency; 2] = [&*dv, &*dp];
        let task = Task::spawn_after(&deps, move || {
            let key = &k;
            let start = std::time::Instant::now();
            let (vertex, pixel) = (vertex.wait()?, pixel.wait()?);
            let ready = std::time::Instant::now();
            let desc = MTLRenderPipelineDescriptor::new();
            desc.setVertexFunction(Some(&vertex));
            desc.setFragmentFunction(Some(&pixel));
            for (i, &raw) in key.formats.iter().enumerate() {
                if raw == 0 {
                    continue;
                }
                let att = unsafe { desc.colorAttachments().objectAtIndexedSubscript(i) };
                att.setPixelFormat(MTLPixelFormat(raw));
                att.setWriteMask(MTLColorWriteMask(key.write_masks[i] as usize));
            }
            if key.depth != 0 {
                desc.setDepthAttachmentPixelFormat(MTLPixelFormat(key.depth));
                if key.stencil {
                    desc.setStencilAttachmentPixelFormat(MTLPixelFormat(key.depth));
                }
            }
            if crate::timing::labels() {
                desc.setLabel(Some(&NSString::from_str(&format!(
                    "vertex {:#x} pixel {:#x}",
                    key.vertex, key.pixel
                ))));
            }
            let t = crate::timing::start();
            let made = device.newRenderPipelineStateWithDescriptor_error(&desc);
            crate::timing::add(crate::timing::Part::Compile, t);
            log_compile(
                "render pipeline",
                &format!("vertex {:#x} pixel {:#x}", key.vertex, key.pixel),
                0,
                queued,
                start,
                Some(ready),
            );
            made.map_err(|e| {
                let msg = e.localizedDescription().to_string();
                diag::problems::record_with_detail(
                    diag::problems::Kind::PipelineFailed,
                    "render pipeline",
                    format!("vs {:#x} ps {:#x}", key.vertex, key.pixel),
                    &msg,
                );
                late_report(format!("render pipeline failed: {msg}"))
            })
            .ok()
        });
        self.pipelines.insert(key.clone(), task.clone());
        task
    }
}

/// Compiles `msl` and returns its function `entry`; any thread.
pub(super) fn compile(
    device: &Device,
    msl: &str,
    entry: &str,
    label: &str,
) -> Option<Retained<Function>> {
    let t = crate::timing::start();
    let unique;
    let msl = if diag::config().gpu.msl_unique {
        unique = format!("{msl}\n// {:?}\n", std::time::SystemTime::now());
        unique.as_str()
    } else {
        msl
    };
    // Safe math keeps IEEE NaN/infinity semantics matching GFX10.
    let options = objc2_metal::MTLCompileOptions::new();
    options.setMathMode(objc2_metal::MTLMathMode::Safe);
    let lib = device.newLibraryWithSource_options_error(&NSString::from_str(msl), Some(&options));
    crate::timing::add(crate::timing::Part::Compile, t);
    match lib {
        Ok(lib) => {
            let f = lib.newFunctionWithName(&NSString::from_str(entry))?;
            if crate::timing::labels() {
                f.setLabel(Some(&NSString::from_str(label)));
            }
            Some(f)
        }
        Err(e) => {
            let msg = e.localizedDescription().to_string();
            diag::problems::record_with_detail(
                diag::problems::Kind::PipelineFailed,
                "MSL compile",
                label,
                &msg,
            );
            late_report(format!("MSL did not compile: {msg}"));
            tracing::debug!(target: "gpu::msl", "{msl}");
            None
        }
    }
}

/// Seconds since the first compile log line.
pub(super) fn log_clock() -> f64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// Logs a compile job: its time in the queue, then (for pipelines) waiting
/// for its functions from `start` to `ready`, then its own work.
fn log_compile(
    kind: &str,
    label: &str,
    bytes: usize,
    queued: std::time::Instant,
    start: std::time::Instant,
    ready: Option<std::time::Instant>,
) {
    if !tracing::enabled!(target: "gpu::compile", tracing::Level::DEBUG) {
        return;
    }
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let now = std::time::Instant::now();
    let ready = ready.unwrap_or(start);
    tracing::debug!(
        target: "gpu::compile",
        "{:.3} s {kind} {label} bytes {bytes} queued {:.1} ms waited {:.1} ms ran {:.1} ms on {}",
        log_clock(),
        ms(start - queued),
        ms(ready - start),
        ms(now - ready),
        std::thread::current().name().unwrap_or("?")
    );
}

/// Logs a failure from the compile threads, the first [`REPORTS`] times.
fn late_report(what: String) {
    pub(super) static LIMIT: diag::FirstN = diag::FirstN::new(REPORTS as u32);
    if LIMIT.allow() {
        tracing::error!("{what}");
    }
}

fn rect_source(msl: &str) -> Option<String> {
    let start = msl.find("vertex VsOut gen5_vs(")?;
    let open = start + "vertex VsOut gen5_vs(".len();
    let close = open + msl[open..].find(") {")?;
    let params = msl[open..close].split_once("[[instance_id]], ")?.1;
    let call = msl[close..].find("return gen5_vs_body(vid, iid")?
        + close
        + "return gen5_vs_body(vid, iid".len();
    let call_end = call + msl[call..].find(");")?;
    let args = &msl[call..call_end];
    let fields: Vec<&str> = msl[msl.find("struct VsOut {")?..]
        .lines()
        .take_while(|l| !l.starts_with("};"))
        .filter_map(|l| l.trim().strip_prefix("float4 ")?.split_whitespace().next())
        .collect();
    let combine: String = fields
        .iter()
        .map(|f| format!("    o.{f} = b.{f} + c.{f} - a.{f};\n"))
        .collect();
    Some(format!(
        "{msl}\nvertex VsOut gen5_vs_rect(uint rv [[vertex_id]], uint first_iid [[instance_id]], uint rect_offset [[base_instance]], {params}) {{\n    uint iid = first_iid - rect_offset;\n    uint base = rect_offset + rv / 6 * 3;\n    uint k = (0x231210u >> ((rv % 6) * 4)) & 0xFu;\n    if (k < 3) return gen5_vs_body(base + k, iid{args});\n    VsOut a = gen5_vs_body(base, iid{args});\n    VsOut b = gen5_vs_body(base + 1, iid{args});\n    VsOut c = gen5_vs_body(base + 2, iid{args});\n    VsOut o = a;\n{combine}    return o;\n}}\n"
    ))
}

fn with_point_size(msl: &str, size: f32) -> String {
    msl.replacen(
        "    float4 pos [[position]];\n",
        "    float4 pos [[position]];\n    float psize [[point_size]];\n",
        1,
    )
    .replacen(
        "    VsOut o = {};\n",
        &format!("    VsOut o = {{}};\n    o.psize = {size:?};\n"),
        1,
    )
}

// OpenGL-style clip space puts z in [-w, w]; Metal clips z below 0.
fn gl_clip_space(msl: &str) -> String {
    let mut out = String::with_capacity(msl.len() + 128);
    let mut rest = msl;
    while let Some(i) = rest.find("o.pos = float4(") {
        let Some(end) = rest[i..].find(");").map(|e| i + e) else {
            break;
        };
        out.push_str(&rest[..i]);
        out.push_str("o.pos = gl_z(");
        out.push_str(&rest[i + "o.pos = ".len()..end + 1]);
        out.push_str(");");
        rest = &rest[end + 2..];
    }
    out.push_str(rest);
    out.replacen("struct VsOut {", "static inline float4 gl_z(float4 p) { p.z = (p.z + p.w) * 0.5; return p; }\n\nstruct VsOut {", 1)
}

/// Whether MRT0 blends with a `SRC1` factor (`Request::dual_source`).
pub(super) fn dual_source(c: &[u32]) -> bool {
    let b = Blend::read(c, 0);
    b.enable
        && [b.color_src, b.color_dst, b.alpha_src, b.alpha_dst]
            .iter()
            .any(|f| (15..=18).contains(f))
}

/// The MRTs whose target holds 16-bit floats (`Request::half_targets`).
pub(super) fn half_targets(c: &[u32]) -> u8 {
    (0..8)
        .filter(|&n| {
            ColorTarget::read(c, n)
                .is_some_and(|t| t.format.number == 7 && matches!(t.format.layout, 2 | 5 | 12))
        })
        .fold(0, |m, n| m | 1 << n)
}

/// Whether MRT `n` is attached to the draw's pass: a program may read only
/// attached targets through `[[color(n)]]`.
fn attached(c: &[u32], n: usize) -> bool {
    super::draw::attached_targets(c[context::CB_TARGET_MASK], export_formats(c)).any(|a| a == n)
}

/// `Request::blends`: every MRT that blends, as the pixel program does it,
/// since Metal's fixed-function blending differs from the CB's in rounding,
/// NaNs, min and max, and some factors. Integer targets and those with
/// `CB_COLOR_INFO.BLEND_BYPASS` do not blend, as Mesa's `ac_descriptors.c`
/// sets that bit for integer formats. 8-bit unorm targets are read at 12
/// significant bits (measured on RDNA2 with the blend sweep; other unorm
/// widths and sRGB are not measured).
pub(super) fn blends(c: &[u32]) -> [u32; 8] {
    std::array::from_fn(|n| {
        let b = Blend::read(c, n);
        let Some(t) = ColorTarget::read(c, n).filter(|_| attached(c, n)) else {
            return 0;
        };
        let info = c[context::CB_COLOR0_INFO + n * context::CB_COLOR_STRIDE];
        if !b.enable || matches!(t.format.number, 4 | 5) || info >> 16 & 1 != 0 {
            return 0;
        }
        let clamp = match t.format.number {
            _ if info >> 15 & 1 == 0 => 0,
            0 | 6 => gfx10::translate::BLEND_CLAMP_UNORM,
            1 => gfx10::translate::BLEND_CLAMP_SNORM,
            _ => 0,
        };
        let dst_bits = match (t.format.number, t.format.layout) {
            (0, 1 | 3 | 10) => gfx10::translate::BLEND_DST_12_BITS,
            _ => 0,
        };
        b.color_src
            | b.color_op << 5
            | b.color_dst << 8
            | clamp
            | dst_bits
            | b.alpha_src << 16
            | b.alpha_op << 21
            | b.alpha_dst << 24
            | 1 << 30
    })
}

/// `CB_BLEND_RED`..`ALPHA` for `binding::BLEND_COLOR`. The CB keeps 12
/// significant bits of each, rounded toward zero (measured on RDNA2 with 0.6
/// only, so whether this is a float or a fixed-point form is not known).
pub(super) fn blend_color(c: &[u32]) -> [u32; 4] {
    std::array::from_fn(|i| c[context::CB_BLEND_RED + i] & 0xFFFF_F000)
}

/// The export component each channel of a target in `format` stores, as
/// `Request::swizzles` packs it; 0 for the order exported, and for the
/// swaps the target's Metal format makes itself (BGRA). Mesa's
/// `ac_translate_colorswap` gives the four-channel swaps and some others;
/// the rest follow the per-channel-count table of AMD's R6xx-era register
/// documentation (not checked on hardware).
pub(super) fn export_swizzle(format: regs::ColorFormat) -> u8 {
    const X: u8 = 0;
    const Y: u8 = 1;
    const Z: u8 = 2;
    const W: u8 = 3;
    let channels = match format.layout {
        1 | 2 | 4 => 1,
        3 | 5 | 11 => 2,
        6 | 7 | 16 | 24 => 3,
        _ => 4,
    };
    if matches!(
        color_format(format),
        Some(
            MTLPixelFormat::BGRA8Unorm
                | MTLPixelFormat::BGRA8Unorm_sRGB
                | MTLPixelFormat::BGR10A2Unorm
        )
    ) {
        return 0;
    }
    let order: &[u8] = match (channels, format.swap) {
        (_, 0) => return 0,
        (1, 1) => &[Y],
        (1, 2) => &[Z],
        (1, _) => &[W],
        (2, 1) => &[X, W],
        (2, 2) => &[Y, X],
        (2, _) => &[W, X],
        (3, 1) => &[X, Y, W],
        (3, 2) => &[Z, Y, X],
        (3, _) => &[W, Y, X],
        (_, 1) => &[Z, Y, X, W],
        (_, 2) => &[W, Z, Y, X],
        _ => &[W, X, Y, Z],
    };
    (0..4).fold(0, |s, i| {
        s | order.get(i).copied().unwrap_or(i as u8) << (2 * i)
    })
}

/// `CB_COLOR_CONTROL.ROP3` and the targets it applies to
/// (`Request::logic_targets`): unorm and integer targets of 8, 16 or 32 bits
/// per channel that do not blend and whose `CB_BLENDn_CONTROL.DISABLE_ROP3`
/// is clear. Only in `CB_NORMAL` mode: context registers start at 0 here
/// (CLEAR_STATE's defaults are not modelled), and a ROP3 of 0 would clear.
pub(super) fn logic_targets(c: &[u32]) -> (u8, [u8; 8]) {
    let control = c[context::CB_COLOR_CONTROL];
    let rop = if control >> 4 & 7 == 1 {
        (control >> 16) as u8
    } else {
        0xCC
    };
    let mut targets = [0; 8];
    if rop == 0xCC {
        return (rop, targets);
    }
    for (n, t) in targets.iter_mut().enumerate() {
        let Some(ct) = ColorTarget::read(c, n).filter(|_| attached(c, n)) else {
            continue;
        };
        if Blend::read(c, n).enable || c[context::CB_BLEND0_CONTROL + n] >> 31 != 0 {
            continue;
        }
        let bits = match ct.format.layout {
            1 | 3 | 10 => 8,
            2 | 5 | 12 => 16,
            4 | 11 | 14 => 32,
            _ => continue,
        };
        let kind = match ct.format.number {
            0 => 0,
            4 => 1,
            5 => 2,
            _ => continue,
        };
        *t = bits | kind << 6;
    }
    (rop, targets)
}

/// `SPI_SHADER_COL_FORMAT` with each integer target's export made the
/// matching 16-bit integer format, so the bits the CB receives reach it as
/// they are (Metal rejects float outputs to integer attachments).
pub(super) fn export_formats(c: &[u32]) -> u32 {
    let mut formats = c[context::SPI_SHADER_COL_FORMAT];
    for n in 0..8 {
        let Some(t) = ColorTarget::read(c, n) else {
            continue;
        };
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
