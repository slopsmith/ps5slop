//! Translated programs, their Metal functions and pipelines, compiled on the compile threads.

use super::*;

/// A vertex program, whether it draws rectangles or uses OpenGL clip space,
/// and the attributes it must add.
pub(super) type VariantKey = (u64, bool, bool, Vec<u32>, Vec<u32>);

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
    /// Compute: threads past a grid size read from a buffer do nothing.
    pub(super) grid_bounds: bool,
    /// Pixel: no depth target is bound, so a depth export is dropped.
    pub(super) drop_depth_export: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct PipelineKey {
    pub(super) vertex: u64,
    pub(super) pixel: u64,
    pub(super) formats: [usize; 8],
    pub(super) blends: [(Blend, u8); 8],
    pub(super) depth: usize,
    pub(super) stencil: bool,
    pub(super) rect: bool,
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
        compile(&self.device, msl, entry)
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
            let f = compile(&device, &msl, &entry);
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
                slot, ref source, ..
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
                    _ => {}
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
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (stage as u8, user_sgprs, ngg, variant).hash(&mut h);
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
                grid_bounds: variant.grid_bounds,
                drop_depth_export: variant.drop_depth_export,
                probe_flat: diag::config().gpu.probe_flat,
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
                            let p = device.newComputePipelineStateWithFunction_error(&function);
                            crate::timing::add(crate::timing::Part::Compile, t);
                            log_compile("compute pipeline", &label, 0, queued, start, Some(made));
                            p.map_err(|e| {
                                late_report(format!(
                                    "compute pipeline failed: {}",
                                    e.localizedDescription()
                                ))
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
        rect: bool,
        gl_clip: bool,
    ) -> Arc<Task<Retained<Function>>> {
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
        let flat: Vec<u32> = if rect { Vec::new() } else { pixel.flat.clone() };
        if missing.is_empty() && flat.is_empty() && !rect && !gl_clip {
            return vertex.function.clone();
        }
        let key = (vertex.id, rect, gl_clip, missing.clone(), flat.clone());
        if let Some(f) = self.variants.get(&key) {
            return f.clone();
        }
        let (device, source, entry) = (
            self.device.clone(),
            vertex.translated.msl.clone(),
            vertex.translated.entry.clone(),
        );
        let label = format!(
            "vertex {:#x} variant rect {rect} gl_clip {gl_clip} missing {missing:?} flat {flat:?}",
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
            let f = if rect {
                rect_source(&msl).and_then(|m| compile(&device, &m, "gen5_vs_rect"))
            } else {
                compile(&device, &msl, &entry)
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
            log_compile(
                "render pipeline",
                &format!("vertex {:#x} pixel {:#x}", key.vertex, key.pixel),
                0,
                queued,
                start,
                Some(ready),
            );
            made.map_err(|e| {
                late_report(format!(
                    "render pipeline failed: {}",
                    e.localizedDescription()
                ))
            })
            .ok()
        });
        self.pipelines.insert(key.clone(), task.clone());
        task
    }
}

/// Compiles `msl` and returns its function `entry`; any thread.
pub(super) fn compile(device: &Device, msl: &str, entry: &str) -> Option<Retained<Function>> {
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
        Ok(lib) => lib.newFunctionWithName(&NSString::from_str(entry)),
        Err(e) => {
            late_report(format!("MSL did not compile: {}", e.localizedDescription()));
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
