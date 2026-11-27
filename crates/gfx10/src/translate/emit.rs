//! Entry-point emission, stage I/O setup and movrel expansion.

use super::*;

impl<'a> Tx<'a> {
    pub(super) fn finish(mut self) -> Translated {
        let mut resources = Vec::new();
        for (i, (_, kind, source, access, view)) in self.textures.iter().enumerate() {
            resources.push(Resource::Texture {
                slot: i as u8,
                source: source.clone(),
                kind: *kind,
                access: *access,
                view: *view,
            });
        }
        for (i, source) in self.samplers.iter().enumerate() {
            resources.push(Resource::Sampler {
                slot: i as u8,
                source: source.clone(),
            });
        }
        for (i, (base, bytes)) in self.constants.iter().enumerate() {
            resources.push(Resource::Constants {
                slot: i as u8,
                base: base.clone(),
                bytes: *bytes,
            });
        }
        let msl = self.emit();
        let entry = match self.req.stage {
            Stage::Vertex => "gen5_vs",
            Stage::Pixel => "gen5_ps",
            Stage::Compute => "gen5_cs",
        };
        Translated {
            msl,
            entry: entry.into(),
            resources,
            vertex_outputs: self.params.iter().next_back().map_or(0, |&p| p as u32 + 1),
            color_outputs: self.mrts,
            writes_depth: self.depth,
            guest_memory: self.mem,
            writes_memory: self.mem_write,
            store_buffers: self.stores.take(),
            lds_bytes: if self.lds { self.lds_bytes() } else { 0 },
            gds: self.gds,
            loop_guard: self.guarded,
            lockstep: self.lockstep,
            reads_blend_color: self.req.stage == Stage::Pixel
                && !self.blend_statements().is_empty(),
            flat_inputs: self.attrs.values().any(|&m| m == Interp::Flat),
        }
    }

    /// Clears EXEC for threads past the grid size in `bounds`. A lockstep
    /// group is one-dimensional, so each wave's active lanes are a prefix.
    fn bounds_init(&self) -> String {
        let t = self.req.threads.map(|n| n.max(1));
        let outside = format!(
            "any(gid * uint3({}u, {}u, {}u) + lid >= uint3(bounds[0], bounds[1], bounds[2]))",
            t[0], t[1], t[2]
        );
        if !self.lockstep {
            return if self.regs.contains("exec_lo") {
                format!("    if ({outside}) exec_lo = 0u;\n")
            } else {
                String::new()
            };
        }
        let mut s = format!(
            "    {{ uint first = gid.x * {}u + (uint(sgi) >> 1) * 64u; uint n = (bounds[0] > first && gid.y < bounds[1] && gid.z < bounds[2]) ? min(bounds[0] - first, 64u) : 0u;\n",
            t[0]
        );
        if self.regs.contains("exec_lo") {
            s.push_str("      exec_lo = n >= 32u ? 0xffffffffu : (1u << n) - 1u;\n");
        }
        if self.regs.contains("exec_hi") {
            s.push_str(
                "      exec_hi = n >= 64u ? 0xffffffffu : n > 32u ? (1u << (n - 32u)) - 1u : 0u;\n",
            );
        }
        s.push_str("    }\n");
        s
    }

    /// Threadgroup memory: `LDS_SIZE` of `COMPUTE_PGM_RSRC2` in 512-byte units,
    /// within what Metal allows.
    fn lds_bytes(&self) -> u32 {
        let declared = (self.req.compute_rsrc2 >> 15 & 0x1FF) * 512;
        let most = (32 << 10) - if self.lockstep { LOCKSTEP_LDS } else { 0 };
        if declared == 0 {
            most
        } else {
            declared.clamp(16, most)
        }
    }

    fn emit(&mut self) -> String {
        let mut out = String::from(PRELUDE);
        if self.floats {
            out.push_str(&float::mode_defines(self.req.rsrc1, self.req.plain_float));
            out.push_str(float::PRELUDE_HERE);
        }
        if self.filters {
            out.push_str(image::FILTER_PRELUDE);
        }

        if self.lockstep {
            self.simd_lane = true;
            out.push_str(&format!(
                "#define LOCKSTEP_WAVES {}u\n#define MEET_SPINS {MEET_SPINS}u\n{WAVE_PRELUDE}",
                self.waves()
            ));
        }
        if self.lds {
            out.push_str(&format!(
                "#define LDS_BYTES {}u\n{LDS_PRELUDE}",
                self.lds_bytes()
            ));
        }
        if self.addtid_local {
            let shift = if self.addtid_aligned128 { 7u32 } else { 2u32 };
            let slots = ((self.addtid_max_off >> shift) + 1).clamp(16, 512);
            out.push_str(&format!(
                "#define ADDTID_SHIFT {shift}u\n#define ADDTID_SLOTS {slots}u\n"
            ));
        }
        if self.scratch_stores {
            out.push_str(&format!(
                "#define SCRATCH_BYTES {}u\n{SCRATCH_PRELUDE}",
                self.scratch_bytes()
            ));
        }
        if self.guarded {
            out.push_str(&format!(
                "#define GUARD_LIMIT {MAX_COMPUTE_BACK_BRANCHES}u\n#define GUARD_BUDGET {GUARD_BUDGET}u\n#define GUARD_UNIT {GUARD_UNIT}u\n{GUARD_PRELUDE}"
            ));
            if self.lockstep {
                out.push_str(LOCKSTEP_GUARD_PRELUDE);
            }
        }
        let stage = self.req.stage;
        let mem_ty = if self.mem_write {
            "device uchar*"
        } else {
            "device const uchar*"
        };
        let mut params = vec!["constant uint* user".to_string()];
        let mut args = vec!["user".to_string()];
        if self.mem {
            params.push(format!("{mem_ty} mem"));
            params.push("Win mem_base".into());
            args.push("mem".into());
            args.push("mem_base".into());
        }
        let mut entry_params = vec![format!("constant uint* user [[buffer({})]]", binding::USER)];
        if self.mem {
            entry_params.push(format!("{mem_ty} mem [[buffer({})]]", binding::MEM));
            entry_params.push(format!(
                "constant Win& mem_base [[buffer({})]]",
                binding::MEM_BASE
            ));
        }
        if self.gds {
            params.push("device atomic_uint* gds".into());
            args.push("gds".into());
            entry_params.push(format!(
                "device atomic_uint* gds [[buffer({})]]",
                binding::GDS
            ));
        }
        if self.guarded {
            entry_params.push(format!(
                "device atomic_uint* guard [[buffer({})]]",
                binding::GUARD
            ));
        }
        for (i, &(_, kind, _, access, _)) in self.textures.iter().enumerate() {
            let base = match kind {
                TextureKind::Tex2d => "texture2d",
                TextureKind::Tex2dArray => "texture2d_array",
                TextureKind::Tex3d => "texture3d",
                TextureKind::Cube => "texturecube",
            };
            let elem = self.texel(i).0;
            let ty = match access {
                Access::Sample => format!("{base}<{elem}>"),
                Access::Write => format!("{base}<{elem}, access::write>"),
                Access::ReadWrite | Access::Atomic => format!("{base}<{elem}, access::read_write>"),
            };
            params.push(format!("{ty} t{i}"));
            args.push(format!("t{i}"));
            entry_params.push(format!("{ty} t{i} [[texture({i})]]"));
        }
        // Metal may read `constant` memory as it was when the dispatch began, but
        // scalar loads see what the program stored before them.
        let space = if self.mem_write {
            "device const"
        } else {
            "constant"
        };
        for i in 0..self.constants.len() {
            params.push(format!("{space} uint* cb{i}"));
            args.push(format!("cb{i}"));
            entry_params.push(format!(
                "{space} uint* cb{i} [[buffer({})]]",
                binding::CONSTANTS as usize + i
            ));
        }
        for i in 0..self.samplers.len() {
            params.push(format!("sampler smp{i}"));
            args.push(format!("smp{i}"));
            entry_params.push(format!("sampler smp{i} [[sampler({i})]]"));
        }
        if self.quad_lane {
            entry_params.push("ushort qlane [[thread_index_in_quadgroup]]".into());
        }
        if self.simd_lane {
            entry_params.push("ushort slane [[thread_index_in_simdgroup]]".into());
        }

        if self.movrel {
            self.fill_movrel();
        }
        let mut init = String::new();
        let users = self.req.user_sgprs as u16;
        for r in self.regs.clone() {
            match r.as_str() {
                "scc" => init.push_str("    bool scc = false;\n"),
                "hw_mode" => init.push_str(&format!(
                    "    uint hw_mode = {:#x}u;\n",
                    self.initial_mode()
                )),
                _ => init.push_str(&format!("    uint {r} = 0u;\n")),
            }
        }
        for i in 0..users {
            let r = self.user_base + i;
            if r < SGPR_COUNT && self.regs.contains(&format!("s{r}")) {
                init.push_str(&format!("    s{r} = user[{i}];\n"));
            }
        }
        if self.addtid_local {
            init.push_str("    uint addtid_lds[ADDTID_SLOTS] = {};\n");
        }
        if self.scratch_stores {
            init.push_str("    uint scratch_mem[SCRATCH_BYTES / 4u] = {};\n");
        }
        if self.lockstep {
            init.push_str("    uint wl = (uint(sgi) & 1u) * 32u + uint(slane);\n    Wave wv = { lockstep_vote, lockstep_seq, sgi, 0u };\n");
            for r in ["exec_lo", "exec_hi"] {
                if self.regs.contains(r) {
                    init.push_str(&format!("    {r} = 0xffffffffu;\n"));
                }
            }
        } else if self.regs.contains("exec_lo") {
            init.push_str("    exec_lo = 1u;\n");
        }
        match stage {
            Stage::Vertex => {
                out.push_str("struct VsOut {\n    float4 pos [[position]];\n");
                for p in &self.params {
                    out.push_str(&format!("    float4 p{p} [[user(locn{p})]];\n"));
                }
                out.push_str("};\n\n");
                let (vid, iid) = if self.req.ngg { (5, 8) } else { (0, 3) };
                for (reg, val) in [(format!("v{vid}"), "vid"), (format!("v{iid}"), "iid")] {
                    if self.regs.contains(&reg) {
                        init.push_str(&format!("    {reg} = {val};\n"));
                    }
                }
                if self.req.ngg && self.regs.contains("s3") {
                    init.push_str("    s3 = 0x101u;\n");
                }
                out.push_str(&format!(
                    "static VsOut gen5_vs_body(uint vid, uint iid, {}) {{\n    VsOut o = {{}};\n{init}{}    return o;\n}}\n\n",
                    params.join(", "),
                    self.body
                ));
                out.push_str(&format!(
                    "vertex VsOut gen5_vs(uint vid [[vertex_id]], uint iid [[instance_id]], {}) {{\n    return gen5_vs_body(vid, iid, {});\n}}\n",
                    entry_params.join(", "),
                    args.join(", ")
                ));
            }
            Stage::Pixel => {
                let loc_of = |a: u8| {
                    self.req
                        .ps_input_cntl
                        .get(a as usize)
                        .map_or(a as u32, |c| c & 0x1F)
                };
                let mut first = BTreeMap::new();
                for &a in self.attrs.keys() {
                    first.entry(loc_of(a)).or_insert(a);
                }
                // Inputs sharing a location read the same export; MSL wants
                // each location once, so those are copied out of `raw`.
                let shared = first.len() != self.attrs.len();
                out.push_str(if shared {
                    "struct PsInRaw {\n"
                } else {
                    "struct PsIn {\n"
                });
                out.push_str("    float4 pos [[position]];\n    bool front [[front_facing]];\n");
                for (&a, &mode) in &self.attrs {
                    let loc = loc_of(a);
                    if first[&loc] != a {
                        continue;
                    }
                    let q = qualifier(mode);
                    // Flat inputs carry raw bits (often integers), which a float
                    // varying may not keep; vertex variants export them as uint4.
                    let ty = if mode == Interp::Flat {
                        "uint4"
                    } else {
                        "float4"
                    };
                    out.push_str(&format!("    {ty} a{a} [[user(locn{loc}), {q}]];\n"));
                }
                for (&a, &mode) in &self.half_attrs {
                    let (loc, q) = (loc_of(a), qualifier(mode));
                    for half in ['l', 'h'] {
                        out.push_str(&format!(
                            "    float4 a{a}{half} [[user(locn{loc}{half}), {q}]];\n"
                        ));
                    }
                }
                if shared {
                    out.push_str("};\n\nstruct PsIn {\n    float4 pos;\n    bool front;\n");
                    let mut fill = String::from("    PsIn in = { raw.pos, raw.front");
                    for (&a, &mode) in &self.attrs {
                        let ty = if mode == Interp::Flat {
                            "uint4"
                        } else {
                            "float4"
                        };
                        out.push_str(&format!("    {ty} a{a};\n"));
                        let src = first[&loc_of(a)];
                        if (self.attrs[&src] == Interp::Flat) == (mode == Interp::Flat) {
                            fill.push_str(&format!(", raw.a{src}"));
                        } else {
                            fill.push_str(&format!(", as_type<{ty}>(raw.a{src})"));
                        }
                    }
                    for &a in self.half_attrs.keys() {
                        out.push_str(&format!("    float4 a{a}l;\n    float4 a{a}h;\n"));
                        fill.push_str(&format!(", raw.a{a}l, raw.a{a}h"));
                    }
                    init.insert_str(0, &(fill + " };\n"));
                }
                out.push_str("};\n\nstruct PsOut {\n");
                let src1 = self.req.dual_source && self.mrts & 2 != 0;
                for i in 0..8 {
                    if self.mrts >> i & 1 != 0 && !(i == 1 && src1) {
                        let blended = self.req.blends[i] != 0;
                        let ty = match self.req.col_format >> (4 * i) & 0xF {
                            _ if self.req.packed_targets[i] != 0 => "uint4",
                            7 => "uint4",
                            8 => "int4",
                            _ if blended && self.req.half_targets >> i & 1 != 0 => "half4",
                            _ => "float4",
                        };
                        out.push_str(&format!("    {ty} c{i} [[color({i})]];\n"));
                        if blended {
                            init.push_str(&format!("    float4 blend_src{i} = 0.0f;\n"));
                        }
                    }
                }
                if src1 {
                    init.push_str("    float4 src1 = 0.0f;\n");
                }
                if self.depth {
                    out.push_str("    float depth [[depth(any)]];\n");
                }
                out.push_str("};\n\n");
                init.push_str(&self.ps_vgprs());
                if self.regs.contains("m0") {
                    init.push_str("    m0 = 0u;\n");
                }
                let mut body = self.body.clone();
                let mut tail =
                    String::from("    if (!exported) discard_fragment();\n    return o;\n");
                let blend = self.blend_statements();
                if !blend.is_empty() {
                    let with_blend = format!("{{ {blend}return o; }}");
                    body = body.replace("return o;", &with_blend);
                    tail = tail.replace("return o;", &with_blend);
                    entry_params.push(format!(
                        "constant float4& blend_color [[buffer({})]]",
                        binding::BLEND_COLOR
                    ));
                }
                if self.req.probe_flat
                    && self.mrts & 2 != 0
                    && self.req.blends[1] == 0
                    && !self.req.dual_source
                    && self.req.col_format >> 4 & 0xF != 7
                    && self.req.col_format >> 4 & 0xF != 8
                    && let Some((&a, _)) = self.attrs.iter().find(|(_, m)| **m == Interp::Flat)
                {
                    let probe = format!(
                        "{{ o.c1 = float4(float(in.a{a}.x & 0xffu) / 255.0, float((in.a{a}.x >> 8) & 0xffu) / 255.0, float((in.a{a}.x >> 16) & 0xffu) / 255.0, 1.0); return o; }}"
                    );
                    body = body.replace("return o;", &probe);
                    tail = tail.replace("return o;", &probe);
                }
                for n in 0..8 {
                    if self.mrts >> n & 1 == 0 || n == 1 && src1 {
                        continue;
                    }
                    let logic = self.req.logic_targets[n];
                    let packed = self.req.packed_targets[n];
                    if packed != 0
                        && (self.req.blends[n] != 0
                            || mem::packed_write_bits(packed) != [u32::MAX; 4])
                    {
                        entry_params.push(format!("uint4 d{n} [[color({n})]]"));
                    } else if self.req.logic_op != 0xCC && logic != 0 {
                        let ty = ["float4", "uint4", "int4"][(logic >> 6).min(2) as usize];
                        entry_params.push(format!("{ty} d{n} [[color({n})]]"));
                    } else if self.req.blends[n] != 0 {
                        let ty = if self.req.half_targets >> n & 1 != 0 {
                            "half4"
                        } else {
                            "float4"
                        };
                        entry_params.push(format!("{ty} d{n} [[color({n})]]"));
                    }
                }
                out.push_str(&format!(
                    "fragment PsOut gen5_ps({} [[stage_in]], {}) {{\n    PsOut o = {{}};\n    bool exported = false;\n{init}{}{tail}}}\n",
                    if shared { "PsInRaw raw" } else { "PsIn in" },
                    entry_params.join(", "),
                    body
                ));
            }
            Stage::Compute => {
                init.push_str(&self.cs_inputs());
                if self.req.grid_bounds {
                    entry_params.push(format!(
                        "device const uint* bounds [[buffer({})]]",
                        binding::BOUNDS
                    ));
                    init.push_str(&self.bounds_init());
                }
                let mut lds = if self.lds {
                    "    threadgroup uint lds[LDS_BYTES / 4];\n"
                } else {
                    ""
                }
                .to_string();
                if self.guarded && (self.has_barrier || self.lockstep) {
                    lds.push_str("    threadgroup bool tg_bail;\n    uint bail_at = 0;\n    if (all(lid == 0)) tg_bail = false;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
                }
                let mut params = entry_params.join(", ");
                if self.lockstep {
                    let waves = self.waves();
                    lds.push_str(&format!(
                        "    threadgroup uint lockstep_vote[{}];\n    threadgroup atomic_uint lockstep_seq[{}];\n",
                        4 * waves,
                        2 * waves
                    ));
                    if waves > 1 {
                        lds.push_str("    if (slane == 0) atomic_store_explicit(&lockstep_seq[sgi], 0u, memory_order_relaxed);\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
                    }
                    params.push_str(", ushort sgi [[simdgroup_index_in_threadgroup]]");
                }
                out.push_str(&format!(
                    "kernel void gen5_cs({params}, uint3 lid [[thread_position_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {{\n{lds}{init}{}}}\n",
                    self.body
                ));
            }
        }
        if self.floats {
            out = out.replacen(float::PRELUDE_HERE, &float::prelude_for(&out), 1);
        }
        out
    }

    /// The MODE register a wave starts with: the float mode, DX10 clamp, IEEE
    /// mode and FP16 overflow bits of `PGM_RSRC1`, and a compute program's
    /// exception enables from `COMPUTE_PGM_RSRC2`.
    fn initial_mode(&self) -> u32 {
        let r1 = self.req.rsrc1;
        let mut mode = (r1 >> 12 & 0xFF) | (r1 >> 21 & 1) << 8 | (r1 >> 23 & 1) << 9;
        mode |= (r1 >> 26 & 1) << 23;
        if self.req.stage == Stage::Compute {
            let r2 = self.req.compute_rsrc2;
            mode |= (r2 >> 24 & 0x7F) << 12 | (r2 >> 13 & 3) << 19;
        }
        mode
    }

    /// Replaces the `VREAD`/`VWRITE` markers by one case per VGPR up to the
    /// highest the program names, all of which are then declared.
    fn fill_movrel(&mut self) {
        let top = self
            .regs
            .iter()
            .filter_map(|r| r.strip_prefix('v')?.parse::<u16>().ok())
            .max()
            .unwrap_or(0);
        for n in 0..=top {
            self.regs.insert(format!("v{n}"));
        }
        let mut body = std::mem::take(&mut self.body);
        for (marker, read) in [(VREAD, true), (VWRITE, false)] {
            while let Some(at) = body.find(marker) {
                let rest = &body[at + marker.len()..];
                let end = rest.find(VEND).unwrap();
                let var = &rest[..end];
                let cases: String = (0..=top)
                    .map(|n| {
                        if read {
                            format!("case {n}: {var} = v{n}; break; ")
                        } else {
                            format!("case {n}: v{n} = {var}; break; ")
                        }
                    })
                    .collect();
                body = format!(
                    "{}{cases}default: break;{}",
                    &body[..at],
                    &rest[end + VEND.len()..]
                );
            }
        }
        self.body = body;
    }

    // Workgroup ids after the user SGPRs and thread ids in v0..v2, as `COMPUTE_PGM_RSRC2` enables them.
    fn cs_inputs(&self) -> String {
        let rsrc2 = self.req.compute_rsrc2;
        let mut s = String::new();
        let mut next = self.req.user_sgprs as u16;
        let threads = self.req.threads;
        let ids = [
            (7, "gid.x".to_string()),
            (8, "gid.y".to_string()),
            (9, "gid.z".to_string()),
            (10, format!("{}u", threads[0] * threads[1] * threads[2])),
        ];
        for (bit, value) in ids {
            if rsrc2 >> bit & 1 != 0 {
                if self.regs.contains(&format!("s{next}")) {
                    s.push_str(&format!("    s{next} = {value};\n"));
                }
                next += 1;
            }
        }
        let comps = (rsrc2 >> 11 & 3).min(2);
        for (i, c) in ["x", "y", "z"].iter().enumerate().take(comps as usize + 1) {
            if self.regs.contains(&format!("v{i}")) {
                s.push_str(&format!("    v{i} = lid.{c};\n"));
            }
        }
        s
    }

    /// The CB's blend of each exported MRT that `Request::blends` names with
    /// the target it reads, run before the pixel program returns.
    fn blend_statements(&self) -> String {
        let src1 = if self.req.dual_source && self.mrts & 2 != 0 {
            "src1"
        } else {
            "float4(0.0f)"
        };
        let mut out = String::new();
        for n in 0..8 {
            let w = self.req.blends[n];
            if w == 0 || self.mrts >> n & 1 == 0 || n == 1 && src1 == "src1" {
                continue;
            }
            let packed = self.req.packed_targets[n];
            if packed != 0 {
                let (layout, number) = (packed & 0x1F, packed >> 5 & 7);
                let word = format!(
                    "cb_pack(cb_blend(blend_src{n}, cb_unpack(d{n}, {layout}u, {number}u), {src1}, blend_color, {w:#x}u), {layout}u, {number}u)"
                );
                out.push_str(&format!(
                    "o.c{n} = {}; ",
                    mem::packed_store(n, &word, packed)
                ));
                continue;
            }
            let blend =
                format!("cb_blend(blend_src{n}, float4(d{n}), {src1}, blend_color, {w:#x}u)");
            if self.req.half_targets >> n & 1 != 0 {
                out.push_str(&format!("o.c{n} = cb_half({blend}); "));
            } else if self.req.srgb_targets >> n & 1 != 0 {
                out.push_str(&format!("o.c{n} = cb_srgb_snap({blend}); "));
            } else {
                out.push_str(&format!("o.c{n} = {blend}; "));
            }
        }
        out
    }

    // Initial VGPRs of a pixel shader, laid out by `SPI_PS_INPUT_ADDR`.
    fn ps_vgprs(&self) -> String {
        let mut s = String::new();
        let mut reg = 0u16;
        const SIZES: [u16; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        for (bit, &n) in SIZES.iter().enumerate() {
            if self.req.ps_input_addr >> bit & 1 == 0 {
                continue;
            }
            let on = self.req.ps_input_ena >> bit & 1 != 0;
            let value = match bit {
                8 => Some("U(in.pos.x)"),
                9 => Some("U(in.pos.y)"),
                10 => Some("U(in.pos.z)"),
                // Metal's position w is 1/w; the hardware's POS_W is w.
                11 => Some("U(1.0f / in.pos.w)"),
                12 => Some("(in.front ? 0x3f800000u : 0xbf800000u)"),
                // ANCILLARY: the Deck gives 2 in bits 0 and 1 for triangles
                // (other primitive types not measured), sample 0 in bits 8 to 11.
                13 => Some("2u"),
                // Single-sampled targets: the one sample is covered.
                14 => Some("1u"),
                15 => Some("(uint(in.pos.x) | uint(in.pos.y) << 16)"),
                _ => None,
            };
            if let Some(v) = value.filter(|_| on && self.regs.contains(&format!("v{reg}"))) {
                s.push_str(&format!("    v{reg} = {v};\n"));
            }
            reg += n;
        }
        s
    }
}

/// The Metal interpolation qualifier of a pixel input.
fn qualifier(mode: Interp) -> &'static str {
    match mode {
        Interp::Perspective => "center_perspective",
        Interp::PerspectiveCentroid => "centroid_perspective",
        Interp::PerspectiveSample => "sample_perspective",
        Interp::Linear => "center_no_perspective",
        Interp::LinearCentroid => "centroid_no_perspective",
        Interp::LinearSample => "sample_no_perspective",
        Interp::Flat => "flat",
    }
}
