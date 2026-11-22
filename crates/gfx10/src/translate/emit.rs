//! Entry-point emission, stage I/O setup and movrel expansion.

use super::*;

impl<'a> Tx<'a> {
    pub(super) fn finish(mut self) -> Translated {
        let mut resources = Vec::new();
        for (i, (_, kind, source, access)) in self.textures.iter().enumerate() {
            resources.push(Resource::Texture {
                slot: i as u8,
                source: source.clone(),
                kind: *kind,
                access: *access,
            });
        }
        for (i, source) in self.samplers.iter().enumerate() {
            resources.push(Resource::Sampler {
                slot: i as u8,
                source: source.clone(),
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
        }
    }

    /// Clears EXEC for threads past the grid size in `bounds`. A lockstep
    /// wave is its whole one-dimensional group, so its active lanes are a prefix.
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
            "    {{ uint first = gid.x * {}u; uint n = (bounds[0] > first && gid.y < bounds[1] && gid.z < bounds[2]) ? min(bounds[0] - first, 64u) : 0u;\n",
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
        if declared == 0 {
            32 << 10
        } else {
            declared.clamp(16, 32 << 10)
        }
    }

    fn emit(&mut self) -> String {
        let mut out = String::from(PRELUDE);
        if self.lockstep {
            self.simd_lane = true;
            out.push_str(WAVE_PRELUDE);
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
        if self.guarded {
            out.push_str(&format!(
                "#define GUARD_LIMIT {MAX_COMPUTE_BACK_BRANCHES}u\n#define GUARD_BUDGET {GUARD_BUDGET}u\n#define GUARD_UNIT {GUARD_UNIT}u\n{GUARD_PRELUDE}"
            ));
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
        for (i, &(_, kind, _, access)) in self.textures.iter().enumerate() {
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
            init.push_str(
                &format!("    uint {r} = 0u;\n").replace("uint scc = 0u", "bool scc = false"),
            );
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
        if self.lockstep {
            init.push_str("    uint wl = uint(sgi) * 32u + uint(slane);\n    uint wk = 0u;\n");
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
                    let q = match mode {
                        Interp::Perspective => "center_perspective",
                        Interp::PerspectiveCentroid => "centroid_perspective",
                        Interp::PerspectiveSample => "sample_perspective",
                        Interp::Linear => "center_no_perspective",
                        Interp::LinearCentroid => "centroid_no_perspective",
                        Interp::LinearSample => "sample_no_perspective",
                        Interp::Flat => "flat",
                    };
                    // Flat inputs carry raw bits (often integers), which a float
                    // varying may not keep; vertex variants export them as uint4.
                    let ty = if mode == Interp::Flat {
                        "uint4"
                    } else {
                        "float4"
                    };
                    out.push_str(&format!("    {ty} a{a} [[user(locn{loc}), {q}]];\n"));
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
                    init.insert_str(0, &(fill + " };\n"));
                }
                out.push_str("};\n\nstruct PsOut {\n");
                for i in 0..8 {
                    if self.mrts >> i & 1 != 0 {
                        let ty = match self.req.col_format >> (4 * i) & 0xF {
                            7 => "uint4",
                            8 => "int4",
                            _ => "float4",
                        };
                        out.push_str(&format!("    {ty} c{i} [[color({i})]];\n"));
                    }
                }
                if self.depth {
                    out.push_str("    float depth [[depth(any)]];\n");
                }
                if self.sample_mask {
                    out.push_str("    uint mask [[sample_mask]];\n");
                }
                out.push_str("};\n\n");
                init.push_str(&self.ps_vgprs());
                if self.regs.contains("m0") {
                    init.push_str("    m0 = 0u;\n");
                }
                let mut body = self.body.clone();
                let mut tail =
                    String::from("    if (!exported) discard_fragment();\n    return o;\n");
                if self.req.probe_flat
                    && self.mrts & 2 != 0
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
                    lds.push_str("    threadgroup uint lockstep_vote[4];\n");
                    params.push_str(", ushort sgi [[simdgroup_index_in_threadgroup]]");
                }
                out.push_str(&format!(
                    "kernel void gen5_cs({params}, uint3 lid [[thread_position_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {{\n{lds}{init}{}}}\n",
                    self.body
                ));
            }
        }
        out
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
                11 => Some("U(in.pos.w)"),
                12 => Some("(in.front ? 0x3f800000u : 0xbf800000u)"),
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
