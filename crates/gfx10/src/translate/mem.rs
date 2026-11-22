//! Interpolation, exports, flat/global, buffer and LDS/GDS instructions.

use super::*;

impl<'a> Tx<'a> {
    fn input_interp(&self, vsrc: u16) -> Interp {
        self.ij_vgprs
            .get(&vsrc)
            .copied()
            .unwrap_or(Interp::Perspective)
    }

    pub(super) fn vintrp(
        &mut self,
        pc: usize,
        name: &str,
        vdst: u16,
        vsrc: u16,
        attr: u8,
        chan: u8,
    ) -> Result<(), TranslateError> {
        if self.req.stage != Stage::Pixel {
            return Err(err(pc, "interpolation outside a pixel shader"));
        }
        // SPI_PS_INPUT_CNTL's FLAT_SHADE (bit 10) makes interpolation return the
        // provoking vertex's value whatever the instruction.
        let flat_shade = self
            .req
            .ps_input_cntl
            .get(attr as usize)
            .is_some_and(|c| c & 0x400 != 0);
        let mode = if name == "v_interp_mov_f32" || flat_shade {
            Interp::Flat
        } else {
            self.input_interp(vsrc)
        };
        match self.attrs.get(&attr) {
            Some(m) if *m != mode => {
                return Err(err(
                    pc,
                    format!("attribute {attr} read with two interpolation modes"),
                ));
            }
            _ => {
                self.attrs.insert(attr, mode);
            }
        }
        if name == "v_interp_mov_f32" && vsrc > 2 {
            return Err(err(pc, "bad interpolation parameter"));
        }
        let c = ['x', 'y', 'z', 'w'][chan as usize];
        let d = self.v(vdst);
        if mode == Interp::Flat {
            self.guard(&format!("{d} = in.a{attr}.{c};"));
        } else {
            self.guard(&format!("{d} = U(in.a{attr}.{c});"));
        }
        Ok(())
    }

    pub(super) fn exp(
        &mut self,
        pc: usize,
        en: u8,
        target: u8,
        compr: bool,
        done: bool,
        vsrc: [u8; 4],
    ) -> Result<(), TranslateError> {
        let mut comps = Vec::new();
        for i in 0..4 {
            let reg = if compr { vsrc[i / 2] } else { vsrc[i] } as u16;
            comps.push((en >> i & 1 != 0, self.v(reg)));
        }
        let float4 = |c: &[(bool, String)]| {
            let parts: Vec<String> = c
                .iter()
                .map(|(on, r)| {
                    if *on {
                        format!("F({r})")
                    } else {
                        "0.0f".into()
                    }
                })
                .collect();
            format!("float4({})", parts.join(", "))
        };
        let stmt = match (self.req.stage, target) {
            (Stage::Vertex, 12) => format!("o.pos = {};", float4(&comps)),
            (Stage::Vertex, 13..=16) | (Stage::Vertex, 20) | (_, 9) => String::new(),
            (Stage::Vertex, 32..=63) => {
                let n = target - 32;
                self.params.insert(n);
                format!("o.p{n} = {};", float4(&comps))
            }
            (Stage::Pixel, 0..=7) => {
                self.mrts |= 1 << target;
                let fmt = self.req.col_format >> (4 * target) & 0xF;
                let value = if compr {
                    let (lo, hi) = (&comps[0].1, &comps[2].1);
                    match fmt {
                        5 => format!(
                            "float4(unpack_unorm2x16_to_float({lo}), unpack_unorm2x16_to_float({hi}))"
                        ),
                        6 => format!(
                            "float4(unpack_snorm2x16_to_float({lo}), unpack_snorm2x16_to_float({hi}))"
                        ),
                        7 => {
                            format!("uint4({lo} & 0xffffu, {lo} >> 16, {hi} & 0xffffu, {hi} >> 16)")
                        }
                        8 => format!(
                            "int4(short({lo}), short({lo} >> 16), short({hi}), short({hi} >> 16))"
                        ),
                        _ => format!(
                            "float4(float2(as_type<half2>({lo})), float2(as_type<half2>({hi})))"
                        ),
                    }
                } else {
                    match fmt {
                        7 => format!(
                            "uint4({})",
                            comps
                                .iter()
                                .map(|(on, r)| if *on { r.clone() } else { "0u".into() })
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                        8 => format!(
                            "int4({})",
                            comps
                                .iter()
                                .map(|(on, r)| if *on { format!("int({r})") } else { "0".into() })
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                        _ => float4(&comps),
                    }
                };
                format!("o.c{target} = {value};")
            }
            (Stage::Pixel, 8) => {
                // MRTZ: depth in R, stencil in G, the sample mask in B.
                let mut stmt = String::new();
                if en & 1 != 0 && !self.req.drop_depth_export {
                    self.depth = true;
                    stmt += &format!("o.depth = F({});", comps[0].1);
                }
                if en & 4 != 0 {
                    self.sample_mask = true;
                    stmt += &format!(" o.mask = {};", comps[2].1);
                }
                stmt
            }
            _ => return Err(err(pc, format!("unsupported export target {target}"))),
        };
        let e = self.s(126);
        if self.req.stage == Stage::Pixel && done {
            self.line(&format!("if (({e} & 1u) == 0u) discard_fragment();"));
        }
        if !stmt.is_empty() {
            self.line(&format!("if ({e} & 1u) {{ {stmt} }}"));
        }
        if self.req.stage == Stage::Pixel {
            self.line("exported = true;");
        }
        Ok(())
    }

    /// FLAT and GLOBAL loads and stores, through the memory window.
    pub(super) fn flat(
        &mut self,
        pc: usize,
        name: &str,
        f: &crate::decode::Flat,
    ) -> Result<(), TranslateError> {
        let op = name.split_once('_').map_or("", |(_, x)| x);
        let store = match op {
            "store_byte" => Some((1u16, "8")),
            "store_short" => Some((1, "16")),
            "store_dword" => Some((1, "")),
            "store_dwordx2" => Some((2, "")),
            "store_dwordx3" => Some((3, "")),
            "store_dwordx4" => Some((4, "")),
            _ => None,
        };
        let (n, call): (u16, &str) = match op {
            _ if store.is_some() => (0, ""),
            "load_ubyte" => (1, "flat_load_ubyte"),
            "load_sbyte" => (1, "flat_load_sbyte"),
            "load_ushort" => (1, "flat_load_ushort"),
            "load_sshort" => (1, "flat_load_sshort"),
            "load_dword" => (1, "flat_load_raw"),
            "load_dwordx2" => (2, "flat_load_raw"),
            "load_dwordx3" => (3, "flat_load_raw"),
            "load_dwordx4" => (4, "flat_load_raw"),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        if name.starts_with("scratch_load") && !self.scratch_stores && !f.lds {
            // Nothing this program runs writes scratch, so a read finds nothing
            // defined; GTA V's ray-tracing stacks read it only past their LDS part.
            let mut s = String::new();
            for i in 0..n {
                write!(s, "{} = 0u; ", self.v(f.vdst as u16 + i)).unwrap();
            }
            self.guard(s.trim_end());
            return Ok(());
        }
        if f.lds || !(name.starts_with("flat_") || name.starts_with("global_")) {
            return Err(err(pc, format!("unsupported instruction {name}")));
        }
        self.mem = true;
        let lo = self.v(f.addr as u16);
        let addr = if name.starts_with("global_") && (f.saddr as u16) < SGPR_COUNT {
            let (s0, s1) = (self.s(f.saddr as u16), self.s(f.saddr as u16 + 1));
            format!("((ulong({s0}) | (ulong({s1}) << 32)) + ulong({lo}))")
        } else {
            let hi = self.v(f.addr as u16 + 1);
            format!("(ulong({lo}) | (ulong({hi}) << 32))")
        };
        if let Some((n, width)) = store {
            self.mem_write = true;
            self.stores = None;
            let mut s = format!(
                "long o = flat_store_offset(mem_base, {addr} + ulong(long({}))); if (o != OOB_OFF) {{",
                f.offset
            );
            for i in 0..n {
                let v = self.v(f.data as u16 + i);
                match width {
                    "8" => write!(s, " mem[o] = uchar({v});").unwrap(),
                    "16" => write!(s, " *(device ushort*)(mem + o) = ushort({v});").unwrap(),
                    _ if f.glc => write!(s, " store_atomic(mem, o + {}, {v});", 4 * i).unwrap(),
                    _ => write!(s, " *(device uint*)(mem + o + {}) = {v};", 4 * i).unwrap(),
                }
            }
            s.push_str(" }");
            self.guard(&s);
            return Ok(());
        }
        let mut s = format!(
            "uint4 r = {call}(mem, goff(mem_base, {addr} + ulong(long({}))));",
            f.offset
        );
        for i in 0..n {
            let d = self.v(f.vdst as u16 + i);
            write!(s, " {d} = r[{i}];").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    pub(super) fn mubuf(
        &mut self,
        pc: usize,
        name: &str,
        b: &crate::decode::Buf,
        format: Option<u8>,
    ) -> Result<(), TranslateError> {
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            return Ok(());
        }
        if b.lds || b.tfe {
            return Err(err(pc, "unsupported LDS or TFE buffer access"));
        }
        self.mem = true;
        let r = b.srsrc as u16;
        let mut vs: Vec<String> = (0..4).map(|i| self.s(r + i)).collect();
        if let Some(f) = format {
            vs[3] = format!("(({} & ~0x7f000u) | {}u)", vs[3], (f as u32 & 0x7f) << 12);
        }
        let index = if b.idxen {
            self.v(b.vaddr as u16)
        } else {
            "0u".into()
        };
        let voff = if b.offen {
            self.v(b.vaddr as u16 + b.idxen as u16)
        } else {
            "0u".into()
        };
        let so = self.src32(b.soffset as u16, None, pc)?;
        let args = format!(
            "mem_base, uint4({}, {}, {}, {}), {index}, {voff} + {so} + {}u, {}u",
            vs[0], vs[1], vs[2], vs[3], b.offset, b.idxen as u32
        );
        let vd = b.vdata as u16;
        if let Some(op) = name.strip_prefix("buffer_atomic_") {
            self.mem_write = true;
            self.note_store(pc, r, b.idxen);
            let data = self.v(vd);
            let call = match op {
                "add" | "sub" | "and" | "or" | "xor" | "umin" | "umax" | "smin" | "smax"
                | "swap" | "fmin" | "fmax" => {
                    format!("atomic_rmw_{op}(mem, buffer_store_offset({args}), {data})")
                }
                "cmpswap" => {
                    let cmp = self.v(vd + 1);
                    format!("atomic_cas(mem, buffer_store_offset({args}), {data}, {cmp})")
                }
                // No 64-bit atomics here: each half alone, the high half (a
                // flag in look-back scans) read before and written after the low.
                "and_x2" | "or_x2" | "xor_x2" | "swap_x2" => {
                    let f = &op[..op.len() - 3];
                    let hi = self.v(vd + 1);
                    let (first, second) = if f == "swap" {
                        (
                            format!("lo = atomic_rmw_swap(mem, o, {data});"),
                            format!("hi = atomic_rmw_swap(mem, o4, {hi});"),
                        )
                    } else {
                        (
                            format!("hi = atomic_rmw_{f}(mem, o4, {hi});"),
                            format!("lo = atomic_rmw_{f}(mem, o, {data});"),
                        )
                    };
                    let stmt = format!(
                        "{{ long o = buffer_store_offset({args}); long o4 = o == OOB_OFF ? o : o + 4; uint lo, hi; {first} {second}{} }}",
                        if b.glc {
                            format!(" {data} = lo; {hi} = hi;")
                        } else {
                            String::new()
                        }
                    );
                    self.guard(&stmt);
                    return Ok(());
                }
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            let stmt = if b.glc {
                format!("{data} = {call};")
            } else {
                format!("(void){call};")
            };
            self.guard(&stmt);
            return Ok(());
        }
        let store: Option<(u16, &str)> = match name {
            "buffer_store_dword" => Some((1, "")),
            "buffer_store_dwordx2" => Some((2, "")),
            "buffer_store_dwordx3" => Some((3, "")),
            "buffer_store_dwordx4" => Some((4, "")),
            "buffer_store_byte" => Some((1, "8")),
            "buffer_store_short" => Some((1, "16")),
            _ => None,
        };
        if let Some((n, width)) = store {
            self.mem_write = true;
            self.note_store(pc, r, b.idxen);
            let mut s = format!("long o = buffer_store_offset({args}); if (o != OOB_OFF) {{");
            for i in 0..n {
                let v = self.v(vd + i);
                match width {
                    "8" => write!(s, " mem[o] = uchar({v});").unwrap(),
                    "16" => write!(s, " *(device ushort*)(mem + o) = ushort({v});").unwrap(),
                    _ if b.glc => write!(s, " store_atomic(mem, o + {}, {v});", 4 * i).unwrap(),
                    _ => write!(s, " *(device uint*)(mem + o + {}) = {v};", 4 * i).unwrap(),
                }
            }
            s.push_str(" }");
            self.guard(&s);
            return Ok(());
        }
        let store_format = match name {
            "buffer_store_format_x" => Some(1u16),
            "buffer_store_format_xy" => Some(2),
            "buffer_store_format_xyz" => Some(3),
            "buffer_store_format_xyzw" => Some(4),
            _ => None,
        };
        if let Some(n) = store_format {
            self.mem_write = true;
            self.note_store(pc, r, b.idxen);
            let vals: Vec<String> = (0..4)
                .map(|i| if i < n { self.v(vd + i) } else { "0u".into() })
                .collect();
            let stmt = format!(
                "buffer_store_format(mem, {args}, uint4({}, {}, {}, {}), {n}u);",
                vals[0], vals[1], vals[2], vals[3]
            );
            self.guard(&stmt);
            return Ok(());
        }
        let (n, call): (u16, &str) = match name {
            "buffer_load_format_x" => (1, "buffer_load_format"),
            "buffer_load_format_xy" => (2, "buffer_load_format"),
            "buffer_load_format_xyz" => (3, "buffer_load_format"),
            "buffer_load_format_xyzw" => (4, "buffer_load_format"),
            "buffer_load_dword" => (1, "buffer_load_raw"),
            "buffer_load_dwordx2" => (2, "buffer_load_raw"),
            "buffer_load_dwordx3" => (3, "buffer_load_raw"),
            "buffer_load_dwordx4" => (4, "buffer_load_raw"),
            "buffer_load_ubyte" => (1, "buffer_load_ubyte"),
            "buffer_load_sbyte" => (1, "buffer_load_sbyte"),
            "buffer_load_ushort" => (1, "buffer_load_ushort"),
            "buffer_load_sshort" => (1, "buffer_load_sshort"),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let mut s = if b.glc && call == "buffer_load_raw" {
            let coherent = if self.mem_write {
                "buffer_load_atomic"
            } else {
                "buffer_load_volatile"
            };
            format!("uint4 r = {coherent}(mem, {args}, {n}u);")
        } else {
            format!("uint4 r = {call}(mem, {args});")
        };
        for i in 0..n {
            let d = self.v(vd + i);
            write!(s, " {d} = r[{i}];").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    pub(super) fn ds(
        &mut self,
        pc: usize,
        name: &str,
        offset0: u8,
        offset1: u8,
        gds: bool,
        regs: [u16; 4],
    ) -> Result<(), TranslateError> {
        if matches!(name, "ds_append" | "ds_consume") {
            return self.ds_counter(
                pc,
                name,
                offset0 as u32 | (offset1 as u32) << 8,
                gds,
                regs[3],
            );
        }
        if name == "ds_bpermute_b32" && !gds && self.req.stage != Stage::Vertex {
            // Metal SIMD groups are 32 lanes, so the source lane is taken mod 32.
            let [addr, data0, _, vdst] = regs;
            let (a, d0, d) = (self.v(addr), self.v(data0), self.v(vdst));
            let off = offset0 as u32 | (offset1 as u32) << 8;
            self.line(&format!(
                "{{ uint bp = simd_shuffle({d0}, ushort((({a} + {off}u) >> 2) & 31u));"
            ));
            self.guard(&format!("{d} = bp;"));
            self.line("}");
            return Ok(());
        }
        if gds || self.req.stage != Stage::Compute {
            return Err(err(pc, format!("unsupported LDS access {name}")));
        }
        self.lds = true;
        if name == "ds_write_addtid_b32" || name == "ds_read_addtid_b32" {
            // Wave64 assumed: the lane is the thread's index in the group mod 64.
            let [_, data0, _, vdst] = regs;
            let m0 = self.s(124);
            let [tx, ty, _] = self.req.threads;
            let a = format!(
                "(({m0} & 0xffffu) + {}u + 4u * ((lid.x + lid.y * {tx}u + lid.z * {}u) & 63u))",
                offset0 as u32 | (offset1 as u32) << 8,
                tx * ty
            );
            let stmt = if name == "ds_write_addtid_b32" {
                let d = self.v(data0);
                format!("lds_wr(lds, {a}, {d});")
            } else {
                let d = self.v(vdst);
                format!("{d} = lds_rd(lds, {a});")
            };
            self.guard(&stmt);
            return Ok(());
        }
        let [addr, data0, data1, vdst] = regs;
        let a = self.v(addr);
        let off = offset0 as u32 | (offset1 as u32) << 8;
        let pair = |scale: u32| (offset0 as u32 * scale, offset1 as u32 * scale);
        let d0 = |t: &mut Self, i: u16| t.v(data0 + i);
        let d1 = |t: &mut Self, i: u16| t.v(data1 + i);
        let dst = |t: &mut Self, i: u16| t.v(vdst + i);
        let mut s = format!("uint a = {a};");
        match name {
            "ds_nop" => return Ok(()),
            "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" => {
                let n = match name {
                    "ds_write_b32" => 1,
                    "ds_write_b64" => 2,
                    "ds_write_b96" => 3,
                    _ => 4,
                };
                for i in 0..n {
                    let v = d0(self, i);
                    write!(s, " lds_wr(lds, a + {}u, {v});", off + 4 * i as u32).unwrap();
                }
            }
            "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
                let wide = name.ends_with("b64");
                let scale = if wide { 8 } else { 4 } * if name.contains("st64") { 64 } else { 1 };
                let (o0, o1) = pair(scale);
                for i in 0..(1 + wide as u16) {
                    let (x, y) = (d0(self, i), d1(self, i));
                    write!(
                        s,
                        " lds_wr(lds, a + {}u, {x}); lds_wr(lds, a + {}u, {y});",
                        o0 + 4 * i as u32,
                        o1 + 4 * i as u32
                    )
                    .unwrap();
                }
            }
            "ds_write_b8" | "ds_write_b16" => {
                let v = d0(self, 0);
                let ty = if name.ends_with("b8") {
                    "uchar"
                } else {
                    "ushort"
                };
                write!(s, " if ((a + {off}u) < LDS_BYTES) *(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u) = {ty}({v});").unwrap();
            }
            "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" => {
                let n = match name {
                    "ds_read_b32" => 1,
                    "ds_read_b64" => 2,
                    "ds_read_b96" => 3,
                    _ => 4,
                };
                for i in 0..n {
                    write!(s, " uint t{i} = lds_rd(lds, a + {}u);", off + 4 * i as u32).unwrap();
                }
                for i in 0..n {
                    let d = dst(self, i);
                    write!(s, " {d} = t{i};").unwrap();
                }
            }
            "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
                let wide = name.ends_with("b64");
                let scale = if wide { 8 } else { 4 } * if name.contains("st64") { 64 } else { 1 };
                let (o0, o1) = pair(scale);
                let n = 1 + wide as u32;
                let mut k = 0;
                for o in [o0, o1] {
                    for i in 0..n {
                        write!(s, " uint t{k} = lds_rd(lds, a + {}u);", o + 4 * i).unwrap();
                        k += 1;
                    }
                }
                for i in 0..k {
                    let d = dst(self, i);
                    write!(s, " {d} = t{i};").unwrap();
                }
            }
            "ds_read_u8" | "ds_read_i8" | "ds_read_u16" | "ds_read_i16" => {
                let ty = match name {
                    "ds_read_u8" => "uchar",
                    "ds_read_i8" => "char",
                    "ds_read_u16" => "ushort",
                    _ => "short",
                };
                let d = dst(self, 0);
                write!(s, " {d} = (a + {off}u) < LDS_BYTES ? uint(int(*(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u))) : 0u;").unwrap();
            }
            _ => {
                let base = name.strip_prefix("ds_").unwrap();
                let rtn = base.contains("_rtn_");
                let op = base.replace("_rtn", "");
                let v = d0(self, 0);
                let call = match op.as_str() {
                    "add_u32" | "sub_u32" | "and_b32" | "or_b32" | "xor_b32" | "min_u32"
                    | "max_u32" | "min_i32" | "max_i32" | "wrxchg_b32" => {
                        let f = op.split('_').next().unwrap();
                        let f = match (f, op.ends_with("i32")) {
                            ("min", true) => "smin",
                            ("max", true) => "smax",
                            ("min", false) => "umin",
                            ("max", false) => "umax",
                            ("wrxchg", _) => "swap",
                            (f, _) => f,
                        };
                        format!("lds_rmw_{f}(lds, a + {off}u, {v})")
                    }
                    "rsub_u32" => format!("lds_rmw_rsub(lds, a + {off}u, {v})"),
                    "min_f32" | "max_f32" => {
                        let f = &op[..3];
                        format!("lds_rmw_f{f}(lds, a + {off}u, {v})")
                    }
                    "cmpst_b32" => {
                        let new = d1(self, 0);
                        format!("lds_cas(lds, a + {off}u, {new}, {v})")
                    }
                    _ => return Err(err(pc, format!("unsupported instruction {name}"))),
                };
                if rtn {
                    let d = dst(self, 0);
                    write!(s, " {d} = {call};").unwrap();
                } else {
                    write!(s, " (void){call};").unwrap();
                }
            }
        }
        self.guard(&s);
        Ok(())
    }

    // `ds_append`/`ds_consume`: a counter at M0's base plus `offset` (for the GDS,
    // the base is M0's high half: Astro Bot's packets clear its counters there). Lanes
    // are threads here (`v_mbcnt` gives 0), so each thread counts itself and
    // gets its own previous value; a wave on hardware adds its active lanes at once.
    fn ds_counter(
        &mut self,
        pc: usize,
        name: &str,
        offset: u32,
        gds: bool,
        vdst: u16,
    ) -> Result<(), TranslateError> {
        let m0 = self.s(124);
        let d = self.v(vdst);
        let op = if name == "ds_append" { "add" } else { "sub" };
        let stmt = if gds {
            self.gds = true;
            format!(
                "{{ uint a = (({m0} >> 16) + {offset}u) & 0xfffcu; {d} = atomic_fetch_{op}_explicit(&gds[a >> 2], 1u, memory_order_relaxed); }}"
            )
        } else {
            if self.req.stage != Stage::Compute {
                return Err(err(pc, format!("unsupported LDS access {name}")));
            }
            self.lds = true;
            format!("{{ {d} = lds_rmw_{op}(lds, ({m0} & 0xffffu) + {offset}u, 1u); }}")
        };
        self.guard(&stmt);
        Ok(())
    }

    /// Records the V# at `srsrc` that a buffer store or atomic at `pc` uses.
    fn note_store(&mut self, pc: usize, srsrc: u16, indexed: bool) {
        let source = self.descriptor_source(pc, srsrc, 4).ok();
        match (source, &mut self.stores) {
            (Some(s), Some(list)) => {
                if !list.contains(&(s.clone(), indexed)) {
                    list.push((s, indexed));
                }
            }
            _ => self.stores = None,
        }
    }
}
