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
        const FLAT_SHADE: u32 = 1 << 10;
        let flat_shaded = self
            .req
            .ps_input_cntl
            .get(attr as usize)
            .is_some_and(|c| c & FLAT_SHADE != 0);
        let mode = if name == "v_interp_mov_f32" || flat_shaded {
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
                            "float4(f16_float({lo} & 0xffffu), f16_float({lo} >> 16), f16_float({hi} & 0xffffu), f16_float({hi} >> 16))"
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
                        _ if self.req.half_targets >> target & 1 != 0
                            && self.req.blends[target as usize] == 0 =>
                        {
                            let parts: Vec<String> = comps
                                .iter()
                                .map(|(on, r)| {
                                    if *on {
                                        format!("f16_trunc(F({r}))")
                                    } else {
                                        "0.0f".into()
                                    }
                                })
                                .collect();
                            format!("float4({})", parts.join(", "))
                        }
                        _ => float4(&comps),
                    }
                };
                let swizzle = self.req.swizzles[target as usize];
                let value = if swizzle == 0 {
                    value
                } else {
                    let sel: String = (0..4)
                        .map(|i| ['x', 'y', 'z', 'w'][(swizzle >> (2 * i) & 3) as usize])
                        .collect();
                    format!("({value}).{sel}")
                };
                let logic = self.req.logic_targets[target as usize];
                let packed = self.req.packed_targets[target as usize];
                if self.req.logic_op != 0xCC && logic != 0 {
                    logic_op(target, &value, self.req.logic_op, logic)
                } else if target == 1 && self.req.dual_source {
                    format!("src1 = {value};")
                } else if self.req.blends[target as usize] != 0 {
                    format!("blend_src{target} = {value};")
                } else if packed != 0 {
                    let word = match packed >> 5 & 7 {
                        4 => format!("cb_pack_u(uint4({value}), {}u)", packed & 0x1F),
                        5 => format!("cb_pack_s(int4({value}), {}u)", packed & 0x1F),
                        n => format!("cb_pack(float4({value}), {}u, {n}u)", packed & 0x1F),
                    };
                    format!("o.c{target} = {};", packed_store(target, &word, packed))
                } else if self.req.srgb_targets >> target & 1 != 0 && !matches!(fmt, 7 | 8) {
                    format!("o.c{target} = cb_srgb_snap({value});")
                } else {
                    format!("o.c{target} = {value};")
                }
            }
            (Stage::Pixel, 8) => {
                // MRTZ: depth in R, stencil in G, the sample mask in B.
                let mut stmt = String::new();
                if en & 1 != 0 && !self.req.drop_depth_export {
                    self.depth = true;
                    stmt += &format!("o.depth = F({});", comps[0].1);
                }
                // Targets are single-sampled, and a `[[sample_mask]]` output
                // turns off Metal's stencil test (measured, gpu.md).
                if en & 4 != 0 {
                    stmt += &format!(" if (({} & 1u) == 0u) discard_fragment();", comps[2].1);
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

    /// FLAT and GLOBAL loads, stores and atomics, through the memory window.
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
            "store_byte_d16_hi" => Some((1, "8hi")),
            "store_short_d16_hi" => Some((1, "16hi")),
            "store_dword" => Some((1, "")),
            "store_dwordx2" => Some((2, "")),
            "store_dwordx3" => Some((3, "")),
            "store_dwordx4" => Some((4, "")),
            _ => None,
        };
        let (n, call, d16): (u16, &str, Option<bool>) = match op {
            _ if store.is_some() || op.starts_with("atomic_") => (0, "", None),
            "load_ubyte" => (1, "flat_load_ubyte", None),
            "load_sbyte" => (1, "flat_load_sbyte", None),
            "load_ushort" => (1, "flat_load_ushort", None),
            "load_sshort" => (1, "flat_load_sshort", None),
            "load_ubyte_d16" => (1, "flat_load_ubyte", Some(false)),
            "load_ubyte_d16_hi" => (1, "flat_load_ubyte", Some(true)),
            "load_sbyte_d16" => (1, "flat_load_sbyte", Some(false)),
            "load_sbyte_d16_hi" => (1, "flat_load_sbyte", Some(true)),
            "load_short_d16" => (1, "flat_load_ushort", Some(false)),
            "load_short_d16_hi" => (1, "flat_load_ushort", Some(true)),
            "load_dword" => (1, "flat_load_raw", None),
            "load_dwordx2" => (2, "flat_load_raw", None),
            "load_dwordx3" => (3, "flat_load_raw", None),
            "load_dwordx4" => (4, "flat_load_raw", None),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        if name.starts_with("scratch_load") && !self.scratch_stores && !f.lds {
            // Unwritten scratch reads zero rather than rejecting the program.
            let mut s = String::new();
            for i in 0..n {
                write!(s, "{} = 0u; ", self.v(f.vdst as u16 + i)).unwrap();
            }
            self.guard(s.trim_end());
            return Ok(());
        }
        if name.starts_with("scratch_") && !f.lds {
            return self.scratch(pc, op, f, store, n, d16);
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
        let at = format!("{addr} + ulong(long({}))", f.offset);
        if let Some(atom) = op.strip_prefix("atomic_") {
            self.mem_write = true;
            self.stores = None;
            let off = |_: u32| format!("flat_store_offset(mem_base, {at})");
            let ret = f.glc.then_some(f.vdst as u16);
            let stmt = self.atomic(pc, name, atom, &off, f.data as u16, ret)?;
            self.guard(&stmt);
            return Ok(());
        }
        if let Some((n, width)) = store {
            self.mem_write = true;
            self.stores = None;
            let mut s = format!("long o = flat_store_offset(mem_base, {at}); if (o != OOB_OFF) {{");
            for i in 0..n {
                let v = self.v(f.data as u16 + i);
                match width {
                    "8" => write!(s, " mem[o] = uchar({v});").unwrap(),
                    "8hi" => write!(s, " mem[o] = uchar({v} >> 16);").unwrap(),
                    "16" => write!(s, " *(device ushort*)(mem + o) = ushort({v});").unwrap(),
                    "16hi" => {
                        write!(s, " *(device ushort*)(mem + o) = ushort({v} >> 16);").unwrap()
                    }
                    _ if f.glc => write!(s, " store_atomic(mem, o + {}, {v});", 4 * i).unwrap(),
                    _ => write!(s, " *(device uint*)(mem + o + {}) = {v};", 4 * i).unwrap(),
                }
            }
            s.push_str(" }");
            self.guard(&s);
            return Ok(());
        }
        let mut s = format!("uint4 r = {call}(mem, goff(mem_base, {at}));");
        let d = self.v(f.vdst as u16);
        match d16 {
            Some(true) => write!(s, " {d} = ({d} & 0xffffu) | (r.x << 16);").unwrap(),
            Some(false) => write!(s, " {d} = ({d} & 0xffff0000u) | (r.x & 0xffffu);").unwrap(),
            None => {
                for i in 0..n {
                    let d = self.v(f.vdst as u16 + i);
                    write!(s, " {d} = r[{i}];").unwrap();
                }
            }
        }
        self.guard(&s);
        Ok(())
    }

    /// SCRATCH loads and stores, in the lane's own `scratch_mem` (a program that
    /// never stores to scratch reads zero instead, above).
    fn scratch(
        &mut self,
        pc: usize,
        op: &str,
        f: &crate::decode::Flat,
        store: Option<(u16, &str)>,
        n: u16,
        d16: Option<bool>,
    ) -> Result<(), TranslateError> {
        if self.scratch_bytes() == 0 {
            return Err(err(
                pc,
                "scratch store with no scratch memory (TMPRING_SIZE)",
            ));
        }
        let base = if (f.saddr as u16) < SGPR_COUNT {
            self.s(f.saddr as u16)
        } else {
            self.v(f.addr as u16)
        };
        let a = format!("{base} + uint({})", f.offset as i32);
        let mut s = format!("uint a = {a};");
        if let Some((n, width)) = store {
            for i in 0..n {
                let v = self.v(f.data as u16 + i);
                let (bytes, v) = match width {
                    "8" => (1, v),
                    "8hi" | "16hi" => (1 + (width == "16hi") as u32, format!("({v} >> 16)")),
                    "16" => (2, v),
                    _ => (4, v),
                };
                write!(
                    s,
                    " scratch_st(scratch_mem, a + {}u, {bytes}u, {v});",
                    4 * i
                )
                .unwrap();
            }
            self.guard(&s);
            return Ok(());
        }
        let (bytes, signed) = match op {
            "load_ubyte" | "load_ubyte_d16" | "load_ubyte_d16_hi" => (1, false),
            "load_sbyte" | "load_sbyte_d16" | "load_sbyte_d16_hi" => (1, true),
            "load_ushort" | "load_short_d16" | "load_short_d16_hi" => (2, false),
            "load_sshort" => (2, true),
            _ => (4, false),
        };
        let read = |k: u16| {
            let r = format!("scratch_ld(scratch_mem, a + {}u, {bytes}u)", 4 * k);
            match (signed, bytes) {
                (true, 1) => format!("uint(int(char({r})))"),
                (true, _) => format!("uint(int(short({r})))"),
                _ => r,
            }
        };
        let d = self.v(f.vdst as u16);
        match d16 {
            Some(true) => write!(s, " {d} = ({d} & 0xffffu) | ({} << 16);", read(0)).unwrap(),
            Some(false) => {
                write!(s, " {d} = ({d} & 0xffff0000u) | ({} & 0xffffu);", read(0)).unwrap()
            }
            None => {
                for k in 0..n {
                    write!(s, " uint t{k} = {};", read(k)).unwrap();
                }
                for k in 0..n {
                    let d = self.v(f.vdst as u16 + k);
                    write!(s, " {d} = t{k};").unwrap();
                }
            }
        }
        self.guard(&s);
        Ok(())
    }

    /// Bytes of scratch each lane owns.
    pub(super) fn scratch_bytes(&self) -> u32 {
        (self.req.scratch_wave_bytes / if self.wave32 { 32 } else { 64 }) & !3
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
        let rsrc = format!("uint4({}, {}, {}, {})", vs[0], vs[1], vs[2], vs[3]);
        let access = format!(
            "{rsrc}, {index}, {voff} + {}u, {so}, {}u",
            b.offset, b.idxen as u32
        );
        let args = format!("mem_base, {access}");
        let sel = format.is_none() as u32;
        let vd = b.vdata as u16;
        if let Some(op) = name.strip_prefix("buffer_atomic_") {
            self.mem_write = true;
            self.note_store(pc, r, b.idxen);
            let off = |bytes: u32| format!("buffer_store_offset({args}, {bytes}u)");
            let stmt = self.atomic(pc, name, op, &off, vd, b.glc.then_some(vd))?;
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
            "buffer_store_byte_d16_hi" => Some((1, "8hi")),
            "buffer_store_short_d16_hi" => Some((1, "16hi")),
            _ => None,
        };
        if let Some((n, width)) = store {
            self.mem_write = true;
            self.note_store(pc, r, b.idxen);
            let v = self.v(vd);
            let narrow = match width {
                "8" => Some((1, format!("mem[o] = uchar({v});"))),
                "8hi" => Some((1, format!("mem[o] = uchar({v} >> 16);"))),
                "16" => Some((2, format!("*(device ushort*)(mem + o) = ushort({v});"))),
                "16hi" => Some((
                    2,
                    format!("*(device ushort*)(mem + o) = ushort({v} >> 16);"),
                )),
                _ => None,
            };
            if let Some((bytes, write)) = narrow {
                self.guard(&format!(
                    "long o = buffer_store_offset({args}, {bytes}u); if (o != OOB_OFF) {write}"
                ));
                return Ok(());
            }
            let mut s = format!("long o0 = buffer_store_dwords({args});");
            for i in 0..n {
                let v = self.v(vd + i);
                let write = if b.glc {
                    format!("store_atomic(mem, o, {v});")
                } else {
                    format!("*(device uint*)(mem + o) = {v};")
                };
                write!(
                    s,
                    " {{ long o = buffer_dword(o0, {access}, {i}u); if (o != OOB_OFF) {write} }}"
                )
                .unwrap();
            }
            self.guard(&s);
            return Ok(());
        }
        if let Some(comps) = name.strip_prefix("buffer_store_format_") {
            self.mem_write = true;
            self.note_store(pc, r, b.idxen);
            let (d16, hi, comps) = match comps.strip_prefix("d16_") {
                Some(c) => match c.strip_prefix("hi_") {
                    Some(c) => (true, true, c),
                    None => (true, false, c),
                },
                None => (false, false, comps),
            };
            let n = comps.len() as u16;
            let vals: Vec<String> = if d16 {
                // Two 16-bit components per VGPR, low half first.
                (0..4)
                    .map(|i| {
                        if i >= n {
                            "0u".into()
                        } else if hi {
                            format!("({} >> 16)", self.v(vd))
                        } else {
                            let v = self.v(vd + i / 2);
                            if i % 2 == 0 {
                                format!("({v} & 0xffffu)")
                            } else {
                                format!("({v} >> 16)")
                            }
                        }
                    })
                    .collect()
            } else {
                (0..4)
                    .map(|i| if i < n { self.v(vd + i) } else { "0u".into() })
                    .collect()
            };
            let call = if d16 {
                "buffer_store_format_d16"
            } else {
                "buffer_store_format"
            };
            self.guard(&format!(
                "{call}(mem, {args}, uint4({}, {}, {}, {}), {n}u, {sel}u);",
                vals[0], vals[1], vals[2], vals[3]
            ));
            return Ok(());
        }
        if let Some(comps) = name.strip_prefix("buffer_load_format_d16_") {
            let (hi, comps) = match comps.strip_prefix("hi_") {
                Some(c) => (true, c),
                None => (false, comps),
            };
            let n = comps.len() as u16;
            let mut s = format!("uint4 r = buffer_load_format_d16(mem, {args}, {sel}u);");
            if hi {
                let d = self.v(vd);
                write!(s, " {d} = ({d} & 0xffffu) | (r.x << 16);").unwrap();
            }
            for i in (0..n).step_by(2).filter(|_| !hi) {
                let d = self.v(vd + i / 2);
                let k = i as usize;
                if i + 1 < n {
                    write!(s, " {d} = r[{k}] | (r[{}] << 16);", k + 1).unwrap();
                } else {
                    write!(s, " {d} = ({d} & 0xffff0000u) | r[{k}];").unwrap();
                }
            }
            self.guard(&s);
            return Ok(());
        }
        let sub16 = match name {
            "buffer_load_ubyte_d16" => Some(("buffer_load_ubyte", false)),
            "buffer_load_ubyte_d16_hi" => Some(("buffer_load_ubyte", true)),
            "buffer_load_sbyte_d16" => Some(("buffer_load_sbyte", false)),
            "buffer_load_sbyte_d16_hi" => Some(("buffer_load_sbyte", true)),
            "buffer_load_short_d16" => Some(("buffer_load_ushort", false)),
            "buffer_load_short_d16_hi" => Some(("buffer_load_ushort", true)),
            _ => None,
        };
        if let Some((call, hi)) = sub16 {
            let d = self.v(vd);
            let stmt = if hi {
                format!("uint4 r = {call}(mem, {args}); {d} = ({d} & 0xffffu) | (r.x << 16);")
            } else {
                format!(
                    "uint4 r = {call}(mem, {args}); {d} = ({d} & 0xffff0000u) | (r.x & 0xffffu);"
                )
            };
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
        let mut s = if call == "buffer_load_raw" {
            let call = match (b.glc, self.mem_write) {
                (true, true) => "buffer_load_atomic",
                (true, false) => "buffer_load_volatile",
                _ => call,
            };
            format!("uint4 r = {call}(mem, {args}, {n}u);")
        } else if call == "buffer_load_format" {
            format!("uint4 r = {call}(mem, {args}, {sel}u);")
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

    /// A buffer, global or flat atomic: `off` gives the offset of an access
    /// of so many bytes, `vd` holds the data and `ret` takes the old value.
    fn atomic(
        &mut self,
        pc: usize,
        name: &str,
        op: &str,
        off: &dyn Fn(u32) -> String,
        vd: u16,
        ret: Option<u16>,
    ) -> Result<String, TranslateError> {
        let data = self.v(vd);
        if let Some(f) = op.strip_suffix("_x2") {
            let hi = self.v(vd + 1);
            let (c0, c1) = (self.v(vd + 2), self.v(vd + 3));
            let call = match f {
                // Bitwise ops are exact per half: the high half (a flag in look-back
                // scans) is read before the low half and written after it.
                "and" | "or" | "xor" | "swap" => {
                    let (first, second) = if f == "swap" {
                        (
                            format!("lo = atomic_rmw_swap(mem, o, {data});"),
                            format!("hi = atomic_rmw_swap(mem, o + 4, {hi});"),
                        )
                    } else {
                        (
                            format!("hi = atomic_rmw_{f}(mem, o + 4, {hi});"),
                            format!("lo = atomic_rmw_{f}(mem, o, {data});"),
                        )
                    };
                    format!(
                        "uint lo = 0u, hi = 0u; if (o != OOB_OFF) {{ {first} {second} }} ulong old = ulong(hi) << 32 | lo;"
                    )
                }
                "add" | "sub" => {
                    format!("ulong old = atomic_{f}64(mem, o, ulong({hi}) << 32 | {data});")
                }
                "umin" | "umax" | "smin" | "smax" | "inc" | "dec" | "fmin" | "fmax" | "cmpswap"
                | "fcmpswap" => {
                    let code = match f {
                        "umin" => 0,
                        "umax" => 1,
                        "smin" => 2,
                        "smax" => 3,
                        "inc" => 4,
                        "dec" => 5,
                        "fmin" => 6,
                        "fmax" => 7,
                        "cmpswap" => 8,
                        _ => 9,
                    };
                    format!(
                        "ulong old = atomic_rmw64(mem, o, ulong({hi}) << 32 | {data}, ulong({c1}) << 32 | {c0}, {code}u);"
                    )
                }
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            let ret = match ret {
                Some(r) => {
                    let (lo, hi) = (self.v(r), self.v(r + 1));
                    format!(" {lo} = uint(old); {hi} = uint(old >> 32);")
                }
                None => String::new(),
            };
            return Ok(format!("{{ long o = {}; {call}{ret} }}", off(8)));
        }
        let call = match op {
            "add" | "sub" | "and" | "or" | "xor" | "umin" | "umax" | "smin" | "smax" | "swap"
            | "fmin" | "fmax" | "inc" | "dec" => {
                format!("atomic_rmw_{op}(mem, {}, {data})", off(4))
            }
            "cmpswap" | "fcmpswap" => {
                let cmp = self.v(vd + 1);
                let f = if op == "fcmpswap" {
                    "atomic_fcas"
                } else {
                    "atomic_cas"
                };
                format!("{f}(mem, {}, {data}, {cmp})", off(4))
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        Ok(match ret {
            Some(r) => format!("{} = {call};", self.v(r)),
            None => format!("(void){call};"),
        })
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
        if name == "ds_permute_b32" && !gds && self.req.stage != Stage::Vertex {
            // Wave64 permutes run as two 32-lane halves; the highest sender wins and
            // lanes nobody sends to get 0 (measured on gfx1033).
            let [addr, data0, _, vdst] = regs;
            let (a, d0, d) = (self.v(addr), self.v(data0), self.v(vdst));
            let off = offset0 as u32 | (offset1 as u32) << 8;
            let on = self.mask_on(126);
            self.simd_lane = true;
            self.line(&format!(
                "{{ uint pt = {on} ? (({a} + {off}u) >> 2) & 31u : 32u; uint pr = 0u; uint pl = uint(slane);"
            ));
            self.line(&format!(
                "for (ushort j = 0; j < 32; j++) {{ uint dj = simd_shuffle({d0}, j); if (simd_shuffle(pt, j) == pl) pr = dj; }}"
            ));
            self.guard(&format!("{d} = pr;"));
            self.line("}");
            return Ok(());
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
        if name == "ds_swizzle_b32" && !gds && self.req.stage != Stage::Vertex {
            // Every mode stays within groups of 32 lanes, as a Metal SIMD group does.
            let [src, _, _, vdst] = regs;
            let (v, d) = (self.v(src), self.v(vdst));
            let from = swizzle_source(offset0 as u16 | (offset1 as u16) << 8);
            let on = self.mask_on(126);
            self.simd_lane = true;
            self.line(&format!(
                "{{ ushort sj = {from}; uint sv = simd_shuffle({v}, sj); bool son = simd_shuffle(uint({on}), sj) != 0u;"
            ));
            self.guard(&format!("{d} = son ? sv : 0u;"));
            self.line("}");
            return Ok(());
        }
        if !gds
            && self.req.stage != Stage::Compute
            && matches!(name, "ds_write_addtid_b32" | "ds_read_addtid_b32")
        {
            // Fragment/vertex stages have no threadgroup memory in Metal; addtid
            // accesses a lane's own per-wave spill slot, so map it to a local array.
            let [_, data0, _, vdst] = regs;
            let m0 = self.s(124);
            let off = offset0 as u32 | (offset1 as u32) << 8;
            self.addtid_local = true;
            self.addtid_max_off = self.addtid_max_off.max(off);
            self.addtid_aligned128 &= off & 127 == 0;
            let stmt = if name == "ds_write_addtid_b32" {
                let d = self.v(data0);
                format!(
                    "{{ uint a = (({m0} & 0xffffu) + {off}u) >> ADDTID_SHIFT; if (a < ADDTID_SLOTS) addtid_lds[a] = {d}; }}"
                )
            } else {
                let d = self.v(vdst);
                format!(
                    "{{ uint a = (({m0} & 0xffffu) + {off}u) >> ADDTID_SHIFT; {d} = a < ADDTID_SLOTS ? addtid_lds[a] : 0u; }}"
                )
            };
            self.guard(&stmt);
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
            "ds_read_u8_d16" | "ds_read_u8_d16_hi" | "ds_read_i8_d16" | "ds_read_i8_d16_hi"
            | "ds_read_u16_d16" | "ds_read_u16_d16_hi" => {
                let ty = match &name[8..10] {
                    "u8" => "uchar",
                    "i8" => "char",
                    _ => "ushort",
                };
                let d = dst(self, 0);
                write!(s, " uint h = (a + {off}u) < LDS_BYTES ? uint(int(*(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u))) & 0xffffu : 0u;").unwrap();
                if name.ends_with("_hi") {
                    write!(s, " {d} = ({d} & 0xffffu) | (h << 16);").unwrap();
                } else {
                    write!(s, " {d} = ({d} & 0xffff0000u) | h;").unwrap();
                }
            }
            "ds_write_b8_d16_hi" | "ds_write_b16_d16_hi" => {
                let v = d0(self, 0);
                let ty = if name.contains("b8") {
                    "uchar"
                } else {
                    "ushort"
                };
                write!(s, " if ((a + {off}u) < LDS_BYTES) *(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u) = {ty}({v} >> 16);").unwrap();
            }
            "ds_wrxchg2_rtn_b32"
            | "ds_wrxchg2st64_rtn_b32"
            | "ds_wrxchg2_rtn_b64"
            | "ds_wrxchg2st64_rtn_b64" => {
                // The whole wave makes its first exchange before its second (measured).
                let wide = name.ends_with("b64");
                let scale = if wide { 8 } else { 4 } * if name.contains("st64") { 64 } else { 1 };
                let (o0, o1) = pair(scale);
                let swap = |t: &mut Self, i: u16, o: u32| {
                    let (lo, hi) = (
                        t.v(if i == 0 { data0 } else { data1 }),
                        t.v(if i == 0 { data0 } else { data1 } + 1),
                    );
                    if wide {
                        format!(
                            "x2t{i} = lds_rmw64(lds, {a} + {o}u, ulong({hi}) << 32 | {lo}, 0ul, 13u);"
                        )
                    } else {
                        format!("x2t{i} = lds_rmw_swap(lds, {a} + {o}u, {lo});")
                    }
                };
                let (s0, s1) = (swap(self, 0, o0), swap(self, 1, o1));
                self.line("{ ulong x2t0 = 0ul, x2t1 = 0ul;");
                if self.lockstep {
                    self.lane_order(wide, &mut |t| {
                        t.guard(&s0);
                        Ok(())
                    })?;
                    self.line("wave_meet(wv);");
                    self.lane_order(wide, &mut |t| {
                        t.guard(&s1);
                        Ok(())
                    })?;
                } else {
                    self.guard(&format!("{s0} {s1}"));
                }
                let mut r = String::new();
                let vals: &[&str] = if wide {
                    &[
                        "uint(x2t0)",
                        "uint(x2t0 >> 32)",
                        "uint(x2t1)",
                        "uint(x2t1 >> 32)",
                    ]
                } else {
                    &["uint(x2t0)", "uint(x2t1)"]
                };
                for (i, v) in vals.iter().enumerate() {
                    let d = dst(self, i as u16);
                    write!(r, " {d} = {v};").unwrap();
                }
                self.guard(r.trim_start());
                self.line("}");
                return Ok(());
            }
            _ => {
                let base = name.strip_prefix("ds_").unwrap();
                let rtn = base.contains("_rtn_");
                let op = base.replace("_rtn", "");
                let v = d0(self, 0);
                let at = format!("a + {off}u");
                if let Some(code) = lds64_op(&op) {
                    let (vh, w, wh) = (d0(self, 1), d1(self, 0), d1(self, 1));
                    // Two-data forms: mskor's bits and cmpst's stored value come second.
                    let second = if matches!(code, 12 | 14 | 15) {
                        format!("ulong({wh}) << 32 | {w}")
                    } else {
                        "0ul".into()
                    };
                    write!(
                        s,
                        " ulong t = lds_rmw64(lds, {at}, ulong({vh}) << 32 | {v}, {second}, {code}u);"
                    )
                    .unwrap();
                    if rtn {
                        let (r0, r1) = (dst(self, 0), dst(self, 1));
                        write!(s, " {r0} = uint(t); {r1} = uint(t >> 32);").unwrap();
                    }
                    self.guard(&s);
                    return Ok(());
                }
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
                        format!("lds_rmw_{f}(lds, {at}, {v})")
                    }
                    "min_f32" => format!("lds_rmw_op(lds, {at}, {v}, 0u, 0u)"),
                    "max_f32" => format!("lds_rmw_op(lds, {at}, {v}, 0u, 1u)"),
                    "rsub_u32" => format!("lds_rmw_op(lds, {at}, {v}, 0u, 2u)"),
                    "inc_u32" => format!("lds_rmw_op(lds, {at}, {v}, 0u, 3u)"),
                    "dec_u32" => format!("lds_rmw_op(lds, {at}, {v}, 0u, 4u)"),
                    "wrap_b32" => {
                        let w = d1(self, 0);
                        format!("lds_rmw_op(lds, {at}, {v}, {w}, 5u)")
                    }
                    "mskor_b32" => {
                        let w = d1(self, 0);
                        format!("lds_rmw_op(lds, {at}, {v}, {w}, 6u)")
                    }
                    "add_f32" => format!("lds_rmw_op(lds, {at}, {v}, 0u, 7u)"),
                    "cmpst_f32" => {
                        let w = d1(self, 0);
                        format!("lds_rmw_op(lds, {at}, {v}, {w}, 8u)")
                    }
                    "cmpst_b32" => {
                        let new = d1(self, 0);
                        format!("lds_cas(lds, {at}, {new}, {v})")
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

    // The counter is at M0's high half plus `offset`, for LDS as for GDS
    // (measured on gfx1033).
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
        if self.lockstep {
            // The wave counts its active lanes once and every lane gets the old value.
            let (lo, hi) = (self.s(126), self.s(127));
            let update = if gds {
                self.gds = true;
                format!(
                    "atomic_fetch_{op}_explicit(&gds[((({m0} >> 16) + {offset}u) & 0xfffcu) >> 2], n, memory_order_relaxed)"
                )
            } else {
                if self.req.stage != Stage::Compute {
                    return Err(err(pc, format!("unsupported LDS access {name}")));
                }
                self.lds = true;
                format!("lds_rmw_{op}(lds, ({m0} >> 16) + {offset}u, n)")
            };
            self.line(&format!(
                "{{ ulong e = ulong({hi}) << 32 | {lo}; uint n = uint(popcount(e)); uint first = e == 0ul ? 0u : uint(ctz(e)); uint old = 0u; if (n != 0u && wl == first) old = {update}; old = wave_readlane(old, first, wv);"
            ));
            self.guard(&format!("{d} = old;"));
            self.line("}");
            return Ok(());
        }
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
            format!("{{ {d} = lds_rmw_{op}(lds, ({m0} >> 16) + {offset}u, 1u); }}")
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

/// `o.cN = value` with ROP3 `rop` applied between the value as the target
/// stores it (integers saturated to the channel, measured) and the target's
/// current bits (`dN`, read through the attachment). The pattern operand is
/// taken as 0.
/// The `uint4` a packed target (`Request::packed_targets`) takes for `words`,
/// keeping the bits of the channels not written from `d{target}`.
pub(super) fn packed_store(target: impl std::fmt::Display, words: &str, packed: u16) -> String {
    let write = packed_write_bits(packed);
    if write == [u32::MAX; 4] {
        return words.to_string();
    }
    let [a, b, c, d] = write;
    let [ka, kb, kc, kd] = write.map(|w| !w);
    format!(
        "(({words}) & uint4({a:#x}u, {b:#x}u, {c:#x}u, {d:#x}u) | d{target} & uint4({ka:#x}u, {kb:#x}u, {kc:#x}u, {kd:#x}u))"
    )
}

/// The bits of each dword of a packed target's pixel that its written
/// channels hold; bits past the pixel count as written.
pub(super) fn packed_write_bits(packed: u16) -> [u32; 4] {
    let widths = packed_widths(u32::from(packed & 0x1F)).unwrap_or([32; 4]);
    let mut bits = [u32::MAX; 4];
    let mut at = 0;
    for (c, w) in widths.into_iter().enumerate() {
        if w != 0 && packed >> (8 + c) & 1 == 0 {
            let field = (((1u64 << w) - 1) << (at % 32)) as u32;
            bits[at / 32] &= !field;
        }
        at += w as usize;
    }
    bits
}

fn logic_op(target: u8, value: &str, rop: u8, logic: u8) -> String {
    let bits = u32::from(logic & 0x3F).min(32);
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    };
    let max = format!("{:?}f", mask as f32);
    let (src, dst) = match logic >> 6 {
        0 => (
            format!("uint4(floor(saturate({value}) * {max} + 0.5f))"),
            format!("uint4(floor(saturate(d{target}) * {max} + 0.5f))"),
        ),
        1 => (
            format!("min(uint4({value}), {mask:#x}u)"),
            format!("d{target}"),
        ),
        _ => {
            let (lo, hi) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
            (
                format!("as_type<uint4>(clamp(int4({value}), int({lo}), int({hi})))"),
                format!("as_type<uint4>(d{target})"),
            )
        }
    };
    let terms: Vec<&str> = [("~s & ~d", 0), ("~s & d", 1), ("s & ~d", 2), ("s & d", 3)]
        .iter()
        .filter(|(_, i)| rop >> i & 1 != 0)
        .map(|(t, _)| *t)
        .collect();
    let combined = if terms.is_empty() {
        "uint4(0u)".to_string()
    } else {
        terms
            .iter()
            .map(|t| format!("({t})"))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let out = match logic >> 6 {
        0 => format!("float4(r) / {max}"),
        1 => "r".to_string(),
        _ if bits == 32 => "as_type<int4>(r)".to_string(),
        _ => format!("as_type<int4>(r << {0}u) >> {0}", 32 - bits),
    };
    format!(
        "uint4 s = {src}; uint4 d = {dst}; uint4 r = ({combined}) & {mask:#x}u; o.c{target} = {out};"
    )
}
/// The lane `ds_swizzle_b32` reads, within the 32-lane group, for the lane in
/// `slane`, by the modes of the ISA guide's pseudo-code.
fn swizzle_source(offset: u16) -> String {
    let field = |shift: u16| u32::from(offset >> shift & 0x1F);
    if offset >= 0xE000 {
        let mask = field(0);
        format!(
            "ushort(((reverse_bits(uint(slane)) >> 27) >> {}u) | (uint(slane) & {mask}u))",
            mask.count_ones()
        )
    } else if offset >= 0xC000 {
        let mask = field(0);
        let rotate = if offset & 0x400 != 0 {
            32 - field(5)
        } else {
            field(5)
        };
        format!(
            "ushort(((uint(slane) & {mask}u) | ((uint(slane) + {rotate}u) & {:#x}u)) & 31u)",
            !mask
        )
    } else if offset & 0x8000 != 0 {
        format!(
            "ushort((uint(slane) & 28u) | (({:#x}u >> ((uint(slane) & 3u) << 1)) & 3u))",
            offset & 0xFF
        )
    } else {
        let (and, or, xor) = (field(0), field(5), field(10));
        format!("ushort(((uint(slane) & {and}u) | {or}u) ^ {xor}u)")
    }
}

/// The `lds_rmw64` code of a 64-bit LDS atomic (its name without `ds_` and `_rtn`).
fn lds64_op(op: &str) -> Option<u32> {
    Some(match op {
        "add_u64" => 0,
        "sub_u64" => 1,
        "rsub_u64" => 2,
        "inc_u64" => 3,
        "dec_u64" => 4,
        "min_i64" => 5,
        "max_i64" => 6,
        "min_u64" => 7,
        "max_u64" => 8,
        "and_b64" => 9,
        "or_b64" => 10,
        "xor_b64" => 11,
        "mskor_b64" => 12,
        "wrxchg_b64" => 13,
        "cmpst_b64" => 14,
        "cmpst_f64" => 15,
        "min_f64" => 16,
        "max_f64" => 17,
        _ => return None,
    })
}
