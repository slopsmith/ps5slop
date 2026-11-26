//! Image sampling, loads, stores, atomics, resinfo and BVH intersection (`MIMG`).

use super::*;

impl<'a> Tx<'a> {
    // The element type of texture slot `t` and how a component of it becomes a VGPR value.
    pub(super) fn texel(&self, t: usize) -> (&'static str, &'static str) {
        if self.req.coded_textures >> t & 1 != 0 || self.textures[t].4 == View::Bits {
            ("uint", "")
        } else if self.atomic_textures >> t & 1 != 0 {
            if self.req.sint_textures >> t & 1 != 0 {
                ("int", "uint")
            } else {
                ("uint", "")
            }
        } else if self.req.uint_textures >> t & 1 != 0 {
            ("uint", "")
        } else if self.req.sint_textures >> t & 1 != 0 {
            ("int", "uint")
        } else {
            ("float", "U")
        }
    }

    // The texture slot for the T# at `srsrc`, merging how it is accessed.
    // Each view of one T# gets its own slot.
    fn texture_slot(
        &mut self,
        pc: usize,
        srsrc: u16,
        kind: TextureKind,
        access: Access,
        view: View,
    ) -> Result<usize, TranslateError> {
        let source = self.descriptor_source(pc, srsrc, 8)?;
        let bits = self.bits_textures.contains(&(srsrc, kind));
        let view = if view == View::Raw && bits {
            View::Bits
        } else {
            view
        };
        let t = match self
            .textures
            .iter()
            .position(|x| x.0 == srsrc && x.1 == kind && x.2 == source && x.4 == view)
        {
            Some(t) => t,
            None => {
                self.textures.push((srsrc, kind, source, access, view));
                self.textures.len() - 1
            }
        };
        let entry = &mut self.textures[t];
        entry.3 = match (entry.3, access) {
            (a, b) if a == b => a,
            (Access::Atomic, _) | (_, Access::Atomic) => Access::Atomic,
            (Access::Write, Access::Sample)
            | (Access::Sample, Access::Write)
            | (_, Access::ReadWrite)
            | (Access::ReadWrite, _) => Access::ReadWrite,
            (a, _) => a,
        };
        Ok(t)
    }

    pub(super) fn mimg(&mut self, pc: usize, name: &str, m: &Mimg) -> Result<(), TranslateError> {
        let rw = name.starts_with("image_load") || name.starts_with("image_store");
        if (m.a16 && !rw) || m.tfe || m.lwe {
            return Err(err(pc, "unsupported image modifiers"));
        }
        if name.starts_with("image_bvh") {
            return self.bvh(m, name.starts_with("image_bvh64"));
        }
        let kind = match m.dim {
            1 => TextureKind::Tex2d,
            5 => TextureKind::Tex2dArray,
            2 => TextureKind::Tex3d,
            3 => TextureKind::Cube,
            d => {
                return Err(err(
                    pc,
                    format!("unsupported image dimension {d} in {name}"),
                ));
            }
        };
        let n = mimg_addr_count(name, m.dim, m.a16) as usize;
        let addr_regs: Vec<u16> = (0..n)
            .map(|i| {
                if m.nsa > 0 {
                    m.vaddr[i] as u16
                } else {
                    m.vaddr[0] as u16 + i as u16
                }
            })
            .collect();
        match name {
            _ if rw => {
                return self.image_rw(pc, name, m, kind, &addr_regs);
            }
            "image_get_resinfo" => return self.image_resinfo(pc, m, kind, &addr_regs),
            n if n.starts_with("image_atomic_") => {
                return self.image_atomic(pc, name, m, kind, &addr_regs);
            }
            _ => {}
        }
        let gather = name.starts_with("image_gather4");
        let variant = name
            .strip_prefix("image_sample")
            .or_else(|| name.strip_prefix("image_gather4"))
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        let t = self.texture_slot(pc, m.srsrc as u16, kind, Access::Sample, View::Sampled)?;
        let s_source = self.descriptor_source(pc, m.ssamp as u16, 4)?;
        let sm = match self.samplers.iter().position(|x| *x == s_source) {
            Some(i) => i,
            None => {
                self.samplers.push(s_source);
                self.samplers.len() - 1
            }
        };
        let addr: Vec<String> = addr_regs
            .iter()
            .map(|&r| format!("F({})", self.v(r)))
            .collect();
        let raw: Vec<String> = addr_regs.iter().map(|&r| self.v(r)).collect();
        // Address order: offset, bias, depth reference, coordinates, LOD.
        let mut flags: Vec<&str> = variant.split('_').filter(|p| !p.is_empty()).collect();
        let offset = flags.last() == Some(&"o");
        if offset {
            flags.pop();
        }
        let compare = flags.first() == Some(&"c");
        if compare {
            flags.remove(0);
        }
        let mut at = offset as usize;
        let derivs = matches!(flags.as_slice(), ["d"] | ["d", "cl"]);
        let (mut level, bias) = match flags.as_slice() {
            [] => (None, false),
            ["lz"] => (Some("level(0.0f)".to_string()), false),
            ["l"] => (Some(format!("level({})", addr[addr.len() - 1])), false),
            ["b"] => (Some(format!("bias({})", addr[at])), true),
            ["d"] | ["d", "cl"] => (None, false),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        at += bias as usize;
        let dref = compare.then(|| addr[at].clone());
        at += compare as usize;
        if derivs {
            // Derivatives come before the coordinates: d/dx of each, then d/dy.
            let g = &addr[at..];
            let grad = match kind {
                TextureKind::Tex2d | TextureKind::Tex2dArray => format!(
                    "gradient2d(float2({}, {}), float2({}, {}))",
                    g[0], g[1], g[2], g[3]
                ),
                TextureKind::Tex3d => format!(
                    "gradient3d(float3({}, {}, {}), float3({}, {}, {}))",
                    g[0], g[1], g[2], g[3], g[4], g[5]
                ),
                TextureKind::Cube => return Err(err(pc, format!("unsupported form of {name}"))),
            };
            at += if kind == TextureKind::Tex3d { 6 } else { 4 };
            level = Some(if flags.len() == 2 {
                format!("{grad}, min_lod_clamp({})", addr[addr.len() - 1])
            } else {
                grad
            });
        }
        let c = &addr[at..];
        let shift = |i: usize, size: &str| {
            if offset {
                format!(
                    " + float(int(({} >> {}u) << 26) >> 26) / float(t{t}.get_{size}(0))",
                    raw[0],
                    8 * i
                )
            } else {
                String::new()
            }
        };
        let (dx, dy) = (shift(0, "width"), shift(1, "height"));
        let coord = match kind {
            TextureKind::Tex2d => format!("float2({}{dx}, {}{dy})", c[0], c[1]),
            TextureKind::Tex2dArray => {
                format!("float2({}{dx}, {}{dy}), uint(rint({}))", c[0], c[1], c[2])
            }
            TextureKind::Tex3d if !offset => format!("float3({}, {}, {})", c[0], c[1], c[2]),
            TextureKind::Cube if !offset => {
                format!("cube_dir(float3({}, {}, {}))", c[0], c[1], c[2])
            }
            _ => return Err(err(pc, format!("unsupported form of {name}"))),
        };
        let level = match (level, self.req.stage) {
            (Some(l), _) => format!(", {l}"),
            (None, Stage::Pixel) => String::new(),
            (None, _) => ", level(0.0f)".into(),
        };
        let (elem, conv) = self.texel(t);
        let mut s = format!("{elem}4 r = t{t}.sample(smp{sm}, {coord}{level});");
        if let Some(dref) = &dref {
            if gather || elem != "float" {
                return Err(err(pc, format!("unsupported form of {name}")));
            }
            write!(
                s,
                " r = float4(depth_compare((user[{}] >> 12) & 7u, {dref}, r.x) ? 1.0f : 0.0f);",
                binding::SAMPLER_WORDS + sm as u32
            )
            .unwrap();
        }
        if gather {
            if !matches!(kind, TextureKind::Tex2d | TextureKind::Tex2dArray)
                || !matches!(variant, "" | "_lz" | "_o" | "_lz_o")
                || m.dmask.count_ones() != 1
            {
                return Err(err(pc, format!("unsupported form of {name}")));
            }
            let c = ["x", "y", "z", "w"][m.dmask.trailing_zeros() as usize];
            s = format!("{elem}4 r = t{t}.gather(smp{sm}, {coord}, int2(0), component::{c});");
        }
        let comps: Vec<usize> = (0..4)
            .filter(|&i| gather || m.dmask >> i & 1 != 0)
            .collect();
        self.write_results(&mut s, m, elem, conv, &comps);
        self.guard(&s);
        Ok(())
    }

    // Moves the components `comps` of `r` into the destination VGPRs; with
    // `d16`, two 16-bit values per VGPR, low half first.
    fn write_results(&mut self, s: &mut String, m: &Mimg, elem: &str, conv: &str, comps: &[usize]) {
        let d = m.vdata as u16;
        for (j, &i) in comps.iter().enumerate() {
            if !m.d16 {
                let r = self.v(d + j as u16);
                write!(s, " {r} = {conv}(r[{i}]);").unwrap();
                continue;
            }
            let r = self.v(d + j as u16 / 2);
            let bits = match elem {
                "float" => format!("uint(as_type<ushort>(half(r[{i}])))"),
                _ => format!("(uint(r[{i}]) & 0xffffu)"),
            };
            if j % 2 == 0 {
                write!(s, " {r} = {bits};").unwrap();
            } else {
                write!(s, " {r} |= {bits} << 16;").unwrap();
            }
        }
    }

    // `image_load`, `image_store` and their `_mip` and `_pck` forms:
    // unfiltered texel access.
    fn image_rw(
        &mut self,
        pc: usize,
        name: &str,
        m: &Mimg,
        kind: TextureKind,
        addr: &[u16],
    ) -> Result<(), TranslateError> {
        let store = name.starts_with("image_store");
        let access = if store { Access::Write } else { Access::Sample };
        let pck = name.contains("_pck");
        let view = if pck { View::Bits } else { View::Raw };
        let t = self.texture_slot(pc, m.srsrc as u16, kind, access, view)?;
        let regs: Vec<String> = addr.iter().map(|&r| self.v(r)).collect();
        let a: Vec<String> = if m.a16 {
            (0..2 * regs.len())
                .map(|k| format!("(({} >> {}u) & 0xffffu)", regs[k / 2], 16 * (k % 2)))
                .collect()
        } else {
            regs
        };
        let mip = name.contains("_mip");
        let level = |i: usize| if mip { a[i].clone() } else { "0u".into() };
        let coord = match kind {
            TextureKind::Tex2d => format!("uint2({}, {})", a[0], a[1]),
            TextureKind::Tex2dArray => format!("uint2({}, {}), {}", a[0], a[1], a[2]),
            TextureKind::Tex3d => format!("uint3({}, {}, {})", a[0], a[1], a[2]),
            TextureKind::Cube => return Err(err(pc, "cube image access is not done yet")),
        };
        let lvl = match kind {
            TextureKind::Tex2d => level(2),
            _ => level(3),
        };
        let d = m.vdata as u16;
        let (elem, conv) = self.texel(t);
        let coded = self.req.coded_textures >> t & 1 != 0 || self.textures[t].4 == View::Bits;
        if (coded || pck) && m.d16 {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        let w = binding::TEXTURE_WORDS + 2 * t as u32;
        // The `_pck` forms move the element's dwords, one VGPR per DMASK bit.
        let dwords = m.dmask.count_ones() as usize;
        if store {
            let mut comps = Vec::new();
            let mut j = 0u16;
            for i in 0..4 {
                if pck {
                    comps.push(if i < dwords {
                        self.v(d + i as u16)
                    } else {
                        "0u".into()
                    });
                } else if m.dmask >> i & 1 != 0 {
                    let v = if m.d16 {
                        self.v(d + j / 2)
                    } else {
                        self.v(d + j)
                    };
                    let half = format!("({v} >> {}u)", 16 * (j % 2));
                    comps.push(match (elem, m.d16) {
                        ("float", false) => format!("F({v})"),
                        ("int", false) => format!("int({v})"),
                        (_, false) => v,
                        ("float", true) => format!("float(as_type<half>(ushort({half})))"),
                        ("int", true) => format!("int(short({half}))"),
                        (_, true) => format!("({half} & 0xffffu)"),
                    });
                    j += 1;
                } else {
                    comps.push(if elem == "float" {
                        "0.0f".into()
                    } else {
                        "0".into()
                    });
                }
            }
            // Later reads of the texture in this thread see the store.
            let fence = if self.read_write_textures >> t & 1 != 0 {
                format!(" t{t}.fence();")
            } else {
                String::new()
            };
            let storable = format!("if (image_storable((user[{w}] >> 20) & 0x1ffu)) ");
            let (encode, when) = if pck {
                ("image_pack", storable)
            } else if coded {
                ("image_encode", storable)
            } else {
                ("image_store_channels", String::new())
            };
            self.guard(&format!(
                "{when}t{t}.write({encode}({elem}4({}), user[{w}]), {coord}, {lvl});{fence}",
                comps.join(", ")
            ));
        } else {
            let mut s = format!("{elem}4 r = t{t}.read({coord}, {lvl});");
            if pck {
                let sgn = name.ends_with("_sgn");
                write!(s, " r = image_unpack(r, user[{w}], {sgn});").unwrap();
            } else if coded {
                write!(s, " r = image_decode(r, user[{w}], user[{}]);", w + 1).unwrap();
            } else {
                if elem == "float" {
                    write!(s, " r = image_srgb(r, user[{w}]);").unwrap();
                }
                write!(s, " r = image_sel(r, user[{w}], user[{}]);", w + 1).unwrap();
            }
            let comps: Vec<usize> = if pck {
                (0..dwords).collect()
            } else {
                (0..4).filter(|&i| m.dmask >> i & 1 != 0).collect()
            };
            self.write_results(&mut s, m, elem, conv, &comps);
            self.guard(&s);
        }
        Ok(())
    }

    /// Image atomics through Metal's texture atomics on a 32-bit integer
    /// view; `glc` returns the value before the operation.
    fn image_atomic(
        &mut self,
        pc: usize,
        name: &str,
        m: &Mimg,
        kind: TextureKind,
        addr: &[u16],
    ) -> Result<(), TranslateError> {
        let op = &name["image_atomic_".len()..];
        let f = match op {
            "swap" => "atomic_exchange",
            "add" => "atomic_fetch_add",
            "sub" => "atomic_fetch_sub",
            "smin" | "umin" => "atomic_fetch_min",
            "smax" | "umax" => "atomic_fetch_max",
            "and" => "atomic_fetch_and",
            "or" => "atomic_fetch_or",
            "xor" => "atomic_fetch_xor",
            "cmpswap" => "",
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        if m.dmask != 1 && !(op == "cmpswap" && m.dmask == 3) || m.d16 {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        let t = self.texture_slot(pc, m.srsrc as u16, kind, Access::Atomic, View::Raw)?;
        let a: Vec<String> = addr.iter().map(|&r| self.v(r)).collect();
        let coord = match kind {
            TextureKind::Tex2d => format!("uint2({}, {})", a[0], a[1]),
            TextureKind::Tex2dArray => format!("uint2({}, {}), {}", a[0], a[1], a[2]),
            _ => return Err(err(pc, format!("unsupported form of {name}"))),
        };
        // The first pass does not know the atomic slots yet.
        let (elem, conv) = match self.texel(t) {
            ("float", _) => ("uint", ""),
            x => x,
        };
        let d = self.v(m.vdata as u16);
        let value = if elem == "int" {
            format!("int({d})")
        } else {
            d.clone()
        };
        let mut s = if op == "cmpswap" {
            let cmp = self.v(m.vdata as u16 + 1);
            let cmp = if elem == "int" {
                format!("int({cmp})")
            } else {
                cmp
            };
            format!(
                "{elem}4 r = {elem}4({cmp}); for (;;) {{ if (t{t}.atomic_compare_exchange_weak({coord}, &r, {elem}4({value}))) {{ r = {elem}4({cmp}); break; }} if (r.x != {cmp}) break; }}"
            )
        } else {
            format!("{elem}4 r = t{t}.{f}({coord}, {elem}4({value}));")
        };
        if m.glc {
            write!(s, " {d} = {conv}(r.x);").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    fn bvh(&mut self, m: &Mimg, wide: bool) -> Result<(), TranslateError> {
        self.mem = true;
        let n = if wide { 12 } else { 11 };
        let a: Vec<String> = (0..n)
            .map(|i| {
                self.v(if m.nsa > 0 {
                    m.vaddr[i] as u16
                } else {
                    m.vaddr[0] as u16 + i as u16
                })
            })
            .collect();
        let node = if wide {
            format!("(ulong({}) | (ulong({}) << 32))", a[0], a[1])
        } else {
            format!("ulong({})", a[0])
        };
        let r = &a[n - 10..];
        let d: Vec<String> = (0..4).map(|i| self.s(m.srsrc as u16 + i)).collect();
        let mut s = format!(
            "uint4 r = bvh_intersect(mem, mem_base, uint4({}, {}, {}, {}), {node}, F({}), \
             float3(F({}), F({}), F({})), float3(F({}), F({}), F({})), float3(F({}), F({}), F({})));",
            d[0], d[1], d[2], d[3], r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]
        );
        for i in 0..4 {
            let v = self.v(m.vdata as u16 + i);
            write!(s, " {v} = r[{i}];").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    fn image_resinfo(
        &mut self,
        pc: usize,
        m: &Mimg,
        kind: TextureKind,
        addr: &[u16],
    ) -> Result<(), TranslateError> {
        let t = self.texture_slot(pc, m.srsrc as u16, kind, Access::Sample, View::Sampled)?;
        let lod = self.v(addr[0]);
        let third = match kind {
            TextureKind::Tex2dArray => format!("t{t}.get_array_size()"),
            TextureKind::Tex3d => format!("t{t}.get_depth(l)"),
            TextureKind::Cube => "6u".into(),
            TextureKind::Tex2d => "1u".into(),
        };
        let mut s = format!(
            "uint l = {lod}; uint4 r = uint4(t{t}.get_width(l), t{t}.get_height(l), {third}, t{t}.get_num_mip_levels());"
        );
        let mut d = m.vdata as u16;
        for i in 0..4 {
            if m.dmask >> i & 1 != 0 {
                let r = self.v(d);
                write!(s, " {r} = r[{i}];").unwrap();
                d += 1;
            }
        }
        self.guard(&s);
        Ok(())
    }
}
