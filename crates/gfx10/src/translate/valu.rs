//! Vector ALU (`VOP1`, `VOP2`, `VOP3`, `VOPC`), DPP, SDWA and carry ops.

use super::*;

impl<'a> Tx<'a> {
    // Source `i` of a VALU instruction as a `float` expression, with its modifiers.
    pub(super) fn fsrc(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        let u = self.usrc(pc, inst, v, i)?;
        let mut e = format!("F({u})");
        if v.abs >> i & 1 != 0 {
            e = format!("fabs({e})");
        }
        if v.neg >> i & 1 != 0 {
            e = format!("(-{e})");
        }
        Ok(e)
    }

    // Source `i` as a `uint` expression, after any SDWA selection.
    pub(super) fn usrc(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        if i == dpp_operand(inst) && matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) {
            return Ok("dpp0".into());
        }
        let e = self.src32(v.src[i], inst.literal, pc)?;
        if v.enc == VEnc::Sdwa && i < 2 {
            let sel = v.sdwa.src_sel[i];
            let sext = v.sdwa.sext[i];
            return Ok(match (sel, sext) {
                (6, _) => e,
                (0..=3, false) => format!("(({e} >> {}) & 0xffu)", 8 * sel),
                (0..=3, true) => format!("uint(int(char({e} >> {})))", 8 * sel),
                (4 | 5, false) => format!("(({e} >> {}) & 0xffffu)", 16 * (sel - 4)),
                (4 | 5, true) => format!("uint(int(short({e} >> {})))", 16 * (sel - 4)),
                _ => return Err(err(pc, "bad SDWA select")),
            });
        }
        Ok(e)
    }

    /// An SDWA instruction writing a byte or word of its destination: the
    /// whole result, then merged into the old value per `dst_unused`.
    fn sdwa_dst(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let (shift, mask) = match v.sdwa.dst_sel {
            sel @ 0..=3 => (8 * sel as u32, 0xffu32),
            sel @ (4 | 5) => (16 * (sel as u32 - 4), 0xffff),
            _ => return Err(err(pc, "bad SDWA destination select")),
        };
        let d = self.v(v.vdst as u16);
        let old = format!("sdwa_old{pc}");
        self.line("{");
        self.line(&format!("uint {old} = {d};"));
        let mut whole = *v;
        whole.sdwa.dst_sel = 6;
        self.valu(pc, inst, &whole)?;
        let field = format!("(({d} & 0x{mask:x}u) << {shift}u)");
        let merged = match v.sdwa.dst_unused {
            0 => field,
            1 => {
                let top = shift + mask.count_ones();
                if top == 32 {
                    field
                } else {
                    format!("uint(int({field} << {}u) >> {}u)", 32 - top, 32 - top)
                }
            }
            _ => format!("(({old} & ~(0x{mask:x}u << {shift}u)) | {field})"),
        };
        self.guard(&format!("{d} = {merged};"));
        self.line("}");
        Ok(())
    }

    fn lane_bit(&mut self, pc: usize, v: &Valu, i: usize) -> Result<String, TranslateError> {
        let code = v.src[i];
        if self.lockstep {
            return Ok(format!(
                "uint(({} >> wl) & 1ul)",
                self.src64(code, None, pc)?
            ));
        }
        Ok(format!("({} & 1u)", self.src32(code, None, pc)?))
    }

    // This thread's bit of VCC as a `uint`.
    pub(super) fn vcc_bit(&mut self) -> String {
        format!("uint({})", self.mask_on(106))
    }

    pub(super) fn valu(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let r = self.valu_inner(pc, inst, v);
        if self.dpp_lane_off {
            self.dpp_lane_off = false;
            self.line("}");
        }
        r
    }

    fn valu_inner(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let op = inst.op;
        let name = op.name;
        let indexed_read = matches!(
            name,
            "v_movrels_b32" | "v_movrelsd_b32" | "v_movrelsd_2_b32"
        );
        if matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) && !indexed_read {
            let a = self.src32(v.src[dpp_operand(inst)], None, pc)?;
            self.dpp_src0(pc, inst, v, &a)?;
        }
        if v.enc == VEnc::Sdwa && op.fmt != Fmt::Vopc && v.sdwa.dst_sel != 6 {
            return self.sdwa_dst(pc, inst, v);
        }
        if op.fmt == Fmt::Vopc {
            return self.vopc(pc, inst, v);
        }
        if double::is_float64(name) {
            return self.float64(pc, inst, v);
        }
        if matches!(
            name,
            "v_interp_p1ll_f16" | "v_interp_p1lv_f16" | "v_interp_p2_f16"
        ) {
            return self.interp_f16(pc, name, v);
        }
        // `v_clrexcp` leaves TRAPSTS alone on gfx1033 (measured).
        if matches!(name, "v_nop" | "v_illegal" | "v_pipeflush" | "v_clrexcp") {
            return Ok(());
        }
        if name == "v_swap_b32" {
            if v.src[0] < 256 || v.enc != VEnc::E32 {
                return Err(err(pc, format!("unsupported form of {name}")));
            }
            let (a, d) = (self.v(v.src[0] - 256), self.v(v.vdst as u16));
            self.guard(&format!("uint t = {d}; {d} = {a}; {a} = t;"));
            return Ok(());
        }
        // Lanes are not modelled: a register is one value per thread. A
        // constant lane gets its own variable, which covers SGPR spills.
        let lane = |src: u16| (128..=192).contains(&src).then(|| src - 128);
        if name == "v_writelane_b32" && self.lockstep {
            let s = self.usrc(pc, inst, v, 0)?;
            let k = self.src32(v.src[1], inst.literal, pc)?;
            let d = self.v(v.vdst as u16);
            self.line(&format!("if (wl == ({k} & 63u)) {d} = {s};"));
            return Ok(());
        }
        if name == "v_writelane_b32" {
            let s = self.usrc(pc, inst, v, 0)?;
            let target = match lane(v.src[1]) {
                Some(k) => format!("v{}_l{k}", v.vdst),
                None => self.v(v.vdst as u16),
            };
            self.regs.insert(target.clone());
            self.line(&format!("{target} = {s};"));
            return Ok(());
        }
        if name == "v_readfirstlane_b32" || name == "v_readlane_b32" {
            let spilled = (name == "v_readlane_b32" && v.src[0] >= 256)
                .then(|| lane(v.src[1]).map(|k| format!("v{}_l{k}", v.src[0] - 256)))
                .flatten()
                .filter(|n| self.regs.contains(n));
            let s = match spilled {
                Some(n) => n,
                None if self.lockstep && v.src[0] >= 256 => {
                    let a = self.usrc(pc, inst, v, 0)?;
                    let k = if name == "v_readfirstlane_b32" {
                        let e = self.src64(126, None, pc)?;
                        format!("({e} == 0ul ? 0u : uint(ctz({e})))")
                    } else {
                        match lane(v.src[1]) {
                            Some(k) => format!("{}u", k & 63),
                            None => format!("({} & 63u)", self.src32(v.src[1], inst.literal, pc)?),
                        }
                    };
                    format!("wave_readlane({a}, {k}, wv)")
                }
                None => {
                    let a = self.usrc(pc, inst, v, 0)?;
                    match lane(v.src[1]) {
                        // Metal SIMD groups are 32 lanes: lane k reads lane k mod 32 of its group.
                        Some(k) if name == "v_readlane_b32" && self.req.stage != Stage::Vertex => {
                            format!("simd_broadcast({a}, ushort({}))", k & 31)
                        }
                        _ => a,
                    }
                }
            };
            if let Some(d) = self.sdst(v.vdst as u16) {
                self.line(&format!("{d} = {s};"));
            }
            return Ok(());
        }
        if matches!(
            name,
            "v_add_co_u32"
                | "v_sub_co_u32"
                | "v_subrev_co_u32"
                | "v_add_co_ci_u32"
                | "v_sub_co_ci_u32"
                | "v_subrev_co_ci_u32"
        ) {
            return self.carry_op(pc, inst, v);
        }
        if name == "v_mad_u64_u32" || name == "v_mad_i64_i32" {
            return self.mad64(pc, inst, v);
        }
        if matches!(
            name,
            "v_qsad_pk_u16_u8" | "v_mqsad_pk_u16_u8" | "v_mqsad_u32_u8"
        ) {
            return self.qsad(pc, inst, v);
        }
        if matches!(name, "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64") {
            let shift = self.usrc(pc, inst, v, 0)?;
            let x = self.vsrc64(pc, inst, v.src[1])?;
            let value = match name {
                "v_lshlrev_b64" => format!("{x} << ({shift} & 63u)"),
                "v_lshrrev_b64" => format!("{x} >> ({shift} & 63u)"),
                _ => format!("ulong(long({x}) >> ({shift} & 63u))"),
            };
            let exec = self.mask_on(126);
            let (lo, hi) = (self.v(v.vdst as u16), self.v(v.vdst as u16 + 1));
            self.line(&format!(
                "{{ ulong r = {value}; if ({exec}) {{ {lo} = uint(r); {hi} = uint(r >> 32); }} }}"
            ));
            return Ok(());
        }
        if matches!(
            name,
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16"
        ) {
            return self.fma_mix(pc, inst, v);
        }
        if name.starts_with("v_pk_") && (name.ends_with("16") && !name.ends_with("_f16")) {
            return self.pk_int16(pc, inst, v);
        }
        if PACKED_F16.contains(&name) {
            return self.pk_f16(pc, inst, v);
        }
        if float::is_float16(name) {
            return self.float16(pc, inst, v);
        }
        if HALF_OPS.contains(&name) {
            return self.half16(pc, inst, v);
        }
        if self.float32(pc, inst, v)? {
            return Ok(());
        }
        if name.starts_with("v_movrel") || name == "v_swaprel_b32" {
            return self.movrel(pc, inst, v);
        }
        if name == "v_permlane16_b32" || name == "v_permlanex16_b32" {
            if self.req.stage == Stage::Vertex {
                return Err(err(pc, format!("{name} in a vertex program")));
            }
            let a = self.usrc(pc, inst, v, 0)?;
            let (lo, hi) = (self.usrc(pc, inst, v, 1)?, self.usrc(pc, inst, v, 2)?);
            self.simd_lane = true;
            let row = if name == "v_permlanex16_b32" {
                "((slane & 16u) ^ 16u)"
            } else {
                "(slane & 16u)"
            };
            let d = self.v(v.vdst as u16);
            self.line(&format!(
                "{{ uint i = slane & 15u; uint sel = (i < 8u ? {lo} >> (4u * i) : {hi} >> (4u * (i - 8u))) & 15u; uint r = simd_shuffle({a}, ushort({row} | sel));"
            ));
            self.guard(&format!("{d} = r;"));
            self.line("}");
            return Ok(());
        }
        let f = |t: &mut Self, i: usize| t.fsrc(pc, inst, v, i);
        let u = |t: &mut Self, i: usize| t.usrc(pc, inst, v, i);
        let vdst = v.vdst as u16;
        self.regs.insert(format!("v{vdst}"));
        match name {
            "v_mov_b32" => {
                let a = u(self, 0)?;
                let d = self.v(vdst);
                self.guard(&format!("{d} = {a};"));
                return Ok(());
            }
            "v_med3_i32" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                let ui = format!("uint(med3_i(int({a}), int({b}), int({c})))");
                let d = self.v(vdst);
                self.guard(&format!("{d} = {ui};"));
                return Ok(());
            }
            "v_med3_u32" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                let ui = format!("max(min({a}, {b}), min(max({a}, {b}), {c}))");
                let d = self.v(vdst);
                self.guard(&format!("{d} = {ui};"));
                return Ok(());
            }
            "v_min3_u32" | "v_max3_u32" | "v_min3_i32" | "v_max3_i32" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                let f = if name.starts_with("v_min") {
                    "min"
                } else {
                    "max"
                };
                let ui = if name.ends_with("i32") {
                    format!("uint({f}({f}(int({a}), int({b})), int({c})))")
                } else {
                    format!("{f}({f}({a}, {b}), {c})")
                };
                let d = self.v(vdst);
                self.guard(&format!("{d} = {ui};"));
                return Ok(());
            }
            _ => {}
        }
        let ui = match name {
            "v_cndmask_b32" => {
                let sel = if matches!(v.enc, VEnc::E64) {
                    self.lane_bit(pc, v, 2)?
                } else {
                    self.vcc_bit()
                };
                let (a, b) = if v.neg != 0 || v.abs != 0 {
                    (format!("U({})", f(self, 0)?), format!("U({})", f(self, 1)?))
                } else {
                    (u(self, 0)?, u(self, 1)?)
                };
                format!("{sel} != 0u ? {b} : {a}")
            }
            "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" => {
                let (a, b) = (u(self, 0)?, u(self, 1)?);
                let (a, b) = if name == "v_subrev_nc_u32" {
                    (b, a)
                } else {
                    (a, b)
                };
                match (name == "v_add_nc_u32", v.clamp) {
                    (true, false) => format!("{a} + {b}"),
                    (true, true) => format!("{a} + min({b}, ~{a})"),
                    (false, false) => format!("{a} - {b}"),
                    (false, true) => format!("{a} - min({a}, {b})"),
                }
            }
            "v_add_nc_i32" | "v_sub_nc_i32" => {
                let (a, b) = (u(self, 0)?, u(self, 1)?);
                let op = if name == "v_add_nc_i32" { '+' } else { '-' };
                if v.clamp {
                    sat_i32(&format!("long(int({a})) {op} long(int({b}))"))
                } else {
                    format!("{a} {op} {b}")
                }
            }
            "v_and_b32" => format!("{} & {}", u(self, 0)?, u(self, 1)?),
            "v_or_b32" => format!("{} | {}", u(self, 0)?, u(self, 1)?),
            "v_xor_b32" => format!("{} ^ {}", u(self, 0)?, u(self, 1)?),
            "v_xnor_b32" => format!("~({} ^ {})", u(self, 0)?, u(self, 1)?),
            "v_not_b32" => format!("~{}", u(self, 0)?),
            "v_bfrev_b32" => format!("reverse_bits({})", u(self, 0)?),
            "v_lshlrev_b32" => format!("{} << ({} & 31u)", u(self, 1)?, u(self, 0)?),
            "v_lshrrev_b32" => format!("{} >> ({} & 31u)", u(self, 1)?, u(self, 0)?),
            "v_ashrrev_i32" => format!("uint(int({}) >> ({} & 31u))", u(self, 1)?, u(self, 0)?),
            "v_min_i32" => format!("uint(min(int({}), int({})))", u(self, 0)?, u(self, 1)?),
            "v_max_i32" => format!("uint(max(int({}), int({})))", u(self, 0)?, u(self, 1)?),
            "v_min_u32" => format!("min({}, {})", u(self, 0)?, u(self, 1)?),
            "v_max_u32" => format!("max({}, {})", u(self, 0)?, u(self, 1)?),
            "v_mul_lo_u32" | "v_mul_lo_i32" => format!("{} * {}", u(self, 0)?, u(self, 1)?),
            "v_mul_hi_u32" => format!("mulhi({}, {})", u(self, 0)?, u(self, 1)?),
            "v_mul_hi_i32" => format!("uint(mulhi(int({}), int({})))", u(self, 0)?, u(self, 1)?),
            "v_mul_u32_u24" | "v_mul_hi_u32_u24" | "v_mad_u32_u24" => {
                let (a, b) = (u(self, 0)?, u(self, 1)?);
                let mut e = format!("ulong({a} & 0xffffffu) * ulong({b} & 0xffffffu)");
                if name == "v_mad_u32_u24" {
                    e = format!("{e} + ulong({})", u(self, 2)?);
                }
                match name {
                    "v_mul_hi_u32_u24" => format!("uint(({e}) >> 32)"),
                    _ if v.clamp => format!("uint(min({e}, 0xfffffffful))"),
                    _ => format!("uint({e})"),
                }
            }
            "v_mul_i32_i24" | "v_mul_hi_i32_i24" | "v_mad_i32_i24" => {
                let (a, b) = (u(self, 0)?, u(self, 1)?);
                let i24 = |x: &str| format!("long((int({x}) << 8) >> 8)");
                let mut e = format!("{} * {}", i24(&a), i24(&b));
                if name == "v_mad_i32_i24" {
                    e = format!("{e} + long(int({}))", u(self, 2)?);
                }
                match name {
                    "v_mul_hi_i32_i24" => format!("uint(ulong({e}) >> 32)"),
                    _ if v.clamp => sat_i32(&e),
                    _ => format!("uint({e})"),
                }
            }
            "v_bfe_u32" => format!(
                "bfe_u32({}, {} & 31u, {} & 31u)",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_bfe_i32" => format!(
                "bfe_i32({}, {} & 31u, {} & 31u)",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_bfi_b32" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                format!("({a} & {b}) | (~{a} & {c})")
            }
            "v_bfm_b32" => format!(
                "((1u << ({} & 31u)) - 1u) << ({} & 31u)",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_lshl_add_u32" => format!(
                "({} << ({} & 31u)) + {}",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_add_lshl_u32" => format!(
                "({} + {}) << ({} & 31u)",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_lshl_or_b32" => format!(
                "({} << ({} & 31u)) | {}",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_and_or_b32" => format!("({} & {}) | {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_or3_b32" => format!("{} | {} | {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_xor3_b32" => format!("{} ^ {} ^ {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_add3_u32" => format!("{} + {} + {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_xad_u32" => format!("({} ^ {}) + {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_alignbyte_b32" => format!(
                "uint((ulong({}) << 32 | ulong({})) >> (8u * ({} & 3u)))",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_alignbit_b32" => format!(
                "uint((ulong({}) << 32 | ulong({})) >> ({} & 31u))",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_cvt_pk_u16_u32" => format!(
                "(min({}, 0xffffu)) | (min({}, 0xffffu) << 16)",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_bcnt_u32_b32" => format!("popcount({}) + {}", u(self, 0)?, u(self, 1)?),
            "v_ffbh_u32" => {
                let a = u(self, 0)?;
                format!("({a} == 0u ? 0xffffffffu : clz({a}))")
            }
            "v_ffbh_i32" => format!("ffbh_i32({})", u(self, 0)?),
            "v_cvt_pk_i16_i32" => {
                let (a, b) = (u(self, 0)?, u(self, 1)?);
                let s16 = |x: &str| format!("(uint(clamp(int({x}), -32768, 32767)) & 0xffffu)");
                format!("{} | ({} << 16)", s16(&a), s16(&b))
            }
            "v_sat_pk_u8_i16" => {
                let a = u(self, 0)?;
                let u8 = |x: String| format!("uint(clamp(int(short({x})), 0, 255))");
                format!("{} | ({} << 8)", u8(a.clone()), u8(format!("({a} >> 16)")))
            }
            "v_lerp_u8" => format!("lerp_u8({}, {}, {})", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_sad_u8" | "v_sad_hi_u8" | "v_sad_u16" | "v_sad_u32" | "v_msad_u8" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                let e = match name {
                    "v_sad_u8" => format!("ulong(sad_u8({a}, {b}, false)) + ulong({c})"),
                    "v_msad_u8" => format!("ulong(sad_u8({a}, {b}, true)) + ulong({c})"),
                    "v_sad_hi_u8" => format!("(ulong(sad_u8({a}, {b}, false)) << 16) + ulong({c})"),
                    "v_sad_u16" => format!(
                        "ulong(absdiff({a} >> 16, {b} >> 16) + absdiff({a} & 0xffffu, {b} & 0xffffu)) + ulong({c})"
                    ),
                    _ => format!("ulong(absdiff({a}, {b})) + ulong({c})"),
                };
                if v.clamp {
                    format!("uint(min({e}, 0xfffffffful))")
                } else {
                    format!("uint({e})")
                }
            }
            "v_perm_b32" => format!(
                "perm_b32({}, {}, {})",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_ffbl_b32" => {
                let a = u(self, 0)?;
                format!("({a} == 0u ? 0xffffffffu : ctz({a}))")
            }
            "v_mbcnt_lo_u32_b32" if self.lockstep => format!(
                "popcount({} & (wl >= 32u ? 0xffffffffu : (1u << wl) - 1u)) + {}",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_mbcnt_hi_u32_b32" if self.lockstep => format!(
                "popcount({} & (wl < 32u ? 0u : (1u << (wl - 32u)) - 1u)) + {}",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => u(self, 1)?.to_string(),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let d = self.v(vdst);
        self.guard(&format!("{d} = {ui};"));
        Ok(())
    }

    // A DPP quad permutation that reads one lane of the quad becomes a quad
    // broadcast of src0 into `dpp0`, which then stands in for src0.
    pub(super) fn dpp_src0(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        a: &str,
    ) -> Result<(), TranslateError> {
        let ctrl = v.dpp.ctrl;
        let lane = ctrl & 3;
        let broadcast = v.enc == VEnc::Dpp16
            && ctrl <= 0xFF
            && (0..4).all(|i| (ctrl >> (2 * i)) & 3 == lane)
            && v.dpp.row_mask == 0xF
            && v.dpp.bank_mask == 0xF;
        let permute = v.enc == VEnc::Dpp16
            && ctrl <= 0xFF
            && v.dpp.row_mask == 0xF
            && v.dpp.bank_mask == 0xF
            && self.req.stage != Stage::Vertex;
        if !broadcast && !permute && self.req.stage != Stage::Vertex {
            return self.dpp_general(pc, v, a);
        }
        if !broadcast && !permute {
            return Err(err(pc, format!("unsupported DPP form of {}", inst.op.name)));
        }
        self.regs.insert("dpp0".into());
        if broadcast {
            self.line(&format!("dpp0 = quad_broadcast({a}, {lane}u);"));
        } else {
            self.quad_lane = true;
            self.line(&format!(
                "dpp0 = quad_shuffle({a}, ushort(({ctrl:#x}u >> (2u * uint(qlane))) & 3u));"
            ));
        }
        Ok(())
    }

    /// DPP forms other than full-mask quad permutations, as `simd_shuffle`
    /// within 32-lane Metal SIMD groups.
    fn dpp_general(&mut self, pc: usize, v: &Valu, a: &str) -> Result<(), TranslateError> {
        let ctrl = v.dpp.ctrl as u32;
        let (src, inside) = if v.enc == VEnc::Dpp8 {
            (
                format!(
                    "((i & 24u) | (({:#x}u >> (3u * (i & 7u))) & 7u))",
                    v.dpp.sel8
                ),
                "true".to_string(),
            )
        } else {
            let n = ctrl & 0xF;
            match ctrl {
                0x00..=0xFF => (
                    format!("((i & 28u) | (({ctrl:#x}u >> (2u * (i & 3u))) & 3u))"),
                    "true".into(),
                ),
                0x101..=0x10F => (format!("(i + {n}u)"), format!("(i & 15u) + {n}u < 16u")),
                0x111..=0x11F => (format!("(i - {n}u)"), format!("(i & 15u) >= {n}u")),
                0x121..=0x12F => (format!("((i & 16u) | ((i - {n}u) & 15u))"), "true".into()),
                0x140 => ("((i & 16u) | (15u - (i & 15u)))".into(), "true".into()),
                0x141 => ("((i & 24u) | (7u - (i & 7u)))".into(), "true".into()),
                0x150..=0x15F => (format!("((i & 16u) | {n}u)"), "true".into()),
                0x160..=0x16F => (format!("(i ^ {n}u)"), "true".into()),
                _ => return Err(err(pc, format!("unsupported DPP control {ctrl:#x}"))),
            }
        };
        self.simd_lane = true;
        self.regs.insert("dpp0".into());
        let (rows, banks) = if v.enc == VEnc::Dpp8 {
            (0xF, 0xF)
        } else {
            (v.dpp.row_mask, v.dpp.bank_mask)
        };
        let row = if self.lockstep { "wl" } else { "i" };
        self.line(&format!(
            "{{ uint i = slane; dpp0 = simd_shuffle({a}, ushort({src} & 31u)); bool dpp_on = ({rows:#x}u >> (({row} >> 4) & 3u) & 1u) != 0u && ({banks:#x}u >> ((i >> 2) & 3u) & 1u) != 0u; bool dpp_in = {inside};"
        ));
        if v.dpp.bound_ctrl {
            self.line("if (!dpp_in) dpp0 = 0u; ");
        } else {
            self.line("dpp_on = dpp_on && dpp_in;");
        }
        self.dpp_lane_off = true;
        Ok(())
    }

    // Integer add and subtract with a carry or borrow out to a lane mask, and maybe in from one.
    fn carry_op(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let e64 = v.enc == VEnc::E64;
        let a = self.usrc(pc, inst, v, 0)?;
        let b = self.usrc(pc, inst, v, 1)?;
        let cin = if !name.contains("_ci_") {
            "0u".to_string()
        } else if e64 {
            self.lane_bit(pc, v, 2)?
        } else {
            self.vcc_bit()
        };
        let (a, b) = if name.starts_with("v_subrev") {
            (b, a)
        } else {
            (a, b)
        };
        let (value, carry) = if name.starts_with("v_add") {
            ("ulong(a) + ulong(b) + ulong(c)", "(r >> 32) != 0ul")
        } else {
            (
                "ulong(a) - ulong(b) - ulong(c)",
                "ulong(b) + ulong(c) > ulong(a)",
            )
        };
        let dst = if e64 {
            v.sdst.map_or(106, |s| s as u16)
        } else {
            106
        };
        let mut exec = self.mask_on(126);
        if self.dpp_lane_off {
            exec = format!("{exec} && dpp_on");
        }
        let d = self.v(v.vdst as u16);
        let mut s = format!(
            "{{ uint a = {a}; uint b = {b}; uint c = {cin}; ulong r = {value}; bool on = {exec}; bool k = ({carry}) && on;"
        );
        let result = match (v.clamp, name.starts_with("v_add")) {
            (false, _) => "uint(r)",
            (true, true) => "(k ? 0xffffffffu : uint(r))",
            (true, false) => "(k ? 0u : uint(r))",
        };
        write!(s, " if (on) {d} = {result};").unwrap();
        if dst != src::NULL {
            s.push_str(&self.set_mask(dst, "k"));
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }

    /// A 64-bit vector-instruction source: a VGPR pair or a scalar source.
    fn vsrc64(&mut self, pc: usize, inst: &Inst, code: u16) -> Result<String, TranslateError> {
        Ok(match code {
            n @ 256.. => {
                let (lo, hi) = (self.v(n - 256), self.v(n - 255));
                format!("(ulong({hi}) << 32 | ulong({lo}))")
            }
            n => self.src64(n, inst.literal, pc)?,
        })
    }

    // `v_mad_u64_u32`/`v_mad_i64_i32`: a 64-bit product plus a 64-bit addend
    // into a VGPR pair, with the carry (unsigned) or overflow (signed) out.
    fn mad64(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let a = self.usrc(pc, inst, v, 0)?;
        let b = self.usrc(pc, inst, v, 1)?;
        let c = self.vsrc64(pc, inst, v.src[2])?;
        let signed = inst.op.name == "v_mad_i64_i32";
        // The carry out is bit 64 of the exact sum, for the signed form that
        // of the sign-extended product and addend.
        let (product, carry) = if signed {
            (
                format!("ulong(long(int({a})) * long(int({b})))"),
                "((((p ^ c) >> 63) ^ ulong(r < c)) != 0ul)",
            )
        } else {
            (format!("ulong({a}) * ulong({b})"), "r < c")
        };
        let mut value = "r".to_string();
        if v.clamp {
            value = if signed {
                "(((p ^ r) & (c ^ r)) >> 63 != 0ul ? (long(r) < 0 ? 0x7ffffffffffffffful : 0x8000000000000000ul) : r)".into()
            } else {
                "(r < c ? ~0ul : r)".into()
            };
        }
        let exec = self.mask_on(126);
        let (lo, hi) = (self.v(v.vdst as u16), self.v(v.vdst as u16 + 1));
        let mut s = format!(
            "{{ ulong p = {product}; ulong c = {c}; ulong r = p + c; bool on = {exec}; bool k = ({carry}) && on; ulong d = {value}; if (on) {{ {lo} = uint(d); {hi} = uint(d >> 32); }}"
        );
        let dst = v.sdst.map_or(src::NULL, |s| s as u16);
        if dst != src::NULL {
            s.push_str(&self.set_mask(dst, "k"));
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }

    /// The quad-byte SADs: four SADs of `src1` against the byte windows of
    /// `src0` that start at bytes 0 to 3, each added to its part of `src2`.
    fn qsad(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let a = self.vsrc64(pc, inst, v.src[0])?;
        let b = self.usrc(pc, inst, v, 1)?;
        let masked = name.starts_with("v_mqsad");
        let exec = self.mask_on(126);
        let mut s = format!("{{ ulong a = {a}; uint b = {b}; bool on = {exec};");
        if name == "v_mqsad_u32_u8" {
            let c = v.src[2];
            if c < 256 {
                return Err(err(pc, format!("{name} with a scalar accumulator")));
            }
            for k in 0..4u16 {
                let acc = self.v(c - 256 + k);
                let d = self.v(v.vdst as u16 + k);
                let sum = format!(
                    "ulong(sad_u8(uint(a >> {}), b, true)) + ulong({acc})",
                    8 * k
                );
                let r = if v.clamp {
                    format!("uint(min({sum}, 0xfffffffful))")
                } else {
                    format!("uint({sum})")
                };
                write!(s, " uint r{k} = {r};").unwrap();
                write!(s, " if (on) {d} = r{k};").unwrap();
            }
        } else {
            let c = self.vsrc64(pc, inst, v.src[2])?;
            write!(s, " ulong c = {c}; ulong r = 0ul;").unwrap();
            let limit = if v.clamp { "0xffffu" } else { "0xffffffffu" };
            for k in 0..4 {
                write!(
                    s,
                    " r |= ulong(min(sad_u8(uint(a >> {}), b, {masked}) + ((uint(c >> {}) & 0xffffu)), {limit}) & 0xffffu) << {};",
                    8 * k,
                    16 * k,
                    16 * k
                )
                .unwrap();
            }
            let (lo, hi) = (self.v(v.vdst as u16), self.v(v.vdst as u16 + 1));
            write!(s, " if (on) {{ {lo} = uint(r); {hi} = uint(r >> 32); }}").unwrap();
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }

    pub(super) fn vopc(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let cmpx = name.starts_with("v_cmpx_");
        let rest = name
            .trim_start_matches("v_cmpx_")
            .trim_start_matches("v_cmp_");
        let (op, ty) = rest
            .rsplit_once('_')
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        let float = if ty == "f64" {
            Some(self.fcompare64(pc, inst, v, op)?)
        } else {
            self.fcompare(pc, inst, v, op, ty)?
        };
        let cond = match ty {
            _ if float.is_some() => float.unwrap_or_default(),
            "i32" | "u32" | "i16" | "u16" => {
                let (a, b) = (self.usrc(pc, inst, v, 0)?, self.usrc(pc, inst, v, 1)?);
                let (a, b) = match ty {
                    "i32" => (format!("int({a})"), format!("int({b})")),
                    "i16" => (format!("int(short({a}))"), format!("int(short({b}))")),
                    "u16" => (format!("({a} & 0xffffu)"), format!("({b} & 0xffffu)")),
                    _ => (a, b),
                };
                match op {
                    "f" => "false".into(),
                    "t" => "true".into(),
                    _ => format!(
                        "{a} {} {b}",
                        cmp_op(op)
                            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?
                    ),
                }
            }
            "i64" | "u64" => {
                if v.enc == VEnc::Sdwa || matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) {
                    return Err(err(pc, format!("unsupported form of {name}")));
                }
                let (a, b) = (
                    self.vsrc64(pc, inst, v.src[0])?,
                    self.vsrc64(pc, inst, v.src[1])?,
                );
                let (a, b) = if ty == "i64" {
                    (format!("long({a})"), format!("long({b})"))
                } else {
                    (a, b)
                };
                match op {
                    "f" => "false".into(),
                    "t" => "true".into(),
                    _ => format!(
                        "{a} {} {b}",
                        cmp_op(op)
                            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?
                    ),
                }
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let e = self.mask_on(126);
        let dst = if cmpx {
            126
        } else {
            match (v.enc, v.sdst) {
                (VEnc::E64, _) => v.vdst as u16,
                (_, Some(s)) => s as u16,
                _ => 106,
            }
        };
        if dst == src::NULL {
            return Ok(());
        }
        let set = self.set_mask(dst, "c");
        self.line(&format!("{{ bool c = ({cond}) && {e};{set} }}"));
        Ok(())
    }
}

/// A 64-bit signed expression saturated to a 32-bit signed result.
fn sat_i32(e: &str) -> String {
    format!("uint(int(clamp({e}, long(-2147483648), long(2147483647))))")
}

/// The source DPP takes from other lanes: src0, except in the integer
/// opcodes with reversed operands, where RDNA2 was measured to apply it to
/// the second source (`v_subrev_f32` and `v_subrev_f16` keep src0).
fn dpp_operand(inst: &Inst) -> usize {
    match inst.op.name {
        "v_subrev_nc_u32" | "v_subrev_co_ci_u32" | "v_lshlrev_b32" | "v_lshrrev_b32"
        | "v_ashrrev_i32" => 1,
        _ => 0,
    }
}
