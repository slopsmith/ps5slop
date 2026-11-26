//! 16-bit and packed half-precision VALU ops, mixed FMA and indexed VGPR moves.

use super::*;

/// The float inline constants 240 to 248 as halves.
const F16_CONSTANTS: [u32; 9] = [
    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
];
pub(super) const HALF_OPS: &[&str] = &[
    "v_add_nc_u16",
    "v_add_nc_i16",
    "v_sub_nc_u16",
    "v_sub_nc_i16",
    "v_mul_lo_u16",
    "v_lshlrev_b16",
    "v_lshrrev_b16",
    "v_ashrrev_i16",
    "v_max_u16",
    "v_min_u16",
    "v_max_i16",
    "v_min_i16",
    "v_mad_u16",
    "v_mad_i16",
    "v_mad_u32_u16",
    "v_mad_i32_i16",
    "v_min3_u16",
    "v_min3_i16",
    "v_max3_u16",
    "v_max3_i16",
    "v_med3_u16",
    "v_med3_i16",
];
pub(super) const PACKED_F16: &[&str] = &[
    "v_pk_add_f16",
    "v_pk_mul_f16",
    "v_pk_max_f16",
    "v_pk_min_f16",
    "v_pk_fma_f16",
    "v_pk_fmac_f16",
];

impl<'a> Tx<'a> {
    /// `v_fma_mix*`: each source is a float, or with its `op_sel_hi` bit the
    /// half `op_sel` picks; `neg_hi` is its abs. Inline constants keep their
    /// value as halves.
    pub(super) fn fma_mix(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        let mut srcs = Vec::new();
        for i in 0..3 {
            let u = self.usrc(pc, inst, v, i)?;
            let inline = (240..=248).contains(&v.src[i]);
            let mut e = if v.opsel_hi >> i & 1 != 0 && !inline {
                let shift = if v.opsel >> i & 1 != 0 { " >> 16" } else { "" };
                format!("widen16({u}{shift})")
            } else {
                format!("in32({u})")
            };
            if v.neg_hi >> i & 1 != 0 {
                e = format!("({e} & 0x7fffffffu)");
            }
            if v.neg >> i & 1 != 0 {
                e = format!("({e} ^ 0x80000000u)");
            }
            srcs.push(e);
        }
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        let stmt = if inst.op.name == "v_fma_mix_f32" {
            let mut r = format!("fma32({})", srcs.join(", "));
            if v.clamp {
                r = format!("clamp32({r})");
            }
            format!("{d} = {r};")
        } else {
            let mut r = format!("fma_mix16({})", srcs.join(", "));
            if v.clamp {
                r = format!("clamp16({r})");
            }
            if inst.op.name == "v_fma_mixlo_f16" {
                format!("{d} = ({d} & 0xffff0000u) | {r};")
            } else {
                format!("{d} = ({d} & 0xffffu) | {r} << 16;")
            }
        };
        self.guard(&stmt);
        Ok(())
    }

    /// Packed 16-bit integer operations: `op_sel` picks each source's half
    /// for the low result, `op_sel_hi` for the high one. Clamp saturates the
    /// arithmetic ones.
    pub(super) fn pk_int16(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let n = inst.op.srcs.len().min(3);
        let mut srcs = Vec::new();
        for i in 0..n {
            srcs.push(self.usrc(pc, inst, v, i)?);
        }
        let signed = name.ends_with("_i16");
        let saturates = v.clamp
            && matches!(
                name,
                "v_pk_add_u16"
                    | "v_pk_add_i16"
                    | "v_pk_sub_u16"
                    | "v_pk_sub_i16"
                    | "v_pk_mad_u16"
                    | "v_pk_mad_i16"
            );
        // `neg_lo`/`neg_hi` flip bit 15 of the half, as for a half float.
        let lane = |hi_sel: u8, neg: u8| -> Option<String> {
            let x: Vec<String> = (0..n)
                .map(|i| {
                    let half = if hi_sel >> i & 1 != 0 {
                        format!("({} >> 16)", srcs[i])
                    } else {
                        format!("({} & 0xffffu)", srcs[i])
                    };
                    if neg >> i & 1 != 0 {
                        format!("({half} ^ 0x8000u)")
                    } else {
                        half
                    }
                })
                .collect();
            let int = |e: &str| format!("int(short({e}))");
            let wide = |e: &str| {
                if signed {
                    format!("long(short({e}))")
                } else {
                    format!("long({e})")
                }
            };
            let r = match name {
                "v_pk_add_u16" | "v_pk_add_i16" if saturates => {
                    format!("{} + {}", wide(&x[0]), wide(&x[1]))
                }
                "v_pk_sub_u16" | "v_pk_sub_i16" if saturates => {
                    format!("{} - {}", wide(&x[0]), wide(&x[1]))
                }
                "v_pk_mad_u16" | "v_pk_mad_i16" if saturates => {
                    format!("{} * {} + {}", wide(&x[0]), wide(&x[1]), wide(&x[2]))
                }
                "v_pk_add_u16" | "v_pk_add_i16" => format!("{} + {}", x[0], x[1]),
                "v_pk_sub_u16" | "v_pk_sub_i16" => format!("{} - {}", x[0], x[1]),
                "v_pk_mad_u16" | "v_pk_mad_i16" => format!("{} * {} + {}", x[0], x[1], x[2]),
                "v_pk_mul_lo_u16" => format!("{} * {}", x[0], x[1]),
                "v_pk_lshlrev_b16" => format!("{} << ({} & 15u)", x[1], x[0]),
                "v_pk_lshrrev_b16" => format!("{} >> ({} & 15u)", x[1], x[0]),
                "v_pk_ashrrev_i16" => format!("uint({} >> ({} & 15u))", int(&x[1]), x[0]),
                "v_pk_max_i16" => format!("uint(max({}, {}))", int(&x[0]), int(&x[1])),
                "v_pk_min_i16" => format!("uint(min({}, {}))", int(&x[0]), int(&x[1])),
                "v_pk_max_u16" => format!("max({}, {})", x[0], x[1]),
                "v_pk_min_u16" => format!("min({}, {})", x[0], x[1]),
                _ => return None,
            };
            let (lo, hi) = if signed { (-32768, 32767) } else { (0, 65535) };
            Some(if saturates {
                format!("(uint(clamp({r}, long({lo}), long({hi}))) & 0xffffu)")
            } else {
                format!("(({r}) & 0xffffu)")
            })
        };
        let (Some(lo), Some(hi)) = (lane(v.opsel, v.neg), lane(v.opsel_hi, v.neg_hi)) else {
            return Err(err(pc, format!("unsupported instruction {name}")));
        };
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        self.guard(&format!("{d} = {lo} | ({hi} << 16);"));
        Ok(())
    }

    /// A 16-bit source as a `uint` with the value in its low half: float
    /// inline constants are halves, and a VOP3 `op_sel` bit picks the high half.
    pub(super) fn src16(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        let code = v.src[i];
        let e = if (240..=248).contains(&code) && v.enc != VEnc::Sdwa {
            format!("0x{:x}u", F16_CONSTANTS[code as usize - 240])
        } else {
            self.usrc(pc, inst, v, i)?
        };
        Ok(if v.enc == VEnc::E64 && v.opsel >> i & 1 != 0 {
            format!("({e} >> 16)")
        } else {
            e
        })
    }

    /// Writes a 16-bit result to the low half of the destination, or with
    /// VOP3 `op_sel[3]` to the high half; the other half is kept (GFX10).
    pub(super) fn write16(&mut self, v: &Valu, r: &str) {
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        if v.enc == VEnc::Sdwa {
            self.guard(&format!("{d} = {r} & 0xffffu;"));
        } else if v.enc == VEnc::E64 && v.opsel & 8 != 0 {
            self.guard(&format!("{d} = ({d} & 0xffffu) | (({r} & 0xffffu) << 16);"));
        } else {
            self.guard(&format!("{d} = ({d} & 0xffff0000u) | ({r} & 0xffffu);"));
        }
    }

    /// Scalar 16-bit integer operations.
    pub(super) fn half16(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        let name = inst.op.name;
        if matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        let a = self.src16(pc, inst, v, 0)?;
        let b = self.src16(pc, inst, v, 1)?;
        let (ua, ub) = (format!("({a} & 0xffffu)"), format!("({b} & 0xffffu)"));
        let (ia, ib) = (format!("int(short({a}))"), format!("int(short({b}))"));
        // The 32-bit addend's `op_sel` bit takes its high half; the
        // destination's shifts the result's low half into the high one.
        if matches!(name, "v_mad_u32_u16" | "v_mad_i32_i16") {
            let unsigned = name == "v_mad_u32_u16";
            let c = self.usrc(pc, inst, v, 2)?;
            let opsel = |i: u8| v.enc == VEnc::E64 && v.opsel >> i & 1 != 0;
            let c = match (opsel(2), unsigned) {
                (false, true) => format!("long({c})"),
                (false, false) => format!("long(int({c}))"),
                (true, true) => format!("long({c} >> 16)"),
                (true, false) => format!("long(int({c}) >> 16)"),
            };
            let e = if unsigned {
                format!("long({ua} * {ub}) + {c}")
            } else {
                format!("long({ia} * {ib}) + {c}")
            };
            let mut r = match (v.clamp, unsigned) {
                (false, _) => format!("uint({e})"),
                (true, true) => format!("uint(clamp({e}, long(0), long(0xffffffffu)))"),
                (true, false) => {
                    format!("uint(int(clamp({e}, long(-2147483648), long(2147483647))))")
                }
            };
            if opsel(3) {
                r = format!("({r} << 16)");
            }
            let d = self.v(v.vdst as u16);
            self.regs.insert(format!("v{}", v.vdst));
            self.guard(&format!("{d} = {r};"));
            return Ok(());
        }
        if inst.op.srcs.len() >= 3 {
            let c = self.src16(pc, inst, v, 2)?;
            let (uc, ic) = (format!("({c} & 0xffffu)"), format!("int(short({c}))"));
            let signed = name.ends_with("_i16");
            let (x, y, z) = if signed {
                (ia.clone(), ib.clone(), ic)
            } else {
                (ua.clone(), ub.clone(), uc)
            };
            let r = match name {
                "v_mad_u16" | "v_mad_i16" => {
                    let e = format!("long({x}) * long({y}) + long({z})");
                    let (lo, hi) = if signed { (-32768, 32767) } else { (0, 65535) };
                    if v.clamp {
                        format!("uint(clamp({e}, long({lo}), long({hi})))")
                    } else {
                        format!("uint({e})")
                    }
                }
                "v_min3_u16" | "v_min3_i16" => format!("uint(min(min({x}, {y}), {z}))"),
                "v_max3_u16" | "v_max3_i16" => format!("uint(max(max({x}, {y}), {z}))"),
                _ => format!("uint(max(min({x}, {y}), min(max({x}, {y}), {z})))"),
            };
            self.write16(v, &format!("({r})"));
            return Ok(());
        }
        let r = match name {
            "v_add_nc_u16" if v.clamp => format!("min({ua} + {ub}, 0xffffu)"),
            "v_sub_nc_u16" if v.clamp => format!("({ua} - min({ua}, {ub}))"),
            "v_add_nc_i16" if v.clamp => format!("uint(clamp({ia} + {ib}, -32768, 32767))"),
            "v_sub_nc_i16" if v.clamp => format!("uint(clamp({ia} - {ib}, -32768, 32767))"),
            "v_add_nc_u16" | "v_add_nc_i16" => format!("{ua} + {ub}"),
            "v_sub_nc_u16" | "v_sub_nc_i16" => format!("{ua} - {ub}"),
            "v_mul_lo_u16" => format!("{ua} * {ub}"),
            "v_lshlrev_b16" => format!("{ub} << ({ua} & 15u)"),
            "v_lshrrev_b16" => format!("{ub} >> ({ua} & 15u)"),
            "v_ashrrev_i16" => format!("uint({ib} >> ({ua} & 15u))"),
            "v_max_u16" => format!("max({ua}, {ub})"),
            "v_min_u16" => format!("min({ua}, {ub})"),
            "v_max_i16" => format!("uint(max({ia}, {ib}))"),
            "v_min_i16" => format!("uint(min({ia}, {ib}))"),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        if v.clamp && !name.contains("_nc_") {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        self.write16(v, &format!("({r})"));
        Ok(())
    }

    /// Packed half operations: `op_sel` picks each source's half for the low
    /// result, `op_sel_hi` for the high one; `neg_lo`/`neg_hi` negate them.
    pub(super) fn pk_f16(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let (opsel, opsel_hi) = if v.enc == VEnc::P {
            (v.opsel, v.opsel_hi)
        } else if v.enc == VEnc::E32 {
            (0, 7)
        } else {
            return Err(err(pc, format!("unsupported form of {name}")));
        };
        let n = if name == "v_pk_fmac_f16" {
            2
        } else {
            inst.op.srcs.len().min(3)
        };
        let mut lo = Vec::new();

        let mut hi = Vec::new();
        for i in 0..n {
            let code = v.src[i];
            // In the VOP2 form an inline constant fills the low half only
            // (RDNA2 hardware); VOP3P is assumed to replicate it.
            let u = if (240..=248).contains(&code) {
                let c = F16_CONSTANTS[code as usize - 240];
                let hi = if v.enc == VEnc::E32 { 0 } else { c << 16 };
                format!("0x{:x}u", c | hi)
            } else {
                self.usrc(pc, inst, v, i)?
            };
            for (sel, neg, out) in [
                (opsel >> i & 1, v.neg >> i & 1, &mut lo),
                (opsel_hi >> i & 1, v.neg_hi >> i & 1, &mut hi),
            ] {
                let shift = if sel != 0 { " >> 16" } else { "" };
                let mut e = format!("(({u}{shift}) & 0xffffu)");
                if neg != 0 {
                    e = format!("({e} ^ 0x8000u)");
                }
                out.push(e);
            }
        }
        if name == "v_pk_fmac_f16" {
            let d = self.v(v.vdst as u16);
            lo.push(format!("({d} & 0xffffu)"));
            hi.push(format!("({d} >> 16)"));
        }
        let helper = match name {
            "v_pk_add_f16" => "add16",
            "v_pk_mul_f16" => "mul16",
            "v_pk_max_f16" => "max16",
            "v_pk_min_f16" => "min16",
            "v_pk_fma_f16" | "v_pk_fmac_f16" => "fma16",
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let (mut l, mut h) = (
            format!("{helper}({})", lo.join(", ")),
            format!("{helper}({})", hi.join(", ")),
        );
        if v.clamp {
            (l, h) = (format!("clamp16({l})"), format!("clamp16({h})"));
        }
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        self.guard(&format!("{d} = {l} | {h} << 16;"));
        Ok(())
    }
    /// Indexed VGPR moves through M0. VGPRs are named locals, so an indexed
    /// access is a `switch` whose cases `emit` fills in (`VREAD`, `VWRITE`).
    pub(super) fn movrel(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let dwords = v.sdwa.dst_sel == 6 && v.sdwa.src_sel[0] == 6 && !v.sdwa.sext[0];
        let plain = match v.enc {
            VEnc::E32 | VEnc::Dpp16 | VEnc::Dpp8 => true,
            VEnc::E64 => v.abs == 0 && v.neg == 0 && !v.clamp && v.omod == 0,
            VEnc::Sdwa => dwords,
            _ => false,
        };
        if !plain || (name == "v_swaprel_b32" && v.enc != VEnc::E32 && v.enc != VEnc::E64) {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        let dpp = matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8);
        let m0 = self.s(124);
        self.movrel = true;
        let src = v.src[0];
        let dst = v.vdst as u16;
        let (src_index, dst_index) = match name {
            "v_movrelsd_2_b32" | "v_swaprel_b32" => (
                format!("({m0} & 0x3ffu)"),
                format!("(({m0} >> 16) & 0x3ffu)"),
            ),
            _ => (m0.clone(), m0.clone()),
        };
        let read = |out: &str, base: u16, index: &str| {
            format!("uint {out} = 0u; switch ({index} + {base}u) {{ {VREAD}{out}{VEND} }}")
        };
        let write = |base: u16, index: &str, val: &str| {
            format!("switch ({index} + {base}u) {{ {VWRITE}{val}{VEND} }}")
        };
        let base = (src >= 256).then(|| src - 256);
        if let Some(b) = base {
            self.v(b);
        }
        self.v(dst);
        let need_base = || base.ok_or_else(|| err(pc, format!("{name} from a non-VGPR")));
        // DPP reads the indexed source from the lane it picks, so every lane
        // reads its own first.
        let mut moved = "rv";
        if dpp && name != "v_movreld_b32" {
            self.line(&format!(
                "uint movrel_src{pc} = 0u; switch ({src_index} + {}u) {{ {VREAD}movrel_src{pc}{VEND} }}",
                need_base()?
            ));
            self.dpp_src0(pc, inst, v, &format!("movrel_src{pc}"))?;
            moved = "dpp0";
        }
        let stmt = match name {
            "v_movrels_b32" if dpp => {
                let d = self.v(dst);
                format!("{d} = dpp0;")
            }
            "v_movrels_b32" => {
                let d = self.v(dst);
                format!("{{ {} {d} = rv; }}", read("rv", need_base()?, &src_index))
            }
            "v_movrelsd_b32" | "v_movrelsd_2_b32" if dpp => {
                format!("{{ {} }}", write(dst, &dst_index, moved))
            }
            "v_movreld_b32" => {
                let s = self.usrc(pc, inst, v, 0)?;
                format!("{{ uint wv = {s}; {} }}", write(dst, &dst_index, "wv"))
            }
            "v_movrelsd_b32" | "v_movrelsd_2_b32" => {
                let b = need_base()?;
                format!(
                    "{{ {} {} }}",
                    read("rv", b, &src_index),
                    write(dst, &dst_index, "rv")
                )
            }
            "v_swaprel_b32" => {
                let b = need_base()?;
                format!(
                    "{{ {} {} {} {} }}",
                    read("rv", b, &src_index),
                    read("rw", dst, &dst_index),
                    write(dst, &dst_index, "rv"),
                    write(b, &src_index, "rw")
                )
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        self.guard(&stmt);
        Ok(())
    }
}
