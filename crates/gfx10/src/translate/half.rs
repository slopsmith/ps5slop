//! 16-bit and packed half-precision VALU ops, mixed FMA and indexed VGPR moves.

use super::*;

/// The float inline constants 240 to 248 as halves.
const F16_CONSTANTS: [u32; 9] = [
    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
];
pub(super) const HALF_OPS: &[&str] = &[
    "v_add_f16",
    "v_sub_f16",
    "v_subrev_f16",
    "v_mul_f16",
    "v_max_f16",
    "v_min_f16",
    "v_fma_f16",
    "v_fmac_f16",
    "v_mad_f16",
    "v_max3_f16",
    "v_min3_f16",
    "v_med3_f16",
    "v_rcp_f16",
    "v_sqrt_f16",
    "v_rsq_f16",
    "v_floor_f16",
    "v_ceil_f16",
    "v_trunc_f16",
    "v_rndne_f16",
    "v_fract_f16",
    "v_exp_f16",
    "v_log_f16",
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
                let half = if v.opsel >> i & 1 != 0 { "y" } else { "x" };
                format!("float(as_type<half2>({u}).{half})")
            } else {
                format!("F({u})")
            };
            if v.neg_hi >> i & 1 != 0 {
                e = format!("fabs({e})");
            }
            if v.neg >> i & 1 != 0 {
                e = format!("(-{e})");
            }
            srcs.push(e);
        }
        let mut r = format!("fma({}, {}, {})", srcs[0], srcs[1], srcs[2]);
        if v.clamp {
            r = format!("saturate({r})");
        }
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        let stmt = match inst.op.name {
            "v_fma_mix_f32" => format!("{d} = U({r});"),
            "v_fma_mixlo_f16" => {
                format!("{d} = ({d} & 0xffff0000u) | uint(as_type<ushort>(half({r})));")
            }
            _ => format!("{d} = ({d} & 0xffffu) | (uint(as_type<ushort>(half({r}))) << 16);"),
        };
        self.guard(&stmt);
        Ok(())
    }

    /// Packed 16-bit integer adds and subtracts: `op_sel` picks each source's
    /// half for the low result, `op_sel_hi` for the high one; wrapping.
    pub(super) fn pk_add16(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        if v.clamp {
            return Err(err(pc, format!("unsupported form of {}", inst.op.name)));
        }
        let half = |u: &str, hi: bool| {
            if hi {
                format!("({u} >> 16)")
            } else {
                format!("({u} & 0xffffu)")
            }
        };
        let (a, b) = (self.usrc(pc, inst, v, 0)?, self.usrc(pc, inst, v, 1)?);
        let op = if inst.op.name.starts_with("v_pk_add") {
            "+"
        } else {
            "-"
        };
        let lo = format!(
            "(({} {op} {}) & 0xffffu)",
            half(&a, v.opsel & 1 != 0),
            half(&b, v.opsel & 2 != 0)
        );
        let hi = format!(
            "(({} {op} {}) << 16)",
            half(&a, v.opsel_hi & 1 != 0),
            half(&b, v.opsel_hi & 2 != 0)
        );
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        self.guard(&format!("{d} = {lo} | {hi};"));
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

    pub(super) fn hsrc(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        let mut e = format!("as_type<half>(ushort({}))", self.src16(pc, inst, v, i)?);
        if v.abs >> i & 1 != 0 {
            e = format!("abs({e})");
        }
        if v.neg >> i & 1 != 0 {
            e = format!("(-{e})");
        }
        Ok(e)
    }

    /// Writes a 16-bit result to the low half of the destination, or with
    /// VOP3 `op_sel[3]` to the high half; the other half is kept (GFX10).
    fn write16(&mut self, v: &Valu, r: &str) {
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        if v.enc == VEnc::E64 && v.opsel & 8 != 0 {
            self.guard(&format!("{d} = ({d} & 0xffffu) | (({r} & 0xffffu) << 16);"));
        } else {
            self.guard(&format!("{d} = ({d} & 0xffff0000u) | ({r} & 0xffffu);"));
        }
    }

    /// Scalar 16-bit float and integer operations.
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
        let n = inst.op.srcs.len().min(3);
        if name.ends_with("_f16") {
            let mut s = Vec::new();
            for i in 0..n {
                s.push(self.hsrc(pc, inst, v, i)?);
            }
            if name == "v_fmac_f16" {
                let d = self.v(v.vdst as u16);
                s.push(format!("as_type<half>(ushort({d}))"));
            }
            let r = match name {
                "v_add_f16" => format!("{} + {}", s[0], s[1]),
                "v_sub_f16" => format!("{} - {}", s[0], s[1]),
                "v_subrev_f16" => format!("{} - {}", s[1], s[0]),
                "v_mul_f16" => format!("{} * {}", s[0], s[1]),
                "v_max_f16" => format!("fmax({}, {})", s[0], s[1]),
                "v_min_f16" => format!("fmin({}, {})", s[0], s[1]),
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" => {
                    format!("fma({}, {}, {})", s[0], s[1], s[2])
                }
                "v_max3_f16" => format!("fmax(fmax({}, {}), {})", s[0], s[1], s[2]),
                "v_min3_f16" => format!("fmin(fmin({}, {}), {})", s[0], s[1], s[2]),
                "v_med3_f16" => format!(
                    "fmax(fmin({a}, {b}), fmin(fmax({a}, {b}), {c}))",
                    a = s[0],
                    b = s[1],
                    c = s[2]
                ),
                "v_rcp_f16" => format!("1.0h / {}", s[0]),
                "v_sqrt_f16" => format!("sqrt({})", s[0]),
                "v_rsq_f16" => format!("rsqrt({})", s[0]),
                "v_floor_f16" => format!("floor({})", s[0]),
                "v_ceil_f16" => format!("ceil({})", s[0]),
                "v_trunc_f16" => format!("trunc({})", s[0]),
                "v_rndne_f16" => format!("rint({})", s[0]),
                "v_fract_f16" => format!("fract({})", s[0]),
                "v_exp_f16" => format!("exp2({})", s[0]),
                "v_log_f16" => format!("log2({})", s[0]),
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            let mut r = format!("half({r})");
            match v.omod {
                1 => r = format!("({r} * 2.0h)"),
                2 => r = format!("({r} * 4.0h)"),
                3 => r = format!("({r} * 0.5h)"),
                _ => {}
            }
            if v.clamp {
                r = format!("saturate({r})");
            }
            self.write16(v, &format!("uint(as_type<ushort>({r}))"));
            return Ok(());
        }
        let a = self.src16(pc, inst, v, 0)?;
        let b = self.src16(pc, inst, v, 1)?;
        let (ua, ub) = (format!("({a} & 0xffffu)"), format!("({b} & 0xffffu)"));
        let (ia, ib) = (format!("int(short({a}))"), format!("int(short({b}))"));
        let r = match name {
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
        if v.clamp {
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
        let n = inst.op.srcs.len().min(3);
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for i in 0..n {
            let code = v.src[i];
            let u = if (240..=248).contains(&code) {
                let c = F16_CONSTANTS[code as usize - 240];
                format!("0x{:x}u", c | c << 16)
            } else {
                self.usrc(pc, inst, v, i)?
            };
            for (sel, neg, out) in [
                (opsel >> i & 1, v.neg >> i & 1, &mut lo),
                (opsel_hi >> i & 1, v.neg_hi >> i & 1, &mut hi),
            ] {
                let shift = if sel != 0 { " >> 16" } else { "" };
                let mut e = format!("as_type<half>(ushort({u}{shift}))");
                if neg != 0 {
                    e = format!("(-{e})");
                }
                out.push(e);
            }
        }
        if name == "v_pk_fmac_f16" {
            let d = self.v(v.vdst as u16);
            lo.push(format!("as_type<half>(ushort({d}))"));
            hi.push(format!("as_type<half>(ushort({d} >> 16))"));
        }
        let f = |s: &[String]| -> Option<String> {
            Some(match name {
                "v_pk_add_f16" => format!("{} + {}", s[0], s[1]),
                "v_pk_mul_f16" => format!("{} * {}", s[0], s[1]),
                "v_pk_max_f16" => format!("fmax({}, {})", s[0], s[1]),
                "v_pk_min_f16" => format!("fmin({}, {})", s[0], s[1]),
                "v_pk_fma_f16" | "v_pk_fmac_f16" => format!("fma({}, {}, {})", s[0], s[1], s[2]),
                _ => return None,
            })
        };
        let (Some(l), Some(h)) = (f(&lo), f(&hi)) else {
            return Err(err(pc, format!("unsupported instruction {name}")));
        };
        let (l, h) = if v.clamp {
            (
                format!("saturate(half({l}))"),
                format!("saturate(half({h}))"),
            )
        } else {
            (format!("half({l})"), format!("half({h})"))
        };
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        self.guard(&format!("{d} = as_type<uint>(half2({l}, {h}));"));
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
        if !matches!(v.enc, VEnc::E32 | VEnc::E64)
            || (v.enc == VEnc::E64 && (v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0))
        {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
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
        let stmt = match name {
            "v_movrels_b32" => {
                let d = self.v(dst);
                format!("{{ {} {d} = rv; }}", read("rv", need_base()?, &src_index))
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
