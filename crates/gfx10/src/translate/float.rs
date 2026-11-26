//! Float VALU instructions (f32 and f16, scalar and packed) under the
//! program's float mode, through the helpers of `float_prelude.metal`.

use super::*;
use crate::opinfo::Ty;

pub(super) const FLOAT_PRELUDE: &str = include_str!("float_prelude.metal");

/// The `#define`s `FLOAT_PRELUDE` reads, from a `PGM_RSRC1`.
pub(super) fn mode_defines(rsrc1: u32) -> String {
    let mode = rsrc1 >> 12 & 0xFF;
    format!(
        "#define FM_RND32 {}u\n#define FM_RND16 {}u\n#define FM_DN32 {}u\n#define FM_DN16 {}u\n#define FM_DX10 {}\n#define FM_IEEE {}\n",
        mode & 3,
        mode >> 2 & 3,
        mode >> 4 & 3,
        mode >> 6 & 3,
        rsrc1 >> 21 & 1,
        rsrc1 >> 23 & 1,
    )
}

/// Float ops with an f32 result and their helper, by operand count.
const F32_OPS: &[(&str, &str)] = &[
    ("v_add_f32", "add32"),
    ("v_sub_f32", "sub32"),
    ("v_mul_f32", "mul32"),
    ("v_mul_legacy_f32", "mul_legacy32"),
    ("v_max_f32", "max32"),
    ("v_min_f32", "min32"),
    ("v_fma_f32", "fma32"),
    ("v_mad_f32", "mad32"),
    ("v_mad_legacy_f32", "mad_legacy32"),
    ("v_med3_f32", "med3_32"),
    ("v_rcp_f32", "rcp32"),
    ("v_rcp_iflag_f32", "rcp32"),
    ("v_rsq_f32", "rsq32"),
    ("v_sqrt_f32", "sqrt32"),
    ("v_sin_f32", "sin32"),
    ("v_cos_f32", "cos32"),
    ("v_exp_f32", "exp32"),
    ("v_log_f32", "log32"),
    ("v_fract_f32", "fract32"),
    ("v_floor_f32", "floor32"),
    ("v_ceil_f32", "ceil32"),
    ("v_trunc_f32", "trunc32"),
    ("v_rndne_f32", "rndne32"),
    ("v_cubeid_f32", "cubeid32"),
    ("v_cubesc_f32", "cubesc32"),
    ("v_cubetc_f32", "cubetc32"),
    ("v_cubema_f32", "cubema32"),
];

/// Float ops with an f16 result and their helper.
const F16_OPS: &[(&str, &str)] = &[
    ("v_add_f16", "add16"),
    ("v_sub_f16", "sub16"),
    ("v_mul_f16", "mul16"),
    ("v_max_f16", "max16"),
    ("v_min_f16", "min16"),
    ("v_fma_f16", "fma16"),
    ("v_mad_f16", "mad16"),
    ("v_med3_f16", "med3_16"),
    ("v_rcp_f16", "rcp16"),
    ("v_sqrt_f16", "sqrt16"),
    ("v_rsq_f16", "rsq16"),
    ("v_log_f16", "log16"),
    ("v_exp_f16", "exp16"),
    ("v_sin_f16", "sin16"),
    ("v_cos_f16", "cos16"),
    ("v_floor_f16", "floor16"),
    ("v_ceil_f16", "ceil16"),
    ("v_trunc_f16", "trunc16"),
    ("v_rndne_f16", "rndne16"),
    ("v_fract_f16", "fract16"),
];

/// f16 ops with a 16-bit integer result and their helper.
const F16_TO_INT: &[(&str, &str)] = &[
    ("v_frexp_exp_i16_f16", "frexp_exp16"),
    ("v_cvt_i16_f16", "cvt_i16_f16"),
    ("v_cvt_u16_f16", "cvt_u16_f16"),
    ("v_cvt_norm_i16_f16", "cvt_snorm16_f16"),
    ("v_cvt_norm_u16_f16", "cvt_unorm16_f16"),
];

pub(super) fn is_float16(name: &str) -> bool {
    F16_OPS.iter().chain(F16_TO_INT).any(|(n, _)| *n == name)
        || matches!(
            name,
            "v_subrev_f16"
                | "v_fmac_f16"
                | "v_min3_f16"
                | "v_max3_f16"
                | "v_ldexp_f16"
                | "v_frexp_mant_f16"
                | "v_div_fixup_f16"
                | "v_fmaak_f16"
                | "v_fmamk_f16"
                | "v_cvt_f16_i16"
                | "v_cvt_f16_u16"
                | "v_cvt_pknorm_i16_f16"
                | "v_cvt_pknorm_u16_f16"
                | "v_pack_b32_f16"
        )
}

impl<'a> Tx<'a> {
    /// Source `i` as f32 bits with its abs and neg modifiers.
    pub(super) fn fbits(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        self.floats = true;
        let mut e = self.usrc(pc, inst, v, i)?;
        if v.abs >> i & 1 != 0 {
            e = format!("({e} & 0x7fffffffu)");
        }
        if v.neg >> i & 1 != 0 {
            e = format!("({e} ^ 0x80000000u)");
        }
        Ok(e)
    }

    /// Source `i` as f16 bits (low half) with its abs and neg modifiers.
    pub(super) fn hbits(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        self.floats = true;
        let mut e = format!("({} & 0xffffu)", self.src16(pc, inst, v, i)?);

        if v.abs >> i & 1 != 0 {
            e = format!("({e} & 0x7fffu)");
        }
        if v.neg >> i & 1 != 0 {
            e = format!("({e} ^ 0x8000u)");
        }
        Ok(e)
    }

    /// An f32 result with the instruction's output modifier and clamp.
    fn mods32(&self, v: &Valu, e: String) -> String {
        let e = match v.omod {
            1 => format!("omod32({e}, 2.0f)"),
            2 => format!("omod32({e}, 4.0f)"),
            3 => format!("omod32({e}, 0.5f)"),
            _ => e,
        };
        if v.clamp { format!("clamp32({e})") } else { e }
    }

    fn mods16(&self, v: &Valu, e: String) -> String {
        let e = match v.omod {
            1 => format!("omod16({e}, 2.0f)"),
            2 => format!("omod16({e}, 4.0f)"),
            3 => format!("omod16({e}, 0.5f)"),
            _ => e,
        };
        if v.clamp { format!("clamp16({e})") } else { e }
    }

    /// Translates `inst` if it is a float instruction with a 32-bit result
    /// (f32, or an integer from an f32); `false` if it is not one.
    pub(super) fn float32(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<bool, TranslateError> {
        let name = inst.op.name;
        let f = |t: &mut Self, i: usize| t.fbits(pc, inst, v, i);
        let u = |t: &mut Self, i: usize| t.usrc(pc, inst, v, i);
        let vdst = v.vdst as u16;
        let k = format!("0x{:x}u", inst.literal.unwrap_or(0));
        let call = |helper: &str, args: &[String]| format!("{helper}({})", args.join(", "));
        let n = inst.op.srcs.len().min(3);
        let float = if let Some((_, helper)) = F32_OPS.iter().find(|(o, _)| *o == name) {
            let mut args = Vec::new();
            for i in 0..n {
                args.push(f(self, i)?);
            }
            call(helper, &args)
        } else {
            match name {
                "v_subrev_f32" => format!("add32({} ^ 0x80000000u, {})", f(self, 0)?, f(self, 1)?),
                "v_min3_f32" => {
                    let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                    format!("min32(min32({a}, {b}), {c})")
                }
                "v_max3_f32" => {
                    let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                    format!("max32(max32({a}, {b}), {c})")
                }
                "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                    let (a, b) = (f(self, 0)?, f(self, 1)?);
                    let d = self.v(vdst);
                    let helper = match name {
                        "v_fmac_f32" => "fma32",
                        "v_mac_f32" => "mad32",
                        _ => "mad_legacy32",
                    };
                    format!("{helper}({a}, {b}, {d})")
                }
                "v_madmk_f32" => format!("mad32({}, {k}, {})", f(self, 0)?, f(self, 1)?),
                "v_madak_f32" => format!("mad32({}, {}, {k})", f(self, 0)?, f(self, 1)?),
                "v_fmamk_f32" => format!("fma32({}, {k}, {})", f(self, 0)?, f(self, 1)?),
                "v_fmaak_f32" => format!("fma32({}, {}, {k})", f(self, 0)?, f(self, 1)?),
                "v_ldexp_f32" => format!("ldexp32({}, int({}))", f(self, 0)?, u(self, 1)?),
                "v_cvt_f32_i32" => format!("cvt_f32_i32(int({}))", u(self, 0)?),
                "v_cvt_f32_u32" => format!("cvt_f32_u32({})", u(self, 0)?),
                "v_cvt_f32_ubyte0" => format!("cvt_f32_u32({} & 0xffu)", u(self, 0)?),
                "v_cvt_f32_ubyte1" => format!("cvt_f32_u32(({} >> 8) & 0xffu)", u(self, 0)?),
                "v_cvt_f32_ubyte2" => format!("cvt_f32_u32(({} >> 16) & 0xffu)", u(self, 0)?),
                "v_cvt_f32_ubyte3" => format!("cvt_f32_u32({} >> 24)", u(self, 0)?),
                "v_cvt_off_f32_i4" => {
                    format!("U(float(int({} << 28) >> 28) / 16.0f)", u(self, 0)?)
                }
                "v_cvt_f32_f16" => format!("cvt_f32_f16({})", self.hbits(pc, inst, v, 0)?),
                "v_frexp_mant_f32" => format!("frexp_mant32({})", f(self, 0)?),
                "v_mullit_f32" => {
                    let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                    format!("mullit32({a}, {b}, {c})")
                }

                "v_div_fixup_f32" => {
                    let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                    format!("div_fixup32({a}, {b}, {c})")
                }
                "v_div_fmas_f32" => {
                    let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                    let vcc = self.vcc_bit();
                    format!("div_fmas32({a}, {b}, {c}, {vcc} != 0u)")
                }
                "v_div_scale_f32" => {
                    let (a, b, c) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                    let d = self.v(vdst);
                    let dst = v.sdst.map_or(106, |s| s as u16);
                    let set = if dst == src::NULL {
                        String::new()
                    } else {
                        self.set_mask(dst, "k")
                    };
                    let exec = self.mask_on(126);
                    self.line(&format!(
                        "{{ bool k = false; uint r = div_scale32({a}, {b}, {c}, k); bool on = {exec}; k = k && on; if (on) {d} = {};{set} }}",
                        self.mods32(v, "r".into())
                    ));
                    return Ok(true);
                }
                _ => String::new(),
            }
        };
        let value = if float.is_empty() {
            match name {
                "v_cvt_i32_f32" => format!("cvt_i32_f32({})", f(self, 0)?),
                "v_frexp_exp_i32_f32" => format!("frexp_exp32({})", f(self, 0)?),
                "v_cvt_pk_u8_f32" => format!(
                    "cvt_pk_u8_f32({}, {}, {})",
                    f(self, 0)?,
                    u(self, 1)?,
                    u(self, 2)?
                ),

                "v_cvt_u32_f32" => format!("cvt_u32_f32({})", f(self, 0)?),
                "v_cvt_flr_i32_f32" => format!("cvt_flr_i32_f32({})", f(self, 0)?),
                "v_cvt_rpi_i32_f32" => format!("cvt_rpi_i32_f32({})", f(self, 0)?),
                "v_cvt_pkrtz_f16_f32" => format!(
                    "cvt_f16_f32({}, 3u) | cvt_f16_f32({}, 3u) << 16",
                    f(self, 0)?,
                    f(self, 1)?
                ),
                "v_cvt_pknorm_i16_f32" => format!(
                    "cvt_snorm16({}) | cvt_snorm16({}) << 16",
                    f(self, 0)?,
                    f(self, 1)?
                ),
                "v_cvt_pknorm_u16_f32" => format!(
                    "cvt_unorm16({}) | cvt_unorm16({}) << 16",
                    f(self, 0)?,
                    f(self, 1)?
                ),
                "v_cvt_f16_f32" => {
                    let e = format!("cvt_f16_f32({}, FM_RND16)", f(self, 0)?);
                    let e = self.mods16(v, e);
                    self.write16(v, &e);
                    return Ok(true);
                }
                _ => return Ok(false),
            }
        } else {
            self.mods32(v, float)
        };
        self.floats = true;
        self.regs.insert(format!("v{vdst}"));
        let d = self.v(vdst);
        self.guard(&format!("{d} = {value};"));
        Ok(true)
    }

    /// Scalar f16 float instructions.
    pub(super) fn float16(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        self.floats = true;
        let name = inst.op.name;
        let n = inst.op.srcs.len().min(3);
        let mut s = Vec::new();

        for i in 0..n {
            s.push(match inst.op.srcs[i] {
                Ty::F16 | Ty::PkF16 => self.hbits(pc, inst, v, i)?,
                _ => String::new(),
            });
        }
        let k = format!("0x{:x}u", inst.literal.unwrap_or(0) & 0xFFFF);
        if let Some((_, helper)) = F16_TO_INT.iter().find(|(o, _)| *o == name) {
            let e = format!("{helper}({})", s[0]);
            self.write16(v, &e);
            return Ok(());
        }
        let dword = match name {
            "v_cvt_pknorm_i16_f16" => Some(format!(
                "cvt_snorm16_f16({}) | cvt_snorm16_f16({}) << 16",
                s[0], s[1]
            )),
            "v_cvt_pknorm_u16_f16" => Some(format!(
                "cvt_unorm16_f16({}) | cvt_unorm16_f16({}) << 16",
                s[0], s[1]
            )),
            "v_pack_b32_f16" => Some(format!(
                "{} | {} << 16",
                self.mods16(v, format!("pack16({})", s[0])),
                self.mods16(v, format!("pack16({})", s[1]))
            )),

            _ => None,
        };
        if let Some(e) = dword {
            let d = self.v(v.vdst as u16);
            self.guard(&format!("{d} = {e};"));
            return Ok(());
        }
        let e = if let Some((_, helper)) = F16_OPS.iter().find(|(o, _)| *o == name) {
            format!("{helper}({})", s.join(", "))
        } else {
            match name {
                "v_subrev_f16" => format!("add16({} ^ 0x8000u, {})", s[0], s[1]),

                "v_fmac_f16" => {
                    let d = self.v(v.vdst as u16);
                    format!("fma16({}, {}, {d})", s[0], s[1])
                }
                "v_min3_f16" => format!("min16(min16({}, {}), {})", s[0], s[1], s[2]),
                "v_max3_f16" => format!("max16(max16({}, {}), {})", s[0], s[1], s[2]),
                "v_ldexp_f16" => {
                    let b = self.src16(pc, inst, v, 1)?;
                    format!("ldexp16({}, int(short({b})))", s[0])
                }
                "v_frexp_mant_f16" => format!("frexp_mant16({})", s[0]),
                "v_div_fixup_f16" => format!("div_fixup16({}, {}, {})", s[0], s[1], s[2]),
                "v_fmaak_f16" => format!("fma16({}, {}, {k})", s[0], s[1]),
                "v_fmamk_f16" => format!("fma16({}, {k}, {})", s[0], s[1]),
                "v_cvt_f16_i16" | "v_cvt_f16_u16" => {
                    let a = self.src16(pc, inst, v, 0)?;
                    let x = if name == "v_cvt_f16_i16" {
                        format!("float(short({a}))")
                    } else {
                        format!("float({a} & 0xffffu)")
                    };
                    format!("h16({x}, 0.0f, FM_RND16, (FM_DN16 & 2) != 0)")
                }
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            }
        };
        let e = self.mods16(v, e);
        self.write16(v, &e);
        Ok(())
    }

    /// A VOPC float compare's condition, or `None` for other types.
    pub(super) fn fcompare(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        op: &str,
        ty: &str,
    ) -> Result<Option<String>, TranslateError> {
        let name = inst.op.name;
        let (a, b, bits) = match ty {
            "f32" => (self.fbits(pc, inst, v, 0)?, self.usrc(pc, inst, v, 1)?, 32),
            "f16" => (self.hbits(pc, inst, v, 0)?, self.src16(pc, inst, v, 1)?, 16),
            _ => return Ok(None),
        };
        if op == "class" {
            return Ok(Some(format!("class{bits}({a}, {b})")));
        }
        let b = match ty {
            "f32" => self.fbits(pc, inst, v, 1)?,
            _ => self.hbits(pc, inst, v, 1)?,
        };
        let code = match op {
            "f" => 0,
            "lt" => 1,
            "eq" => 2,
            "le" => 3,
            "gt" => 4,
            "lg" => 5,
            "ge" => 6,
            "o" => 7,
            "u" => 8,
            "nge" => 9,
            "nlg" => 10,
            "ngt" => 11,
            "nle" => 12,
            "neq" => 13,
            "nlt" => 14,
            "t" | "tru" => 15,

            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        Ok(Some(format!("cmp{bits}({a}, {b}, {code}u)")))
    }
}
