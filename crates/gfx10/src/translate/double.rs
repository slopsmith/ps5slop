//! f64 VALU instructions, through the helpers of `double_prelude.metal`,
//! which compute IEEE binary64 on the bits in 64-bit integers.

use super::*;

/// f64 ops with an f64 result and their helper; their sources are all f64.
const F64_OPS: &[(&str, &str)] = &[
    ("v_add_f64", "add64"),
    ("v_mul_f64", "mul64"),
    ("v_fma_f64", "fma64"),
    ("v_min_f64", "min64"),
    ("v_max_f64", "max64"),
    ("v_div_fixup_f64", "div_fixup64"),
    ("v_trunc_f64", "trunc64"),
    ("v_ceil_f64", "ceil64"),
    ("v_floor_f64", "floor64"),
    ("v_rndne_f64", "rndne64"),
    ("v_fract_f64", "fract64"),
    ("v_frexp_mant_f64", "frexp_mant64"),
    ("v_rcp_f64", "rcp64"),
    ("v_rsq_f64", "rsq64"),
    ("v_sqrt_f64", "sqrt64"),
];

/// Whether `name` is a non-compare VALU instruction with an f64 operand or
/// result.
pub(super) fn is_float64(name: &str) -> bool {
    name.starts_with("v_") && name.contains("f64") && !name.starts_with("v_cmp")
}

impl<'a> Tx<'a> {
    /// Source `i` as f64 bits with its abs and neg modifiers.
    fn dbits(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        self.floats = true;
        let code = v.src[i];
        // A 32-bit literal is the high half of an f64 operand.
        let mut e = match code {
            src::LITERAL => format!(
                "0x{:x}ul",
                u64::from(inst.literal.ok_or_else(|| err(pc, "missing literal"))?) << 32
            ),
            256.. => {
                let (lo, hi) = (self.v(code - 256), self.v(code - 255));
                format!("(ulong({hi}) << 32 | ulong({lo}))")
            }
            _ => self.src64(code, inst.literal, pc)?,
        };
        if v.abs >> i & 1 != 0 {
            e = format!("({e} & 0x7ffffffffffffffful)");
        }
        if v.neg >> i & 1 != 0 {
            e = format!("({e} ^ 0x8000000000000000ul)");
        }
        Ok(e)
    }

    /// An f64 result with the instruction's output modifier and clamp.
    fn mods64(&self, v: &Valu, e: String) -> String {
        let e = match v.omod {
            1 => format!("omod64({e}, 1)"),
            2 => format!("omod64({e}, 2)"),
            3 => format!("omod64({e}, -1)"),
            _ => e,
        };
        if v.clamp { format!("clamp64({e})") } else { e }
    }

    fn write64(&mut self, v: &Valu, e: &str) {
        let d = v.vdst as u16;
        let (lo, hi) = (self.v(d), self.v(d + 1));
        self.guard(&format!(
            "ulong r = {e}; {lo} = uint(r); {hi} = uint(r >> 32);"
        ));
    }

    /// Non-compare f64 instructions.
    pub(super) fn float64(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
    ) -> Result<(), TranslateError> {
        let name = inst.op.name;
        if !matches!(v.enc, VEnc::E32 | VEnc::E64) {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        self.floats = true;
        let d = |t: &mut Self, i: usize| t.dbits(pc, inst, v, i);
        if let Some((_, helper)) = F64_OPS.iter().find(|(o, _)| *o == name) {
            let mut args = Vec::new();
            for i in 0..inst.op.srcs.len().min(3) {
                args.push(d(self, i)?);
            }
            let e = self.mods64(v, format!("{helper}({})", args.join(", ")));
            self.write64(v, &e);
            return Ok(());
        }
        let wide = match name {
            "v_ldexp_f64" => {
                let a = d(self, 0)?;
                format!("ldexp64({a}, int({}))", self.usrc(pc, inst, v, 1)?)
            }
            "v_trig_preop_f64" => {
                let a = d(self, 0)?;
                format!("trig_preop64({a}, {})", self.usrc(pc, inst, v, 1)?)
            }
            "v_cvt_f64_i32" => format!("cvt_f64_i32(int({}))", self.usrc(pc, inst, v, 0)?),
            "v_cvt_f64_u32" => format!("cvt_f64_u32({})", self.usrc(pc, inst, v, 0)?),
            "v_cvt_f64_f32" => format!("cvt_f64_f32({})", self.fbits(pc, inst, v, 0)?),
            "v_div_fmas_f64" => {
                let (a, b, c) = (d(self, 0)?, d(self, 1)?, d(self, 2)?);
                let vcc = self.vcc_bit();
                format!("div_fmas64({a}, {b}, {c}, {vcc} != 0u)")
            }
            "v_div_scale_f64" => {
                let (a, b, c) = (d(self, 0)?, d(self, 1)?, d(self, 2)?);
                let dst = v.sdst.map_or(106, |s| s as u16);
                let set = if dst == src::NULL {
                    String::new()
                } else {
                    self.set_mask(dst, "k")
                };
                let exec = self.mask_on(126);
                let (lo, hi) = (self.v(v.vdst as u16), self.v(v.vdst as u16 + 1));
                let r = self.mods64(v, "r".into());
                self.line(&format!(
                    "{{ bool k = false; ulong r = div_scale64({a}, {b}, {c}, k); bool on = {exec}; k = k && on; if (on) {{ r = {r}; {lo} = uint(r); {hi} = uint(r >> 32); }}{set} }}"
                ));
                return Ok(());
            }
            _ => String::new(),
        };
        if !wide.is_empty() {
            let e = self.mods64(v, wide);
            self.write64(v, &e);
            return Ok(());
        }
        let narrow = match name {
            "v_cvt_i32_f64" => format!("cvt_i32_f64({})", d(self, 0)?),
            "v_cvt_u32_f64" => format!("cvt_u32_f64({})", d(self, 0)?),
            "v_frexp_exp_i32_f64" => format!("frexp_exp64({})", d(self, 0)?),
            "v_cvt_f32_f64" => {
                let a = d(self, 0)?;
                self.mods32(v, format!("cvt_f32_f64({a})"))
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let dst = self.v(v.vdst as u16);
        self.guard(&format!("{dst} = {narrow};"));
        Ok(())
    }

    /// The condition of a VOPC f64 compare.
    pub(super) fn fcompare64(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        op: &str,
    ) -> Result<String, TranslateError> {
        let name = inst.op.name;
        if !matches!(v.enc, VEnc::E32 | VEnc::E64) {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        let a = self.dbits(pc, inst, v, 0)?;
        if op == "class" {
            return Ok(format!("class64({a}, {})", self.usrc(pc, inst, v, 1)?));
        }
        let b = self.dbits(pc, inst, v, 1)?;
        let code = float::cmp_code(op)
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        Ok(format!("cmp64({a}, {b}, {code}u)"))
    }
}
