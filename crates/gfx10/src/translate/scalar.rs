//! Scalar ALU (`SOP1`, `SOP2`, `SOPC`, `SOPK`) and scalar memory (`SMEM`).

use super::*;

/// Hardware registers `s_getreg_b32` reads (ISA guide, 5.8) and their fields.
pub(super) mod hwreg {
    pub const MODE: u16 = 1;
    pub const STATUS: u16 = 2;
    pub const TRAPSTS: u16 = 3;
    pub const GPR_ALLOC: u16 = 5;
    pub const LDS_ALLOC: u16 = 6;
    pub const STATUS_EXECZ: u32 = 1 << 9;
    pub const STATUS_VCCZ: u32 = 1 << 10;
    pub const STATUS_IN_WG: u32 = 1 << 11;
    pub const STATUS_VALID: u32 = 1 << 16;
    /// GFX10 gives every wave 128 SGPRs (measured on gfx1033).
    pub const GPR_ALLOC_SGPRS: u32 = 0xF << 24;
}

/// The operands of a scalar memory instruction.
pub(super) struct Smem {
    pub sdata: u16,
    pub sbase: u16,
    pub offset: i32,
    pub soffset: u16,
    pub glc: bool,
}

impl<'a> Tx<'a> {
    pub(super) fn sop1(
        &mut self,
        pc: usize,
        name: &str,
        sdst: u16,
        s0: u16,
        lit: Option<u32>,
    ) -> Result<(), TranslateError> {
        let b64 = name.ends_with("_b64");
        if name.starts_with("s_movrel") {
            return self.sgpr_movrel(pc, name, sdst, s0, lit);
        }
        if name == "s_cmov_b32" {
            let a = self.src32(s0, lit, pc)?;
            self.regs.insert("scc".into());
            if let Some(d) = self.sdst(sdst) {
                self.line(&format!("if (scc) {d} = {a};"));
            }
            return Ok(());
        }
        if name == "s_cmov_b64" {
            let a = self.src64(s0, lit, pc)?;
            self.regs.insert("scc".into());
            self.line("if (scc) {");
            self.set64(sdst, &a);
            self.line("}");
            return Ok(());
        }
        // `wrexec` writes the new EXEC to the destination, `saveexec` the old one.
        let wrexec = name.contains("_wrexec_");
        if name.ends_with("saveexec_b32") || name.ends_with("wrexec_b32") {
            let a = self.src32(s0, lit, pc)?;
            let op = name
                .trim_start_matches("s_")
                .trim_end_matches("_saveexec_b32")
                .trim_end_matches("_wrexec_b32");
            let new = match op {
                "and" => format!("{a} & old"),
                "or" => format!("{a} | old"),
                "xor" => format!("{a} ^ old"),
                "andn2" => format!("{a} & ~old"),
                "orn2" => format!("{a} | ~old"),
                "nand" => format!("~({a} & old)"),
                "nor" => format!("~({a} | old)"),
                "xnor" => format!("~({a} ^ old)"),
                "andn1" => format!("~{a} & old"),
                "orn1" => format!("~{a} | old"),
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            self.regs.insert("scc".into());
            let lo = self.s(126);
            self.line(&format!(
                "{{ uint old = {lo}; uint n = {new}; scc = n != 0u; {lo} = n;"
            ));
            if let Some(d) = self.sdst(sdst) {
                let v = if wrexec { "n" } else { "old" };
                self.line(&format!("{d} = {v};"));
            }
            self.line("}");
            return Ok(());
        }
        if name.ends_with("saveexec_b64") || name.ends_with("wrexec_b64") {
            let a = self.src64(s0, lit, pc)?;
            let exec = self.src64(126, None, pc)?;
            let op = name
                .trim_start_matches("s_")
                .trim_end_matches("_saveexec_b64")
                .trim_end_matches("_wrexec_b64");
            let new = match op {
                "and" => format!("{a} & old"),
                "or" => format!("{a} | old"),
                "xor" => format!("{a} ^ old"),
                "andn2" => format!("{a} & ~old"),
                "orn2" => format!("{a} | ~old"),
                "nand" => format!("~({a} & old)"),
                "nor" => format!("~({a} | old)"),
                "xnor" => format!("~({a} ^ old)"),
                "andn1" => format!("~{a} & old"),
                "orn1" => format!("~{a} | old"),
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            self.regs.insert("scc".into());
            let (lo, hi) = (self.s(126), self.s(127));
            self.line(&format!("{{ ulong old = {exec}; ulong n = {new}; scc = n != 0ul; {lo} = uint(n); {hi} = uint(n >> 32);"));
            self.set64(sdst, if wrexec { "n" } else { "old" });
            self.line("}");
            return Ok(());
        }
        if name == "s_bitset0_b32" || name == "s_bitset1_b32" {
            let a = self.src32(s0, lit, pc)?;
            let Some(d) = self.sdst(sdst) else {
                return Ok(());
            };
            let e = if name == "s_bitset1_b32" {
                format!("{d} | (1u << ({a} & 31u))")
            } else {
                format!("{d} & ~(1u << ({a} & 31u))")
            };
            self.line(&format!("{d} = {e};"));
            return Ok(());
        }
        if name == "s_bitset0_b64" || name == "s_bitset1_b64" {
            let a = self.src32(s0, lit, pc)?;
            let d = self.src64(sdst, None, pc)?;
            let e = if name == "s_bitset1_b64" {
                format!("{d} | (1ul << ({a} & 63u))")
            } else {
                format!("{d} & ~(1ul << ({a} & 63u))")
            };
            self.set64(sdst, &e);
            return Ok(());
        }
        // The called code follows the call, inlined (`calls.rs`).
        if matches!(name, "s_getpc_b64" | "s_swappc_b64") {
            let next = calls::Expanded::address(&self.runs, pc) + 4;
            self.set64(sdst, &format!("{next:#x}ul"));
            return Ok(());
        }
        if name == "s_setpc_b64" {
            return Ok(());
        }
        if let Some(e) = match name {
            "s_bcnt1_i32_b64" => Some(("popcount({a})", true)),
            "s_bcnt0_i32_b64" => Some(("64ul - popcount({a})", true)),
            "s_ff1_i32_b64" => Some(("({a} == 0ul ? 0xffffffffu : uint(ctz({a})))", false)),
            "s_ff0_i32_b64" => Some(("(~{a} == 0ul ? 0xffffffffu : uint(ctz(~{a})))", false)),
            "s_flbit_i32_b64" => Some(("({a} == 0ul ? 0xffffffffu : uint(clz({a})))", false)),
            "s_flbit_i32_i64" => Some(("ffbh_i64({a})", false)),
            _ => None,
        } {
            let a = self.src64(s0, lit, pc)?;
            let (e, scc) = (e.0.replace("{a}", &a), e.1);
            let Some(d) = self.sdst(sdst) else {
                return Ok(());
            };
            if scc {
                self.regs.insert("scc".into());
            }
            let scc = if scc { " scc = r != 0u;" } else { "" };
            self.line(&format!("{{ uint r = uint({e});{scc} {d} = r; }}"));
            return Ok(());
        }
        if name == "s_bitreplicate_b64_b32" {
            let a = self.src32(s0, lit, pc)?;
            self.line(&format!(
                "{{ uint a = {a}; ulong r = 0ul; for (uint i = 0u; i < 32u; i++) r |= ulong((a >> i) & 1u) * 3ul << (2u * i);"
            ));
            self.set64(sdst, "r");
            self.line("}");
            return Ok(());
        }
        if b64 {
            let a = self.src64(s0, lit, pc)?;
            let (e, scc) = match name {
                "s_mov_b64" => (a, false),
                "s_not_b64" => (format!("~{a}"), true),
                "s_brev_b64" => (format!("brev64({a})"), false),
                "s_wqm_b64" => (format!("wqm64({a})"), true),
                "s_quadmask_b64" => (format!("ulong(quadmask64({a}))"), true),
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            self.line(&format!("{{ ulong r = {e};"));
            if scc {
                self.regs.insert("scc".into());
                self.line("scc = r != 0ul;");
            }
            self.set64(sdst, "r");
            self.line("}");
            return Ok(());
        }
        let a = self.src32(s0, lit, pc)?;
        let (e, scc) = match name {
            "s_mov_b32" => (a, false),
            "s_not_b32" => (format!("~{a}"), true),
            "s_brev_b32" => (format!("reverse_bits({a})"), false),
            "s_bcnt1_i32_b32" => (format!("popcount({a})"), true),
            "s_bcnt0_i32_b32" => (format!("32u - popcount({a})"), true),
            "s_ff1_i32_b32" => (format!("({a} == 0u ? 0xffffffffu : ctz({a}))"), false),
            "s_ff0_i32_b32" => (format!("(~{a} == 0u ? 0xffffffffu : ctz(~{a}))"), false),
            "s_flbit_i32_b32" => (format!("({a} == 0u ? 0xffffffffu : clz({a}))"), false),
            "s_flbit_i32" => (format!("ffbh_i32({a})"), false),
            "s_wqm_b32" => (format!("uint(wqm64(ulong({a})))"), true),
            "s_sext_i32_i8" => (format!("uint(int(char({a})))"), false),
            "s_sext_i32_i16" => (format!("uint(int(short({a})))"), false),
            "s_abs_i32" => (format!("uint(abs(int({a})))"), true),
            "s_quadmask_b32" => (format!("quadmask64(ulong({a}))"), true),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let Some(d) = self.sdst(sdst) else {
            return Ok(());
        };
        if scc {
            self.regs.insert("scc".into());
            self.line(&format!("{{ uint r = {e}; scc = r != 0u; {d} = r; }}"));
        } else {
            self.line(&format!("{d} = {e};"));
        }
        Ok(())
    }

    /// The variable holding a hardware register programs can write.
    pub(super) fn hwreg_var(&mut self, id: u16) -> String {
        let name = if id == hwreg::MODE {
            "hw_mode"
        } else {
            "hw_trapsts"
        };
        self.regs.insert(name.into());
        name.into()
    }

    /// What `s_getreg_b32` reads from hardware register `id`. What depends on
    /// where the hardware placed the wave, or on how the machine is set up
    /// (bases, ids, apertures, trace and debug bits), reads 0.
    fn hwreg(&mut self, id: u16) -> String {
        match id {
            hwreg::MODE | hwreg::TRAPSTS => self.hwreg_var(id),
            hwreg::STATUS => {
                self.regs.insert("scc".into());
                let execz = self.mask_any(126);
                let vccz = self.mask_any(106);
                let [x, y, z] = self.req.threads;
                let in_wg = if self.req.stage.grouped() && x * y * z > 64 {
                    hwreg::STATUS_IN_WG
                } else {
                    0
                };
                format!(
                    "(uint(scc) | ({execz} ? 0u : {:#x}u) | ({vccz} ? 0u : {:#x}u) | {:#x}u)",
                    hwreg::STATUS_EXECZ,
                    hwreg::STATUS_VCCZ,
                    hwreg::STATUS_VALID | in_wg
                )
            }
            hwreg::GPR_ALLOC => format!(
                "{:#x}u",
                (self.req.rsrc1 & 0x3F) << 8 | hwreg::GPR_ALLOC_SGPRS
            ),
            hwreg::LDS_ALLOC if self.req.stage == Stage::Compute => {
                let granules = self.req.compute_rsrc2 >> 15 & 0x1FF;
                format!("{:#x}u", (granules * 2) << 12)
            }
            _ => "0u".into(),
        }
    }

    // M0 indexes s0 to s105 and VCC; the ISA guide calls indices past them
    // illegal, and here they read 0 and write nothing.
    fn sgpr_movrel(
        &mut self,
        pc: usize,
        name: &str,
        sdst: u16,
        s0: u16,
        lit: Option<u32>,
    ) -> Result<(), TranslateError> {
        const LAST: u16 = 107;
        let words = if name.ends_with("_b64") { 2 } else { 1 };
        let m0 = self.s(124);
        let (from, to) = if name == "s_movrelsd_2_b32" {
            (
                format!("({m0} & 0x3ffu)"),
                format!("(({m0} >> 16) & 0x3ffu)"),
            )
        } else {
            (m0.clone(), m0)
        };
        let indexed_src = name != "s_movreld_b32" && name != "s_movreld_b64";
        let indexed_dst = name != "s_movrels_b32" && name != "s_movrels_b64";
        if (indexed_src && s0 + words - 1 > LAST) || (indexed_dst && sdst + words - 1 > LAST) {
            return Err(err(pc, format!("{name} indexing past VCC")));
        }
        let mut stmt = String::from("{ ");
        let mut values = Vec::new();
        for k in 0..words {
            let value = if indexed_src {
                let cases: String = (0..=LAST)
                    .map(|c| format!("case {c}u: mv{k} = {}; break; ", self.s(c)))
                    .collect();
                format!("switch ({from} + {}u) {{ {cases}default: break; }}", s0 + k)
            } else if words == 2 {
                format!("({}) >> {}", self.src64(s0, lit, pc)?, 32 * k)
            } else {
                self.src32(s0, lit, pc)?
            };
            if indexed_src {
                write!(stmt, "uint mv{k} = 0u; {value} ").unwrap();
            } else {
                write!(stmt, "uint mv{k} = uint({value}); ").unwrap();
            }
            values.push(format!("mv{k}"));
        }
        for (k, value) in values.iter().enumerate() {
            let k = k as u16;
            if indexed_dst {
                let cases: String = (0..=LAST)
                    .map(|c| {
                        self.written(c);
                        format!("case {c}u: {} = {value}; break; ", self.sreg(c))
                    })
                    .collect();
                write!(
                    stmt,
                    "switch ({to} + {}u) {{ {cases}default: break; }} ",
                    sdst + k
                )
                .unwrap();
            } else if let Some(d) = self.sdst(sdst + k) {
                write!(stmt, "{d} = {value}; ").unwrap();
            }
        }
        stmt.push('}');
        self.line(&stmt);
        Ok(())
    }

    pub(super) fn sop2(
        &mut self,
        pc: usize,
        name: &str,
        sdst: u16,
        s0: u16,
        s1: u16,
        lit: Option<u32>,
    ) -> Result<(), TranslateError> {
        self.regs.insert("scc".into());
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if name == "s_bfm_b64" {
            let (a, b) = (self.src32(s0, lit, pc)?, self.src32(s1, lit, pc)?);
            self.set64(
                sdst,
                &format!("(((1ul << ({a} & 63u)) - 1ul) << ({b} & 63u))"),
            );
            return Ok(());
        }
        if wide {
            let a = self.src64(s0, lit, pc)?;
            let b = if matches!(
                name,
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" | "s_bfe_u64" | "s_bfe_i64"
            ) {
                let b = self.src32(s1, lit, pc)?;
                format!("ulong({b})")
            } else {
                self.src64(s1, lit, pc)?
            };
            let (e, scc) = match name {
                "s_and_b64" => (format!("{a} & {b}"), true),
                "s_or_b64" => (format!("{a} | {b}"), true),
                "s_xor_b64" => (format!("{a} ^ {b}"), true),
                "s_andn2_b64" => (format!("{a} & ~{b}"), true),
                "s_orn2_b64" => (format!("{a} | ~{b}"), true),
                "s_nand_b64" => (format!("~({a} & {b})"), true),
                "s_nor_b64" => (format!("~({a} | {b})"), true),
                "s_xnor_b64" => (format!("~({a} ^ {b})"), true),
                "s_lshl_b64" => (format!("{a} << ({b} & 63ul)"), true),
                "s_lshr_b64" => (format!("{a} >> ({b} & 63ul)"), true),
                "s_ashr_i64" => (format!("ulong(long({a}) >> ({b} & 63ul))"), true),
                "s_bfe_u64" => (format!("bfe_u64({a}, uint({b}))"), true),
                "s_bfe_i64" => (format!("bfe_i64({a}, uint({b}))"), true),
                "s_cselect_b64" => (format!("scc ? {a} : {b}"), false),
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            self.line(&format!("{{ ulong r = {e};"));
            if scc {
                self.line("scc = r != 0ul;");
            }
            self.set64(sdst, "r");
            self.line("}");
            return Ok(());
        }
        let ea = self.src32(s0, lit, pc)?;
        let eb = self.src32(s1, lit, pc)?;
        let (a, b) = ("a", "b");
        let (e, scc) = match name {
            "s_add_u32" => (format!("{a} + {b}"), "r < a"),
            "s_sub_u32" => (format!("{a} - {b}"), "b > a"),
            "s_add_i32" => (format!("{a} + {b}"), "((~(a ^ b)) & (a ^ r)) >> 31 != 0u"),
            "s_sub_i32" => (format!("{a} - {b}"), "((a ^ b) & (a ^ r)) >> 31 != 0u"),
            "s_addc_u32" => (
                format!("{a} + {b} + uint(scc)"),
                "(ulong(a) + ulong(b) + ulong(scc)) >> 32 != 0ul",
            ),
            "s_subb_u32" => (
                format!("{a} - {b} - uint(scc)"),
                "ulong(b) + ulong(scc) > ulong(a)",
            ),
            "s_min_i32" => (format!("uint(min(int({a}), int({b})))"), "int(a) < int(b)"),
            "s_min_u32" => (format!("min({a}, {b})"), "a < b"),
            "s_max_i32" => (format!("uint(max(int({a}), int({b})))"), "int(a) > int(b)"),
            "s_max_u32" => (format!("max({a}, {b})"), "a > b"),
            "s_cselect_b32" => (format!("scc ? {a} : {b}"), ""),
            "s_and_b32" => (format!("{a} & {b}"), "r != 0u"),
            "s_or_b32" => (format!("{a} | {b}"), "r != 0u"),
            "s_xor_b32" => (format!("{a} ^ {b}"), "r != 0u"),
            "s_andn2_b32" => (format!("{a} & ~{b}"), "r != 0u"),
            "s_orn2_b32" => (format!("{a} | ~{b}"), "r != 0u"),
            "s_nand_b32" => (format!("~({a} & {b})"), "r != 0u"),
            "s_nor_b32" => (format!("~({a} | {b})"), "r != 0u"),
            "s_xnor_b32" => (format!("~({a} ^ {b})"), "r != 0u"),
            "s_lshl_b32" => (format!("{a} << ({b} & 31u)"), "r != 0u"),
            "s_lshr_b32" => (format!("{a} >> ({b} & 31u)"), "r != 0u"),
            "s_ashr_i32" => (format!("uint(int({a}) >> ({b} & 31u))"), "r != 0u"),
            "s_bfm_b32" => (format!("((1u << ({a} & 31u)) - 1u) << ({b} & 31u)"), ""),
            "s_mul_i32" => (format!("uint(int({a}) * int({b}))"), ""),
            "s_mul_hi_u32" => (format!("mulhi({a}, {b})"), ""),
            "s_mul_hi_i32" => (format!("uint(mulhi(int({a}), int({b})))"), ""),
            "s_bfe_u32" => (
                format!("bfe_u32({a}, {b} & 31u, ({b} >> 16) & 0x7fu)"),
                "r != 0u",
            ),
            "s_bfe_i32" => (
                format!("bfe_i32({a}, {b} & 31u, ({b} >> 16) & 0x7fu)"),
                "r != 0u",
            ),
            "s_absdiff_i32" => (format!("uint(abs(int({a}) - int({b})))"), "r != 0u"),
            "s_lshl1_add_u32" => (
                format!("({a} << 1) + {b}"),
                "(ulong(a) << 1) + ulong(b) > 0xfffffffful",
            ),
            "s_lshl2_add_u32" => (
                format!("({a} << 2) + {b}"),
                "(ulong(a) << 2) + ulong(b) > 0xfffffffful",
            ),
            "s_lshl3_add_u32" => (
                format!("({a} << 3) + {b}"),
                "(ulong(a) << 3) + ulong(b) > 0xfffffffful",
            ),
            "s_lshl4_add_u32" => (
                format!("({a} << 4) + {b}"),
                "(ulong(a) << 4) + ulong(b) > 0xfffffffful",
            ),
            "s_pack_ll_b32_b16" => (format!("({a} & 0xffffu) | ({b} << 16)"), ""),
            "s_pack_lh_b32_b16" => (format!("({a} & 0xffffu) | ({b} & 0xffff0000u)"), ""),
            "s_pack_hh_b32_b16" => (format!("({a} >> 16) | ({b} & 0xffff0000u)"), ""),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let d = self.sdst(sdst);
        let mut s = format!("{{ uint a = {ea}; uint b = {eb}; uint r = {e}; (void)a; (void)b;");
        if !scc.is_empty() {
            write!(s, " scc = {scc};").unwrap();
        }
        if let Some(d) = d {
            write!(s, " {d} = r;").unwrap();
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }

    pub(super) fn sopc(
        &mut self,
        pc: usize,
        name: &str,
        s0: u16,
        s1: u16,
        lit: Option<u32>,
    ) -> Result<(), TranslateError> {
        self.regs.insert("scc".into());
        let rest = name.trim_start_matches("s_cmp_");
        if name.starts_with("s_bitcmp") {
            let a = if name.ends_with("b64") {
                self.src64(s0, lit, pc)?
            } else {
                self.src32(s0, lit, pc)?
            };
            let b = self.src32(s1, lit, pc)?;
            let want = if name.starts_with("s_bitcmp1") {
                "1"
            } else {
                "0"
            };
            let mask = if name.ends_with("b64") { "63u" } else { "31u" };
            self.line(&format!("scc = (({a} >> ({b} & {mask})) & 1) == {want};"));
            return Ok(());
        }
        let (op, ty) = rest
            .split_once('_')
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        let (a, b) = if ty == "u64" {
            (self.src64(s0, lit, pc)?, self.src64(s1, lit, pc)?)
        } else if ty == "i32" {
            let (a, b) = (self.src32(s0, lit, pc)?, self.src32(s1, lit, pc)?);
            (format!("int({a})"), format!("int({b})"))
        } else {
            (self.src32(s0, lit, pc)?, self.src32(s1, lit, pc)?)
        };
        let c = cmp_op(op).ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        self.line(&format!("scc = {a} {c} {b};"));
        Ok(())
    }

    pub(super) fn sopk(
        &mut self,
        pc: usize,
        name: &str,
        sdst: u16,
        simm16: u16,
        lit: Option<u32>,
    ) -> Result<(), TranslateError> {
        let k = simm16 as i16 as i32;
        let (id, offset, size) = (simm16 & 0x3F, simm16 >> 6 & 0x1F, (simm16 >> 11 & 0x1F) + 1);
        let field = u32::MAX >> (32 - size);
        match name {
            "s_getreg_b32" => {
                let value = self.hwreg(id);
                if let Some(d) = self.sdst(sdst) {
                    self.line(&format!("{d} = ({value} >> {offset}u) & {field:#x}u;"));
                }
            }
            // The float mode stays the one `PGM_RSRC1` gives at translation.
            "s_setreg_b32" | "s_setreg_imm32_b32" if matches!(id, hwreg::MODE | hwreg::TRAPSTS) => {
                let value = match lit {
                    Some(v) if name == "s_setreg_imm32_b32" => format!("{v:#x}u"),
                    _ => self.s(sdst),
                };
                let var = self.hwreg_var(id);
                let mask = field << offset;
                self.line(&format!(
                    "{var} = ({var} & ~{mask:#x}u) | (({value} << {offset}u) & {mask:#x}u);"
                ));
            }
            "s_call_b64" => {
                let next = calls::Expanded::address(&self.runs, pc) + 4;
                self.set64(sdst, &format!("{next:#x}ul"));
            }
            "s_setreg_b32" | "s_setreg_imm32_b32" => {}
            "s_movk_i32" => {
                if let Some(d) = self.sdst(sdst) {
                    self.line(&format!("{d} = 0x{:x}u;", k as u32));
                }
            }
            "s_cmovk_i32" => {
                self.regs.insert("scc".into());
                if let Some(d) = self.sdst(sdst) {
                    self.line(&format!("if (scc) {d} = 0x{:x}u;", k as u32));
                }
            }
            "s_version" => {}
            "s_addk_i32" => {
                self.regs.insert("scc".into());
                let d = self.s(sdst);
                self.line(&format!("{{ uint a = {d}; uint r = a + 0x{:x}u; scc = ((~(a ^ 0x{:x}u)) & (a ^ r)) >> 31 != 0u; {d} = r; }}", k as u32, k as u32));
            }
            "s_mulk_i32" => {
                let d = self.s(sdst);
                self.line(&format!("{d} = uint(int({d}) * {k});"));
            }
            "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" => {}
            n if n.starts_with("s_cmpk_") => {
                self.regs.insert("scc".into());
                let rest = &n[7..];
                let (op, ty) = rest.split_once('_').unwrap();
                let c =
                    cmp_op(op).ok_or_else(|| err(pc, format!("unsupported instruction {n}")))?;
                let d = self.s(sdst);
                if ty == "i32" {
                    self.line(&format!("scc = int({d}) {c} {k};"));
                } else {
                    self.line(&format!("scc = {d} {c} {simm16}u;"));
                }
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        }
        Ok(())
    }

    /// The constant buffer slot and first dword of a load through a traced base.
    fn constant_load(
        &mut self,
        pc: usize,
        name: &str,
        sbase: u16,
        offset: i32,
        soffset: u16,
        n: u16,
    ) -> Option<(usize, u32)> {
        // A program that invalidates the scalar cache may read what it wrote.
        let invalidates =
            |(_, i): &(usize, Inst)| matches!(i.op.name, "s_dcache_inv" | "s_gl1_inv");
        if self.insts.iter().any(invalidates) {
            return None;
        }
        let st = self.syms.as_ref()?.get(&pc)?;
        let at = |r: u16| st.get(r as usize).cloned().flatten();
        let words = if name.starts_with("s_buffer") { 4 } else { 2 };
        let first = at(sbase)?;
        if matches!(first, Source::Scaled { .. } | Source::Constant(_))
            || (1..words).any(|i| at(sbase + i) != Some(first.advance(4 * i as u32)))
            || soffset != src::NULL
            || offset < 0
            || offset % 4 != 0
        {
            return None;
        }
        let base = if words == 4 {
            Base::Buffer(first)
        } else {
            Base::Pointer(first)
        };
        let end = offset as u32 + 4 * n as u32;
        let slot = match self.constants.iter().position(|(b, _)| *b == base) {
            Some(i) => i,
            None if self.constants.len() < MAX_CONSTANTS => {
                self.constants.push((base, 0));
                self.constants.len() - 1
            }
            None => return None,
        };
        let bytes = &mut self.constants[slot].1;
        *bytes = (*bytes).max(end);
        Some((slot, offset as u32 / 4))
    }

    pub(super) fn smem(&mut self, pc: usize, name: &str, op: Smem) -> Result<(), TranslateError> {
        let Smem {
            sdata,
            sbase,
            offset,
            soffset,
            glc,
        } = op;
        let n = match name {
            "s_load_dword" | "s_buffer_load_dword" => 1,
            "s_load_dwordx2" | "s_buffer_load_dwordx2" => 2,
            "s_load_dwordx4" | "s_buffer_load_dwordx4" => 4,
            "s_load_dwordx8" | "s_buffer_load_dwordx8" => 8,
            "s_load_dwordx16" | "s_buffer_load_dwordx16" => 16,
            "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb" => return Ok(()),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let dests = sdata..sdata + n;
        if let Some(read) = &self.read_anywhere
            && dests.end <= SGPR_COUNT
            && !dests.clone().any(|r| read.contains(&r))
        {
            // Descriptors only: the host binds what they describe.
            for r in dests {
                self.sdst(r);
            }
            return Ok(());
        }
        let traced = (!glc).then(|| self.constant_load(pc, name, sbase, offset, soffset, n));
        if let Some((cb, first)) = traced.flatten() {
            let mut s = String::from("{");
            for i in 0..n {
                if let Some(d) = self.sdst(sdata + i) {
                    write!(s, " {d} = cb{cb}[{}];", first + i as u32).unwrap();
                }
            }
            s.push_str(" }");
            self.line(&s);
            return Ok(());
        }
        self.mem = true;
        let buffer = name.starts_with("s_buffer");
        let mut bases = vec![sbase, sbase + 1];
        if buffer {
            bases.push(sbase + 2);
        }
        if soffset != src::NULL {
            bases.push(soffset);
        }
        self.load_bases.push((dests.collect(), bases));
        let lo = self.sreg(sbase);
        let hi = self.sreg(sbase + 1);
        let base = if buffer {
            format!("(ulong({hi} & 0xffffu) << 32 | ulong({lo}))")
        } else {
            format!("(ulong({hi}) << 32 | ulong({lo}))")
        };
        let so = if soffset == src::NULL {
            "0u".to_string()
        } else {
            self.sreg(soffset)
        };
        let mut s = format!("{{ ulong a = {base} + ulong({so}) + ulong(long({offset}));");
        if n == 1 {
            s.push_str(" uint t0 = ld32(mem, mem_base, a);");
        } else {
            // One window lookup per instruction, not per dword: the lookup's 64-bit
            // compares cost GTA V's lighting programs most of their time.
            let temps: Vec<String> = (0..n).map(|i| format!("t{i}")).collect();
            write!(s, " uint {};", temps.join(", ")).unwrap();
            write!(s, " long o = goff_span(mem_base, a, {}ul);", 4 * n).unwrap();
            s.push_str(
                " if (o != OOB_OFF) { device const uint* p = (device const uint*)(mem + o);",
            );
            for i in 0..n {
                write!(s, " t{i} = p[{i}];").unwrap();
            }
            s.push_str(
                " } else { ulong r1, r2; long o1 = goff_run(mem_base, a, r1);\
                 long o2 = goff_run(mem_base, a + r1, r2);",
            );
            for i in 0..n {
                write!(s, " t{i} = ld_runs(mem, o1, r1, o2, r2, {}ul);", 4 * i).unwrap();
            }
            s.push_str(" }");
        }
        if buffer {
            // Dwords at or past the descriptor's range read as zero. The range
            // with a nonzero stride (num_records strides) is unverified.
            let records = self.sreg(sbase + 2);
            write!(
                s,
                " ulong st = ulong({hi} >> 16); ulong lim = st == 0ul ? ulong({records}) : ulong({records}) * st;\
                 ulong o0 = (ulong({so}) + ulong({offset})) & ~3ul;"
            )
            .unwrap();
            for i in 0..n {
                write!(s, " if (o0 + {}ul >= lim) t{i} = 0u;", 4 * i).unwrap();
            }
        }
        for i in 0..n {
            if let Some(d) = self.sdst(sdata + i) {
                write!(s, " {d} = t{i};").unwrap();
            }
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }
}
