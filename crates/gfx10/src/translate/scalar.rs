//! Scalar ALU (`SOP1`, `SOP2`, `SOPC`, `SOPK`) and scalar memory (`SMEM`).

use super::*;

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
        if name.ends_with("saveexec_b32") {
            let a = self.src32(s0, lit, pc)?;
            let op = name
                .trim_start_matches("s_")
                .trim_end_matches("_saveexec_b32");
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
                self.line(&format!("{d} = old;"));
            }
            self.line("}");
            return Ok(());
        }
        if name.ends_with("saveexec_b64") {
            let a = self.src64(s0, lit, pc)?;
            let exec = self.src64(126, None, pc)?;
            let new = match name {
                "s_and_saveexec_b64" => format!("{a} & old"),
                "s_or_saveexec_b64" => format!("{a} | old"),
                "s_xor_saveexec_b64" => format!("{a} ^ old"),
                "s_andn2_saveexec_b64" => format!("{a} & ~old"),
                "s_orn2_saveexec_b64" => format!("{a} | ~old"),
                "s_nand_saveexec_b64" => format!("~({a} & old)"),
                "s_nor_saveexec_b64" => format!("~({a} | old)"),
                "s_xnor_saveexec_b64" => format!("~({a} ^ old)"),
                "s_andn1_saveexec_b64" => format!("~{a} & old"),
                "s_orn1_saveexec_b64" => format!("~{a} | old"),
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            self.regs.insert("scc".into());
            let (lo, hi) = (self.s(126), self.s(127));
            self.line(&format!("{{ ulong old = {exec}; ulong n = {new}; scc = n != 0ul; {lo} = uint(n); {hi} = uint(n >> 32);"));
            self.set64(sdst, "old");
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
        if name == "s_getpc_b64" {
            let next = self.req.address + 4 * (pc as u64 + 1);
            self.set64(sdst, &format!("{next:#x}ul"));
            return Ok(());
        }
        if let Some(e) = match name {
            "s_bcnt1_i32_b64" => Some(("popcount({a})", true)),
            "s_ff1_i32_b64" => Some(("({a} == 0ul ? 0xffffffffu : uint(ctz({a})))", false)),
            "s_flbit_i32_b64" => Some(("({a} == 0ul ? 0xffffffffu : uint(clz({a})))", false)),
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
            "s_ff1_i32_b32" => (format!("({a} == 0u ? 0xffffffffu : ctz({a}))"), false),
            "s_flbit_i32_b32" => (format!("({a} == 0u ? 0xffffffffu : clz({a}))"), false),
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
            "s_min_i32" => (format!("uint(min(int({a}), int({b})))"), "r == a"),
            "s_min_u32" => (format!("min({a}, {b})"), "r == a"),
            "s_max_i32" => (format!("uint(max(int({a}), int({b})))"), "r == a"),
            "s_max_u32" => (format!("max({a}, {b})"), "r == a"),
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
    ) -> Result<(), TranslateError> {
        let k = simm16 as i16 as i32;
        match name {
            // Hardware registers (rounding and denormal modes) are not modelled.
            "s_setreg_b32" | "s_setreg_imm32_b32" => {}
            "s_movk_i32" => {
                if let Some(d) = self.sdst(sdst) {
                    self.line(&format!("{d} = 0x{:x}u;", k as u32));
                }
            }
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

    pub(super) fn smem(
        &mut self,
        pc: usize,
        name: &str,
        sdata: u16,
        sbase: u16,
        offset: i32,
        soffset: u16,
    ) -> Result<(), TranslateError> {
        let n = match name {
            "s_load_dword" | "s_buffer_load_dword" => 1,
            "s_load_dwordx2" | "s_buffer_load_dwordx2" => 2,
            "s_load_dwordx4" | "s_buffer_load_dwordx4" => 4,
            "s_load_dwordx8" | "s_buffer_load_dwordx8" => 8,
            "s_load_dwordx16" | "s_buffer_load_dwordx16" => 16,
            "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb" => return Ok(()),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        self.mem = true;
        let lo = self.s(sbase);
        let hi = self.s(sbase + 1);
        let base = if name.starts_with("s_buffer") {
            format!("(ulong({hi} & 0xffffu) << 32 | ulong({lo}))")
        } else {
            format!("(ulong({hi}) << 32 | ulong({lo}))")
        };
        let so = if soffset == src::NULL {
            "0u".to_string()
        } else {
            self.s(soffset)
        };
        let mut s = format!("{{ ulong a = {base} + ulong({so}) + ulong(long({offset}));");
        for i in 0..n {
            write!(s, " uint t{i} = ld32(mem, mem_base, a + {}ul);", 4 * i).unwrap();
        }
        for i in 0..n {
            if let Some(d) = self.sdst(sdata + i as u16) {
                write!(s, " {d} = t{i};").unwrap();
            }
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }
}
