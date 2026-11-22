//! Control flow, basic-block structuring and loop guards.

use super::*;

/// The loops of a program as `(first, last)` block leaders: each backward
/// branch's target and the last block that branches back to it. `None`
/// when two of them overlap without nesting.
pub(super) fn loop_regions(
    insts: &[(usize, Inst)],
    leaders: &BTreeSet<usize>,
) -> Option<Vec<(usize, usize)>> {
    let mut ends: BTreeMap<usize, usize> = BTreeMap::new();
    for (pc, inst) in insts {
        if let Some(t) = branch_target(*pc, inst)
            && t <= *pc
        {
            let block = *leaders.range(..=*pc).next_back()?;
            let end = ends.entry(t).or_insert(block);
            *end = (*end).max(block);
        }
    }
    let loops: Vec<(usize, usize)> = ends.into_iter().collect();
    for (i, a) in loops.iter().enumerate() {
        if !leaders.contains(&a.0) {
            return None;
        }
        for b in &loops[i + 1..] {
            if b.0 <= a.1 && b.1 > a.1 {
                return None;
            }
        }
    }
    Some(loops)
}

/// Whether `inst` needs lane masks as whole-wave values: it reads another lane, counts
/// lanes, or picks lanes out of EXEC or VCC with a constant (`s_and_b64 s[0:1], exec, 1`).
pub(super) fn crosses_lanes(inst: &Inst, insts: &[(usize, Inst)]) -> bool {
    let name = inst.op.name;
    let mask = |c: u16| c == 106 || c == 126;
    let picks = |c: u16| (129..=208).contains(&c) && c != 193 || c == src::LITERAL;
    match inst.kind {
        Kind::Valu(v) if name == "v_readlane_b32" && v.src[0] >= 256 => {
            !insts.iter().any(|(_, w)| {
                w.op.name == "v_writelane_b32"
                    && matches!(w.kind, Kind::Valu(x) if x.vdst as u16 + 256 == v.src[0])
            })
        }
        Kind::Valu(v) if matches!(name, "v_lshrrev_b64" | "v_lshlrev_b64") => mask(v.src[1]),
        Kind::Valu(_) => name.starts_with("v_mbcnt_"),
        Kind::Sopc { .. } => name.ends_with("_u64"),
        Kind::Sop1 { sdst, ssrc0 } => {
            matches!(
                name,
                "s_bcnt1_i32_b64" | "s_ff1_i32_b64" | "s_flbit_i32_b64"
            ) || name.ends_with("_b64") && (mask(sdst as u16) && picks(ssrc0))
        }
        Kind::Sop2 { sdst, ssrc0, ssrc1 } => {
            name.ends_with("_b64")
                && (mask(sdst as u16) && (picks(ssrc0) || picks(ssrc1))
                    || mask(ssrc0) && picks(ssrc1)
                    || mask(ssrc1) && picks(ssrc0))
        }
        _ => false,
    }
}

/// A scalar instruction that writes the lane mask.
pub(super) fn writes_exec(inst: &Inst) -> bool {
    let exec = |d: u16| d == src::EXEC_LO || d == src::EXEC_HI;
    match inst.kind {
        Kind::Sop1 { sdst, .. } => exec(sdst as u16) || inst.op.name.contains("exec"),
        Kind::Sop2 { sdst, .. } | Kind::Sopk { sdst, .. } => exec(sdst as u16),
        _ => false,
    }
}

pub(super) fn branch_target(pc: usize, inst: &Inst) -> Option<usize> {
    let name = inst.op.name;
    match inst.kind {
        Kind::Sopp { simm16 } if name == "s_branch" || name.starts_with("s_cbranch") => {
            Some((pc as i64 + 1 + simm16 as i16 as i64) as usize)
        }
        _ => None,
    }
}

impl<'a> Tx<'a> {
    pub(super) fn run(&mut self) -> Result<(), TranslateError> {
        let mut leaders: BTreeSet<usize> = BTreeSet::new();
        leaders.insert(0);
        for (pc, inst) in &self.insts {
            if let Some(t) = branch_target(*pc, inst) {
                leaders.insert(t);
                leaders.insert(pc + 1);
            }
        }
        let structured = leaders.len() > 1;
        let loops = loop_regions(&self.insts, &leaders);
        self.regions = structured && loops.is_some();
        let loops = loops.unwrap_or_default();
        self.has_barrier = self.insts.iter().any(|(_, i)| i.op.name == "s_barrier");
        self.scratch_stores = self
            .insts
            .iter()
            .any(|(_, i)| i.op.name.starts_with("scratch_store"));
        self.guarded = self.req.stage == Stage::Compute
            && self
                .insts
                .iter()
                .any(|(pc, i)| branch_target(*pc, i).is_some_and(|t| t <= *pc));
        let insts = std::mem::take(&mut self.insts);
        let ret = if self.req.stage == Stage::Compute {
            "return;"
        } else {
            "return o;"
        };
        let mut block = None;
        for (pc, inst) in &insts {
            if self.regions && leaders.contains(pc) {
                self.close_guard();
                if let Some(prev) = block {
                    if !self.block_ended {
                        self.line(&format!("pc = {pc};"));
                    }
                    self.body.push_str("    }\n");
                    for &(start, _) in loops.iter().rev().filter(|l| l.1 == prev) {
                        self.body
                            .push_str(&format!("    if (pc != {start}u) break; }}\n"));
                    }
                }
                for _ in loops.iter().filter(|l| l.0 == *pc) {
                    self.body.push_str("    for (;;) {\n");
                }
                self.body.push_str(&format!("    if (pc == {pc}u) {{\n"));
                block = Some(*pc);
                self.block_ended = false;
            } else if structured && !self.regions && leaders.contains(pc) {
                self.close_guard();
                self.body.push_str(&format!("    case {pc}:\n"));
            }
            self.pc = *pc;
            let text = print(inst, Syntax { wave64: true });
            self.line(&format!("// {text}"));
            let step = self.lockstep && matches!(inst.kind, Kind::Ds { gds: false, .. });
            if step {
                self.line(LOCKSTEP_BARRIER);
            }
            self.inst(*pc, inst)?;
            if !self.lockstep && writes_exec(inst) {
                // Other lanes are other threads: keep only this lane's bit.
                let (lo, hi) = (self.s(src::EXEC_LO), self.s(src::EXEC_HI));
                self.line(&format!("{lo} &= 1u; {hi} = 0u;"));
            }
            if step {
                self.line(LOCKSTEP_BARRIER);
            }
        }
        self.close_guard();
        self.insts = insts;
        if self.regions {
            self.body = format!(
                "    uint pc = 0;\n    uint back = 0;\n{}    }}\n{}    {ret}\n",
                self.body,
                loops
                    .iter()
                    .rev()
                    .filter(|l| Some(l.1) == block)
                    .map(|l| format!("    if (pc != {}u) break; }}\n", l.0))
                    .collect::<String>()
            );
        } else if structured {
            self.body = format!(
                "    uint pc = 0;\n    uint back = 0;\n    for (;;) {{\n    switch (pc) {{\n{}    default: {}\n    }}\n    }}\n",
                self.body,
                if self.req.stage == Stage::Compute {
                    "return;"
                } else {
                    "return o;"
                }
            );
        }
        Ok(())
    }

    fn inst(&mut self, pc: usize, inst: &Inst) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let lit = inst.literal;
        match inst.kind {
            Kind::Sopp { simm16 } => self.sopp(pc, inst, simm16),
            Kind::Sop1 { sdst, ssrc0 } => self.sop1(pc, name, sdst as u16, ssrc0, lit),
            Kind::Sop2 { sdst, ssrc0, ssrc1 } => {
                self.sop2(pc, name, sdst as u16, ssrc0, ssrc1, lit)
            }
            Kind::Sopc { ssrc0, ssrc1 } => self.sopc(pc, name, ssrc0, ssrc1, lit),
            Kind::Sopk { sdst, simm16 } => self.sopk(pc, name, sdst as u16, simm16),
            Kind::Smem {
                sdata,
                sbase,
                offset,
                soffset,
                ..
            } => self.smem(pc, name, sdata as u16, sbase as u16, offset, soffset as u16),
            Kind::Valu(v) => self.valu(pc, inst, &v),
            Kind::Vintrp {
                vdst,
                vsrc,
                attr,
                chan,
            } => self.vintrp(pc, name, vdst as u16, vsrc as u16, attr, chan),
            Kind::Exp {
                en,
                target,
                compr,
                done,
                vsrc,
                ..
            } => self.exp(pc, en, target, compr, done, vsrc),
            Kind::Mubuf(b) => self.mubuf(pc, name, &b, None),
            Kind::Flat(f) => self.flat(pc, name, &f),
            Kind::Mtbuf(b) => match name.strip_prefix('t') {
                Some(plain)
                    if plain.starts_with("buffer_load_format_")
                        || plain.starts_with("buffer_store_format_") =>
                {
                    self.mubuf(pc, plain, &b, Some(b.format))
                }
                _ => Err(err(pc, format!("unsupported instruction {name}"))),
            },
            Kind::Ds {
                offset0,
                offset1,
                gds,
                addr,
                data0,
                data1,
                vdst,
            } => self.ds(
                pc,
                name,
                offset0,
                offset1,
                gds,
                [addr as u16, data0 as u16, data1 as u16, vdst as u16],
            ),
            Kind::Mimg(m) => self.mimg(pc, name, &m),
        }
    }

    fn sopp(&mut self, pc: usize, inst: &Inst, _simm16: u16) -> Result<(), TranslateError> {
        let name = inst.op.name;
        match name {
            "s_endpgm" => {
                self.end_program();
            }
            "s_code_end" | "s_nop" | "s_waitcnt" | "s_inst_prefetch" | "s_clause" | "s_sendmsg"
            | "s_waitcnt_depctr" | "s_setprio" | "s_sleep" | "s_icache_inv" | "s_trap" => {}
            "s_barrier" => {
                if self.req.stage == Stage::Compute {
                    self.line(
                        "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);",
                    );
                    if self.guarded {
                        self.line("{ bool leave = tg_bail; threadgroup_barrier(mem_flags::mem_threadgroup); if (leave) return; }");
                    }
                }
            }
            "s_branch" => {
                let t = branch_target(pc, inst).unwrap();
                let guard = self.loop_guard(pc, t);
                if self.regions {
                    self.line(&format!("{guard}pc = {t};"));
                    self.block_ended = true;
                } else {
                    self.line(&format!("{guard}pc = {t}; continue;"));
                }
            }
            n if n.starts_with("s_cbranch") => {
                let t = branch_target(pc, inst).unwrap();
                let cond = match n {
                    "s_cbranch_scc0" | "s_cbranch_scc1" if self.lockstep => {
                        let any = "wave_any(scc, lockstep_vote, sgi, wk)";
                        if n.ends_with('0') {
                            format!("!{any}")
                        } else {
                            any.to_string()
                        }
                    }
                    "s_cbranch_scc0" => "!scc".to_string(),
                    "s_cbranch_scc1" => "scc".to_string(),
                    "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz"
                    | "s_cbranch_execnz" => {
                        let any = self.mask_any(if n.contains("vcc") { 106 } else { 126 });
                        let any = if self.lockstep {
                            format!("wave_any({any}, lockstep_vote, sgi, wk)")
                        } else if self.req.stage == Stage::Compute {
                            format!("simd_any({any})")
                        } else {
                            any
                        };
                        if n.ends_with("nz") {
                            any
                        } else {
                            format!("!{any}")
                        }
                    }
                    _ => return Err(err(pc, format!("unsupported branch {n}"))),
                };
                if cond.contains("scc") {
                    self.regs.insert("scc".into());
                }
                let guard = self.loop_guard(pc, t);
                if self.regions {
                    let next = pc + 1;
                    self.line(&format!(
                        "if ({cond}) {{ {guard}pc = {t}; }} else {{ pc = {next}; }}"
                    ));
                    self.block_ended = true;
                } else {
                    self.line(&format!("if ({cond}) {{ {guard}pc = {t}; continue; }}"));
                }
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        }
        Ok(())
    }

    /// Code that ends an invocation stuck in a loop (see `GUARD_PRELUDE`); with a
    /// barrier, the group leaves together at the next one.
    fn loop_guard(&self, pc: usize, target: usize) -> String {
        if target > pc {
            return String::new();
        }
        match self.req.stage {
            Stage::Pixel => {
                format!("if (++back > {MAX_BACK_BRANCHES}u) {{ discard_fragment(); return o; }} ")
            }
            Stage::Vertex => format!("if (++back > {MAX_BACK_BRANCHES}u) {{ return o; }} "),
            Stage::Compute if self.lockstep => {
                "if (lockstep_leave(guard, ++back, tg_bail)) { return; } ".into()
            }
            Stage::Compute if self.has_barrier => {
                "if (guard_leave(guard, ++back, bail_at, tg_bail)) { return; } ".into()
            }
            Stage::Compute => "if (guard_stop(guard, ++back)) { return; } ".into(),
        }
    }

    fn end_program(&mut self) {
        match self.req.stage {
            Stage::Pixel => self.line("if (!exported) discard_fragment(); return o;"),
            Stage::Vertex => self.line("return o;"),
            Stage::Compute => self.line("return;"),
        }
    }
}
