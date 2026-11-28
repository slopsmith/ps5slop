//! Control flow, basic-block structuring and loop guards.

use super::*;

/// `s_sendmsg` message ids are in `SIMM16[3:0]`; 9 is `MSG_GS_ALLOC_REQ`.
const SENDMSG_GS_ALLOC_REQ: u16 = 9;

/// Branches whose condition is false on the machines programs run on.
const NEVER_TAKEN: [&str; 2] = ["s_cbranch_cdbgsys", "s_cbranch_cdbgsys_and_user"];

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
        Kind::Valu(v) if name == "v_writelane_b32" => !insts.iter().any(|(_, r)| {
            r.op.name == "v_readlane_b32"
                && matches!(r.kind, Kind::Valu(x) if x.src[0] == v.vdst as u16 + 256)
        }),
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

/// An LDS atomic whose result depends on the order of the lanes that run it.
pub(super) fn ordered_lds(inst: &Inst) -> bool {
    matches!(inst.kind, Kind::Ds { gds: false, .. }) && inst.op.name.contains("_rtn_")
}

/// An LDS atomic built from a compare-and-swap loop or from plain accesses, whose
/// lanes Metal does not apply in lane order when they hit one place.
pub(super) fn serial_lds(inst: &Inst) -> bool {
    let name = inst.op.name;
    matches!(inst.kind, Kind::Ds { gds: false, .. })
        && !name.starts_with("ds_read")
        && !name.starts_with("ds_write")
        && ["f32", "64", "rsub", "inc_", "dec_", "wrap", "mskor"]
            .iter()
            .any(|k| name.contains(k))
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

/// Where a branch goes, if it can be taken.
///
/// Branches on the system debug flag are never taken (see `branch_cond`), so
/// they have no target: titles guard debug code after `s_endpgm` with them,
/// which would otherwise be decoded and translated.
pub(super) fn branch_target(pc: usize, inst: &Inst) -> Option<usize> {
    let name = inst.op.name;
    if NEVER_TAKEN.contains(&name) {
        return None;
    }
    match inst.kind {
        Kind::Sopp { simm16 } if name == "s_branch" || name.starts_with("s_cbranch") => {
            Some((pc as i64 + 1 + simm16 as i16 as i64) as usize)
        }
        _ => None,
    }
}

/// A program as nested regions, so the MSL needs no `pc` dispatcher. Indices
/// are into the instruction list.
enum Node {
    /// An instruction other than a branch.
    Inst(usize),
    /// A forward branch over `then`; with `els`, the unconditional branch that
    /// ends `then` and the code it skips.
    If {
        branch: usize,
        then: Vec<Node>,
        els: Option<(usize, Vec<Node>)>,
    },
    Loop(Vec<Node>),
    /// A branch back to the start of the innermost loop.
    Continue(usize),
    /// A branch to the instruction after the innermost loop.
    Break(usize),
}

/// Finds the nesting the guest compiler's structured code has.
struct Nest<'i> {
    insts: &'i [(usize, Inst)],
    /// Loop starts, with the index of the last branch back to each.
    loops: BTreeMap<usize, usize>,
}

impl<'i> Nest<'i> {
    fn new(insts: &'i [(usize, Inst)]) -> Self {
        let mut loops = BTreeMap::new();
        for (i, (pc, inst)) in insts.iter().enumerate() {
            if let Some(t) = branch_target(*pc, inst)
                && t <= *pc
            {
                loops.insert(t, i);
            }
        }
        Nest { insts, loops }
    }

    /// The index of the instruction at `pc`, or the length for the end of the code.
    fn index(&self, pc: usize) -> Option<usize> {
        match self.insts.binary_search_by_key(&pc, |(p, _)| *p) {
            Ok(i) => Some(i),
            Err(i) if i == self.insts.len() => Some(i),
            Err(_) => None,
        }
    }

    fn pc(&self, i: usize) -> usize {
        self.insts.get(i).map_or(usize::MAX, |(p, _)| *p)
    }

    /// Instructions `a..b` as nodes, inside the loops `ctx` lists as (start,
    /// end) addresses, innermost last; `None` when they do not nest.
    fn region(&self, a: usize, b: usize, ctx: &mut Vec<(usize, usize)>) -> Option<Vec<Node>> {
        let mut out = Vec::new();
        let mut i = a;
        while i < b {
            let (pc, inst) = &self.insts[i];
            let pc = *pc;
            if let Some(&last) = self.loops.get(&pc)
                && last < b
                && !ctx.iter().any(|l| l.0 == pc)
            {
                ctx.push((pc, self.pc(last + 1)));
                let body = self.region(i, last + 1, ctx);
                ctx.pop();
                out.push(Node::Loop(body?));
                i = last + 1;
                continue;
            }
            let Some(t) = branch_target(pc, inst) else {
                out.push(Node::Inst(i));
                i += 1;
                continue;
            };
            let innermost = ctx.last().copied();
            if t <= pc {
                if innermost?.0 != t {
                    return None;
                }
                out.push(Node::Continue(i));
                i += 1;
                continue;
            }
            if innermost.is_some_and(|l| l.1 == t) {
                out.push(Node::Break(i));
                i += 1;
                continue;
            }
            let to = self.index(t)?;
            if to > b {
                return None;
            }
            if inst.op.name == "s_branch" {
                if to != i + 1 {
                    return None;
                }
                i += 1;
                continue;
            }
            let j = to - 1;
            let skip = (j > i && self.insts[j].1.op.name == "s_branch")
                .then(|| branch_target(self.insts[j].0, &self.insts[j].1))
                .flatten()
                .filter(|&u| u > t && !innermost.is_some_and(|l| l.1 == u))
                .and_then(|u| self.index(u))
                .filter(|&u| u <= b);
            if let Some(end) = skip {
                let then = self.region(i + 1, j, ctx)?;
                let els = self.region(to, end, ctx)?;
                out.push(Node::If {
                    branch: i,
                    then,
                    els: Some((j, els)),
                });
                i = end;
            } else {
                let then = self.region(i + 1, to, ctx)?;
                out.push(Node::If {
                    branch: i,
                    then,
                    els: None,
                });
                i = to;
            }
        }
        Some(out)
    }
}

impl<'a> Tx<'a> {
    /// Emits one instruction other than a branch, with its disassembly.
    fn emit_inst(&mut self, pc: usize, inst: &Inst) -> Result<(), TranslateError> {
        self.pc = pc;
        let text = print(inst, Syntax { wave64: true });
        self.line(&format!("// {text}"));
        let step = self.lockstep && matches!(inst.kind, Kind::Ds { gds: false, .. });
        if step {
            self.line("wave_meet(wv);");
        }
        let two_phase = inst.op.name.contains("wrxchg2");
        if step && !two_phase && (ordered_lds(inst) || serial_lds(inst)) {
            self.lane_order(serial_lds(inst), &mut |t| t.inst(pc, inst))?;
        } else {
            self.inst(pc, inst)?;
        }
        if !self.lockstep && writes_exec(inst) {
            // Other lanes are other threads: keep only this lane's bit.
            let (lo, hi) = (self.s(src::EXEC_LO), self.s(src::EXEC_HI));
            self.line(&format!("{lo} &= 1u; {hi} = 0u;"));
        }
        if step {
            self.line("wave_meet(wv);");
        }
        Ok(())
    }

    /// Emits an LDS access so a lockstep wave's lanes take effect in lane order:
    /// within a SIMD group Metal's native atomics were seen to, so the low half
    /// goes first; `serial` ones (compare-and-swap loops, plain accesses) run one
    /// lane at a time.
    pub(super) fn lane_order(
        &mut self,
        serial: bool,
        body: &mut dyn FnMut(&mut Self) -> Result<(), TranslateError>,
    ) -> Result<(), TranslateError> {
        for half in ["==", "!="] {
            if half == "!=" {
                self.line("wave_meet(wv);");
            }
            self.line(&format!("if ((wv.sgi & 1u) {half} 0u) {{"));
            if serial {
                self.line("for (uint sl = 0u; sl < 32u; sl++) {");
                self.line("if (uint(slane) == sl) {");
            }
            body(self)?;
            if serial {
                self.line("}");
                self.line("simdgroup_barrier(mem_flags::mem_threadgroup);");
                self.line("}");
            }
            self.line("}");
        }
        Ok(())
    }

    fn emit_nodes(
        &mut self,
        insts: &[(usize, Inst)],
        nodes: &[Node],
    ) -> Result<(), TranslateError> {
        for node in nodes {
            match node {
                Node::Inst(i) => self.emit_inst(insts[*i].0, &insts[*i].1)?,
                Node::If { branch, then, els } => {
                    let cond = self.branch_head(insts, *branch)?;
                    self.line(&format!("if (!({cond})) {{"));
                    self.emit_nodes(insts, then)?;
                    if let Some((skip, els)) = els {
                        self.branch_head(insts, *skip)?;
                        self.line("} else {");
                        self.emit_nodes(insts, els)?;
                    }
                    self.line("}");
                }
                Node::Loop(body) => {
                    self.line("for (;;) {");
                    self.emit_nodes(insts, body)?;
                    self.line("break;");
                    self.line("}");
                }
                Node::Continue(i) | Node::Break(i) => {
                    let (pc, inst) = &insts[*i];
                    let cond = self.branch_head(insts, *i)?;
                    let jump = match node {
                        Node::Continue(_) => {
                            let t = branch_target(*pc, inst).unwrap_or(*pc);
                            format!("{}continue;", self.loop_guard(*pc, t))
                        }
                        _ => "break;".to_string(),
                    };
                    if cond == "true" {
                        self.line(&jump);
                    } else {
                        self.line(&format!("if ({cond}) {{ {jump} }}"));
                    }
                }
            }
        }
        Ok(())
    }

    /// The disassembly of branch `i` as a comment, and its condition
    /// (`true` for `s_branch`).
    fn branch_head(&mut self, insts: &[(usize, Inst)], i: usize) -> Result<String, TranslateError> {
        let (pc, inst) = &insts[i];
        self.pc = *pc;
        let text = print(inst, Syntax { wave64: true });
        self.line(&format!("// {text}"));
        if inst.op.name == "s_branch" {
            return Ok("true".into());
        }
        self.branch_cond(*pc, inst.op.name)
    }

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
        self.guarded = self.req.stage.grouped()
            && self
                .insts
                .iter()
                .any(|(pc, i)| branch_target(*pc, i).is_some_and(|t| t <= *pc));
        let insts = std::mem::take(&mut self.insts);
        let ret = if self.req.stage.grouped() {
            "return;"
        } else {
            "return o;"
        };
        let nested = structured
            .then(|| Nest::new(&insts).region(0, insts.len(), &mut Vec::new()))
            .flatten();
        if let Some(nodes) = nested {
            self.regions = false;
            let result = self.emit_nodes(&insts, &nodes);
            self.close_guard();
            self.insts = insts;
            result?;
            self.body = format!("    uint back = 0;\n{}    {ret}\n", self.body);
            return Ok(());
        }
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
            self.emit_inst(*pc, inst)?;
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
                if self.req.stage.grouped() {
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
            Kind::Sopk { sdst, simm16 } => self.sopk(pc, name, sdst as u16, simm16, inst.literal),
            Kind::Smem {
                sdata,
                sbase,
                offset,
                soffset,
                glc,
                ..
            } => self.smem(
                pc,
                name,
                scalar::Smem {
                    sdata: sdata as u16,
                    sbase: sbase as u16,
                    offset,
                    soffset: soffset as u16,
                    glc,
                },
            ),
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

    fn sopp(&mut self, pc: usize, inst: &Inst, simm16: u16) -> Result<(), TranslateError> {
        let name = inst.op.name;
        match name {
            "s_endpgm" => {
                self.end_program();
            }
            "s_sendmsg"
                if self.req.stage == Stage::Mesh && simm16 & 0xF == SENDMSG_GS_ALLOC_REQ =>
            {
                // M0: vertices in bits 0 to 10, primitives from bit 12.
                let m0 = self.s(124);
                self.line(&format!(
                    "if ((lid.x & 63u) == 0u) m.set_primitive_count(min(({m0} >> 12) & 0x7ffu, MESH_PRIMITIVES));"
                ));
            }
            "s_code_end" | "s_nop" | "s_waitcnt" | "s_inst_prefetch" | "s_clause" | "s_sendmsg"
            | "s_waitcnt_depctr" | "s_setprio" | "s_sleep" | "s_icache_inv" | "s_trap"
            | "s_wakeup" | "s_incperflevel" | "s_decperflevel" | "s_ttracedata"
            | "s_ttracedata_imm" | "s_wait_idle" => {}
            "s_round_mode" | "s_denorm_mode" => {
                let shift = if name == "s_round_mode" { 0 } else { 4 };
                let mode = self.hwreg_var(hwreg::MODE);
                self.line(&format!(
                    "{mode} = ({mode} & ~{:#x}u) | {:#x}u;",
                    0xF << shift,
                    (simm16 as u32 & 0xF) << shift
                ));
            }
            "s_barrier" => {
                if self.req.stage.grouped() {
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
            n if NEVER_TAKEN.contains(&n) => {}
            n if n.starts_with("s_cbranch") => {
                let t = branch_target(pc, inst).unwrap();
                let cond = self.branch_cond(pc, n)?;
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

    /// The condition under which the conditional branch `name` is taken.
    fn branch_cond(&mut self, pc: usize, n: &str) -> Result<String, TranslateError> {
        let cond = match n {
            "s_cbranch_scc0" | "s_cbranch_scc1" if self.lockstep => {
                let any = "wave_any(scc, wv)";
                if n.ends_with('0') {
                    format!("!{any}")
                } else {
                    any.to_string()
                }
            }
            "s_cbranch_scc0" => "!scc".to_string(),
            "s_cbranch_scc1" => "scc".to_string(),
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let any = self.mask_any(if n.contains("vcc") { 106 } else { 126 });
                let any = if self.lockstep {
                    format!("wave_any({any}, wv)")
                } else if self.req.stage.grouped() {
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
            // The conditional debug flags as the reference RDNA2 machine runs
            // programs: the system flag clear, the user flag set.
            "s_cbranch_cdbgsys" | "s_cbranch_cdbgsys_and_user" => "false".to_string(),
            "s_cbranch_cdbguser" | "s_cbranch_cdbgsys_or_user" => "true".to_string(),
            _ => return Err(err(pc, format!("unsupported branch {n}"))),
        };
        if cond.contains("scc") {
            self.regs.insert("scc".into());
        }
        Ok(cond)
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
            Stage::Compute | Stage::Mesh if self.lockstep => {
                "if (lockstep_leave(guard, ++back, wv, tg_bail)) { return; } ".into()
            }
            Stage::Compute | Stage::Mesh if self.has_barrier => {
                "if (guard_leave(guard, ++back, bail_at, tg_bail)) { return; } ".into()
            }
            Stage::Compute | Stage::Mesh => "if (guard_stop(guard, ++back)) { return; } ".into(),
        }
    }

    fn end_program(&mut self) {
        match self.req.stage {
            Stage::Pixel => self.line("if (!exported) discard_fragment(); return o;"),
            Stage::Vertex => self.line("return o;"),
            Stage::Compute | Stage::Mesh => self.line("return;"),
        }
    }
}
