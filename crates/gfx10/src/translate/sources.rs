//! Data-flow analysis tracing descriptor dwords back to user SGPRs and tables.

use super::*;

/// SGPRs `sgpr_sources` follows: the general ones and VCC.
pub(super) const TRACKED_SGPRS: u16 = 108;

impl<'a> Tx<'a> {
    pub(super) fn descriptor_source(
        &self,
        pc: usize,
        first: u16,
        n: u16,
    ) -> Result<Source, TranslateError> {
        let Some(syms) = &self.syms else {
            return Ok(Source::UserSgpr(0));
        };
        let none = || err(pc, "descriptor not from user SGPRs or memory they point to");
        let st = syms.get(&pc).ok_or_else(none)?;
        let at = |r: u16| st.get(r as usize).cloned().flatten();
        let first_src = at(first)
            .filter(|s| !matches!(s, Source::Scaled { .. }))
            .ok_or_else(none)?;
        if let Source::Constant(_) = first_src {
            let mut dwords = Vec::new();
            for i in 0..n {
                match at(first + i) {
                    Some(Source::Constant(v)) if v.len() == 1 => dwords.push(v[0]),
                    _ => return Err(none()),
                }
            }
            return Ok(Source::Constant(dwords));
        }
        for i in 1..n {
            if at(first + i) != Some(first_src.advance(4 * i as u32)) {
                return Err(none());
            }
        }
        Ok(first_src)
    }

    /// Follows user SGPRs through moves and scalar loads, over every path
    /// through the program, to find where descriptor dwords come from.
    pub(super) fn sgpr_sources(&self) -> BTreeMap<usize, Syms> {
        let index: BTreeMap<usize, usize> = self
            .insts
            .iter()
            .enumerate()
            .map(|(i, (pc, _))| (*pc, i))
            .collect();
        let mut entry: Syms = vec![None; TRACKED_SGPRS as usize];
        for i in 0..self.req.user_sgprs as u16 {
            let r = (self.user_base + i) as usize;
            if r < entry.len() {
                entry[r] = Some(Source::UserSgpr(i as u8));
            }
        }
        let mut before: BTreeMap<usize, Syms> = BTreeMap::new();
        before.insert(0, entry);
        let mut work = vec![0usize];
        while let Some(pc) = work.pop() {
            let Some(&i) = index.get(&pc) else { continue };
            let (_, inst) = &self.insts[i];
            let mut st = before[&pc].clone();
            let pointer = |st: &Syms, r: u16| {
                let lo = st.get(r as usize).cloned().flatten()?;
                (st.get(r as usize + 1).cloned().flatten() == Some(lo.advance(4))).then_some(lo)
            };
            let mut new: BTreeMap<u16, Option<Source>> = BTreeMap::new();
            match inst.kind {
                Kind::Sop1 { sdst, ssrc0 } if inst.op.name == "s_mov_b32" && ssrc0 < SGPR_COUNT => {
                    new.insert(sdst as u16, st[ssrc0 as usize].clone());
                }
                Kind::Sop1 { sdst, ssrc0 }
                    if inst.op.name.starts_with("s_mov_b") && constant(inst, ssrc0).is_some() =>
                {
                    let v = constant(inst, ssrc0).unwrap_or(0);
                    new.insert(sdst as u16, Some(Source::Constant(vec![v])));
                    if inst.op.name == "s_mov_b64" {
                        let high = if v as i32 >= 0 || ssrc0 == src::LITERAL {
                            0
                        } else {
                            u32::MAX
                        };
                        new.insert(sdst as u16 + 1, Some(Source::Constant(vec![high])));
                    }
                }
                Kind::Sopk { sdst, simm16 } if inst.op.name == "s_movk_i32" => {
                    new.insert(
                        sdst as u16,
                        Some(Source::Constant(vec![simm16 as i16 as i32 as u32])),
                    );
                }
                Kind::Sop2 { sdst, ssrc0, ssrc1 } if inst.op.name == "s_bfm_b32" => {
                    let mask =
                        constant(inst, ssrc0)
                            .zip(constant(inst, ssrc1))
                            .map(|(width, offset)| {
                                1u32.checked_shl(width & 31)
                                    .map_or(u32::MAX, |b| b.wrapping_sub(1))
                                    << (offset & 31)
                            });
                    new.insert(sdst as u16, mask.map(|m| Source::Constant(vec![m])));
                }
                Kind::Sop2 { sdst, ssrc0, ssrc1 } if inst.op.name == "s_bfm_b64" => {
                    let mask =
                        constant(inst, ssrc0)
                            .zip(constant(inst, ssrc1))
                            .map(|(width, offset)| {
                                1u64.checked_shl(width & 63)
                                    .map_or(u64::MAX, |b| b.wrapping_sub(1))
                                    << (offset & 63)
                            });
                    new.insert(sdst as u16, mask.map(|m| Source::Constant(vec![m as u32])));
                    new.insert(
                        sdst as u16 + 1,
                        mask.map(|m| Source::Constant(vec![(m >> 32) as u32])),
                    );
                }
                // Bits a program ORs or ANDs into a descriptor dword at run
                // time are dropped; the result keeps where the dword came
                // from (the destination's, when it is one of the two).
                Kind::Sop2 { sdst, ssrc0, ssrc1 }
                    if matches!(inst.op.name, "s_or_b32" | "s_and_b32" | "s_andn2_b32")
                        && let Some(origin) = {
                            let origin = |r: u16| {
                                (r < TRACKED_SGPRS)
                                    .then(|| st[r as usize].clone())
                                    .flatten()
                                    .filter(|s| {
                                        matches!(
                                            s,
                                            Source::Buffer { .. }
                                                | Source::Memory { .. }
                                                | Source::UserSgpr(_)
                                        )
                                    })
                            };
                            match (origin(ssrc0), origin(ssrc1)) {
                                (Some(a), None) | (None, Some(a)) => Some(a),
                                (Some(_), Some(_)) if ssrc0 == sdst as u16 => origin(ssrc0),
                                (Some(_), Some(_)) if ssrc1 == sdst as u16 => origin(ssrc1),
                                _ => None,
                            }
                        } =>
                {
                    new.insert(sdst as u16, Some(origin));
                }
                Kind::Sop1 { sdst, ssrc0 }
                    if inst.op.name == "s_mov_b64" && ssrc0 + 1 < SGPR_COUNT =>
                {
                    new.insert(sdst as u16, st[ssrc0 as usize].clone());
                    new.insert(sdst as u16 + 1, st[ssrc0 as usize + 1].clone());
                }
                Kind::Sop2 { sdst, ssrc0, ssrc1 }
                    if inst.op.name == "s_lshl_b32"
                        && ssrc0 < TRACKED_SGPRS
                        && (src::ZERO..=src::ZERO + 31).contains(&ssrc1) =>
                {
                    let shift = (ssrc1 - src::ZERO) as u8;
                    let scaled = match &st[ssrc0 as usize] {
                        Some(of @ (Source::UserSgpr(_) | Source::Buffer { .. })) => {
                            Some(Source::Scaled {
                                of: Box::new(of.clone()),
                                shift,
                            })
                        }
                        _ => None,
                    };
                    new.insert(sdst as u16, scaled);
                }
                Kind::Sop1 { sdst, .. } if inst.op.name == "s_getpc_b64" => {
                    let next = calls::Expanded::address(&self.runs, pc) + 4;
                    new.insert(sdst as u16, Some(Source::Constant(vec![next as u32])));
                    new.insert(
                        sdst as u16 + 1,
                        Some(Source::Constant(vec![(next >> 32) as u32])),
                    );
                }
                // Code addresses: `s_getpc_b64` plus an offset in two halves.
                Kind::Sop2 { sdst, ssrc0, ssrc1 }
                    if matches!(inst.op.name, "s_add_u32" | "s_addc_u32") =>
                {
                    let known = |st: &Syms, inst: &Inst, r: u16| match st.get(r as usize) {
                        Some(Some(Source::Constant(v))) if v.len() == 1 => Some(v[0]),
                        _ => constant(inst, r),
                    };
                    let sum = |st: &Syms, inst: &Inst, s0, s1| {
                        Some(known(st, inst, s0)? as u64 + known(st, inst, s1)? as u64)
                    };
                    let carry = if inst.op.name == "s_add_u32" {
                        Some(0)
                    } else {
                        i.checked_sub(1)
                            .map(|p| &self.insts[p])
                            .filter(|(_, p)| p.op.name == "s_add_u32")
                            .and_then(|(p_pc, p)| match p.kind {
                                Kind::Sop2 { ssrc0, ssrc1, .. } => {
                                    sum(before.get(p_pc)?, p, ssrc0, ssrc1).map(|s| s >> 32)
                                }
                                _ => None,
                            })
                    };
                    let value = carry
                        .zip(sum(&st, inst, ssrc0, ssrc1))
                        .map(|(c, s)| Source::Constant(vec![(s + c) as u32]));
                    new.insert(sdst as u16, value);
                }
                Kind::Smem {
                    sdata,
                    sbase,
                    offset,
                    soffset,
                    ..
                } if inst.op.name.starts_with("s_buffer_load_dword") => {
                    let vsharp = st.get(sbase as usize).cloned().flatten().filter(|v| {
                        (1..4u16).all(|k| {
                            st.get(sbase as usize + k as usize).cloned().flatten()
                                == Some(v.advance(4 * k as u32))
                        })
                    });
                    let index = match soffset as u16 {
                        src::NULL => Some(None),
                        r if r < TRACKED_SGPRS => match &st[r as usize] {
                            Some(Source::Scaled { of, shift })
                                if !matches!(**of, Source::Constant(_)) =>
                            {
                                Some(Some((of.clone(), *shift)))
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    if let (Some(v), Some(index)) = (vsharp, index) {
                        let n = smem_dwords(inst.op.name);
                        for k in 0..n {
                            new.insert(
                                sdata as u16 + k,
                                Some(Source::Buffer {
                                    buffer: Box::new(v.clone()),
                                    offset: (offset as u32).wrapping_add(4 * k as u32),
                                    index: index.clone(),
                                }),
                            );
                        }
                    }
                }
                Kind::Smem {
                    sdata,
                    sbase,
                    offset,
                    soffset,
                    ..
                } if inst.op.name.starts_with("s_load_dword") && soffset as u16 == src::NULL => {
                    if let Some(p) = pointer(&st, sbase as u16) {
                        let n = smem_dwords(inst.op.name);
                        for k in 0..n {
                            new.insert(
                                sdata as u16 + k,
                                Some(Source::Memory {
                                    pointer: Box::new(p.clone()),
                                    offset: (offset as u32).wrapping_add(4 * k as u32),
                                }),
                            );
                        }
                    }
                }
                _ => {}
            }
            if let Some(w) = self.writes.get(&pc) {
                for &r in w {
                    st[r as usize] = new.remove(&r).flatten();
                }
            }
            let vcc = |d: u8| d == 106 || d == 107;
            let vcc_written = match &inst.kind {
                Kind::Valu(v) => match v.sdst {
                    Some(d) => vcc(d),
                    None => {
                        let n = inst.op.name;
                        (n.starts_with("v_cmp") && !n.starts_with("v_cmpx_"))
                            || n.contains("_co_")
                            || (n.starts_with("v_read") && n.contains("lane") && vcc(v.vdst))
                    }
                },
                _ => false,
            };
            if vcc_written {
                st[106] = None;
                st[107] = None;
            }
            let next = pc + inst.words as usize;
            let name = inst.op.name;
            let mut succ = Vec::new();
            if let Some(t) = branch_target(pc, inst) {
                succ.push(t);
            }
            if name != "s_endpgm" && name != "s_branch" && name != "s_code_end" {
                succ.push(next);
            }
            for t in succ {
                match before.get_mut(&t) {
                    None => {
                        before.insert(t, st.clone());
                        work.push(t);
                    }
                    Some(old) => {
                        let mut changed = false;
                        for (o, n) in old.iter_mut().zip(&st) {
                            if o.is_some() && o != n {
                                *o = None;
                                changed = true;
                            }
                        }
                        if changed {
                            work.push(t);
                        }
                    }
                }
            }
        }
        before
    }
}
