//! Calls and jumps (`s_swappc_b64`, `s_call_b64`, `s_setpc_b64`) whose target
//! is known, inlined into one instruction list. A target is known when the
//! program builds it from constants (`s_getpc_b64` plus offsets), when it is
//! relative (`s_call_b64`), or when the renderer read it for the dispatch
//! from where [`call_sites`] says it comes from ([`Request::calls`]).

use std::collections::HashMap;

use super::*;

/// The most code a program may have once its calls are inlined, in dwords.
const MAX_DWORDS: usize = 1 << 16;
/// How deep calls may nest.
const MAX_DEPTH: usize = 8;
/// `s_branch` with a zero offset; a return becomes one.
const S_BRANCH: u32 = 0xBF82_0000;

/// A call or jump whose target is in SGPRs that hold no constant: the
/// address of the instruction and where the 64-bit target comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallSite {
    pub site: u64,
    pub target: Source,
}

/// The target of the call or jump at `site` for one dispatch, and the code
/// there (up to its end, or as much as the renderer could read).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Callee {
    pub site: u64,
    pub target: u64,
    pub code: Vec<u32>,
}

/// The calls and jumps through SGPRs in `req`'s program (and in the code it
/// calls that `req.calls` already gives) whose target is known only per
/// dispatch. The renderer reads each target from its source, then passes it
/// with the code there in `Request::calls`; called code can hold calls of
/// its own, which a further call then reports.
pub fn call_sites(req: &Request) -> Result<Vec<CallSite>, TranslateError> {
    resolve(req).map(|(_, sites)| sites)
}

/// A program with its calls inlined.
pub(super) struct Expanded {
    pub(super) insts: Vec<(usize, Inst)>,
    /// The first pc and guest address of each run of instructions that lie
    /// one after another in guest memory.
    pub(super) runs: Vec<(usize, u64)>,
    /// Calls and jumps whose target is not known: pc, address, the SGPR pair.
    pub(super) open: Vec<(usize, u64, u16)>,
}

impl Expanded {
    /// The guest address of the instruction at `pc`.
    pub(super) fn address(runs: &[(usize, u64)], pc: usize) -> u64 {
        let i = runs.partition_point(|&(start, _)| start <= pc);
        let (start, address) = runs[i.saturating_sub(1)];
        address + 4 * (pc - start) as u64
    }
}

/// Inlines every call whose target is known, following constant targets
/// until no new one turns up, and lists the targets known only per dispatch.
pub(super) fn resolve(req: &Request) -> Result<(Expanded, Vec<CallSite>), TranslateError> {
    let mut targets: BTreeMap<u64, u64> = req.calls.iter().map(|c| (c.site, c.target)).collect();
    let mut unreachable = BTreeSet::new();
    for _ in 0..=MAX_DEPTH {
        let mut exp = expand(req, &targets)?;
        exp.open.retain(|(_, site, _)| !unreachable.contains(site));
        if exp.open.is_empty() {
            return Ok((exp, Vec::new()));
        }
        let mut dry = Tx::new(req, exp.insts.clone());
        dry.runs = exp.runs.clone();
        // Sources before a site do not depend on what the dry run refuses after it.
        let _ = dry.run();
        let syms = dry.sgpr_sources();
        let mut sites = Vec::new();
        let mut found = false;
        for &(pc, site, _) in &exp.open {
            if !syms.contains_key(&pc) {
                unreachable.insert(site);
                found = true;
            }
        }
        dry.syms = Some(syms);
        for &(pc, site, reg) in &exp.open {
            match dry.descriptor_source(pc, reg, 2) {
                Ok(Source::Constant(v)) => {
                    let target = v[0] as u64 | (v[1] as u64) << 32;
                    if code_at(req, target).is_some() {
                        targets.insert(site, target);
                        found = true;
                    } else {
                        sites.push(CallSite {
                            site,
                            target: Source::Constant(v),
                        });
                    }
                }
                Ok(target) => sites.push(CallSite { site, target }),
                Err(_) => {}
            }
        }
        if !found {
            return Ok((exp, sites));
        }
    }
    Err(at_address(req.address, "calls nested too deep"))
}

fn at_address(address: u64, message: impl std::fmt::Display) -> TranslateError {
    TranslateError {
        pc: None,
        message: format!("at {address:#x}: {message}"),
    }
}

/// The code at `address` the request holds: its program's or a callee's.
fn code_at<'r>(req: &Request<'r>, address: u64) -> Option<&'r [u32]> {
    let offset = address.checked_sub(req.address)?;
    if offset % 4 == 0 && offset / 4 < req.code.len() as u64 {
        return Some(&req.code[offset as usize / 4..]);
    }
    req.calls
        .iter()
        .find(|c| c.target == address)
        .map(|c| &c.code[..])
}

/// How a call or jump moves on.
enum Transfer {
    /// To code that returns to the address it writes to the SGPR pair `ret`.
    Call { ret: u16, to: To },
    /// To code that does not come back here.
    Jump(u16),
    /// Back to the instruction after the call.
    Return,
}

enum To {
    Sgprs(u16),
    /// Dwords past the next instruction.
    Relative(i64),
}

// `s_swappc_b64` that discards the return address is `s_setpc_b64`; GTA VI
// returns that way.
fn transfer(inst: &Inst, ret: Option<u16>) -> Option<Transfer> {
    match (inst.op.name, inst.kind) {
        ("s_swappc_b64", Kind::Sop1 { sdst, ssrc0 }) if sdst as u16 == src::NULL => {
            Some(if ret == Some(ssrc0) {
                Transfer::Return
            } else {
                Transfer::Jump(ssrc0)
            })
        }
        ("s_swappc_b64", Kind::Sop1 { sdst, ssrc0 }) => Some(Transfer::Call {
            ret: sdst as u16,
            to: To::Sgprs(ssrc0),
        }),
        ("s_call_b64", Kind::Sopk { sdst, simm16 }) => Some(Transfer::Call {
            ret: sdst as u16,
            to: To::Relative(simm16 as i16 as i64),
        }),
        ("s_setpc_b64", Kind::Sop1 { ssrc0, .. }) if ret == Some(ssrc0) => Some(Transfer::Return),
        ("s_setpc_b64", Kind::Sop1 { ssrc0, .. }) => Some(Transfer::Jump(ssrc0)),
        _ => None,
    }
}

struct Entry {
    inst: Inst,
    /// Which inlined copy of a piece of code it belongs to.
    copy: usize,
    /// Its dword offset in that code.
    local: usize,
    address: u64,
    /// A return: the end marker of the call it returns from.
    returns_to: Option<usize>,
}

struct Builder<'r> {
    req: &'r Request<'r>,
    targets: &'r BTreeMap<u64, u64>,
    entries: Vec<Entry>,
    /// Per call, the index of the entry after the inlined code.
    ends: Vec<usize>,
    copies: usize,
    dwords: usize,
    open: Vec<(usize, u64, u16)>,
}

impl<'r> Builder<'r> {
    /// Appends the code at `address`, which the call at `site` reaches;
    /// `ret` is the SGPR pair a return reads and the end marker of the call
    /// it returns from.
    fn code(
        &mut self,
        site: u64,
        address: u64,
        ret: Option<(u16, usize)>,
        depth: usize,
    ) -> Result<(), TranslateError> {
        if depth > MAX_DEPTH {
            return Err(at_address(site, "calls nested too deep"));
        }
        let code = code_at(self.req, address)
            .ok_or_else(|| at_address(site, format!("no code at the target {address:#x}")))?;
        let insts = program(code).map_err(|e| match e.pc {
            Some(pc) if depth > 0 => at_address(address + 4 * pc as u64, e.message),
            _ => e,
        })?;
        let copy = self.copies;
        self.copies += 1;
        for (local, inst) in insts {
            let at = address + 4 * local as u64;
            self.dwords += inst.words as usize;
            if self.dwords > MAX_DWORDS {
                return Err(at_address(site, "too much code once calls are inlined"));
            }
            let here = self.entries.len();
            let mut entry = Entry {
                inst,
                copy,
                local,
                address: at,
                returns_to: None,
            };
            match transfer(&inst, ret.map(|r| r.0)) {
                Some(Transfer::Return) => {
                    entry.inst = decode(&[S_BRANCH]).expect("s_branch decodes");
                    entry.returns_to = ret.map(|r| r.1);
                    self.entries.push(entry);
                }
                Some(Transfer::Call { ret: r, to }) => {
                    self.entries.push(entry);
                    let target = match to {
                        To::Relative(k) => Some(at.wrapping_add_signed(4 + 4 * k)),
                        To::Sgprs(_) => self.targets.get(&at).copied(),
                    };
                    match (target, to) {
                        (Some(t), _) => {
                            let end = self.ends.len();
                            self.ends.push(0);
                            self.code(at, t, Some((r, end)), depth + 1)?;
                            self.ends[end] = self.entries.len();
                        }
                        (None, To::Sgprs(s)) => self.open.push((here, at, s)),
                        (None, To::Relative(_)) => unreachable!("relative targets are known"),
                    }
                }
                Some(Transfer::Jump(s)) => {
                    self.entries.push(entry);
                    match self.targets.get(&at) {
                        Some(&t) => self.code(at, t, ret, depth + 1)?,
                        None => self.open.push((here, at, s)),
                    }
                }
                None => self.entries.push(entry),
            }
        }
        Ok(())
    }
}

// Each return becomes a branch past its inlined code.
fn expand(req: &Request, targets: &BTreeMap<u64, u64>) -> Result<Expanded, TranslateError> {
    let mut b = Builder {
        req,
        targets,
        entries: Vec::new(),
        ends: Vec::new(),
        copies: 0,
        dwords: 0,
        open: Vec::new(),
    };
    b.code(req.address, req.address, None, 0)?;
    if b.copies == 1 {
        let open = b
            .open
            .iter()
            .map(|&(i, address, reg)| (b.entries[i].local, address, reg))
            .collect();
        let insts = b.entries.into_iter().map(|e| (e.local, e.inst)).collect();
        return Ok(Expanded {
            insts,
            runs: vec![(0, req.address)],
            open,
        });
    }
    let mut pcs = Vec::with_capacity(b.entries.len() + 1);
    let mut pc = 0;
    for e in &b.entries {
        pcs.push(pc);
        pc += e.inst.words as usize;
    }
    pcs.push(pc);
    let at: HashMap<(usize, usize), usize> = b
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| ((e.copy, e.local), i))
        .collect();
    let mut runs: Vec<(usize, u64)> = Vec::new();
    let mut insts = Vec::with_capacity(b.entries.len());
    for (i, e) in b.entries.iter().enumerate() {
        let mut inst = e.inst;
        let target = match e.returns_to {
            Some(end) => Some(b.ends[end]),
            None => branch_target(e.local, &inst)
                .map(|t| {
                    at.get(&(e.copy, t))
                        .copied()
                        .ok_or_else(|| at_address(e.address, "branch out of called code"))
                })
                .transpose()?,
        };
        if let (Some(t), Kind::Sopp { simm16 }) = (target, &mut inst.kind) {
            let offset = pcs[t] as i64 - pcs[i] as i64 - 1;
            *simm16 = i16::try_from(offset)
                .map_err(|_| at_address(e.address, "branch too far once calls are inlined"))?
                as u16;
        }
        let follows = i > 0 && {
            let p = &b.entries[i - 1];
            p.address + 4 * p.inst.words as u64 == e.address
        };
        if !follows {
            runs.push((pcs[i], e.address));
        }
        insts.push((pcs[i], inst));
    }
    let open = b
        .open
        .into_iter()
        .map(|(i, address, reg)| (pcs[i], address, reg))
        .collect();
    Ok(Expanded { insts, runs, open })
}
