//! Translates a GFX10 shader program to Metal Shading Language.
//!
//! The model is a straight register one, see `docs/shaders.md`: every Metal thread
//! is one lane, SGPRs and VGPRs are local `uint`s, a lane mask holds this lane in
//! bit 0 of its low half, and branches become a `switch` inside a loop.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::decode::{Inst, Kind, Mimg, VEnc, Valu, decode, src};
use crate::opinfo::Fmt;
use crate::print::{Syntax, mimg_addr_count, print};

/// Pipeline stage of a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// A vertex program: the merged NGG ES/GS program in `SPI_SHADER_PGM_LO_ES`, or a legacy VS.
    Vertex,
    Pixel,
    Compute,
}

/// What to translate and the register state the program starts with.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub stage: Stage,
    /// Code from the program address on; the end is found by decoding.
    pub code: &'a [u32],
    /// Number of user SGPRs (`USER_SGPR` of the stage's `SPI_SHADER_PGM_RSRC2`).
    pub user_sgprs: u32,
    /// Vertex stage: the program is a merged NGG one (user SGPRs from s8, vertex index
    /// in v5, instance index in v8) rather than a legacy VS (from s0, v0 and v3).
    pub ngg: bool,
    /// Pixel stage: `SPI_PS_INPUT_ENA` and `SPI_PS_INPUT_ADDR`.
    pub ps_input_ena: u32,
    pub ps_input_addr: u32,
    /// Pixel stage: `SPI_PS_INPUT_CNTL_n`; attribute n reads the vertex export at its
    /// `OFFSET` field. Empty means attribute n reads export n.
    pub ps_input_cntl: &'a [u32],
    /// Pixel stage: `SPI_SHADER_COL_FORMAT`, 4 bits per MRT.
    pub col_format: u32,
}

/// Where a descriptor's dwords come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The descriptor's dwords start at this user SGPR.
    UserSgpr(u8),
    /// A 64-bit pointer in user SGPRs `sgpr`, `sgpr + 1`; the descriptor sits `offset` bytes past it.
    Memory { sgpr: u8, offset: u32 },
}

/// Image dimension a texture is read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    Tex2d,
    Tex2dArray,
    Tex3d,
    Cube,
}

/// A Metal object the host binds for the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resource {
    /// A T#, bound at `[[texture(slot)]]`.
    Texture {
        slot: u8,
        source: Source,
        kind: TextureKind,
    },
    /// An S#, bound at `[[sampler(slot)]]`.
    Sampler { slot: u8, source: Source },
}

/// The translated program.
#[derive(Clone, Debug)]
pub struct Translated {
    /// One MSL library source.
    pub msl: String,
    /// Name of the `vertex`, `fragment` or `kernel` function.
    pub entry: String,
    pub resources: Vec<Resource>,
    /// Vertex stage: number of parameter exports (`float4 pN [[user(locnN)]]`).
    pub vertex_outputs: u32,
    /// Pixel stage: mask of the MRTs written (`float4 cN [[color(N)]]`).
    pub color_outputs: u8,
    /// Pixel stage: the program writes depth.
    pub writes_depth: bool,
    /// The program reads guest memory (scalar loads, buffer loads), so the host binds
    /// the guest memory buffer at `[[buffer(29)]]` and its guest address at `[[buffer(28)]]`.
    pub guest_memory: bool,
}

/// Why a program could not be translated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslateError {
    /// Dword offset of the instruction, if one is at fault.
    pub pc: Option<usize>,
    pub message: String,
}

impl std::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.pc {
            Some(pc) => write!(f, "at dword {pc:#x}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for TranslateError {}

/// Binding indices of the generated MSL.
pub mod binding {
    /// `constant uint* user`: all user SGPR values in order.
    pub const USER: u32 = 0;
    /// `constant ulong& mem_base`: guest address of byte 0 of the memory buffer.
    pub const MEM_BASE: u32 = 28;
    /// `device uchar* mem`: guest memory.
    pub const MEM: u32 = 29;
}

fn err(pc: usize, message: impl Into<String>) -> TranslateError {
    TranslateError {
        pc: Some(pc),
        message: message.into(),
    }
}

/// Decodes from the start of `code` to the `s_endpgm` that no branch jumps past.
pub fn program(code: &[u32]) -> Result<Vec<(usize, Inst)>, TranslateError> {
    let mut out = Vec::new();
    let mut pc = 0;
    let mut furthest = 0;
    loop {
        let inst =
            decode(&code[pc.min(code.len())..]).map_err(|e| err(pc, format!("decode: {e:?}")))?;
        let name = inst.op.name;
        let next = pc + inst.words as usize;
        if let Some(t) = branch_target(pc, &inst) {
            furthest = furthest.max(t);
        }
        out.push((pc, inst));
        if (name == "s_endpgm" || name == "s_code_end") && pc >= furthest {
            return Ok(out);
        }
        pc = next;
        if pc >= code.len() {
            return Err(err(pc, "code ends before s_endpgm"));
        }
    }
}

fn branch_target(pc: usize, inst: &Inst) -> Option<usize> {
    let name = inst.op.name;
    match inst.kind {
        Kind::Sopp { simm16 } if name == "s_branch" || name.starts_with("s_cbranch") => {
            Some((pc as i64 + 1 + simm16 as i16 as i64) as usize)
        }
        _ => None,
    }
}

/// Translates a program.
pub fn translate(req: &Request) -> Result<Translated, TranslateError> {
    let insts = program(req.code)?;
    let mut tx = Tx::new(req, insts);
    tx.run()?;
    Ok(tx.finish())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Interp {
    Perspective,
    PerspectiveCentroid,
    PerspectiveSample,
    Linear,
    LinearCentroid,
    LinearSample,
    Flat,
}

struct Tx<'a> {
    req: &'a Request<'a>,
    insts: Vec<(usize, Inst)>,
    body: String,
    regs: BTreeSet<String>,
    written_sgprs: BTreeSet<u16>,
    /// Descriptor register, dimension and where the descriptor comes from,
    /// as of the first instruction reading it.
    textures: Vec<(u16, TextureKind, Source)>,
    samplers: Vec<(u16, Source)>,
    attrs: BTreeMap<u8, Interp>,
    params: BTreeSet<u8>,
    mrts: u8,
    depth: bool,
    mem: bool,
    ij_vgprs: BTreeMap<u16, Interp>,
    user_base: u16,
}

const SGPR_COUNT: u16 = 106;

impl<'a> Tx<'a> {
    fn new(req: &'a Request<'a>, insts: Vec<(usize, Inst)>) -> Self {
        let user_base = if req.stage == Stage::Vertex && req.ngg {
            8
        } else {
            0
        };
        let mut ij_vgprs = BTreeMap::new();
        if req.stage == Stage::Pixel {
            const SIZES: [u16; 7] = [2, 2, 2, 3, 2, 2, 2];
            const MODES: [Option<Interp>; 7] = [
                Some(Interp::PerspectiveSample),
                Some(Interp::Perspective),
                Some(Interp::PerspectiveCentroid),
                None,
                Some(Interp::LinearSample),
                Some(Interp::Linear),
                Some(Interp::LinearCentroid),
            ];
            let mut reg = 0u16;
            for bit in 0..7 {
                if req.ps_input_addr >> bit & 1 != 0 {
                    if let Some(m) = MODES[bit] {
                        ij_vgprs.insert(reg, m);
                        ij_vgprs.insert(reg + 1, m);
                    }
                    reg += SIZES[bit];
                }
            }
        }
        Tx {
            req,
            insts,
            body: String::new(),
            regs: BTreeSet::new(),
            written_sgprs: BTreeSet::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            attrs: BTreeMap::new(),
            params: BTreeSet::new(),
            mrts: 0,
            depth: false,
            mem: false,
            ij_vgprs,
            user_base,
        }
    }

    fn line(&mut self, s: &str) {
        self.body.push_str("        ");
        self.body.push_str(s);
        self.body.push('\n');
    }

    // The local holding scalar register `code` (32 bits).
    fn s(&mut self, code: u16) -> String {
        let name = match code {
            0..=105 => format!("s{code}"),
            106 => "vcc_lo".into(),
            107 => "vcc_hi".into(),
            108..=123 => format!("ttmp{}", code - 108),
            124 => "m0".into(),
            125 => return "0u".into(),
            126 => "exec_lo".into(),
            127 => "exec_hi".into(),
            _ => format!("<bad sreg {code}>"),
        };
        self.regs.insert(name.clone());
        name
    }

    fn v(&mut self, n: u16) -> String {
        let name = format!("v{n}");
        self.regs.insert(name.clone());
        name
    }

    fn sdst(&mut self, code: u16) -> Option<String> {
        if code < 106 {
            self.written_sgprs.insert(code);
        }
        (code != src::NULL).then(|| self.s(code))
    }

    // A 32-bit source operand as a `uint` expression; `f16` picks 16-bit float constants.
    fn src32(
        &mut self,
        code: u16,
        literal: Option<u32>,
        pc: usize,
    ) -> Result<String, TranslateError> {
        Ok(match code {
            0..=127 => self.s(code),
            128..=192 => format!("{}u", code - 128),
            193..=208 => format!("0x{:x}u", (192 - code as i32) as u32),
            240..=248 => format!(
                "0x{:x}u",
                [
                    0x3F00_0000u32,
                    0xBF00_0000,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x4000_0000,
                    0xC000_0000,
                    0x4080_0000,
                    0xC080_0000,
                    0x3E22_F983
                ][code as usize - 240]
            ),
            src::VCCZ => {
                let (lo, hi) = (self.s(106), self.s(107));
                format!("uint(({lo} | {hi}) == 0u)")
            }
            src::EXECZ => {
                let (lo, hi) = (self.s(126), self.s(127));
                format!("uint(({lo} | {hi}) == 0u)")
            }
            src::SCC => {
                self.regs.insert("scc".into());
                "uint(scc)".into()
            }
            src::LITERAL => format!(
                "0x{:x}u",
                literal.ok_or_else(|| err(pc, "missing literal"))?
            ),
            256..=511 => self.v(code - 256),
            _ => return Err(err(pc, format!("unsupported operand {code}"))),
        })
    }

    // A 64-bit source operand as a `ulong` expression.
    fn src64(
        &mut self,
        code: u16,
        literal: Option<u32>,
        pc: usize,
    ) -> Result<String, TranslateError> {
        Ok(match code {
            0..=104 | 106 | 108..=122 | 126 => {
                let (lo, hi) = (self.s(code), self.s(code + 1));
                format!("(ulong({hi}) << 32 | ulong({lo}))")
            }
            125 => "0ul".into(),
            128..=192 => format!("{}ul", code - 128),
            193..=208 => format!("0x{:x}ul", (192 - code as i64) as u64),
            src::LITERAL => format!(
                "0x{:x}ul",
                literal.ok_or_else(|| err(pc, "missing literal"))?
            ),
            _ => {
                let v = self.src32(code, literal, pc)?;
                format!("ulong({v})")
            }
        })
    }

    fn set64(&mut self, code: u16, value: &str) {
        if code == src::NULL {
            return;
        }
        if code < 105 {
            self.written_sgprs.insert(code);
            self.written_sgprs.insert(code + 1);
        }
        let (lo, hi) = (self.s(code), self.s(code + 1));
        self.line(&format!(
            "{{ ulong t = {value}; {lo} = uint(t); {hi} = uint(t >> 32); }}"
        ));
    }

    fn run(&mut self) -> Result<(), TranslateError> {
        let mut leaders: BTreeSet<usize> = BTreeSet::new();
        leaders.insert(0);
        for (pc, inst) in &self.insts {
            if let Some(t) = branch_target(*pc, inst) {
                leaders.insert(t);
                leaders.insert(pc + 1);
            }
        }
        let structured = leaders.len() > 1;
        let insts = std::mem::take(&mut self.insts);
        for (pc, inst) in &insts {
            if structured && leaders.contains(pc) {
                self.body.push_str(&format!("    case {pc}:\n"));
            }
            let text = print(inst, Syntax { wave64: true });
            self.line(&format!("// {text}"));
            self.inst(*pc, inst)?;
        }
        self.insts = insts;
        if structured {
            self.body = format!(
                "    uint pc = 0;\n    for (;;) {{\n    switch (pc) {{\n{}    default: return o;\n    }}\n    }}\n",
                self.body
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
            Kind::Mubuf(b) => self.mubuf(pc, name, &b),
            Kind::Mimg(m) => self.mimg(pc, name, &m),
            _ => Err(err(pc, format!("unsupported instruction {name}"))),
        }
    }

    fn sopp(&mut self, pc: usize, inst: &Inst, _simm16: u16) -> Result<(), TranslateError> {
        let name = inst.op.name;
        match name {
            "s_endpgm" => {
                self.end_program();
            }
            "s_code_end" | "s_nop" | "s_waitcnt" | "s_inst_prefetch" | "s_clause" | "s_sendmsg"
            | "s_waitcnt_depctr" | "s_barrier" | "s_setprio" | "s_sleep" | "s_icache_inv" => {}
            "s_branch" => {
                let t = branch_target(pc, inst).unwrap();
                self.line(&format!("pc = {t}; continue;"));
            }
            n if n.starts_with("s_cbranch") => {
                let t = branch_target(pc, inst).unwrap();
                let cond = match n {
                    "s_cbranch_scc0" => "!scc".to_string(),
                    "s_cbranch_scc1" => "scc".to_string(),
                    "s_cbranch_vccz" | "s_cbranch_vccnz" => {
                        let (lo, hi) = (self.s(106), self.s(107));
                        let op = if n.ends_with("nz") { "!=" } else { "==" };
                        format!("({lo} | {hi}) {op} 0u")
                    }
                    "s_cbranch_execz" | "s_cbranch_execnz" => {
                        let (lo, hi) = (self.s(126), self.s(127));
                        let op = if n.ends_with("nz") { "!=" } else { "==" };
                        format!("({lo} | {hi}) {op} 0u")
                    }
                    _ => return Err(err(pc, format!("unsupported branch {n}"))),
                };
                if cond.contains("scc") {
                    self.regs.insert("scc".into());
                }
                self.line(&format!("if ({cond}) {{ pc = {t}; continue; }}"));
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        }
        Ok(())
    }

    fn end_program(&mut self) {
        match self.req.stage {
            Stage::Pixel => self.line("if (!exported) discard_fragment(); return o;"),
            Stage::Vertex => self.line("return o;"),
            Stage::Compute => self.line("return;"),
        }
    }

    fn sop1(
        &mut self,
        pc: usize,
        name: &str,
        sdst: u16,
        s0: u16,
        lit: Option<u32>,
    ) -> Result<(), TranslateError> {
        let b64 = name.ends_with("_b64");
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
        if b64 {
            let a = self.src64(s0, lit, pc)?;
            let (e, scc) = match name {
                "s_mov_b64" => (a, false),
                "s_not_b64" => (format!("~{a}"), true),
                "s_wqm_b64" => (format!("wqm64({a})"), true),
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

    fn sop2(
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

    fn sopc(
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

    fn sopk(
        &mut self,
        pc: usize,
        name: &str,
        sdst: u16,
        simm16: u16,
    ) -> Result<(), TranslateError> {
        let k = simm16 as i16 as i32;
        match name {
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

    fn smem(
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

    fn guard(&mut self, stmt: &str) {
        let e = self.s(126);
        self.line(&format!("if ({e} & 1u) {{ {stmt} }}"));
    }

    // Source `i` of a VALU instruction as a `float` expression, with its modifiers.
    fn fsrc(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        let u = self.usrc(pc, inst, v, i)?;
        let mut e = format!("F({u})");
        if v.abs >> i & 1 != 0 {
            e = format!("fabs({e})");
        }
        if v.neg >> i & 1 != 0 {
            e = format!("(-{e})");
        }
        Ok(e)
    }

    // Source `i` as a `uint` expression, after any SDWA selection.
    fn usrc(
        &mut self,
        pc: usize,
        inst: &Inst,
        v: &Valu,
        i: usize,
    ) -> Result<String, TranslateError> {
        let e = self.src32(v.src[i], inst.literal, pc)?;
        if v.enc == VEnc::Sdwa && i < 2 {
            let sel = v.sdwa.src_sel[i];
            let sext = v.sdwa.sext[i];
            return Ok(match (sel, sext) {
                (6, _) => e,
                (0..=3, false) => format!("(({e} >> {}) & 0xffu)", 8 * sel),
                (0..=3, true) => format!("uint(int(char({e} >> {})))", 8 * sel),
                (4 | 5, false) => format!("(({e} >> {}) & 0xffffu)", 16 * (sel - 4)),
                (4 | 5, true) => format!("uint(int(short({e} >> {})))", 16 * (sel - 4)),
                _ => return Err(err(pc, "bad SDWA select")),
            });
        }
        Ok(e)
    }

    fn lane_bit(&mut self, pc: usize, v: &Valu, i: usize) -> Result<String, TranslateError> {
        let code = v.src[i];
        Ok(format!("({} & 1u)", self.src32(code, None, pc)?))
    }

    fn valu(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let op = inst.op;
        let name = op.name;
        if matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) {
            return Err(err(pc, format!("unsupported DPP form of {name}")));
        }
        if v.enc == VEnc::Sdwa && op.fmt != Fmt::Vopc && (v.sdwa.dst_sel != 6) {
            return Err(err(pc, "unsupported SDWA destination select"));
        }
        if op.fmt == Fmt::Vopc {
            return self.vopc(pc, inst, v);
        }
        if name == "v_nop" || name == "v_illegal" || name == "v_pipeflush" {
            return Ok(());
        }
        if name == "v_readfirstlane_b32" || name == "v_readlane_b32" {
            let s = self.usrc(pc, inst, v, 0)?;
            if let Some(d) = self.sdst(v.vdst as u16) {
                self.line(&format!("{d} = {s};"));
            }
            return Ok(());
        }
        let f = |t: &mut Self, i: usize| t.fsrc(pc, inst, v, i);
        let u = |t: &mut Self, i: usize| t.usrc(pc, inst, v, i);
        let k = || format!("F(0x{:x}u)", inst.literal.unwrap_or(0));
        let vdst = v.vdst as u16;
        let old = format!("F(v{vdst})");
        self.regs.insert(format!("v{vdst}"));
        let fl: Option<String> = Some(match name {
            "v_mov_b32" => {
                let a = u(self, 0)?;
                let d = self.v(vdst);
                self.guard(&format!("{d} = {a};"));
                return Ok(());
            }
            "v_add_f32" => format!("{} + {}", f(self, 0)?, f(self, 1)?),
            "v_sub_f32" => format!("{} - {}", f(self, 0)?, f(self, 1)?),
            "v_subrev_f32" => format!("{} - {}", f(self, 1)?, f(self, 0)?),
            "v_mul_f32" => format!("{} * {}", f(self, 0)?, f(self, 1)?),
            "v_mul_legacy_f32" => {
                let (a, b) = (f(self, 0)?, f(self, 1)?);
                format!("(({a}) == 0.0f || ({b}) == 0.0f) ? 0.0f : ({a}) * ({b})")
            }
            "v_max_f32" => format!("fmax({}, {})", f(self, 0)?, f(self, 1)?),
            "v_min_f32" => format!("fmin({}, {})", f(self, 0)?, f(self, 1)?),
            "v_mac_f32" | "v_mac_legacy_f32" => {
                format!("{} * {} + {old}", f(self, 0)?, f(self, 1)?)
            }
            "v_fmac_f32" => format!("fma({}, {}, {old})", f(self, 0)?, f(self, 1)?),
            "v_madmk_f32" => format!("{} * {} + {}", f(self, 0)?, k(), f(self, 1)?),
            "v_madak_f32" => format!("{} * {} + {}", f(self, 0)?, f(self, 1)?, k()),
            "v_fmamk_f32" => format!("fma({}, {}, {})", f(self, 0)?, k(), f(self, 1)?),
            "v_fmaak_f32" => format!("fma({}, {}, {})", f(self, 0)?, f(self, 1)?, k()),
            "v_mad_f32" | "v_mad_legacy_f32" => {
                format!("{} * {} + {}", f(self, 0)?, f(self, 1)?, f(self, 2)?)
            }
            "v_fma_f32" => format!("fma({}, {}, {})", f(self, 0)?, f(self, 1)?, f(self, 2)?),
            "v_min3_f32" => format!(
                "fmin(fmin({}, {}), {})",
                f(self, 0)?,
                f(self, 1)?,
                f(self, 2)?
            ),
            "v_max3_f32" => format!(
                "fmax(fmax({}, {}), {})",
                f(self, 0)?,
                f(self, 1)?,
                f(self, 2)?
            ),
            "v_med3_f32" => format!("med3({}, {}, {})", f(self, 0)?, f(self, 1)?, f(self, 2)?),
            "v_rcp_f32" | "v_rcp_iflag_f32" => format!("1.0f / {}", f(self, 0)?),
            "v_rsq_f32" => format!("rsqrt({})", f(self, 0)?),
            "v_sqrt_f32" => format!("sqrt({})", f(self, 0)?),
            "v_sin_f32" => format!("sin({} * 6.2831853071795864f)", f(self, 0)?),
            "v_cos_f32" => format!("cos({} * 6.2831853071795864f)", f(self, 0)?),
            "v_exp_f32" => format!("exp2({})", f(self, 0)?),
            "v_log_f32" => format!("log2({})", f(self, 0)?),
            "v_fract_f32" => format!("fract({})", f(self, 0)?),
            "v_floor_f32" => format!("floor({})", f(self, 0)?),
            "v_ceil_f32" => format!("ceil({})", f(self, 0)?),
            "v_trunc_f32" => format!("trunc({})", f(self, 0)?),
            "v_rndne_f32" => format!("rint({})", f(self, 0)?),
            "v_ldexp_f32" => {
                let a = f(self, 0)?;
                let b = u(self, 1)?;
                format!("ldexp({a}, int({b}))")
            }
            "v_cvt_f32_i32" => format!("float(int({}))", u(self, 0)?),
            "v_cvt_f32_u32" => format!("float({})", u(self, 0)?),
            "v_cvt_f32_ubyte0" => format!("float({} & 0xffu)", u(self, 0)?),
            "v_cvt_f32_ubyte1" => format!("float(({} >> 8) & 0xffu)", u(self, 0)?),
            "v_cvt_f32_ubyte2" => format!("float(({} >> 16) & 0xffu)", u(self, 0)?),
            "v_cvt_f32_ubyte3" => format!("float({} >> 24)", u(self, 0)?),
            "v_cvt_f32_f16" => format!("float(as_type<half2>({}).x)", u(self, 0)?),
            _ => String::new(),
        })
        .filter(|s| !s.is_empty());
        if let Some(e) = fl {
            let e = self.output_mods(v, &e);
            let d = self.v(vdst);
            self.guard(&format!("{d} = U({e});"));
            return Ok(());
        }
        let ui = match name {
            "v_cndmask_b32" => {
                let sel = if matches!(v.enc, VEnc::E64) {
                    self.lane_bit(pc, v, 2)?
                } else {
                    format!("({} & 1u)", self.s(106))
                };
                let (a, b) = if v.neg != 0 || v.abs != 0 {
                    (format!("U({})", f(self, 0)?), format!("U({})", f(self, 1)?))
                } else {
                    (u(self, 0)?, u(self, 1)?)
                };
                format!("{sel} != 0u ? {b} : {a}")
            }
            "v_add_nc_u32" => format!("{} + {}", u(self, 0)?, u(self, 1)?),
            "v_sub_nc_u32" => format!("{} - {}", u(self, 0)?, u(self, 1)?),
            "v_subrev_nc_u32" => format!("{} - {}", u(self, 1)?, u(self, 0)?),
            "v_and_b32" => format!("{} & {}", u(self, 0)?, u(self, 1)?),
            "v_or_b32" => format!("{} | {}", u(self, 0)?, u(self, 1)?),
            "v_xor_b32" => format!("{} ^ {}", u(self, 0)?, u(self, 1)?),
            "v_xnor_b32" => format!("~({} ^ {})", u(self, 0)?, u(self, 1)?),
            "v_not_b32" => format!("~{}", u(self, 0)?),
            "v_bfrev_b32" => format!("reverse_bits({})", u(self, 0)?),
            "v_lshlrev_b32" => format!("{} << ({} & 31u)", u(self, 1)?, u(self, 0)?),
            "v_lshrrev_b32" => format!("{} >> ({} & 31u)", u(self, 1)?, u(self, 0)?),
            "v_ashrrev_i32" => format!("uint(int({}) >> ({} & 31u))", u(self, 1)?, u(self, 0)?),
            "v_min_i32" => format!("uint(min(int({}), int({})))", u(self, 0)?, u(self, 1)?),
            "v_max_i32" => format!("uint(max(int({}), int({})))", u(self, 0)?, u(self, 1)?),
            "v_min_u32" => format!("min({}, {})", u(self, 0)?, u(self, 1)?),
            "v_max_u32" => format!("max({}, {})", u(self, 0)?, u(self, 1)?),
            "v_mul_lo_u32" | "v_mul_lo_i32" => format!("{} * {}", u(self, 0)?, u(self, 1)?),
            "v_mul_hi_u32" => format!("mulhi({}, {})", u(self, 0)?, u(self, 1)?),
            "v_mul_u32_u24" => format!(
                "({} & 0xffffffu) * ({} & 0xffffffu)",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_mul_i32_i24" => format!(
                "uint(((int({}) << 8) >> 8) * ((int({}) << 8) >> 8))",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_mad_u32_u24" => format!(
                "({} & 0xffffffu) * ({} & 0xffffffu) + {}",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_bfe_u32" => format!(
                "bfe_u32({}, {} & 31u, {} & 31u)",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_bfe_i32" => format!(
                "bfe_i32({}, {} & 31u, {} & 31u)",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_bfi_b32" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                format!("({a} & {b}) | (~{a} & {c})")
            }
            "v_bfm_b32" => format!(
                "((1u << ({} & 31u)) - 1u) << ({} & 31u)",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_lshl_add_u32" => format!(
                "({} << ({} & 31u)) + {}",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_add_lshl_u32" => format!(
                "({} + {}) << ({} & 31u)",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_lshl_or_b32" => format!(
                "({} << ({} & 31u)) | {}",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_and_or_b32" => format!("({} & {}) | {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_or3_b32" => format!("{} | {} | {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_xor3_b32" => format!("{} ^ {} ^ {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_add3_u32" => format!("{} + {} + {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_alignbit_b32" => format!(
                "uint((ulong({}) << 32 | ulong({})) >> ({} & 31u))",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
            "v_cvt_u32_f32" => format!("uint(clamp({}, 0.0f, 4294967295.0f))", f(self, 0)?),
            "v_cvt_i32_f32" => format!(
                "uint(int(clamp({}, -2147483648.0f, 2147483647.0f)))",
                f(self, 0)?
            ),
            "v_cvt_flr_i32_f32" => format!("uint(int(floor({})))", f(self, 0)?),
            "v_cvt_rpi_i32_f32" => format!("uint(int(floor({} + 0.5f)))", f(self, 0)?),
            "v_cvt_pkrtz_f16_f32" => format!(
                "as_type<uint>(half2(half({}), half({})))",
                f(self, 0)?,
                f(self, 1)?
            ),
            "v_cvt_f16_f32" => format!("uint(as_type<ushort>(half({})))", f(self, 0)?),
            "v_cvt_pknorm_u16_f32" => {
                format!("pack_unorm2x16(float2({}, {}))", f(self, 0)?, f(self, 1)?)
            }
            "v_cvt_pknorm_i16_f32" => {
                format!("pack_snorm2x16(float2({}, {}))", f(self, 0)?, f(self, 1)?)
            }
            "v_cvt_pk_u16_u32" => format!(
                "(min({}, 0xffffu)) | (min({}, 0xffffu) << 16)",
                u(self, 0)?,
                u(self, 1)?
            ),
            "v_bcnt_u32_b32" => format!("popcount({}) + {}", u(self, 0)?, u(self, 1)?),
            "v_ffbh_u32" => {
                let a = u(self, 0)?;
                format!("({a} == 0u ? 0xffffffffu : clz({a}))")
            }
            "v_ffbl_b32" => {
                let a = u(self, 0)?;
                format!("({a} == 0u ? 0xffffffffu : ctz({a}))")
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => u(self, 1)?.to_string(),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let d = self.v(vdst);
        self.guard(&format!("{d} = {ui};"));
        Ok(())
    }

    fn output_mods(&self, v: &Valu, e: &str) -> String {
        let mut e = format!("({e})");
        match v.omod {
            1 => e = format!("({e} * 2.0f)"),
            2 => e = format!("({e} * 4.0f)"),
            3 => e = format!("({e} * 0.5f)"),
            _ => {}
        }
        if v.clamp {
            e = format!("saturate({e})");
        }
        e
    }

    fn vopc(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let cmpx = name.starts_with("v_cmpx_");
        let rest = name
            .trim_start_matches("v_cmpx_")
            .trim_start_matches("v_cmp_");
        let (op, ty) = rest
            .rsplit_once('_')
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        let cond = match ty {
            "f32" => {
                let (a, b) = (self.fsrc(pc, inst, v, 0)?, self.fsrc(pc, inst, v, 1)?);
                fcmp(op, &a, &b)
                    .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?
            }
            "i32" | "u32" => {
                let (a, b) = (self.usrc(pc, inst, v, 0)?, self.usrc(pc, inst, v, 1)?);
                let (a, b) = if ty == "i32" {
                    (format!("int({a})"), format!("int({b})"))
                } else {
                    (a, b)
                };
                match op {
                    "f" => "false".into(),
                    "t" => "true".into(),
                    _ => format!(
                        "{a} {} {b}",
                        cmp_op(op)
                            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?
                    ),
                }
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let e = self.s(126);
        let dst = if cmpx {
            126
        } else {
            match (v.enc, v.sdst) {
                (VEnc::E64, _) => v.vdst as u16,
                (_, Some(s)) => s as u16,
                _ => 106,
            }
        };
        if dst == src::NULL {
            return Ok(());
        }
        if dst < 105 {
            self.written_sgprs.insert(dst);
            self.written_sgprs.insert(dst + 1);
        }
        let (lo, hi) = (self.s(dst), self.s(dst + 1));
        self.line(&format!(
            "{{ bool c = ({cond}) && ({e} & 1u) != 0u; {lo} = uint(c); {hi} = 0u; }}"
        ));
        Ok(())
    }

    fn input_interp(&self, vsrc: u16) -> Interp {
        self.ij_vgprs
            .get(&vsrc)
            .copied()
            .unwrap_or(Interp::Perspective)
    }

    fn vintrp(
        &mut self,
        pc: usize,
        name: &str,
        vdst: u16,
        vsrc: u16,
        attr: u8,
        chan: u8,
    ) -> Result<(), TranslateError> {
        if self.req.stage != Stage::Pixel {
            return Err(err(pc, "interpolation outside a pixel shader"));
        }
        let mode = if name == "v_interp_mov_f32" {
            Interp::Flat
        } else {
            self.input_interp(vsrc)
        };
        match self.attrs.get(&attr) {
            Some(m) if *m != mode => {
                return Err(err(
                    pc,
                    format!("attribute {attr} read with two interpolation modes"),
                ));
            }
            _ => {
                self.attrs.insert(attr, mode);
            }
        }
        if name == "v_interp_mov_f32" && vsrc > 2 {
            return Err(err(pc, "bad interpolation parameter"));
        }
        let c = ['x', 'y', 'z', 'w'][chan as usize];
        let d = self.v(vdst);
        self.guard(&format!("{d} = U(in.a{attr}.{c});"));
        Ok(())
    }

    fn exp(
        &mut self,
        pc: usize,
        en: u8,
        target: u8,
        compr: bool,
        done: bool,
        vsrc: [u8; 4],
    ) -> Result<(), TranslateError> {
        let mut comps = Vec::new();
        for i in 0..4 {
            let reg = if compr { vsrc[i / 2] } else { vsrc[i] } as u16;
            comps.push((en >> i & 1 != 0, self.v(reg)));
        }
        let float4 = |c: &[(bool, String)]| {
            let parts: Vec<String> = c
                .iter()
                .map(|(on, r)| {
                    if *on {
                        format!("F({r})")
                    } else {
                        "0.0f".into()
                    }
                })
                .collect();
            format!("float4({})", parts.join(", "))
        };
        let stmt = match (self.req.stage, target) {
            (Stage::Vertex, 12) => format!("o.pos = {};", float4(&comps)),
            (Stage::Vertex, 13..=16) | (Stage::Vertex, 20) | (_, 9) => String::new(),
            (Stage::Vertex, 32..=63) => {
                let n = target - 32;
                self.params.insert(n);
                format!("o.p{n} = {};", float4(&comps))
            }
            (Stage::Pixel, 0..=7) => {
                self.mrts |= 1 << target;
                let fmt = self.req.col_format >> (4 * target) & 0xF;
                let value = if compr {
                    let (lo, hi) = (&comps[0].1, &comps[2].1);
                    match fmt {
                        5 => format!(
                            "float4(unpack_unorm2x16_to_float({lo}), unpack_unorm2x16_to_float({hi}))"
                        ),
                        6 => format!(
                            "float4(unpack_snorm2x16_to_float({lo}), unpack_snorm2x16_to_float({hi}))"
                        ),
                        7 => format!(
                            "float4(float({lo} & 0xffffu), float({lo} >> 16), float({hi} & 0xffffu), float({hi} >> 16))"
                        ),
                        8 => format!(
                            "float4(float(short({lo})), float(short({lo} >> 16)), float(short({hi})), float(short({hi} >> 16)))"
                        ),
                        _ => format!(
                            "float4(float2(as_type<half2>({lo})), float2(as_type<half2>({hi})))"
                        ),
                    }
                } else {
                    float4(&comps)
                };
                format!("o.c{target} = {value};")
            }
            (Stage::Pixel, 8) => {
                self.depth = true;
                format!("o.depth = F({});", comps[0].1)
            }
            _ => return Err(err(pc, format!("unsupported export target {target}"))),
        };
        let e = self.s(126);
        if self.req.stage == Stage::Pixel && done {
            self.line(&format!("if (({e} & 1u) == 0u) discard_fragment();"));
        }
        if !stmt.is_empty() {
            self.line(&format!("if ({e} & 1u) {{ {stmt} }}"));
        }
        if self.req.stage == Stage::Pixel {
            self.line("exported = true;");
        }
        Ok(())
    }

    fn mubuf(
        &mut self,
        pc: usize,
        name: &str,
        b: &crate::decode::Buf,
    ) -> Result<(), TranslateError> {
        let n: u16 = match name {
            "buffer_load_format_x" => 1,
            "buffer_load_format_xy" => 2,
            "buffer_load_format_xyz" => 3,
            "buffer_load_format_xyzw" => 4,
            "buffer_load_dword" => 1,
            "buffer_load_dwordx2" => 2,
            "buffer_load_dwordx3" => 3,
            "buffer_load_dwordx4" => 4,
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        if b.lds || b.tfe {
            return Err(err(pc, "unsupported LDS or TFE buffer load"));
        }
        self.mem = true;
        let r = b.srsrc as u16;
        let vs: Vec<String> = (0..4).map(|i| self.s(r + i)).collect();
        let index = if b.idxen {
            self.v(b.vaddr as u16)
        } else {
            "0u".into()
        };
        let voff = if b.offen {
            self.v(b.vaddr as u16 + b.idxen as u16)
        } else {
            "0u".into()
        };
        let so = self.src32(b.soffset as u16, None, pc)?;
        let call = if name.contains("format") {
            "buffer_load_format"
        } else {
            "buffer_load_raw"
        };
        let mut s = format!(
            "uint4 r = {call}(mem, mem_base, uint4({}, {}, {}, {}), {index}, {voff} + {so} + {}u, {}u);",
            vs[0], vs[1], vs[2], vs[3], b.offset, b.idxen as u32
        );
        for i in 0..n {
            let d = self.v(b.vdata as u16 + i);
            write!(s, " {d} = r[{i}];").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    fn descriptor_source(&self, pc: usize, first: u16, n: u16) -> Result<Source, TranslateError> {
        let base = self.user_base;
        let users = self.req.user_sgprs as u16;
        let user = first >= base && first + n <= base + users;
        let written = (first..first + n).any(|r| self.written_sgprs.contains(&r));
        if user && !written {
            Ok(Source::UserSgpr((first - base) as u8))
        } else {
            Err(err(pc, "descriptor not in unmodified user SGPRs"))
        }
    }

    fn mimg(&mut self, pc: usize, name: &str, m: &Mimg) -> Result<(), TranslateError> {
        let variant = name
            .strip_prefix("image_sample")
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        if m.d16 || m.a16 || m.tfe || m.lwe {
            return Err(err(pc, "unsupported image modifiers"));
        }
        let kind = match m.dim {
            1 => TextureKind::Tex2d,
            5 => TextureKind::Tex2dArray,
            2 => TextureKind::Tex3d,
            3 => TextureKind::Cube,
            _ => return Err(err(pc, "unsupported image dimension")),
        };
        let t_source = self.descriptor_source(pc, m.srsrc as u16, 8)?;
        let t = match self.textures.iter().position(|x| x.0 == m.srsrc as u16 && x.2 == t_source) {
            Some(t) => t,
            None => {
                self.textures.push((m.srsrc as u16, kind, t_source));
                self.textures.len() - 1
            }
        };
        if self.textures[t].1 != kind {
            return Err(err(pc, "texture read with two dimensions"));
        }
        let s_source = self.descriptor_source(pc, m.ssamp as u16, 4)?;
        let sm = match self.samplers.iter().position(|&x| x == (m.ssamp as u16, s_source)) {
            Some(i) => i,
            None => {
                self.samplers.push((m.ssamp as u16, s_source));
                self.samplers.len() - 1
            }
        };
        let n = mimg_addr_count(name, m.dim, m.a16) as usize;
        let addr: Vec<String> = (0..n)
            .map(|i| {
                if m.nsa > 0 {
                    m.vaddr[i] as u16
                } else {
                    m.vaddr[0] as u16 + i as u16
                }
            })
            .map(|r| format!("F({})", self.v(r)))
            .collect();
        let (coords_at, level) = match variant {
            "" => (0, None),
            "_lz" => (0, Some("level(0.0f)".to_string())),
            "_l" => (0, Some(format!("level({})", addr[addr.len() - 1]))),
            "_b" => (1, Some(format!("bias({})", addr[0]))),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let c = &addr[coords_at..];
        let coord = match kind {
            TextureKind::Tex2d => format!("float2({}, {})", c[0], c[1]),
            TextureKind::Tex2dArray => format!("float2({}, {}), uint(rint({}))", c[0], c[1], c[2]),
            TextureKind::Tex3d => format!("float3({}, {}, {})", c[0], c[1], c[2]),
            TextureKind::Cube => return Err(err(pc, "cube sampling is not done yet")),
        };
        let level = match (level, self.req.stage) {
            (Some(l), _) => format!(", {l}"),
            (None, Stage::Pixel) => String::new(),
            (None, _) => ", level(0.0f)".into(),
        };
        let mut s = format!("float4 r = t{t}.sample(smp{sm}, {coord}{level});");
        let mut d = m.vdata as u16;
        for i in 0..4 {
            if m.dmask >> i & 1 != 0 {
                let r = self.v(d);
                write!(s, " {r} = U(r[{i}]);").unwrap();
                d += 1;
            }
        }
        self.guard(&s);
        Ok(())
    }

    fn finish(mut self) -> Translated {
        let mut resources = Vec::new();
        for (i, &(_, kind, source)) in self.textures.iter().enumerate() {
            resources.push(Resource::Texture {
                slot: i as u8,
                source,
                kind,
            });
        }
        for (i, &(_, source)) in self.samplers.iter().enumerate() {
            resources.push(Resource::Sampler {
                slot: i as u8,
                source,
            });
        }
        let msl = self.emit();
        let entry = match self.req.stage {
            Stage::Vertex => "gen5_vs",
            Stage::Pixel => "gen5_ps",
            Stage::Compute => "gen5_cs",
        };
        Translated {
            msl,
            entry: entry.into(),
            resources,
            vertex_outputs: self.params.iter().next_back().map_or(0, |&p| p as u32 + 1),
            color_outputs: self.mrts,
            writes_depth: self.depth,
            guest_memory: self.mem,
        }
    }

    fn emit(&mut self) -> String {
        let mut out = String::from(PRELUDE);
        let stage = self.req.stage;
        let mut params = vec!["constant uint* user".to_string()];
        let mut args = vec!["user".to_string()];
        if self.mem {
            params.push("device const uchar* mem".into());
            params.push("ulong mem_base".into());
            args.push("mem".into());
            args.push("mem_base".into());
        }
        let mut entry_params = vec![format!("constant uint* user [[buffer({})]]", binding::USER)];
        if self.mem {
            entry_params.push(format!(
                "device const uchar* mem [[buffer({})]]",
                binding::MEM
            ));
            entry_params.push(format!(
                "constant ulong& mem_base [[buffer({})]]",
                binding::MEM_BASE
            ));
        }
        for (i, &(_, kind, _)) in self.textures.iter().enumerate() {
            let ty = match kind {
                TextureKind::Tex2d => "texture2d<float>",
                TextureKind::Tex2dArray => "texture2d_array<float>",
                TextureKind::Tex3d => "texture3d<float>",
                TextureKind::Cube => "texturecube<float>",
            };
            params.push(format!("{ty} t{i}"));
            args.push(format!("t{i}"));
            entry_params.push(format!("{ty} t{i} [[texture({i})]]"));
        }
        for i in 0..self.samplers.len() {
            params.push(format!("sampler smp{i}"));
            args.push(format!("smp{i}"));
            entry_params.push(format!("sampler smp{i} [[sampler({i})]]"));
        }

        let mut init = String::new();
        let users = self.req.user_sgprs as u16;
        for r in self.regs.clone() {
            init.push_str(
                &format!("    uint {r} = 0u;\n").replace("uint scc = 0u", "bool scc = false"),
            );
        }
        for i in 0..users {
            let r = self.user_base + i;
            if r < SGPR_COUNT && self.regs.contains(&format!("s{r}")) {
                init.push_str(&format!("    s{r} = user[{i}];\n"));
            }
        }
        if self.regs.contains("exec_lo") {
            init.push_str("    exec_lo = 1u;\n");
        }
        match stage {
            Stage::Vertex => {
                out.push_str("struct VsOut {\n    float4 pos [[position]];\n");
                for p in &self.params {
                    out.push_str(&format!("    float4 p{p} [[user(locn{p})]];\n"));
                }
                out.push_str("};\n\n");
                let (vid, iid) = if self.req.ngg { (5, 8) } else { (0, 3) };
                for (reg, val) in [(format!("v{vid}"), "vid"), (format!("v{iid}"), "iid")] {
                    if self.regs.contains(&reg) {
                        init.push_str(&format!("    {reg} = {val};\n"));
                    }
                }
                if self.req.ngg && self.regs.contains("s3") {
                    init.push_str("    s3 = 0x101u;\n");
                }
                out.push_str(&format!(
                    "static VsOut gen5_vs_body(uint vid, uint iid, {}) {{\n    VsOut o = {{}};\n{init}{}    return o;\n}}\n\n",
                    params.join(", "),
                    self.body
                ));
                out.push_str(&format!(
                    "vertex VsOut gen5_vs(uint vid [[vertex_id]], uint iid [[instance_id]], {}) {{\n    return gen5_vs_body(vid, iid, {});\n}}\n",
                    entry_params.join(", "),
                    args.join(", ")
                ));
            }
            Stage::Pixel => {
                out.push_str("struct PsIn {\n    float4 pos [[position]];\n    bool front [[front_facing]];\n");
                for (&a, &mode) in &self.attrs {
                    let loc = self
                        .req
                        .ps_input_cntl
                        .get(a as usize)
                        .map_or(a as u32, |c| c & 0x1F);
                    let q = match mode {
                        Interp::Perspective => "center_perspective",
                        Interp::PerspectiveCentroid => "centroid_perspective",
                        Interp::PerspectiveSample => "sample_perspective",
                        Interp::Linear => "center_no_perspective",
                        Interp::LinearCentroid => "centroid_no_perspective",
                        Interp::LinearSample => "sample_no_perspective",
                        Interp::Flat => "flat",
                    };
                    out.push_str(&format!("    float4 a{a} [[user(locn{loc}), {q}]];\n"));
                }
                out.push_str("};\n\nstruct PsOut {\n");
                for i in 0..8 {
                    if self.mrts >> i & 1 != 0 {
                        out.push_str(&format!("    float4 c{i} [[color({i})]];\n"));
                    }
                }
                if self.depth {
                    out.push_str("    float depth [[depth(any)]];\n");
                }
                out.push_str("};\n\n");
                init.push_str(&self.ps_vgprs());
                if self.regs.contains("m0") {
                    init.push_str("    m0 = 0u;\n");
                }
                out.push_str(&format!(
                    "fragment PsOut gen5_ps(PsIn in [[stage_in]], {}) {{\n    PsOut o = {{}};\n    bool exported = false;\n{init}{}    if (!exported) discard_fragment();\n    return o;\n}}\n",
                    entry_params.join(", "),
                    self.body
                ));
            }
            Stage::Compute => {
                out.push_str(&format!(
                    "kernel void gen5_cs({}) {{\n{init}{}}}\n",
                    entry_params.join(", "),
                    self.body
                ));
            }
        }
        out
    }

    // Initial VGPRs of a pixel shader, laid out by `SPI_PS_INPUT_ADDR`.
    fn ps_vgprs(&self) -> String {
        let mut s = String::new();
        let mut reg = 0u16;
        const SIZES: [u16; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        for (bit, &n) in SIZES.iter().enumerate() {
            if self.req.ps_input_addr >> bit & 1 == 0 {
                continue;
            }
            let on = self.req.ps_input_ena >> bit & 1 != 0;
            let value = match bit {
                8 => Some("U(in.pos.x)"),
                9 => Some("U(in.pos.y)"),
                10 => Some("U(in.pos.z)"),
                11 => Some("U(in.pos.w)"),
                12 => Some("(in.front ? 0x3f800000u : 0xbf800000u)"),
                _ => None,
            };
            if let Some(v) = value.filter(|_| on && self.regs.contains(&format!("v{reg}"))) {
                s.push_str(&format!("    v{reg} = {v};\n"));
            }
            reg += n;
        }
        s
    }
}

fn cmp_op(op: &str) -> Option<&'static str> {
    Some(match op {
        "eq" => "==",
        "lg" | "ne" => "!=",
        "gt" => ">",
        "ge" => ">=",
        "lt" => "<",
        "le" => "<=",
        _ => return None,
    })
}

fn fcmp(op: &str, a: &str, b: &str) -> Option<String> {
    Some(match op {
        "f" => "false".into(),
        "tru" => "true".into(),
        "o" => format!("(!isnan({a}) && !isnan({b}))"),
        "u" => format!("(isnan({a}) || isnan({b}))"),
        "lg" => format!("(({a}) < ({b}) || ({a}) > ({b}))"),
        "neq" => format!("!(({a}) == ({b}))"),
        "nge" => format!("!(({a}) >= ({b}))"),
        "nlg" => format!("!(({a}) < ({b}) || ({a}) > ({b}))"),
        "ngt" => format!("!(({a}) > ({b}))"),
        "nle" => format!("!(({a}) <= ({b}))"),
        "nlt" => format!("!(({a}) < ({b}))"),
        _ => format!("({a}) {} ({b})", cmp_op(op)?),
    })
}

const PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;

#define F(x) as_type<float>(uint(x))
#define U(x) as_type<uint>(float(x))

static inline uint bfe_u32(uint v, uint o, uint w) {
    w = min(w, 32u);
    if (w == 0u) return 0u;
    uint m = w >= 32u ? 0xffffffffu : ((1u << w) - 1u);
    return (v >> (o & 31u)) & m;
}

static inline uint bfe_i32(uint v, uint o, uint w) {
    w = min(w, 32u);
    if (w == 0u) return 0u;
    int s = int(v << (32u - min(32u, (o & 31u) + w))) >> (32u - w);
    return uint(s);
}

static inline ulong bfe_u64(ulong v, uint ctrl) {
    uint o = ctrl & 63u;
    uint w = min((ctrl >> 16) & 0x7fu, 64u);
    if (w == 0u) return 0ul;
    ulong m = w >= 64u ? ~0ul : ((1ul << w) - 1ul);
    return (v >> o) & m;
}

static inline ulong wqm64(ulong v) {
    ulong r = 0ul;
    for (uint q = 0u; q < 16u; q++) {
        if ((v >> (4u * q)) & 0xful) r |= 0xful << (4u * q);
    }
    return r;
}

static inline float med3(float a, float b, float c) {
    return fmax(fmin(a, b), fmin(fmax(a, b), c));
}

static inline uint ld32(device const uchar* mem, ulong base, ulong addr) {
    return *(device const uint*)(mem + (addr - base));
}

static inline uint4 buffer_load_raw(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    ulong base = ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32);
    uint stride = (vs.y >> 16) & 0x3fffu;
    if (stride != 0u ? index >= vs.z : offset >= vs.z) return uint4(0u);
    ulong a = base + ulong(index) * ulong(stride) * ulong(idxen) + ulong(offset);
    device const uint* p = (device const uint*)(mem + (a - mem_base));
    return uint4(p[0], p[1], p[2], p[3]);
}

static inline uint fmt_sel(uint sel, uint4 v, uint one) {
    switch (sel) {
    case 1u: return one;
    case 4u: return v.x;
    case 5u: return v.y;
    case 6u: return v.z;
    case 7u: return v.w;
    default: return 0u;
    }
}

// Buffer formats (GFX10 unified encoding, BUF_FMT_*), as raw result dwords.
static inline uint4 buffer_load_format(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    ulong base = ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32);
    uint stride = (vs.y >> 16) & 0x3fffu;
    if (stride != 0u ? index >= vs.z : offset >= vs.z) return uint4(0u);
    ulong a = base + ulong(index) * ulong(stride) * ulong(idxen) + ulong(offset);
    device const uchar* p = mem + (a - mem_base);
    uint fmt = (vs.w >> 12) & 0x7fu;
    uint one = 0x3f800000u;
    uint4 r = uint4(0u);
    switch (fmt) {
    case 20u: case 21u: one = 1u; r.x = *(device const uint*)p; break;
    case 22u: r.x = *(device const uint*)p; break;
    case 62u: case 63u: one = 1u; r.xy = *(device const packed_uint2*)p; break;
    case 64u: r.xy = *(device const packed_uint2*)p; break;
    case 72u: case 73u: one = 1u; r.xyz = *(device const packed_uint3*)p; break;
    case 74u: r.xyz = *(device const packed_uint3*)p; break;
    case 75u: case 76u: one = 1u; r = *(device const packed_uint4*)p; break;
    case 77u: r = *(device const packed_uint4*)p; break;
    case 1u: r.x = as_type<uint>(float(p[0]) / 255.0f); break;
    case 14u: r.xy = as_type<uint2>(float2(p[0], p[1]) / 255.0f); break;
    case 56u: r = as_type<uint4>(float4(p[0], p[1], p[2], p[3]) / 255.0f); break;
    case 57u: r = as_type<uint4>(max(float4(char4(p[0], p[1], p[2], p[3])) / 127.0f, -1.0f)); break;
    case 60u: one = 1u; r = uint4(p[0], p[1], p[2], p[3]); break;
    case 5u: one = 1u; r.x = p[0]; break;
    case 18u: one = 1u; r.xy = uint2(p[0], p[1]); break;
    case 13u: r.x = as_type<uint>(float(*(device const half*)p)); break;
    case 29u: r.xy = as_type<uint2>(float2(*(device const packed_half2*)p)); break;
    case 71u: r = as_type<uint4>(float4(*(device const packed_half4*)p)); break;
    case 23u: r.xy = as_type<uint2>(float2(*(device const packed_ushort2*)p) / 65535.0f); break;
    case 65u: r = as_type<uint4>(float4(*(device const packed_ushort4*)p) / 65535.0f); break;
    case 27u: one = 1u; r.xy = uint2(*(device const packed_ushort2*)p); break;
    case 69u: one = 1u; r = uint4(*(device const packed_ushort4*)p); break;
    default: break;
    }
    return uint4(fmt_sel(vs.w & 7u, r, one), fmt_sel((vs.w >> 3) & 7u, r, one),
                 fmt_sel((vs.w >> 6) & 7u, r, one), fmt_sel((vs.w >> 9) & 7u, r, one));
}

"#;
