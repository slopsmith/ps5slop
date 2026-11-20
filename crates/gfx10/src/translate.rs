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
    /// Compute stage: `COMPUTE_PGM_RSRC2`, which says which workgroup ids and
    /// thread ids the program starts with.
    pub compute_rsrc2: u32,
    /// Compute stage: `COMPUTE_NUM_THREAD_X/Y/Z`, the threadgroup size.
    pub threads: [u32; 3],
    /// Texture slots (bit n for slot n of [`Translated::resources`]) whose image
    /// format is unsigned or signed integer, declared `<uint>` or `<int>`.
    /// Slot numbers do not depend on these masks.
    pub uint_textures: u32,
    pub sint_textures: u32,
}

/// Where a descriptor's dwords come from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// The descriptor's dwords start at this user SGPR.
    UserSgpr(u8),
    /// The dwords sit `offset` bytes past a 64-bit pointer, which is the first
    /// two dwords at `pointer`.
    Memory { pointer: Box<Source>, offset: u32 },
}

impl Source {
    /// The source of the dwords `bytes` further on.
    fn advance(&self, bytes: u32) -> Source {
        match self {
            Source::UserSgpr(n) => Source::UserSgpr(n.wrapping_add((bytes / 4) as u8)),
            Source::Memory { pointer, offset } => Source::Memory {
                pointer: pointer.clone(),
                offset: offset.wrapping_add(bytes),
            },
        }
    }
}

/// Image dimension a texture is read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureKind {
    Tex2d,
    Tex2dArray,
    Tex3d,
    Cube,
}

/// How a program uses a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Sampled or read (`access::sample`).
    Sample,
    /// Only written (`access::write`).
    Write,
    /// Read and written (`access::read_write`).
    ReadWrite,
}

/// A Metal object the host binds for the program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resource {
    /// A T#, bound at `[[texture(slot)]]`.
    Texture {
        slot: u8,
        source: Source,
        kind: TextureKind,
        access: Access,
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
    /// The program also writes guest memory (buffer stores and atomics).
    pub writes_memory: bool,
    /// Compute stage: bytes of threadgroup memory (LDS) the kernel declares.
    pub lds_bytes: u32,
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
    let mut dry = Tx::new(req, insts.clone());
    dry.run()?;
    let syms = dry.sgpr_sources();
    let mut tx = Tx::new(req, insts);
    tx.syms = Some(syms);
    tx.run()?;
    Ok(tx.finish())
}

/// What each SGPR holds before an instruction, where it is a descriptor dword
/// whose origin is known.
type Syms = Vec<Option<Source>>;

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
    /// SGPRs each instruction writes, by dword offset.
    writes: BTreeMap<usize, BTreeSet<u16>>,
    pc: usize,
    /// SGPR origins before each instruction, from a first pass; `None` in that pass.
    syms: Option<BTreeMap<usize, Syms>>,
    /// Descriptor register, dimension, source and access, as of the first
    /// instruction using it.
    textures: Vec<(u16, TextureKind, Source, Access)>,
    samplers: Vec<(u16, Source)>,
    attrs: BTreeMap<u8, Interp>,
    params: BTreeSet<u8>,
    mrts: u8,
    depth: bool,
    mem: bool,
    mem_write: bool,
    lds: bool,
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
            writes: BTreeMap::new(),
            pc: 0,
            syms: None,
            textures: Vec::new(),
            samplers: Vec::new(),
            attrs: BTreeMap::new(),
            params: BTreeSet::new(),
            mrts: 0,
            depth: false,
            mem: false,
            mem_write: false,
            lds: false,
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

    fn written(&mut self, code: u16) {
        if code < SGPR_COUNT {
            self.writes.entry(self.pc).or_default().insert(code);
        }
    }

    fn sdst(&mut self, code: u16) -> Option<String> {
        self.written(code);
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
                let lo = self.s(106);
                format!("uint(({lo} & 1u) == 0u)")
            }
            src::EXECZ => {
                let lo = self.s(126);
                format!("uint(({lo} & 1u) == 0u)")
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
        self.written(code);
        self.written(code + 1);
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
            self.pc = *pc;
            let text = print(inst, Syntax { wave64: true });
            self.line(&format!("// {text}"));
            self.inst(*pc, inst)?;
        }
        self.insts = insts;
        if structured {
            self.body = format!(
                "    uint pc = 0;\n    for (;;) {{\n    switch (pc) {{\n{}    default: {}\n    }}\n    }}\n",
                self.body,
                if self.req.stage == Stage::Compute { "return;" } else { "return o;" }
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
            | "s_waitcnt_depctr" | "s_setprio" | "s_sleep" | "s_icache_inv" => {}
            "s_barrier" => {
                if self.req.stage == Stage::Compute {
                    self.line("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
                }
            }
            "s_branch" => {
                let t = branch_target(pc, inst).unwrap();
                self.line(&format!("pc = {t}; continue;"));
            }
            n if n.starts_with("s_cbranch") => {
                let t = branch_target(pc, inst).unwrap();
                let cond = match n {
                    "s_cbranch_scc0" => "!scc".to_string(),
                    "s_cbranch_scc1" => "scc".to_string(),
                    "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz"
                    | "s_cbranch_execnz" => {
                        let mask = if n.contains("vcc") { self.s(106) } else { self.s(126) };
                        let any = if self.req.stage == Stage::Compute {
                            format!("simd_any(({mask} & 1u) != 0u)")
                        } else {
                            format!("(({mask} & 1u) != 0u)")
                        };
                        if n.ends_with("nz") { any } else { format!("!{any}") }
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
        if i == 0 && matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) {
            return Ok("dpp0".into());
        }
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

    /// An SDWA instruction writing a byte or word of its destination: the
    /// whole result, then merged into the old value per `dst_unused`.
    fn sdwa_dst(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let (shift, mask) = match v.sdwa.dst_sel {
            sel @ 0..=3 => (8 * sel as u32, 0xffu32),
            sel @ (4 | 5) => (16 * (sel as u32 - 4), 0xffff),
            _ => return Err(err(pc, "bad SDWA destination select")),
        };
        let d = self.v(v.vdst as u16);
        let old = format!("sdwa_old{pc}");
        self.line("{");
        self.line(&format!("uint {old} = {d};"));
        let mut whole = *v;
        whole.sdwa.dst_sel = 6;
        self.valu(pc, inst, &whole)?;
        let field = format!("(({d} & 0x{mask:x}u) << {shift}u)");
        let merged = match v.sdwa.dst_unused {
            0 => field,
            1 => {
                let top = shift + mask.count_ones();
                if top == 32 {
                    field
                } else {
                    format!("uint(int({field} << {}u) >> {}u)", 32 - top, 32 - top)
                }
            }
            _ => format!("(({old} & ~(0x{mask:x}u << {shift}u)) | {field})"),
        };
        self.guard(&format!("{d} = {merged};"));
        self.line("}");
        Ok(())
    }

    fn lane_bit(&mut self, pc: usize, v: &Valu, i: usize) -> Result<String, TranslateError> {
        let code = v.src[i];
        Ok(format!("({} & 1u)", self.src32(code, None, pc)?))
    }

    fn valu(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let op = inst.op;
        let name = op.name;
        if matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) {
            self.dpp_src0(pc, inst, v)?;
        }
        if v.enc == VEnc::Sdwa && op.fmt != Fmt::Vopc && v.sdwa.dst_sel != 6 {
            return self.sdwa_dst(pc, inst, v);
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
        if matches!(
            name,
            "v_add_co_u32"
                | "v_sub_co_u32"
                | "v_subrev_co_u32"
                | "v_add_co_ci_u32"
                | "v_sub_co_ci_u32"
                | "v_subrev_co_ci_u32"
        ) {
            return self.carry_op(pc, inst, v);
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
            "v_med3_i32" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                let ui = format!("uint(med3_i(int({a}), int({b}), int({c})))");
                let d = self.v(vdst);
                self.guard(&format!("{d} = {ui};"));
                return Ok(());
            }
            "v_med3_u32" => {
                let (a, b, c) = (u(self, 0)?, u(self, 1)?, u(self, 2)?);
                let ui = format!("max(min({a}, {b}), min(max({a}, {b}), {c}))");
                let d = self.v(vdst);
                self.guard(&format!("{d} = {ui};"));
                return Ok(());
            }
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
            "v_cvt_off_f32_i4" => format!("float(int({} << 28) >> 28) / 16.0f", u(self, 0)?),
            "v_cvt_f32_ubyte1" => format!("float(({} >> 8) & 0xffu)", u(self, 0)?),
            "v_cvt_f32_ubyte2" => format!("float(({} >> 16) & 0xffu)", u(self, 0)?),
            "v_cvt_f32_ubyte3" => format!("float({} >> 24)", u(self, 0)?),
            "v_cvt_f32_f16" => format!("float(as_type<half2>({}).x)", u(self, 0)?),
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => format!(
                "cube_{}({}, {}, {})",
                &name[6..8],
                f(self, 0)?,
                f(self, 1)?,
                f(self, 2)?
            ),
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
            "v_mul_hi_i32" => format!("uint(mulhi(int({}), int({})))", u(self, 0)?, u(self, 1)?),
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

    // A DPP quad permutation that reads one lane of the quad becomes a quad
    // broadcast of src0 into `dpp0`, which then stands in for src0.
    fn dpp_src0(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let ctrl = v.dpp.ctrl;
        let lane = ctrl & 3;
        let broadcast = v.enc == VEnc::Dpp16
            && ctrl <= 0xFF
            && (0..4).all(|i| (ctrl >> (2 * i)) & 3 == lane)
            && v.dpp.row_mask == 0xF
            && v.dpp.bank_mask == 0xF;
        if !broadcast {
            return Err(err(pc, format!("unsupported DPP form of {}", inst.op.name)));
        }
        let a = self.src32(v.src[0], None, pc)?;
        self.regs.insert("dpp0".into());
        self.line(&format!("dpp0 = quad_broadcast({a}, {lane}u);"));
        Ok(())
    }

    // Integer add and subtract with a carry or borrow out to a lane mask, and maybe in from one.
    fn carry_op(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let name = inst.op.name;
        let e64 = v.enc == VEnc::E64;
        let a = self.usrc(pc, inst, v, 0)?;
        let b = self.usrc(pc, inst, v, 1)?;
        let cin = if !name.contains("_ci_") {
            "0u".to_string()
        } else if e64 {
            self.lane_bit(pc, v, 2)?
        } else {
            format!("({} & 1u)", self.s(106))
        };
        let (a, b) = if name.starts_with("v_subrev") { (b, a) } else { (a, b) };
        let (value, carry) = if name.starts_with("v_add") {
            ("ulong(a) + ulong(b) + ulong(c)", "(r >> 32) != 0ul")
        } else {
            ("ulong(a) - ulong(b) - ulong(c)", "ulong(b) + ulong(c) > ulong(a)")
        };
        let dst = if e64 { v.sdst.map_or(106, |s| s as u16) } else { 106 };
        let exec = self.s(126);
        let d = self.v(v.vdst as u16);
        let mut s = format!(
            "{{ uint a = {a}; uint b = {b}; uint c = {cin}; ulong r = {value}; bool on = ({exec} & 1u) != 0u; bool k = ({carry}) && on;"
        );
        if v.clamp {
            return Err(err(pc, "clamped carry arithmetic"));
        }
        write!(s, " if (on) {d} = uint(r);").unwrap();
        if dst != src::NULL {
            self.written(dst);
            self.written(dst + 1);
            let (lo, hi) = (self.s(dst), self.s(dst + 1));
            write!(s, " {lo} = uint(k); {hi} = 0u;").unwrap();
        }
        s.push_str(" }");
        self.line(&s);
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
        self.written(dst);
        self.written(dst + 1);
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
        // SPI_PS_INPUT_CNTL's FLAT_SHADE (bit 10) makes interpolation return the
        // provoking vertex's value whatever the instruction.
        let flat_shade = self.req.ps_input_cntl.get(attr as usize).is_some_and(|c| c & 0x400 != 0);
        let mode = if name == "v_interp_mov_f32" || flat_shade {
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
        if mode == Interp::Flat {
            self.guard(&format!("{d} = in.a{attr}.{c};"));
        } else {
            self.guard(&format!("{d} = U(in.a{attr}.{c});"));
        }
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
                            "uint4({lo} & 0xffffu, {lo} >> 16, {hi} & 0xffffu, {hi} >> 16)"
                        ),
                        8 => format!(
                            "int4(short({lo}), short({lo} >> 16), short({hi}), short({hi} >> 16))"
                        ),
                        _ => format!(
                            "float4(float2(as_type<half2>({lo})), float2(as_type<half2>({hi})))"
                        ),
                    }
                } else {
                    match fmt {
                        7 => format!("uint4({})", comps.iter().map(|(on, r)| if *on { r.clone() } else { "0u".into() }).collect::<Vec<_>>().join(", ")),
                        8 => format!("int4({})", comps.iter().map(|(on, r)| if *on { format!("int({r})") } else { "0".into() }).collect::<Vec<_>>().join(", ")),
                        _ => float4(&comps),
                    }
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
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            return Ok(());
        }
        if b.lds || b.tfe {
            return Err(err(pc, "unsupported LDS or TFE buffer access"));
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
        let args = format!(
            "mem_base, uint4({}, {}, {}, {}), {index}, {voff} + {so} + {}u, {}u",
            vs[0], vs[1], vs[2], vs[3], b.offset, b.idxen as u32
        );
        let vd = b.vdata as u16;
        if let Some(op) = name.strip_prefix("buffer_atomic_") {
            self.mem_write = true;
            let data = self.v(vd);
            let call = match op {
                "add" | "sub" | "and" | "or" | "xor" | "umin" | "umax" | "smin" | "smax" | "swap" => {
                    format!("atomic_rmw_{op}(mem, buffer_offset({args}), {data})")
                }
                "cmpswap" => {
                    let cmp = self.v(vd + 1);
                    format!("atomic_cas(mem, buffer_offset({args}), {data}, {cmp})")
                }
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            let stmt = if b.glc { format!("{data} = {call};") } else { format!("(void){call};") };
            self.guard(&stmt);
            return Ok(());
        }
        let store: Option<(u16, &str)> = match name {
            "buffer_store_dword" => Some((1, "")),
            "buffer_store_dwordx2" => Some((2, "")),
            "buffer_store_dwordx3" => Some((3, "")),
            "buffer_store_dwordx4" => Some((4, "")),
            "buffer_store_byte" => Some((1, "8")),
            "buffer_store_short" => Some((1, "16")),
            _ => None,
        };
        if let Some((n, width)) = store {
            self.mem_write = true;
            let mut s = format!("long o = buffer_offset({args}); if (o >= 0) {{");
            for i in 0..n {
                let v = self.v(vd + i);
                match width {
                    "8" => write!(s, " mem[o] = uchar({v});").unwrap(),
                    "16" => write!(s, " *(device ushort*)(mem + o) = ushort({v});").unwrap(),
                    _ => write!(s, " *(device uint*)(mem + o + {}) = {v};", 4 * i).unwrap(),
                }
            }
            s.push_str(" }");
            self.guard(&s);
            return Ok(());
        }
        let (n, call): (u16, &str) = match name {
            "buffer_load_format_x" => (1, "buffer_load_format"),
            "buffer_load_format_xy" => (2, "buffer_load_format"),
            "buffer_load_format_xyz" => (3, "buffer_load_format"),
            "buffer_load_format_xyzw" => (4, "buffer_load_format"),
            "buffer_load_dword" => (1, "buffer_load_raw"),
            "buffer_load_dwordx2" => (2, "buffer_load_raw"),
            "buffer_load_dwordx3" => (3, "buffer_load_raw"),
            "buffer_load_dwordx4" => (4, "buffer_load_raw"),
            "buffer_load_ubyte" => (1, "buffer_load_ubyte"),
            "buffer_load_sbyte" => (1, "buffer_load_sbyte"),
            "buffer_load_ushort" => (1, "buffer_load_ushort"),
            "buffer_load_sshort" => (1, "buffer_load_sshort"),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        let mut s = format!("uint4 r = {call}(mem, {args});");
        for i in 0..n {
            let d = self.v(vd + i);
            write!(s, " {d} = r[{i}];").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    fn ds(
        &mut self,
        pc: usize,
        name: &str,
        offset0: u8,
        offset1: u8,
        gds: bool,
        regs: [u16; 4],
    ) -> Result<(), TranslateError> {
        if gds || self.req.stage != Stage::Compute {
            return Err(err(pc, format!("unsupported LDS access {name}")));
        }
        self.lds = true;
        let [addr, data0, data1, vdst] = regs;
        let a = self.v(addr);
        let off = offset0 as u32 | (offset1 as u32) << 8;
        let pair = |scale: u32| (offset0 as u32 * scale, offset1 as u32 * scale);
        let d0 = |t: &mut Self, i: u16| t.v(data0 + i);
        let d1 = |t: &mut Self, i: u16| t.v(data1 + i);
        let dst = |t: &mut Self, i: u16| t.v(vdst + i);
        let mut s = format!("uint a = {a};");
        match name {
            "ds_nop" => return Ok(()),
            "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" => {
                let n = match name { "ds_write_b32" => 1, "ds_write_b64" => 2, "ds_write_b96" => 3, _ => 4 };
                for i in 0..n {
                    let v = d0(self, i);
                    write!(s, " lds_wr(lds, a + {}u, {v});", off + 4 * i as u32).unwrap();
                }
            }
            "ds_write2_b32" | "ds_write2st64_b32" | "ds_write2_b64" | "ds_write2st64_b64" => {
                let wide = name.ends_with("b64");
                let scale = if wide { 8 } else { 4 } * if name.contains("st64") { 64 } else { 1 };
                let (o0, o1) = pair(scale);
                for i in 0..(1 + wide as u16) {
                    let (x, y) = (d0(self, i), d1(self, i));
                    write!(s, " lds_wr(lds, a + {}u, {x}); lds_wr(lds, a + {}u, {y});", o0 + 4 * i as u32, o1 + 4 * i as u32).unwrap();
                }
            }
            "ds_write_b8" | "ds_write_b16" => {
                let v = d0(self, 0);
                let ty = if name.ends_with("b8") { "uchar" } else { "ushort" };
                write!(s, " if ((a + {off}u) < LDS_BYTES) *(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u) = {ty}({v});").unwrap();
            }
            "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" => {
                let n = match name { "ds_read_b32" => 1, "ds_read_b64" => 2, "ds_read_b96" => 3, _ => 4 };
                for i in 0..n {
                    write!(s, " uint t{i} = lds_rd(lds, a + {}u);", off + 4 * i as u32).unwrap();
                }
                for i in 0..n {
                    let d = dst(self, i);
                    write!(s, " {d} = t{i};").unwrap();
                }
            }
            "ds_read2_b32" | "ds_read2st64_b32" | "ds_read2_b64" | "ds_read2st64_b64" => {
                let wide = name.ends_with("b64");
                let scale = if wide { 8 } else { 4 } * if name.contains("st64") { 64 } else { 1 };
                let (o0, o1) = pair(scale);
                let n = 1 + wide as u32;
                let mut k = 0;
                for o in [o0, o1] {
                    for i in 0..n {
                        write!(s, " uint t{k} = lds_rd(lds, a + {}u);", o + 4 * i).unwrap();
                        k += 1;
                    }
                }
                for i in 0..k {
                    let d = dst(self, i);
                    write!(s, " {d} = t{i};").unwrap();
                }
            }
            "ds_read_u8" | "ds_read_i8" | "ds_read_u16" | "ds_read_i16" => {
                let ty = match name { "ds_read_u8" => "uchar", "ds_read_i8" => "char", "ds_read_u16" => "ushort", _ => "short" };
                let d = dst(self, 0);
                write!(s, " {d} = (a + {off}u) < LDS_BYTES ? uint(int(*(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u))) : 0u;").unwrap();
            }
            _ => {
                let base = name.strip_prefix("ds_").unwrap();
                let rtn = base.contains("_rtn_");
                let op = base.replace("_rtn", "");
                let v = d0(self, 0);
                let call = match op.as_str() {
                    "add_u32" | "sub_u32" | "and_b32" | "or_b32" | "xor_b32" | "min_u32" | "max_u32" | "min_i32" | "max_i32" | "wrxchg_b32" => {
                        let f = op.split('_').next().unwrap();
                        let f = match (f, op.ends_with("i32")) {
                            ("min", true) => "smin",
                            ("max", true) => "smax",
                            ("min", false) => "umin",
                            ("max", false) => "umax",
                            ("wrxchg", _) => "swap",
                            (f, _) => f,
                        };
                        format!("lds_rmw_{f}(lds, a + {off}u, {v})")
                    }
                    "rsub_u32" => format!("lds_rmw_rsub(lds, a + {off}u, {v})"),
                    "cmpst_b32" => {
                        let new = d1(self, 0);
                        format!("lds_cas(lds, a + {off}u, {new}, {v})")
                    }
                    _ => return Err(err(pc, format!("unsupported instruction {name}"))),
                };
                if rtn {
                    let d = dst(self, 0);
                    write!(s, " {d} = {call};").unwrap();
                } else {
                    write!(s, " (void){call};").unwrap();
                }
            }
        }
        self.guard(&s);
        Ok(())
    }

    fn descriptor_source(&self, pc: usize, first: u16, n: u16) -> Result<Source, TranslateError> {
        let Some(syms) = &self.syms else {
            return Ok(Source::UserSgpr(0));
        };
        let none = || err(pc, "descriptor not from user SGPRs or memory they point to");
        let st = syms.get(&pc).ok_or_else(none)?;
        let at = |r: u16| st.get(r as usize).cloned().flatten();
        let first_src = at(first).ok_or_else(none)?;
        for i in 1..n {
            if at(first + i) != Some(first_src.advance(4 * i as u32)) {
                return Err(none());
            }
        }
        Ok(first_src)
    }

    /// Follows user SGPRs through moves and scalar loads, over every path
    /// through the program, to find where descriptor dwords come from.
    fn sgpr_sources(&self) -> BTreeMap<usize, Syms> {
        let index: BTreeMap<usize, usize> =
            self.insts.iter().enumerate().map(|(i, (pc, _))| (*pc, i)).collect();
        let mut entry: Syms = vec![None; SGPR_COUNT as usize];
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
                Kind::Sop1 { sdst, ssrc0 } if inst.op.name == "s_mov_b64" && ssrc0 + 1 < SGPR_COUNT => {
                    new.insert(sdst as u16, st[ssrc0 as usize].clone());
                    new.insert(sdst as u16 + 1, st[ssrc0 as usize + 1].clone());
                }
                Kind::Smem { sdata, sbase, offset, soffset, .. }
                    if inst.op.name.starts_with("s_load_dword") && soffset as u16 == src::NULL =>
                {
                    if let Some(p) = pointer(&st, sbase as u16) {
                        let n = match inst.op.name {
                            "s_load_dword" => 1,
                            "s_load_dwordx2" => 2,
                            "s_load_dwordx4" => 4,
                            "s_load_dwordx8" => 8,
                            _ => 16,
                        };
                        for k in 0..n {
                            new.insert(sdata as u16 + k, Some(Source::Memory {
                                pointer: Box::new(p.clone()),
                                offset: (offset as u32).wrapping_add(4 * k as u32),
                            }));
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

    // The element type of texture slot `t` and how a component of it becomes a VGPR value.
    fn texel(&self, t: usize) -> (&'static str, &'static str) {
        if self.req.uint_textures >> t & 1 != 0 {
            ("uint", "")
        } else if self.req.sint_textures >> t & 1 != 0 {
            ("int", "uint")
        } else {
            ("float", "U")
        }
    }

    // The texture slot for the T# at `srsrc`, merging how it is accessed.
    fn texture_slot(
        &mut self,
        pc: usize,
        srsrc: u16,
        kind: TextureKind,
        access: Access,
    ) -> Result<usize, TranslateError> {
        let source = self.descriptor_source(pc, srsrc, 8)?;
        let t = match self.textures.iter().position(|x| x.0 == srsrc && x.1 == kind && x.2 == source) {
            Some(t) => t,
            None => {
                self.textures.push((srsrc, kind, source, access));
                self.textures.len() - 1
            }
        };
        let entry = &mut self.textures[t];
        entry.3 = match (entry.3, access) {
            (a, b) if a == b => a,
            (Access::Write, Access::Sample) | (Access::Sample, Access::Write) | (_, Access::ReadWrite) | (Access::ReadWrite, _) => Access::ReadWrite,
            (a, _) => a,
        };
        Ok(t)
    }

    fn mimg(&mut self, pc: usize, name: &str, m: &Mimg) -> Result<(), TranslateError> {
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
        let n = mimg_addr_count(name, m.dim, m.a16) as usize;
        let addr_regs: Vec<u16> = (0..n)
            .map(|i| {
                if m.nsa > 0 {
                    m.vaddr[i] as u16
                } else {
                    m.vaddr[0] as u16 + i as u16
                }
            })
            .collect();
        match name {
            "image_load" | "image_load_mip" | "image_store" | "image_store_mip" => {
                return self.image_rw(pc, name, m, kind, &addr_regs);
            }
            "image_get_resinfo" => return self.image_resinfo(pc, m, kind, &addr_regs),
            _ => {}
        }
        let gather = name.starts_with("image_gather4");
        let variant = name
            .strip_prefix("image_sample")
            .or_else(|| name.strip_prefix("image_gather4"))
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        let t = self.texture_slot(pc, m.srsrc as u16, kind, Access::Sample)?;
        let s_source = self.descriptor_source(pc, m.ssamp as u16, 4)?;
        let sm = match self.samplers.iter().position(|x| *x == (m.ssamp as u16, s_source.clone())) {
            Some(i) => i,
            None => {
                self.samplers.push((m.ssamp as u16, s_source));
                self.samplers.len() - 1
            }
        };
        let addr: Vec<String> = addr_regs.iter().map(|&r| format!("F({})", self.v(r))).collect();
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
            TextureKind::Cube => format!("cube_dir(float3({}, {}, {}))", c[0], c[1], c[2]),
        };
        let level = match (level, self.req.stage) {
            (Some(l), _) => format!(", {l}"),
            (None, Stage::Pixel) => String::new(),
            (None, _) => ", level(0.0f)".into(),
        };
        let (elem, conv) = self.texel(t);
        let mut s = format!("{elem}4 r = t{t}.sample(smp{sm}, {coord}{level});");
        if gather {
            if !matches!(kind, TextureKind::Tex2d | TextureKind::Tex2dArray) || !matches!(variant, "" | "_lz") || m.dmask.count_ones() != 1 {
                return Err(err(pc, format!("unsupported form of {name}")));
            }
            let c = ["x", "y", "z", "w"][m.dmask.trailing_zeros() as usize];
            s = format!("{elem}4 r = t{t}.gather(smp{sm}, {coord}, int2(0), component::{c});");
        }
        let mut d = m.vdata as u16;
        for i in 0..4 {
            if gather || m.dmask >> i & 1 != 0 {
                let r = self.v(d);
                write!(s, " {r} = {conv}(r[{i}]);").unwrap();
                d += 1;
            }
        }
        self.guard(&s);
        Ok(())
    }

    // `image_load`, `image_store` and their `_mip` forms: unfiltered texel access.
    fn image_rw(
        &mut self,
        pc: usize,
        name: &str,
        m: &Mimg,
        kind: TextureKind,
        addr: &[u16],
    ) -> Result<(), TranslateError> {
        let store = name.starts_with("image_store");
        let access = if store { Access::Write } else { Access::Sample };
        let t = self.texture_slot(pc, m.srsrc as u16, kind, access)?;
        let a: Vec<String> = addr.iter().map(|&r| self.v(r)).collect();
        let mip = name.ends_with("_mip");
        let level = |i: usize| if mip { a[i].clone() } else { "0u".into() };
        let coord = match kind {
            TextureKind::Tex2d => format!("uint2({}, {})", a[0], a[1]),
            TextureKind::Tex2dArray => format!("uint2({}, {}), {}", a[0], a[1], a[2]),
            TextureKind::Tex3d => format!("uint3({}, {}, {})", a[0], a[1], a[2]),
            TextureKind::Cube => return Err(err(pc, "cube image access is not done yet")),
        };
        let lvl = match kind {
            TextureKind::Tex2d => level(2),
            _ => level(3),
        };
        let mut d = m.vdata as u16;
        let (elem, conv) = self.texel(t);
        if store {
            let mut comps = Vec::new();
            for i in 0..4 {
                if m.dmask >> i & 1 != 0 {
                    let v = self.v(d);
                    comps.push(match elem {
                        "float" => format!("F({v})"),
                        "int" => format!("int({v})"),
                        _ => v,
                    });
                    d += 1;
                } else {
                    comps.push(if elem == "float" { "0.0f".into() } else { "0".into() });
                }
            }
            self.guard(&format!("t{t}.write({elem}4({}), {coord}, {lvl});", comps.join(", ")));
        } else {
            let mut s = format!("{elem}4 r = t{t}.read({coord}, {lvl});");
            for i in 0..4 {
                if m.dmask >> i & 1 != 0 {
                    let r = self.v(d);
                    write!(s, " {r} = {conv}(r[{i}]);").unwrap();
                    d += 1;
                }
            }
            self.guard(&s);
        }
        Ok(())
    }

    fn image_resinfo(
        &mut self,
        pc: usize,
        m: &Mimg,
        kind: TextureKind,
        addr: &[u16],
    ) -> Result<(), TranslateError> {
        let t = self.texture_slot(pc, m.srsrc as u16, kind, Access::Sample)?;
        let lod = self.v(addr[0]);
        let third = match kind {
            TextureKind::Tex2dArray => format!("t{t}.get_array_size()"),
            TextureKind::Tex3d => format!("t{t}.get_depth(l)"),
            TextureKind::Cube => "6u".into(),
            TextureKind::Tex2d => "1u".into(),
        };
        let mut s = format!(
            "uint l = {lod}; uint4 r = uint4(t{t}.get_width(l), t{t}.get_height(l), {third}, t{t}.get_num_mip_levels());"
        );
        let mut d = m.vdata as u16;
        for i in 0..4 {
            if m.dmask >> i & 1 != 0 {
                let r = self.v(d);
                write!(s, " {r} = r[{i}];").unwrap();
                d += 1;
            }
        }
        self.guard(&s);
        Ok(())
    }

    fn finish(mut self) -> Translated {
        let mut resources = Vec::new();
        for (i, (_, kind, source, access)) in self.textures.iter().enumerate() {
            resources.push(Resource::Texture {
                slot: i as u8,
                source: source.clone(),
                kind: *kind,
                access: *access,
            });
        }
        for (i, (_, source)) in self.samplers.iter().enumerate() {
            resources.push(Resource::Sampler {
                slot: i as u8,
                source: source.clone(),
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
            writes_memory: self.mem_write,
            lds_bytes: if self.lds { self.lds_bytes() } else { 0 },
        }
    }

    /// Threadgroup memory: `LDS_SIZE` of `COMPUTE_PGM_RSRC2` in 512-byte units,
    /// within what Metal allows.
    fn lds_bytes(&self) -> u32 {
        let declared = (self.req.compute_rsrc2 >> 15 & 0x1FF) * 512;
        if declared == 0 { 32 << 10 } else { declared.clamp(16, 32 << 10) }
    }

    fn emit(&mut self) -> String {
        let mut out = String::from(PRELUDE);
        if self.lds {
            out.push_str(&format!("#define LDS_BYTES {}u\n{LDS_PRELUDE}", self.lds_bytes()));
        }
        let stage = self.req.stage;
        let mem_ty = if self.mem_write { "device uchar*" } else { "device const uchar*" };
        let mut params = vec!["constant uint* user".to_string()];
        let mut args = vec!["user".to_string()];
        if self.mem {
            params.push(format!("{mem_ty} mem"));
            params.push("ulong mem_base".into());
            args.push("mem".into());
            args.push("mem_base".into());
        }
        let mut entry_params = vec![format!("constant uint* user [[buffer({})]]", binding::USER)];
        if self.mem {
            entry_params.push(format!(
                "{mem_ty} mem [[buffer({})]]",
                binding::MEM
            ));
            entry_params.push(format!(
                "constant ulong& mem_base [[buffer({})]]",
                binding::MEM_BASE
            ));
        }
        for (i, &(_, kind, _, access)) in self.textures.iter().enumerate() {
            let base = match kind {
                TextureKind::Tex2d => "texture2d",
                TextureKind::Tex2dArray => "texture2d_array",
                TextureKind::Tex3d => "texture3d",
                TextureKind::Cube => "texturecube",
            };
            let elem = self.texel(i).0;
            let ty = match access {
                Access::Sample => format!("{base}<{elem}>"),
                Access::Write => format!("{base}<{elem}, access::write>"),
                Access::ReadWrite => format!("{base}<{elem}, access::read_write>"),
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
                let loc_of = |a: u8| {
                    self.req
                        .ps_input_cntl
                        .get(a as usize)
                        .map_or(a as u32, |c| c & 0x1F)
                };
                let mut first = BTreeMap::new();
                for &a in self.attrs.keys() {
                    first.entry(loc_of(a)).or_insert(a);
                }
                // Inputs sharing a location read the same export; MSL wants
                // each location once, so those are copied out of `raw`.
                let shared = first.len() != self.attrs.len();
                out.push_str(if shared { "struct PsInRaw {\n" } else { "struct PsIn {\n" });
                out.push_str("    float4 pos [[position]];\n    bool front [[front_facing]];\n");
                for (&a, &mode) in &self.attrs {
                    let loc = loc_of(a);
                    if first[&loc] != a {
                        continue;
                    }
                    let q = match mode {
                        Interp::Perspective => "center_perspective",
                        Interp::PerspectiveCentroid => "centroid_perspective",
                        Interp::PerspectiveSample => "sample_perspective",
                        Interp::Linear => "center_no_perspective",
                        Interp::LinearCentroid => "centroid_no_perspective",
                        Interp::LinearSample => "sample_no_perspective",
                        Interp::Flat => "flat",
                    };
                    // Flat inputs carry raw bits (often integers), which a float
                    // varying may not keep; vertex variants export them as uint4.
                    let ty = if mode == Interp::Flat { "uint4" } else { "float4" };
                    out.push_str(&format!("    {ty} a{a} [[user(locn{loc}), {q}]];\n"));
                }
                if shared {
                    out.push_str("};\n\nstruct PsIn {\n    float4 pos;\n    bool front;\n");
                    let mut fill = String::from("    PsIn in = { raw.pos, raw.front");
                    for &a in self.attrs.keys() {
                        out.push_str(&format!("    float4 a{a};\n"));
                        fill.push_str(&format!(", raw.a{}", first[&loc_of(a)]));
                    }
                    init.insert_str(0, &(fill + " };\n"));
                }
                out.push_str("};\n\nstruct PsOut {\n");
                for i in 0..8 {
                    if self.mrts >> i & 1 != 0 {
                        let ty = match self.req.col_format >> (4 * i) & 0xF {
                            7 => "uint4",
                            8 => "int4",
                            _ => "float4",
                        };
                        out.push_str(&format!("    {ty} c{i} [[color({i})]];\n"));
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
                let mut body = self.body.clone();
                let mut tail = String::from("    if (!exported) discard_fragment();\n    return o;\n");
                // Debugging aid: the first flat input's raw value in colour output 1.
                if std::env::var_os("PS5SLOP_PROBE_FLAT").is_some()
                    && self.mrts & 2 != 0
                    && self.req.col_format >> 4 & 0xF != 7
                    && self.req.col_format >> 4 & 0xF != 8
                    && let Some((&a, _)) = self.attrs.iter().find(|(_, m)| **m == Interp::Flat)
                {
                    let probe = format!(
                        "{{ o.c1 = float4(float(in.a{a}.x & 0xffu) / 255.0, float((in.a{a}.x >> 8) & 0xffu) / 255.0, float((in.a{a}.x >> 16) & 0xffu) / 255.0, 1.0); return o; }}"
                    );
                    body = body.replace("return o;", &probe);
                    tail = tail.replace("return o;", &probe);
                }
                out.push_str(&format!(
                    "fragment PsOut gen5_ps({} [[stage_in]], {}) {{\n    PsOut o = {{}};\n    bool exported = false;\n{init}{}{tail}}}\n",
                    if shared { "PsInRaw raw" } else { "PsIn in" },
                    entry_params.join(", "),
                    body
                ));
            }
            Stage::Compute => {
                init.push_str(&self.cs_inputs());
                let lds = if self.lds {
                    "    threadgroup uint lds[LDS_BYTES / 4];\n"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "kernel void gen5_cs({}, uint3 lid [[thread_position_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {{\n{lds}{init}{}}}\n",
                    entry_params.join(", "),
                    self.body
                ));
            }
        }
        out
    }

    // Workgroup ids after the user SGPRs and thread ids in v0..v2, as `COMPUTE_PGM_RSRC2` enables them.
    fn cs_inputs(&self) -> String {
        let rsrc2 = self.req.compute_rsrc2;
        let mut s = String::new();
        let mut next = self.req.user_sgprs as u16;
        let threads = self.req.threads;
        let ids = [
            (7, "gid.x".to_string()),
            (8, "gid.y".to_string()),
            (9, "gid.z".to_string()),
            (10, format!("{}u", threads[0] * threads[1] * threads[2])),
        ];
        for (bit, value) in ids {
            if rsrc2 >> bit & 1 != 0 {
                if self.regs.contains(&format!("s{next}")) {
                    s.push_str(&format!("    s{next} = {value};\n"));
                }
                next += 1;
            }
        }
        let comps = (rsrc2 >> 11 & 3).min(2);
        for (i, c) in ["x", "y", "z"].iter().enumerate().take(comps as usize + 1) {
            if self.regs.contains(&format!("v{i}")) {
                s.push_str(&format!("    v{i} = lid.{c};\n"));
            }
        }
        s
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

static inline int med3_i(int a, int b, int c) {
    return max(min(a, b), min(max(a, b), c));
}

static inline uint ld32(device const uchar* mem, ulong base, ulong addr) {
    return *(device const uint*)(mem + (addr - base));
}

// Byte offset into the memory buffer of a buffer element, or -1 when out of range.
static inline long buffer_offset(ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    ulong base = ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32);
    uint stride = (vs.y >> 16) & 0x3fffu;
    if (stride != 0u && idxen != 0u ? index >= vs.z : offset >= vs.z) return -1;
    return long(base + ulong(index) * ulong(stride) * ulong(idxen) + ulong(offset) - mem_base);
}

static inline uint4 buffer_load_raw(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    if (o < 0) return uint4(0u);
    device const uint* p = (device const uint*)(mem + o);
    return uint4(p[0], p[1], p[2], p[3]);
}

static inline uint4 buffer_load_ubyte(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o < 0 ? 0u : uint(mem[o]), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sbyte(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o < 0 ? 0u : uint(int(char(mem[o]))), 0u, 0u, 0u);
}

static inline uint4 buffer_load_ushort(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o < 0 ? 0u : uint(*(device const ushort*)(mem + o)), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sshort(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o < 0 ? 0u : uint(int(*(device const short*)(mem + o))), 0u, 0u, 0u);
}

static inline uint atomic_rmw_add(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_add_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_sub(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_sub_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_and(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_and_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_or(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_or_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_xor(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_xor_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_umin(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_min_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_umax(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_max_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_smin(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_min_explicit((device atomic_int*)(mem + o), int(v), memory_order_relaxed));
}

static inline uint atomic_rmw_smax(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_fetch_max_explicit((device atomic_int*)(mem + o), int(v), memory_order_relaxed));
}

static inline uint atomic_rmw_swap(device uchar* mem, long o, uint v) {
    if (o < 0) return 0u;
    return uint(atomic_exchange_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_cas(device uchar* mem, long o, uint v, uint cmp) {
    if (o < 0) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = cmp;
    while (!atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed) && e == cmp) {}
    return e;
}

static inline float cube_ma(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return 2.0f * z;
    if (abs(y) >= abs(x)) return 2.0f * y;
    return 2.0f * x;
}

static inline float cube_id(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? 5.0f : 4.0f;
    if (abs(y) >= abs(x)) return y < 0.0f ? 3.0f : 2.0f;
    return x < 0.0f ? 1.0f : 0.0f;
}

static inline float cube_sc(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? -x : x;
    if (abs(y) >= abs(x)) return x;
    return x < 0.0f ? z : -z;
}

static inline float cube_tc(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return -y;
    if (abs(y) >= abs(x)) return y < 0.0f ? -z : z;
    return -y;
}

// The direction for cube coordinates as image_sample takes them: face
// coordinates in [1, 2] (v_cubesc / |ma| + 1.5) and the face id.
static inline float3 cube_dir(float3 c) {
    float s = (c.x - 1.5f) * 2.0f;
    float t = (c.y - 1.5f) * 2.0f;
    switch (uint(rint(c.z))) {
    case 0u: return float3(1.0f, -t, -s);
    case 1u: return float3(-1.0f, -t, s);
    case 2u: return float3(s, 1.0f, t);
    case 3u: return float3(s, -1.0f, -t);
    case 4u: return float3(s, -t, 1.0f);
    default: return float3(-s, -t, -1.0f);
    }
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

// One component of `bits` bits converted as `kind`: 0 UNORM, 1 SNORM, 2 USCALED,
// 3 SSCALED, 4 UINT, 5 SINT, 6 FLOAT.
static inline uint fmt_conv(uint v, uint bits, uint kind) {
    uint m = bits == 32u ? 0xffffffffu : (1u << bits) - 1u;
    v &= m;
    int s = bits == 32u ? int(v) : int(v << (32u - bits)) >> (32u - bits);
    switch (kind) {
    case 0u: return as_type<uint>(float(v) / float(m));
    case 1u: return as_type<uint>(max(float(s) / float(m >> 1), -1.0f));
    case 2u: return as_type<uint>(float(v));
    case 3u: return as_type<uint>(float(s));
    case 4u: return v;
    case 5u: return uint(s);
    default: return bits == 16u ? as_type<uint>(float(as_type<half>(ushort(v)))) : v;
    }
}

static inline uint ld_le(device const uchar* p, uint bytes) {
    uint v = 0u;
    for (uint i = 0u; i < bytes; i++) v |= uint(p[i]) << (8u * i);
    return v;
}

// Buffer formats (GFX10 unified encoding, BUF_FMT_*), as raw result dwords.
static inline uint4 buffer_load_format(device const uchar* mem, ulong mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    if (o < 0) return uint4(0u);
    device const uchar* p = mem + o;
    uint fmt = (vs.w >> 12) & 0x7fu;
    uint n = 0u, bits = 0u, kind = 0u;
    if (fmt >= 1u && fmt <= 6u) { n = 1u; bits = 8u; kind = fmt - 1u; }
    else if (fmt >= 7u && fmt <= 13u) { n = 1u; bits = 16u; kind = fmt - 7u; }
    else if (fmt >= 14u && fmt <= 19u) { n = 2u; bits = 8u; kind = fmt - 14u; }
    else if (fmt >= 20u && fmt <= 22u) { n = 1u; bits = 32u; kind = fmt - 16u; }
    else if (fmt >= 23u && fmt <= 29u) { n = 2u; bits = 16u; kind = fmt - 23u; }
    else if (fmt >= 50u && fmt <= 55u) { n = 4u; bits = 10u; kind = fmt - 50u; }
    else if (fmt >= 56u && fmt <= 61u) { n = 4u; bits = 8u; kind = fmt - 56u; }
    else if (fmt >= 62u && fmt <= 64u) { n = 2u; bits = 32u; kind = fmt - 58u; }
    else if (fmt >= 65u && fmt <= 71u) { n = 4u; bits = 16u; kind = fmt - 65u; }
    else if (fmt >= 72u && fmt <= 74u) { n = 3u; bits = 32u; kind = fmt - 68u; }
    else if (fmt >= 75u && fmt <= 77u) { n = 4u; bits = 32u; kind = fmt - 71u; }
    uint4 r = uint4(0u);
    if (bits == 10u) {
        uint w = ld_le(p, 4u);
        r = uint4(fmt_conv(w, 10u, kind), fmt_conv(w >> 10, 10u, kind), fmt_conv(w >> 20, 10u, kind), fmt_conv(w >> 30, 2u, kind));
    } else {
        uint bytes = bits / 8u;
        for (uint i = 0u; i < n; i++) r[i] = fmt_conv(ld_le(p + i * bytes, bytes), bits, kind);
    }
    uint one = kind == 4u || kind == 5u ? 1u : 0x3f800000u;
    return uint4(fmt_sel(vs.w & 7u, r, one), fmt_sel((vs.w >> 3) & 7u, r, one),
                 fmt_sel((vs.w >> 6) & 7u, r, one), fmt_sel((vs.w >> 9) & 7u, r, one));
}

"#;

const LDS_PRELUDE: &str = r#"static inline uint lds_rmw_add(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_add_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_sub(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_sub_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_and(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_and_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_or(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_or_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_xor(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_xor_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_umin(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_min_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_umax(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_max_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_smin(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_min_explicit((threadgroup atomic_int*)&lds[a >> 2], int(v), memory_order_relaxed));
}

static inline uint lds_rmw_smax(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_max_explicit((threadgroup atomic_int*)&lds[a >> 2], int(v), memory_order_relaxed));
}

static inline uint lds_rmw_swap(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_exchange_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rd(threadgroup uint* lds, uint a) {
    return a < LDS_BYTES ? lds[a >> 2] : 0u;
}

static inline void lds_wr(threadgroup uint* lds, uint a, uint v) {
    if (a < LDS_BYTES) lds[a >> 2] = v;
}

static inline uint lds_cas(threadgroup uint* lds, uint a, uint v, uint cmp) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = cmp;
    while (!atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed) && e == cmp) {}
    return e;
}

static inline uint lds_rmw_rsub(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &e, v - e, memory_order_relaxed, memory_order_relaxed)) {}
    return e;
}
"#;
