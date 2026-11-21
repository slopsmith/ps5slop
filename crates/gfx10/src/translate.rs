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
    /// The program's guest address, for `s_getpc_b64` (0 when unknown).
    pub address: u64,
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
    /// The dwords sit `offset` bytes, plus `dword << shift` when `index`
    /// is given, into the buffer the V# at `buffer` describes.
    Buffer {
        buffer: Box<Source>,
        offset: u32,
        index: Option<(Box<Source>, u8)>,
    },
    /// One dword (a user SGPR or a buffer dword) shifted left, used as a
    /// buffer load offset.
    Scaled { of: Box<Source>, shift: u8 },
    /// Dwords the program builds from constants.
    Constant(Vec<u32>),
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
            Source::Buffer {
                buffer,
                offset,
                index,
            } => Source::Buffer {
                buffer: buffer.clone(),
                offset: offset.wrapping_add(bytes),
                index: index.clone(),
            },
            Source::Scaled { .. } => Source::Scaled {
                of: Box::new(Source::Constant(Vec::new())),
                shift: 0,
            },
            Source::Constant(v) => Source::Constant(
                v.get(bytes as usize / 4..)
                    .map_or(Vec::new(), <[u32]>::to_vec),
            ),
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
    /// Used by image atomics (`access::read_write`, `uint` or `int` texels):
    /// the host binds a 32-bit integer view made with atomic usage.
    Atomic,
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
    /// The V#s every buffer store and atomic goes through, with whether the
    /// access is indexed; `None` when one of them is not known.
    pub store_buffers: Option<Vec<(Source, bool)>>,
    /// Compute stage: bytes of threadgroup memory (LDS) the kernel declares.
    pub lds_bytes: u32,
    /// The program uses the global data share, bound at `[[buffer(27)]]`.
    pub gds: bool,
    /// Compute stage: the program loops, and gives up after too many backward
    /// branches, counted in the words bound at `[[buffer(25)]]` (zeroed per dispatch).
    pub loop_guard: bool,
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
    /// `constant Win& mem_base`: the guest range the memory buffer covers, a
    /// second range reached at a fixed distance from it (see `goff`), and a
    /// range stores and atomics skip.
    pub const MEM_BASE: u32 = 28;
    /// `device uchar* mem`: guest memory.
    pub const MEM: u32 = 29;
    /// Where in `user` the renderer puts dword 0 of each sampler slot's S#,
    /// after the 32 user SGPRs (for the depth compare function).
    pub const SAMPLER_WORDS: u32 = 32;
    /// `device atomic_uint* gds`: the 64 KiB global data share.
    pub const GDS: u32 = 27;
    /// `device atomic_uint* guard`: two words per dispatch for the loop guard
    /// of compute programs ([`super::Translated::loop_guard`]): backward
    /// branches taken so far in [`super::GUARD_UNIT`]s, and threads that gave up.
    pub const GUARD: u32 = 25;
}

/// Backward branches a vertex or pixel invocation may take before it gives up: a
/// GPU hang makes macOS ignore all of the process's later Metal work.
pub const MAX_BACK_BRANCHES: u32 = 1024;

/// The same for a compute thread. Persistent loops take a turn per batch of
/// work, so this is higher.
pub const MAX_COMPUTE_BACK_BRANCHES: u32 = 1 << 14;

/// Backward branches all threads of one dispatch may take together, in units
/// of [`GUARD_UNIT`], so a runaway in a large grid also ends quickly.
pub const GUARD_BUDGET: u32 = 1 << 20;
pub const GUARD_UNIT: u32 = 1024;

fn err(pc: usize, message: impl Into<String>) -> TranslateError {
    TranslateError {
        pc: Some(pc),
        message: message.into(),
    }
}

/// Decodes from the start of `code` to the first instruction that does not
/// fall through (`s_endpgm`, `s_branch`) and that no branch jumps past.
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
        let ends = matches!(name, "s_endpgm" | "s_code_end" | "s_branch");
        if ends && pc >= furthest {
            return Ok(out);
        }
        pc = next;
        if pc >= code.len() {
            return Err(err(pc, "code ends before s_endpgm"));
        }
    }
}

/// The loops of a program as `(first, last)` block leaders: each backward
/// branch's target and the last block that branches back to it. `None`
/// when two of them overlap without nesting.
fn loop_regions(insts: &[(usize, Inst)], leaders: &BTreeSet<usize>) -> Option<Vec<(usize, usize)>> {
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
    tx.mem_write = dry.mem_write;
    tx.atomic_textures = dry
        .textures
        .iter()
        .enumerate()
        .filter(|(_, t)| t.3 == Access::Atomic)
        .fold(0, |m, (i, _)| m | 1 << i);
    tx.run()?;
    Ok(tx.finish())
}

/// What each SGPR holds before an instruction, where it is a descriptor dword
/// whose origin is known.
type Syms = Vec<Option<Source>>;

/// The value of a scalar source that is an integer inline constant or a
/// literal. A 32-bit literal in a 64-bit move is zero-extended (our reading
/// of the ISA guide); inline constants are sign-extended.
fn constant(inst: &Inst, code: u16) -> Option<u32> {
    match code {
        src::LITERAL => inst.literal,
        c if (src::ZERO..=src::ZERO + 64).contains(&c) => Some((c - src::ZERO) as u32),
        c if (193..=208).contains(&c) => Some((192i32 - c as i32) as u32),
        _ => None,
    }
}

/// Dwords a scalar memory load reads, from its name's suffix.
fn smem_dwords(name: &str) -> u16 {
    match name.rsplit_once("_dword").map_or("", |(_, x)| x) {
        "" => 1,
        "x2" => 2,
        "x4" => 4,
        "x8" => 8,
        _ => 16,
    }
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
    has_barrier: bool,
    /// A compute program with backward branches (see [`Translated::loop_guard`]).
    guarded: bool,
    body: String,
    /// The condition of the `exec` guard still open at the end of `body`.
    open_guard: Option<String>,
    /// Blocks are guarded by `pc` in address order, with a loop around
    /// each backward branch's range, instead of a `switch` in one loop.
    regions: bool,
    /// The current block ended with a branch, which set `pc`.
    block_ended: bool,
    regs: BTreeSet<String>,
    /// SGPRs each instruction writes, by dword offset.
    writes: BTreeMap<usize, BTreeSet<u16>>,
    pc: usize,
    /// SGPR origins before each instruction, from a first pass; `None` in that pass.
    syms: Option<BTreeMap<usize, Syms>>,
    /// Descriptor register, dimension, source and access, as of the first
    /// instruction using it.
    textures: Vec<(u16, TextureKind, Source, Access)>,
    samplers: Vec<Source>,
    attrs: BTreeMap<u8, Interp>,
    params: BTreeSet<u8>,
    mrts: u8,
    depth: bool,
    mem: bool,
    mem_write: bool,
    /// See [`Translated::store_buffers`].
    stores: Option<Vec<(Source, bool)>>,
    lds: bool,
    gds: bool,
    /// The entry point declares `qlane`, the thread's lane in its quad.
    quad_lane: bool,
    /// Texture slots image atomics use, from the first pass.
    atomic_textures: u32,
    /// The entry point declares `slane`, the thread's lane in its SIMD group.
    simd_lane: bool,
    /// The current DPP instruction leaves some lanes unwritten (`dpp_on`).
    dpp_lane_off: bool,
    /// The program runs in wave32: lane masks a VALU instruction writes are one SGPR.
    wave32: bool,
    /// A compute group that is one wave64 and talks through LDS without `s_barrier`,
    /// relying on its lanes running together: its two Metal SIMD groups are kept in step.
    lockstep: bool,
    /// The body holds indexed VGPR accesses (`movrel`) to fill in.
    movrel: bool,
    ij_vgprs: BTreeMap<u16, Interp>,
    user_base: u16,
}

const SGPR_COUNT: u16 = 106;
const VREAD: &str = "/*@VREAD:";
const VWRITE: &str = "/*@VWRITE:";
const VEND: &str = "@*/";
/// SGPRs `sgpr_sources` follows: the general ones and VCC.
const TRACKED_SGPRS: u16 = 108;

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
        let wave32 = insts.iter().any(|(_, i)| {
            let n = i.op.name;
            n.contains("saveexec_b32")
                || n.ends_with("_b32")
                    && matches!(
                        i.kind,
                        Kind::Sop1 { sdst: 126, .. } | Kind::Sop2 { sdst: 126, .. }
                    )
        });
        let lockstep = req.stage == Stage::Compute
            && !wave32
            && req.threads.iter().product::<u32>() == 64
            && !insts.iter().any(|(_, i)| i.op.name == "s_barrier")
            && insts
                .iter()
                .any(|(_, i)| matches!(i.kind, Kind::Ds { gds: false, .. }));
        Tx {
            req,
            insts,
            wave32,
            lockstep,
            has_barrier: false,
            guarded: false,
            body: String::new(),
            open_guard: None,
            regions: false,
            block_ended: false,
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
            stores: Some(Vec::new()),
            lds: false,
            gds: false,
            quad_lane: false,
            atomic_textures: 0,
            simd_lane: false,
            dpp_lane_off: false,
            movrel: false,
            ij_vgprs,
            user_base,
        }
    }

    fn line(&mut self, s: &str) {
        if !s.starts_with("//") {
            self.close_guard();
        }
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
        if code < TRACKED_SGPRS {
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

    // Assigns a VALU lane mask (this thread's bit) to `dst`: an SGPR pair, or one in wave32.
    fn set_mask(&mut self, dst: u16, bit: &str) -> String {
        self.written(dst);
        let lo = self.s(dst);
        if self.wave32 {
            return format!(" {lo} = uint({bit});");
        }
        self.written(dst + 1);
        let hi = self.s(dst + 1);
        format!(" {lo} = uint({bit}); {hi} = 0u;")
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
        let loops = loop_regions(&self.insts, &leaders);
        self.regions = structured && loops.is_some();
        let loops = loops.unwrap_or_default();
        self.has_barrier = self.insts.iter().any(|(_, i)| i.op.name == "s_barrier");
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
                        let any = "wave_any(scc, lockstep_vote, sgi)";
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
                        let mask = if n.contains("vcc") {
                            self.s(106)
                        } else {
                            self.s(126)
                        };
                        let any = if self.lockstep {
                            format!("wave_any(({mask} & 1u) != 0u, lockstep_vote, sgi)")
                        } else if self.req.stage == Stage::Compute {
                            format!("simd_any(({mask} & 1u) != 0u)")
                        } else {
                            format!("(({mask} & 1u) != 0u)")
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

    fn sop1(
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

    // Consecutive guarded statements share one `if`: a branch per
    // instruction made Metal's compiler several times slower.
    fn guard(&mut self, stmt: &str) {
        let e = self.s(126);
        let extra = if self.dpp_lane_off { " && dpp_on" } else { "" };
        let cond = format!("({e} & 1u) != 0u{extra}");
        if self.open_guard.as_ref() != Some(&cond) {
            self.close_guard();
            self.body.push_str(&format!("        if ({cond}) {{\n"));
            self.open_guard = Some(cond);
        }
        self.body.push_str(&format!("        {{ {stmt} }}\n"));
        if stmt.contains(&format!("{e} =")) || stmt.contains("dpp_on =") {
            self.close_guard();
        }
    }

    fn close_guard(&mut self) {
        if self.open_guard.take().is_some() {
            self.body.push_str("        }\n");
        }
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
        let r = self.valu_inner(pc, inst, v);
        if self.dpp_lane_off {
            self.dpp_lane_off = false;
            self.line("}");
        }
        r
    }

    fn valu_inner(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
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
        // Lanes are not modelled: a register is one value per thread. A
        // constant lane gets its own variable, which covers SGPR spills.
        let lane = |src: u16| (128..=192).contains(&src).then(|| src - 128);
        if name == "v_writelane_b32" {
            let s = self.usrc(pc, inst, v, 0)?;
            let target = match lane(v.src[1]) {
                Some(k) => format!("v{}_l{k}", v.vdst),
                None => self.v(v.vdst as u16),
            };
            self.regs.insert(target.clone());
            self.line(&format!("{target} = {s};"));
            return Ok(());
        }
        if name == "v_readfirstlane_b32" || name == "v_readlane_b32" {
            let spilled = (name == "v_readlane_b32" && v.src[0] >= 256)
                .then(|| lane(v.src[1]).map(|k| format!("v{}_l{k}", v.src[0] - 256)))
                .flatten()
                .filter(|n| self.regs.contains(n));
            let s = match spilled {
                Some(n) => n,
                None => {
                    let a = self.usrc(pc, inst, v, 0)?;
                    match lane(v.src[1]) {
                        // Metal SIMD groups are 32 lanes: lane k reads lane k mod 32 of its group.
                        Some(k) if name == "v_readlane_b32" && self.req.stage != Stage::Vertex => {
                            format!("simd_broadcast({a}, ushort({}))", k & 31)
                        }
                        _ => a,
                    }
                }
            };
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
        if name == "v_mad_u64_u32" || name == "v_mad_i64_i32" {
            return self.mad64(pc, inst, v);
        }
        if matches!(name, "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64") {
            let shift = self.usrc(pc, inst, v, 0)?;
            let x = self.vsrc64(pc, inst, v.src[1])?;
            let value = match name {
                "v_lshlrev_b64" => format!("{x} << ({shift} & 63u)"),
                "v_lshrrev_b64" => format!("{x} >> ({shift} & 63u)"),
                _ => format!("ulong(long({x}) >> ({shift} & 63u))"),
            };
            let exec = self.s(126);
            let (lo, hi) = (self.v(v.vdst as u16), self.v(v.vdst as u16 + 1));
            self.line(&format!(
                "{{ ulong r = {value}; if (({exec} & 1u) != 0u) {{ {lo} = uint(r); {hi} = uint(r >> 32); }} }}"
            ));
            return Ok(());
        }
        if matches!(
            name,
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16"
        ) {
            return self.fma_mix(pc, inst, v);
        }
        if name.starts_with("v_movrel") || name == "v_swaprel_b32" {
            return self.movrel(pc, inst, v);
        }
        if name == "v_permlane16_b32" || name == "v_permlanex16_b32" {
            if self.req.stage == Stage::Vertex {
                return Err(err(pc, format!("{name} in a vertex program")));
            }
            let a = self.usrc(pc, inst, v, 0)?;
            let (lo, hi) = (self.usrc(pc, inst, v, 1)?, self.usrc(pc, inst, v, 2)?);
            self.simd_lane = true;
            let row = if name == "v_permlanex16_b32" {
                "((slane & 16u) ^ 16u)"
            } else {
                "(slane & 16u)"
            };
            let d = self.v(v.vdst as u16);
            self.line(&format!(
                "{{ uint i = slane & 15u; uint sel = (i < 8u ? {lo} >> (4u * i) : {hi} >> (4u * (i - 8u))) & 15u; uint r = simd_shuffle({a}, ushort({row} | sel));"
            ));
            self.guard(&format!("{d} = r;"));
            self.line("}");
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
            "v_xad_u32" => format!("({} ^ {}) + {}", u(self, 0)?, u(self, 1)?, u(self, 2)?),
            "v_alignbyte_b32" => format!(
                "uint((ulong({}) << 32 | ulong({})) >> (8u * ({} & 3u)))",
                u(self, 0)?,
                u(self, 1)?,
                u(self, 2)?
            ),
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
                "as_type<uint>(half2(half_rtz({}), half_rtz({})))",
                f(self, 0)?,
                f(self, 1)?
            ),
            "v_cvt_f16_f32" => format!("uint(as_type<ushort>(half({})))", f(self, 0)?),
            "v_cvt_pknorm_u16_f32" => {
                format!(
                    "pack_float_to_unorm2x16(float2({}, {}))",
                    f(self, 0)?,
                    f(self, 1)?
                )
            }
            "v_cvt_pknorm_i16_f32" => {
                format!(
                    "pack_float_to_snorm2x16(float2({}, {}))",
                    f(self, 0)?,
                    f(self, 1)?
                )
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
        let permute = v.enc == VEnc::Dpp16
            && ctrl <= 0xFF
            && v.dpp.row_mask == 0xF
            && v.dpp.bank_mask == 0xF
            && self.req.stage != Stage::Vertex;
        if !broadcast && !permute && self.req.stage != Stage::Vertex {
            return self.dpp_general(pc, v);
        }
        if !broadcast && !permute {
            return Err(err(pc, format!("unsupported DPP form of {}", inst.op.name)));
        }
        let a = self.src32(v.src[0], None, pc)?;
        self.regs.insert("dpp0".into());
        if broadcast {
            self.line(&format!("dpp0 = quad_broadcast({a}, {lane}u);"));
        } else {
            self.quad_lane = true;
            self.line(&format!(
                "dpp0 = quad_shuffle({a}, ushort(({ctrl:#x}u >> (2u * uint(qlane))) & 3u));"
            ));
        }
        Ok(())
    }

    /// DPP forms other than full-mask quad permutations, as `simd_shuffle`
    /// within 32-lane Metal SIMD groups.
    fn dpp_general(&mut self, pc: usize, v: &Valu) -> Result<(), TranslateError> {
        let ctrl = v.dpp.ctrl as u32;
        let (src, inside) = if v.enc == VEnc::Dpp8 {
            (
                format!(
                    "((i & 24u) | (({:#x}u >> (3u * (i & 7u))) & 7u))",
                    v.dpp.sel8
                ),
                "true".to_string(),
            )
        } else {
            let n = ctrl & 0xF;
            match ctrl {
                0x00..=0xFF => (
                    format!("((i & 28u) | (({ctrl:#x}u >> (2u * (i & 3u))) & 3u))"),
                    "true".into(),
                ),
                0x101..=0x10F => (format!("(i + {n}u)"), format!("(i & 15u) + {n}u < 16u")),
                0x111..=0x11F => (format!("(i - {n}u)"), format!("(i & 15u) >= {n}u")),
                0x121..=0x12F => (format!("((i & 16u) | ((i - {n}u) & 15u))"), "true".into()),
                0x140 => ("((i & 16u) | (15u - (i & 15u)))".into(), "true".into()),
                0x141 => ("((i & 24u) | (7u - (i & 7u)))".into(), "true".into()),
                0x150..=0x15F => (format!("((i & 16u) | {n}u)"), "true".into()),
                0x160..=0x16F => (format!("(i ^ {n}u)"), "true".into()),
                _ => return Err(err(pc, format!("unsupported DPP control {ctrl:#x}"))),
            }
        };
        let a = self.src32(v.src[0], None, pc)?;
        self.simd_lane = true;
        self.regs.insert("dpp0".into());
        let (rows, banks) = if v.enc == VEnc::Dpp8 {
            (0xF, 0xF)
        } else {
            (v.dpp.row_mask, v.dpp.bank_mask)
        };
        self.line(&format!(
            "{{ uint i = slane; dpp0 = simd_shuffle({a}, ushort({src} & 31u)); bool dpp_on = ({rows:#x}u >> ((i >> 4) & 3u) & 1u) != 0u && ({banks:#x}u >> ((i >> 2) & 3u) & 1u) != 0u; bool dpp_in = {inside};"
        ));
        if v.dpp.bound_ctrl {
            self.line("if (!dpp_in) dpp0 = 0u; ");
        } else {
            self.line("dpp_on = dpp_on && dpp_in;");
        }
        self.dpp_lane_off = true;
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
        let (a, b) = if name.starts_with("v_subrev") {
            (b, a)
        } else {
            (a, b)
        };
        let (value, carry) = if name.starts_with("v_add") {
            ("ulong(a) + ulong(b) + ulong(c)", "(r >> 32) != 0ul")
        } else {
            (
                "ulong(a) - ulong(b) - ulong(c)",
                "ulong(b) + ulong(c) > ulong(a)",
            )
        };
        let dst = if e64 {
            v.sdst.map_or(106, |s| s as u16)
        } else {
            106
        };
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
            s.push_str(&self.set_mask(dst, "k"));
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }

    /// A 64-bit vector-instruction source: a VGPR pair or a scalar source.
    fn vsrc64(&mut self, pc: usize, inst: &Inst, code: u16) -> Result<String, TranslateError> {
        Ok(match code {
            n @ 256.. => {
                let (lo, hi) = (self.v(n - 256), self.v(n - 255));
                format!("(ulong({hi}) << 32 | ulong({lo}))")
            }
            n => self.src64(n, inst.literal, pc)?,
        })
    }

    // `v_mad_u64_u32`/`v_mad_i64_i32`: a 64-bit product plus a 64-bit addend
    // into a VGPR pair, with the carry (unsigned) or overflow (signed) out.
    fn mad64(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        if v.clamp {
            return Err(err(pc, "clamped 64-bit multiply-add"));
        }
        let a = self.usrc(pc, inst, v, 0)?;
        let b = self.usrc(pc, inst, v, 1)?;
        let c = self.vsrc64(pc, inst, v.src[2])?;
        let signed = inst.op.name == "v_mad_i64_i32";
        let (value, carry) = if signed {
            (
                format!("ulong(long(int({a})) * long(int({b})) + long({c}))"),
                format!(
                    "(long(int({a})) * long(int({b}))) > 0 ? long(r) < long({c}) : long(r) > long({c})"
                ),
            )
        } else {
            (format!("ulong({a}) * ulong({b}) + {c}"), format!("r < {c}"))
        };
        let exec = self.s(126);
        let (lo, hi) = (self.v(v.vdst as u16), self.v(v.vdst as u16 + 1));
        let mut s = format!(
            "{{ ulong r = {value}; bool on = ({exec} & 1u) != 0u; bool k = ({carry}) && on; if (on) {{ {lo} = uint(r); {hi} = uint(r >> 32); }}"
        );
        let dst = v.sdst.map_or(src::NULL, |s| s as u16);
        if dst != src::NULL {
            s.push_str(&self.set_mask(dst, "k"));
        }
        s.push_str(" }");
        self.line(&s);
        Ok(())
    }

    /// `v_fma_mix*`: each source is a float, or with its `op_sel_hi` bit the
    /// half `op_sel` picks; `neg_hi` is its abs. Inline constants keep their
    /// value as halves.
    fn fma_mix(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let mut srcs = Vec::new();
        for i in 0..3 {
            let u = self.usrc(pc, inst, v, i)?;
            let inline = (240..=248).contains(&v.src[i]);
            let mut e = if v.opsel_hi >> i & 1 != 0 && !inline {
                let half = if v.opsel >> i & 1 != 0 { "y" } else { "x" };
                format!("float(as_type<half2>({u}).{half})")
            } else {
                format!("F({u})")
            };
            if v.neg_hi >> i & 1 != 0 {
                e = format!("fabs({e})");
            }
            if v.neg >> i & 1 != 0 {
                e = format!("(-{e})");
            }
            srcs.push(e);
        }
        let mut r = format!("fma({}, {}, {})", srcs[0], srcs[1], srcs[2]);
        if v.clamp {
            r = format!("saturate({r})");
        }
        let d = self.v(v.vdst as u16);
        self.regs.insert(format!("v{}", v.vdst));
        let stmt = match inst.op.name {
            "v_fma_mix_f32" => format!("{d} = U({r});"),
            "v_fma_mixlo_f16" => {
                format!("{d} = ({d} & 0xffff0000u) | uint(as_type<ushort>(half({r})));")
            }
            _ => format!("{d} = ({d} & 0xffffu) | (uint(as_type<ushort>(half({r}))) << 16);"),
        };
        self.guard(&stmt);
        Ok(())
    }

    /// Indexed VGPR moves through M0. VGPRs are named locals, so an indexed
    /// access is a `switch` whose cases `emit` fills in (`VREAD`, `VWRITE`).
    fn movrel(&mut self, pc: usize, inst: &Inst, v: &Valu) -> Result<(), TranslateError> {
        let name = inst.op.name;
        if !matches!(v.enc, VEnc::E32 | VEnc::E64)
            || (v.enc == VEnc::E64 && (v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0))
        {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        let m0 = self.s(124);
        self.movrel = true;
        let src = v.src[0];
        let dst = v.vdst as u16;
        let (src_index, dst_index) = match name {
            "v_movrelsd_2_b32" | "v_swaprel_b32" => (
                format!("({m0} & 0x3ffu)"),
                format!("(({m0} >> 16) & 0x3ffu)"),
            ),
            _ => (m0.clone(), m0.clone()),
        };
        let read = |out: &str, base: u16, index: &str| {
            format!("uint {out} = 0u; switch ({index} + {base}u) {{ {VREAD}{out}{VEND} }}")
        };
        let write = |base: u16, index: &str, val: &str| {
            format!("switch ({index} + {base}u) {{ {VWRITE}{val}{VEND} }}")
        };
        let base = (src >= 256).then(|| src - 256);
        if let Some(b) = base {
            self.v(b);
        }
        self.v(dst);
        let need_base = || base.ok_or_else(|| err(pc, format!("{name} from a non-VGPR")));
        let stmt = match name {
            "v_movrels_b32" => {
                let d = self.v(dst);
                format!("{{ {} {d} = rv; }}", read("rv", need_base()?, &src_index))
            }
            "v_movreld_b32" => {
                let s = self.usrc(pc, inst, v, 0)?;
                format!("{{ uint wv = {s}; {} }}", write(dst, &dst_index, "wv"))
            }
            "v_movrelsd_b32" | "v_movrelsd_2_b32" => {
                let b = need_base()?;
                format!(
                    "{{ {} {} }}",
                    read("rv", b, &src_index),
                    write(dst, &dst_index, "rv")
                )
            }
            "v_swaprel_b32" => {
                let b = need_base()?;
                format!(
                    "{{ {} {} {} {} }}",
                    read("rv", b, &src_index),
                    read("rw", dst, &dst_index),
                    write(dst, &dst_index, "rv"),
                    write(b, &src_index, "rw")
                )
            }
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        self.guard(&stmt);
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
            "f32" if op == "class" => {
                let a = self.fsrc(pc, inst, v, 0)?;
                let m = self.usrc(pc, inst, v, 1)?;
                format!("fclass({a}, {m})")
            }
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
            "i64" | "u64" => {
                if v.enc == VEnc::Sdwa || matches!(v.enc, VEnc::Dpp16 | VEnc::Dpp8) {
                    return Err(err(pc, format!("unsupported form of {name}")));
                }
                let (a, b) = (
                    self.vsrc64(pc, inst, v.src[0])?,
                    self.vsrc64(pc, inst, v.src[1])?,
                );
                let (a, b) = if ty == "i64" {
                    (format!("long({a})"), format!("long({b})"))
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
        let set = self.set_mask(dst, "c");
        self.line(&format!(
            "{{ bool c = ({cond}) && ({e} & 1u) != 0u;{set} }}"
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
        let flat_shade = self
            .req
            .ps_input_cntl
            .get(attr as usize)
            .is_some_and(|c| c & 0x400 != 0);
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
                        7 => {
                            format!("uint4({lo} & 0xffffu, {lo} >> 16, {hi} & 0xffffu, {hi} >> 16)")
                        }
                        8 => format!(
                            "int4(short({lo}), short({lo} >> 16), short({hi}), short({hi} >> 16))"
                        ),
                        _ => format!(
                            "float4(float2(as_type<half2>({lo})), float2(as_type<half2>({hi})))"
                        ),
                    }
                } else {
                    match fmt {
                        7 => format!(
                            "uint4({})",
                            comps
                                .iter()
                                .map(|(on, r)| if *on { r.clone() } else { "0u".into() })
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                        8 => format!(
                            "int4({})",
                            comps
                                .iter()
                                .map(|(on, r)| if *on { format!("int({r})") } else { "0".into() })
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
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

    /// FLAT and GLOBAL loads, reading guest memory through the memory window.
    fn flat(
        &mut self,
        pc: usize,
        name: &str,
        f: &crate::decode::Flat,
    ) -> Result<(), TranslateError> {
        let (n, call): (u16, &str) = match name.split_once('_').map_or("", |(_, x)| x) {
            "load_ubyte" => (1, "flat_load_ubyte"),
            "load_sbyte" => (1, "flat_load_sbyte"),
            "load_ushort" => (1, "flat_load_ushort"),
            "load_sshort" => (1, "flat_load_sshort"),
            "load_dword" => (1, "flat_load_raw"),
            "load_dwordx2" => (2, "flat_load_raw"),
            "load_dwordx3" => (3, "flat_load_raw"),
            "load_dwordx4" => (4, "flat_load_raw"),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        if f.lds || !(name.starts_with("flat_") || name.starts_with("global_")) {
            return Err(err(pc, format!("unsupported instruction {name}")));
        }
        self.mem = true;
        let lo = self.v(f.addr as u16);
        let addr = if name.starts_with("global_") && (f.saddr as u16) < SGPR_COUNT {
            let (s0, s1) = (self.s(f.saddr as u16), self.s(f.saddr as u16 + 1));
            format!("((ulong({s0}) | (ulong({s1}) << 32)) + ulong({lo}))")
        } else {
            let hi = self.v(f.addr as u16 + 1);
            format!("(ulong({lo}) | (ulong({hi}) << 32))")
        };
        let mut s = format!(
            "uint4 r = {call}(mem, goff(mem_base, {addr} + ulong(long({}))));",
            f.offset
        );
        for i in 0..n {
            let d = self.v(f.vdst as u16 + i);
            write!(s, " {d} = r[{i}];").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    fn mubuf(
        &mut self,
        pc: usize,
        name: &str,
        b: &crate::decode::Buf,
        format: Option<u8>,
    ) -> Result<(), TranslateError> {
        if name == "buffer_gl0_inv" || name == "buffer_gl1_inv" {
            return Ok(());
        }
        if b.lds || b.tfe {
            return Err(err(pc, "unsupported LDS or TFE buffer access"));
        }
        self.mem = true;
        let r = b.srsrc as u16;
        let mut vs: Vec<String> = (0..4).map(|i| self.s(r + i)).collect();
        if let Some(f) = format {
            vs[3] = format!("(({} & ~0x7f000u) | {}u)", vs[3], (f as u32 & 0x7f) << 12);
        }
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
            self.note_store(pc, r, b.idxen);
            let data = self.v(vd);
            let call = match op {
                "add" | "sub" | "and" | "or" | "xor" | "umin" | "umax" | "smin" | "smax"
                | "swap" | "fmin" | "fmax" => {
                    format!("atomic_rmw_{op}(mem, buffer_store_offset({args}), {data})")
                }
                "cmpswap" => {
                    let cmp = self.v(vd + 1);
                    format!("atomic_cas(mem, buffer_store_offset({args}), {data}, {cmp})")
                }
                // No 64-bit atomics here: each half alone, the high half (a
                // flag in look-back scans) read before and written after the low.
                "and_x2" | "or_x2" | "xor_x2" | "swap_x2" => {
                    let f = &op[..op.len() - 3];
                    let hi = self.v(vd + 1);
                    let (first, second) = if f == "swap" {
                        (
                            format!("lo = atomic_rmw_swap(mem, o, {data});"),
                            format!("hi = atomic_rmw_swap(mem, o4, {hi});"),
                        )
                    } else {
                        (
                            format!("hi = atomic_rmw_{f}(mem, o4, {hi});"),
                            format!("lo = atomic_rmw_{f}(mem, o, {data});"),
                        )
                    };
                    let stmt = format!(
                        "{{ long o = buffer_store_offset({args}); long o4 = o == OOB_OFF ? o : o + 4; uint lo, hi; {first} {second}{} }}",
                        if b.glc {
                            format!(" {data} = lo; {hi} = hi;")
                        } else {
                            String::new()
                        }
                    );
                    self.guard(&stmt);
                    return Ok(());
                }
                _ => return Err(err(pc, format!("unsupported instruction {name}"))),
            };
            let stmt = if b.glc {
                format!("{data} = {call};")
            } else {
                format!("(void){call};")
            };
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
            self.note_store(pc, r, b.idxen);
            let mut s = format!("long o = buffer_store_offset({args}); if (o != OOB_OFF) {{");
            for i in 0..n {
                let v = self.v(vd + i);
                match width {
                    "8" => write!(s, " mem[o] = uchar({v});").unwrap(),
                    "16" => write!(s, " *(device ushort*)(mem + o) = ushort({v});").unwrap(),
                    _ if b.glc => write!(s, " store_atomic(mem, o + {}, {v});", 4 * i).unwrap(),
                    _ => write!(s, " *(device uint*)(mem + o + {}) = {v};", 4 * i).unwrap(),
                }
            }
            s.push_str(" }");
            self.guard(&s);
            return Ok(());
        }
        let store_format = match name {
            "buffer_store_format_x" => Some(1u16),
            "buffer_store_format_xy" => Some(2),
            "buffer_store_format_xyz" => Some(3),
            "buffer_store_format_xyzw" => Some(4),
            _ => None,
        };
        if let Some(n) = store_format {
            self.mem_write = true;
            self.note_store(pc, r, b.idxen);
            let vals: Vec<String> = (0..4)
                .map(|i| if i < n { self.v(vd + i) } else { "0u".into() })
                .collect();
            let stmt = format!(
                "buffer_store_format(mem, {args}, uint4({}, {}, {}, {}), {n}u);",
                vals[0], vals[1], vals[2], vals[3]
            );
            self.guard(&stmt);
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
        let mut s = if b.glc && call == "buffer_load_raw" {
            let coherent = if self.mem_write {
                "buffer_load_atomic"
            } else {
                "buffer_load_volatile"
            };
            format!("uint4 r = {coherent}(mem, {args}, {n}u);")
        } else {
            format!("uint4 r = {call}(mem, {args});")
        };
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
        if matches!(name, "ds_append" | "ds_consume") {
            return self.ds_counter(
                pc,
                name,
                offset0 as u32 | (offset1 as u32) << 8,
                gds,
                regs[3],
            );
        }
        if name == "ds_bpermute_b32" && !gds && self.req.stage != Stage::Vertex {
            // Metal SIMD groups are 32 lanes, so the source lane is taken mod 32.
            let [addr, data0, _, vdst] = regs;
            let (a, d0, d) = (self.v(addr), self.v(data0), self.v(vdst));
            let off = offset0 as u32 | (offset1 as u32) << 8;
            self.line(&format!(
                "{{ uint bp = simd_shuffle({d0}, ushort((({a} + {off}u) >> 2) & 31u));"
            ));
            self.guard(&format!("{d} = bp;"));
            self.line("}");
            return Ok(());
        }
        if gds || self.req.stage != Stage::Compute {
            return Err(err(pc, format!("unsupported LDS access {name}")));
        }
        self.lds = true;
        if name == "ds_write_addtid_b32" || name == "ds_read_addtid_b32" {
            // Wave64 assumed: the lane is the thread's index in the group mod 64.
            let [_, data0, _, vdst] = regs;
            let m0 = self.s(124);
            let [tx, ty, _] = self.req.threads;
            let a = format!(
                "(({m0} & 0xffffu) + {}u + 4u * ((lid.x + lid.y * {tx}u + lid.z * {}u) & 63u))",
                offset0 as u32 | (offset1 as u32) << 8,
                tx * ty
            );
            let stmt = if name == "ds_write_addtid_b32" {
                let d = self.v(data0);
                format!("lds_wr(lds, {a}, {d});")
            } else {
                let d = self.v(vdst);
                format!("{d} = lds_rd(lds, {a});")
            };
            self.guard(&stmt);
            return Ok(());
        }
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
                let n = match name {
                    "ds_write_b32" => 1,
                    "ds_write_b64" => 2,
                    "ds_write_b96" => 3,
                    _ => 4,
                };
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
                    write!(
                        s,
                        " lds_wr(lds, a + {}u, {x}); lds_wr(lds, a + {}u, {y});",
                        o0 + 4 * i as u32,
                        o1 + 4 * i as u32
                    )
                    .unwrap();
                }
            }
            "ds_write_b8" | "ds_write_b16" => {
                let v = d0(self, 0);
                let ty = if name.ends_with("b8") {
                    "uchar"
                } else {
                    "ushort"
                };
                write!(s, " if ((a + {off}u) < LDS_BYTES) *(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u) = {ty}({v});").unwrap();
            }
            "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" => {
                let n = match name {
                    "ds_read_b32" => 1,
                    "ds_read_b64" => 2,
                    "ds_read_b96" => 3,
                    _ => 4,
                };
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
                let ty = match name {
                    "ds_read_u8" => "uchar",
                    "ds_read_i8" => "char",
                    "ds_read_u16" => "ushort",
                    _ => "short",
                };
                let d = dst(self, 0);
                write!(s, " {d} = (a + {off}u) < LDS_BYTES ? uint(int(*(threadgroup {ty}*)((threadgroup uchar*)lds + a + {off}u))) : 0u;").unwrap();
            }
            _ => {
                let base = name.strip_prefix("ds_").unwrap();
                let rtn = base.contains("_rtn_");
                let op = base.replace("_rtn", "");
                let v = d0(self, 0);
                let call = match op.as_str() {
                    "add_u32" | "sub_u32" | "and_b32" | "or_b32" | "xor_b32" | "min_u32"
                    | "max_u32" | "min_i32" | "max_i32" | "wrxchg_b32" => {
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
                    "min_f32" | "max_f32" => {
                        let f = &op[..3];
                        format!("lds_rmw_f{f}(lds, a + {off}u, {v})")
                    }
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

    // `ds_append`/`ds_consume`: a counter at M0's base plus `offset` (for the GDS,
    // the base is M0's high half: Astro Bot's packets clear its counters there). Lanes
    // are threads here (`v_mbcnt` gives 0), so each thread counts itself and
    // gets its own previous value; a wave on hardware adds its active lanes at once.
    fn ds_counter(
        &mut self,
        pc: usize,
        name: &str,
        offset: u32,
        gds: bool,
        vdst: u16,
    ) -> Result<(), TranslateError> {
        let m0 = self.s(124);
        let d = self.v(vdst);
        let op = if name == "ds_append" { "add" } else { "sub" };
        let stmt = if gds {
            self.gds = true;
            format!(
                "{{ uint a = (({m0} >> 16) + {offset}u) & 0xfffcu; {d} = atomic_fetch_{op}_explicit(&gds[a >> 2], 1u, memory_order_relaxed); }}"
            )
        } else {
            if self.req.stage != Stage::Compute {
                return Err(err(pc, format!("unsupported LDS access {name}")));
            }
            self.lds = true;
            format!("{{ {d} = lds_rmw_{op}(lds, ({m0} & 0xffffu) + {offset}u, 1u); }}")
        };
        self.guard(&stmt);
        Ok(())
    }

    /// Records the V# at `srsrc` that a buffer store or atomic at `pc` uses.
    fn note_store(&mut self, pc: usize, srsrc: u16, indexed: bool) {
        let source = self.descriptor_source(pc, srsrc, 4).ok();
        match (source, &mut self.stores) {
            (Some(s), Some(list)) => {
                if !list.contains(&(s.clone(), indexed)) {
                    list.push((s, indexed));
                }
            }
            _ => self.stores = None,
        }
    }

    fn descriptor_source(&self, pc: usize, first: u16, n: u16) -> Result<Source, TranslateError> {
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
    fn sgpr_sources(&self) -> BTreeMap<usize, Syms> {
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
                        n.starts_with("v_cmp")
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

    // The element type of texture slot `t` and how a component of it becomes a VGPR value.
    fn texel(&self, t: usize) -> (&'static str, &'static str) {
        if self.atomic_textures >> t & 1 != 0 {
            if self.req.sint_textures >> t & 1 != 0 {
                ("int", "uint")
            } else {
                ("uint", "")
            }
        } else if self.req.uint_textures >> t & 1 != 0 {
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
        let t = match self
            .textures
            .iter()
            .position(|x| x.0 == srsrc && x.1 == kind && x.2 == source)
        {
            Some(t) => t,
            None => {
                self.textures.push((srsrc, kind, source, access));
                self.textures.len() - 1
            }
        };
        let entry = &mut self.textures[t];
        entry.3 = match (entry.3, access) {
            (a, b) if a == b => a,
            (Access::Atomic, _) | (_, Access::Atomic) => Access::Atomic,
            (Access::Write, Access::Sample)
            | (Access::Sample, Access::Write)
            | (_, Access::ReadWrite)
            | (Access::ReadWrite, _) => Access::ReadWrite,
            (a, _) => a,
        };
        Ok(t)
    }

    fn mimg(&mut self, pc: usize, name: &str, m: &Mimg) -> Result<(), TranslateError> {
        if m.a16 || m.tfe || m.lwe {
            return Err(err(pc, "unsupported image modifiers"));
        }
        if name.starts_with("image_bvh") {
            return self.bvh(m, name.starts_with("image_bvh64"));
        }
        let kind = match m.dim {
            1 => TextureKind::Tex2d,
            5 => TextureKind::Tex2dArray,
            2 => TextureKind::Tex3d,
            3 => TextureKind::Cube,
            d => {
                return Err(err(
                    pc,
                    format!("unsupported image dimension {d} in {name}"),
                ));
            }
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
            n if n.starts_with("image_atomic_") => {
                return self.image_atomic(pc, name, m, kind, &addr_regs);
            }
            _ => {}
        }
        let gather = name.starts_with("image_gather4");
        let variant = name
            .strip_prefix("image_sample")
            .or_else(|| name.strip_prefix("image_gather4"))
            .ok_or_else(|| err(pc, format!("unsupported instruction {name}")))?;
        let t = self.texture_slot(pc, m.srsrc as u16, kind, Access::Sample)?;
        let s_source = self.descriptor_source(pc, m.ssamp as u16, 4)?;
        let sm = match self.samplers.iter().position(|x| *x == s_source) {
            Some(i) => i,
            None => {
                self.samplers.push(s_source);
                self.samplers.len() - 1
            }
        };
        let addr: Vec<String> = addr_regs
            .iter()
            .map(|&r| format!("F({})", self.v(r)))
            .collect();
        let raw: Vec<String> = addr_regs.iter().map(|&r| self.v(r)).collect();
        // Address order: offset, bias, depth reference, coordinates, LOD.
        let mut flags: Vec<&str> = variant.split('_').filter(|p| !p.is_empty()).collect();
        let offset = flags.last() == Some(&"o");
        if offset {
            flags.pop();
        }
        let compare = flags.first() == Some(&"c");
        if compare {
            flags.remove(0);
        }
        let mut at = offset as usize;
        let (level, bias) = match flags.as_slice() {
            [] => (None, false),
            ["lz"] => (Some("level(0.0f)".to_string()), false),
            ["l"] => (Some(format!("level({})", addr[addr.len() - 1])), false),
            ["b"] => (Some(format!("bias({})", addr[at])), true),
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        at += bias as usize;
        let dref = compare.then(|| addr[at].clone());
        at += compare as usize;
        let c = &addr[at..];
        let shift = |i: usize, size: &str| {
            if offset {
                format!(
                    " + float(int(({} >> {}u) << 26) >> 26) / float(t{t}.get_{size}(0))",
                    raw[0],
                    8 * i
                )
            } else {
                String::new()
            }
        };
        let (dx, dy) = (shift(0, "width"), shift(1, "height"));
        let coord = match kind {
            TextureKind::Tex2d => format!("float2({}{dx}, {}{dy})", c[0], c[1]),
            TextureKind::Tex2dArray => {
                format!("float2({}{dx}, {}{dy}), uint(rint({}))", c[0], c[1], c[2])
            }
            TextureKind::Tex3d if !offset => format!("float3({}, {}, {})", c[0], c[1], c[2]),
            TextureKind::Cube if !offset => {
                format!("cube_dir(float3({}, {}, {}))", c[0], c[1], c[2])
            }
            _ => return Err(err(pc, format!("unsupported form of {name}"))),
        };
        let level = match (level, self.req.stage) {
            (Some(l), _) => format!(", {l}"),
            (None, Stage::Pixel) => String::new(),
            (None, _) => ", level(0.0f)".into(),
        };
        let (elem, conv) = self.texel(t);
        let mut s = format!("{elem}4 r = t{t}.sample(smp{sm}, {coord}{level});");
        if let Some(dref) = &dref {
            if gather || elem != "float" {
                return Err(err(pc, format!("unsupported form of {name}")));
            }
            write!(
                s,
                " r = float4(depth_compare((user[{}] >> 12) & 7u, {dref}, r.x) ? 1.0f : 0.0f);",
                binding::SAMPLER_WORDS + sm as u32
            )
            .unwrap();
        }
        if gather {
            if !matches!(kind, TextureKind::Tex2d | TextureKind::Tex2dArray)
                || !matches!(variant, "" | "_lz")
                || m.dmask.count_ones() != 1
            {
                return Err(err(pc, format!("unsupported form of {name}")));
            }
            let c = ["x", "y", "z", "w"][m.dmask.trailing_zeros() as usize];
            s = format!("{elem}4 r = t{t}.gather(smp{sm}, {coord}, int2(0), component::{c});");
        }
        let comps: Vec<usize> = (0..4)
            .filter(|&i| gather || m.dmask >> i & 1 != 0)
            .collect();
        self.write_results(&mut s, m, elem, conv, &comps);
        self.guard(&s);
        Ok(())
    }

    // Moves the components `comps` of `r` into the destination VGPRs; with
    // `d16`, two 16-bit values per VGPR, low half first.
    fn write_results(&mut self, s: &mut String, m: &Mimg, elem: &str, conv: &str, comps: &[usize]) {
        let d = m.vdata as u16;
        for (j, &i) in comps.iter().enumerate() {
            if !m.d16 {
                let r = self.v(d + j as u16);
                write!(s, " {r} = {conv}(r[{i}]);").unwrap();
                continue;
            }
            let r = self.v(d + j as u16 / 2);
            let bits = match elem {
                "float" => format!("uint(as_type<ushort>(half(r[{i}])))"),
                _ => format!("(uint(r[{i}]) & 0xffffu)"),
            };
            if j % 2 == 0 {
                write!(s, " {r} = {bits};").unwrap();
            } else {
                write!(s, " {r} |= {bits} << 16;").unwrap();
            }
        }
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
        let d = m.vdata as u16;
        let (elem, conv) = self.texel(t);
        if store {
            let mut comps = Vec::new();
            let mut j = 0u16;
            for i in 0..4 {
                if m.dmask >> i & 1 != 0 {
                    let v = if m.d16 {
                        self.v(d + j / 2)
                    } else {
                        self.v(d + j)
                    };
                    let half = format!("({v} >> {}u)", 16 * (j % 2));
                    comps.push(match (elem, m.d16) {
                        ("float", false) => format!("F({v})"),
                        ("int", false) => format!("int({v})"),
                        (_, false) => v,
                        ("float", true) => format!("float(as_type<half>(ushort({half})))"),
                        ("int", true) => format!("int(short({half}))"),
                        (_, true) => format!("({half} & 0xffffu)"),
                    });
                    j += 1;
                } else {
                    comps.push(if elem == "float" {
                        "0.0f".into()
                    } else {
                        "0".into()
                    });
                }
            }
            self.guard(&format!(
                "t{t}.write({elem}4({}), {coord}, {lvl});",
                comps.join(", ")
            ));
        } else {
            let mut s = format!("{elem}4 r = t{t}.read({coord}, {lvl});");
            let comps: Vec<usize> = (0..4).filter(|&i| m.dmask >> i & 1 != 0).collect();
            self.write_results(&mut s, m, elem, conv, &comps);
            self.guard(&s);
        }
        Ok(())
    }

    /// Image atomics through Metal's texture atomics on a 32-bit integer
    /// view; `glc` returns the value before the operation.
    fn image_atomic(
        &mut self,
        pc: usize,
        name: &str,
        m: &Mimg,
        kind: TextureKind,
        addr: &[u16],
    ) -> Result<(), TranslateError> {
        let op = &name["image_atomic_".len()..];
        let f = match op {
            "swap" => "atomic_exchange",
            "add" => "atomic_fetch_add",
            "sub" => "atomic_fetch_sub",
            "smin" | "umin" => "atomic_fetch_min",
            "smax" | "umax" => "atomic_fetch_max",
            "and" => "atomic_fetch_and",
            "or" => "atomic_fetch_or",
            "xor" => "atomic_fetch_xor",
            "cmpswap" => "",
            _ => return Err(err(pc, format!("unsupported instruction {name}"))),
        };
        if m.dmask != 1 && !(op == "cmpswap" && m.dmask == 3) || m.d16 {
            return Err(err(pc, format!("unsupported form of {name}")));
        }
        let t = self.texture_slot(pc, m.srsrc as u16, kind, Access::Atomic)?;
        let a: Vec<String> = addr.iter().map(|&r| self.v(r)).collect();
        let coord = match kind {
            TextureKind::Tex2d => format!("uint2({}, {})", a[0], a[1]),
            TextureKind::Tex2dArray => format!("uint2({}, {}), {}", a[0], a[1], a[2]),
            _ => return Err(err(pc, format!("unsupported form of {name}"))),
        };
        // The first pass does not know the atomic slots yet.
        let (elem, conv) = match self.texel(t) {
            ("float", _) => ("uint", ""),
            x => x,
        };
        let d = self.v(m.vdata as u16);
        let value = if elem == "int" {
            format!("int({d})")
        } else {
            d.clone()
        };
        let mut s = if op == "cmpswap" {
            let cmp = self.v(m.vdata as u16 + 1);
            let cmp = if elem == "int" {
                format!("int({cmp})")
            } else {
                cmp
            };
            format!(
                "{elem}4 r = {elem}4({cmp}); for (;;) {{ if (t{t}.atomic_compare_exchange_weak({coord}, &r, {elem}4({value}))) {{ r = {elem}4({cmp}); break; }} if (r.x != {cmp}) break; }}"
            )
        } else {
            format!("{elem}4 r = t{t}.{f}({coord}, {elem}4({value}));")
        };
        if m.glc {
            write!(s, " {d} = {conv}(r.x);").unwrap();
        }
        self.guard(&s);
        Ok(())
    }

    fn bvh(&mut self, m: &Mimg, wide: bool) -> Result<(), TranslateError> {
        self.mem = true;
        let n = if wide { 12 } else { 11 };
        let a: Vec<String> = (0..n)
            .map(|i| {
                self.v(if m.nsa > 0 {
                    m.vaddr[i] as u16
                } else {
                    m.vaddr[0] as u16 + i as u16
                })
            })
            .collect();
        let node = if wide {
            format!("(ulong({}) | (ulong({}) << 32))", a[0], a[1])
        } else {
            format!("ulong({})", a[0])
        };
        let r = &a[n - 10..];
        let d: Vec<String> = (0..4).map(|i| self.s(m.srsrc as u16 + i)).collect();
        let mut s = format!(
            "uint4 r = bvh_intersect(mem, mem_base, uint4({}, {}, {}, {}), {node}, F({}), \
             float3(F({}), F({}), F({})), float3(F({}), F({}), F({})), float3(F({}), F({}), F({})));",
            d[0], d[1], d[2], d[3], r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]
        );
        for i in 0..4 {
            let v = self.v(m.vdata as u16 + i);
            write!(s, " {v} = r[{i}];").unwrap();
        }
        self.guard(&s);
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
        for (i, source) in self.samplers.iter().enumerate() {
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
            store_buffers: self.stores.take(),
            lds_bytes: if self.lds { self.lds_bytes() } else { 0 },
            gds: self.gds,
            loop_guard: self.guarded,
        }
    }

    /// Threadgroup memory: `LDS_SIZE` of `COMPUTE_PGM_RSRC2` in 512-byte units,
    /// within what Metal allows.
    fn lds_bytes(&self) -> u32 {
        let declared = (self.req.compute_rsrc2 >> 15 & 0x1FF) * 512;
        if declared == 0 {
            32 << 10
        } else {
            declared.clamp(16, 32 << 10)
        }
    }

    fn emit(&mut self) -> String {
        let mut out = String::from(PRELUDE);
        if self.lds {
            out.push_str(&format!(
                "#define LDS_BYTES {}u\n{LDS_PRELUDE}",
                self.lds_bytes()
            ));
        }
        if self.guarded {
            out.push_str(&format!(
                "#define GUARD_LIMIT {MAX_COMPUTE_BACK_BRANCHES}u\n#define GUARD_BUDGET {GUARD_BUDGET}u\n#define GUARD_UNIT {GUARD_UNIT}u\n{GUARD_PRELUDE}"
            ));
        }
        let stage = self.req.stage;
        let mem_ty = if self.mem_write {
            "device uchar*"
        } else {
            "device const uchar*"
        };
        let mut params = vec!["constant uint* user".to_string()];
        let mut args = vec!["user".to_string()];
        if self.mem {
            params.push(format!("{mem_ty} mem"));
            params.push("Win mem_base".into());
            args.push("mem".into());
            args.push("mem_base".into());
        }
        let mut entry_params = vec![format!("constant uint* user [[buffer({})]]", binding::USER)];
        if self.mem {
            entry_params.push(format!("{mem_ty} mem [[buffer({})]]", binding::MEM));
            entry_params.push(format!(
                "constant Win& mem_base [[buffer({})]]",
                binding::MEM_BASE
            ));
        }
        if self.gds {
            params.push("device atomic_uint* gds".into());
            args.push("gds".into());
            entry_params.push(format!(
                "device atomic_uint* gds [[buffer({})]]",
                binding::GDS
            ));
        }
        if self.guarded {
            entry_params.push(format!(
                "device atomic_uint* guard [[buffer({})]]",
                binding::GUARD
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
                Access::ReadWrite | Access::Atomic => format!("{base}<{elem}, access::read_write>"),
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
        if self.quad_lane {
            entry_params.push("ushort qlane [[thread_index_in_quadgroup]]".into());
        }
        if self.simd_lane {
            entry_params.push("ushort slane [[thread_index_in_simdgroup]]".into());
        }

        if self.movrel {
            self.fill_movrel();
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
                out.push_str(if shared {
                    "struct PsInRaw {\n"
                } else {
                    "struct PsIn {\n"
                });
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
                    let ty = if mode == Interp::Flat {
                        "uint4"
                    } else {
                        "float4"
                    };
                    out.push_str(&format!("    {ty} a{a} [[user(locn{loc}), {q}]];\n"));
                }
                if shared {
                    out.push_str("};\n\nstruct PsIn {\n    float4 pos;\n    bool front;\n");
                    let mut fill = String::from("    PsIn in = { raw.pos, raw.front");
                    for (&a, &mode) in &self.attrs {
                        let ty = if mode == Interp::Flat {
                            "uint4"
                        } else {
                            "float4"
                        };
                        out.push_str(&format!("    {ty} a{a};\n"));
                        let src = first[&loc_of(a)];
                        if (self.attrs[&src] == Interp::Flat) == (mode == Interp::Flat) {
                            fill.push_str(&format!(", raw.a{src}"));
                        } else {
                            fill.push_str(&format!(", as_type<{ty}>(raw.a{src})"));
                        }
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
                let mut tail =
                    String::from("    if (!exported) discard_fragment();\n    return o;\n");
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
                let mut lds = if self.lds {
                    "    threadgroup uint lds[LDS_BYTES / 4];\n"
                } else {
                    ""
                }
                .to_string();
                if self.guarded && (self.has_barrier || self.lockstep) {
                    lds.push_str("    threadgroup bool tg_bail;\n    uint bail_at = 0;\n    if (all(lid == 0)) tg_bail = false;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
                }
                let mut params = entry_params.join(", ");
                if self.lockstep {
                    lds.push_str("    threadgroup uint lockstep_vote[2];\n");
                    params.push_str(", ushort sgi [[simdgroup_index_in_threadgroup]]");
                }
                out.push_str(&format!(
                    "kernel void gen5_cs({params}, uint3 lid [[thread_position_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {{\n{lds}{init}{}}}\n",
                    self.body
                ));
            }
        }
        out
    }

    /// Replaces the `VREAD`/`VWRITE` markers by one case per VGPR up to the
    /// highest the program names, all of which are then declared.
    fn fill_movrel(&mut self) {
        let top = self
            .regs
            .iter()
            .filter_map(|r| r.strip_prefix('v')?.parse::<u16>().ok())
            .max()
            .unwrap_or(0);
        for n in 0..=top {
            self.regs.insert(format!("v{n}"));
        }
        let mut body = std::mem::take(&mut self.body);
        for (marker, read) in [(VREAD, true), (VWRITE, false)] {
            while let Some(at) = body.find(marker) {
                let rest = &body[at + marker.len()..];
                let end = rest.find(VEND).unwrap();
                let var = &rest[..end];
                let cases: String = (0..=top)
                    .map(|n| {
                        if read {
                            format!("case {n}: {var} = v{n}; break; ")
                        } else {
                            format!("case {n}: v{n} = {var}; break; ")
                        }
                    })
                    .collect();
                body = format!(
                    "{}{cases}default: break;{}",
                    &body[..at],
                    &rest[end + VEND.len()..]
                );
            }
        }
        self.body = body;
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

/// The compute loop guard (see `Tx::loop_guard`). A thread checks the
/// dispatch's budget at its first backward branch and adds to it every unit.
const GUARD_PRELUDE: &str = r#"
static inline bool guard_stop(device atomic_uint* g, uint back) {
    bool stop = back > GUARD_LIMIT;
    if (back % GUARD_UNIT == 0u) {
        stop = atomic_fetch_add_explicit(&g[0], 1u, memory_order_relaxed) >= GUARD_BUDGET || stop;
    } else if (back == 1u) {
        stop = atomic_load_explicit(&g[0], memory_order_relaxed) >= GUARD_BUDGET || stop;
    }
    if (stop) atomic_fetch_add_explicit(&g[1], 1u, memory_order_relaxed);
    return stop;
}

// The group leaves together once any thread's loop budget is spent (lockstep programs).
static inline bool lockstep_leave(device atomic_uint* g, uint back, threadgroup bool& tg_bail) {
    if (guard_stop(g, back)) tg_bail = true;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool leave = tg_bail;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return leave;
}

static inline bool guard_leave(device atomic_uint* g, uint back, thread uint& bail_at, threadgroup bool& tg_bail) {
    if (bail_at == 0u && guard_stop(g, back)) {
        bail_at = back;
        tg_bail = true;
    }
    return bail_at != 0u && back - bail_at > GUARD_UNIT;
}
"#;

const LOCKSTEP_BARRIER: &str = "threadgroup_barrier(mem_flags::mem_threadgroup);";

const PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;

#define F(x) as_type<float>(uint(x))
#define U(x) as_type<uint>(float(x))

// v_cmp_class_f32's mask: signalling NaN, quiet NaN, -inf, -normal,
// -denormal, -0, +0, +denormal, +normal, +inf.
static inline bool fclass(float x, uint mask) {
    uint b = as_type<uint>(x);
    bool neg = (b >> 31) != 0u;
    uint e = (b >> 23) & 0xffu, f = b & 0x7fffffu;
    uint k;
    if (e == 0xffu) k = f != 0u ? ((f >> 22) != 0u ? 1u : 0u) : (neg ? 2u : 9u);
    else if (e == 0u) k = f != 0u ? (neg ? 4u : 7u) : (neg ? 5u : 6u);
    else k = neg ? 3u : 8u;
    return (mask >> k & 1u) != 0u;
}

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

static inline uint quadmask64(ulong v) {
    uint r = 0u;
    for (uint q = 0u; q < 16u; q++) {
        if ((v >> (4u * q)) & 0xful) r |= 1u << q;
    }
    return r;
}

static inline float med3(float a, float b, float c) {
    return fmax(fmin(a, b), fmin(fmax(a, b), c));
}

static inline int med3_i(int a, int b, int c) {
    return max(min(a, b), min(max(a, b), c));
}

// Float to half, rounding toward zero: values past the largest half become
// +-65504, not infinity.
static inline half half_rtz(float x) {
    half h = half(x);
    if (abs(float(h)) > abs(x)) h = as_type<half>(ushort(as_type<ushort>(h) - 1u));
    return h;
}

// `mem` covers [base, end); windows over [base2, end2) and [base3, end3) lie `delta`
// and `delta3` bytes from `mem` in the GPU's address space. Stores skip [ro, ro_end).
struct Win { ulong base; ulong end; ulong base2; ulong end2; long delta; ulong ro; ulong ro_end; ulong base3; ulong end3; long delta3; };
#define OOB_OFF (-0x7fffffffffffffffl - 1l)

// Byte offset from `mem` of guest address `a`, or OOB_OFF outside both windows.
static inline long goff(Win w, ulong a) {
    if (a >= w.base && a < w.end) return long(a - w.base);
    if (a >= w.base2 && a < w.end2) return long(a - w.base2) + w.delta;
    if (a >= w.base3 && a < w.end3) return long(a - w.base3) + w.delta3;
    return OOB_OFF;
}

static inline uint ld32(device const uchar* mem, Win w, ulong addr) {
    long o = goff(w, addr);
    return o == OOB_OFF ? 0u : *(device const uint*)(mem + o);
}

// image_bvh_intersect_ray as Mesa radv's software emulation states it. Box nodes
// (types 4, 5) give the children the ray enters, nearest first; triangles (t, det, i, j).
static inline uint4 bvh_intersect(device const uchar* mem, Win w, uint4 d, ulong node, float tmax, float3 o, float3 dir, float3 inv) {
    ulong a = ((ulong(d.x) | (ulong(d.y & 0xffu) << 32)) << 8) + ((node & ~7ul) << 3);
    uint type = uint(node) & 7u;
    if (type == 4u || type == 5u) {
        float4 dist = float4(INFINITY);
        uint4 child = uint4(0xffffffffu);
        inv = clamp(inv, float3(-FLT_MAX), float3(FLT_MAX));
        for (uint i = 0u; i < 4u; i++) {
            float3 lo, hi;
            if (type == 5u) {
                ulong b = a + 16ul + 24ul * i;
                lo = float3(F(ld32(mem, w, b)), F(ld32(mem, w, b + 4ul)), F(ld32(mem, w, b + 8ul)));
                hi = float3(F(ld32(mem, w, b + 12ul)), F(ld32(mem, w, b + 16ul)), F(ld32(mem, w, b + 20ul)));
            } else {
                ulong b = a + 16ul + 12ul * i;
                half2 h0 = as_type<half2>(ld32(mem, w, b));
                half2 h1 = as_type<half2>(ld32(mem, w, b + 4ul));
                half2 h2 = as_type<half2>(ld32(mem, w, b + 8ul));
                lo = float3(float(h0.x), float(h0.y), float(h1.x));
                hi = float3(float(h1.y), float(h2.x), float(h2.y));
            }
            float3 b0 = (lo - o) * inv, b1 = (hi - o) * inv;
            float3 near = min(b0, b1), far = max(b0, b1);
            float t0 = max(max(near.x, near.y), near.z);
            float t1 = min(min(far.x, far.y), far.z);
            if (!isnan(lo.x) && t1 >= max(0.0f, t0) && t0 < tmax) {
                dist[i] = t0;
                child[i] = ld32(mem, w, a + 4ul * i);
            }
        }
        if ((d.y >> 31) != 0u) {
            const uint2 pairs[5] = { uint2(0, 1), uint2(2, 3), uint2(0, 2), uint2(1, 3), uint2(1, 2) };
            for (uint p = 0u; p < 5u; p++) {
                uint x = pairs[p].x, y = pairs[p].y;
                if (dist[y] < dist[x]) {
                    float td = dist[x]; dist[x] = dist[y]; dist[y] = td;
                    uint tc = child[x]; child[x] = child[y]; child[y] = tc;
                }
            }
        }
        return child;
    }
    if (type > 3u) return uint4(0xffffffffu);
    float3 v[3];
    for (uint i = 0u; i < 3u; i++) {
        ulong b = a + 12ul * i;
        v[i] = float3(F(ld32(mem, w, b)), F(ld32(mem, w, b + 4ul)), F(ld32(mem, w, b + 8ul))) - o;
    }
    // Watertight test (Woop, Benthin and Wald, JCGT 2013), as radv does it.
    float3 ad = abs(dir);
    uint kz = ad.x >= ad.y ? (ad.x >= ad.z ? 0u : 2u) : (ad.y >= ad.z ? 1u : 2u);
    uint kx = (kz + 1u) % 3u, ky = (kx + 1u) % 3u;
    if (dir[kz] < 0.0f) { uint t = kx; kx = ky; ky = t; }
    float sz = 1.0f / dir[kz], sx = dir[kx] * sz, sy = dir[ky] * sz;
    float ax = v[0][kx] - sx * v[0][kz], ay = v[0][ky] - sy * v[0][kz];
    float bx = v[1][kx] - sx * v[1][kz], by = v[1][ky] - sy * v[1][kz];
    float cx = v[2][kx] - sx * v[2][kz], cy = v[2][ky] - sy * v[2][kz];
    float u = cx * by - cy * bx, vv = ax * cy - ay * cx, ww = bx * ay - by * ax;
    bool back = u < 0.0f || vv < 0.0f || ww < 0.0f;
    bool front = u > 0.0f || vv > 0.0f || ww > 0.0f;
    if (!(back && front)) {
        float det = u + vv + ww;
        float t = u * sz * v[0][kz] + vv * sz * v[1][kz] + ww * sz * v[2][kz];
        if (!(sign(det) * t < 0.0f)) return uint4(U(t), U(det), U(vv), U(ww));
    }
    return uint4(U(INFINITY), U(1.0f), 0u, 0u);
}

// Byte offset into the memory buffer of a buffer element, or OOB_OFF when out of range.
static inline long buffer_offset(Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    ulong base = ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32);
    uint stride = (vs.y >> 16) & 0x3fffu;
    if (stride != 0u && idxen != 0u ? index >= vs.z : offset >= vs.z) return OOB_OFF;
    return goff(mem_base, base + ulong(index) * ulong(stride) * ulong(idxen) + ulong(offset));
}

// As `buffer_offset`, for stores and atomics: OOB_OFF inside [ro, ro_end) too.
static inline long buffer_store_offset(Win w, uint4 vs, uint index, uint offset, uint idxen) {
    ulong a = (ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32))
        + ulong(index) * ulong((vs.y >> 16) & 0x3fffu) * ulong(idxen) + ulong(offset);
    if (a >= w.ro && a < w.ro_end) return OOB_OFF;
    return buffer_offset(w, vs, index, offset, idxen);
}

static inline uint4 buffer_load_raw(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    if (o == OOB_OFF) return uint4(0u);
    device const uint* p = (device const uint*)(mem + o);
    return uint4(p[0], p[1], p[2], p[3]);
}

// GLC loads and stores, which other groups may be polling or writing while
// this one runs: atomics, so the compiler keeps them in loops and in order.
static inline uint4 buffer_load_atomic(device uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen, uint n) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    uint4 r = uint4(0u);
    if (o == OOB_OFF) return r;
    if ((o & 3) != 0) return buffer_load_raw(mem, mem_base, vs, index, offset, idxen);
    for (uint i = 0u; i < n; i++) r[i] = atomic_load_explicit((device atomic_uint*)(mem + o) + i, memory_order_relaxed);
    return r;
}

static inline uint4 buffer_load_volatile(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen, uint n) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    uint4 r = uint4(0u);
    if (o == OOB_OFF) return r;
    for (uint i = 0u; i < n; i++) r[i] = ((volatile device const uint*)(mem + o))[i];
    return r;
}

static inline void store_atomic(device uchar* mem, long o, uint v) {
    if ((o & 3) != 0) { *(device uint*)(mem + o) = v; return; }
    atomic_store_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed);
}

static inline uint4 buffer_load_ubyte(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o == OOB_OFF ? 0u : uint(mem[o]), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sbyte(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o == OOB_OFF ? 0u : uint(int(char(mem[o]))), 0u, 0u, 0u);
}

static inline uint4 buffer_load_ushort(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o == OOB_OFF ? 0u : uint(*(device const ushort*)(mem + o)), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sshort(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o == OOB_OFF ? 0u : uint(int(*(device const short*)(mem + o))), 0u, 0u, 0u);
}

static inline uint4 flat_load_raw(device const uchar* mem, long o) {
    if (o == OOB_OFF) return uint4(0u);
    device const uint* p = (device const uint*)(mem + o);
    return uint4(p[0], p[1], p[2], p[3]);
}

static inline uint4 flat_load_ubyte(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(mem[o]), 0u, 0u, 0u);
}

static inline uint4 flat_load_sbyte(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(int(char(mem[o]))), 0u, 0u, 0u);
}

static inline uint4 flat_load_ushort(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(*(device const ushort*)(mem + o)), 0u, 0u, 0u);
}

static inline uint4 flat_load_sshort(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(int(*(device const short*)(mem + o))), 0u, 0u, 0u);
}

static inline uint atomic_rmw_add(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_add_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_sub(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_sub_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_and(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_and_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_or(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_or_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_xor(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_xor_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_umin(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_min_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_umax(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_max_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_smin(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_min_explicit((device atomic_int*)(mem + o), int(v), memory_order_relaxed));
}

static inline uint atomic_rmw_smax(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_max_explicit((device atomic_int*)(mem + o), int(v), memory_order_relaxed));
}

static inline uint atomic_rmw_fmin(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmin(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint atomic_rmw_fmax(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmax(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint atomic_rmw_swap(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_exchange_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_cas(device uchar* mem, long o, uint v, uint cmp) {
    if (o == OOB_OFF) return 0u;
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

// SQ_TEX_DEPTH_COMPARE: the reference against the fetched value.
static inline bool depth_compare(uint f, float ref, float v) {
    switch (f) {
    case 1u: return ref < v;
    case 2u: return ref == v;
    case 3u: return ref <= v;
    case 4u: return ref > v;
    case 5u: return ref != v;
    case 6u: return ref >= v;
    case 7u: return true;
    default: return false;
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

// Components, bits per component and conversion of a buffer format
// (GFX10 unified encoding, BUF_FMT_*); n is 0 for formats not handled.
static inline uint3 buffer_fmt(uint fmt) {
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
    return uint3(n, bits, kind);
}

// Buffer formats as raw result dwords.
static inline uint4 buffer_load_format(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    if (o == OOB_OFF) return uint4(0u);
    device const uchar* p = mem + o;
    uint3 f = buffer_fmt((vs.w >> 12) & 0x7fu);
    uint n = f.x, bits = f.y, kind = f.z;
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

// The inverse of fmt_conv: a result dword packed into a `bits`-bit component.
static inline uint fmt_pack(uint v, uint bits, uint kind) {
    uint m = bits == 32u ? 0xffffffffu : (1u << bits) - 1u;
    float f = as_type<float>(v);
    switch (kind) {
    case 0u: return uint(rint(clamp(f, 0.0f, 1.0f) * float(m))) & m;
    case 1u: return uint(int(rint(clamp(f, -1.0f, 1.0f) * float(m >> 1)))) & m;
    case 2u: return uint(clamp(f, 0.0f, float(m))) & m;
    case 3u: return uint(int(clamp(f, -float(m >> 1) - 1.0f, float(m >> 1)))) & m;
    case 4u: case 5u: return v & m;
    default: return bits == 16u ? uint(as_type<ushort>(half(f))) : v;
    }
}

static inline void st_le(device uchar* p, uint bytes, uint v) {
    for (uint i = 0u; i < bytes; i++) p[i] = uchar(v >> (8u * i));
}

// Typed store: the first n data components go to the format's components in
// order; components past n are left as they are.
static inline void buffer_store_format(device uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen, uint4 d, uint n) {
    long o = buffer_store_offset(mem_base, vs, index, offset, idxen);
    if (o == OOB_OFF) return;
    device uchar* p = mem + o;
    uint3 f = buffer_fmt((vs.w >> 12) & 0x7fu);
    uint bits = f.y, kind = f.z;
    if (f.x == 0u) return;
    if (bits == 10u) {
        uint w = ld_le(p, 4u);
        uint shift[4] = {0u, 10u, 20u, 30u};
        for (uint i = 0u; i < min(n, 4u); i++) {
            uint b = i == 3u ? 2u : 10u;
            uint m = ((1u << b) - 1u) << shift[i];
            w = (w & ~m) | ((fmt_pack(d[i], b, kind) << shift[i]) & m);
        }
        st_le(p, 4u, w);
        return;
    }
    uint bytes = bits / 8u;
    for (uint i = 0u; i < min(n, f.x); i++) st_le(p + i * bytes, bytes, fmt_pack(d[i], bits, kind));
}

"#;

const LDS_PRELUDE: &str = r#"// A branch vote over a wave64 group run as two SIMD groups (lockstep programs).
static inline bool wave_any(bool x, threadgroup uint* vote, ushort sgi) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool a = simd_any(x);
    if (simd_is_first()) vote[sgi & 1] = a ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (vote[0] | vote[1]) != 0u;
}

static inline uint lds_rmw_add(threadgroup uint* lds, uint a, uint v) {
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

static inline uint lds_rmw_fmin(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmin(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint lds_rmw_fmax(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmax(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint lds_rmw_rsub(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &e, v - e, memory_order_relaxed, memory_order_relaxed)) {}
    return e;
}
"#;
