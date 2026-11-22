//! Translates a GFX10 shader program to Metal Shading Language.
//!
//! The model is a straight register one, see `docs/shaders.md`: every Metal thread
//! is one lane, SGPRs and VGPRs are local `uint`s, a lane mask holds this lane in
//! bit 0 of its low half (the whole wave's mask in lockstep programs), and branches
//! become a `switch` inside a loop.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::decode::{Inst, Kind, Mimg, VEnc, Valu, decode, src};
use crate::opinfo::Fmt;
use crate::print::{Syntax, mimg_addr_count, print};

mod emit;
mod flow;
mod half;
mod image;
mod mem;
mod prelude;
mod scalar;
mod sources;
mod valu;

use flow::{branch_target, crosses_lanes};
use half::{HALF_OPS, PACKED_F16};
use prelude::{GUARD_PRELUDE, LDS_PRELUDE, LOCKSTEP_BARRIER, PRELUDE, WAVE_PRELUDE};
use sources::TRACKED_SGPRS;
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
    /// Compute stage: threads at or past the grid size in `bounds` (three
    /// dwords at [`binding::BOUNDS`]) start with EXEC clear.
    pub grid_bounds: bool,
    /// Pixel stage: no depth target is bound, so a depth export is dropped.
    pub drop_depth_export: bool,
    /// Pixel stage, for debugging: the first flat input's raw value goes to
    /// colour output 1 in place of what the program computes.
    pub probe_flat: bool,
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
    /// `device const uint* bounds`: the grid size in threads, for [`Request::grid_bounds`].
    pub const BOUNDS: u32 = 24;
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
    if req.grid_bounds && tx.lockstep && req.threads[1..].iter().any(|&t| t > 1) {
        return Err(err(
            0,
            "bounds of a lockstep program with a multi-dimensional group",
        ));
    }
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
    /// The program stores to scratch (private per-thread memory).
    scratch_stores: bool,
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
    sample_mask: bool,
    mem: bool,
    mem_write: bool,
    /// See [`Translated::store_buffers`].
    stores: Option<Vec<(Source, bool)>>,
    lds: bool,
    gds: bool,
    /// Non-compute per-lane `ds_write_addtid_b32`/`ds_read_addtid_b32` spill slots.
    addtid_local: bool,
    addtid_max_off: u32,
    addtid_aligned128: bool,
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
            && insts.iter().any(|(_, i)| {
                matches!(i.kind, Kind::Ds { gds: false, .. }) || crosses_lanes(i, &insts)
            });
        Tx {
            req,
            insts,
            wave32,
            lockstep,
            has_barrier: false,
            scratch_stores: false,
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
            sample_mask: false,
            mem: false,
            mem_write: false,
            stores: Some(Vec::new()),
            lds: false,
            gds: false,
            addtid_local: false,
            addtid_max_off: 0,
            addtid_aligned128: true,
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
            src::VCCZ => format!("uint(!{})", self.mask_any(106)),
            src::EXECZ => format!("uint(!{})", self.mask_any(126)),
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

    // Assigns a VALU lane mask to `dst` (an SGPR pair, or one in wave32): this
    // thread's bit, or in lockstep programs the whole wave's mask.
    fn set_mask(&mut self, dst: u16, bit: &str) -> String {
        self.written(dst);
        let lo = self.s(dst);
        if self.wave32 {
            return format!(" {lo} = uint({bit});");
        }
        self.written(dst + 1);
        let hi = self.s(dst + 1);
        if self.lockstep {
            return format!(
                " {{ ulong m = wave_ballot({bit}, lockstep_vote, sgi, wk); {lo} = uint(m); {hi} = uint(m >> 32); }}"
            );
        }
        format!(" {lo} = uint({bit}); {hi} = 0u;")
    }

    // Whether this thread's lane is set in the mask in SGPRs `code` and `code + 1`.
    fn mask_on(&mut self, code: u16) -> String {
        let lo = self.s(code);
        if self.lockstep {
            let hi = self.s(code + 1);
            return format!("lane_on({lo}, {hi}, wl)");
        }
        format!("(({lo} & 1u) != 0u)")
    }

    // Whether any lane is set in the mask in SGPRs `code` and `code + 1`.
    fn mask_any(&mut self, code: u16) -> String {
        let lo = self.s(code);
        if self.lockstep {
            let hi = self.s(code + 1);
            return format!("(({lo} | {hi}) != 0u)");
        }
        format!("(({lo} & 1u) != 0u)")
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

    // Consecutive guarded statements share one `if`: a branch per
    // instruction made Metal's compiler several times slower.
    fn guard(&mut self, stmt: &str) {
        let e = self.s(126);
        let on = self.mask_on(126);
        let extra = if self.dpp_lane_off { " && dpp_on" } else { "" };
        let cond = format!("{on}{extra}");
        if self.open_guard.as_ref() != Some(&cond) {
            self.close_guard();
            self.body.push_str(&format!("        if ({cond}) {{\n"));
            self.open_guard = Some(cond);
        }
        self.body.push_str(&format!("        {{ {stmt} }}\n"));
        if stmt.contains(&format!("{e} ="))
            || stmt.contains("exec_hi =")
            || stmt.contains("dpp_on =")
        {
            self.close_guard();
        }
    }

    fn close_guard(&mut self) {
        if self.open_guard.take().is_some() {
            self.body.push_str("        }\n");
        }
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
