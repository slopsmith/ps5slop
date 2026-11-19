//! Prints instructions in LLVM's AMDGPU assembly syntax, as
//! `llvm-mc --disassemble -mcpu=gfx1013` does.

use std::fmt::Write;

use crate::decode::{Buf, Flat, Inst, Kind, Mimg, VEnc, Valu, is_vop3b, src};
use crate::opinfo::{Fmt, OpInfo, Ty};

/// Printing options.
#[derive(Clone, Copy, Debug)]
pub struct Syntax {
    /// Lane masks are 64 bits wide (`vcc`) rather than 32 (`vcc_lo`).
    pub wave64: bool,
}

/// Formats `inst`. Encodings LLVM rejects still print, with the odd parts marked.
pub fn print(inst: &Inst, syn: Syntax) -> String {
    run(inst, syn).0
}

/// Formats `inst`, or returns `None` for an encoding LLVM's disassembler rejects or
/// flags as invalid (reserved operands, misaligned or out-of-range registers, fields
/// that must be zero, modifiers the opcode does not take).
pub fn print_checked(inst: &Inst, syn: Syntax) -> Option<String> {
    let (s, ok) = run(inst, syn);
    ok.then_some(s)
}

fn run(inst: &Inst, syn: Syntax) -> (String, bool) {
    let mut p = Printer {
        inst,
        syn,
        out: String::new(),
        nargs: 0,
        ok: true,
    };
    p.inst();
    (p.out, p.ok)
}

struct Printer<'a> {
    inst: &'a Inst,
    syn: Syntax,
    out: String,
    nargs: usize,
    ok: bool,
}

/// Name of the scalar register range of `dwords` starting at operand code `code`,
/// and whether LLVM accepts it (ranges are aligned down to their size, as LLVM does).
pub fn sreg(code: u16, dwords: u8) -> (String, bool) {
    let n = dwords as u16;
    let shift = match n {
        1 => 0,
        2 => 1,
        _ => 2,
    };
    let range = |prefix: &str, first: u16, limit: u16| {
        let first = first >> shift << shift;
        let ok = first + n <= limit;
        if n == 1 {
            (format!("{prefix}{first}"), ok)
        } else {
            (format!("{prefix}[{}:{}]", first, first + n - 1), ok)
        }
    };
    match code {
        0..=105 => range("s", code, 106),
        108..=123 => range("ttmp", code - 108, 16),
        src::VCC_LO if n == 2 => ("vcc".into(), true),
        src::EXEC_LO if n == 2 => ("exec".into(), true),
        src::NULL => ("null".into(), true),
        _ if n > 1 => (format!("<sreg {code}>"), false),
        src::VCC_LO => ("vcc_lo".into(), true),
        src::VCC_HI => ("vcc_hi".into(), true),
        src::M0 => ("m0".into(), true),
        src::EXEC_LO => ("exec_lo".into(), true),
        src::EXEC_HI => ("exec_hi".into(), true),
        _ => (format!("<sreg {code}>"), false),
    }
}

/// Name of a vector register range.
pub fn vreg(first: u16, dwords: u8) -> String {
    if dwords <= 1 {
        format!("v{first}")
    } else {
        format!("v[{}:{}]", first, first + dwords as u16 - 1)
    }
}

const F16_INLINE: [u16; 9] = [
    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
];

fn float_inline(code: u16, bits: u8) -> &'static str {
    match code {
        240 => "0.5",
        241 => "-0.5",
        242 => "1.0",
        243 => "-1.0",
        244 => "2.0",
        245 => "-2.0",
        246 => "4.0",
        247 => "-4.0",
        _ if bits == 64 => "0.15915494309189532",
        _ => "0.15915494",
    }
}

/// Whether a source operand code is an inline constant or literal.
pub fn is_constant(code: u16) -> bool {
    matches!(code, 128..=208 | 240..=248 | src::LITERAL)
}

/// Formats a 9-bit source operand of type `ty`, and whether it is a valid operand.
pub fn operand(code: u16, ty: Ty, literal: Option<u32>, wave64: bool) -> (String, bool) {
    let dw = ty.dwords(wave64);
    match code {
        0..=127 => sreg(code, dw),
        128..=192 => (format!("{}", code as i32 - 128), true),
        193..=208 => (format!("{}", 192 - code as i32), true),
        235 => ("src_shared_base".into(), true),
        236 => ("src_shared_limit".into(), true),
        237 => ("src_private_base".into(), true),
        238 => ("src_private_limit".into(), true),
        239 => ("src_pops_exiting_wave_id".into(), true),
        240..=248 if ty == Ty::U16 => (format!("0x{:x}", F16_INLINE[code as usize - 240]), true),
        240..=248 => (float_inline(code, ty.bits()).into(), true),
        src::VCCZ => ("src_vccz".into(), true),
        src::EXECZ => ("src_execz".into(), true),
        src::SCC => ("src_scc".into(), true),
        src::LDS_DIRECT => ("src_lds_direct".into(), dw == 1),
        src::LITERAL => match literal {
            Some(v) if ty.bits() == 16 && !matches!(ty, Ty::PkF16 | Ty::PkU16) => {
                (literal16(v as u16), true)
            }
            Some(v) => (literal32(v), true),
            None => ("<literal>".into(), false),
        },
        256..=511 => (vreg(code - 256, dw), code + dw as u16 <= 512),
        _ => (format!("<src {code}>"), false),
    }
}

const F32_INLINE: [u32; 9] = [
    0x3F00_0000,
    0xBF00_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x4000_0000,
    0xC000_0000,
    0x4080_0000,
    0xC080_0000,
    0x3E22_F983,
];

// A 32-bit literal; LLVM prints values that equal an inline constant like the constant.
fn literal32(v: u32) -> String {
    if v <= 64 || (v as i32) >= -16 && (v as i32) < 0 {
        return format!("{}", v as i32);
    }
    match F32_INLINE.iter().position(|&f| f == v) {
        Some(i) => float_inline(240 + i as u16, 32).into(),
        None => format!("0x{v:x}"),
    }
}

fn literal16(v: u16) -> String {
    if v <= 64 || (v as i16) >= -16 && (v as i16) < 0 {
        return format!("{}", v as i16);
    }
    match F16_INLINE.iter().position(|&f| f == v) {
        Some(i) => float_inline(240 + i as u16, 16).into(),
        None => format!("0x{v:x}"),
    }
}

// An immediate as LLVM prints a plain integer: decimal in the inline range, else hex.
fn imm(v: u16) -> String {
    if v <= 64 {
        format!("{v}")
    } else {
        format!("0x{v:x}")
    }
}

fn explicit_srcs(op: &OpInfo) -> Vec<Ty> {
    op.srcs
        .iter()
        .copied()
        .filter(|t| !matches!(t, Ty::Scc | Ty::M0 | Ty::Exec))
        .collect()
}

fn explicit_defs(op: &OpInfo) -> Vec<Ty> {
    op.defs
        .iter()
        .copied()
        .filter(|t| !matches!(t, Ty::Scc | Ty::Exec))
        .collect()
}

/// Whether the opcode takes floating-point input modifiers (neg, abs, omod).
pub fn has_fp_mods(op: &OpInfo) -> bool {
    op.name == "v_cndmask_b32" || op.srcs.iter().any(|t| t.is_float())
}

/// Whether VOP3 `op_sel` applies: opcodes on 16-bit values, and the byte and bit aligners.
pub fn is_16bit(op: &OpInfo) -> bool {
    matches!(op.name, "v_alignbit_b32" | "v_alignbyte_b32")
        || op
            .srcs
            .iter()
            .chain(op.defs)
            .any(|t| matches!(t, Ty::F16 | Ty::U16))
}

fn is_64bit(op: &OpInfo) -> bool {
    op.srcs
        .iter()
        .chain(op.defs)
        .any(|t| matches!(t, Ty::F64 | Ty::U64 | Ty::I64))
}

fn omod_ok(op: &OpInfo) -> bool {
    op.fmt != Fmt::Vopc
        && op.name != "v_cvt_pk_u8_f32"
        && op.name != "v_cndmask_b32"
        && op.srcs.iter().chain(op.defs).any(|t| t.is_float())
}

fn clamp_ok(op: &OpInfo) -> bool {
    let n = op.name;
    if op.fmt == Fmt::Vopc {
        return op.srcs.iter().any(|t| t.is_float()) && !n.contains("class");
    }
    if n == "v_cndmask_b32" {
        return false;
    }
    if op.srcs.iter().chain(op.defs).any(|t| t.is_float()) {
        return true;
    }
    [
        "_nc_", "_co_", "v_mad_", "v_sad_", "v_msad_", "qsad", "_u32_u24", "_i32_i24", "v_cvt_",
    ]
    .iter()
    .any(|p| n.contains(p))
}

// Number of source fields a VOP3 or VOP3P opcode uses.
fn encoded_srcs(op: &OpInfo) -> usize {
    let n = op.name;
    if n.starts_with("v_mac_") || n.starts_with("v_fmac_") || n == "v_writelane_b32" {
        return 2;
    }
    op.srcs
        .iter()
        .filter(|t| !matches!(t, Ty::M0 | Ty::Exec | Ty::Scc))
        .count()
        .min(3)
}

fn mod_ok(op: &OpInfo, t: Ty) -> bool {
    t.is_float() || op.name == "v_cndmask_b32"
}

// VALU opcodes LLVM knows only in their 32-bit form, printed without `_e32`.
fn e32_only(name: &str) -> bool {
    matches!(
        name,
        "v_readfirstlane_b32" | "v_swap_b32" | "v_swaprel_b32" | "v_pk_fmac_f16" | "v_illegal"
    ) || name.starts_with("v_madmk")
        || name.starts_with("v_madak")
        || name.starts_with("v_fmamk")
        || name.starts_with("v_fmaak")
}

impl Printer<'_> {
    fn w(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn arg(&mut self, s: &str) {
        self.out.push_str(if self.nargs == 0 { " " } else { ", " });
        self.nargs += 1;
        self.out.push_str(s);
    }

    fn checked(&mut self, (s, ok): (String, bool)) -> String {
        self.ok &= ok;
        s
    }

    fn vreg_arg(&mut self, first: u16, n: u8) {
        self.require(first + n as u16 <= 256);
        self.arg(&vreg(first, n));
    }

    fn sreg_arg(&mut self, code: u16, n: u8) {
        let s = self.checked(sreg(code, n));
        self.arg(&s);
    }

    fn src(&mut self, code: u16, ty: Ty) -> String {
        let r = operand(code, ty, self.inst.literal, self.syn.wave64);
        self.checked(r)
    }

    fn flag(&mut self, on: bool, s: &str) {
        if on {
            self.out.push(' ');
            self.out.push_str(s);
        }
    }

    fn require(&mut self, cond: bool) {
        self.ok &= cond;
    }

    fn lane(&self) -> u8 {
        1 + self.syn.wave64 as u8
    }

    fn inst(&mut self) {
        let op = self.inst.op;
        match self.inst.kind {
            Kind::Sop2 { sdst, ssrc0, ssrc1 } => {
                self.w(op.name);
                let d = explicit_defs(op);
                let s = explicit_srcs(op);
                self.sreg_arg(sdst as u16, d[0].dwords(true));
                self.scalar_src(ssrc0, s[0]);
                self.scalar_src(ssrc1, s[1]);
            }
            Kind::Sop1 { sdst, ssrc0 } => {
                self.w(op.name);
                let d = explicit_defs(op);
                let s = explicit_srcs(op);
                if let Some(t) = d.first() {
                    self.sreg_arg(sdst as u16, t.dwords(true));
                }
                if s.is_empty() {
                    self.require(ssrc0 != src::LITERAL);
                }
                if let Some(&t) = s.first() {
                    let reg_only = matches!(
                        op.name,
                        "s_setpc_b64" | "s_rfe_b64" | "s_movrels_b32" | "s_movrels_b64"
                    );
                    if reg_only {
                        self.require(!is_constant(ssrc0));
                    }
                    self.scalar_src(ssrc0, t);
                }
            }
            Kind::Sopc { ssrc0, ssrc1 } => {
                self.w(op.name);
                let s = explicit_srcs(op);
                self.scalar_src(ssrc0, s[0]);
                self.scalar_src(ssrc1, s[1]);
            }
            Kind::Sopk { sdst, simm16 } => self.sopk(sdst, simm16),
            Kind::Sopp { simm16 } => self.sopp(simm16),
            Kind::Smem {
                sdata,
                sbase,
                glc,
                dlc,
                offset,
                soffset,
            } => self.smem(sdata, sbase, glc, dlc, offset, soffset),
            Kind::Valu(v) => self.valu(&v),
            Kind::Vintrp {
                vdst,
                vsrc,
                attr,
                chan,
            } => {
                write!(self.out, "{}_e32", op.name).unwrap();
                self.vreg_arg(vdst as u16, 1);
                if op.name == "v_interp_mov_f32" {
                    let p = match vsrc {
                        0 => "p10".to_string(),
                        1 => "p20".to_string(),
                        2 => "p0".to_string(),
                        n => format!("invalid_param_{n}"),
                    };
                    self.require(vsrc <= 2);
                    self.arg(&p);
                } else {
                    self.vreg_arg(vsrc as u16, 1);
                }
                self.arg(&format!(
                    "attr{attr}.{}",
                    ['x', 'y', 'z', 'w'][chan as usize]
                ));
            }
            Kind::Ds {
                offset0,
                offset1,
                gds,
                addr,
                data0,
                data1,
                vdst,
            } => self.ds(offset0, offset1, gds, addr, data0, data1, vdst),
            Kind::Mubuf(b) => self.buf(&b, false),
            Kind::Mtbuf(b) => self.buf(&b, true),
            Kind::Mimg(m) => self.mimg(&m),
            Kind::Exp {
                en,
                target,
                compr,
                done,
                vm,
                vsrc,
            } => self.exp(en, target, compr, done, vm, vsrc),
            Kind::Flat(f) => self.flat(&f),
        }
    }

    fn scalar_src(&mut self, code: u16, t: Ty) {
        self.require(!matches!(
            code,
            src::LDS_DIRECT | src::SDWA | src::DPP16 | 209..=234
        ));
        let s = self.src(code, t);
        self.arg(&s);
    }

    fn sopk(&mut self, sdst: u8, simm16: u16) {
        let name = self.inst.op.name;
        self.w(name);
        match name {
            "s_getreg_b32" => {
                self.sreg_arg(sdst as u16, 1);
                self.arg(&hwreg(simm16));
            }
            "s_setreg_b32" => {
                self.arg(&hwreg(simm16));
                self.sreg_arg(sdst as u16, 1);
            }
            "s_setreg_imm32_b32" => {
                self.arg(&hwreg(simm16));
                let lit = self.inst.literal.unwrap_or(0);
                self.arg(&format!("0x{lit:x}"));
            }
            "s_version" => self.arg(&version(simm16)),
            "s_call_b64" => {
                self.sreg_arg(sdst as u16, 2);
                self.arg(&format!("{simm16}"));
            }
            "s_subvector_loop_begin" | "s_subvector_loop_end" => {
                self.sreg_arg(sdst as u16, 1);
                self.arg(&format!("{simm16}"));
            }
            _ => {
                self.sreg_arg(sdst as u16, 1);
                self.arg(&format!("0x{simm16:x}"));
            }
        }
    }

    fn sopp(&mut self, simm16: u16) {
        let name = self.inst.op.name;
        self.w(name);
        match name {
            "s_endpgm" => {
                if simm16 != 0 {
                    self.arg(&format!("{simm16}"));
                }
            }
            "s_endpgm_saved"
            | "s_barrier"
            | "s_wakeup"
            | "s_icache_inv"
            | "s_code_end"
            | "s_ttracedata"
            | "s_endpgm_ordered_ps_done"
            | "s_wait_idle" => self.require(simm16 == 0),
            "s_waitcnt" => {
                let s = waitcnt(simm16);
                self.arg(&s);
            }
            "s_sendmsg" | "s_sendmsghalt" => self.arg(&sendmsg(simm16)),
            n if n.starts_with("s_branch") || n.starts_with("s_cbranch") => {
                self.arg(&format!("{simm16}"))
            }
            "s_waitcnt_depctr" => self.arg(&depctr(simm16)),
            "s_round_mode" | "s_inst_prefetch" | "s_clause" | "s_ttracedata_imm" => {
                self.arg(&format!("0x{simm16:x}"))
            }
            _ => self.arg(&imm(simm16)),
        }
    }

    fn smem(&mut self, sdata: u8, sbase: u8, glc: bool, dlc: bool, offset: i32, soffset: u8) {
        let name = self.inst.op.name;
        self.w(name);
        let size = smem_dwords(name);
        let base = if name.starts_with("s_buffer") || name == "s_atc_probe_buffer" {
            4
        } else {
            2
        };
        let no_addr = matches!(
            name,
            "s_memtime" | "s_memrealtime" | "s_get_waveid_in_workgroup"
        );
        if matches!(name, "s_dcache_inv" | "s_dcache_wb" | "s_gl1_inv") {
            return;
        }
        let probe = name.starts_with("s_atc_probe") || name.starts_with("s_dcache_discard");
        if name.starts_with("s_atc_probe") {
            self.arg(&imm(sdata as u16));
        } else if name.starts_with("s_dcache_discard") {
        } else {
            self.sreg_arg(sdata as u16, size);
        }
        self.require(probe || !matches!(sdata, 124 | 126 | 127));
        if no_addr {
            return;
        }
        self.sreg_arg(sbase as u16, base);
        if soffset as u16 != src::NULL {
            self.sreg_arg(soffset as u16, 1);
            if offset != 0 {
                write!(self.out, " offset:{}", hex_signed(offset)).unwrap();
            }
        } else if offset == 0 {
            self.arg("null");
        } else {
            self.arg(&hex_signed(offset));
        }
        if !probe {
            self.flag(glc, "glc");
            self.flag(dlc, "dlc");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ds(&mut self, o0: u8, o1: u8, gds: bool, addr: u8, d0: u8, d1: u8, vdst: u8) {
        let name = self.inst.op.name;
        self.w(name);
        let s = ds_shape(name);
        if s.ret > 0 {
            self.vreg_arg(vdst as u16, s.ret);
        } else {
            self.require(vdst == 0);
        }
        if s.addr {
            self.vreg_arg(addr as u16, 1);
        } else {
            self.require(addr == 0);
        }
        if s.data > 0 {
            self.vreg_arg(d0 as u16, s.data);
        } else {
            self.require(d0 == 0);
        }
        if s.data1 {
            self.vreg_arg(d1 as u16, s.data);
        } else {
            self.require(d1 == 0);
        }
        if s.two_offsets {
            if o0 != 0 {
                write!(self.out, " offset0:{o0}").unwrap();
            }
            if o1 != 0 {
                write!(self.out, " offset1:{o1}").unwrap();
            }
        } else {
            let off = o0 as u16 | (o1 as u16) << 8;
            if name == "ds_swizzle_b32" && off != 0 {
                write!(self.out, " offset:{}", swizzle(off)).unwrap();
            } else if off != 0 {
                write!(self.out, " offset:{off}").unwrap();
            }
        }
        self.flag(gds, "gds");
    }

    fn buf(&mut self, b: &Buf, mtbuf: bool) {
        let name = self.inst.op.name;
        self.w(name);
        let atomic = name.contains("atomic");
        if matches!(name, "buffer_gl0_inv" | "buffer_gl1_inv") {
            let zero = Buf {
                offset: 0,
                offen: false,
                idxen: false,
                glc: false,
                dlc: false,
                slc: false,
                lds: false,
                tfe: false,
                format: 0,
                vaddr: 0,
                vdata: 0,
                srsrc: 0,
                soffset: 0,
            };
            self.require(*b == zero);
            return;
        }
        if b.lds {
            let small = ["ubyte", "sbyte", "ushort", "sshort", "dword"]
                .iter()
                .any(|s| name == format!("buffer_load_{s}"));
            self.require(small && !b.tfe);
        } else {
            let data = buf_dwords(name) + b.tfe as u8;
            self.vreg_arg(b.vdata as u16, data);
        }
        self.require(!(atomic && (b.dlc || b.tfe)));
        self.require(!(mtbuf && b.tfe));
        if b.offen || b.idxen {
            self.vreg_arg(b.vaddr as u16, if b.offen && b.idxen { 2 } else { 1 });
        } else {
            self.arg("off");
        }
        self.sreg_arg(b.srsrc as u16, 4);
        self.require(!matches!(
            b.soffset as u16,
            209..=234 | 249 | 250 | 254 | 255
        ));
        let so = operand(b.soffset as u16, Ty::U32, None, self.syn.wave64);
        let so = self.checked(so);
        self.arg(&so);
        if mtbuf && b.format != 1 {
            match BUF_FORMATS.get(b.format as usize) {
                Some(f) => write!(self.out, " format:[BUF_FMT_{f}]").unwrap(),
                None => write!(self.out, " format:{}", b.format).unwrap(),
            }
        }
        self.flag(b.idxen, "idxen");
        self.flag(b.offen, "offen");
        if b.offset != 0 {
            write!(self.out, " offset:{}", b.offset).unwrap();
        }
        self.flag(b.glc, "glc");
        self.flag(b.slc, "slc");
        self.flag(b.dlc, "dlc");
        self.flag(b.lds, "lds");
        self.flag(b.tfe, "tfe");
    }

    fn mimg(&mut self, m: &Mimg) {
        let name = self.inst.op.name;
        self.w(name);
        if m.nsa > 0 && mimg_has_sampler(name) {
            self.w(" ");
        }
        let bvh = name.starts_with("image_bvh");
        self.require(!(bvh && (m.tfe || m.lwe || m.d16)));
        let mut data = if name.starts_with("image_gather4") || bvh {
            4
        } else {
            m.dmask.count_ones() as u8
        }
        .max(1);
        if m.d16 && !bvh {
            data = data.div_ceil(2);
        }
        data += m.tfe as u8;
        self.vreg_arg(m.vdata as u16, data);
        let n = mimg_addr_count(name, m.dim, m.a16);
        if name.starts_with("image_gather4") {
            self.require(m.dmask.count_ones() == 1 && !(m.d16 && m.tfe));
        }
        if name.starts_with("image_atomic") {
            self.require(matches!(m.dmask, 1 | 3 | 0xF));
        }
        if name == "image_get_lod" || name.contains("_pck") || name.starts_with("image_atomic") {
            self.require(!m.d16);
        }
        if m.nsa > 0 {
            self.require(n > 1 && m.nsa as usize == (n as usize - 1).div_ceil(4));
            self.require(sampler_addr_range(name).is_none_or(|(_, hi)| n <= hi));
            let regs: Vec<String> = (0..n as usize)
                .map(|i| vreg(m.vaddr[i] as u16, 1))
                .collect();
            self.arg(&format!("[{}]", regs.join(", ")));
        } else {
            let regs = mimg_vaddr_regs(name, n);
            self.require(regs.is_some());
            self.vreg_arg(m.vaddr[0] as u16, regs.unwrap_or(n));
        }
        self.sreg_arg(m.srsrc as u16, if bvh { 4 } else { 8 });
        if mimg_has_sampler(name) {
            self.sreg_arg(m.ssamp as u16, 4);
        } else {
            self.require(m.ssamp == 0);
        }
        if bvh {
            self.flag(m.a16, "a16");
            return;
        }
        if m.dmask != 0 {
            write!(self.out, " dmask:0x{:x}", m.dmask).unwrap();
        }
        write!(self.out, " dim:SQ_RSRC_IMG_{}", DIMS[m.dim as usize]).unwrap();
        self.flag(m.unorm, "unorm");
        self.flag(m.glc, "glc");
        self.flag(m.slc, "slc");
        self.flag(m.dlc, "dlc");
        self.flag(m.r128, "r128");
        self.flag(m.a16, "a16");
        self.flag(m.tfe, "tfe");
        self.flag(m.lwe, "lwe");
        self.flag(m.d16, "d16");
    }

    fn exp(&mut self, en: u8, target: u8, compr: bool, done: bool, vm: bool, vsrc: [u8; 4]) {
        self.w("exp");
        let t = match target {
            0..=7 => format!("mrt{target}"),
            8 => "mrtz".into(),
            9 => "null".into(),
            12..=16 => format!("pos{}", target - 12),
            20 => "prim".into(),
            32..=63 => format!("param{}", target - 32),
            _ => format!("invalid_target_{target}"),
        };
        self.require(matches!(target, 0..=9 | 12..=16 | 20 | 32..=63));
        write!(self.out, " {t}").unwrap();
        for i in 0..4 {
            let reg = if compr { vsrc[i / 2] } else { vsrc[i] };
            let s = if en >> i & 1 != 0 {
                vreg(reg as u16, 1)
            } else {
                "off".into()
            };
            self.arg(&s);
        }
        self.flag(done, "done");
        self.flag(compr, "compr");
        self.flag(vm, "vm");
    }

    fn flat(&mut self, f: &Flat) {
        let name = self.inst.op.name;
        self.w(name);
        let seg = self.inst.op.fmt;
        let (load, store, size) = flat_shape(name);
        let atomic = name.contains("atomic");
        let off = f.saddr as u16 == 0x7D;
        self.require(!f.nv && !matches!(f.saddr, 124 | 126 | 127));
        if seg == Fmt::Flat {
            self.require(off && !f.lds);
        }
        if f.lds {
            let small = ["ubyte", "sbyte", "ushort", "sshort", "dword"]
                .iter()
                .any(|s| name.ends_with(&format!("_load_{s}")));
            self.require(small);
        }
        let returns = (load && !f.lds) || (atomic && f.glc);
        if returns {
            self.vreg_arg(
                f.vdst as u16,
                if atomic && name.contains("cmpswap") {
                    size / 2
                } else {
                    size
                },
            );
        }
        match seg {
            Fmt::Scratch if !off => self.arg("off"),
            Fmt::Scratch => self.vreg_arg(f.addr as u16, 1),
            Fmt::Global if !off => self.vreg_arg(f.addr as u16, 1),
            _ => self.vreg_arg(f.addr as u16, 2),
        }
        if store || atomic {
            self.vreg_arg(f.data as u16, size);
        }
        match seg {
            Fmt::Flat => {}
            _ if off => self.arg("off"),
            Fmt::Global => self.sreg_arg(f.saddr as u16, 2),
            _ => self.sreg_arg(f.saddr as u16, 1),
        }
        if f.offset != 0 {
            write!(self.out, " offset:{}", f.offset).unwrap();
        }
        self.flag(f.glc, "glc");
        self.flag(f.slc, "slc");
        self.flag(f.dlc, "dlc");
        self.flag(f.lds, "lds");
    }

    fn valu(&mut self, v: &Valu) {
        let op = self.inst.op;
        let name = op.name;
        let promoted = matches!(op.fmt, Fmt::Vop1 | Fmt::Vop2 | Fmt::Vopc | Fmt::Vintrp);
        let no_operands = op.defs.is_empty() && op.srcs.is_empty();
        let suffix = match v.enc {
            VEnc::E32 if e32_only(name) || no_operands => "",
            VEnc::E32 => "_e32",
            VEnc::E64 if promoted => "_e64",
            VEnc::Sdwa => "_sdwa",
            VEnc::Dpp16 | VEnc::Dpp8 => "_dpp",
            _ => "",
        };
        write!(self.out, "{name}{suffix}").unwrap();
        let fp = has_fp_mods(op);
        let lane = self.lane();
        let cmpx = name.starts_with("v_cmpx");
        if name.starts_with("v_interp_") && op.fmt == Fmt::Vop3 {
            return self.vop3_interp(v);
        }
        if op.fmt == Fmt::Vintrp {
            return self.vop3_vintrp(v);
        }
        if no_operands {
            self.require(v.vdst == 0 && v.src[0] == 0);
            return;
        }
        if matches!(v.enc, VEnc::Sdwa | VEnc::Dpp16 | VEnc::Dpp8) {
            self.require(
                !is_64bit(op)
                    && !op.srcs.contains(&Ty::Imm)
                    && name != "v_mac_legacy_f32"
                    && !e32_only(name),
            );
            if op.fmt == Fmt::Vop1 {
                self.require(v.neg & 2 == 0 && v.abs & 2 == 0);
            }
            self.require(!(op.fmt == Fmt::Vopc && v.enc != VEnc::Sdwa));
            if !fp {
                self.require(v.neg == 0 && v.abs == 0);
            }
        }
        if v.enc == VEnc::E64 {
            if !fp {
                self.require(v.neg == 0 && v.abs == 0);
            }
            self.require(v.omod == 0 || omod_ok(op));
            self.require(!v.clamp || clamp_ok(op));
        }
        if matches!(v.enc, VEnc::E64 | VEnc::P) {
            let used = encoded_srcs(op);
            for i in used..3 {
                self.require(v.src[i] == 0 && v.neg >> i & 1 == 0 && v.neg_hi >> i & 1 == 0);
                if v.enc == VEnc::E64 && !is_vop3b(op) {
                    self.require(v.abs >> i & 1 == 0 && v.opsel >> i & 1 == 0);
                }
                if v.enc == VEnc::P {
                    self.require(v.opsel >> i & 1 == 0);
                }
            }
        }
        if v.enc == VEnc::Sdwa {
            self.require(
                v.sdwa.src_sel[0] != 7
                    && v.sdwa.src_sel[1] != 7
                    && v.sdwa.dst_sel != 7
                    && v.sdwa.dst_unused != 3,
            );
            if fp {
                self.require(!v.sdwa.sext[0] && !v.sdwa.sext[1]);
            }
            self.require(v.omod == 0);
            for i in 0..2 {
                self.require(!matches!(v.src[i], 108..=123 | 209..=239 | 249..=255));
            }
        }

        if op.fmt == Fmt::Vopc {
            if cmpx {
                match v.enc {
                    VEnc::E64 => self.require(v.vdst as u16 == src::EXEC_LO),
                    VEnc::Sdwa => self.require(v.sdst.is_none()),
                    _ => {}
                }
            } else {
                match (v.enc, v.sdst) {
                    (VEnc::E64, _) => self.sreg_arg(v.vdst as u16, lane),
                    (_, Some(s)) => self.sreg_arg(s as u16, lane),
                    _ => self.sreg_arg(src::VCC_LO, lane),
                }
            }
        } else if let Some(&d) = op.defs.first() {
            if matches!(name, "v_readlane_b32" | "v_readfirstlane_b32") {
                self.sreg_arg(v.vdst as u16, 1);
            } else {
                self.vreg_arg(v.vdst as u16, d.dwords(self.syn.wave64));
            }
            if op.defs.len() > 1 && op.defs[1] == Ty::Lane {
                match v.sdst {
                    Some(s) if v.enc == VEnc::E64 => self.sreg_arg(s as u16, lane),
                    _ => self.sreg_arg(src::VCC_LO, lane),
                }
            }
        }

        let mut srcs: Vec<Ty> = op
            .srcs
            .iter()
            .copied()
            .filter(|t| !matches!(t, Ty::M0 | Ty::Exec))
            .collect();
        if name.starts_with("v_mac_")
            || name.starts_with("v_fmac_")
            || name == "v_pk_fmac_f16"
            || name == "v_writelane_b32"
        {
            srcs.truncate(2);
        }
        if matches!(name, "v_swap_b32" | "v_swaprel_b32") {
            srcs.truncate(1);
            self.require(v.src[0] >= 256);
        }
        if matches!(name, "v_readfirstlane_b32" | "v_readlane_b32") {
            self.require(v.src[0] >= src::VGPR0 || v.src[0] == src::LDS_DIRECT);
        }
        if name.starts_with("v_permlane") {
            self.require(v.src[0] >= src::VGPR0 && v.src[1] < src::VGPR0 && v.src[2] < src::VGPR0);
        }
        if name == "v_mqsad_u32_u8" {
            self.require(v.src[2] >= src::VGPR0);
        }
        if v.enc == VEnc::Sdwa && (name.starts_with("v_mac_") || name.starts_with("v_fmac_")) {
            self.require(v.src[0] >= src::VGPR0 && v.src[1] >= src::VGPR0);
        }
        if v.enc == VEnc::Sdwa && name.starts_with("v_cmp_class") {
            self.require(v.neg == 0 && v.abs == 0);
        }
        if name.starts_with("v_movrel") {
            self.require(v.src[0] >= 256 || name == "v_movreld_b32");
        }
        let mix = name.starts_with("v_fma_mix");
        let mut vi = 0;
        for &t in &srcs {
            if t == Ty::Imm {
                let k = format!("0x{:x}", self.inst.literal.unwrap_or(0));
                self.arg(&k);
                continue;
            }
            if t == Ty::Lane && !matches!(v.enc, VEnc::E64 | VEnc::P) {
                self.sreg_arg(src::VCC_LO, lane);
                continue;
            }
            if vi >= 3 {
                break;
            }
            let code = v.src[vi];
            let ty = if mix { Ty::F16 } else { t };
            let mut s = if t == Ty::Lane {
                let r = sreg(code, lane);
                self.require(code < 126);
                self.checked(r)
            } else {
                self.src(code, ty)
            };
            if v.enc == VEnc::P && !mix {
                vi += 1;
                self.arg(&s);
                continue;
            }
            let (neg, abs) = if mix {
                (v.neg >> vi & 1 != 0, v.neg_hi >> vi & 1 != 0)
            } else {
                (v.neg >> vi & 1 != 0, v.abs >> vi & 1 != 0)
            };
            if mix || mod_ok(op, t) {
                if abs {
                    s = format!("|{s}|");
                }
                if neg {
                    s = if !abs && is_constant(code) {
                        format!("neg({s})")
                    } else {
                        format!("-{s}")
                    };
                }
            } else if v.enc == VEnc::E64 && (neg || abs) {
                self.ok = false;
            }
            if v.enc == VEnc::Sdwa && vi < 2 && v.sdwa.sext[vi] {
                s = format!("sext({s})");
            }
            vi += 1;
            self.arg(&s);
        }
        if name.starts_with("v_madmk") || name.starts_with("v_fmamk") {
            let (head, args) = self.out.split_once(' ').unwrap();
            let a: Vec<&str> = args.split(", ").collect();
            self.out = format!("{head} {}, {}, {}, {}", a[0], a[1], a[3], a[2]);
        }
        self.modifiers(v, srcs.len().min(3), mix);
    }

    fn modifiers(&mut self, v: &Valu, nsrc: usize, mix: bool) {
        let op = self.inst.op;
        let arr = |m: u8, n: usize| {
            (0..n)
                .map(|i| format!("{}", m >> i & 1))
                .collect::<Vec<_>>()
                .join(",")
        };
        let omod = match v.omod {
            1 => " mul:2",
            2 => " mul:4",
            3 => " div:2",
            _ => "",
        };
        match v.enc {
            VEnc::E64 => {
                if is_16bit(op) && v.opsel != 0 {
                    let mut parts: Vec<String> =
                        (0..nsrc).map(|i| format!("{}", v.opsel >> i & 1)).collect();
                    parts.push(format!("{}", v.opsel >> 3 & 1));
                    write!(self.out, " op_sel:[{}]", parts.join(",")).unwrap();
                }
                self.flag(v.clamp, "clamp");
                self.w(omod);
            }
            VEnc::P => {
                if v.opsel != 0 {
                    write!(self.out, " op_sel:[{}]", arr(v.opsel, nsrc)).unwrap();
                }
                let full = (1u8 << nsrc) - 1;
                let hi = v.opsel_hi & full;
                if (mix && hi != 0) || (!mix && hi != full) {
                    write!(self.out, " op_sel_hi:[{}]", arr(v.opsel_hi, nsrc)).unwrap();
                }
                if !mix {
                    if nsrc == 2 && !has_fp_mods(op) {
                        self.require(v.neg == 0 && v.neg_hi == 0);
                    }
                    if v.neg != 0 {
                        write!(self.out, " neg_lo:[{}]", arr(v.neg, nsrc)).unwrap();
                    }
                    if v.neg_hi != 0 {
                        write!(self.out, " neg_hi:[{}]", arr(v.neg_hi, nsrc)).unwrap();
                    }
                }
                self.flag(v.clamp, "clamp");
            }
            VEnc::Sdwa => {
                self.flag(v.clamp, "clamp");
                self.w(omod);
                const SEL: [&str; 8] = [
                    "BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD", "<7>",
                ];
                if op.fmt != Fmt::Vopc {
                    write!(self.out, " dst_sel:{}", SEL[v.sdwa.dst_sel as usize]).unwrap();
                    const UNUSED: [&str; 4] =
                        ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE", "<3>"];
                    write!(
                        self.out,
                        " dst_unused:{}",
                        UNUSED[v.sdwa.dst_unused as usize]
                    )
                    .unwrap();
                }
                write!(self.out, " src0_sel:{}", SEL[v.sdwa.src_sel[0] as usize]).unwrap();
                if op.fmt != Fmt::Vop1 {
                    write!(self.out, " src1_sel:{}", SEL[v.sdwa.src_sel[1] as usize]).unwrap();
                }
            }
            VEnc::Dpp16 => {
                let c = v.dpp.ctrl;
                let s = match c {
                    0..=0xFF => format!(
                        "quad_perm:[{},{},{},{}]",
                        c & 3,
                        c >> 2 & 3,
                        c >> 4 & 3,
                        c >> 6 & 3
                    ),
                    0x101..=0x10F => format!("row_shl:{}", c & 15),
                    0x111..=0x11F => format!("row_shr:{}", c & 15),
                    0x121..=0x12F => format!("row_ror:{}", c & 15),
                    0x140 => "row_mirror".into(),
                    0x141 => "row_half_mirror".into(),
                    0x150..=0x15F => format!("row_share:{}", c & 15),
                    0x160..=0x16F => format!("row_xmask:{}", c & 15),
                    _ => {
                        self.ok = false;
                        "/* Invalid dpp_ctrl value */".into()
                    }
                };
                write!(
                    self.out,
                    " {s} row_mask:0x{:x} bank_mask:0x{:x}",
                    v.dpp.row_mask, v.dpp.bank_mask
                )
                .unwrap();
                self.flag(v.dpp.bound_ctrl, "bound_ctrl:1");
                self.flag(v.dpp.fi, "fi:1");
            }
            VEnc::Dpp8 => {
                let sel: Vec<String> = (0..8)
                    .map(|i| format!("{}", v.dpp.sel8 >> (3 * i) & 7))
                    .collect();
                write!(self.out, " dpp8:[{}]", sel.join(",")).unwrap();
                self.flag(v.dpp.fi, "fi:1");
            }
            VEnc::E32 => {}
        }
    }

    // `v_interp_p1_f32`, `v_interp_p2_f32`, `v_interp_mov_f32` in VOP3 form.
    fn vop3_vintrp(&mut self, v: &Valu) {
        let a = v.src[0];
        self.require(a < 0x100 && v.src[2] == 0 && v.neg & 5 == 0 && v.abs & 5 == 0);
        self.vreg_arg(v.vdst as u16, 1);
        if self.inst.op.name == "v_interp_mov_f32" {
            self.require(v.src[1] <= 2 && v.neg == 0 && v.abs == 0);
            let p = match v.src[1] {
                0 => "p10".to_string(),
                1 => "p20".to_string(),
                2 => "p0".to_string(),
                n => format!("invalid_param_{n}"),
            };
            self.arg(&p);
        } else {
            self.require(v.src[1] >= src::VGPR0);
            let mut s = self.src(v.src[1], Ty::F32);
            if v.abs & 2 != 0 {
                s = format!("|{s}|");
            }
            if v.neg & 2 != 0 {
                s = format!("-{s}");
            }
            self.arg(&s);
        }
        self.arg(&format!(
            "attr{}.{}",
            a & 0x3F,
            ['x', 'y', 'z', 'w'][(a >> 6 & 3) as usize]
        ));
        self.flag(v.clamp, "clamp");
        match v.omod {
            1 => self.w(" mul:2"),
            2 => self.w(" mul:4"),
            3 => self.w(" div:2"),
            _ => {}
        }
    }

    // `v_interp_p1ll_f16`, `v_interp_p1lv_f16`, `v_interp_p2_f16`: the attribute sits in src0.
    fn vop3_interp(&mut self, v: &Valu) {
        let name = self.inst.op.name;
        self.vreg_arg(v.vdst as u16, 1);
        self.require(v.src[1] >= src::VGPR0 && v.omod == 0);
        if name == "v_interp_p1ll_f16" {
            self.require(v.src[2] == 0);
        } else {
            self.require(v.src[2] >= src::VGPR0);
        }
        let m = |p: &mut Self, i: usize| {
            p.require(v.src[i] != src::LITERAL);
            let s = p.src(v.src[i], Ty::F32);
            let s = if v.abs >> i & 1 != 0 {
                format!("|{s}|")
            } else {
                s
            };
            let s = if v.neg >> i & 1 != 0 {
                format!("-{s}")
            } else {
                s
            };
            p.arg(&s);
        };
        m(self, 1);
        let a = v.src[0];
        self.arg(&format!(
            "attr{}.{}",
            a & 0x3F,
            ['x', 'y', 'z', 'w'][(a >> 6 & 3) as usize]
        ));
        if name != "v_interp_p1ll_f16" {
            m(self, 2);
        }
        self.flag(a & 0x100 != 0, "high");
        self.flag(v.clamp, "clamp");
        match v.omod {
            1 => self.w(" mul:2"),
            2 => self.w(" mul:4"),
            3 => self.w(" div:2"),
            _ => {}
        }
    }
}

fn hex_signed(v: i32) -> String {
    if v < 0 {
        format!("-0x{:x}", -(v as i64))
    } else {
        format!("0x{v:x}")
    }
}

const DIMS: [&str; 8] = [
    "1D",
    "2D",
    "3D",
    "CUBE",
    "1D_ARRAY",
    "2D_ARRAY",
    "2D_MSAA",
    "2D_MSAA_ARRAY",
];

// GFX10 buffer formats 0 to 77, named as in Mesa's `gfx10-rsrc.json` (MIT).
const BUF_FORMATS: [&str; 78] = [
    "INVALID",
    "8_UNORM",
    "8_SNORM",
    "8_USCALED",
    "8_SSCALED",
    "8_UINT",
    "8_SINT",
    "16_UNORM",
    "16_SNORM",
    "16_USCALED",
    "16_SSCALED",
    "16_UINT",
    "16_SINT",
    "16_FLOAT",
    "8_8_UNORM",
    "8_8_SNORM",
    "8_8_USCALED",
    "8_8_SSCALED",
    "8_8_UINT",
    "8_8_SINT",
    "32_UINT",
    "32_SINT",
    "32_FLOAT",
    "16_16_UNORM",
    "16_16_SNORM",
    "16_16_USCALED",
    "16_16_SSCALED",
    "16_16_UINT",
    "16_16_SINT",
    "16_16_FLOAT",
    "10_11_11_UNORM",
    "10_11_11_SNORM",
    "10_11_11_USCALED",
    "10_11_11_SSCALED",
    "10_11_11_UINT",
    "10_11_11_SINT",
    "10_11_11_FLOAT",
    "11_11_10_UNORM",
    "11_11_10_SNORM",
    "11_11_10_USCALED",
    "11_11_10_SSCALED",
    "11_11_10_UINT",
    "11_11_10_SINT",
    "11_11_10_FLOAT",
    "10_10_10_2_UNORM",
    "10_10_10_2_SNORM",
    "10_10_10_2_USCALED",
    "10_10_10_2_SSCALED",
    "10_10_10_2_UINT",
    "10_10_10_2_SINT",
    "2_10_10_10_UNORM",
    "2_10_10_10_SNORM",
    "2_10_10_10_USCALED",
    "2_10_10_10_SSCALED",
    "2_10_10_10_UINT",
    "2_10_10_10_SINT",
    "8_8_8_8_UNORM",
    "8_8_8_8_SNORM",
    "8_8_8_8_USCALED",
    "8_8_8_8_SSCALED",
    "8_8_8_8_UINT",
    "8_8_8_8_SINT",
    "32_32_UINT",
    "32_32_SINT",
    "32_32_FLOAT",
    "16_16_16_16_UNORM",
    "16_16_16_16_SNORM",
    "16_16_16_16_USCALED",
    "16_16_16_16_SSCALED",
    "16_16_16_16_UINT",
    "16_16_16_16_SINT",
    "16_16_16_16_FLOAT",
    "32_32_32_UINT",
    "32_32_32_SINT",
    "32_32_32_FLOAT",
    "32_32_32_32_UINT",
    "32_32_32_32_SINT",
    "32_32_32_32_FLOAT",
];

fn mimg_has_sampler(name: &str) -> bool {
    name.starts_with("image_sample") || name.starts_with("image_gather") || name == "image_get_lod"
}

/// Number of address dwords an image instruction reads.
pub fn mimg_addr_count(name: &str, dim: u8, a16: bool) -> u8 {
    if name.starts_with("image_bvh64") {
        return if a16 { 9 } else { 12 };
    }
    if name.starts_with("image_bvh") {
        return if a16 { 8 } else { 11 };
    }
    if name == "image_get_resinfo" {
        return 1;
    }
    let coords: u8 = match dim {
        0 => 1,
        1 | 4 => 2,
        2 | 3 | 5 | 6 => 3,
        _ => 4,
    };
    let grads: u8 = match dim {
        0 | 4 => 2,
        2 => 6,
        _ => 4,
    };
    let parts: Vec<&str> = name.trim_start_matches("image_").split('_').collect();
    let mut extra = 0u8;
    let mut lod = 0u8;
    let mut derivs = false;
    let g16 = parts.contains(&"g16");
    for p in &parts[1..] {
        match *p {
            "o" | "b" | "c" => extra += 1,
            "l" | "cl" | "mip" => lod = 1,
            "d" | "cd" => derivs = true,
            _ => {}
        }
    }
    let comps = coords + lod;
    let mut n = extra + if a16 { comps.div_ceil(2) } else { comps };
    if derivs {
        n += if g16 {
            (grads / 2).next_multiple_of(2)
        } else {
            grads
        };
    }
    n
}

/// Size of the VGPR range that holds `n` address dwords outside NSA: LLVM only has
/// register classes for the counts a sampler opcode can need without MSAA, and
/// rounds up to the next one (its `MIMG_Sampler_AddrSizes`).
pub fn mimg_vaddr_regs(name: &str, n: u8) -> Option<u8> {
    let Some((lo, hi)) = sampler_addr_range(name) else {
        return Some(n);
    };
    const RANGES: [(u8, u8); 13] = [
        (1, 1),
        (2, 2),
        (3, 3),
        (3, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (5, 8),
        (9, 9),
        (10, 10),
        (11, 11),
        (12, 12),
        (13, 16),
    ];
    RANGES
        .iter()
        .filter(|r| r.0 <= hi && r.1 >= lo && r.1 >= n)
        .map(|r| r.1)
        .min()
}

// The fewest and most address dwords a sampler opcode needs over all non-MSAA dims.
fn sampler_addr_range(name: &str) -> Option<(u8, u8)> {
    if !mimg_has_sampler(name) {
        return None;
    }
    let parts: Vec<&str> = name.trim_start_matches("image_").split('_').collect();
    let extra = parts
        .iter()
        .filter(|p| matches!(**p, "o" | "b" | "c"))
        .count() as u8;
    let lod = parts.iter().any(|p| matches!(*p, "l" | "cl")) as u8;
    let grads = parts.iter().any(|p| matches!(*p, "d" | "cd"));
    let g16 = parts.contains(&"g16");
    let (lo, hi) = match (grads, g16) {
        (true, true) => (2, 7 + lod),
        (true, false) => (2, 9 + lod),
        _ => (1, 3 + lod),
    };
    Some((lo + extra, hi + extra))
}

fn smem_dwords(name: &str) -> u8 {
    let base = if name.ends_with("x16") {
        16
    } else if name.ends_with("x8") {
        8
    } else if name.ends_with("x4") {
        4
    } else if name.ends_with("x2")
        || name.starts_with("s_memtime")
        || name.starts_with("s_memrealtime")
    {
        2
    } else {
        1
    };
    if name.contains("cmpswap") {
        base * 2
    } else {
        base
    }
}

fn buf_dwords(name: &str) -> u8 {
    let comps: u8 = if name.ends_with("xyzw") {
        4
    } else if name.ends_with("xyz") {
        3
    } else if name.ends_with("xy") {
        2
    } else if name.ends_with("dwordx4") {
        4
    } else if name.ends_with("dwordx3") {
        3
    } else if name.ends_with("dwordx2") || name.ends_with("_x2") {
        2
    } else {
        1
    };
    let comps = if name.contains("cmpswap") {
        comps * 2
    } else {
        comps
    };
    if name.contains("format_d16") && !name.contains("d16_hi") {
        comps.div_ceil(2)
    } else {
        comps
    }
}

struct DsShape {
    ret: u8,
    addr: bool,
    data: u8,
    data1: bool,
    two_offsets: bool,
}

fn ds_shape(name: &str) -> DsShape {
    let size = if name.ends_with("b128") {
        4
    } else if name.ends_with("b96") {
        3
    } else if name.ends_with("64") {
        2
    } else {
        1
    };
    let two = name.contains("read2") || name.contains("write2") || name.contains("xchg2");
    let no_addr = matches!(
        name,
        "ds_nop"
            | "ds_consume"
            | "ds_append"
            | "ds_gws_sema_release_all"
            | "ds_gws_sema_v"
            | "ds_gws_sema_p"
            | "ds_read_addtid_b32"
            | "ds_write_addtid_b32"
            | "ds_gws_init"
            | "ds_gws_sema_br"
            | "ds_gws_barrier"
    );
    let returns = name.contains("read")
        || name == "ds_ordered_count"
        || name.contains("rtn")
        || name.contains("swizzle")
        || name.contains("permute")
        || matches!(name, "ds_consume" | "ds_append");
    let data1 = name.contains("mskor")
        || name.contains("cmpst")
        || name.contains("write2")
        || name.contains("xchg2")
        || name == "ds_wrap_rtn_b32";
    let no_data = name.contains("read")
        || name == "ds_ordered_count"
        || name.contains("swizzle")
        || name.contains("src2")
        || matches!(
            name,
            "ds_nop"
                | "ds_consume"
                | "ds_append"
                | "ds_gws_sema_release_all"
                | "ds_gws_sema_v"
                | "ds_gws_sema_p"
        );
    let data = if no_data {
        0
    } else if name == "ds_condxchg32_rtn_b64" {
        2
    } else {
        size
    };
    let ret = if !returns {
        0
    } else if two && !name.contains("write") {
        size * 2
    } else {
        size
    };
    DsShape {
        ret,
        addr: !no_addr,
        data,
        data1,
        two_offsets: two,
    }
}

fn swizzle(off: u16) -> String {
    if off & 0x8000 != 0 {
        let s: Vec<String> = (0..4).map(|i| format!("{}", off >> (2 * i) & 3)).collect();
        format!("swizzle(QUAD_PERM,{})", s.join(","))
    } else {
        format!("swizzle(BITMASK_PERM,\"{}\")", bitmask_perm(off))
    }
}

fn bitmask_perm(off: u16) -> String {
    let and = off & 0x1F;
    let or = off >> 5 & 0x1F;
    let xor = off >> 10 & 0x1F;
    (0..5)
        .rev()
        .map(|i| match (and >> i & 1, or >> i & 1, xor >> i & 1) {
            (0, 0, _) => '0',
            (0, 1, _) => '1',
            (1, _, 1) => 'i',
            _ => 'p',
        })
        .collect()
}

// (is a load, is a store, data dwords) of a FLAT-family opcode.
fn flat_shape(name: &str) -> (bool, bool, u8) {
    let load = name.contains("load");
    let store = name.contains("store");
    let mut size = if name.ends_with("x4") {
        4
    } else if name.ends_with("x3") {
        3
    } else if name.ends_with("x2") {
        2
    } else {
        1
    };
    if name.contains("cmpswap") {
        size *= 2;
    }
    (load, store, size)
}

fn waitcnt(s: u16) -> String {
    let vm = (s & 0xF) | (s >> 14 & 3) << 4;
    let exp = s >> 4 & 7;
    let lgkm = s >> 8 & 0x3F;
    let mut parts = Vec::new();
    if vm != 63 {
        parts.push(format!("vmcnt({vm})"));
    }
    if exp != 7 {
        parts.push(format!("expcnt({exp})"));
    }
    if lgkm != 63 {
        parts.push(format!("lgkmcnt({lgkm})"));
    }
    if parts.is_empty() {
        return "vmcnt(63) expcnt(7) lgkmcnt(63)".into();
    }
    parts.join(" ")
}

fn version(s: u16) -> String {
    if s & 0x1F00 != 0 {
        return format!("0x{s:x}");
    }
    let mut out = format!("{}", s & 0xFF);
    for (bit, name) in [
        (13, "UC_VERSION_W64_BIT"),
        (14, "UC_VERSION_W32_BIT"),
        (15, "UC_VERSION_MDP_BIT"),
    ] {
        if s >> bit & 1 != 0 {
            out = if out.contains('|') {
                format!("({out})|{name}")
            } else {
                format!("{out}|{name}")
            };
        }
    }
    out
}

// `sendmsg(...)` as LLVM prints it: by name when the message, operation and stream
// are valid together (bits above 9 are then ignored), else as numbers or a plain integer.
fn sendmsg(s: u16) -> String {
    match sendmsg_fields(s) {
        Some(f) => f,
        None if s & !0x3FF != 0 => format!("{s}"),
        None => format!("sendmsg({}, {}, {})", s & 0xF, s >> 4 & 7, s >> 8 & 3),
    }
}

fn sendmsg_fields(s: u16) -> Option<String> {
    let id = s & 0xF;
    let op = s >> 4 & 7;
    let stream = s >> 8 & 3;
    let name = match id {
        1 => "MSG_INTERRUPT",
        2 => "MSG_GS",
        3 => "MSG_GS_DONE",
        4 => "MSG_SAVEWAVE",
        5 => "MSG_STALL_WAVE_GEN",
        6 => "MSG_HALT_WAVES",
        7 => "MSG_ORDERED_PS_DONE",
        8 => "MSG_EARLY_PRIM_DEALLOC",
        9 => "MSG_GS_ALLOC_REQ",
        10 => "MSG_GET_DOORBELL",
        11 => "MSG_GET_DDID",
        15 => "MSG_SYSMSG",
        _ => "",
    };
    match id {
        2 | 3 => {
            let ops = ["GS_OP_NOP", "GS_OP_CUT", "GS_OP_EMIT", "GS_OP_EMIT_CUT"];
            if op > 3 || (id == 2 && op == 0) || (op == 0 && stream != 0) {
                return None;
            }
            Some(if op == 0 {
                format!("sendmsg({name}, {})", ops[0])
            } else {
                format!("sendmsg({name}, {}, {stream})", ops[op as usize])
            })
        }
        15 => {
            let ops = [
                "",
                "SYSMSG_OP_ECC_ERR_INTERRUPT",
                "SYSMSG_OP_REG_RD",
                "",
                "SYSMSG_OP_TTRACE_PC",
            ];
            if !matches!(op, 1 | 2 | 4) || stream != 0 {
                return None;
            }
            Some(format!("sendmsg({name}, {})", ops[op as usize]))
        }
        _ if name.is_empty() || op != 0 || stream != 0 => None,
        _ => Some(format!("sendmsg({name})")),
    }
}

fn depctr(s: u16) -> String {
    if s & 0xE0 != 0 {
        return format!("0x{s:x}");
    }
    let fields = [
        ("sa_sdst", s & 1, 1),
        ("va_vdst", s >> 12 & 15, 15),
        ("va_sdst", s >> 9 & 7, 7),
        ("va_ssrc", s >> 8 & 1, 1),
        ("va_vcc", s >> 1 & 1, 1),
        ("vm_vsrc", s >> 2 & 7, 7),
    ];
    let all = fields.iter().all(|f| f.1 == f.2);
    let parts: Vec<String> = fields
        .iter()
        .filter(|f| all || f.1 != f.2)
        .map(|f| format!("depctr_{}({})", f.0, f.1))
        .collect();
    parts.join(" ")
}

fn hwreg(s: u16) -> String {
    let id = s & 0x3F;
    let off = s >> 6 & 0x1F;
    let size = (s >> 11 & 0x1F) + 1;
    let name = match id {
        1 => "HW_REG_MODE",
        2 => "HW_REG_STATUS",
        3 => "HW_REG_TRAPSTS",
        5 => "HW_REG_GPR_ALLOC",
        6 => "HW_REG_LDS_ALLOC",
        7 => "HW_REG_IB_STS",
        15 => "HW_REG_SH_MEM_BASES",
        16 => "HW_REG_TBA_LO",
        17 => "HW_REG_TBA_HI",
        18 => "HW_REG_TMA_LO",
        19 => "HW_REG_TMA_HI",
        20 => "HW_REG_FLAT_SCR_LO",
        21 => "HW_REG_FLAT_SCR_HI",
        22 => "HW_REG_XNACK_MASK",
        23 => "HW_REG_HW_ID1",
        24 => "HW_REG_HW_ID2",
        25 => "HW_REG_POPS_PACKER",
        _ => "",
    };
    let id = if name.is_empty() {
        format!("{id}")
    } else {
        name.into()
    };
    if off == 0 && size == 32 {
        format!("hwreg({id})")
    } else {
        format!("hwreg({id}, {off}, {size})")
    }
}
