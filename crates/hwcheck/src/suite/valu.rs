//! Vector ALU tests: every VOP1, VOP2, VOPC, VOP3 and VOP3P opcode whose
//! operands are plain values, in its own encoding and promoted to VOP3,
//! with input and output modifiers, under each float mode. Each lane loads
//! eight words (three operands of up to two words, a carry bit, the
//! destination's starting value), runs the instruction and stores eight
//! (the destination, its lane's bit of the mask written and of EXEC).

use gfx10::decode::{decode, is_vop3b, vop3_op};
use gfx10::opinfo::{Fmt, OpInfo, Ty, all};
use gfx10::print::{Syntax, print_checked};

use super::{BASE_MODE, MODES, Mode, Rng, Test, build, dst, edge, edges, is_value, random};
use crate::Case;

/// Groups of one wave64 per test.
pub const GROUPS: u32 = 32;
const IN_WORDS: usize = 8;
const OUT_NAMES: [&str; 8] = ["dst", "dst+1", "mask", "-", "exec", "-", "-", "-"];

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
const PROLOGUE: [u32; 17] = [
    0x3428_0085, // v_lshlrev_b32_e32 v20, 5, v0
    0x8F08_8B04, // s_lshl_b32 s8, s4, 11
    0x4A28_2808, // v_add_nc_u32_e32 v20, s8, v20
    0xDC38_8000,
    0x0100_0014, // global_load_dwordx4 v[1:4], v20, s[0:1]
    0xDC38_8010,
    0x0500_0014, // global_load_dwordx4 v[5:8], v20, s[0:1] offset:16
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xD4C5_0016,
    0x0002_0E80, // v_cmp_ne_u32_e64 s[22:23], 0, v7
    0xBEEA_0416, // s_mov_b64 vcc, s[22:23]
    0x7E14_0308, // v_mov_b32_e32 v10, v8
    0x7E16_0308, // v_mov_b32_e32 v11, v8
    0xBE94_0480, // s_mov_b64 s[20:21], 0
    0xBF80_0004, // s_nop 4
    0xBF80_0004, // s_nop 4
    0xBF80_0004, // s_nop 4
];
const NOP4: u32 = 0xBF80_0004;
// Only the lane's own mask bit: outside lockstep programs the translator
// keeps masks per thread (docs/shaders.md).
const MASK_FROM_VCC: [u32; 3] = [
    0xD501_000C,
    0x01A9_0280, // v_cndmask_b32_e64 v12, 0, 1, vcc
    0x7E1A_0280, // v_mov_b32_e32 v13, 0
];
const MASK_FROM_SGPRS: [u32; 3] = [
    0xD501_000C,
    0x0051_0280, // v_cndmask_b32_e64 v12, 0, 1, s[20:21]
    0x7E1A_0280, // v_mov_b32_e32 v13, 0
];
const RESTORE_EXEC: [u32; 2] = [
    0xBE98_047E, // s_mov_b64 s[24:25], exec
    0xBEFE_04C1, // s_mov_b64 exec, -1
];
const EPILOGUE: [u32; 10] = [
    0xD501_000E,
    0x0061_0280, // v_cndmask_b32_e64 v14, 0, 1, s[24:25]
    0x7E1E_0280, // v_mov_b32_e32 v15, 0
    0x7E20_0280, // v_mov_b32_e32 v16, 0
    0x7E22_0280, // v_mov_b32_e32 v17, 0
    0xDC78_8000,
    0x0002_0A14, // global_store_dwordx4 v20, v[10:13], s[2:3]
    0xDC78_8010,
    0x0002_0E14, // global_store_dwordx4 v20, v[14:17], s[2:3] offset:16
    0xBF81_0000, // s_endpgm
];
/// VGPRs holding the operands and the destination.
const SRC_VGPRS: [u16; 3] = [1, 3, 5];
const DST_VGPR: u8 = 10;
/// SGPR pairs for a VOP3 lane-mask result and carry-in.
const SDST: u8 = 20;
const CARRY_IN: u16 = 22;
const EXEC_LO: u8 = 126;

/// How an instruction is encoded for a test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Its own 32-bit encoding (VOP1, VOP2, VOPC).
    E32,
    /// VOP3 (or VOP3P), no modifiers.
    E64,
    /// VOP3 with neg on the first and third operands and abs on the second
    /// (VOP3P: neg and neg_hi on the first and third).
    NegAbs,
    /// VOP3 with clamp and output multiplied by two.
    ClampOmod,
    /// VOP3 with the high halves of 16-bit operands and destination
    /// (VOP3P: op_sel swapped halves).
    OpSel,
    /// The 32-bit encoding with its first operand taken from lanes of the
    /// same row (DPP16), one entry of [`DPP`].
    Dpp(usize),
    /// The 32-bit encoding with its first operand from the lanes [`DPP8`]
    /// picks within each group of eight.
    Dpp8,
    /// The 32-bit encoding with sub-dword operand and result selects, one
    /// entry of [`SDWA`].
    Sdwa(usize),
    /// The 32-bit encoding with a literal constant as its first operand.
    Literal,
    /// The 32-bit encoding with an inline constant (by operand code) first.
    Inline(u16),
    /// The 32-bit encoding of an opcode that carries a constant K, with this K.
    K(u32),
}

/// DPP16 controls tested: name, DPP_CTRL, BOUND_CTRL, ROW_MASK, BANK_MASK.
pub const DPP: [(&str, u32, bool, u32, u32); 9] = [
    ("quad_perm:[1,0,3,2]", 0x0B1, false, 0xF, 0xF),
    ("row_shl:1", 0x101, false, 0xF, 0xF),
    ("row_shr:15 bound_ctrl:0", 0x11F, true, 0xF, 0xF),
    ("row_ror:4", 0x124, false, 0xF, 0xF),
    ("row_mirror", 0x140, false, 0xF, 0xF),
    ("row_half_mirror", 0x141, false, 0xF, 0xF),
    ("row_share:3", 0x153, false, 0xF, 0xF),
    ("row_xmask:5", 0x165, false, 0xF, 0xF),
    (
        "row_shl:1 row_mask:0xa bank_mask:0x5",
        0x101,
        false,
        0xA,
        0x5,
    ),
];

/// The DPP8 lane selects tested.
pub const DPP8: [u32; 8] = [7, 6, 5, 4, 3, 2, 1, 0];

/// SDWA selects tested: name, DST_SEL, DST_UNUSED, SRC0_SEL, SRC0_SEXT, SRC1_SEL.
pub const SDWA: [(&str, u32, u32, u32, bool, u32); 4] = [
    ("dword selects", 6, 0, 6, false, 6),
    ("src0_sel:BYTE_1 sext", 6, 0, 1, true, 6),
    (
        "src1_sel:WORD_1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE",
        5,
        2,
        6,
        false,
        5,
    ),
    ("dst_sel:BYTE_0 dst_unused:UNUSED_SEXT", 0, 1, 6, false, 6),
];

/// Literal constants tested, by the first operand's type.
fn literal(t: Ty) -> u32 {
    match t {
        Ty::F64 => 0x3FF8_0000,
        Ty::F16 | Ty::PkF16 => 0x3E00,
        _ => 0x3FC0_0000,
    }
}

impl Form {
    fn name(self) -> String {
        match self {
            Form::E32 => "e32".into(),
            Form::E64 => "e64".into(),
            Form::NegAbs => "e64 neg abs".into(),
            Form::ClampOmod => "e64 clamp omod".into(),
            Form::OpSel => "e64 op_sel".into(),
            Form::Dpp(i) => format!("dpp {}", DPP[i].0),
            Form::Dpp8 => "dpp8".into(),
            Form::Sdwa(i) => format!("sdwa {}", SDWA[i].0),
            Form::Literal => "e32 literal".into(),
            Form::Inline(c) => format!("e32 inline constant {c:#x}"),
            Form::K(k) => format!("e32 K {k:#010x}"),
        }
    }

    /// Whether the form is the 32-bit encoding (lane masks go to VCC).
    fn short(self) -> bool {
        !matches!(
            self,
            Form::E64 | Form::NegAbs | Form::ClampOmod | Form::OpSel
        )
    }
}

/// `op` in the 32-bit encoding with its first operand replaced by one of
/// the extended forms: the operand code and the dword that follows.
fn extended(op: &OpInfo, form: Form) -> Option<Vec<u32>> {
    if let Form::K(k) = form {
        if op.srcs.last() != Some(&Ty::Imm) || op.fmt != Fmt::Vop2 {
            return None;
        }
        let w = (op.code as u32) << 25
            | (DST_VGPR as u32) << 17
            | (SRC_VGPRS[1] as u32) << 9
            | vgpr(SRC_VGPRS[0]) as u32;
        return Some(vec![w, k]);
    }
    let base = encode(op, Form::E32)?[0] & !0x1FF;
    let src0 = vgpr(SRC_VGPRS[0]) as u32 - 256;
    let (code, extra) = match form {
        Form::Dpp(i) => {
            if !matches!(op.fmt, Fmt::Vop1 | Fmt::Vop2) {
                return None;
            }
            let (_, ctrl, bc, rows, banks) = DPP[i];
            (
                0xFA,
                src0 | ctrl << 8 | (bc as u32) << 19 | banks << 24 | rows << 28,
            )
        }
        Form::Dpp8 => {
            if !matches!(op.fmt, Fmt::Vop1 | Fmt::Vop2) {
                return None;
            }
            let sel = DPP8
                .iter()
                .enumerate()
                .fold(0, |w, (i, &l)| w | l << (8 + 3 * i));
            (0xE9, src0 | sel)
        }
        Form::Sdwa(i) => {
            let (_, dst_sel, dst_unused, src0_sel, sext, src1_sel) = SDWA[i];
            let dst = if op.fmt == Fmt::Vopc {
                0
            } else {
                dst_sel << 8 | dst_unused << 11
            };
            let src1_sel = if op.fmt == Fmt::Vop1 { 0 } else { src1_sel };
            (
                0xF9,
                src0 | dst | src0_sel << 16 | (sext as u32) << 19 | src1_sel << 24,
            )
        }
        Form::Literal => (0xFF, literal(op.srcs[0])),
        Form::Inline(c) => return Some(vec![base | c as u32]),
        _ => return None,
    };
    Some(vec![base | code, extra])
}

/// Why an opcode has no generated test, or `None` if it has.
pub fn unsupported(op: &OpInfo) -> Option<&'static str> {
    if !matches!(
        op.fmt,
        Fmt::Vop1 | Fmt::Vop2 | Fmt::Vopc | Fmt::Vop3 | Fmt::Vop3p
    ) {
        return Some("not a vector ALU instruction");
    }
    const LANES: [&str; 12] = [
        "readlane",
        "writelane",
        "readfirstlane",
        "permlane",
        "movrel",
        "swaprel",
        "v_swap_b32",
        "v_nop",
        "v_pipeflush",
        "v_clrexcp",
        "v_interp",
        "v_illegal",
    ];
    if LANES.iter().any(|n| op.name.contains(n)) {
        return Some("lane-crossing or no-value instruction, tested by hand");
    }
    if op
        .srcs
        .iter()
        .any(|&t| matches!(t, Ty::M0 | Ty::Scc | Ty::Exec))
    {
        return Some("takes a special register");
    }
    if op.defs.is_empty() || op.defs.iter().filter(|&&t| is_value(t)).count() > 1 {
        return Some("no single value result");
    }
    if op.srcs.iter().take(3).filter(|&&t| is_value(t)).count() == 0 {
        return Some("no value operand");
    }
    None
}

fn vgpr(n: u16) -> u16 {
    256 + n
}

/// The instruction words of `op` in `form`, or `None` if `op` has no such form.
fn encode(op: &OpInfo, form: Form) -> Option<Vec<u32>> {
    if !matches!(
        form,
        Form::E32 | Form::E64 | Form::NegAbs | Form::ClampOmod | Form::OpSel
    ) {
        return extended(op, form);
    }
    let value_srcs: Vec<usize> = (0..op.srcs.len().min(3))
        .filter(|&i| is_value(op.srcs[i]))
        .collect();
    let src = |i: usize| -> u16 {
        match op.srcs.get(i) {
            Some(&t) if is_value(t) => vgpr(SRC_VGPRS[i]),
            Some(Ty::Lane) => CARRY_IN,
            _ => 0,
        }
    };
    let floats: u8 = (0..op.srcs.len().min(3))
        .filter(|&i| op.srcs[i].is_float())
        .fold(0, |m, i| m | 1 << i);
    let sixteen = op
        .srcs
        .iter()
        .chain(op.defs)
        .any(|&t| t.bits() == 16 && is_value(t));
    match form {
        Form::E32 => {
            let w = match op.fmt {
                Fmt::Vop1 => {
                    0x7E00_0000 | (DST_VGPR as u32) << 17 | (op.code as u32) << 9 | src(0) as u32
                }
                Fmt::Vop2 => {
                    if op.srcs.len() > 1 && !is_value(op.srcs[1]) {
                        return None;
                    }
                    (op.code as u32) << 25
                        | (DST_VGPR as u32) << 17
                        | (SRC_VGPRS[1] as u32) << 9
                        | src(0) as u32
                }
                Fmt::Vopc => {
                    0x7C00_0000
                        | (op.code as u32) << 17
                        | (SRC_VGPRS[1] as u32) << 9
                        | src(0) as u32
                }
                _ => return None,
            };
            Some(vec![w])
        }
        Form::E64 | Form::NegAbs | Form::ClampOmod | Form::OpSel if op.fmt == Fmt::Vop3p => {
            let n = value_srcs.len() as u32;
            let all = (1u32 << n) - 1;
            let (neg, neg_hi, opsel, opsel_hi, clamp) = match form {
                Form::E64 => (0, 0, 0, all, 0),
                Form::NegAbs => (0b101 & all, 0b101 & all, 0, all, 0),
                Form::ClampOmod => (0, 0, 0, all, 1),
                Form::OpSel => (0, 0, all, 0, 0),
                _ => unreachable!("short forms are encoded by `extended`"),
            };
            let w0 = 0xCC00_0000
                | (op.code as u32) << 16
                | clamp << 15
                | (opsel_hi >> 2 & 1) << 14
                | opsel << 11
                | neg_hi << 8
                | DST_VGPR as u32;
            let w1 = neg << 29
                | (opsel_hi & 3) << 27
                | (src(2) as u32) << 18
                | (src(1) as u32) << 9
                | src(0) as u32;
            Some(vec![w0, w1])
        }
        _ => {
            let code = match op.fmt {
                Fmt::Vopc => op.code,
                Fmt::Vop2 => op.code + 0x100,
                Fmt::Vop1 => op.code + 0x180,
                _ => op.code,
            };
            if vop3_op(code).map(|o| o.name) != Some(op.name) {
                return None;
            }
            let (mut abs, mut neg, mut clamp, mut omod, mut opsel) = (0u32, 0u32, 0u32, 0u32, 0u32);
            match form {
                Form::NegAbs => {
                    if floats == 0 {
                        return None;
                    }
                    neg = (floats & 0b101) as u32;
                    abs = (floats & 0b010) as u32;
                }
                Form::ClampOmod => {
                    clamp = 1;
                    if op
                        .defs
                        .first()
                        .is_some_and(|t| matches!(t, Ty::F32 | Ty::F64))
                    {
                        omod = 1;
                    }
                }
                Form::OpSel => {
                    if !sixteen {
                        return None;
                    }
                    opsel = 0b1111;
                }
                _ => {}
            }
            let vdst = match op.defs.first() {
                Some(Ty::Lane) => SDST,
                Some(Ty::Exec) => EXEC_LO,
                _ => DST_VGPR,
            };
            let w0 = if is_vop3b(op) {
                0xD400_0000 | (code as u32) << 16 | clamp << 15 | (SDST as u32) << 8 | vdst as u32
            } else {
                0xD400_0000
                    | (code as u32) << 16
                    | clamp << 15
                    | opsel << 11
                    | abs << 8
                    | vdst as u32
            };
            let tied = op.fmt == Fmt::Vop2 && op.name.contains("mac");
            let src2 = if tied { 0 } else { src(2) as u32 };
            let w1 = neg << 29 | omod << 27 | src2 << 18 | (src(1) as u32) << 9 | src(0) as u32;
            Some(vec![w0, w1])
        }
    }
}

/// Every lane's input words for `op`: pairs of edge cases of the first two
/// operands, then random mixes of edge cases and random values.
fn operands(op: &OpInfo, seed: &str, lanes: usize) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let srcs: Vec<Ty> = op.srcs.iter().take(3).copied().collect();
    let dst = op
        .defs
        .first()
        .copied()
        .filter(|&t| is_value(t))
        .unwrap_or(Ty::U32);
    let n0 = srcs.first().map_or(1, |&t| edges(t));
    let n1 = srcs.get(1).map_or(1, |&t| edges(t));
    let mut out = Vec::with_capacity(lanes * IN_WORDS);
    for lane in 0..lanes {
        let mut words = [0u32; IN_WORDS];
        for (i, &t) in srcs.iter().enumerate() {
            if !is_value(t) {
                continue;
            }
            let garbage = rng.next();
            let v = if lane < n0 * n1 {
                let k = match i {
                    0 => lane % n0,
                    1 => lane / n0 % n1,
                    _ => (lane * 7 + 3) % edges(t),
                };
                edge(t, k, garbage)
            } else if rng.next() & 3 == 0 {
                edge(t, rng.below(edges(t)), garbage)
            } else {
                random(t, &mut rng)
            };
            words[2 * i] = v as u32;
            words[2 * i + 1] = (v >> 32) as u32;
        }
        words[6] = rng.next() as u32 & 1;
        let fill = random(dst, &mut rng);
        words[7] = if rng.next() & 3 == 0 {
            edge(dst, rng.below(edges(dst)), rng.next()) as u32
        } else {
            fill as u32
        };
        out.extend_from_slice(&words);
    }
    out
}

/// The test of `op` in `form` under `mode`, or `None` if `op` has no such form.
pub fn case(op: &OpInfo, form: Form, mode: Mode) -> Option<Case> {
    let inst = encode(op, form)?;
    let mask_in_sgprs = op.defs.contains(&Ty::Lane) && !form.short();
    let variant = if mode == BASE_MODE {
        form.name()
    } else {
        format!("{}, {}", form.name(), mode.name)
    };
    let body = Body {
        setup: &[],
        inst: &inst,
        after: &[],
    };
    around(op, variant, body, mask_in_sgprs, mode, None)
}

/// The instruction under test with what runs just before and just after it.
pub(super) struct Body<'a> {
    pub setup: &'a [u32],
    pub inst: &'a [u32],
    pub after: &'a [u32],
}

/// The per-lane test program around `body.inst`, a form of `op`; the mask
/// stored is VCC or, with `mask_in_sgprs`, `s[20:21]`. `inputs` replaces
/// the operands generated from `op`'s types.
pub(super) fn around(
    op: &OpInfo,
    variant: String,
    body: Body,
    mask_in_sgprs: bool,
    mode: Mode,
    inputs: Option<Vec<u32>>,
) -> Option<Case> {
    let Body { setup, inst, after } = body;
    let decoded = decode(inst).ok()?;
    if decoded.op.name != op.name {
        return None;
    }
    let text = print_checked(&decoded, Syntax { wave64: true })?;
    let mut code: Vec<u32> = PROLOGUE.to_vec();
    code.extend(setup);
    code.extend(inst);
    code.extend([NOP4, NOP4]);
    code.extend(after);
    code.extend(RESTORE_EXEC);
    code.extend(if mask_in_sgprs {
        MASK_FROM_SGPRS
    } else {
        MASK_FROM_VCC
    });
    code.extend(EPILOGUE);
    let seed = format!("{} {variant}", op.name);
    Some(build(Test {
        opcode: op.name,
        variant,
        text,
        code,
        mode,
        groups: GROUPS,
        per_wave: false,
        in_words: IN_WORDS as u32,
        out_names: &OUT_NAMES,
        dst: dst(op.defs.first()),
        lds_bytes: 0,
        inputs: inputs.unwrap_or_else(|| operands(op, &seed, (GROUPS * 64) as usize)),
    }))
}

/// Every vector ALU test, in opcode table order.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for op in all() {
        if unsupported(op).is_some() {
            continue;
        }
        let float = op.srcs.iter().chain(op.defs).any(|t| t.is_float());
        for form in [
            Form::E32,
            Form::E64,
            Form::NegAbs,
            Form::ClampOmod,
            Form::OpSel,
        ] {
            out.extend(case(op, form, BASE_MODE));
        }
        if float {
            for mode in MODES {
                out.extend(case(op, Form::E64, mode).or_else(|| case(op, Form::E32, mode)));
            }
        }
        let extended = (0..DPP.len())
            .map(Form::Dpp)
            .chain([
                Form::Dpp8,
                Form::Literal,
                Form::Inline(0xF8),
                Form::Inline(0xF3),
            ])
            .chain((0..SDWA.len()).map(Form::Sdwa))
            .chain([Form::K(0x3FC0_0000), Form::K(0x8000_0001)]);
        for form in extended {
            out.extend(case(op, form, BASE_MODE));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::opinfo::by_name;

    #[test]
    fn encodings_decode_to_the_opcode_they_test() {
        let add = by_name("v_add_f32").unwrap();
        assert_eq!(
            case(add, Form::E32, BASE_MODE).unwrap().text,
            "v_add_f32_e32 v10, v1, v3"
        );
        assert_eq!(
            case(add, Form::NegAbs, BASE_MODE).unwrap().text,
            "v_add_f32_e64 v10, -v1, |v3|"
        );
        let cmp = by_name("v_cmp_lt_f32").unwrap();
        assert_eq!(
            case(cmp, Form::E64, BASE_MODE).unwrap().text,
            "v_cmp_lt_f32_e64 s[20:21], v1, v3"
        );
        let carry = by_name("v_add_co_ci_u32").unwrap();
        assert_eq!(
            case(carry, Form::E64, BASE_MODE).unwrap().text,
            "v_add_co_ci_u32_e64 v10, s[20:21], v1, v3, s[22:23]"
        );
    }

    #[test]
    fn the_suite_covers_most_vector_alu_opcodes() {
        let cases = cases();
        let ops: std::collections::BTreeSet<&str> =
            cases.iter().map(|c| c.opcode.as_str()).collect();
        assert!(ops.len() > 350, "{} opcodes", ops.len());
        let lanes = cases[0].record_inputs();
        assert_eq!(lanes.len(), (GROUPS * 64) as usize * IN_WORDS);
    }
}
