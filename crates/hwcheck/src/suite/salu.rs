//! Scalar ALU tests: every SOP1, SOP2, SOPC and SOPK opcode that computes
//! a value from plain operands. Each wave loads eight words (two operands
//! of up to two words, two unused, an SCC bit, the destination's starting
//! value), runs the instruction and stores the destination and SCC.

use gfx10::decode::decode;
use gfx10::opinfo::{Fmt, OpInfo, Ty, all};
use gfx10::print::{Syntax, print_checked};

use super::{BASE_MODE, Rng, Test, build, dst, edge, edges, is_value, random};
use crate::Case;

/// Groups of one wave64 per test, one record each.
const GROUPS: u32 = 1024;
const IN_WORDS: usize = 8;
const OUT_NAMES: [&str; 8] = ["dst", "dst+1", "scc", "-", "-", "-", "-", "-"];

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
const PROLOGUE: [u32; 7] = [
    0x8F08_8504, // s_lshl_b32 s8, s4, 5
    0xF40C_0600,
    0x1000_0000, // s_load_dwordx8 s[24:31], s[0:1], s8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF07_801E, // s_cmp_lg_u32 s30, 0
    0xBE8A_031F, // s_mov_b32 s10, s31
    0xBE8B_031F, // s_mov_b32 s11, s31
];
const EPILOGUE: [u32; 15] = [
    0x850C_8081, // s_cselect_b32 s12, 1, 0
    0xBEFE_0481, // s_mov_b64 exec, 1
    0x7E02_020A, // v_mov_b32_e32 v1, s10
    0x7E04_020B, // v_mov_b32_e32 v2, s11
    0x7E06_020C, // v_mov_b32_e32 v3, s12
    0x7E08_0280, // v_mov_b32_e32 v4, 0
    0x7E0A_0280, // v_mov_b32_e32 v5, 0
    0x7E0C_0280, // v_mov_b32_e32 v6, 0
    0x7E0E_0280, // v_mov_b32_e32 v7, 0
    0x7E12_0208, // v_mov_b32_e32 v9, s8
    0xDC78_8000,
    0x0002_0109, // global_store_dwordx4 v9, v[1:4], s[2:3]
    0xDC78_8010,
    0x0002_0409, // global_store_dwordx4 v9, v[4:7], s[2:3] offset:16
    0xBF81_0000, // s_endpgm
];

const SDST: u32 = 10;
const SRC0: u32 = 24;
const SRC1: u32 = 26;

/// Immediates SOPK tests run with, each its own test.
const SIMM16: [u16; 6] = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x1234];

/// Why an opcode has no generated scalar ALU test, or `None` if it has.
pub fn unsupported(op: &OpInfo) -> Option<&'static str> {
    if !matches!(op.fmt, Fmt::Sop1 | Fmt::Sop2 | Fmt::Sopc | Fmt::Sopk) {
        return Some("not a scalar ALU instruction");
    }
    const OTHER: [&str; 12] = [
        "pc",
        "call",
        "rfe",
        "movrel",
        "getreg",
        "setreg",
        "version",
        "subvector",
        "waitcnt",
        "cbranch",
        "wrexec",
        "saveexec",
    ];
    if OTHER.iter().any(|n| op.name.contains(n)) {
        return Some("control flow, hardware registers or EXEC, tested by hand");
    }
    if op
        .srcs
        .iter()
        .chain(op.defs)
        .any(|t| matches!(t, Ty::Exec | Ty::Lane | Ty::M0))
    {
        return Some("reads or writes EXEC, a lane mask or M0");
    }
    if op.defs.iter().all(|t| !is_value(*t) && *t != Ty::Scc) {
        return Some("no result");
    }
    None
}

fn encode(op: &OpInfo, simm16: u16) -> u32 {
    let c = op.code as u32;
    match op.fmt {
        Fmt::Sop2 => 0x8000_0000 | c << 23 | SDST << 16 | SRC1 << 8 | SRC0,
        Fmt::Sop1 => 0xBE80_0000 | SDST << 16 | c << 8 | SRC0,
        Fmt::Sopc => 0xBF00_0000 | c << 16 | SRC1 << 8 | SRC0,
        _ => 0xB000_0000 | c << 23 | SDST << 16 | simm16 as u32,
    }
}

/// Every wave's input words for `op`: pairs of edge cases of the two
/// operands, then random mixes.
fn operands(op: &OpInfo, seed: &str) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let srcs: Vec<Ty> = op
        .srcs
        .iter()
        .copied()
        .filter(|&t| is_value(t))
        .take(2)
        .collect();
    let dst_ty = op
        .defs
        .first()
        .copied()
        .filter(|&t| is_value(t))
        .unwrap_or(Ty::U32);
    let n0 = srcs.first().map_or(1, |&t| edges(t));
    let n1 = srcs.get(1).map_or(1, |&t| edges(t));
    let mut out = Vec::with_capacity(GROUPS as usize * IN_WORDS);
    for wave in 0..GROUPS as usize {
        let mut w = [0u32; IN_WORDS];
        for (i, &t) in srcs.iter().enumerate() {
            let v = if wave < n0 * n1 {
                edge(
                    t,
                    if i == 0 { wave % n0 } else { wave / n0 % n1 },
                    rng.next(),
                )
            } else if rng.next() & 3 == 0 {
                edge(t, rng.below(edges(t)), rng.next())
            } else {
                random(t, &mut rng)
            };
            w[2 * i] = v as u32;
            w[2 * i + 1] = (v >> 32) as u32;
        }
        w[6] = rng.next() as u32 & 1;
        w[7] = random(dst_ty, &mut rng) as u32;
        out.extend_from_slice(&w);
    }
    out
}

fn case(op: &OpInfo, simm16: u16) -> Option<Case> {
    let inst = encode(op, simm16);
    let decoded = decode(&[inst, 0]).ok()?;
    if decoded.op.name != op.name {
        return None;
    }
    let text = print_checked(&decoded, Syntax { wave64: true })?;
    let mut code = PROLOGUE.to_vec();
    code.push(inst);
    code.extend(EPILOGUE);
    let variant = if op.fmt == Fmt::Sopk {
        format!("simm16 {simm16:#x}")
    } else {
        "base".to_string()
    };
    let seed = format!("{} {variant}", op.name);
    Some(build(Test {
        opcode: op.name,
        variant,
        text,
        code,
        mode: BASE_MODE,
        groups: GROUPS,
        per_wave: true,
        in_words: IN_WORDS as u32,
        out_names: &OUT_NAMES,
        dst: dst(None),
        lds_bytes: 0,
        inputs: operands(op, &seed),
    }))
}

/// Every scalar ALU test, in opcode table order.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for op in all() {
        if unsupported(op).is_some() {
            continue;
        }
        if op.fmt == Fmt::Sopk {
            out.extend(SIMM16.iter().filter_map(|&k| case(op, k)));
        } else {
            out.extend(case(op, 0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::opinfo::by_name;

    #[test]
    fn scalar_encodings_decode_to_the_opcode_they_test() {
        let text = |n: &str, k| case(by_name(n).unwrap(), k).unwrap().text;
        assert_eq!(text("s_add_u32", 0), "s_add_u32 s10, s24, s26");
        assert_eq!(text("s_lshl_b64", 0), "s_lshl_b64 s[10:11], s[24:25], s26");
        assert_eq!(text("s_bcnt1_i32_b32", 0), "s_bcnt1_i32_b32 s10, s24");
        assert_eq!(text("s_cmp_lt_i32", 0), "s_cmp_lt_i32 s24, s26");
        assert_eq!(text("s_addk_i32", 0x8000), "s_addk_i32 s10, 0x8000");
    }

    #[test]
    fn the_scalar_suite_covers_most_scalar_alu_opcodes() {
        let ops: std::collections::BTreeSet<String> =
            cases().into_iter().map(|c| c.opcode).collect();
        assert!(
            ops.len() > 110,
            "only {} scalar opcodes would be tested",
            ops.len()
        );
    }
}
