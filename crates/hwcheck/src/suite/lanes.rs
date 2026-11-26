//! Tests of vector instructions that read or write other lanes or registers
//! picked at run time: `v_readlane_b32`, `v_readfirstlane_b32`,
//! `v_writelane_b32`, `v_permlane16_b32`, `v_permlanex16_b32`,
//! `v_swap_b32` and the `v_movrel*` family, each with several lane
//! selects, EXEC masks or M0 values set just before it.

use gfx10::opinfo::{OpInfo, by_name};

use super::BASE_MODE;
use super::valu::{Body, around};
use crate::Case;

const V1: u32 = 257;
const VDST: u32 = 10;
const SDST: u32 = 20;
const S21: u32 = 21;
const S22: u32 = 22;
const S23: u32 = 23;
const M0: u32 = 124;
const EXEC_LO: u32 = 126;
const EXEC_HI: u32 = 127;

/// `s_mov_b32 dst, k`, with `k` inline when it can be.
fn s_mov(dst: u32, k: u32) -> Vec<u32> {
    let w = 0xBE80_0300 | dst << 16;
    if k <= 64 {
        vec![w | (0x80 + k)]
    } else {
        vec![w | 0xFF, k]
    }
}

fn vop3(op: &OpInfo, vdst: u32, opsel: u32, srcs: [u32; 3]) -> Vec<u32> {
    vec![
        0xD400_0000 | (op.code as u32) << 16 | opsel << 11 | vdst,
        srcs[2] << 18 | srcs[1] << 9 | srcs[0],
    ]
}

fn vop1(op: &OpInfo, vdst: u32, src0: u32) -> Vec<u32> {
    vec![0x7E00_0000 | vdst << 17 | (op.code as u32) << 9 | src0]
}

/// `v_mov_b32 vdst, src`.
fn v_mov(vdst: u32, src: u32) -> u32 {
    0x7E00_0200 | vdst << 17 | src
}

fn test(
    name: &str,
    variant: String,
    setup: Vec<u32>,
    inst: Vec<u32>,
    after: Vec<u32>,
) -> Option<Case> {
    let op = by_name(name)?;
    let body = Body {
        setup: &setup,
        inst: &inst,
        after: &after,
    };
    around(op, variant, body, false, BASE_MODE, None)
}

/// Every lane-crossing test.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for lane in [0, 5, 31, 32, 63, 69] {
        let op = by_name("v_readlane_b32").expect("in the opcode table");
        out.extend(test(
            op.name,
            format!("lane {lane}"),
            s_mov(S22, lane),
            vop3(op, SDST, 0, [V1, S22, 0]),
            vec![v_mov(VDST, SDST)],
        ));
        let op = by_name("v_writelane_b32").expect("in the opcode table");
        out.extend(test(
            op.name,
            format!("lane {lane}"),
            [s_mov(S21, 0x1234_5678), s_mov(S22, lane)].concat(),
            vop3(op, VDST, 0, [S21, S22, 0]),
            Vec::new(),
        ));
    }
    let masks: [(&str, u32, u32); 3] = [
        ("all lanes", u32::MAX, u32::MAX),
        ("lanes 4 and up", 0xFFFF_FFF0, u32::MAX),
        ("high half only", 0, 0x8000_0001),
    ];
    for (name, lo, hi) in masks {
        let op = by_name("v_readfirstlane_b32").expect("in the opcode table");
        out.extend(test(
            op.name,
            format!("exec {name}"),
            [s_mov(EXEC_LO, lo), s_mov(EXEC_HI, hi)].concat(),
            vop1(op, SDST, V1),
            vec![0xBEFE_04C1, v_mov(VDST, SDST)],
        ));
    }
    let selects: [(&str, u32, u32); 3] = [
        ("identity", 0x7654_3210, 0xFEDC_BA98),
        ("reversed", 0x89AB_CDEF, 0x0123_4567),
        ("broadcast 5", 0x5555_5555, 0x5555_5555),
    ];
    for name in ["v_permlane16_b32", "v_permlanex16_b32"] {
        let op = by_name(name).expect("in the opcode table");
        for (sel, lo, hi) in selects {
            for (flags, opsel) in [("", 0), (" fi bound_ctrl", 0b11)] {
                out.extend(test(
                    op.name,
                    format!("{sel}{flags}"),
                    [s_mov(S22, lo), s_mov(S23, hi)].concat(),
                    vop3(op, VDST, opsel, [V1, S22, S23]),
                    Vec::new(),
                ));
            }
        }
    }
    let op = by_name("v_swap_b32").expect("in the opcode table");
    out.extend(test(
        op.name,
        "src stored as dst+1".into(),
        Vec::new(),
        vop1(op, VDST, V1),
        vec![v_mov(11, V1)],
    ));
    for (name, m0s) in [
        ("v_movrels_b32", &[0u32, 2, 4][..]),
        ("v_movreld_b32", &[0, 1][..]),
        ("v_movrelsd_b32", &[0, 1][..]),
        ("v_movrelsd_2_b32", &[0, 1][..]),
    ] {
        let Some(op) = by_name(name) else {
            continue;
        };
        for &m0 in m0s {
            out.extend(test(
                op.name,
                format!("m0 {m0}"),
                s_mov(M0, m0),
                vop1(op, VDST, V1),
                Vec::new(),
            ));
        }
        let m0 = m0s[m0s.len() - 1];
        let e32 = vop1(op, VDST, V1)[0] & !0x1FF;
        const QUAD_PERM_1032: u32 = 0xB1 << 8;
        const ALL_ROWS_AND_BANKS: u32 = 0xFF << 24;
        const SDWA_DWORDS: u32 = 6 << 8 | 6 << 16;
        let forms = [
            (
                "e64",
                vec![0xD400_0000 | (op.code as u32 + 0x180) << 16 | VDST, V1],
            ),
            (
                "dpp quad_perm:[1,0,3,2]",
                vec![e32 | 0xFA, 1 | QUAD_PERM_1032 | ALL_ROWS_AND_BANKS],
            ),
            ("sdwa dwords", vec![e32 | 0xF9, 1 | SDWA_DWORDS]),
        ];
        for (form, inst) in forms {
            out.extend(test(
                op.name,
                format!("{form} m0 {m0}"),
                s_mov(M0, m0),
                inst,
                Vec::new(),
            ));
        }
    }
    let op = by_name("v_swaprel_b32").expect("in the opcode table");
    for m0 in [0, 1] {
        out.extend(test(
            op.name,
            format!("m0 {m0}, src stored as dst+1"),
            s_mov(M0, m0),
            vop1(op, VDST, V1),
            vec![v_mov(11, V1)],
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_crossing_tests_decode_to_what_they_test() {
        let cases = cases();
        let text = |op: &str, variant: &str| {
            cases
                .iter()
                .find(|c| c.opcode == op && c.variant == variant)
                .map(|c| c.text.clone())
                .unwrap_or_else(|| panic!("no test {op} ({variant})"))
        };
        assert_eq!(
            text("v_readlane_b32", "lane 5"),
            "v_readlane_b32 s20, v1, s22"
        );
        assert_eq!(
            text("v_writelane_b32", "lane 0"),
            "v_writelane_b32 v10, s21, s22"
        );
        assert_eq!(
            text("v_permlane16_b32", "identity fi bound_ctrl"),
            "v_permlane16_b32 v10, v1, s22, s23 op_sel:[1,1]"
        );
        assert_eq!(text("v_movrels_b32", "m0 2"), "v_movrels_b32_e32 v10, v1");
        assert!(cases.iter().any(|c| c.opcode == "v_readfirstlane_b32"));
        assert!(cases.iter().any(|c| c.opcode == "v_swap_b32"));
        assert!(cases.iter().any(|c| c.opcode == "v_swaprel_b32"));
    }
}
