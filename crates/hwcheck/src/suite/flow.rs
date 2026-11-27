//! Tests of program flow and machine state, in the vector ALU test's
//! per-lane program: every conditional and unconditional branch (taken and
//! not, from conditions that are the same across a wave and from ones that
//! differ per lane), the `s_*_saveexec` family, `s_getpc_b64`, calls and
//! jumps, `s_setreg` on the float mode, and instructions that must leave the
//! results alone (waits, nops, priorities, prefetch and cache invalidates).

use gfx10::opinfo::{Fmt, OpInfo, all, by_name};

use super::valu::{Body, GROUPS, around};
use super::{BASE_MODE, Rng};
use crate::coverage::instance;
use crate::{COMPUTE_DISPATCH_INITIATOR, COMPUTE_FIRST, CS_W32_EN, Case, write_pages};

const IN_WORDS: usize = 8;
const V1: u32 = 257;
const V3: u32 = 259;
const VCC: u32 = 106;
const EXEC: u32 = 126;

fn op(name: &str) -> &'static OpInfo {
    by_name(name).expect("in the opcode table")
}

fn sopp(name: &str, simm16: u16) -> u32 {
    0xBF80_0000 | (op(name).code as u32) << 16 | simm16 as u32
}

fn sop1(name: &str, sdst: u32, ssrc0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | (op(name).code as u32) << 8 | ssrc0
}

fn sopc(name: &str, ssrc0: u32, ssrc1: u32) -> u32 {
    0xBF00_0000 | (op(name).code as u32) << 16 | ssrc1 << 8 | ssrc0
}

fn sop2(name: &str, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0x8000_0000 | (op(name).code as u32) << 23 | sdst << 16 | ssrc1 << 8 | ssrc0
}

/// `v_cmp_ne_u32_e64 sdst, 0, vsrc`: a lane mask from a lane's operand.
fn mask_from(sdst: u32, vsrc: u32) -> [u32; 2] {
    let code = op("v_cmp_ne_u32").code as u32;
    [0xD400_0000 | code << 16 | sdst, vsrc << 9 | 0x80]
}

/// `v_mov_b32 vdst, src`.
fn v_mov(vdst: u32, src: u32) -> u32 {
    0x7E00_0200 | vdst << 17 | src
}

/// `v_add_f32_e32 v10, v1, v3`: a result that shows whether anything changed.
fn v_add() -> u32 {
    (op("v_add_f32").code as u32) << 25 | 10 << 17 | 3 << 9 | V1
}

const S4: u32 = 4;
const INLINE_0: u32 = 0x80;
const INLINE_1: u32 = 0x81;
const INLINE_MINUS_1: u32 = 0xC1;

/// Random lane operands: bit 0 of the first and third words choose masks,
/// the words are also floats for `v_add_f32`.
fn inputs(seed: &str) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..GROUPS as usize * 64 * IN_WORDS)
        .map(|i| match i % IN_WORDS {
            0 | 2 => {
                let f = (rng.next() % 4001) as f32 / 1000.0 - 2.0;
                f.to_bits() & !1 | (rng.next() as u32 & 1)
            }
            _ => rng.next() as u32,
        })
        .collect()
}

fn test(
    name: &str,
    variant: &str,
    setup: Vec<u32>,
    inst: Vec<u32>,
    after: Vec<u32>,
) -> Option<Case> {
    let seed = format!("{name} {variant}");
    let body = Body {
        setup: &setup,
        inst: &inst,
        after: &after,
    };
    around(
        op(name),
        variant.into(),
        body,
        false,
        BASE_MODE,
        Some(inputs(&seed)),
    )
}

// From the group number a whole wave agrees on the condition; from a lane
// operand the lanes disagree.
fn conditions() -> Vec<(&'static str, Vec<u32>)> {
    let group_bit = |bit: u32| sopc("s_bitcmp1_b32", S4, INLINE_0 + bit);
    vec![
        ("scc from the group's bit 0", vec![group_bit(0)]),
        (
            "vcc from the group's bit 1",
            vec![
                group_bit(1),
                sop2("s_cselect_b64", VCC, INLINE_MINUS_1, INLINE_0),
            ],
        ),
        ("vcc from each lane", mask_from(VCC, V1).to_vec()),
        (
            "exec from the group's bit 0",
            vec![
                group_bit(0),
                sop2("s_cselect_b64", EXEC, INLINE_MINUS_1, INLINE_0),
            ],
        ),
        ("exec from each lane", mask_from(EXEC, V1).to_vec()),
        (
            "exec from each lane, then s_nop 4",
            [mask_from(EXEC, V1).to_vec(), vec![sopp("s_nop", 4)]].concat(),
        ),
    ]
}

fn branches() -> Vec<Case> {
    let mut out = Vec::new();
    let names: Vec<&str> = all()
        .iter()
        .filter(|o| o.fmt == Fmt::Sopp && (o.name.starts_with("s_cbranch") || o.name == "s_branch"))
        .map(|o| o.name)
        .collect();
    for name in names {
        for (cond, setup) in conditions() {
            out.extend(test(
                name,
                cond,
                setup,
                vec![sopp(name, 3)],
                vec![v_mov(10, INLINE_1)],
            ));
        }
    }
    out
}

fn saveexec() -> Vec<Case> {
    let mut out = Vec::new();
    for o in all().iter().filter(|o| o.name.contains("_saveexec_")) {
        let wide = o.name.ends_with("b64");
        let setup = [mask_from(24, V1), mask_from(26, V3)].concat();
        let set_exec = sop1(if wide { "s_mov_b64" } else { "s_mov_b32" }, EXEC, 26);
        let inst = vec![0xBE80_0000 | 20 << 16 | (o.code as u32) << 8 | 24];
        let seed = format!("{} saveexec", o.name);
        let setup = [setup, vec![set_exec]].concat();
        let body = || Body {
            setup: &setup,
            inst: &inst,
            after: &[],
        };
        out.extend(around(
            o,
            "lane masks".into(),
            body(),
            true,
            BASE_MODE,
            Some(inputs(&seed)),
        ));
        if !wide {
            let seed = format!("{} saveexec wave32", o.name);
            let mut case = around(
                o,
                "lane masks, wave32".into(),
                body(),
                true,
                BASE_MODE,
                Some(inputs(&seed)),
            );
            if let Some(c) = &mut case {
                c.compute[COMPUTE_DISPATCH_INITIATOR - COMPUTE_FIRST] = CS_W32_EN;
            }
            out.extend(case);
        }
    }
    out
}

fn machine_state() -> Vec<Case> {
    let mut out = Vec::new();
    out.extend(test(
        "s_getpc_b64",
        "program counter",
        Vec::new(),
        vec![sop1("s_getpc_b64", 20, 0)],
        vec![v_mov(10, 20), v_mov(11, 21)],
    ));
    const HW_REG_MODE: u16 = 1;
    let hwreg = |size: u16| HW_REG_MODE | (size - 1) << 11;
    for (name, mode) in [
        ("round toward +inf", 0x5u32),
        ("round toward -inf", 0xA),
        ("round toward zero", 0xF),
        ("all denormals kept", 0xF0),
    ] {
        let setreg = op("s_setreg_imm32_b32").code as u32;
        out.extend(test(
            "s_setreg_imm32_b32",
            name,
            Vec::new(),
            vec![0xB000_0000 | setreg << 23 | hwreg(8) as u32, mode],
            vec![v_add()],
        ));
        let setreg = op("s_setreg_b32").code as u32;
        out.extend(test(
            "s_setreg_b32",
            name,
            vec![sop1("s_mov_b32", 22, 0xFF), mode],
            vec![0xB000_0000 | setreg << 23 | 22 << 16 | hwreg(8) as u32],
            vec![v_add()],
        ));
    }
    out.extend(hardware_registers());
    out
}

/// `s_getreg_b32` of the fields of hardware registers that do not depend on
/// where the wave runs or on how the machine is set up, into s20 (stored in
/// v10), also after writes by `s_setreg`, `s_round_mode` and
/// `s_denorm_mode`, and TRAPSTS around `v_clrexcp`. Writes leave out MODE's
/// DEBUG and exception enables, which could make the wave trap.
fn hardware_registers() -> Vec<Case> {
    const S_CMP_EQ_U32_0_0: u32 = 0xBF06_8080;
    let getreg = |id: u32, offset: u32, size: u32| {
        0xB000_0000
            | (op("s_getreg_b32").code as u32) << 23
            | 20 << 16
            | (size - 1) << 11
            | offset << 6
            | id
    };
    let setreg_imm = |id: u32, value: u32| {
        vec![
            0xB000_0000 | (op("s_setreg_imm32_b32").code as u32) << 23 | 31 << 11 | id,
            value,
        ]
    };
    let s_mov_b64 = |sdst: u32, ssrc0: u32| sop1("s_mov_b64", sdst, ssrc0);
    let store = vec![v_mov(10, 20)];
    let mut out = Vec::new();
    let reads = [
        ("MODE", Vec::new(), (1, 0, 32)),
        ("MODE bits 4 to 7", Vec::new(), (1, 4, 4)),
        (
            "STATUS bits 0 to 11, SCC set",
            vec![S_CMP_EQ_U32_0_0, s_mov_b64(VCC, INLINE_MINUS_1)],
            (2, 0, 12),
        ),
        (
            "STATUS bits 0 to 11, VCC zero",
            vec![s_mov_b64(VCC, INLINE_0)],
            (2, 0, 12),
        ),
        ("STATUS VALID", Vec::new(), (2, 16, 1)),
        ("TRAPSTS exceptions", Vec::new(), (3, 0, 9)),
        ("GPR_ALLOC VGPR_SIZE", Vec::new(), (5, 8, 6)),
        ("GPR_ALLOC SGPR_SIZE", Vec::new(), (5, 24, 4)),
        ("LDS_ALLOC LDS_SIZE", Vec::new(), (6, 12, 9)),
        ("IB_STS", Vec::new(), (7, 0, 32)),
        ("TBA_LO", Vec::new(), (16, 0, 32)),
        ("TMA_LO", Vec::new(), (18, 0, 32)),
        (
            "MODE after s_setreg_imm32_b32 0x80055a",
            setreg_imm(1, 0x0080_055A),
            (1, 0, 32),
        ),
    ];
    for (variant, setup, (id, offset, size)) in reads {
        out.extend(test(
            "s_getreg_b32",
            variant,
            setup,
            vec![getreg(id, offset, size)],
            store.clone(),
        ));
    }
    out.extend(test(
        "s_getreg_b32",
        "STATUS bits 0 to 11, EXEC zero",
        vec![s_mov_b64(EXEC, INLINE_0)],
        vec![getreg(2, 0, 12)],
        [vec![s_mov_b64(EXEC, INLINE_MINUS_1)], store.clone()].concat(),
    ));
    out.extend(test(
        "s_round_mode",
        "9, then s_getreg_b32 MODE",
        Vec::new(),
        vec![sopp("s_round_mode", 9)],
        [vec![getreg(1, 0, 32)], store.clone()].concat(),
    ));
    out.extend(test(
        "s_denorm_mode",
        "6, then s_getreg_b32 MODE",
        Vec::new(),
        vec![sopp("s_denorm_mode", 6)],
        [vec![getreg(1, 0, 32)], store.clone()].concat(),
    ));
    out.extend(test(
        "v_clrexcp",
        "after s_setreg_imm32_b32 TRAPSTS 0x1ff, then s_getreg_b32 TRAPSTS",
        setreg_imm(3, 0x1FF),
        vec![0x7E00_0000 | (op("v_clrexcp").code as u32) << 9],
        [vec![getreg(3, 0, 9)], store].concat(),
    ));
    out
}

/// Where the code that calls through memory or constants go to lies.
const CALLEE: u64 = 0x18_0000;

/// Calls and jumps to code in the program, after a branch over it, and to
/// code elsewhere, the target built from `s_getpc_b64`, relative, constant
/// or loaded from memory. The called code runs `v_add_f32` into v10 and
/// returns through s[30:31] (`s_setpc_b64`, or `s_swappc_b64` that drops
/// the address it would save); v11 keeps the return address.
fn calls() -> Vec<Case> {
    const S30: u32 = 30;
    let swappc = sop1("s_swappc_b64", S30, S4);
    let setpc = |s: u32| 0xBE80_0000 | (op("s_setpc_b64").code as u32) << 8 | s;
    let callee = [v_add(), setpc(S30)];
    // s_getpc_b64 s[4:5] gives the address of the s_add_u32 after it.
    let getpc_plus = |bytes: u32| {
        vec![
            sop1("s_getpc_b64", S4, 0),
            sop2("s_add_u32", S4, S4, INLINE_0 + bytes),
            sop2("s_addc_u32", S4 + 1, S4 + 1, INLINE_0),
        ]
    };
    let s_mov_literal = |sdst: u32, k: u32| [sop1("s_mov_b32", sdst, 0xFF), k];
    let load_target = [
        0xF400_0000 | (op("s_load_dwordx2").code as u32) << 18 | S4 << 6,
        0xFA00_0000, // s_load_dwordx2 s[4:5], s[0:1], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    let over_callee = sopp("s_branch", 2);
    let mut out = Vec::new();
    out.extend(test(
        "s_swappc_b64",
        "target from s_getpc_b64, called code after a branch",
        getpc_plus(16),
        [&[swappc, over_callee][..], &callee].concat(),
        vec![v_mov(11, S30)],
    ));
    const NULL: u32 = 125;
    out.extend(test(
        "s_swappc_b64",
        "returning with s_swappc_b64 null",
        getpc_plus(16),
        vec![
            swappc,
            over_callee,
            v_add(),
            sop1("s_swappc_b64", NULL, S30),
        ],
        vec![v_mov(11, S30)],
    ));
    let call = 0xB000_0000 | (op("s_call_b64").code as u32) << 23 | S30 << 16 | 1;
    out.extend(test(
        "s_call_b64",
        "called code after a branch",
        Vec::new(),
        [&[call, over_callee][..], &callee].concat(),
        vec![v_mov(11, S30)],
    ));
    out.extend(test(
        "s_setpc_b64",
        "jump over an instruction, target from s_getpc_b64",
        getpc_plus(16),
        vec![setpc(S4), v_mov(11, INLINE_0 + 5), v_add()],
        Vec::new(),
    ));
    out.extend(test(
        "s_swappc_b64",
        "constant target elsewhere",
        [s_mov_literal(S4, CALLEE as u32), s_mov_literal(S4 + 1, 0)].concat(),
        vec![swappc],
        vec![v_mov(11, S30)],
    ));
    let mut inputs = inputs("s_swappc_b64 target from memory");
    inputs[..2].copy_from_slice(&[CALLEE as u32, (CALLEE >> 32) as u32]);
    let body = Body {
        setup: &load_target,
        inst: &[swappc],
        after: &[v_mov(11, S30)],
    };
    out.extend(around(
        op("s_swappc_b64"),
        "target loaded from memory".into(),
        body,
        false,
        BASE_MODE,
        Some(inputs),
    ));
    for case in out
        .iter_mut()
        .filter(|c| c.variant.contains("elsewhere") || c.variant.contains("memory"))
    {
        let bytes: Vec<u8> = callee.iter().flat_map(|w| w.to_le_bytes()).collect();
        write_pages(&mut case.input, CALLEE, &bytes);
    }
    out
}

/// Opcodes whose effect on a program's results is none at all.
const NO_EFFECT: [&str; 18] = [
    "s_nop",
    "s_sleep",
    "s_setprio",
    "s_waitcnt",
    "s_waitcnt_vmcnt",
    "s_waitcnt_expcnt",
    "s_waitcnt_lgkmcnt",
    "s_waitcnt_vscnt",
    "s_clause",
    "s_inst_prefetch",
    "s_icache_inv",
    "s_dcache_inv",
    "s_gl1_inv",
    "buffer_gl0_inv",
    "buffer_gl1_inv",
    "v_nop",
    "v_pipeflush",
    "ds_nop",
];

fn no_effect() -> Vec<Case> {
    let mut out = Vec::new();
    for name in NO_EFFECT {
        let Some(o) = by_name(name) else {
            continue;
        };
        let inst = match o.fmt {
            Fmt::Vop1 => vec![0x7E00_0000 | (o.code as u32) << 9],
            _ => match instance(o) {
                Some(i) => i,
                None => continue,
            },
        };
        out.extend(test(
            name,
            "leaves v_add_f32's result alone",
            Vec::new(),
            inst,
            vec![v_add()],
        ));
        if o.fmt == Fmt::Vop1 {
            let e64 = vec![0xD400_0000 | (o.code as u32 + 0x180) << 16, 0];
            out.extend(test(
                name,
                "e64 leaves v_add_f32's result alone",
                Vec::new(),
                e64,
                vec![v_add()],
            ));
        }
    }
    out.extend(test(
        "s_barrier",
        "one wave",
        Vec::new(),
        vec![sopp("s_barrier", 0)],
        vec![v_add()],
    ));
    out.extend(test(
        "s_endpgm",
        "before any store",
        vec![v_add()],
        vec![sopp("s_endpgm", 0)],
        Vec::new(),
    ));
    out
}

/// Every flow and machine-state test.
pub fn cases() -> Vec<Case> {
    [
        branches(),
        saveexec(),
        machine_state(),
        calls(),
        no_effect(),
    ]
    .concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_tests_decode_to_what_they_test() {
        let cases = cases();
        let text = |o: &str| {
            cases
                .iter()
                .find(|c| c.opcode == o)
                .map(|c| c.text.clone())
                .unwrap_or_else(|| panic!("no test of {o}"))
        };
        assert_eq!(text("s_cbranch_scc1"), "s_cbranch_scc1 3");
        assert_eq!(
            text("s_and_saveexec_b64"),
            "s_and_saveexec_b64 s[20:21], s[24:25]"
        );
        assert_eq!(text("s_getpc_b64"), "s_getpc_b64 s[20:21]");
        assert_eq!(text("s_swappc_b64"), "s_swappc_b64 s[30:31], s[4:5]");
        assert_eq!(text("s_call_b64"), "s_call_b64 s[30:31], 1");
        assert_eq!(text("s_setpc_b64"), "s_setpc_b64 s[4:5]");
        assert_eq!(
            cases
                .iter()
                .filter(|c| c.opcode.ends_with("pc_b64"))
                .count(),
            6
        );
        assert!(text("s_setreg_imm32_b32").starts_with("s_setreg_imm32_b32 hwreg(HW_REG_MODE"));
        for name in NO_EFFECT {
            text(name);
        }
    }
}
