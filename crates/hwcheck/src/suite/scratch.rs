//! Scratch tests: every SCRATCH load and store as a round trip through each
//! lane's private memory. Each wave points FLAT_SCRATCH at its own 4 KiB so
//! waves cannot race; a load test first stores the lane's operands, a store
//! test reads back what it stored.

use gfx10::opinfo::{Fmt, OpInfo, all};

use super::BASE_MODE;
use super::valu::{Body, GROUPS, around};
use crate::{Case, write_pages};

const SCRATCH: u32 = 0x80_0000;
const WAVE_BYTES: u32 = 0x1000;

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
fn setup() -> Vec<u32> {
    vec![
        0x9328_FF04,
        WAVE_BYTES, // s_mul_i32 s40, s4, 0x1000
        0x8028_FF28,
        SCRATCH,     // s_add_u32 s40, s40, 0x800000
        0xBEA9_0380, // s_mov_b32 s41, 0
        0xB9A8_F814, // s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s40
        0xB9A9_F815, // s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s41
        0x7E12_0290, // v_mov_b32_e32 v9, 16
        0xDC78_4000,
        0x007D_0109, // scratch_store_dwordx4 v9, v[1:4], off
        0xBBFD_0000, // s_waitcnt_vscnt null, 0x0
    ]
}

const WAIT_LOADS: u32 = 0xBF8C_3F70;
const WAIT_STORES: u32 = 0xBBFD_0000;

fn scratch(op: &OpInfo, vdst: u32, data: u32) -> [u32; 2] {
    [
        0xDC00_4000 | (op.code as u32) << 18,
        vdst << 24 | 0x7D << 16 | data << 8 | 9,
    ]
}

fn case(op: &OpInfo) -> Option<Case> {
    let store = op.name.contains("store");
    let load_back = all().iter().find(|o| o.name == "scratch_load_dwordx2")?;
    let (inst, after) = if store {
        (
            scratch(op, 0, 5),
            [
                vec![WAIT_STORES],
                scratch(load_back, 10, 0).to_vec(),
                vec![WAIT_LOADS],
            ]
            .concat(),
        )
    } else {
        (scratch(op, 10, 0), vec![WAIT_LOADS])
    };
    let body = Body {
        setup: &setup(),
        inst: &inst,
        after: &after,
    };
    let mut case = around(op, "round trip".into(), body, false, BASE_MODE, None)?;
    const SCRATCH_EN: u32 = 1;
    const WAVES: u32 = 32;
    case.compute[0x213 - crate::COMPUTE_FIRST] |= SCRATCH_EN;
    case.compute[0x218 - crate::COMPUTE_FIRST] = WAVES | (WAVE_BYTES / 1024) << 12;
    let zeros = vec![0; (GROUPS * WAVE_BYTES) as usize];
    write_pages(&mut case.input, SCRATCH as u64, &zeros);
    Some(case)
}

/// Every scratch test.
pub fn cases() -> Vec<Case> {
    all()
        .iter()
        .filter(|o| o.fmt == Fmt::Scratch)
        .filter_map(case)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scratch_opcode_has_a_round_trip() {
        let n = all().iter().filter(|o| o.fmt == Fmt::Scratch).count();
        assert_eq!(cases().len(), n);
    }
}
