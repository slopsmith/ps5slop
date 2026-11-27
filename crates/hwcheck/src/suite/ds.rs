//! LDS tests: every DS load, store, atomic, two-address form and permute
//! (`ds_swizzle_b32` is with the lane-crossing tests), in one-wave groups with 4 KiB of LDS. Each lane fills its
//! 64-byte LDS slot from its record, runs the instruction at a random
//! address (lanes collide on purpose, so atomics and their order are
//! tested too), then stores what the instruction returned and its slot.

use gfx10::decode::decode;
use gfx10::opinfo::{Fmt, OpInfo, Ty, all};
use gfx10::print::{Syntax, print_checked};

use super::{BASE_MODE, Rng, Test, build, edge, edges};
use crate::{Case, Dst};

const GROUPS: u32 = 16;
const IN_WORDS: usize = 16;
const OUT_NAMES: [&str; 8] = [
    "ret0", "ret1", "ret2", "ret3", "slot0", "slot1", "slot2", "slot3",
];
const LDS_BYTES: u32 = 4096;

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
const PROLOGUE: [u32; 29] = [
    0x343C_0086, // v_lshlrev_b32_e32 v30, 6, v0
    0x8F08_8C04, // s_lshl_b32 s8, s4, 12
    0x4A3C_3C08, // v_add_nc_u32_e32 v30, s8, v30
    0x343E_0085, // v_lshlrev_b32_e32 v31, 5, v0
    0x8F09_8B04, // s_lshl_b32 s9, s4, 11
    0x4A3E_3E09, // v_add_nc_u32_e32 v31, s9, v31
    0xDC38_8000,
    0x0100_001E, // global_load_dwordx4 v[1:4], v30, s[0:1]
    0xDC38_8010,
    0x0500_001E, // global_load_dwordx4 v[5:8], v30, s[0:1] offset:16
    0xDC38_8020,
    0x0900_001E, // global_load_dwordx4 v[9:12], v30, s[0:1] offset:32
    0xDC38_8030,
    0x0D00_001E, // global_load_dwordx4 v[13:16], v30, s[0:1] offset:48
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x343A_0086, // v_lshlrev_b32_e32 v29, 6, v0
    0xDB7C_0000,
    0x0000_0B1D, // ds_write_b128 v29, v[11:14]
    0xDB7C_0010,
    0x0000_0B1D, // ds_write_b128 v29, v[11:14] offset:16
    0xDB7C_0020,
    0x0000_0B1D, // ds_write_b128 v29, v[11:14] offset:32
    0xDB7C_0030,
    0x0000_0B1D, // ds_write_b128 v29, v[11:14] offset:48
    0x7E28_030F, // v_mov_b32_e32 v20, v15
    0x7E2A_030F, // v_mov_b32_e32 v21, v15
    0x7E2C_030F, // v_mov_b32_e32 v22, v15
    0x7E2E_030F, // v_mov_b32_e32 v23, v15
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];
const EPILOGUE: [u32; 9] = [
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xDBFC_0000,
    0x1800_001D, // ds_read_b128 v[24:27], v29
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xDC78_8000,
    0x0002_141F, // global_store_dwordx4 v31, v[20:23], s[2:3]
    0xDC78_8010,
    0x0002_181F, // global_store_dwordx4 v31, v[24:27], s[2:3] offset:16
    0xBF81_0000, // s_endpgm
];

const ADDR: u32 = 1;
/// The LDS address M0 holds for the forms that take theirs from it.
const M0_ADDRESS: u32 = 0x100;
const DATA0: u32 = 3;
const DATA1: u32 = 7;
const VDST: u32 = 20;

/// What a DS opcode reads and writes, from its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    /// Dwords per element.
    width: u32,
    /// Data operands (0, 1 or 2).
    data: u32,
    /// Whether it returns a value.
    returns: bool,
    /// Two addresses from `offset0` and `offset1` (`read2`, `write2`, `wrxchg2`).
    two: bool,
    /// Lanes exchange values (permute): the address is a lane.
    lanes: bool,
    /// The address comes from M0 rather than a VGPR (append, consume, addtid).
    m0: bool,
}

fn shape(name: &str) -> Option<Shape> {
    const SKIP: [&str; 7] = [
        "gws", "ordered", "src2", "nop", "barrier", "condxchg", "swizzle",
    ];
    if SKIP.iter().any(|s| name.contains(s)) {
        return None;
    }
    let width = if name.contains("b128") {
        4
    } else if name.contains("b96") {
        3
    } else if ["b64", "u64", "i64", "f64"]
        .iter()
        .any(|w| name.contains(w))
    {
        2
    } else {
        1
    };
    let two = name.contains("2_") || name.contains("2st64");
    let lanes = name.contains("permute");
    let m0 = ["append", "consume", "addtid"]
        .iter()
        .any(|n| name.contains(n));
    let s = if name.starts_with("ds_read") || name.contains("append") || name.contains("consume") {
        Shape {
            width,
            data: 0,
            returns: true,
            two,
            lanes,
            m0,
        }
    } else if name.starts_with("ds_write") && !name.contains("rtn") {
        Shape {
            width,
            data: if two { 2 } else { 1 },
            returns: false,
            two,
            lanes,
            m0,
        }
    } else if lanes {
        Shape {
            width,
            data: 1,
            returns: true,
            two,
            lanes,
            m0,
        }
    } else {
        let data = if two || name.contains("cmpst") || name.contains("mskor") {
            2
        } else {
            1
        };
        Shape {
            width,
            data,
            returns: name.contains("rtn"),
            two,
            lanes,
            m0,
        }
    };
    Some(s)
}

/// Why an opcode has no generated LDS test, or `None` if it has.
pub fn unsupported(op: &OpInfo) -> Option<&'static str> {
    if op.fmt != Fmt::Ds {
        return Some("not an LDS instruction");
    }
    if shape(op.name).is_none() {
        return Some("GDS, GWS, M0, a 10.1-only form or a swizzle, tested elsewhere or by hand");
    }
    None
}

/// Offsets tested: `(offset0, offset1)`, a 16-bit offset when not `two`.
fn offsets(s: Shape) -> Vec<(u32, u32)> {
    if s.two {
        return vec![(0, 1), (3, 7)];
    }
    let align = if s.width >= 3 { 16 } else { 8 };
    vec![(0, 0), (align * 3, 0)]
}

fn encode(op: &OpInfo, s: Shape, offset0: u32, offset1: u32) -> [u32; 2] {
    let vdst = if s.returns { VDST } else { 0 };
    let data1 = if s.data == 2 { DATA1 } else { 0 };
    let data0 = if s.data >= 1 { DATA0 } else { 0 };
    [
        0xD800_0000 | (op.code as u32) << 18 | offset1 << 8 | offset0,
        vdst << 24 | data1 << 16 | data0 << 8 | if s.m0 { 0 } else { ADDR },
    ]
}

fn data_type(name: &str) -> Ty {
    if name.ends_with("f32") {
        Ty::F32
    } else if name.ends_with("f64") {
        Ty::F64
    } else {
        Ty::U32
    }
}

// Stores go where no other lane stores, so the order lanes store in, which
// the hardware defines and Metal does not, stays out of the test.
fn records(s: Shape, name: &str, seed: &str) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let t = data_type(name);
    let writes = name.starts_with("ds_write") && !name.contains("rtn");
    let align = if s.width >= 3 { 16 } else { 4 * s.width };
    let lanes = (GROUPS * 64) as usize;
    let mut out = Vec::with_capacity(lanes * IN_WORDS);
    for _ in 0..lanes {
        let mut w = [0u32; IN_WORDS];
        let lane = (out.len() / IN_WORDS % 64) as u32;
        w[0] = if s.lanes {
            rng.below(128) as u32 * 4
        } else if writes && name.contains("st64") {
            lane * 4 * s.width
        } else if writes {
            lane * 64
        } else {
            (rng.below(3072) as u32) / align * align
        };
        let value = |rng: &mut Rng| {
            if rng.next() & 1 == 0 {
                edge(t, rng.below(edges(t)), rng.next())
            } else if t == Ty::U32 {
                rng.next() & if rng.next() & 1 == 0 { 0xFF } else { u64::MAX }
            } else {
                rng.next()
            }
        };
        for d in 0..2 {
            let mut k = 0;
            while k < 4 {
                let v = value(&mut rng);
                w[2 + 4 * d + k] = v as u32;
                if t == Ty::F64 && k + 1 < 4 {
                    w[2 + 4 * d + k + 1] = (v >> 32) as u32;
                    k += 1;
                }
                k += 1;
            }
        }
        for word in &mut w[10..15] {
            *word = rng.next() as u32;
        }
        out.extend_from_slice(&w);
    }
    out
}

fn case(op: &OpInfo, offset0: u32, offset1: u32) -> Option<Case> {
    let s = shape(op.name)?;
    let inst = encode(op, s, offset0, offset1);
    let decoded = decode(&inst).ok()?;
    if decoded.op.name != op.name {
        return None;
    }
    let text = print_checked(&decoded, Syntax { wave64: true })?;
    let mut code = PROLOGUE.to_vec();
    if s.m0 {
        const S_MOV_B32_M0_LITERAL: u32 = 0xBEFC_03FF;
        code.extend([S_MOV_B32_M0_LITERAL, M0_ADDRESS]);
    }
    code.extend(inst);
    code.extend(EPILOGUE);
    let variant = if s.two {
        format!("offset0 {offset0} offset1 {offset1}")
    } else {
        format!("offset {:#x}", offset1 << 8 | offset0)
    };
    let seed = format!("{} {variant}", op.name);
    Some(build(Test {
        opcode: op.name,
        variant,
        text,
        code,
        mode: BASE_MODE,
        groups: GROUPS,
        per_wave: false,
        in_words: IN_WORDS as u32,
        out_names: &OUT_NAMES,
        dst: Dst::Bits,
        lds_bytes: LDS_BYTES,
        inputs: records(s, op.name, &seed),
    }))
}

/// Every LDS test, in opcode table order.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for op in all().iter().filter(|op| unsupported(op).is_none()) {
        let s = shape(op.name).expect("checked by unsupported");
        for (o0, o1) in offsets(s) {
            out.extend(case(op, o0, o1));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::opinfo::by_name;

    #[test]
    fn lds_tests_decode_to_what_they_test() {
        let text = |n: &str, a, b| case(by_name(n).unwrap(), a, b).unwrap().text;
        assert_eq!(text("ds_add_rtn_u32", 0, 0), "ds_add_rtn_u32 v20, v1, v3");
        assert_eq!(
            text("ds_write_b64", 24, 0),
            "ds_write_b64 v1, v[3:4] offset:24"
        );
        assert_eq!(
            text("ds_read2_b32", 3, 7),
            "ds_read2_b32 v[20:21], v1 offset0:3 offset1:7"
        );
        assert_eq!(
            text("ds_cmpst_rtn_b32", 0, 0),
            "ds_cmpst_rtn_b32 v20, v1, v3, v7"
        );
        let ops: std::collections::BTreeSet<String> =
            cases().into_iter().map(|c| c.opcode).collect();
        assert!(
            ops.len() > 90,
            "only {} LDS opcodes would be tested",
            ops.len()
        );
    }
}
