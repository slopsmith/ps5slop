//! Vector memory tests: every MUBUF, MTBUF, GLOBAL and FLAT load, store
//! and atomic. Each wave takes one of [`DESCRIPTORS`] (formats, strides,
//! sizes, swizzles, destination selects, out-of-bounds modes) and each lane
//! a random index, offset and data; buffer opcodes run with offen, idxen,
//! both, and an immediate offset with an SGPR offset. Loads and atomics
//! store what they returned; stores land in memory that is compared whole.

use gfx10::decode::decode;
use gfx10::opinfo::{Fmt, OpInfo, all};
use gfx10::print::{Syntax, print_checked};

use super::{BASE_MODE, INPUTS, Rng, Test, build};
use crate::{Case, Dst, write_pages};

const GROUPS: u32 = 16;
const IN_WORDS: usize = 16;
const OUT_NAMES: [&str; 8] = ["ret0", "ret1", "ret2", "ret3", "-", "-", "-", "-"];
/// Random words loads read.
const DATA: u64 = 0x30_0000;
/// Random words stores and atomics change, compared whole afterwards.
const STORE: u64 = 0x50_0000;
/// Room for the farthest access any descriptor reaches from its index and
/// offset, so every access the range check lets through is mapped.
const AREA_BYTES: u32 = 0x2_0000;
/// Each wave's buffer descriptor, 16 bytes apart.
const DESC: u64 = INPUTS + 0x8_0000;

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
const LOADS: [u32; 18] = [
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
    0x8F0A_8404, // s_lshl_b32 s10, s4, 4
    0x800A_FF0A,
    0x0008_0000, // s_add_u32 s10, s10, 0x80000
    0xBEAC_0390, // s_mov_b32 s44, 16
];
const DESCRIPTOR_AND_WAIT: [u32; 7] = [
    0xF408_0A00,
    0x1400_0000, // s_load_dwordx4 s[40:43], s[0:1], s10
    0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
    0x7E28_030F, // v_mov_b32_e32 v20, v15
    0x7E2A_030F, // v_mov_b32_e32 v21, v15
    0x7E2C_030F, // v_mov_b32_e32 v22, v15
    0x7E2E_030F, // v_mov_b32_e32 v23, v15
];
const EPILOGUE: [u32; 10] = [
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E30_0280, // v_mov_b32_e32 v24, 0
    0x7E32_0280, // v_mov_b32_e32 v25, 0
    0x7E34_0280, // v_mov_b32_e32 v26, 0
    0x7E36_0280, // v_mov_b32_e32 v27, 0
    0xDC78_8000,
    0x0002_141F, // global_store_dwordx4 v31, v[20:23], s[2:3]
    0xDC78_8010,
    0x0002_181F, // global_store_dwordx4 v31, v[24:27], s[2:3] offset:16
    0xBF81_0000, // s_endpgm
];

/// `s_mov_b32 s46, lo`, `s_mov_b32 s47, hi`: the area GLOBAL loads use as `saddr`.
fn saddr(base: u64) -> [u32; 4] {
    [0xBEAE_03FF, base as u32, 0xBEAF_03FF, (base >> 32) as u32]
}

const INDEX: u32 = 1;
const OFFSET: u32 = 2;
/// Index and offset for accesses with both, apart from the ones `idxen` and
/// `offen` use alone, so each can keep lanes apart.
const BOTH: u32 = 11;
const GLOBAL_OFFSET: u32 = 3;
const FLAT_ADDRESS: u32 = 9;
const DATA_VGPR: u32 = 4;
const VDST: u32 = 20;
const SRSRC: u32 = 40;
const SOFFSET: u32 = 44;
const SADDR: u32 = 46;
const NULL: u32 = 0x7D;

/// A buffer descriptor: name, FORMAT, STRIDE, NUM_RECORDS, DST_SEL,
/// SWIZZLE_ENABLE and INDEX_STRIDE, OOB_SELECT.
pub type Descriptor = (&'static str, u32, u32, u32, [u32; 4], u32, u32);

/// The descriptors waves use.
pub const DESCRIPTORS: [Descriptor; 16] = [
    ("32_FLOAT raw", 22, 0, 0x1_0000, [4, 5, 6, 7], 0, 3),
    ("32_FLOAT stride 16", 22, 16, 0x800, [4, 5, 6, 7], 0, 0),
    ("8_8_8_8_UNORM stride 4", 56, 4, 0x1000, [4, 5, 6, 7], 0, 1),
    ("16_16_SNORM stride 8", 24, 8, 0x1000, [4, 5, 6, 7], 0, 0),
    (
        "32_32_UINT stride 12, 100 records",
        62,
        12,
        100,
        [4, 5, 6, 7],
        0,
        1,
    ),
    ("10_10_10_2_UNORM wzyx", 44, 4, 0x1000, [7, 6, 5, 4], 0, 1),
    (
        "16_16_16_16_FLOAT stride 8",
        71,
        8,
        0x1000,
        [4, 5, 6, 7],
        0,
        1,
    ),
    ("8_8_8_8_SNORM 7 records", 57, 4, 7, [4, 5, 6, 7], 0, 0),
    ("32_32_32_32_FLOAT x01w", 77, 16, 0x800, [4, 0, 1, 7], 0, 1),
    ("11_11_10_FLOAT stride 4", 43, 4, 0x1000, [4, 5, 6, 7], 0, 1),
    (
        "8_8_8_8_UINT swizzled stride 20",
        60,
        20,
        0x400,
        [4, 5, 6, 7],
        1 | 1 << 2,
        1,
    ),
    ("16_UNORM stride 2", 7, 2, 0x2000, [4, 5, 6, 7], 0, 1),
    (
        "2_10_10_10_SNORM checks off",
        51,
        4,
        0x1000,
        [4, 5, 6, 7],
        0,
        2,
    ),
    ("32_SINT raw 64 bytes", 21, 0, 64, [4, 5, 6, 7], 0, 3),
    ("5_6_5_UNORM stride 2", 133, 2, 0x2000, [4, 5, 6, 7], 0, 1),
    (
        "16_16_16_16_UINT checks off",
        69,
        8,
        0x1000,
        [4, 5, 6, 7],
        0,
        2,
    ),
];

// For stores: the group's own span, and no stride 0, where every index
// would store to one place.
fn descriptor(base: u64, group: u64, writes: bool) -> [u32; 4] {
    let (_, format, stride, records, sel, swizzle, oob) =
        DESCRIPTORS[group as usize % DESCRIPTORS.len()];
    let (base, stride) = if writes {
        (base + group * STORE_SPAN, stride.max(16))
    } else {
        (base, stride)
    };
    const RESOURCE_LEVEL: u32 = 1 << 24;
    [
        base as u32,
        (base >> 32) as u32 & 0xFFFF | stride << 16 | (swizzle & 1) << 31,
        records,
        sel[0]
            | sel[1] << 3
            | sel[2] << 6
            | sel[3] << 9
            | format << 12
            | (swizzle >> 1) << 21
            | RESOURCE_LEVEL
            | oob << 28,
    ]
}

/// How a buffer instruction addresses memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Offen,
    Idxen,
    Both,
    /// offen with an immediate offset and an SGPR offset.
    Offsets,
    /// GLOBAL: a VGPR offset from an SGPR base.
    Saddr,
    /// GLOBAL with a negative immediate offset, FLAT with a positive one.
    Imm,
    /// GLOBAL and FLAT: a 64-bit VGPR address.
    Vaddr,
}

fn is_store(name: &str) -> bool {
    name.contains("store")
}

fn is_atomic(name: &str) -> bool {
    name.contains("atomic")
}

/// Why an opcode has no generated memory test, or `None` if it has.
pub fn unsupported(op: &OpInfo) -> Option<&'static str> {
    if !matches!(op.fmt, Fmt::Mubuf | Fmt::Mtbuf | Fmt::Global | Fmt::Flat) {
        return Some("not a vector memory instruction tested here");
    }
    if op.name.contains("lds") || op.name.contains("wbinv") || op.name.contains("gl1_inv") {
        return Some("cache control or LDS loads, tested by hand");
    }
    if op.name.contains("addtid") {
        return Some("lane-indexed form, tested by hand");
    }
    None
}

fn encode(op: &OpInfo, mode: Mode, format: u32) -> Option<Vec<u32>> {
    let store = is_store(op.name);
    let atomic = is_atomic(op.name);
    let glc = atomic as u32;
    let vdata = if store || atomic { DATA_VGPR } else { VDST };
    let c = op.code as u32;
    match op.fmt {
        Fmt::Mubuf | Fmt::Mtbuf => {
            let (offen, idxen, vaddr, offset, soffset) = match mode {
                Mode::Offen => (1, 0, OFFSET, 0, 0x80),
                Mode::Idxen => (0, 1, INDEX, 0, 0x80),
                Mode::Both => (1, 1, BOTH, 0, 0x80),
                Mode::Offsets => (1, 0, OFFSET, 0x40, SOFFSET),
                _ => return None,
            };
            let w0 = offen << 12 | idxen << 13 | glc << 14 | offset;
            let w1 = soffset << 24 | (SRSRC >> 2) << 16 | vdata << 8 | vaddr;
            Some(if op.fmt == Fmt::Mubuf {
                vec![0xE000_0000 | (c & 0x7F) << 18 | (c >> 7) << 25 | w0, w1]
            } else {
                vec![
                    0xE800_0000 | format << 19 | (c & 7) << 16 | w0,
                    w1 | (c >> 3) << 21,
                ]
            })
        }
        Fmt::Global | Fmt::Flat => {
            let seg = if op.fmt == Fmt::Global { 2 } else { 0 };
            let (addr, saddr, offset) = match (mode, op.fmt) {
                (Mode::Saddr, Fmt::Global) => (GLOBAL_OFFSET, SADDR, 0),
                (Mode::Imm, Fmt::Global) => (GLOBAL_OFFSET, SADDR, 0xFF0),
                (Mode::Vaddr, _) => (FLAT_ADDRESS, NULL, 0),
                (Mode::Imm, Fmt::Flat) => (FLAT_ADDRESS, NULL, 0x40),
                _ => return None,
            };
            let vdst = if store { 0 } else { VDST };
            let data = if store || atomic { DATA_VGPR } else { 0 };
            Some(vec![
                0xDC00_0000 | c << 18 | glc << 16 | seg << 14 | offset,
                vdst << 24 | saddr << 16 | data << 8 | addr,
            ])
        }
        _ => None,
    }
}

/// Bytes of the store area each wave writes in.
const STORE_SPAN: u64 = 0x2000;

/// Room below the first GLOBAL and FLAT address for the negative immediate offset.
const GLOBAL_START: u64 = 0x40;

// Stores go where no other lane stores: Metal leaves the lane order undefined.
// Swizzled stores take every fourth index, as wide accesses there overlap.
fn records(base: u64, writes: bool, seed: &str) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let lanes = (GROUPS * 64) as usize;
    let mut out = Vec::with_capacity(lanes * IN_WORDS);
    for i in 0..lanes {
        let (lane, group) = ((i % 64) as u32, (i / 64) as u64);
        let mut w = [0u32; IN_WORDS];
        let global = if writes {
            let swizzled = DESCRIPTORS[group as usize % DESCRIPTORS.len()].5 & 1 != 0;
            w[0] = if swizzled { lane * 4 } else { lane };
            w[1] = lane * 16;
            (w[10], w[11]) = if swizzled {
                (lane * 4, lane % 4 * 4)
            } else {
                (lane, lane * 16)
            };
            group * STORE_SPAN + GLOBAL_START + lane as u64 * 16
        } else {
            w[0] = if rng.below(8) == 0 {
                rng.below(0x3000) as u32
            } else {
                rng.below(300) as u32
            };
            w[1] = rng.below(0x800) as u32 & !3;
            (w[10], w[11]) = (w[0], rng.below(0x800) as u32 & !3);
            GLOBAL_START + (rng.below(0x8000) as u64 & !3)
        };
        w[2] = global as u32;
        for word in &mut w[3..7] {
            *word = rng.next() as u32;
        }
        let flat = base + global;
        w[8] = flat as u32;
        w[9] = (flat >> 32) as u32;
        w[14] = rng.next() as u32;
        out.extend_from_slice(&w);
    }
    out
}

fn case(op: &OpInfo, mode: Mode, format: u32) -> Option<Case> {
    let inst = encode(op, mode, format)?;
    let decoded = decode(&inst).ok()?;
    if decoded.op.name != op.name {
        return None;
    }
    let text = print_checked(&decoded, Syntax { wave64: true })?;
    let writes = is_store(op.name) || is_atomic(op.name);
    let base = if writes { STORE } else { DATA };
    let mut code = LOADS.to_vec();
    code.extend(saddr(base));
    code.extend(DESCRIPTOR_AND_WAIT);
    code.extend(&inst);
    code.extend(EPILOGUE);
    let variant = match op.fmt {
        Fmt::Mtbuf => format!("{mode:?} format {format}").to_lowercase(),
        _ => format!("{mode:?}").to_lowercase(),
    };
    let seed = format!("{} {variant}", op.name);
    let mut case = build(Test {
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
        lds_bytes: 0,
        inputs: records(base, writes, &seed),
    });
    let table: Vec<u8> = (0..GROUPS as usize)
        .flat_map(|g| descriptor(base, g as u64, writes))
        .flat_map(u32::to_le_bytes)
        .collect();
    write_pages(&mut case.input, DESC, &table);
    for (area, name) in [(DATA, "load data"), (STORE, "store data")] {
        let mut rng = Rng::new(name);
        let bytes: Vec<u8> = (0..AREA_BYTES / 4)
            .flat_map(|_| (rng.next() as u32).to_le_bytes())
            .collect();
        write_pages(&mut case.input, area, &bytes);
    }
    if writes {
        case.checked = (STORE, AREA_BYTES);
    }
    Some(case)
}

/// Every vector memory test, in opcode table order.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for op in all().iter().filter(|op| unsupported(op).is_none()) {
        match op.fmt {
            Fmt::Mubuf => {
                for mode in [Mode::Offen, Mode::Idxen, Mode::Both, Mode::Offsets] {
                    out.extend(case(op, mode, 0));
                }
            }
            Fmt::Mtbuf => {
                for format in [22, 56, 44, 71] {
                    for mode in [Mode::Both, Mode::Offsets] {
                        out.extend(case(op, mode, format));
                    }
                }
            }
            _ => {
                for mode in [Mode::Saddr, Mode::Imm, Mode::Vaddr] {
                    out.extend(case(op, mode, 0));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::opinfo::by_name;

    #[test]
    fn memory_tests_decode_to_what_they_test() {
        let text = |n: &str, m, f| case(by_name(n).unwrap(), m, f).unwrap().text;
        assert_eq!(
            text("buffer_load_dword", Mode::Both, 0),
            "buffer_load_dword v20, v[11:12], s[40:43], 0 idxen offen"
        );
        assert_eq!(
            text("buffer_store_dwordx2", Mode::Offsets, 0),
            "buffer_store_dwordx2 v[4:5], v2, s[40:43], s44 offen offset:64"
        );
        assert_eq!(
            text("global_load_dword", Mode::Imm, 0),
            "global_load_dword v20, v3, s[46:47] offset:-16"
        );
        assert_eq!(
            text("flat_store_dword", Mode::Vaddr, 0),
            "flat_store_dword v[9:10], v4"
        );
        let ops: std::collections::BTreeSet<String> =
            cases().into_iter().map(|c| c.opcode).collect();
        assert!(
            ops.len() > 150,
            "only {} memory opcodes would be tested",
            ops.len()
        );
    }
}
