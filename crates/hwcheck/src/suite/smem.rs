//! Scalar memory tests: every `s_load_dword*` and `s_buffer_load_dword*`
//! with immediate and SGPR offsets, the buffer forms through descriptors of
//! varied size and out-of-bounds mode. Each wave loads its record (an SGPR
//! offset, the data's address, a buffer descriptor), runs the load into
//! `s[40:55]` (cleared first) and stores those sixteen words.

use gfx10::decode::decode;
use gfx10::opinfo::{Fmt, OpInfo, all};
use gfx10::print::{Syntax, print_checked};

use super::{BASE_MODE, Rng, Test, build};
use crate::{Case, Dst, write_pages};

const GROUPS: u32 = 1024;
const IN_WORDS: usize = 8;
const OUT_NAMES: [&str; 16] = [
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14",
    "d15",
];
/// Random words the loads read.
const DATA: u64 = 0x30_0000;
const DATA_BYTES: u32 = 0x1_0000;

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
const PROLOGUE: [u32; 5] = [
    0x8F08_8504, // s_lshl_b32 s8, s4, 5
    0x8F09_8604, // s_lshl_b32 s9, s4, 6
    0xF40C_0600,
    0x1000_0000, // s_load_dwordx8 s[24:31], s[0:1], s8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];
const WAIT: u32 = 0xBF8C_C07F;
const EXEC_ONE: u32 = 0xBEFE_0481;
const STORES: [u32; 9] = [
    0x7E12_0209, // v_mov_b32_e32 v9, s9
    0xDC78_8000,
    0x0002_0109, // global_store_dwordx4 v9, v[1:4], s[2:3]
    0xDC78_8010,
    0x0002_0509, // global_store_dwordx4 v9, v[5:8], s[2:3] offset:16
    0xDC78_8020,
    0x0002_0A09, // global_store_dwordx4 v9, v[10:13], s[2:3] offset:32
    0xDC78_8030,
    0x0002_0E09, // global_store_dwordx4 v9, v[14:17], s[2:3] offset:48
];
const S_ENDPGM: u32 = 0xBF81_0000;

const SDATA: u32 = 40;
const SOFFSET: u32 = 24;
const ADDRESS: u32 = 26;
const DESCRIPTOR: u32 = 28;
const NO_SOFFSET: u32 = 0x7D;

/// Immediate offsets tested, each its own test; negative ones only for `s_load`.
const OFFSETS: [i32; 4] = [0, 0x10, 0x1004, -16];

/// Why an opcode has no generated scalar memory test, or `None` if it has.
pub fn unsupported(op: &OpInfo) -> Option<&'static str> {
    if op.fmt != Fmt::Smem {
        return Some("not a scalar memory instruction");
    }
    if !(op.name.starts_with("s_load_dword") || op.name.starts_with("s_buffer_load_dword")) {
        return Some("not a scalar load");
    }
    None
}

fn encode(op: &OpInfo, offset: i32, soffset: bool) -> [u32; 2] {
    let sbase = if op.name.starts_with("s_buffer") {
        DESCRIPTOR
    } else {
        ADDRESS
    };
    let so = if soffset { SOFFSET } else { NO_SOFFSET };
    [
        0xF400_0000 | (op.code as u32) << 18 | SDATA << 6 | sbase >> 1,
        so << 25 | (offset as u32 & 0x1F_FFFF),
    ]
}

// The address is 256 bytes into the data so negative offsets stay in it:
// past mapped memory, a fault and Metal's zeros cannot be compared.
fn records(seed: &str) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    const SIZES: [u32; 7] = [0, 4, 60, 64, 0x100, 0x8000, 0xFFFF_FFFF];
    let mut out = Vec::with_capacity(GROUPS as usize * IN_WORDS);
    for _ in 0..GROUPS {
        let size = SIZES[rng.below(SIZES.len())];
        let soffset = match rng.below(4) {
            0 => (rng.next() as u32 % (DATA_BYTES / 2)) & !3,
            1 => rng.next() as u32 % (DATA_BYTES / 2),
            2 => {
                size.min(DATA_BYTES / 2)
                    .saturating_sub(rng.below(64) as u32)
                    & !3
            }
            _ => rng.below(64) as u32 * 4,
        };
        let oob = rng.below(4) as u32;
        const DST_SEL_XYZW: u32 = 4 | 5 << 3 | 6 << 6 | 7 << 9;
        const FORMAT_32_FLOAT: u32 = 22 << 12;
        const RESOURCE_LEVEL: u32 = 1 << 24;
        out.extend_from_slice(&[
            soffset,
            0,
            (DATA + 0x100) as u32,
            ((DATA + 0x100) >> 32) as u32,
            DATA as u32,
            (DATA >> 32) as u32,
            size,
            DST_SEL_XYZW | FORMAT_32_FLOAT | RESOURCE_LEVEL | oob << 28,
        ]);
    }
    out
}

fn case(op: &OpInfo, offset: i32, soffset: bool) -> Option<Case> {
    let inst = encode(op, offset, soffset);
    let decoded = decode(&inst).ok()?;
    if decoded.op.name != op.name {
        return None;
    }
    let text = print_checked(&decoded, Syntax { wave64: true })?;
    let mut code = PROLOGUE.to_vec();
    const S_MOV_B32_ZERO: u32 = 0xBE80_0380;
    code.extend((SDATA..SDATA + 16).map(|s| S_MOV_B32_ZERO | s << 16));
    code.extend(inst);
    code.extend([WAIT, EXEC_ONE]);
    const V_MOV_B32: u32 = 0x7E00_0200;
    let vgprs = (1..=8).chain(10..=17);
    code.extend(vgprs.zip(SDATA..).map(|(v, s)| V_MOV_B32 | v << 17 | s));
    code.extend(STORES);
    code.push(S_ENDPGM);
    let variant = format!(
        "offset {offset:#x}{}",
        if soffset { " + soffset" } else { "" }
    );
    let seed = format!("{} {variant}", op.name);
    let mut case = build(Test {
        opcode: op.name,
        variant,
        text,
        code,
        mode: BASE_MODE,
        groups: GROUPS,
        per_wave: true,
        in_words: IN_WORDS as u32,
        out_names: &OUT_NAMES,
        dst: Dst::Bits,
        lds_bytes: 0,
        inputs: records(&seed),
    });
    let mut rng = Rng::new("smem data");
    let data: Vec<u8> = (0..DATA_BYTES / 4)
        .flat_map(|_| (rng.next() as u32).to_le_bytes())
        .collect();
    write_pages(&mut case.input, DATA, &data);
    Some(case)
}

/// Every scalar memory test, in opcode table order.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for op in all().iter().filter(|op| unsupported(op).is_none()) {
        for offset in OFFSETS {
            if offset < 0 && op.name.starts_with("s_buffer") {
                continue;
            }
            out.extend(case(op, offset, true));
        }
        out.extend(case(op, 0x40, false));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::opinfo::by_name;

    #[test]
    fn scalar_loads_decode_to_the_opcode_they_test() {
        let text = |n: &str, o, s| case(by_name(n).unwrap(), o, s).unwrap().text;
        assert_eq!(
            text("s_load_dwordx16", 0x10, true),
            "s_load_dwordx16 s[40:55], s[26:27], s24 offset:0x10"
        );
        assert_eq!(
            text("s_buffer_load_dword", 0x40, false),
            "s_buffer_load_dword s40, s[28:31], 0x40"
        );
    }
}
