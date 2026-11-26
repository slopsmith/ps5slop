//! Generated instruction tests, one module per instruction family. Every
//! test is a small program around one instruction under test; operands are
//! edge cases of each operand's type (signed zeros, denormals, infinities,
//! quiet and signalling NaNs, integer boundaries, garbage in bits an operand
//! does not use) combined pairwise, then seeded random values.

use gfx10::opinfo::Ty;

use crate::{COMPUTE_COUNT, COMPUTE_FIRST, Case, Dst, write_pages};

mod draw;
mod ds;
mod flow;
mod formats;
mod lanes;
mod mem;
mod mimg;
mod salu;
mod scratch;
mod smem;
mod valu;

pub use draw::texture::{
    BIAS as TEXTURE_DRAW_BIAS, DESC as TEXTURE_DRAW_DESC, LOD as TEXTURE_DRAW_LOD, TextureDraw,
    draws as texture_draws,
};
pub use mimg::{DESC as IMAGE_DESC, sampling_heavy};
pub use valu::{Form, case as valu_case};

/// Where every test puts its program, operands and results.
pub const PROGRAM: u64 = 0x10_0000;
pub const INPUTS: u64 = 0x20_0000;
pub const OUTPUTS: u64 = 0x40_0000;

/// A float mode a program runs under: `COMPUTE_PGM_RSRC1` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    pub name: &'static str,
    /// FLOAT_MODE: f32 rounding (bits 0-1), f16/f64 rounding (2-3), f32
    /// denormals (4-5), f16/f64 denormals (6-7).
    pub float_mode: u8,
    pub dx10_clamp: bool,
    pub ieee: bool,
}

/// The mode tests run in unless they test modes: f32 denormals flushed,
/// f16 and f64 denormals kept, round to nearest even, DX10 clamp on, IEEE
/// mode off, as compute shaders commonly set.
pub const BASE_MODE: Mode = Mode {
    name: "base",
    float_mode: 0xC0,
    dx10_clamp: true,
    ieee: false,
};

/// The other modes float instructions are tested under.
pub const MODES: [Mode; 7] = [
    Mode {
        name: "all denormals kept",
        float_mode: 0xF0,
        ..BASE_MODE
    },
    Mode {
        name: "all denormals flushed",
        float_mode: 0x00,
        ..BASE_MODE
    },
    Mode {
        name: "round toward +inf",
        float_mode: 0xC5,
        ..BASE_MODE
    },
    Mode {
        name: "round toward -inf",
        float_mode: 0xCA,
        ..BASE_MODE
    },
    Mode {
        name: "round toward zero",
        float_mode: 0xCF,
        ..BASE_MODE
    },
    Mode {
        name: "IEEE mode",
        float_mode: 0xC0,
        dx10_clamp: false,
        ieee: true,
    },
    Mode {
        name: "IEEE mode with DX10 clamp",
        float_mode: 0xC0,
        dx10_clamp: true,
        ieee: true,
    },
];

impl Mode {
    pub(crate) fn rsrc1(self) -> u32 {
        const VGPR_BLOCKS: u32 = 7;
        const MEM_ORDERED: u32 = 1 << 25;
        VGPR_BLOCKS
            | (self.float_mode as u32) << 12
            | (self.dx10_clamp as u32) << 21
            | (self.ieee as u32) << 23
            | MEM_ORDERED
    }
}

pub(crate) fn is_value(t: Ty) -> bool {
    !matches!(t, Ty::Lane | Ty::Exec | Ty::Scc | Ty::M0 | Ty::Imm)
}

/// splitmix64, seeded from a test's name so every run makes the same operands.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: &str) -> Self {
        Rng(seed.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        }))
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const F32_EDGES: [u32; 26] = [
    0x0000_0000,
    0x8000_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x3F00_0000,
    0xBFC0_0000,
    0x4000_0000,
    0x4040_0000,
    0x0000_0001,
    0x807F_FFFF,
    0x0080_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0x7F80_0001,
    0xFFC0_0001,
    0x4049_0FDB,
    0x3F7F_FFFF,
    0x4B80_0001,
    0x5015_02F9,
    0x2EDB_E6FF,
    0x477F_E000,
    0x437F_8000,
    0xBE99_999A,
];
const F16_EDGES: [u16; 20] = [
    0x0000, 0x8000, 0x3C00, 0xBC00, 0x3800, 0x4000, 0x0001, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00,
    0xFC00, 0x7E00, 0x7C01, 0xFE01, 0x4248, 0x3BFF, 0x5640, 0xB4CD,
];
const U32_EDGES: [u32; 24] = [
    0,
    1,
    2,
    3,
    31,
    32,
    33,
    63,
    64,
    0xFF,
    0x100,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x1_0000,
    0x7FFF_FFFF,
    0x8000_0000,
    0x8000_0001,
    0xFFFF_FFFE,
    0xFFFF_FFFF,
    0x5555_5555,
    0xAAAA_AAAA,
    0x1234_5678,
    1_000_000,
];
const U16_EDGES: [u16; 14] = [
    0, 1, 2, 15, 16, 17, 0x7FFF, 0x8000, 0xFFFF, 0xFF, 0x100, 0x1234, 0x7F, 0x80,
];
const F64_EDGES: [u64; 20] = [
    0,
    0x8000_0000_0000_0000,
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000,
    0x3FE0_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x000F_FFFF_FFFF_FFFF,
    0x0010_0000_0000_0000,
    0x7FEF_FFFF_FFFF_FFFF,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x7FF8_0000_0000_0000,
    0x7FF0_0000_0000_0001,
    0x4009_21FB_5444_2D18,
    0x7E37_E43C_8800_759C,
    0x01A5_6E1F_C2F8_F359,
    0x4340_0000_0000_0001,
    0x3FB9_9999_9999_999A,
    0xC000_0000_0000_0000,
    0x41EF_FFFF_FFE0_0000,
];
const U64_EDGES: [u64; 12] = [
    0,
    1,
    0xFFFF_FFFF_FFFF_FFFF,
    0x7FFF_FFFF_FFFF_FFFF,
    0x8000_0000_0000_0000,
    0xFFFF_FFFF,
    0x1_0000_0000,
    63,
    64,
    65,
    0x0123_4567_89AB_CDEF,
    0xFFFF_FFFF_0000_0000,
];

pub(crate) fn edges(t: Ty) -> usize {
    match t {
        Ty::F32 => F32_EDGES.len(),
        Ty::F16 => F16_EDGES.len(),
        Ty::PkF16 => F16_EDGES.len() * 2,
        Ty::U16 => U16_EDGES.len(),
        Ty::PkU16 => U16_EDGES.len() * 2,
        Ty::F64 => F64_EDGES.len(),
        Ty::U64 | Ty::I64 => U64_EDGES.len(),
        _ => U32_EDGES.len(),
    }
}

/// Edge case `i` of type `t`, with `garbage` in bits a 16-bit operand does not use.
pub(crate) fn edge(t: Ty, i: usize, garbage: u64) -> u64 {
    let g16 = garbage as u32 & 0xFFFF_0000;
    match t {
        Ty::F32 => F32_EDGES[i] as u64,
        Ty::F16 => (F16_EDGES[i] as u32 | g16) as u64,
        Ty::U16 => (U16_EDGES[i] as u32 | g16) as u64,
        Ty::PkF16 => {
            let n = F16_EDGES.len();
            (F16_EDGES[i % n] as u64) | (F16_EDGES[(i * 7 + 3) % n] as u64) << 16
        }
        Ty::PkU16 => {
            let n = U16_EDGES.len();
            (U16_EDGES[i % n] as u64) | (U16_EDGES[(i * 5 + 1) % n] as u64) << 16
        }
        Ty::F64 => F64_EDGES[i],
        Ty::U64 | Ty::I64 => U64_EDGES[i],
        _ => U32_EDGES[i] as u64,
    }
}

/// A random value of type `t`: random bits half the time, otherwise a
/// value in a range where arithmetic is interesting.
pub(crate) fn random(t: Ty, rng: &mut Rng) -> u64 {
    let bits = rng.next();
    if bits & 1 == 0 {
        return match t.dwords(true) {
            2 => rng.next(),
            _ => rng.next() & 0xFFFF_FFFF,
        };
    }
    let small = (rng.next() % 2001) as f64 - 1000.0 + (rng.next() % 1000) as f64 / 1000.0;
    match t {
        Ty::F32 => (small as f32).to_bits() as u64,
        Ty::F64 => small.to_bits(),
        Ty::F16 | Ty::PkF16 => {
            let h = |r: u64| (0x3000 + (r % 0x1800)) | (r >> 20 & 1) << 15;
            h(rng.next()) | h(rng.next()) << 16
        }
        Ty::U16 | Ty::PkU16 => (rng.next() % 300) | ((rng.next() % 300) << 16),
        Ty::U64 | Ty::I64 => rng.next() % 200,
        _ => rng.next() % 100,
    }
}

/// What a test program is and how its records are laid out.
pub(crate) struct Test<'a> {
    pub opcode: &'a str,
    pub variant: String,
    pub text: String,
    pub code: Vec<u32>,
    pub mode: Mode,
    pub groups: u32,
    pub per_wave: bool,
    pub in_words: u32,
    pub out_names: &'a [&'a str],
    pub dst: Dst,
    /// LDS the program uses.
    pub lds_bytes: u32,
    /// Every record's input words.
    pub inputs: Vec<u32>,
}

/// A case running `t.code` at [`PROGRAM`] over wave64 groups, with
/// `s[0:1]` pointing at the inputs, `s[2:3]` at the outputs and `s4` the group.
pub(crate) fn build(t: Test) -> Case {
    let mut compute = vec![0u32; COMPUTE_COUNT];
    let mut set = |reg: usize, v: u32| compute[reg - COMPUTE_FIRST] = v;
    set(0x207, 64);
    set(0x208, 1);
    set(0x209, 1);
    set(0x20C, (PROGRAM >> 8) as u32);
    set(0x20D, (PROGRAM >> 40) as u32);
    set(0x212, t.mode.rsrc1());
    const USER_SGPRS: u32 = 4;
    const TGID_X_EN: u32 = 1 << 7;
    const LDS_GRANULE: u32 = 512;
    set(
        0x213,
        USER_SGPRS << 1 | TGID_X_EN | t.lds_bytes.div_ceil(LDS_GRANULE) << 15,
    );
    set(0x240, INPUTS as u32);
    set(0x241, (INPUTS >> 32) as u32);
    set(0x242, OUTPUTS as u32);
    set(0x243, (OUTPUTS >> 32) as u32);
    let mut case = Case {
        opcode: t.opcode.to_string(),
        variant: t.variant,
        text: t.text,
        program: PROGRAM,
        compute,
        groups: t.groups,
        per_wave: t.per_wave,
        inputs: INPUTS,
        outputs: OUTPUTS,
        in_words: t.in_words,
        out_words: t.out_names.len() as u32,
        out_names: t.out_names.iter().map(|s| s.to_string()).collect(),
        dst: t.dst,
        ..Case::default()
    };
    let bytes: Vec<u8> = t.code.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_pages(&mut case.input, PROGRAM, &bytes);
    let bytes: Vec<u8> = t.inputs.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_pages(&mut case.input, INPUTS, &bytes);
    case
}

/// The destination type of a result of type `t`, for classifying differences.
pub(crate) fn dst(t: Option<&Ty>) -> Dst {
    match t {
        Some(Ty::F16) => Dst::F16,
        Some(Ty::PkF16) => Dst::PkF16,
        Some(Ty::F32) => Dst::F32,
        Some(Ty::F64) => Dst::F64,
        _ => Dst::Bits,
    }
}

/// Every generated test.
pub fn all_cases() -> Vec<Case> {
    let mut out = valu::cases();
    out.extend(salu::cases());
    out.extend(smem::cases());
    out.extend(lanes::cases());
    out.extend(ds::cases());
    out.extend(mem::cases());
    out.extend(mimg::cases());
    out.extend(draw::cases());
    out.extend(flow::cases());
    out.extend(scratch::cases());
    out
}
