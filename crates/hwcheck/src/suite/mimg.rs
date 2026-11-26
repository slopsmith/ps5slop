//! Image tests: every MIMG load, store, atomic, sample, gather, LOD query and
//! size query over linear textures in memory, so both sides read the same
//! bytes. Each opcode runs on every image dimension it allows with an RGBA8
//! texture and a bilinear wrapping sampler; the commonest opcodes also run
//! over a sweep of formats and of samplers. Stores and atomics write
//! texels no other lane writes and are read back with `image_load`.

use gfx10::decode::decode;
use gfx10::opinfo::{Fmt, OpInfo, all};
use gfx10::print::{Syntax, print_checked};

use super::formats::FORMATS;
use super::{BASE_MODE, INPUTS, Rng, Test, build};
use crate::{Case, Dst, write_pages};

const GROUPS: u32 = 16;
const IN_WORDS: usize = 16;
const OUT_NAMES: [&str; 8] = ["d0", "d1", "d2", "d3", "-", "-", "-", "-"];
/// The texture's texels.
pub(super) const TEXTURE: u64 = 0x60_0000;
/// The image descriptor (32 bytes), then the sampler (16 bytes).
pub const DESC: u64 = INPUTS + 0x8_0000;

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
const PROLOGUE: [u32; 23] = [
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
    0xF40C_0A00,
    0xFA08_0000, // s_load_dwordx8 s[40:47], s[0:1], 0x80000
    0xF408_0C00,
    0xFA08_0020, // s_load_dwordx4 s[48:51], s[0:1], 0x80020
    0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
    0x7E28_030F, // v_mov_b32_e32 v20, v15
    0x7E2A_030F, // v_mov_b32_e32 v21, v15
    0x7E2C_030F, // v_mov_b32_e32 v22, v15
    0x7E2E_030F, // v_mov_b32_e32 v23, v15
];
const WAIT_STORES: u32 = 0xBBFD_0000;
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

const VADDR: u32 = 1;
const VDATA_STORE: u32 = 9;
pub(super) const VDST: u32 = 20;
pub(super) const SRSRC: u32 = 40;
pub(super) const SSAMP: u32 = 48;

/// Image dimensions: name, MIMG DIM, descriptor TYPE, size (width, height,
/// depth or layers), coordinates, whether it is an array (last coordinate a slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim {
    pub name: &'static str,
    pub dim: u32,
    pub kind: u32,
    pub size: (u32, u32, u32),
    pub coords: usize,
    pub derivs: usize,
}

pub const DIMS: [Dim; 6] = [
    Dim {
        name: "1d",
        dim: 0,
        kind: 8,
        size: (256, 1, 1),
        coords: 1,
        derivs: 2,
    },
    Dim {
        name: "2d",
        dim: 1,
        kind: 9,
        size: (64, 64, 1),
        coords: 2,
        derivs: 4,
    },
    Dim {
        name: "3d",
        dim: 2,
        kind: 10,
        size: (64, 16, 8),
        coords: 3,
        derivs: 6,
    },
    Dim {
        name: "cube",
        dim: 3,
        kind: 11,
        size: (64, 64, 6),
        coords: 3,
        derivs: 4,
    },
    Dim {
        name: "1d array",
        dim: 4,
        kind: 12,
        size: (256, 1, 4),
        coords: 2,
        derivs: 2,
    },
    Dim {
        name: "2d array",
        dim: 5,
        kind: 13,
        size: (64, 64, 4),
        coords: 3,
        derivs: 4,
    },
];

/// Samplers: name, CLAMP_X/Y/Z, XY_MAG filter, XY_MIN filter, MIP_FILTER,
/// DEPTH_COMPARE_FUNC, BORDER_COLOR_TYPE, FORCE_UNNORMALIZED, LOD_BIAS (s5.8).
pub type Sampler = (&'static str, u32, u32, u32, u32, u32, u32, bool, u32);

pub const SAMPLERS: [Sampler; 14] = [
    ("bilinear wrap", 0, 1, 1, 0, 0, 0, false, 0),
    ("point wrap", 0, 0, 0, 0, 0, 0, false, 0),
    ("point mag bilinear min", 0, 0, 1, 0, 0, 0, false, 0),
    ("bilinear mirror", 1, 1, 1, 0, 0, 0, false, 0),
    ("bilinear clamp to edge", 2, 1, 1, 0, 0, 0, false, 0),
    ("point mirror once", 3, 0, 0, 0, 0, 0, false, 0),
    (
        "bilinear clamp to transparent black border",
        6,
        1,
        1,
        0,
        0,
        0,
        false,
        0,
    ),
    (
        "bilinear clamp to opaque black border",
        6,
        1,
        1,
        0,
        0,
        1,
        false,
        0,
    ),
    (
        "bilinear clamp to opaque white border",
        6,
        1,
        1,
        0,
        0,
        2,
        false,
        0,
    ),
    ("point unnormalized clamp", 2, 0, 0, 0, 0, 0, true, 0),
    ("bilinear compare less", 2, 1, 1, 0, 1, 0, false, 0),
    ("point compare greater or equal", 2, 0, 0, 0, 6, 0, false, 0),
    ("trilinear wrap", 0, 1, 1, 2, 0, 0, false, 0),
    (
        "point mips wrap, lod bias +1.5",
        0,
        0,
        0,
        1,
        0,
        0,
        false,
        0x180,
    ),
];

/// What one address operand holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arg {
    Offset,
    Bias,
    Compare,
    Deriv,
    /// Two derivatives as 16-bit floats in one dword (`_g16`).
    Deriv16,
    Coord(usize),
    Slice,
    Lod,
    Clamp,
    Mip,
}

/// The address operands of `name` on `dim`, in the order the ISA lays them out.
pub(super) fn args(name: &str, dim: Dim) -> Vec<Arg> {
    let sample = name.starts_with("image_sample") || name.starts_with("image_gather");
    let mut out = Vec::new();
    if name.ends_with("_o") || name.contains("_o_") {
        out.push(Arg::Offset);
    }
    if name.contains("_b_") || name.ends_with("_b") || name.contains("_b_cl") {
        out.push(Arg::Bias);
    }
    if sample && (name.contains("_c_") || name.ends_with("_c")) {
        out.push(Arg::Compare);
    }
    if name.contains("_d_") || name.ends_with("_d") || name.contains("_cd") {
        let packed = name.contains("g16");
        let n = if packed {
            dim.derivs.div_ceil(2)
        } else {
            dim.derivs
        };
        out.extend(std::iter::repeat_n(
            if packed { Arg::Deriv16 } else { Arg::Deriv },
            n,
        ));
    }
    let array = matches!(dim.dim, 4 | 5);
    let plain = dim.coords - array as usize;
    out.extend((0..plain).map(Arg::Coord));
    if array {
        out.push(Arg::Slice);
    }
    if name.contains("_l_") || name.ends_with("_l") {
        out.push(Arg::Lod);
    }
    if name.contains("_cl") {
        out.push(Arg::Clamp);
    }
    if name.contains("_mip") {
        out.push(Arg::Mip);
    }
    if name == "image_get_resinfo" {
        return vec![Arg::Mip];
    }
    out
}

fn is_store(name: &str) -> bool {
    name.starts_with("image_store")
}

fn is_atomic(name: &str) -> bool {
    name.starts_with("image_atomic")
}

/// Why an opcode has no generated image test, or `None` if it has.
pub fn unsupported(op: &OpInfo) -> Option<&'static str> {
    if op.fmt != Fmt::Mimg {
        return Some("not an image instruction");
    }
    if op.name.contains("bvh") {
        return Some("ray tracing, tested by `bvh_cases`");
    }
    if op.name.contains("msaa") {
        return Some("MSAA loads need tiled multisampled surfaces");
    }
    None
}

/// The dims `name` runs on.
fn dims(name: &str) -> Vec<Dim> {
    let gather = name.starts_with("image_gather");
    DIMS.iter()
        .copied()
        .filter(|d| !gather || matches!(d.dim, 1 | 3 | 5))
        .collect()
}

fn encode(op: &OpInfo, dim: Dim, dmask: u32, unorm: bool) -> [u32; 2] {
    encode_at(op, dim, dmask, unorm, VADDR)
}

/// The instruction with its address operands from `vaddr` on; it reads
/// [`SRSRC`] and [`SSAMP`] and writes from [`VDST`] on.
pub(super) fn encode_at(op: &OpInfo, dim: Dim, dmask: u32, unorm: bool, vaddr: u32) -> [u32; 2] {
    let c = op.code as u32;
    let glc = is_atomic(op.name) as u32;
    let vdata = if is_store(op.name) || is_atomic(op.name) {
        VDATA_STORE
    } else {
        VDST
    };
    [
        0xF000_0000
            | (c & 0x7F) << 18
            | (c >> 7) & 1
            | dim.dim << 3
            | dmask << 8
            | (unorm as u32) << 12
            | glc << 13,
        (uses_sampler(op.name) as u32 * (SSAMP >> 2)) << 21
            | (SRSRC >> 2) << 16
            | vdata << 8
            | vaddr,
    ]
}

fn uses_sampler(name: &str) -> bool {
    name.starts_with("image_sample") || name.starts_with("image_gather") || name == "image_get_lod"
}

/// An `image_load` of the same texel, to read back what a store wrote.
fn load_back(dim: Dim, mip: bool) -> [u32; 2] {
    let code = if mip { 1 } else { 0 };
    [
        0xF000_0000 | code << 18 | dim.dim << 3 | 0xF << 8,
        (SRSRC >> 2) << 16 | VDST << 8 | VADDR,
    ]
}

/// Bytes of a linear texture of `bits` per texel (per 4x4 block when
/// `block`), rows padded to 256 bytes, every level. Each slice of an array
/// and each depth slice of a 3D image holds its own mip chain.
fn texture_bytes(dim: Dim, bits: u32, block: bool, levels: u32) -> u32 {
    let (w, h, d) = dim.size;
    let mut total = 0;
    for l in 0..levels {
        let (lw, lh) = ((w >> l).max(1), (h >> l).max(1));
        let (cols, rows) = if block {
            (lw.div_ceil(4), lh.div_ceil(4))
        } else {
            (lw, lh)
        };
        let row = (cols * bits / 8).next_multiple_of(256);
        total += row * rows;
    }
    total * d
}

fn descriptor(dim: Dim, format: u32, levels: u32) -> [u32; 8] {
    descriptor_at(TEXTURE, dim, format, levels)
}

/// The image descriptor of a linear texture at `address`.
pub(super) fn descriptor_at(address: u64, dim: Dim, format: u32, levels: u32) -> [u32; 8] {
    let (w, h, d) = dim.size;
    let depth = match dim.dim {
        2 => d - 1,
        3..=5 => d - 1,
        _ => 0,
    };
    const RESOURCE_LEVEL: u32 = 1 << 31;
    const XYZW: u32 = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    [
        (address >> 8) as u32,
        (address >> 40) as u32 & 0xFF | format << 20 | ((w - 1) & 3) << 30,
        (w - 1) >> 2 | (h - 1) << 14 | RESOURCE_LEVEL,
        XYZW | (levels - 1) << 16 | dim.kind << 28,
        depth,
        (levels - 1) << 4,
        0,
        0,
    ]
}

fn sampler(s: usize) -> [u32; 4] {
    sampler_words(SAMPLERS[s], 0)
}

/// The sampler descriptor of `s` with `MAX_ANISO_RATIO` `aniso` (0 for
/// 1x, up to 4 for 16x); an anisotropic sampler filters as Mesa's
/// `ac_build_sampler_descriptor` sets it up.
pub(super) fn sampler_words(s: Sampler, aniso: u32) -> [u32; 4] {
    let (_, clamp, mag, min, mip, compare, border, unnormalized, bias) = s;
    const ANISO_FILTER: u32 = 2;
    let (mag, min, perf_mip) = if aniso > 0 {
        (mag | ANISO_FILTER, min | ANISO_FILTER, aniso + 6)
    } else {
        (mag, min, 0)
    };
    [
        clamp
            | clamp << 3
            | clamp << 6
            | aniso << 9
            | compare << 12
            | (unnormalized as u32) << 15
            | aniso >> 1 << 16
            | aniso << 21,
        15 << 8 << 12 | perf_mip << 24,
        bias & 0x3FFF | mag << 20 | min << 22 | mip << 26,
        border << 30,
    ]
}

/// Texel bytes: random bits, or for float formats random values in [-2, 2]
/// with a few infinities and NaNs, as 32-bit or (`half`) 16-bit floats.
fn texels(bytes: u32, float: bool, half: bool, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes as usize);
    while out.len() < bytes as usize {
        if !float {
            out.extend((rng.next() as u32).to_le_bytes());
            continue;
        }
        let v = (rng.next() % 4001) as f32 / 1000.0 - 2.0;
        let v = match rng.below(64) {
            0 => f32::INFINITY,
            1 => f32::NAN,
            _ => v,
        };
        if half {
            out.extend(f16_bits(v).to_le_bytes());
        } else {
            out.extend(v.to_bits().to_le_bytes());
        }
    }
    out.truncate(bytes as usize);
    out
}

pub(super) fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = (b >> 16 & 0x8000) as u16;
    if v.is_nan() {
        return sign | 0x7E00;
    }
    let exp = (b >> 23 & 0xFF) as i32 - 127 + 15;
    if exp >= 31 {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        return sign;
    }
    sign | (exp as u16) << 10 | (b >> 13 & 0x3FF) as u16
}

/// `dim`, widened until every lane can write a texel of its own.
fn room_for_every_lane(mut dim: Dim) -> Dim {
    let (w, h, d) = dim.size;
    dim.size.0 = w.max((GROUPS * 64).div_ceil(h * d));
    dim
}

/// How a test's float coordinates are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coords {
    /// Normalized, random per lane.
    Random,
    /// In texels (the instruction's UNORM), random per lane.
    Texels,
    /// Each quad of lanes a small 2x2 block (top left, top right, bottom
    /// left, bottom right), its spacing sweeping the LOD from -1.5 to 3.5
    /// over the quads; the bottom right lane is off the grid, so per-lane
    /// and per-quad derivatives differ.
    Quads,
}

/// Every lane's record: its address operands, store data, the destination's start.
fn records(
    args: &[Arg],
    dim: Dim,
    unique: bool,
    integer: bool,
    coords: Coords,
    seed: &str,
) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let lanes = (GROUPS * 64) as usize;
    let mut out = Vec::with_capacity(lanes * IN_WORDS);
    let (w, h, d) = dim.size;
    let extent = [w, h, d];
    let mut quad = [0.0f32; 3];
    for i in 0..lanes {
        if coords == Coords::Quads && i % 4 == 0 {
            let lod = -1.5 + 5.0 * (i / 4) as f32 / (lanes / 4) as f32;
            quad = [
                rng.below(1000) as f32 / 1000.0,
                rng.below(1000) as f32 / 1000.0,
                lod.exp2(),
            ];
        }
        let mut rec = [0u32; IN_WORDS];
        for (k, &a) in args.iter().enumerate() {
            let f = |v: f32| v.to_bits();
            rec[k] = match a {
                Arg::Offset => rng.next() as u32 & 0x3F3F3F,
                Arg::Bias => f((rng.next() % 4001) as f32 / 1000.0 - 2.0),
                Arg::Compare => f((rng.next() % 1001) as f32 / 1000.0),
                Arg::Deriv => f((rng.next() % 2001) as f32 / 10000.0 - 0.1),
                Arg::Deriv16 => {
                    let h = |r: u64| f16_bits((r % 2001) as f32 / 10000.0 - 0.1) as u32;
                    h(rng.next()) | h(rng.next()) << 16
                }
                Arg::Coord(c) if unique => match c {
                    0 => (i as u32) % w,
                    1 => (i as u32 / w) % h,
                    _ => (i as u32 / w) % d,
                },
                Arg::Coord(c) if integer => {
                    (rng.below(extent[c].max(1) as usize + 4) as i32 - 2) as u32
                }
                Arg::Coord(c) if coords == Coords::Quads && c < 2 => {
                    let lane = i % 4;
                    let step = quad[2] / extent[c] as f32;
                    let (on, skew) = if c == 0 {
                        (lane & 1, 0.5)
                    } else {
                        (lane >> 1, -0.3)
                    };
                    let skew = if lane == 3 { skew } else { 0.0 };
                    f(quad[c] + step * (on as f32 + skew))
                }
                Arg::Coord(c) => {
                    let v = (rng.next() % 1501) as f32 / 1000.0 - 0.25;
                    f(if coords == Coords::Texels {
                        v * extent[c] as f32
                    } else {
                        v
                    })
                }
                Arg::Slice if unique => (i as u32 / w) % d,
                Arg::Slice if integer => rng.below(d as usize + 1) as u32,
                Arg::Slice => f(rng.below(d as usize + 1) as f32 + 0.3),
                Arg::Lod => f((rng.next() % 801) as f32 / 100.0 - 1.0),
                Arg::Clamp => f((rng.next() % 801) as f32 / 100.0),
                Arg::Mip => rng.below(3) as u32,
            };
        }
        for word in rec[8..12].iter_mut().skip(args.len().saturating_sub(8)) {
            *word = rng.next() as u32;
        }
        rec[14] = rng.next() as u32;
        out.extend_from_slice(&rec);
    }
    out
}

/// One image test.
#[allow(clippy::too_many_arguments)]
fn case(
    op: &OpInfo,
    dim: Dim,
    format: usize,
    samp: usize,
    dmask: u32,
    coords: Coords,
) -> Option<Case> {
    let unorm = coords == Coords::Texels;
    let args = args(op.name, dim);
    if args.len() > 13 {
        return None;
    }
    let inst = encode(op, dim, dmask, unorm);
    let decoded = decode(&inst).ok()?;
    if decoded.op.name != op.name {
        return None;
    }
    let text = print_checked(&decoded, Syntax { wave64: true })?;
    let (fname, fcode, bits, kind) = FORMATS[format];
    let float = fname.contains("FLOAT") && !fname.starts_with("BC");
    let levels = if SAMPLERS[samp].4 != 0 || op.name.contains("mip") {
        3
    } else {
        1
    };
    let store = is_store(op.name);
    let writes = store || is_atomic(op.name);
    let dim = if writes {
        room_for_every_lane(dim)
    } else {
        dim
    };
    let mut code = PROLOGUE.to_vec();
    code.extend(inst);
    if store {
        code.push(WAIT_STORES);
        code.extend(load_back(dim, op.name.contains("mip")));
    }
    code.extend(EPILOGUE);
    let mut variant = format!("{} {fname} {}", dim.name, SAMPLERS[samp].0);
    if dmask != 0xF {
        variant += &format!(" dmask {dmask:#x}");
    }
    match coords {
        Coords::Random => {}
        Coords::Texels => variant += " unorm",
        Coords::Quads => variant += " quad derivatives",
    }
    let seed = format!("{} {variant}", op.name);
    let integer = !uses_sampler(op.name);
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
        dst: if kind == "UINT" || kind == "SINT" {
            Dst::Bits
        } else {
            Dst::F32
        },
        lds_bytes: 0,
        inputs: records(&args, dim, writes, integer, coords, &seed),
    });
    let mut table: Vec<u32> = descriptor(dim, fcode, levels).to_vec();
    table.extend(sampler(samp));
    let table: Vec<u8> = table.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_pages(&mut case.input, DESC, &table);
    let bytes = texture_bytes(dim, bits, fname.starts_with("BC"), levels);
    let mut rng = Rng::new(&format!("texture {fname} {}", dim.name));
    let half = fname.starts_with("16");
    write_pages(
        &mut case.input,
        TEXTURE,
        &texels(bytes, float, half, &mut rng),
    );
    if writes {
        case.checked = (TEXTURE, bytes);
    }
    Some(case)
}

fn format(name: &str) -> usize {
    FORMATS
        .iter()
        .position(|f| f.0 == name)
        .expect("in Mesa's format table")
}

/// Opcodes that run over every format and every sampler.
const SWEEP: [&str; 11] = [
    "image_load",
    "image_load_mip",
    "image_load_pck",
    "image_store",
    "image_store_mip",
    "image_store_pck",
    "image_sample",
    "image_sample_l",
    "image_sample_c_lz",
    "image_gather4",
    "image_gather4_c",
];

/// Opcodes that also run over quads whose LOD is known ([`Coords::Quads`]).
const QUADS: [&str; 3] = ["image_sample", "image_gather4", "image_get_lod"];
const TRILINEAR: usize = 12;
const POINT_MIPS: usize = 13;

/// Opcodes that compare against a reference also run over the formats
/// depth images use, with a comparing sampler.
const COMPARED: [&str; 2] = ["image_sample_c_lz", "image_gather4_c"];
const COMPARE_LESS: usize = 10;
const DEPTH_FORMATS: [&str; 8] = [
    "32_FLOAT",
    "16_UNORM",
    "16_FLOAT",
    "8_UNORM",
    "32_32_FLOAT",
    "8_24_UNORM",
    "24_8_UNORM",
    "X24_8_32_FLOAT",
];

/// Every image test, in opcode table order.
pub fn cases() -> Vec<Case> {
    let mut out = bvh_cases();
    out.extend(image_cases());
    out
}

/// Where the ray tracing tests' nodes start, 128 bytes each.
const NODES: u64 = TEXTURE;
const NODE_COUNT: u64 = 64;

/// A ray tracing node's pointer as the instruction takes it: the address
/// over 8, the node type in the low bits (radv's `bvh_defines.h`).
fn node_pointer(index: u64) -> u64 {
    const TRIANGLE: u64 = 0;
    const BOX32: u64 = 5;
    let kind = if index.is_multiple_of(2) {
        BOX32
    } else {
        TRIANGLE
    };
    (NODES + index * 128) >> 3 | kind
}

/// Box nodes at even indices and triangle nodes at odd ones, laid out as
/// radv's `radv_bvh_box32_node` and `radv_bvh_triangle_node`.
fn nodes(rng: &mut Rng) -> Vec<u8> {
    let mut f = |lo: f32, hi: f32| lo + (rng.next() % 10_000) as f32 / 10_000.0 * (hi - lo);
    let mut words = Vec::new();
    for i in 0..NODE_COUNT {
        let mut node = [0u32; 32];
        if i.is_multiple_of(2) {
            for c in 0..4 {
                let child = (i + 1 + c * 7) % NODE_COUNT;
                node[c as usize] = if c == 3 && i % 4 == 0 {
                    u32::MAX
                } else {
                    node_pointer(child) as u32
                };
                let (x, y, z) = (f(-1.0, 1.0), f(-1.0, 1.0), f(-1.0, 1.0));
                let e = f(0.05, 0.8);
                let b = 4 + c as usize * 6;
                node[b..b + 6]
                    .copy_from_slice(&[x - e, y - e, z - e, x + e, y + e, z + e].map(f32::to_bits));
            }
        } else {
            for word in &mut node[..9] {
                *word = f(-1.0, 1.0).to_bits();
            }
            node[12] = i as u32 * 3;
            node[13] = i as u32;
            node[15] = i as u32;
        }
        words.extend(node);
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn uniform(rng: &mut Rng, lo: f32, hi: f32) -> f32 {
    lo + (rng.next() % 10_000) as f32 / 10_000.0 * (hi - lo)
}

/// Ray tracing tests: every lane casts a random ray (some with zero
/// direction components) at one of [`NODE_COUNT`] random nodes.
fn bvh_cases() -> Vec<Case> {
    const BOX_SORT: u32 = 1 << 31;
    const RANGE: u64 = (1 << 42) - 1;
    const RETURN_IJ: u32 = 1 << 24;
    const BVH_TYPE: u32 = 1 << 31;
    let descriptor = [
        0,
        BOX_SORT,
        RANGE as u32,
        (RANGE >> 32) as u32 | RETURN_IJ | BVH_TYPE,
    ];
    let mut out = Vec::new();
    for op in all().iter().filter(|o| o.name.contains("bvh")) {
        let wide = op.name.contains("bvh64");
        let c = op.code as u32;
        const DMASK_XYZW: u32 = 0xF << 8;
        const UNORM: u32 = 1 << 12;
        const R128: u32 = 1 << 15;
        let inst = [
            0xF000_0000 | (c & 0x7F) << 18 | (c >> 7) & 1 | DMASK_XYZW | UNORM | R128,
            (SRSRC >> 2) << 16 | VDST << 8 | VADDR,
        ];
        let Some(decoded) = decode(&inst).ok().filter(|d| d.op.name == op.name) else {
            continue;
        };
        let Some(text) = print_checked(&decoded, Syntax { wave64: true }) else {
            continue;
        };
        let mut code = PROLOGUE.to_vec();
        code.extend(inst);
        code.extend(EPILOGUE);
        let seed = format!("{} rays", op.name);
        let mut rng = Rng::new(&seed);
        let lanes = (GROUPS * 64) as usize;
        let mut records = Vec::with_capacity(lanes * IN_WORDS);
        for _ in 0..lanes {
            let node = node_pointer(rng.below(NODE_COUNT as usize) as u64);
            let origin = [0; 3].map(|_| uniform(&mut rng, -2.0, 2.0));
            let mut dir = [0; 3].map(|_| uniform(&mut rng, -1.0, 1.0));
            if rng.below(8) == 0 {
                dir[rng.below(3)] = 0.0;
            }
            let mut rec = vec![node as u32];
            if wide {
                rec.push((node >> 32) as u32);
            }
            rec.push(uniform(&mut rng, 0.5, 10.0).to_bits());
            rec.extend(origin.map(f32::to_bits));
            rec.extend(dir.map(f32::to_bits));
            rec.extend(dir.map(|d| (1.0 / d).to_bits()));
            rec.resize(IN_WORDS, 0);
            rec[14] = rng.next() as u32;
            records.extend(rec);
        }
        let mut case = build(Test {
            opcode: op.name,
            variant: "random rays at box and triangle nodes".into(),
            text,
            code,
            mode: BASE_MODE,
            groups: GROUPS,
            per_wave: false,
            in_words: IN_WORDS as u32,
            out_names: &OUT_NAMES,
            dst: Dst::F32,
            lds_bytes: 0,
            inputs: records,
        });
        let table: Vec<u8> = descriptor.iter().flat_map(|w| w.to_le_bytes()).collect();
        write_pages(&mut case.input, DESC, &table);
        write_pages(&mut case.input, NODES, &nodes(&mut Rng::new("bvh nodes")));
        out.push(case);
    }
    out
}

fn image_cases() -> Vec<Case> {
    let rgba8 = format("8_8_8_8_UNORM");
    let r32_uint = format("32_UINT");
    let r32_float = format("32_FLOAT");
    let rgba16_float = format("16_16_16_16_FLOAT");
    let two_d = DIMS[1];
    let mut out = Vec::new();
    for op in all().iter().filter(|op| unsupported(op).is_none()) {
        let atomic = op.name.starts_with("image_atomic");
        let float_atomic = ["fcmp", "fmin", "fmax"].iter().any(|f| op.name.contains(f));
        let base = match (atomic, float_atomic) {
            (true, true) => r32_float,
            (true, false) => r32_uint,
            _ => rgba8,
        };
        let dmask = if op.name.starts_with("image_gather") {
            1
        } else if atomic {
            if op.name.contains("cmpswap") { 3 } else { 1 }
        } else {
            0xF
        };
        for dim in dims(op.name) {
            out.extend(case(op, dim, base, 0, dmask, Coords::Random));
        }
        if !atomic {
            for f in [r32_float, rgba16_float] {
                out.extend(case(op, two_d, f, 0, dmask, Coords::Random));
            }
        }
        if QUADS.contains(&op.name) {
            for s in [TRILINEAR, POINT_MIPS] {
                out.extend(case(op, two_d, base, s, dmask, Coords::Quads));
            }
        }
        if COMPARED.contains(&op.name) {
            for f in DEPTH_FORMATS {
                out.extend(case(
                    op,
                    two_d,
                    format(f),
                    COMPARE_LESS,
                    dmask,
                    Coords::Random,
                ));
            }
        }
        if SWEEP.contains(&op.name) {
            let writes = op.name.starts_with("image_store");
            let storable = |f: usize| !(writes && FORMATS[f].0.starts_with("BC"));
            for f in (0..FORMATS.len()).filter(|&f| f != base && storable(f)) {
                out.extend(case(op, two_d, f, 0, dmask, Coords::Random));
            }
            for s in 1..SAMPLERS.len() {
                out.extend(case(op, two_d, base, s, dmask, Coords::Random));
            }
            out.extend(case(op, two_d, base, 0, 0x5, Coords::Random));
            // Unnormalized coordinates need a clamping sampler (Vulkan's rule).
            let clamp = if uses_sampler(op.name) { 4 } else { 0 };
            out.extend(case(op, two_d, base, clamp, dmask, Coords::Texels));
        }
    }
    out
}

/// A sampling-heavy program for measuring the cost of sampling, not a
/// hardware test: every lane of `groups` waves takes `samples` bilinear
/// `image_sample_l`s of an RGBA8 texture, moving `u` between them, and sums
/// them so none is dead.
pub fn sampling_heavy(samples: u32, groups: u32) -> Case {
    let op = gfx10::opinfo::by_name("image_sample_l").expect("in the opcode table");
    let dim = DIMS[1];
    let add = gfx10::opinfo::by_name("v_add_f32")
        .expect("in the opcode table")
        .code as u32;
    let vop2 = |vdst: u32, src0: u32, vsrc1: u32| add << 25 | vdst << 17 | vsrc1 << 9 | src0;
    const LITERAL: u32 = 0xFF;
    const VGPR: u32 = 256;
    let mut code = PROLOGUE.to_vec();
    for _ in 0..samples {
        code.extend(encode(op, dim, 0xF, false));
        code.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
        code.extend([vop2(VADDR, LITERAL, VADDR), (1.0f32 / 97.0).to_bits()]);
        for c in 0..4 {
            code.push(vop2(9 + c, VGPR + VDST + c, 9 + c));
        }
    }
    for c in 0..4 {
        code.push(0x7E00_0000 | (VDST + c) << 17 | 1 << 9 | (VGPR + 9 + c));
    }
    code.extend(EPILOGUE);
    let rgba8 = format("8_8_8_8_UNORM");
    let (_, fcode, bits, _) = FORMATS[rgba8];
    let args = args(op.name, dim);
    let mut case = build(Test {
        opcode: op.name,
        variant: format!("{samples} samples per lane"),
        text: String::new(),
        code,
        mode: BASE_MODE,
        groups,
        per_wave: false,
        in_words: IN_WORDS as u32,
        out_names: &OUT_NAMES,
        dst: Dst::F32,
        lds_bytes: 0,
        inputs: records(&args, dim, false, false, Coords::Random, "sampling heavy")
            .into_iter()
            .cycle()
            .take((groups * 64) as usize * IN_WORDS)
            .collect(),
    });
    let mut table: Vec<u32> = descriptor(dim, fcode, 1).to_vec();
    table.extend(sampler(0));
    let table: Vec<u8> = table.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_pages(&mut case.input, DESC, &table);
    let bytes = texture_bytes(dim, bits, false, 1);
    write_pages(
        &mut case.input,
        TEXTURE,
        &texels(bytes, false, false, &mut Rng::new("sampling heavy")),
    );
    case
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::opinfo::by_name;

    #[test]
    fn image_tests_decode_to_what_they_test() {
        let text = |n: &str, d: usize| {
            case(by_name(n).unwrap(), DIMS[d], 0, 0, 0xF, Coords::Random)
                .unwrap()
                .text
        };
        assert_eq!(
            text("image_sample", 1),
            "image_sample v[20:23], v[1:2], s[40:47], s[48:51] dmask:0xf dim:SQ_RSRC_IMG_2D"
        );
        assert_eq!(
            text("image_sample_c_d_o", 1),
            "image_sample_c_d_o v[20:23], v[1:8], s[40:47], s[48:51] dmask:0xf dim:SQ_RSRC_IMG_2D"
        );
        assert_eq!(
            text("image_load_mip", 5),
            "image_load_mip v[20:23], v[1:4], s[40:47] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY"
        );
        let ops: std::collections::BTreeSet<String> =
            cases().into_iter().map(|c| c.opcode).collect();
        assert!(
            ops.len() > 80,
            "only {} image opcodes would be tested",
            ops.len()
        );
    }
}
