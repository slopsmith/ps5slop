//! Draw tests: a pass-through vertex program, a pixel program built for the
//! case and a linear render target, set up register by register as a PS5
//! command buffer would, and compared pixel by pixel. The hardware also gets
//! what a driver sends around its command buffers (CONTEXT_CONTROL,
//! CLEAR_STATE, compute-unit masks before; a cache flush after).
//!
//! Suites: every colour format and number type through both export paths,
//! the component swaps, the interpolation modes and pixel inputs, every
//! blend factor and function, logic ops and write masks, and raster state
//! (culling, fill rule, viewport sign, scissors, points and lines), and
//! texture sampling (see [`texture`]).

use gfx10::opinfo::by_name;

use super::mimg::f16_bits;
use super::{BASE_MODE, INPUTS, OUTPUTS, PROGRAM, Rng};
use crate::{Case, write_pages};

pub(super) mod texture;

/// The pixel program's address; the vertex program sits at [`PROGRAM`].
const PIXEL_PROGRAM: u64 = PROGRAM + 0x1000;
/// The render target.
const TARGET: u64 = OUTPUTS;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
/// Loads vertex `v0`'s position and one attribute (8 floats at `s[0:1]`)
/// and exports them.
const VERTEX: [u32; 11] = [
    0x3402_0085, // v_lshlrev_b32_e32 v1, 5, v0
    0xDC38_8000,
    0x0200_0001, // global_load_dwordx4 v[2:5], v1, s[0:1]
    0xDC38_8010,
    0x0600_0001, // global_load_dwordx4 v[6:9], v1, s[0:1] offset:16
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xF800_08CF,
    0x0504_0302, // exp pos0 v2, v3, v4, v5 done
    0xF800_020F,
    0x0908_0706, // exp param0 v6, v7, v8, v9
    0xBF81_0000, // s_endpgm
];

// Assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`.
/// A mesh program of two waves, one group per input point: wave 0's
/// first lane hands the group index (v5) to the group through LDS, every
/// thread stores its index in LDS and after `s_barrier` reads the one 64
/// threads away (the other wave's), and thread `t` of the first 8 exports
/// vertex `8 * group + (that index & 7)` of the table at `s[8:9]`
/// (position and one attribute). Wave 0 allocates 8 vertices and 4
/// triangles (t, t + 1, t + 2), the last of them null.
const MESH: [u32; 67] = [
    0xBEFE_04C1, // s_mov_b64 exec, -1
    0x9384_FF03,
    0x0004_0018, // s_bfe_u32 s4, s3, 0x40018
    0xD765_0000,
    0x0001_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
    0xD766_0000,
    0x0002_00C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
    0xD746_0000,
    0x0401_0C04, // v_lshl_add_u32 v0, s4, 6, v0
    0xBF07_8004, // s_cmp_lg_u32 s4, 0
    0xBF85_0006, // s_cbranch_scc1 6
    0xBEFE_0481, // s_mov_b64 exec, 1
    0x7E02_02FF,
    0x0000_0200, // v_mov_b32_e32 v1, 0x200
    0xD834_0000,
    0x0000_0501, // ds_write_b32 v1, v5
    0xBEFE_04C1, // s_mov_b64 exec, -1
    0x3404_0082, // v_lshlrev_b32_e32 v2, 2, v0
    0xD834_0000,
    0x0000_0002, // ds_write_b32 v2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0x3A06_00C0, // v_xor_b32_e32 v3, 64, v0
    0x3406_0682, // v_lshlrev_b32_e32 v3, 2, v3
    0xD8D8_0000,
    0x0400_0003, // ds_read_b32 v4, v3
    0x7E02_02FF,
    0x0000_0200, // v_mov_b32_e32 v1, 0x200
    0xD8D8_0000,
    0x0100_0001, // ds_read_b32 v1, v1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x3608_0887, // v_and_b32_e32 v4, 7, v4
    0xBF07_8004, // s_cmp_lg_u32 s4, 0
    0xBF85_0003, // s_cbranch_scc1 3
    0xBEFC_03FF,
    0x0000_4008, // s_mov_b32 m0, 0x4008
    0xBF90_0009, // s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
    0x7DA8_0084, // v_cmpx_gt_u32_e32 4, v0
    0xBF88_000C, // s_cbranch_execz 12
    0x4A0C_0081, // v_add_nc_u32_e32 v6, 1, v0
    0x4A0E_0082, // v_add_nc_u32_e32 v7, 2, v0
    0x340C_0C8A, // v_lshlrev_b32_e32 v6, 10, v6
    0x340E_0E94, // v_lshlrev_b32_e32 v7, 20, v7
    0xD772_0006,
    0x041E_0D00, // v_or3_b32 v6, v0, v6, v7
    0x7E12_7081, // v_bfrev_b32_e32 v9, 1
    0x3812_0D09, // v_or_b32_e32 v9, v9, v6
    0x7D84_0083, // v_cmp_eq_u32_e32 vcc, 3, v0
    0x020C_1306, // v_cndmask_b32_e32 v6, v6, v9, vcc
    0xF800_0941,
    0x0000_0006, // exp prim v6, off, off, off done
    0xBEFE_04C1, // s_mov_b64 exec, -1
    0x7DA8_0088, // v_cmpx_gt_u32_e32 8, v0
    0xBF88_000C, // s_cbranch_execz 12
    0xD746_000A,
    0x0411_0701, // v_lshl_add_u32 v10, v1, 3, v4
    0x3414_1485, // v_lshlrev_b32_e32 v10, 5, v10
    0xDC38_8000,
    0x0C08_000A, // global_load_dwordx4 v[12:15], v10, s[8:9]
    0xDC38_8010,
    0x1008_000A, // global_load_dwordx4 v[16:19], v10, s[8:9] offset:16
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xF800_08CF,
    0x0F0E_0D0C, // exp pos0 v12, v13, v14, v15 done
    0xF800_020F,
    0x1312_1110, // exp param0 v16, v17, v18, v19
    0xBF81_0000, // s_endpgm
];
/// Threads in a [`MESH`] group, and the vertices and primitives it may export.
const MESH_THREADS: u32 = 128;
const MESH_VERTICES: usize = 8;
const MESH_PRIMITIVES: u32 = 4;

const CONTEXT_CONTROL: u32 = 0x28;
const CLEAR_STATE: u32 = 0x12;
const SET_CONTEXT_REG: u32 = 0x69;
const SET_SH_REG: u32 = 0x76;
const SET_UCONFIG_REG: u32 = 0x79;
const NUM_INSTANCES: u32 = 0x2F;
const DRAW_INDEX_AUTO: u32 = 0x2D;

fn packet(op: u32, body: &[u32]) -> Vec<u32> {
    let mut p = vec![(3 << 30) | ((body.len() as u32 - 1) << 16) | (op << 8)];
    p.extend_from_slice(body);
    p
}

/// A register write by the register's byte address.
fn set(reg: u32, values: &[u32]) -> Vec<u32> {
    let (op, base) = match reg {
        0x2_8000..=0x2_8FFF => (SET_CONTEXT_REG, 0x2_8000),
        0xB000..=0xBFFF => (SET_SH_REG, 0xB000),
        _ => (SET_UCONFIG_REG, 0x3_0000),
    };
    let mut body = vec![(reg - base) / 4];
    body.extend_from_slice(values);
    packet(op, &body)
}

/// What only the hardware gets first: the state Mesa's drivers send once per
/// command stream on GFX10 (`ac_init_graphics_preamble_state`), and for
/// `mesh` the compute units geometry waves may use, as radv sets them.
fn preamble(mesh: bool) -> Vec<u32> {
    const LOAD_ENABLES: u32 = 1 << 31;
    const SHADOW_ENABLES: u32 = 1 << 31;
    const CU_EN_ALL: u32 = 0xFFFF;
    const WAVE_LIMIT: u32 = 0x3F << 16;
    [
        packet(CONTEXT_CONTROL, &[LOAD_ENABLES, SHADOW_ENABLES]),
        packet(CLEAR_STATE, &[0]),
        set(0xB01C, &[CU_EN_ALL | WAVE_LIMIT]),
        set(0xB004, &[CU_EN_ALL]),
        set(0xB118, &[CU_EN_ALL | WAVE_LIMIT]),
        set(0xB104, &[CU_EN_ALL]),
        set(0x2_8708, &[1]),
        if mesh {
            [
                set(0xB21C, &[CU_EN_ALL | WAVE_LIMIT]),
                set(0xB204, &[CU_EN_ALL]),
            ]
            .concat()
        } else {
            Vec::new()
        },
    ]
    .concat()
}

// An end-of-pipe colour cache flush with an L2 write-back; the GCR bits
// as Linux's `nvd.h` names them.
fn postamble() -> Vec<u32> {
    const RELEASE_MEM: u32 = 0x49;
    const CACHE_FLUSH_AND_INV_TS_EVENT: u32 = 20;
    const EVENT_INDEX_EOP: u32 = 5 << 8;
    const GLM_WB: u32 = 1 << 12;
    const GLM_INV: u32 = 1 << 13;
    const GLV_INV: u32 = 1 << 14;
    const GL1_INV: u32 = 1 << 15;
    const GL2_INV: u32 = 1 << 20;
    const GL2_WB: u32 = 1 << 21;
    let event = CACHE_FLUSH_AND_INV_TS_EVENT
        | EVENT_INDEX_EOP
        | GLM_WB
        | GLM_INV
        | GLV_INV
        | GL1_INV
        | GL2_INV
        | GL2_WB;
    packet(RELEASE_MEM, &[event, 0, 0, 0, 0, 0, 0])
}

/// How a pixel program gets attribute 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    PerspCenter,
    PerspCentroid,
    PerspSample,
    LinearCenter,
    LinearCentroid,
    /// `SPI_PS_INPUT_CNTL` FLAT_SHADE: the provoking vertex's value.
    Flat,
    /// `v_interp_mov_f32` of vertex P0.
    MovP0,
    /// The pixel's position (`POS_X`, `POS_Y`, `POS_Z`, `POS_W` inputs).
    FragCoord,
    /// Front face, the ancillary word, sample coverage and the fixed-point position.
    Misc,
    /// Attribute 0 as packed halves (`FP16_INTERP_MODE`), the low ones read
    /// with `v_interp_p1ll_f16` and `v_interp_p2_f16`, widened to f32.
    F16Low,
    /// The same with the high halves.
    F16High,
    /// The high halves with `v_interp_p1lv_f16`, P0 from `v_interp_mov_f32`.
    F16HighP0,
}

/// How the pixel program turns its four floats into the export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// Four 32-bit floats (`SPI_SHADER_32_ABGR`).
    F32,
    /// Scaled to [0, 300) and converted to unsigned integers, 32 bits each.
    U32,
    /// Scaled to [-150, 150) and converted to signed integers, 32 bits each.
    I32,
    /// Packed to half floats (`FP16_ABGR`, `compr`).
    F16,
    /// Packed to 16-bit unorm (`UNORM16_ABGR`).
    Unorm16,
    /// Packed to 16-bit snorm (`SNORM16_ABGR`).
    Snorm16,
    /// Packed to 16-bit unsigned integers (`UINT16_ABGR`).
    Uint16,
    /// Packed to 16-bit signed integers (`SINT16_ABGR`).
    Sint16,
}

impl Interp {
    /// The opcodes a pixel program for this input uses to get its values,
    /// and `exp`, which every draw tests.
    fn covers(self) -> Vec<&'static str> {
        match self {
            Interp::FragCoord | Interp::Misc => vec!["exp"],
            Interp::MovP0 => vec!["exp", "v_interp_mov_f32"],
            Interp::F16Low | Interp::F16High => vec!["exp", "v_interp_p1ll_f16", "v_interp_p2_f16"],
            Interp::F16HighP0 => vec!["exp", "v_interp_p1lv_f16", "v_interp_p2_f16"],
            _ => vec!["exp", "v_interp_p1_f32", "v_interp_p2_f32"],
        }
    }
}

impl Out {
    /// `SPI_SHADER_COL_FORMAT` for MRT0.
    fn export_format(self) -> u32 {
        match self {
            Out::F32 | Out::U32 | Out::I32 => 9,
            Out::F16 => 4,
            Out::Unorm16 => 5,
            Out::Snorm16 => 6,
            Out::Uint16 => 7,
            Out::Sint16 => 8,
        }
    }
}

/// VGPRs each `SPI_PS_INPUT_ADDR` bit takes, from bit 0.
const INPUT_VGPRS: [u32; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];

fn input_vgpr(addr: u32, bit: u32) -> u32 {
    (0..bit)
        .filter(|&b| addr >> b & 1 != 0)
        .map(|b| INPUT_VGPRS[b as usize])
        .sum()
}

fn vop1(name: &str, vdst: u32, src0: u32) -> u32 {
    let op = by_name(name).expect("in the opcode table").code as u32;
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

fn vop2(name: &str, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    let op = by_name(name).expect("in the opcode table").code as u32;
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

fn vop3(name: &str, vdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    let op = by_name(name).expect("in the opcode table").code as u32;
    [0xD400_0000 | op << 16 | vdst, src1 << 9 | src0]
}

fn vintrp(op: u32, vdst: u32, vsrc: u32, chan: u32) -> u32 {
    0xC800_0000 | vdst << 18 | op << 16 | chan << 8 | vsrc
}

/// `exp mrtN` of `srcs` with the valid mask, `done` on the last export;
/// packed (`compr`) exports name each register twice and carry two of them.
fn export(target: u32, compr: bool, done: bool, srcs: [u32; 4]) -> [u32; 2] {
    let regs = if compr {
        srcs[2] << 8 | srcs[0]
    } else {
        srcs[3] << 24 | srcs[2] << 16 | srcs[1] << 8 | srcs[0]
    };
    [
        0xF800_0000 | 1 << 12 | (done as u32) << 11 | (compr as u32) << 10 | target << 4 | 0xF,
        regs,
    ]
}

const LITERAL: u32 = 0xFF;
const VGPR: u32 = 256;

/// A pixel program: `SPI_PS_INPUT_ENA`/`ADDR`, `SPI_PS_INPUT_CNTL_0`, its code.
pub struct Pixel {
    pub inputs: u32,
    pub cntl: u32,
    pub code: Vec<u32>,
    pub dual: bool,
}

/// The pixel program for `interp` and `out`: four floats into v20..v23,
/// converted, exported to MRT0, and with `dual` also to MRT1 with its halves
/// swapped, the second source of dual-source blending.
pub fn pixel(interp: Interp, out: Out, dual: bool) -> Pixel {
    const PERSP_SAMPLE: u32 = 0;
    const PERSP_CENTER: u32 = 1;
    const PERSP_CENTROID: u32 = 2;
    const LINEAR_CENTER: u32 = 5;
    const LINEAR_CENTROID: u32 = 6;
    let (bary, extra): (u32, &[u32]) = match interp {
        Interp::PerspCenter
        | Interp::Flat
        | Interp::MovP0
        | Interp::F16Low
        | Interp::F16High
        | Interp::F16HighP0 => (PERSP_CENTER, &[]),
        Interp::PerspCentroid => (PERSP_CENTROID, &[]),
        Interp::PerspSample => (PERSP_SAMPLE, &[]),
        Interp::LinearCenter => (LINEAR_CENTER, &[]),
        Interp::LinearCentroid => (LINEAR_CENTROID, &[]),
        Interp::FragCoord => (PERSP_CENTER, &[8, 9, 10, 11]),
        Interp::Misc => (PERSP_CENTER, &[12, 13, 14, 15]),
    };
    let inputs = extra.iter().fold(1 << bary, |m, b| m | 1 << b);
    let i = input_vgpr(inputs, bary);
    let mut code = vec![0xBEFC_0300]; // s_mov_b32 m0, s0
    match interp {
        Interp::FragCoord | Interp::Misc => {
            for (k, &b) in extra.iter().enumerate() {
                code.push(vop1(
                    "v_mov_b32",
                    20 + k as u32,
                    VGPR + input_vgpr(inputs, b),
                ));
            }
        }
        Interp::MovP0 => {
            const P0: u32 = 2;
            for c in 0..4 {
                code.push(vintrp(2, 20 + c, P0, c));
            }
        }
        Interp::F16Low | Interp::F16High | Interp::F16HighP0 => {
            const P0: u32 = 2;
            let high = (interp != Interp::F16Low) as u32;
            let interp_f16 = |name: &str, vdst: u32, src: [u32; 3]| {
                let op = by_name(name).expect("in the opcode table").code as u32;
                [
                    0xD400_0000 | op << 16 | vdst,
                    src[2] << 18 | src[1] << 9 | src[0],
                ]
            };
            for c in 0..4 {
                let attr = c << 6 | high << 8;
                let d = 20 + c;
                if interp == Interp::F16HighP0 {
                    code.push(vintrp(2, 26, P0, c));
                    code.extend(interp_f16(
                        "v_interp_p1lv_f16",
                        d,
                        [attr, VGPR + i, VGPR + 26],
                    ));
                } else {
                    code.extend(interp_f16("v_interp_p1ll_f16", d, [attr, VGPR + i, 0]));
                }
                code.extend(interp_f16(
                    "v_interp_p2_f16",
                    d,
                    [attr, VGPR + i + 1, VGPR + d],
                ));
                code.push(vop1("v_cvt_f32_f16", d, VGPR + d));
            }
        }
        _ => {
            for c in 0..4 {
                code.push(vintrp(0, 20 + c, i, c));
                code.push(vintrp(1, 20 + c, i + 1, c));
            }
        }
    }
    let scale = |code: &mut Vec<u32>, k: f32, sub: f32| {
        for c in 0..4 {
            code.extend([vop2("v_mul_f32", 20 + c, LITERAL, 20 + c), k.to_bits()]);
            if sub != 0.0 {
                code.extend([vop2("v_subrev_f32", 20 + c, LITERAL, 20 + c), sub.to_bits()]);
            }
        }
    };
    let (compr, srcs) = match out {
        Out::F32 => (false, [20, 21, 22, 23]),
        Out::U32 | Out::I32 => {
            let signed = out == Out::I32;
            scale(&mut code, 300.0, if signed { 150.0 } else { 0.0 });
            let cvt = if signed {
                "v_cvt_i32_f32"
            } else {
                "v_cvt_u32_f32"
            };
            for c in 0..4 {
                code.push(vop1(cvt, 20 + c, VGPR + 20 + c));
            }
            (false, [20, 21, 22, 23])
        }
        Out::F16 => {
            code.push(vop2("v_cvt_pkrtz_f16_f32", 24, VGPR + 20, 21));
            code.push(vop2("v_cvt_pkrtz_f16_f32", 25, VGPR + 22, 23));
            (true, [24, 24, 25, 25])
        }
        Out::Unorm16 | Out::Snorm16 => {
            let name = if out == Out::Unorm16 {
                "v_cvt_pknorm_u16_f32"
            } else {
                "v_cvt_pknorm_i16_f32"
            };
            code.extend(vop3(name, 24, VGPR + 20, VGPR + 21));
            code.extend(vop3(name, 25, VGPR + 22, VGPR + 23));
            (true, [24, 24, 25, 25])
        }
        Out::Uint16 | Out::Sint16 => {
            let signed = out == Out::Sint16;
            scale(&mut code, 300.0, if signed { 150.0 } else { 0.0 });
            let (cvt, pack) = if signed {
                ("v_cvt_i32_f32", "v_cvt_pk_i16_i32")
            } else {
                ("v_cvt_u32_f32", "v_cvt_pk_u16_u32")
            };
            for c in 0..4 {
                code.push(vop1(cvt, 20 + c, VGPR + 20 + c));
            }
            code.extend(vop3(pack, 24, VGPR + 20, VGPR + 21));
            code.extend(vop3(pack, 25, VGPR + 22, VGPR + 23));
            (true, [24, 24, 25, 25])
        }
    };
    if dual {
        code.extend(export(0, compr, false, srcs));
        let [a, b, c, d] = srcs;
        code.extend(export(1, compr, true, [c, d, a, b]));
    } else {
        code.extend(export(0, compr, true, srcs));
    }
    code.push(0xBF81_0000);
    Pixel {
        inputs,
        cntl: match interp {
            Interp::Flat => FLAT_SHADE,
            Interp::F16Low | Interp::F16High | Interp::F16HighP0 => FP16_ATTRS,
            _ => 0,
        },
        code,
        dual,
    }
}

/// A pixel program that exports `t * scale[c] + offset[c]` in channel `c`,
/// where `t` is the pixel's index (x + 64 y) in the target: exact values
/// that step through a conversion's whole curve.
fn ramp_pixel(scale: [f32; 4], offset: [f32; 4]) -> Pixel {
    const PERSP_CENTER: u32 = 1;
    let inputs = 1 << PERSP_CENTER | 1 << 8 | 1 << 9;
    let (x, y) = (input_vgpr(inputs, 8), input_vgpr(inputs, 9));
    let mut code = vec![
        vop2("v_mul_f32", 24, LITERAL, y),
        (WIDTH as f32).to_bits(),
        vop2("v_add_f32", 24, VGPR + x, 24),
        vop2("v_subrev_f32", 24, LITERAL, 24),
        (WIDTH as f32 / 2.0 + 0.5).to_bits(),
    ];
    for c in 0..4 {
        code.extend([
            vop2("v_mul_f32", 20 + c, LITERAL, 24),
            scale[c as usize].to_bits(),
        ]);
        code.extend([
            vop2("v_add_f32", 20 + c, LITERAL, 20 + c),
            offset[c as usize].to_bits(),
        ]);
    }
    code.extend(export(0, false, true, [20, 21, 22, 23]));
    code.push(0xBF81_0000);
    Pixel {
        inputs,
        cntl: 0,
        code,
        dual: false,
    }
}

/// Two triangles over the whole target.
fn full_screen() -> Vec<[f32; 8]> {
    let v = |x: f32, y: f32| [x, y, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0];
    vec![
        v(-1.0, -1.0),
        v(1.0, -1.0),
        v(-1.0, 1.0),
        v(1.0, -1.0),
        v(1.0, 1.0),
        v(-1.0, 1.0),
    ]
}

/// One draw test's settings.
pub struct Draw {
    pub name: String,
    pub variant: String,
    pub pixel: Pixel,
    pub out: Out,
    /// `CB_COLOR0_INFO` FORMAT, NUMBER_TYPE and COMP_SWAP, and bytes per pixel.
    pub format: (u32, u32, u32, u32),
    /// Other `CB_COLOR0_INFO` bits.
    pub info: u32,
    /// `CB_BLEND0_CONTROL`.
    pub blend: u32,
    /// `CB_BLEND_RED`..`ALPHA`.
    pub blend_color: [f32; 4],
    /// `CB_COLOR_CONTROL` ROP3.
    pub rop3: u32,
    /// `CB_TARGET_MASK` for MRT0.
    pub write_mask: u32,
    /// `PA_SU_SC_MODE_CNTL`.
    pub raster: u32,
    /// `VGT_PRIMITIVE_TYPE`.
    pub primitive: u32,
    /// `PA_SU_LINE_CNTL` WIDTH: half the line width in 12.4 fixed point.
    pub line_half_width: u32,
    /// A negative `PA_CL_VPORT_YSCALE`, as PS5 titles set it.
    pub flip_y: bool,
    /// `PA_SC_GENERIC_SCISSOR` top left and bottom right.
    pub scissor: (u32, u32, u32, u32),
    /// Vertices: position (x, y, z, w) and attribute 0.
    pub vertices: Vec<[f32; 8]>,
    /// The target's dwords before the draw; random when `None`.
    pub fill: Option<Vec<u32>>,
    /// Opcodes the draw exercises: its exports and interpolation.
    pub covers: Vec<&'static str>,
    /// The pixel program's user SGPRs (`SPI_SHADER_USER_DATA_PS_*`).
    pub user_data: Vec<u32>,
    /// Memory the draw reads besides its programs and vertices.
    pub memory: Vec<(u64, Vec<u8>)>,
    /// The vertices come from [`MESH`], one group per [`MESH_VERTICES`] of
    /// them, launched as this says.
    pub mesh: Option<MeshLaunch>,
}

/// How the hardware is told to run [`MESH`] groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshLaunch {
    /// `GS_FAST_LAUNCH`, as radv launches mesh shaders on GFX10.3.
    Fast,
    /// GS subgroups of one ES vertex with amplified primitives, as PS5
    /// titles set it up (SharpEmu-mac's reading of Black Myth: Wukong's
    /// registers: `VGT_SHADER_STAGES_EN` 0x2030, `ES_VERTS_PER_SUBGRP` 1).
    Amplified,
}

impl Draw {
    fn new(name: &str, variant: String, interp: Interp, out: Out, vertices: Vec<[f32; 8]>) -> Self {
        Draw {
            name: name.into(),
            variant,
            pixel: pixel(interp, out, false),
            out,
            format: (COLOR_8_8_8_8, NUMBER_UNORM, 0, 4),
            info: 0,
            blend: 0,
            blend_color: [0.25, 0.5, 0.75, 0.6],
            rop3: 0xCC,
            write_mask: 0xF,
            raster: 0,
            primitive: TRILIST,
            line_half_width: 4 << 4,
            flip_y: true,
            scissor: (0, 0, WIDTH, HEIGHT),
            vertices,
            fill: None,
            covers: interp.covers(),
            user_data: Vec::new(),
            memory: Vec::new(),
            mesh: None,
        }
    }
}

const TRILIST: u32 = 4;
/// `SPI_PS_INPUT_CNTL_n.FLAT_SHADE`.
const FLAT_SHADE: u32 = 1 << 10;
/// `SPI_PS_INPUT_CNTL_n` FP16_INTERP_MODE, ATTR0_VALID and ATTR1_VALID, as
/// radv sets them for 16-bit inputs.
const FP16_ATTRS: u32 = 1 << 19 | 1 << 24 | 1 << 25;
const COLOR_8_8_8_8: u32 = 10;
const NUMBER_UNORM: u32 = 0;

const BLEND_ENABLE: u32 = 1 << 30;

// As radeonsi's `si_create_blend_state`: a hint that does not fit the blend
// scrambles the target on RDNA2.
fn sx_blend_opt(blend: u32, dual: bool) -> u32 {
    const PRESERVE_ALL: u32 = 1;
    const BLEND_DISABLED: u32 = 6;
    if blend & BLEND_ENABLE == 0 {
        return PRESERVE_ALL
            | PRESERVE_ALL << 4
            | BLEND_DISABLED << 8
            | PRESERVE_ALL << 16
            | PRESERVE_ALL << 20
            | BLEND_DISABLED << 24;
    }
    if dual {
        return 0;
    }
    let separate = blend & 1 << 29 != 0;
    let field = |at: u32| blend >> at & 0x1F;
    let (src, func, dst) = (field(0), field(5) & 7, field(8));
    let (asrc, afunc, adst) = if separate {
        (field(16), field(21) & 7, field(24))
    } else {
        (src, func, dst)
    };
    let half = |src: u32, func: u32, dst: u32, alpha: bool| {
        let src_opt = opt_factor(src, alpha);
        let uses_dst = matches!(src, 6..=10);
        let dst_opt = if src == SRC_ALPHA_SATURATE && matches!(dst, 0 | 4 | 10) {
            NONE_IGNORE_A0
        } else if uses_dst {
            NONE_IGNORE_NONE
        } else {
            opt_factor(dst, alpha)
        };
        src_opt | dst_opt << 4 | (func + 1) << 8
    };
    half(src, func, dst, false) | half(asrc, afunc, adst, true) << 16
}

const SRC_ALPHA_SATURATE: u32 = 10;
const NONE_IGNORE_A0: u32 = 6;
const NONE_IGNORE_NONE: u32 = 7;

// radeonsi's `si_translate_blend_opt_factor`.
fn opt_factor(factor: u32, alpha: bool) -> u32 {
    const NONE_IGNORE_ALL: u32 = 0;
    const ALL_IGNORE_NONE: u32 = 1;
    const C1_IGNORE_C0: u32 = 2;
    const C0_IGNORE_C1: u32 = 3;
    const A1_IGNORE_A0: u32 = 4;
    const A0_IGNORE_A1: u32 = 5;
    match factor {
        0 => NONE_IGNORE_ALL,
        1 => ALL_IGNORE_NONE,
        2 if alpha => A1_IGNORE_A0,
        2 => C1_IGNORE_C0,
        3 if alpha => A0_IGNORE_A1,
        3 => C0_IGNORE_C1,
        4 => A1_IGNORE_A0,
        5 => A0_IGNORE_A1,
        SRC_ALPHA_SATURATE if alpha => ALL_IGNORE_NONE,
        SRC_ALPHA_SATURATE => NONE_IGNORE_A0,
        _ => NONE_IGNORE_NONE,
    }
}

fn rsrc1(vgprs: u32) -> u32 {
    const DX10_CLAMP: u32 = 1 << 21;
    (vgprs / 4).saturating_sub(1) | (BASE_MODE.float_mode as u32) << 12 | DX10_CLAMP
}

/// The command buffer of `d`: every register the draw reads, then the draw.
fn stream(d: &Draw, vertex_buffer: u64) -> Vec<u32> {
    let (format, number, swap, _) = d.format;
    const CB_NORMAL: u32 = 1 << 4;
    const WINDOW_OFFSET_DISABLE: u32 = 1 << 31;
    const MAX_PRIMGRP_IN_WAVE: u32 = 2 << 15;
    const AUTO_INDEX: u32 = 2;
    const VTX_W0_FMT: u32 = 1 << 10;
    let br = |w: u32, h: u32| w | h << 16;
    let half = |n: u32| (n as f32 / 2.0).to_bits();
    let yscale = if d.flip_y {
        -(HEIGHT as f32) / 2.0
    } else {
        HEIGHT as f32 / 2.0
    };
    let (sx0, sy0, sx1, sy1) = d.scissor;
    let dual = d.pixel.dual && d.blend & BLEND_ENABLE != 0;
    let blend_controls = [d.blend, BLEND_ENABLE];
    let vertex_regs = if let Some(launch) = d.mesh {
        mesh_regs(vertex_buffer, launch)
    } else {
        [
            set(0xB120, &[(PROGRAM >> 8) as u32, (PROGRAM >> 40) as u32]),
            set(0xB128, &[rsrc1(16), 2 << 1]),
            set(
                0xB130,
                &[vertex_buffer as u32, (vertex_buffer >> 32) as u32],
            ),
            set(0x2_8B54, &[MAX_PRIMGRP_IN_WAVE]),
            set(0x3_096C, &[128 | 256 << 9]),
            set(0x3_0908, &[d.primitive, 0]),
        ]
        .concat()
    };
    let count = if d.mesh.is_some() {
        (d.vertices.len() / MESH_VERTICES) as u32
    } else {
        d.vertices.len() as u32
    };
    [
        vertex_regs,
        set(
            0xB020,
            &[(PIXEL_PROGRAM >> 8) as u32, (PIXEL_PROGRAM >> 40) as u32],
        ),
        set(0xB028, &[rsrc1(32), (d.user_data.len() as u32) << 1]),
        if d.user_data.is_empty() {
            Vec::new()
        } else {
            set(0xB030, &d.user_data)
        },
        set(0x2_8A40, &[0]),
        set(0x3_0924, &[0, 0, 0]),
        set(0x3_0964, &[u32::MAX, 0]),
        set(0x3_097C, &[0]),
        set(0x3_0988, &[0]),
        set(0x2_86C4, &[0]),
        set(0x2_870C, &[4]),
        set(0x2_881C, &[0]),
        set(0x2_86CC, &[d.pixel.inputs, d.pixel.inputs]),
        set(0x2_86D8, &[1]),
        set(0x2_8644, &[d.pixel.cntl]),
        // SPI_INTERP_CONTROL_0.FLAT_SHADE_ENA, which radeonsi and radv
        // always set: without it the hardware ignores FLAT_SHADE. Written
        // only where an input is flat, which leaves the other draws as they
        // were measured.
        if d.pixel.cntl & FLAT_SHADE != 0 {
            set(0x2_86D4, &[1])
        } else {
            Vec::new()
        },
        set(0x2_8710, &[0]),
        set(
            0x2_8714,
            &[d.out.export_format() * if d.pixel.dual { 0x11 } else { 1 }],
        ),
        set(
            0x2_8238,
            &[d.write_mask, if d.pixel.dual { 0xFF } else { 0xF }],
        ),
        set(0x2_8808, &[CB_NORMAL | d.rop3 << 16 | dual as u32]),
        set(0x2_8780, &blend_controls[..1 + dual as usize]),
        set(0x2_8414, &d.blend_color.map(f32::to_bits)),
        set(0x2_8C60, &[(TARGET >> 8) as u32]),
        set(0x2_8E40, &[(TARGET >> 40) as u32]),
        set(
            0x2_8C6C,
            &[0, format << 2 | number << 8 | swap << 11 | d.info, 0],
        ),
        set(0x2_8EC0, &[(HEIGHT - 1) | (WIDTH - 1) << 14]),
        set(0x2_8EE0, &[1 << 24 | 1 << 27]),
        set(0x2_8C78, &[0]),
        set(0x2_8800, &[0]),
        set(0x2_8040, &[0, 0]),
        set(0x2_880C, &[0]),
        set(0x2_8204, &[WINDOW_OFFSET_DISABLE, br(WIDTH, HEIGHT)]),
        set(
            0x2_8240,
            &[WINDOW_OFFSET_DISABLE | br(sx0, sy0), br(sx1, sy1)],
        ),
        set(0x2_8250, &[0, br(WIDTH, HEIGHT)]),
        set(0x2_8030, &[0, br(16384, 16384)]),
        set(0x2_82D0, &[0, 1f32.to_bits()]),
        set(
            0x2_843C,
            &[
                half(WIDTH),
                half(WIDTH),
                yscale.to_bits(),
                half(HEIGHT),
                1f32.to_bits(),
                0,
            ],
        ),
        set(0x2_8818, &[0x3F | VTX_W0_FMT]),
        set(0x2_8810, &[1 << 19]),
        set(0x2_8814, &[d.raster]),
        set(0x2_8BE4, &[1 | 2 << 1 | 5 << 3]),
        set(0x2_8BE0, &[0]),
        set(0x2_8C38, &[u32::MAX, u32::MAX]),
        set(0x2_8C44, &[2 | 1 << 18]),
        set(0x2_8A00, &[8 << 16 | 8, 0xFFFF_0000]),
        set(0x2_8A08, &[d.line_half_width]),
        set(0x2_8754, &[0, 0, 0]),
        set(0x2_8760, &[sx_blend_opt(d.blend, dual); 8]),
        packet(NUM_INSTANCES, &[1]),
        packet(DRAW_INDEX_AUTO, &[count, AUTO_INDEX]),
    ]
    .concat()
}

/// The registers of a [`MESH`] draw, as radv sets them for a mesh shader on
/// GFX10.3 (`radv_precompute_registers_hw_ngg`, `radv_precompute_registers_hw_ms`,
/// `radv_emit_vgt_shader_config_gfx6`): a point list whose every point
/// launches one group of [`MESH_THREADS`] threads. `Amplified` only changes
/// the stages to the PS5's.
fn mesh_regs(vertex_buffer: u64, launch: MeshLaunch) -> Vec<u32> {
    const ES_EN_REAL: u32 = 2 << 3;
    const GS_EN: u32 = 1 << 5;
    const PRIMGEN_EN: u32 = 1 << 13;
    const MAX_PRIMGRP_IN_WAVE: u32 = 2 << 15;
    const GS_FAST_LAUNCH: u32 = 1 << 19;
    const GS_VGPR_COMP_CNT_1: u32 = 1 << 29;
    const LDS_GRANULES: u32 = 4;
    const POINTLIST: u32 = 1;
    const TRISTRIP: u32 = 2;
    [
        set(0xB320, &[(PROGRAM >> 8) as u32, (PROGRAM >> 40) as u32]),
        set(
            0xB228,
            &[rsrc1(24) | GS_VGPR_COMP_CNT_1, 2 << 1 | LDS_GRANULES << 19],
        ),
        set(
            0xB230,
            &[vertex_buffer as u32, (vertex_buffer >> 32) as u32],
        ),
        set(
            0x2_8B54,
            &[match launch {
                MeshLaunch::Fast => GS_EN | PRIMGEN_EN | MAX_PRIMGRP_IN_WAVE | GS_FAST_LAUNCH,
                MeshLaunch::Amplified => ES_EN_REAL | GS_EN | PRIMGEN_EN,
            }],
        ),
        set(0x2_87FC, &[MESH_VERTICES as u32]),
        set(0x2_8B4C, &[MESH_PRIMITIVES]),
        set(0x2_8B38, &[MESH_THREADS]),
        set(0x2_8A44, &[1 | 1 << 11 | 1 << 22]),
        set(0x2_8A6C, &[TRISTRIP]),
        set(0x2_8AAC, &[1]),
        set(0x3_096C, &[1 | 1 << 9]),
        set(0x3_0908, &[POINTLIST, 0]),
    ]
    .concat()
}

/// Rows of a linear render target start 256 bytes apart at least (Mesa's
/// `ac_surface`, confirmed by the hardware oracle).
const LINEAR_PITCH_ALIGN: u32 = 256;

/// The case for `d`, its target filled with random bytes first.
pub fn case(d: Draw) -> Case {
    let vertex_buffer = INPUTS;
    let bpp = d.format.3;
    let pitch = (WIDTH * bpp).next_multiple_of(LINEAR_PITCH_ALIGN);
    let mut case = Case {
        opcode: d.name.clone(),
        variant: d.variant.clone(),
        text: format!("{} vertices", d.vertices.len()),
        graphics: true,
        stream: stream(&d, vertex_buffer),
        preamble: preamble(d.mesh.is_some()),
        postamble: postamble(),
        target: (TARGET, WIDTH, HEIGHT, pitch),
        checked: (TARGET, pitch * HEIGHT),
        covers: d.covers.iter().map(|s| s.to_string()).collect(),
        ..Case::default()
    };
    let vertex: &[u32] = if d.mesh.is_some() { &MESH } else { &VERTEX };
    let code: Vec<u8> = vertex.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_pages(&mut case.input, PROGRAM, &code);
    let code: Vec<u8> = d.pixel.code.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_pages(&mut case.input, PIXEL_PROGRAM, &code);
    let verts: Vec<u8> = d
        .vertices
        .iter()
        .flatten()
        .flat_map(|f| f.to_bits().to_le_bytes())
        .collect();
    write_pages(&mut case.input, vertex_buffer, &verts);
    for (addr, bytes) in &d.memory {
        write_pages(&mut case.input, *addr, bytes);
    }
    let mut rng = Rng::new(&format!("target {} {}", d.name, d.variant));
    let finite = finite_mask(d.format.0, d.format.1);
    let fill: Vec<u8> = match &d.fill {
        Some(dwords) => dwords.iter().flat_map(|w| w.to_le_bytes()).collect(),
        None => (0..pitch * HEIGHT / 4)
            .flat_map(|_| (rng.next() as u32 & finite).to_le_bytes())
            .collect(),
    };
    write_pages(&mut case.input, TARGET, &fill);
    case
}

/// The bits a random fill of a target keeps so that every value is finite:
/// packed float formats lose each channel's top exponent bit. Other formats
/// keep every bit.
fn finite_mask(format: u32, number: u32) -> u32 {
    const FLOAT: u32 = 7;
    let top_bits = match (format, number) {
        (6, FLOAT) => 1 << 10 | 1 << 21 | 1 << 31,
        (7, FLOAT) => 1 << 9 | 1 << 20 | 1 << 31,
        _ => 0,
    };
    !top_bits
}

/// `n` random triangles in clip space, x and y up to `reach` times w, with
/// random attributes in [0, 1].
fn triangles_reaching(n: usize, seed: &str, reach: f32) -> Vec<[f32; 8]> {
    let mut rng = Rng::new(seed);
    let mut f = |lo: f32, hi: f32| lo + (rng.next() % 10_000) as f32 / 10_000.0 * (hi - lo);
    (0..n * 3)
        .map(|_| {
            let w = f(0.5, 2.0);
            [
                f(-reach, reach) * w,
                f(-reach, reach) * w,
                f(0.0, 1.0) * w,
                w,
                f(0.0, 1.0),
                f(0.0, 1.0),
                f(0.0, 1.0),
                f(0.0, 1.0),
            ]
        })
        .collect()
}

/// `n` random triangles inside the clip volume, so nothing is clipped.
fn triangles(n: usize, seed: &str) -> Vec<[f32; 8]> {
    triangles_reaching(n, seed, 1.0)
}

/// Colour formats: name, `CB_COLOR0_INFO` FORMAT, bytes per pixel.
const COLOR_FORMATS: [(&str, u32, u32); 18] = [
    ("8", 1, 1),
    ("16", 2, 2),
    ("8_8", 3, 2),
    ("32", 4, 4),
    ("16_16", 5, 4),
    ("10_11_11", 6, 4),
    ("11_11_10", 7, 4),
    ("10_10_10_2", 8, 4),
    ("2_10_10_10", 9, 4),
    ("8_8_8_8", 10, 4),
    ("32_32", 11, 8),
    ("16_16_16_16", 12, 8),
    ("32_32_32_32", 14, 16),
    ("5_6_5", 16, 2),
    ("1_5_5_5", 17, 2),
    ("5_5_5_1", 18, 2),
    ("4_4_4_4", 19, 2),
    ("5_9_9_9", 24, 4),
];

/// Whether RB+ takes 16-bit exports into `format` only as FP16 (or 16-bit
/// integers for integer number types), as Mesa's `ac_choose_spi_color_formats`
/// requires: UNORM16 and SNORM16 exports into these are invalid there.
fn needs_fp16_export(format: u32) -> bool {
    matches!(format, 1 | 3 | 6..=10 | 16..=19 | 24)
}

/// Number types: name, `NUMBER_TYPE`, whether it holds integers.
const NUMBER_TYPES: [(&str, u32, Option<bool>); 6] = [
    ("unorm", 0, None),
    ("snorm", 1, None),
    ("uint", 4, Some(false)),
    ("sint", 5, Some(true)),
    ("srgb", 6, None),
    ("float", 7, None),
];

/// Blend factors by `BlendOp` number.
const FACTORS: [&str; 21] = [
    "zero",
    "one",
    "src color",
    "one minus src color",
    "src alpha",
    "one minus src alpha",
    "dst alpha",
    "one minus dst alpha",
    "dst color",
    "one minus dst color",
    "src alpha saturate",
    "both src alpha",
    "both inv src alpha",
    "constant color",
    "one minus constant color",
    "src1 color",
    "inv src1 color",
    "src1 alpha",
    "inv src1 alpha",
    "constant alpha",
    "one minus constant alpha",
];

const FUNCTIONS: [&str; 5] = ["add", "src minus dst", "min", "max", "dst minus src"];

fn blend(src: u32, dst: u32, func: u32, alpha: Option<(u32, u32, u32)>) -> u32 {
    const ENABLE: u32 = 1 << 30;
    const SEPARATE_ALPHA: u32 = 1 << 29;
    let (asrc, adst, afunc) = alpha.unwrap_or((src, dst, func));
    src | (func << 5)
        | (dst << 8)
        | (asrc << 16)
        | (afunc << 21)
        | (adst << 24)
        | ENABLE
        | ((alpha.is_some() as u32) * SEPARATE_ALPHA)
}

/// Alphas of the blend sweep's bands.
const SWEEP_ALPHAS: u32 = 16;

/// The blend sweep's RGBA8 destination: pixel `i` of each band of four
/// rows holds `i` in every channel.
fn sweep_destination() -> Vec<u32> {
    (0..WIDTH * HEIGHT)
        .map(|p| (p % 256) * 0x0101_0101)
        .collect()
}

/// One rectangle per band of four rows, attribute 0 a grey of alpha
/// `band / 15`, so every pixel blends one destination value with one alpha.
fn sweep_bands() -> Vec<[f32; 8]> {
    let rows = HEIGHT / SWEEP_ALPHAS;
    (0..SWEEP_ALPHAS)
        .flat_map(|band| {
            let alpha = band as f32 / (SWEEP_ALPHAS - 1) as f32;
            let top = 1.0 - (band * rows) as f32 / (HEIGHT as f32 / 2.0);
            let bottom = 1.0 - ((band + 1) * rows) as f32 / (HEIGHT as f32 / 2.0);
            let v = |x: f32, y: f32| [x, y, 0.5, 1.0, 0.75, 0.75, 0.75, alpha];
            [
                v(-1.0, top),
                v(1.0, top),
                v(-1.0, bottom),
                v(1.0, top),
                v(1.0, bottom),
                v(-1.0, bottom),
            ]
        })
        .collect()
}

/// Whether `CB_BLEND0_CONTROL` uses a `src1` factor anywhere.
fn reads_src1(blend: u32) -> bool {
    [0, 8, 16, 24]
        .iter()
        .any(|&at| (15..=18).contains(&(blend >> at & 0x1F)))
}

// Pairings a driver would set up; the CB's results for others are not modelled.
fn offered(format: u32, number: u32, out: Out) -> bool {
    const UNORM: u32 = 0;
    const SNORM: u32 = 1;
    const SINT: u32 = 5;
    const SRGB: u32 = 6;
    const FLOAT: u32 = 7;
    let wide = matches!(out, Out::F32 | Out::U32 | Out::I32);
    match format {
        4 | 11 | 14 => wide && !matches!(number, UNORM | SNORM | SRGB),
        24 => number == FLOAT,
        16 | 19 => number != FLOAT,
        17 | 18 => !matches!(number, SNORM | SINT | FLOAT),
        _ => !(number == SRGB && out == Out::Unorm16),
    }
}

/// Every draw test.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let tris = |seed: &str| triangles(8, seed);
    let v = |x: f32, y: f32, r: f32, g: f32| [x, y, 0.5, 1.0, r, g, 0.25, 1.0];

    for (fname, format, bpp) in COLOR_FORMATS {
        for (nname, number, int) in NUMBER_TYPES {
            let outs: &[Out] = match int {
                Some(false) => &[Out::U32, Out::Uint16],
                Some(true) => &[Out::I32, Out::Sint16],
                None if number == 7 || needs_fp16_export(format) => &[Out::F32, Out::F16],
                None if number == 1 => &[Out::F32, Out::Snorm16],
                None => &[Out::F32, Out::Unorm16],
            };
            for &o in outs.iter().filter(|&&o| offered(format, number, o)) {
                let mut d = Draw::new(
                    "draw format",
                    format!("{fname} {nname} export {o:?}"),
                    Interp::PerspCenter,
                    o,
                    tris("formats"),
                );
                d.format = (format, number, 0, bpp);
                out.push(case(d));
            }
        }
    }
    for swap in 1..4 {
        let mut d = Draw::new(
            "draw format",
            format!("8_8_8_8 unorm comp swap {swap}"),
            Interp::PerspCenter,
            Out::F32,
            tris("formats"),
        );
        d.format.2 = swap;
        out.push(case(d));
    }

    for interp in [
        Interp::PerspCenter,
        Interp::PerspCentroid,
        Interp::PerspSample,
        Interp::LinearCenter,
        Interp::LinearCentroid,
        Interp::Flat,
        Interp::MovP0,
        Interp::FragCoord,
        Interp::Misc,
    ] {
        out.push(case(Draw::new(
            "draw input",
            format!("{interp:?}"),
            interp,
            Out::F32,
            tris("inputs"),
        )));
        let mut d = Draw::new(
            "draw input",
            format!("{interp:?} to 32_32_32_32 float"),
            interp,
            Out::F32,
            tris("inputs"),
        );
        d.format = (14, 7, 0, 16);
        out.push(case(d));
    }
    // Each attribute dword packs two halves in [0, 2).
    let packed = |v: Vec<[f32; 8]>| -> Vec<[f32; 8]> {
        v.into_iter()
            .map(|mut x| {
                let a = [x[4], x[5], x[6], x[7]];
                for c in 0..4 {
                    let (lo, hi) = (f16_bits(a[c]), f16_bits(2.0 * a[(c + 1) % 4]));
                    x[4 + c] = f32::from_bits(u32::from(hi) << 16 | u32::from(lo));
                }
                x
            })
            .collect()
    };
    for interp in [Interp::F16Low, Interp::F16High, Interp::F16HighP0] {
        let mut d = Draw::new(
            "draw input",
            format!("{interp:?} to 32_32_32_32 float"),
            interp,
            Out::F32,
            packed(tris("inputs")),
        );
        d.format = (14, 7, 0, 16);
        out.push(case(d));
    }
    for (name, half) in [("ramp", 0.0), ("ramp, half a step up", 0.5)] {
        let scale = [12, 16, 20, 12].map(|e: i32| 2f32.powi(-e));
        let mut d = Draw::new(
            "draw format",
            format!("8_8_8_8 srgb {name}"),
            Interp::FragCoord,
            Out::F32,
            full_screen(),
        );
        d.pixel = ramp_pixel(scale, scale.map(|k| k * half));
        d.format = (COLOR_8_8_8_8, 6, 0, 4);
        out.push(case(d));
    }
    for (suffix, format) in [
        ("", (COLOR_8_8_8_8, NUMBER_UNORM, 0, 4)),
        (" to 32_32_32_32 float", (14, 7, 0, 16)),
    ] {
        let mut d = Draw::new(
            "draw input",
            format!("MovP0 flat shaded{suffix}"),
            Interp::MovP0,
            Out::F32,
            tris("inputs"),
        );
        d.pixel.cntl = FLAT_SHADE;
        d.format = format;
        out.push(case(d));
    }
    for interp in [Interp::PerspCenter, Interp::LinearCenter] {
        out.push(case(Draw::new(
            "draw clip",
            format!("{interp:?} past the viewport's edges"),
            interp,
            Out::F32,
            triangles_reaching(8, "clip", 1.5),
        )));
    }

    for (fmt, number, bpp, fname) in [(10, 0, 4, "8_8_8_8 unorm"), (12, 7, 8, "16_16_16_16 float")]
    {
        let mut add = |variant: String, blend_state: u32| {
            let mut d = Draw::new(
                "draw blend",
                format!("{fname} {variant}"),
                Interp::PerspCenter,
                Out::F16,
                tris("blend"),
            );
            d.format = (fmt, number, 0, bpp);
            d.blend = blend_state;
            if reads_src1(blend_state) {
                d.pixel = pixel(Interp::PerspCenter, Out::F16, true);
            }
            out.push(case(d));
        };
        for (f, name) in FACTORS.iter().enumerate() {
            add(format!("src {name}"), blend(f as u32, 5, 0, None));
            add(format!("dst {name}"), blend(4, f as u32, 0, None));
        }
        for (f, name) in FUNCTIONS.iter().enumerate() {
            add(format!("function {name}"), blend(4, 5, f as u32, None));
        }
        add("separate alpha".into(), blend(4, 5, 0, Some((1, 1, 3))));
    }
    const ZERO: u32 = 0;
    const ONE: u32 = 1;
    const SRC_ALPHA: u32 = 4;
    const ONE_MINUS_SRC_ALPHA: u32 = 5;
    const CONSTANT_COLOR: u32 = 13;
    for (name, factor) in [
        ("dst x 0.5", CONSTANT_COLOR),
        ("dst x (1 - src alpha)", ONE_MINUS_SRC_ALPHA),
    ] {
        let mut d = Draw::new(
            "draw blend",
            format!("8_8_8_8 unorm sweep {name}"),
            Interp::PerspCenter,
            Out::F16,
            sweep_bands(),
        );
        d.blend = blend(ZERO, factor, 0, None);
        d.blend_color = [0.5; 4];
        d.fill = Some(sweep_destination());
        out.push(case(d));
    }
    for (fmt, number, bpp, fname) in [(10, 0, 4, "8_8_8_8 unorm"), (12, 7, 8, "16_16_16_16 float")]
    {
        for f in [13, 14, 19, 20] {
            for (role, state) in [
                ("src", blend(f, ONE_MINUS_SRC_ALPHA, 0, None)),
                ("dst", blend(SRC_ALPHA, f, 0, None)),
            ] {
                let mut d = Draw::new(
                    "draw blend",
                    format!("{fname} {role} {} 0.3 0.7 0.1 0.9", FACTORS[f as usize]),
                    Interp::PerspCenter,
                    Out::F16,
                    tris("blend"),
                );
                d.format = (fmt, number, 0, bpp);
                d.blend = state;
                d.blend_color = [0.3, 0.7, 0.1, 0.9];
                out.push(case(d));
            }
        }
        const BLEND_CLAMP: u32 = 1 << 15;
        let outside: Vec<[f32; 8]> = tris("blend clamp")
            .into_iter()
            .map(|mut v| {
                for a in &mut v[4..] {
                    *a = *a * 4.0 - 1.5;
                }
                v
            })
            .collect();
        for (name, state) in [
            ("one plus one", blend(ONE, ONE, 0, None)),
            ("src alpha", blend(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, 0, None)),
        ] {
            for clamp in [false, true] {
                let mut d = Draw::new(
                    "draw blend",
                    format!(
                        "{fname} {name}, sources in [-1.5, 2.5], blend clamp {}",
                        if clamp { "set" } else { "cleared" }
                    ),
                    Interp::PerspCenter,
                    Out::F16,
                    outside.clone(),
                );
                d.format = (fmt, number, 0, bpp);
                d.blend = state;
                d.info = if clamp { BLEND_CLAMP } else { 0 };
                out.push(case(d));
            }
        }
    }
    for rop3 in [0x00, 0xFF, 0x88, 0xEE, 0x66, 0x33, 0x55, 0x99] {
        for (fmt, number, name) in [(10, 0, "8_8_8_8 unorm"), (10, 4, "8_8_8_8 uint")] {
            let o = if number == 4 { Out::U32 } else { Out::F32 };
            let mut d = Draw::new(
                "draw rop3",
                format!("{name} rop3 {rop3:#04x}"),
                Interp::PerspCenter,
                o,
                tris("rop3"),
            );
            d.format = (fmt, number, 0, 4);
            d.rop3 = rop3;
            out.push(case(d));
        }
    }
    for mask in [0x1, 0x6, 0x8, 0xB] {
        let mut d = Draw::new(
            "draw write mask",
            format!("mask {mask:#x}"),
            Interp::PerspCenter,
            Out::F32,
            tris("mask"),
        );
        d.write_mask = mask;
        out.push(case(d));
    }

    const CULL_FRONT: u32 = 1;
    const CULL_BACK: u32 = 2;
    const FACE_CW: u32 = 4;
    const PROVOKING_LAST: u32 = 1 << 19;
    for (name, raster) in [
        ("cull front", CULL_FRONT),
        ("cull back", CULL_BACK),
        ("cull back, clockwise front", CULL_BACK | FACE_CW),
        ("flat shading, last vertex provokes", PROVOKING_LAST),
    ] {
        let interp = if raster == PROVOKING_LAST {
            Interp::Flat
        } else {
            Interp::PerspCenter
        };
        let mut d = Draw::new("draw raster", name.into(), interp, Out::F32, tris("raster"));
        d.raster = raster;
        out.push(case(d));
    }
    let mut d = Draw::new(
        "draw raster",
        "positive viewport y scale".into(),
        Interp::PerspCenter,
        Out::F32,
        tris("raster"),
    );
    d.flip_y = false;
    out.push(case(d));
    let mut d = Draw::new(
        "draw raster",
        "scissor 10,7 to 50,41".into(),
        Interp::PerspCenter,
        Out::F32,
        tris("raster"),
    );
    d.scissor = (10, 7, 50, 41);
    out.push(case(d));
    let edge = |x: f32, y: f32| v(x / 32.0 - 1.0, y / 32.0 - 1.0, x / 64.0, y / 64.0);
    out.push(case(Draw::new(
        "draw raster",
        "fill rule: shared edges through pixel centres".into(),
        Interp::PerspCenter,
        Out::F32,
        vec![
            edge(8.5, 8.5),
            edge(40.5, 8.5),
            edge(8.5, 40.5),
            edge(40.5, 8.5),
            edge(40.5, 40.5),
            edge(8.5, 40.5),
            edge(20.5, 50.5),
            edge(60.5, 50.5),
            edge(40.5, 20.5),
        ],
    )));
    for (name, prim) in [
        ("points", 1),
        ("lines", 2),
        ("line strip", 3),
        ("triangle strip", 6),
        ("triangle fan", 5),
        ("rectangles", 17),
    ] {
        let mut d = Draw::new(
            "draw primitive",
            name.into(),
            Interp::PerspCenter,
            Out::F32,
            tris("primitives"),
        );
        d.primitive = prim;
        out.push(case(d));
    }
    let mut d = Draw::new(
        "draw primitive",
        "screen-aligned rectangles, right angle at each vertex".into(),
        Interp::PerspCenter,
        Out::F32,
        vec![
            edge(4.0, 4.0),
            edge(28.0, 4.0),
            edge(4.0, 20.0),
            edge(36.0, 4.0),
            edge(36.0, 28.0),
            edge(60.0, 28.0),
            edge(56.0, 60.0),
            edge(8.0, 36.0),
            edge(56.0, 36.0),
        ],
    );
    d.primitive = 17;
    out.push(case(d));
    for (name, transpose) in [("x-major", false), ("y-major", true)] {
        let mut d = Draw::new(
            "draw primitive",
            format!("wide line ends, {name}"),
            Interp::PerspCenter,
            Out::F32,
            line_ends(transpose),
        );
        d.primitive = 2;
        d.line_half_width = 1 << 4;
        out.push(case(d));
    }
    for (name, launch) in [
        ("fast launch", MeshLaunch::Fast),
        ("amplified", MeshLaunch::Amplified),
    ] {
        let mut d = Draw::new(
            "draw mesh",
            format!("LDS across waves and barrier, {name}"),
            Interp::PerspCenter,
            Out::F32,
            mesh_strips(),
        );
        d.mesh = Some(launch);
        d.covers
            .extend(["ds_write_b32", "ds_read_b32", "s_barrier", "s_sendmsg"]);
        out.push(case(d));
    }
    out.extend(texture::cases());
    out
}

/// Vertex `k` of the strip in the quadrant from (`x0`, `y0`).
fn mesh_corner(x0: f32, y0: f32, k: usize) -> (f32, f32) {
    (
        x0 + 0.05 + 0.3 * (k / 2) as f32,
        y0 + 0.15 + 0.7 * (k % 2) as f32 - 0.01 * k as f32,
    )
}

/// The vertex table of the mesh draw: four groups, each a strip of
/// [`MESH_VERTICES`] vertices in its own quadrant, coloured by group and vertex.
fn mesh_strips() -> Vec<[f32; 8]> {
    let mut out = Vec::new();
    for g in 0..4 {
        let (x0, y0) = ((g % 2) as f32 - 1.0, (g / 2) as f32 - 1.0);
        for k in 0..MESH_VERTICES {
            let (x, y) = mesh_corner(x0, y0, k);
            out.push([x, y, 0.5, 1.0, g as f32 / 4.0, k as f32 / 8.0, 0.25, 1.0]);
        }
    }
    out
}

/// Short 2-pixel-wide lines, 16 going each way, whose ends step through
/// sixteenths of a pixel along their major axis, so the hardware's rule for
/// which end pixels a line covers can be read off. `transpose` swaps x and y.
fn line_ends(transpose: bool) -> Vec<[f32; 8]> {
    let mut out = Vec::new();
    for k in 0..16 {
        let f = k as f32 / 16.0;
        let g = ((k * 7) % 16) as f32 / 16.0;
        let across = 2.0 + 4.0 * k as f32 + 0.375;
        for (from, to) in [(2.0 + f, 26.0 + g), (62.0 - f, 34.0 - g)] {
            for along in [from, to] {
                let (x, y) = if transpose {
                    (across, along)
                } else {
                    (along, across)
                };
                let t = (along - 2.0) / 60.0;
                out.push([x / 32.0 - 1.0, 1.0 - y / 32.0, 0.5, 1.0, t, 1.0 - t, f, 1.0]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::decode::decode;
    use gfx10::print::{Syntax, print};

    fn disassemble(code: &[u32]) -> Vec<String> {
        let mut out = Vec::new();
        let mut pc = 0;
        while pc < code.len() {
            let inst = decode(&code[pc..]).expect("pixel programs decode");
            out.push(print(&inst, Syntax { wave64: true }));
            pc += inst.words as usize;
        }
        out
    }

    #[test]
    fn pixel_programs_read_their_inputs_where_the_addr_bits_put_them() {
        let p = pixel(Interp::LinearCenter, Out::F32, false);
        let text = disassemble(&p.code);
        assert_eq!(text[1], "v_interp_p1_f32_e32 v20, v0, attr0.x");
        let p = pixel(Interp::FragCoord, Out::Unorm16, false);
        let text = disassemble(&p.code);
        assert_eq!(text[1], "v_mov_b32_e32 v20, v2");
        assert!(
            text.iter()
                .any(|t| t.starts_with("exp mrt0 v24, v24, v25, v25 done compr vm"))
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mesh_groups_draw_the_vertices_their_lds_exchange_picks() {
        for mut c in cases().into_iter().filter(|c| c.opcode == "draw mesh") {
            crate::metal::run(&mut c).unwrap();
            assert_eq!(c.untranslated, None);
            let (drawn, before) = (c.metal_checked(), {
                let mut b = vec![0u8; c.checked.1 as usize];
                crate::read_pages(&[&c.input], c.checked.0, &mut b);
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| u32::from_le_bytes(*w))
                    .collect::<Vec<_>>()
            });
            let pitch = c.target.3 as usize / 4;
            let at = |(x, y): (f32, f32)| {
                let (px, py) = ((x + 1.0) * 32.0, (1.0 - y) * 32.0);
                py as usize * pitch + px as usize
            };
            let mix = |p: [(f32, f32); 3], w: [f32; 3]| {
                (
                    p.iter().zip(w).map(|(p, w)| p.0 * w).sum::<f32>(),
                    p.iter().zip(w).map(|(p, w)| p.1 * w).sum::<f32>(),
                )
            };
            for g in 0..4 {
                let (x0, y0) = ((g % 2) as f32 - 1.0, (g / 2) as f32 - 1.0);
                let v = |k| mesh_corner(x0, y0, k);
                let first = drawn[at(mix([v(0), v(1), v(2)], [1.0 / 3.0; 3]))].to_le_bytes();
                assert!(
                    first[0].abs_diff(g as u8 * 64) <= 2 && first[1].abs_diff(32) <= 4,
                    "group {g}: {first:?}"
                );
                let null = at(mix([v(3), v(4), v(5)], [0.1, 0.1, 0.8]));
                assert_eq!(
                    drawn[null], before[null],
                    "group {g} drew its null triangle"
                );
            }
        }
    }

    #[test]
    fn the_draw_suite_covers_formats_blending_and_raster_state() {
        assert!(cases().len() > 300);
    }
}
