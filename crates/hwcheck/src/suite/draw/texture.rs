//! Texture draws: a full-screen quad whose pixel program samples a texture
//! at coordinates computed from the pixel's position, so both sides sample
//! at the same coordinates and only filtering can differ. A small texture is
//! magnified, a large mipmapped one minified evenly and anisotropically,
//! under the common samplers, for 8-bit unorm, sRGB and half-float textures,
//! into 32-bit float and 8-bit unorm targets.

use super::super::INPUTS;
use super::super::formats::FORMATS;
use super::super::mimg::{
    Arg, DIMS, Dim, SRSRC, SSAMP, Sampler, TEXTURE, VDST, args, descriptor_at, encode_at, f16_bits,
    sampler_words,
};
use super::{Draw, Interp, LITERAL, Out, Pixel, case, export, input_vgpr, vop1, vop2};
use crate::Case;
use gfx10::opinfo::by_name;

/// The image descriptor (32 bytes), then the sampler (16 bytes); the pixel
/// program gets this address in its first two user SGPRs.
pub const DESC: u64 = INPUTS + 0x1000;
/// Where the pixel program puts the sample's address operands.
const VADDR: u32 = 4;

/// Sampled at `u = x * u[0] + y * u[1] + u[2]` (and `v` alike).
struct Scene {
    name: &'static str,
    size: u32,
    levels: u32,
    u: [f32; 3],
    v: [f32; 3],
}

const MAGNIFIED: Scene = Scene {
    name: "magnified 8x8",
    size: 8,
    levels: 1,
    u: [1.13 / 64.0, 0.07 / 64.0, -0.06],
    v: [-0.05 / 64.0, 1.07 / 64.0, -0.04],
};

const MINIFIED: Scene = Scene {
    name: "minified 128x128",
    size: 128,
    levels: 8,
    u: [3.3 / 64.0, 0.4 / 64.0, -0.1],
    v: [-0.3 / 64.0, 2.9 / 64.0, 0.05],
};

const ANISOTROPIC: Scene = Scene {
    name: "anisotropic 128x128",
    size: 128,
    levels: 8,
    u: [1.1 / 64.0, 0.2 / 64.0, 0.02],
    v: [0.5 / 64.0, 6.7 / 64.0, -0.1],
};

/// Samplers for magnification: point and bilinear under each address mode.
const MAGNIFYING: [Sampler; 6] = [
    ("point wrap", 0, 0, 0, 0, 0, 0, false, 0),
    ("bilinear wrap", 0, 1, 1, 0, 0, 0, false, 0),
    ("bilinear mirror", 1, 1, 1, 0, 0, 0, false, 0),
    ("bilinear clamp to edge", 2, 1, 1, 0, 0, 0, false, 0),
    (
        "bilinear transparent black border",
        6,
        1,
        1,
        0,
        0,
        0,
        false,
        0,
    ),
    ("bilinear opaque white border", 6, 1, 1, 0, 0, 2, false, 0),
];

/// Samplers for minification, with `MAX_ANISO_RATIO`.
const MINIFYING: [(Sampler, u32); 5] = [
    (("point, mips point", 0, 0, 0, 1, 0, 0, false, 0), 0),
    (("bilinear, mips point", 0, 1, 1, 1, 0, 0, false, 0), 0),
    (("trilinear", 0, 1, 1, 2, 0, 0, false, 0), 0),
    (("trilinear aniso 2x", 0, 1, 1, 2, 0, 0, false, 0), 1),
    (("trilinear aniso 16x", 0, 1, 1, 2, 0, 0, false, 0), 4),
];

const TEXTURE_FORMATS: [&str; 3] = ["8_8_8_8_UNORM", "8_8_8_8_SRGB", "16_16_16_16_FLOAT"];

/// Render targets: name and `Draw::format`.
type Target = (&'static str, (u32, u32, u32, u32));

const TARGETS: [Target; 2] = [
    ("32_32_32_32 float", (14, 7, 0, 16)),
    ("8_8_8_8 unorm", (10, 0, 0, 4)),
];

/// The LOD of `image_sample_l` and the bias of `image_sample_b`: between
/// two mips, so trilinear filtering blends them.
pub const LOD: f32 = 1.37;
pub const BIAS: f32 = 0.6;

/// The pixel program: loads the descriptors, computes the address from the
/// pixel position, samples into v20..v23 and exports them as 32-bit floats.
fn program(op: &'static str, scene: &Scene, dmask: u32) -> Pixel {
    const PERSP_CENTER: u32 = 1;
    const POS_X: u32 = 8;
    const POS_Y: u32 = 9;
    let inputs = 1 << PERSP_CENTER | 1 << POS_X | 1 << POS_Y;
    let x = input_vgpr(inputs, POS_X);
    let y = input_vgpr(inputs, POS_Y);
    let mut code = vec![
        0xBEFC_0302, // s_mov_b32 m0, s2
        0xF40C_0000 | SRSRC << 6,
        0xFA00_0000, // s_load_dwordx8 s[40:47], s[0:1], null
        0xF408_0000 | SSAMP << 6,
        0xFA00_0020, // s_load_dwordx4 s[48:51], s[0:1], 0x20
    ];
    let dim = DIMS[1];
    for (k, arg) in args(op, dim).into_iter().enumerate() {
        let r = VADDR + k as u32;
        let affine = |c: [f32; 3]| {
            [
                vop2("v_mul_f32", r, LITERAL, x),
                c[0].to_bits(),
                vop2("v_fmac_f32", r, LITERAL, y),
                c[1].to_bits(),
                vop2("v_add_f32", r, LITERAL, r),
                c[2].to_bits(),
            ]
        };
        match arg {
            Arg::Coord(0) => code.extend(affine(scene.u)),
            Arg::Coord(1) => code.extend(affine(scene.v)),
            Arg::Lod => code.extend([vop1("v_mov_b32", r, LITERAL), LOD.to_bits()]),
            Arg::Bias => code.extend([vop1("v_mov_b32", r, LITERAL), BIAS.to_bits()]),
            a => panic!("{op} takes {a:?}, which texture draws do not set"),
        }
    }
    code.push(0xBF8C_0070); // s_waitcnt vmcnt(0) lgkmcnt(0)
    code.extend(encode_at(
        by_name(op).expect("in the opcode table"),
        dim,
        dmask,
        false,
        VADDR,
    ));
    code.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    code.extend(export(0, false, true, [VDST, VDST + 1, VDST + 2, VDST + 3]));
    code.push(0xBF81_0000); // s_endpgm
    Pixel {
        inputs,
        cntl: 0,
        code,
        dual: false,
    }
}

/// Texel `(x, y)` of mip `level` of a square texture of `size` texels: a
/// horizontal gradient, a checkerboard of 8 by 8 cells (single texels on
/// small mips), a vertical gradient with a sharp diagonal step, and a value
/// that tells the mips apart.
pub(crate) fn texel(x: u32, y: u32, size: u32, level: u32, levels: u32) -> [f32; 4] {
    let n = (size >> level).max(1);
    let cell = (n / 8).max(1);
    let checker = ((x / cell) ^ (y / cell)) & 1;
    let step = if x + y >= n { 0.5 } else { 0.0 };
    [
        (x as f32 + 0.5) / n as f32,
        checker as f32,
        (y as f32 + 0.5) / n as f32 * 0.5 + step,
        0.15 + 0.7 * level as f32 / (levels - 1).max(1) as f32,
    ]
}

/// A texture's stored value of `v` in [0, 1]: 8-bit (unorm or sRGB-encoded
/// bytes alike) or half floats spread over [-1, 2].
fn encode_texel(v: [f32; 4], half: bool, out: &mut Vec<u8>) {
    for c in v {
        if half {
            out.extend(f16_bits(c * 3.0 - 1.0).to_le_bytes());
        } else {
            out.push((c * 255.0).round() as u8);
        }
    }
}

/// The texture's bytes: the mips from the smallest up, rows padded to 256
/// bytes, as RDNA2 lays out linear textures (see `gpu::tiling`).
pub(crate) fn texture_bytes(size: u32, levels: u32, half: bool) -> Vec<u8> {
    let bpp = if half { 8 } else { 4 };
    let mut out = Vec::new();
    for level in (0..levels).rev() {
        let n = (size >> level).max(1);
        let row = (n * bpp).next_multiple_of(256) as usize;
        for y in 0..n {
            let start = out.len();
            for x in 0..n {
                encode_texel(texel(x, y, size, level, levels), half, &mut out);
            }
            out.resize(start + row, 0);
        }
    }
    out
}

fn quad() -> Vec<[f32; 8]> {
    let v = |x: f32, y: f32| [x, y, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0];
    vec![
        v(-1.0, -1.0),
        v(1.0, -1.0),
        v(-1.0, 1.0),
        v(1.0, -1.0),
        v(1.0, 1.0),
        v(-1.0, 1.0),
    ]
}

#[allow(clippy::too_many_arguments)]
fn draw(
    op: &'static str,
    dmask: u32,
    scene: &Scene,
    format: &str,
    sampler: Sampler,
    aniso: u32,
    target: Target,
) -> TextureDraw {
    let (_, fcode, bits, _) = FORMATS
        .iter()
        .copied()
        .find(|f| f.0 == format)
        .expect("in Mesa's format table");
    let dim = Dim {
        size: (scene.size, scene.size, 1),
        ..DIMS[1]
    };
    let mut table: Vec<u32> = descriptor_at(TEXTURE, dim, fcode, scene.levels).to_vec();
    table.extend(sampler_words(sampler, aniso));
    let table: Vec<u8> = table.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut d = Draw::new(
        "draw texture",
        format!("{} {op} {format} {} to {}", scene.name, sampler.0, target.0),
        Interp::PerspCenter,
        Out::F32,
        quad(),
    );
    d.pixel = program(op, scene, dmask);
    d.format = target.1;
    d.user_data = vec![DESC as u32, (DESC >> 32) as u32];
    d.memory = vec![
        (DESC, table),
        (TEXTURE, texture_bytes(scene.size, scene.levels, bits == 64)),
    ];
    d.covers = vec!["exp", op];
    TextureDraw {
        case: case(d),
        op,
        u: scene.u,
        v: scene.v,
        float_target: target.1.0 == 14,
    }
}

/// A texture draw and how its pixel program samples.
pub struct TextureDraw {
    pub case: Case,
    pub op: &'static str,
    /// The coordinates at pixel centre `(x, y)`: `u[0] * x + u[1] * y + u[2]`,
    /// the second term fused.
    pub u: [f32; 3],
    pub v: [f32; 3],
    /// Whether the target holds 32-bit floats rather than 8-bit unorm.
    pub float_target: bool,
}

/// Every texture draw.
pub fn cases() -> Vec<Case> {
    draws().into_iter().map(|d| d.case).collect()
}

/// Every texture draw with how it samples.
pub fn draws() -> Vec<TextureDraw> {
    let mut out = Vec::new();
    for format in TEXTURE_FORMATS {
        for target in TARGETS {
            for s in MAGNIFYING {
                out.push(draw("image_sample", 0xF, &MAGNIFIED, format, s, 0, target));
            }
            for scene in [&MINIFIED, &ANISOTROPIC] {
                for (s, aniso) in MINIFYING {
                    out.push(draw("image_sample", 0xF, scene, format, s, aniso, target));
                }
            }
        }
        let f32_target = TARGETS[0];
        for op in ["image_sample_l", "image_sample_b"] {
            for (s, aniso) in &MINIFYING[1..3] {
                out.push(draw(op, 0xF, &MINIFIED, format, *s, *aniso, f32_target));
            }
        }
        for s in [MAGNIFYING[0], MAGNIFYING[1], MAGNIFYING[3], MAGNIFYING[5]] {
            out.push(draw(
                "image_gather4",
                1,
                &MAGNIFIED,
                format,
                s,
                0,
                f32_target,
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gfx10::decode::decode;
    use gfx10::print::{Syntax, print};

    #[test]
    fn texture_programs_load_their_descriptors_and_sample_at_the_pixel() {
        let p = program("image_sample_b", &MINIFIED, 0xF);
        let mut text = Vec::new();
        let mut pc = 0;
        while pc < p.code.len() {
            let inst = decode(&p.code[pc..]).expect("texture programs decode");
            text.push(print(&inst, Syntax { wave64: true }));
            pc += inst.words as usize;
        }
        assert_eq!(text[0], "s_mov_b32 m0, s2");
        assert_eq!(text[1], "s_load_dwordx8 s[40:47], s[0:1], null");
        assert_eq!(text[2], "s_load_dwordx4 s[48:51], s[0:1], 0x20");
        assert!(text[4].starts_with("v_mul_f32_e32 v5, 0x3d533333, v2"));
        assert!(text[5].starts_with("v_fmac_f32_e32 v5, 0x3bcccccd, v3"));
        assert!(text[9].starts_with("v_add_f32_e32 v6, 0x3d4ccccd, v6"));
        assert!(text.iter().any(|t| t
            == "image_sample_b v[20:23], v[4:6], s[40:47], s[48:51] dmask:0xf dim:SQ_RSRC_IMG_2D"));
        assert_eq!(text.last().unwrap(), "s_endpgm");
    }

    #[test]
    fn texture_bytes_hold_every_mip() {
        assert_eq!(texture_bytes(8, 1, false).len(), 8 * 256);
        let rows: u32 = (0..8).map(|l| (128u32 >> l).max(1)).sum();
        assert_eq!(
            texture_bytes(128, 8, true).len(),
            256 * 128 * 4 + 256 * (rows as usize - 128 - 64) + 512 * 64
        );
    }
}
