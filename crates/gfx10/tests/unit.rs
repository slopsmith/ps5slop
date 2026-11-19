//! Self-contained checks on hand-written programs (no llvm-mc, no game files).

use gfx10::translate::{Request, Resource, Source, Stage, TextureKind};
use gfx10::{Syntax, decode, print, translate};

fn text(words: &[u32]) -> String {
    print(&decode(words).unwrap(), Syntax { wave64: true })
}

#[test]
fn prints_llvm_syntax() {
    assert_eq!(text(&[0x7E00_0000]), "v_nop");
    assert_eq!(text(&[0x7E02_0302]), "v_mov_b32_e32 v1, v2");
    assert_eq!(text(&[0xD541_0001, 0x0412_0702]), "v_mad_f32 v1, v2, v3, v4");
    assert_eq!(text(&[0xBF8C_3F70]), "s_waitcnt vmcnt(0)");
    assert_eq!(text(&[0xF800_180F, 0x0403_0201]), "exp mrt0 v1, v2, v3, v4 done vm");
}

fn pixel(code: &[u32], user_sgprs: u32) -> gfx10::Translated {
    translate(&Request {
        stage: Stage::Pixel,
        code,
        user_sgprs,
        ngg: false,
        ps_input_ena: 2,
        ps_input_addr: 2,
        ps_input_cntl: &[],
        col_format: 9,
    })
    .unwrap()
}

#[test]
fn constant_colour_pixel_shader() {
    let code = [
        0x7E00_0200, // v_mov_b32 v0, s0
        0x7E02_0201, // v_mov_b32 v1, s1
        0x7E04_0202, // v_mov_b32 v2, s2
        0x7E06_0203, // v_mov_b32 v3, s3
        0xF800_180F, // exp mrt0 v0, v1, v2, v3 done vm
        0x0302_0100,
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 4);
    assert_eq!(t.color_outputs, 1);
    assert!(t.resources.is_empty());
    assert!(t.msl.contains("s0 = user[0];"));
    assert!(t.msl.contains("o.c0 = float4(F(v0), F(v1), F(v2), F(v3));"));
}

#[test]
fn textured_pixel_shader_binds_user_sgpr_descriptors() {
    let code = [
        0xC800_0200, // v_interp_p1_f32 v0, v0, attr0.x
        0xC801_0201, // v_interp_p2_f32 v0, v1, attr0.x
        0xC804_0300, // v_interp_p1_f32 v1, v0, attr0.y
        0xC805_0301, // v_interp_p2_f32 v1, v1, attr0.y
        0xF080_0F08, // image_sample v[2:5], v[0:1], s[0:7], s[8:11] dmask:0xf dim:2D
        0x0040_0200,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_180F, // exp mrt0 v2, v3, v4, v5 done vm
        0x0504_0302,
        0xBF81_0000,
    ];
    let t = pixel(&code, 12);
    assert_eq!(
        t.resources,
        vec![
            Resource::Texture { slot: 0, source: Source::UserSgpr(0), kind: TextureKind::Tex2d },
            Resource::Sampler { slot: 0, source: Source::UserSgpr(8) },
        ]
    );
    assert!(t.msl.contains("[[user(locn0), center_perspective]]"));
}
