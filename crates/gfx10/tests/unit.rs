//! Self-contained checks on hand-written programs (no llvm-mc, no game files).

use gfx10::translate::{Access, Request, Resource, Source, Stage, TextureKind};
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

#[cfg(target_os = "macos")]
fn compile(msl: &str) {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let Some(device) = MTLCreateSystemDefaultDevice() else { return };
    if let Err(e) = device.newLibraryWithSource_options_error(&NSString::from_str(msl), None) {
        panic!("Metal rejected the MSL: {}\n{msl}", e.localizedDescription());
    }
}

#[cfg(not(target_os = "macos"))]
fn compile(_: &str) {}

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
        compute_rsrc2: 0,
        threads: [1, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
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
            Resource::Texture { slot: 0, source: Source::UserSgpr(0), kind: TextureKind::Tex2d, access: Access::Sample },
            Resource::Sampler { slot: 0, source: Source::UserSgpr(8) },
        ]
    );
    assert!(t.msl.contains("[[user(locn0), center_perspective]]"));
}

fn compute(code: &[u32], user_sgprs: u32, rsrc2: u32) -> gfx10::Translated {
    translate(&Request {
        stage: Stage::Compute,
        code,
        user_sgprs,
        ngg: false,
        ps_input_ena: 0,
        ps_input_addr: 0,
        ps_input_cntl: &[],
        col_format: 0,
        compute_rsrc2: rsrc2,
        threads: [64, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
    })
    .unwrap()
}

#[test]
fn compute_program_uses_group_ids_lds_barriers_stores_and_atomics() {
    let code = [
        0xD746_0002, 0x0401_0C04, // v_lshl_add_u32 v2, s4, 6, v0
        0x3406_0082, // v_lshlrev_b32_e32 v3, 2, v0
        0xD834_0000, 0x0000_0203, // ds_write_b32 v3, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0xD8D8_0004, 0x0400_0003, // ds_read_b32 v4, v3 offset:4
        0xD800_0000, 0x0000_0203, // ds_add_u32 v3, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_2000, 0x8000_0402, // buffer_store_dword v4, v2, s[0:3], 0 idxen
        0xE0C8_6000, 0x8000_0402, // buffer_atomic_add v4, v2, s[0:3], 0 idxen glc
        0xBF81_0000, // s_endpgm
    ];
    // Four user SGPRs, TGID_X_EN, LDS_SIZE of 2 (1 KiB).
    let t = compute(&code, 4, 4 << 1 | 1 << 7 | 2 << 15);
    assert!(t.guest_memory && t.writes_memory);
    assert_eq!(t.lds_bytes, 1024);
    assert!(t.msl.contains("s4 = gid.x;"));
    assert!(t.msl.contains("v0 = lid.x;"));
    assert!(t.msl.contains("threadgroup_barrier"));
    assert!(t.msl.contains("device uchar* mem"));
    compile(&t.msl);
}

#[test]
fn descriptors_loaded_through_user_pointers_become_memory_sources() {
    let code = [
        0xF404_0101, 0xFA00_0010, // s_load_dwordx2 s[4:5], s[2:3], 0x10
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF40C_0202, 0xFA00_0020, // s_load_dwordx8 s[8:15], s[4:5], 0x20
        0xF408_0401, 0xFA00_0000, // s_load_dwordx4 s[16:19], s[2:3], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF080_0F08, 0x0082_0200, // image_sample v[2:5], v[0:1], s[8:15], s[16:19] dmask:0xf dim:2D
        0xF038_0308, 0x0002_0600, // image_get_resinfo v[6:7], v0, s[8:15] dmask:0x3 dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_180F, 0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 4);
    let table = Source::Memory { pointer: Box::new(Source::UserSgpr(2)), offset: 0x10 };
    assert_eq!(
        t.resources,
        vec![
            Resource::Texture {
                slot: 0,
                source: Source::Memory { pointer: Box::new(table), offset: 0x20 },
                kind: TextureKind::Tex2d,
                access: Access::Sample,
            },
            Resource::Sampler { slot: 0, source: Source::Memory { pointer: Box::new(Source::UserSgpr(2)), offset: 0 } },
        ]
    );
    assert!(t.msl.contains("get_width"));
    compile(&t.msl);
}

#[test]
fn image_load_and_store_bind_read_and_write_textures() {
    let code = [
        0xF000_0F08, 0x0000_0200, // image_load v[2:5], v[0:1], s[0:7] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF020_0F08, 0x0002_0200, // image_store v[2:5], v[0:1], s[8:15] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 16, 16 << 1);
    let access: Vec<_> = t
        .resources
        .iter()
        .map(|r| match r {
            Resource::Texture { access, .. } => *access,
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(access, [Access::Sample, Access::Write]);
    assert!(t.msl.contains("access::write"));
    compile(&t.msl);
}

#[test]
fn carry_arithmetic_quad_broadcasts_and_cube_ops_translate() {
    let code = [
        0xD70F_6A02, 0x0002_0300, // v_add_co_u32 v2, vcc, v0, v1
        0xD528_0403, 0x01AA_0300, // v_add_co_ci_u32_e64 v3, s[4:5], v0, v1, vcc
        0x5208_0300, // v_sub_co_ci_u32_e32 v4, vcc, v0, v1, vcc
        0x7E0A_02FA, 0xFF08_5500, // v_mov_b32_dpp v5, v0 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
        0x080C_02FA, 0xFF00_AA00, // v_sub_f32_dpp v6, v0, v1 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
        0xD547_0007, 0x040A_0300, // v_cubema_f32 v7, v0, v1, v2
        0xD544_0008, 0x040A_0300, // v_cubeid_f32 v8, v0, v1, v2
        0xD558_0009, 0x040A_0300, // v_med3_i32 v9, v0, v1, v2
        0xD56C_000A, 0x0002_0300, // v_mul_hi_i32 v10, v0, v1
        0xF800_180F, 0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 0);
    assert!(t.msl.contains("quad_broadcast(v0, 1u)"));
    assert!(t.msl.contains("quad_broadcast(v0, 2u)"));
    assert!(t.msl.contains("cube_ma("));
    compile(&t.msl);
}

#[test]
fn gather_from_a_texture_array() {
    let code = [
        0xF11C_0128, 0x0040_0200, // image_gather4_lz v[2:5], v[0:2], s[0:7], s[8:11] dmask:0x1 dim:2D_ARRAY
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_180F, 0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 12);
    assert!(t.msl.contains(".gather(smp0"));
    assert!(t.msl.contains("texture2d_array<float>"));
    compile(&t.msl);
}

#[test]
fn integer_colour_formats_export_integers() {
    let code = [
        0xF800_180F, 0x0302_0100, // exp mrt0 v0, v1, v2, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let mut req = Request {
        stage: Stage::Pixel,
        code: &code,
        user_sgprs: 0,
        ngg: false,
        ps_input_ena: 2,
        ps_input_addr: 2,
        ps_input_cntl: &[],
        col_format: 7,
        compute_rsrc2: 0,
        threads: [1, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
    };
    let t = translate(&req).unwrap();
    assert!(t.msl.contains("uint4 c0 [[color(0)]]"));
    compile(&t.msl);
    req.col_format = 8;
    let t = translate(&req).unwrap();
    assert!(t.msl.contains("int4 c0 [[color(0)]]"));
    compile(&t.msl);
}

#[test]
fn integer_texture_slots_read_and_write_integers() {
    let code = [
        0xF000_0F08, 0x0000_0200, // image_load v[2:5], v[0:1], s[0:7] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF020_0F08, 0x0002_0200, // image_store v[2:5], v[0:1], s[8:15] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let request = |uint_textures, sint_textures| Request {
        stage: Stage::Compute,
        code: &code,
        user_sgprs: 16,
        ngg: false,
        ps_input_ena: 0,
        ps_input_addr: 0,
        ps_input_cntl: &[],
        col_format: 0,
        compute_rsrc2: 16 << 1,
        threads: [64, 1, 1],
        uint_textures,
        sint_textures,
    };
    let float = translate(&request(0, 0)).unwrap();
    let typed = translate(&request(1, 2)).unwrap();
    assert_eq!(float.resources, typed.resources);
    assert!(typed.msl.contains("texture2d<uint> t0"));
    assert!(typed.msl.contains("texture2d<int, access::write> t1"));
    compile(&typed.msl);
}

#[test]
fn version_1_captures_still_load() {
    let code = [0xBF81_0000u32];
    let mut words = vec![u32::from_le_bytes(*b"g10c"), 1, 2, 0, 0, 0, 0, 0, 0, 64, 1, 1, 0];
    words.extend_from_slice(&code);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let c = gfx10::capture::Capture::from_bytes(&bytes).unwrap();
    assert_eq!((c.threads, c.uint_textures, c.code.as_slice()), ([64, 1, 1], 0, &code[..]));
    let again = gfx10::capture::Capture::from_bytes(&c.to_bytes()).unwrap();
    assert_eq!((again.threads, again.code), (c.threads, c.code));
}
