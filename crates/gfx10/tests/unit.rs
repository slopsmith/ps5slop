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
    assert_eq!(
        text(&[0xD541_0001, 0x0412_0702]),
        "v_mad_f32 v1, v2, v3, v4"
    );
    assert_eq!(text(&[0xBF8C_3F70]), "s_waitcnt vmcnt(0)");
    assert_eq!(
        text(&[0xF800_180F, 0x0403_0201]),
        "exp mrt0 v1, v2, v3, v4 done vm"
    );
}

#[cfg(target_os = "macos")]
fn compile(msl: &str) {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    if let Err(e) = device.newLibraryWithSource_options_error(&NSString::from_str(msl), None) {
        panic!(
            "Metal rejected the MSL: {}\n{msl}",
            e.localizedDescription()
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn compile(_: &str) {}

fn pixel(code: &[u32], user_sgprs: u32) -> gfx10::Translated {
    translate(&Request {
        stage: Stage::Pixel,
        code,
        address: 0,
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
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
        scratch_wave_bytes: 0,
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
            Resource::Texture {
                slot: 0,
                source: Source::UserSgpr(0),
                kind: TextureKind::Tex2d,
                access: Access::Sample
            },
            Resource::Sampler {
                slot: 0,
                source: Source::UserSgpr(8)
            },
        ]
    );
    assert!(t.msl.contains("[[user(locn0), center_perspective]]"));
}

#[test]
fn pixel_inputs_sharing_a_location_read_one_interpolant() {
    let code = [
        0xC800_0200, // v_interp_p1_f32 v0, v0, attr0.x
        0xC801_0201, // v_interp_p2_f32 v0, v1, attr0.x
        0xC804_0600, // v_interp_p1_f32 v1, v0, attr1.x
        0xC805_0601, // v_interp_p2_f32 v1, v1, attr1.x
        0xF800_180F, // exp mrt0 v0, v1, v0, v1 done vm
        0x0100_0100,
        0xBF81_0000,
    ];
    let t = translate(&Request {
        stage: Stage::Pixel,
        code: &code,
        address: 0,
        user_sgprs: 0,
        ngg: false,
        ps_input_ena: 2,
        ps_input_addr: 2,
        ps_input_cntl: &[0, 0],
        col_format: 9,
        compute_rsrc2: 0,
        threads: [1, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
        scratch_wave_bytes: 0,
    })
    .unwrap();
    assert_eq!(t.msl.matches("[[user(locn0)").count(), 1);
    assert!(
        t.msl
            .contains("PsIn in = { raw.pos, raw.front, raw.a0, raw.a0 };")
    );
}

fn compute(code: &[u32], user_sgprs: u32, rsrc2: u32) -> gfx10::Translated {
    compute_group(code, user_sgprs, rsrc2, 64)
}

fn compute_group(code: &[u32], user_sgprs: u32, rsrc2: u32, threads: u32) -> gfx10::Translated {
    translate(&Request {
        stage: Stage::Compute,
        code,
        address: 0,
        user_sgprs,
        ngg: false,
        ps_input_ena: 0,
        ps_input_addr: 0,
        ps_input_cntl: &[],
        col_format: 0,
        compute_rsrc2: rsrc2,
        threads: [threads, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
        scratch_wave_bytes: 0,
    })
    .unwrap()
}

#[test]
fn compute_program_uses_group_ids_lds_barriers_stores_and_atomics() {
    let code = [
        0xD746_0002,
        0x0401_0C04, // v_lshl_add_u32 v2, s4, 6, v0
        0x3406_0082, // v_lshlrev_b32_e32 v3, 2, v0
        0xD834_0000,
        0x0000_0203, // ds_write_b32 v3, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0xD8D8_0004,
        0x0400_0003, // ds_read_b32 v4, v3 offset:4
        0xD800_0000,
        0x0000_0203, // ds_add_u32 v3, v2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_2000,
        0x8000_0402, // buffer_store_dword v4, v2, s[0:3], 0 idxen
        0xE0C8_6000,
        0x8000_0402, // buffer_atomic_add v4, v2, s[0:3], 0 idxen glc
        0xBF81_0000, // s_endpgm
    ];
    // Four user SGPRs, TGID_X_EN, LDS_SIZE of 2 (1 KiB).
    let t = compute(&code, 4, 4 << 1 | 1 << 7 | 2 << 15);
    assert!(t.guest_memory && t.writes_memory);
    assert_eq!(t.store_buffers, Some(vec![(Source::UserSgpr(0), true)]));
    assert_eq!(t.lds_bytes, 1024);
    assert!(t.msl.contains("s4 = gid.x;"));
    assert!(t.msl.contains("v0 = lid.x;"));
    assert!(t.msl.contains("threadgroup_barrier"));
    assert!(t.msl.contains("device uchar* mem"));
    compile(&t.msl);
}

#[test]
fn stores_through_a_v_sharp_the_program_changes_are_not_known() {
    let code = [
        0x8000_8400, // s_add_u32 s0, s0, 4
        0xE070_2000,
        0x8000_0402, // buffer_store_dword v4, v2, s[0:3], 0 idxen
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 4, 4 << 1);
    assert!(t.writes_memory);
    assert_eq!(t.store_buffers, None);
    let loaded = [
        0xF408_0001,
        0xFA00_0040, // s_load_dwordx4 s[0:3], s[2:3], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_0000,
        0x8000_0402, // buffer_store_dword v4, v2, s[0:3], 0
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&loaded, 4, 4 << 1);
    let table = Source::Memory {
        pointer: Box::new(Source::UserSgpr(2)),
        offset: 0x40,
    };
    assert_eq!(t.store_buffers, Some(vec![(table, false)]));
}

#[test]
fn descriptors_loaded_through_user_pointers_become_memory_sources() {
    let code = [
        0xF404_0101,
        0xFA00_0010, // s_load_dwordx2 s[4:5], s[2:3], 0x10
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF40C_0202,
        0xFA00_0020, // s_load_dwordx8 s[8:15], s[4:5], 0x20
        0xF408_0401,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[2:3], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF080_0F08,
        0x0082_0200, // image_sample v[2:5], v[0:1], s[8:15], s[16:19] dmask:0xf dim:2D
        0xF038_0308,
        0x0002_0600, // image_get_resinfo v[6:7], v0, s[8:15] dmask:0x3 dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_180F,
        0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 4);
    let table = Source::Memory {
        pointer: Box::new(Source::UserSgpr(2)),
        offset: 0x10,
    };
    assert_eq!(
        t.resources,
        vec![
            Resource::Texture {
                slot: 0,
                source: Source::Memory {
                    pointer: Box::new(table),
                    offset: 0x20
                },
                kind: TextureKind::Tex2d,
                access: Access::Sample,
            },
            Resource::Sampler {
                slot: 0,
                source: Source::Memory {
                    pointer: Box::new(Source::UserSgpr(2)),
                    offset: 0
                }
            },
        ]
    );
    assert!(t.msl.contains("get_width"));
    compile(&t.msl);
}

#[test]
fn image_load_and_store_bind_read_and_write_textures() {
    let code = [
        0xF000_0F08,
        0x0000_0200, // image_load v[2:5], v[0:1], s[0:7] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF020_0F08,
        0x0002_0200, // image_store v[2:5], v[0:1], s[8:15] dmask:0xf dim:2D
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
        0xD70F_6A02,
        0x0002_0300, // v_add_co_u32 v2, vcc, v0, v1
        0xD528_0403,
        0x01AA_0300, // v_add_co_ci_u32_e64 v3, s[4:5], v0, v1, vcc
        0x5208_0300, // v_sub_co_ci_u32_e32 v4, vcc, v0, v1, vcc
        0x7E0A_02FA,
        0xFF08_5500, // v_mov_b32_dpp v5, v0 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
        0x080C_02FA,
        0xFF00_AA00, // v_sub_f32_dpp v6, v0, v1 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
        0xD547_0007,
        0x040A_0300, // v_cubema_f32 v7, v0, v1, v2
        0xD544_0008,
        0x040A_0300, // v_cubeid_f32 v8, v0, v1, v2
        0xD558_0009,
        0x040A_0300, // v_med3_i32 v9, v0, v1, v2
        0xD56C_000A,
        0x0002_0300, // v_mul_hi_i32 v10, v0, v1
        0xF800_180F,
        0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
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
        0xF11C_0128,
        0x0040_0200, // image_gather4_lz v[2:5], v[0:2], s[0:7], s[8:11] dmask:0x1 dim:2D_ARRAY
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_180F,
        0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
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
        0xF800_180F,
        0x0302_0100, // exp mrt0 v0, v1, v2, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let mut req = Request {
        stage: Stage::Pixel,
        code: &code,
        address: 0,
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
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
        scratch_wave_bytes: 0,
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
        0xF000_0F08,
        0x0000_0200, // image_load v[2:5], v[0:1], s[0:7] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF020_0F08,
        0x0002_0200, // image_store v[2:5], v[0:1], s[8:15] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let request = |uint_textures, sint_textures| Request {
        stage: Stage::Compute,
        code: &code,
        address: 0,
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
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
        scratch_wave_bytes: 0,
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
    let mut words = vec![
        u32::from_le_bytes(*b"g10c"),
        1,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        64,
        1,
        1,
        0,
    ];
    words.extend_from_slice(&code);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let c = gfx10::capture::Capture::from_bytes(&bytes).unwrap();
    assert_eq!(
        (c.threads, c.uint_textures, c.code.as_slice()),
        ([64, 1, 1], 0, &code[..])
    );
    let again = gfx10::capture::Capture::from_bytes(&c.to_bytes()).unwrap();
    assert_eq!((again.threads, again.code), (c.threads, c.code));
}

#[test]
fn captures_keep_the_grid_bounds_and_depth_export_flags() {
    let code = [0xBF81_0000u32];
    let req = Request {
        stage: Stage::Compute,
        code: &code,
        address: 0x1000,
        user_sgprs: 0,
        ngg: false,
        ps_input_ena: 0,
        ps_input_addr: 0,
        ps_input_cntl: &[],
        col_format: 0,
        compute_rsrc2: 0,
        threads: [64, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
        grid_bounds: true,
        drop_depth_export: false,
        half_targets: 5,
        logic_op: 0x88,
        logic_targets: [0, 0x48, 0, 0, 0, 0, 0, 8],
        swizzles: [0x1B, 0, 0, 0, 0, 0, 0, 0x93],
        probe_flat: false,
        scratch_wave_bytes: 0,
    };
    let bytes = gfx10::capture::Capture::new(&req).to_bytes();
    let c = gfx10::capture::Capture::from_bytes(&bytes).unwrap();
    let again = c.request();
    assert_eq!(
        (
            again.grid_bounds,
            again.drop_depth_export,
            again.half_targets,
            again.logic_op,
            again.logic_targets,
            again.swizzles,
            again.address,
            again.code
        ),
        (
            true,
            false,
            5,
            0x88,
            [0, 0x48, 0, 0, 0, 0, 0, 8],
            [0x1B, 0, 0, 0, 0, 0, 0, 0x93],
            0x1000,
            &code[..]
        )
    );
}

#[test]
fn sdwa_writes_to_a_word_keep_or_clear_the_rest() {
    let code = [
        0x7E00_0200, // v_mov_b32 v0, s0
        0x7E02_0201, // v_mov_b32 v1, s1
        0x7E02_02F9, // v_mov_b32_sdwa v1, v0 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
        0x0004_1500,
        0x7E04_02F9, // v_mov_b32_sdwa v2, v0 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD
        0x0006_0100,
        0xBF82_0000, // s_branch 1 (a branch makes the body a switch)
        0xF800_180F, // exp mrt0 v0, v1, v2, v3 done vm
        0x0302_0100,
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 2);
    assert!(t.msl.contains("switch"), "{}", t.msl);
    compile(&t.msl);
    assert!(
        t.msl
            .contains("v1 = ((sdwa_old2 & ~(0xffffu << 16u)) | ((v1 & 0xffffu) << 16u));"),
        "{}",
        t.msl
    );
    assert!(t.msl.contains("v2 = ((v2 & 0xffu) << 8u);"), "{}", t.msl);
}

#[test]
fn gds_appends_count_through_the_bound_global_data_share() {
    let code = [
        0xD8FA_0008,
        0x0100_0000, // ds_append v1 offset:8 gds
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 0, 0);
    assert!(t.gds);
    assert!(t.msl.contains("atomic_fetch_add_explicit(&gds["));
    assert!(t.msl.contains("(m0 >> 16) + 8u"), "{}", t.msl);
    compile(&t.msl);
}

#[test]
fn bvh_intersection_reads_nodes_through_guest_memory() {
    let code = [
        0xF198_9F01,
        0x0000_0004, // image_bvh_intersect_ray v[0:3], v[4:14], s[0:3]
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 4, 0);
    assert!(
        t.msl.contains(
            "uint4 r = bvh_intersect(mem, mem_base, uint4(s0, s1, s2, s3), ulong(v4), F(v5)"
        ),
        "{}",
        t.msl
    );
    assert!(t.msl.contains("v3 = r[3];"));
    compile(&t.msl);
}

#[test]
fn wave32_masks_are_one_sgpr() {
    let code = [
        0xBE80_3C6A, // s_and_saveexec_b32 s0, vcc_lo
        0xD4C2_0002,
        0x0002_0300, // v_cmp_eq_u32_e64 s2, v0, v1
        0xD70F_0402,
        0x0002_0300, // v_add_co_u32 v2, s4, v0, v1
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 0, 0);
    assert!(!t.msl.contains("s3 = 0u"), "{}", t.msl);
    assert!(!t.msl.contains("s5 = 0u"), "{}", t.msl);
    compile(&t.msl);
}

#[test]
fn one_wave64_group_talking_through_lds_runs_in_step() {
    let code = [
        0x7E02_0280, // v_mov_b32_e32 v1, 0
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 1
        0xD8D8_0004,
        0x0200_0001, // ds_read_b32 v2, v1 offset:4
        0x7D84_0480, // v_cmp_eq_u32_e32 vcc, 0, v2
        0xBF86_0001, // s_cbranch_vccz 1
        0xBF80_0000, // s_nop 0
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 1, 0);
    assert!(t.msl.contains("wave_any(scc, wv)"), "{}", t.msl);
    assert!(
        t.msl.contains("wave_any(((vcc_lo | vcc_hi) != 0u), wv)"),
        "{}",
        t.msl
    );
    assert!(t.msl.contains("wave_ballot("), "{}", t.msl);
    assert!(t.msl.matches("wave_meet(wv);").count() >= 4);
    compile(&t.msl);
}

#[test]
fn getpc_gives_the_address_of_the_next_instruction() {
    let code = [
        0xBE80_1F00, // s_getpc_b64 s[0:1]
        0xBF81_0000, // s_endpgm
    ];
    let mut req = Request {
        stage: Stage::Compute,
        code: &code,
        address: 0x5_0000_1000,
        user_sgprs: 0,
        ngg: false,
        ps_input_ena: 0,
        ps_input_addr: 0,
        ps_input_cntl: &[],
        col_format: 0,
        compute_rsrc2: 0,
        threads: [64, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
        scratch_wave_bytes: 0,
    };
    let t = translate(&req).unwrap();
    assert!(t.msl.contains("0x500001004ul"));
    req.address = 0;
    assert!(translate(&req).unwrap().msl.contains("0x4ul"));
}

#[test]
fn sixty_four_bit_integer_ops_compile() {
    let code = [
        0xD576_0402,
        0x040A_0300, // v_mad_u64_u32 v[2:3], s[4:5], v0, v1, v[2:3]
        0x7DC8_0902, // v_cmp_gt_u64_e32 vcc, v[2:3], v[4:5]
        0x9282_0504, // s_bfm_b64 s[2:3], s4, s5
        0xBE82_1D04, // s_bitset1_b32 s2, s4
        0xBF81_0000, // s_endpgm
    ];
    compile(&compute(&code, 6, 6 << 1).msl);
}

#[test]
fn quad_masks_and_sixty_four_bit_bit_scans_compile() {
    let code = [
        0xBE82_2D04, // s_quadmask_b64 s[2:3], s[4:5]
        0xBE82_2C04, // s_quadmask_b32 s2, s4
        0xBE82_1404, // s_ff1_i32_b64 s2, s[4:5]
        0xBE82_1004, // s_bcnt1_i32_b64 s2, s[4:5]
        0xBE82_1604, // s_flbit_i32_b64 s2, s[4:5]
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 6, 6 << 1);
    assert!(t.msl.contains("quadmask64("));
    compile(&t.msl);
}

#[test]
fn sixty_four_bit_shifts_alignbyte_subb_and_setreg_compile() {
    let code = [
        0xD6FF_0002,
        0x0002_0900, // v_lshlrev_b64 v[2:3], v0, v[4:5]
        0xD700_0002,
        0x0002_0883, // v_lshrrev_b64 v[2:3], 3, v[4:5]
        0xD701_0002,
        0x0002_0900, // v_ashrrev_i64 v[2:3], v0, v[4:5]
        0xD54F_0001,
        0x0412_0702, // v_alignbyte_b32 v1, v2, v3, v4
        0x8282_0504, // s_subb_u32 s2, s4, s5
        0xB982_1801, // s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), s2
        0xBF81_0000, // s_endpgm
    ];
    compile(&compute(&code, 6, 6 << 1).msl);
}

#[test]
fn typed_buffer_stores_and_d16_image_stores_pack_their_data() {
    let code = [
        0xE014_2000,
        0x8000_0200, // buffer_store_format_xy v[2:3], v0, s[0:3], 0 idxen
        0xF020_0308,
        0x8002_0102, // image_store v1, v[2:3], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D d16
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 16, 16 << 1);
    assert!(t.msl.contains("buffer_store_format("));
    assert!(t.msl.contains("as_type<half>(ushort((v1 >> 16u)))"));
    compile(&t.msl);
}

#[test]
fn depth_compare_samples_read_the_compare_function_from_the_sampler() {
    let code = [
        0xF0BC_0108,
        0x0082_0102, // image_sample_c_lz v1, v[2:4], s[8:15], s[16:19] dmask:0x1
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 20, 20 << 1);
    assert!(t.msl.contains("depth_compare((user[32] >> 12) & 7u"));
    compile(&t.msl);
}

#[test]
fn explicit_derivative_samples_become_gradient_samples() {
    let code = [
        0xF088_0F08,
        0x0082_0000, // image_sample_d v[0:3], v[0:5], s[8:15], s[16:19] dmask:0xf dim:2D
        0xF08C_0F08,
        0x0082_0000, // image_sample_d_cl v[0:3], v[0:6], s[8:15], s[16:19] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 20, 20 << 1);
    assert!(
        t.msl
            .contains("gradient2d(float2(F(v0), F(v1)), float2(F(v2), F(v3)))")
    );
    assert!(t.msl.contains("float2(F(v4), F(v5))"));
    assert!(t.msl.contains("min_lod_clamp(F(v6))"));
    compile(&t.msl);
}

#[test]
fn sixteen_bit_compares_read_the_low_halves() {
    let code = [
        0x7D32_0300, // v_cmpx_lt_i16_e32 v0, v1
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 0, 0);
    assert!(t.msl.contains("int(short(v0)) < int(short(v1))"));
    compile(&t.msl);
}

#[test]
fn bit_replication_doubles_each_bit() {
    let code = [
        0xBE82_3B00, // s_bitreplicate_b64_b32 s[2:3], s0
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 0, 0);
    assert!(t.msl.contains("* 3ul << (2u * i)"));
    compile(&t.msl);
}

#[test]
fn flat_and_global_stores_write_guest_memory() {
    let code = [
        0xDC70_0000,
        0x007D_0200, // flat_store_dword v[0:1], v2
        0xDC74_8010,
        0x0004_0200, // global_store_dwordx2 v0, v[2:3], s[4:5] offset:16
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 0, 0);
    assert!(t.writes_memory);
    assert!(
        t.msl
            .contains("flat_store_offset(mem_base, (ulong(v0) | (ulong(v1) << 32))")
    );
    assert!(t.msl.contains("*(device uint*)(mem + o + 4) = v3;"));
    compile(&t.msl);
}

#[test]
fn packed_sixteen_bit_adds_pick_halves_by_op_sel() {
    let code = [
        0xCC02_4001,
        0x1802_02FF,
        0x0004_0100, // v_pk_add_i16 v1, 0x40100, v1
        0xCC03_4802,
        0x1002_0903, // v_pk_sub_i16 v2, v3, v4 op_sel:[1,0] op_sel_hi:[0,1]
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 0, 0);
    assert!(
        t.msl
            .contains("(((0x40100u & 0xffffu) + (v1 & 0xffffu)) & 0xffffu)")
    );
    assert!(t.msl.contains(
        "v2 = (((v3 >> 16) - (v4 & 0xffffu)) & 0xffffu) | ((((v3 & 0xffffu) - (v4 >> 16)) & 0xffffu) << 16);"
    ));
    compile(&t.msl);
}

#[test]
fn quad_permutations_shuffle_within_the_quad() {
    let code = [
        0x7E02_02FA,
        0xFF00_B100, // v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2]
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 0, 0);
    assert!(t.msl.contains("quad_shuffle(") && t.msl.contains("[[thread_index_in_quadgroup]]"));
    compile(&t.msl);
}

#[test]
fn sampler_adjust_samples_decode_as_their_base_form() {
    assert_eq!(
        text(&[0xF080_0109, 0x00C0_0904]),
        text(&[0xF080_0108, 0x00C0_0904])
    );
    assert_eq!(
        text(&[0xF0A0_0109, 0x00C0_0904]),
        text(&[0xF0A0_0108, 0x00C0_0904])
    );
}

#[test]
fn samplers_from_indexed_buffer_loads_and_constants_resolve() {
    let code = [
        0x8F6A_8402, // s_lshl_b32 vcc_lo, s2, 4
        0xF428_0202,
        0xD400_0000, // s_buffer_load_dwordx4 s[8:11], s[4:7], vcc_lo
        0xBE8E_04FF,
        0x0650_0000, // s_mov_b64 s[14:15], 0x6500000
        0xB00C_0092, // s_movk_i32 s12, 0x92
        0x920D_8C8C, // s_bfm_b32 s13, 12, 12
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF080_0F08,
        0x0044_0000, // image_sample v[0:3], v[0:1], s[16:23], s[8:11] dmask:0xf dim:2D
        0xF080_0F08,
        0x0064_0400, // image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 24, 24 << 1);
    let buffer = Source::Buffer {
        buffer: Box::new(Source::UserSgpr(4)),
        offset: 0,
        index: Some((Box::new(Source::UserSgpr(2)), 4)),
    };
    assert!(t.resources.contains(&Resource::Sampler {
        slot: 0,
        source: buffer
    }));
    let constant = Source::Constant(vec![0x92, 0xFF_F000, 0x650_0000, 0]);
    assert!(t.resources.contains(&Resource::Sampler {
        slot: 1,
        source: constant
    }));
    compile(&t.msl);
}

#[test]
fn samplers_indexed_by_a_buffer_dword_resolve() {
    let code = [
        0xF420_1A82,
        0xFA00_0000, // s_buffer_load_dword vcc_lo, s[4:7], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x8F6A_846A, // s_lshl_b32 vcc_lo, vcc_lo, 4
        0xF428_0202,
        0xD400_0000, // s_buffer_load_dwordx4 s[8:11], s[4:7], vcc_lo
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF080_0F08,
        0x0044_0000, // image_sample v[0:3], v[0:1], s[16:23], s[8:11] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 24, 24 << 1);
    let dword = Source::Buffer {
        buffer: Box::new(Source::UserSgpr(4)),
        offset: 0,
        index: None,
    };
    let buffer = Source::Buffer {
        buffer: Box::new(Source::UserSgpr(4)),
        offset: 0,
        index: Some((Box::new(dword), 4)),
    };
    assert!(t.resources.contains(&Resource::Sampler {
        slot: 0,
        source: buffer
    }));
    compile(&t.msl);
}

#[test]
fn a_descriptor_dword_masked_with_constants_through_another_register_keeps_its_origin() {
    let code = [
        0x876A_FF17,
        0x0FFF_FFFF, // s_and_b32 vcc_lo, s23, 0xfffffff
        0x8817_FF6A,
        0xD000_0000, // s_or_b32 s23, vcc_lo, 0xd0000000
        0xF080_0F08,
        0x0044_0000, // image_sample v[0:3], v[0:1], s[16:23], s[8:11] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 24, 24 << 1);
    assert!(t.resources.iter().any(|r| matches!(
        r,
        Resource::Texture {
            source: Source::UserSgpr(16),
            ..
        }
    )));
    compile(&t.msl);
}

#[test]
fn one_sampler_copied_into_other_registers_takes_one_slot() {
    let code = [
        0xBE98_0308, // s_mov_b32 s24, s8
        0xBE99_0309, // s_mov_b32 s25, s9
        0xBE9A_030A, // s_mov_b32 s26, s10
        0xBE9B_030B, // s_mov_b32 s27, s11
        0xF080_0F08,
        0x0044_0000, // image_sample v[0:3], v[0:1], s[16:23], s[8:11] dmask:0xf dim:2D
        0xF080_0F08,
        0x00C4_0400, // image_sample v[4:7], v[0:1], s[16:23], s[24:27] dmask:0xf dim:2D
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 24, 24 << 1);
    let samplers: Vec<_> = t
        .resources
        .iter()
        .filter(|r| matches!(r, Resource::Sampler { .. }))
        .collect();
    assert_eq!(
        samplers,
        [&Resource::Sampler {
            slot: 0,
            source: Source::UserSgpr(8)
        }]
    );
    compile(&t.msl);
}

#[test]
fn flat_and_global_loads_read_guest_memory() {
    let code = [
        0xDC20_0000,
        0x007D_0001, // flat_load_ubyte v0, v[1:2]
        0xDC30_8010,
        0x0300_0004, // global_load_dword v3, v4, s[0:1] offset:16
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 2, 2 << 1);
    assert!(t.guest_memory);
    assert!(
        t.msl
            .contains("flat_load_ubyte(mem, goff(mem_base, (ulong(v1) | (ulong(v2) << 32))")
    );
    assert!(t.msl.contains("flat_load_raw(mem, goff(mem_base, ((ulong(s0) | (ulong(s1) << 32)) + ulong(v4)) + ulong(long(16))"));
    assert!(t.msl.contains("constant Win& mem_base [[buffer(28)]]"));
    compile(&t.msl);
}

#[test]
fn typed_buffer_loads_take_the_instruction_format_and_fma_mix_reads_halves() {
    let code = [
        0xEA6B_2000,
        0x8000_0200, // tbuffer_load_format_xyzw v[2:5], v0, s[0:3], 0 format:[BUF_FMT_32_32_32_32_FLOAT] idxen
        0xCC20_4006,
        0x0C12_0702, // v_fma_mix_f32 v6, v2, v3, v4 op_sel_hi:[1,0,1]
        0xCC21_C807,
        0x5BCA_0702, // v_fma_mixlo_f16 v7, v2, -v3, 1.0 op_sel:[1,0,0] op_sel_hi:[1,1,1] clamp
        0xCC22_0107,
        0x1412_0702, // v_fma_mixhi_f16 v7, |v2|, v3, v4 op_sel_hi:[0,1,0]
        0xE078_2000,
        0x8000_0400, // buffer_store_dwordx4 v[4:7], v0, s[0:3], 0 idxen
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 4, 4 << 1);
    assert!(t.msl.contains(&format!("& ~0x7f000u) | {}u)", 77 << 12)));
    assert!(
        t.msl
            .contains("fma(float(as_type<half2>(v2).x), F(v3), float(as_type<half2>(v4).x))")
    );
    assert!(
        t.msl.contains(
            "saturate(fma(float(as_type<half2>(v2).y), (-float(as_type<half2>(v3).x)), F("
        )
    );
    assert!(
        t.msl
            .contains("fma(fabs(F(v2)), float(as_type<half2>(v3).x), F(v4))")
    );
    compile(&t.msl);
}

#[test]
fn indexed_vgpr_moves_switch_over_every_vgpr() {
    let code = [
        0xBEFC_0300, // s_mov_b32 m0, s0
        0x7E00_8704, // v_movrels_b32_e32 v0, v4
        0x7E0A_8501, // v_movreld_b32_e32 v5, v1
        0x7E0C_8904, // v_movrelsd_b32_e32 v6, v4
        0x7E0C_9104, // v_movrelsd_2_b32_e32 v6, v4
        0x7E0C_D104, // v_swaprel_b32 v6, v4
        0xF800_180F,
        0x0302_0100, // exp mrt0 v0, v1, v2, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 1);
    assert!(t.msl.contains("switch (m0 + 4u) { case 0: rv = v0; break;"));
    assert!(t.msl.contains("case 6: v6 = wv; break;"));
    assert!(t.msl.contains("switch (((m0 >> 16) & 0x3ffu) + 6u)"));
    assert!(!t.msl.contains("@"));
    compile(&t.msl);
}

#[test]
fn a_program_may_end_with_a_branch_back() {
    let code = [
        0xBF85_0002, // s_cbranch_scc1 2 (to 3)
        0xBF81_0000, // s_endpgm
        0xBF81_0000, // s_endpgm
        0xBEFC_0300, // s_mov_b32 m0, s0
        0xBF82_FFFC, // s_branch -4 (to 1)
        0x3030_6C73, // the trailer, not an instruction
    ];
    assert_eq!(gfx10::translate::program(&code).unwrap().len(), 5);
}

#[test]
fn wave32_saveexec_xad_class_addtid_and_image_atomics() {
    let code = [
        0xBE82_3C03, // s_and_saveexec_b32 s2, s3
        0xBE82_4403, // s_andn1_saveexec_b32 s2, s3
        0xD745_0001,
        0x0412_0702, // v_xad_u32 v1, v2, v3, v4
        0x7D10_0501, // v_cmp_class_f32_e32 vcc, v1, v2
        0xD488_0004,
        0x0001_0701, // v_cmp_class_f32_e64 s[4:5], v1, 3
        0xDAC0_0010,
        0x0000_0000, // ds_write_addtid_b32 v0 offset:16
        0xDAC4_0010,
        0x0100_0000, // ds_read_addtid_b32 v1 offset:16
        0xF044_2108,
        0x0002_0104, // image_atomic_add v1, v[4:5], s[8:15] dmask:0x1 dim:2D glc
        0xF03C_0108,
        0x0002_0204, // image_atomic_swap v2, v[4:5], s[8:15] dmask:0x1 dim:2D
        0xF040_2308,
        0x0002_0204, // image_atomic_cmpswap v[2:3], v[4:5], s[8:15] dmask:0x3 dim:2D glc
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&code, 16, 16 << 1);
    assert!(t.msl.contains("uint old = exec_lo; uint n = s3 & old;"));
    assert!(t.msl.contains("uint n = ~s3 & old;"));
    assert!(t.msl.contains("(v2 ^ v3) + v4"));
    assert!(t.msl.contains("fclass(F(v1), v2)"));
    assert!(
        t.msl
            .contains("4u * ((lid.x + lid.y * 64u + lid.z * 64u) & 63u)")
    );
    assert!(t.msl.contains("texture2d<uint, access::read_write> t0"));
    assert!(
        t.msl
            .contains("t0.atomic_fetch_add(uint2(v4, v5), uint4(v1)); v1 = (r.x);")
    );
    assert!(t.msl.contains("atomic_compare_exchange_weak"));
    assert!(matches!(
        t.resources[0],
        Resource::Texture {
            access: Access::Atomic,
            ..
        }
    ));
    compile(&t.msl);
}

#[test]
fn pixel_shader_addtid_spills_use_per_thread_local_slots() {
    let code = [
        0xBEFC_0380, // s_mov_b32 m0, 0
        0x7E00_0200, // v_mov_b32 v0, s0
        0xDAC0_0100,
        0x0000_0000, // ds_write_addtid_b32 v0 offset:256
        0xDAC4_0100,
        0x0100_0000, // ds_read_addtid_b32 v1 offset:256
        0xF800_180F,
        0x0101_0101, // exp mrt0 v1, v1, v1, v1 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 1);
    assert_eq!(t.lds_bytes, 0);
    assert!(t.msl.contains("#define ADDTID_SHIFT 7u"));
    assert!(t.msl.contains("uint addtid_lds[ADDTID_SLOTS] = {};"));
    assert!(t.msl.contains("addtid_lds[a] = v0;"));
    assert!(
        t.msl
            .contains("v1 = a < ADDTID_SLOTS ? addtid_lds[a] : 0u;")
    );
    compile(&t.msl);
}

#[test]
fn row_dpp_permlanes_readlane_and_bpermute_use_simd_shuffles() {
    let code = [
        0x3814_14FA,
        0xFF01_110A, // v_or_b32_dpp v10, v10, v10 row_shr:1 row_mask:0xf bank_mask:0xf
        0x7E02_02FA,
        0xFF09_4002, // v_mov_b32_dpp v1, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
        0x7E02_02FA,
        0x3F01_6302, // v_mov_b32_dpp v1, v2 row_xmask:3 row_mask:0x3 bank_mask:0xf
        0x0602_06FA,
        0x5F00_B102, // v_add_f32_dpp v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0x5 bank_mask:0xf
        0x7E02_02E9,
        0x0539_7702, // v_mov_b32_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
        0xD778_0009,
        0x0305_830A, // v_permlanex16_b32 v9, v10, -1, -1
        0xD777_0009,
        0x0008_030A, // v_permlane16_b32 v9, v10, s1, s2
        0xD760_0000,
        0x0001_3F0A, // v_readlane_b32 s0, v10, 31
        0xDACC_0004,
        0x0100_0302, // ds_bpermute_b32 v1, v2, v3 offset:4
        0xF800_180F,
        0x0302_0100, // exp mrt0 v0, v1, v2, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 3);
    assert!(t.msl.contains("ushort slane [[thread_index_in_simdgroup]]"));
    assert!(t.msl.contains("simd_shuffle(v10, ushort((i - 1u) & 31u))"));
    assert!(t.msl.contains("dpp_on = dpp_on && dpp_in;"));
    assert!(t.msl.contains("if (!dpp_in) dpp0 = 0u;"));
    assert!(t.msl.contains("((slane & 16u) ^ 16u)"));
    assert!(t.msl.contains("s0 = simd_broadcast(v10, ushort(31));"));
    assert!(
        t.msl
            .contains("simd_shuffle(v3, ushort(((v2 + 4u) >> 2) & 31u))")
    );
    compile(&t.msl);
}

#[test]
fn pixel_loops_give_up_after_too_many_branches_back() {
    let code = [
        0x7E00_0200, // v_mov_b32 v0, s0
        0xBF84_FFFE, // s_cbranch_scc0 -2 (to 0)
        0x7E02_0201, // v_mov_b32 v1, s1
        0xF800_180F,
        0x0302_0100, // exp mrt0 v0, v1, v2, v3 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = pixel(&code, 2);
    let limit = gfx10::translate::MAX_BACK_BRANCHES;
    assert!(t.msl.contains(&format!(
        "if (!scc) {{ if (++back > {limit}u) {{ discard_fragment(); return o; }} continue; }}"
    )));
    assert!(t.msl.contains("for (;;) {") && !t.msl.contains("pc = 0"));
    compile(&t.msl);
}

#[test]
fn branches_become_nested_ifs_unless_one_leaves_its_region() {
    let if_else = [
        0xBF06_8005, // s_cmp_eq_u32 s5, 0
        0xBF84_0002, // s_cbranch_scc0 (to else)
        0xBE85_0387, // s_mov_b32 s5, 7
        0xBF82_0001, // s_branch (to end)
        0xBE85_0389, // else: s_mov_b32 s5, 9
        0xBF81_0000, // end: s_endpgm
    ];
    let t = compute(&if_else, 6, 0);
    assert!(t.msl.contains("if (!(!scc)) {") && t.msl.contains("} else {"));
    assert!(!t.msl.contains("if (pc == "));
    compile(&t.msl);
    let leaving = [
        0xBF06_8005, // s_cmp_eq_u32 s5, 0
        0xBF84_0002, // s_cbranch_scc0 (to the add)
        0xBF85_0002, // s_cbranch_scc1 (past the add)
        0xBE85_0385, // s_mov_b32 s5, 5
        0x8005_8A05, // s_add_u32 s5, s5, 10
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&leaving, 6, 0);
    assert!(t.msl.contains("if (pc == "));
    compile(&t.msl);
}

#[test]
fn compute_loops_give_up_alone_or_together_at_a_barrier() {
    let looping = [
        0x7E00_0200, // v_mov_b32 v0, s0
        0xBF84_FFFE, // s_cbranch_scc0 -2 (to 0)
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&looping, 1, 0);
    assert!(t.loop_guard);
    assert!(
        t.msl
            .contains("if (!scc) { if (guard_stop(guard, ++back)) { return; } continue; }")
    );
    assert!(t.msl.contains("device atomic_uint* guard [[buffer(25)]]"));
    compile(&t.msl);
    let with_barrier = [
        0x7E00_0200, // v_mov_b32 v0, s0
        0xBF8A_0000, // s_barrier
        0xBF84_FFFD, // s_cbranch_scc0 -3 (to 0)
        0xBF81_0000, // s_endpgm
    ];
    let t = compute(&with_barrier, 1, 0);
    assert!(
        t.msl
            .contains("guard_leave(guard, ++back, bail_at, tg_bail)")
    );
    assert!(t.msl.contains(
        "bool leave = tg_bail; threadgroup_barrier(mem_flags::mem_threadgroup); if (leave) return;"
    ));
    compile(&t.msl);
    let straight = [0x7E00_0200, 0xBF81_0000];
    let t = compute(&straight, 1, 0);
    assert!(!t.loop_guard && !t.msl.contains("guard"));
}

#[test]
fn a_sample_mask_export_discards_without_writing_depth_or_a_mask() {
    let code = [
        0xF800_0884, // exp mrtz off, off, v9, off done
        0x0009_0000,
        0xBF81_0000, // s_endpgm
    ];
    assert_eq!(text(&code[..2]), "exp mrtz off, off, v9, off done");
    let t = pixel(&code, 0);
    assert!(!t.writes_depth);
    assert!(!t.msl.contains("sample_mask"));
    assert!(t.msl.contains("if ((v9 & 1u) == 0u) discard_fragment();"));
    compile(&t.msl);
}

#[test]
fn waves_of_a_bigger_group_that_cross_halves_meet_pairwise() {
    let code = [
        0xD880_0000,
        0x0300_0504, // ds_add_rtn_u32 v3, v4, v5
        0x92FE_A0A0, // s_bfm_b64 exec, 32, 32
        0xD760_0008,
        0x0001_3F02, // v_readlane_b32 s8, v2, 31
        0xBEFE_04C1, // s_mov_b64 exec, -1
        0xBF8A_0000, // s_barrier
        0xBF84_FFF9, // s_cbranch_scc0 -7 (to 0)
        0xBF81_0000, // s_endpgm
    ];
    let t = compute_group(&code, 1, 0, 256);
    assert!(t.msl.contains("#define LOCKSTEP_WAVES 4u"), "{}", t.msl);
    assert!(t.msl.contains("wave_readlane(v2, 31u, wv)"), "{}", t.msl);
    assert!(t.msl.contains("if ((wv.sgi & 1u) == 0u) {"), "{}", t.msl);
    assert!(
        t.msl.contains("lockstep_leave(guard, ++back, wv, tg_bail)"),
        "{}",
        t.msl
    );
    compile(&t.msl);
    let plain = compute_group(
        &[
            0xD834_0000,
            0x0000_0804, // ds_write_b32 v4, v8
            0xBF81_0000, // s_endpgm
        ],
        1,
        0,
        256,
    );
    assert!(!plain.msl.contains("LOCKSTEP_WAVES"));
}
