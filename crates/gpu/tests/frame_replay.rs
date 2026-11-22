//! End-to-end GPU frame capture, deterministic offline replay, pass timing
//! and golden-image regression tests.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use gpu::capture::{FrameCapture, FrameRecorder, read_command_buffer};
use gpu::metal::{Metal, read_rgba};
use gpu::regs::{context, sh, uconfig};
use gpu::replay::{
    ReplayOptions, compare_rgba_to_ppm, read_image, read_png, read_ppm, replay, write_png,
    write_ppm,
};
use gpu::{Flip, Host, Processor};

const IT_NOP: u32 = 0x10;
const IT_DRAW_INDEX_AUTO: u32 = 0x2D;
const IT_SET_CONTEXT_REG: u32 = 0x69;
const IT_SET_SH_REG: u32 = 0x76;
const IT_SET_UCONFIG_REG: u32 = 0x79;
const NOP_FLIP: u32 = 0x17;

fn pkt(op: u32, code: u32, payload: &[u32]) -> Vec<u32> {
    let len = payload.len() as u32 + 1;
    let header = 0xC000_0000 | ((len - 2) << 16) | (op << 8) | (code << 2);
    let mut out = Vec::with_capacity(payload.len() + 1);
    out.push(header);
    out.extend_from_slice(payload);
    out
}

fn set_reg(op: u32, reg: usize, values: &[u32]) -> Vec<u32> {
    let mut payload = Vec::with_capacity(values.len() + 1);
    payload.push(reg as u32);
    payload.extend_from_slice(values);
    pkt(op, 0, &payload)
}

struct TestMem {
    base: *mut u8,
    len: u64,
    display: u64,
}

impl TestMem {
    fn new(len: u64, display: u64) -> Self {
        // SAFETY: anonymous private mapping freed in Drop.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        } as *mut u8;
        assert!(!base.is_null() && base != libc::MAP_FAILED.cast());
        Self { base, len, display }
    }

    fn write_words(&self, addr: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(self.write(addr, &bytes));
    }
}

impl Drop for TestMem {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.cast(), self.len as usize);
        }
    }
}

impl Host for TestMem {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        if addr + buf.len() as u64 > self.len {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.base.add(addr as usize),
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        true
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        if addr + data.len() as u64 > self.len {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(addr as usize), data.len());
        }
        true
    }

    fn interrupt(&self, _: u32) {}
    fn flip(&self, _: Flip) {}
    fn exiting(&self) -> bool {
        false
    }
    fn extent(&self, addr: u64) -> Option<(u64, u64)> {
        (addr < self.len).then_some((0, self.len))
    }
    fn host_pointer(&self, addr: u64) -> Option<*mut u8> {
        (addr < self.len).then(|| unsafe { self.base.add(addr as usize) })
    }
    fn display_buffer(&self, index: i32) -> Option<u64> {
        (index == 0).then_some(self.display)
    }
    fn gpu_area(&self) -> Option<(u64, u64)> {
        Some((0, self.len))
    }
    fn back(&self, _start: u64, end: u64) -> bool {
        end <= self.len
    }
}

#[test]
fn synthetic_frame_capture_and_replay_matches_expected_pixels() {
    let Some(mut metal) = Metal::new() else {
        return;
    };
    metal.set_wait_for_compiles(true);

    let target_addr: u64 = 0x4_0000;
    let vs_addr: u64 = 0x1_0000;
    let ps_addr: u64 = 0x1_4000;
    let vb_addr: u64 = 0x2_0000;
    let cb_addr: u64 = 0x3_0000;
    let (w, h) = (32u32, 32u32);

    let mem = TestMem::new(1 << 20, target_addr);

    // Vertex shader (legacy VS, v0 = vertex_id): loads float4 clip pos from V# in s[0:3].
    let vs_code: [u32; 6] = [
        0xE038_2000, // buffer_load_dwordx4 v[1:4], v0, s[0:3], 0 idxen
        0x8000_0100,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_08CF, // exp pos0 v1, v2, v3, v4 done
        0x0403_0201,
        0xBF81_0000, // s_endpgm
    ];
    // Pixel shader: exports constant RGBA from user SGPRs s[0:3].
    let ps_code: [u32; 7] = [
        0x7E00_0200, // v_mov_b32 v0, s0
        0x7E02_0201, // v_mov_b32 v1, s1
        0x7E04_0202, // v_mov_b32 v2, s2
        0x7E06_0203, // v_mov_b32 v3, s3
        0xF800_180F, // exp mrt0 v0, v1, v2, v3 done vm
        0x0302_0100,
        0xBF81_0000, // s_endpgm
    ];
    mem.write_words(vs_addr, &vs_code);
    mem.write_words(ps_addr, &ps_code);

    // Fullscreen triangle covering [-1, 1]^2.
    let verts: [[f32; 4]; 3] = [
        [-1.0, -1.0, 0.5, 1.0],
        [3.0, -1.0, 0.5, 1.0],
        [-1.0, 3.0, 0.5, 1.0],
    ];
    let vb_words: Vec<u32> = verts
        .iter()
        .flat_map(|v| v.iter().map(|f| f.to_bits()))
        .collect();
    mem.write_words(vb_addr, &vb_words);

    // Build a PM4 command stream with two draws (green then red) + flip.
    let mut stream = Vec::new();
    // Color target 0: 32x32 RGBA8Unorm at target_addr.
    stream.extend(set_reg(
        IT_SET_CONTEXT_REG,
        context::CB_COLOR0_BASE,
        &[(target_addr >> 8) as u32],
    ));
    stream.extend(set_reg(
        IT_SET_CONTEXT_REG,
        context::CB_COLOR0_INFO,
        &[10 << 2], // RGBA8 Unorm
    ));
    stream.extend(set_reg(
        IT_SET_CONTEXT_REG,
        context::CB_COLOR0_ATTRIB2,
        &[(w - 1) | ((h - 1) << 14)],
    ));
    stream.extend(set_reg(IT_SET_CONTEXT_REG, context::CB_TARGET_MASK, &[0xF]));
    stream.extend(set_reg(
        IT_SET_CONTEXT_REG,
        context::SPI_SHADER_COL_FORMAT,
        &[9], // 32_ABGR
    ));
    // Viewport covering 32x32.
    stream.extend(set_reg(
        IT_SET_CONTEXT_REG,
        context::PA_CL_VPORT_XSCALE,
        &[
            (w as f32 * 0.5).to_bits(),
            (w as f32 * 0.5).to_bits(),
            (h as f32 * 0.5).to_bits(),
            (h as f32 * 0.5).to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
        ],
    ));
    // Legacy VS + PS program addresses and user SGPRs.
    stream.extend(set_reg(
        IT_SET_SH_REG,
        sh::SPI_SHADER_PGM_LO_VS,
        &[(vs_addr >> 8) as u32, 0],
    ));
    stream.extend(set_reg(
        IT_SET_SH_REG,
        sh::SPI_SHADER_PGM_RSRC2_VS,
        &[4 << 1],
    ));
    // V# for vertex buffer: base=vb_addr, stride=16, num_records=3.
    stream.extend(set_reg(
        IT_SET_SH_REG,
        sh::SPI_SHADER_USER_DATA_VS_0,
        &[vb_addr as u32, 16 << 16, 3, 0],
    ));
    stream.extend(set_reg(
        IT_SET_SH_REG,
        sh::SPI_SHADER_PGM_LO_PS,
        &[(ps_addr >> 8) as u32, 0],
    ));
    stream.extend(set_reg(
        IT_SET_SH_REG,
        sh::SPI_SHADER_PGM_RSRC2_PS,
        &[4 << 1],
    ));
    stream.extend(set_reg(
        IT_SET_UCONFIG_REG,
        uconfig::VGT_PRIMITIVE_TYPE,
        &[4], // TriangleList
    ));
    // Draw 0: solid green (0, 1, 0, 1).
    stream.extend(set_reg(
        IT_SET_SH_REG,
        sh::SPI_SHADER_USER_DATA_PS_0,
        &[
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
        ],
    ));
    stream.extend(pkt(IT_DRAW_INDEX_AUTO, 0, &[3]));
    // Draw 1: solid red (1, 0, 0, 1).
    stream.extend(set_reg(
        IT_SET_SH_REG,
        sh::SPI_SHADER_USER_DATA_PS_0,
        &[
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
        ],
    ));
    stream.extend(pkt(IT_DRAW_INDEX_AUTO, 0, &[3]));
    // Flip display buffer 0.
    stream.extend(pkt(IT_NOP, NOP_FLIP, &[1, 0, 0, 0, 0]));

    mem.write_words(cb_addr, &stream);

    // Record the frame through FrameRecorder + Processor + Metal.
    let recorder = FrameRecorder::new(mem.gpu_area(), Vec::new());
    let mut proc = Processor::default();
    proc.backend = Some(Box::new(metal));
    recorder.note_queue(u64::MAX, || proc.saved());
    let cmd_buf = read_command_buffer(&mem, cb_addr, stream.len() as u32);
    let rec_host = recorder.recording_host(&mem);
    proc.run(&rec_host, cb_addr, stream.len() as u32).unwrap();
    proc.end_submission(&rec_host, true);
    recorder.record_submission(u64::MAX, vec![cmd_buf], true);

    // Serialize and reload the capture from bytes.
    let cap = recorder.finish();
    let mut encoded = Vec::new();
    cap.write_to(&mut encoded).unwrap();
    let loaded = FrameCapture::read_from(&encoded[..]).unwrap();

    // 1. Replay full frame with pass timing enabled: must draw 2 draws and output red.
    let full = replay(
        &loaded,
        ReplayOptions {
            time_passes: true,
            ..ReplayOptions::default()
        },
    )
    .unwrap();
    assert_eq!(full.draws.len(), 2);
    assert_eq!(full.draws[0].vs, vs_addr);
    assert_eq!(full.draws[0].ps, ps_addr);
    assert_eq!(full.shown, Some(target_addr));
    assert!(!full.timings.is_empty());
    let (_, tex) = full
        .targets
        .iter()
        .find(|(a, _)| *a == target_addr)
        .expect("color target present");
    let (rw, rh, rgba) = read_rgba(tex).expect("read back color target");
    assert_eq!((rw, rh), (w, h));
    assert_eq!(&rgba[..4], &[255, 0, 0, 255], "draw 1 painted solid red");

    // 2. Replay stopping after draw 0: target must hold draw 0's green pixels.
    let after_draw0 = replay(
        &loaded,
        ReplayOptions {
            stop_after_draw: Some(0),
            ..ReplayOptions::default()
        },
    )
    .unwrap();
    assert_eq!(after_draw0.draws.len(), 1);
    let (_, tex0) = after_draw0
        .targets
        .iter()
        .find(|(a, _)| *a == target_addr)
        .expect("color target present after draw 0");
    let (_, _, rgba0) = read_rgba(tex0).unwrap();
    assert_eq!(&rgba0[..4], &[0, 255, 0, 255], "draw 0 painted solid green");

    // 3. Verify golden PNG and PPM round-trip and mismatch detection.
    let tmp_ppm = std::env::temp_dir().join(format!(
        "ps5slop-frame-replay-test-{}.ppm",
        std::process::id()
    ));
    write_ppm(&tmp_ppm, rw, rh, &rgba).unwrap();
    let (gw, gh, mut grgb) = read_ppm(&tmp_ppm).unwrap();
    let _ = std::fs::remove_file(&tmp_ppm);
    let same = compare_rgba_to_ppm(rw, rh, &rgba, gw, gh, &grgb).unwrap();
    assert_eq!(same.max_channel_diff, 0);
    assert_eq!(same.differing_pixels, 0);

    let tmp_png = std::env::temp_dir().join(format!(
        "ps5slop-frame-replay-test-{}.png",
        std::process::id()
    ));
    write_png(&tmp_png, rw, rh, &rgba).unwrap();
    let (pw, ph, prgb) = read_png(&tmp_png).unwrap();
    let _ = std::fs::remove_file(&tmp_png);
    let same_png = compare_rgba_to_ppm(rw, rh, &rgba, pw, ph, &prgb).unwrap();
    assert_eq!(same_png.max_channel_diff, 0);
    assert_eq!(same_png.differing_pixels, 0);

    grgb[0] = 0;
    let diff = compare_rgba_to_ppm(rw, rh, &rgba, gw, gh, &grgb).unwrap();
    assert_eq!(diff.max_channel_diff, 255);
    assert_eq!(diff.differing_pixels, 1);
}

#[test]
fn local_game_captures_match_golden_images() {
    let dir = diag::config()
        .gpu
        .frame_captures
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../captures"));
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: no local captures dir at {}", dir.display());
        return;
    };
    let mut captures: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ps5frame"))
        .collect();
    captures.sort();
    if captures.is_empty() {
        eprintln!("skipped: no .ps5frame captures in {}", dir.display());
        return;
    }
    let mut checked = 0usize;
    for cap_path in captures {
        let golden_path = [
            cap_path.with_extension("golden.png"),
            cap_path.with_extension("golden.ppm"),
        ]
        .into_iter()
        .find(|p| p.is_file());
        let Some(golden_path) = golden_path else {
            continue;
        };
        let cap = FrameCapture::load(&cap_path).expect("load local frame capture");
        let res = replay(&cap, ReplayOptions::default()).expect("replay local frame capture");
        let target_addr = res
            .shown
            .or_else(|| res.targets.first().map(|(a, _)| *a))
            .expect("capture produced a render target");
        let (_, tex) = res
            .targets
            .iter()
            .find(|(a, _)| *a == target_addr)
            .expect("target found");
        let (w, h, rgba) = read_rgba(tex).expect("read back target");
        let (gw, gh, grgb) = read_image(&golden_path).expect("read golden image");
        let diff = compare_rgba_to_ppm(w, h, &rgba, gw, gh, &grgb).expect("compare images");
        assert!(
            diff.max_channel_diff <= 2,
            "{}: max channel diff {} ({}/{} differing pixels, mae {:.3})",
            cap_path.display(),
            diff.max_channel_diff,
            diff.differing_pixels,
            diff.total_pixels,
            diff.mean_abs_error
        );
        checked += 1;
    }
    eprintln!("checked {checked} local frame capture(s) against goldens");
}
