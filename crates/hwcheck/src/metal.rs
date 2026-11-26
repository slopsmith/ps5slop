//! Running a case through ps5slop's own path: its command buffer through the
//! PM4 command processor into the Metal renderer, as a one-dispatch frame replay.

use std::collections::HashMap;

use gfx10::translate::{Request, Stage, translate};
use gpu::capture::{CaptureMeta, CommandBuffer, FrameCapture, Submission};
use gpu::replay::{ReplayOptions, replay};

use crate::{Case, Memory, PAGE, read_pages, write_pages};

/// Where the command buffer goes in the replayed memory.
const COMMANDS: u64 = 0x8_0000;
/// The memory replay holds: everything a case uses.
const MEMORY: (u64, u64) = (0, 0x100_0000);

/// Why the translator refuses the case's program, if it does.
pub fn untranslated(case: &Case) -> Option<String> {
    let code = case.code(case.program);
    let req = Request {
        stage: Stage::Compute,
        code: &code,
        address: case.program,
        user_sgprs: case.reg(0x213) >> 1 & 0x1F,
        ngg: false,
        ps_input_ena: 0,
        ps_input_addr: 0,
        ps_input_cntl: &[],
        col_format: 0,
        compute_rsrc2: case.reg(0x213),
        threads: [64, 1, 1],
        uint_textures: 0,
        sint_textures: 0,
        coded_textures: 0,
        grid_bounds: false,
        drop_depth_export: false,
        half_targets: 0,
        dual_source: false,
        logic_op: 0xCC,
        logic_targets: [0; 8],
        swizzles: [0; 8],
        probe_flat: false,
        scratch_wave_bytes: (case.reg(0x218) >> 12 & 0x1FFF) << 10,
        rsrc1: case.reg(0x212),
    };
    translate(&req).err().map(|e| match e.pc {
        Some(pc) => format!("at dword {pc}: {}", e.message),
        None => e.message,
    })
}

/// Runs `case` through Metal and records what it changed in `case.output`,
/// or why the translator refused it in `case.untranslated`.
pub fn run(case: &mut Case) -> Result<(), String> {
    case.output.clear();
    if case.graphics {
        return draw(case);
    }
    case.untranslated = untranslated(case);
    if case.untranslated.is_some() {
        return Ok(());
    }
    let res = replay(
        &capture(case),
        ReplayOptions {
            extract: Some(0),
            ..ReplayOptions::default()
        },
    )?;
    let e = res.extracted.ok_or("the dispatch did not run")?;
    let commands_end = COMMANDS + case.commands().len() as u64 * 4;
    case.output = e
        .after
        .into_iter()
        .filter(|&(addr, _)| addr + PAGE <= COMMANDS || addr >= commands_end)
        .collect();
    Ok(())
}

/// Runs a draw case and lays the render target Metal drew out in
/// `case.output` at the target's address, in the linear layout the
/// hardware uses (rows padded to 256 bytes).
fn draw(case: &mut Case) -> Result<(), String> {
    let res = replay(&capture(case), ReplayOptions::default())?;
    let (addr, _, height, pitch) = case.target;
    let Some((_, texture)) = res.targets.iter().find(|(a, _)| *a == addr) else {
        case.untranslated = Some(format!("Metal drew nothing ({})", res.summary));
        return Ok(());
    };
    let Some((_, h, bytes)) = gpu::metal::read_bytes(texture) else {
        case.untranslated = Some("Metal's render target format cannot be read back".into());
        return Ok(());
    };
    // Row padding keeps what memory held.
    let mut surface = vec![0u8; pitch as usize * height as usize];
    read_pages(&[&case.input], addr, &mut surface);
    let row = bytes.len() / h.max(1) as usize;
    let used = row.min(pitch as usize);
    for y in 0..height.min(h) as usize {
        surface[y * pitch as usize..][..used].copy_from_slice(&bytes[y * row..][..used]);
    }
    write_pages(&mut case.output, addr, &surface);
    Ok(())
}

/// A one-submission frame capture holding the case's memory and command buffer.
fn capture(case: &Case) -> FrameCapture {
    let commands = case.commands();
    let mut pages = case.input.clone();
    let bytes: Vec<u8> = commands.iter().flat_map(|w| w.to_le_bytes()).collect();
    write_pages(&mut pages, COMMANDS, &bytes);
    FrameCapture {
        meta: CaptureMeta {
            gpu_area: Some(MEMORY),
            extents: vec![MEMORY],
            ..CaptureMeta::default()
        },
        queues: HashMap::new(),
        textures: Vec::new(),
        submissions: vec![Submission {
            queue: 0,
            buffers: vec![CommandBuffer {
                address: COMMANDS,
                dwords: commands,
            }],
            ends_frame: false,
        }],
        pages,
        live: None,
    }
}
