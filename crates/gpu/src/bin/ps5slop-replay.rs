//! Standalone GPU frame replay tool: lists draws, dumps render targets after
//! any draw, times passes, checks golden images and captures Metal `.gputrace`s
//! without running the game or hypervisor.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ps5slop-replay requires macOS Metal");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    diag::init_logging();
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "macos")]
fn usage() -> &'static str {
    "\
usage:
  ps5slop-replay list CAPTURE
  ps5slop-replay dump CAPTURE OUT_DIR [--draw N] [--target HEX_ADDR] [--gputrace PATH]
  ps5slop-replay time CAPTURE [--warmup]
  ps5slop-replay check CAPTURE GOLDEN.ppm [--draw N] [--target HEX_ADDR] [--tolerance N] [--update]"
}

#[cfg(target_os = "macos")]
fn parse_hex(s: &str) -> Result<u64, String> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16)
        .map_err(|_| format!("invalid hex address: {s}"))
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), String> {
    use gpu::capture::FrameCapture;
    use gpu::metal::{read_depth_rgba, read_rgba};
    use gpu::replay::{ReplayOptions, compare_rgba_to_ppm, read_ppm, replay, write_ppm};
    use std::path::PathBuf;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first().map(String::as_str) else {
        return Err(usage().into());
    };
    match cmd {
        "list" => {
            let path = args.get(1).ok_or_else(|| usage().to_owned())?;
            let cap = FrameCapture::load(std::path::Path::new(path))?;
            let res = replay(&cap, ReplayOptions::default())?;
            println!(
                "capture: {} submissions, {} draws, {} dispatches, shown {:?}",
                cap.submissions.len(),
                res.draws.len(),
                res.dispatches.len(),
                res.shown.map(|a| format!("{a:#x}"))
            );
            for d in &res.draws {
                let targets: Vec<String> = d
                    .color_targets
                    .iter()
                    .map(|(a, w, h)| format!("{a:#x}({w}x{h})"))
                    .collect();
                let depth = d
                    .depth_target
                    .map(|(a, w, h)| format!(" depth={a:#x}({w}x{h})"))
                    .unwrap_or_default();
                println!(
                    "  draw #{:04} q={:#x} vs={:#x} ps={:#x} prim={:#x} {:?} targets=[{}]{depth} scissor={:?}",
                    d.index,
                    d.queue,
                    d.vs,
                    d.ps,
                    d.primitive,
                    d.draw,
                    targets.join(", "),
                    d.scissor
                );
            }
            for c in &res.dispatches {
                println!(
                    "  dispatch #{:04} q={:#x} cs={:#x} groups={:?} indirect={:x?}",
                    c.index, c.queue, c.cs, c.groups, c.indirect_at
                );
            }
            println!("Metal: {}", res.summary);
        }
        "dump" => {
            let cap_path = args.get(1).ok_or_else(|| usage().to_owned())?;
            let out_dir = PathBuf::from(args.get(2).ok_or_else(|| usage().to_owned())?);
            let mut opts = ReplayOptions::default();
            let mut target_filter: Option<u64> = None;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--draw" => {
                        i += 1;
                        opts.stop_after_draw = Some(
                            args.get(i)
                                .ok_or("missing --draw N")?
                                .parse()
                                .map_err(|_| "invalid --draw N")?,
                        );
                    }
                    "--target" => {
                        i += 1;
                        target_filter =
                            Some(parse_hex(args.get(i).ok_or("missing --target ADDR")?)?);
                    }
                    "--gputrace" => {
                        i += 1;
                        opts.metal_capture =
                            Some(PathBuf::from(args.get(i).ok_or("missing --gputrace PATH")?));
                    }
                    other => return Err(format!("unknown option {other}\n{}", usage())),
                }
                i += 1;
            }
            std::fs::create_dir_all(&out_dir)
                .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
            let cap = FrameCapture::load(std::path::Path::new(cap_path))?;
            let res = replay(&cap, opts)?;
            let mut written = 0usize;
            for (addr, tex) in &res.targets {
                if target_filter.is_some_and(|f| f != *addr) {
                    continue;
                }
                if let Some((w, h, rgba)) = read_rgba(tex) {
                    let file = out_dir.join(format!("color-{addr:x}.ppm"));
                    write_ppm(&file, w, h, &rgba)?;
                    written += 1;
                    if res.shown == Some(*addr) {
                        write_ppm(&out_dir.join("shown.ppm"), w, h, &rgba)?;
                    }
                }
            }
            for (addr, tex) in &res.depths {
                if target_filter.is_some_and(|f| f != *addr) {
                    continue;
                }
                if let Some((w, h, rgba)) = read_depth_rgba(tex) {
                    let file = out_dir.join(format!("depth-{addr:x}.ppm"));
                    write_ppm(&file, w, h, &rgba)?;
                    written += 1;
                }
            }
            println!(
                "wrote {written} target(s) to {} ({} draws replayed)",
                out_dir.display(),
                res.draws.len()
            );
        }
        "time" => {
            let cap_path = args.get(1).ok_or_else(|| usage().to_owned())?;
            let warmup = args.iter().any(|a| a == "--warmup");
            let cap = FrameCapture::load(std::path::Path::new(cap_path))?;
            if warmup {
                let _ = replay(&cap, ReplayOptions::default())?;
            }
            let res = replay(
                &cap,
                ReplayOptions {
                    time_passes: true,
                    ..ReplayOptions::default()
                },
            )?;
            let mut timings = res.timings;
            timings.sort_by(|a, b| b.gpu_ms.total_cmp(&a.gpu_ms));
            let total_gpu: f64 = timings.iter().map(|t| t.gpu_ms).sum();
            let total_cpu: f64 = timings.iter().map(|t| t.cpu_ms).sum();
            println!(
                "passes: {} total, GPU {total_gpu:.2} ms, CPU {total_cpu:.2} ms",
                timings.len()
            );
            for t in &timings {
                let targets: Vec<String> = t.targets.iter().map(|a| format!("{a:#x}")).collect();
                println!(
                    "  pass #{:03} {:<26} draws={:<4} gpu={:>7.3} ms  cpu={:>7.3} ms  targets=[{}]",
                    t.index,
                    t.kind,
                    t.draws,
                    t.gpu_ms,
                    t.cpu_ms,
                    targets.join(", ")
                );
            }
        }
        "check" => {
            let cap_path = args.get(1).ok_or_else(|| usage().to_owned())?;
            let golden_path = PathBuf::from(args.get(2).ok_or_else(|| usage().to_owned())?);
            let mut opts = ReplayOptions::default();
            let mut target_addr: Option<u64> = None;
            let mut tolerance: u8 = 2;
            let mut update = false;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--draw" => {
                        i += 1;
                        opts.stop_after_draw = Some(
                            args.get(i)
                                .ok_or("missing --draw N")?
                                .parse()
                                .map_err(|_| "invalid --draw N")?,
                        );
                    }
                    "--target" => {
                        i += 1;
                        target_addr = Some(parse_hex(args.get(i).ok_or("missing --target ADDR")?)?);
                    }
                    "--tolerance" => {
                        i += 1;
                        tolerance = args
                            .get(i)
                            .ok_or("missing --tolerance N")?
                            .parse()
                            .map_err(|_| "invalid --tolerance N")?;
                    }
                    "--update" => update = true,
                    other => return Err(format!("unknown option {other}\n{}", usage())),
                }
                i += 1;
            }
            let cap = FrameCapture::load(std::path::Path::new(cap_path))?;
            let res = replay(&cap, opts)?;
            let want = target_addr
                .or(res.shown)
                .or_else(|| res.targets.first().map(|(a, _)| *a))
                .ok_or("capture produced no render targets")?;
            let (_, tex) = res
                .targets
                .iter()
                .find(|(a, _)| *a == want)
                .ok_or_else(|| format!("render target {want:#x} not found"))?;
            let (w, h, rgba) =
                read_rgba(tex).ok_or_else(|| format!("cannot read target {want:#x}"))?;
            if update {
                write_ppm(&golden_path, w, h, &rgba)?;
                println!("updated golden {} ({w}x{h})", golden_path.display());
                return Ok(());
            }
            let (gw, gh, grgb) = read_ppm(&golden_path)?;
            let diff = compare_rgba_to_ppm(w, h, &rgba, gw, gh, &grgb)?;
            println!(
                "target {want:#x} ({w}x{h}): max_diff={} differing={}/{} mae={:.3}",
                diff.max_channel_diff,
                diff.differing_pixels,
                diff.total_pixels,
                diff.mean_abs_error
            );
            if diff.max_channel_diff > tolerance {
                return Err(format!(
                    "image differs from {} (max channel diff {} > tolerance {tolerance})",
                    golden_path.display(),
                    diff.max_channel_diff
                ));
            }
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}
