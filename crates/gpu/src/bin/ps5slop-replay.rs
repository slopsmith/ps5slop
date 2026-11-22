//! Standalone GPU frame replay tool: renders captured `.ps5frame` files to
//! PNG, lists draws, dumps render targets after any draw, times passes, checks
//! golden images and captures Metal `.gputrace` bundles without running the
//! game or hypervisor.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

/// Replays captured `.ps5frame` GPU frames offline through the Metal renderer.
#[derive(Parser)]
#[command(name = "ps5slop-replay")]
struct Cli {
    /// Skips every draw and dispatch that uses this program (vertex, pixel or compute address).
    #[arg(long, global = true, value_name = "HEX", value_parser = parse_hex)]
    skip: Vec<u64>,
    #[command(subcommand)]
    command: Command,
}

/// Output image format for multi-target dumps.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum ImageFormat {
    /// Portable Network Graphics (`.png`, default).
    #[default]
    Png,
    /// Binary Portable Pixmap (`.ppm`).
    Ppm,
}

impl ImageFormat {
    fn ext(self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Ppm => "ppm",
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Renders the final frame (or a specific draw/target) to one PNG file.
    Render {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Output image path (defaults to `<capture>.png` or `<capture>.draw<N>.png`).
        #[arg(value_name = "OUT.png")]
        output: Option<PathBuf>,
        /// Stop right after this 0-based draw index and save its render target.
        #[arg(long, value_name = "N")]
        draw: Option<usize>,
        /// Guest hex address of a specific color or depth target to save.
        #[arg(long, value_name = "HEX", value_parser = parse_hex)]
        target: Option<u64>,
    },
    /// Lists every draw and compute dispatch recorded in a capture.
    List {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
    },
    /// Dumps every color and depth target (plus `shown.png`) after a draw or the full frame.
    Dump {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Directory to write target images into (created if missing).
        out_dir: PathBuf,
        /// Stop right after this 0-based draw index.
        #[arg(long, value_name = "N")]
        draw: Option<usize>,
        /// Only write the target at this guest hex address.
        #[arg(long, value_name = "HEX", value_parser = parse_hex)]
        target: Option<u64>,
        /// Also record an Apple Metal `.gputrace` bundle (needs `METAL_CAPTURE_ENABLED=1`).
        #[arg(long, value_name = "PATH")]
        gputrace: Option<PathBuf>,
        /// Image file format to write.
        #[arg(long, value_enum, default_value_t = ImageFormat::Png)]
        format: ImageFormat,
    },
    /// Commits and times every render pass and compute dispatch, sorted by GPU time.
    Time {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Run one untimed replay first to warm Metal's pipeline cache.
        #[arg(long)]
        warmup: bool,
    },
    /// Compares a replayed target against a golden PNG or PPM image.
    Check {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Path to the golden `.png` or `.ppm` image.
        golden: PathBuf,
        /// Stop right after this 0-based draw index.
        #[arg(long, value_name = "N")]
        draw: Option<usize>,
        /// Guest hex address of the color target to compare (defaults to the shown buffer).
        #[arg(long, value_name = "HEX", value_parser = parse_hex)]
        target: Option<u64>,
        /// Maximum allowed per-channel difference (0 to 255).
        #[arg(long, value_name = "N", default_value_t = 2)]
        tolerance: u8,
        /// Overwrite `GOLDEN` with the rendered output instead of comparing.
        #[arg(long)]
        update: bool,
    },
}

fn parse_hex(s: &str) -> Result<u64, String> {
    u64::from_str_radix(s.trim().trim_start_matches("0x"), 16)
        .map_err(|_| format!("invalid hex address: {s}"))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    diag::init_logging();
    let _ = diag::config();
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cli;
        tracing::error!("ps5slop-replay requires macOS Metal");
        return ExitCode::FAILURE;
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(mut skip) = gpu::metal::SKIP.lock() {
            skip.extend(&cli.skip);
        }
    }
    #[cfg(target_os = "macos")]
    match run(cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn default_render_path(capture: &std::path::Path, draw: Option<usize>) -> PathBuf {
    match draw {
        Some(n) => capture.with_extension(format!("draw{n}.png")),
        None => capture.with_extension("png"),
    }
}

#[cfg(target_os = "macos")]
fn select_target(
    res: &gpu::replay::ReplayResult,
    requested: Option<u64>,
) -> Result<(u64, u32, u32, Vec<u8>), String> {
    use gpu::metal::{read_depth_rgba, read_rgba};

    if let Some(addr) = requested {
        if let Some((_, tex)) = res.targets.iter().find(|(a, _)| *a == addr) {
            let (w, h, rgba) =
                read_rgba(tex).ok_or_else(|| format!("cannot read color target {addr:#x}"))?;
            return Ok((addr, w, h, rgba));
        }
        if let Some((_, tex)) = res.depths.iter().find(|(a, _)| *a == addr) {
            let (w, h, rgba) = read_depth_rgba(tex)
                .ok_or_else(|| format!("cannot read depth target {addr:#x}"))?;
            return Ok((addr, w, h, rgba));
        }
        return Err(format!("target {addr:#x} not found"));
    }

    // If we stopped at a draw, prefer that draw's first color (or depth) target.
    if res.shown.is_none()
        && let Some(last) = res.draws.last()
    {
        if let Some(&(addr, _, _)) = last.color_targets.first()
            && let Some((_, tex)) = res.targets.iter().find(|(a, _)| *a == addr)
            && let Some((w, h, rgba)) = read_rgba(tex)
        {
            return Ok((addr, w, h, rgba));
        }
        if let Some((addr, _, _)) = last.depth_target
            && let Some((_, tex)) = res.depths.iter().find(|(a, _)| *a == addr)
            && let Some((w, h, rgba)) = read_depth_rgba(tex)
        {
            return Ok((addr, w, h, rgba));
        }
    }

    let addr = res
        .shown
        .or_else(|| res.targets.first().map(|(a, _)| *a))
        .ok_or("capture produced no render targets")?;
    let (_, tex) = res
        .targets
        .iter()
        .find(|(a, _)| *a == addr)
        .ok_or_else(|| format!("render target {addr:#x} not found"))?;
    let (w, h, rgba) = read_rgba(tex).ok_or_else(|| format!("cannot read target {addr:#x}"))?;
    Ok((addr, w, h, rgba))
}

#[cfg(target_os = "macos")]
fn run(cmd: Command) -> Result<(), String> {
    use gpu::capture::FrameCapture;
    use gpu::metal::{read_depth_rgba, read_rgba};
    use gpu::replay::{
        ReplayOptions, compare_rgba_to_ppm, primitive_name, read_image, replay, write_image,
        write_png, write_ppm,
    };

    match cmd {
        Command::Render {
            capture,
            output,
            draw,
            target,
        } => {
            let out_path = output.unwrap_or_else(|| default_render_path(&capture, draw));
            let cap = FrameCapture::load(&capture)?;
            let res = replay(
                &cap,
                ReplayOptions {
                    stop_after_draw: draw,
                    ..ReplayOptions::default()
                },
            )?;
            let (addr, w, h, rgba) = select_target(&res, target)?;
            write_image(&out_path, w, h, &rgba)?;
            println!(
                "wrote {} ({w}x{h}, target {addr:#x}, {} draws replayed)",
                out_path.display(),
                res.draws.len()
            );
        }
        Command::List { capture } => {
            let cap = FrameCapture::load(&capture)?;
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
                    "  draw #{:04} q={:#x} vs={:#x} ps={:#x} prim={}({:#x}) {:?} targets=[{}]{depth} scissor={:?}",
                    d.index,
                    d.queue,
                    d.vs,
                    d.ps,
                    primitive_name(d.primitive),
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
        Command::Dump {
            capture,
            out_dir,
            draw,
            target,
            gputrace,
            format,
        } => {
            std::fs::create_dir_all(&out_dir)
                .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
            let cap = FrameCapture::load(&capture)?;
            let res = replay(
                &cap,
                ReplayOptions {
                    stop_after_draw: draw,
                    metal_capture: gputrace,
                    ..ReplayOptions::default()
                },
            )?;
            let write_fmt = |path: &std::path::Path, w, h, rgba: &[u8]| match format {
                ImageFormat::Png => write_png(path, w, h, rgba),
                ImageFormat::Ppm => write_ppm(path, w, h, rgba),
            };
            let ext = format.ext();
            let mut written = 0usize;
            for (addr, tex) in &res.targets {
                if target.is_some_and(|f| f != *addr) {
                    continue;
                }
                if let Some((w, h, rgba)) = read_rgba(tex) {
                    let file = out_dir.join(format!("color-{addr:x}.{ext}"));
                    write_fmt(&file, w, h, &rgba)?;
                    written += 1;
                    if res.shown == Some(*addr) {
                        write_fmt(&out_dir.join(format!("shown.{ext}")), w, h, &rgba)?;
                    }
                }
            }
            for (addr, tex) in &res.depths {
                if target.is_some_and(|f| f != *addr) {
                    continue;
                }
                if let Some((w, h, rgba)) = read_depth_rgba(tex) {
                    let file = out_dir.join(format!("depth-{addr:x}.{ext}"));
                    write_fmt(&file, w, h, &rgba)?;
                    written += 1;
                }
            }
            println!(
                "wrote {written} target(s) to {} ({} draws replayed)",
                out_dir.display(),
                res.draws.len()
            );
        }
        Command::Time { capture, warmup } => {
            let cap = FrameCapture::load(&capture)?;
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
        Command::Check {
            capture,
            golden,
            draw,
            target,
            tolerance,
            update,
        } => {
            let cap = FrameCapture::load(&capture)?;
            let res = replay(
                &cap,
                ReplayOptions {
                    stop_after_draw: draw,
                    ..ReplayOptions::default()
                },
            )?;
            let (addr, w, h, rgba) = select_target(&res, target)?;
            if update {
                write_image(&golden, w, h, &rgba)?;
                println!("updated golden {} ({w}x{h})", golden.display());
                return Ok(());
            }
            let (gw, gh, grgb) = read_image(&golden)?;
            let diff = compare_rgba_to_ppm(w, h, &rgba, gw, gh, &grgb)?;
            println!(
                "target {addr:#x} ({w}x{h}): max_diff={} differing={}/{} mae={:.3}",
                diff.max_channel_diff,
                diff.differing_pixels,
                diff.total_pixels,
                diff.mean_abs_error
            );
            if diff.max_channel_diff > tolerance {
                return Err(format!(
                    "image differs from {} (max channel diff {} > tolerance {tolerance})",
                    golden.display(),
                    diff.max_channel_diff
                ));
            }
        }
    }
    Ok(())
}
