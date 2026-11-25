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
    /// Prints the minimum, maximum and mean of every channel of a float or unorm target.
    Stats {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Guest hex address of the color target.
        #[arg(long, value_name = "HEX", value_parser = parse_hex)]
        target: u64,
        /// Stop right after this 0-based draw index.
        #[arg(long, value_name = "N")]
        draw: Option<usize>,
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
        /// Time each draw and dispatch alone, by program; each pays its own target loads and stores.
        #[arg(long)]
        draws: bool,
        /// Rows to print per table with `--draws`.
        #[arg(long, default_value_t = 30)]
        top: usize,
        /// Runs the frame this many times in one renderer and reports each pass's or draw's median.
        #[arg(long, default_value_t = 1)]
        runs: usize,
    },
    /// Measures the CPU time of processing and encoding the frame, as the GPU thread spends it.
    ///
    /// Runs the frame `--warmup` times untimed, then `--runs` times, in one
    /// renderer so its caches stay warm as in a game, and prints each run and
    /// the median. Loading the capture and resetting guest memory between runs
    /// are not counted; waits for command buffers and compiles are left out of encoding.
    Bench {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Timed runs.
        #[arg(long, value_name = "N", default_value_t = 5)]
        runs: usize,
        /// Untimed runs first, to warm the renderer's caches.
        #[arg(long, value_name = "N", default_value_t = 1)]
        warmup: usize,
    },
    /// Prints one texel of a target after every draw that binds the target, marking changes.
    History {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Guest hex address of the color or depth target.
        #[arg(long, value_name = "HEX", value_parser = parse_hex)]
        target: u64,
        /// Texel column.
        x: u32,
        /// Texel row.
        y: u32,
        /// Also list draws that left the texel unchanged.
        #[arg(long)]
        all: bool,
    },
    /// Prints captured guest memory as it was before the frame, as hex and as 32-bit floats.
    Mem {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Guest hex address.
        #[arg(value_parser = parse_hex)]
        address: u64,
        /// Number of bytes (decimal).
        #[arg(default_value_t = 64)]
        len: u64,
    },
    /// Compares the picture the game showed while capturing with the replay of the same frame.
    Live {
        /// Path to the `.ps5frame` capture file.
        capture: PathBuf,
        /// Where to save the live picture (defaults to `<capture>.live.png`).
        out: Option<PathBuf>,
        /// Also save the replay with the pixels that differ from the live picture in red.
        #[arg(long, value_name = "PNG")]
        diff: Option<PathBuf>,
        /// Largest per-channel difference not counted as differing (0 to 255).
        #[arg(long, value_name = "N", default_value_t = 8)]
        tolerance: u8,
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
        Command::Stats {
            capture,
            target,
            draw,
        } => {
            let cap = FrameCapture::load(&capture)?;
            let res = replay(
                &cap,
                ReplayOptions {
                    stop_after_draw: draw,
                    ..ReplayOptions::default()
                },
            )?;
            let (_, tex) = res
                .targets
                .iter()
                .find(|(a, _)| *a == target)
                .ok_or_else(|| format!("target {target:#x} not found"))?;
            let (w, h, channels, values) = gpu::metal::read_values(tex)
                .ok_or_else(|| format!("cannot read the values of target {target:#x}"))?;
            println!("target {target:#x} {w}x{h}, {channels} channel(s)");
            for c in 0..channels {
                let (mut min, mut max, mut sum, mut bad) = (f32::MAX, f32::MIN, 0f64, 0u64);
                for v in values.iter().skip(c).step_by(channels) {
                    if v.is_finite() {
                        min = min.min(*v);
                        max = max.max(*v);
                        sum += f64::from(*v);
                    } else {
                        bad += 1;
                    }
                }
                let finite = (w as u64 * h as u64).saturating_sub(bad).max(1);
                println!(
                    "  channel {c}: min {min} max {max} mean {} non-finite {bad}",
                    sum / finite as f64
                );
            }
        }
        Command::History {
            capture,
            target,
            x,
            y,
            all,
        } => {
            let cap = FrameCapture::load(&capture)?;
            let res = replay(
                &cap,
                ReplayOptions {
                    history: Some((target, x, y)),
                    ..ReplayOptions::default()
                },
            )?;
            let Some(first) = res.history.first() else {
                return Err(format!("no draw binds target {target:#x}"));
            };
            println!("target {target:#x} ({}) texel ({x}, {y})", first.format);
            let mut last: Option<&[u8]> = None;
            for t in &res.history {
                let changed = last.is_some_and(|l| l != t.bytes.as_slice());
                if changed || all || last.is_none() {
                    let d = &res.draws[t.draw];
                    let hex: String = t.bytes.iter().map(|b| format!("{b:02x}")).collect();
                    println!(
                        "  draw #{:04} {} {hex} vs={:#x} ps={:#x}",
                        t.draw,
                        match (last, changed) {
                            (None, _) => "first  ",
                            (_, true) => "changed",
                            _ => "same   ",
                        },
                        d.vs,
                        d.ps
                    );
                }
                last = Some(&t.bytes);
            }
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
        Command::Bench {
            capture,
            runs,
            warmup,
        } => {
            use gpu::timing::{self, Part};
            let cap = FrameCapture::load(&capture)?;
            let runs = runs.max(1);
            timing::set(true);
            let res = replay(
                &cap,
                ReplayOptions {
                    frames: warmup + runs,
                    ..ReplayOptions::default()
                },
            )?;
            timing::set(false);
            let draws = res.draws.len();
            let ms = |ns: u64| ns as f64 / 1e6;
            println!(
                "run  processing  draws(ms)  dispatch(ms)  sync(ms)  GPU wait(ms)  compile wait(ms)  encode(ms)  us/draw  GPU busy(ms)"
            );
            let mut encode = Vec::new();
            for (run, f) in res.frames.iter().enumerate().skip(warmup) {
                let processing = f.processing.as_secs_f64() * 1e3;
                let sync = ms(f.parts[Part::Sync as usize]);
                let gpu_wait = ms(f.parts[Part::CommandWait as usize]);
                let wait = ms(f.parts[Part::CompileWait as usize]);
                let enc = processing - gpu_wait - wait;
                println!(
                    "{run:>3}  {processing:>10.2}  {:>9.2}  {:>12.2}  {sync:>8.2}  {gpu_wait:>12.2}  {wait:>16.2}  {enc:>10.2}  {:>7.2}  {:>12.2}",
                    ms(f.parts[Part::Draw as usize]),
                    ms(f.parts[Part::Dispatch as usize]),
                    enc * 1e3 / draws.max(1) as f64,
                    ms(f.parts[Part::GpuBusy as usize]),
                );
                encode.push(enc);
            }
            encode.sort_by(f64::total_cmp);
            let enc = encode[encode.len() / 2];
            println!(
                "median encode {enc:.2} ms/frame, {:.2} us/draw ({draws} draws, {} dispatches)",
                enc * 1e3 / draws.max(1) as f64,
                res.dispatches.len()
            );
            println!("Metal, all runs: {}", res.summary);
        }
        Command::Time {
            capture,
            warmup,
            draws: true,
            top,
            runs,
        } => {
            let cap = FrameCapture::load(&capture)?;
            if warmup {
                let _ = replay(&cap, ReplayOptions::default())?;
            }
            let res = replay(
                &cap,
                ReplayOptions {
                    time_draws: true,
                    frames: runs,
                    ..ReplayOptions::default()
                },
            )?;
            print_draw_times(&res, top);
        }
        Command::Time {
            capture,
            warmup,
            runs,
            ..
        } => {
            let cap = FrameCapture::load(&capture)?;
            if warmup {
                let _ = replay(&cap, ReplayOptions::default())?;
            }
            let res = replay(
                &cap,
                ReplayOptions {
                    time_passes: true,
                    frames: runs,
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
        Command::Mem {
            capture,
            address,
            len,
        } => {
            let cap = FrameCapture::load(&capture)?;
            let page = gpu::capture::PAGE_SIZE;
            for line in (address..address + len).step_by(16) {
                let bytes: Vec<u8> = (line..line + 16)
                    .map(|a| {
                        cap.pages
                            .get(&(a & !(page - 1)))
                            .map_or(0, |p| p[(a & (page - 1)) as usize])
                    })
                    .collect();
                let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
                let floats: Vec<String> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| format!("{:>12.5}", f32::from_le_bytes(*w)))
                    .collect();
                println!("{line:#012x}  {}  {}", hex.join(" "), floats.join(" "));
            }
        }
        Command::Live {
            capture,
            out,
            diff,
            tolerance,
        } => {
            let cap = FrameCapture::load(&capture)?;
            let live = cap.live.as_ref().ok_or(
                "the capture holds no live picture (captures made before 2026-11-25 do not)",
            )?;
            let out = out.unwrap_or_else(|| capture.with_extension("live.png"));
            write_png(&out, live.width, live.height, &live.rgba)?;
            println!("wrote {} ({}x{})", out.display(), live.width, live.height);
            let res = replay(&cap, ReplayOptions::default())?;
            let (addr, w, h, rgba) = select_target(&res, Some(live.address))?;
            if (w, h) != (live.width, live.height) {
                return Err(format!(
                    "replay target {addr:#x} is {w}x{h}, the live picture {}x{}",
                    live.width, live.height
                ));
            }
            let d = diff_pixels(w, &rgba, &live.rgba, tolerance);
            println!(
                "target {addr:#x} ({w}x{h}): {} of {} pixels differ by more than {tolerance}, largest difference {}",
                d.count,
                w as usize * h as usize,
                d.largest
            );
            if let Some((x0, y0, x1, y1)) = d.bounds {
                println!("differing pixels lie in ({x0}, {y0})..=({x1}, {y1})");
            }
            if let Some(path) = diff {
                write_png(&path, w, h, &d.image)?;
                println!("wrote {}", path.display());
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

#[cfg(target_os = "macos")]
struct PixelDiff {
    count: usize,
    largest: u8,
    bounds: Option<(u32, u32, u32, u32)>,
    /// The replay dimmed, with differing pixels in red.
    image: Vec<u8>,
}

#[cfg(target_os = "macos")]
fn diff_pixels(width: u32, replayed: &[u8], live: &[u8], tolerance: u8) -> PixelDiff {
    let mut d = PixelDiff {
        count: 0,
        largest: 0,
        bounds: None,
        image: Vec::with_capacity(replayed.len()),
    };
    let pairs = replayed
        .as_chunks::<4>()
        .0
        .iter()
        .zip(live.as_chunks::<4>().0);
    for (i, (r, l)) in pairs.enumerate() {
        let most = (0..3).map(|c| r[c].abs_diff(l[c])).max().unwrap_or(0);
        d.largest = d.largest.max(most);
        if most > tolerance {
            d.count += 1;
            let (x, y) = ((i % width as usize) as u32, (i / width as usize) as u32);
            d.bounds = Some(match d.bounds {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
            d.image.extend_from_slice(&[255, 0, 0, 255]);
        } else {
            d.image.extend(r[..3].iter().map(|&c| c / 3));
            d.image.push(255);
        }
    }
    d
}

fn print_draw_times(res: &gpu::replay::ReplayResult, top: usize) {
    struct Row {
        programs: String,
        count: usize,
        total: f64,
        max: f64,
    }
    let mut by_program: std::collections::HashMap<String, Row> = std::collections::HashMap::new();
    let mut add = |programs: String, ms: f64| {
        let row = by_program.entry(programs.clone()).or_insert(Row {
            programs,
            count: 0,
            total: 0.0,
            max: 0.0,
        });
        row.count += 1;
        row.total += ms;
        row.max = row.max.max(ms);
    };
    for d in &res.draws {
        add(
            format!("vs={:#x} ps={:#x}", d.vs, d.ps),
            d.gpu_ms.unwrap_or(0.0),
        );
    }
    for c in &res.dispatches {
        add(format!("cs={:#x}", c.cs), c.gpu_ms.unwrap_or(0.0));
    }
    let total: f64 = by_program.values().map(|r| r.total).sum();
    println!(
        "{} draws and {} dispatches timed one by one: GPU {total:.2} ms",
        res.draws.len(),
        res.dispatches.len()
    );

    let mut rows: Vec<Row> = by_program.into_values().collect();
    rows.sort_by(|a, b| b.total.total_cmp(&a.total));
    println!("by program ({} distinct):", rows.len());
    for r in rows.iter().take(top) {
        println!(
            "  {:>8.3} ms {:>5.1}%  {:>4}x  max {:>7.3} ms  {}",
            r.total,
            r.total * 100.0 / total.max(f64::MIN_POSITIVE),
            r.count,
            r.max,
            r.programs
        );
    }

    let mut draws: Vec<&gpu::replay::DrawRecord> = res.draws.iter().collect();
    draws.sort_by(|a, b| b.gpu_ms.unwrap_or(0.0).total_cmp(&a.gpu_ms.unwrap_or(0.0)));
    println!("slowest draws:");
    for d in draws.iter().take(top) {
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
            "  draw #{:04} {:>7.3} ms  vs={:#x} ps={:#x} {} {:?} targets=[{}]{depth}",
            d.index,
            d.gpu_ms.unwrap_or(0.0),
            d.vs,
            d.ps,
            gpu::replay::primitive_name(d.primitive),
            d.draw,
            targets.join(", ")
        );
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::diff_pixels;

    #[test]
    fn pixels_past_the_tolerance_are_counted_bounded_and_marked_red() {
        let replayed = [
            [10, 10, 10, 255],
            [90, 90, 90, 255],
            [30, 30, 30, 255],
            [60, 0, 0, 255],
        ]
        .concat();
        let live = [
            [12, 10, 10, 255],
            [90, 90, 90, 255],
            [30, 30, 30, 255],
            [0, 0, 0, 255],
        ]
        .concat();
        let d = diff_pixels(2, &replayed, &live, 2);
        assert_eq!((d.count, d.largest, d.bounds), (1, 60, Some((1, 1, 1, 1))));
        assert_eq!(&d.image[12..], &[255, 0, 0, 255]);
        assert_eq!(&d.image[..4], &[3, 3, 3, 255]);
    }
}
