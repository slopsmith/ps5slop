use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use pkg::keys::{DEFAULT_PASSCODE, Passcode};
use ps5slop::Process;
use ps5slop::title::Title;

/// A PS5 compatibility layer for Apple Silicon Macs.
#[derive(Parser)]
#[command(name = "ps5slop")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Runs an executable: a plain or fake-signed ELF, an app folder
    /// holding `eboot.bin`, or a `.pkg` file read in place.
    Run {
        path: PathBuf,
        /// Runs without a window; frames are not shown.
        #[arg(long)]
        headless: bool,
        /// Writes the screen to this file (binary PPM) and stops the guest,
        /// `--after` seconds in.
        #[arg(long, value_name = "PATH")]
        dump_frame: Option<PathBuf>,
        #[arg(long, value_name = "SECONDS", default_value_t = 5.0)]
        after: f64,
        /// Runs guest code in the interpreter instead of the JIT.
        #[arg(long)]
        interpreter: bool,
        /// The host folder the guest sees as `/data`, with saves next to it
        /// (default: one per title in ~/Library/Application Support/ps5slop).
        #[arg(long, value_name = "DIR")]
        data: Option<PathBuf>,
        /// The folder with open-source stand-ins for system fonts, filled by
        /// `scripts/fetch-fonts` (default: ~/Library/Application
        /// Support/ps5slop/fonts).
        #[arg(long, value_name = "DIR")]
        fonts: Option<PathBuf>,
        /// A PS4 patch package to lay over the base package `PATH`. Not
        /// needed when `PATH` is the patch and its base is in the same folder.
        #[arg(long, value_name = "PKG")]
        patch: Option<PathBuf>,
        /// A fake-signed PS4 package's 32-character passcode (all zeros by
        /// default).
        #[arg(long)]
        passcode: Option<String>,
        /// The Unix socket for debugging commands (default:
        /// `ps5slop-PID.sock` in the temporary folder).
        #[arg(long, value_name = "PATH")]
        control: Option<PathBuf>,
        /// Goes on from a save state written by the control socket's `save`
        /// instead of starting the title (`PATH` must be the same title).
        #[arg(long, value_name = "STATE")]
        load_state: Option<PathBuf>,
    },
    /// Lists the games in `.pkg` files, app folders and folders searched for
    /// them: name, title id, version, icon and compatibility status.
    Info {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Prints one JSON array instead of a table.
        #[arg(long)]
        json: bool,
        /// Writes each game's icon (`icon0.png`) into this folder.
        #[arg(long, value_name = "DIR")]
        icons: Option<PathBuf>,
    },
}

fn info(paths: &[PathBuf], json: bool, icons: Option<&std::path::Path>) -> ExitCode {
    let games = ps5slop::library::scan(paths, icons);
    if json {
        let list: Vec<_> = games.iter().map(|g| g.to_json()).collect();
        println!("{}", serde_json::Value::Array(list));
        return ExitCode::SUCCESS;
    }
    for g in &games {
        let status = g.compat.as_ref().map_or("-", |c| c.status.as_str());
        println!(
            "{:<10} {:<10} {:<9} {}  {}",
            g.title_id.as_deref().unwrap_or("-"),
            g.version.as_deref().unwrap_or("-"),
            status,
            g.name.as_deref().or(g.error.as_deref()).unwrap_or("-"),
            g.path.display()
        );
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let command = Cli::parse().command;
    if let Command::Info { paths, json, icons } = &command {
        return info(paths, *json, icons.as_deref());
    }
    let Command::Run {
        path,
        headless,
        dump_frame,
        after,
        interpreter,
        data,
        fonts,
        patch,
        passcode,
        control,
        load_state,
    } = command
    else {
        unreachable!("info returned above");
    };
    diag::init_logging();
    let control = control.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("ps5slop-{}.sock", std::process::id()))
    });
    let passcode = match passcode
        .as_deref()
        .map(Passcode::new)
        .unwrap_or(Ok(DEFAULT_PASSCODE))
    {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let title = match Title::open(&path, patch.as_deref(), &passcode) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let file = match title.eboot() {
        Ok(f) => f,
        Err(e) => {
            tracing::error!("cannot read the executable of {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let process = match Process::load(file, title.files().as_deref()) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if !process.unimplemented.is_empty() {
        tracing::warn!(
            "{} imported functions are not implemented yet",
            process.unimplemented.len()
        );
        for (library, nid, name) in &process.unimplemented {
            tracing::debug!(
                "not implemented: {library} {nid} {}",
                name.as_deref().unwrap_or("")
            );
        }
    }
    let label = title.label();
    let title_id = title.id();
    let capture_tag = title_id
        .clone()
        .or_else(|| {
            title
                .fallback_name()
                .map(|n| n.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "capture".into());
    tracing::info!("running {label}");
    let data = data.or_else(|| default_data(&title));
    let temp = data.as_ref().map(|d| beside(d, "temp"));
    let saves = data.as_ref().map(|d| beside(d, "savedata"));
    let download = data.as_ref().map(|d| beside(d, "download0"));
    for d in data.iter().chain(&temp).chain(&saves).chain(&download) {
        if let Err(e) = std::fs::create_dir_all(d) {
            tracing::warn!("cannot make {}: {e}", d.display());
        }
    }
    let mut vfs = hle::Vfs::new(None, data);
    title.mount(&vfs);
    if let Some(t) = temp {
        vfs.mount("/temp0", t, true);
    }
    if let Some(d) = download {
        vfs.mount("/download0", d, true);
    }
    let config = diag::config();
    if let Some(d) = &config.run.devlog {
        vfs.mount("/devlog", d.clone(), true);
    }
    if let Some(s) = saves {
        vfs.set_saves(s);
    }
    if let Some(f) = fonts.or_else(default_fonts) {
        vfs.set_system_fonts(f);
    }
    let status = process
        .boot(
            Box::new(std::io::stdout()),
            Box::new(std::io::stderr()),
            vfs,
        )
        .and_then(|mut booted| {
            if let Some(state) = load_state {
                booted.load_state(state);
            }
            booted.kernel.use_jit(!interpreter);
            if let Some(id) = title_id {
                booted.kernel.set_title_id(id);
            }
            if let Some(out) = dump_frame {
                dump_after(booted.kernel.clone(), out, after);
            }
            if let Some(script) = &config.run.input {
                play_input(booted.kernel.clone(), script.clone());
            }
            if let Some(every) = config.run.check_code {
                check_code(booted.kernel.memory(), every);
            }
            ps5slop::control::serve(booted.kernel.clone(), control.clone());
            if let Some((path, filter)) = &config.run.calls
                && let Ok(f) = std::fs::File::create(path)
            {
                hle::trace_calls(Some(f), filter.clone());
            }
            let kernel_for_report = booted.kernel.clone();
            let result = run(booted, headless, label, capture_tag);
            kernel_for_report.finish_captures();
            let report = ps5slop::problem_report(&kernel_for_report);
            if !report.is_empty() {
                if let Some(path) = &config.run.report {
                    match std::fs::write(path, &report) {
                        Ok(()) => tracing::info!("wrote problem report to {}", path.display()),
                        Err(e) => {
                            tracing::warn!(
                                "cannot write problem report to {}: {e}",
                                path.display()
                            );
                        }
                    }
                }
                tracing::info!(target: "ps5slop::report", "\n{}", report.trim_end());
            }
            result
        });
    let _ = std::fs::remove_file(&control);
    match status {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            tracing::error!("{e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(all(target_os = "macos", feature = "window"))]
fn run(
    booted: ps5slop::Booted,
    headless: bool,
    label: String,
    capture_tag: String,
) -> Result<i32, ps5slop::Error> {
    if headless {
        booted.run()
    } else {
        booted.kernel.use_audio_device();
        ps5slop::window::run(booted, label, capture_tag)
    }
}

#[cfg(not(all(target_os = "macos", feature = "window")))]
fn run(
    booted: ps5slop::Booted,
    _headless: bool,
    _label: String,
    _capture_tag: String,
) -> Result<i32, ps5slop::Error> {
    booted.run()
}

/// Compares every non-writable region (code and read-only data) with its
/// contents at start, `every` so often, and reports changes.
fn check_code(mem: memory::Memory, every: std::time::Duration) {
    let start = std::time::Instant::now();
    let mut code: Vec<(u64, Vec<u8>)> = mem
        .region_spans()
        .into_iter()
        .filter(|r| {
            !r.trap && r.prot != memory::Prot::NONE && !r.prot.allows(memory::Access::Write)
        })
        .filter_map(|r| {
            let mut b = vec![0; (r.end - r.start) as usize];
            read_any(&mem, r.start, &mut b).then_some((r.start, b))
        })
        .collect();
    let total: usize = code.iter().map(|(_, b)| b.len()).sum();
    tracing::info!(
        "checking {} code regions ({} MiB) every {} ms",
        code.len(),
        total >> 20,
        every.as_millis()
    );
    std::thread::spawn(move || {
        let mut now = vec![0u8; 1 << 14];
        loop {
            std::thread::sleep(every);
            for (base, old) in &mut code {
                for (k, page) in old.chunks_mut(1 << 14).enumerate() {
                    let at = *base + (k << 14) as u64;
                    let now = &mut now[..page.len()];
                    if !read_any(&mem, at, now) || now == page {
                        continue;
                    }
                    let first = page.iter().zip(&*now).position(|(a, b)| a != b).unwrap();
                    let last = page.iter().zip(&*now).rposition(|(a, b)| a != b).unwrap();
                    let differ = page.iter().zip(&*now).filter(|(a, b)| a != b).count();
                    let show = first..(first + 32).min(page.len());
                    tracing::warn!(
                        "code changed at {:.3} s: {:#x}..{:#x} ({differ} bytes differ)\n  was {:02x?}\n  now {:02x?}",
                        start.elapsed().as_secs_f64(),
                        at + first as u64,
                        at + last as u64 + 1,
                        &page[show.clone()],
                        &now[show],
                    );
                    page.copy_from_slice(now);
                }
            }
        }
    });
}

fn read_any(mem: &memory::Memory, addr: u64, buf: &mut [u8]) -> bool {
    mem.read(addr, buf).is_ok() || mem.fetch(addr, buf).is_ok()
}

/// After `seconds`, writes the screen to `out` and ends the process.
fn dump_after(kernel: std::sync::Arc<hle::Kernel>, out: PathBuf, seconds: f64) {
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
        #[cfg(target_os = "macos")]
        if diag::config().gpu.dump_targets {
            for (addr, t) in hle::gpu::metal::all_frames() {
                if let Some((w, h, px)) = hle::gpu::metal::read_rgba(&t) {
                    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                    ppm.extend(
                        px.as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|p| [p[0], p[1], p[2]]),
                    );
                    let _ = std::fs::write(out.with_extension(format!("{addr:x}.ppm")), ppm);
                }
            }
        }
        let (series, every) = (
            diag::config().run.dump_series,
            diag::config().run.dump_every,
        );
        for i in 0..series {
            if i > 0 {
                std::thread::sleep(every);
            }
            let path = if series > 1 {
                out.with_extension(format!("{i}.ppm"))
            } else {
                out.clone()
            };
            write_frame(&kernel, &path);
        }
        kernel.exit(Ok(0));
    });
}

/// Plays `script`: `seconds:buttons` steps, comma-separated; buttons are
/// joined by `+`, and an empty list releases everything.
fn play_input(kernel: std::sync::Arc<hle::Kernel>, script: String) {
    let start = std::time::Instant::now();
    let steps: Vec<(f64, hle::PadInput)> = script
        .split(',')
        .filter_map(|step| {
            let (t, keys) = step.split_once(':')?;
            Some((t.parse().ok()?, ps5slop::pad_input(keys)))
        })
        .collect();
    std::thread::spawn(move || {
        for (at, input) in steps {
            if let Some(wait) = std::time::Duration::from_secs_f64(at).checked_sub(start.elapsed())
            {
                std::thread::sleep(wait);
            }
            kernel.set_pad_input(input);
        }
    });
}

fn write_frame(kernel: &hle::Kernel, out: &std::path::Path) {
    if let Err(e) = ps5slop::write_frame(kernel, out) {
        tracing::error!("{e}");
    }
}

// `<data>.<what>` unless the folder is named `data`, so that two data
// folders side by side never share saves.
fn beside(data: &std::path::Path, what: &str) -> PathBuf {
    match data.file_name().and_then(|n| n.to_str()) {
        Some("data") | None => data.with_file_name(what),
        Some(name) => data.with_file_name(format!("{name}.{what}")),
    }
}

/// `~/Library/Application Support/ps5slop/fonts`.
fn default_fonts() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join("Library/Application Support/ps5slop/fonts"))
}

/// `~/Library/Application Support/ps5slop/<title id or name>/data`, moving a
/// folder kept under the name to the title id.
fn default_data(title: &Title) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let root = PathBuf::from(home).join("Library/Application Support/ps5slop");
    let fallback = title.fallback_name().map(|n| root.join(n));
    let Some(id) = title.id() else {
        return fallback.map(|d| d.join("data"));
    };
    let dir = root.join(&id);
    if let Some(old) = fallback.filter(|old| old.is_dir() && !dir.exists() && *old != dir) {
        match std::fs::rename(&old, &dir) {
            Ok(()) => tracing::info!("moved {} to {}", old.display(), dir.display()),
            Err(e) => tracing::warn!("cannot move {} to {}: {e}", old.display(), dir.display()),
        }
    }
    Some(dir.join("data"))
}
