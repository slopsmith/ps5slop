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
    },
}

fn main() -> ExitCode {
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
    } = Cli::parse().command;
    let control = control.unwrap_or_else(|| std::env::temp_dir().join(format!("ps5slop-{}.sock", std::process::id())));
    let passcode = match passcode
        .as_deref()
        .map(Passcode::new)
        .unwrap_or(Ok(DEFAULT_PASSCODE))
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ps5slop: {e}");
            return ExitCode::FAILURE;
        }
    };
    let title = match Title::open(&path, patch.as_deref(), &passcode) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ps5slop: {e}");
            return ExitCode::FAILURE;
        }
    };
    let file = match title.eboot() {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "ps5slop: cannot read the executable of {}: {e}",
                path.display()
            );
            return ExitCode::FAILURE;
        }
    };
    let process = match Process::load(file, title.files().as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ps5slop: {e}");
            return ExitCode::FAILURE;
        }
    };
    if !process.unimplemented.is_empty() {
        eprintln!(
            "ps5slop: {} imported functions are not implemented yet",
            process.unimplemented.len()
        );
        if std::env::var_os("PS5SLOP_LIST_UNIMPLEMENTED").is_some() {
            for (library, nid, name) in &process.unimplemented {
                eprintln!("  {library} {nid} {}", name.as_deref().unwrap_or(""));
            }
        }
    }
    let label = title.label();
    eprintln!("ps5slop: running {label}");
    let data = data.or_else(|| default_data(&title));
    let temp = data.as_ref().map(|d| beside(d, "temp"));
    let saves = data.as_ref().map(|d| beside(d, "savedata"));
    for d in data.iter().chain(&temp).chain(&saves) {
        if let Err(e) = std::fs::create_dir_all(d) {
            eprintln!("ps5slop: cannot make {}: {e}", d.display());
        }
    }
    let mut vfs = hle::Vfs::new(None, data);
    title.mount(&vfs);
    if let Some(t) = temp {
        vfs.mount("/temp0", t, true);
    }
    if let Some(d) = std::env::var_os("PS5SLOP_DEVLOG") {
        vfs.mount("/devlog", std::path::PathBuf::from(d), true);
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
        .and_then(|booted| {
            booted.kernel.use_jit(!interpreter);
            if let Some(out) = dump_frame {
                dump_after(booted.kernel.clone(), out, after);
            }
            if let Ok(script) = std::env::var("PS5SLOP_INPUT") {
                play_input(booted.kernel.clone(), script);
            }
            ps5slop::control::serve(booted.kernel.clone(), control.clone());
            run(booted, headless, label)
        });
    let _ = std::fs::remove_file(&control);
    match status {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("ps5slop: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(all(target_os = "macos", feature = "window"))]
fn run(booted: ps5slop::Booted, headless: bool, label: String) -> Result<i32, ps5slop::Error> {
    if headless {
        booted.run()
    } else {
        booted.kernel.use_audio_device();
        ps5slop::window::run(booted, label)
    }
}

#[cfg(not(all(target_os = "macos", feature = "window")))]
fn run(booted: ps5slop::Booted, _headless: bool, _label: String) -> Result<i32, ps5slop::Error> {
    booted.run()
}

/// After `seconds`, writes the screen to `out` and ends the process.
fn dump_after(kernel: std::sync::Arc<hle::Kernel>, out: PathBuf, seconds: f64) {
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
        #[cfg(target_os = "macos")]
        if std::env::var_os("PS5SLOP_DUMP_TARGETS").is_some() {
            for (addr, t) in hle::gpu::metal::all_frames() {
                if let Some((w, h, px)) = hle::gpu::metal::read_rgba(&t) {
                    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                    ppm.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
                    let _ = std::fs::write(out.with_extension(format!("{addr:x}.ppm")), ppm);
                }
            }
        }
        let series: u32 = std::env::var("PS5SLOP_DUMP_SERIES").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let every: u64 = std::env::var("PS5SLOP_DUMP_EVERY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
        for i in 0..series {
            if i > 0 {
                std::thread::sleep(std::time::Duration::from_millis(every));
            }
            let path = if series > 1 { out.with_extension(format!("{i}.ppm")) } else { out.clone() };
            write_frame(&kernel, &path);
        }
        kernel.exit(Ok(0));
    });
}

// `PS5SLOP_INPUT` is `seconds:buttons` steps, comma-separated; buttons are
// joined by `+`, and an empty list releases everything (see README).
fn play_input(kernel: std::sync::Arc<hle::Kernel>, script: String) {
    let start = std::time::Instant::now();
    let steps: Vec<(f64, u32)> = script
        .split(',')
        .filter_map(|step| {
            let (t, keys) = step.split_once(':')?;
            let buttons = ps5slop::pad_buttons(keys);
            Some((t.parse().ok()?, buttons))
        })
        .collect();
    std::thread::spawn(move || {
        for (at, buttons) in steps {
            if let Some(wait) = std::time::Duration::from_secs_f64(at).checked_sub(start.elapsed()) {
                std::thread::sleep(wait);
            }
            kernel.set_pad_input(hle::PadInput { buttons, ..Default::default() });
        }
    });
}

fn write_frame(kernel: &hle::Kernel, out: &std::path::Path) {
    if let Err(e) = ps5slop::write_frame(kernel, out) {
        eprintln!("ps5slop: {e}");
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
            Ok(()) => eprintln!("ps5slop: moved {} to {}", old.display(), dir.display()),
            Err(e) => eprintln!(
                "ps5slop: cannot move {} to {}: {e}",
                old.display(),
                dir.display()
            ),
        }
    }
    Some(dir.join("data"))
}
