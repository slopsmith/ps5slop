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
        /// The host folder the guest sees as `/data` (default: a folder per
        /// title under ~/Library/Application Support/ps5slop).
        #[arg(long, value_name = "DIR")]
        data: Option<PathBuf>,
        /// A fake-signed PS4 package's 32-character passcode (all zeros by
        /// default).
        #[arg(long)]
        passcode: Option<String>,
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
        passcode,
    } = Cli::parse().command;
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
    let title = match Title::open(&path, &passcode) {
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
    let data = data.or_else(|| default_data(&title));
    let temp = data.as_ref().map(|d| d.with_file_name("temp"));
    let saves = data.as_ref().map(|d| d.with_file_name("savedata"));
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
    if let Some(s) = saves {
        vfs.set_saves(s);
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
            run(booted, headless)
        });
    match status {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("ps5slop: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(all(target_os = "macos", feature = "window"))]
fn run(booted: ps5slop::Booted, headless: bool) -> Result<i32, ps5slop::Error> {
    if headless {
        booted.run()
    } else {
        booted.kernel.use_audio_device();
        ps5slop::window::run(booted)
    }
}

#[cfg(not(all(target_os = "macos", feature = "window")))]
fn run(booted: ps5slop::Booted, _headless: bool) -> Result<i32, ps5slop::Error> {
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
        for i in 0..series {
            if i > 0 {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            let path = if series > 1 { out.with_extension(format!("{i}.ppm")) } else { out.clone() };
            write_frame(&kernel, &path);
        }
        kernel.exit(Ok(0));
    });
}

/// Writes the screen to `out` as a binary PPM.
fn write_frame(kernel: &hle::Kernel, out: &std::path::Path) {
    match kernel.frame() {
        Some(mut f) => {
            #[cfg(target_os = "macos")]
            if let Some((w, h, px)) = f.texture.as_ref().and_then(hle::gpu::metal::read_rgba) {
                (f.width, f.height, f.rgba) = (w, h, px);
            }
            let mut ppm = format!("P6\n{} {}\n255\n", f.width, f.height).into_bytes();
            ppm.extend(f.rgba.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
            if let Err(e) = std::fs::write(out, ppm) {
                eprintln!("ps5slop: cannot write {}: {e}", out.display());
            }
        }
        None => eprintln!("ps5slop: no frame on screen to dump"),
    }
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
