use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use ps5slop::Process;

/// A PS5 compatibility layer for Apple Silicon Macs.
#[derive(Parser)]
#[command(name = "ps5slop")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Runs an executable: a plain or fake-signed ELF, or an app folder
    /// holding `eboot.bin`.
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
    } = Cli::parse().command;
    let path = if path.is_dir() {
        path.join("eboot.bin")
    } else {
        path
    };
    let file = match std::fs::read(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ps5slop: cannot read {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let process = match Process::load(file, path.parent()) {
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
    let app0 = path.parent().map(|p| p.to_path_buf());
    let data = data.or_else(|| default_data(&path));
    let temp = data.as_ref().map(|d| d.with_file_name("temp"));
    for d in data.iter().chain(&temp) {
        if let Err(e) = std::fs::create_dir_all(d) {
            eprintln!("ps5slop: cannot make {}: {e}", d.display());
        }
    }
    let mut vfs = hle::Vfs::new(app0, data);
    if let Some(t) = temp {
        vfs.mount("/temp0", t, true);
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
        match kernel.frame() {
            Some(f) => {
                let mut ppm = format!("P6\n{} {}\n255\n", f.width, f.height).into_bytes();
                ppm.extend(
                    f.rgba
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .flat_map(|p| [p[0], p[1], p[2]]),
                );
                if let Err(e) = std::fs::write(&out, ppm) {
                    eprintln!("ps5slop: cannot write {}: {e}", out.display());
                }
            }
            None => eprintln!("ps5slop: no frame on screen to dump"),
        }
        kernel.exit(Ok(0));
    });
}

/// `~/Library/Application Support/ps5slop/<title>/data`, where `<title>` is
/// the app folder's name (or the executable's, outside one).
fn default_data(eboot: &std::path::Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let title = match eboot.parent() {
        Some(dir) if eboot.file_name().is_some_and(|n| n == "eboot.bin") => dir.file_name()?,
        _ => eboot.file_stem()?,
    };
    let root = PathBuf::from(home).join("Library/Application Support/ps5slop");
    Some(root.join(title).join("data"))
}
