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
        /// Runs guest code in the JIT instead of the interpreter.
        #[arg(long)]
        jit: bool,
    },
}

fn main() -> ExitCode {
    let Command::Run {
        path,
        headless,
        dump_frame,
        after,
        jit,
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
    let process = match Process::load(file) {
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
            for (library, nid) in &process.unimplemented {
                eprintln!("  {library} {nid}");
            }
        }
    }
    let app0 = path.parent().map(|p| p.to_path_buf());
    let status = process
        .boot(
            Box::new(std::io::stdout()),
            Box::new(std::io::stderr()),
            app0,
        )
        .and_then(|booted| {
            booted.kernel.use_jit(jit);
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
