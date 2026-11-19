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
    Run { path: PathBuf },
}

fn main() -> ExitCode {
    let Command::Run { path } = Cli::parse().command;
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
    }
    let status = process.run(Box::new(std::io::stdout()), Box::new(std::io::stderr()));
    match status {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("ps5slop: {e}");
            ExitCode::FAILURE
        }
    }
}
