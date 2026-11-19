use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use pkg::keys::{DEFAULT_PASSCODE, Passcode};
use pkg::{Package, entry_id};

/// Inspects and extracts fake-signed PS4 packages.
#[derive(Parser)]
#[command(name = "ps5slop-pkg")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prints the header and entry table.
    Info { package: PathBuf },
    /// Extracts the app's files into a new folder.
    Extract {
        package: PathBuf,
        dest: PathBuf,
        /// The package's 32-character passcode (all zeros by default).
        #[arg(long)]
        passcode: Option<String>,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ps5slop-pkg: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> pkg::Result<()> {
    match cli.command {
        Command::Info { package } => {
            let p = Package::open(&package)?;
            let h = p.header();
            println!("content id:   {}", h.content_id);
            println!(
                "content type: {:#x}, flags {:#x}",
                h.content_type, h.content_flags
            );
            println!(
                "pfs image:    {:#x} bytes at {:#x}, flags {:#x}",
                h.pfs_image_size, h.pfs_image_offset, h.pfs_flags
            );
            for e in p.entries() {
                let enc = if e.is_encrypted() {
                    format!("encrypted, key {}", e.key_index())
                } else {
                    "clear".into()
                };
                println!(
                    "entry {:#06x}: {:#9x} bytes at {:#9x} ({enc})",
                    e.id, e.size, e.offset
                );
            }
            if let Some(e) = p.entry(entry_id::PARAM_SFO) {
                println!("param.sfo:    {} bytes", p.read_entry(e)?.len());
            }
        }
        Command::Extract {
            package,
            dest,
            passcode,
        } => {
            let passcode = match passcode {
                Some(s) => Passcode::new(&s)?,
                None => DEFAULT_PASSCODE,
            };
            let done = Package::open(&package)?.extract(&passcode, &dest)?;
            println!(
                "extracted {} files, {} bytes, into {}",
                done.files,
                done.bytes,
                dest.display()
            );
        }
    }
    Ok(())
}
