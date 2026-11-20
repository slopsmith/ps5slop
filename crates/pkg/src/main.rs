use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use pkg::keys::{DEFAULT_PASSCODE, Passcode};
use pkg::{Package, entry_id};

/// Inspects and extracts fake-signed PS4 packages and plaintext PS5 packages.
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
    /// Lists the app's files with their sizes.
    Ls {
        package: PathBuf,
        /// The package's 32-character passcode (all zeros by default).
        #[arg(long)]
        passcode: Option<String>,
    },
    /// Writes one app file to stdout.
    Cat {
        package: PathBuf,
        /// The file's path inside the app, as `ls` prints it.
        path: String,
        /// The package's 32-character passcode (all zeros by default).
        #[arg(long)]
        passcode: Option<String>,
    },
    /// Prints the block map of a PS5 package's inner image.
    Naps { package: PathBuf },
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
            println!("format:       {:x?}", p.format());
            println!("content id:   {}", h.content_id);
            println!("title id:     {}", p.title_id().as_deref().unwrap_or("?"));
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
        Command::Ls { package, passcode } => {
            let app = Package::open(&package)?.app_pfs(&passcode_or_default(passcode)?)?;
            for (path, inode) in app.files()? {
                println!(
                    "{:>12} {}",
                    app.inode(inode)?.size_compressed,
                    path.display()
                );
            }
        }
        Command::Cat {
            package,
            path,
            passcode,
        } => {
            let app = Package::open(&package)?.app_pfs(&passcode_or_default(passcode)?)?;
            let src = app.open_file(app.lookup(&path)?.inode)?;
            let mut out = std::io::stdout().lock();
            let mut buf = vec![0u8; 1 << 20];
            let mut at = 0;
            while at < src.len() {
                let n = (src.len() - at).min(buf.len() as u64) as usize;
                src.read_exact_at(at, &mut buf[..n])?;
                std::io::Write::write_all(&mut out, &buf[..n])?;
                at += n as u64;
            }
        }
        Command::Naps { package } => {
            let l = Package::open(&package)?.naps_layout(&DEFAULT_PASSCODE)?;
            println!("header:         {:02x?}", l.header);
            println!("compression:    {}", l.compression());
            println!("files:          {}", l.file_count);
            println!(
                "logical blocks: {} ({:#x} bytes)",
                l.logical_blocks,
                l.mount_size()
            );
            println!("outer blocks:   {}", l.outer_blocks);
            println!("u2c groups:     {}", l.u2c.len());
            println!("records:        {}", l.records.len());
            for (i, f) in l.files.iter().enumerate() {
                println!("file {i:6}: {:#12x} kind {:#04x}", f.offset, f.kind);
            }
        }
        Command::Extract {
            package,
            dest,
            passcode,
        } => {
            let passcode = passcode_or_default(passcode)?;
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

fn passcode_or_default(passcode: Option<String>) -> pkg::Result<Passcode> {
    match passcode {
        Some(s) => Passcode::new(&s),
        None => Ok(DEFAULT_PASSCODE),
    }
}
