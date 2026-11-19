use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use mkelf::{ImportSpec, convert};

/// Turns an ELF linked with `ld.lld -shared` for x86_64-unknown-freebsd into an
/// Orbis executable (our own test programs only).
#[derive(Parser)]
#[command(name = "ps5slop-mkelf")]
struct Cli {
    input: PathBuf,
    output: PathBuf,
    /// Lines of `SYMBOL LIBRARY [MODULE]` naming where each import comes from.
    #[arg(long)]
    imports: PathBuf,
    /// The module name written into the output (default: the output's file stem).
    #[arg(long)]
    name: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let run = || -> Result<(), Box<dyn std::error::Error>> {
        let spec = ImportSpec::parse(&std::fs::read_to_string(&cli.imports)?)?;
        let name = match &cli.name {
            Some(n) => n.clone(),
            None => cli
                .output
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("main")
                .to_owned(),
        };
        let out = convert(&std::fs::read(&cli.input)?, &spec, &name)?;
        std::fs::write(&cli.output, out)?;
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ps5slop-mkelf: {e}");
            ExitCode::FAILURE
        }
    }
}
