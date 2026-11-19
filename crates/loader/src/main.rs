use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use loader::elf::Container;
use loader::nid::nid;
use loader::{Dynamic, ElfFile};

/// Inspects Orbis executables and modules (plain ELF or fake-signed SELF).
#[derive(Parser)]
#[command(name = "ps5slop-elf")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prints the headers, segments, modules and libraries.
    Info { file: PathBuf },
    /// Lists every imported symbol by library.
    Imports {
        file: PathBuf,
        /// A file of symbol names, one per line, to label the NIDs they hash to.
        #[arg(long)]
        names: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ps5slop-elf: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> loader::Result<()> {
    match cli.command {
        Command::Info { file } => {
            let elf = ElfFile::parse(std::fs::read(file)?)?;
            let h = elf.header();
            let container = match elf.container() {
                Container::Elf => "ELF",
                Container::SelfPs4 => "fake-signed SELF (PS4 magic)",
                Container::SelfPs5 => "fake-signed SELF (PS5 magic)",
            };
            println!("container: {container}");
            println!("type:      {:#06x}", h.e_type);
            println!("entry:     {:#x}", h.entry);
            for (i, p) in elf.program_headers().iter().enumerate() {
                println!(
                    "segment {i:2}: type {:#010x} flags {} vaddr {:#10x} filesz {:#9x} memsz {:#9x} align {:#x}",
                    p.p_type,
                    flags(p.flags),
                    p.vaddr,
                    p.filesz,
                    p.memsz,
                    p.align
                );
            }
            let d = Dynamic::parse(&elf)?;
            if let Some(name) = &d.original_filename {
                println!("original filename: {name}");
            }
            for m in &d.modules {
                println!(
                    "module:        {} {}.{}",
                    m.name, m.version_major, m.version_minor
                );
            }
            for m in &d.needed_modules {
                println!(
                    "needs module:  {} {}.{} (id {})",
                    m.name, m.version_major, m.version_minor, m.id
                );
            }
            for l in &d.export_libs {
                println!("exports lib:   {} v{} (id {})", l.name, l.version, l.id);
            }
            for l in &d.import_libs {
                println!("imports lib:   {} v{} (id {})", l.name, l.version, l.id);
            }
            println!(
                "{} symbols, {} relocations, {} jump slots",
                d.symbols.len(),
                d.relocations.len(),
                d.jump_slots.len()
            );
        }
        Command::Imports { file, names } => {
            let labels: HashMap<String, String> = match names {
                Some(path) => std::fs::read_to_string(path)?
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(|n| (nid(n), n.to_owned()))
                    .collect(),
                None => HashMap::new(),
            };
            let elf = ElfFile::parse(std::fs::read(file)?)?;
            let d = Dynamic::parse(&elf)?;
            let mut by_lib: BTreeMap<(&str, &str), Vec<&str>> = BTreeMap::new();
            for i in d.imports()? {
                by_lib
                    .entry((i.module.name.as_str(), i.library.name.as_str()))
                    .or_default()
                    .push(i.nid);
            }
            for ((module, library), nids) in by_lib {
                println!("{module} / {library}: {} imports", nids.len());
                for n in nids {
                    match labels.get(n) {
                        Some(name) => println!("  {n} {name}"),
                        None => println!("  {n}"),
                    }
                }
            }
        }
    }
    Ok(())
}

fn flags(f: u32) -> String {
    [(4, 'r'), (2, 'w'), (1, 'x')]
        .iter()
        .map(|&(bit, c)| if f & bit != 0 { c } else { '-' })
        .collect()
}
