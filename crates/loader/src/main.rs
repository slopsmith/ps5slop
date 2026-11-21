use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use loader::elf::Container;
use loader::image::NoImports;
use loader::nid::nid;
use loader::{Dynamic, ElfFile, Image};

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
    /// Prints the raw tags of the dynamic section.
    Tags { file: PathBuf },
    /// Lists every imported symbol by library.
    Imports {
        file: PathBuf,
        /// A file of symbol names, one per line, to label the NIDs they hash to.
        #[arg(long)]
        names: Option<PathBuf>,
    },
    /// Lays the file out at a base address and applies its relocations, with
    /// no imports provided, then reports what is missing.
    Load {
        file: PathBuf,
        #[arg(long, default_value = "0x400000", value_parser = parse_hex)]
        base: u64,
    },
    /// Writes the file bytes of segment `index` to `out`, for local analysis
    /// of a title's own code (see docs/legal.md).
    Segment {
        file: PathBuf,
        index: usize,
        out: PathBuf,
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
        Command::Tags { file } => {
            let elf = ElfFile::parse(std::fs::read(file)?)?;
            let (i, _) = elf
                .find(loader::elf::PT_DYNAMIC)
                .ok_or_else(|| loader::Error::Invalid("no PT_DYNAMIC".into()))?;
            for t in elf.segment_data(i)?.as_chunks::<16>().0 {
                let tag = u64::from_le_bytes(t[..8].try_into().unwrap());
                let value = u64::from_le_bytes(t[8..].try_into().unwrap());
                println!("{tag:#010x} {value:#018x}");
                if tag == 0 {
                    break;
                }
            }
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
        Command::Segment { file, index, out } => {
            let elf = ElfFile::parse(std::fs::read(file)?)?;
            std::fs::write(out, elf.segment_data(index)?)?;
        }
        Command::Load { file, base } => {
            let elf = ElfFile::parse(std::fs::read(file)?)?;
            let d = Dynamic::parse(&elf)?;
            let img = Image::load(&elf, &d, base, 1, &mut NoImports)?;
            println!(
                "image: {:#x}..{:#x}, entry {:#x}",
                img.base,
                img.base + img.bytes.len() as u64,
                img.entry
            );
            for r in &img.regions {
                let p = r.protection;
                let prot: String = [(p.read, 'r'), (p.write, 'w'), (p.execute, 'x')]
                    .iter()
                    .map(|&(on, c)| if on { c } else { '-' })
                    .collect();
                println!(
                    "  {:#10x}..{:#10x} {prot}",
                    img.base + r.offset,
                    img.base + r.offset + r.size
                );
            }
            if let Some(t) = img.tls {
                println!(
                    "tls: {:#x} bytes ({:#x} initialised), align {:#x}",
                    t.size, t.init_size, t.align
                );
            }
            if let Some(p) = img.proc_param {
                println!("proc param at {p:#x}");
            }
            let mut by_lib: BTreeMap<&str, usize> = BTreeMap::new();
            for m in &img.missing {
                *by_lib.entry(m.library.as_str()).or_default() += 1;
            }
            println!("{} imports missing", img.missing.len());
            for (lib, n) in by_lib {
                println!("  {lib}: {n}");
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

fn parse_hex(s: &str) -> Result<u64, String> {
    let digits = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(digits, 16).map_err(|e| e.to_string())
}
