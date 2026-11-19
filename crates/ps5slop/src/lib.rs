//! A guest process: an executable loaded into its own address space, run by
//! the interpreter, with system calls served by the HLE layer.

use std::fmt;
use std::io::Write;
use std::sync::Arc;

use hle::{Hle, Kernel, ThreadError, TlsTemplate};
use loader::{Dynamic, ElfFile, Image};
use memory::{MapError, Memory, PAGE_SIZE, Prot};
use x86::Cpu;
use x86::cpu::reg::*;

#[cfg(all(target_os = "macos", feature = "window"))]
pub mod window;

/// The guest address space spans 1 TiB, enough for the PS4 layout in
/// `docs/memory.md` (GPU addresses stay below 2^40).
pub const ADDRESS_SPACE: u64 = 1 << 40;
/// Where executables load. PS4 titles expect 0x400000 (shadPS4); unverified for PS5.
pub const IMAGE_BASE: u64 = 0x40_0000;
pub const STACK_TOP: u64 = 0x7_0000_0000;
pub const STACK_SIZE: u64 = 1 << 20;
/// The entry parameter block and its strings.
pub const PROCESS_ARGS: u64 = 0x7_0000_0000;

#[derive(Debug)]
pub enum Error {
    Load(loader::Error),
    Map(MapError),
    Thread(ThreadError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Load(e) => write!(f, "cannot load: {e}"),
            Error::Map(e) => write!(f, "cannot map: {e}"),
            Error::Thread(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<loader::Error> for Error {
    fn from(e: loader::Error) -> Self {
        Error::Load(e)
    }
}

impl From<MapError> for Error {
    fn from(e: MapError) -> Self {
        Error::Map(e)
    }
}

impl From<ThreadError> for Error {
    fn from(e: ThreadError) -> Self {
        Error::Thread(e)
    }
}

pub struct Process {
    cpu: Cpu,
    mem: Memory,
    hle: Hle,
    tls: TlsTemplate,
    /// Imports no host function implements yet, as `(library, NID)`.
    pub unimplemented: Vec<(String, String, Option<String>)>,
}

impl Process {
    /// Loads an executable (plain ELF or fake-signed SELF) and prepares its
    /// main thread to start at the entry point.
    pub fn load(file: Vec<u8>) -> Result<Self, Error> {
        let elf = ElfFile::parse(file)?;
        let dynamic = Dynamic::parse(&elf)?;
        let mut hle = Hle::new();
        let image = Image::load(&elf, &dynamic, IMAGE_BASE, 1, &mut hle)?;
        let mem = Memory::new(ADDRESS_SPACE).map_err(|e| Error::Map(MapError::Host(e)))?;
        for r in &image.regions {
            let p = r.protection;
            let addr = image.base + r.offset;
            mem.map(addr, r.size, Prot::new(p.read, p.write, p.execute), "image")?;
            let bytes = &image.bytes[r.offset as usize..(r.offset + r.size) as usize];
            mem.write_host(addr, bytes).expect("region was just mapped");
        }
        mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Prot::RW, "main stack")?;
        let teardown = hle.internal(hle::LOADER_TEARDOWN);
        hle.map(&mem)?;

        let tls = match image.tls {
            Some(t) => TlsTemplate {
                init: image.bytes[t.init_offset as usize..(t.init_offset + t.init_size) as usize]
                    .to_vec(),
                size: t.size,
                align: t.align,
            },
            None => TlsTemplate::default(),
        };
        let mut cpu = Cpu {
            rip: image.entry,
            ..Cpu::default()
        };
        let params = setup_args(&mem, &["eboot.bin"])?;
        cpu.regs[RDI] = params;
        cpu.regs[RSI] = teardown;
        // The kernel leaves argc and argv[0] on top of the stack, with RSP 8 off a
        // 16-byte boundary (shadPS4 RunMainEntry; OpenOrbis's startup reads them).
        let mut sp = STACK_TOP - 8 - 16;
        for word in [
            mem.read_u64(params + 8).unwrap(),
            mem.read_u64(params).unwrap(),
        ] {
            sp -= 8;
            mem.write_u64(sp, word).expect("stack was just mapped");
        }
        cpu.regs[RSP] = sp;
        let unimplemented = hle.unimplemented();
        Ok(Self {
            cpu,
            mem,
            hle,
            tls,
            unimplemented,
        })
    }

    /// Starts the kernel, so the host can attach to it (a window, input)
    /// before the guest runs, with the guest's file system `vfs`.
    pub fn boot(
        self,
        stdout: Box<dyn Write + Send>,
        stderr: Box<dyn Write + Send>,
        vfs: hle::Vfs,
    ) -> Result<Booted, Error> {
        let Process {
            mut cpu,
            mem,
            hle,
            tls,
            ..
        } = self;
        let kernel = Kernel::new(hle, mem, tls, stdout, stderr, vfs);
        cpu.fs_base = kernel.new_tls(&kernel.memory())?;
        Ok(Booted { kernel, cpu })
    }
}

/// A process whose kernel is up and whose main thread has not started.
pub struct Booted {
    pub kernel: Arc<Kernel>,
    cpu: Cpu,
}

impl Booted {
    /// Runs the main thread on this host thread until the guest exits, and
    /// returns its exit status.
    pub fn run(self) -> Result<i32, Error> {
        Ok(self.kernel.run_main(self.cpu)?)
    }
}

/// Writes the entry parameter block `{ i32 argc; u32 pad; char *argv[]; }`
/// (shadPS4's `EntryParams`) with its strings, and returns its address.
fn setup_args(mem: &Memory, args: &[&str]) -> Result<u64, Error> {
    mem.map(PROCESS_ARGS, PAGE_SIZE, Prot::RW, "process args")?;
    let mut block = (args.len() as u64).to_le_bytes().to_vec();
    let mut strings = Vec::new();
    let strings_at = PROCESS_ARGS + 8 + 8 * (args.len() as u64 + 1);
    for a in args {
        block.extend_from_slice(&(strings_at + strings.len() as u64).to_le_bytes());
        strings.extend_from_slice(a.as_bytes());
        strings.push(0);
    }
    block.extend_from_slice(&0u64.to_le_bytes());
    block.extend_from_slice(&strings);
    mem.write(PROCESS_ARGS, &block)
        .expect("args page was just mapped");
    Ok(PROCESS_ARGS)
}
