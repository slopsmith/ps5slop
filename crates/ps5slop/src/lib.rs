//! A guest process: an executable loaded into its own address space, run by
//! the interpreter, with system calls served by the HLE layer.

use std::fmt;
use std::io::Write;

use hle::{CallError, Env, Hle, Outcome};
use loader::{Dynamic, ElfFile, Image};
use memory::{MapError, Memory, PAGE_SIZE, Prot};
use x86::cpu::reg::*;
use x86::{Cpu, CpuFault, Exit};

/// The guest address space spans 1 TiB, enough for the PS4 layout in
/// `docs/memory.md` (GPU addresses stay below 2^40).
pub const ADDRESS_SPACE: u64 = 1 << 40;
/// Where executables load. PS4 titles expect 0x400000 (shadPS4); unverified for PS5.
pub const IMAGE_BASE: u64 = 0x40_0000;
pub const STACK_TOP: u64 = 0x7_0000_0000;
pub const STACK_SIZE: u64 = 1 << 20;
/// The entry parameter block and its strings.
pub const PROCESS_ARGS: u64 = 0x7_0000_0000;
/// The main thread's static TLS and thread control block.
pub const MAIN_TLS: u64 = 0x7_0001_0000;
/// Room for the thread control block above the thread pointer.
const TCB_SIZE: u64 = 0x1000;

/// Instructions between checks of the run loop.
const SLICE: u64 = 1 << 20;

#[derive(Debug)]
pub enum Error {
    Load(loader::Error),
    Map(MapError),
    /// The guest faulted, with the CPU state at the faulting instruction.
    Guest {
        fault: CpuFault,
        cpu: Box<Cpu>,
    },
    /// A host function failed; `caller` is the return address it was called with.
    Call {
        error: CallError,
        caller: u64,
        cpu: Box<Cpu>,
    },
    /// A raw `syscall` instruction; we only serve calls through libraries.
    Syscall {
        number: u64,
        cpu: Box<Cpu>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Load(e) => write!(f, "cannot load: {e}"),
            Error::Map(e) => write!(f, "cannot map: {e}"),
            Error::Guest { fault, cpu } => {
                write!(f, "guest fault {fault:?} at {:#x}\n{}", cpu.rip, dump(cpu))
            }
            Error::Call { error, caller, cpu } => {
                write!(f, "{error}, called from {caller:#x}\n{}", dump(cpu))
            }
            Error::Syscall { number, cpu } => {
                write!(
                    f,
                    "raw syscall {number} at {:#x} is not supported\n{}",
                    cpu.rip - 2,
                    dump(cpu)
                )
            }
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

fn dump(cpu: &Cpu) -> String {
    const NAMES: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    let mut s = String::new();
    for (i, n) in NAMES.iter().enumerate() {
        s += &format!(
            "{n:>3} {:016x}{}",
            cpu.regs[i],
            if i % 4 == 3 { "\n" } else { "  " }
        );
    }
    s + &format!("rip {:016x}  rflags {:08x}", cpu.rip, cpu.rflags)
}

pub struct Process {
    pub cpu: Cpu,
    pub mem: Memory,
    pub hle: Hle,
    /// Imports no host function implements yet, as `(library, NID)`.
    pub unimplemented: Vec<(String, String)>,
}

impl Process {
    /// Loads an executable (plain ELF or fake-signed SELF) and prepares its
    /// main thread to start at the entry point.
    pub fn load(file: Vec<u8>) -> Result<Self, Error> {
        let elf = ElfFile::parse(file)?;
        let dynamic = Dynamic::parse(&elf)?;
        let mut hle = Hle::new();
        let image = Image::load(&elf, &dynamic, IMAGE_BASE, 1, &mut hle)?;
        let mut mem = Memory::new(ADDRESS_SPACE).map_err(|e| Error::Map(MapError::Host(e)))?;
        for r in &image.regions {
            let p = r.protection;
            let addr = image.base + r.offset;
            mem.map(addr, r.size, Prot::new(p.read, p.write, p.execute), "image")?;
            let bytes = &image.bytes[r.offset as usize..(r.offset + r.size) as usize];
            mem.write_host(addr, bytes).expect("region was just mapped");
        }
        mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Prot::RW, "main stack")?;
        let exit = hle.stub_by_name(hle::INTERNAL, hle::PROGRAM_EXIT);
        hle.map(&mut mem)?;

        let mut cpu = Cpu {
            rip: image.entry,
            ..Cpu::default()
        };
        cpu.fs_base = setup_tls(&mut mem, &image)?;
        let params = setup_args(&mut mem, &["eboot.bin"])?;
        cpu.regs[RDI] = params;
        cpu.regs[RSI] = exit;
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
            unimplemented,
        })
    }

    /// Runs until the guest exits, and returns its exit status.
    pub fn run(&mut self, stdout: &mut dyn Write, stderr: &mut dyn Write) -> Result<i32, Error> {
        loop {
            match x86::run(&mut self.cpu, &mut self.mem, SLICE) {
                Exit::Continue | Exit::Limit => {}
                Exit::Trap(addr) => {
                    let mut env = Env {
                        cpu: &mut self.cpu,
                        mem: &mut self.mem,
                        stdout,
                        stderr,
                    };
                    match self.hle.call(addr, &mut env) {
                        Ok(Outcome::Return(_)) => {}
                        Ok(Outcome::Exit(status)) => return Ok(status),
                        Err(error) => {
                            let caller = self.mem.read_u64(self.cpu.regs[RSP]).unwrap_or(0);
                            return Err(Error::Call {
                                error,
                                caller,
                                cpu: Box::new(self.cpu.clone()),
                            });
                        }
                    }
                }
                Exit::Syscall => {
                    return Err(Error::Syscall {
                        number: self.cpu.regs[RAX],
                        cpu: Box::new(self.cpu.clone()),
                    });
                }
                Exit::Fault { kind, .. } => {
                    return Err(Error::Guest {
                        fault: kind,
                        cpu: Box::new(self.cpu.clone()),
                    });
                }
            }
        }
    }
}

/// Writes the entry parameter block `{ i32 argc; u32 pad; char *argv[]; }`
/// (shadPS4's `EntryParams`) with its strings, and returns its address.
fn setup_args(mem: &mut Memory, args: &[&str]) -> Result<u64, Error> {
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

/// Sets up the main thread's static TLS (x86-64 variant II: FS points at a
/// self-pointer with the TLS image below it) and returns the FS base.
fn setup_tls(mem: &mut Memory, image: &Image) -> Result<u64, Error> {
    let (init, init_size, size, align) = match image.tls {
        Some(t) => (t.init_offset, t.init_size, t.size, t.align.max(16)),
        None => (0, 0, 0, 16),
    };
    let offset = size.next_multiple_of(align);
    let len = (offset + TCB_SIZE).next_multiple_of(PAGE_SIZE);
    mem.map(MAIN_TLS, len, Prot::RW, "main thread tls")?;
    let tcb = MAIN_TLS + offset;
    let data = &image.bytes[init as usize..(init + init_size) as usize];
    mem.write(tcb - offset, data).expect("tls was just mapped");
    mem.write_u64(tcb, tcb).expect("tls was just mapped");
    Ok(tcb)
}
