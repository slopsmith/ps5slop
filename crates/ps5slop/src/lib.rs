//! A guest process: an executable loaded into its own address space, run by
//! the interpreter, with system calls served by the HLE layer.

use std::fmt;
use std::io::Write;
use std::sync::Arc;

use std::collections::HashMap;

use hle::{Hle, Kernel, ThreadError, TlsTemplate};
use loader::Image;
use loader::elf::PT_TLS;
use memory::{MapError, Memory, PAGE_SIZE, Prot};
use modules::Unit;
use x86::Cpu;
use x86::cpu::reg::*;

pub mod control;
pub mod modules;
pub mod title;
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
    /// Module initialisers, to run before the entry point.
    inits: Vec<u64>,
    module_infos: Vec<hle::ModuleInfo>,
    proc_param: Option<u64>,
    /// Modules loaded besides the executable: name and base address.
    pub modules: Vec<(String, u64)>,
    /// Imports nothing provides yet, as `(library, NID, name if known)`.
    pub unimplemented: Vec<(String, String, Option<String>)>,
}

impl Process {
    /// Loads an executable (plain ELF or fake-signed SELF), and the modules it
    /// imports that its title's `files` hold, and prepares its main thread to
    /// start at the entry point.
    pub fn load(file: Vec<u8>, files: Option<&dyn modules::Files>) -> Result<Self, Error> {
        let mut main = Unit::parse("eboot".into(), file)?;
        main.file = "eboot.bin".into();
        main.base = IMAGE_BASE;
        let mut units = vec![main];
        let mut next = modules::MODULE_AREA;
        let chosen = files.map_or_else(Vec::new, |files| {
            modules::choose(&units[0], &modules::candidates(files), files)
        });
        for (name, path, file) in chosen {
            let mut unit = Unit::parse(name, file)?;
            unit.file = path.rsplit('/').next().unwrap_or_default().to_owned();
            unit.base = next;
            next = (next + unit.size() + PAGE_SIZE).next_multiple_of(PAGE_SIZE);
            units.push(unit);
        }
        let mut exports = HashMap::new();
        for u in &units {
            for (key, addr) in u.exports() {
                exports.entry(key).or_insert(addr);
            }
        }

        let mut hle = Hle::new();
        let mem = Memory::new(ADDRESS_SPACE).map_err(|e| Error::Map(MapError::Host(e)))?;
        let mut tls = TlsTemplate::default();
        let mut images = Vec::new();
        let mut module_infos = Vec::new();
        for (k, u) in units.iter().enumerate() {
            let tls_segment = u
                .elf
                .find(PT_TLS)
                .map(|(_, p)| (p.vaddr, p.filesz, p.memsz, p.align));
            let tls_id = match tls_segment {
                Some((_, _, size, align)) => tls.add(Vec::new(), size, align),
                None if k == 0 => tls.add(Vec::new(), 0, 1),
                None => 0,
            };
            let mut linker = modules::Linker {
                exports: &exports,
                hle: &mut hle,
            };
            let image = Image::load(&u.elf, &u.dynamic, u.base, tls_id, &mut linker)?;
            if let Some((vaddr, filesz, _, _)) = tls_segment {
                let init = image.bytes[vaddr as usize..(vaddr + filesz) as usize].to_vec();
                tls.blocks[tls_id as usize - 1].init = init;
            }
            for r in &image.regions {
                let p = r.protection;
                let addr = image.base + r.offset;
                mem.map(addr, r.size, Prot::new(p.read, p.write, p.execute), &u.name)?;
                let bytes = &image.bytes[r.offset as usize..(r.offset + r.size) as usize];
                mem.write_host(addr, bytes).expect("region was just mapped");
            }
            module_infos.push(modules::info(u, &image));
            images.push(image);
        }
        let inits = initialisers(&units, &images);
        mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Prot::RW, "main stack")?;
        let teardown = hle.internal(hle::LOADER_TEARDOWN);
        hle.map(&mem)?;

        let mut cpu = Cpu {
            rip: images[0].entry,
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
        let modules = units[1..]
            .iter()
            .map(|u| (u.name.clone(), u.base))
            .collect();
        let proc_param = images[0].proc_param;
        Ok(Self {
            cpu,
            mem,
            hle,
            tls,
            inits,
            module_infos,
            proc_param,
            modules,
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
            inits,
            proc_param,
            module_infos,
            ..
        } = self;
        let kernel = Kernel::new(hle, mem, tls, stdout, stderr, vfs);
        cpu.fs_base = kernel.new_tls(&kernel.memory())?;
        if let Some(p) = proc_param {
            kernel.set_proc_param(p);
        }
        for info in module_infos {
            kernel.add_module(info);
        }
        Ok(Booted { kernel, cpu, inits })
    }
}

/// A process whose kernel is up and whose main thread has not started.
pub struct Booted {
    pub kernel: Arc<Kernel>,
    cpu: Cpu,
    inits: Vec<u64>,
}

impl Booted {
    /// Runs the main thread on this host thread until the guest exits, and
    /// returns its exit status.
    pub fn run(self) -> Result<i32, Error> {
        Ok(self.kernel.run_main(self.cpu, &self.inits)?)
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

/// The modules' initialisers, dependencies first: a module's entry point
/// (`module_start`) if it has one, else its `DT_INIT` and init array.
fn initialisers(units: &[Unit], images: &[Image]) -> Vec<u64> {
    fn visit(k: usize, units: &[Unit], seen: &mut Vec<bool>, order: &mut Vec<usize>) {
        if seen[k] {
            return;
        }
        seen[k] = true;
        for need in &units[k].dynamic.needed_modules {
            if let Some(d) = units.iter().position(|u| u.name == need.name) {
                visit(d, units, seen, order);
            }
        }
        order.push(k);
    }
    let (mut seen, mut order) = (vec![false; units.len()], Vec::new());
    seen[0] = true;
    for k in 1..units.len() {
        visit(k, units, &mut seen, &mut order);
    }
    let mut out = Vec::new();
    for (u, image) in order.into_iter().map(|k| (&units[k], &images[k])) {
        if u.elf.header().entry != 0 {
            out.push(image.entry);
            continue;
        }
        if let Some(init) = u.dynamic.init {
            out.push(u.base + init);
        }
        if let Some((addr, size)) = u.dynamic.init_array {
            for k in 0..size / 8 {
                let at = (addr + 8 * k) as usize;
                let f = u64::from_le_bytes(image.bytes[at..at + 8].try_into().expect("8 bytes"));
                if f != 0 && f != u64::MAX {
                    out.push(f);
                }
            }
        }
    }
    out
}

/// Writes the screen to `out` as a binary PPM.
pub fn write_frame(kernel: &Kernel, out: &std::path::Path) -> Result<(), String> {
    let Some(mut f) = kernel.frame() else {
        return Err("no frame on screen to dump".into());
    };
    #[cfg(target_os = "macos")]
    if let Some((w, h, px)) = f.texture.as_ref().and_then(hle::gpu::metal::read_rgba) {
        (f.width, f.height, f.rgba) = (w, h, px);
    }
    let mut ppm = format!("P6\n{} {}\n255\n", f.width, f.height).into_bytes();
    ppm.extend(f.rgba.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
    std::fs::write(out, ppm).map_err(|e| format!("cannot write {}: {e}", out.display()))
}

/// Pad buttons named in `keys`, joined by `+` (`cross+up`); unknown names
/// count for nothing.
pub fn pad_buttons(keys: &str) -> u32 {
    keys.split('+').filter(|k| !k.is_empty()).fold(0, |b, k| {
        b | match k {
            "options" => 0x8,
            "up" => 0x10,
            "right" => 0x20,
            "down" => 0x40,
            "left" => 0x80,
            "l1" => 0x400,
            "r1" => 0x800,
            "triangle" => 0x1000,
            "circle" => 0x2000,
            "cross" => 0x4000,
            "square" => 0x8000,
            _ => 0,
        }
    })
}
