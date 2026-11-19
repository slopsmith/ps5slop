//! Finding, choosing and linking the modules an executable imports from.
//! See `docs/loader.md`, "Modules".

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hle::Hle;
use loader::dynamic::{Import, STB_GLOBAL, STB_WEAK};
use loader::image::{PAGE_SIZE, Resolver};
use loader::{Dynamic, ElfFile};

/// Sony modules from a title's `sce_module` folder that we load instead of
/// providing their libraries in the HLE layer. Each must only need modules
/// we provide (see `docs/decisions.md`).
pub const ALLOWED_SYSTEM_MODULES: &[&str] = &["libc"];

/// Where modules are laid out, one after another (our choice; see
/// `docs/memory.md`).
pub const MODULE_AREA: u64 = 0x8_0000_0000;

/// A module file found in the title folder.
pub struct Candidate {
    pub path: PathBuf,
    /// In `sce_module`, so a Sony module that needs allowing.
    pub system: bool,
}

/// The module files in `dir` (and `dir/sce_module`), by the module name
/// their own tables give. Files that do not parse are skipped.
pub fn candidates(dir: &Path) -> HashMap<String, Candidate> {
    let mut out = HashMap::new();
    for (folder, system) in [(dir.to_path_buf(), false), (dir.join("sce_module"), true)] {
        let Ok(entries) = std::fs::read_dir(&folder) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            let prx = path
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case("prx") || x.eq_ignore_ascii_case("sprx"));
            if !prx {
                continue;
            }
            let Some(name) = module_name(&path) else {
                continue;
            };
            out.entry(name).or_insert(Candidate { path, system });
        }
    }
    out
}

fn module_name(path: &Path) -> Option<String> {
    let elf = ElfFile::parse(std::fs::read(path).ok()?).ok()?;
    let dynamic = Dynamic::parse(&elf).ok()?;
    dynamic.modules.first().map(|m| m.name.clone())
}

/// A file to load: the executable or a module.
pub struct Unit {
    pub name: String,
    /// The file it comes from (its name only).
    pub file: String,
    pub elf: ElfFile,
    pub dynamic: Dynamic,
    pub base: u64,
}

impl Unit {
    pub fn parse(name: String, file: Vec<u8>) -> Result<Self, loader::Error> {
        let elf = ElfFile::parse(file)?;
        let dynamic = Dynamic::parse(&elf)?;
        Ok(Self {
            name,
            file: String::new(),
            elf,
            dynamic,
            base: 0,
        })
    }

    /// How much address space the unit's segments need.
    pub fn size(&self) -> u64 {
        self.elf
            .program_headers()
            .iter()
            .filter(|p| {
                p.memsz != 0 && matches!(p.p_type, loader::elf::PT_LOAD | loader::elf::PT_SCE_RELRO)
            })
            .map(|p| (p.vaddr + p.memsz).next_multiple_of(PAGE_SIZE))
            .max()
            .unwrap_or(0)
    }

    /// The libraries and NIDs this unit exports, with their addresses.
    pub fn exports(&self) -> Vec<((String, String), u64)> {
        let mut out = Vec::new();
        for (i, sym) in self.dynamic.symbols.iter().enumerate() {
            if sym.value == 0 || !matches!(sym.bind(), STB_GLOBAL | STB_WEAK) {
                continue;
            }
            if let Ok(Some(e)) = self.dynamic.resolve_name(i as u32) {
                out.push((
                    (e.library.name.clone(), e.nid.to_owned()),
                    self.base + sym.value,
                ));
            }
        }
        out
    }
}

/// Chooses the modules to load for `main`: those it (or a loaded module)
/// imports from, found in `found`, and allowed if they are system modules.
/// Returns them in the order they were found, dependencies after dependents.
pub fn choose(main: &Unit, found: &HashMap<String, Candidate>) -> Vec<(String, PathBuf)> {
    let mut chosen: Vec<(String, PathBuf)> = Vec::new();
    let mut queue: Vec<String> = main
        .dynamic
        .needed_modules
        .iter()
        .map(|m| m.name.clone())
        .collect();
    while let Some(name) = queue.pop() {
        if chosen.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let Some(c) = found.get(&name) else { continue };
        if c.system && !ALLOWED_SYSTEM_MODULES.contains(&name.as_str()) {
            continue;
        }
        let Ok(file) = std::fs::read(&c.path) else {
            continue;
        };
        let Ok(unit) = Unit::parse(name.clone(), file) else {
            continue;
        };
        queue.extend(unit.dynamic.needed_modules.iter().map(|m| m.name.clone()));
        chosen.push((name, c.path.clone()));
    }
    chosen
}

/// Binds imports to loaded modules' exports first, then to HLE stubs.
pub struct Linker<'a> {
    pub exports: &'a HashMap<(String, String), u64>,
    pub hle: &'a mut Hle,
}

impl Resolver for Linker<'_> {
    fn resolve(&mut self, import: &Import<'_>) -> Option<u64> {
        let key = (import.library.name.clone(), import.nid.to_owned());
        match self.exports.get(&key) {
            Some(&addr) => Some(addr),
            None => self.hle.resolve(import),
        }
    }
}

/// What the kernel's loader functions report about `unit`, laid out as
/// `image` (unwind tables found through its `PT_GNU_EH_FRAME` segment).
pub fn info(unit: &Unit, image: &loader::Image) -> hle::ModuleInfo {
    let phdrs = unit.elf.program_headers();
    let text = phdrs
        .iter()
        .find(|p| p.p_type == loader::elf::PT_LOAD)
        .map_or((0, 0), |p| (unit.base + p.vaddr, p.memsz));
    let mut info = hle::ModuleInfo {
        name: unit.name.clone(),
        file: unit.file.clone(),
        start: unit.base,
        end: unit.base + unit.size(),
        text,
        ..Default::default()
    };
    let Some((_, hdr)) = unit.elf.find(loader::elf::PT_GNU_EH_FRAME) else { return info };
    info.eh_frame_hdr = unit.base + hdr.vaddr;
    let at = hdr.vaddr as usize;
    let Some(&enc) = image.bytes.get(at + 1) else { return info };
    let Some(frame) = read_encoded(&image.bytes, at + 4, enc).map(|v| v.wrapping_add(unit.base)) else {
        return info;
    };
    let end = phdrs
        .iter()
        .filter(|p| p.p_type == loader::elf::PT_LOAD)
        .map(|p| (unit.base + p.vaddr, unit.base + p.vaddr + p.memsz))
        .find(|&(s, e)| (s..e).contains(&frame))
        .map_or(frame, |(_, e)| e);
    info.eh_frame = (frame, end - frame);
    info
}

/// A DWARF pointer-encoded value at `at` (relative to the image's start),
/// for the encodings `.eh_frame_hdr` uses.
fn read_encoded(bytes: &[u8], at: usize, enc: u8) -> Option<u64> {
    let raw = |n: usize| bytes.get(at..at + n);
    let v = match enc & 0x0f {
        0x03 => u32::from_le_bytes(raw(4)?.try_into().ok()?) as u64,
        0x0b => i32::from_le_bytes(raw(4)?.try_into().ok()?) as i64 as u64,
        0x04 | 0x0c => u64::from_le_bytes(raw(8)?.try_into().ok()?),
        _ => return None,
    };
    match enc & 0x70 {
        0x00 => Some(v),
        0x10 => Some(v.wrapping_add(at as u64)),
        _ => None,
    }
}
