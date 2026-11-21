//! Finding, choosing and linking the modules an executable imports from.
//! See `docs/loader.md`, "Modules".

use std::collections::HashMap;
use std::path::Path;

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

/// Where a title's files come from: a host folder or a package.
pub trait Files {
    /// The names of the files (not folders) in folder `dir`; `""` is the root.
    fn list(&self, dir: &str) -> Vec<String>;
    /// The names of the folders in folder `dir`.
    fn folders(&self, dir: &str) -> Vec<String>;
    /// The bytes of the file at `path`, relative to the root.
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>>;
}

impl Files for &Path {
    fn list(&self, dir: &str) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(self.join(dir)) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| !t.is_dir()))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect()
    }

    fn folders(&self, dir: &str) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(self.join(dir)) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect()
    }

    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        std::fs::read(self.join(path))
    }
}

/// A module file found in the title's files.
pub struct Candidate {
    /// Relative to the title's root.
    pub path: String,
    /// In `sce_module`, so a Sony module that needs allowing.
    pub system: bool,
}

/// How deep below the root modules are looked for (Unity titles keep theirs
/// in `Media/Modules` and `Media/Plugins`).
const MODULE_SEARCH_DEPTH: usize = 3;

/// The module files in `files`, by the module name their own tables give,
/// shallowest first: the root, `sce_module` (Sony's), then any other folder
/// but `sce_sys`. Files that do not parse are skipped.
pub fn candidates(files: &dyn Files) -> HashMap<String, Candidate> {
    let mut out = HashMap::new();
    let mut level = vec![String::new()];
    for depth in 0..=MODULE_SEARCH_DEPTH {
        let mut next = Vec::new();
        for folder in &level {
            if depth < MODULE_SEARCH_DEPTH {
                next.extend(
                    files
                        .folders(folder)
                        .into_iter()
                        .filter(|f| f != "sce_sys")
                        .map(|f| {
                            if folder.is_empty() {
                                f
                            } else {
                                format!("{folder}/{f}")
                            }
                        }),
                );
            }
            let system = folder.split('/').any(|c| c == "sce_module");
            add_candidates(files, folder, system, &mut out);
        }
        next.sort_by_key(|f| !f.ends_with("sce_module"));
        level = next;
    }
    out
}

fn add_candidates(
    files: &dyn Files,
    folder: &str,
    system: bool,
    out: &mut HashMap<String, Candidate>,
) {
    for file in files.list(folder) {
        let prx = Path::new(&file)
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("prx") || x.eq_ignore_ascii_case("sprx"));
        if !prx {
            continue;
        }
        let path = if folder.is_empty() {
            file
        } else {
            format!("{folder}/{file}")
        };
        let Some(name) = module_name(files, &path) else {
            continue;
        };
        out.entry(name).or_insert(Candidate { path, system });
    }
}

fn module_name(files: &dyn Files, path: &str) -> Option<String> {
    let elf = ElfFile::parse(files.read(path).ok()?).ok()?;
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

/// Chooses the modules to load for `main`: every game module in `found`,
/// and the system modules it (or a loaded module) imports from that are
/// allowed.
/// Returns them with their bytes, in the order they were found,
/// dependencies after dependents.
pub fn choose(
    main: &Unit,
    found: &HashMap<String, Candidate>,
    files: &dyn Files,
) -> Vec<(String, String, Vec<u8>)> {
    let mut chosen: Vec<(String, String, Vec<u8>)> = Vec::new();
    let mut games: Vec<&String> = found
        .iter()
        .filter(|(_, c)| !c.system)
        .map(|(n, _)| n)
        .collect();
    games.sort_by(|a, b| b.cmp(a));
    let mut queue: Vec<String> = games.into_iter().cloned().collect();
    queue.extend(main.dynamic.needed_modules.iter().map(|m| m.name.clone()));
    while let Some(name) = queue.pop() {
        if chosen.iter().any(|(n, _, _)| *n == name) {
            continue;
        }
        let Some(c) = found.get(&name) else { continue };
        if c.system && !ALLOWED_SYSTEM_MODULES.contains(&name.as_str()) {
            continue;
        }
        let Ok(file) = files.read(&c.path) else {
            continue;
        };
        let Ok(unit) = Unit::parse(name.clone(), file.clone()) else {
            continue;
        };
        queue.extend(unit.dynamic.needed_modules.iter().map(|m| m.name.clone()));
        chosen.push((name, c.path.clone(), file));
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
    let Some((_, hdr)) = unit.elf.find(loader::elf::PT_GNU_EH_FRAME) else {
        return info;
    };
    info.eh_frame_hdr = unit.base + hdr.vaddr;
    let at = hdr.vaddr as usize;
    let Some(&enc) = image.bytes.get(at + 1) else {
        return info;
    };
    let Some(frame) = read_encoded(&image.bytes, at + 4, enc).map(|v| v.wrapping_add(unit.base))
    else {
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
