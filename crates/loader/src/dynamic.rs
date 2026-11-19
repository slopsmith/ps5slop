//! The dynamic section of an Orbis ELF. `PT_DYNAMIC` holds the tags; the
//! Sony-specific ones (`DT_SCE_*`) point into the `PT_SCE_DYNLIBDATA`
//! segment, which holds the string, symbol and relocation tables. Facts and
//! sources are in `docs/loader.md`.

use crate::elf::{ElfFile, PT_DYNAMIC, PT_SCE_DYNLIBDATA, u32_le, u64_le};
use crate::error::invalid;
use crate::nid::SymbolRef;
use crate::{Error, Result};

pub const DT_NULL: u64 = 0;
pub const DT_NEEDED: u64 = 1;
pub const DT_INIT: u64 = 12;
pub const DT_FINI: u64 = 13;
pub const DT_INIT_ARRAY: u64 = 25;
pub const DT_FINI_ARRAY: u64 = 26;
pub const DT_INIT_ARRAYSZ: u64 = 27;
pub const DT_FINI_ARRAYSZ: u64 = 28;
pub const DT_PREINIT_ARRAY: u64 = 32;
pub const DT_PREINIT_ARRAYSZ: u64 = 33;

pub const DT_SCE_ORIGINAL_FILENAME: u64 = 0x6100_0009;
pub const DT_SCE_MODULE_INFO: u64 = 0x6100_000d;
pub const DT_SCE_NEEDED_MODULE: u64 = 0x6100_000f;
pub const DT_SCE_EXPORT_LIB: u64 = 0x6100_0013;
pub const DT_SCE_IMPORT_LIB: u64 = 0x6100_0015;
pub const DT_SCE_PLTGOT: u64 = 0x6100_0027;
pub const DT_SCE_JMPREL: u64 = 0x6100_0029;
pub const DT_SCE_PLTRELSZ: u64 = 0x6100_002d;
pub const DT_SCE_RELA: u64 = 0x6100_002f;
pub const DT_SCE_RELASZ: u64 = 0x6100_0031;
pub const DT_SCE_STRTAB: u64 = 0x6100_0035;
pub const DT_SCE_STRSZ: u64 = 0x6100_0037;
pub const DT_SCE_SYMTAB: u64 = 0x6100_0039;
pub const DT_SCE_SYMTABSZ: u64 = 0x6100_003f;

pub const SYM_SIZE: usize = 24;
pub const RELA_SIZE: usize = 24;

pub const STB_GLOBAL: u8 = 1;
pub const STB_WEAK: u8 = 2;
pub const STT_OBJECT: u8 = 1;
pub const STT_FUNC: u8 = 2;
pub const STT_TLS: u8 = 6;

pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_GLOB_DAT: u32 = 6;
pub const R_X86_64_JUMP_SLOT: u32 = 7;
pub const R_X86_64_RELATIVE: u32 = 8;
pub const R_X86_64_DTPMOD64: u32 = 16;

/// A module this file is (`DT_SCE_MODULE_INFO`) or needs (`DT_SCE_NEEDED_MODULE`).
/// Packed as name offset (bits 0-31), minor and major version (32-39, 40-47), id (48-63).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub name: String,
    pub id: u16,
    pub version_major: u8,
    pub version_minor: u8,
}

/// A library this file exports or imports. Packed as name offset (bits 0-31),
/// version (32-47), id (48-63).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Library {
    pub name: String,
    pub id: u16,
    pub version: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub info: u8,
    pub value: u64,
    pub size: u64,
}

impl Symbol {
    pub fn bind(&self) -> u8 {
        self.info >> 4
    }

    pub fn kind(&self) -> u8 {
        self.info & 0xf
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rela {
    pub offset: u64,
    pub r_type: u32,
    pub symbol: u32,
    pub addend: i64,
}

/// `(address, size in bytes)` of an init, fini or preinit array.
pub type Array = Option<(u64, u64)>;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Dynamic {
    pub original_filename: Option<String>,
    pub needed: Vec<String>,
    pub modules: Vec<Module>,
    pub needed_modules: Vec<Module>,
    pub export_libs: Vec<Library>,
    pub import_libs: Vec<Library>,
    pub symbols: Vec<Symbol>,
    pub relocations: Vec<Rela>,
    pub jump_slots: Vec<Rela>,
    pub pltgot: Option<u64>,
    pub init: Option<u64>,
    pub fini: Option<u64>,
    pub init_array: Array,
    pub fini_array: Array,
    pub preinit_array: Array,
}

/// A symbol another module must provide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import<'a> {
    pub symbol: u32,
    pub nid: &'a str,
    pub library: &'a Library,
    pub module: &'a Module,
}

impl Dynamic {
    pub fn parse(elf: &ElfFile) -> Result<Self> {
        let (dyn_index, _) = elf
            .find(PT_DYNAMIC)
            .ok_or_else(|| invalid("no PT_DYNAMIC segment"))?;
        let (data_index, _) = elf
            .find(PT_SCE_DYNLIBDATA)
            .ok_or_else(|| invalid("no PT_SCE_DYNLIBDATA segment"))?;
        let tags: Vec<(u64, u64)> = elf
            .segment_data(dyn_index)?
            .as_chunks::<16>()
            .0
            .iter()
            .map(|t| (u64_le(t, 0), u64_le(t, 8)))
            .take_while(|&(tag, _)| tag != DT_NULL)
            .collect();
        let data = elf.segment_data(data_index)?;
        let one = |tag| tags.iter().find(|t| t.0 == tag).map(|t| t.1);
        let table = |off_tag, size_tag, what: &str| -> Result<&[u8]> {
            match (one(off_tag), one(size_tag)) {
                (Some(off), Some(size)) => slice(data, off, size)
                    .ok_or_else(|| invalid(format!("{what} is outside PT_SCE_DYNLIBDATA"))),
                (None, None) => Ok(&[]),
                _ => Err(invalid(format!(
                    "{what} has an offset or a size but not both"
                ))),
            }
        };

        let strtab = table(DT_SCE_STRTAB, DT_SCE_STRSZ, "string table")?;
        let string = |off: u64| -> Result<String> {
            let rest = usize::try_from(off)
                .ok()
                .and_then(|o| strtab.get(o..))
                .ok_or_else(|| invalid("string offset out of range"))?;
            let end = rest
                .iter()
                .position(|&c| c == 0)
                .ok_or_else(|| invalid("unterminated string"))?;
            String::from_utf8(rest[..end].to_vec()).map_err(|_| invalid("string is not UTF-8"))
        };

        let mut d = Dynamic {
            pltgot: one(DT_SCE_PLTGOT),
            init: one(DT_INIT),
            fini: one(DT_FINI),
            init_array: one(DT_INIT_ARRAY).zip(one(DT_INIT_ARRAYSZ)),
            fini_array: one(DT_FINI_ARRAY).zip(one(DT_FINI_ARRAYSZ)),
            preinit_array: one(DT_PREINIT_ARRAY).zip(one(DT_PREINIT_ARRAYSZ)),
            ..Default::default()
        };
        for &(tag, v) in &tags {
            let name_off = v & 0xffff_ffff;
            let id = (v >> 48) as u16;
            let module = || -> Result<Module> {
                Ok(Module {
                    name: string(name_off)?,
                    id,
                    version_minor: (v >> 32) as u8,
                    version_major: (v >> 40) as u8,
                })
            };
            let library = || -> Result<Library> {
                Ok(Library {
                    name: string(name_off)?,
                    id,
                    version: (v >> 32) as u16,
                })
            };
            match tag {
                DT_NEEDED => d.needed.push(string(v)?),
                DT_SCE_ORIGINAL_FILENAME => d.original_filename = Some(string(v)?),
                DT_SCE_MODULE_INFO => d.modules.push(module()?),
                DT_SCE_NEEDED_MODULE => d.needed_modules.push(module()?),
                DT_SCE_EXPORT_LIB => d.export_libs.push(library()?),
                DT_SCE_IMPORT_LIB => d.import_libs.push(library()?),
                _ => {}
            }
        }

        let symtab = table(DT_SCE_SYMTAB, DT_SCE_SYMTABSZ, "symbol table")?;
        d.symbols = symtab
            .as_chunks::<SYM_SIZE>()
            .0
            .iter()
            .map(|s| {
                Ok(Symbol {
                    name: string(u32_le(s, 0).into())?,
                    info: s[4],
                    value: u64_le(s, 8),
                    size: u64_le(s, 16),
                })
            })
            .collect::<Result<_>>()?;
        d.relocations = parse_relas(table(DT_SCE_RELA, DT_SCE_RELASZ, "relocation table")?);
        d.jump_slots = parse_relas(table(DT_SCE_JMPREL, DT_SCE_PLTRELSZ, "jump slot table")?);
        Ok(d)
    }

    /// Splits a symbol's `NID#LIB#MOD` name and finds its library and module.
    /// Exported symbols name this file's own libraries and modules.
    pub fn resolve_name(&self, symbol: u32) -> Result<Option<Import<'_>>> {
        let sym = self
            .symbols
            .get(symbol as usize)
            .ok_or_else(|| invalid(format!("symbol {symbol} does not exist")))?;
        let Some(r) = SymbolRef::parse(&sym.name) else {
            return Ok(None);
        };
        let exported = sym.value != 0;
        let (libs, mods) = if exported {
            (&self.export_libs, &self.modules)
        } else {
            (&self.import_libs, &self.needed_modules)
        };
        let library = libs.iter().find(|l| l.id == r.library).ok_or_else(|| {
            Error::Invalid(format!(
                "symbol {} names unknown library {}",
                sym.name, r.library
            ))
        })?;
        let module = mods.iter().find(|m| m.id == r.module).ok_or_else(|| {
            Error::Invalid(format!(
                "symbol {} names unknown module {}",
                sym.name, r.module
            ))
        })?;
        Ok(Some(Import {
            symbol,
            nid: r.nid,
            library,
            module,
        }))
    }

    /// Every symbol this file needs from other modules.
    pub fn imports(&self) -> Result<Vec<Import<'_>>> {
        let mut out = Vec::new();
        for (i, sym) in self.symbols.iter().enumerate() {
            let global = matches!(sym.bind(), STB_GLOBAL | STB_WEAK);
            if global
                && sym.value == 0
                && let Some(import) = self.resolve_name(i as u32)?
            {
                out.push(import);
            }
        }
        Ok(out)
    }
}

fn slice(data: &[u8], off: u64, size: u64) -> Option<&[u8]> {
    let off = usize::try_from(off).ok()?;
    let size = usize::try_from(size).ok()?;
    data.get(off..off.checked_add(size)?)
}

fn parse_relas(b: &[u8]) -> Vec<Rela> {
    b.as_chunks::<RELA_SIZE>()
        .0
        .iter()
        .map(|r| {
            let info = u64_le(r, 8);
            Rela {
                offset: u64_le(r, 0),
                r_type: info as u32,
                symbol: (info >> 32) as u32,
                addend: u64_le(r, 16) as i64,
            }
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::elf::tests::{Seg, build_elf};
    use crate::elf::{ET_SCE_DYNEXEC, PF_R, PF_X, PT_LOAD};
    use crate::nid::{encode_id, nid};

    /// Builds the two dynamic segments of a test executable.
    #[derive(Default)]
    pub struct DynBuilder {
        strtab: Vec<u8>,
        symtab: Vec<u8>,
        rela: Vec<u8>,
        jmprel: Vec<u8>,
        tags: Vec<(u64, u64)>,
    }

    impl DynBuilder {
        pub fn new() -> Self {
            Self {
                strtab: vec![0],
                ..Default::default()
            }
        }

        pub fn string(&mut self, s: &str) -> u64 {
            let off = self.strtab.len() as u64;
            self.strtab.extend_from_slice(s.as_bytes());
            self.strtab.push(0);
            off
        }

        pub fn needed_module(&mut self, name: &str, id: u16) {
            let v = self.string(name) | (1 << 40) | (u64::from(id) << 48);
            self.tags.push((DT_SCE_NEEDED_MODULE, v));
        }

        pub fn import_lib(&mut self, name: &str, id: u16) {
            let v = self.string(name) | (1 << 32) | (u64::from(id) << 48);
            self.tags.push((DT_SCE_IMPORT_LIB, v));
        }

        /// Adds a symbol and returns its index.
        pub fn symbol(&mut self, name: &str, info: u8, value: u64) -> u32 {
            let off = self.string(name) as u32;
            let mut s = [0u8; SYM_SIZE];
            s[0..4].copy_from_slice(&off.to_le_bytes());
            s[4] = info;
            s[8..16].copy_from_slice(&value.to_le_bytes());
            self.symtab.extend_from_slice(&s);
            (self.symtab.len() / SYM_SIZE - 1) as u32
        }

        /// Adds an imported function `name` from library `lib` of module `module`.
        pub fn import(&mut self, name: &str, lib: u16, module: u16) -> u32 {
            let s = format!("{}#{}#{}", nid(name), encode_id(lib), encode_id(module));
            self.symbol(&s, (STB_GLOBAL << 4) | STT_FUNC, 0)
        }

        pub fn rela(
            &mut self,
            jump_slot: bool,
            offset: u64,
            r_type: u32,
            symbol: u32,
            addend: i64,
        ) {
            let mut r = [0u8; RELA_SIZE];
            r[0..8].copy_from_slice(&offset.to_le_bytes());
            r[8..16]
                .copy_from_slice(&((u64::from(symbol) << 32) | u64::from(r_type)).to_le_bytes());
            r[16..24].copy_from_slice(&addend.to_le_bytes());
            if jump_slot {
                self.jmprel.extend_from_slice(&r);
            } else {
                self.rela.extend_from_slice(&r);
            }
        }

        pub fn tag(&mut self, tag: u64, v: u64) {
            self.tags.push((tag, v));
        }

        /// Returns (PT_DYNAMIC contents, PT_SCE_DYNLIBDATA contents).
        pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
            let mut data = Vec::new();
            let mut place = |bytes: &[u8], off_tag, size_tag, tags: &mut Vec<(u64, u64)>| {
                tags.push((off_tag, data.len() as u64));
                tags.push((size_tag, bytes.len() as u64));
                data.extend_from_slice(bytes);
                while data.len() % 8 != 0 {
                    data.push(0);
                }
            };
            place(&self.strtab, DT_SCE_STRTAB, DT_SCE_STRSZ, &mut self.tags);
            place(&self.symtab, DT_SCE_SYMTAB, DT_SCE_SYMTABSZ, &mut self.tags);
            place(&self.rela, DT_SCE_RELA, DT_SCE_RELASZ, &mut self.tags);
            place(&self.jmprel, DT_SCE_JMPREL, DT_SCE_PLTRELSZ, &mut self.tags);
            let mut dynamic = Vec::new();
            for (t, v) in self.tags.iter().chain([&(DT_NULL, 0)]) {
                dynamic.extend_from_slice(&t.to_le_bytes());
                dynamic.extend_from_slice(&v.to_le_bytes());
            }
            (dynamic, data)
        }
    }

    pub fn elf_with(b: DynBuilder, code: Vec<u8>) -> Vec<u8> {
        let (dynamic, data) = b.finish();
        let code_len = code.len() as u64;
        build_elf(
            ET_SCE_DYNEXEC,
            0,
            &[
                Seg {
                    p_type: PT_LOAD,
                    flags: PF_R | PF_X,
                    vaddr: 0,
                    data: code,
                    memsz: code_len,
                },
                Seg {
                    p_type: PT_DYNAMIC,
                    flags: PF_R,
                    vaddr: 0,
                    data: dynamic,
                    memsz: 0,
                },
                Seg {
                    p_type: PT_SCE_DYNLIBDATA,
                    flags: PF_R,
                    vaddr: 0,
                    data,
                    memsz: 0,
                },
            ],
        )
    }

    fn sample() -> Vec<u8> {
        let mut b = DynBuilder::new();
        b.needed_module("libkernel", 1);
        b.import_lib("libkernel", 1);
        b.import_lib("libc", 2);
        let write = b.import("sceKernelWrite", 1, 1);
        b.import("puts", 2, 1);
        b.symbol("local", 0, 0x10);
        b.rela(true, 0x100, R_X86_64_JUMP_SLOT, write, 0);
        b.rela(false, 0x108, R_X86_64_RELATIVE, 0, 0x40);
        b.tag(DT_INIT, 0x20);
        elf_with(b, vec![0xcc; 0x200])
    }

    #[test]
    fn reads_modules_libraries_and_tables() {
        let d = Dynamic::parse(&ElfFile::parse(sample()).unwrap()).unwrap();
        assert_eq!(d.needed_modules.len(), 1);
        assert_eq!(d.needed_modules[0].name, "libkernel");
        assert_eq!(
            (
                d.needed_modules[0].version_major,
                d.needed_modules[0].version_minor
            ),
            (1, 0)
        );
        assert_eq!(
            d.import_libs
                .iter()
                .map(|l| (l.name.as_str(), l.id, l.version))
                .collect::<Vec<_>>(),
            [("libkernel", 1, 1), ("libc", 2, 1)]
        );
        assert_eq!(d.symbols.len(), 3);
        assert_eq!(
            d.jump_slots,
            [Rela {
                offset: 0x100,
                r_type: R_X86_64_JUMP_SLOT,
                symbol: 0,
                addend: 0
            }]
        );
        assert_eq!(d.relocations[0].addend, 0x40);
        assert_eq!(d.init, Some(0x20));
    }

    #[test]
    fn lists_imports_with_their_library_and_module() {
        let d = Dynamic::parse(&ElfFile::parse(sample()).unwrap()).unwrap();
        let imports = d.imports().unwrap();
        let got: Vec<_> = imports
            .iter()
            .map(|i| (i.nid, i.library.name.as_str(), i.module.name.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                (nid("sceKernelWrite").as_str(), "libkernel", "libkernel"),
                (nid("puts").as_str(), "libc", "libkernel")
            ]
        );
    }

    #[test]
    fn unknown_library_ids_are_errors() {
        let mut b = DynBuilder::new();
        b.needed_module("libkernel", 1);
        b.import("sceKernelWrite", 9, 1);
        let d = Dynamic::parse(&ElfFile::parse(elf_with(b, vec![0; 16])).unwrap()).unwrap();
        assert!(d.imports().is_err());
    }

    #[test]
    fn missing_dynamic_segments_are_errors() {
        let elf = build_elf(
            ET_SCE_DYNEXEC,
            0,
            &[Seg {
                p_type: PT_LOAD,
                flags: PF_R,
                vaddr: 0,
                data: vec![0; 4],
                memsz: 4,
            }],
        );
        assert!(Dynamic::parse(&ElfFile::parse(elf).unwrap()).is_err());
    }
}
