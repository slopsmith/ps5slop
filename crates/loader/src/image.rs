//! Lays an ELF's segments out in memory at a base address and applies its
//! relocations. The result is a byte image for the guest address space to map.

use crate::dynamic::{
    Dynamic, Import, R_X86_64_64, R_X86_64_DTPMOD64, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT,
    R_X86_64_RELATIVE, Rela,
};
use crate::elf::{ElfFile, PF_R, PF_W, PF_X, PT_LOAD, PT_SCE_PROCPARAM, PT_SCE_RELRO, PT_TLS};
use crate::error::invalid;
use crate::{Error, Result};

/// Orbis allocates in 16 KiB pages (see `docs/memory.md`).
pub const PAGE_SIZE: u64 = 0x4000;

/// Supplies the address of each imported symbol.
pub trait Resolver {
    /// The address `import` should bind to, or `None` if nothing provides it.
    fn resolve(&mut self, import: &Import<'_>) -> Option<u64>;
}

/// Resolves nothing, so every import is reported as missing.
pub struct NoImports;

impl Resolver for NoImports {
    fn resolve(&mut self, _: &Import<'_>) -> Option<u64> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Protection {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Protection {
    fn from_flags(f: u32) -> Self {
        Self {
            read: f & PF_R != 0,
            write: f & PF_W != 0,
            execute: f & PF_X != 0,
        }
    }
}

/// A page-aligned range of the image and how it is protected once loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub offset: u64,
    pub size: u64,
    pub protection: Protection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tls {
    /// Image offset of the initialisation data.
    pub init_offset: u64,
    pub init_size: u64,
    pub size: u64,
    pub align: u64,
}

/// An import nothing resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Missing {
    pub nid: String,
    pub library: String,
    pub module: String,
}

pub struct Image {
    pub base: u64,
    /// The image's memory, starting at `base`.
    pub bytes: Vec<u8>,
    pub regions: Vec<Region>,
    pub entry: u64,
    pub tls: Option<Tls>,
    /// Address of the `PT_SCE_PROCPARAM` data, if any.
    pub proc_param: Option<u64>,
    pub missing: Vec<Missing>,
}

impl Image {
    /// Loads `elf` at `base` (a multiple of [`PAGE_SIZE`]). `tls_module` is the
    /// TLS module id `R_X86_64_DTPMOD64` relocations receive.
    pub fn load(
        elf: &ElfFile,
        dynamic: &Dynamic,
        base: u64,
        tls_module: u64,
        resolver: &mut dyn Resolver,
    ) -> Result<Self> {
        if !base.is_multiple_of(PAGE_SIZE) {
            return Err(invalid(format!("base {base:#x} is not page aligned")));
        }
        let mapped = |t| t == PT_LOAD || t == PT_SCE_RELRO;
        let mut size = 0u64;
        for p in elf
            .program_headers()
            .iter()
            .filter(|p| mapped(p.p_type) && p.memsz != 0)
        {
            let end = p
                .vaddr
                .checked_add(p.memsz)
                .ok_or_else(|| invalid("segment wraps around"))?;
            size = size.max(end.next_multiple_of(PAGE_SIZE));
        }
        let len = usize::try_from(size).map_err(|_| invalid("image too large"))?;
        let mut bytes = vec![0u8; len];
        let mut regions = Vec::new();
        for (i, p) in elf.program_headers().iter().enumerate() {
            if !mapped(p.p_type) || p.memsz == 0 {
                continue;
            }
            if p.filesz > p.memsz {
                return Err(invalid(format!(
                    "segment {i} has more file bytes than memory"
                )));
            }
            let data = elf.segment_data(i)?;
            let at = p.vaddr as usize;
            bytes[at..at + data.len()].copy_from_slice(data);
            // PS5 files carry their dynamic tables in a segment the guest may
            // not touch; it keeps its bytes here but gets no region.
            if p.flags & (PF_R | PF_W | PF_X) == 0 {
                continue;
            }
            let start = p.vaddr / PAGE_SIZE * PAGE_SIZE;
            regions.push(Region {
                offset: start,
                size: (p.vaddr + p.memsz).next_multiple_of(PAGE_SIZE) - start,
                protection: Protection::from_flags(p.flags),
            });
        }
        regions.sort_by_key(|r| r.offset);
        if regions
            .windows(2)
            .any(|w| w[0].offset + w[0].size > w[1].offset)
        {
            return Err(Error::Unsupported("segments that share a page".into()));
        }

        let tls = elf.find(PT_TLS).map(|(_, p)| Tls {
            init_offset: p.vaddr,
            init_size: p.filesz,
            size: p.memsz,
            align: p.align,
        });
        let proc_param = elf.find(PT_SCE_PROCPARAM).map(|(_, p)| base + p.vaddr);

        let mut image = Self {
            base,
            bytes,
            regions,
            entry: base + elf.header().entry,
            tls,
            proc_param,
            missing: Vec::new(),
        };
        let mut resolved: Vec<Option<Option<u64>>> = vec![None; dynamic.symbols.len()];
        for rela in dynamic.relocations.iter().chain(&dynamic.jump_slots) {
            image.relocate(rela, dynamic, tls_module, resolver, &mut resolved)?;
        }
        Ok(image)
    }

    fn relocate(
        &mut self,
        r: &Rela,
        dynamic: &Dynamic,
        tls_module: u64,
        resolver: &mut dyn Resolver,
        resolved: &mut [Option<Option<u64>>],
    ) -> Result<()> {
        let value = match r.r_type {
            R_X86_64_RELATIVE => self.base.wrapping_add_signed(r.addend),
            R_X86_64_DTPMOD64 => tls_module,
            R_X86_64_64 | R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT => {
                let s = self.symbol(r.symbol, dynamic, resolver, resolved)?;
                if r.r_type == R_X86_64_64 {
                    s.wrapping_add_signed(r.addend)
                } else {
                    s
                }
            }
            other => return Err(Error::Unsupported(format!("relocation type {other}"))),
        };
        let at = usize::try_from(r.offset)
            .ok()
            .filter(|&a| a.checked_add(8).is_some_and(|end| end <= self.bytes.len()))
            .ok_or_else(|| {
                invalid(format!(
                    "relocation at {:#x} is outside the image",
                    r.offset
                ))
            })?;
        self.bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// The address a symbol binds to: this image's own definition, or what the
    /// resolver provides. Missing imports bind to 0 and are recorded once.
    fn symbol(
        &mut self,
        index: u32,
        dynamic: &Dynamic,
        resolver: &mut dyn Resolver,
        resolved: &mut [Option<Option<u64>>],
    ) -> Result<u64> {
        let slot = resolved.get_mut(index as usize).ok_or_else(|| {
            invalid(format!(
                "relocation names symbol {index}, which does not exist"
            ))
        })?;
        if let Some(done) = *slot {
            return Ok(done.unwrap_or(0));
        }
        let sym = &dynamic.symbols[index as usize];
        let addr = if sym.value != 0 {
            Some(self.base + sym.value)
        } else {
            match dynamic.resolve_name(index)? {
                Some(import) => {
                    let found = resolver.resolve(&import);
                    if found.is_none() {
                        self.missing.push(Missing {
                            nid: import.nid.to_owned(),
                            library: import.library.name.clone(),
                            module: import.module.name.clone(),
                        });
                    }
                    found
                }
                None => {
                    return Err(invalid(format!(
                        "import {:?} is not a NID#LIB#MOD name",
                        sym.name
                    )));
                }
            }
        };
        *slot = Some(addr);
        Ok(addr.unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic::tests::{DynBuilder, elf_with};
    use crate::nid::nid;

    struct Table(Vec<(String, u64)>);

    impl Resolver for Table {
        fn resolve(&mut self, i: &Import<'_>) -> Option<u64> {
            self.0.iter().find(|(n, _)| n == i.nid).map(|e| e.1)
        }
    }

    fn word(img: &Image, at: usize) -> u64 {
        u64::from_le_bytes(img.bytes[at..at + 8].try_into().unwrap())
    }

    fn load(b: DynBuilder, resolver: &mut dyn Resolver) -> Image {
        let elf = ElfFile::parse(elf_with(b, vec![0xcc; 0x200])).unwrap();
        let d = Dynamic::parse(&elf).unwrap();
        Image::load(&elf, &d, 0x40_0000, 1, resolver).unwrap()
    }

    #[test]
    fn applies_relative_and_symbol_relocations() {
        let mut b = DynBuilder::new();
        b.needed_module("libkernel", 1);
        b.import_lib("libkernel", 1);
        let write = b.import("sceKernelWrite", 1, 1);
        let local = b.symbol("local", 0x12, 0x80);
        b.rela(false, 0x100, R_X86_64_RELATIVE, 0, 0x40);
        b.rela(false, 0x108, R_X86_64_64, local, 8);
        b.rela(false, 0x110, R_X86_64_DTPMOD64, 0, 0);
        b.rela(true, 0x118, R_X86_64_JUMP_SLOT, write, 0);
        let img = load(b, &mut Table(vec![(nid("sceKernelWrite"), 0xdead_0000)]));
        assert_eq!(word(&img, 0x100), 0x40_0040);
        assert_eq!(word(&img, 0x108), 0x40_0088);
        assert_eq!(word(&img, 0x110), 1);
        assert_eq!(word(&img, 0x118), 0xdead_0000);
        assert!(img.missing.is_empty());
        assert_eq!(img.bytes.len() as u64, PAGE_SIZE);
        assert_eq!(
            img.regions[0].protection,
            Protection {
                read: true,
                write: false,
                execute: true
            }
        );
    }

    #[test]
    fn reports_each_missing_import_once() {
        let mut b = DynBuilder::new();
        b.needed_module("libkernel", 1);
        b.import_lib("libkernel", 1);
        let write = b.import("sceKernelWrite", 1, 1);
        b.rela(true, 0x100, R_X86_64_JUMP_SLOT, write, 0);
        b.rela(false, 0x108, R_X86_64_GLOB_DAT, write, 0);
        let img = load(b, &mut NoImports);
        assert_eq!(
            img.missing,
            [Missing {
                nid: nid("sceKernelWrite"),
                library: "libkernel".into(),
                module: "libkernel".into()
            }]
        );
        assert_eq!(word(&img, 0x100), 0);
    }

    #[test]
    fn relocations_outside_the_image_are_errors() {
        let mut b = DynBuilder::new();
        b.rela(false, 0x1_0000, R_X86_64_RELATIVE, 0, 0);
        let elf = ElfFile::parse(elf_with(b, vec![0; 0x10])).unwrap();
        let d = Dynamic::parse(&elf).unwrap();
        assert!(Image::load(&elf, &d, 0x40_0000, 1, &mut NoImports).is_err());
    }

    #[test]
    fn base_must_be_page_aligned() {
        let elf = ElfFile::parse(elf_with(DynBuilder::new(), vec![0; 0x10])).unwrap();
        let d = Dynamic::parse(&elf).unwrap();
        assert!(Image::load(&elf, &d, 0x1000, 1, &mut NoImports).is_err());
    }
}
