//! Turns an ELF that `ld.lld -shared` linked for `x86_64-unknown-freebsd` into
//! an Orbis executable: `ET_SCE_DYNEXEC`, Sony dynamic tags, and imports
//! renamed to `NID#LIB#MOD`. Only for test programs we write ourselves; the
//! format facts are in `docs/loader.md`.

use std::collections::BTreeMap;
use std::fmt;

use loader::dynamic::{
    DT_SCE_IMPORT_LIB, DT_SCE_JMPREL, DT_SCE_MODULE_INFO, DT_SCE_NEEDED_MODULE, DT_SCE_PLTGOT,
    DT_SCE_PLTRELSZ, DT_SCE_RELA, DT_SCE_RELASZ, DT_SCE_STRSZ, DT_SCE_STRTAB, DT_SCE_SYMTAB,
    DT_SCE_SYMTABSZ, R_X86_64_64, R_X86_64_DTPMOD64, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT,
    R_X86_64_RELATIVE, RELA_SIZE, SYM_SIZE,
};
use loader::elf::{
    ELF_HEADER_SIZE, EM_X86_64, ET_SCE_DYNEXEC, PF_R, PHDR_SIZE, PT_DYNAMIC, PT_LOAD,
    PT_SCE_DYNLIBDATA, PT_TLS,
};
use loader::nid::{encode_id, nid};

const ET_DYN: u16 = 3;
const SHT_DYNSYM: u32 = 11;
const SHDR_SIZE: usize = 0x40;

const DT_NULL: u64 = 0;
const DT_PLTRELSZ: u64 = 2;
const DT_PLTGOT: u64 = 3;
const DT_STRTAB: u64 = 5;
const DT_SYMTAB: u64 = 6;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_STRSZ: u64 = 10;
const DT_JMPREL: u64 = 23;
/// Standard tags whose values are image addresses we pass through unchanged:
/// init, fini, their arrays and sizes, and the preinit array and its size.
const KEPT_TAGS: [u64; 8] = [12, 13, 25, 26, 27, 28, 32, 33];

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn err<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error(msg.into()))
}

/// Which library and module provide each imported symbol.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ImportSpec {
    providers: BTreeMap<String, (String, String)>,
}

impl ImportSpec {
    /// Parses lines of `SYMBOL LIBRARY [MODULE]`; the module defaults to the
    /// library's name. `#` starts a comment.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let mut providers = BTreeMap::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let words: Vec<&str> = line.split_whitespace().collect();
            let (symbol, library, module) = match words[..] {
                [s, l] => (s, l, l),
                [s, l, m] => (s, l, m),
                _ => return err(format!("line {}: expected SYMBOL LIBRARY [MODULE]", n + 1)),
            };
            providers.insert(symbol.to_owned(), (library.to_owned(), module.to_owned()));
        }
        Ok(Self { providers })
    }
}

struct Input<'a> {
    file: &'a [u8],
    entry: u64,
    phdrs: Vec<Phdr>,
    tags: Vec<(u64, u64)>,
    dynsym_count: usize,
}

#[derive(Clone, Copy)]
struct Phdr {
    p_type: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

fn rd16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}

fn rd32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn rd64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn get(b: &[u8], at: u64, len: u64) -> Result<&[u8], Error> {
    let at = usize::try_from(at).map_err(|_| Error("offset out of range".into()))?;
    let len = usize::try_from(len).map_err(|_| Error("length out of range".into()))?;
    match at.checked_add(len).and_then(|end| b.get(at..end)) {
        Some(s) => Ok(s),
        None => err(format!(
            "{len} bytes at {at:#x} are past the end of the input"
        )),
    }
}

impl<'a> Input<'a> {
    fn parse(file: &'a [u8]) -> Result<Self, Error> {
        let h = get(file, 0, ELF_HEADER_SIZE as u64)?;
        if &h[..4] != b"\x7fELF" || h[4] != 2 || h[5] != 1 {
            return err("input is not a 64-bit little-endian ELF");
        }
        if rd16(h, 0x10) != ET_DYN || rd16(h, 0x12) != EM_X86_64 {
            return err("input must be an x86-64 shared object (link with ld.lld -shared)");
        }
        let phoff = rd64(h, 0x20);
        let phnum = rd16(h, 0x38) as u64;
        let phdrs: Vec<Phdr> = get(file, phoff, phnum * PHDR_SIZE as u64)?
            .chunks(PHDR_SIZE)
            .map(|p| Phdr {
                p_type: rd32(p, 0),
                flags: rd32(p, 4),
                offset: rd64(p, 8),
                vaddr: rd64(p, 16),
                filesz: rd64(p, 32),
                memsz: rd64(p, 40),
                align: rd64(p, 48),
            })
            .collect();
        let shoff = rd64(h, 0x28);
        let shnum = rd16(h, 0x3c) as u64;
        let dynsym_count = get(file, shoff, shnum * SHDR_SIZE as u64)?
            .chunks(SHDR_SIZE)
            .find(|s| rd32(s, 4) == SHT_DYNSYM)
            .map(|s| rd64(s, 0x20) as usize / SYM_SIZE)
            .ok_or_else(|| Error("input has no .dynsym section".into()))?;
        let dynamic = phdrs
            .iter()
            .find(|p| p.p_type == PT_DYNAMIC)
            .ok_or_else(|| Error("input has no PT_DYNAMIC".into()))?;
        let tags = get(file, dynamic.offset, dynamic.filesz)?
            .chunks(16)
            .filter(|t| t.len() == 16)
            .map(|t| (rd64(t, 0), rd64(t, 8)))
            .take_while(|t| t.0 != DT_NULL)
            .collect();
        Ok(Self {
            file,
            entry: rd64(h, 0x18),
            phdrs,
            tags,
            dynsym_count,
        })
    }

    fn tag(&self, t: u64) -> Option<u64> {
        self.tags.iter().find(|x| x.0 == t).map(|x| x.1)
    }

    /// The input bytes at an image address.
    fn at_vaddr(&self, vaddr: u64, len: u64) -> Result<&'a [u8], Error> {
        let p = self
            .phdrs
            .iter()
            .find(|p| p.p_type == PT_LOAD && vaddr >= p.vaddr && vaddr + len <= p.vaddr + p.filesz)
            .ok_or_else(|| Error(format!("address {vaddr:#x} is not in a loaded segment")))?;
        get(self.file, p.offset + (vaddr - p.vaddr), len)
    }

    fn table(&self, addr_tag: u64, size_tag: u64) -> Result<&'a [u8], Error> {
        match (self.tag(addr_tag), self.tag(size_tag)) {
            (Some(a), Some(s)) => self.at_vaddr(a, s),
            (None, None) => Ok(&[]),
            _ => err(format!("dynamic tag {addr_tag} or {size_tag} is missing")),
        }
    }
}

const SUPPORTED_RELOCATIONS: [u32; 5] = [
    R_X86_64_64,
    R_X86_64_GLOB_DAT,
    R_X86_64_JUMP_SLOT,
    R_X86_64_RELATIVE,
    R_X86_64_DTPMOD64,
];

/// Converts `input` into an Orbis executable whose module is named `module_name`.
pub fn convert(input: &[u8], imports: &ImportSpec, module_name: &str) -> Result<Vec<u8>, Error> {
    let inp = Input::parse(input)?;
    let strsz = inp
        .tag(DT_STRSZ)
        .ok_or_else(|| Error("input has no DT_STRSZ".into()))?;
    let old_strtab = inp.at_vaddr(inp.tag(DT_STRTAB).unwrap_or(0), strsz)?;
    let old_symtab = inp.at_vaddr(
        inp.tag(DT_SYMTAB)
            .ok_or_else(|| Error("input has no DT_SYMTAB".into()))?,
        (inp.dynsym_count * SYM_SIZE) as u64,
    )?;
    let rela = inp.table(DT_RELA, DT_RELASZ)?;
    let jmprel = inp.table(DT_JMPREL, DT_PLTRELSZ)?;
    for r in rela.chunks(RELA_SIZE).chain(jmprel.chunks(RELA_SIZE)) {
        let t = rd64(r, 8) as u32;
        if !SUPPORTED_RELOCATIONS.contains(&t) {
            return err(format!("relocation type {t} is not supported"));
        }
    }

    let mut strtab = vec![0u8];
    let mut add_str = |s: &str| {
        let off = strtab.len() as u64;
        strtab.extend_from_slice(s.as_bytes());
        strtab.push(0);
        off
    };
    let mut libs: Vec<String> = Vec::new();
    let mut mods: Vec<String> = Vec::new();
    let mut symtab = Vec::with_capacity(old_symtab.len());
    for (i, s) in old_symtab.chunks(SYM_SIZE).enumerate() {
        let name = old_strtab
            .get(rd32(s, 0) as usize..)
            .and_then(|r| r.split(|&c| c == 0).next())
            .and_then(|n| std::str::from_utf8(n).ok())
            .ok_or_else(|| Error(format!("symbol {i} has a bad name")))?;
        let undefined = rd16(s, 6) == 0;
        let new_name = if i > 0 && undefined && !name.is_empty() {
            let (lib, module) = imports
                .providers
                .get(name)
                .ok_or_else(|| Error(format!("no library given for import {name}")))?;
            let lib_id = id_of(&mut libs, lib);
            let mod_id = id_of(&mut mods, module);
            format!("{}#{}#{}", nid(name), encode_id(lib_id), encode_id(mod_id))
        } else {
            name.to_owned()
        };
        let off = if new_name.is_empty() {
            0
        } else {
            add_str(&new_name)
        };
        let mut sym = s.to_vec();
        sym[0..4].copy_from_slice(&(off as u32).to_le_bytes());
        symtab.extend_from_slice(&sym);
    }

    let mut tags: Vec<(u64, u64)> = vec![(DT_SCE_MODULE_INFO, add_str(module_name) | (1 << 40))];
    for (i, m) in mods.iter().enumerate() {
        tags.push((
            DT_SCE_NEEDED_MODULE,
            add_str(m) | (1 << 40) | ((i as u64 + 1) << 48),
        ));
    }
    for (i, l) in libs.iter().enumerate() {
        tags.push((
            DT_SCE_IMPORT_LIB,
            add_str(l) | (1 << 32) | ((i as u64 + 1) << 48),
        ));
    }

    let mut data = Vec::new();
    let mut place = |bytes: &[u8], off_tag, size_tag, tags: &mut Vec<(u64, u64)>| {
        tags.push((off_tag, data.len() as u64));
        tags.push((size_tag, bytes.len() as u64));
        data.extend_from_slice(bytes);
        data.resize(data.len().next_multiple_of(16), 0);
    };
    place(&strtab, DT_SCE_STRTAB, DT_SCE_STRSZ, &mut tags);
    place(&symtab, DT_SCE_SYMTAB, DT_SCE_SYMTABSZ, &mut tags);
    place(rela, DT_SCE_RELA, DT_SCE_RELASZ, &mut tags);
    place(jmprel, DT_SCE_JMPREL, DT_SCE_PLTRELSZ, &mut tags);
    if let Some(v) = inp.tag(DT_PLTGOT) {
        tags.push((DT_SCE_PLTGOT, v));
    }
    tags.extend(inp.tags.iter().filter(|t| KEPT_TAGS.contains(&t.0)));
    tags.push((DT_NULL, 0));
    let dynamic: Vec<u8> = tags
        .iter()
        .flat_map(|&(t, v)| t.to_le_bytes().into_iter().chain(v.to_le_bytes()))
        .collect();

    let mut out = input.to_vec();
    out.resize(out.len().next_multiple_of(16), 0);
    let mut phdrs: Vec<Phdr> = inp
        .phdrs
        .iter()
        .copied()
        .filter(|p| p.p_type == PT_LOAD || p.p_type == PT_TLS)
        .collect();
    let data_off = out.len() as u64;
    out.extend_from_slice(&data);
    let dyn_off = out.len() as u64;
    out.extend_from_slice(&dynamic);
    for (p_type, offset, len) in [
        (PT_SCE_DYNLIBDATA, data_off, data.len()),
        (PT_DYNAMIC, dyn_off, dynamic.len()),
    ] {
        phdrs.push(Phdr {
            p_type,
            flags: PF_R,
            offset,
            vaddr: 0,
            filesz: len as u64,
            memsz: 0,
            align: 16,
        });
    }
    out.resize(out.len().next_multiple_of(8), 0);
    let phoff = out.len() as u64;
    for p in &phdrs {
        let mut b = [0u8; PHDR_SIZE];
        b[0..4].copy_from_slice(&p.p_type.to_le_bytes());
        b[4..8].copy_from_slice(&p.flags.to_le_bytes());
        b[8..16].copy_from_slice(&p.offset.to_le_bytes());
        b[16..24].copy_from_slice(&p.vaddr.to_le_bytes());
        b[24..32].copy_from_slice(&p.vaddr.to_le_bytes());
        b[32..40].copy_from_slice(&p.filesz.to_le_bytes());
        b[40..48].copy_from_slice(&p.memsz.to_le_bytes());
        b[48..56].copy_from_slice(&p.align.to_le_bytes());
        out.extend_from_slice(&b);
    }
    let h = &mut out[..ELF_HEADER_SIZE];
    h[0x10..0x12].copy_from_slice(&ET_SCE_DYNEXEC.to_le_bytes());
    h[0x18..0x20].copy_from_slice(&inp.entry.to_le_bytes());
    h[0x20..0x28].copy_from_slice(&phoff.to_le_bytes());
    h[0x28..0x30].fill(0);
    h[0x38..0x3a].copy_from_slice(&(phdrs.len() as u16).to_le_bytes());
    h[0x3c..0x40].fill(0);
    Ok(out)
}

/// The 1-based id of `name` in `list`, adding it if new.
fn id_of(list: &mut Vec<String>, name: &str) -> u16 {
    match list.iter().position(|n| n == name) {
        Some(i) => i as u16 + 1,
        None => {
            list.push(name.to_owned());
            list.len() as u16
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loader::image::Resolver;
    use loader::{Dynamic, ElfFile, Image};

    /// A tiny shared object laid out as `ld.lld -shared` would (vaddr equals
    /// file offset), with a section header table naming `.dynsym`.
    fn shared_object() -> Vec<u8> {
        let mut f = vec![0u8; 0x1000];
        let strtab = b"\0write\0exit\0_start\0";
        let str_at = 0x200;
        f[str_at..str_at + strtab.len()].copy_from_slice(strtab);
        let sym_at = 0x300;
        let syms: [(u32, u8, u16, u64); 4] = [
            (0, 0, 0, 0),
            (1, 0x12, 0, 0),
            (7, 0x12, 0, 0),
            (12, 0x12, 1, 0x800),
        ];
        for (i, (name, info, shndx, value)) in syms.iter().enumerate() {
            let s = &mut f[sym_at + i * SYM_SIZE..][..SYM_SIZE];
            s[0..4].copy_from_slice(&name.to_le_bytes());
            s[4] = *info;
            s[6..8].copy_from_slice(&shndx.to_le_bytes());
            s[8..16].copy_from_slice(&value.to_le_bytes());
        }
        let rela_at = 0x400;
        let relas: [(u64, u64, i64); 3] = [
            (0x900, R_X86_64_RELATIVE as u64, 0x800),
            (0x908, (1 << 32) | R_X86_64_JUMP_SLOT as u64, 0),
            (0x910, (2 << 32) | R_X86_64_JUMP_SLOT as u64, 0),
        ];
        for (i, (off, info, add)) in relas.iter().enumerate() {
            let r = &mut f[rela_at + i * RELA_SIZE..][..RELA_SIZE];
            r[0..8].copy_from_slice(&off.to_le_bytes());
            r[8..16].copy_from_slice(&info.to_le_bytes());
            r[16..24].copy_from_slice(&add.to_le_bytes());
        }
        let dyn_at = 0x500;
        let tags = [
            (DT_STRTAB, str_at as u64),
            (DT_STRSZ, strtab.len() as u64),
            (DT_SYMTAB, sym_at as u64),
            (DT_RELA, rela_at as u64),
            (DT_RELASZ, RELA_SIZE as u64),
            (DT_JMPREL, (rela_at + RELA_SIZE) as u64),
            (DT_PLTRELSZ, 2 * RELA_SIZE as u64),
            (DT_NULL, 0),
        ];
        for (i, (t, v)) in tags.iter().enumerate() {
            f[dyn_at + i * 16..][..8].copy_from_slice(&t.to_le_bytes());
            f[dyn_at + i * 16 + 8..][..8].copy_from_slice(&v.to_le_bytes());
        }
        let sh_at = 0x600;
        f[sh_at + SHDR_SIZE + 4..][..4].copy_from_slice(&SHT_DYNSYM.to_le_bytes());
        f[sh_at + SHDR_SIZE + 0x20..][..8]
            .copy_from_slice(&((syms.len() * SYM_SIZE) as u64).to_le_bytes());
        let ph_at = 0x40;
        let phdrs = [
            (PT_LOAD, 5u32, 0u64, 0x1000u64),
            (PT_DYNAMIC, 4, dyn_at as u64, tags.len() as u64 * 16),
        ];
        for (i, (t, fl, off, sz)) in phdrs.iter().enumerate() {
            let p = &mut f[ph_at + i * PHDR_SIZE..][..PHDR_SIZE];
            p[0..4].copy_from_slice(&t.to_le_bytes());
            p[4..8].copy_from_slice(&fl.to_le_bytes());
            p[8..16].copy_from_slice(&off.to_le_bytes());
            p[16..24].copy_from_slice(&off.to_le_bytes());
            p[32..40].copy_from_slice(&sz.to_le_bytes());
            p[40..48].copy_from_slice(&sz.to_le_bytes());
            p[48..56].copy_from_slice(&0x1000u64.to_le_bytes());
        }
        f[..4].copy_from_slice(b"\x7fELF");
        f[4] = 2;
        f[5] = 1;
        f[0x10..0x12].copy_from_slice(&ET_DYN.to_le_bytes());
        f[0x12..0x14].copy_from_slice(&EM_X86_64.to_le_bytes());
        f[0x18..0x20].copy_from_slice(&0x800u64.to_le_bytes());
        f[0x20..0x28].copy_from_slice(&(ph_at as u64).to_le_bytes());
        f[0x28..0x30].copy_from_slice(&(sh_at as u64).to_le_bytes());
        f[0x36..0x38].copy_from_slice(&(PHDR_SIZE as u16).to_le_bytes());
        f[0x38..0x3a].copy_from_slice(&(phdrs.len() as u16).to_le_bytes());
        f[0x3a..0x3c].copy_from_slice(&(SHDR_SIZE as u16).to_le_bytes());
        f[0x3c..0x3e].copy_from_slice(&2u16.to_le_bytes());
        f
    }

    struct ByName(Vec<(&'static str, u64)>);

    impl Resolver for ByName {
        fn resolve(&mut self, i: &loader::dynamic::Import<'_>) -> Option<u64> {
            self.0.iter().find(|(n, _)| nid(n) == i.nid).map(|e| e.1)
        }
    }

    fn spec() -> ImportSpec {
        ImportSpec::parse("write libkernel\nexit libkernel # comment\n").unwrap()
    }

    #[test]
    fn output_loads_with_imports_under_their_library() {
        let out = convert(&shared_object(), &spec(), "hello").unwrap();
        let elf = ElfFile::parse(out).unwrap();
        assert_eq!(elf.header().e_type, ET_SCE_DYNEXEC);
        let d = Dynamic::parse(&elf).unwrap();
        assert_eq!(d.modules[0].name, "hello");
        let imports = d.imports().unwrap();
        let got: Vec<_> = imports
            .iter()
            .map(|i| (i.nid.to_owned(), i.library.name.as_str()))
            .collect();
        assert_eq!(
            got,
            [(nid("write"), "libkernel"), (nid("exit"), "libkernel")]
        );

        let mut r = ByName(vec![("write", 0x1111), ("exit", 0x2222)]);
        let img = Image::load(&elf, &d, 0x40_0000, 1, &mut r).unwrap();
        let word = |at: usize| u64::from_le_bytes(img.bytes[at..at + 8].try_into().unwrap());
        assert_eq!(word(0x900), 0x40_0800);
        assert_eq!((word(0x908), word(0x910)), (0x1111, 0x2222));
        assert_eq!(img.entry, 0x40_0800);
        assert!(img.missing.is_empty());
    }

    #[test]
    fn imports_without_a_library_are_errors() {
        let spec = ImportSpec::parse("write libkernel").unwrap();
        assert!(convert(&shared_object(), &spec, "hello").is_err());
    }

    #[test]
    fn spec_lines_take_an_optional_module() {
        let s = ImportSpec::parse("puts libc libSceLibcInternal\nwrite libkernel").unwrap();
        assert_eq!(
            s.providers["puts"],
            ("libc".into(), "libSceLibcInternal".into())
        );
        assert_eq!(
            s.providers["write"],
            ("libkernel".into(), "libkernel".into())
        );
        assert!(ImportSpec::parse("lonely").is_err());
    }

    #[test]
    fn rejects_inputs_that_are_not_shared_objects() {
        let mut f = shared_object();
        f[0x10] = 2;
        assert!(convert(&f, &spec(), "hello").is_err());
    }
}
