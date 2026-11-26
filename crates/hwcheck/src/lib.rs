//! Instruction tests for the hardware oracle: small compute programs that run
//! one GFX10 instruction over many lanes of operands, once through ps5slop's
//! own path (PM4 processor, translator, Metal) and once as raw PM4 on a real
//! AMD GPU, and the lane-by-lane comparison of the two. See docs/shaders.md.
//!
//! A [`Case`] is self-contained: the compute registers, the memory the
//! program starts with (its code and the operands), and the memory Metal
//! left. The Mac writes cases; the machine with the AMD GPU runs them.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

#[cfg(target_os = "linux")]
pub mod amdgpu;
pub mod chart;
pub mod compat;
pub mod coverage;
pub mod hardware;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "linux")]
pub mod run;
pub mod suite;

/// Page granularity of a case's memory.
pub const PAGE: u64 = 16 << 10;

/// First SH register a case keeps (`COMPUTE_DISPATCH_INITIATOR`), as an
/// offset in the SH bank.
pub const COMPUTE_FIRST: usize = 0x200;
/// Number of SH registers a case keeps, through `COMPUTE_USER_DATA_15`.
pub const COMPUTE_COUNT: usize = 0x50;

const COMPUTE_START_X: usize = 0x204;
const COMPUTE_PGM_LO: usize = 0x20C;
const COMPUTE_PGM_RSRC1: usize = 0x212;
const COMPUTE_RESOURCE_LIMITS: usize = 0x215;
const COMPUTE_STATIC_THREAD_MGMT_SE0: usize = 0x216;
const COMPUTE_TMPRING_SIZE: usize = 0x218;
const COMPUTE_STATIC_THREAD_MGMT_SE2: usize = 0x219;
const COMPUTE_PGM_RSRC3: usize = 0x228;
const COMPUTE_USER_DATA_0: usize = 0x240;

const HEADER: [u8; 8] = *b"PS5ITEST";
const FORMAT: u32 = 1;
const TAG_META: [u8; 4] = *b"META";
const TAG_INPUT: [u8; 4] = *b"INPT";
const TAG_OUTPUT: [u8; 4] = *b"OUTP";

/// One test program with its operands and, once run on the Mac, Metal's result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Case {
    /// The opcode under test, by its LLVM name.
    pub opcode: String,
    /// What this variant of the test is about, e.g. "e64 neg abs" or "denormals kept".
    pub variant: String,
    /// The instruction under test in LLVM syntax.
    pub text: String,
    /// The program's address (`COMPUTE_PGM_LO/HI`).
    pub program: u64,
    /// SH registers `COMPUTE_FIRST..COMPUTE_FIRST + COMPUTE_COUNT`.
    pub compute: Vec<u32>,
    /// Groups of one wave64 each.
    pub groups: u32,
    /// One record per wave instead of one per lane.
    pub per_wave: bool,
    /// Where record `i` reads its `in_words` dwords and writes its `out_words`.
    pub inputs: u64,
    pub outputs: u64,
    pub in_words: u32,
    pub out_words: u32,
    /// What each output dword holds, for reports.
    pub out_names: Vec<String>,
    /// Memory compared as a whole besides the records, `(address, bytes)`:
    /// where stores land. Zero bytes for none.
    pub checked: (u64, u32),
    /// A draw rather than a dispatch: `stream` holds its command buffer and
    /// it runs on the graphics queue.
    pub graphics: bool,
    /// The command buffer when it is not the compute dispatch [`Case::commands`] builds.
    pub stream: Vec<u32>,
    /// Packets only the hardware gets before `stream`: the state a driver
    /// sets up once (CLEAR_STATE and the like), which ps5slop never sees.
    pub preamble: Vec<u32>,
    /// Packets only the hardware gets after `stream`: the cache flushes a
    /// driver ends its command buffer with, so render target writes reach memory.
    pub postamble: Vec<u32>,
    /// Opcodes besides `opcode` whose behaviour the outcome also shows (a
    /// draw's exports and interpolation); results list it under each.
    pub covers: Vec<String>,
    /// The render target a draw writes: address, width, height, row pitch in
    /// bytes. Metal's target is laid out there for comparing.
    pub target: (u64, u32, u32, u32),
    /// Every page the program starts with.
    pub input: BTreeMap<u64, Vec<u8>>,
    /// Every page Metal's run changed, as Metal left it.
    pub output: BTreeMap<u64, Vec<u8>>,
    /// Why Metal did not run it (the translator refused it), if so.
    pub untranslated: Option<String>,
    /// How to read the destination when classifying differences.
    pub dst: Dst,
}

/// The type of a test's destination, as far as comparing it cares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dst {
    #[default]
    Bits,
    F16,
    PkF16,
    F32,
    F64,
}

impl Dst {
    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(c: u8) -> Self {
        [Dst::Bits, Dst::F16, Dst::PkF16, Dst::F32, Dst::F64]
            .get(c as usize)
            .copied()
            .unwrap_or_default()
    }

    /// The float lanes of a destination `(lo, hi)`, as (bits, width) pairs.
    fn floats(self, lo: u32, hi: u32) -> Vec<(u64, u32)> {
        match self {
            Dst::Bits => Vec::new(),
            Dst::F16 => vec![(lo as u64 & 0xFFFF, 16)],
            Dst::PkF16 => vec![(lo as u64 & 0xFFFF, 16), (lo as u64 >> 16, 16)],
            Dst::F32 => vec![(lo as u64, 32)],
            Dst::F64 => vec![(lo as u64 | (hi as u64) << 32, 64)],
        }
    }
}

/// What kind of difference a lane shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Both results are NaN with different bits (payload, sign, quieting).
    NanBits,
    /// One side flushed a denormal (or zero's sign) the other kept.
    Denormal,
    /// Any other difference.
    Value,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::NanBits => "NaN bits",
            Kind::Denormal => "denormal or zero sign",
            Kind::Value => "value",
        }
    }
}

fn float_parts(bits: u64, width: u32) -> (bool, bool) {
    let (exp_bits, man_bits) = match width {
        16 => (5, 10),
        32 => (8, 23),
        _ => (11, 52),
    };
    let exp = bits >> man_bits & ((1 << exp_bits) - 1);
    let man = bits & ((1 << man_bits) - 1);
    let nan = exp == (1 << exp_bits) - 1 && man != 0;
    (nan, exp == 0)
}

struct Meta {
    opcode: String,
    variant: String,
    text: String,
    program: u64,
    compute: Vec<u32>,
    groups: u32,
    per_wave: bool,
    inputs: u64,
    outputs: u64,
    in_words: u32,
    out_words: u32,
    out_names: Vec<String>,
    checked: (u64, u32),
    graphics: bool,
    stream: Vec<u32>,
    preamble: Vec<u32>,
    postamble: Vec<u32>,
    covers: Vec<String>,
    target: (u64, u32, u32, u32),
    untranslated: Option<String>,
    dst: u8,
}

savestate::record!(Meta {
    opcode,
    variant,
    text,
    program,
    compute,
    groups,
    per_wave,
    inputs,
    outputs,
    in_words,
    out_words,
    out_names,
    checked,
    graphics,
    stream,
    preamble,
    postamble,
    covers,
    target,
    untranslated,
    dst
});

/// `kept` results lines with `new` ones replacing those of the same test
/// (opcode and variant), sorted.
pub fn merge_results(kept: &str, new: &str) -> String {
    let key = |l: &str| {
        let mut f = l.splitn(3, '\t');
        (
            f.next().unwrap_or("").to_string(),
            f.next().unwrap_or("").to_string(),
        )
    };
    let mut lines: BTreeMap<(String, String), String> = BTreeMap::new();
    for l in kept.lines().chain(new.lines()).filter(|l| !l.is_empty()) {
        lines.insert(key(l), l.to_string());
    }
    let mut out = String::new();
    for l in lines.into_values() {
        out.push_str(&l);
        out.push('\n');
    }
    out
}

/// Guest memory as a test program sees it.
pub trait Memory {
    /// Reads `buf.len()` bytes at `addr`; memory nobody holds reads as zeros.
    fn read(&self, addr: u64, buf: &mut [u8]);
    /// Writes `bytes` at `addr`.
    fn write(&mut self, addr: u64, bytes: &[u8]);

    /// The machine code at `addr`, as 64 KiB of words.
    fn code(&self, addr: u64) -> Vec<u32> {
        let mut bytes = vec![0u8; 64 << 10];
        self.read(addr, &mut bytes);
        words(&bytes)
    }
}

/// Little-endian dwords of `bytes`.
pub fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Calls `f` with each page-sized piece of `[addr, addr + len)`: the page,
/// the offset in it, and the range in the whole.
pub fn by_page(addr: u64, len: usize, mut f: impl FnMut(u64, usize, std::ops::Range<usize>)) {
    let mut done = 0;
    while done < len {
        let at = addr + done as u64;
        let page = at & !(PAGE - 1);
        let offset = (at - page) as usize;
        let n = (PAGE as usize - offset).min(len - done);
        f(page, offset, done..done + n);
        done += n;
    }
}

/// Reads `buf` at `addr` from the first of `layers` holding each page, zeros elsewhere.
pub fn read_pages(layers: &[&BTreeMap<u64, Vec<u8>>], addr: u64, buf: &mut [u8]) {
    by_page(addr, buf.len(), |page, offset, range| {
        let dst = &mut buf[range];
        match layers.iter().find_map(|l| l.get(&page)) {
            Some(src) => dst.copy_from_slice(&src[offset..offset + dst.len()]),
            None => dst.fill(0),
        }
    });
}

/// Writes `bytes` at `addr` into `pages`, adding zeroed pages as needed.
pub fn write_pages(pages: &mut BTreeMap<u64, Vec<u8>>, addr: u64, bytes: &[u8]) {
    by_page(addr, bytes.len(), |page, offset, range| {
        pages.entry(page).or_insert_with(|| vec![0; PAGE as usize])[offset..offset + range.len()]
            .copy_from_slice(&bytes[range]);
    });
}

impl Memory for Case {
    fn read(&self, addr: u64, buf: &mut [u8]) {
        read_pages(&[&self.input], addr, buf);
    }

    fn write(&mut self, addr: u64, bytes: &[u8]) {
        write_pages(&mut self.input, addr, bytes);
    }
}

impl Case {
    /// The value of SH register `reg` (an offset in the SH bank).
    pub fn reg(&self, reg: usize) -> u32 {
        reg.checked_sub(COMPUTE_FIRST)
            .and_then(|i| self.compute.get(i))
            .copied()
            .unwrap_or(0)
    }

    /// How many records the test reads and writes: one per lane or per wave.
    pub fn records(&self) -> u32 {
        if self.per_wave {
            self.groups
        } else {
            self.groups * 64
        }
    }

    /// A short name for reports.
    pub fn name(&self) -> String {
        format!("{} ({})", self.opcode, self.variant)
    }

    /// The command buffer that runs the case: its compute registers, then
    /// its dispatch as wave64 groups. The same packets go to ps5slop's
    /// command processor and to the hardware; registers that pick compute
    /// units or scratch are set for whatever GPU runs them.
    pub fn commands(&self) -> Vec<u32> {
        if !self.stream.is_empty() {
            return self.stream.clone();
        }
        const SET_SH_REG: u32 = 0x76;
        const DISPATCH_DIRECT: u32 = 0x15;
        const NOP: u32 = 0x10;
        const COMPUTE_SHADER_EN: u32 = 1;
        let header =
            |op: u32, body: usize| (3 << 30) | ((body as u32 - 1) << 16) | (op << 8) | (1 << 1);
        let mut ib = Vec::new();
        let mut set = |reg: usize, values: &[u32]| {
            ib.push(header(SET_SH_REG, values.len() + 1));
            ib.push(reg as u32);
            ib.extend_from_slice(values);
        };
        let regs = |first: usize, n: usize| -> Vec<u32> {
            (first..first + n).map(|r| self.reg(r)).collect()
        };
        set(COMPUTE_START_X, &regs(COMPUTE_START_X, 6));
        set(COMPUTE_PGM_LO, &regs(COMPUTE_PGM_LO, 2));
        set(COMPUTE_PGM_RSRC1, &regs(COMPUTE_PGM_RSRC1, 2));
        set(COMPUTE_RESOURCE_LIMITS, &[0]);
        set(COMPUTE_STATIC_THREAD_MGMT_SE0, &[u32::MAX; 2]);
        set(COMPUTE_TMPRING_SIZE, &regs(COMPUTE_TMPRING_SIZE, 1));
        set(COMPUTE_STATIC_THREAD_MGMT_SE2, &[u32::MAX; 2]);
        set(COMPUTE_PGM_RSRC3, &regs(COMPUTE_PGM_RSRC3, 1));
        set(COMPUTE_USER_DATA_0, &regs(COMPUTE_USER_DATA_0, 16));
        ib.push(header(DISPATCH_DIRECT, 4));
        ib.extend_from_slice(&[self.groups, 1, 1, COMPUTE_SHADER_EN]);
        while ib.len() % 8 != 0 {
            ib.push((3 << 30) | (0x3FFF << 16) | (NOP << 8));
        }
        ib
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        self.write_to(std::io::BufWriter::new(file))
    }

    pub fn write_to<W: Write>(&self, out: W) -> Result<(), String> {
        let mut header = HEADER.to_vec();
        header.extend_from_slice(&FORMAT.to_le_bytes());
        let mut w = savestate::Writer::new(out, &header).map_err(|e| e.to_string())?;
        let meta = Meta {
            opcode: self.opcode.clone(),
            variant: self.variant.clone(),
            text: self.text.clone(),
            program: self.program,
            compute: self.compute.clone(),
            groups: self.groups,
            per_wave: self.per_wave,
            inputs: self.inputs,
            outputs: self.outputs,
            in_words: self.in_words,
            out_words: self.out_words,
            out_names: self.out_names.clone(),
            checked: self.checked,
            graphics: self.graphics,
            stream: self.stream.clone(),
            preamble: self.preamble.clone(),
            postamble: self.postamble.clone(),
            covers: self.covers.clone(),
            target: self.target,
            untranslated: self.untranslated.clone(),
            dst: self.dst.code(),
        };
        w.section(TAG_META, &savestate::to_bytes(&meta))
            .map_err(|e| e.to_string())?;
        for (tag, pages) in [(TAG_INPUT, &self.input), (TAG_OUTPUT, &self.output)] {
            let mut raw = Vec::with_capacity(8 + pages.len() * (8 + PAGE as usize));
            raw.extend_from_slice(&(pages.len() as u64).to_le_bytes());
            for addr in pages.keys() {
                raw.extend_from_slice(&addr.to_le_bytes());
            }
            for bytes in pages.values() {
                raw.extend_from_slice(bytes);
            }
            w.section(tag, &raw).map_err(|e| e.to_string())?;
        }
        w.finish().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        Self::read_from(std::io::BufReader::new(file))
    }

    pub fn read_from<R: Read>(input: R) -> Result<Self, String> {
        let mut r = savestate::Reader::new(input).map_err(|e| e.to_string())?;
        if r.header.len() != 12 || r.header[..8] != HEADER {
            return Err("not a ps5slop instruction test".into());
        }
        let format = u32::from_le_bytes(r.header[8..12].try_into().expect("4 bytes"));
        if format != FORMAT {
            return Err(format!(
                "instruction test format {format}, but this build reads format {FORMAT}"
            ));
        }
        let meta: Meta = savestate::from_bytes(&r.expect(TAG_META).map_err(|e| e.0)?)
            .map_err(|e| e.to_string())?;
        let mut case = Case {
            opcode: meta.opcode,
            variant: meta.variant,
            text: meta.text,
            program: meta.program,
            compute: meta.compute,
            groups: meta.groups,
            per_wave: meta.per_wave,
            inputs: meta.inputs,
            outputs: meta.outputs,
            in_words: meta.in_words,
            out_words: meta.out_words,
            out_names: meta.out_names,
            checked: meta.checked,
            graphics: meta.graphics,
            stream: meta.stream,
            preamble: meta.preamble,
            postamble: meta.postamble,
            covers: meta.covers,
            target: meta.target,
            untranslated: meta.untranslated,
            dst: Dst::from_code(meta.dst),
            ..Case::default()
        };
        for tag in [TAG_INPUT, TAG_OUTPUT] {
            let raw = r.expect(tag).map_err(|e| e.0)?;
            let bad = || "corrupt page section in instruction test".to_string();
            let count = u64::from_le_bytes(raw.get(..8).ok_or_else(bad)?.try_into().unwrap());
            let count = count as usize;
            let data_at = 8 + count * 8;
            if raw.len() != data_at + count * PAGE as usize {
                return Err(bad());
            }
            let pages = if tag == TAG_INPUT {
                &mut case.input
            } else {
                &mut case.output
            };
            for i in 0..count {
                let addr = u64::from_le_bytes(raw[8 + i * 8..16 + i * 8].try_into().unwrap());
                let at = data_at + i * PAGE as usize;
                pages.insert(addr, raw[at..at + PAGE as usize].to_vec());
            }
        }
        Ok(case)
    }

    /// Every record's operands.
    pub fn record_inputs(&self) -> Vec<u32> {
        let mut bytes = vec![0u8; (self.records() * self.in_words * 4) as usize];
        read_pages(&[&self.input], self.inputs, &mut bytes);
        words(&bytes)
    }

    /// Every record's results as Metal left them.
    pub fn metal_outputs(&self) -> Vec<u32> {
        let mut bytes = vec![0u8; (self.records() * self.out_words * 4) as usize];
        read_pages(&[&self.output, &self.input], self.outputs, &mut bytes);
        words(&bytes)
    }

    fn kind(&self, m: &[u32], h: &[u32]) -> Kind {
        if m[2..] != h[2..] {
            return Kind::Value;
        }
        let ms = self.dst.floats(m[0], m[1]);
        let hs = self.dst.floats(h[0], h[1]);
        let mut kind = if ms.is_empty() {
            Kind::Value
        } else {
            Kind::NanBits
        };
        for (&(a, w), &(b, _)) in ms.iter().zip(&hs) {
            if a == b {
                continue;
            }
            let ((an, az), (bn, bz)) = (float_parts(a, w), float_parts(b, w));
            let k = if an && bn {
                Kind::NanBits
            } else if az && bz {
                Kind::Denormal
            } else {
                Kind::Value
            };
            kind = kind.max(k);
        }
        kind
    }

    /// The checked memory as Metal left it.
    pub fn metal_checked(&self) -> Vec<u32> {
        let mut bytes = vec![0u8; self.checked.1 as usize];
        read_pages(&[&self.output, &self.input], self.checked.0, &mut bytes);
        words(&bytes)
    }

    /// Compares Metal's results with `hardware`'s: every record's output
    /// words, then the checked memory.
    pub fn compare(&self, hardware: &[u32]) -> Outcome {
        let inputs = self.record_inputs();
        let metal = self.metal_outputs();
        let (hardware, hardware_memory) = hardware.split_at(metal.len());
        let memory = self
            .metal_checked()
            .iter()
            .zip(hardware_memory)
            .enumerate()
            .filter(|(_, (m, h))| m != h)
            .map(|(i, (&m, &h))| (self.checked.0 + 4 * i as u64, m, h))
            .collect();
        let (iw, ow) = (self.in_words as usize, self.out_words as usize);
        let mut diffs = Vec::new();
        for r in 0..self.records() as usize {
            let o = r * ow..(r + 1) * ow;
            let (m, h) = (&metal[o.clone()], &hardware[o]);
            if m != h {
                diffs.push(Diff {
                    kind: self.kind(m, h),
                    record: r as u32,
                    input: inputs[r * iw..(r + 1) * iw].to_vec(),
                    metal: m.to_vec(),
                    hardware: h.to_vec(),
                });
            }
        }
        Outcome {
            records: self.records(),
            per_wave: self.per_wave,
            out_names: self.out_names.clone(),
            diffs,
            memory,
        }
    }
}

/// One record whose results differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diff {
    pub kind: Kind,
    pub record: u32,
    pub input: Vec<u32>,
    pub metal: Vec<u32>,
    pub hardware: Vec<u32>,
}

/// The comparison of one case.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub records: u32,
    pub per_wave: bool,
    pub out_names: Vec<String>,
    pub diffs: Vec<Diff>,
    /// Dwords of the checked memory that differ: address, Metal's, hardware's.
    pub memory: Vec<(u64, u32, u32)>,
}

fn show(w: u32) -> String {
    let f = f32::from_bits(w);
    if f.is_finite() && w != 0 && (f.abs() > 1e-6 && f.abs() < 1e9) {
        format!("{w:08x} ({f})")
    } else {
        format!("{w:08x}")
    }
}

impl Outcome {
    pub fn agrees(&self) -> bool {
        self.diffs.is_empty() && self.memory.is_empty()
    }

    /// How many records differ in each way, e.g. "150 NaN bits, 8 value".
    pub fn kinds(&self) -> String {
        let mut counts = std::collections::BTreeMap::new();
        for d in &self.diffs {
            *counts.entry(d.kind).or_insert(0) += 1;
        }
        counts
            .iter()
            .map(|(k, n)| format!("{n} {}", k.name()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn unit(&self) -> &'static str {
        if self.per_wave { "waves" } else { "lanes" }
    }

    /// A report: how many records differ and, for the first `limit`, each
    /// record's operands and the output words that differ.
    pub fn report(&self, limit: usize) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        if self.agrees() {
            let _ = writeln!(s, "all {} {} agree", self.records, self.unit());
            return s;
        }
        if !self.memory.is_empty() {
            let _ = writeln!(s, "{} dwords of stored memory differ", self.memory.len());
            for &(addr, m, h) in self.memory.iter().take(limit) {
                let _ = writeln!(s, "    {addr:#x}: metal {}  hardware {}", show(m), show(h));
            }
        }
        let _ = writeln!(
            s,
            "{} of {} {} differ: {}",
            self.diffs.len(),
            self.records,
            self.unit(),
            self.kinds()
        );
        for d in self.diffs.iter().take(limit) {
            let ins: Vec<String> = d.input.iter().map(|&w| show(w)).collect();
            let _ = writeln!(
                s,
                "  {} {} ({}): in {}",
                &self.unit()[..self.unit().len() - 1],
                d.record,
                d.kind.name(),
                ins.join(" ")
            );
            for k in 0..d.metal.len() {
                if d.metal[k] != d.hardware[k] {
                    let name = self.out_names.get(k).map_or("?", String::as_str);
                    let _ = writeln!(
                        s,
                        "    {name:<6} metal {}  hardware {}",
                        show(d.metal[k]),
                        show(d.hardware[k])
                    );
                }
            }
        }
        if self.diffs.len() > limit {
            let _ = writeln!(s, "  ... {} more", self.diffs.len() - limit);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> Case {
        let mut case = Case {
            opcode: "v_add_f32".into(),
            variant: "e32".into(),
            text: "v_add_f32_e32 v10, v1, v3".into(),
            program: 0x10_0000,
            compute: (0..COMPUTE_COUNT as u32).collect(),
            groups: 2,
            inputs: 0x20_0000,
            outputs: 0x40_0000,
            in_words: 8,
            out_words: 8,
            out_names: vec!["dst".into(), "dst+1".into(), "mask".into()],
            ..Case::default()
        };
        let ins: Vec<u8> = (0..128 * 8u32).flat_map(u32::to_le_bytes).collect();
        write_pages(&mut case.input, case.inputs, &ins);
        let outs: Vec<u8> = (0..128 * 8u32).flat_map(u32::to_le_bytes).collect();
        write_pages(&mut case.output, case.outputs, &outs);
        case
    }

    #[test]
    fn cases_survive_a_round_trip() {
        let case = case();
        let mut bytes = Vec::new();
        case.write_to(&mut bytes).unwrap();
        assert_eq!(Case::read_from(bytes.as_slice()).unwrap(), case);
    }

    #[test]
    fn comparison_names_the_lanes_that_differ() {
        let case = case();
        let mut hw: Vec<u32> = (0..128 * 8u32).collect();
        assert!(case.compare(&hw).diffs.is_empty());
        hw[70 * 8 + 2] ^= 1;
        let out = case.compare(&hw);
        assert_eq!(out.diffs.len(), 1);
        assert_eq!(out.diffs[0].record, 70);
        assert_eq!(out.diffs[0].input[0], 70 * 8);
        assert!(out.report(4).contains("mask"));
    }

    #[test]
    fn commands_end_in_one_dispatch_of_the_groups() {
        let c = case().commands();
        assert_eq!(c.len() % 8, 0);
        let at = c.iter().position(|&w| w >> 8 & 0xFF == 0x15).unwrap();
        assert_eq!(&c[at + 1..at + 5], &[2, 1, 1, 1]);
    }
}
