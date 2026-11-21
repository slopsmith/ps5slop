//! The PM4 type-3 packet walker.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::{Backend, Draw, Error, Flip, Host, State};

const IT_NOP: u32 = 0x10;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_DRAW_INDIRECT: u32 = 0x24;
const IT_DRAW_INDEX_INDIRECT: u32 = 0x25;
const IT_DRAW_INDEX_2: u32 = 0x27;
const IT_DRAW_INDIRECT_MULTI: u32 = 0x2C;
const IT_DRAW_INDEX_AUTO: u32 = 0x2D;
const IT_DRAW_INDEX_MULTI_AUTO: u32 = 0x30;
const IT_DRAW_INDEX_OFFSET_2: u32 = 0x35;
const IT_WRITE_DATA: u32 = 0x37;
const IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
const IT_WAIT_REG_MEM: u32 = 0x3C;
const IT_INDIRECT_BUFFER: u32 = 0x3F;
const IT_EVENT_WRITE_EOP: u32 = 0x47;
const IT_EVENT_WRITE_EOS: u32 = 0x48;
const IT_RELEASE_MEM: u32 = 0x49;
const IT_SET_CONTEXT_REG: u32 = 0x69;
const IT_SET_SH_REG: u32 = 0x76;
const IT_SET_UCONFIG_REG: u32 = 0x79;
const IT_SET_UCONFIG_REG_INDEX: u32 = 0x7A;
const IT_WAIT_REG_MEM_64: u32 = 0x93;
const IT_INDEX_BASE: u32 = 0x26;
const IT_NUM_INSTANCES: u32 = 0x2F;
const IT_SET_SH_REG_INDIRECT: u32 = 0x63;
const IT_SET_UCONFIG_REG_INDIRECT: u32 = 0x64;
const IT_SET_CONTEXT_REG_INDIRECT: u32 = 0x9F;
const IT_SET_BASE: u32 = 0x11;
const IT_DMA_DATA: u32 = 0x50;
/// The global data share's size on GFX10.
const GDS_BYTES: usize = 64 << 10;
/// Largest DMA_DATA transfer carried out, a guard against garbage.
const DMA_LIMIT: usize = 64 << 20;
/// Bits of a register-table offset that select something other than the register.
const TABLE_SELECTOR: u32 = 0x7000_0000;

/// Codes in a NOP's shader-type/index field that AGC's builders use for
/// their own packets (SharpEmu's names).
const NOP_WAIT_FLIP_DONE: u32 = 0x06;
const NOP_WAIT_MEMORY_32: u32 = 0x0A;
const NOP_WRITE_DATA: u32 = 0x15;
const NOP_WAIT_MEMORY_64: u32 = 0x16;
const NOP_FLIP: u32 = 0x17;
const NOP_RELEASE_MEM: u32 = 0x18;

/// A plain NOP whose first payload dword starts with this is a marker.
const MARKER_MAGIC: u32 = 0x6875_0000;
const MARKER_FLIP: u32 = 0x777;
const MARKER_FLIP_WITH_LABEL: u32 = 0x778;
const MARKER_FLIP_WITH_INTERRUPT: u32 = 0x781;

/// Whether releases and memory writes wait for the GPU instead of being
/// queued (`PS5SLOP_GPU_SYNC_RELEASES`).
fn synchronous_releases() -> bool {
    static SYNCHRONOUS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SYNCHRONOUS.get_or_init(|| std::env::var_os("PS5SLOP_GPU_SYNC_RELEASES").is_some())
}

/// Destination selects that mean memory in WRITE_DATA.
const WRITE_DATA_MEMORY: [u32; 4] = [1, 2, 4, 5];
const MAX_DEPTH: u32 = 8;
/// How long a wait polls before giving up and moving on.
const WAIT_LIMIT: Duration = Duration::from_secs(5);
/// Flips wait for the display's next refresh, at 60 Hz.
const VBLANK: Duration = Duration::from_micros(16_667);

/// What the processor has seen, for logs.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub submissions: u64,
    pub packets: u64,
    pub draws: u64,
    pub dispatches: u64,
    pub flips: u64,
    pub interrupts: u64,
    /// Waits that gave up after [`WAIT_LIMIT`].
    pub timeouts: u64,
    /// Waits a pending release satisfied without waiting for the GPU.
    pub deferred_waits: u64,
    /// Packets by opcode (and NOP code, as `0x100 | code`).
    pub opcodes: BTreeMap<u32, u64>,
}

/// One GPU queue's command processor and its register state.
pub struct Processor {
    pub context: Vec<u32>,
    pub sh: Vec<u32>,
    pub uconfig: Vec<u32>,
    pub stats: Stats,
    pub backend: Option<Box<dyn Backend>>,
    index_base: u64,
    /// `SET_BASE` index 1: where indirect draw and dispatch arguments are.
    indirect_base: u64,
    instances: u32,
    /// The global data share when the backend does not keep one.
    gds: Vec<u8>,
    last_flip: Option<Instant>,
    /// Releases and memory writes whose earlier GPU work is committed but
    /// not known to be done, in command order.
    pending: Vec<Pending>,
}

enum Pending {
    Release { data: u32, interrupt: u32, dest: u64, value: u64, context: u32 },
    Write { dest: u64, values: Vec<u32>, fixed: bool },
}

impl Pending {
    /// The 32-bit value this leaves at `address`, if it writes there.
    fn word_at(&self, address: u64) -> Option<u32> {
        match *self {
            Pending::Release { data: 1 | 2, dest, value, .. } if dest == address => Some(value as u32),
            Pending::Release { data: 2, dest, value, .. } if dest + 4 == address => Some((value >> 32) as u32),
            Pending::Write { dest, ref values, fixed } => {
                if fixed {
                    values.last().copied().filter(|_| dest == address)
                } else {
                    let i = address.checked_sub(dest)?;
                    (i % 4 == 0).then(|| values.get((i / 4) as usize).copied()).flatten()
                }
            }
            _ => None,
        }
    }
}

impl Default for Processor {
    fn default() -> Self {
        Self {
            context: vec![0; 0x400],
            sh: vec![0; 0x400],
            uconfig: vec![0; 0x4000],
            stats: Stats::default(),
            backend: None,
            index_base: 0,
            indirect_base: 0,
            instances: 1,
            gds: vec![0; GDS_BYTES],
            pending: Vec::new(),
            last_flip: None,
        }
    }
}

fn read_dwords(host: &dyn Host, addr: u64, n: usize) -> Result<Vec<u32>, Error> {
    let mut b = vec![0u8; n * 4];
    if !host.read(addr, &mut b) {
        return Err(Error::Unreadable(addr));
    }
    Ok(b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

fn read_u64(host: &dyn Host, addr: u64) -> u64 {
    let mut b = [0u8; 8];
    host.read(addr, &mut b);
    u64::from_le_bytes(b)
}

fn addr(lo: u32, hi: u32) -> u64 {
    lo as u64 | (hi as u64) << 32
}

/// WAIT_REG_MEM's comparison `function` of the masked value with `reference`.
fn compare(function: u32, value: u64, reference: u64) -> bool {
    match function & 7 {
        0 => true,
        1 => value < reference,
        2 => value <= reference,
        3 => value == reference,
        4 => value != reference,
        5 => value >= reference,
        6 => value > reference,
        _ => false,
    }
}

impl Processor {
    /// Runs the `dwords`-dword command buffer at `addr`.
    pub fn run(&mut self, host: &dyn Host, addr: u64, dwords: u32) -> Result<(), Error> {
        self.stats.submissions += 1;
        self.walk(host, addr, dwords, 0)
    }

    /// Runs a command buffer. A chained INDIRECT_BUFFER (CHAIN, bit 20 of its
    /// size dword) ends the buffer and continues at its target, at the same depth.
    fn walk(&mut self, host: &dyn Host, base: u64, dwords: u32, depth: u32) -> Result<(), Error> {
        if depth > MAX_DEPTH {
            return Err(Error::TooDeep(base));
        }
        let (mut base, mut dwords) = (base, dwords);
        'chain: loop {
            let words = read_dwords(host, base, dwords as usize)?;
            let mut at = 0usize;
            while at < words.len() {
                let header = words[at];
                let here = base + 4 * at as u64;
                if header >> 30 != 3 {
                    at += 1;
                    continue;
                }
                let len = (((header >> 16) & 0x3FFF) + 2) as usize;
                if at + len > words.len() {
                    return Err(Error::Truncated { addr: here, header });
                }
                let payload = &words[at + 1..at + len];
                if (header >> 8) & 0xFF == IT_INDIRECT_BUFFER && payload.len() >= 3 && payload[2] & (1 << 20) != 0 {
                    base = (payload[0] & !3) as u64 | ((payload[1] & 0xFFFF) as u64) << 32;
                    dwords = payload[2] & 0xF_FFFF;
                    continue 'chain;
                }
                self.packet(host, header, payload, depth)?;
                at += len;
            }
            return Ok(());
        }
    }

    fn packet(&mut self, host: &dyn Host, header: u32, p: &[u32], depth: u32) -> Result<(), Error> {
        let op = (header >> 8) & 0xFF;
        let code = (header >> 2) & 0x3F;
        self.stats.packets += 1;
        let key = if op == IT_NOP && code != 0 { 0x100 | code } else { op };
        *self.stats.opcodes.entry(key).or_default() += 1;
        match op {
            IT_NOP => self.nop(host, code, p),
            IT_WRITE_DATA if p.len() >= 3 => {
                let control = p[0];
                if WRITE_DATA_MEMORY.contains(&((control >> 8) & 0xF)) {
                    self.write_values(host, addr(p[1], p[2]), &p[3..], control & (1 << 16) != 0);
                }
            }
            IT_WAIT_REG_MEM if p.len() >= 5 && p[0] & 0x10 != 0 => {
                let address = (p[1] & !3) as u64 | ((p[2] & 0x3FFFF) as u64) << 32;
                self.wait(host, p[0], address, p[3] as u64, p[4] as u64, false);
            }
            IT_WAIT_REG_MEM_64 if p.len() >= 7 && p[0] & 0x10 != 0 => {
                let address = (p[1] & !7) as u64 | ((p[2] & 0x3FFFF) as u64) << 32;
                self.wait(host, p[0], address, addr(p[3], p[4]), addr(p[5], p[6]), true);
            }
            IT_INDIRECT_BUFFER if p.len() >= 3 => {
                let target = (p[0] & !3) as u64 | ((p[1] & 0xFFFF) as u64) << 32;
                self.walk(host, target, p[2] & 0xF_FFFF, depth + 1)?;
            }
            IT_EVENT_WRITE_EOP if p.len() >= 5 => {
                let data = (p[2] >> 29) & 7;
                let interrupt = (p[2] >> 24) & 7;
                let dest = p[1] as u64 | ((p[2] & 0xFFFF) as u64) << 32;
                self.release(host, data, interrupt, dest, addr(p[3], p[4]), 0);
            }
            IT_EVENT_WRITE_EOS if p.len() >= 4 => {
                let dest = p[1] as u64 | ((p[2] & 0xFFFF) as u64) << 32;
                self.release(host, 1, 0, dest, p[3] as u64, 0);
            }
            IT_RELEASE_MEM if p.len() >= 7 => {
                let control = p[1];
                self.release(
                    host,
                    (control >> 29) & 7,
                    (control >> 24) & 7,
                    addr(p[2], p[3]),
                    addr(p[4], p[5]),
                    p[6] & 0x07FF_FFFF,
                );
            }
            IT_SET_CONTEXT_REG => Self::set_registers(&mut self.context, p),
            IT_SET_SH_REG => Self::set_registers(&mut self.sh, p),
            IT_SET_UCONFIG_REG | IT_SET_UCONFIG_REG_INDEX => Self::set_registers(&mut self.uconfig, p),
            IT_SET_CONTEXT_REG_INDIRECT | IT_SET_SH_REG_INDIRECT | IT_SET_UCONFIG_REG_INDIRECT if p.len() >= 4 => {
                let table = (p[0] & !3) as u64 | (p[1] as u64) << 32;
                let entries = read_dwords(host, table, 2 * (p[3] & 0x3FFF) as usize)?;
                let bank = match op {
                    IT_SET_CONTEXT_REG_INDIRECT => &mut self.context,
                    IT_SET_SH_REG_INDIRECT => &mut self.sh,
                    _ => &mut self.uconfig,
                };
                for e in entries.as_chunks::<2>().0 {
                    if e[0] == u32::MAX {
                        continue;
                    }
                    if let Some(r) = bank.get_mut((e[0] & !TABLE_SELECTOR) as usize) {
                        *r = e[1];
                    }
                }
            }
            IT_INDEX_BASE if p.len() >= 2 => self.index_base = addr(p[0], p[1]),
            IT_NUM_INSTANCES if !p.is_empty() => self.instances = p[0].max(1),
            IT_DRAW_INDEX_AUTO if !p.is_empty() => {
                self.draw(host, op, p, Draw::Auto { count: p[0], instances: self.instances })
            }
            IT_DRAW_INDEX_2 if p.len() >= 4 => {
                let draw = Draw::Indexed { count: p[3], address: addr(p[1], p[2]), instances: self.instances };
                self.draw(host, op, p, draw)
            }
            IT_DRAW_INDEX_OFFSET_2 if p.len() >= 3 => {
                let size = if self.uconfig[0x243] & 3 == 1 { 4 } else { 2 };
                let address = self.index_base + p[1] as u64 * size;
                let draw = Draw::Indexed { count: p[2], address, instances: self.instances };
                self.draw(host, op, p, draw)
            }
            IT_SET_BASE if p.len() >= 3 && p[0] & 0xF == 1 => self.indirect_base = addr(p[1], p[2]),
            IT_DMA_DATA if p.len() >= 6 => self.dma_data(host, p),
            IT_DRAW_INDEX_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 5);
                let size = if self.uconfig[0x243] & 3 == 1 { 4 } else { 2 };
                let address = self.index_base + a[2] as u64 * size;
                self.draw(host, op, p, Draw::Indexed { count: a[0], address, instances: a[1].max(1) })
            }
            IT_DRAW_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 4);
                self.draw(host, op, p, Draw::Auto { count: a[0], instances: a[1].max(1) })
            }
            IT_DRAW_INDIRECT_MULTI | IT_DRAW_INDEX_INDIRECT_MULTI | IT_DRAW_INDEX_MULTI_AUTO => self.stats.draws += 1,
            IT_DISPATCH_DIRECT if p.len() >= 3 => self.dispatch(host, [p[0], p[1], p[2]]),
            IT_DISPATCH_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 3);
                self.dispatch(host, [a[0], a[1], a[2]]);
            }
            _ => {}
        }
        Ok(())
    }

    /// `n` argument dwords of an indirect draw or dispatch at `offset` past
    /// the indirect base, read once earlier GPU work (which may write them) is done.
    fn indirect_args(&mut self, host: &dyn Host, offset: u32, n: usize) -> Vec<u32> {
        self.drain(host);
        self.sync();
        read_dwords(host, self.indirect_base + offset as u64, n).unwrap_or_else(|_| vec![0; n])
    }

    /// `DMA_DATA`: copies or fills between memory, the GDS and the packet's
    /// own data dword. Done on the CPU in command order, after the GPU work
    /// before it (which may write the source) has finished.
    fn dma_data(&mut self, host: &dyn Host, p: &[u32]) {
        let control = p[0];
        let src_sel = (control >> 29) & 3;
        let dst_sel = (control >> 20) & 3;
        let command = p[5];
        let bytes = (command & 0x3FF_FFFF) as usize;
        let (src_register, dst_register) = (command >> 26 & 1 != 0, command >> 27 & 1 != 0);
        let (src_fixed, dst_fixed) = (command >> 28 & 1 != 0, command >> 29 & 1 != 0);
        if bytes == 0 || bytes > DMA_LIMIT || dst_sel == 2 || src_register || dst_register {
            return;
        }
        self.drain(host);
        self.sync();
        let src = addr(p[1], p[2]);
        let dst = addr(p[3], p[4]);
        let mut data = vec![0u8; bytes];
        match src_sel {
            2 => {
                for c in data.chunks_mut(4) {
                    c.copy_from_slice(&p[1].to_le_bytes()[..c.len()]);
                }
            }
            1 => {
                let at = src as usize;
                self.with_gds(&mut |gds| {
                    for (i, b) in data.iter_mut().enumerate() {
                        let a = if src_fixed { at + i % 4 } else { at + i };
                        *b = gds.get(a).copied().unwrap_or(0);
                    }
                });
            }
            _ if src_fixed => {
                let mut word = [0u8; 4];
                host.read(src, &mut word);
                for c in data.chunks_mut(4) {
                    c.copy_from_slice(&word[..c.len()]);
                }
            }
            _ => {
                host.read(src, &mut data);
            }
        }
        if dst_fixed {
            let n = data.len().min(4);
            data.drain(..data.len() - n);
        }
        if dst_sel == 1 {
            self.with_gds(&mut |gds| {
                let start = (dst as usize).min(gds.len());
                let end = (start + data.len()).min(gds.len());
                gds[start..end].copy_from_slice(&data[..end - start]);
            });
        } else {
            host.write(dst, &data);
        }
    }

    fn with_gds(&mut self, f: &mut dyn FnMut(&mut [u8])) {
        if !self.backend.as_mut().is_some_and(|b| b.with_gds(f)) {
            f(&mut self.gds);
        }
    }

    fn dispatch(&mut self, host: &dyn Host, groups: [u32; 3]) {
        self.stats.dispatches += 1;
        if let Some(b) = self.backend.as_mut() {
            let state = State { context: &self.context, sh: &self.sh, uconfig: &self.uconfig };
            let t = crate::timing::start();
            b.dispatch(host, &state, groups);
            crate::timing::add(crate::timing::Part::Dispatch, t);
        }
    }

    fn draw(&mut self, host: &dyn Host, op: u32, p: &[u32], draw: Draw) {
        self.stats.draws += 1;
        self.dump_draw(op, p);
        if let Some(b) = self.backend.as_mut() {
            let state = State { context: &self.context, sh: &self.sh, uconfig: &self.uconfig };
            let t = crate::timing::start();
            b.draw(host, &state, draw);
            crate::timing::add(crate::timing::Part::Draw, t);
        }
    }

    fn sync(&mut self) {
        if let Some(b) = self.backend.as_mut() {
            let t = crate::timing::start();
            b.sync();
            crate::timing::add(crate::timing::Part::Sync, t);
        }
    }

    fn dump_draw(&self, op: u32, p: &[u32]) {
        static LEFT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
        use std::sync::atomic::Ordering::Relaxed;
        if LEFT.load(Relaxed) == i64::MIN {
            let n = std::env::var("PS5SLOP_GPU_DUMP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            LEFT.store(n, Relaxed);
        }
        static SKIP: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
        if SKIP.load(Relaxed) == i64::MIN {
            let n = std::env::var("PS5SLOP_GPU_DUMP_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            SKIP.store(n, Relaxed);
        }
        if SKIP.load(Relaxed) > 0 {
            SKIP.fetch_sub(1, Relaxed);
            return;
        }
        if LEFT.fetch_sub(1, Relaxed) <= 0 {
            return;
        }
        let c = &self.context;
        let sh = &self.sh;
        let u = &self.uconfig;
        eprintln!("DRAW op={op:#x} payload={:x?}", p);
        for (name, lo) in [("PS", 0x08usize), ("VS", 0x48), ("GS", 0x88), ("ES", 0xC8), ("HS", 0x108), ("LS", 0x148)] {
            if sh[lo] != 0 || sh[lo + 1] != 0 {
                eprintln!("  {name} pgm={:#x} rsrc1={:#x} rsrc2={:#x} user={:x?}", (sh[lo] as u64 | (sh[lo + 1] as u64) << 32) << 8, sh[lo + 2], sh[lo + 3], &sh[lo + 4..lo + 20]);
            }
        }
        eprintln!("  stages={:#x} prim={:#x} idx={:#x} cb0 base={:#x} ext={:#x} view={:#x} info={:#x} attrib={:#x} attrib2={:#x} attrib3={:#x} target_mask={:#x} color_ctl={:#x} blend0={:#x}",
            c[0x2D5], u[0x242], u[0x243], c[0x318], c[0x390], c[0x31B], c[0x31C], c[0x31D], c[0x3B0], c[0x3B8], c[0x8E], c[0x202], c[0x1E0]);
        eprintln!("  vport scale/off={:?} scissor={:#x}/{:#x} vte={:#x} spi_ps_in={:#x}/{:#x} col_fmt={:#x} z_info={:#x} depth_ctl={:#x} ps_input_cntl0={:#x} vs_out={:#x} pos_fmt={:#x}",
            c[0x10F..0x115].iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), c[0x90], c[0x91], c[0x206], c[0x1B3], c[0x1B4], c[0x1C5], c[0x10], c[0x200], c[0x191], c[0x1B1], c[0x1C3]);
    }

    /// SET_*_REG: a register offset (relative to the bank), then values.
    fn set_registers(bank: &mut [u32], p: &[u32]) {
        let Some((&first, values)) = p.split_first() else { return };
        let start = (first & 0xFFFF) as usize;
        for (i, &v) in values.iter().enumerate() {
            if let Some(r) = bank.get_mut(start + i) {
                *r = v;
            }
        }
    }

    fn nop(&mut self, host: &dyn Host, code: u32, p: &[u32]) {
        match code {
            0 if p.first().is_some_and(|w| w & 0xFFFF_0000 == MARKER_MAGIC) => self.marker(host, p),
            NOP_FLIP if p.len() >= 5 => self.flip(
                host,
                Flip {
                    handle: p[0] as i32,
                    index: p[1] as i32,
                    mode: p[2] as i32,
                    arg: addr(p[3], p[4]) as i64,
                },
            ),
            NOP_WRITE_DATA if p.len() >= 3 => {
                let control = p[0];
                if WRITE_DATA_MEMORY.contains(&(control & 0xFF)) {
                    let fixed = (control >> 16) & 0xFF != 0;
                    self.write_values(host, addr(p[1], p[2]), &p[3..], fixed);
                }
            }
            NOP_RELEASE_MEM if p.len() >= 7 => {
                let control = p[1];
                let interrupt = (control >> 24) & 0xFF;
                let data = (control >> 16) & 0xFF;
                self.release(host, data, interrupt, addr(p[2], p[3]), addr(p[4], p[5]), p[6] & 0x07FF_FFFF);
            }
            NOP_WAIT_MEMORY_32 if p.len() >= 5 => {
                let (mask, reference, control) = if p.len() == 5 {
                    (p[2], p[4], p[3])
                } else {
                    (p[2], p[3], p[4])
                };
                self.wait(host, control, addr(p[0], p[1]), reference as u64, mask as u64, false);
            }
            NOP_WAIT_MEMORY_64 if p.len() >= 7 => {
                self.wait(host, p[6], addr(p[0], p[1]), addr(p[4], p[5]), addr(p[2], p[3]), true);
            }
            NOP_WAIT_FLIP_DONE => {}
            _ => {}
        }
    }

    fn marker(&mut self, host: &dyn Host, p: &[u32]) {
        let flip = Flip {
            handle: 0,
            index: -1,
            mode: 0,
            arg: 0,
        };
        match p[0] & 0xFFF {
            MARKER_FLIP => self.flip(host, flip),
            MARKER_FLIP_WITH_LABEL if p.len() >= 4 => {
                self.flip(host, flip);
                host.write(addr(p[1], p[2]), &p[3].to_le_bytes());
            }
            MARKER_FLIP_WITH_INTERRUPT if p.len() >= 6 => {
                self.flip(host, flip);
                host.write(addr(p[1], p[2]), &p[3].to_le_bytes());
                self.stats.interrupts += 1;
                host.interrupt(p[5]);
            }
            _ => {}
        }
    }

    fn flip(&mut self, host: &dyn Host, flip: Flip) {
        self.drain(host);
        if let Some(b) = self.backend.as_mut() {
            b.frame_done(host.display_buffer(flip.index));
        }
        if let Some(last) = self.last_flip
            && let Some(wait) = (last + VBLANK).checked_duration_since(Instant::now())
        {
            let t = crate::timing::start();
            std::thread::sleep(wait);
            crate::timing::add(crate::timing::Part::Pacing, t);
        }
        self.last_flip = Some(Instant::now());
        self.stats.flips += 1;
        host.flip(flip);
    }

    fn write_values(&mut self, host: &dyn Host, dest: u64, values: &[u32], fixed: bool) {
        if self.backend.is_some() && !synchronous_releases() {
            self.pending.push(Pending::Write { dest, values: values.to_vec(), fixed });
            return;
        }
        Self::apply_write(host, dest, values, fixed);
    }

    fn apply_write(host: &dyn Host, dest: u64, values: &[u32], fixed: bool) {
        for (i, v) in values.iter().enumerate() {
            let at = if fixed { dest } else { dest + 4 * i as u64 };
            host.write(at, &v.to_le_bytes());
        }
    }

    /// An end-of-pipe release: data select 1 writes 32 bits, 2 writes 64,
    /// 3 and 4 a clock; interrupt selects 1 and 2 raise an interrupt.
    // Selectors 5 and 6 write nothing and interrupt only while the label at
    // `dest` is at or below `value` (SharpEmu-mac's `InterruptDecision`).
    fn release(&mut self, host: &dyn Host, data: u32, interrupt: u32, dest: u64, value: u64, context: u32) {
        if self.backend.is_some() && !synchronous_releases() {
            self.pending.push(Pending::Release { data, interrupt, dest, value, context });
        } else {
            self.sync();
            self.apply_release(host, data, interrupt, dest, value, context);
        }
    }

    /// Waits for the GPU work behind pending releases, then makes them take
    /// effect in order. Called before anything that could observe them.
    pub fn drain(&mut self, host: &dyn Host) {
        if self.pending.is_empty() {
            return;
        }
        self.sync();
        for p in std::mem::take(&mut self.pending) {
            match p {
                Pending::Release { data, interrupt, dest, value, context } => {
                    self.apply_release(host, data, interrupt, dest, value, context)
                }
                Pending::Write { dest, values, fixed } => Self::apply_write(host, dest, &values, fixed),
            }
        }
    }

    fn apply_release(&mut self, host: &dyn Host, data: u32, interrupt: u32, dest: u64, value: u64, context: u32) {
        let raise = match interrupt {
            1 | 2 | 4 => true,
            5 => dest != 0 && (read_u64(host, dest) as u32) <= value as u32,
            6 => dest != 0 && read_u64(host, dest) <= value,
            _ => false,
        };
        if interrupt <= 3 {
            match data {
                1 => {
                    host.write(dest, &(value as u32).to_le_bytes());
                }
                2 => {
                    host.write(dest, &value.to_le_bytes());
                }
                3 | 4 => {
                    let clock = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_nanos() as u64);
                    host.write(dest, &clock.to_le_bytes());
                }
                _ => {}
            }
        }
        if raise {
            self.stats.interrupts += 1;
            host.interrupt(context);
        }
    }

    /// Polls guest memory until the masked value passes the comparison in
    /// `control`, the process exits, or [`WAIT_LIMIT`] passes.
    fn wait(&mut self, host: &dyn Host, control: u32, address: u64, reference: u64, mask: u64, wide: bool) {
        // GPU work runs in order, so a wait on a label that a pending release
        // will satisfy passes without waiting for the GPU.
        let word = |at: u64| self.pending.iter().rev().find_map(|p| p.word_at(at));
        let value = match (word(address), wide) {
            (Some(lo), false) => Some(lo as u64),
            (Some(lo), true) => word(address + 4).map(|hi| lo as u64 | (hi as u64) << 32),
            (None, _) => None,
        };
        let satisfied = value.is_some_and(|v| compare(control, v & mask, reference));
        if satisfied {
            self.stats.deferred_waits += 1;
            return;
        }
        self.drain(host);
        let start = Instant::now();
        loop {
            let value = if wide {
                read_u64(host, address)
            } else {
                let mut b = [0u8; 4];
                host.read(address, &mut b);
                u32::from_le_bytes(b) as u64
            };
            if compare(control, value & mask, reference) || host.exiting() {
                return;
            }
            if start.elapsed() > WAIT_LIMIT {
                self.stats.timeouts += 1;
                if self.stats.timeouts <= 16 {
                    eprintln!(
                        "ps5slop: GPU: wait timed out: {address:#x} & {mask:#x} = {value:#x}, function {}, reference {reference:#x}",
                        control & 7
                    );
                }
                return;
            }
            let t = crate::timing::start();
            std::thread::sleep(Duration::from_micros(200));
            crate::timing::add(crate::timing::Part::WaitMemory, t);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    #[derive(Default)]
    struct Fake {
        mem: RefCell<Vec<u8>>,
        interrupts: RefCell<Vec<u32>>,
        flips: RefCell<Vec<Flip>>,
    }

    impl Host for Fake {
        fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
            let mem = self.mem.borrow();
            let Some(src) = mem.get(addr as usize..addr as usize + buf.len()) else { return false };
            buf.copy_from_slice(src);
            true
        }
        fn write(&self, addr: u64, data: &[u8]) -> bool {
            let mut mem = self.mem.borrow_mut();
            let Some(dst) = mem.get_mut(addr as usize..addr as usize + data.len()) else { return false };
            dst.copy_from_slice(data);
            true
        }
        fn interrupt(&self, context: u32) {
            self.interrupts.borrow_mut().push(context);
        }
        fn flip(&self, flip: Flip) {
            self.flips.borrow_mut().push(flip);
        }
        fn exiting(&self) -> bool {
            false
        }
    }

    fn header(len: u32, op: u32, code: u32) -> u32 {
        0xC000_0000 | ((len - 2) << 16) | (op << 8) | (code << 2)
    }

    fn run(packets: &[u32]) -> (Fake, Processor) {
        let host = Fake::default();
        host.mem.replace(vec![0; 0x1000]);
        let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
        host.write(0x100, &bytes);
        let mut p = Processor::default();
        p.run(&host, 0x100, packets.len() as u32).unwrap();
        (host, p)
    }

    fn word(host: &Fake, at: usize) -> u32 {
        u32::from_le_bytes(host.mem.borrow()[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn dma_data_fills_the_gds_and_copies_it_to_memory() {
        let (host, _) = run(&[
            // Data 0x2A into GDS bytes 8..12.
            header(7, IT_DMA_DATA, 0),
            2 << 29 | 1 << 20,
            0x2A,
            0,
            8,
            0,
            4,
            // GDS bytes 8..12 to memory at 0x900.
            header(7, IT_DMA_DATA, 0),
            1 << 29 | 3 << 20,
            8,
            0,
            0x900,
            0,
            4,
        ]);
        assert_eq!(word(&host, 0x900), 0x2A);
    }

    #[test]
    fn indirect_dispatches_read_their_group_counts_past_the_base() {
        let host = Fake::default();
        host.mem.replace(vec![0; 0x1000]);
        host.write(0xA08, &[2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0]);
        let packets = [header(4, IT_SET_BASE, 0), 1, 0xA00, 0, header(3, IT_DISPATCH_INDIRECT, 0), 8, 0];
        let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
        host.write(0x100, &bytes);
        struct Groups(std::sync::Arc<std::sync::Mutex<Vec<[u32; 3]>>>);
        impl Backend for Groups {
            fn draw(&mut self, _: &dyn Host, _: &State, _: Draw) {}
            fn dispatch(&mut self, _: &dyn Host, _: &State, groups: [u32; 3]) {
                self.0.lock().unwrap().push(groups);
            }
            fn sync(&mut self) {}
            fn frame_done(&mut self, _: Option<u64>) {}
        }
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut p = Processor { backend: Some(Box::new(Groups(seen.clone()))), ..Processor::default() };
        p.run(&host, 0x100, packets.len() as u32).unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![[2, 3, 1]]);
    }

    #[test]
    fn queued_writes_and_releases_report_what_they_leave_in_memory() {
        let w = Pending::Write { dest: 0x100, values: vec![1, 2, 3], fixed: false };
        assert_eq!((w.word_at(0x104), w.word_at(0x10c), w.word_at(0x102)), (Some(2), None, None));
        let f = Pending::Write { dest: 0x100, values: vec![1, 2, 3], fixed: true };
        assert_eq!((f.word_at(0x100), f.word_at(0x104)), (Some(3), None));
        let r = Pending::Release { data: 2, interrupt: 0, dest: 0x200, value: 0x5_0000_0007, context: 0 };
        assert_eq!((r.word_at(0x200), r.word_at(0x204)), (Some(7), Some(5)));
    }

    #[test]
    fn write_data_and_release_mem_reach_memory_and_interrupts() {
        let (host, p) = run(&[
            header(5, IT_WRITE_DATA, 0),
            5 << 8,
            0x800,
            0,
            0xAABB_CCDD,
            header(8, IT_RELEASE_MEM, 0),
            0,
            (2 << 29) | (2 << 24),
            0x810,
            0,
            0x1122_3344,
            0x5566_7788,
            7,
        ]);
        assert_eq!(word(&host, 0x800), 0xAABB_CCDD);
        assert_eq!(word(&host, 0x810), 0x1122_3344);
        assert_eq!(word(&host, 0x814), 0x5566_7788);
        assert_eq!(*host.interrupts.borrow(), [7]);
        assert_eq!(p.stats.packets, 2);
    }

    #[test]
    fn registers_are_recorded_and_indirect_buffers_followed() {
        let host = Fake::default();
        host.mem.replace(vec![0; 0x1000]);
        let inner = [header(4, IT_SET_SH_REG, 0), 0x8, 0x1234, 0x5678];
        host.write(0x400, &inner.iter().flat_map(|d| d.to_le_bytes()).collect::<Vec<u8>>());
        let outer = [header(4, IT_INDIRECT_BUFFER, 0), 0x400, 0, 4, header(2, IT_DRAW_INDEX_AUTO, 0), 3];
        host.write(0x100, &outer.iter().flat_map(|d| d.to_le_bytes()).collect::<Vec<u8>>());
        let mut p = Processor::default();
        p.run(&host, 0x100, outer.len() as u32).unwrap();
        assert_eq!((p.sh[8], p.sh[9]), (0x1234, 0x5678));
        assert_eq!(p.stats.draws, 1);
    }

    #[test]
    fn chained_indirect_buffers_continue_without_nesting() {
        let host = Fake::default();
        host.mem.replace(vec![0; 0x2000]);
        let segments = 2 * MAX_DEPTH as u64;
        for k in 0..segments {
            let at = 0x100 + 0x100 * k;
            let mut seg = vec![header(2, IT_DRAW_INDEX_AUTO, 0), 3];
            if k + 1 < segments {
                seg.extend([header(4, IT_INDIRECT_BUFFER, 0), (at + 0x100) as u32, 0, 7 | 1 << 20]);
                seg.extend([header(2, IT_DRAW_INDEX_AUTO, 0), 3]);
            }
            host.write(at, &seg.iter().flat_map(|d| d.to_le_bytes()).collect::<Vec<u8>>());
        }
        let mut p = Processor::default();
        p.run(&host, 0x100, 8).unwrap();
        assert_eq!(p.stats.draws, segments);
    }

    #[test]
    fn waits_end_when_memory_matches_and_flips_are_passed_on() {
        let (host, p) = run(&[
            header(7, IT_WAIT_REG_MEM, 0),
            0x10 | 3,
            0x900,
            0,
            0,
            0xFFFF_FFFF,
            0,
            header(6, IT_NOP, NOP_FLIP),
            1,
            2,
            0,
            0x42,
            0,
        ]);
        assert_eq!(p.stats.timeouts, 0);
        assert_eq!(
            *host.flips.borrow(),
            [Flip {
                handle: 1,
                index: 2,
                mode: 0,
                arg: 0x42
            }]
        );
    }
}
