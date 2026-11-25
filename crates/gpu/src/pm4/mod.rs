//! The PM4 type-3 packet walker.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::{Backend, Draw, Error, Flip, Host, State};

mod copy;
mod sync;
#[cfg(test)]
mod tests;
mod work;

use sync::gpu_clock;
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
const IT_REWIND: u32 = 0x59;
const IT_ATOMIC_MEM: u32 = 0x1E;
const IT_MEM_SEMAPHORE: u32 = 0x39;
const IT_COPY_DATA: u32 = 0x40;
const IT_COND_EXEC: u32 = 0x22;
/// REWIND's valid bit: the CPU has finished writing the buffer past the packet.
const REWIND_VALID: u32 = 1 << 31;
const CB_COLOR_CONTROL: u32 = 0x202;
/// The global data share's size on GFX10.
const GDS_BYTES: usize = 64 << 10;
/// Largest DMA_DATA transfer carried out, a guard against garbage.
const DMA_LIMIT: usize = 64 << 20;
/// DMA_DATA fills up to this size are queued like WRITE_DATA instead of
/// waiting for the GPU (AGC clears labels this way before waiting on them).
const DEFERRED_FILL: usize = 64;
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

/// Reports a packet form the processor drops, once per form and register.
fn unhandled(what: std::fmt::Arguments) {
    static SEEN: diag::Distinct = diag::Distinct::new(usize::MAX);
    let what = what.to_string();
    if SEEN.first(&what) {
        tracing::warn!("{what} dropped");
    }
}

/// Logs a submission, wait, release or memory write with its queue (the `gpu::sync` target).
fn trace(what: std::fmt::Arguments) {
    tracing::debug!(
        target: "gpu::sync",
        "{:.3} [{}]: {what}",
        crate::seconds(),
        std::thread::current().name().unwrap_or("?")
    );
}

/// A line in the sync trace from outside the command processors.
pub fn trace_sync_event(what: std::fmt::Arguments) {
    trace(what);
}

/// Destination selects that mean memory in WRITE_DATA.
const WRITE_DATA_MEMORY: [u32; 4] = [1, 2, 4, 5];
const MAX_DEPTH: u32 = 8;
/// How long a wait polls before giving up and moving on.
const WAIT_LIMIT: Duration = Duration::from_secs(5);

/// Dispatches with more thread groups than this are skipped: Metal would run
/// them for minutes, and a real title's are far smaller.
pub(crate) const MAX_GROUPS: u64 = 1 << 24;
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
    /// Dispatches skipped for their size ([`MAX_GROUPS`]).
    pub huge_dispatches: u64,
    /// Indirect dispatches whose dimensions the backend read on the GPU.
    pub gpu_indirect: u64,
    /// The longest time between two flips.
    pub longest_flip_gap: Duration,
    /// Times between two flips over a second, for the next report to print.
    pub long_flip_gaps: Vec<Duration>,
    /// Packets by opcode (and NOP code, as `0x100 | code`).
    pub opcodes: BTreeMap<u32, u64>,
    /// Waits for the GPU by the packet that needed them.
    pub gpu_waits: BTreeMap<&'static str, u64>,
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
    /// The display buffer the last flip showed.
    pub shown: Option<u64>,
    /// Releases and memory writes whose earlier GPU work is committed but
    /// not known to be done, in command order.
    pending: Vec<Pending>,
    /// Pending work of ended submissions, by the backend's commit point that
    /// must be done before it takes effect (see [`Processor::retire`]).
    retiring: std::collections::VecDeque<(u64, Vec<Pending>)>,
    watched: Option<u64>,
    /// A register table wrote `CB_COLOR_CONTROL` since the last draw.
    color_control_written: bool,
}

/// What a save state keeps of a [`Processor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Saved {
    pub context: Vec<u32>,
    pub sh: Vec<u32>,
    pub uconfig: Vec<u32>,
    pub index_base: u64,
    pub indirect_base: u64,
    pub instances: u32,
    pub gds: Vec<u8>,
}

savestate::record!(Saved {
    context,
    sh,
    uconfig,
    index_base,
    indirect_base,
    instances,
    gds
});

enum Pending {
    Release {
        data: u32,
        interrupt: u32,
        dest: u64,
        value: u64,
        context: u32,
    },
    Write {
        dest: u64,
        values: Vec<u32>,
        fixed: bool,
        /// The backend already wrote it in GPU order; kept for waits only.
        landed: bool,
    },
    /// The queue's interrupt at the end of a submission.
    Interrupt,
}

impl Pending {
    /// The 32-bit value this leaves at `address`, if it writes there.
    fn word_at(&self, address: u64) -> Option<u32> {
        match *self {
            Pending::Release {
                data: 1 | 2,
                dest,
                value,
                ..
            } if dest == address => Some(value as u32),
            Pending::Release {
                data: 2,
                dest,
                value,
                ..
            } if dest + 4 == address => Some((value >> 32) as u32),
            Pending::Write {
                dest,
                ref values,
                fixed,
                ..
            } => {
                if fixed {
                    values.last().copied().filter(|_| dest == address)
                } else {
                    let i = address.checked_sub(dest)?;
                    (i % 4 == 0)
                        .then(|| values.get((i / 4) as usize).copied())
                        .flatten()
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
            retiring: Default::default(),
            watched: None,
            last_flip: None,
            shown: None,
            color_control_written: false,
        }
    }
}

fn read_dwords(host: &dyn Host, addr: u64, n: usize) -> Result<Vec<u32>, Error> {
    let mut b = vec![0u8; n * 4];
    if !host.read(addr, &mut b) {
        return Err(Error::Unreadable(addr));
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

fn read_u64(host: &dyn Host, addr: u64) -> u64 {
    let mut b = [0u8; 8];
    host.read(addr, &mut b);
    u64::from_le_bytes(b)
}

fn addr(lo: u32, hi: u32) -> u64 {
    lo as u64 | (hi as u64) << 32
}

impl Processor {
    /// Runs the `dwords`-dword command buffer at `addr`.
    pub fn run(&mut self, host: &dyn Host, addr: u64, dwords: u32) -> Result<(), Error> {
        self.stats.submissions += 1;
        trace(format_args!("submission {addr:#x} {dwords} dwords"));
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
            host.commands_read(base, &words);
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
                if (header >> 8) & 0xFF == IT_INDIRECT_BUFFER
                    && payload.len() >= 3
                    && payload[2] & (1 << 20) != 0
                {
                    base = (payload[0] & !3) as u64 | ((payload[1] & 0xFFFF) as u64) << 32;
                    dwords = payload[2] & 0xF_FFFF;
                    continue 'chain;
                }
                if (header >> 8) & 0xFF == IT_REWIND && !payload.is_empty() {
                    *self.stats.opcodes.entry(IT_REWIND).or_default() += 1;
                    self.rewind(host, here + 4);
                    let next = (at + len) as u32;
                    base += 4 * next as u64;
                    dwords -= next;
                    continue 'chain;
                }
                if (header >> 8) & 0xFF == IT_COND_EXEC && payload.len() >= 4 {
                    *self.stats.opcodes.entry(IT_COND_EXEC).or_default() += 1;
                    self.wait_gpu(host, "COND_EXEC");
                    let flag =
                        read_dwords(host, addr(payload[0] & !3, payload[1]), 1).map_or(0, |w| w[0]);
                    at += len;
                    if flag == 0 {
                        at += (payload[3] & 0x3FFF) as usize;
                    }
                    continue;
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
        let key = if op == IT_NOP && code != 0 {
            0x100 | code
        } else if op == 0x46 && !p.is_empty() {
            0x4600 | (p[0] & 0x3F) | ((p.len() as u32 > 1) as u32) << 7
        } else {
            op
        };
        *self.stats.opcodes.entry(key).or_default() += 1;
        match op {
            IT_NOP => self.nop(host, code, p),
            IT_WRITE_DATA if p.len() >= 3 => {
                let control = p[0];
                if WRITE_DATA_MEMORY.contains(&((control >> 8) & 0xF)) {
                    self.write_values(host, addr(p[1], p[2]), &p[3..], control & (1 << 16) != 0);
                } else {
                    unhandled(format_args!(
                        "WRITE_DATA to select {} at {:#x}",
                        (control >> 8) & 0xF,
                        p[1]
                    ));
                }
            }
            IT_WAIT_REG_MEM if p.len() >= 5 && p[0] & 0x10 != 0 => {
                let address = (p[1] & !3) as u64 | ((p[2] & 0x3FFFF) as u64) << 32;
                self.wait(host, p[0], address, p[3] as u64, p[4] as u64, false);
            }
            IT_WAIT_REG_MEM_64 if p.len() >= 7 && p[0] & 0x10 != 0 => {
                let address = (p[1] & !7) as u64 | ((p[2] & 0x3FFFF) as u64) << 32;
                self.wait(
                    host,
                    p[0],
                    address,
                    addr(p[3], p[4]),
                    addr(p[5], p[6]),
                    true,
                );
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
            IT_COPY_DATA if p.len() >= 5 => self.copy_data(host, p),
            IT_ATOMIC_MEM if p.len() >= 8 => self.atomic_mem(host, p),
            IT_MEM_SEMAPHORE if p.len() >= 3 => self.mem_semaphore(host, p),
            IT_SET_CONTEXT_REG => Self::set_registers(&mut self.context, p),
            IT_SET_SH_REG => Self::set_registers(&mut self.sh, p),
            IT_SET_UCONFIG_REG | IT_SET_UCONFIG_REG_INDEX => {
                Self::set_registers(&mut self.uconfig, p)
            }
            IT_SET_CONTEXT_REG_INDIRECT | IT_SET_SH_REG_INDIRECT | IT_SET_UCONFIG_REG_INDIRECT
                if p.len() >= 4 =>
            {
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
                    if op == IT_SET_CONTEXT_REG_INDIRECT
                        && e[0] & !TABLE_SELECTOR == CB_COLOR_CONTROL
                    {
                        self.color_control_written = true;
                    }
                    if let Some(r) = bank.get_mut((e[0] & !TABLE_SELECTOR) as usize) {
                        *r = e[1];
                    }
                }
            }
            IT_INDEX_BASE if p.len() >= 2 => self.index_base = addr(p[0], p[1]),
            IT_NUM_INSTANCES if !p.is_empty() => self.instances = p[0].max(1),
            IT_DRAW_INDEX_AUTO if !p.is_empty() => self.draw(
                host,
                op,
                Draw::Auto {
                    count: p[0],
                    instances: self.instances,
                },
            ),
            IT_DRAW_INDEX_2 if p.len() >= 4 => {
                let draw = Draw::Indexed {
                    count: p[3],
                    address: addr(p[1], p[2]),
                    instances: self.instances,
                };
                self.draw(host, op, draw)
            }
            IT_DRAW_INDEX_OFFSET_2 if p.len() >= 3 => {
                let size = if self.uconfig[0x243] & 3 == 1 { 4 } else { 2 };
                let address = self.index_base + p[1] as u64 * size;
                let draw = Draw::Indexed {
                    count: p[2],
                    address,
                    instances: self.instances,
                };
                self.draw(host, op, draw)
            }
            IT_SET_BASE if p.len() >= 3 && p[0] & 0xF == 1 => self.indirect_base = addr(p[1], p[2]),
            IT_DMA_DATA if p.len() >= 6 => self.dma_data(host, p),
            IT_DRAW_INDEX_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 5);
                let size = if self.uconfig[0x243] & 3 == 1 { 4 } else { 2 };
                let address = self.index_base + a[2] as u64 * size;
                if a[1] != 0 {
                    self.draw(
                        host,
                        op,
                        Draw::Indexed {
                            count: a[0],
                            address,
                            instances: a[1],
                        },
                    )
                }
            }
            IT_DRAW_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 4);
                if a[1] != 0 {
                    self.draw(
                        host,
                        op,
                        Draw::Auto {
                            count: a[0],
                            instances: a[1],
                        },
                    )
                }
            }
            IT_DRAW_INDIRECT_MULTI | IT_DRAW_INDEX_INDIRECT_MULTI | IT_DRAW_INDEX_MULTI_AUTO => {
                self.stats.draws += 1
            }
            IT_DISPATCH_DIRECT if p.len() >= 3 => {
                let initiator = p.get(3).copied().unwrap_or(0);
                self.dispatch(host, [p[0], p[1], p[2]], initiator)
            }
            // The compute queues' form has a 64-bit address instead of an
            // offset from SET_BASE (Mesa's radv builds both).
            IT_DISPATCH_INDIRECT if p.len() >= 3 => {
                self.dispatch_indirect(host, addr(p[0], p[1]), p[2]);
            }
            IT_DISPATCH_INDIRECT if !p.is_empty() => {
                let at = self.indirect_base + p[0] as u64;
                self.dispatch_indirect(host, at, p.get(1).copied().unwrap_or(0));
            }
            _ => {}
        }
        Ok(())
    }

    fn with_gds(&mut self, f: &mut dyn FnMut(&mut [u8])) {
        if !self.backend.as_mut().is_some_and(|b| b.with_gds(f)) {
            f(&mut self.gds);
        }
    }

    /// SET_*_REG: a register offset (relative to the bank), then values.
    fn set_registers(bank: &mut [u32], p: &[u32]) {
        let Some((&first, values)) = p.split_first() else {
            return;
        };
        let start = (first & 0xFFFF) as usize;
        for (i, &v) in values.iter().enumerate() {
            if let Some(r) = bank.get_mut(start + i) {
                *r = v;
            }
        }
    }

    /// The register state and global data share, for a save state. Call
    /// [`Processor::finish`] first, so that no GPU work is in flight.
    pub fn saved(&mut self) -> Saved {
        let mut gds = Vec::new();
        self.with_gds(&mut |g| gds = g.to_vec());
        Saved {
            context: self.context.clone(),
            sh: self.sh.clone(),
            uconfig: self.uconfig.clone(),
            index_base: self.index_base,
            indirect_base: self.indirect_base,
            instances: self.instances,
            gds,
        }
    }

    /// Takes over a saved register state; the global data share too when
    /// `gds` (it is shared by every queue of a backend).
    pub fn restore(&mut self, s: Saved, gds: bool) {
        self.context = s.context;
        self.sh = s.sh;
        self.uconfig = s.uconfig;
        self.index_base = s.index_base;
        self.indirect_base = s.indirect_base;
        self.instances = s.instances;
        if gds {
            self.with_gds(&mut |g| {
                let n = g.len().min(s.gds.len());
                g[..n].copy_from_slice(&s.gds[..n]);
            });
        }
    }
}
