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

/// Whether releases and memory writes wait for the GPU instead of being
/// queued (`PS5SLOP_GPU_SYNC_RELEASES`).
fn synchronous_releases() -> bool {
    static SYNCHRONOUS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SYNCHRONOUS.get_or_init(|| std::env::var_os("PS5SLOP_GPU_SYNC_RELEASES").is_some())
}

/// Whether each submission waits for its GPU work before its releases take
/// effect, as before 2026-11-21 (`PS5SLOP_GPU_WAIT_SUBMISSIONS`).
fn wait_submissions() -> bool {
    static WAIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WAIT.get_or_init(|| std::env::var_os("PS5SLOP_GPU_WAIT_SUBMISSIONS").is_some())
}

/// Whether waits, releases and memory writes are logged with their queue
/// (`PS5SLOP_GPU_TRACE_SYNC`).
fn trace_sync() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("PS5SLOP_GPU_TRACE_SYNC").is_some())
}

/// Whether every dispatch is waited for (`PS5SLOP_GPU_SYNC_DISPATCHES`), so a
/// program that never ends is the last one traced.
fn sync_dispatches() -> bool {
    static SYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SYNC.get_or_init(|| std::env::var_os("PS5SLOP_GPU_SYNC_DISPATCHES").is_some())
}

/// `PS5SLOP_GPU_PROBE=ADDR`: dispatches of the compute program at ADDR are
/// printed with their groups and user data.
fn probed(program: u64) -> bool {
    static PROBE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *PROBE.get_or_init(|| {
        std::env::var("PS5SLOP_GPU_PROBE")
            .ok()
            .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
    }) == Some(program)
}

/// Reports a packet form the processor drops, once per form and register.
fn unhandled(what: std::fmt::Arguments) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
        std::sync::Mutex::new(None);
    let what = what.to_string();
    if SEEN
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(what.clone())
    {
        eprintln!("ps5slop: GPU: {what} dropped");
    }
}

fn trace(what: std::fmt::Arguments) {
    if trace_sync() {
        eprintln!(
            "ps5slop: GPU sync {:.3} [{}]: {what}",
            crate::seconds(),
            std::thread::current().name().unwrap_or("?")
        );
    }
}

/// `PS5SLOP_GPU_TRACE_SYNC`: a line in the sync trace from outside the command processors.
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
const MAX_GROUPS: u64 = 1 << 24;
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
            color_control_written: false,
        }
    }
}

/// Tells the host the command processor polls memory, for as long as it lives.
struct Polling<'a>(&'a dyn Host);

impl<'a> Polling<'a> {
    fn start(host: &'a dyn Host) -> Self {
        host.waiting(true);
        Self(host)
    }
}

impl Drop for Polling<'_> {
    fn drop(&mut self) {
        self.0.waiting(false);
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

fn gpu_clock() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

/// ATOMIC_MEM's operation `op` (the TC op numbers SharpEmu-mac uses) on
/// `current`: whether it applied (a compare-swap may not) and the new value.
fn atomic_op(op: u32, current: u64, source: u64, compare: u64) -> (bool, u64) {
    let v = match op {
        7 => source,
        8 if current == compare => source,
        8 => return (false, current),
        15 => current.wrapping_add(source),
        16 => current.wrapping_sub(source),
        17 => (current as i64).min(source as i64) as u64,
        18 => current.min(source),
        19 => (current as i64).max(source as i64) as u64,
        20 => current.max(source),
        21 => current & source,
        22 => current | source,
        23 => current ^ source,
        24 if current >= source => 0,
        24 => current + 1,
        25 if current == 0 || current > source => source,
        25 => current - 1,
        _ => current,
    };
    (true, v)
}

/// Applies `f` to the 32- or 64-bit value at `address` atomically where the
/// host maps it, and returns what `f` said about it.
fn atomic(
    host: &dyn Host,
    address: u64,
    wide: bool,
    f: impl Fn(u64) -> (bool, u64),
) -> Option<bool> {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::SeqCst};
    let size = if wide { 8 } else { 4 };
    let mut probe = [0u8; 8];
    if !address.is_multiple_of(size) || !host.read(address, &mut probe[..size as usize]) {
        return None;
    }
    let mut result = true;
    match host.host_pointer(address) {
        // SAFETY: the address is aligned and readable guest memory, which the
        // host maps for the life of the process.
        Some(ptr) if wide => {
            let cell = unsafe { AtomicU64::from_ptr(ptr.cast()) };
            let _ = cell.try_update(SeqCst, SeqCst, |v| {
                let (ok, new) = f(v);
                result = ok;
                Some(new)
            });
        }
        Some(ptr) => {
            let cell = unsafe { AtomicU32::from_ptr(ptr.cast()) };
            let _ = cell.try_update(SeqCst, SeqCst, |v| {
                let (ok, new) = f(v as u64);
                result = ok;
                Some(new as u32)
            });
        }
        None => {
            let (ok, new) = f(if wide {
                u64::from_le_bytes(probe)
            } else {
                u32::from_le_bytes(probe[..4].try_into().unwrap()) as u64
            });
            result = ok;
            host.write(address, &new.to_le_bytes()[..size as usize]);
        }
    }
    Some(result)
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
                p,
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
                self.draw(host, op, p, draw)
            }
            IT_DRAW_INDEX_OFFSET_2 if p.len() >= 3 => {
                let size = if self.uconfig[0x243] & 3 == 1 { 4 } else { 2 };
                let address = self.index_base + p[1] as u64 * size;
                let draw = Draw::Indexed {
                    count: p[2],
                    address,
                    instances: self.instances,
                };
                self.draw(host, op, p, draw)
            }
            IT_SET_BASE if p.len() >= 3 && p[0] & 0xF == 1 => self.indirect_base = addr(p[1], p[2]),
            IT_DMA_DATA if p.len() >= 6 => self.dma_data(host, p),
            IT_DRAW_INDEX_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 5);
                let size = if self.uconfig[0x243] & 3 == 1 { 4 } else { 2 };
                let address = self.index_base + a[2] as u64 * size;
                self.draw(
                    host,
                    op,
                    p,
                    Draw::Indexed {
                        count: a[0],
                        address,
                        instances: a[1].max(1),
                    },
                )
            }
            IT_DRAW_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 4);
                self.draw(
                    host,
                    op,
                    p,
                    Draw::Auto {
                        count: a[0],
                        instances: a[1].max(1),
                    },
                )
            }
            IT_DRAW_INDIRECT_MULTI | IT_DRAW_INDEX_INDIRECT_MULTI | IT_DRAW_INDEX_MULTI_AUTO => {
                self.stats.draws += 1
            }
            IT_DISPATCH_DIRECT if p.len() >= 3 => self.dispatch(host, [p[0], p[1], p[2]]),
            IT_DISPATCH_INDIRECT if !p.is_empty() => {
                let a = self.indirect_args(host, p[0], 3);
                self.dispatch(host, [a[0], a[1], a[2]]);
            }
            _ => {}
        }
        Ok(())
    }

    /// COPY_DATA from an immediate or memory into memory. Selectors are the
    /// compute form; the graphics form has the same low bits (Mesa's `sid.h`).
    fn copy_data(&mut self, host: &dyn Host, p: &[u32]) {
        let control = p[0];
        let (src_sel, dst_sel) = (control & 0xF, (control >> 8) & 0xF);
        let wide = control & (1 << 16) != 0;
        let n = if wide { 2 } else { 1 };
        if !matches!(dst_sel, 1 | 2 | 5) {
            unhandled(format_args!(
                "COPY_DATA from select {src_sel} to select {dst_sel} at {:#x}",
                p[3]
            ));
            return;
        }
        let dest = addr(p[3], p[4]);
        let values = match src_sel {
            5 => vec![p[1], p[2]][..n].to_vec(),
            1 | 2 => {
                self.wait_gpu(host, "COPY_DATA");
                read_dwords(host, addr(p[1], p[2]) & !3, n).unwrap_or_else(|_| vec![0; n])
            }
            9 => {
                let clock = gpu_clock();
                vec![clock as u32, (clock >> 32) as u32][..n].to_vec()
            }
            _ => return,
        };
        self.write_values(host, dest, &values, false);
    }

    /// ATOMIC_MEM: one read-modify-write after the work before it, or with
    /// command 1 a compare-swap retried until it succeeds.
    fn atomic_mem(&mut self, host: &dyn Host, p: &[u32]) {
        let op = p[0] & 0x1F;
        let wide = p[0] & 0x20 != 0;
        let command = (p[0] >> 8) & 0xF;
        let (address, source, compare) = (addr(p[1], p[2]), addr(p[3], p[4]), addr(p[5], p[6]));
        self.wait_gpu(host, "ATOMIC_MEM");
        let start = Instant::now();
        let mut polling = None;
        loop {
            let passed = atomic(host, address, wide, |v| atomic_op(op, v, source, compare));
            if command != 1 || op != 8 || passed != Some(false) || host.exiting() {
                return;
            }
            polling.get_or_insert_with(|| Polling::start(host));
            if start.elapsed() > WAIT_LIMIT {
                self.stats.timeouts += 1;
                return;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// MEM_SEMAPHORE: select 6 signals (adds one to the 64-bit count, or sets
    /// it to 1), select 7 waits until the count is not zero and takes one.
    fn mem_semaphore(&mut self, host: &dyn Host, p: &[u32]) {
        let address = (p[0] & !7) as u64 | (p[1] as u64) << 32;
        let select = p[2] >> 29;
        match select {
            6 => {
                self.wait_gpu(host, "MEM_SEMAPHORE signal");
                let set = p[2] & (1 << 20) != 0;
                atomic(host, address, true, |v| {
                    (true, if set { 1 } else { v.wrapping_add(1) })
                });
            }
            7 => {
                self.drain(host);
                let start = Instant::now();
                let mut polling = None;
                while atomic(host, address, true, |v| (v != 0, v.saturating_sub(1))) == Some(false)
                {
                    polling.get_or_insert_with(|| Polling::start(host));
                    if host.exiting() {
                        return;
                    }
                    if start.elapsed() > WAIT_LIMIT {
                        self.stats.timeouts += 1;
                        eprintln!("ps5slop: GPU: semaphore wait at {address:#x} timed out");
                        return;
                    }
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
            _ => {}
        }
    }

    /// REWIND: waits until the CPU sets the valid bit at `body`; the caller then
    /// fetches the rest of the buffer again, which the CPU may have written since.
    fn rewind(&mut self, host: &dyn Host, body: u64) {
        let valid = |host: &dyn Host| {
            let mut b = [0u8; 4];
            host.read(body, &mut b);
            u32::from_le_bytes(b) & REWIND_VALID != 0
        };
        if valid(host) {
            return;
        }
        self.drain(host);
        trace(format_args!("rewind {:#x} waits", body - 4));
        let start = Instant::now();
        let mut warned = false;
        let _polling = Polling::start(host);
        while !valid(host) && !host.exiting() {
            if !warned && start.elapsed() > WAIT_LIMIT {
                warned = true;
                eprintln!(
                    "ps5slop: GPU: REWIND at {:#x} still not valid after {} s",
                    body - 4,
                    WAIT_LIMIT.as_secs()
                );
            }
            let t = crate::timing::start();
            std::thread::sleep(Duration::from_micros(50));
            crate::timing::add(crate::timing::Part::WaitMemory, t);
        }
    }

    /// `n` argument dwords of an indirect draw or dispatch at `offset` past
    /// the indirect base, read once earlier GPU work (which may write them) is done.
    fn indirect_args(&mut self, host: &dyn Host, offset: u32, n: usize) -> Vec<u32> {
        self.wait_gpu(host, "indirect arguments");
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
        let src = addr(p[1], p[2]);
        let dst = addr(p[3], p[4]);
        if src_sel == 2
            && dst_sel != 1
            && bytes.is_multiple_of(4)
            && bytes <= DEFERRED_FILL
            && self.backend.is_some()
            && !synchronous_releases()
        {
            if std::env::var_os("PS5SLOP_TRACE_DMA").is_some() {
                eprintln!(
                    "ps5slop: DMA_DATA fill {dst:#x} {bytes:#x} bytes with {:#x}, queued",
                    p[1]
                );
            }
            let n = if dst_fixed { 1 } else { bytes / 4 };
            self.write_values(host, dst, &vec![p[1]; n], dst_fixed);
            return;
        }
        self.wait_gpu(host, "DMA_DATA");
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
        if std::env::var_os("PS5SLOP_TRACE_DMA").is_some() {
            eprintln!(
                "ps5slop: DMA_DATA src_sel {src_sel} {src:#x} dst_sel {dst_sel} {dst:#x} {bytes:#x} bytes, first word {:#x}",
                u32::from_le_bytes([
                    data[0],
                    *data.get(1).unwrap_or(&0),
                    *data.get(2).unwrap_or(&0),
                    *data.get(3).unwrap_or(&0)
                ])
            );
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
        let total = groups.iter().map(|&g| g as u64).product::<u64>();
        let program = (self.sh[crate::regs::sh::COMPUTE_PGM_LO] as u64
            | (self.sh[crate::regs::sh::COMPUTE_PGM_LO + 1] as u64) << 32)
            << 8;
        trace(format_args!("dispatch {groups:?} program {program:#x}"));
        if probed(program) {
            let user = &self.sh
                [crate::regs::sh::COMPUTE_USER_DATA_0..crate::regs::sh::COMPUTE_USER_DATA_0 + 16];
            eprintln!(
                "ps5slop: GPU probe [{}]: dispatch {groups:?} of {program:#x}, user data {user:x?}",
                std::thread::current().name().unwrap_or("?")
            );
        }
        if total > MAX_GROUPS {
            self.stats.huge_dispatches += 1;
            if self.stats.huge_dispatches <= 16 {
                eprintln!(
                    "ps5slop: GPU: dispatch of {groups:?} groups of the program at {program:#x} skipped"
                );
            }
            return;
        }
        if let Some(b) = self.backend.as_mut() {
            let state = State {
                context: &self.context,
                sh: &self.sh,
                uconfig: &self.uconfig,
                color_control_written: self.color_control_written,
            };
            let t = crate::timing::start();
            b.dispatch(host, &state, groups);
            crate::timing::add(crate::timing::Part::Dispatch, t);
        }
        if sync_dispatches() {
            self.sync();
            trace(format_args!("dispatch done"));
        }
        self.gpu_watch(host, || format!("dispatch {groups:?} of {program:#x}"));
    }

    /// `PS5SLOP_GPU_WATCH=ADDR`: waits for each draw and dispatch and reports
    /// those that change the 64-bit word at ADDR.
    fn gpu_watch(&mut self, host: &dyn Host, what: impl FnOnce() -> String) {
        static AT: std::sync::OnceLock<Option<(u64, Instant)>> = std::sync::OnceLock::new();
        let Some((at, from)) = *AT.get_or_init(|| {
            let v = std::env::var("PS5SLOP_GPU_WATCH").ok()?;
            let (addr, secs) = v.split_once('@').unwrap_or((&v, "0"));
            Some((
                u64::from_str_radix(addr.trim_start_matches("0x"), 16).ok()?,
                Instant::now() + Duration::from_secs(secs.parse().ok()?),
            ))
        }) else {
            return;
        };
        if Instant::now() < from {
            return;
        }
        self.sync();
        let now = read_u64(host, at);
        if self.watched != Some(now) {
            if self.watched.is_some() {
                let user = &self.sh[crate::regs::sh::COMPUTE_USER_DATA_0
                    ..crate::regs::sh::COMPUTE_USER_DATA_0 + 16];
                eprintln!(
                    "ps5slop: GPU watch {:.3}: {at:#x} {:#x?} -> {now:#x} after {} on {} (compute user {user:x?})",
                    crate::seconds(),
                    self.watched,
                    what(),
                    std::thread::current().name().unwrap_or("?")
                );
            }
            self.watched = Some(now);
        }
    }

    fn draw(&mut self, host: &dyn Host, op: u32, p: &[u32], draw: Draw) {
        self.stats.draws += 1;
        self.dump_draw(op, p);
        if let Some(b) = self.backend.as_mut() {
            let state = State {
                context: &self.context,
                sh: &self.sh,
                uconfig: &self.uconfig,
                color_control_written: self.color_control_written,
            };
            let t = crate::timing::start();
            b.draw(host, &state, draw);
            crate::timing::add(crate::timing::Part::Draw, t);
        }
        self.color_control_written = false;
        self.gpu_watch(host, || format!("draw {op:#x}"));
    }

    /// Waits for the GPU work so far and makes pending releases take effect,
    /// before `why` reads what that work may have written.
    fn wait_gpu(&mut self, host: &dyn Host, why: &'static str) {
        *self.stats.gpu_waits.entry(why).or_default() += 1;
        self.drain_now(host);
        self.sync();
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
            let n = std::env::var("PS5SLOP_GPU_DUMP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            LEFT.store(n, Relaxed);
        }
        static SKIP: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
        if SKIP.load(Relaxed) == i64::MIN {
            let n = std::env::var("PS5SLOP_GPU_DUMP_SKIP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
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
        for (name, lo) in [
            ("PS", 0x08usize),
            ("VS", 0x48),
            ("GS", 0x88),
            ("ES", 0xC8),
            ("HS", 0x108),
            ("LS", 0x148),
        ] {
            if sh[lo] != 0 || sh[lo + 1] != 0 {
                eprintln!(
                    "  {name} pgm={:#x} rsrc1={:#x} rsrc2={:#x} user={:x?}",
                    (sh[lo] as u64 | (sh[lo + 1] as u64) << 32) << 8,
                    sh[lo + 2],
                    sh[lo + 3],
                    &sh[lo + 4..lo + 20]
                );
            }
        }
        eprintln!(
            "  stages={:#x} prim={:#x} idx={:#x} cb0 base={:#x} ext={:#x} view={:#x} info={:#x} attrib={:#x} attrib2={:#x} attrib3={:#x} target_mask={:#x} color_ctl={:#x} blend0={:#x}",
            c[0x2D5],
            u[0x242],
            u[0x243],
            c[0x318],
            c[0x390],
            c[0x31B],
            c[0x31C],
            c[0x31D],
            c[0x3B0],
            c[0x3B8],
            c[0x8E],
            c[0x202],
            c[0x1E0]
        );
        eprintln!(
            "  vport scale/off={:?} scissor={:#x}/{:#x} vte={:#x} spi_ps_in={:#x}/{:#x} col_fmt={:#x} z_info={:#x} depth_ctl={:#x} ps_input_cntl0={:#x} vs_out={:#x} pos_fmt={:#x}",
            c[0x10F..0x115]
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>(),
            c[0x90],
            c[0x91],
            c[0x206],
            c[0x1B3],
            c[0x1B4],
            c[0x1C5],
            c[0x10],
            c[0x200],
            c[0x191],
            c[0x1B1],
            c[0x1C3]
        );
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
                self.release(
                    host,
                    data,
                    interrupt,
                    addr(p[2], p[3]),
                    addr(p[4], p[5]),
                    p[6] & 0x07FF_FFFF,
                );
            }
            NOP_WAIT_MEMORY_32 if p.len() >= 5 => {
                let (mask, reference, control) = if p.len() == 5 {
                    (p[2], p[4], p[3])
                } else {
                    (p[2], p[3], p[4])
                };
                self.wait(
                    host,
                    control,
                    addr(p[0], p[1]),
                    reference as u64,
                    mask as u64,
                    false,
                );
            }
            NOP_WAIT_MEMORY_64 if p.len() >= 7 => {
                self.wait(
                    host,
                    p[6],
                    addr(p[0], p[1]),
                    addr(p[4], p[5]),
                    addr(p[2], p[3]),
                    true,
                );
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
        let now = Instant::now();
        if let Some(last) = self.last_flip {
            let gap = now - last;
            self.stats.longest_flip_gap = self.stats.longest_flip_gap.max(gap);
            if gap > Duration::from_secs(1) && self.stats.long_flip_gaps.len() < 64 {
                self.stats.long_flip_gaps.push(gap);
            }
        }
        self.last_flip = Some(now);
        self.stats.flips += 1;
        host.flip(flip);
    }

    fn write_values(&mut self, host: &dyn Host, dest: u64, values: &[u32], fixed: bool) {
        trace(format_args!(
            "write {dest:#x} {:x?}",
            &values[..values.len().min(4)]
        ));
        if self.backend.is_some() && !synchronous_releases() {
            self.pending.push(Pending::Write {
                dest,
                values: values.to_vec(),
                fixed,
            });
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
    fn release(
        &mut self,
        host: &dyn Host,
        data: u32,
        interrupt: u32,
        dest: u64,
        value: u64,
        context: u32,
    ) {
        trace(format_args!(
            "release data {data} interrupt {interrupt} {dest:#x} = {value:#x}"
        ));
        if self.backend.is_some() && !synchronous_releases() {
            self.pending.push(Pending::Release {
                data,
                interrupt,
                dest,
                value,
                context,
            });
        } else {
            self.sync();
            self.apply_release(host, data, interrupt, dest, value, context);
        }
    }

    /// Ends a submission: pending releases take effect, and the GPU work
    /// encoded so far is done.
    pub fn finish(&mut self, host: &dyn Host) {
        self.drain(host);
        self.sync();
    }

    /// Ends a submission and raises the queue's interrupt. One that ends a
    /// frame waits for the GPU ([`Processor::finish`]); for others the work is
    /// committed and its pending releases, then the interrupt, take effect once
    /// the GPU has done it ([`Processor::retire`]), without waiting here.
    pub fn end_submission(&mut self, host: &dyn Host, ends_frame: bool) {
        let queued = !self.pending.is_empty() || !self.retiring.is_empty();
        if !ends_frame
            && queued
            && !synchronous_releases()
            && !wait_submissions()
            && let Some(b) = self.backend.as_mut()
        {
            let point = b.commit_point();
            self.pending.push(Pending::Interrupt);
            self.retiring
                .push_back((point, std::mem::take(&mut self.pending)));
            self.retire(host);
            return;
        }
        if ends_frame {
            self.finish(host);
        } else {
            self.drain(host);
        }
        host.interrupt(0);
    }

    /// Makes the pending work of ended submissions whose GPU work is done take
    /// effect, in order. Returns whether some is still waiting for the GPU.
    pub fn retire(&mut self, host: &dyn Host) -> bool {
        while let Some(&(point, _)) = self.retiring.front() {
            if !self.backend.as_mut().is_none_or(|b| b.done(point)) {
                break;
            }
            let (_, list) = self.retiring.pop_front().expect("checked above");
            for p in list {
                self.apply(host, p);
            }
        }
        !self.retiring.is_empty()
    }

    /// Waits for the GPU work behind pending releases, then makes them take
    /// effect in order. Called before anything that could observe them.
    pub fn drain(&mut self, host: &dyn Host) {
        if self.pending.is_empty() && self.retiring.is_empty() {
            return;
        }
        *self
            .stats
            .gpu_waits
            .entry("releases before a poll or flip")
            .or_default() += 1;
        self.drain_now(host);
    }

    fn drain_now(&mut self, host: &dyn Host) {
        self.sync();
        for (_, list) in std::mem::take(&mut self.retiring) {
            for p in list {
                self.apply(host, p);
            }
        }
        for p in std::mem::take(&mut self.pending) {
            self.apply(host, p);
        }
    }

    fn apply(&mut self, host: &dyn Host, p: Pending) {
        match p {
            Pending::Release {
                data,
                interrupt,
                dest,
                value,
                context,
            } => self.apply_release(host, data, interrupt, dest, value, context),
            Pending::Write {
                dest,
                values,
                fixed,
            } => Self::apply_write(host, dest, &values, fixed),
            Pending::Interrupt => host.interrupt(0),
        }
    }

    /// The 32-bit value queued work will leave at `address`, the latest first.
    fn queued_word(&self, address: u64) -> Option<u32> {
        self.pending
            .iter()
            .rev()
            .chain(self.retiring.iter().rev().flat_map(|(_, l)| l.iter().rev()))
            .find_map(|p| p.word_at(address))
    }

    fn apply_release(
        &mut self,
        host: &dyn Host,
        data: u32,
        interrupt: u32,
        dest: u64,
        value: u64,
        context: u32,
    ) {
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
                    host.write(dest, &gpu_clock().to_le_bytes());
                }
                7 => {
                    atomic(host, dest, false, |v| {
                        (true, v.wrapping_add(value & 0xFFFF_FFFF))
                    });
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
    fn wait(
        &mut self,
        host: &dyn Host,
        control: u32,
        address: u64,
        reference: u64,
        mask: u64,
        wide: bool,
    ) {
        trace(format_args!(
            "wait {address:#x} & {mask:#x} function {} {reference:#x}",
            control & 7
        ));
        // GPU work runs in order, so queued releases and writes come first; with
        // none at the label, memory holds what the wait sees (unless a program writes it).
        let word = |at: u64| self.queued_word(at);
        let read = |host: &dyn Host| {
            if wide {
                read_u64(host, address)
            } else {
                let mut b = [0u8; 4];
                host.read(address, &mut b);
                u32::from_le_bytes(b) as u64
            }
        };
        let value = match (word(address), wide.then(|| word(address + 4))) {
            (Some(lo), None) => Some(lo as u64),
            (Some(lo), Some(Some(hi))) => Some(lo as u64 | (hi as u64) << 32),
            (None, None | Some(None)) => Some(read(host)),
            _ => None,
        };
        if value.is_some_and(|v| compare(control, v & mask, reference)) {
            if word(address).is_some() {
                self.stats.deferred_waits += 1;
            }
            return;
        }
        self.drain(host);
        let start = Instant::now();
        let mut polling = None;
        loop {
            let value = read(host);
            if compare(control, value & mask, reference) || host.exiting() {
                return;
            }
            if start.elapsed() > WAIT_LIMIT {
                self.stats.timeouts += 1;
                if self.stats.timeouts <= 16 {
                    eprintln!(
                        "ps5slop: GPU: wait timed out on {}: {address:#x} & {mask:#x} = {value:#x}, function {}, reference {reference:#x}",
                        std::thread::current().name().unwrap_or("?"),
                        control & 7
                    );
                }
                return;
            }
            polling.get_or_insert_with(|| Polling::start(host));
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
            let Some(src) = mem.get(addr as usize..addr as usize + buf.len()) else {
                return false;
            };
            buf.copy_from_slice(src);
            true
        }
        fn write(&self, addr: u64, data: &[u8]) -> bool {
            let mut mem = self.mem.borrow_mut();
            let Some(dst) = mem.get_mut(addr as usize..addr as usize + data.len()) else {
                return false;
            };
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
        let packets = [
            header(4, IT_SET_BASE, 0),
            1,
            0xA00,
            0,
            header(3, IT_DISPATCH_INDIRECT, 0),
            8,
            0,
        ];
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
        let mut p = Processor {
            backend: Some(Box::new(Groups(seen.clone()))),
            ..Processor::default()
        };
        p.run(&host, 0x100, packets.len() as u32).unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![[2, 3, 1]]);
    }

    #[test]
    fn label_waits_pass_without_the_gpu_and_releases_wait_for_it() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
        #[derive(Clone, Default)]
        struct Gpu {
            syncs: std::sync::Arc<AtomicU32>,
            done: std::sync::Arc<AtomicBool>,
        }
        impl Backend for Gpu {
            fn draw(&mut self, _: &dyn Host, _: &State, _: Draw) {}
            fn dispatch(&mut self, _: &dyn Host, _: &State, _: [u32; 3]) {}
            fn sync(&mut self) {
                self.syncs.fetch_add(1, Relaxed);
                self.done.store(true, Relaxed);
            }
            fn flush(&mut self) {}
            fn commit_point(&mut self) -> u64 {
                7
            }
            fn done(&mut self, point: u64) -> bool {
                assert_eq!(point, 7);
                self.done.load(Relaxed)
            }
            fn frame_done(&mut self, _: Option<u64>) {}
        }
        let host = Fake::default();
        host.mem.replace(vec![0; 0x1000]);
        host.write(0x800, &5u32.to_le_bytes());
        // AGC's barrier: zero a label, release 1 into it, wait for 1.
        let packets = [
            header(7, IT_DMA_DATA, 0),
            2 << 29,
            0,
            0x800,
            0,
            0,
            4,
            header(8, IT_RELEASE_MEM, 0),
            0,
            (1 << 29) | (2 << 24),
            0x800,
            0,
            1,
            0,
            3,
            header(7, IT_WAIT_REG_MEM, 0),
            0x13,
            0x800,
            0,
            1,
            0xFFFF_FFFF,
            0,
        ];
        let bytes: Vec<u8> = packets.iter().flat_map(|d| d.to_le_bytes()).collect();
        host.write(0x100, &bytes);
        let gpu = Gpu::default();
        let mut p = Processor {
            backend: Some(Box::new(gpu.clone())),
            ..Processor::default()
        };
        p.run(&host, 0x100, packets.len() as u32).unwrap();
        p.end_submission(&host, false);
        assert_eq!(gpu.syncs.load(Relaxed), 0);
        assert_eq!(
            word(&host, 0x800),
            5,
            "nothing takes effect before the GPU is done"
        );
        assert!(host.interrupts.borrow().is_empty());
        assert!(p.retire(&host));
        gpu.done.store(true, Relaxed);
        assert!(!p.retire(&host));
        assert_eq!(word(&host, 0x800), 1);
        assert_eq!(*host.interrupts.borrow(), [3, 0]);
        assert_eq!(gpu.syncs.load(Relaxed), 0);
    }

    #[test]
    fn queued_writes_and_releases_report_what_they_leave_in_memory() {
        let w = Pending::Write {
            dest: 0x100,
            values: vec![1, 2, 3],
            fixed: false,
        };
        assert_eq!(
            (w.word_at(0x104), w.word_at(0x10c), w.word_at(0x102)),
            (Some(2), None, None)
        );
        let f = Pending::Write {
            dest: 0x100,
            values: vec![1, 2, 3],
            fixed: true,
        };
        assert_eq!((f.word_at(0x100), f.word_at(0x104)), (Some(3), None));
        let r = Pending::Release {
            data: 2,
            interrupt: 0,
            dest: 0x200,
            value: 0x5_0000_0007,
            context: 0,
        };
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
        host.write(
            0x400,
            &inner
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let outer = [
            header(4, IT_INDIRECT_BUFFER, 0),
            0x400,
            0,
            4,
            header(2, IT_DRAW_INDEX_AUTO, 0),
            3,
        ];
        host.write(
            0x100,
            &outer
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
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
                seg.extend([
                    header(4, IT_INDIRECT_BUFFER, 0),
                    (at + 0x100) as u32,
                    0,
                    7 | 1 << 20,
                ]);
                seg.extend([header(2, IT_DRAW_INDEX_AUTO, 0), 3]);
            }
            host.write(
                at,
                &seg.iter()
                    .flat_map(|d| d.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
        }
        let mut p = Processor::default();
        p.run(&host, 0x100, 8).unwrap();
        assert_eq!(p.stats.draws, segments);
    }

    #[test]
    fn a_valid_rewind_runs_what_was_written_after_it() {
        let host = Fake::default();
        host.mem.replace(vec![0; 0x1000]);
        let first = [header(2, IT_REWIND, 0), REWIND_VALID];
        host.write(
            0x100,
            &first
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let late = [header(5, IT_WRITE_DATA, 0), 5 << 8, 0x800, 0, 9];
        struct Late<'a>(&'a Fake, [u32; 5]);
        impl Host for Late<'_> {
            fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
                if addr == 0x108 {
                    self.0.write(
                        0x108,
                        &self
                            .1
                            .iter()
                            .flat_map(|d| d.to_le_bytes())
                            .collect::<Vec<u8>>(),
                    );
                }
                self.0.read(addr, buf)
            }
            fn write(&self, addr: u64, data: &[u8]) -> bool {
                self.0.write(addr, data)
            }
            fn interrupt(&self, _: u32) {}
            fn flip(&self, _: Flip) {}
            fn exiting(&self) -> bool {
                false
            }
        }
        let mut p = Processor::default();
        p.run(&Late(&host, late), 0x100, 7).unwrap();
        assert_eq!(word(&host, 0x800), 9);
    }

    #[test]
    fn cond_exec_skips_its_dwords_when_the_flag_is_zero() {
        let (host, _) = run(&[
            header(5, IT_COND_EXEC, 0),
            0x900,
            0,
            0,
            5,
            header(5, IT_WRITE_DATA, 0),
            5 << 8,
            0x800,
            0,
            1,
            header(5, IT_WRITE_DATA, 0),
            5 << 8,
            0x804,
            0,
            2,
        ]);
        assert_eq!((word(&host, 0x800), word(&host, 0x804)), (0, 2));
    }

    #[test]
    fn copy_data_atomics_and_semaphores_update_memory() {
        let (host, _) = run(&[
            header(6, IT_COPY_DATA, 0),
            5 | 5 << 8,
            0x77,
            0,
            0x800,
            0,
            header(9, IT_ATOMIC_MEM, 0),
            15 | 0x40,
            0x800,
            0,
            3,
            0,
            0,
            0,
            0,
            header(4, IT_MEM_SEMAPHORE, 0),
            0x808,
            0,
            6 << 29,
            header(8, IT_RELEASE_MEM, 0),
            0,
            7 << 29,
            0x810,
            0,
            5,
            0,
            0,
        ]);
        assert_eq!(
            (word(&host, 0x800), word(&host, 0x808), word(&host, 0x810)),
            (0x7A, 1, 5)
        );
    }

    #[test]
    fn a_semaphore_wait_takes_one_count() {
        let host = Fake::default();
        host.mem.replace(vec![0; 0x1000]);
        host.write(0x808, &2u64.to_le_bytes());
        let packets = [header(4, IT_MEM_SEMAPHORE, 0), 0x808, 0, 7 << 29];
        host.write(
            0x100,
            &packets
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let mut p = Processor::default();
        p.run(&host, 0x100, 4).unwrap();
        assert_eq!((word(&host, 0x808), p.stats.timeouts), (1, 0));
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
