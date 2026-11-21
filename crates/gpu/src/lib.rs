//! The PS5 GPU as titles drive it: a command processor that walks the PM4
//! packets in submitted command buffers. Packets the CPU can observe (memory
//! writes, end-of-pipe labels and interrupts, waits, flips) take effect;
//! register writes are recorded; draws go to a [`Backend`], the Metal
//! renderer on macOS. See `docs/gpu.md`. Packet meanings follow AMD's
//! PM4 documentation and SharpEmu's interpreter (GPL-2.0-or-later).

#[cfg(target_os = "macos")]
mod compile;
#[cfg(target_os = "macos")]
pub mod metal;
mod pm4;
pub mod regs;
pub mod tiling;
pub mod timing;

pub use pm4::{Processor, Stats};

/// The register banks as a draw sees them.
pub struct State<'a> {
    pub context: &'a [u32],
    pub sh: &'a [u32],
    pub uconfig: &'a [u32],
    /// `CB_COLOR_CONTROL` was written since the previous draw.
    pub color_control_written: bool,
}

/// A draw's vertex source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Draw {
    Auto { count: u32, instances: u32 },
    Indexed { count: u32, address: u64, instances: u32 },
}

/// Where draws and dispatches are carried out.
pub trait Backend: Send {
    fn draw(&mut self, host: &dyn Host, state: &State, draw: Draw);
    /// A compute dispatch of `groups` thread groups.
    fn dispatch(&mut self, host: &dyn Host, state: &State, groups: [u32; 3]);
    /// Finishes GPU work recorded so far, before the CPU may observe its results.
    fn sync(&mut self);
    /// Starts the GPU on the work recorded so far without waiting for it.
    fn flush(&mut self) {
        self.sync();
    }
    /// As [`Backend::sync`], at a flip that puts the buffer at guest address
    /// `shown` on screen.
    fn frame_done(&mut self, shown: Option<u64>);
    /// Runs `f` on the global data share (64 KiB) programs use, if the
    /// backend keeps it; the caller has finished earlier GPU work first.
    fn with_gds(&mut self, _f: &mut dyn FnMut(&mut [u8])) -> bool {
        false
    }
    /// A line of counters for logs.
    fn summary(&self) -> String {
        String::new()
    }
}

/// One backend shared by the command processors of several queues; each
/// call holds it for its duration, so work is encoded in the order calls come.
#[derive(Clone)]
pub struct SharedBackend(pub std::sync::Arc<std::sync::Mutex<Box<dyn Backend>>>);

impl SharedBackend {
    fn lock(&self) -> std::sync::MutexGuard<'_, Box<dyn Backend>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Backend for SharedBackend {
    fn draw(&mut self, host: &dyn Host, state: &State, draw: Draw) {
        self.lock().draw(host, state, draw)
    }
    fn dispatch(&mut self, host: &dyn Host, state: &State, groups: [u32; 3]) {
        self.lock().dispatch(host, state, groups)
    }
    fn sync(&mut self) {
        self.lock().sync()
    }
    fn flush(&mut self) {
        self.lock().flush()
    }
    fn frame_done(&mut self, shown: Option<u64>) {
        self.lock().frame_done(shown)
    }
    fn with_gds(&mut self, f: &mut dyn FnMut(&mut [u8])) -> bool {
        self.lock().with_gds(f)
    }
    fn summary(&self) -> String {
        self.lock().summary()
    }
}

/// Runs `f` inside an Objective-C autorelease pool where the platform has one.
pub fn with_pool<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_os = "macos")]
    {
        objc2::rc::autoreleasepool(|_| f())
    }
    #[cfg(not(target_os = "macos"))]
    {
        f()
    }
}

/// What the command processor needs from the rest of the emulator.
pub trait Host {
    /// Reads guest memory; false if the range is not readable.
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool;
    /// Writes guest memory; false if the range is not writable.
    fn write(&self, addr: u64, data: &[u8]) -> bool;
    /// An end-of-pipe interrupt for the GPU event queues, with its context id.
    fn interrupt(&self, context: u32);
    /// A flip the command buffer asked for.
    fn flip(&self, flip: Flip);
    /// True once the process is ending, so waits give up.
    fn exiting(&self) -> bool;
    /// The host-backed guest range around `addr`, as `(start, end)`.
    fn extent(&self, _addr: u64) -> Option<(u64, u64)> {
        None
    }
    /// The host address of guest address `addr`.
    fn host_pointer(&self, _addr: u64) -> Option<*mut u8> {
        None
    }
    /// The guest address of display buffer `index`.
    fn display_buffer(&self, _index: i32) -> Option<u64> {
        None
    }
    /// A number that changes whenever guest mappings change.
    fn generation(&self) -> u64 {
        0
    }
    /// Ranges given new host pages since mark `since`, and the next mark;
    /// `None` means anything may have changed.
    fn replaced_since(&self, _since: u64) -> (u64, Option<Vec<(u64, u64)>>) {
        (0, None)
    }
    /// The guest range where memory the GPU uses is mapped, as `(start, end)`.
    fn gpu_area(&self) -> Option<(u64, u64)> {
        None
    }
    /// Gives every page in `[start, end)` host memory (zeros where nothing
    /// is mapped), so a window can span the holes; false if it cannot.
    fn back(&self, _start: u64, _end: u64) -> bool {
        false
    }
}

/// A flip request from a command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flip {
    pub handle: i32,
    pub index: i32,
    pub mode: i32,
    pub arg: i64,
}

/// Why a command buffer could not be processed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A packet or buffer could not be read at this address.
    Unreadable(u64),
    /// A packet header claims more dwords than its buffer has.
    Truncated { addr: u64, header: u32 },
    /// Indirect buffers nested too deep.
    TooDeep(u64),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unreadable(a) => write!(f, "command buffer memory at {a:#x} is not readable"),
            Error::Truncated { addr, header } => {
                write!(f, "packet {header:#010x} at {addr:#x} runs past its buffer")
            }
            Error::TooDeep(a) => write!(f, "indirect buffers nest too deep at {a:#x}"),
        }
    }
}
