//! The PS5 GPU as titles drive it: a command processor that walks the PM4
//! packets in submitted command buffers. Packets the CPU can observe (memory
//! writes, end-of-pipe labels and interrupts, waits, flips) take effect;
//! register writes are recorded; draws and dispatches are counted until the
//! Metal renderer exists. See `docs/gpu.md`. Packet meanings follow AMD's
//! PM4 documentation and SharpEmu's interpreter (GPL-2.0-or-later).

mod pm4;

pub use pm4::{Processor, Stats};

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
