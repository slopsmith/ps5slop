//! Where the GPU thread's time goes, for the control socket's `profile`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Part {
    /// Encoding draws: state, programs, textures, Metal commands.
    Draw,
    Dispatch,
    /// Waiting for Metal to finish submitted work.
    Sync,
    /// The sleep that paces flips to 60 Hz.
    Pacing,
    /// Polling memory for a WAIT_REG_MEM condition.
    WaitMemory,
    /// Waiting for the next submission.
    Idle,
    /// Time the GPU itself spent executing our command buffers.
    GpuBusy,
    /// Command buffers committed and waited for.
    Syncs,
    /// From the kernel scheduling a command buffer to the GPU starting it.
    Scheduling,
    /// From our commit to the kernel scheduling it.
    Submission,
    /// From the GPU finishing to our wait returning.
    WakeUp,
    /// Compiling MSL and pipelines, summed over the compile threads.
    Compile,
    /// Encoding waiting for a program or pipeline still compiling.
    CompileWait,
    /// Waiting for a command buffer to finish, wherever it happens (also
    /// inside draw and dispatch encoding, and inside [`Part::Sync`]).
    CommandWait,
}

pub const PARTS: usize = 14;

pub const NAMES: [&str; PARTS] = [
    "draw encoding",
    "dispatch encoding",
    "waiting for Metal",
    "flip pacing",
    "memory waits",
    "idle",
    "GPU busy",
    "syncs",
    "scheduled to start",
    "commit to scheduled",
    "done to woken",
    "compiling (compile threads)",
    "waiting for compiles (in encoding)",
    "waiting for command buffers (anywhere)",
];

static ON: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];

/// Starts counting from zero, or stops.
pub fn set(on: bool) {
    for n in &NS {
        n.store(0, Ordering::Relaxed);
    }
    ON.store(on, Ordering::Relaxed);
}

/// Now, if counting.
pub fn start() -> Option<Instant> {
    ON.load(Ordering::Relaxed).then(Instant::now)
}

pub fn add(part: Part, since: Option<Instant>) {
    if let Some(t) = since {
        NS[part as usize].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Adds `n` (nanoseconds, or a count for [`Part::Syncs`]) to `part`.
pub fn add_raw(part: Part, n: u64) {
    if ON.load(Ordering::Relaxed) {
        NS[part as usize].fetch_add(n, Ordering::Relaxed);
    }
}

/// Nanoseconds per [`Part`] since counting started.
pub fn totals() -> [u64; PARTS] {
    NS.each_ref().map(|n| n.load(Ordering::Relaxed))
}

/// Whether to label Metal encoders for GPU traces.
pub fn labels() -> bool {
    diag::config().gpu.labels
}

/// Host time in seconds, on the clock Metal's command buffer times use.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
pub fn host_seconds() -> f64 {
    static BASE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let scale = *BASE.get_or_init(|| {
        let mut info = libc::mach_timebase_info { numer: 0, denom: 0 };
        // SAFETY: fills the struct we pass.
        unsafe { libc::mach_timebase_info(&mut info) };
        info.numer as f64 / info.denom as f64 / 1e9
    });
    // SAFETY: no arguments.
    (unsafe { libc::mach_absolute_time() }) as f64 * scale
}
