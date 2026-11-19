//! Submitting command buffers: `sceAgcDriverSubmitDcb` and friends hand
//! them to a GPU thread, which runs them through the `gpu` crate's command
//! processor in submission order. Ported in shape from SharpEmu's
//! `AgcExports.Submissions.cs`.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use super::{ERROR_INVALID_ARGUMENT, ret};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Kernel, Outcome};

/// The event filter of GPU interrupts (Orbis's `EVFILT_GRAPHICS_CORE`, as
/// public emulators number it; unverified).
pub(crate) const GRAPHICS_FILTER: i16 = -14;
/// The event ident the graphics queue's end-of-pipe interrupts use.
const GRAPHICS_EOP_EVENT: u64 = 0;
/// SharpEmu's limit on one submission's size.
const MAX_DWORDS: u32 = 0xF_FFFF;

/// Command buffers to run, in order, on one queue.
struct Job {
    buffers: Vec<(u64, u32)>,
}

/// The GPU thread's queue, started by the first submission.
#[derive(Default)]
pub(super) struct Gpu {
    jobs: Mutex<Option<Sender<Job>>>,
    pub(super) stats: Arc<Mutex<gpu::Stats>>,
}

pub(super) static FUNCTIONS: &[Function] = &[
    d("sceAgcDriverSubmitDcb", submit_dcb),
    d("sceAgcDriverAgrSubmitDcb", submit_dcb),
    d("sceAgcDriverSubmitAcb", submit_acb),
    d("sceAgcDriverSubmitMultiDcbs", submit_multi_dcbs),
    d("sceAgcDriverAgrSubmitMultiDcbs", submit_multi_dcbs),
    d("sceAgcDriverSubmitMultiAcbs", submit_multi_acbs),
    Function {
        library: "libSceAgc",
        name: "sceAgcSuspendPoint",
        func: suspend_point,
    },
];

const fn d(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgcDriver",
        name,
        func,
    }
}

/// The guest memory, events and display, as the command processor sees them.
struct Host {
    kernel: Arc<Kernel>,
    mem: memory::Memory,
}

impl gpu::Host for Host {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.mem.read(addr, buf).is_ok()
    }

    fn write(&self, addr: u64, data: &[u8]) -> bool {
        self.mem.write(addr, data).is_ok()
    }

    fn interrupt(&self, context: u32) {
        self.kernel
            .trigger_event(GRAPHICS_EOP_EVENT, GRAPHICS_FILTER, context as u64);
    }

    fn flip(&self, flip: gpu::Flip) {
        self.kernel.flip(flip.index, flip.arg as u64);
    }

    fn exiting(&self) -> bool {
        self.kernel.has_exited()
    }
}

fn gpu_thread(host: Host, jobs: Receiver<Job>, stats: Arc<Mutex<gpu::Stats>>) {
    let mut processor = gpu::Processor::default();
    let report = std::env::var_os("PS5SLOP_GPU_STATS").is_some();
    let mut last_report = std::time::Instant::now();
    while let Ok(job) = jobs.recv() {
        for (addr, dwords) in job.buffers {
            if let Err(e) = processor.run(&host, addr, dwords) {
                eprintln!("ps5slop: GPU: {e}");
            }
        }
        *lock(&stats) = processor.stats.clone();
        if report && last_report.elapsed().as_secs() >= 1 {
            last_report = std::time::Instant::now();
            let s = &processor.stats;
            eprintln!(
                "ps5slop: GPU: {} submissions, {} packets, {} draws, {} dispatches, {} flips, {} interrupts, {} wait timeouts",
                s.submissions, s.packets, s.draws, s.dispatches, s.flips, s.interrupts, s.timeouts
            );
        }
        if host.kernel.has_exited() {
            break;
        }
    }
}

fn enqueue(env: &Env, buffers: Vec<(u64, u32)>) {
    let gpu = &env.kernel.agc.gpu;
    let mut jobs = lock(&gpu.jobs);
    let sender = jobs.get_or_insert_with(|| {
        let (tx, rx) = channel();
        let host = Host {
            kernel: env.kernel.this(),
            mem: env.kernel.memory(),
        };
        let stats = gpu.stats.clone();
        std::thread::Builder::new()
            .name("gpu".into())
            .spawn(move || gpu_thread(host, rx, stats))
            .expect("the GPU thread starts");
        tx
    });
    let _ = sender.send(Job { buffers });
}

/// A submit descriptor: the command address (u64 at 0) and dword count (u32 at 8).
fn descriptor(env: &Env, at: u64) -> Result<Option<(u64, u32)>, CallError> {
    if at == 0 {
        return Ok(None);
    }
    let addr = env.mem.read_u64(at)?;
    let mut b = [0u8; 4];
    env.mem.read(at + 8, &mut b)?;
    let dwords = u32::from_le_bytes(b);
    Ok((dwords <= MAX_DWORDS).then_some((addr, dwords)))
}

/// `sceAgcDriverSubmitDcb(descriptor)`.
fn submit_dcb(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffer) = descriptor(env, env.arg(0))? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, vec![buffer]);
    ret(0)
}

/// `sceAgcDriverSubmitAcb(owner, descriptor)`.
fn submit_acb(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffer) = descriptor(env, env.arg(1))? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, vec![buffer]);
    ret(0)
}

/// The `count` buffers whose addresses (u64s) and sizes (u32s) are listed
/// at `addrs` and `sizes`.
fn buffer_list(
    env: &Env,
    addrs: u64,
    sizes: u64,
    count: u64,
) -> Result<Option<Vec<(u64, u32)>>, CallError> {
    if addrs == 0 || sizes == 0 || count > 1024 {
        return Ok(None);
    }
    let mut out = Vec::new();
    for i in 0..count {
        let addr = env.mem.read_u64(addrs + 8 * i)?;
        let mut b = [0u8; 4];
        env.mem.read(sizes + 4 * i, &mut b)?;
        out.push((addr, u32::from_le_bytes(b)));
    }
    Ok(Some(out))
}

/// `sceAgcDriverSubmitMultiDcbs(addresses, sizes, count)`.
fn submit_multi_dcbs(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffers) = buffer_list(env, env.arg(0), env.arg(1), env.arg(2) as u32 as u64)? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, buffers);
    ret(0)
}

/// `sceAgcDriverSubmitMultiAcbs(owner, addresses, sizes, count)`.
fn submit_multi_acbs(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffers) = buffer_list(env, env.arg(1), env.arg(2), env.arg(3) as u32 as u64)? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, buffers);
    ret(0)
}

/// `sceAgcSuspendPoint()`: where a title may be suspended; nothing to do.
fn suspend_point(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}
