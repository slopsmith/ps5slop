//! Submitting command buffers: `sceAgcDriverSubmitDcb` and friends hand
//! them to a GPU thread per queue (graphics, and each compute owner), which
//! runs them through the `gpu` crate's command processor in submission
//! order. The queues share one Metal renderer, and a queue waiting on a
//! label does not hold up the others. Ported in shape from SharpEmu's
//! `AgcExports.Submissions.cs`.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};

use super::{ERROR_INVALID_ARGUMENT, ret};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Kernel, Outcome};

/// The event filter of GPU interrupts (Orbis's `EVFILT_GRAPHICS_CORE`, as
/// public emulators number it; unverified).
pub(crate) const GRAPHICS_FILTER: i16 = -14;
/// The event ident the graphics queue's end-of-pipe interrupts use.
const GRAPHICS_EOP_EVENT: u64 = 0;
/// Submissions waiting for the GPU before a submitter blocks, as a full ring
/// would block it (our choice of depth, unverified).
const MAX_QUEUED: usize = 1;
/// SharpEmu's limit on one submission's size.
const MAX_DWORDS: u32 = 0xF_FFFF;

/// Command buffers to run, in order, on one queue.
struct Job {
    buffers: Vec<(u64, u32)>,
}

/// The graphics queue's key; compute queues are keyed by their owner.
const GRAPHICS: u64 = u64::MAX;

/// The GPU threads' queues, each started by its first submission.
#[derive(Default)]
pub(super) struct Gpu {
    jobs: Mutex<HashMap<u64, SyncSender<Job>>>,
    backend: Arc<OnceLock<Option<gpu::SharedBackend>>>,
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

    fn extent(&self, addr: u64) -> Option<(u64, u64)> {
        self.mem.mapped_extent(addr)
    }

    fn host_pointer(&self, addr: u64) -> Option<*mut u8> {
        (addr < self.mem.size()).then(|| self.mem.host_base().wrapping_add(addr as usize))
    }

    fn display_buffer(&self, index: i32) -> Option<u64> {
        self.kernel.display_buffer(index)
    }

    fn generation(&self) -> u64 {
        self.mem.generation()
    }

    fn replaced_since(&self, since: u64) -> (u64, Option<Vec<(u64, u64)>>) {
        self.mem.replaced_since(since)
    }

    fn gpu_area(&self) -> Option<(u64, u64)> {
        Some(crate::dmem::GPU_AREA)
    }

    fn back(&self, start: u64, end: u64) -> bool {
        self.mem.back(start, end).is_ok()
    }
}

fn new_backend() -> Option<gpu::SharedBackend> {
    #[cfg(target_os = "macos")]
    if std::env::var_os("PS5SLOP_GPU_RENDER").is_none_or(|v| v != "0") {
        let metal = gpu::metal::Metal::new()?;
        return Some(gpu::SharedBackend(Arc::new(Mutex::new(Box::new(metal)))));
    }
    None
}

fn gpu_thread(host: Host, jobs: Receiver<Job>, backend: Arc<OnceLock<Option<gpu::SharedBackend>>>, name: String) {
    let mut processor = gpu::Processor::default();
    if let Some(b) = backend.get_or_init(new_backend) {
        processor.backend = Some(Box::new(b.clone()));
    }
    let report = std::env::var_os("PS5SLOP_GPU_STATS").is_some();
    let mut last_report = std::time::Instant::now();
    loop {
        let idle = gpu::timing::start();
        let Ok(job) = jobs.recv() else { break };
        gpu::timing::add(gpu::timing::Part::Idle, idle);
        gpu::with_pool(|| {
            for (addr, dwords) in job.buffers {
                if let Err(e) = processor.run(&host, addr, dwords) {
                    eprintln!("ps5slop: GPU: {e}");
                }
            }
            processor.drain(&host);
        });
        // Unverified: the driver signals each submission's completion; Minecraft
        // waits for it when its label poll loses the race.
        gpu::Host::interrupt(&host, 0);
        if report && last_report.elapsed().as_secs() >= 1 {
            last_report = std::time::Instant::now();
            let s = &processor.stats;
            eprintln!(
                "ps5slop: GPU {name}: {} submissions, {} packets, {} draws, {} dispatches, {} flips, {} interrupts, {} wait timeouts, {} waits met by pending releases, {} memory writes, longest flip gap {} ms",
                s.submissions,
                s.packets,
                s.draws,
                s.dispatches,
                s.flips,
                s.interrupts,
                s.timeouts,
                s.deferred_waits,
                s.opcodes.get(&0x37).unwrap_or(&0) + s.opcodes.get(&0x115).unwrap_or(&0),
                s.longest_flip_gap.as_millis()
            );
            if !processor.stats.long_flip_gaps.is_empty() {
                let gaps: Vec<u128> = processor.stats.long_flip_gaps.drain(..).map(|g| g.as_millis()).collect();
                eprintln!("ps5slop: GPU {name}: flip gaps over 1 s since the last report (ms): {gaps:?}");
            }
            if let Some(b) = processor.backend.as_ref().filter(|_| name == "graphics") {
                eprintln!("ps5slop: GPU: Metal: {}", b.summary());
            }
            if std::env::var_os("PS5SLOP_GPU_OPCODES").is_some() {
                eprintln!("ps5slop: GPU: packets by opcode: {:x?}", processor.stats.opcodes);
            }
        }
        if host.kernel.has_exited() {
            break;
        }
    }
}

fn enqueue(env: &Env, queue: u64, buffers: Vec<(u64, u32)>) {
    let gpu = &env.kernel.agc.gpu;
    let mut jobs = lock(&gpu.jobs);
    let sender = jobs.entry(queue).or_insert_with(|| {
        let (tx, rx) = sync_channel(MAX_QUEUED);
        let host = Host {
            kernel: env.kernel.this(),
            mem: env.kernel.memory(),
        };
        let backend = gpu.backend.clone();
        let name = if queue == GRAPHICS { "graphics".to_string() } else { format!("compute {queue:#x}") };
        std::thread::Builder::new()
            .name(format!("gpu {name}"))
            .spawn(move || gpu_thread(host, rx, backend, name))
            .expect("the GPU thread starts");
        tx
    });
    let sender = sender.clone();
    drop(jobs);
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
    enqueue(env, GRAPHICS, vec![buffer]);
    ret(0)
}

/// `sceAgcDriverSubmitAcb(owner, descriptor)`.
fn submit_acb(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffer) = descriptor(env, env.arg(1))? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, env.arg(0), vec![buffer]);
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
    enqueue(env, GRAPHICS, buffers);
    ret(0)
}

/// `sceAgcDriverSubmitMultiAcbs(owner, addresses, sizes, count)`.
fn submit_multi_acbs(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(buffers) = buffer_list(env, env.arg(1), env.arg(2), env.arg(3) as u32 as u64)? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    enqueue(env, env.arg(0), buffers);
    ret(0)
}

/// `sceAgcSuspendPoint()`: where a title may be suspended; nothing to do.
fn suspend_point(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}
