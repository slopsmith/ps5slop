//! libSceGnmDriver, the PS4's GPU driver, for now only as far as a title's
//! frame loop needs it. Submitted command buffers run in order on a GPU
//! thread through the `gpu` crate's command processor with no renderer:
//! labels, waits and end-of-pipe interrupts take effect, draws are skipped,
//! and a submit-and-flip flips once its buffers have run. The driver calls
//! that build packets fill their space with a NOP. See `docs/gpu.md`, "PS4
//! titles".

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};

use crate::pthread::lock;
use crate::{CallError, Env, Function, Kernel, Outcome};

/// `EVFILT_GRAPHICS_CORE`, as for AGC.
const GRAPHICS_FILTER: i16 = -14;
/// The event id Deltarune registers with `sceGnmAddEqEvent` and waits on
/// after each submission; we raise it for every end-of-pipe interrupt.
const END_OF_PIPE: u64 = 0x40;
/// Submissions waiting for the GPU thread before a submitter blocks.
const MAX_QUEUED: usize = 1;
const IT_NOP: u32 = 0x10;
/// A PM4 packet's length field holds at most this many dwords.
const MAX_PACKET: u32 = 0x3FFF + 2;

/// Command buffers to run, then the flip to make, if any.
struct Job {
    buffers: Vec<(u64, u32)>,
    flip: Option<(i32, u64)>,
}

/// The GPU thread's queue, started by the first submission.
#[derive(Default)]
pub struct Gnm {
    jobs: Mutex<Option<SyncSender<Job>>>,
}

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceGnmDriver",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

pub static FUNCTIONS: &[Function] = &[
    f("sceGnmSubmitDone", ok),
    f("sceGnmAddEqEvent", add_eq_event),
    f("sceGnmDeleteEqEvent", delete_eq_event),
    f("sceGnmSubmitCommandBuffers", submit),
    f("sceGnmSubmitAndFlipCommandBuffers", submit_and_flip),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f("sceGnmDrawInitDefaultHardwareState350", ok),
    f("sceGnmInsertWaitFlipDone", nop),
    f("sceGnmSetVsShader", nop),
    f("sceGnmUpdateVsShader", nop),
    f("sceGnmSetPsShader350", nop),
    f("sceGnmUpdatePsShader350", nop),
    f("sceGnmDrawIndex", nop),
];

/// `(buffer, dwords, ...)`: one NOP over the space the title reserved.
fn nop(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, dwords) = (env.arg(0), env.arg(1) as u32);
    if at != 0 && (2..=MAX_PACKET).contains(&dwords) {
        let header = 3 << 30 | (dwords - 2) << 16 | IT_NOP << 8;
        env.mem.write(at, &header.to_le_bytes())?;
    }
    ret(0)
}

/// `sceGnmAddEqEvent(queue, id, udata)`.
fn add_eq_event(env: &mut Env) -> Result<Outcome, CallError> {
    let added = env
        .kernel
        .add_event(env.arg(0), env.arg(1), GRAPHICS_FILTER, env.arg(2));
    ret(if added {
        0
    } else {
        crate::dmem::sce_error(crate::errno::EBADF)
    })
}

/// `sceGnmDeleteEqEvent(queue, id)`.
fn delete_eq_event(env: &mut Env) -> Result<Outcome, CallError> {
    let removed = env
        .kernel
        .remove_event(env.arg(0), env.arg(1), GRAPHICS_FILTER);
    ret(if removed {
        0
    } else {
        crate::dmem::sce_error(crate::errno::ENOENT)
    })
}

/// The draw command buffers of `(count, addresses, byte_sizes, ...)`. The
/// constant command buffers that may come with them are not run.
fn buffers(env: &Env) -> Result<Vec<(u64, u32)>, CallError> {
    let (count, addrs, sizes) = (env.arg(0) as u32, env.arg(1), env.arg(2));
    let mut out = Vec::new();
    for i in 0..u64::from(count.min(16)) {
        let addr = env.mem.read_u64(addrs + 8 * i)?;
        let mut size = [0u8; 4];
        env.mem.read(sizes + 4 * i, &mut size)?;
        out.push((addr, u32::from_le_bytes(size) / 4));
    }
    Ok(out)
}

/// `sceGnmSubmitCommandBuffers(count, dcbs, dcb_sizes, ccbs, ccb_sizes)`.
fn submit(env: &mut Env) -> Result<Outcome, CallError> {
    let buffers = buffers(env)?;
    enqueue(
        env,
        Job {
            buffers,
            flip: None,
        },
    );
    ret(0)
}

/// `sceGnmSubmitAndFlipCommandBuffers(count, dcbs, dcb_sizes, ccbs,
/// ccb_sizes, video_out, index, flip_mode, flip_arg)`.
fn submit_and_flip(env: &mut Env) -> Result<Outcome, CallError> {
    let buffers = buffers(env)?;
    let flip = Some((env.arg_n(6)? as i32, env.arg_n(8)?));
    enqueue(env, Job { buffers, flip });
    ret(0)
}

fn enqueue(env: &Env, job: Job) {
    let mut jobs = lock(&env.kernel.gnm.jobs);
    let sender = jobs.get_or_insert_with(|| {
        let (tx, rx) = sync_channel(MAX_QUEUED);
        let host = Host {
            kernel: env.kernel.this(),
            mem: env.kernel.memory(),
        };
        std::thread::Builder::new()
            .name("gnm".into())
            .spawn(move || gpu_thread(host, rx))
            .expect("the GPU thread starts");
        tx
    });
    let sender = sender.clone();
    drop(jobs);
    let _ = sender.send(job);
}

fn gpu_thread(host: Host, jobs: Receiver<Job>) {
    let mut processor = gpu::Processor::default();
    while let Ok(job) = jobs.recv() {
        for (addr, dwords) in job.buffers {
            if let Err(e) = processor.run(&host, addr, dwords) {
                eprintln!("ps5slop: GPU: {e}");
            }
        }
        processor.drain(&host);
        if let Some((index, arg)) = job.flip {
            host.kernel.flip(index, arg);
        }
        if host.kernel.has_exited() {
            break;
        }
    }
}

/// The guest memory and events, as the command processor sees them.
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

    fn interrupt(&self, _context: u32) {
        self.kernel.trigger_event(END_OF_PIPE, GRAPHICS_FILTER, 0);
    }

    fn flip(&self, flip: gpu::Flip) {
        self.kernel.flip(flip.index, flip.arg as u64);
    }

    fn exiting(&self) -> bool {
        self.kernel.has_exited()
    }

    fn display_buffer(&self, index: i32) -> Option<u64> {
        self.kernel.display_buffer(index)
    }
}

impl Gnm {
    /// Whether the PS4 command path has run, which save states do not cover.
    pub(crate) fn in_use(&self) -> bool {
        lock(&self.jobs).is_some()
    }
}
