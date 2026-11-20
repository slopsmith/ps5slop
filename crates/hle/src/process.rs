//! The running process: guest threads, each on its own host thread, sharing
//! one address space and one [`Kernel`].

use std::fmt;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

use memory::{MapError, Memory, PAGE_SIZE, Prot};
use x86::cpu::reg::*;
use x86::{Cpu, CpuFault, Exit};

use crate::{CallError, Env, Hle, Outcome};

/// Instructions a thread runs between checks for process exit.
const SLICE: u64 = 1 << 16;

/// Room above a thread pointer for the thread control block.
const TCB_SIZE: u64 = 0x1000;
/// Where in its thread control block a thread's `errno` lives (our choice).
pub(crate) const ERRNO_OFFSET: u64 = 0x800;
/// Where in its thread control block a thread's values for the first
/// [`SPECIFIC_KEYS`] thread-specific keys live (our choice, read by the JIT's
/// native `pthread_getspecific`).
pub const SPECIFIC_OFFSET: u64 = 0x400;
pub const SPECIFIC_KEYS: u32 = 128;
/// Where thread TLS blocks and stacks are allocated from (our choice).
const TLS_AREA: u64 = 0x7_0001_0000;
const STACK_AREA: u64 = 0x5_0000_0000;

/// What a guest thread did wrong, with its CPU state at that point.
#[derive(Debug)]
pub enum ThreadError {
    Guest {
        fault: CpuFault,
        cpu: Box<Cpu>,
        /// What the faulting address is, when we know (an import's stub).
        note: Option<String>,
    },
    /// A host function failed; `caller` is the return address it was called with.
    Call {
        error: CallError,
        caller: u64,
        cpu: Box<Cpu>,
    },
    /// A raw `syscall` instruction; the kernel is only reached through libraries.
    Syscall {
        number: u64,
        cpu: Box<Cpu>,
    },
    Map(MapError),
}

impl fmt::Display for ThreadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThreadError::Guest { fault, cpu, note } => {
                write!(f, "guest fault {fault:?} at {:#x}", cpu.rip)?;
                if let Some(n) = note {
                    write!(f, " ({n})")?;
                }
                write!(f, "\n{}", dump(cpu))
            }
            ThreadError::Call { error, caller, cpu } => {
                write!(f, "{error}, called from {caller:#x}\n{}", dump(cpu))
            }
            ThreadError::Syscall { number, cpu } => write!(
                f,
                "raw syscall {number} at {:#x} is not supported\n{}",
                cpu.rip - 2,
                dump(cpu)
            ),
            ThreadError::Map(e) => write!(f, "cannot map: {e}"),
        }
    }
}

impl std::error::Error for ThreadError {}

impl From<MapError> for ThreadError {
    fn from(e: MapError) -> Self {
        ThreadError::Map(e)
    }
}

fn dump(cpu: &Cpu) -> String {
    const NAMES: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    let mut s = String::new();
    for (i, n) in NAMES.iter().enumerate() {
        let sep = if i % 4 == 3 { "\n" } else { "  " };
        s += &format!("{n:>3} {:016x}{sep}", cpu.regs[i]);
    }
    s + &format!("rip {:016x}  rflags {:08x}", cpu.rip, cpu.rflags)
}

/// One module's static TLS: its initial image, its size, and how far below
/// the thread pointer it sits.
#[derive(Debug, Clone, Default)]
pub struct TlsBlock {
    pub init: Vec<u8>,
    pub size: u64,
    pub offset: u64,
}

/// The static TLS of the executable and its modules, copied into every new
/// thread. Block `i` belongs to TLS module id `i + 1`; the executable is 1.
#[derive(Debug, Clone, Default)]
pub struct TlsTemplate {
    pub blocks: Vec<TlsBlock>,
}

impl TlsTemplate {
    /// Adds a module's TLS image (`size` bytes, aligned to `align`) below the
    /// blocks already there, and returns its TLS module id.
    pub fn add(&mut self, init: Vec<u8>, size: u64, align: u64) -> u64 {
        let below = self.blocks.last().map_or(0, |b| b.offset);
        let offset = (below + size).next_multiple_of(align.max(1));
        self.blocks.push(TlsBlock { init, size, offset });
        self.blocks.len() as u64
    }

    /// The distance from the lowest block to the thread pointer.
    fn span(&self) -> u64 {
        self.blocks
            .last()
            .map_or(0, |b| b.offset)
            .next_multiple_of(64)
    }
}

/// What a host function knows about the guest thread that called it.
#[derive(Debug, Clone, Copy)]
pub struct Thread {
    /// The thread pointer (FS base), which also serves as its `pthread_t`.
    pub tcb: u64,
}

impl Thread {
    pub fn errno_addr(&self) -> u64 {
        self.tcb + ERRNO_OFFSET
    }
}

/// State shared by all guest threads of a process.
pub struct Kernel {
    pub hle: Hle,
    /// Kept only to hand out clones; each thread uses its own handle.
    mem: Mutex<Memory>,
    stdout: Mutex<Box<dyn Write + Send>>,
    stderr: Mutex<Box<dyn Write + Send>>,
    tls: TlsTemplate,
    exited: AtomicBool,
    result: Mutex<Option<Result<i32, ThreadError>>>,
    exit_signal: Condvar,
    pub(crate) objects: crate::pthread::Objects,
    pub(crate) threads: crate::thread::Threads,
    /// A weak handle on this kernel, for starting threads from host functions.
    me: Weak<Kernel>,
    /// Guest threads still running; the process ends when the last one does.
    live: AtomicU64,
    pub(crate) heap: crate::heap::Heap,
    pub(crate) sems: crate::sem::Semaphores,
    pub(crate) dmem: crate::dmem::DirectMemory,
    pub(crate) equeues: crate::equeue::Queues,
    pub(crate) pad: crate::pad::Pad,
    pub(crate) audio: crate::audio::Audio,
    pub(crate) ngs2: crate::ngs2::Ngs2,
    pub(crate) videoout: crate::videoout::VideoOut,
    pub(crate) rtld: crate::rtld::Rtld,
    /// Whether guest code runs in the JIT instead of the interpreter.
    jit: AtomicBool,
    pub(crate) libc: crate::libc::State,
    /// The guest's file system.
    pub(crate) vfs: crate::Vfs,
    pub(crate) saves: crate::savedata::Saves,
    pub(crate) agc: crate::agc::Agc,
    pub(crate) started: std::time::Instant,
    /// Every thread pointer `new_tls` handed out.
    pub(crate) tcbs: Mutex<Vec<u64>>,
    /// Names guest code gave its threads, by thread pointer.
    pub(crate) names: Mutex<std::collections::HashMap<u64, String>>,
    running: Mutex<Vec<(u64, Arc<Whereabouts>)>>,
}

/// The last host function a thread called, for [`Kernel::threads`].
#[derive(Default)]
struct Whereabouts {
    stub: AtomicU64,
    caller: AtomicU64,
    calls: AtomicU64,
    done: AtomicBool,
}

/// A guest thread as [`Kernel::threads`] sees it.
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    pub tcb: u64,
    pub name: Option<String>,
    /// The host function it called last, and where from.
    pub stub: u64,
    pub caller: u64,
    pub calls: u64,
    pub running: bool,
}

impl Kernel {
    pub fn new(
        hle: Hle,
        mem: Memory,
        tls: TlsTemplate,
        stdout: Box<dyn Write + Send>,
        stderr: Box<dyn Write + Send>,
        vfs: crate::Vfs,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            hle,
            mem: Mutex::new(mem),
            stdout: Mutex::new(stdout),
            stderr: Mutex::new(stderr),
            tls,
            exited: AtomicBool::new(false),
            result: Mutex::new(None),
            exit_signal: Condvar::new(),
            objects: Default::default(),
            threads: Default::default(),
            me: me.clone(),
            live: AtomicU64::new(0),
            heap: Default::default(),
            sems: Default::default(),
            dmem: Default::default(),
            equeues: Default::default(),
            pad: Default::default(),
            audio: Default::default(),
            ngs2: Default::default(),
            videoout: Default::default(),
            rtld: Default::default(),
            jit: AtomicBool::new(false),
            libc: Default::default(),
            vfs,
            saves: Default::default(),
            agc: Default::default(),

            started: std::time::Instant::now(),
            tcbs: Mutex::default(),
            names: Mutex::default(),
            running: Mutex::default(),
        })
    }

    /// Every guest thread started so far.
    pub fn threads(&self) -> Vec<ThreadInfo> {
        let names = self.names.lock().unwrap_or_else(|e| e.into_inner());
        let running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        running
            .iter()
            .map(|(tcb, w)| ThreadInfo {
                tcb: *tcb,
                name: names.get(tcb).cloned(),
                stub: w.stub.load(Ordering::Relaxed),
                caller: w.caller.load(Ordering::Relaxed),
                calls: w.calls.load(Ordering::Relaxed),
                running: !w.done.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// A fresh handle on the address space, for a new thread.
    pub fn memory(&self) -> Memory {
        self.mem.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Writes guest output to stdout (fd 1) or stderr (fd 2).
    pub fn write_output(&self, fd: i32, data: &[u8]) -> std::io::Result<()> {
        let sink = match fd {
            1 => &self.stdout,
            2 => &self.stderr,
            _ => return Err(std::io::ErrorKind::InvalidInput.into()),
        };
        let mut sink = sink.lock().unwrap_or_else(|e| e.into_inner());
        sink.write_all(data)?;
        sink.flush()
    }

    pub fn has_exited(&self) -> bool {
        self.exited.load(Ordering::Acquire)
    }

    /// Ends the process; the first result wins.
    pub fn exit(&self, result: Result<i32, ThreadError>) {
        let mut slot = self.result.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(result);
        }
        self.exited.store(true, Ordering::Release);
        self.exit_signal.notify_all();
        if self.jit.load(Ordering::Relaxed) {
            jit::kick_all();
            jit::report(&self.memory());
            crate::report_calls();
        }
    }

    /// Runs guest code in the JIT (translated, inside a VM) instead of the
    /// interpreter. Set before the guest starts.
    pub fn use_jit(&self, on: bool) {
        self.jit.store(on, Ordering::Relaxed);
        if !on {
            return;
        }
        let natives = [
            ("memset", jit::Native::Memset),
            ("memcpy", jit::Native::Memmove),
            ("memmove", jit::Native::Memmove),
        ];
        for (name, native) in natives {
            if let Some(addr) = self.hle.bound(crate::libc::LIBRARY, name) {
                jit::set_native(addr, native);
            }
        }
        let specific = jit::Native::Getspecific {
            specific: SPECIFIC_OFFSET as u32,
            keys: SPECIFIC_KEYS,
        };
        let thread = [
            ("pthread_getspecific", specific),
            ("scePthreadGetspecific", specific),
            ("scePthreadGetthreadid", jit::Native::ThreadId),
            ("pthread_self", jit::Native::SelfThread),
            ("scePthreadSelf", jit::Native::SelfThread),
        ];
        for (name, native) in thread {
            for addr in self.hle.bound_anywhere(name) {
                jit::set_native(addr, native);
            }
        }
    }

    /// Guest instructions run so far: (interpreted, translated).
    pub fn instructions(&self) -> (u64, u64) {
        jit::instructions()
    }

    /// Runs guest code on this thread until it calls into the host, faults
    /// or the process exits.
    pub(crate) fn execute(&self, cpu: &mut Cpu, mem: &Memory) -> Exit {
        if self.jit.load(Ordering::Relaxed) {
            jit::run(cpu, mem, &|| self.has_exited())
        } else {
            jit::interpret(cpu, mem, SLICE)
        }
    }

    /// Makes a thread's static TLS block in the x86-64 "variant II" layout
    /// (FS points at a word holding its own address, the TLS image below it)
    /// and returns the thread pointer.
    pub fn new_tls(&self, mem: &Memory) -> Result<u64, ThreadError> {
        let offset = self.tls.span();
        let len = (offset + TCB_SIZE).next_multiple_of(PAGE_SIZE);
        let at = mem
            .find_free(TLS_AREA, len)
            .ok_or(ThreadError::Map(MapError::OutOfRange))?;
        mem.map(at, len, Prot::RW, "thread tls")?;
        let tcb = at + offset;
        for b in &self.tls.blocks {
            mem.write(tcb - b.offset, &b.init)
                .expect("tls was just mapped");
        }
        mem.write_u64(tcb, tcb).expect("tls was just mapped");
        self.tcbs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(tcb);
        Ok(tcb)
    }

    /// The address of a TLS variable on thread `tcb`: TLS module `module`,
    /// offset `offset`, for `__tls_get_addr`.
    pub fn tls_address(&self, tcb: u64, module: u64, offset: u64) -> Option<u64> {
        let b = self.tls.blocks.get(module.checked_sub(1)? as usize)?;
        Some(tcb - b.offset + offset)
    }

    /// Maps a stack of `size` bytes and returns its top.
    pub fn new_stack(&self, mem: &Memory, size: u64) -> Result<u64, ThreadError> {
        let size = size.next_multiple_of(PAGE_SIZE);
        let at = mem
            .find_free(STACK_AREA, size)
            .ok_or(ThreadError::Map(MapError::OutOfRange))?;
        mem.map(at, size, Prot::RW, "thread stack")?;
        Ok(at + size)
    }

    /// Runs each module initialiser in `inits`, then the main thread, on the
    /// calling host thread until the process exits, and returns how it ended.
    pub fn run_main(self: &Arc<Self>, mut cpu: Cpu, inits: &[u64]) -> Result<i32, ThreadError> {
        let thread = Thread { tcb: cpu.fs_base };
        self.live.fetch_add(1, Ordering::AcqRel);
        let mem = self.memory();
        for &init in inits {
            let mut env = Env {
                cpu: &mut cpu,
                mem: &mem,
                kernel: self,
                thread,
            };
            if let Err(error) = env.call_guest(init, &[0, 0, 0]) {
                let caller = init;
                let cpu = Box::new(cpu.clone());
                self.exit(Err(ThreadError::Call { error, caller, cpu }));
            }
        }
        self.run_thread(cpu, self.memory(), thread);
        let mut slot = self.result.lock().unwrap_or_else(|e| e.into_inner());
        while slot.is_none() {
            slot = self
                .exit_signal
                .wait(slot)
                .unwrap_or_else(|e| e.into_inner());
        }
        slot.take().expect("checked above")
    }

    /// Starts a guest thread on a new host thread.
    pub(crate) fn spawn_thread(&self, cpu: Cpu) {
        let kernel = self.this();
        let thread = Thread { tcb: cpu.fs_base };
        self.live.fetch_add(1, Ordering::AcqRel);
        let spawned = std::thread::Builder::new()
            .name(format!("guest {:#x}", thread.tcb))
            .spawn(move || {
                let mem = kernel.memory();
                kernel.run_thread(cpu, mem, thread);
            });
        if let Err(e) = spawned {
            panic!("cannot start a host thread: {e}");
        }
    }

    pub(crate) fn this(&self) -> Arc<Kernel> {
        self.me.upgrade().expect("the kernel outlives its threads")
    }

    fn run_thread(&self, mut cpu: Cpu, mem: Memory, thread: Thread) {
        let whereabouts = Arc::new(Whereabouts::default());
        self.running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((thread.tcb, whereabouts.clone()));
        loop {
            if self.has_exited() {
                break;
            }
            match self.execute(&mut cpu, &mem) {
                Exit::Continue | Exit::Limit => {}
                Exit::Trap(addr) => {
                    let caller = mem.read_u64(cpu.regs[RSP]).unwrap_or(0);
                    whereabouts.stub.store(addr, Ordering::Relaxed);
                    whereabouts.caller.store(caller, Ordering::Relaxed);
                    whereabouts.calls.fetch_add(1, Ordering::Relaxed);
                    if memory::watch::active() {
                        memory::watch::set_site(caller | memory::watch::HOST_CALL);
                    }
                    let mut env = Env {
                        cpu: &mut cpu,
                        mem: &mem,
                        kernel: self,
                        thread,
                    };
                    match self.hle.call(addr, &mut env) {
                        Ok(Outcome::Return(_) | Outcome::Resume) => {}
                        Ok(Outcome::Exit(status)) => self.exit(Ok(status)),
                        Ok(Outcome::ThreadExit) => break,
                        Err(error) => {
                            let caller = mem.read_u64(cpu.regs[RSP]).unwrap_or(0);
                            let cpu = Box::new(cpu.clone());
                            self.exit(Err(ThreadError::Call { error, caller, cpu }));
                        }
                    }
                }
                Exit::Syscall => {
                    let number = cpu.regs[RAX];
                    self.exit(Err(ThreadError::Syscall {
                        number,
                        cpu: Box::new(cpu.clone()),
                    }));
                }
                Exit::Fault { kind, .. } => {
                    let bytes = || {
                        let mut b = [0u8; 15];
                        mem.fetch(cpu.rip, &mut b).ok().map(|()| {
                            let hex: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
                            format!("bytes {}", hex.join(" "))
                        })
                    };
                    let mut note = match kind {
                        CpuFault::Memory(f) => self.hle.describe(f.addr).or_else(bytes),
                        _ => bytes(),
                    };
                    let recent = jit::recent_blocks();
                    if std::env::var_os("PS5SLOP_JIT_TRACE").is_some() && !recent.is_empty() {
                        let list: Vec<String> = recent.iter().map(|r| format!("{r:#x}")).collect();
                        note = Some(format!("{}; blocks before: {}", note.unwrap_or_default(), list.join(" ")));
                    }
                    self.exit(Err(ThreadError::Guest {
                        fault: kind,
                        cpu: Box::new(cpu.clone()),
                        note,
                    }));
                }
            }
        }
        whereabouts.done.store(true, Ordering::Relaxed);
        if self.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.exit(Ok(0));
        }
    }
}
