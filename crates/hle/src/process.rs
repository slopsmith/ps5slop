//! The running process: guest threads, each on its own host thread, sharing
//! one address space and one [`Kernel`].

use std::fmt;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

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
/// Where thread TLS blocks and stacks are allocated from (our choice).
const TLS_AREA: u64 = 0x7_0001_0000;
const STACK_AREA: u64 = 0x5_0000_0000;

/// What a guest thread did wrong, with its CPU state at that point.
#[derive(Debug)]
pub enum ThreadError {
    Guest {
        fault: CpuFault,
        cpu: Box<Cpu>,
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
            ThreadError::Guest { fault, cpu } => {
                write!(f, "guest fault {fault:?} at {:#x}\n{}", cpu.rip, dump(cpu))
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

/// The executable's TLS image, copied into every new thread's TLS block.
#[derive(Debug, Clone, Default)]
pub struct TlsTemplate {
    pub init: Vec<u8>,
    pub size: u64,
    pub align: u64,
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
}

impl Kernel {
    pub fn new(
        hle: Hle,
        mem: Memory,
        tls: TlsTemplate,
        stdout: Box<dyn Write + Send>,
        stderr: Box<dyn Write + Send>,
    ) -> Arc<Self> {
        Arc::new(Self {
            hle,
            mem: Mutex::new(mem),
            stdout: Mutex::new(stdout),
            stderr: Mutex::new(stderr),
            tls,
            exited: AtomicBool::new(false),
            result: Mutex::new(None),
            exit_signal: Condvar::new(),
        })
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
    }

    /// Makes a thread's static TLS block in the x86-64 "variant II" layout
    /// (FS points at a word holding its own address, the TLS image below it)
    /// and returns the thread pointer.
    pub fn new_tls(&self, mem: &Memory) -> Result<u64, ThreadError> {
        let align = self.tls.align.max(16);
        let offset = self.tls.size.next_multiple_of(align);
        let len = (offset + TCB_SIZE).next_multiple_of(PAGE_SIZE);
        let at = mem
            .find_free(TLS_AREA, len)
            .ok_or(ThreadError::Map(MapError::OutOfRange))?;
        mem.map(at, len, Prot::RW, "thread tls")?;
        let tcb = at + offset;
        mem.write(tcb - offset, &self.tls.init)
            .expect("tls was just mapped");
        mem.write_u64(tcb, tcb).expect("tls was just mapped");
        Ok(tcb)
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

    /// Runs the main thread on the calling host thread until the process
    /// exits, and returns how it ended.
    pub fn run_main(self: &Arc<Self>, cpu: Cpu) -> Result<i32, ThreadError> {
        let thread = Thread { tcb: cpu.fs_base };
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
    pub fn spawn(self: &Arc<Self>, cpu: Cpu) {
        let kernel = self.clone();
        let thread = Thread { tcb: cpu.fs_base };
        std::thread::spawn(move || {
            let mem = kernel.memory();
            kernel.run_thread(cpu, mem, thread);
        });
    }

    fn run_thread(&self, mut cpu: Cpu, mut mem: Memory, thread: Thread) {
        loop {
            if self.has_exited() {
                break;
            }
            match x86::run(&mut cpu, &mut mem, SLICE) {
                Exit::Continue | Exit::Limit => {}
                Exit::Trap(addr) => {
                    let mut env = Env {
                        cpu: &mut cpu,
                        mem: &mem,
                        kernel: self,
                        thread,
                    };
                    match self.hle.call(addr, &mut env) {
                        Ok(Outcome::Return(_) | Outcome::Resume) => {}
                        Ok(Outcome::Exit(status)) => self.exit(Ok(status)),
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
                    self.exit(Err(ThreadError::Guest {
                        fault: kind,
                        cpu: Box::new(cpu.clone()),
                    }));
                }
            }
        }
    }
}
