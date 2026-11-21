//! libSceFiber: user-mode contexts that a thread runs and that switch to each
//! other. A fiber's state lives on the host, keyed by the address of the
//! guest's fiber object, whose bytes we never touch. Prototypes follow
//! shadPS4's PS4 notes (from memory, unverified for the PS5): the entry is
//! `entry(arg_on_initialize, arg_on_run_to)` and must never return.

use std::collections::HashMap;
use std::sync::Mutex;

use x86::Cpu;
use x86::cpu::reg::*;

use crate::pthread::lock;
use crate::{CallError, Env, Function, INTERNAL, Outcome};

/// `SCE_FIBER_ERROR_INVALID` and `SCE_FIBER_ERROR_PERMISSION` (shadPS4, unverified).
const ERROR_INVALID: u64 = 0x8059_0004;
const ERROR_PERMISSION: u64 = 0x8059_0005;

/// The internal function a fiber's entry returns to.
const FIBER_RETURN: &str = "fiber_return";

struct Fiber {
    entry: u64,
    arg_on_initialize: u64,
    stack_top: u64,
    /// Where it stopped (in a switch or a return to its thread), and where to
    /// write the argument it is resumed with. `None` before its first run.
    suspended: Option<(Box<Cpu>, u64)>,
    /// Running on some thread now.
    running: bool,
}

/// What a thread running fibers keeps: the fiber it runs, and the thread's
/// own context inside `sceFiberRun` with where to put the return argument.
struct Host {
    current: u64,
    thread: Box<Cpu>,
    arg_on_return: u64,
}

#[derive(Default)]
pub struct Fibers {
    fibers: Mutex<HashMap<u64, Fiber>>,
    /// By thread pointer, while a thread is inside `sceFiberRun`.
    hosts: Mutex<HashMap<u64, Host>>,
}

pub static FUNCTIONS: &[Function] = &[
    f("_sceFiberInitializeImpl", initialize),
    f("sceFiberFinalize", finalize),
    f("sceFiberRun", run),
    f("sceFiberSwitch", switch),
    f("sceFiberReturnToThread", return_to_thread),
    f("sceFiberGetSelf", get_self),
    Function {
        library: INTERNAL,
        name: FIBER_RETURN,
        func: fiber_return,
    },
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceFiber",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

/// `_sceFiberInitializeImpl(fiber, name, entry, arg_on_initialize,
/// context, context_size, options, ...)`: the context is the fiber's stack.
fn initialize(env: &mut Env) -> Result<Outcome, CallError> {
    let (fiber, entry, arg) = (env.arg(0), env.arg(2), env.arg(3));
    let (context, size) = (env.arg(4), env.arg(5));
    if fiber == 0 || entry == 0 || context == 0 || size == 0 {
        return ret(ERROR_INVALID);
    }
    let state = Fiber {
        entry,
        arg_on_initialize: arg,
        stack_top: context + size,
        suspended: None,
        running: false,
    };
    lock(&env.kernel.fibers.fibers).insert(fiber, state);
    ret(0)
}

fn finalize(env: &mut Env) -> Result<Outcome, CallError> {
    let mut fibers = lock(&env.kernel.fibers.fibers);
    match fibers.get(&env.arg(0)) {
        Some(f) if f.running => ret(ERROR_PERMISSION),
        Some(_) => {
            fibers.remove(&env.arg(0));
            ret(0)
        }
        None => ret(ERROR_INVALID),
    }
}

/// The caller's context as it will be after the host call returns 0.
fn after_return(env: &Env) -> Result<Box<Cpu>, CallError> {
    let mut cpu = Box::new(env.cpu.clone());
    let sp = cpu.regs[RSP];
    cpu.rip = env.mem.read_u64(sp)?;
    cpu.regs[RSP] = sp + 8;
    cpu.regs[RAX] = 0;
    Ok(cpu)
}

/// Puts `fiber` on the CPU, resumed with `arg`. The caller has checked that
/// it exists and is not running.
fn enter(
    env: &mut Env,
    fibers: &mut HashMap<u64, Fiber>,
    fiber: u64,
    arg: u64,
) -> Result<(), CallError> {
    let f = fibers.get_mut(&fiber).expect("checked by the caller");
    f.running = true;
    let fs_base = env.cpu.fs_base;
    match f.suspended.take() {
        Some((cpu, arg_on_run)) => {
            *env.cpu = *cpu;
            env.cpu.fs_base = fs_base;
            if arg_on_run != 0 {
                env.mem.write_u64(arg_on_run, arg)?;
            }
        }
        None => {
            let sp = (f.stack_top & !15) - 8;
            env.mem
                .write_u64(sp, env.kernel.hle.internal(FIBER_RETURN))?;
            let mut cpu = Cpu {
                rip: f.entry,
                fs_base,
                mxcsr: env.cpu.mxcsr,
                fcw: env.cpu.fcw,
                ..Cpu::default()
            };
            cpu.regs[RDI] = f.arg_on_initialize;
            cpu.regs[RSI] = arg;
            cpu.regs[RSP] = sp;
            *env.cpu = cpu;
        }
    }
    Ok(())
}

/// `sceFiberRun(fiber, arg_on_run_to, *arg_on_return)`: runs the fiber on
/// this thread until it returns to the thread.
fn run(env: &mut Env) -> Result<Outcome, CallError> {
    let (fiber, arg, arg_on_return) = (env.arg(0), env.arg(1), env.arg(2));
    let tcb = env.thread.tcb;
    if lock(&env.kernel.fibers.hosts).contains_key(&tcb) {
        return ret(ERROR_PERMISSION);
    }
    let mut fibers = lock(&env.kernel.fibers.fibers);
    match fibers.get(&fiber) {
        None => return ret(ERROR_INVALID),
        Some(f) if f.running => return ret(ERROR_PERMISSION),
        Some(_) => {}
    }
    let thread = after_return(env)?;
    enter(env, &mut fibers, fiber, arg)?;
    let host = Host {
        current: fiber,
        thread,
        arg_on_return,
    };
    lock(&env.kernel.fibers.hosts).insert(tcb, host);
    Ok(Outcome::Resume)
}

/// `sceFiberSwitch(fiber, arg_on_run_to, *arg_on_run)`: from a fiber, runs
/// another one; this one goes on when it is resumed.
fn switch(env: &mut Env) -> Result<Outcome, CallError> {
    let (fiber, arg, arg_on_run) = (env.arg(0), env.arg(1), env.arg(2));
    let tcb = env.thread.tcb;
    let mut hosts = lock(&env.kernel.fibers.hosts);
    let Some(host) = hosts.get_mut(&tcb) else {
        return ret(ERROR_PERMISSION);
    };
    let mut fibers = lock(&env.kernel.fibers.fibers);
    match fibers.get(&fiber) {
        None => return ret(ERROR_INVALID),
        Some(f) if f.running => return ret(ERROR_PERMISSION),
        Some(_) => {}
    }
    let me = after_return(env)?;
    if let Some(f) = fibers.get_mut(&host.current) {
        f.running = false;
        f.suspended = Some((me, arg_on_run));
    }
    host.current = fiber;
    enter(env, &mut fibers, fiber, arg)?;
    Ok(Outcome::Resume)
}

/// `sceFiberReturnToThread(arg_on_return, *arg_on_run)`: back to the
/// thread's `sceFiberRun`; this fiber goes on when it is resumed.
fn return_to_thread(env: &mut Env) -> Result<Outcome, CallError> {
    let (arg, arg_on_run) = (env.arg(0), env.arg(1));
    let tcb = env.thread.tcb;
    let Some(host) = lock(&env.kernel.fibers.hosts).remove(&tcb) else {
        return ret(ERROR_PERMISSION);
    };
    let me = after_return(env)?;
    if let Some(f) = lock(&env.kernel.fibers.fibers).get_mut(&host.current) {
        f.running = false;
        f.suspended = Some((me, arg_on_run));
    }
    let fs_base = env.cpu.fs_base;
    *env.cpu = *host.thread;
    env.cpu.fs_base = fs_base;
    if host.arg_on_return != 0 {
        env.mem.write_u64(host.arg_on_return, arg)?;
    }
    Ok(Outcome::Resume)
}

/// `sceFiberGetSelf(*fiber)`.
fn get_self(env: &mut Env) -> Result<Outcome, CallError> {
    let current = lock(&env.kernel.fibers.hosts)
        .get(&env.thread.tcb)
        .map(|h| h.current);
    match current {
        Some(fiber) => {
            env.mem.write_u64(env.arg(0), fiber)?;
            ret(0)
        }
        None => ret(ERROR_PERMISSION),
    }
}

/// A fiber's entry returned, which Orbis does not allow.
fn fiber_return(_: &mut Env) -> Result<Outcome, CallError> {
    Err(CallError::Abort("a fiber's entry function returned"))
}

savestate::record!(Fiber {
    entry,
    arg_on_initialize,
    stack_top,
    suspended,
    running
});
savestate::record!(Host {
    current,
    thread,
    arg_on_return
});

impl Fibers {
    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        e.put(&*lock(&self.fibers)).put(&*lock(&self.hosts));
    }

    pub(crate) fn load(&self, d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
        *lock(&self.fibers) = d.get()?;
        *lock(&self.hosts) = d.get()?;
        Ok(())
    }
}
