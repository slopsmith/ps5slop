//! Exception handlers (`libkernel_unity`), the signals of this layer: a title
//! installs a handler for a signal number with
//! `sceKernelInstallExceptionHandler` and raises it on a thread with
//! `sceKernelRaiseException`. Unity's garbage collector uses them to stop
//! threads (judging by Superliminal's calls). Names were found by hashing
//! guesses; the handler's arguments are unverified. See `docs/hle-kernel.md`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use x86::cpu::reg::*;

use crate::dmem::sce_error;
use crate::errno::{EINVAL, ESRCH};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

/// Room for the context on the stack.
const CONTEXT_SIZE: u64 = 1024;
/// Where the register block starts in the context. Superliminal's handler
/// reads the stack pointer at 248, FreeBSD's `mc_rsp` (184) past 64.
const MCONTEXT: u64 = 64;
const RED_ZONE: u64 = 128;

#[derive(Default)]
pub struct Exceptions {
    /// Handler address by signal number.
    handlers: Mutex<HashMap<i32, u64>>,
    /// Raised and not yet delivered, by thread pointer.
    pending: Mutex<HashMap<u64, Vec<i32>>>,
    any_pending: AtomicBool,
}

pub static FUNCTIONS: &[Function] = &[
    f("sceKernelInstallExceptionHandler", install),
    f("sceKernelRaiseException", raise),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel_unity",
        name,
        func,
    }
}

fn ret(errno: i32) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(if errno == 0 {
        0
    } else {
        sce_error(errno)
    }))
}

/// `sceKernelInstallExceptionHandler(signal, handler)`.
fn install(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.exceptions.handlers).insert(env.arg(0) as i32, env.arg(1));
    ret(0)
}

/// `sceKernelRaiseException(thread, signal)`: queues the exception and
/// kicks running threads out of translated code so the target notices.
fn raise(env: &mut Env) -> Result<Outcome, CallError> {
    let (thread, signal) = (env.arg(0), env.arg(1) as i32);
    if !lock(&env.kernel.exceptions.handlers).contains_key(&signal) {
        return ret(EINVAL);
    }
    if !lock(&env.kernel.tcbs).contains(&thread) {
        return ret(ESRCH);
    }
    let e = &env.kernel.exceptions;
    lock(&e.pending).entry(thread).or_default().push(signal);
    e.any_pending.store(true, Ordering::Release);
    if thread == env.thread.tcb {
        deliver(env)?;
    } else {
        jit::kick_all();
    }
    ret(0)
}

/// Whether the calling thread has an exception waiting. A blocking host
/// function that sees this returns early where its contract allows, or
/// else returns [`Outcome::Resume`] untouched so that the call restarts
/// after the handler has run.
pub(crate) fn interrupted(env: &Env) -> bool {
    env.kernel.exception_pending(env.thread.tcb)
}

impl crate::Kernel {
    /// Whether thread `tcb` has an exception waiting to be delivered.
    pub(crate) fn exception_pending(&self, tcb: u64) -> bool {
        let e = &self.exceptions;
        e.any_pending.load(Ordering::Acquire)
            && lock(&e.pending).get(&tcb).is_some_and(|v| !v.is_empty())
    }
}

/// Runs the handlers of the calling thread's pending exceptions, oldest
/// first, and returns whether there were any.
pub(crate) fn deliver(env: &mut Env) -> Result<bool, CallError> {
    let mut ran = false;
    loop {
        let signal = {
            let e = &env.kernel.exceptions;
            let mut pending = lock(&e.pending);
            let next = pending
                .get_mut(&env.thread.tcb)
                .and_then(|v| (!v.is_empty()).then(|| v.remove(0)));
            if pending.values().all(Vec::is_empty) {
                e.any_pending.store(false, Ordering::Release);
            }
            next
        };
        let Some(signal) = signal else {
            return Ok(ran);
        };
        let handler = lock(&env.kernel.exceptions.handlers).get(&signal).copied();
        if let Some(handler) = handler {
            run_handler(env, handler, signal)?;
            ran = true;
        }
    }
}

fn run_handler(env: &mut Env, handler: u64, signal: i32) -> Result<(), CallError> {
    let rsp = env.cpu.regs[RSP];
    let context = (rsp - RED_ZONE - CONTEXT_SIZE) & !63;
    env.mem.write(context, &context_bytes(env.cpu))?;
    env.cpu.regs[RSP] = context;
    let result = env.call_guest(handler, &[signal as u64, context, context]);
    env.cpu.regs[RSP] = rsp;
    result.map(|_| ())
}

/// The registers of `cpu` in FreeBSD's amd64 `mcontext_t` layout at
/// [`MCONTEXT`] (no floating-point state).
fn context_bytes(cpu: &x86::Cpu) -> [u8; CONTEXT_SIZE as usize] {
    let mut b = [0u8; CONTEXT_SIZE as usize];
    let mut put = |at: u64, v: u64| {
        let at = (MCONTEXT + at) as usize;
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    let r = &cpu.regs;
    for (k, v) in [
        r[RDI], r[RSI], r[RDX], r[RCX], r[R8], r[R9], r[RAX], r[RBX], r[RBP], r[10], r[11], r[12],
        r[13], r[14], r[15],
    ]
    .into_iter()
    .enumerate()
    {
        put(8 + 8 * k as u64, v);
    }
    put(160, cpu.rip);
    put(176, cpu.rflags);
    put(184, r[RSP]);
    put(200, 800);
    put(736, cpu.fs_base);
    put(744, cpu.gs_base);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_puts_registers_at_freebsd_mcontext_offsets_past_64_bytes() {
        let mut cpu = x86::Cpu::default();
        cpu.regs[RDI] = 1;
        cpu.regs[15] = 15;
        cpu.regs[RSP] = 0x7000;
        cpu.rip = 0x40_1234;
        let b = context_bytes(&cpu);
        let at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        assert_eq!(
            (at(72), at(184), at(224), at(248)),
            (1, 15, 0x40_1234, 0x7000)
        );
    }
}
