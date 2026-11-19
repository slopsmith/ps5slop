//! libkernel's run-time linker interface: what loaded modules (Sony's libc
//! first) ask the loader about while they start.

use std::sync::Mutex;

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

/// What the loader knows about a loaded module (or the executable).
#[derive(Debug, Clone, Default)]
pub struct ModuleInfo {
    pub name: String,
    /// The file it was loaded from (its name only).
    pub file: String,
    pub start: u64,
    pub end: u64,
    /// The first segment (code): address and size.
    pub text: (u64, u64),
    /// `.eh_frame_hdr`, and `.eh_frame` with its size, if the module has them.
    pub eh_frame_hdr: u64,
    pub eh_frame: (u64, u64),
}

/// What modules registered with the loader.
#[derive(Default)]
pub struct Rtld {
    modules: Mutex<Vec<ModuleInfo>>,
    /// Thread-exit callbacks libc registers (TLS destructors, the atexit
    /// count and report hooks); stored, not called yet.
    thread_dtors: Mutex<Option<u64>>,
    atexit_count: Mutex<Option<u64>>,
    atexit_report: Mutex<Option<u64>>,
    /// The heap function table libc registers (`_sceKernelRtldSetApplicationHeapAPI`).
    heap_api: Mutex<Option<u64>>,
    /// The executable's process parameters (`PT_SCE_PROCPARAM`), or 0.
    proc_param: Mutex<u64>,
}

pub static FUNCTIONS: &[Function] = &[
    f("_sceKernelSetThreadDtors", set_thread_dtors),
    f("_sceKernelSetThreadAtexitCount", set_atexit_count),
    f("_sceKernelSetThreadAtexitReport", set_atexit_report),
    f("sceKernelGetProcParam", get_proc_param),
    f("sceKernelGetModuleInfoForUnwind", module_info_for_unwind),
    f("sceKernelLoadStartModule", load_start_module),
    f("_sceKernelRtldSetApplicationHeapAPI", set_heap_api),
    f("sceKernelGetSanitizerMallocReplace", no_sanitizer),
    f("sceKernelGetSanitizerMallocReplaceExternal", no_sanitizer),
    f("sceKernelGetSanitizerNewReplace", no_sanitizer),
    f("sceKernelGetSanitizerNewReplaceExternal", no_sanitizer),
    f("sceKernelIsAddressSanitizerEnabled", no_sanitizer),
    Function {
        library: "libSceLibcInternalExt",
        name: "sceLibcHeapGetTraceInfo",
        func: heap_trace_info,
    },
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

fn set_thread_dtors(env: &mut Env) -> Result<Outcome, CallError> {
    *lock(&env.kernel.rtld.thread_dtors) = Some(env.arg(0));
    Ok(Outcome::Return(0))
}

fn set_atexit_count(env: &mut Env) -> Result<Outcome, CallError> {
    *lock(&env.kernel.rtld.atexit_count) = Some(env.arg(0));
    Ok(Outcome::Return(0))
}

fn set_atexit_report(env: &mut Env) -> Result<Outcome, CallError> {
    *lock(&env.kernel.rtld.atexit_report) = Some(env.arg(0));
    Ok(Outcome::Return(0))
}

fn set_heap_api(env: &mut Env) -> Result<Outcome, CallError> {
    *lock(&env.kernel.rtld.heap_api) = Some(env.arg(0));
    Ok(Outcome::Return(0))
}

/// The sanitizer hooks: no AddressSanitizer replaces `malloc` or `new`.
fn no_sanitizer(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

/// `sceLibcHeapGetTraceInfo(info)`: heap tracing is off; reports success
/// without writing (the structure's layout is unknown).
fn heap_trace_info(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(0))
}

fn get_proc_param(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(*lock(&env.kernel.rtld.proc_param)))
}

/// `sceKernelGetModuleInfoForUnwind(addr, flags, info)`: the unwind tables
/// of the module holding `addr`, in shadPS4's 0x130-byte layout (PS4).
fn module_info_for_unwind(env: &mut Env) -> Result<Outcome, CallError> {
    let (addr, info) = (env.arg(0), env.arg(2));
    let found = lock(&env.kernel.rtld.modules)
        .iter()
        .find(|m| (m.start..m.end).contains(&addr))
        .cloned();
    let Some(m) = found else {
        return Ok(Outcome::Return(crate::dmem::sce_error(crate::errno::ESRCH)));
    };
    let mut b = vec![0u8; 0x130];
    b[0..8].copy_from_slice(&0x130u64.to_le_bytes());
    let name = m.name.as_bytes();
    let n = name.len().min(255);
    b[8..8 + n].copy_from_slice(&name[..n]);
    for (k, v) in [
        m.eh_frame_hdr,
        m.eh_frame.0,
        m.eh_frame.1,
        m.text.0,
        m.text.1,
    ]
    .iter()
    .enumerate()
    {
        let at = 0x108 + 8 * k;
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    env.mem.write(info, &b)?;
    Ok(Outcome::Return(0))
}

/// `sceKernelLoadStartModule(path, args, argp, flags, opt, result)`: the
/// handle of the loaded module with that file name (modules load at start).
fn load_start_module(env: &mut Env) -> Result<Outcome, CallError> {
    let path = crate::libc::cstr(env, env.arg(0))?;
    let file = path
        .rsplit(|&c| c == b'/')
        .next()
        .unwrap_or(&path)
        .to_ascii_lowercase();
    let found = lock(&env.kernel.rtld.modules)
        .iter()
        .position(|m| m.file.to_ascii_lowercase().as_bytes() == file.as_slice());
    let Some(handle) = found else {
        return Ok(Outcome::Return(crate::dmem::sce_error(
            crate::errno::ENOENT,
        )));
    };
    if env.arg(5) != 0 {
        env.mem.write(env.arg(5), &0u32.to_le_bytes())?;
    }
    Ok(Outcome::Return(handle as u64))
}

impl crate::Kernel {
    /// Records a loaded module (or the executable) for the loader functions.
    pub fn add_module(&self, info: ModuleInfo) {
        lock(&self.rtld.modules).push(info);
    }

    /// Records where the executable's process parameters are.
    pub fn set_proc_param(&self, addr: u64) {
        *lock(&self.rtld.proc_param) = addr;
    }
}
