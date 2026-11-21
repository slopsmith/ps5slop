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
    /// The modules it needs, by name.
    pub needs: Vec<String>,
    /// Initialisers still to run when the title starts it with
    /// `sceKernelLoadStartModule`: its entry point or `DT_INIT`, then its
    /// init array. Empty once started.
    pub inits: Vec<u64>,
    /// What it exports, by NID (any library), with addresses.
    pub exports: Vec<(String, u64)>,
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
    /// Guest block behind `sceLibcHeapGetTraceInfo`'s pointers, or 0.
    heap_trace: Mutex<u64>,
}

pub static FUNCTIONS: &[Function] = &[
    f("_sceKernelSetThreadDtors", set_thread_dtors),
    f("_sceKernelSetThreadAtexitCount", set_atexit_count),
    f("_sceKernelSetThreadAtexitReport", set_atexit_report),
    f("sceKernelGetProcParam", get_proc_param),
    f("sceKernelGetModuleInfoForUnwind", module_info_for_unwind),
    f("sceKernelLoadStartModule", load_start_module),
    f("sceKernelDlsym", dlsym),
    f("_sceKernelRtldSetApplicationHeapAPI", set_heap_api),
    f("sceKernelGetSanitizerMallocReplace", no_sanitizer),
    f("sceKernelGetSanitizerMallocReplaceExternal", no_sanitizer),
    f("sceKernelGetSanitizerNewReplace", no_sanitizer),
    f("sceKernelGetSanitizerNewReplaceExternal", no_sanitizer),
    f("sceKernelIsAddressSanitizerEnabled", no_sanitizer),
];

/// Functions of libSceLibcInternalExt that Sony's libc calls while it starts.
pub static LIBC_EXT: &[Function] = &[Function {
    library: "libSceLibcInternalExt",
    name: "sceLibcHeapGetTraceInfo",
    func: heap_trace_info,
}];

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

/// `sceLibcHeapGetTraceInfo(info)`: heap tracing is off. Fills in the
/// layout shadPS4 documents: `u32` segment info at 0xc, then pointers to an
/// atomic id mask (0x10) and a 64-entry state table (0x18), all zero. Sony's
/// libc writes through both pointers, so they must point at real memory.
fn heap_trace_info(env: &mut Env) -> Result<Outcome, CallError> {
    const MASK_AND_TABLE: u64 = 8 + 64 * 8;
    let info = env.arg(0);
    let block = {
        let mut slot = lock(&env.kernel.rtld.heap_trace);
        if *slot == 0 {
            let at = env
                .kernel
                .heap
                .alloc(env.mem, MASK_AND_TABLE)
                .ok_or(CallError::Abort("out of memory for the heap trace table"))?;
            env.mem.write(at, &[0; MASK_AND_TABLE as usize])?;
            *slot = at;
        }
        *slot
    };
    env.mem.write(info + 0xc, &0u32.to_le_bytes())?;
    env.mem.write(info + 0x10, &block.to_le_bytes())?;
    env.mem.write(info + 0x18, &(block + 8).to_le_bytes())?;
    Ok(Outcome::Return(0))
}

fn get_proc_param(env: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(*lock(&env.kernel.rtld.proc_param)))
}

/// `sceKernelGetModuleInfoForUnwind(addr, flags, info)`, also
/// `sceSysmoduleGetModuleInfoForUnwind`: the unwind tables of the module
/// holding `addr`, in shadPS4's 0x130-byte layout (PS4).
pub(crate) fn module_info_for_unwind(env: &mut Env) -> Result<Outcome, CallError> {
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
/// handle of the loaded module with that file name. Modules are loaded and
/// linked at start; one the executable does not need is started here, after
/// the modules it needs. Its first initialiser gets `(args, argp, 0)`, as
/// Superliminal's plugins expect, and its result goes to `result`.
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
    let (args, argp, result) = (env.arg(1), env.arg(2), env.arg(5));
    let status = start_module(env, handle, args, argp)?;
    if result != 0 {
        env.mem.write(result, &(status as u32).to_le_bytes())?;
    }
    Ok(Outcome::Return(handle as u64))
}

/// Runs module `handle`'s initialisers if it has not started, after those
/// of the modules it needs, and returns what its first initialiser returned.
fn start_module(env: &mut Env, handle: usize, args: u64, argp: u64) -> Result<u64, CallError> {
    let (needs, inits) = {
        let mut modules = lock(&env.kernel.rtld.modules);
        let m = &mut modules[handle];
        (m.needs.clone(), std::mem::take(&mut m.inits))
    };
    if inits.is_empty() {
        return Ok(0);
    }
    for need in needs {
        let found = lock(&env.kernel.rtld.modules)
            .iter()
            .position(|m| m.name == need);
        if let Some(k) = found {
            start_module(env, k, 0, 0)?;
        }
    }
    let mut status = 0;
    for (k, init) in inits.into_iter().enumerate() {
        let r = env.call_guest(init, &[args, argp, 0])?;
        if k == 0 {
            status = r;
        }
    }
    Ok(status)
}

/// `sceKernelDlsym(handle, name, *addr)`: the address of the symbol `name`
/// exported by module `handle`, found by its NID.
fn dlsym(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, out) = (env.arg(0) as usize, env.arg(2));
    let name = crate::libc::cstr(env, env.arg(1))?;
    let nid = loader::nid::nid(&String::from_utf8_lossy(&name));
    let found = lock(&env.kernel.rtld.modules).get(handle).and_then(|m| {
        m.exports
            .iter()
            .find(|(n, _)| *n == nid)
            .map(|&(_, addr)| addr)
    });
    let Some(addr) = found else {
        return Ok(Outcome::Return(crate::dmem::sce_error(crate::errno::ESRCH)));
    };
    env.mem.write_u64(out, addr)?;
    Ok(Outcome::Return(0))
}

impl crate::Kernel {
    /// Records a loaded module (or the executable) for the loader functions.
    pub fn add_module(&self, info: ModuleInfo) {
        lock(&self.rtld.modules).push(info);
    }

    /// The loaded module (or executable) whose image holds `addr`, as `(start, end)`.
    pub fn module_image(&self, addr: u64) -> Option<(u64, u64)> {
        lock(&self.rtld.modules)
            .iter()
            .find(|m| m.start <= addr && addr < m.end)
            .map(|m| (m.start, m.end))
    }

    /// Records where the executable's process parameters are.
    pub fn set_proc_param(&self, addr: u64) {
        *lock(&self.rtld.proc_param) = addr;
    }
}

savestate::record!(ModuleInfo {
    name,
    file,
    start,
    end,
    text,
    eh_frame_hdr,
    eh_frame,
    needs,
    inits,
    exports
});

impl Rtld {
    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        e.put(&*lock(&self.modules))
            .put(&*lock(&self.thread_dtors))
            .put(&*lock(&self.atexit_count))
            .put(&*lock(&self.atexit_report))
            .put(&*lock(&self.heap_api))
            .put(&*lock(&self.proc_param))
            .put(&*lock(&self.heap_trace));
    }

    pub(crate) fn load(&self, d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
        *lock(&self.modules) = d.get()?;
        *lock(&self.thread_dtors) = d.get()?;
        *lock(&self.atexit_count) = d.get()?;
        *lock(&self.atexit_report) = d.get()?;
        *lock(&self.heap_api) = d.get()?;
        *lock(&self.proc_param) = d.get()?;
        *lock(&self.heap_trace) = d.get()?;
        Ok(())
    }
}
