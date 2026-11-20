//! Asynchronous paging reads (APR): libSceAmpr builds command buffers of
//! file reads, and libkernel's `sceKernelApr*` resolves paths to file ids and
//! runs the buffers. Names were found by hashing guesses; the argument order
//! is read off Superliminal's calls, so it is unverified. Commands are kept
//! on the host, by the guest address of the command buffer object (whose
//! layout we never touch), and run synchronously at submission.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::dmem::sce_error;
use crate::errno::{EINVAL, ENOENT, ESRCH};
use crate::pthread::lock;
use crate::vfs::{ImageFile, Target};
use crate::{CallError, Env, Function, Outcome};

/// What an unresolved path gets as its id (SharpProspero's docs).
const FILE_ID_INVALID: u32 = 0xFFFF_FFFF;
/// Most paths one resolve call takes (SharpProspero's docs).
const RESOLVE_MAX: u64 = 1024;

enum AprFile {
    Host(File),
    Image(Arc<dyn ImageFile>),
}

impl AprFile {
    fn len(&self) -> u64 {
        match self {
            AprFile::Host(f) => f.metadata().map_or(0, |m| m.len()),
            AprFile::Image(f) => f.len(),
        }
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            AprFile::Host(f) => f.read_at(buf, offset),
            AprFile::Image(f) => f.read_at(offset, buf),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Read {
    file: u32,
    dest: u64,
    len: u64,
    offset: u64,
}

#[derive(Default)]
pub struct Apr {
    /// Resolved files; a file's id is its index.
    files: Mutex<Vec<Arc<AprFile>>>,
    by_path: Mutex<HashMap<Vec<u8>, u32>>,
    /// Commands by command buffer address.
    buffers: Mutex<HashMap<u64, Vec<Read>>>,
    submissions: AtomicU64,
}

pub static FUNCTIONS: &[Function] = &[
    f("libkernel", "sceKernelAprResolveFilepathsToIdsAndFileSizes", resolve),
    f("libkernel", "sceKernelAprSubmitCommandBufferAndGetResult", submit),
    f("libkernel", "sceKernelAprWaitCommandBuffer", wait),
    f("libSceAmpr", "sceAmprCommandBufferConstructor", construct),
    f("libSceAmpr", "sceAmprAprCommandBufferConstructor", construct),
    f("libSceAmpr", "sceAmprCommandBufferSetBuffer", set_buffer),
    f("libSceAmpr", "sceAmprCommandBufferReset", reset),
    f("libSceAmpr", "sceAmprAprCommandBufferReadFile", read_file),
    f("libSceAmpr", "sceAmprCommandBufferDestructor", destruct),
    f("libSceAmpr", "sceAmprAprCommandBufferDestructor", destruct),
    f("libSceAmpr", "sceAmprCommandBufferClearBuffer", reset),
    f("libSceAmpr", "sceAmprCommandBufferGetNumCommands", num_commands),
    f("libSceAmpr", "sceAmprCommandBufferGetCurrentOffset", current_offset),
    f("libSceAmpr", "sceAmprCommandBufferNop", accept),
    f("libSceAmpr", "sceAmprCommandBufferSetMarker", accept),
    f("libSceAmpr", "sceAmprCommandBufferPushMarker", accept),
    f("libSceAmpr", "sceAmprCommandBufferPopMarker", accept),
    f("libSceAmpr", "sceAmprMeasureCommandSizeNop", command_size),
    f("libSceAmpr", "sceAmprMeasureCommandSizeReadFile", command_size),
    f("libSceAmpr", "sceAmprMeasureCommandSizeReadFileGather", command_size),
    f("libSceAmpr", "sceAmprMeasureCommandSizeReadFileScatter", command_size),
    f("libSceAmpr", "sceAmprMeasureCommandSizeReadFileGatherScatter", command_size),
    f("libSceAmpr", "sceAmprMeasureCommandSizeSetMarker", command_size),
    f("libSceAmpr", "sceAmprMeasureCommandSizePushMarker", command_size),
    f("libSceAmpr", "sceAmprMeasureCommandSizePopMarker", command_size),
];

/// What `sceAmprMeasureCommandSize*` report for any command (our choice:
/// commands live on the host, so the guest buffer only needs to look big
/// enough).
const COMMAND_SIZE: u64 = 64;

const fn f(library: &'static str, name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library,
        name,
        func,
    }
}

fn ret(errno: i32) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(if errno == 0 { 0 } else { sce_error(errno) }))
}

fn open(env: &Env, path: &[u8]) -> Option<AprFile> {
    match env.kernel.vfs.lookup(path)? {
        Target::Host(p, _) => File::open(p).ok().map(AprFile::Host),
        Target::Image(fs, p) => fs.open(&p).map(AprFile::Image),
    }
}

/// `(paths, count, ids, sizes, failed)`: 32-bit ids, 64-bit sizes; the
/// index of the first unresolved path goes to `failed`.
fn resolve(env: &mut Env) -> Result<Outcome, CallError> {
    let (paths, count, ids, sizes, failed) =
        (env.arg(0), env.arg(1), env.arg(2), env.arg(3), env.arg(4));
    if count == 0 || count > RESOLVE_MAX {
        return ret(EINVAL);
    }
    let mut missing = None;
    for k in 0..count {
        let path = crate::libc::cstr(env, env.mem.read_u64(paths + 8 * k)?)?;
        let known = lock(&env.kernel.apr.by_path).get(&path).copied();
        let found = match known {
            Some(id) => Some((id, lock(&env.kernel.apr.files)[id as usize].len())),
            None => open(env, &path).map(|file| {
                let len = file.len();
                let mut files = lock(&env.kernel.apr.files);
                let id = files.len() as u32;
                files.push(Arc::new(file));
                lock(&env.kernel.apr.by_path).insert(path.clone(), id);
                (id, len)
            }),
        };
        let (id, len) = found.unwrap_or((FILE_ID_INVALID, 0));
        if found.is_none() && missing.is_none() {
            missing = Some(k as u32);
        }
        if ids != 0 {
            env.mem.write(ids + 4 * k, &id.to_le_bytes())?;
        }
        if sizes != 0 {
            env.mem.write_u64(sizes + 8 * k, len)?;
        }
    }
    match missing {
        Some(k) => {
            if failed != 0 {
                env.mem.write(failed, &k.to_le_bytes())?;
            }
            ret(ENOENT)
        }
        None => ret(0),
    }
}

/// `sceAmprCommandBufferConstructor(buffer)` and the APR variant.
fn construct(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.apr.buffers).insert(env.arg(0), Vec::new());
    ret(0)
}

/// `sceAmprCommandBufferSetBuffer(buffer, memory, size)`: commands stay on
/// the host, so the memory is not used.
fn set_buffer(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn destruct(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.apr.buffers).remove(&env.arg(0));
    ret(0)
}

/// `sceAmprCommandBufferGetNumCommands(buffer)`: the commands added since
/// the buffer was made or cleared.
fn num_commands(env: &mut Env) -> Result<Outcome, CallError> {
    let n = lock(&env.kernel.apr.buffers)
        .get(&env.arg(0))
        .map_or(0, Vec::len);
    Ok(Outcome::Return(n as u64))
}

/// `sceAmprCommandBufferGetCurrentOffset(buffer)`: as if each command
/// took [`COMMAND_SIZE`] bytes.
fn current_offset(env: &mut Env) -> Result<Outcome, CallError> {
    let n = lock(&env.kernel.apr.buffers)
        .get(&env.arg(0))
        .map_or(0, Vec::len);
    Ok(Outcome::Return(n as u64 * COMMAND_SIZE))
}

/// Markers and no-ops: nothing to run.
fn accept(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

fn command_size(_: &mut Env) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(COMMAND_SIZE))
}

fn reset(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.apr.buffers).insert(env.arg(0), Vec::new());
    ret(0)
}

/// `(buffer, ?, ?, file, dest, len, offset)`; the offset is the first
/// stack argument. What the second and third are is unknown.
fn read_file(env: &mut Env) -> Result<Outcome, CallError> {
    let offset = env.mem.read_u64(env.cpu.regs[x86::cpu::reg::RSP] + 8)?;
    let read = Read {
        file: env.arg(3) as u32,
        dest: env.arg(4),
        len: env.arg(5),
        offset,
    };
    lock(&env.kernel.apr.buffers)
        .entry(env.arg(0))
        .or_default()
        .push(read);
    ret(0)
}

/// `(buffer, priority, result, id)`: runs the reads now, writes a zero
/// result (status and error offset, 32 bits each) and a submission id.
fn submit(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(reads) = lock(&env.kernel.apr.buffers).get(&env.arg(0)).cloned() else {
        return ret(ESRCH);
    };
    for r in reads {
        let file = lock(&env.kernel.apr.files).get(r.file as usize).cloned();
        let Some(file) = file else {
            return ret(EINVAL);
        };
        let mut buf = vec![0u8; r.len as usize];
        let mut done = 0;
        while done < buf.len() {
            match file.read_at(r.offset + done as u64, &mut buf[done..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => done += n,
            }
        }
        env.mem.write(r.dest, &buf[..done])?;
    }
    let (result, id) = (env.arg(2), env.arg(3));
    if result != 0 {
        env.mem.write(result, &[0; 8])?;
    }
    if id != 0 {
        let n = env.kernel.apr.submissions.fetch_add(1, Ordering::Relaxed) as u32;
        env.mem.write(id, &n.to_le_bytes())?;
    }
    ret(0)
}

/// `sceKernelAprWaitCommandBuffer(...)`: submissions finish before they
/// return, so there is nothing to wait for.
fn wait(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}
