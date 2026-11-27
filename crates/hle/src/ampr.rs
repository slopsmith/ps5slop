//! libSceAmpr's command buffers: APR (asynchronous paging reads, file reads
//! that libkernel's `sceKernelApr*` resolves and runs) and AMM (address
//! space changes in an area of its own). Names were found by hashing
//! guesses; argument orders are read off titles' calls, so they are
//! unverified. Commands are kept on the host, by the guest address of the
//! command buffer object (whose layout we never touch), and run
//! synchronously at submission.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::dmem::sce_error;
use crate::errno::{EINVAL, ENOENT, ESRCH};
use crate::pthread::lock;
use crate::vfs::{ImageFile, Target};
use crate::{CallError, Env, Function, Outcome};
use memory::Prot;

/// What an unresolved path gets as its id (SharpProspero's docs).
const FILE_ID_INVALID: u32 = 0xFFFF_FFFF;
/// Most paths one resolve call takes (SharpProspero's docs).
const RESOLVE_MAX: u64 = 1024;

/// Where the AMM area and its multimap area sit, and their sizes: our
/// choice (the console's are unknown). GTA VI uses 0x6_1700_0000 bytes.
const AMM_AREA: (u64, u64) = (0x40_0000_0000, 64 << 30);
const AMM_MULTIMAP_AREA: (u64, u64) = (0x50_0000_0000, 64 << 30);

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
enum Command {
    Read {
        file: u32,
        dest: u64,
        len: u64,
        offset: u64,
    },
    /// Writes `value` (64 bits) to `addr` once the commands before it ran.
    WriteAddress {
        addr: u64,
        value: u64,
    },
    /// AMM: `[va, va + len)` becomes partially resident (sparse) memory.
    MapAsPrt {
        va: u64,
        len: u64,
    },
    /// AMM: memory of the AMM's own is mapped at `[va, va + len)`.
    Map {
        va: u64,
        len: u64,
    },
    /// AMM: `[va, va + len)` goes back to being partially resident.
    UnmapToPrt {
        va: u64,
        len: u64,
    },
    Signal {
        counter: u32,
        value: u64,
    },
    /// Holds the commands after it until `counter` reaches `value`.
    Wait {
        counter: u32,
        value: u64,
    },
}

#[derive(Default)]
pub struct Apr {
    /// Resolved files; a file's id is its index.
    files: Mutex<Vec<Arc<AprFile>>>,
    paths: Mutex<Vec<Vec<u8>>>,
    by_path: Mutex<HashMap<Vec<u8>, u32>>,
    /// Commands by command buffer address.
    buffers: Mutex<HashMap<u64, Vec<Command>>>,
    /// The guest memory each command buffer was given, by buffer address.
    bases: Mutex<HashMap<u64, u64>>,
    counters: Mutex<HashMap<u32, u64>>,
    /// The rest of submitted buffers that wait for a counter, oldest first.
    parked: Mutex<Vec<VecDeque<Command>>>,
    submissions: AtomicU64,
}

pub static FUNCTIONS: &[Function] = &[
    f(
        "libkernel",
        "sceKernelAprResolveFilepathsToIdsAndFileSizes",
        resolve,
    ),
    f(
        "libkernel",
        "sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes",
        resolve_with_prefix,
    ),
    f(
        "libkernel",
        "sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizesForEach",
        resolve_with_prefix_each,
    ),
    f(
        "libkernel",
        "sceKernelAprSubmitCommandBufferAndGetResult",
        submit_and_get_result,
    ),
    f("libkernel", "sceKernelAprSubmitCommandBuffer", submit),
    f("libkernel", "sceKernelAprWaitCommandBuffer", wait),
    f("libSceAmpr", "sceAmprCommandBufferConstructor", construct),
    f(
        "libSceAmpr",
        "sceAmprAprCommandBufferConstructor",
        construct,
    ),
    f("libSceAmpr", "sceAmprCommandBufferSetBuffer", set_buffer),
    f(
        "libSceAmpr",
        "sceAmprCommandBufferGetBufferBaseAddress",
        base_address,
    ),
    f(
        "libSceAmpr",
        "sceAmprAmmCommandBufferConstructor",
        construct,
    ),
    f("libSceAmpr", "sceAmprAmmCommandBufferDestructor", destruct),
    f("libSceAmpr", "sceAmprAmmCommandBufferMapAsPrt", map_as_prt),
    f("libSceAmpr", "#d3lKWPM6odc", amm_map),
    f("libSceAmpr", "#jK+yuYCI7MA", signal),
    f("libSceAmpr", "#cQb8Zr8Q0Y0", wait_counter),
    f(
        "libSceAmpr",
        "sceAmprAmmCommandBufferUnmapToPrt",
        unmap_to_prt,
    ),
    f("libSceAmpr", "sceAmprAmmSubmitCommandBuffer", amm_submit),
    f(
        "libSceAmpr",
        "sceAmprAmmGetVirtualAddressRanges",
        virtual_address_ranges,
    ),
    f("libSceAmpr", "#Q07J7XpvhrU", crate::dmem::allocate),
    f("libSceAmpr", "sceAmprCommandBufferReset", reset),
    f("libSceAmpr", "sceAmprAprCommandBufferReadFile", read_file),
    f("libSceAmpr", "#j0+3uJMxYJY", write_address),
    f("libSceAmpr", "sceAmprCommandBufferDestructor", destruct),
    f("libSceAmpr", "sceAmprAprCommandBufferDestructor", destruct),
    f("libSceAmpr", "sceAmprCommandBufferClearBuffer", reset),
    f(
        "libSceAmpr",
        "sceAmprCommandBufferGetNumCommands",
        num_commands,
    ),
    f(
        "libSceAmpr",
        "sceAmprCommandBufferGetCurrentOffset",
        current_offset,
    ),
    f("libSceAmpr", "sceAmprCommandBufferNop", accept),
    f("libSceAmpr", "sceAmprCommandBufferSetMarker", accept),
    f("libSceAmpr", "sceAmprCommandBufferPushMarker", accept),
    f("libSceAmpr", "sceAmprCommandBufferPopMarker", accept),
    f("libSceAmpr", "sceAmprMeasureCommandSizeNop", command_size),
    f(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeReadFile",
        command_size,
    ),
    f(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeReadFileGather",
        command_size,
    ),
    f(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeReadFileScatter",
        command_size,
    ),
    f(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeReadFileGatherScatter",
        command_size,
    ),
    f(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeSetMarker",
        command_size,
    ),
    f(
        "libSceAmpr",
        "sceAmprMeasureCommandSizePushMarker",
        command_size,
    ),
    f(
        "libSceAmpr",
        "sceAmprMeasureCommandSizePopMarker",
        command_size,
    ),
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
    Ok(Outcome::Return(if errno == 0 {
        0
    } else {
        sce_error(errno)
    }))
}

fn open(env: &Env, path: &[u8]) -> Option<AprFile> {
    open_in(&env.kernel.vfs, path)
}

fn open_in(vfs: &crate::Vfs, path: &[u8]) -> Option<AprFile> {
    match vfs.lookup(path)? {
        Target::Host(p, _) => File::open(p).ok().map(AprFile::Host),
        Target::Image(fs, p) => fs.open(&p).map(AprFile::Image),
    }
}

/// `(paths, count, ids, sizes, failed)`: 32-bit ids, 64-bit sizes; the
/// index of the first unresolved path goes to `failed`.
fn resolve(env: &mut Env) -> Result<Outcome, CallError> {
    let [paths, count, ids, sizes, failed] = [0, 1, 2, 3, 4].map(|n| env.arg(n));
    resolve_paths(env, &[], [paths, count, ids, sizes, failed, 0])
}

/// `(prefix, paths, count, ids, sizes, failed)`: as `resolve`, with
/// `prefix` put in front of each path. GTA V passes an empty prefix.
fn resolve_with_prefix(env: &mut Env) -> Result<Outcome, CallError> {
    let prefix = crate::libc::cstr(env, env.arg(0))?;
    let [paths, count, ids, sizes, failed] = [1, 2, 3, 4, 5].map(|n| env.arg(n));
    resolve_paths(env, &prefix, [paths, count, ids, sizes, failed, 0])
}

/// `(prefix, paths, count, ids, sizes, results)`: as `resolve_with_prefix`,
/// with a 32-bit status per path in `results` instead of the first failure.
fn resolve_with_prefix_each(env: &mut Env) -> Result<Outcome, CallError> {
    let prefix = crate::libc::cstr(env, env.arg(0))?;
    let [paths, count, ids, sizes, results] = [1, 2, 3, 4, 5].map(|n| env.arg(n));
    match resolve_paths(env, &prefix, [paths, count, ids, sizes, 0, results])? {
        Outcome::Return(r) if r == sce_error(EINVAL) => ret(EINVAL),
        _ => ret(0),
    }
}

fn resolve_paths(
    env: &Env,
    prefix: &[u8],
    [paths, count, ids, sizes, failed, results]: [u64; 6],
) -> Result<Outcome, CallError> {
    if count == 0 || count > RESOLVE_MAX {
        return ret(EINVAL);
    }
    let mut missing = None;
    for k in 0..count {
        let mut path = prefix.to_vec();
        path.extend(crate::libc::cstr(env, env.mem.read_u64(paths + 8 * k)?)?);
        let known = lock(&env.kernel.apr.by_path).get(&path).copied();
        let found = match known {
            Some(id) => Some((id, lock(&env.kernel.apr.files)[id as usize].len())),
            None => open(env, &path).map(|file| {
                let len = file.len();
                let mut files = lock(&env.kernel.apr.files);
                let id = files.len() as u32;
                files.push(Arc::new(file));
                lock(&env.kernel.apr.paths).push(path.clone());
                lock(&env.kernel.apr.by_path).insert(path.clone(), id);
                (id, len)
            }),
        };
        let (id, len) = found.unwrap_or((FILE_ID_INVALID, 0));
        tracing::debug!(
            "resolve {}: id {id:#x}, {len:#x} bytes",
            String::from_utf8_lossy(&path)
        );
        if found.is_none() && missing.is_none() {
            missing = Some(k as u32);
        }
        if ids != 0 {
            env.mem.write(ids + 4 * k, &id.to_le_bytes())?;
        }
        if sizes != 0 {
            env.mem.write_u64(sizes + 8 * k, len)?;
        }
        if results != 0 {
            let status = if found.is_some() {
                0
            } else {
                sce_error(ENOENT) as u32
            };
            env.mem.write(results + 4 * k, &status.to_le_bytes())?;
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
/// the host, so the memory is only remembered, for AMM submissions.
fn set_buffer(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.apr.bases).insert(env.arg(0), env.arg(1));
    ret(0)
}

fn base_address(env: &mut Env) -> Result<Outcome, CallError> {
    let base = lock(&env.kernel.apr.bases).get(&env.arg(0)).copied();
    Ok(Outcome::Return(base.unwrap_or(0)))
}

fn destruct(env: &mut Env) -> Result<Outcome, CallError> {
    lock(&env.kernel.apr.buffers).remove(&env.arg(0));
    lock(&env.kernel.apr.bases).remove(&env.arg(0));
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
    let read = Command::Read {
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

/// `#j0+3uJMxYJY` (name unknown) `(buffer, addr, value, ?)`: inferred from
/// GTA V's use, see docs/hle-kernel.md.
fn write_address(env: &mut Env) -> Result<Outcome, CallError> {
    let command = Command::WriteAddress {
        addr: env.arg(1),
        value: env.arg(2),
    };
    lock(&env.kernel.apr.buffers)
        .entry(env.arg(0))
        .or_default()
        .push(command);
    ret(0)
}

fn push(env: &Env, command: Command) -> Result<Outcome, CallError> {
    lock(&env.kernel.apr.buffers)
        .entry(env.arg(0))
        .or_default()
        .push(command);
    ret(0)
}

/// `(buffer, va, len, ...)`; later arguments unknown.
fn map_as_prt(env: &mut Env) -> Result<Outcome, CallError> {
    let (va, len) = (env.arg(1), env.arg(2));
    push(env, Command::MapAsPrt { va, len })
}

/// `#d3lKWPM6odc` (name unknown) `(buffer, va, len, ?, ?)`: GTA VI maps
/// pieces of its AMM area with it, passing no physical address.
fn amm_map(env: &mut Env) -> Result<Outcome, CallError> {
    let (va, len) = (env.arg(1), env.arg(2));
    push(env, Command::Map { va, len })
}

/// `#jK+yuYCI7MA` (name unknown) `(buffer, counter, ?, value, 0, 0)`: GTA VI
/// signals with it on AMM buffers what its APR buffers wait for.
fn signal(env: &mut Env) -> Result<Outcome, CallError> {
    let (counter, value) = (env.arg(1) as u32, env.arg(3));
    push(env, Command::Signal { counter, value })
}

/// `#cQb8Zr8Q0Y0` (name unknown) `(buffer, counter, ?, value, ?, ...)`; that
/// it waits until the counter is at least `value` is assumed.
fn wait_counter(env: &mut Env) -> Result<Outcome, CallError> {
    let (counter, value) = (env.arg(1) as u32, env.arg(3));
    push(env, Command::Wait { counter, value })
}

/// `(buffer, va, len, ...)`; later arguments unknown.
fn unmap_to_prt(env: &mut Env) -> Result<Outcome, CallError> {
    let (va, len) = (env.arg(1), env.arg(2));
    push(env, Command::UnmapToPrt { va, len })
}

/// `(memory, size, flags)`: the buffer is found by its memory, and `size`
/// is what `sceAmprCommandBufferGetCurrentOffset` returned.
fn amm_submit(env: &mut Env) -> Result<Outcome, CallError> {
    let (base, size) = (env.arg(0), env.arg(1));
    let buffer = lock(&env.kernel.apr.bases)
        .iter()
        .find(|&(_, &b)| b == base)
        .map(|(&buffer, _)| buffer);
    let Some(buffer) = buffer else {
        return ret(ESRCH);
    };
    let errno = run(env, buffer, (size / COMMAND_SIZE) as usize)?;
    ret(errno)
}

/// Four u64 out-pointers, assumed to be area start and end, then multimap
/// start and end; GTA VI reads only the first.
fn virtual_address_ranges(env: &mut Env) -> Result<Outcome, CallError> {
    for (area, name) in [
        (AMM_AREA, "AMM area"),
        (AMM_MULTIMAP_AREA, "AMM multimap area"),
    ] {
        if env.mem.describe_region(area.0).is_none() {
            env.mem
                .map_over(area.0, area.1, Prot::NONE, name)
                .map_err(|_| CallError::Abort("cannot reserve the AMM areas"))?;
        }
    }
    let values = [
        AMM_AREA.0,
        AMM_AREA.0 + AMM_AREA.1,
        AMM_MULTIMAP_AREA.0,
        AMM_MULTIMAP_AREA.0 + AMM_MULTIMAP_AREA.1,
    ];
    for (k, v) in values.into_iter().enumerate() {
        let out = env.arg(k);
        if out != 0 {
            env.mem.write_u64(out, v)?;
        }
    }
    ret(0)
}

/// `(buffer, priority, result, id)`: runs the commands now, writes a zero
/// result (status and error offset, 32 bits each) and a submission id.
fn submit_and_get_result(env: &mut Env) -> Result<Outcome, CallError> {
    let errno = run(env, env.arg(0), usize::MAX)?;
    if errno != 0 {
        return ret(errno);
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

/// `(buffer, priority)`: runs the commands now.
fn submit(env: &mut Env) -> Result<Outcome, CallError> {
    let errno = run(env, env.arg(0), usize::MAX)?;
    ret(errno)
}

/// Runs the first `count` commands of `buffer`, then any parked batch its
/// counter signals let go on.
fn run(env: &Env, buffer: u64, count: usize) -> Result<i32, CallError> {
    let Some(commands) = lock(&env.kernel.apr.buffers).get(&buffer).cloned() else {
        return Ok(ESRCH);
    };
    let errno = run_batch(env, commands.into_iter().take(count).collect())?;
    if errno != 0 {
        return Ok(errno);
    }
    loop {
        let parked = std::mem::take(&mut *lock(&env.kernel.apr.parked));
        let before = parked.len();
        for batch in parked {
            let errno = run_batch(env, batch)?;
            if errno != 0 {
                return Ok(errno);
            }
        }
        if lock(&env.kernel.apr.parked).len() >= before {
            return Ok(0);
        }
    }
}

/// Runs `batch` until it ends or waits for a counter that has not reached
/// its value; the rest is then parked.
fn run_batch(env: &Env, mut batch: VecDeque<Command>) -> Result<i32, CallError> {
    while let Some(c) = batch.pop_front() {
        match c {
            Command::Read {
                file,
                dest,
                len,
                offset,
            } => {
                if !read(env, file, dest, len, offset)? {
                    return Ok(EINVAL);
                }
            }
            Command::WriteAddress { addr, value } => env.mem.write_u64(addr, value)?,
            Command::MapAsPrt { va, len }
            | Command::Map { va, len }
            | Command::UnmapToPrt { va, len } => {
                if let Err(e) = amm_change(env, c, va, len) {
                    tracing::error!("AMM command {c:?} failed: {e:?}");
                    return Ok(EINVAL);
                }
            }
            Command::Signal { counter, value } => {
                lock(&env.kernel.apr.counters).insert(counter, value);
            }
            Command::Wait { counter, value } => {
                let now = lock(&env.kernel.apr.counters)
                    .get(&counter)
                    .copied()
                    .unwrap_or(0);
                if now < value {
                    batch.push_front(c);
                    lock(&env.kernel.apr.parked).push(batch);
                    return Ok(0);
                }
            }
        }
    }
    Ok(0)
}

/// Reads `len` bytes of `file` at `offset` to `dest`; false if there is no
/// such file.
fn read(env: &Env, file: u32, dest: u64, len: u64, offset: u64) -> Result<bool, CallError> {
    let id = file;
    let file = lock(&env.kernel.apr.files).get(file as usize).cloned();
    let Some(file) = file else {
        return Ok(false);
    };
    let mut buf = vec![0u8; len as usize];
    let mut done = 0;
    while done < buf.len() {
        match file.read_at(offset + done as u64, &mut buf[done..]) {
            Ok(0) | Err(_) => break,
            Ok(n) => done += n,
        }
    }
    if tracing::enabled!(tracing::Level::DEBUG) {
        let path = lock(&env.kernel.apr.paths)[id as usize].clone();
        tracing::debug!(
            "read {} at {offset:#x} len {len:#x} ({done:#x} read) to {dest:#x}",
            String::from_utf8_lossy(&path)
        );
    }
    env.mem.write(dest, &buf[..done])?;
    Ok(true)
}

/// Pieces keep their pages: fresh host pages for ranges this large make
/// every vCPU and GPU window map them again.
fn amm_change(env: &Env, c: Command, va: u64, len: u64) -> Result<(), memory::MapError> {
    let inside = |area: (u64, u64)| va >= area.0 && va + len <= area.0 + area.1;
    if !inside(AMM_AREA) && !inside(AMM_MULTIMAP_AREA) {
        return Err(memory::MapError::OutOfRange);
    }
    match c {
        Command::MapAsPrt { .. } => env.mem.map_over(va, len, Prot::RW, "AMM"),
        _ => env.mem.protect_range(va, len, Prot::RW),
    }
}

/// `sceKernelAprWaitCommandBuffer(...)`: submissions finish before they
/// return, so there is nothing to wait for.
fn wait(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

impl savestate::Encode for Command {
    fn encode(&self, e: &mut savestate::Encoder) {
        match *self {
            Command::Read {
                file,
                dest,
                len,
                offset,
            } => e.put(&0u8).put(&(file, dest, len, offset)),
            Command::WriteAddress { addr, value } => e.put(&1u8).put(&(addr, value)),
            Command::MapAsPrt { va, len } => e.put(&2u8).put(&(va, len)),
            Command::Map { va, len } => e.put(&3u8).put(&(va, len)),
            Command::UnmapToPrt { va, len } => e.put(&4u8).put(&(va, len)),
            Command::Signal { counter, value } => e.put(&5u8).put(&(counter, value)),
            Command::Wait { counter, value } => e.put(&6u8).put(&(counter, value)),
        };
    }
}

impl savestate::Decode for Command {
    fn decode(d: &mut savestate::Decoder) -> Result<Self, savestate::Error> {
        let tag = d.get::<u8>()?;
        Ok(match tag {
            0 => {
                let (file, dest, len, offset) = d.get()?;
                Command::Read {
                    file,
                    dest,
                    len,
                    offset,
                }
            }
            1 => {
                let (addr, value) = d.get()?;
                Command::WriteAddress { addr, value }
            }
            2..=4 => {
                let (va, len) = d.get()?;
                match tag {
                    2 => Command::MapAsPrt { va, len },
                    3 => Command::Map { va, len },
                    _ => Command::UnmapToPrt { va, len },
                }
            }
            5 | 6 => {
                let (counter, value) = d.get()?;
                if tag == 5 {
                    Command::Signal { counter, value }
                } else {
                    Command::Wait { counter, value }
                }
            }
            n => return Err(savestate::error(format!("bad APR command {n}"))),
        })
    }
}

impl Apr {
    pub(crate) fn save_amm(&self, e: &mut savestate::Encoder) {
        let parked: Vec<Vec<Command>> = lock(&self.parked)
            .iter()
            .map(|b| b.iter().copied().collect())
            .collect();
        e.put(&*lock(&self.bases))
            .put(&*lock(&self.counters))
            .put(&parked);
    }

    pub(crate) fn load_amm(&self, d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
        *lock(&self.bases) = d.get()?;
        *lock(&self.counters) = d.get()?;
        let parked: Vec<Vec<Command>> = d.get()?;
        *lock(&self.parked) = parked.into_iter().map(VecDeque::from).collect();
        Ok(())
    }

    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        e.put(&*lock(&self.paths))
            .put(&*lock(&self.buffers))
            .put(&self.submissions.load(Ordering::Relaxed));
    }

    /// Opens the resolved files again by path, keeping their ids.
    pub(crate) fn load(
        &self,
        d: &mut savestate::Decoder,
        vfs: &crate::Vfs,
    ) -> Result<(), savestate::Error> {
        let paths: Vec<Vec<u8>> = d.get()?;
        let mut files = Vec::new();
        let mut by_path = HashMap::new();
        for (id, path) in paths.iter().enumerate() {
            let file = open_in(vfs, path).ok_or_else(|| {
                savestate::error(format!(
                    "cannot open APR file {} again",
                    String::from_utf8_lossy(path)
                ))
            })?;
            files.push(Arc::new(file));
            by_path.insert(path.clone(), id as u32);
        }
        *lock(&self.files) = files;
        *lock(&self.paths) = paths;
        *lock(&self.by_path) = by_path;
        *lock(&self.buffers) = d.get()?;
        self.submissions.store(d.get()?, Ordering::Relaxed);
        Ok(())
    }
}
