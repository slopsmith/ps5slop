//! libkernel's file descriptors: the POSIX calls (`open`, `read`, `close`),
//! which fail with errno and -1, and their `sceKernel*` twins, which return
//! `0x80020000` + errno. Paths go through the guest's file system (see
//! `vfs.rs`). Flag values are FreeBSD's.

use std::fs::OpenOptions;

use crate::dmem::sce_error;
use crate::libc::cstr;
use crate::vfs::io_errno;
use crate::{CallError, Env, Function, Outcome, errno};

pub static FUNCTIONS: &[Function] = &[
    f("open", open),
    f("_open", open),
    f("read", read),
    f("_read", read),
    f("close", close),
    f("_close", close),
    f("sceKernelOpen", sce_open),
    f("sceKernelRead", sce_read),
    f("sceKernelWrite", sce_write),
    f("sceKernelClose", sce_close),
    f("sceKernelMkdir", sce_mkdir),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libkernel",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

const O_ACCMODE: u64 = 3;
const O_WRONLY: u64 = 1;
const O_RDWR: u64 = 2;
const O_APPEND: u64 = 0x8;
const O_CREAT: u64 = 0x200;
const O_TRUNC: u64 = 0x400;
const O_EXCL: u64 = 0x800;
/// Reads larger than this are cut short.
const MAX_READ: u64 = 1 << 24;

/// The two error conventions: errno and -1, or a `sceKernel` error code.
#[derive(Clone, Copy)]
enum Style {
    Posix,
    Sce,
}

fn finish(env: &Env, style: Style, r: Result<u64, i32>) -> Result<Outcome, CallError> {
    match (r, style) {
        (Ok(v), _) => ret(v),
        (Err(e), Style::Posix) => env.fail(e),
        (Err(e), Style::Sce) => ret(sce_error(e)),
    }
}

/// `open(path, flags, mode)`: files under `/app0` for reading, under
/// `/data` for anything.
fn open_file(env: &Env) -> Result<Result<u64, i32>, CallError> {
    let (path, flags) = (cstr(env, env.arg(0))?, env.arg(1));
    let Some((host, writable)) = env.kernel.vfs.resolve(&path) else {
        return Ok(Err(errno::ENOENT));
    };
    let access = flags & O_ACCMODE;
    let wants_write = access != 0 || flags & (O_CREAT | O_TRUNC | O_APPEND) != 0;
    if wants_write && !writable {
        return Ok(Err(errno::EROFS));
    }
    let mut o = OpenOptions::new();
    o.read(access != O_WRONLY)
        .write(matches!(access, O_WRONLY | O_RDWR))
        .append(flags & O_APPEND != 0)
        .truncate(flags & O_TRUNC != 0);
    if flags & O_CREAT != 0 {
        if flags & O_EXCL != 0 {
            o.create_new(true);
        } else {
            o.create(true);
        }
    }
    Ok(match o.open(host) {
        Ok(file) => Ok(env.kernel.libc.files.open_fd(file) as u64),
        Err(e) => Err(io_errno(&e)),
    })
}

/// `read(fd, buf, len)`: descriptor files, and stdin, which is always at
/// end of file.
fn read_file(env: &Env) -> Result<Result<u64, i32>, CallError> {
    let (fd, buf, len) = (env.arg(0) as i32, env.arg(1), env.arg(2).min(MAX_READ));
    if fd == 0 {
        return Ok(Ok(0));
    }
    let mut data = vec![0u8; len as usize];
    match env.kernel.libc.files.read_fd(fd, &mut data) {
        Some(Ok(n)) => {
            env.mem.write(buf, &data[..n])?;
            Ok(Ok(n as u64))
        }
        Some(Err(e)) => Ok(Err(e.raw_os_error().map_or(errno::EINVAL, host_errno))),
        None => Ok(Err(errno::EBADF)),
    }
}

/// The FreeBSD number for a host (macOS) errno; the two agree on the
/// numbers below 35 that reading a file can return.
fn host_errno(e: i32) -> i32 {
    if e < 35 { e } else { errno::EINVAL }
}

fn close_file(env: &Env) -> Result<u64, i32> {
    if env.kernel.libc.files.close_fd(env.arg(0) as i32) {
        Ok(0)
    } else {
        Err(errno::EBADF)
    }
}

fn open(env: &mut Env) -> Result<Outcome, CallError> {
    let r = open_file(env)?;
    finish(env, Style::Posix, r)
}

fn sce_open(env: &mut Env) -> Result<Outcome, CallError> {
    let r = open_file(env)?;
    finish(env, Style::Sce, r)
}

fn read(env: &mut Env) -> Result<Outcome, CallError> {
    let r = read_file(env)?;
    finish(env, Style::Posix, r)
}

fn sce_read(env: &mut Env) -> Result<Outcome, CallError> {
    let r = read_file(env)?;
    finish(env, Style::Sce, r)
}

fn close(env: &mut Env) -> Result<Outcome, CallError> {
    finish(env, Style::Posix, close_file(env))
}

fn sce_close(env: &mut Env) -> Result<Outcome, CallError> {
    finish(env, Style::Sce, close_file(env))
}

/// `write(fd, buf, len)`: fds 1 and 2 go to the host's stdout and stderr,
/// other descriptors to their files.
pub(crate) fn write_file(env: &Env) -> Result<Result<u64, i32>, CallError> {
    let (fd, buf, len) = (env.arg(0) as i32, env.arg(1), env.arg(2).min(MAX_READ));
    let mut data = vec![0u8; len as usize];
    env.mem.read(buf, &mut data)?;
    if matches!(fd, 1 | 2) {
        return Ok(env
            .kernel
            .write_output(fd, &data)
            .map(|_| len)
            .map_err(|_| errno::EIO));
    }
    Ok(match env.kernel.libc.files.write_fd(fd, &data) {
        Some(Ok(())) => Ok(len),
        Some(Err(e)) => Err(io_errno(&e)),
        None => Err(errno::EBADF),
    })
}

fn sce_write(env: &mut Env) -> Result<Outcome, CallError> {
    let r = write_file(env)?;
    finish(env, Style::Sce, r)
}

/// `write` and `_write`, with errno.
pub(crate) fn write(env: &mut Env) -> Result<Outcome, CallError> {
    let r = write_file(env)?;
    finish(env, Style::Posix, r)
}

fn sce_mkdir(env: &mut Env) -> Result<Outcome, CallError> {
    let path = cstr(env, env.arg(0))?;
    let r = env.kernel.vfs.mkdir(&path).map(|_| 0);
    finish(env, Style::Sce, r)
}
