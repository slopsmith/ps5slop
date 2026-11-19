//! libkernel's file descriptors: the POSIX calls (`open`, `read`, `close`),
//! which fail with errno and -1, and their `sceKernel*` twins, which return
//! `0x80020000` + errno. Only `/app0` exists, read-only. Flag values are
//! FreeBSD's.

use std::fs::File;

use crate::dmem::sce_error;
use crate::libc::{cstr, host_path};
use crate::posix::mkdir_error;
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

/// `open(path, flags, mode)`: opens a file under `/app0` for reading.
fn open_file(env: &Env) -> Result<Result<u64, i32>, CallError> {
    let (path, flags) = (cstr(env, env.arg(0))?, env.arg(1));
    let Some(host) = host_path(env.kernel.app0.as_deref(), &path) else {
        return Ok(Err(errno::ENOENT));
    };
    if flags & O_ACCMODE != 0 {
        return Ok(Err(errno::EROFS));
    }
    Ok(match File::open(host) {
        Ok(file) => Ok(env.kernel.libc.files.open_fd(file) as u64),
        Err(_) => Err(errno::ENOENT),
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

/// `sceKernelWrite(fd, buf, len)`: fds 1 and 2 go to the host's stdout and
/// stderr; descriptor files are read-only.
fn sce_write(env: &mut Env) -> Result<Outcome, CallError> {
    let (fd, buf, len) = (env.arg(0) as i32, env.arg(1), env.arg(2).min(MAX_READ));
    if !matches!(fd, 1 | 2) {
        return ret(sce_error(errno::EBADF));
    }
    let mut data = vec![0u8; len as usize];
    env.mem.read(buf, &mut data)?;
    match env.kernel.write_output(fd, &data) {
        Ok(()) => ret(len),
        Err(_) => ret(sce_error(errno::EINVAL)),
    }
}

fn sce_mkdir(env: &mut Env) -> Result<Outcome, CallError> {
    let path = cstr(env, env.arg(0))?;
    ret(sce_error(mkdir_error(&path)))
}
