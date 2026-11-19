//! libkernel's file descriptors: the POSIX calls (`open`, `read`, `close`),
//! which fail with errno and -1, and their `sceKernel*` twins, which return
//! `0x80020000` + errno. Paths go through the guest's file system (see
//! `vfs.rs`). Flag values are FreeBSD's.

use std::fs::OpenOptions;

use crate::dmem::sce_error;
use crate::libc::cstr;
use crate::vfs::{GuestFile, io_errno};
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
    f("lseek", lseek),
    f("sceKernelLseek", sce_lseek),
    f("unlink", unlink),
    f("sceKernelUnlink", sce_unlink),
    f("rmdir", rmdir),
    f("sceKernelRmdir", sce_rmdir),
    f("rename", rename),
    f("sceKernelRename", sce_rename),
    f("fsync", fsync),
    f("sceKernelFsync", sce_fsync),
    f("ftruncate", ftruncate),
    f("sceKernelFtruncate", sce_ftruncate),
    f("truncate", truncate),
    f("sceKernelTruncate", sce_truncate),
    f("getdents", getdents),
    f("sceKernelGetdents", sce_getdents),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f("sceKernelChmod", sce_accept),
    f("sceKernelUtimes", sce_accept),
    f("sceKernelSync", sce_accept),
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
const MAX_READ: u64 = 1 << 30;

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
    let access = flags & O_ACCMODE;
    let wants_write = access != 0 || flags & (O_CREAT | O_TRUNC | O_APPEND) != 0;
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
    Ok(env
        .kernel
        .vfs
        .open(&path, &o, wants_write)
        .map(|(file, dir)| {
            let fd = env.kernel.libc.files.open_fd(file);
            if let Some(dir) = dir {
                env.kernel.libc.files.set_directory(fd, dir);
            }
            fd as u64
        }))
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

/// `lseek(fd, offset, whence)` with FreeBSD's `SEEK_SET`, `SEEK_CUR` and
/// `SEEK_END` (0 to 2).
fn seek_file(env: &Env) -> Result<u64, i32> {
    use std::io::{Seek, SeekFrom};
    let (fd, off) = (env.arg(0) as i32, env.arg(1) as i64);
    let to = match env.arg(2) {
        0 => SeekFrom::Start(u64::try_from(off).map_err(|_| errno::EINVAL)?),
        1 => SeekFrom::Current(off),
        2 => SeekFrom::End(off),
        _ => return Err(errno::EINVAL),
    };
    match env.kernel.libc.files.with_fd(fd, |f| f.seek(to)) {
        Some(Ok(at)) => Ok(at),
        Some(Err(e)) => Err(io_errno(&e)),
        None => Err(errno::EBADF),
    }
}

fn lseek(env: &mut Env) -> Result<Outcome, CallError> {
    finish(env, Style::Posix, seek_file(env))
}

fn sce_lseek(env: &mut Env) -> Result<Outcome, CallError> {
    finish(env, Style::Sce, seek_file(env))
}

/// The host path of the guest path in argument `n`, if it may be written.
fn writable_path(env: &Env, n: usize) -> Result<Result<std::path::PathBuf, i32>, CallError> {
    let path = cstr(env, env.arg(n))?;
    Ok(match env.kernel.vfs.resolve(&path) {
        None => Err(errno::ENOENT),
        Some((_, false)) => Err(errno::EROFS),
        Some((host, true)) => Ok(host),
    })
}

fn on_path(
    env: &Env,
    op: impl FnOnce(&std::path::Path) -> std::io::Result<()>,
) -> Result<Result<u64, i32>, CallError> {
    Ok(writable_path(env, 0)?.and_then(|p| op(&p).map(|()| 0).map_err(|e| io_errno(&e))))
}

fn unlink(env: &mut Env) -> Result<Outcome, CallError> {
    let r = on_path(env, |p| std::fs::remove_file(p))?;
    finish(env, Style::Posix, r)
}

fn sce_unlink(env: &mut Env) -> Result<Outcome, CallError> {
    let r = on_path(env, |p| std::fs::remove_file(p))?;
    finish(env, Style::Sce, r)
}

fn rmdir(env: &mut Env) -> Result<Outcome, CallError> {
    let r = on_path(env, |p| std::fs::remove_dir(p))?;
    finish(env, Style::Posix, r)
}

fn sce_rmdir(env: &mut Env) -> Result<Outcome, CallError> {
    let r = on_path(env, |p| std::fs::remove_dir(p))?;
    finish(env, Style::Sce, r)
}

fn rename_file(env: &Env) -> Result<Result<u64, i32>, CallError> {
    Ok(match (writable_path(env, 0)?, writable_path(env, 1)?) {
        (Ok(from), Ok(to)) => std::fs::rename(from, to)
            .map(|()| 0)
            .map_err(|e| io_errno(&e)),
        (Err(e), _) | (_, Err(e)) => Err(e),
    })
}

fn rename(env: &mut Env) -> Result<Outcome, CallError> {
    let r = rename_file(env)?;
    finish(env, Style::Posix, r)
}

fn sce_rename(env: &mut Env) -> Result<Outcome, CallError> {
    let r = rename_file(env)?;
    finish(env, Style::Sce, r)
}

fn on_fd(env: &Env, op: impl FnOnce(&mut GuestFile) -> std::io::Result<()>) -> Result<u64, i32> {
    match env.kernel.libc.files.with_fd(env.arg(0) as i32, op) {
        Some(Ok(())) => Ok(0),
        Some(Err(e)) => Err(io_errno(&e)),
        None => Err(errno::EBADF),
    }
}

fn fsync(env: &mut Env) -> Result<Outcome, CallError> {
    finish(env, Style::Posix, on_fd(env, |f| f.sync_all()))
}

fn sce_fsync(env: &mut Env) -> Result<Outcome, CallError> {
    finish(env, Style::Sce, on_fd(env, |f| f.sync_all()))
}

fn ftruncate(env: &mut Env) -> Result<Outcome, CallError> {
    let len = env.arg(1);
    finish(env, Style::Posix, on_fd(env, |f| f.set_len(len)))
}

fn sce_ftruncate(env: &mut Env) -> Result<Outcome, CallError> {
    let len = env.arg(1);
    finish(env, Style::Sce, on_fd(env, |f| f.set_len(len)))
}

fn truncate_path(env: &Env) -> Result<Result<u64, i32>, CallError> {
    let len = env.arg(1);
    on_path(env, |p| {
        OpenOptions::new().write(true).open(p)?.set_len(len)
    })
}

fn truncate(env: &mut Env) -> Result<Outcome, CallError> {
    let r = truncate_path(env)?;
    finish(env, Style::Posix, r)
}

fn sce_truncate(env: &mut Env) -> Result<Outcome, CallError> {
    let r = truncate_path(env)?;
    finish(env, Style::Sce, r)
}

/// Permissions and file times are not kept; `sceKernelSync` has nothing to do.
fn sce_accept(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// FreeBSD `DT_DIR` and `DT_REG`.
const DT_DIR: u8 = 4;
const DT_REG: u8 = 8;

/// `getdents(fd, buf, len)` with FreeBSD's pre-12 `struct dirent`; 0 at
/// the end.
fn read_directory(env: &Env) -> Result<Result<u64, i32>, CallError> {
    let (fd, buf, len) = (env.arg(0) as i32, env.arg(1), env.arg(2) as usize);
    let out = env.kernel.libc.files.with_directory(fd, |dir, pos| {
        let mut names: Vec<(String, u8)> = dir
            .names()
            .into_iter()
            .map(|(name, is_dir)| (name, if is_dir { DT_DIR } else { DT_REG }))
            .collect();
        names.splice(0..0, [(".".to_owned(), DT_DIR), ("..".to_owned(), DT_DIR)]);
        let mut records = Vec::new();
        for (i, (name, kind)) in names.iter().enumerate().skip(*pos) {
            let name = &name.as_bytes()[..name.len().min(255)];
            let reclen = (8 + name.len() + 1).next_multiple_of(4);
            if records.len() + reclen > len {
                break;
            }
            let start = records.len();
            records.resize(start + reclen, 0);
            records[start..start + 4].copy_from_slice(&(i as u32 + 1).to_le_bytes());
            records[start + 4..start + 6].copy_from_slice(&(reclen as u16).to_le_bytes());
            records[start + 6] = *kind;
            records[start + 7] = name.len() as u8;
            records[start + 8..start + 8 + name.len()].copy_from_slice(name);
            *pos += 1;
        }
        records
    });
    let Some(records) = out else {
        return Ok(Err(errno::EBADF));
    };
    env.mem.write(buf, &records)?;
    Ok(Ok(records.len() as u64))
}

fn getdents(env: &mut Env) -> Result<Outcome, CallError> {
    let r = read_directory(env)?;
    finish(env, Style::Posix, r)
}

fn sce_getdents(env: &mut Env) -> Result<Outcome, CallError> {
    let r = read_directory(env)?;
    finish(env, Style::Sce, r)
}
