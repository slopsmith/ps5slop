//! libSceFios2's synchronous file handle calls, which a PS4 title's own
//! `libc.prx` uses for its `FILE` functions. A handle is one of our file
//! descriptors (see `file.rs`). Argument positions are what Deltarune's libc
//! passes; flag bits other than zero (read only) and the error codes are
//! assumptions (`docs/hle-kernel.md`).

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom};

use crate::dmem::sce_error;
use crate::libc::cstr;
use crate::vfs::io_errno;
use crate::{CallError, Env, Function, Outcome, errno};

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceFios2",
        name,
        func,
    }
}

pub static FUNCTIONS: &[Function] = &[
    f("sceFiosFHOpenWithModeSync", open),
    f("sceFiosFHOpenSync", open),
    f("sceFiosFHReadSync", read),
    f("sceFiosFHWriteSync", write),
    f("sceFiosFHSeek", seek),
    f("sceFiosFHCloseSync", close),
    f("sceFiosFHToFileno", same),
    f("sceFiosFilenoToFH", same),
    f("sceFiosIsValidHandle", is_valid),
    f("sceFiosDeallocatePassthruFH", |_| ret(0)),
    f("sceFiosRenameSync", rename),
    f("sceFiosDeleteSync", delete),
];

const O_WRITE: u32 = 0x2;
const O_APPEND: u32 = 0x4;
const O_CREAT: u32 = 0x8;
const O_TRUNC: u32 = 0x10;
/// Reads larger than this are cut short.
const MAX_READ: u64 = 1 << 30;

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

fn result(r: Result<u64, i32>) -> Result<Outcome, CallError> {
    ret(r.unwrap_or_else(sce_error))
}

/// `(attr, out_handle, path, params, mode)`: the open flags are the first
/// u16 of `params`, which may be null (read only).
fn open(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, path, params) = (env.arg(1), cstr(env, env.arg(2))?, env.arg(3));
    let flags = if params == 0 {
        0
    } else {
        let mut b = [0u8; 2];
        env.mem.read(params, &mut b)?;
        u32::from(u16::from_le_bytes(b))
    };
    let writes = flags & (O_WRITE | O_APPEND | O_CREAT | O_TRUNC) != 0;
    let mut o = OpenOptions::new();
    o.read(true)
        .write(flags & O_WRITE != 0)
        .append(flags & O_APPEND != 0)
        .truncate(flags & O_TRUNC != 0)
        .create(flags & O_CREAT != 0);
    match env.kernel.vfs.open(&path, &o, writes) {
        Ok((file, dir)) => {
            let origin = crate::libc::Origin {
                path: path.clone(),
                write: flags & (O_WRITE | O_APPEND) != 0,
                append: flags & O_APPEND != 0,
            };
            let fd = env.kernel.libc.files.open_fd(file, origin);
            if let Some(dir) = dir {
                env.kernel.libc.files.set_directory(fd, dir);
            }
            env.mem.write(out, &fd.to_le_bytes())?;
            ret(0)
        }
        Err(e) => {
            env.mem.write(out, &(-1i32).to_le_bytes())?;
            result(Err(e))
        }
    }
}

/// `(attr, handle, buf, len)`: the count read.
fn read(env: &mut Env) -> Result<Outcome, CallError> {
    let (fd, buf, len) = (env.arg(1) as i32, env.arg(2), env.arg(3).min(MAX_READ));
    let mut data = vec![0u8; len as usize];
    match env.kernel.libc.files.read_fd(fd, &mut data) {
        Some(Ok(n)) => {
            env.mem.write(buf, &data[..n])?;
            ret(n as u64)
        }
        Some(Err(e)) => result(Err(io_errno(&e))),
        None => result(Err(errno::EBADF)),
    }
}

/// `(attr, handle, buf, len)`: the count written.
fn write(env: &mut Env) -> Result<Outcome, CallError> {
    let (fd, buf, len) = (env.arg(1) as i32, env.arg(2), env.arg(3).min(MAX_READ));
    let mut data = vec![0u8; len as usize];
    env.mem.read(buf, &mut data)?;
    result(match env.kernel.libc.files.write_fd(fd, &data) {
        Some(Ok(())) => Ok(len),
        Some(Err(e)) => Err(io_errno(&e)),
        None => Err(errno::EBADF),
    })
}

/// `(handle, offset, whence)`, whence as in `lseek`: the new position.
fn seek(env: &mut Env) -> Result<Outcome, CallError> {
    let (fd, off) = (env.arg(0) as i32, env.arg(1) as i64);
    let to = match env.arg(2) {
        0 => match u64::try_from(off) {
            Ok(o) => SeekFrom::Start(o),
            Err(_) => return result(Err(errno::EINVAL)),
        },
        1 => SeekFrom::Current(off),
        2 => SeekFrom::End(off),
        _ => return result(Err(errno::EINVAL)),
    };
    result(match env.kernel.libc.files.with_fd(fd, |f| f.seek(to)) {
        Some(Ok(at)) => Ok(at),
        Some(Err(e)) => Err(io_errno(&e)),
        None => Err(errno::EBADF),
    })
}

/// `(attr, handle)`.
fn close(env: &mut Env) -> Result<Outcome, CallError> {
    if env.kernel.libc.files.close_fd(env.arg(1) as i32) {
        ret(0)
    } else {
        result(Err(errno::EBADF))
    }
}

/// A handle is its descriptor, and the other way round.
fn same(env: &mut Env) -> Result<Outcome, CallError> {
    ret(env.arg(0) as i32 as u64)
}

fn is_valid(env: &mut Env) -> Result<Outcome, CallError> {
    let fd = env.arg(0) as i32;
    ret(u64::from(
        env.kernel.libc.files.with_fd(fd, |_| ()).is_some(),
    ))
}

fn writable(env: &Env, n: usize) -> Result<Result<std::path::PathBuf, i32>, CallError> {
    let path = cstr(env, env.arg(n))?;
    Ok(match env.kernel.vfs.resolve(&path) {
        None => Err(errno::ENOENT),
        Some((_, false)) => Err(errno::EROFS),
        Some((host, true)) => Ok(host),
    })
}

/// `(attr, from, to)`.
fn rename(env: &mut Env) -> Result<Outcome, CallError> {
    result(match (writable(env, 1)?, writable(env, 2)?) {
        (Ok(from), Ok(to)) => std::fs::rename(from, to)
            .map(|()| 0)
            .map_err(|e| io_errno(&e)),
        (Err(e), _) | (_, Err(e)) => Err(e),
    })
}

/// `(attr, path)`: a file or an empty directory.
fn delete(env: &mut Env) -> Result<Outcome, CallError> {
    result(writable(env, 1)?.and_then(|p| {
        let r = if p.is_dir() {
            std::fs::remove_dir(&p)
        } else {
            std::fs::remove_file(&p)
        };
        r.map(|()| 0).map_err(|e| io_errno(&e))
    }))
}
