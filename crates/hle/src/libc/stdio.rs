//! `<stdio.h>` streams. A `FILE *` is an address we hand out; the stream
//! behind it lives on the host. The guest never sees the inside of a FILE.
//! Paths under `/app0` open files in the title folder, read-only.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use super::format::{Args, printf};
use super::{STDERR_FILE, STDIN_FILE, STDOUT_FILE, cstr, f, ret};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome, errno};

pub static FUNCTIONS: &[Function] = &[
    f("fopen", fopen),
    f("fclose", fclose),
    f("fread", fread),
    f("fwrite", fwrite),
    f("fgets", fgets),
    f("fprintf", fprintf),
    f("fflush", fflush),
    f("fseeko", fseeko),
    f("ftello", ftello),
    f("ferror", ferror),
    f("fileno", fileno),
];

/// How much guest memory a FILE handle takes; the guest only uses its address.
const FILE_SIZE: u64 = 0x100;
/// The first descriptor number given to an opened file.
const FIRST_FD: i32 = 3;
const EOF: u64 = -1i32 as u32 as u64;
const EROFS: i32 = 30;

enum Target {
    Console(i32),
    Host(File),
}

struct Stream {
    target: Target,
    fd: i32,
    error: bool,
    eof: bool,
}

/// The open streams, by FILE address.
pub(crate) struct Files {
    streams: Mutex<HashMap<u64, Stream>>,
}

/// What a file descriptor from a stream refers to.
pub(crate) enum FdInfo {
    Console,
    File(std::fs::Metadata),
}

impl Files {
    pub(crate) fn fd_info(&self, fd: i32) -> Option<FdInfo> {
        let streams = lock(&self.streams);
        let s = streams.values().find(|s| s.fd == fd)?;
        match &s.target {
            Target::Console(_) => Some(FdInfo::Console),
            Target::Host(f) => f.metadata().ok().map(FdInfo::File),
        }
    }
}

impl Default for Files {
    fn default() -> Self {
        let console = |fd| Stream {
            target: Target::Console(fd),
            fd,
            error: false,
            eof: false,
        };
        let streams = HashMap::from([
            (STDIN_FILE, console(0)),
            (STDOUT_FILE, console(1)),
            (STDERR_FILE, console(2)),
        ]);
        Self {
            streams: Mutex::new(streams),
        }
    }
}

/// The host path for guest path `path`, if it names something in the
/// title folder.
pub(crate) fn host_path(app0: Option<&Path>, path: &[u8]) -> Option<PathBuf> {
    let path = std::str::from_utf8(path).ok()?;
    let rest = path.strip_prefix("/app0")?;
    if !(rest.is_empty() || rest.starts_with('/')) {
        return None;
    }
    let rest = Path::new(rest.trim_start_matches('/'));
    if rest
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return None;
    }
    Some(app0?.join(rest))
}

fn fopen(env: &mut Env) -> Result<Outcome, CallError> {
    let path = cstr(env, env.arg(0))?;
    let mode = cstr(env, env.arg(1))?;
    if mode.first() != Some(&b'r') || mode.contains(&b'+') {
        env.set_errno(EROFS)?;
        return ret(0);
    }
    let file = host_path(env.kernel.app0.as_deref(), &path)
        .and_then(|p| OpenOptions::new().read(true).open(p).ok());
    let Some(file) = file else {
        env.set_errno(errno::ENOENT)?;
        return ret(0);
    };
    let Some(at) = env.kernel.heap.alloc(env.mem, FILE_SIZE) else {
        env.set_errno(errno::ENOMEM)?;
        return ret(0);
    };
    let mut streams = lock(&env.kernel.libc.files.streams);
    let fd = (FIRST_FD..)
        .find(|fd| streams.values().all(|s| s.fd != *fd))
        .expect("a free fd");
    streams.insert(
        at,
        Stream {
            target: Target::Host(file),
            fd,
            error: false,
            eof: false,
        },
    );
    ret(at)
}

/// Runs `op` on the stream at `file`, or fails with EBADF.
fn with_stream(
    env: &Env,
    file: u64,
    fail: u64,
    op: impl FnOnce(&mut Stream) -> Result<u64, CallError>,
) -> Result<Outcome, CallError> {
    let mut streams = lock(&env.kernel.libc.files.streams);
    match streams.get_mut(&file) {
        Some(s) => ret(op(s)?),
        None => {
            drop(streams);
            env.set_errno(errno::EBADF)?;
            ret(fail)
        }
    }
}

fn fclose(env: &mut Env) -> Result<Outcome, CallError> {
    let file = env.arg(0);
    let removed = match lock(&env.kernel.libc.files.streams).remove(&file) {
        Some(Stream {
            target: Target::Host(_),
            ..
        }) => true,
        Some(console) => {
            lock(&env.kernel.libc.files.streams).insert(file, console);
            false
        }
        None => {
            env.set_errno(errno::EBADF)?;
            return ret(EOF);
        }
    };
    if removed {
        env.kernel.heap.free(file);
    }
    ret(0)
}

/// Reads up to `buf.len()` bytes; stops early only at end of file or on error.
fn read_stream(s: &mut Stream, buf: &mut [u8]) -> usize {
    let Target::Host(file) = &mut s.target else {
        s.eof = true;
        return 0;
    };
    let mut done = 0;
    while done < buf.len() {
        match file.read(&mut buf[done..]) {
            Ok(0) => {
                s.eof = true;
                break;
            }
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => {
                s.error = true;
                break;
            }
        }
    }
    done
}

fn write_stream(env: &Env, s: &mut Stream, data: &[u8]) -> bool {
    let ok = match &mut s.target {
        Target::Console(fd @ (1 | 2)) => env.kernel.write_output(*fd, data).is_ok(),
        Target::Console(_) => false,
        Target::Host(file) => file.write_all(data).is_ok(),
    };
    s.error |= !ok;
    ok
}

fn fread(env: &mut Env) -> Result<Outcome, CallError> {
    let (dst, size, count, file) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    let total = size.saturating_mul(count);
    if total == 0 {
        return ret(0);
    }
    let mut buf = vec![0u8; total as usize];
    with_stream(env, file, 0, |s| {
        let n = read_stream(s, &mut buf);
        env.mem.write(dst, &buf[..n])?;
        Ok(n as u64 / size)
    })
}

fn fwrite(env: &mut Env) -> Result<Outcome, CallError> {
    let (src, size, count, file) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    let total = size.saturating_mul(count);
    if total == 0 {
        return ret(0);
    }
    let mut buf = vec![0u8; total as usize];
    env.mem.read(src, &mut buf)?;
    with_stream(env, file, 0, |s| {
        Ok(if write_stream(env, s, &buf) { count } else { 0 })
    })
}

/// `fgets(buf, n, file)`: reads one byte at a time up to and including a
/// newline, so the file position stays exact.
fn fgets(env: &mut Env) -> Result<Outcome, CallError> {
    let (dst, n, file) = (env.arg(0), env.arg(1) as i32, env.arg(2));
    if n <= 0 {
        return ret(0);
    }
    with_stream(env, file, 0, |s| {
        let mut line = Vec::new();
        let mut byte = [0u8];
        while line.len() + 1 < n as usize && read_stream(s, &mut byte) == 1 {
            line.push(byte[0]);
            if byte[0] == b'\n' {
                break;
            }
        }
        if line.is_empty() && n > 1 {
            return Ok(0);
        }
        line.push(0);
        env.mem.write(dst, &line)?;
        Ok(dst)
    })
}

fn fprintf(env: &mut Env) -> Result<Outcome, CallError> {
    let file = env.arg(0);
    let fmt = cstr(env, env.arg(1))?;
    let text = printf(env, &fmt, &mut Args::call(env, 2))?;
    with_stream(env, file, EOF, |s| {
        Ok(if write_stream(env, s, &text) {
            text.len() as u64
        } else {
            EOF
        })
    })
}

/// Nothing is buffered on our side, so there is nothing to flush.
fn fflush(env: &mut Env) -> Result<Outcome, CallError> {
    let file = env.arg(0);
    if file == 0 {
        return ret(0);
    }
    with_stream(env, file, EOF, |_| Ok(0))
}

fn fseeko(env: &mut Env) -> Result<Outcome, CallError> {
    let (file, offset, whence) = (env.arg(0), env.arg(1) as i64, env.arg(2) as i32);
    let pos = match whence {
        0 if offset >= 0 => SeekFrom::Start(offset as u64),
        1 => SeekFrom::Current(offset),
        2 => SeekFrom::End(offset),
        _ => return env.fail(errno::EINVAL),
    };
    let mut bad = false;
    let out = with_stream(env, file, EOF, |s| {
        let moved = match &mut s.target {
            Target::Host(f) => f.seek(pos).is_ok(),
            Target::Console(_) => false,
        };
        s.eof &= !moved;
        bad = !moved;
        Ok(if moved { 0 } else { EOF })
    })?;
    if bad {
        env.set_errno(errno::EINVAL)?;
    }
    Ok(out)
}

fn ftello(env: &mut Env) -> Result<Outcome, CallError> {
    let file = env.arg(0);
    with_stream(env, file, u64::MAX, |s| match &mut s.target {
        Target::Host(f) => Ok(f.stream_position().unwrap_or(u64::MAX)),
        Target::Console(_) => Ok(u64::MAX),
    })
}

fn ferror(env: &mut Env) -> Result<Outcome, CallError> {
    let file = env.arg(0);
    with_stream(env, file, 0, |s| Ok(s.error as u64))
}

fn fileno(env: &mut Env) -> Result<Outcome, CallError> {
    let file = env.arg(0);
    with_stream(env, file, EOF, |s| Ok(s.fd as u32 as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_paths_inside_app0_map_to_the_title_folder() {
        let app0 = Path::new("/titles/X");
        let map = |p: &[u8]| host_path(Some(app0), p);
        assert_eq!(
            map(b"/app0/sce_sys/param.json"),
            Some(app0.join("sce_sys/param.json"))
        );
        assert_eq!(map(b"/app0"), Some(app0.to_path_buf()));
        assert_eq!(map(b"/app0/../etc/passwd"), None);
        assert_eq!(map(b"/app0x/file"), None);
        assert_eq!(map(b"/data/file"), None);
        assert_eq!(host_path(None, b"/app0/file"), None);
    }
}
