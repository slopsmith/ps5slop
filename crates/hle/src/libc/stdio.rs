//! `<stdio.h>` streams. A `FILE *` is an address we hand out; the stream
//! behind it lives on the host. The guest never sees the inside of a FILE.
//! Paths go through the guest's file system (see `vfs.rs`).

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Mutex;

use super::format::{Args, printf};
use super::{STDERR_FILE, STDIN_FILE, STDOUT_FILE, cstr, f, ret};
use crate::pthread::lock;
use crate::vfs::{DirSource, GuestFile, Meta};
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

enum Target {
    Console(i32),
    Host(GuestFile),
}

struct Stream {
    target: Target,
    fd: i32,
    error: bool,
    eof: bool,
    origin: Option<Origin>,
}

/// How a file was opened, so that a loaded save state can open it again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Origin {
    pub path: Vec<u8>,
    pub write: bool,
    pub append: bool,
}

savestate::record!(Origin {
    path,
    write,
    append
});

/// The open streams, by FILE address, and the files opened as bare
/// descriptors (libkernel's `open`). Both share one descriptor space.
pub(crate) struct Files {
    streams: Mutex<HashMap<u64, Stream>>,
    descriptors: Mutex<HashMap<i32, GuestFile>>,
    origins: Mutex<HashMap<i32, Origin>>,
    /// Descriptors open on directories: what to list and how many entries
    /// `getdents` has returned.
    directories: Mutex<HashMap<i32, (DirSource, usize)>>,
}

/// What a file descriptor from a stream refers to.
pub(crate) enum FdInfo {
    Console,
    File(Meta),
}

impl Files {
    pub(crate) fn fd_info(&self, fd: i32) -> Option<FdInfo> {
        let streams = lock(&self.streams);
        if let Some(f) = lock(&self.descriptors).get(&fd) {
            return f.meta().ok().map(FdInfo::File);
        }
        let s = streams.values().find(|s| s.fd == fd)?;
        match &s.target {
            Target::Console(_) => Some(FdInfo::Console),
            Target::Host(f) => f.meta().ok().map(FdInfo::File),
        }
    }

    /// Gives `file` the lowest free descriptor number.
    pub(crate) fn open_fd(&self, file: GuestFile, origin: Origin) -> i32 {
        let streams = lock(&self.streams);
        let mut descriptors = lock(&self.descriptors);
        let fd = free_fd(&streams, &descriptors);
        descriptors.insert(fd, file);
        lock(&self.origins).insert(fd, origin);
        fd
    }

    /// Reads from descriptor `fd` into `buf`; `None` when `fd` is not an
    /// open descriptor file.
    pub(crate) fn read_fd(&self, fd: i32, buf: &mut [u8]) -> Option<std::io::Result<usize>> {
        lock(&self.descriptors).get_mut(&fd).map(|f| f.read(buf))
    }

    /// Writes `data` to descriptor `fd`; `None` when `fd` is not an open
    /// descriptor file.
    pub(crate) fn write_fd(&self, fd: i32, data: &[u8]) -> Option<std::io::Result<()>> {
        lock(&self.descriptors)
            .get_mut(&fd)
            .map(|f| f.write_all(data))
    }

    /// Runs `f` on descriptor `fd`'s file; `None` when it is not open.
    pub(crate) fn with_fd<T>(&self, fd: i32, f: impl FnOnce(&mut GuestFile) -> T) -> Option<T> {
        lock(&self.descriptors).get_mut(&fd).map(f)
    }

    /// Closes descriptor `fd`; false when it was not open.
    pub(crate) fn close_fd(&self, fd: i32) -> bool {
        lock(&self.directories).remove(&fd);
        lock(&self.origins).remove(&fd);
        lock(&self.descriptors).remove(&fd).is_some()
    }

    /// Marks `fd` as open on directory `dir`.
    pub(crate) fn set_directory(&self, fd: i32, dir: DirSource) {
        lock(&self.directories).insert(fd, (dir, 0));
    }

    /// Runs `f` on directory descriptor `fd`'s directory and read position.
    pub(crate) fn with_directory<T>(
        &self,
        fd: i32,
        f: impl FnOnce(&DirSource, &mut usize) -> T,
    ) -> Option<T> {
        lock(&self.directories)
            .get_mut(&fd)
            .map(|(p, pos)| f(p, pos))
    }
}

fn free_fd(streams: &HashMap<u64, Stream>, descriptors: &HashMap<i32, GuestFile>) -> i32 {
    (FIRST_FD..)
        .find(|fd| !descriptors.contains_key(fd) && streams.values().all(|s| s.fd != *fd))
        .expect("a free fd")
}

impl Default for Files {
    fn default() -> Self {
        let console = |fd| Stream {
            target: Target::Console(fd),
            fd,
            error: false,
            eof: false,
            origin: None,
        };
        let streams = HashMap::from([
            (STDIN_FILE, console(0)),
            (STDOUT_FILE, console(1)),
            (STDERR_FILE, console(2)),
        ]);
        Self {
            streams: Mutex::new(streams),
            descriptors: Mutex::default(),
            origins: Mutex::default(),
            directories: Mutex::default(),
        }
    }
}

/// How `fopen` mode `mode` (`r`, `w`, `a`, each optionally with `+`)
/// opens a file.
fn open_options(mode: &[u8]) -> Option<OpenOptions> {
    let plus = mode.contains(&b'+');
    let mut o = OpenOptions::new();
    match mode.first()? {
        b'r' => o.read(true).write(plus),
        b'w' => o.write(true).read(plus).create(true).truncate(true),
        b'a' => o.append(true).read(plus).create(true),
        _ => return None,
    };
    Some(o)
}

fn fopen(env: &mut Env) -> Result<Outcome, CallError> {
    let path = cstr(env, env.arg(0))?;
    let mode = cstr(env, env.arg(1))?;
    let Some(options) = open_options(&mode) else {
        env.set_errno(errno::EINVAL)?;
        return ret(0);
    };
    let writes = mode.first() != Some(&b'r') || mode.contains(&b'+');
    let file = match env.kernel.vfs.open(&path, &options, writes) {
        Ok((f, _)) => f,
        Err(e) => {
            env.set_errno(e)?;
            return ret(0);
        }
    };
    let Some(at) = env.kernel.heap.alloc(env.mem, FILE_SIZE) else {
        env.set_errno(errno::ENOMEM)?;
        return ret(0);
    };
    let files = &env.kernel.libc.files;
    let mut streams = lock(&files.streams);
    let fd = free_fd(&streams, &lock(&files.descriptors));
    streams.insert(
        at,
        Stream {
            target: Target::Host(file),
            fd,
            error: false,
            eof: false,
            origin: Some(Origin {
                path,
                write: writes,
                append: mode.first() == Some(&b'a'),
            }),
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

/// The guest's position in an open file, from its handle.
fn position(file: &mut GuestFile) -> u64 {
    file.stream_position().unwrap_or(0)
}

impl Files {
    /// Writes every open file as how it was opened and where it is.
    pub(crate) fn save(&self, e: &mut savestate::Encoder) -> Result<(), String> {
        let mut streams = lock(&self.streams);
        let mut descriptors = lock(&self.descriptors);
        let origins = lock(&self.origins);
        let directories = lock(&self.directories);
        let mut saved_streams = Vec::new();
        for (&at, s) in streams.iter_mut() {
            let file = match &mut s.target {
                Target::Console(fd) => (true, *fd as u64),
                Target::Host(f) => (false, position(f)),
            };
            saved_streams.push((at, s.fd, file, (s.error, s.eof), s.origin.clone()));
        }
        let mut saved_fds = Vec::new();
        for (&fd, f) in descriptors.iter_mut() {
            let Some(origin) = origins.get(&fd) else {
                return Err(format!("descriptor {fd} has no recorded path"));
            };
            let dir = directories.get(&fd).map(|(_, n)| *n as u64);
            saved_fds.push((fd, origin.clone(), position(f), dir));
        }
        saved_streams.sort_by_key(|s| s.0);
        saved_fds.sort_by_key(|f| f.0);
        e.put(&saved_streams).put(&saved_fds);
        Ok(())
    }

    /// Opens the saved files again through `vfs`.
    pub(crate) fn load(
        &self,
        d: &mut savestate::Decoder,
        vfs: &crate::Vfs,
    ) -> Result<(), savestate::Error> {
        type SavedStream = (u64, i32, (bool, u64), (bool, bool), Option<Origin>);
        let saved_streams: Vec<SavedStream> = d.get()?;
        let saved_fds: Vec<(i32, Origin, u64, Option<u64>)> = d.get()?;
        let reopen =
            |o: &Origin, pos: u64| -> Result<(GuestFile, Option<DirSource>), savestate::Error> {
                let mut options = OpenOptions::new();
                options
                    .read(true)
                    .write(o.write && !o.append)
                    .append(o.append);
                let (mut file, dir) = vfs.open(&o.path, &options, o.write).map_err(|e| {
                    savestate::error(format!(
                        "cannot open {} again (errno {e})",
                        String::from_utf8_lossy(&o.path)
                    ))
                })?;
                if dir.is_none() {
                    file.seek(SeekFrom::Start(pos))?;
                }
                Ok((file, dir))
            };
        let mut streams = HashMap::new();
        for (at, fd, (console, n), (error, eof), origin) in saved_streams {
            let target = match (&origin, console) {
                (_, true) => Target::Console(n as i32),
                (Some(o), false) => Target::Host(reopen(o, n)?.0),
                (None, false) => return Err(savestate::error("a stream has no recorded path")),
            };
            let s = Stream {
                target,
                fd,
                error,
                eof,
                origin,
            };
            streams.insert(at, s);
        }
        let (mut descriptors, mut origins, mut directories) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        for (fd, origin, pos, dir) in saved_fds {
            let (file, source) = reopen(&origin, pos)?;
            if let (Some(source), Some(n)) = (source, dir) {
                directories.insert(fd, (source, n as usize));
            }
            descriptors.insert(fd, file);
            origins.insert(fd, origin);
        }
        *lock(&self.streams) = streams;
        *lock(&self.descriptors) = descriptors;
        *lock(&self.origins) = origins;
        *lock(&self.directories) = directories;
        Ok(())
    }
}
