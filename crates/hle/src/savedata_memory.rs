//! Save data memory (`sceSaveData*SaveDataMemory2`): a block of bytes per
//! user and slot that a title reads and writes by offset, kept as a host
//! file in a hidden folder of the saves folder, so save searches never list
//! it. Layouts are shadPS4's PS4 ones (from memory), checked against what
//! GTA V passes for setup and get; unverified for the PS5.

use std::path::{Path, PathBuf};

use crate::{CallError, Env, Function, Outcome};

/// `sceSaveDataGetSaveDataMemory2` and the other calls before setup. The
/// value is shadPS4's; GTA V only checks for a failure, then sets up.
const ERROR_MEMORY_NOT_READY: u64 = 0x809F_0012;
const ERROR_PARAMETER: u64 = 0x809F_0000;

const FOLDER: &str = ".memory";
/// One `SceSaveDataMemoryData`: buffer, size and offset, then reserved bytes.
const DATA_SIZE: u64 = 64;
const PARAM_SIZE: usize = 1328;
/// Set calls take at most this many data entries.
const MAX_DATA: u32 = 16;

pub static FUNCTIONS: &[Function] = &[
    f("sceSaveDataSetupSaveDataMemory2", setup),
    f("sceSaveDataGetSaveDataMemory2", get),
    f("sceSaveDataSetSaveDataMemory2", set),
    f("sceSaveDataSyncSaveDataMemory", sync),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceSaveData_native",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

/// The memory of `user`'s slot, and the parameters stored with it.
fn files(saves: &Path, user: u32, slot: u32) -> (PathBuf, PathBuf) {
    let dir = saves.join(FOLDER);
    (
        dir.join(format!("user{user}-slot{slot}.bin")),
        dir.join(format!("user{user}-slot{slot}.param")),
    )
}

/// Makes the memory at least `size` bytes; returns the size it had (0 when new).
fn grow(path: &Path, size: u64) -> std::io::Result<u64> {
    std::fs::create_dir_all(path.parent().unwrap_or(path))?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)?;
    let had = file.metadata()?.len();
    if had < size {
        file.set_len(size)?;
    }
    Ok(had)
}

/// `len` bytes at `offset`, or `None` when the memory is not set up or too small.
fn read_at(path: &Path, offset: u64, len: u64) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    if offset.checked_add(len)? > file.metadata().ok()?.len() {
        return None;
    }
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0; usize::try_from(len).ok()?];
    file.read_exact(&mut buf).ok()?;
    Some(buf)
}

/// Writes `bytes` at `offset`; false when the memory is not set up or too small.
fn write_at(path: &Path, offset: u64, bytes: &[u8]) -> bool {
    use std::io::{Seek, SeekFrom, Write};
    let Ok(mut file) = std::fs::OpenOptions::new().write(true).open(path) else {
        return false;
    };
    let fits = file
        .metadata()
        .is_ok_and(|m| offset.saturating_add(bytes.len() as u64) <= m.len());
    fits && file.seek(SeekFrom::Start(offset)).is_ok() && file.write_all(bytes).is_ok()
}

fn read_u32(env: &Env, at: u64) -> Result<u32, CallError> {
    let mut b = [0u8; 4];
    env.mem.read(at, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// `sceSaveDataSetupSaveDataMemory2(setup, result)`; the result gets the size it had.
fn setup(env: &mut Env) -> Result<Outcome, CallError> {
    let (at, result) = (env.arg(0), env.arg(1));
    let Some(saves) = env.kernel.vfs.saves.clone() else {
        return ret(ERROR_PARAMETER);
    };
    let (user, size, slot) = (
        read_u32(env, at + 4)?,
        env.mem.read_u64(at + 8)?,
        read_u32(env, at + 40)?,
    );
    let (memory, param) = files(&saves, user, slot);
    tracing::debug!("setup user {user} slot {slot}: {size:#x} bytes");
    let had = grow(&memory, size).map_err(|_| CallError::Abort("cannot make save data memory"))?;
    let init = env.mem.read_u64(at + 24)?;
    if had == 0 && init != 0 {
        let mut p = vec![0u8; PARAM_SIZE];
        env.mem.read(init, &mut p)?;
        let _ = std::fs::write(&param, p);
    }
    if result != 0 {
        env.mem.write_u64(result, had)?;
    }
    ret(0)
}

/// `sceSaveDataGetSaveDataMemory2(get)`.
fn get(env: &mut Env) -> Result<Outcome, CallError> {
    let at = env.arg(0);
    let Some(saves) = env.kernel.vfs.saves.clone() else {
        return ret(ERROR_MEMORY_NOT_READY);
    };
    let (user, data, param_at, slot) = (
        read_u32(env, at)?,
        env.mem.read_u64(at + 8)?,
        env.mem.read_u64(at + 16)?,
        read_u32(env, at + 32)?,
    );
    let (memory, param) = files(&saves, user, slot);
    tracing::debug!("get user {user} slot {slot} (set up: {})", memory.exists());
    if !memory.exists() {
        return ret(ERROR_MEMORY_NOT_READY);
    }
    if data != 0 {
        let (buf, len, offset) = (
            env.mem.read_u64(data)?,
            env.mem.read_u64(data + 8)?,
            env.mem.read_u64(data + 16)?,
        );
        let Some(bytes) = read_at(&memory, offset, len) else {
            return ret(ERROR_PARAMETER);
        };
        env.mem.write(buf, &bytes)?;
    }
    if param_at != 0 {
        let mut p = std::fs::read(&param).unwrap_or_default();
        p.resize(PARAM_SIZE, 0);
        env.mem.write(param_at, &p)?;
    }
    ret(0)
}

/// `sceSaveDataSetSaveDataMemory2(set)`.
fn set(env: &mut Env) -> Result<Outcome, CallError> {
    let at = env.arg(0);
    let Some(saves) = env.kernel.vfs.saves.clone() else {
        return ret(ERROR_MEMORY_NOT_READY);
    };
    let (user, data, param_at, count, slot) = (
        read_u32(env, at)?,
        env.mem.read_u64(at + 8)?,
        env.mem.read_u64(at + 16)?,
        read_u32(env, at + 32)?,
        read_u32(env, at + 36)?,
    );
    let (memory, param) = files(&saves, user, slot);
    if !memory.exists() {
        return ret(ERROR_MEMORY_NOT_READY);
    }
    if count > MAX_DATA {
        return ret(ERROR_PARAMETER);
    }
    let entries = match (data, count) {
        (0, _) => 0,
        // GTA V passes one entry with a count of 0.
        (_, 0) => 1,
        (_, n) => n,
    };
    for i in 0..u64::from(entries) {
        let entry = data + i * DATA_SIZE;
        let (buf, len, offset) = (
            env.mem.read_u64(entry)?,
            env.mem.read_u64(entry + 8)?,
            env.mem.read_u64(entry + 16)?,
        );
        tracing::debug!("set user {user} slot {slot}: {len:#x} bytes at {offset:#x}");
        let len = usize::try_from(len)
            .map_err(|_| CallError::Abort("save data memory write too large"))?;
        let mut bytes = vec![0u8; len];
        env.mem.read(buf, &mut bytes)?;
        if !write_at(&memory, offset, &bytes) {
            return ret(ERROR_PARAMETER);
        }
    }
    if param_at != 0 {
        let mut p = vec![0u8; PARAM_SIZE];
        env.mem.read(param_at, &mut p)?;
        let _ = std::fs::write(&param, p);
    }
    ret(0)
}

/// `sceSaveDataSyncSaveDataMemory(sync)`: writes go straight to the file.
fn sync(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

#[cfg(test)]
mod tests {
    use super::{grow, read_at, write_at};

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ps5slop-sdmemory-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("memory.bin")
    }

    #[test]
    fn memory_reads_back_what_was_written_after_setup() {
        let path = scratch("roundtrip");
        assert_eq!(grow(&path, 0x400).unwrap(), 0);
        assert!(write_at(&path, 0x10, b"settings"));
        assert_eq!(read_at(&path, 0x10, 8).unwrap(), b"settings");
        assert_eq!(read_at(&path, 0, 4).unwrap(), [0; 4]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn memory_that_was_never_set_up_cannot_be_read_or_written() {
        let path = scratch("missing");
        assert!(read_at(&path, 0, 4).is_none());
        assert!(!write_at(&path, 0, b"x"));
    }

    #[test]
    fn access_past_the_end_fails_and_setup_again_keeps_the_bytes() {
        let path = scratch("bounds");
        grow(&path, 16).unwrap();
        assert!(write_at(&path, 8, b"12345678"));
        assert!(!write_at(&path, 9, b"12345678"));
        assert!(read_at(&path, 12, 8).is_none());
        assert_eq!(grow(&path, 8).unwrap(), 16);
        assert_eq!(read_at(&path, 8, 8).unwrap(), b"12345678");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
