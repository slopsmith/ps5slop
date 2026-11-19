//! Save data (`libSceSaveData`): each save is a host folder in the saves
//! folder (see `Vfs::set_saves`), mounted at `/savedataN` while in use.
//! Structure layouts are read off what Minecraft passes and shadPS4's PS4
//! layouts (from memory); both are unverified for the PS5.

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

/// Mount mode bits (PS4 values).
const MOUNT_CREATE: u32 = 4;
const MOUNT_CREATE2: u32 = 32;

/// Error codes (PS4 values from shadPS4, unverified).
const ERROR_PARAMETER: u64 = 0x809F_0000;
const ERROR_EXISTS: u64 = 0x809F_0007;
const ERROR_NOT_FOUND: u64 = 0x809F_0008;
const ERROR_BUSY: u64 = 0x809F_000A;
const ERROR_NOT_MOUNTED: u64 = 0x809F_0004;

/// Offsets in the mount parameters and result.
const PARAM_DIR_NAME: u64 = 8;
const PARAM_BLOCKS: u64 = 16;
const PARAM_MODE: u64 = 32;
const RESULT_STATUS: usize = 28;

/// `mountStatus` when the save was made by this mount.
const STATUS_CREATED: u32 = 1;
/// Mount points a title may have at once.
const MOUNT_POINTS: usize = 16;
const DIR_NAME_MAX: usize = 31;

/// Which save folder each mount point holds.
#[derive(Default)]
pub struct Saves {
    mounted: std::sync::Mutex<[Option<Mounted>; MOUNT_POINTS]>,
}

struct Mounted {
    name: String,
    host: std::path::PathBuf,
    blocks: u64,
}

pub static FUNCTIONS: &[Function] = &[
    f("sceSaveDataMount3", mount3),
    f("sceSaveDataUmount", umount),
    f("sceSaveDataUmount2", umount2),
    f("sceSaveDataGetMountInfo", get_mount_info),
    f("sceSaveDataDirNameSearch", dir_name_search),
    // Name unknown; Minecraft passes it the same condition and result.
    f("#X4MYzukPc3g", dir_name_search),
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

fn valid_dir_name(name: &[u8]) -> bool {
    !name.is_empty()
        && name.len() <= DIR_NAME_MAX
        && name
            .iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
}

/// `sceSaveDataMount3(param, result)`.
fn mount3(env: &mut Env) -> Result<Outcome, CallError> {
    let (param, result) = (env.arg(0), env.arg(1));
    let name = env
        .mem
        .read_cstr(env.mem.read_u64(param + PARAM_DIR_NAME)?, DIR_NAME_MAX + 1)?;
    let mut mode = [0u8; 4];
    env.mem.read(param + PARAM_MODE, &mut mode)?;
    let mode = u32::from_le_bytes(mode);
    let Some(root) = env.kernel.vfs.saves.clone() else {
        return ret(ERROR_NOT_FOUND);
    };
    if !valid_dir_name(&name) {
        return ret(ERROR_PARAMETER);
    }
    let name = String::from_utf8_lossy(&name).into_owned();
    let host = root.join(&name);
    let mut mounted = lock(&env.kernel.saves.mounted);
    if mounted.iter().flatten().any(|m| m.name == name) {
        return ret(ERROR_BUSY);
    }
    let Some(slot) = mounted.iter().position(Option::is_none) else {
        return ret(ERROR_BUSY);
    };
    let exists = host.is_dir();
    let mut status = 0;
    match (exists, mode & MOUNT_CREATE != 0, mode & MOUNT_CREATE2 != 0) {
        (true, true, false) => return ret(ERROR_EXISTS),
        (false, false, false) => return ret(ERROR_NOT_FOUND),
        (false, _, _) => {
            std::fs::create_dir_all(&host)
                .map_err(|_| CallError::Abort("cannot make a save folder"))?;
            status = STATUS_CREATED;
        }
        _ => {}
    }
    let point = format!("/savedata{slot}");
    env.kernel.vfs.mount(&point, host.clone(), true);
    let blocks = env.mem.read_u64(param + PARAM_BLOCKS)?;
    mounted[slot] = Some(Mounted { name, host, blocks });
    drop(mounted);
    let mut out = [0u8; 32];
    out[..point.len()].copy_from_slice(point.as_bytes());
    out[RESULT_STATUS..RESULT_STATUS + 4].copy_from_slice(&status.to_le_bytes());
    env.mem.write(result, &out)?;
    ret(0)
}

/// The mount point named at `at` and its slot, if it is a valid one.
fn mount_point(env: &Env, at: u64) -> Result<Option<(String, usize)>, CallError> {
    let point = env.mem.read_cstr(at, 16)?;
    let point = String::from_utf8_lossy(&point).into_owned();
    let slot = point
        .strip_prefix("/savedata")
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n < MOUNT_POINTS);
    Ok(slot.map(|s| (point, s)))
}

fn unmount_point(env: &Env, at: u64) -> Result<Outcome, CallError> {
    let Some((point, slot)) = mount_point(env, at)? else {
        return ret(ERROR_PARAMETER);
    };
    let mut mounted = lock(&env.kernel.saves.mounted);
    if mounted[slot].take().is_none() || !env.kernel.vfs.unmount(&point) {
        return ret(ERROR_NOT_MOUNTED);
    }
    ret(0)
}

/// `sceSaveDataUmount(mount_point)`.
fn umount(env: &mut Env) -> Result<Outcome, CallError> {
    unmount_point(env, env.arg(0))
}

/// `sceSaveDataUmount2(mode, mount_point)`; the mode is ignored.
fn umount2(env: &mut Env) -> Result<Outcome, CallError> {
    unmount_point(env, env.arg(1))
}

/// Bytes in one save data block.
const BLOCK_SIZE: u64 = 32 * 1024;

fn used_bytes(dir: &std::path::Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => used_bytes(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

/// `sceSaveDataGetMountInfo(mount_point, info)`: blocks and free blocks
/// (two u64s), the size given at mount time minus what the files use.
fn get_mount_info(env: &mut Env) -> Result<Outcome, CallError> {
    let Some((_, slot)) = mount_point(env, env.arg(0))? else {
        return ret(ERROR_PARAMETER);
    };
    let Some((host, blocks)) = lock(&env.kernel.saves.mounted)[slot]
        .as_ref()
        .map(|m| (m.host.clone(), m.blocks))
    else {
        return ret(ERROR_NOT_MOUNTED);
    };
    let used = used_bytes(&host).div_ceil(BLOCK_SIZE);
    env.mem.write_u64(env.arg(1), blocks)?;
    env.mem
        .write_u64(env.arg(1) + 8, blocks.saturating_sub(used))?;
    ret(0)
}

/// Whether `name` matches search pattern `pat`, where `%` stands for any
/// run of characters and `_` for one.
fn matches(pat: &[u8], name: &[u8]) -> bool {
    match pat.split_first() {
        None => name.is_empty(),
        Some((b'%', rest)) => (0..=name.len()).any(|i| matches(rest, &name[i..])),
        Some((&c, rest)) => name
            .split_first()
            .is_some_and(|(&n, tail)| (c == b'_' || c == n) && matches(rest, tail)),
    }
}

/// Offsets in the search condition and result.
const COND_DIR_NAME: u64 = 16;
const COND_KEY: u64 = 24;
const COND_ORDER: u64 = 28;
const RESULT_HITS: u64 = 0;
const RESULT_DIR_NAMES: u64 = 8;
const RESULT_DIR_NAMES_NUM: u64 = 16;
const RESULT_SET: u64 = 20;
const DIR_NAME_SIZE: u64 = 32;

fn read_u32(env: &Env, at: u64) -> Result<u32, CallError> {
    let mut b = [0u8; 4];
    env.mem.read(at, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// `sceSaveDataDirNameSearch(cond, result)`; keys other than the name
/// sort by modification time.
fn dir_name_search(env: &mut Env) -> Result<Outcome, CallError> {
    let (cond, result) = (env.arg(0), env.arg(1));
    let pattern = match env.mem.read_u64(cond + COND_DIR_NAME)? {
        0 => b"%".to_vec(),
        at => env.mem.read_cstr(at, DIR_NAME_MAX + 1)?,
    };
    let (key, descending) = (
        read_u32(env, cond + COND_KEY)?,
        read_u32(env, cond + COND_ORDER)? != 0,
    );
    let Some(root) = env.kernel.vfs.saves.clone() else {
        return ret(ERROR_NOT_FOUND);
    };
    let mut found: Vec<(std::time::SystemTime, String)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let time = e.metadata().and_then(|m| m.modified()).ok()?;
            (valid_dir_name(name.as_bytes()) && matches(&pattern, name.as_bytes()))
                .then_some((time, name))
        })
        .collect();
    if key == 0 {
        found.sort_by(|a, b| a.1.cmp(&b.1));
    } else {
        found.sort();
    }
    if descending {
        found.reverse();
    }
    let names = env.mem.read_u64(result + RESULT_DIR_NAMES)?;
    let room = read_u32(env, result + RESULT_DIR_NAMES_NUM)? as usize;
    let set = if names == 0 { 0 } else { found.len().min(room) };
    for (i, (_, name)) in found.iter().take(set).enumerate() {
        let mut entry = [0u8; DIR_NAME_SIZE as usize];
        entry[..name.len()].copy_from_slice(name.as_bytes());
        env.mem.write(names + i as u64 * DIR_NAME_SIZE, &entry)?;
    }
    env.mem
        .write(result + RESULT_HITS, &(found.len() as u32).to_le_bytes())?;
    env.mem
        .write(result + RESULT_SET, &(set as u32).to_le_bytes())?;
    ret(0)
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn search_patterns_use_percent_and_underscore_wildcards() {
        assert!(matches(b"%", b"Save1"));
        assert!(matches(b"Save%", b"Save12"));
        assert!(matches(b"Sav_1", b"Save1"));
        assert!(!matches(b"Save_", b"Save12"));
        assert!(matches(b"Exact", b"Exact"));
        assert!(!matches(b"Exact", b"Exacto"));
    }
}
