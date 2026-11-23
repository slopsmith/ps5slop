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
/// Where `sceSaveDataMount2` (PS4) keeps the mount mode.
const PARAM2_MODE: u64 = 24;
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
    f("sceSaveDataInitialize3", accept),
    f("sceSaveDataMount2", mount2),
    f("sceSaveDataMount3", mount3),
    f("sceSaveDataUmount", umount),
    f("sceSaveDataUmount2", umount2),
    f("sceSaveDataGetMountInfo", get_mount_info),
    f("sceSaveDataSetParam", set_param),
    f("sceSaveDataGetParam", get_param),
    f("sceSaveDataDirNameSearch", dir_name_search),
    f("sceSaveDataSaveIcon", save_icon),
    f("sceSaveDataLoadIcon", load_icon),
    f("sceSaveDataSaveIconByPath", save_icon_by_path),
    f("sceSaveDataDelete", delete),
    // Name unknown; Minecraft passes it the same condition and result.
    f("#X4MYzukPc3g", dir_name_search),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f("sceSaveDataCreateTransactionResource", create_transaction),
    f("sceSaveDataDeleteTransactionResource", accept),
    f("sceSaveDataPrepare", accept),
    f("sceSaveDataTransferringMount", |_| ret(ERROR_NOT_FOUND)),
    f("sceSaveDataCommit", accept),
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

/// Transactions are not modelled: files are written straight through, so
/// preparing and committing have nothing to do.
fn accept(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// `sceSaveDataCreateTransactionResource(...)`: a positive id.
fn create_transaction(_: &mut Env) -> Result<Outcome, CallError> {
    ret(NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

// Minecraft's world saves contain `@`; the console's own rules are unknown.
fn valid_dir_name(name: &[u8]) -> bool {
    !name.is_empty()
        && name.len() <= DIR_NAME_MAX
        && name[0] != b'.'
        && name
            .iter()
            .all(|&c| c.is_ascii_graphic() && c != b'/' && c != b'\\')
}

/// `sceSaveDataMount3(param, result)`.
fn mount3(env: &mut Env) -> Result<Outcome, CallError> {
    mount(env, PARAM_MODE)
}

/// `sceSaveDataMount2(param, result)`, the PS4 call: the mode comes right
/// after the block count.
fn mount2(env: &mut Env) -> Result<Outcome, CallError> {
    mount(env, PARAM2_MODE)
}

fn mount(env: &mut Env, mode_at: u64) -> Result<Outcome, CallError> {
    let (param, result) = (env.arg(0), env.arg(1));
    let name = env
        .mem
        .read_cstr(env.mem.read_u64(param + PARAM_DIR_NAME)?, DIR_NAME_MAX + 1)?;
    let mut mode = [0u8; 4];
    env.mem.read(param + mode_at, &mut mode)?;
    let mode = u32::from_le_bytes(mode);
    let Some(root) = env.kernel.vfs.saves.clone() else {
        return ret(ERROR_NOT_FOUND);
    };
    if !valid_dir_name(&name) {
        return ret(ERROR_PARAMETER);
    }
    let name = String::from_utf8_lossy(&name).into_owned();
    tracing::debug!("mount {name} mode {mode:#x}");
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
    let record = host.join("sce_sys").join(BLOCKS_FILE);
    if status == STATUS_CREATED || !record.exists() {
        let _ = std::fs::create_dir_all(host.join("sce_sys"));
        let _ = std::fs::write(&record, blocks.to_string());
    }
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
/// Where a save folder keeps the size (in blocks) it was made with, our own
/// file next to the param files.
const BLOCKS_FILE: &str = "blocks";
/// The size of a save made before we recorded sizes: 1 GiB.
const DEFAULT_BLOCKS: u64 = (1 << 30) / BLOCK_SIZE;

/// The size a save folder was made with, in blocks.
fn save_blocks(dir: &std::path::Path) -> u64 {
    std::fs::read_to_string(dir.join("sce_sys").join(BLOCKS_FILE))
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(DEFAULT_BLOCKS)
}

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
const RESULT_PARAMS: u64 = 24;
const RESULT_INFOS: u64 = 32;
/// `OrbisSaveDataSearchInfo` (blocks, free blocks, reserved) and
/// `OrbisSaveDataParam` (title, subtitle, details, user parameter, mtime).
const INFO_SIZE: u64 = 48;
const PARAM_SIZE: usize = 1328;
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
    let mut found: Vec<(std::time::SystemTime, String)> = std::fs::read_dir(&root)
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
    tracing::debug!(
        "search {} by key {key}: {} found",
        String::from_utf8_lossy(&pattern),
        found.len()
    );
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
    let (params, infos) = (
        env.mem.read_u64(result + RESULT_PARAMS)?,
        env.mem.read_u64(result + RESULT_INFOS)?,
    );
    for (i, (_, name)) in found.iter().take(set).enumerate() {
        let mut entry = [0u8; DIR_NAME_SIZE as usize];
        entry[..name.len()].copy_from_slice(name.as_bytes());
        env.mem.write(names + i as u64 * DIR_NAME_SIZE, &entry)?;
        let dir = root.join(name);
        if infos != 0 {
            let blocks = save_blocks(&dir);
            let used = used_bytes(&dir).div_ceil(BLOCK_SIZE);
            let mut info = [0u8; INFO_SIZE as usize];
            info[..8].copy_from_slice(&blocks.to_le_bytes());
            info[8..16].copy_from_slice(&blocks.saturating_sub(used).to_le_bytes());
            env.mem.write(infos + i as u64 * INFO_SIZE, &info)?;
        }
        if params != 0 {
            env.mem
                .write(params + (i * PARAM_SIZE) as u64, &stamped_param(&dir))?;
        }
    }
    env.mem
        .write(result + RESULT_HITS, &(found.len() as u32).to_le_bytes())?;
    env.mem
        .write(result + RESULT_SET, &(set as u32).to_le_bytes())?;
    ret(0)
}

/// The host file holding parameter block `kind` of the save mounted at the
/// mount point named at `at`.
fn param_file(env: &Env, at: u64, kind: u64) -> Result<Option<std::path::PathBuf>, CallError> {
    let Some((_, slot)) = mount_point(env, at)? else {
        return Ok(None);
    };
    let mounted = lock(&env.kernel.saves.mounted);
    Ok(mounted[slot]
        .as_ref()
        .map(|m| m.host.join("sce_sys").join(format!("param{kind}.bin"))))
}

/// `sceSaveDataSetParam(mount_point, kind, buf, size)`: the bytes are kept
/// as they are, one file per kind (kind 0 is the whole title/detail block).
fn set_param(env: &mut Env) -> Result<Outcome, CallError> {
    let (kind, buf, size) = (env.arg(1), env.arg(2), env.arg(3).min(0x10000) as usize);
    let Some(file) = param_file(env, env.arg(0), kind)? else {
        return ret(ERROR_NOT_MOUNTED);
    };
    let mut data = vec![0u8; size];
    env.mem.read(buf, &mut data)?;
    let written = file
        .parent()
        .map(std::fs::create_dir_all)
        .transpose()
        .and_then(|_| std::fs::write(&file, &data));
    if written.is_err() {
        return Err(CallError::Abort("cannot write save data parameters"));
    }
    ret(0)
}

/// `sceSaveDataGetParam(mount_point, kind, buf, size, &got)`: what
/// `SetParam` stored, zero-filled when there is none.
fn get_param(env: &mut Env) -> Result<Outcome, CallError> {
    let (kind, buf, size, got) = (
        env.arg(1),
        env.arg(2),
        env.arg(3).min(0x10000) as usize,
        env.arg(4),
    );
    let Some(file) = param_file(env, env.arg(0), kind)? else {
        return ret(ERROR_NOT_MOUNTED);
    };
    let mut data = match (kind, file.parent().and_then(|d| d.parent())) {
        (0, Some(dir)) => stamped_param(dir),
        _ => std::fs::read(file).unwrap_or_default(),
    };
    data.resize(size, 0);
    env.mem.write(buf, &data)?;
    if got != 0 {
        env.mem.write_u64(got, size as u64)?;
    }
    ret(0)
}

/// Where `OrbisSaveDataParam` keeps `mtime`, which the system fills in.
const PARAM_MTIME: usize = 0x508;

/// The parameters `SetParam` stored for the save in `dir`, with `mtime` set
/// to when its newest file changed. GTA V loads no profile whose time is 0.
fn stamped_param(dir: &std::path::Path) -> Vec<u8> {
    let mut p = std::fs::read(dir.join("sce_sys").join("param0.bin")).unwrap_or_default();
    p.resize(PARAM_SIZE, 0);
    let secs = last_modified(dir)
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    p[PARAM_MTIME..PARAM_MTIME + 8].copy_from_slice(&secs.to_le_bytes());
    p
}

fn last_modified(dir: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            if m.is_dir() {
                last_modified(&e.path())
            } else {
                m.modified().ok()
            }
        })
        .max()
}

/// `OrbisSaveDataIcon`: buffer, its size and the data size in it.
const ICON_BUF: u64 = 0;
const ICON_BUF_SIZE: u64 = 8;
const ICON_DATA_SIZE: u64 = 16;
/// `OrbisSaveDataDelete`: the folder name.
const DELETE_DIR_NAME: u64 = 16;

/// The icon file of the save mounted at the mount point named at `at`.
fn icon_file(env: &Env, at: u64) -> Result<Option<std::path::PathBuf>, CallError> {
    let Some((_, slot)) = mount_point(env, at)? else {
        return Ok(None);
    };
    let mounted = lock(&env.kernel.saves.mounted);
    Ok(mounted[slot]
        .as_ref()
        .map(|m| m.host.join("sce_sys").join("icon0.png")))
}

/// `sceSaveDataSaveIcon(mount_point, icon)`.
fn save_icon(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(file) = icon_file(env, env.arg(0))? else {
        return ret(ERROR_NOT_MOUNTED);
    };
    let icon = env.arg(1);
    let size = env.mem.read_u64(icon + ICON_DATA_SIZE)?.min(16 << 20) as usize;
    let mut data = vec![0u8; size];
    env.mem
        .read(env.mem.read_u64(icon + ICON_BUF)?, &mut data)?;
    let _ = file.parent().map(std::fs::create_dir_all);
    std::fs::write(file, data).map_err(|_| CallError::Abort("cannot write a save icon"))?;
    ret(0)
}

/// `sceSaveDataSaveIconByPath(mount_point, path)`: the icon is a file the
/// title can read, such as `/app0/sce_sys/save_data.png` (name hashed from
/// a guess; the arguments read off GTA V's call).
fn save_icon_by_path(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(file) = icon_file(env, env.arg(0))? else {
        return ret(ERROR_NOT_MOUNTED);
    };
    let path = env.mem.read_cstr(env.arg(1), 1024)?;
    let Some(data) = env.kernel.vfs.read(&path) else {
        return ret(ERROR_NOT_FOUND);
    };
    let _ = file.parent().map(std::fs::create_dir_all);
    std::fs::write(file, data).map_err(|_| CallError::Abort("cannot write a save icon"))?;
    ret(0)
}

/// `sceSaveDataLoadIcon(mount_point, icon)`.
fn load_icon(env: &mut Env) -> Result<Outcome, CallError> {
    let Some(file) = icon_file(env, env.arg(0))? else {
        return ret(ERROR_NOT_MOUNTED);
    };
    let Ok(mut data) = std::fs::read(file) else {
        return ret(ERROR_NOT_FOUND);
    };
    let icon = env.arg(1);
    data.truncate(env.mem.read_u64(icon + ICON_BUF_SIZE)? as usize);
    env.mem.write(env.mem.read_u64(icon + ICON_BUF)?, &data)?;
    env.mem
        .write_u64(icon + ICON_DATA_SIZE, data.len() as u64)?;
    ret(0)
}

/// `sceSaveDataDelete(del)`: removes an unmounted save folder.
fn delete(env: &mut Env) -> Result<Outcome, CallError> {
    let name = env.mem.read_cstr(
        env.mem.read_u64(env.arg(0) + DELETE_DIR_NAME)?,
        DIR_NAME_MAX + 1,
    )?;
    if !valid_dir_name(&name) {
        return ret(ERROR_PARAMETER);
    }
    let name = String::from_utf8_lossy(&name).into_owned();
    tracing::debug!("delete {name}");
    let Some(root) = env.kernel.vfs.saves.clone() else {
        return ret(ERROR_NOT_FOUND);
    };
    if lock(&env.kernel.saves.mounted)
        .iter()
        .flatten()
        .any(|m| m.name == name)
    {
        return ret(ERROR_BUSY);
    }
    match std::fs::remove_dir_all(root.join(&name)) {
        Ok(()) => ret(0),
        Err(_) => ret(ERROR_NOT_FOUND),
    }
}

savestate::record!(Mounted { name, host, blocks });

impl Saves {
    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        let mounted = lock(&self.mounted);
        e.put(&*mounted)
            .put(&NEXT_ID.load(std::sync::atomic::Ordering::Relaxed));
    }

    /// Mounts the saved mount points again, in this run's save folder.
    pub(crate) fn load(
        &self,
        d: &mut savestate::Decoder,
        vfs: &crate::Vfs,
    ) -> Result<(), savestate::Error> {
        let mut saved: [Option<Mounted>; MOUNT_POINTS] = d.get()?;
        for (slot, m) in saved.iter_mut().enumerate() {
            let Some(m) = m else { continue };
            if let (Some(root), Some(leaf)) = (&vfs.saves, m.host.file_name()) {
                m.host = root.join(leaf);
            }
            let _ = std::fs::create_dir_all(&m.host);
            vfs.mount(&format!("/savedata{slot}"), m.host.clone(), true);
        }
        *lock(&self.mounted) = saved;
        NEXT_ID.store(d.get()?, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PARAM_MTIME, PARAM_SIZE, matches, stamped_param, valid_dir_name};

    #[test]
    fn params_carry_the_time_the_save_last_changed() {
        let dir = std::env::temp_dir().join(format!("ps5slop-param-mtime-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sce_sys")).unwrap();
        std::fs::write(dir.join("sce_sys").join("param0.bin"), b"Title").unwrap();
        std::fs::write(dir.join("Profile"), b"data").unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let p = stamped_param(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(p.len(), PARAM_SIZE);
        assert_eq!(&p[..5], b"Title");
        let mtime = u64::from_le_bytes(p[PARAM_MTIME..PARAM_MTIME + 8].try_into().unwrap());
        assert!(
            now.abs_diff(mtime) < 60,
            "saves look never written, so GTA V drops its settings (mtime {mtime})"
        );
    }

    #[test]
    fn save_names_may_hold_at_signs_but_not_paths() {
        assert!(valid_dir_name(b"BedrockWorld2QAAC4ePE14@P1"));
        assert!(!valid_dir_name(b"a/b"));
        assert!(!valid_dir_name(b".."));
        assert!(!valid_dir_name(b"with space"));
    }

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
