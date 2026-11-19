//! The guest's file system: mount points backed by host folders. `/app0`
//! is the title folder, read-only; `/data` is the title's writable storage;
//! save data appears at `/savedataN` while mounted.

use std::path::{Component, Path, PathBuf};
use std::sync::RwLock;

use crate::errno;

struct Mount {
    guest: String,
    host: PathBuf,
    writable: bool,
}

#[derive(Default)]
pub struct Vfs {
    mounts: RwLock<Vec<Mount>>,
    /// Where save data lives on the host, one folder per save.
    pub(crate) saves: Option<PathBuf>,
}

impl Vfs {
    /// A file system with the title folder at `/app0` and `data` at `/data`.
    pub fn new(app0: Option<PathBuf>, data: Option<PathBuf>) -> Self {
        let vfs = Self::default();
        if let Some(host) = app0 {
            vfs.mount("/app0", host, false);
        }
        if let Some(host) = data {
            vfs.mount("/data", host, true);
        }
        vfs
    }

    /// Adds a mount point: guest path `guest` backed by host folder `host`.
    pub fn mount(&self, guest: &str, host: PathBuf, writable: bool) {
        let mut mounts = self.mounts.write().unwrap_or_else(|e| e.into_inner());
        mounts.retain(|m| m.guest != guest);
        mounts.push(Mount {
            guest: guest.to_owned(),
            host,
            writable,
        });
    }

    /// Removes mount point `guest`; false if nothing was mounted there.
    pub fn unmount(&self, guest: &str) -> bool {
        let mut mounts = self.mounts.write().unwrap_or_else(|e| e.into_inner());
        let before = mounts.len();
        mounts.retain(|m| m.guest != guest);
        mounts.len() != before
    }

    /// Whether something is mounted at `guest`.
    pub fn is_mounted(&self, guest: &str) -> bool {
        let mounts = self.mounts.read().unwrap_or_else(|e| e.into_inner());
        mounts.iter().any(|m| m.guest == guest)
    }

    /// Keeps save data in host folder `saves`.
    pub fn set_saves(&mut self, saves: PathBuf) {
        self.saves = Some(saves);
    }

    /// The host path for guest path `path`, and whether it may be written;
    /// `None` outside every mount, or for paths that climb out with `..`.
    pub fn resolve(&self, path: &[u8]) -> Option<(PathBuf, bool)> {
        let path = std::str::from_utf8(path).ok()?;
        let mounts = self.mounts.read().unwrap_or_else(|e| e.into_inner());
        mounts.iter().find_map(|m| {
            let rest = path.strip_prefix(m.guest.as_str())?;
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
            Some((m.host.join(rest), m.writable))
        })
    }

    /// Makes the directory `path`, returning an errno on failure.
    pub fn mkdir(&self, path: &[u8]) -> Result<(), i32> {
        match self.resolve(path) {
            None => Err(errno::ENOENT),
            Some((_, false)) => Err(errno::EROFS),
            Some((host, true)) => std::fs::create_dir(host).map_err(|e| io_errno(&e)),
        }
    }
}

/// The FreeBSD errno for a host I/O error.
pub(crate) fn io_errno(e: &std::io::Error) -> i32 {
    use std::io::ErrorKind::*;
    match e.kind() {
        NotFound => errno::ENOENT,
        AlreadyExists => errno::EEXIST,
        PermissionDenied => errno::EACCES,
        IsADirectory => errno::EISDIR,
        NotADirectory => errno::ENOTDIR,
        _ => errno::EIO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_resolve_inside_their_mount_only() {
        let vfs = Vfs::new(Some("/titles/X".into()), Some("/saves/X".into()));
        let get = |p: &[u8]| vfs.resolve(p);
        assert_eq!(
            get(b"/app0/sce_sys/param.json"),
            Some(("/titles/X/sce_sys/param.json".into(), false))
        );
        assert_eq!(get(b"/app0"), Some(("/titles/X".into(), false)));
        assert_eq!(
            get(b"/data/log.txt"),
            Some(("/saves/X/log.txt".into(), true))
        );
        assert_eq!(get(b"/app0/../etc/passwd"), None);
        assert_eq!(get(b"/data/../../etc"), None);
        assert_eq!(get(b"/app0x/file"), None);
        assert_eq!(get(b"/system/file"), None);
        assert_eq!(Vfs::default().resolve(b"/app0/file"), None);
    }

    #[test]
    fn directories_can_only_be_made_in_writable_mounts() {
        let tmp = std::env::temp_dir().join(format!("ps5slop-vfs-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let vfs = Vfs::new(Some(tmp.clone()), Some(tmp.clone()));
        assert_eq!(vfs.mkdir(b"/data/logs"), Ok(()));
        assert_eq!(vfs.mkdir(b"/data/logs"), Err(errno::EEXIST));
        assert_eq!(vfs.mkdir(b"/data/a/b"), Err(errno::ENOENT));
        assert_eq!(vfs.mkdir(b"/app0/x"), Err(errno::EROFS));
        std::fs::remove_dir_all(tmp).unwrap();
    }
}
