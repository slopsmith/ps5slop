//! The guest's file system: mount points backed by host folders. `/app0`
//! is the title folder, read-only; `/data` is the title's writable storage.

use std::path::{Component, Path, PathBuf};

use crate::errno;

struct Mount {
    guest: &'static str,
    host: PathBuf,
    writable: bool,
}

#[derive(Default)]
pub struct Vfs {
    mounts: Vec<Mount>,
}

impl Vfs {
    /// A file system with the title folder at `/app0` and `data` at `/data`.
    pub fn new(app0: Option<PathBuf>, data: Option<PathBuf>) -> Self {
        let mut mounts = Vec::new();
        if let Some(host) = app0 {
            mounts.push(Mount {
                guest: "/app0",
                host,
                writable: false,
            });
        }
        if let Some(host) = data {
            mounts.push(Mount {
                guest: "/data",
                host,
                writable: true,
            });
        }
        Self { mounts }
    }

    /// Adds a mount point: guest path `guest` backed by host folder `host`.
    pub fn mount(&mut self, guest: &'static str, host: PathBuf, writable: bool) {
        self.mounts.push(Mount {
            guest,
            host,
            writable,
        });
    }

    /// The host path for guest path `path`, and whether it may be written;
    /// `None` outside every mount, or for paths that climb out with `..`.
    pub fn resolve(&self, path: &[u8]) -> Option<(PathBuf, bool)> {
        let path = std::str::from_utf8(path).ok()?;
        self.mounts.iter().find_map(|m| {
            let rest = path.strip_prefix(m.guest)?;
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
