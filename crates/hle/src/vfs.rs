//! The guest's file system: mount points backed by host folders or by
//! read-only images (a package's PFS). `/app0` is the title, read-only;
//! `/data` is the title's writable storage; save data appears at
//! `/savedataN` while mounted.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use crate::errno;

/// A read-only file tree that is not a host folder, such as a package's
/// image. Paths are `/`-separated and relative to its root, which is `""`.
pub trait ImageFs: Send + Sync {
    fn stat(&self, path: &str) -> Option<ImageStat>;
    /// The names in directory `path`, each with whether it is a directory.
    fn list(&self, path: &str) -> Option<Vec<(String, bool)>>;
    fn open(&self, path: &str) -> Option<Arc<dyn ImageFile>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageStat {
    pub dir: bool,
    pub len: u64,
}

/// A file inside an [`ImageFs`].
pub trait ImageFile: Send + Sync {
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads up to `buf.len()` bytes at `offset`; 0 at or past the end.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize>;
}

#[derive(Clone)]
enum Backing {
    Host(PathBuf),
    Image(Arc<dyn ImageFs>),
}

struct Mount {
    guest: String,
    backing: Backing,
    writable: bool,
}

/// Where a guest path leads.
#[derive(Clone)]
pub enum Target {
    /// A host path, and whether it may be written.
    Host(PathBuf, bool),
    /// A path inside a read-only image.
    Image(Arc<dyn ImageFs>, String),
}

fn exists(target: &Target) -> bool {
    match target {
        Target::Host(p, _) => p.exists(),
        Target::Image(fs, p) => fs.stat(p).is_some(),
    }
}

/// What `stat` reports about a file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Meta {
    pub dir: bool,
    pub len: u64,
    pub modified: Option<SystemTime>,
}

impl From<&std::fs::Metadata> for Meta {
    fn from(m: &std::fs::Metadata) -> Self {
        Self {
            dir: m.is_dir(),
            len: m.len(),
            modified: m.modified().ok(),
        }
    }
}

/// An open guest file: a host file, or a file or directory in an image.
pub(crate) enum GuestFile {
    Host(File),
    Image {
        /// `None` for a directory.
        file: Option<Arc<dyn ImageFile>>,
        pos: u64,
    },
}

impl GuestFile {
    pub(crate) fn meta(&self) -> io::Result<Meta> {
        match self {
            GuestFile::Host(f) => Ok(Meta::from(&f.metadata()?)),
            GuestFile::Image { file, .. } => Ok(Meta {
                dir: file.is_none(),
                len: file.as_ref().map_or(0, |f| f.len()),
                modified: None,
            }),
        }
    }

    pub(crate) fn sync_all(&mut self) -> io::Result<()> {
        match self {
            GuestFile::Host(f) => f.sync_all(),
            GuestFile::Image { .. } => Ok(()),
        }
    }

    pub(crate) fn set_len(&mut self, len: u64) -> io::Result<()> {
        match self {
            GuestFile::Host(f) => f.set_len(len),
            GuestFile::Image { .. } => Err(io::ErrorKind::ReadOnlyFilesystem.into()),
        }
    }
}

impl Read for GuestFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            GuestFile::Host(f) => f.read(buf),
            GuestFile::Image { file: None, .. } => Err(io::ErrorKind::IsADirectory.into()),
            GuestFile::Image { file: Some(f), pos } => {
                let n = f.read_at(*pos, buf)?;
                *pos += n as u64;
                Ok(n)
            }
        }
    }
}

impl Write for GuestFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            GuestFile::Host(f) => f.write(buf),
            GuestFile::Image { .. } => Err(io::ErrorKind::ReadOnlyFilesystem.into()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            GuestFile::Host(f) => f.flush(),
            GuestFile::Image { .. } => Ok(()),
        }
    }
}

impl Seek for GuestFile {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let (file, pos) = match self {
            GuestFile::Host(f) => return f.seek(to),
            GuestFile::Image { file, pos } => (file, pos),
        };
        let len = file.as_ref().map_or(0, |f| f.len());
        let target = match to {
            SeekFrom::Start(n) => Some(n),
            SeekFrom::Current(d) => pos.checked_add_signed(d),
            SeekFrom::End(d) => len.checked_add_signed(d),
        };
        *pos = target.ok_or(io::ErrorKind::InvalidInput)?;
        Ok(*pos)
    }
}

/// A directory being listed with `getdents`.
pub(crate) enum DirSource {
    Host(PathBuf),
    Image(Arc<dyn ImageFs>, String),
}

impl DirSource {
    /// The directory's entries, sorted, each with whether it is a directory.
    pub(crate) fn names(&self) -> Vec<(String, bool)> {
        let mut names = match self {
            DirSource::Host(path) => std::fs::read_dir(path)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|e| {
                    Some((
                        e.file_name().into_string().ok()?,
                        e.file_type().ok()?.is_dir(),
                    ))
                })
                .collect(),
            DirSource::Image(fs, path) => fs.list(path).unwrap_or_default(),
        };
        names.sort();
        names
    }
}

#[derive(Default)]
pub struct Vfs {
    mounts: RwLock<Vec<Mount>>,
    /// Where save data lives on the host, one folder per save.
    pub(crate) saves: Option<PathBuf>,
    /// Host folder with the open-source stand-ins for system fonts.
    system_fonts: Option<PathBuf>,
}

/// System fonts titles name as if they were their own files, and the
/// open-source stand-in for each (see `scripts/fetch-fonts`): file name
/// prefix, stand-in.
const SYSTEM_FONTS: &[(&str, &str)] = &[
    ("SIE-ShinGo", "NotoSansJP[wght].ttf"),
    ("HeiseiMaruGo", "MPLUSRounded1c-Bold.ttf"),
    ("Mohr-", "Jost-Italic[wght].ttf"),
    ("FuturaStd-", "Jost[wght].ttf"),
    ("FuturaNowText-", "Jost[wght].ttf"),
];

/// The stand-in for every system font set (`sceFontOpenFontSet`): Noto
/// Sans JP covers Latin and Japanese.
const FONT_SET_STAND_IN: &str = "NotoSansJP[wght].ttf";

/// The stand-in for a system font file named `name`.
fn system_font(name: &str) -> Option<&'static str> {
    SYSTEM_FONTS
        .iter()
        .find(|(prefix, _)| name.starts_with(prefix))
        .map(|&(_, file)| file)
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
        self.add(guest, Backing::Host(host), writable);
    }

    /// Adds a read-only mount point backed by an image.
    pub fn mount_image(&self, guest: &str, image: Arc<dyn ImageFs>) {
        self.add(guest, Backing::Image(image), false);
    }

    fn add(&self, guest: &str, backing: Backing, writable: bool) {
        let mut mounts = self.mounts.write().unwrap_or_else(|e| e.into_inner());
        mounts.retain(|m| m.guest != guest);
        mounts.push(Mount {
            guest: guest.to_owned(),
            backing,
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

    /// Serves system fonts a title asks for under `/app0` from host folder
    /// `fonts` when the title does not have the file itself.
    pub fn set_system_fonts(&mut self, fonts: PathBuf) {
        self.system_fonts = Some(fonts);
    }

    /// The host file that stands in for system font sets, if a fonts folder
    /// is set.
    pub(crate) fn font_set_file(&self) -> Option<PathBuf> {
        Some(self.system_fonts.as_ref()?.join(FONT_SET_STAND_IN))
    }

    /// Keeps save data in host folder `saves`.
    pub fn set_saves(&mut self, saves: PathBuf) {
        self.saves = Some(saves);
    }

    /// Where guest path `path` leads; `None` outside every mount, or for
    /// paths that climb out with `..`.
    pub fn lookup(&self, path: &[u8]) -> Option<Target> {
        let target = self.lookup_mounted(path)?;
        let stand_in = std::str::from_utf8(path)
            .ok()
            .filter(|p| p.starts_with("/app0/"))
            .and_then(|p| system_font(p.rsplit('/').next()?))
            .zip(self.system_fonts.as_ref());
        match stand_in {
            Some((file, fonts)) if !exists(&target) => Some(Target::Host(fonts.join(file), false)),
            _ => Some(target),
        }
    }

    fn lookup_mounted(&self, path: &[u8]) -> Option<Target> {
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
            Some(match &m.backing {
                Backing::Host(host) => Target::Host(host.join(rest), m.writable),
                Backing::Image(fs) => {
                    let parts: Vec<&str> = rest.iter().filter_map(|c| c.to_str()).collect();
                    Target::Image(fs.clone(), parts.join("/"))
                }
            })
        })
    }

    /// The host path for guest path `path`, and whether it may be written;
    /// `None` outside every host-backed mount.
    pub fn resolve(&self, path: &[u8]) -> Option<(PathBuf, bool)> {
        match self.lookup(path)? {
            Target::Host(p, w) => Some((p, w)),
            Target::Image(..) => None,
        }
    }

    /// The whole contents of the file at guest path `path`.
    pub fn read(&self, path: &[u8]) -> Option<Vec<u8>> {
        match self.lookup(path)? {
            Target::Host(p, _) => std::fs::read(p).ok(),
            Target::Image(fs, p) => {
                let f = fs.open(&p)?;
                let mut out = vec![0; usize::try_from(f.len()).ok()?];
                let mut done = 0;
                while done < out.len() {
                    match f.read_at(done as u64, &mut out[done..]).ok()? {
                        0 => return None,
                        n => done += n,
                    }
                }
                Some(out)
            }
        }
    }

    pub(crate) fn metadata(&self, path: &[u8]) -> Option<Meta> {
        match self.lookup(path)? {
            Target::Host(p, _) => std::fs::metadata(p).ok().map(|m| Meta::from(&m)),
            Target::Image(fs, p) => fs.stat(&p).map(|s| Meta {
                dir: s.dir,
                len: s.len,
                modified: None,
            }),
        }
    }

    /// Opens guest path `path` with `options`; `writes` says whether the
    /// caller wants to change the file, which read-only mounts refuse. A
    /// directory also gives what `getdents` lists.
    pub(crate) fn open(
        &self,
        path: &[u8],
        options: &OpenOptions,
        writes: bool,
    ) -> Result<(GuestFile, Option<DirSource>), i32> {
        match self.lookup(path).ok_or(errno::ENOENT)? {
            Target::Host(_, false) | Target::Image(..) if writes => Err(errno::EROFS),
            Target::Host(host, _) => {
                let file = options.open(&host).map_err(|e| io_errno(&e))?;
                let dir = host.is_dir().then_some(DirSource::Host(host));
                Ok((GuestFile::Host(file), dir))
            }
            Target::Image(fs, p) => {
                let stat = fs.stat(&p).ok_or(errno::ENOENT)?;
                if stat.dir {
                    let file = GuestFile::Image { file: None, pos: 0 };
                    return Ok((file, Some(DirSource::Image(fs, p))));
                }
                let file = fs.open(&p).ok_or(errno::EIO)?;
                Ok((
                    GuestFile::Image {
                        file: Some(file),
                        pos: 0,
                    },
                    None,
                ))
            }
        }
    }

    /// Makes the directory `path`, returning an errno on failure.
    pub fn mkdir(&self, path: &[u8]) -> Result<(), i32> {
        match self.lookup(path) {
            None => Err(errno::ENOENT),
            Some(Target::Host(host, true)) => std::fs::create_dir(host).map_err(|e| io_errno(&e)),
            Some(_) => Err(errno::EROFS),
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
        ReadOnlyFilesystem => errno::EROFS,
        InvalidInput => errno::EINVAL,
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

    struct Tree;

    struct Bytes(&'static [u8]);

    impl ImageFile for Bytes {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
            let rest = self.0.get(offset as usize..).unwrap_or_default();
            let n = rest.len().min(buf.len());
            buf[..n].copy_from_slice(&rest[..n]);
            Ok(n)
        }
    }

    impl ImageFs for Tree {
        fn stat(&self, path: &str) -> Option<ImageStat> {
            match path {
                "" | "sce_sys" => Some(ImageStat { dir: true, len: 0 }),
                "eboot.bin" => Some(ImageStat { dir: false, len: 5 }),
                _ => None,
            }
        }

        fn list(&self, path: &str) -> Option<Vec<(String, bool)>> {
            (path.is_empty()).then(|| vec![("sce_sys".into(), true), ("eboot.bin".into(), false)])
        }

        fn open(&self, path: &str) -> Option<Arc<dyn ImageFile>> {
            (path == "eboot.bin").then(|| Arc::new(Bytes(b"hello")) as Arc<dyn ImageFile>)
        }
    }

    #[test]
    fn image_mounts_serve_reads_and_refuse_writes() {
        let vfs = Vfs::default();
        vfs.mount_image("/app0", Arc::new(Tree));
        let read = OpenOptions::new().read(true).clone();
        let (mut f, dir) = vfs.open(b"/app0/eboot.bin", &read, false).unwrap();
        assert!(dir.is_none());
        let mut buf = [0u8; 8];
        assert_eq!(f.read(&mut buf).unwrap(), 5);
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(f.seek(SeekFrom::End(-2)).unwrap(), 3);
        assert_eq!(f.read(&mut buf).unwrap(), 2);
        assert_eq!(f.read(&mut buf).unwrap(), 0);
        assert!(f.seek(SeekFrom::Current(-10)).is_err());
        assert!(f.write(b"x").is_err());
        assert_eq!(f.meta().unwrap().len, 5);

        assert_eq!(
            vfs.open(b"/app0/eboot.bin", &read, true).err(),
            Some(errno::EROFS)
        );
        assert_eq!(
            vfs.open(b"/app0/missing", &read, false).err(),
            Some(errno::ENOENT)
        );
        assert_eq!(vfs.mkdir(b"/app0/new"), Err(errno::EROFS));
        assert_eq!(vfs.read(b"/app0/eboot.bin").as_deref(), Some(&b"hello"[..]));
        assert!(vfs.metadata(b"/app0/sce_sys").unwrap().dir);
        assert_eq!(vfs.resolve(b"/app0/eboot.bin"), None);

        let (mut d, dir) = vfs.open(b"/app0", &read, false).unwrap();
        assert!(d.read(&mut buf).is_err());
        let names = dir.unwrap().names();
        assert_eq!(
            names,
            [("eboot.bin".into(), false), ("sce_sys".into(), true)]
        );
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
