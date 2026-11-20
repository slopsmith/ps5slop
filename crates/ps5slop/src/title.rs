//! What `ps5slop run` was given: a lone executable, an app folder, or a
//! package whose files are read in place (see `docs/packages.md`).

use std::collections::HashMap;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use hle::{ImageFile, ImageFs, ImageStat};
use pkg::keys::Passcode;
use pkg::pfs::{DirEntry, Kind, Pfs};
use pkg::{Entry, Package, ReadAt};

use crate::modules::Files;

pub enum Title {
    /// A plain or fake-signed ELF on its own.
    Executable(PathBuf),
    /// A folder holding `eboot.bin` (an `app0` dump).
    Folder(PathBuf),
    Package {
        path: PathBuf,
        fs: Arc<PackageFs>,
        id: Option<String>,
    },
}

impl Title {
    /// Opens `path`: a folder, a `.pkg` file (opened with `passcode` if it is
    /// a PS4 package) or an executable.
    pub fn open(path: &Path, passcode: &Passcode) -> Result<Self, String> {
        if path.is_dir() {
            return Ok(Title::Folder(path.to_path_buf()));
        }
        let is_pkg = path
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("pkg"));
        if !is_pkg {
            return Ok(Title::Executable(path.to_path_buf()));
        }
        let package =
            Package::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let id = package.title_id();
        let fs = PackageFs::new(package, passcode)
            .map_err(|e| format!("cannot read the files in {}: {e}", path.display()))?;
        Ok(Title::Package {
            path: path.to_path_buf(),
            fs: Arc::new(fs),
            id,
        })
    }

    /// The executable's bytes.
    pub fn eboot(&self) -> io::Result<Vec<u8>> {
        match self {
            Title::Executable(p) => std::fs::read(p),
            Title::Folder(d) => std::fs::read(d.join("eboot.bin")),
            Title::Package { fs, .. } => Image(&**fs).read("eboot.bin"),
        }
    }

    /// Where the title's modules are looked for.
    pub fn files(&self) -> Option<Box<dyn Files + '_>> {
        match self {
            Title::Executable(p) => p.parent().map(|d| Box::new(d) as Box<dyn Files>),
            Title::Folder(d) => Some(Box::new(d.as_path())),
            Title::Package { fs, .. } => Some(Box::new(Image(&**fs))),
        }
    }

    /// Mounts the title at `/app0`.
    pub fn mount(&self, vfs: &hle::Vfs) {
        match self {
            Title::Executable(p) => {
                if let Some(d) = p.parent() {
                    vfs.mount("/app0", d.to_path_buf(), false);
                }
            }
            Title::Folder(d) => vfs.mount("/app0", d.clone(), false),
            Title::Package { fs, .. } => vfs.mount_image("/app0", fs.clone()),
        }
    }

    /// The title id from `param.json` or `param.sfo` (or a package's content
    /// id), if there is one.
    pub fn id(&self) -> Option<String> {
        match self {
            Title::Executable(_) => None,
            Title::Folder(d) => {
                let sce_sys = d.join("sce_sys");
                std::fs::read(sce_sys.join("param.json"))
                    .ok()
                    .and_then(|j| pkg::title::from_param_json(&j))
                    .or_else(|| {
                        std::fs::read(sce_sys.join("param.sfo"))
                            .ok()
                            .and_then(|s| pkg::title::from_param_sfo(&s))
                    })
            }
            Title::Package { id, .. } => id.clone(),
        }
    }

    /// The name used for the title's data folders when it has no title id:
    /// the folder's name, or the file's without its extension.
    pub fn fallback_name(&self) -> Option<OsString> {
        match self {
            Title::Executable(p) | Title::Package { path: p, .. } => p.file_stem().map(Into::into),
            Title::Folder(d) => d.file_name().map(Into::into),
        }
    }
}

/// A file in the package: from the PFS image, or a metadata entry.
enum Node {
    Image(DirEntry),
    Entry(Entry),
    /// A folder that holds only metadata entries.
    EntryDir,
}

/// A package's app files, read in place: the PFS image's `uroot`, plus the
/// metadata entries for `sce_sys/` that the image does not hold.
pub struct PackageFs {
    package: Package,
    pfs: Pfs,
    root: u32,
    /// Metadata entries by path below the root (`sce_sys/param.json`).
    entries: HashMap<String, Entry>,
    dirs: Mutex<HashMap<u32, Arc<Vec<DirEntry>>>>,
}

impl PackageFs {
    pub fn new(package: Package, passcode: &Passcode) -> pkg::Result<Self> {
        let pfs = package.app_pfs(passcode)?;
        let root = pfs.uroot()?;
        let entries = package
            .sce_sys_entries()
            .into_iter()
            .map(|(name, e)| (format!("sce_sys/{name}"), e))
            .collect();
        Ok(Self {
            package,
            pfs,
            root,
            entries,
            dirs: Mutex::default(),
        })
    }

    fn read_dir(&self, inode: u32) -> Option<Arc<Vec<DirEntry>>> {
        let mut dirs = self.dirs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(d) = dirs.get(&inode) {
            return Some(d.clone());
        }
        let list = Arc::new(self.pfs.read_dir(inode).ok()?);
        dirs.insert(inode, list.clone());
        Some(list)
    }

    fn image_node(&self, path: &str) -> Option<DirEntry> {
        let mut node = DirEntry {
            name: String::new(),
            inode: self.root,
            kind: Kind::Dir,
        };
        for part in path.split('/').filter(|p| !p.is_empty()) {
            if node.kind != Kind::Dir {
                return None;
            }
            node = self
                .read_dir(node.inode)?
                .iter()
                .find(|e| e.name == part)?
                .clone();
        }
        Some(node)
    }

    fn node(&self, path: &str) -> Option<Node> {
        if let Some(n) = self.image_node(path) {
            return Some(Node::Image(n));
        }
        if let Some(e) = self.entries.get(path) {
            return Some(Node::Entry(*e));
        }
        let prefix = format!("{path}/");
        self.entries
            .keys()
            .any(|k| k.starts_with(&prefix))
            .then_some(Node::EntryDir)
    }
}

impl ImageFs for PackageFs {
    fn stat(&self, path: &str) -> Option<ImageStat> {
        Some(match self.node(path)? {
            Node::Image(DirEntry {
                kind: Kind::Dir, ..
            })
            | Node::EntryDir => ImageStat { dir: true, len: 0 },
            Node::Image(e) => {
                let inode = self.pfs.inode(e.inode).ok()?;
                let len = if inode.is_compressed() {
                    self.pfs.open_file(e.inode).ok()?.len()
                } else {
                    inode.size
                };
                ImageStat { dir: false, len }
            }
            Node::Entry(e) => ImageStat {
                dir: false,
                len: e.size.into(),
            },
        })
    }

    fn list(&self, path: &str) -> Option<Vec<(String, bool)>> {
        let mut out: Vec<(String, bool)> = match self.node(path)? {
            Node::Image(DirEntry {
                kind: Kind::Dir,
                inode,
                ..
            }) => self
                .read_dir(inode)?
                .iter()
                .map(|e| (e.name.clone(), e.kind == Kind::Dir))
                .collect(),
            Node::EntryDir => Vec::new(),
            _ => return None,
        };
        let prefix = if path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };
        for key in self.entries.keys() {
            let Some(rest) = key.strip_prefix(&prefix) else {
                continue;
            };
            let (name, is_dir) = match rest.split_once('/') {
                Some((dir, _)) => (dir, true),
                None => (rest, false),
            };
            if !out.iter().any(|(n, _)| n == name) {
                out.push((name.to_owned(), is_dir));
            }
        }
        Some(out)
    }

    fn open(&self, path: &str) -> Option<Arc<dyn ImageFile>> {
        match self.node(path)? {
            Node::Image(e) if e.kind == Kind::File => {
                Some(Arc::new(PfsFile(self.pfs.open_file(e.inode).ok()?)))
            }
            Node::Entry(e) => Some(Arc::new(PfsFile(Arc::new(
                self.package.read_entry(&e).ok()?,
            )))),
            _ => None,
        }
    }
}

struct PfsFile(Arc<dyn ReadAt + Send + Sync>);

impl ImageFile for PfsFile {
    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.0.len().saturating_sub(offset).min(buf.len() as u64) as usize;
        self.0
            .read_exact_at(offset, &mut buf[..n])
            .map_err(io::Error::other)?;
        Ok(n)
    }
}

/// An [`ImageFs`] seen as the place modules come from.
struct Image<'a>(&'a dyn ImageFs);

impl Files for Image<'_> {
    fn list(&self, dir: &str) -> Vec<String> {
        self.0
            .list(dir)
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, is_dir)| !is_dir)
            .map(|(name, _)| name)
            .collect()
    }

    fn folders(&self, dir: &str) -> Vec<String> {
        self.0
            .list(dir)
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, is_dir)| *is_dir)
            .map(|(name, _)| name)
            .collect()
    }

    fn read(&self, path: &str) -> io::Result<Vec<u8>> {
        let f = self.0.open(path).ok_or(io::ErrorKind::NotFound)?;
        let len = usize::try_from(f.len()).map_err(io::Error::other)?;
        let mut out = vec![0; len];
        let mut done = 0;
        while done < len {
            match f.read_at(done as u64, &mut out[done..])? {
                0 => return Err(io::ErrorKind::UnexpectedEof.into()),
                n => done += n,
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads every file of the package `PS5SLOP_TEST_PKG` names through the
    /// guest's file system and checks it against an extraction of the same
    /// package in `PS5SLOP_TEST_DIR`.
    #[test]
    #[ignore = "needs a package and its extraction"]
    fn package_files_match_their_extraction() {
        let pkg = PathBuf::from(std::env::var_os("PS5SLOP_TEST_PKG").unwrap());
        let dir = PathBuf::from(std::env::var_os("PS5SLOP_TEST_DIR").unwrap());
        let title = Title::open(&pkg, &pkg::keys::DEFAULT_PASSCODE).unwrap();
        let vfs = hle::Vfs::default();
        title.mount(&vfs);
        let mut pending = vec![String::new()];
        let mut seen = 0;
        while let Some(at) = pending.pop() {
            let Some(hle::Target::Image(fs, _)) = vfs.lookup(format!("/app0/{at}").as_bytes())
            else {
                panic!("/app0/{at} is not in the image");
            };
            for (name, is_dir) in fs.list(&at).unwrap() {
                let path = if at.is_empty() {
                    name
                } else {
                    format!("{at}/{name}")
                };
                if is_dir {
                    pending.push(path);
                    continue;
                }
                let ours = vfs.read(format!("/app0/{path}").as_bytes()).unwrap();
                assert_eq!(ours, std::fs::read(dir.join(&path)).unwrap(), "{path}");
                seen += 1;
            }
        }
        assert_eq!(seen, walk(&dir));
        assert_eq!(
            title.eboot().unwrap(),
            std::fs::read(dir.join("eboot.bin")).unwrap()
        );
    }

    fn walk(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| {
                if e.path().is_dir() {
                    walk(&e.path())
                } else {
                    1
                }
            })
            .sum()
    }
}
