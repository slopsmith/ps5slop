//! What `ps5slop run` was given: a lone executable, an app folder, or a
//! package whose files are read in place, with a PS4 patch package over it
//! (see `docs/packages.md`).

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
    /// A base package, with its patch's files over it when there is one.
    Package {
        path: PathBuf,
        fs: Arc<dyn ImageFs>,
        id: Option<String>,
    },
}

impl Title {
    /// Opens `path`: a folder, a `.pkg` file (opened with `passcode` if it is
    /// a PS4 package) or an executable. `patch` is a PS4 patch package laid
    /// over a base package. When `path` is itself a patch, its base is looked
    /// for in the same folder.
    pub fn open(path: &Path, patch: Option<&Path>, passcode: &Passcode) -> Result<Self, String> {
        if path.is_dir() {
            return Self::without_patch(patch, Title::Folder(path.to_path_buf()));
        }
        let is_pkg = path
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("pkg"));
        if !is_pkg {
            return Self::without_patch(patch, Title::Executable(path.to_path_buf()));
        }
        let package = open_package(path)?;
        let id = package.title_id();
        let (base_path, base, patch) = if package.is_patch() {
            if patch.is_some() {
                return Err(format!("{} is a patch itself", path.display()));
            }
            let base_path = find_base(path, id.as_deref()).ok_or_else(|| {
                format!(
                    "{} is a patch; its base game package was not found in the same folder \
                     (give the base package, and this one with --patch)",
                    path.display()
                )
            })?;
            eprintln!("ps5slop: base package {}", base_path.display());
            let base = open_package(&base_path)?;
            (base_path, base, Some((path.to_path_buf(), package)))
        } else {
            let patch = patch
                .map(|p| open_package(p).map(|pkg| (p.to_path_buf(), pkg)))
                .transpose()?;
            (path.to_path_buf(), package, patch)
        };
        let id = base.title_id();
        let mut fs: Arc<dyn ImageFs> = Arc::new(package_fs(&base_path, base, passcode)?);
        if let Some((patch_path, patch)) = patch {
            if !patch.is_patch() {
                return Err(format!("{} is not a patch", patch_path.display()));
            }
            if patch.title_id() != id {
                return Err(format!(
                    "{} is a patch for another title ({:?}, not {:?})",
                    patch_path.display(),
                    patch.title_id(),
                    id
                ));
            }
            fs = Arc::new(Layered {
                top: Arc::new(package_fs(&patch_path, patch, passcode)?),
                base: fs,
            });
        }
        Ok(Title::Package {
            path: base_path,
            fs,
            id,
        })
    }

    fn without_patch(patch: Option<&Path>, title: Title) -> Result<Self, String> {
        match patch {
            Some(p) => Err(format!(
                "{}: a patch goes with a base package only",
                p.display()
            )),
            None => Ok(title),
        }
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

fn open_package(path: &Path) -> Result<Package, String> {
    Package::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))
}

fn package_fs(path: &Path, package: Package, passcode: &Passcode) -> Result<PackageFs, String> {
    PackageFs::new(package, passcode)
        .map_err(|e| format!("cannot read the files in {}: {e}", path.display()))
}

/// The first package (by name) that has the same title id as `patch` and is
/// not a patch, next to `patch` or, if it is a link, next to its target.
fn find_base(patch: &Path, id: Option<&str>) -> Option<PathBuf> {
    let real = std::fs::canonicalize(patch).ok()?;
    let mut dirs = vec![match patch.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    }];
    dirs.extend(real.parent().map(Path::to_path_buf));
    dirs.into_iter().find_map(|dir| {
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("pkg")))
            .filter(|p| std::fs::canonicalize(p).ok().as_ref() != Some(&real))
            .collect();
        found.sort();
        found.into_iter().find(|p| {
            Package::open(p).is_ok_and(|pkg| !pkg.is_patch() && pkg.title_id().as_deref() == id)
        })
    })
}

/// A patch's files over its base package's: a path the patch holds comes
/// from the patch, anything else from the base.
pub struct Layered {
    top: Arc<dyn ImageFs>,
    base: Arc<dyn ImageFs>,
}

impl ImageFs for Layered {
    fn stat(&self, path: &str) -> Option<ImageStat> {
        self.top.stat(path).or_else(|| self.base.stat(path))
    }

    fn list(&self, path: &str) -> Option<Vec<(String, bool)>> {
        let top = self.top.list(path);
        let base = self.base.list(path);
        if top.is_none() && base.is_none() {
            return None;
        }
        let mut out = top.unwrap_or_default();
        for (name, is_dir) in base.unwrap_or_default() {
            if !out.iter().any(|(n, _)| *n == name) {
                out.push((name, is_dir));
            }
        }
        Some(out)
    }

    fn open(&self, path: &str) -> Option<Arc<dyn ImageFile>> {
        self.top.open(path).or_else(|| self.base.open(path))
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

    /// A tree of files, each holding its own path as bytes.
    struct Tree(&'static [&'static str]);

    impl ImageFs for Tree {
        fn stat(&self, path: &str) -> Option<ImageStat> {
            if self.0.contains(&path) {
                return Some(ImageStat { dir: false, len: path.len() as u64 });
            }
            let dir = path.is_empty() || self.0.iter().any(|f| f.starts_with(&format!("{path}/")));
            dir.then_some(ImageStat { dir: true, len: 0 })
        }

        fn list(&self, path: &str) -> Option<Vec<(String, bool)>> {
            if !self.stat(path)?.dir {
                return None;
            }
            let prefix = if path.is_empty() { String::new() } else { format!("{path}/") };
            let mut out: Vec<(String, bool)> = Vec::new();
            for f in self.0.iter().filter_map(|f| f.strip_prefix(&prefix)) {
                let entry = match f.split_once('/') {
                    Some((d, _)) => (d.to_owned(), true),
                    None => (f.to_owned(), false),
                };
                if !out.contains(&entry) {
                    out.push(entry);
                }
            }
            Some(out)
        }

        fn open(&self, path: &str) -> Option<Arc<dyn ImageFile>> {
            self.0.iter().find(|f| **f == path).map(|f| Arc::new(Bytes(f)) as Arc<dyn ImageFile>)
        }
    }

    struct Bytes(&'static str);

    impl ImageFile for Bytes {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
            let rest = self.0.as_bytes().get(offset as usize..).unwrap_or_default();
            let n = rest.len().min(buf.len());
            buf[..n].copy_from_slice(&rest[..n]);
            Ok(n)
        }
    }

    #[test]
    fn a_patch_shows_its_files_over_the_base_and_the_base_fills_the_rest() {
        let layered = Layered {
            top: Arc::new(Tree(&["games/game.win", "games/new.ogg"])),
            base: Arc::new(Tree(&["eboot.bin", "games/game.win", "games/old.ogg"])),
        };
        let mut games = layered.list("games").unwrap();
        games.sort();
        let names: Vec<&str> = games.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["game.win", "new.ogg", "old.ogg"]);
        assert!(layered.list("").unwrap().contains(&("eboot.bin".to_owned(), false)));
        assert!(layered.stat("eboot.bin").is_some());
        assert!(layered.open("games/old.ogg").is_some());
        assert!(layered.open("missing").is_none());
        assert!(layered.list("missing").is_none());
    }

    /// Reads every file of the package `PS5SLOP_TEST_PKG` names through the
    /// guest's file system and checks it against an extraction of the same
    /// package in `PS5SLOP_TEST_DIR`.
    #[test]
    #[ignore = "needs a package and its extraction"]
    fn package_files_match_their_extraction() {
        let pkg = PathBuf::from(std::env::var_os("PS5SLOP_TEST_PKG").unwrap());
        let dir = PathBuf::from(std::env::var_os("PS5SLOP_TEST_DIR").unwrap());
        let title = Title::open(&pkg, None, &pkg::keys::DEFAULT_PASSCODE).unwrap();
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
