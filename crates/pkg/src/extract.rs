use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::error::invalid;
use crate::inner::InnerImage;
use crate::keys::Passcode;
use crate::naps::Layout;
use crate::package::Format;
use crate::pfs::{Kind, Pfs, Superblock};
use crate::pfsc::MAGIC as PFSC_MAGIC;
use crate::read::ReadAt;
use crate::{Error, Package, Result};

/// LibOrbisPkg's flag for the variant PFS key schedule ("new crypt"); not yet
/// seen in a real package.
const PFS_FLAG_NEW_CRYPT: u64 = 0x2000_0000_0000_0000;

/// The outer image's single file, a compressed PFS that holds the app's files.
const INNER_IMAGE: &str = "pfs_image.dat";
/// The PS5 outer image's map of `pfs_image.dat`'s blocks.
const NAPS_LAYOUT: &str = "naps_pkg_layout.dat";

const CHUNK: usize = 1 << 20;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Extracted {
    pub files: u64,
    pub bytes: u64,
}

impl Package {
    /// The outer PFS image, decrypted with keys derived from `passcode`. PS5
    /// images are read only when stored in the clear.
    pub fn outer_pfs(&self, passcode: &Passcode) -> Result<Pfs> {
        if let Format::Ps5 {
            pfs_offset,
            pfs_size,
            superblock_offset,
        } = self.format()
        {
            let image = self.file().slice(pfs_offset, pfs_size)?;
            let at = superblock_offset
                .checked_sub(pfs_offset)
                .ok_or_else(|| invalid("PS5 superblock lies before the PFS image"))?;
            let image: Arc<dyn ReadAt + Send + Sync> = Arc::new(image);
            if Superblock::read(&*image, at)?.is_encrypted() {
                return Err(Error::Unsupported(
                    "this PS5 package's PFS image is encrypted; ps5slop only reads images stored in the clear".into(),
                ));
            }
            return Pfs::open_at(image, at, None, false);
        }
        let ekpfs = self.ekpfs(passcode)?;
        let h = self.header();
        let image = self.source().slice(h.pfs_image_offset, h.pfs_image_size)?;
        Pfs::open(
            Arc::new(image),
            Some(&ekpfs),
            h.pfs_flags & PFS_FLAG_NEW_CRYPT != 0,
        )
    }

    /// The inner PFS image with the app's files (`eboot.bin`, `sce_module/`, ...).
    pub fn app_pfs(&self, passcode: &Passcode) -> Result<Pfs> {
        let outer = self.outer_pfs(passcode)?;
        let entry = outer.lookup(INNER_IMAGE)?;
        if entry.kind != Kind::File {
            return Err(invalid(format!("{INNER_IMAGE} is not a file")));
        }
        if matches!(self.format(), Format::Ps5 { .. }) {
            let inode = outer.inode(entry.inode)?;
            let stored = outer.open_file_stored(entry.inode)?;
            if inode.is_compressed() && stored.read_vec(0, 4)?[..] != PFSC_MAGIC[..] {
                return self.data_first_pfs(&outer, stored);
            }
        }
        Pfs::open(outer.open_file(entry.inode)?, None, false)
    }

    /// A PS5 "data-first" inner image: files first, compressed in blocks the
    /// NAPS layout maps, with the inner superblock in the last region.
    fn data_first_pfs(&self, outer: &Pfs, stored: Arc<dyn ReadAt + Send + Sync>) -> Result<Pfs> {
        let inner = Self::inner_image(outer, stored)?;
        let layout = inner.layout().clone();
        let superblock = layout.files[layout.file_count as usize - 1].offset;
        let image: Arc<dyn ReadAt + Send + Sync> = Arc::new(inner);
        let sb = Superblock::read(&*image, superblock)?;
        if sb.dinode_block_count == 0 || !sb.is_flat() {
            return Err(Error::Unsupported(format!(
                "PS5 inner PFS with mode {:#x} (only flat images are read)",
                sb.mode
            )));
        }
        Pfs::open_at(image, superblock, None, false)
    }

    /// The block map of a PS5 package's inner image.
    pub fn naps_layout(&self, passcode: &Passcode) -> Result<Layout> {
        let outer = self.outer_pfs(passcode)?;
        let entry = outer.lookup(NAPS_LAYOUT)?;
        let file = outer.open_file_stored(entry.inode)?;
        Layout::parse(&file.read_vec(0, file.len() as usize)?)
    }

    /// The decompressed view of a PS5 package's `pfs_image.dat`.
    pub fn ps5_inner_image(&self) -> Result<InnerImage> {
        let outer = self.outer_pfs(&crate::keys::DEFAULT_PASSCODE)?;
        let entry = outer.lookup(INNER_IMAGE)?;
        Self::inner_image(&outer, outer.open_file_stored(entry.inode)?)
    }

    fn inner_image(outer: &Pfs, stored: Arc<dyn ReadAt + Send + Sync>) -> Result<InnerImage> {
        let entry = outer.lookup(NAPS_LAYOUT)?;
        let file = outer.open_file_stored(entry.inode)?;
        let layout = Layout::parse(&file.read_vec(0, file.len() as usize)?)?;
        Ok(InnerImage::new(Arc::new(layout), stored))
    }

    /// Writes the app's files below `dest`, which must not hold any of them yet,
    /// then the clear metadata entries that belong in `sce_sys/` (`param.sfo`,
    /// icons, ...) unless the image already had them.
    pub fn extract(&self, passcode: &Passcode, dest: &Path) -> Result<Extracted> {
        let app = self.app_pfs(passcode)?;
        let mut done = Extracted::default();
        let mut buf = vec![0u8; CHUNK];
        for (path, inode) in app.files()? {
            let target = dest.join(&path);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let src = app.open_file(inode)?;
            let mut out = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)?;
            let mut at = 0;
            while at < src.len() {
                let n = (src.len() - at).min(CHUNK as u64) as usize;
                src.read_exact_at(at, &mut buf[..n])?;
                out.write_all(&buf[..n])?;
                at += n as u64;
            }
            done.files += 1;
            done.bytes += src.len();
        }
        for (name, entry) in self.sce_sys_entries() {
            let target = dest.join("sce_sys").join(name);
            if target.exists() {
                continue;
            }
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let data = self.read_entry(&entry)?;
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)?
                .write_all(&data)?;
            done.files += 1;
            done.bytes += data.len() as u64;
        }
        Ok(done)
    }
}
