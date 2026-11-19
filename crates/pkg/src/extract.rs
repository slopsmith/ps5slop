use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::error::invalid;
use crate::keys::Passcode;
use crate::package::SCE_SYS_ENTRIES;
use crate::pfs::{Kind, Pfs};
use crate::read::ReadAt;
use crate::{Package, Result};

/// LibOrbisPkg's flag for the variant PFS key schedule ("new crypt"); not yet
/// seen in a real package.
const PFS_FLAG_NEW_CRYPT: u64 = 0x2000_0000_0000_0000;

/// The outer image's single file, a compressed PFS that holds the app's files.
const INNER_IMAGE: &str = "pfs_image.dat";

const CHUNK: usize = 1 << 20;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Extracted {
    pub files: u64,
    pub bytes: u64,
}

impl Package {
    /// The outer PFS image, decrypted with keys derived from `passcode`.
    pub fn outer_pfs(&self, passcode: &Passcode) -> Result<Pfs> {
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
        Pfs::open(outer.open_file(entry.inode)?, None, false)
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
        for &(id, name) in SCE_SYS_ENTRIES {
            let Some(entry) = self.entry(id).filter(|e| !e.is_encrypted()) else {
                continue;
            };
            let target = dest.join("sce_sys").join(name);
            if target.exists() {
                continue;
            }
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let data = self.read_entry(entry)?;
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
