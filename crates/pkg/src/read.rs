use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use crate::Result;
use crate::error::invalid;

/// Random-access reads from a fixed-size byte source.
pub trait ReadAt {
    fn len(&self) -> u64;

    /// Fills `buf` from `offset`. Reading past the end is an error.
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn read_vec(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0; len];
        self.read_exact_at(offset, &mut buf)?;
        Ok(buf)
    }
}

impl ReadAt for [u8] {
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let src = usize::try_from(offset)
            .ok()
            .and_then(|start| self.get(start..start.checked_add(buf.len())?))
            .ok_or_else(|| {
                invalid(format!(
                    "read of {} bytes at {offset:#x} is out of bounds",
                    buf.len()
                ))
            })?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

impl ReadAt for Vec<u8> {
    fn len(&self) -> u64 {
        self.as_slice().len() as u64
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.as_slice().read_exact_at(offset, buf)
    }
}

impl<R: ReadAt + ?Sized> ReadAt for Arc<R> {
    fn len(&self) -> u64 {
        (**self).len()
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(offset, buf)
    }
}

impl<R: ReadAt + ?Sized> ReadAt for Box<R> {
    fn len(&self) -> u64 {
        (**self).len()
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(offset, buf)
    }
}

/// A window `[base, base + len)` into a shared file.
#[derive(Clone)]
pub struct FileSlice {
    file: Arc<File>,
    base: u64,
    len: u64,
}

impl FileSlice {
    pub fn new(file: Arc<File>, base: u64, len: u64) -> Self {
        Self { file, base, len }
    }

    pub fn whole(file: File) -> io::Result<Self> {
        let len = file.metadata()?.len();
        Ok(Self::new(Arc::new(file), 0, len))
    }

    /// A window into this window.
    pub fn slice(&self, offset: u64, len: u64) -> Result<Self> {
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(invalid(format!(
                "slice {offset:#x}+{len:#x} is outside a source of {:#x} bytes",
                self.len
            )));
        }
        Ok(Self::new(self.file.clone(), self.base + offset, len))
    }
}

impl ReadAt for FileSlice {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(buf.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(invalid(format!(
                "read of {} bytes at {offset:#x} is out of bounds",
                buf.len()
            )));
        }
        self.file.read_exact_at(buf, self.base + offset)?;
        Ok(())
    }
}

pub(crate) fn u32_be(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

pub(crate) fn u64_be(b: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(b[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_reads_reject_out_of_bounds() {
        let data = vec![1u8, 2, 3, 4];
        let mut buf = [0u8; 2];
        assert!(data.read_exact_at(2, &mut buf).is_ok());
        assert_eq!(buf, [3, 4]);
        assert!(data.read_exact_at(3, &mut buf).is_err());
        assert!(data.read_exact_at(u64::MAX, &mut buf).is_err());
    }
}
