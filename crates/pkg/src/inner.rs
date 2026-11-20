//! The decompressed view of a PS5 `pfs_image.dat`: the logical image the inner
//! PFS addresses, rebuilt block by block through the NAPS layout.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::error::invalid;
use crate::kraken::{decode_lz, decode_plain, plain_size};
use crate::naps::{BLOCK_SIZE, CHUNK_SIZE, ChunkKind, Layout};
use crate::read::ReadAt;
use crate::{Error, Result};

type Source = Arc<dyn ReadAt + Send + Sync>;

/// Decoded blocks kept for reuse.
const CACHE_BLOCKS: usize = 16;

/// Decoded blocks by (region, block index), oldest first.
type BlockCache = VecDeque<((usize, u64), Arc<Vec<u8>>)>;

/// A read-only view of the logical image. Files are compressed in 256 KiB
/// blocks counted from each file's start, so a read decodes whole blocks.
pub struct InnerImage {
    layout: Arc<Layout>,
    stored: Source,
    cache: Mutex<BlockCache>,
}

impl InnerImage {
    pub fn new(layout: Arc<Layout>, stored: Source) -> Self {
        Self {
            layout,
            stored,
            cache: Mutex::new(VecDeque::new()),
        }
    }

    pub fn layout(&self) -> &Arc<Layout> {
        &self.layout
    }

    fn block(&self, region: usize, index: u64) -> Result<Arc<Vec<u8>>> {
        let key = (region, index);
        if let Some((_, b)) = self.cache.lock().unwrap().iter().find(|(k, _)| *k == key) {
            return Ok(b.clone());
        }
        let block = Arc::new(self.decode_block(region, index)?);
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= CACHE_BLOCKS {
            cache.pop_front();
        }
        cache.push_back((key, block.clone()));
        Ok(block)
    }

    /// Decodes block `index` of a region (counted from the region's start).
    pub fn decode_block(&self, region: usize, index: u64) -> Result<Vec<u8>> {
        let from = self.layout.region(region).0 + index * BLOCK_SIZE;
        self.decode_block_inner(region, index).map_err(|e| {
            let record = self.layout.piece(from);
            let what = format!(
                "{} (logical {from:#x}, stored {:#x}, flags {:#x})",
                match &e {
                    Error::Invalid(w) | Error::Unsupported(w) => w.clone(),
                    e => e.to_string(),
                },
                record.map_or(0, |r| r.offset),
                record.map_or(0, |r| r.flags),
            );
            match e {
                Error::Unsupported(_) => Error::Unsupported(what),
                Error::Io(e) => Error::Io(e),
                _ => Error::Invalid(what),
            }
        })
    }

    fn decode_block_inner(&self, region: usize, index: u64) -> Result<Vec<u8>> {
        let (start, end, _) = self.layout.region(region);
        let from = start + index * BLOCK_SIZE;
        let len = (end - from).min(BLOCK_SIZE) as usize;
        let record = *self
            .layout
            .piece(from)
            .ok_or_else(|| invalid(format!("no NAPS record for logical offset {from:#x}")))?;
        let len1 = len.min(CHUNK_SIZE as usize);
        let len2 = len - len1;
        let (kind1, kind2) = Layout::chunk_kinds(&record);
        let size1 = record.chunk1 as usize;
        let mut out = Vec::with_capacity(len);
        let first = self.stored.read_vec(record.offset, size1)?;
        decode_chunk(&first, len1, kind1, true, &mut out)?;
        if len2 > 0 {
            let at = record.offset + size1 as u64;
            let size2 = match (self.layout.piece_size(&record), kind2) {
                (Some(total), _) => (total as usize)
                    .checked_sub(size1)
                    .ok_or_else(|| invalid("NAPS piece is smaller than its first chunk"))?,
                (None, ChunkKind::Plain) => {
                    let head_len = 5.min((self.stored.len() - at) as usize);
                    plain_size(&self.stored.read_vec(at, head_len)?, len2)
                }
                (None, ChunkKind::Lz { .. }) => {
                    return Err(invalid(format!(
                        "unknown stored size of the piece at {from:#x}"
                    )));
                }
            };
            let second = self.stored.read_vec(at, size2)?;
            decode_chunk(&second, len2, kind2, false, &mut out)?;
        }
        Ok(out)
    }
}

fn decode_chunk(
    d: &[u8],
    len: usize,
    kind: ChunkKind,
    first: bool,
    out: &mut Vec<u8>,
) -> Result<()> {
    match kind {
        ChunkKind::Plain => decode_plain(d, len, out),
        ChunkKind::Lz { sub } => decode_lz(d, len, first, sub, out),
    }
}

impl ReadAt for InnerImage {
    fn len(&self) -> u64 {
        self.layout.mount_size()
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(buf.len() as u64)
            .is_none_or(|end| end > self.len())
        {
            return Err(invalid("read past the end of the inner image"));
        }
        let mut done = 0;
        while done < buf.len() {
            let pos = offset + done as u64;
            let region = self
                .layout
                .region_at(pos)
                .ok_or_else(|| Error::Invalid(format!("no NAPS region at {pos:#x}")))?;
            let (start, end, sparse) = self.layout.region(region);
            let in_region = pos - start;
            let index = in_region / BLOCK_SIZE;
            let within = (in_region % BLOCK_SIZE) as usize;
            let block_end = (start + (index + 1) * BLOCK_SIZE).min(end);
            let n = ((block_end - pos) as usize).min(buf.len() - done);
            if sparse {
                buf[done..done + n].fill(0);
            } else {
                let block = self.block(region, index)?;
                buf[done..done + n].copy_from_slice(&block[within..within + n]);
            }
            done += n;
        }
        Ok(())
    }
}
