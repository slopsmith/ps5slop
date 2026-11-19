use std::io::Read;

use flate2::read::DeflateDecoder;

use crate::Result;
use crate::error::invalid;
use crate::read::{ReadAt, u32_le, u64_le};

pub(crate) const MAGIC: &[u8; 4] = b"PFSC";
const HEADER_SIZE: usize = 0x30;

/// A PFSC file: data cut into fixed-size blocks, each stored raw or deflated.
///
/// Header: magic, a zero u32, an unknown u32, the block size (u32 at `0x0c`,
/// repeated as u64 at `0x10`), the offset of the block offset table (`0x18`),
/// the data start (`0x20`) and the uncompressed length (`0x28`). The table has
/// one u64 offset per block plus one for the end. A stored block as long as
/// the block size is raw, a longer one reads as zeros, and a shorter one is a
/// 2-byte zlib header followed by raw deflate data.
pub struct PfscReader<R> {
    inner: R,
    block_size: u64,
    len: u64,
    offsets: Vec<u64>,
}

impl<R: ReadAt> PfscReader<R> {
    pub fn new(inner: R) -> Result<Self> {
        let h = inner.read_vec(0, HEADER_SIZE)?;
        if &h[..4] != MAGIC || u32_le(&h, 4) != 0 {
            return Err(invalid("not a PFSC file"));
        }
        let block_size = u64::from(u32_le(&h, 0x0c));
        if block_size == 0 || u64_le(&h, 0x10) != block_size {
            return Err(invalid("PFSC block sizes disagree"));
        }
        let table = u64_le(&h, 0x18);
        let len = u64_le(&h, 0x28);
        let blocks = len.div_ceil(block_size);
        let count = usize::try_from(blocks + 1).map_err(|_| invalid("PFSC file is too large"))?;
        let raw = inner.read_vec(table, count * 8)?;
        let offsets: Vec<u64> = raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64_le(c, 0))
            .collect();
        if offsets.windows(2).any(|w| w[1] < w[0]) || offsets[count - 1] > inner.len() {
            return Err(invalid(
                "PFSC block offsets are out of order or out of bounds",
            ));
        }
        Ok(Self {
            inner,
            block_size,
            len,
            offsets,
        })
    }

    fn read_block(&self, index: usize, out: &mut [u8]) -> Result<()> {
        let start = self.offsets[index];
        let stored = self.offsets[index + 1] - start;
        if stored == self.block_size {
            self.inner.read_exact_at(start, out)?;
        } else if stored > self.block_size {
            out.fill(0);
        } else {
            let data = self.inner.read_vec(start, stored as usize)?;
            let deflated = data
                .get(2..)
                .ok_or_else(|| invalid("PFSC block is too short"))?;
            DeflateDecoder::new(deflated)
                .read_exact(out)
                .map_err(|e| invalid(format!("PFSC block {index} does not inflate: {e}")))?;
        }
        Ok(())
    }
}

impl<R: ReadAt> ReadAt for PfscReader<R> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(buf.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(invalid("read past the end of a PFSC file"));
        }
        let mut block = vec![0u8; self.block_size as usize];
        let mut done = 0;
        while done < buf.len() {
            let pos = offset + done as u64;
            let index = (pos / self.block_size) as usize;
            let within = (pos % self.block_size) as usize;
            let n = (self.block_size as usize - within).min(buf.len() - done);
            self.read_block(index, &mut block)?;
            buf[done..done + n].copy_from_slice(&block[within..within + n]);
            done += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::ZlibEncoder;

    use super::*;

    const BS: usize = 0x100;

    /// Builds a PFSC file from blocks given as (plaintext, how to store it).
    fn build(blocks: &[(Vec<u8>, Store)]) -> Vec<u8> {
        let table = HEADER_SIZE;
        let data_start = table + 8 * (blocks.len() + 1);
        let mut data = Vec::new();
        let mut offsets = vec![];
        for (plain, store) in blocks {
            offsets.push((data_start + data.len()) as u64);
            match store {
                Store::Raw => data.extend_from_slice(plain),
                Store::Zeros => data.extend(std::iter::repeat_n(0xAA, BS + 1)),
                Store::Zlib => {
                    let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
                    z.write_all(plain).unwrap();
                    data.extend(z.finish().unwrap());
                }
            }
        }
        offsets.push((data_start + data.len()) as u64);
        let mut f = vec![0u8; HEADER_SIZE];
        f[..4].copy_from_slice(MAGIC);
        f[0x0c..0x10].copy_from_slice(&(BS as u32).to_le_bytes());
        f[0x10..0x18].copy_from_slice(&(BS as u64).to_le_bytes());
        f[0x18..0x20].copy_from_slice(&(table as u64).to_le_bytes());
        f[0x20..0x28].copy_from_slice(&(data_start as u64).to_le_bytes());
        f[0x28..0x30].copy_from_slice(&((blocks.len() * BS) as u64).to_le_bytes());
        for o in offsets {
            f.extend_from_slice(&o.to_le_bytes());
        }
        f.extend(data);
        f
    }

    enum Store {
        Raw,
        Zeros,
        Zlib,
    }

    #[test]
    fn reads_raw_deflated_and_zero_blocks() {
        let raw: Vec<u8> = (0..BS).map(|i| (i * 7) as u8).collect();
        let text: Vec<u8> = b"pfsc ".iter().copied().cycle().take(BS).collect();
        let f = build(&[
            (raw.clone(), Store::Raw),
            (text.clone(), Store::Zlib),
            (vec![0; BS], Store::Zeros),
        ]);
        let r = PfscReader::new(f).unwrap();
        assert_eq!(r.len(), 3 * BS as u64);
        let all = r.read_vec(0, 3 * BS).unwrap();
        assert_eq!(&all[..BS], &raw[..]);
        assert_eq!(&all[BS..2 * BS], &text[..]);
        assert!(all[2 * BS..].iter().all(|&b| b == 0));
    }

    #[test]
    fn reads_across_a_block_boundary() {
        let a = vec![1u8; BS];
        let b = vec![2u8; BS];
        let r = PfscReader::new(build(&[(a, Store::Zlib), (b, Store::Raw)])).unwrap();
        assert_eq!(r.read_vec(BS as u64 - 2, 4).unwrap(), [1, 1, 2, 2]);
    }

    #[test]
    fn rejects_bad_magic_and_reads_past_the_end() {
        let mut f = build(&[(vec![0; BS], Store::Raw)]);
        let r = PfscReader::new(f.clone()).unwrap();
        assert!(r.read_vec(1, BS).is_err());
        f[0] = b'X';
        assert!(PfscReader::new(f).is_err());
    }
}
