use aes::Aes128;
use aes::cipher::KeyInit;
use xts_mode::{Xts128, get_tweak_default};

use crate::Result;
use crate::keys::PfsKeys;
use crate::read::ReadAt;

pub const SECTOR_SIZE: u64 = 0x1000;

/// Decrypts an AES-XTS encrypted PFS image on the fly. Sectors before
/// `first_encrypted` (the header block) are stored in the clear. The tweak of a
/// sector is its index from the start of the image.
pub struct XtsReader<R> {
    inner: R,
    xts: Xts128<Aes128>,
    first_encrypted: u64,
}

impl<R: ReadAt> XtsReader<R> {
    pub fn new(inner: R, keys: &PfsKeys, first_encrypted: u64) -> Self {
        let data = Aes128::new(&keys.data.into());
        let tweak = Aes128::new(&keys.tweak.into());
        Self {
            inner,
            xts: Xts128::new(data, tweak),
            first_encrypted,
        }
    }
}

impl<R: ReadAt> ReadAt for XtsReader<R> {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let mut sector_buf = [0u8; SECTOR_SIZE as usize];
        let mut done = 0;
        while done < buf.len() {
            let pos = offset + done as u64;
            let sector = pos / SECTOR_SIZE;
            let within = (pos % SECTOR_SIZE) as usize;
            let n = (SECTOR_SIZE as usize - within).min(buf.len() - done);
            let out = &mut buf[done..done + n];
            if sector < self.first_encrypted {
                self.inner.read_exact_at(pos, out)?;
            } else {
                self.inner
                    .read_exact_at(sector * SECTOR_SIZE, &mut sector_buf)?;
                self.xts
                    .decrypt_sector(&mut sector_buf, get_tweak_default(sector.into()));
                out.copy_from_slice(&sector_buf[within..within + n]);
            }
            done += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> PfsKeys {
        PfsKeys {
            tweak: [7; 16],
            data: [9; 16],
        }
    }

    /// Four sectors whose plaintext is the byte offset (mod 251), with every
    /// sector from `first_encrypted` on encrypted.
    fn image(first_encrypted: u64) -> (Vec<u8>, Vec<u8>) {
        let plain: Vec<u8> = (0..4 * SECTOR_SIZE).map(|i| (i % 251) as u8).collect();
        let mut enc = plain.clone();
        let k = keys();
        let xts = Xts128::new(Aes128::new(&k.data.into()), Aes128::new(&k.tweak.into()));
        for (i, s) in enc.chunks_mut(SECTOR_SIZE as usize).enumerate() {
            if i as u64 >= first_encrypted {
                xts.encrypt_sector(s, get_tweak_default(i as u128));
            }
        }
        (plain, enc)
    }

    #[test]
    fn reads_across_plain_and_encrypted_sectors() {
        let (plain, enc) = image(1);
        let r = XtsReader::new(enc, &keys(), 1);
        let mut buf = vec![0u8; 0x2100];
        r.read_exact_at(0xf80, &mut buf).unwrap();
        assert_eq!(buf, plain[0xf80..0xf80 + 0x2100]);
    }

    #[test]
    fn short_read_inside_one_sector() {
        let (plain, enc) = image(0);
        let r = XtsReader::new(enc, &keys(), 0);
        let mut buf = [0u8; 5];
        r.read_exact_at(0x3001, &mut buf).unwrap();
        assert_eq!(buf, plain[0x3001..0x3006]);
    }

    #[test]
    fn wrong_keys_give_other_bytes() {
        let (plain, enc) = image(0);
        let other = PfsKeys {
            tweak: [1; 16],
            data: [2; 16],
        };
        let r = XtsReader::new(enc, &other, 0);
        assert_ne!(r.read_vec(0, 64).unwrap(), plain[..64]);
    }
}
