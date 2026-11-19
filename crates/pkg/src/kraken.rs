//! The Kraken variant in PS5 inner images, as far as our packages use it.
//! Format and sources: `docs/packages.md`.

use crate::error::invalid;
use crate::{Error, Result};

const MAX_CODE_LEN: u32 = 11;
const TABLE_BITS: u32 = MAX_CODE_LEN;
const RAW_PREFIX: usize = 8;
const INITIAL_OFFSET: usize = 8;
const ESCAPE_K: u32 = 6;

fn short(what: &str) -> Error {
    invalid(format!("Kraken data ends early ({what})"))
}

struct ArrayHeader {
    kind: u8,
    header_len: usize,
    decoded: usize,
    coded: usize,
}

fn array_header(d: &[u8], p: usize) -> Result<ArrayHeader> {
    let b = |n: usize| d.get(p..p + n).ok_or_else(|| short("array header"));
    let b0 = *d.get(p).ok_or_else(|| short("array header"))?;
    let short_form = b0 & 0x80 != 0;
    let kind = (b0 >> 4) & 7;
    let be = |s: &[u8]| s.iter().fold(0u64, |v, &x| v << 8 | u64::from(x));
    Ok(match (kind, short_form) {
        (0, true) => {
            let n = (be(b(2)?) & 0xfff) as usize;
            ArrayHeader {
                kind,
                header_len: 2,
                decoded: n,
                coded: n,
            }
        }
        (0, false) => {
            let n = (be(b(3)?) & 0xf_ffff) as usize;
            ArrayHeader {
                kind,
                header_len: 3,
                decoded: n,
                coded: n,
            }
        }
        (_, true) => {
            let w = be(b(3)?);
            let coded = (w & 0x3ff) as usize;
            ArrayHeader {
                kind,
                header_len: 3,
                decoded: ((w >> 10) & 0x3ff) as usize + coded + 1,
                coded,
            }
        }
        (_, false) => {
            let w = be(b(5)?);
            ArrayHeader {
                kind,
                header_len: 5,
                decoded: ((w >> 18) & 0x3_ffff) as usize + 1,
                coded: (w & 0x3_ffff) as usize,
            }
        }
    })
}

/// Bits read most significant first.
struct MsbReader<'a> {
    d: &'a [u8],
    pos: usize,
}

impl MsbReader<'_> {
    fn bit(&mut self) -> Result<u32> {
        let byte = *self
            .d
            .get(self.pos >> 3)
            .ok_or_else(|| short("code description"))?;
        let v = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(u32::from(v))
    }

    fn bits(&mut self, n: u32) -> Result<u32> {
        (0..n).try_fold(0, |v, _| Ok(v << 1 | self.bit()?))
    }

    fn unary(&mut self) -> Result<u32> {
        let mut n = 0;
        while self.bit()? == 0 {
            n += 1;
            if n > 32 {
                return Err(invalid("Kraken unary code is too long"));
            }
        }
        Ok(n)
    }
}

/// Code lengths by symbol (0 = unused).
type Lengths = [u8; 256];

/// The `00` form: a symbol count, then the symbol alone, or a field width
/// and each symbol with its length - 1.
fn lengths_small(r: &mut MsbReader) -> Result<Lengths> {
    let mut out = [0u8; 256];
    let n = r.bits(8)?;
    if n == 1 {
        out[r.bits(8)? as usize] = 1;
        return Ok(out);
    }
    let width = r.bits(3)?;
    for _ in 0..n {
        let sym = r.bits(8)? as usize;
        out[sym] = (r.bits(width)? + 1) as u8;
    }
    Ok(out)
}

/// The `01` form: alternating runs of used and unused symbols, each used
/// symbol followed by a Rice-coded step against a running estimate.
fn lengths_spans(r: &mut MsbReader) -> Result<Lengths> {
    let mut out = [0u8; 256];
    let k = r.bits(2)?;
    let mut take = r.bit()? == 1;
    let mut est: i32 = 30;
    let mut sym = 0usize;
    while sym < 256 {
        let u = r.unary()?;
        let span = ((1u32 << (u + 1)) | r.bits(u + 1)?) as usize - 1;
        if take {
            for _ in 0..span {
                let v = (r.unary()? << k | r.bits(k)?) as i32;
                let step = if v % 2 == 0 { v / 2 } else { -(v + 1) / 2 };
                let len = step + (est >> 2) + 1;
                if !(1..=MAX_CODE_LEN as i32).contains(&len) || sym > 255 {
                    return Err(invalid("Kraken code length out of range"));
                }
                out[sym] = len as u8;
                est = est + len - ((est + 3) >> 2);
                sym += 1;
            }
        } else {
            sym += span;
        }
        take = !take;
    }
    if sym != 256 {
        return Err(invalid("Kraken symbol spans overrun"));
    }
    Ok(out)
}

/// A lookup table indexed by the next 11 bits read least significant first.
fn decode_table(lengths: &Lengths) -> Result<Vec<(u8, u8)>> {
    let mut order: Vec<(u8, u8)> = (0..=255u8)
        .filter(|&s| lengths[s as usize] != 0)
        .map(|s| (lengths[s as usize], s))
        .collect();
    order.sort_unstable();
    let mut table = vec![(0u8, 0u8); 1 << TABLE_BITS];
    let mut code = 0u32;
    let mut prev = 0u32;
    for &(len, sym) in &order {
        let len = u32::from(len);
        code <<= len - prev;
        prev = len;
        let rev = code.reverse_bits() >> (32 - len);
        for hi in 0..1u32 << (TABLE_BITS - len) {
            table[(rev | hi << len) as usize] = (sym, len as u8);
        }
        code += 1;
    }
    if code != 1 << prev {
        return Err(invalid("Kraken Huffman code is not complete"));
    }
    Ok(table)
}

/// A bit stream read least significant bit first, forwards or from the end.
struct LsbStream<'a> {
    d: &'a [u8],
    backward: bool,
    pos: usize,
}

impl LsbStream<'_> {
    fn peek(&self) -> u32 {
        let mut v = 0u32;
        for i in 0..TABLE_BITS as usize {
            let bit = self.pos + i;
            let idx = bit >> 3;
            let byte = if idx < self.d.len() {
                self.d[if self.backward {
                    self.d.len() - 1 - idx
                } else {
                    idx
                }]
            } else {
                0
            };
            v |= u32::from((byte >> (bit & 7)) & 1) << i;
        }
        v
    }
}

fn huffman3(d: &[u8], table: &[(u8, u8)], count: usize, out: &mut Vec<u8>) -> Result<()> {
    let size0 = usize::from(u16::from_le_bytes(
        d.get(..2)
            .ok_or_else(|| short("Huffman streams"))?
            .try_into()
            .unwrap(),
    ));
    let s0 = d
        .get(2..2 + size0)
        .ok_or_else(|| short("Huffman stream 0"))?;
    let rest = &d[2 + size0..];
    let mut streams = [
        LsbStream {
            d: s0,
            backward: false,
            pos: 0,
        },
        LsbStream {
            d: rest,
            backward: true,
            pos: 0,
        },
        LsbStream {
            d: rest,
            backward: false,
            pos: 0,
        },
    ];
    out.reserve(count);
    for i in 0..count {
        let s = &mut streams[i % 3];
        let (sym, len) = table[s.peek() as usize];
        s.pos += usize::from(len);
        out.push(sym);
    }
    if streams[0].pos > s0.len() * 8 || streams[1].pos + streams[2].pos > rest.len() * 8 {
        return Err(invalid("Kraken Huffman streams overrun"));
    }
    Ok(())
}

/// Decodes the array at `p`, appending it to `out`; returns where it ends.
fn decode_array(d: &[u8], p: usize, out: &mut Vec<u8>) -> Result<usize> {
    let h = array_header(d, p)?;
    let start = p + h.header_len;
    let end = start + h.coded;
    let body = d.get(start..end).ok_or_else(|| short("array"))?;
    match h.kind {
        0 => out.extend_from_slice(body),
        2 => {
            let mut r = MsbReader { d: body, pos: 0 };
            let lengths = match r.bits(2)? {
                0 => lengths_small(&mut r)?,
                1 => lengths_spans(&mut r)?,
                form => {
                    return Err(Error::Unsupported(format!(
                        "Kraken Huffman code description form {form:02b}"
                    )));
                }
            };
            let used: Vec<usize> = (0..256).filter(|&s| lengths[s] != 0).collect();
            if used.len() == 1 {
                out.resize(out.len() + h.decoded, used[0] as u8);
            } else {
                let table = decode_table(&lengths)?;
                huffman3(&body[r.pos.div_ceil(8)..], &table, h.decoded, out)?;
            }
        }
        kind => {
            return Err(Error::Unsupported(format!("Kraken array type {kind}")));
        }
    }
    Ok(end)
}

fn array(d: &[u8], p: &mut usize) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    *p = decode_array(d, *p, &mut v)?;
    Ok(v)
}

/// Bits read most significant first, from the front or from the end.
struct BitStream<'a> {
    d: &'a [u8],
    backward: bool,
    pos: usize,
}

impl BitStream<'_> {
    fn bits(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let idx = self.pos >> 3;
            if idx >= self.d.len() {
                return Err(short("offset or length bits"));
            }
            let byte = self.d[if self.backward {
                self.d.len() - 1 - idx
            } else {
                idx
            }];
            v = v << 1 | u32::from((byte >> (7 - (self.pos & 7))) & 1);
            self.pos += 1;
        }
        Ok(v)
    }

    /// An Exp-Golomb code of order [`ESCAPE_K`].
    fn escape(&mut self) -> Result<usize> {
        let mut zeros = 0;
        while self.bits(1)? == 0 {
            zeros += 1;
            if zeros > 24 {
                return Err(invalid("Kraken length escape is too long"));
            }
        }
        let v = (1u32 << (zeros + ESCAPE_K)) | self.bits(zeros + ESCAPE_K)?;
        Ok((v - (1 << ESCAPE_K)) as usize)
    }
}

/// Decodes one LZ chunk of `len` bytes, appending it to `out`, whose earlier
/// bytes (the block's first chunk) matches may copy from.
pub(crate) fn decode_lz(
    d: &[u8],
    len: usize,
    first: bool,
    sub: bool,
    out: &mut Vec<u8>,
) -> Result<()> {
    let base = out.len();
    let mut p = 0;
    if first {
        out.extend_from_slice(d.get(..RAW_PREFIX).ok_or_else(|| short("chunk"))?);
        p = RAW_PREFIX;
    }
    let flag = *d.get(p).ok_or_else(|| short("chunk"))?;
    p += 1;
    if flag & 0x80 == 0 {
        return Err(invalid("Kraken chunk lacks its flag byte"));
    }
    let extra = if flag & 0x20 != 0 {
        let b1 = *d.get(p).ok_or_else(|| short("chunk"))?;
        p += 1;
        usize::from(flag & 0x1f) + 32 * (usize::from(b1) + 1)
    } else {
        usize::from(flag & 0x1f)
    };
    let lits = array(d, &mut p)?;
    let cmds = array(d, &mut p)?;
    let offs = array(d, &mut p)?;
    let lens = array(d, &mut p)?;
    let tail_end = d
        .len()
        .checked_sub(extra)
        .filter(|&e| e >= p)
        .ok_or_else(|| short("chunk tail"))?;
    let tail = &d[p..tail_end];
    let escapes = &d[tail_end..];
    let mut offset_streams = [
        BitStream {
            d: tail,
            backward: false,
            pos: 0,
        },
        BitStream {
            d: tail,
            backward: true,
            pos: 0,
        },
    ];
    let mut escape_streams = [
        BitStream {
            d: escapes,
            backward: false,
            pos: 0,
        },
        BitStream {
            d: escapes,
            backward: true,
            pos: 0,
        },
    ];
    let (mut li, mut ci, mut oi, mut new_offsets, mut esc) = (0, 0, 0, 0usize, 0usize);
    let mut recent = [INITIAL_OFFSET; 3];
    let next_len = |li: &mut usize| -> Result<usize> {
        let v = *lens.get(*li).ok_or_else(|| short("lengths"))?;
        *li += 1;
        Ok(usize::from(v))
    };
    let push_literals = |out: &mut Vec<u8>, from: usize, n: usize, last: usize| -> Result<()> {
        let src = lits.get(from..from + n).ok_or_else(|| short("literals"))?;
        if sub {
            for &l in src {
                let prev = out[out
                    .len()
                    .checked_sub(last)
                    .ok_or_else(|| invalid("Kraken delta literal before the start"))?];
                out.push(l.wrapping_add(prev));
            }
        } else {
            out.extend_from_slice(src);
        }
        Ok(())
    };
    for &c in &cmds {
        let mut lit = usize::from(c & 3);
        let mut ml = usize::from((c >> 2) & 15) + 2;
        let typ = usize::from(c >> 6);
        if lit == 3 {
            lit = next_len(&mut li)? + 3;
            if lit == 258 {
                lit += escape_streams[esc % 2].escape()?;
                esc += 1;
            }
        }
        if ml == 17 {
            ml = next_len(&mut li)? + 17;
            if ml == 272 {
                ml += escape_streams[esc % 2].escape()?;
                esc += 1;
            }
        }
        push_literals(out, ci, lit, recent[0])?;
        ci += lit;
        let off = if typ == 3 {
            let code = *offs.get(oi).ok_or_else(|| short("offsets"))?;
            oi += 1;
            let k = u32::from(code >> 4);
            let hi = offset_streams[new_offsets % 2].bits(4 + k)? + (((1 << k) - 1) << 4);
            new_offsets += 1;
            let off = INITIAL_OFFSET + ((hi as usize) << 4 | usize::from(code & 15));
            recent = [off, recent[0], recent[1]];
            off
        } else {
            let off = recent[typ];
            for i in (1..=typ).rev() {
                recent[i] = recent[i - 1];
            }
            recent[0] = off;
            off
        };
        if off > out.len() {
            return Err(invalid("Kraken match reaches before the block"));
        }
        for _ in 0..ml {
            let b = out[out.len() - off];
            out.push(b);
        }
    }
    let rest = lits
        .len()
        .checked_sub(ci)
        .ok_or_else(|| short("literals"))?;
    push_literals(out, ci, rest, recent[0])?;
    if out.len() - base != len {
        return Err(invalid(format!(
            "Kraken chunk decodes to {} bytes, not {len}",
            out.len() - base
        )));
    }
    Ok(())
}

/// Decodes a chunk stored without LZ: as is, or as one entropy-coded array.
pub(crate) fn decode_plain(d: &[u8], len: usize, out: &mut Vec<u8>) -> Result<()> {
    if d.len() == len {
        out.extend_from_slice(d);
        return Ok(());
    }
    let base = out.len();
    decode_array(d, 0, out)?;
    if out.len() - base != len {
        return Err(invalid("Kraken array chunk has the wrong length"));
    }
    Ok(())
}

/// The stored size of a chunk stored without LZ, from its array header.
pub(crate) fn plain_size(head: &[u8], len: usize) -> usize {
    match array_header(head, 0) {
        Ok(h) if h.decoded == len && matches!(h.kind, 0 | 2) => h.header_len + h.coded,
        _ => len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_single_symbol_array() {
        let mut out = Vec::new();
        decode_plain(
            &[0x27, 0xff, 0xfc, 0x00, 0x03, 0x00, 0x40, 0x00],
            0x20000,
            &mut out,
        )
        .unwrap();
        assert_eq!(out.len(), 0x20000);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn decodes_a_small_alphabet_huffman_array() {
        let a = [
            0x20, 0x00, 0x88, 0x00, 0x10, 0x01, 0x10, 0x02, 0x10, 0x08, 0x45, 0x10, 0x03, 0x00,
            0xa8, 0xea, 0x19, 0xc0, 0x01, 0x2a, 0x20,
        ];
        let mut out = Vec::new();
        decode_array(&a, 0, &mut out).unwrap();
        assert_eq!(out.len(), 35);
        assert_eq!(&out[..12], &[8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8]);
    }

    #[test]
    fn decodes_span_coded_lengths() {
        let mut r = MsbReader {
            d: &[0x6c, 0x72, 0x23, 0x91, 0x51, 0x0b, 0x91, 0x50, 0x36, 0x00],
            pos: 0,
        };
        assert_eq!(r.bits(2).unwrap(), 1);
        let l = lengths_spans(&mut r).unwrap();
        assert_eq!((l[0], l[8], l[16], l[32], l[40]), (2, 1, 4, 3, 4));
        assert_eq!(l.iter().filter(|&&x| x != 0).count(), 5);
        assert_eq!(r.pos.div_ceil(8), 10);
    }

    #[test]
    fn decodes_an_lz_chunk_with_one_match() {
        let mut chunk = b"abcdefgh".to_vec();
        chunk.push(0x80);
        chunk.extend([0x80, 0x00]);
        chunk.extend([0x80, 0x01, 6 << 2]);
        chunk.extend([0x80, 0x00]);
        chunk.extend([0x80, 0x00]);
        let mut out = Vec::new();
        decode_lz(&chunk, 16, true, false, &mut out).unwrap();
        assert_eq!(out, b"abcdefghabcdefgh");
        assert!(decode_lz(&chunk, 17, true, false, &mut Vec::new()).is_err());
    }
}
