//! The save state file format. A file is a header followed by sections,
//! each a four-byte tag and a zstd-compressed payload; payloads hold values
//! in a small binary encoding ([`Encode`], [`Decode`]): little-endian
//! integers, lengths as `u64`, fields in declaration order. There is no
//! self-description: a change to anything encoded changes [`FORMAT`].
//!
//! Also here, because every crate that reads a clock needs it: the base
//! that keeps the guest's wall clock continuous across a save and a load.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Duration;

pub mod clock;

/// The first bytes of every save state file.
pub const MAGIC: [u8; 8] = *b"PS5SLOPS";
/// Bumped whenever anything a save state holds changes shape.
pub const FORMAT: u32 = 1;

/// The tag that ends a file.
pub const END: [u8; 4] = *b"END.";

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error(e.to_string())
    }
}

pub fn error(what: impl fmt::Display) -> Error {
    Error(what.to_string())
}

#[derive(Default)]
pub struct Encoder {
    pub buf: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put<T: Encode + ?Sized>(&mut self, v: &T) -> &mut Self {
        v.encode(self);
        self
    }

    pub fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

pub struct Decoder<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    pub fn get<T: Decode>(&mut self) -> Result<T, Error> {
        T::decode(self)
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| error("save state section ends early"))?;
        let b = &self.buf[self.at..end];
        self.at = end;
        Ok(b)
    }

    pub fn is_empty(&self) -> bool {
        self.at == self.buf.len()
    }
}

pub trait Encode {
    fn encode(&self, e: &mut Encoder);
}

pub trait Decode: Sized {
    fn decode(d: &mut Decoder) -> Result<Self, Error>;
}

/// Encodes `v` on its own.
pub fn to_bytes<T: Encode + ?Sized>(v: &T) -> Vec<u8> {
    let mut e = Encoder::new();
    v.encode(&mut e);
    e.finish()
}

/// Decodes a `T` that must fill all of `b`.
pub fn from_bytes<T: Decode>(b: &[u8]) -> Result<T, Error> {
    let mut d = Decoder::new(b);
    let v = T::decode(&mut d)?;
    if !d.is_empty() {
        return Err(error("save state section has bytes left over"));
    }
    Ok(v)
}

macro_rules! numbers {
    ($($t:ty),*) => {$(
        impl Encode for $t {
            fn encode(&self, e: &mut Encoder) {
                e.bytes(&self.to_le_bytes());
            }
        }
        impl Decode for $t {
            fn decode(d: &mut Decoder) -> Result<Self, Error> {
                let b = d.take(size_of::<$t>())?;
                Ok(<$t>::from_le_bytes(b.try_into().expect("sized above")))
            }
        }
    )*};
}

numbers!(u8, u16, u32, u64, u128, i8, i16, i32, i64, f32, f64);

impl Encode for usize {
    fn encode(&self, e: &mut Encoder) {
        (*self as u64).encode(e);
    }
}

impl Decode for usize {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        usize::try_from(u64::decode(d)?).map_err(|_| error("length out of range"))
    }
}

impl Encode for bool {
    fn encode(&self, e: &mut Encoder) {
        (*self as u8).encode(e);
    }
}

impl Decode for bool {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        match u8::decode(d)? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(error(format!("bad boolean {v}"))),
        }
    }
}

impl Encode for str {
    fn encode(&self, e: &mut Encoder) {
        self.len().encode(e);
        e.bytes(self.as_bytes());
    }
}

impl Encode for String {
    fn encode(&self, e: &mut Encoder) {
        self.as_str().encode(e);
    }
}

impl Decode for String {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        let n = usize::decode(d)?;
        String::from_utf8(d.take(n)?.to_vec()).map_err(|_| error("bad string"))
    }
}

impl Encode for PathBuf {
    fn encode(&self, e: &mut Encoder) {
        use std::os::unix::ffi::OsStrExt;
        self.as_os_str().as_bytes().to_vec().encode(e);
    }
}

impl Decode for PathBuf {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        use std::os::unix::ffi::OsStringExt;
        Ok(PathBuf::from(std::ffi::OsString::from_vec(Vec::decode(d)?)))
    }
}

impl Encode for Duration {
    fn encode(&self, e: &mut Encoder) {
        (self.as_nanos() as u64).encode(e);
    }
}

impl Decode for Duration {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        Ok(Duration::from_nanos(u64::decode(d)?))
    }
}

impl<T: Encode> Encode for [T] {
    fn encode(&self, e: &mut Encoder) {
        self.len().encode(e);
        for v in self {
            v.encode(e);
        }
    }
}

impl<T: Encode> Encode for Vec<T> {
    fn encode(&self, e: &mut Encoder) {
        self.as_slice().encode(e);
    }
}

impl<T: Decode> Decode for Vec<T> {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        let n = usize::decode(d)?;
        let mut v = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            v.push(T::decode(d)?);
        }
        Ok(v)
    }
}

impl<T: Encode> Encode for VecDeque<T> {
    fn encode(&self, e: &mut Encoder) {
        self.len().encode(e);
        for v in self {
            v.encode(e);
        }
    }
}

impl<T: Decode> Decode for VecDeque<T> {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        Ok(Vec::decode(d)?.into())
    }
}

impl<T: Encode, const N: usize> Encode for [T; N] {
    fn encode(&self, e: &mut Encoder) {
        for v in self {
            v.encode(e);
        }
    }
}

impl<T: Decode, const N: usize> Decode for [T; N] {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        let mut v = Vec::with_capacity(N);
        for _ in 0..N {
            v.push(T::decode(d)?);
        }
        Ok(v.try_into().ok().expect("N values"))
    }
}

impl<T: Encode> Encode for Option<T> {
    fn encode(&self, e: &mut Encoder) {
        match self {
            None => 0u8.encode(e),
            Some(v) => {
                1u8.encode(e);
                v.encode(e);
            }
        }
    }
}

impl<T: Decode> Decode for Option<T> {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        match u8::decode(d)? {
            0 => Ok(None),
            1 => Ok(Some(T::decode(d)?)),
            v => Err(error(format!("bad option tag {v}"))),
        }
    }
}

impl<T: Encode + ?Sized> Encode for Box<T> {
    fn encode(&self, e: &mut Encoder) {
        (**self).encode(e);
    }
}

impl<T: Decode> Decode for Box<T> {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        Ok(Box::new(T::decode(d)?))
    }
}

impl<K: Encode + Ord, V: Encode> Encode for BTreeMap<K, V> {
    fn encode(&self, e: &mut Encoder) {
        self.len().encode(e);
        for (k, v) in self {
            k.encode(e);
            v.encode(e);
        }
    }
}

impl<K: Decode + Ord, V: Decode> Decode for BTreeMap<K, V> {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        Ok(Vec::<(K, V)>::decode(d)?.into_iter().collect())
    }
}

// Sorted, so that the same state always encodes to the same bytes.
impl<K: Encode + Ord + Hash, V: Encode> Encode for HashMap<K, V> {
    fn encode(&self, e: &mut Encoder) {
        let mut entries: Vec<(&K, &V)> = self.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        entries.len().encode(e);
        for (k, v) in entries {
            k.encode(e);
            v.encode(e);
        }
    }
}

impl<K: Decode + Eq + Hash, V: Decode> Decode for HashMap<K, V> {
    fn decode(d: &mut Decoder) -> Result<Self, Error> {
        Ok(Vec::<(K, V)>::decode(d)?.into_iter().collect())
    }
}

macro_rules! tuples {
    ($(($($n:ident),+)),*) => {$(
        #[allow(non_snake_case)]
        impl<$($n: Encode),+> Encode for ($($n,)+) {
            fn encode(&self, e: &mut Encoder) {
                let ($($n,)+) = self;
                $($n.encode(e);)+
            }
        }
        impl<$($n: Decode),+> Decode for ($($n,)+) {
            fn decode(d: &mut Decoder) -> Result<Self, Error> {
                Ok(($($n::decode(d)?,)+))
            }
        }
    )*};
}

tuples!(
    (A),
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F)
);

impl Encode for () {
    fn encode(&self, _: &mut Encoder) {}
}

impl Decode for () {
    fn decode(_: &mut Decoder) -> Result<Self, Error> {
        Ok(())
    }
}

/// Implements [`Encode`] and [`Decode`] for a struct by its fields, in the
/// order given (which must name every field).
#[macro_export]
macro_rules! record {
    ($t:ident { $($f:ident),* $(,)? }) => {
        impl $crate::Encode for $t {
            fn encode(&self, e: &mut $crate::Encoder) {
                $($crate::Encode::encode(&self.$f, e);)*
            }
        }
        impl $crate::Decode for $t {
            fn decode(d: &mut $crate::Decoder) -> Result<Self, $crate::Error> {
                Ok(Self { $($f: $crate::Decode::decode(d)?,)* })
            }
        }
    };
}

/// Compresses a section payload.
pub fn compress(raw: &[u8], level: i32) -> Vec<u8> {
    zstd::bulk::compress(raw, level).expect("compressing into memory does not fail")
}

/// Decompresses a section payload of `raw_len` bytes.
pub fn decompress(packed: &[u8], raw_len: usize) -> Result<Vec<u8>, Error> {
    let out = zstd::bulk::decompress(packed, raw_len)
        .map_err(|e| error(format!("corrupt save state section: {e}")))?;
    if out.len() != raw_len {
        return Err(error("save state section has the wrong size"));
    }
    Ok(out)
}

/// Writes a save state file: the header, then sections.
pub struct Writer<W: Write> {
    out: W,
    pub written: u64,
}

impl<W: Write> Writer<W> {
    pub fn new(mut out: W, header: &[u8]) -> io::Result<Self> {
        out.write_all(&MAGIC)?;
        out.write_all(&FORMAT.to_le_bytes())?;
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(header)?;
        Ok(Self {
            out,
            written: (MAGIC.len() + 12 + header.len()) as u64,
        })
    }

    /// A section from its uncompressed payload.
    pub fn section(&mut self, tag: [u8; 4], raw: &[u8]) -> io::Result<()> {
        self.packed(tag, raw.len() as u64, &compress(raw, 3))
    }

    /// A section whose payload of `raw_len` bytes was compressed already.
    pub fn packed(&mut self, tag: [u8; 4], raw_len: u64, packed: &[u8]) -> io::Result<()> {
        self.out.write_all(&tag)?;
        self.out.write_all(&raw_len.to_le_bytes())?;
        self.out.write_all(&(packed.len() as u64).to_le_bytes())?;
        self.out.write_all(packed)?;
        self.written += 20 + packed.len() as u64;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<W> {
        self.packed(END, 0, &[])?;
        self.out.flush()?;
        Ok(self.out)
    }
}

/// A section as read: its tag, uncompressed length and packed payload.
pub struct Section {
    pub tag: [u8; 4],
    pub raw_len: u64,
    pub packed: Vec<u8>,
}

impl Section {
    pub fn unpack(&self) -> Result<Vec<u8>, Error> {
        decompress(&self.packed, self.raw_len as usize)
    }

    pub fn name(&self) -> String {
        String::from_utf8_lossy(&self.tag).into_owned()
    }
}

/// Reads a save state file written by [`Writer`].
pub struct Reader<R: Read> {
    input: R,
    pub header: Vec<u8>,
    done: bool,
}

impl<R: Read> Reader<R> {
    /// Checks the magic number and format version and reads the header.
    pub fn new(mut input: R) -> Result<Self, Error> {
        let mut magic = [0u8; 8];
        input
            .read_exact(&mut magic)
            .map_err(|_| error("not a ps5slop save state (too short)"))?;
        if magic != MAGIC {
            return Err(error("not a ps5slop save state"));
        }
        let mut b = [0u8; 4];
        input.read_exact(&mut b)?;
        let format = u32::from_le_bytes(b);
        if format != FORMAT {
            return Err(error(format!(
                "save state format {format}, but this build reads format {FORMAT}"
            )));
        }
        let mut n = [0u8; 8];
        input.read_exact(&mut n)?;
        let len = u64::from_le_bytes(n);
        if len > 1 << 24 {
            return Err(error("save state header too large"));
        }
        let mut header = vec![0u8; len as usize];
        input.read_exact(&mut header)?;
        Ok(Self {
            input,
            header,
            done: false,
        })
    }

    /// The next section, or `None` after the end tag.
    pub fn next_section(&mut self) -> Result<Option<Section>, Error> {
        if self.done {
            return Ok(None);
        }
        let mut head = [0u8; 20];
        self.input
            .read_exact(&mut head)
            .map_err(|_| error("save state file is cut short"))?;
        let tag: [u8; 4] = head[..4].try_into().expect("4 bytes");
        let raw_len = u64::from_le_bytes(head[4..12].try_into().expect("8 bytes"));
        let len = u64::from_le_bytes(head[12..20].try_into().expect("8 bytes"));
        if tag == END {
            self.done = true;
            return Ok(None);
        }
        if len > 1 << 34 || raw_len > 1 << 34 {
            return Err(error("save state section too large"));
        }
        let mut packed = vec![0u8; len as usize];
        self.input
            .read_exact(&mut packed)
            .map_err(|_| error("save state file is cut short"))?;
        Ok(Some(Section {
            tag,
            raw_len,
            packed,
        }))
    }

    /// The next section, which must have tag `tag`, unpacked.
    pub fn expect(&mut self, tag: [u8; 4]) -> Result<Vec<u8>, Error> {
        match self.next_section()? {
            Some(s) if s.tag == tag => s.unpack(),
            Some(s) => Err(error(format!(
                "save state has section {} where {} belongs",
                s.name(),
                String::from_utf8_lossy(&tag)
            ))),
            None => Err(error(format!(
                "save state ends before section {}",
                String::from_utf8_lossy(&tag)
            ))),
        }
    }
}

/// A 64-bit FNV-1a hash, for fingerprints that must match across processes.
pub fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Default)]
    struct Sample {
        a: u32,
        b: Option<String>,
        c: Vec<(u64, i16)>,
        d: HashMap<u64, bool>,
        e: [u8; 3],
    }

    record!(Sample { a, b, c, d, e });

    #[test]
    fn records_round_trip() {
        let s = Sample {
            a: 7,
            b: Some("hi".into()),
            c: vec![(1, -2), (3, 4)],
            d: [(5, true), (2, false)].into_iter().collect(),
            e: [9, 8, 7],
        };
        let b = to_bytes(&s);
        assert_eq!(from_bytes::<Sample>(&b).unwrap(), s);
        assert_eq!(to_bytes(&from_bytes::<Sample>(&b).unwrap()), b);
    }

    #[test]
    fn short_or_long_input_is_an_error() {
        let b = to_bytes(&(1u64, 2u32));
        assert!(from_bytes::<(u64, u32)>(&b[..10]).is_err());
        assert!(from_bytes::<u64>(&b).is_err());
        assert!(from_bytes::<bool>(&[2]).is_err());
    }

    #[test]
    fn files_round_trip_and_check_their_format() {
        let mut w = Writer::new(Vec::new(), b"head").unwrap();
        w.section(*b"ONE.", &[1, 2, 3]).unwrap();
        w.section(*b"TWO.", &vec![0; 100_000]).unwrap();
        let file = w.finish().unwrap();
        let mut r = Reader::new(&file[..]).unwrap();
        assert_eq!(r.header, b"head");
        assert_eq!(r.expect(*b"ONE.").unwrap(), [1, 2, 3]);
        assert!(r.expect(*b"ONE.").is_err());
        assert!(r.next_section().unwrap().is_none());

        let mut old = file.clone();
        old[8] = 0;
        let e = Reader::new(&old[..]).err().unwrap();
        assert!(e.0.contains("format 0"), "{e}");
        assert!(Reader::new(&b"nonsense"[..]).is_err());
        let cut = &file[..file.len() - 30];
        let mut r = Reader::new(cut).unwrap();
        r.expect(*b"ONE.").unwrap();
        assert!(r.next_section().is_err());
    }
}
