//! The guest address space. Guest address `a` lives at host address
//! `base + a` inside one reservation, so a guest range is always one host
//! range. The host keeps every mapped page readable and writable; guest
//! protections are checked here on each access. See `docs/memory.md`.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::ptr::NonNull;

/// Orbis allocates in 16 KiB pages, and so does macOS on Apple Silicon.
pub const PAGE_SIZE: u64 = 0x4000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prot(u8);

impl Prot {
    pub const NONE: Prot = Prot(0);
    pub const READ: Prot = Prot(1);
    pub const WRITE: Prot = Prot(2);
    pub const EXEC: Prot = Prot(4);
    pub const RW: Prot = Prot(3);
    pub const RX: Prot = Prot(5);

    pub fn new(read: bool, write: bool, exec: bool) -> Self {
        Prot(read as u8 | (write as u8) << 1 | (exec as u8) << 2)
    }

    pub fn allows(self, access: Access) -> bool {
        let bit = match access {
            Access::Read => 1,
            Access::Write => 2,
            Access::Execute => 4,
        };
        self.0 & bit != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    Execute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    Unmapped,
    Protection,
    /// Execution reached a trap region: a call into the host.
    Trap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub addr: u64,
    pub access: Access,
    pub kind: FaultKind,
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} fault ({:?}) at {:#x}",
            self.access, self.kind, self.addr
        )
    }
}

impl std::error::Error for Fault {}

#[derive(Debug)]
pub enum MapError {
    Unaligned,
    OutOfRange,
    Overlaps,
    Host(io::Error),
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::Unaligned => f.write_str("address or length is not page aligned"),
            MapError::OutOfRange => f.write_str("range is outside the guest address space"),
            MapError::Overlaps => f.write_str("range overlaps a mapped region"),
            MapError::Host(e) => write!(f, "host mapping failed: {e}"),
        }
    }
}

impl std::error::Error for MapError {}

#[derive(Debug, Clone)]
struct Region {
    end: u64,
    prot: Prot,
    trap: bool,
    name: String,
}

/// A mapped region, as listed by [`Memory::regions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionInfo {
    pub start: u64,
    pub end: u64,
    pub prot: Prot,
    pub trap: bool,
    pub name: String,
}

pub struct Memory {
    base: NonNull<u8>,
    size: u64,
    regions: BTreeMap<u64, Region>,
    /// The last region an access hit: `(start, end, prot, trap)`.
    last: Cell<Option<(u64, u64, Prot, bool)>>,
}

// The reservation is owned by this value and only reached through it.
unsafe impl Send for Memory {}

impl Memory {
    /// Reserves `size` bytes of guest address space (a multiple of [`PAGE_SIZE`]).
    pub fn new(size: u64) -> io::Result<Self> {
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "size must be a non-zero multiple of the page size",
            ));
        }
        let len = usize::try_from(size)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "size too large"))?;
        // SAFETY: an anonymous PROT_NONE reservation; no existing memory is touched.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            base: NonNull::new(p.cast()).expect("mmap returned null"),
            size,
            regions: BTreeMap::new(),
            last: Cell::new(None),
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    fn check_range(&self, addr: u64, len: u64) -> Result<(), MapError> {
        if !addr.is_multiple_of(PAGE_SIZE) || !len.is_multiple_of(PAGE_SIZE) || len == 0 {
            return Err(MapError::Unaligned);
        }
        if addr.checked_add(len).is_none_or(|end| end > self.size) {
            return Err(MapError::OutOfRange);
        }
        Ok(())
    }

    fn overlaps(&self, addr: u64, end: u64) -> bool {
        self.regions
            .range(..end)
            .next_back()
            .is_some_and(|(_, r)| r.end > addr)
    }

    fn host_protect(&self, addr: u64, len: u64, prot: libc::c_int) -> Result<(), MapError> {
        // SAFETY: the range was checked to lie inside our reservation.
        let r = unsafe {
            libc::mprotect(
                self.base.as_ptr().add(addr as usize).cast(),
                len as usize,
                prot,
            )
        };
        if r != 0 {
            return Err(MapError::Host(io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Maps zero-filled memory at `[addr, addr + len)`.
    pub fn map(&mut self, addr: u64, len: u64, prot: Prot, name: &str) -> Result<(), MapError> {
        self.insert(addr, len, prot, false, name)
    }

    /// Maps a region that faults with [`FaultKind::Trap`] when executed and
    /// cannot be read or written: the guest-visible addresses of host functions.
    pub fn map_trap(&mut self, addr: u64, len: u64, name: &str) -> Result<(), MapError> {
        self.insert(addr, len, Prot::NONE, true, name)
    }

    fn insert(
        &mut self,
        addr: u64,
        len: u64,
        prot: Prot,
        trap: bool,
        name: &str,
    ) -> Result<(), MapError> {
        self.check_range(addr, len)?;
        let end = addr + len;
        if self.overlaps(addr, end) {
            return Err(MapError::Overlaps);
        }
        if !trap {
            self.host_protect(addr, len, libc::PROT_READ | libc::PROT_WRITE)?;
        }
        self.regions.insert(
            addr,
            Region {
                end,
                prot,
                trap,
                name: name.to_owned(),
            },
        );
        self.last.set(None);
        Ok(())
    }

    /// Unmaps the region starting exactly at `addr` and drops its contents.
    pub fn unmap(&mut self, addr: u64) -> Result<(), MapError> {
        let r = self.regions.remove(&addr).ok_or(MapError::OutOfRange)?;
        self.last.set(None);
        if !r.trap {
            let len = r.end - addr;
            // SAFETY: the range is inside our reservation; MAP_FIXED replaces it with a fresh reservation.
            let p = unsafe {
                libc::mmap(
                    self.base.as_ptr().add(addr as usize).cast(),
                    len as usize,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE | libc::MAP_FIXED,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(MapError::Host(io::Error::last_os_error()));
            }
        }
        Ok(())
    }

    /// Changes the guest protection of the region starting exactly at `addr`.
    pub fn protect(&mut self, addr: u64, prot: Prot) -> Result<(), MapError> {
        let r = self.regions.get_mut(&addr).ok_or(MapError::OutOfRange)?;
        if r.trap {
            return Err(MapError::Overlaps);
        }
        r.prot = prot;
        self.last.set(None);
        Ok(())
    }

    /// The lowest page-aligned free range of `len` bytes at or above `min`.
    pub fn find_free(&self, min: u64, len: u64) -> Option<u64> {
        let len = len.next_multiple_of(PAGE_SIZE);
        let mut candidate = min.next_multiple_of(PAGE_SIZE);
        if let Some((_, r)) = self.regions.range(..=candidate).next_back() {
            candidate = candidate.max(r.end);
        }
        for (&start, r) in self.regions.range(candidate..) {
            if start >= candidate + len {
                break;
            }
            candidate = candidate.max(r.end);
        }
        (candidate.checked_add(len)? <= self.size).then_some(candidate)
    }

    pub fn regions(&self) -> Vec<RegionInfo> {
        self.regions
            .iter()
            .map(|(&start, r)| RegionInfo {
                start,
                end: r.end,
                prot: r.prot,
                trap: r.trap,
                name: r.name.clone(),
            })
            .collect()
    }

    /// Checks that `[addr, addr + len)` may be accessed as `access`.
    fn check(&self, addr: u64, len: usize, access: Access) -> Result<(), Fault> {
        let fault = |at, kind| {
            Err(Fault {
                addr: at,
                access,
                kind,
            })
        };
        let end = match addr.checked_add(len as u64) {
            Some(e) => e,
            None => return fault(addr, FaultKind::Unmapped),
        };
        let mut at = addr;
        while at < end {
            let (_, rend, prot, trap) = match self.last.get() {
                Some(l) if at >= l.0 && at < l.1 => l,
                _ => match self.regions.range(..=at).next_back() {
                    Some((&s, r)) if at < r.end => {
                        let l = (s, r.end, r.prot, r.trap);
                        self.last.set(Some(l));
                        l
                    }
                    _ => return fault(at, FaultKind::Unmapped),
                },
            };
            if trap {
                return fault(
                    at,
                    if access == Access::Execute {
                        FaultKind::Trap
                    } else {
                        FaultKind::Protection
                    },
                );
            }
            if !prot.allows(access) {
                return fault(at, FaultKind::Protection);
            }
            at = rend;
        }
        Ok(())
    }

    fn host(&self, addr: u64) -> *mut u8 {
        // SAFETY (for callers): `check` has confirmed the range is mapped inside the reservation.
        unsafe { self.base.as_ptr().add(addr as usize) }
    }

    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.check(addr, buf.len(), Access::Read)?;
        // SAFETY: checked above; host pages of mapped regions are readable.
        unsafe { std::ptr::copy_nonoverlapping(self.host(addr), buf.as_mut_ptr(), buf.len()) };
        Ok(())
    }

    /// Reads instruction bytes: like [`Memory::read`], but needs execute permission.
    pub fn fetch(&self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.check(addr, buf.len(), Access::Execute)?;
        // SAFETY: checked above.
        unsafe { std::ptr::copy_nonoverlapping(self.host(addr), buf.as_mut_ptr(), buf.len()) };
        Ok(())
    }

    pub fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        self.check(addr, data.len(), Access::Write)?;
        // SAFETY: checked above; host pages of mapped regions are writable.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.host(addr), data.len()) };
        Ok(())
    }

    /// Writes regardless of guest protection, for the loader and the host.
    pub fn write_host(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        self.check(addr, data.len(), Access::Read).or_else(|f| {
            if f.kind == FaultKind::Protection {
                Ok(())
            } else {
                Err(f)
            }
        })?;
        // SAFETY: the range is mapped and not a trap region (those fail the check).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.host(addr), data.len()) };
        Ok(())
    }

    pub fn read_u64(&self, addr: u64) -> Result<u64, Fault> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn write_u64(&mut self, addr: u64, v: u64) -> Result<(), Fault> {
        self.write(addr, &v.to_le_bytes())
    }

    /// Reads a NUL-terminated string of at most `max` bytes.
    pub fn read_cstr(&self, addr: u64, max: usize) -> Result<Vec<u8>, Fault> {
        let mut out = Vec::new();
        for i in 0..max as u64 {
            let mut b = [0u8];
            self.read(addr + i, &mut b)?;
            if b[0] == 0 {
                break;
            }
            out.push(b[0]);
        }
        Ok(out)
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: we own the whole reservation.
        unsafe { libc::munmap(self.base.as_ptr().cast(), self.size as usize) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn mapped_memory_reads_back_what_was_written() {
        let mut m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, PAGE_SIZE, Prot::RW, "data").unwrap();
        m.write(0x40_0ff8, &0x1122_3344_5566_7788u64.to_le_bytes())
            .unwrap();
        assert_eq!(m.read_u64(0x40_0ff8).unwrap(), 0x1122_3344_5566_7788);
        let mut fresh = [0xffu8; 4];
        m.read(0x40_1000, &mut fresh).unwrap();
        assert_eq!(fresh, [0; 4]);
    }

    #[test]
    fn unmapped_and_protected_accesses_fault() {
        let mut m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, PAGE_SIZE, Prot::RX, "code").unwrap();
        let mut b = [0u8; 8];
        assert_eq!(
            m.read(0x80_0000, &mut b).unwrap_err().kind,
            FaultKind::Unmapped
        );
        assert_eq!(
            m.write(0x40_0000, &b).unwrap_err().kind,
            FaultKind::Protection
        );
        assert!(m.fetch(0x40_0000, &mut b).is_ok());
        let crossing = m.read(0x40_0000 + PAGE_SIZE - 4, &mut b).unwrap_err();
        assert_eq!(
            (crossing.addr, crossing.kind),
            (0x40_0000 + PAGE_SIZE, FaultKind::Unmapped)
        );
    }

    #[test]
    fn host_writes_ignore_guest_protection() {
        let mut m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, PAGE_SIZE, Prot::RX, "code").unwrap();
        m.write_host(0x40_0000, &[0xc3]).unwrap();
        let mut b = [0u8];
        m.fetch(0x40_0000, &mut b).unwrap();
        assert_eq!(b, [0xc3]);
    }

    #[test]
    fn executing_a_trap_region_reports_a_trap() {
        let mut m = Memory::new(4 * GIB).unwrap();
        m.map_trap(0x1000_0000, PAGE_SIZE, "hle").unwrap();
        let mut b = [0u8; 1];
        let f = m.fetch(0x1000_0010, &mut b).unwrap_err();
        assert_eq!((f.addr, f.kind), (0x1000_0010, FaultKind::Trap));
        assert_eq!(
            m.read(0x1000_0010, &mut b).unwrap_err().kind,
            FaultKind::Protection
        );
    }

    #[test]
    fn regions_cannot_overlap_and_must_be_aligned() {
        let mut m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, 2 * PAGE_SIZE, Prot::RW, "a").unwrap();
        assert!(matches!(
            m.map(0x40_4000, PAGE_SIZE, Prot::RW, "b"),
            Err(MapError::Overlaps)
        ));
        assert!(matches!(
            m.map(0x40_1000, PAGE_SIZE, Prot::RW, "c"),
            Err(MapError::Unaligned)
        ));
        assert!(matches!(
            m.map(4 * GIB, PAGE_SIZE, Prot::RW, "d"),
            Err(MapError::OutOfRange)
        ));
    }

    #[test]
    fn unmapping_drops_contents() {
        let mut m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, PAGE_SIZE, Prot::RW, "a").unwrap();
        m.write(0x40_0000, &[7]).unwrap();
        m.unmap(0x40_0000).unwrap();
        let mut b = [0u8];
        assert!(m.read(0x40_0000, &mut b).is_err());
        m.map(0x40_0000, PAGE_SIZE, Prot::RW, "a").unwrap();
        m.read(0x40_0000, &mut b).unwrap();
        assert_eq!(b, [0]);
    }

    #[test]
    fn finds_free_ranges_between_regions() {
        let mut m = Memory::new(4 * GIB).unwrap();
        m.map(0x10_0000, PAGE_SIZE, Prot::RW, "a").unwrap();
        m.map(0x10_8000, PAGE_SIZE, Prot::RW, "b").unwrap();
        assert_eq!(m.find_free(0x10_0000, PAGE_SIZE), Some(0x10_4000));
        assert_eq!(m.find_free(0x10_0000, 2 * PAGE_SIZE), Some(0x10_c000));
        assert_eq!(m.find_free(4 * GIB - PAGE_SIZE, 2 * PAGE_SIZE), None);
    }
}
