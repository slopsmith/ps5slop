//! The guest address space. Guest address `a` lives at host address
//! `base + a` inside one reservation, so a guest range is always one host
//! range. The host keeps every mapped page readable and writable; guest
//! protections are checked here on each access. Threads share an address
//! space through cloned handles. See `docs/memory.md`.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering, fence};
use std::sync::{Arc, Mutex, RwLock};

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

struct Shared {
    base: NonNull<u8>,
    size: u64,
    regions: RwLock<BTreeMap<u64, Region>>,
    /// Bumped on every change to the region table, so handles drop their caches.
    generation: AtomicU64,
    /// Serialises locked operations on misaligned addresses, which host atomics cannot do.
    misaligned_atomics: Mutex<()>,
}

// The reservation is owned by `Shared` and only reached through `Memory` handles,
// which check every access against the region table.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: we own the whole reservation, and no handle outlives `Shared`.
        unsafe { libc::munmap(self.base.as_ptr().cast(), self.size as usize) };
    }
}

#[derive(Clone, Copy)]
struct Cached {
    start: u64,
    end: u64,
    prot: Prot,
    trap: bool,
    generation: u64,
}

/// A handle on a guest address space. Clones share the address space; give
/// each thread its own clone (a handle caches the last region it touched).
///
/// Accesses follow x86-TSO: a load is followed by a load barrier and a store
/// is preceded by a store barrier, and aligned accesses of 1, 2, 4 or 8 bytes
/// are single host accesses.
pub struct Memory {
    shared: Arc<Shared>,
    last: Cell<Option<Cached>>,
}

impl Clone for Memory {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            last: Cell::new(None),
        }
    }
}

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
            shared: Arc::new(Shared {
                base: NonNull::new(p.cast()).expect("mmap returned null"),
                size,
                regions: RwLock::new(BTreeMap::new()),
                generation: AtomicU64::new(0),
                misaligned_atomics: Mutex::new(()),
            }),
            last: Cell::new(None),
        })
    }

    pub fn size(&self) -> u64 {
        self.shared.size
    }

    /// A number that changes whenever regions are mapped, unmapped or
    /// change protection, for code that mirrors the region table.
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(Ordering::Acquire)
    }

    /// The host address that guest address 0 corresponds to; guest address
    /// `a` is at `host_base() + a`. For mapping guest memory into a VM.
    pub fn host_base(&self) -> *mut u8 {
        self.shared.base.as_ptr()
    }

    fn regions_read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<u64, Region>> {
        self.shared
            .regions
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn regions_write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<u64, Region>> {
        self.shared
            .regions
            .write()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn changed(&self) {
        self.shared.generation.fetch_add(1, Ordering::Release);
    }

    fn check_range(&self, addr: u64, len: u64) -> Result<(), MapError> {
        if !addr.is_multiple_of(PAGE_SIZE) || !len.is_multiple_of(PAGE_SIZE) || len == 0 {
            return Err(MapError::Unaligned);
        }
        if addr
            .checked_add(len)
            .is_none_or(|end| end > self.shared.size)
        {
            return Err(MapError::OutOfRange);
        }
        Ok(())
    }

    // Mapped host pages never become inaccessible again: a thread racing
    // with an unmap reads zeros instead of crashing the host.
    fn host_fresh(&self, addr: u64, len: u64) -> Result<(), MapError> {
        // SAFETY: the range was checked to lie inside our reservation; MAP_FIXED replaces it.
        let p = unsafe {
            libc::mmap(
                self.host(addr).cast(),
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(MapError::Host(io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Maps zero-filled memory at `[addr, addr + len)`.
    pub fn map(&self, addr: u64, len: u64, prot: Prot, name: &str) -> Result<(), MapError> {
        self.insert(addr, len, prot, false, name)
    }

    /// Maps a region that faults with [`FaultKind::Trap`] when executed and
    /// cannot be read or written: the guest-visible addresses of host functions.
    pub fn map_trap(&self, addr: u64, len: u64, name: &str) -> Result<(), MapError> {
        self.insert(addr, len, Prot::NONE, true, name)
    }

    fn insert(
        &self,
        addr: u64,
        len: u64,
        prot: Prot,
        trap: bool,
        name: &str,
    ) -> Result<(), MapError> {
        self.check_range(addr, len)?;
        let end = addr + len;
        let mut regions = self.regions_write();
        if regions
            .range(..end)
            .next_back()
            .is_some_and(|(_, r)| r.end > addr)
        {
            return Err(MapError::Overlaps);
        }
        self.host_fresh(addr, len)?;
        regions.insert(
            addr,
            Region {
                end,
                prot,
                trap,
                name: name.to_owned(),
            },
        );
        drop(regions);
        self.changed();
        Ok(())
    }

    /// Unmaps the region starting exactly at `addr` and drops its contents.
    pub fn unmap(&self, addr: u64) -> Result<(), MapError> {
        let mut regions = self.regions_write();
        let r = regions.remove(&addr).ok_or(MapError::OutOfRange)?;
        drop(regions);
        self.changed();
        self.host_fresh(addr, r.end - addr)
    }

    /// Changes the guest protection of the region starting exactly at `addr`.
    pub fn protect(&self, addr: u64, prot: Prot) -> Result<(), MapError> {
        let mut regions = self.regions_write();
        let r = regions.get_mut(&addr).ok_or(MapError::OutOfRange)?;
        if r.trap {
            return Err(MapError::Overlaps);
        }
        r.prot = prot;
        drop(regions);
        self.changed();
        Ok(())
    }

    /// The lowest page-aligned free range of `len` bytes at or above `min`.
    pub fn find_free(&self, min: u64, len: u64) -> Option<u64> {
        let regions = self.regions_read();
        let len = len.next_multiple_of(PAGE_SIZE);
        let mut candidate = min.next_multiple_of(PAGE_SIZE);
        if let Some((_, r)) = regions.range(..=candidate).next_back() {
            candidate = candidate.max(r.end);
        }
        for (&start, r) in regions.range(candidate..) {
            if start >= candidate + len {
                break;
            }
            candidate = candidate.max(r.end);
        }
        (candidate.checked_add(len)? <= self.shared.size).then_some(candidate)
    }

    pub fn regions(&self) -> Vec<RegionInfo> {
        self.regions_read()
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

    /// The region containing `at`, from this handle's cache when it is current.
    fn region(&self, at: u64) -> Option<Cached> {
        let generation = self.shared.generation.load(Ordering::Acquire);
        if let Some(c) = self.last.get()
            && c.generation == generation
            && at >= c.start
            && at < c.end
        {
            return Some(c);
        }
        let regions = self.regions_read();
        let (&start, r) = regions.range(..=at).next_back()?;
        if at >= r.end {
            return None;
        }
        let c = Cached {
            start,
            end: r.end,
            prot: r.prot,
            trap: r.trap,
            generation,
        };
        self.last.set(Some(c));
        Some(c)
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
            let Some(c) = self.region(at) else {
                return fault(at, FaultKind::Unmapped);
            };
            if c.trap {
                let kind = if access == Access::Execute {
                    FaultKind::Trap
                } else {
                    FaultKind::Protection
                };
                return fault(at, kind);
            }
            if !c.prot.allows(access) {
                return fault(at, FaultKind::Protection);
            }
            at = c.end;
        }
        Ok(())
    }

    fn host(&self, addr: u64) -> *mut u8 {
        // SAFETY (for callers): `check` has confirmed the range is mapped inside the reservation.
        unsafe { self.shared.base.as_ptr().add(addr as usize) }
    }

    /// Copies from guest memory without a barrier; single host loads for 1, 2, 4 and 8 bytes.
    fn copy_out(&self, addr: u64, buf: &mut [u8]) {
        let p = self.host(addr);
        // SAFETY: the caller checked the range; host pages of mapped regions are readable.
        unsafe {
            match buf.len() {
                1 => buf[0] = p.read(),
                2 => buf.copy_from_slice(&p.cast::<u16>().read_unaligned().to_ne_bytes()),
                4 => buf.copy_from_slice(&p.cast::<u32>().read_unaligned().to_ne_bytes()),
                8 => buf.copy_from_slice(&p.cast::<u64>().read_unaligned().to_ne_bytes()),
                n => std::ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), n),
            }
        }
    }

    fn copy_in(&self, addr: u64, data: &[u8]) {
        let p = self.host(addr);
        // SAFETY: the caller checked the range; host pages of mapped regions are writable.
        unsafe {
            match data.len() {
                1 => p.write(data[0]),
                2 => p
                    .cast::<u16>()
                    .write_unaligned(u16::from_ne_bytes(data.try_into().unwrap())),
                4 => p
                    .cast::<u32>()
                    .write_unaligned(u32::from_ne_bytes(data.try_into().unwrap())),
                8 => p
                    .cast::<u64>()
                    .write_unaligned(u64::from_ne_bytes(data.try_into().unwrap())),
                n => std::ptr::copy_nonoverlapping(data.as_ptr(), p, n),
            }
        }
    }

    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.check(addr, buf.len(), Access::Read)?;
        self.copy_out(addr, buf);
        fence(Ordering::Acquire);
        Ok(())
    }

    /// Reads instruction bytes: like [`Memory::read`], but needs execute permission.
    pub fn fetch(&self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.check(addr, buf.len(), Access::Execute)?;
        self.copy_out(addr, buf);
        Ok(())
    }

    pub fn write(&self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        self.check(addr, data.len(), Access::Write)?;
        fence(Ordering::Release);
        self.copy_in(addr, data);
        Ok(())
    }

    /// Writes regardless of guest protection, for the loader and the host.
    pub fn write_host(&self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        self.check(addr, data.len(), Access::Read).or_else(|f| {
            if f.kind == FaultKind::Protection {
                Ok(())
            } else {
                Err(f)
            }
        })?;
        fence(Ordering::Release);
        self.copy_in(addr, data);
        Ok(())
    }

    /// Atomically replaces the `size`-byte value at `addr` with `f(old)`, or
    /// leaves it when `f` returns `None`, and returns the old value. `f` may
    /// run more than once. This is a full barrier, like a locked x86
    /// instruction. Misaligned addresses fall back to a lock that only
    /// excludes other atomic updates.
    pub fn atomic_update(
        &self,
        addr: u64,
        size: u8,
        mut f: impl FnMut(u64) -> Option<u64>,
    ) -> Result<u64, Fault> {
        self.check(addr, size as usize, Access::Read)?;
        self.check(addr, size as usize, Access::Write)?;
        let p = self.host(addr);
        macro_rules! cas {
            ($atomic:ty, $int:ty) => {{
                // SAFETY: checked, mapped and aligned to the atomic's size.
                let a = unsafe { <$atomic>::from_ptr(p.cast::<$int>()) };
                let mut old = a.load(Ordering::SeqCst);
                loop {
                    let Some(new) = f(old as u64) else {
                        fence(Ordering::SeqCst);
                        return Ok(old as u64);
                    };
                    match a.compare_exchange_weak(
                        old,
                        new as $int,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => return Ok(old as u64),
                        Err(cur) => old = cur,
                    }
                }
            }};
        }
        if addr.is_multiple_of(size as u64) {
            match size {
                1 => cas!(AtomicU8, u8),
                2 => cas!(AtomicU16, u16),
                4 => cas!(AtomicU32, u32),
                8 => cas!(AtomicU64, u64),
                _ => {}
            }
        }
        let _guard = self
            .shared
            .misaligned_atomics
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        fence(Ordering::SeqCst);
        let mut b = [0u8; 8];
        self.copy_out(addr, &mut b[..size as usize]);
        let old = u64::from_le_bytes(b);
        if let Some(new) = f(old) {
            self.copy_in(addr, &new.to_le_bytes()[..size as usize]);
        }
        fence(Ordering::SeqCst);
        Ok(old)
    }

    pub fn read_u64(&self, addr: u64) -> Result<u64, Fault> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn write_u64(&self, addr: u64, v: u64) -> Result<(), Fault> {
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

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn mapped_memory_reads_back_what_was_written() {
        let m = Memory::new(4 * GIB).unwrap();
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
        let m = Memory::new(4 * GIB).unwrap();
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
        let m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, PAGE_SIZE, Prot::RX, "code").unwrap();
        m.write_host(0x40_0000, &[0xc3]).unwrap();
        let mut b = [0u8];
        m.fetch(0x40_0000, &mut b).unwrap();
        assert_eq!(b, [0xc3]);
    }

    #[test]
    fn executing_a_trap_region_reports_a_trap() {
        let m = Memory::new(4 * GIB).unwrap();
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
        let m = Memory::new(4 * GIB).unwrap();
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
        let m = Memory::new(4 * GIB).unwrap();
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
    fn atomic_updates_from_many_threads_lose_nothing() {
        let m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, PAGE_SIZE, Prot::RW, "a").unwrap();
        std::thread::scope(|s| {
            for _ in 0..4 {
                let h = m.clone();
                s.spawn(move || {
                    for _ in 0..10_000 {
                        h.atomic_update(0x40_0008, 8, |v| Some(v + 1)).unwrap();
                        h.atomic_update(0x40_0013, 4, |v| Some(v + 1)).unwrap();
                    }
                });
            }
        });
        assert_eq!(m.read_u64(0x40_0008).unwrap(), 40_000);
        let mut b = [0u8; 4];
        m.read(0x40_0013, &mut b).unwrap();
        assert_eq!(u32::from_le_bytes(b), 40_000);
    }

    #[test]
    fn a_declined_update_writes_nothing_and_returns_the_old_value() {
        let m = Memory::new(4 * GIB).unwrap();
        m.map(0x40_0000, PAGE_SIZE, Prot::RW, "a").unwrap();
        m.write_u64(0x40_0000, 5).unwrap();
        assert_eq!(m.atomic_update(0x40_0000, 8, |_| None).unwrap(), 5);
        assert_eq!(m.read_u64(0x40_0000).unwrap(), 5);
    }

    #[test]
    fn other_handles_see_unmaps() {
        let a = Memory::new(4 * GIB).unwrap();
        a.map(0x40_0000, PAGE_SIZE, Prot::RW, "a").unwrap();
        let b = a.clone();
        b.write_u64(0x40_0000, 1).unwrap();
        a.unmap(0x40_0000).unwrap();
        assert_eq!(b.read_u64(0x40_0000).unwrap_err().kind, FaultKind::Unmapped);
    }

    #[test]
    fn finds_free_ranges_between_regions() {
        let m = Memory::new(4 * GIB).unwrap();
        m.map(0x10_0000, PAGE_SIZE, Prot::RW, "a").unwrap();
        m.map(0x10_8000, PAGE_SIZE, Prot::RW, "b").unwrap();
        assert_eq!(m.find_free(0x10_0000, PAGE_SIZE), Some(0x10_4000));
        assert_eq!(m.find_free(0x10_0000, 2 * PAGE_SIZE), Some(0x10_c000));
        assert_eq!(m.find_free(4 * GIB - PAGE_SIZE, 2 * PAGE_SIZE), None);
    }
}
