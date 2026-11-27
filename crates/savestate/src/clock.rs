//! The guest's wall clock: a base (Unix time at host counter zero) plus the
//! host's monotonic counter, `mach_absolute_time`. Translated code computes
//! the same sum from CNTVCT_EL0, which reads that counter. The base starts
//! at the host's Unix time, and a load moves it so that the guest sees time
//! go on from where it was saved instead of jumping.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Unix nanoseconds at counter zero; 0 until first used.
static BASE: AtomicU64 = AtomicU64::new(0);

/// The counter's period as a fraction of a nanosecond: ns = ticks * numer / denom.
#[allow(deprecated)]
pub fn timebase() -> (u32, u32) {
    static TIMEBASE: OnceLock<(u32, u32)> = OnceLock::new();
    *TIMEBASE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            let mut info = libc::mach_timebase_info { numer: 0, denom: 0 };
            // SAFETY: fills the struct we pass.
            unsafe { libc::mach_timebase_info(&mut info) };
            (info.numer, info.denom)
        }
        #[cfg(not(target_os = "macos"))]
        {
            (1, 1)
        }
    })
}

/// The host's monotonic counter, in [`timebase`] units.
#[allow(deprecated)]
pub fn ticks() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: no arguments.
        unsafe { libc::mach_absolute_time() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_nanos() as u64
    }
}

/// `ticks` in nanoseconds. Translated code computes the same with 64-bit
/// MUL and UDIV, which wrap only after centuries of host uptime.
pub fn ticks_to_nanos(ticks: u64) -> u64 {
    let (numer, denom) = timebase();
    ticks.wrapping_mul(numer as u64) / denom as u64
}

/// The guest's Unix time in nanoseconds at counter zero.
pub fn base() -> u64 {
    let b = BASE.load(Ordering::Relaxed);
    if b != 0 {
        return b;
    }
    let host = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let fresh = (host.as_nanos() as u64).wrapping_sub(ticks_to_nanos(ticks()));
    match BASE.compare_exchange(0, fresh, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => fresh,
        Err(set) => set,
    }
}

/// The guest's time since the Unix epoch in nanoseconds.
pub fn unix_nanos() -> u64 {
    base().wrapping_add(ticks_to_nanos(ticks()))
}

/// The guest's time since the Unix epoch.
pub fn unix() -> Duration {
    Duration::from_nanos(unix_nanos())
}

/// Makes the guest's wall clock read `guest` now. Translated code sees the
/// new base at its thread's next entry into the VM.
pub fn set_unix(guest: Duration) {
    let b = (guest.as_nanos() as u64).wrapping_sub(ticks_to_nanos(ticks()));
    BASE.store(b.max(1), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_unix_moves_the_clock_and_it_goes_on_from_there() {
        let guest = Duration::from_secs(1_000_000_000);
        set_unix(guest);
        let first = unix();
        let second = unix();
        assert!(guest <= first && first <= second, "{first:?} {second:?}");
        assert!(second - guest < Duration::from_secs(1));
    }
}
