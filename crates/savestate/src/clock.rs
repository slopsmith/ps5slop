//! The guest's wall clock: the host's, minus an offset a load sets so that
//! the guest sees time go on from where it was saved instead of jumping.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

static BEHIND_NS: AtomicU64 = AtomicU64::new(0);

/// The guest's time since the Unix epoch.
pub fn unix() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .saturating_sub(Duration::from_nanos(BEHIND_NS.load(Ordering::Relaxed)))
}

/// The guest's time since the Unix epoch in nanoseconds.
pub fn unix_nanos() -> u64 {
    unix().as_nanos() as u64
}

/// Makes the guest's wall clock read `guest` now.
pub fn set_unix(guest: Duration) {
    let host = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    BEHIND_NS.store(
        host.saturating_sub(guest).as_nanos() as u64,
        Ordering::Relaxed,
    );
}
