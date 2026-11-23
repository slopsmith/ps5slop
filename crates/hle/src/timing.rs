//! Waits that land on their deadline despite macOS timer coalescing, which
//! may fire a timer up to half its length late (`kern.timer_coalesce_*`).

use std::time::{Duration, Instant};

/// Below this, a wait asks for all the time left: the slack is small.
const SHORT: Duration = Duration::from_micros(300);

/// How long to block when `left` remains until a deadline. Callers loop
/// until the deadline, so asking for less costs a wake-up, not accuracy.
pub(crate) fn step(left: Duration) -> Duration {
    if left > SHORT { left * 2 / 3 } else { left }
}

/// Sleeps until `deadline`.
pub(crate) fn sleep_until(deadline: Instant) {
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        std::thread::sleep(step(left));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_millisecond_sleeps_end_on_time() {
        let mut took: Vec<Duration> = (0..21)
            .map(|_| {
                let start = Instant::now();
                sleep_until(start + Duration::from_millis(5));
                start.elapsed()
            })
            .collect();
        took.sort();
        let median = took[took.len() / 2];
        assert!(
            median < Duration::from_micros(5600),
            "timed waits oversleep, so games polling on short timeouts fall behind ({median:?})"
        );
    }
}
