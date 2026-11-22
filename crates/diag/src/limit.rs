use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Lets the first `limit` calls through, for reports that would otherwise
/// repeat every frame.
pub struct FirstN {
    limit: u32,
    seen: AtomicU32,
}

impl FirstN {
    pub const fn new(limit: u32) -> Self {
        Self {
            limit,
            seen: AtomicU32::new(0),
        }
    }

    pub fn allow(&self) -> bool {
        self.next().is_some()
    }

    /// `Some(last)` for one of the first `limit` calls, where `last` says
    /// that later calls are dropped.
    pub fn next(&self) -> Option<bool> {
        if self.seen.load(Relaxed) >= self.limit {
            return None;
        }
        let n = self.seen.fetch_add(1, Relaxed);
        (n < self.limit).then_some(n + 1 == self.limit)
    }
}

/// Lets each distinct key through once, up to `limit` keys.
pub struct Distinct {
    limit: usize,
    seen: Mutex<Option<HashSet<String>>>,
}

impl Distinct {
    pub const fn new(limit: usize) -> Self {
        Self {
            limit,
            seen: Mutex::new(None),
        }
    }

    pub fn first(&self, key: &str) -> bool {
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_default();
        seen.len() < self.limit && !seen.contains(key) && seen.insert(key.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_n_lets_the_first_calls_through_and_marks_the_last() {
        let limit = FirstN::new(2);
        assert_eq!(limit.next(), Some(false));
        assert_eq!(limit.next(), Some(true));
        assert_eq!(limit.next(), None);
        assert!(!limit.allow());
    }

    #[test]
    fn first_n_with_no_room_lets_nothing_through() {
        assert!(!FirstN::new(0).allow());
    }

    #[test]
    fn distinct_lets_each_key_through_once() {
        let seen = Distinct::new(8);
        assert!(seen.first("a"));
        assert!(!seen.first("a"));
        assert!(seen.first("b"));
    }

    #[test]
    fn distinct_stops_after_its_limit_of_keys() {
        let seen = Distinct::new(1);
        assert!(seen.first("a"));
        assert!(!seen.first("b"));
    }
}
