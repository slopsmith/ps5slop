//! Per-run problem report: collects work skipped, failed or approximated
//! across subsystems and formats it grouped by reason and sorted by impact.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

const MAX_PER_REASON: usize = 32;
const MAX_DETAIL_CHARS: usize = 200;

/// Subsystem category for a recorded problem. Ordered by rendering/runtime
/// priority so lost draws appear first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    DrawSkipped,
    DispatchSkipped,
    ShaderTranslate,
    PipelineFailed,
    UnsupportedFormat,
    MemoryWindow,
    LoopGuard,
    WaitTimeout,
    UnimplementedHle,
    JitFallback,
    AudioLate,
}

impl Kind {
    pub fn heading(self) -> &'static str {
        match self {
            Kind::DrawSkipped => "draws skipped",
            Kind::DispatchSkipped => "dispatches skipped",
            Kind::ShaderTranslate => "programs not translated",
            Kind::PipelineFailed => "failed pipelines",
            Kind::UnsupportedFormat => "unsupported formats, primitives and tile modes",
            Kind::MemoryWindow => "memory windows failed",
            Kind::LoopGuard => "loop-guard trips",
            Kind::WaitTimeout => "wait timeouts",
            Kind::UnimplementedHle => "unimplemented and placeholder HLE calls",
            Kind::JitFallback => "JIT interpreter fallbacks",
            Kind::AudioLate => "audio buffers submitted late",
        }
    }

    fn per_frame(self) -> bool {
        matches!(
            self,
            Kind::DrawSkipped
                | Kind::DispatchSkipped
                | Kind::UnsupportedFormat
                | Kind::MemoryWindow
                | Kind::LoopGuard
                | Kind::WaitTimeout
                | Kind::UnimplementedHle
                | Kind::JitFallback
                | Kind::AudioLate
        )
    }
}

/// One aggregated problem entry, also used to merge counts kept outside the
/// global collector (such as per-stub HLE counters or per-thread JIT counts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub kind: Kind,
    pub reason: String,
    pub subject: String,
    pub count: u64,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    kind: Kind,
    reason: String,
    subject: String,
}

#[derive(Clone, Debug, Default)]
struct Entry {
    count: u64,
    detail: Option<String>,
}

/// Thread-safe problem collector for a run.
pub struct Problems {
    frames: AtomicU64,
    entries: Mutex<HashMap<Key, Entry>>,
}

impl Default for Problems {
    fn default() -> Self {
        Self::new()
    }
}

impl Problems {
    pub fn new() -> Self {
        Self {
            frames: AtomicU64::new(0),
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn note_frame(&self) {
        self.frames.fetch_add(1, Relaxed);
    }

    pub fn frames(&self) -> u64 {
        self.frames.load(Relaxed)
    }

    pub fn record(&self, kind: Kind, reason: impl Into<String>, subject: impl Into<String>) {
        self.add(kind, reason, subject, 1, None);
    }

    pub fn record_with_detail(
        &self,
        kind: Kind,
        reason: impl Into<String>,
        subject: impl Into<String>,
        detail: impl Into<String>,
    ) {
        self.add(kind, reason, subject, 1, Some(detail.into()));
    }

    pub fn add(
        &self,
        kind: Kind,
        reason: impl Into<String>,
        subject: impl Into<String>,
        count: u64,
        detail: Option<String>,
    ) {
        if count == 0 {
            return;
        }
        let key = Key {
            kind,
            reason: reason.into(),
            subject: subject.into(),
        };
        let mut map = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let entry = map.entry(key).or_default();
        entry.count = entry.count.saturating_add(count);
        if entry.detail.is_none()
            && let Some(d) = detail
        {
            entry.detail = Some(one_line(&d));
        }
    }

    /// Formats all recorded problems plus `extra`, grouped by category and
    /// reason and sorted by impact (highest count first). Returns an empty
    /// string when nothing was recorded.
    pub fn format_with(&self, extra: &[Row]) -> String {
        let mut merged: HashMap<Key, Entry> = self
            .entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for row in extra {
            if row.count == 0 {
                continue;
            }
            let key = Key {
                kind: row.kind,
                reason: row.reason.clone(),
                subject: row.subject.clone(),
            };
            let entry = merged.entry(key).or_default();
            entry.count = entry.count.saturating_add(row.count);
            if entry.detail.is_none()
                && let Some(d) = &row.detail
            {
                entry.detail = Some(one_line(d));
            }
        }
        if merged.is_empty() {
            return String::new();
        }
        struct Item {
            subject: String,
            count: u64,
            detail: Option<String>,
        }
        let frames = self.frames();
        let total_events: u64 = merged.values().map(|e| e.count).sum();
        let mut by_kind: HashMap<Kind, HashMap<String, Vec<Item>>> = HashMap::new();
        for (key, entry) in merged {
            by_kind
                .entry(key.kind)
                .or_default()
                .entry(key.reason)
                .or_default()
                .push(Item {
                    subject: key.subject,
                    count: entry.count,
                    detail: entry.detail,
                });
        }
        let mut kinds: Vec<_> = by_kind.into_iter().collect();
        kinds.sort_by_key(|(k, _)| *k);

        let mut out = String::new();
        let _ = writeln!(
            out,
            "problem report ({frames} frames, {total_events} events across {} categories):",
            kinds.len()
        );
        for (kind, reasons_map) in kinds {
            let mut reasons: Vec<(String, u64, Vec<Item>)> = reasons_map
                .into_iter()
                .map(|(reason, mut items)| {
                    items.sort_by(|a, b| {
                        b.count
                            .cmp(&a.count)
                            .then_with(|| a.subject.cmp(&b.subject))
                    });
                    let sum: u64 = items.iter().map(|i| i.count).sum();
                    (reason, sum, items)
                })
                .collect();
            reasons.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let kind_total: u64 = reasons.iter().map(|r| r.1).sum();
            let rate = format_rate(kind_total, frames, kind.per_frame());
            let _ = writeln!(out, "== {} ({kind_total}{rate}) ==", kind.heading());
            for (reason, sum, items) in reasons {
                let rate = format_rate(sum, frames, kind.per_frame());
                let _ = writeln!(out, "  {reason} ({sum}{rate}):");
                for item in items.iter().take(MAX_PER_REASON) {
                    let rate = format_rate(item.count, frames, kind.per_frame());
                    match &item.detail {
                        Some(d) if !d.is_empty() => {
                            let _ =
                                writeln!(out, "    {}: {}{rate} [{d}]", item.subject, item.count);
                        }
                        _ => {
                            let _ = writeln!(out, "    {}: {}{rate}", item.subject, item.count);
                        }
                    }
                }
                if items.len() > MAX_PER_REASON {
                    let _ = writeln!(out, "    ... and {} more", items.len() - MAX_PER_REASON);
                }
            }
        }
        out
    }

    pub fn format(&self) -> String {
        self.format_with(&[])
    }
}

fn format_rate(count: u64, frames: u64, per_frame: bool) -> String {
    if per_frame && frames > 0 {
        format!(", {:.1}/frame", count as f64 / frames as f64)
    } else {
        String::new()
    }
}

fn one_line(s: &str) -> String {
    let flat: String = s
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    if flat.chars().count() <= MAX_DETAIL_CHARS {
        flat
    } else {
        let mut cut: String = flat.chars().take(MAX_DETAIL_CHARS - 3).collect();
        cut.push_str("...");
        cut
    }
}

/// The process-wide problem collector.
pub fn global() -> &'static Problems {
    static GLOBAL: OnceLock<Problems> = OnceLock::new();
    GLOBAL.get_or_init(Problems::new)
}

pub fn note_frame() {
    global().note_frame();
}

pub fn record(kind: Kind, reason: impl Into<String>, subject: impl Into<String>) {
    global().record(kind, reason, subject);
}

pub fn record_with_detail(
    kind: Kind,
    reason: impl Into<String>,
    subject: impl Into<String>,
    detail: impl Into<String>,
) {
    global().record_with_detail(kind, reason, subject, detail);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_collector_formats_empty_report() {
        let p = Problems::new();
        assert!(p.format().is_empty());
    }

    #[test]
    fn report_groups_by_reason_sorts_by_impact_and_shows_per_frame_rates() {
        let p = Problems::new();
        for _ in 0..10 {
            p.note_frame();
        }
        p.add(
            Kind::DrawSkipped,
            "resources",
            "vs 0x100 ps 0x200",
            50,
            None,
        );
        p.add(
            Kind::DrawSkipped,
            "compiling",
            "vs 0x300 ps 0x400",
            30,
            None,
        );
        p.add(
            Kind::DrawSkipped,
            "compiling",
            "vs 0x300 ps 0x500",
            120,
            None,
        );
        p.record_with_detail(
            Kind::ShaderTranslate,
            "unsupported instruction ds_write_addtid_b32",
            "Pixel 0x115ae0dc00",
            "at dword 0x10: unsupported instruction ds_write_addtid_b32",
        );
        p.record_with_detail(
            Kind::PipelineFailed,
            "render pipeline",
            "vs 0x100 ps 0x200",
            "line 1\n  line 2",
        );
        p.record(
            Kind::UnsupportedFormat,
            "primitive type 0x15",
            "vs 0x100 ps 0x200",
        );
        p.record(Kind::MemoryWindow, "no guest memory window", "0x600");
        p.record_with_detail(
            Kind::LoopGuard,
            "compute loop guard",
            "cs 0x700",
            "64 threads, 1024/1048576 units",
        );
        p.record(Kind::WaitTimeout, "WAIT_REG_MEM", "gpu compute 0x800");

        let extra = vec![
            Row {
                kind: Kind::UnimplementedHle,
                reason: "placeholder".into(),
                subject: "libSceFoo::sceFooBar".into(),
                count: 40,
                detail: None,
            },
            Row {
                kind: Kind::JitFallback,
                reason: "interpreter fallback".into(),
                subject: "map 2 0a /0".into(),
                count: 500,
                detail: None,
            },
        ];

        let text = p.format_with(&extra);
        assert!(text.contains("problem report (10 frames,"));
        assert!(text.contains("== draws skipped (200, 20.0/frame) =="));
        // "compiling" (150) must appear before "resources" (50), and ps 0x500 (120) before ps 0x400 (30).
        let idx_compiling = text.find("compiling (150, 15.0/frame):").unwrap();
        let idx_resources = text.find("resources (50, 5.0/frame):").unwrap();
        assert!(idx_compiling < idx_resources);
        let idx_500 = text.find("vs 0x300 ps 0x500: 120, 12.0/frame").unwrap();
        let idx_400 = text.find("vs 0x300 ps 0x400: 30, 3.0/frame").unwrap();
        assert!(idx_500 < idx_400);
        // Every other category is present with its detail flattened to one line.
        assert!(text.contains("ds_write_addtid_b32"));
        assert!(text.contains("Pixel 0x115ae0dc00"));
        assert!(text.contains("[line 1 | line 2]"));
        assert!(text.contains("primitive type 0x15"));
        assert!(text.contains("no guest memory window"));
        assert!(text.contains("cs 0x700"));
        assert!(text.contains("WAIT_REG_MEM"));
        assert!(text.contains("libSceFoo::sceFooBar: 40, 4.0/frame"));
        assert!(text.contains("map 2 0a /0: 500, 50.0/frame"));
    }
}
