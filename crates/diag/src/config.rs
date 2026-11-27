//! The `PS5SLOP_*` debug switches, read from the environment once. Each is
//! described in docs/environment.md.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

/// This process's switches.
pub fn config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| Config::from_vars(|name| std::env::var_os(name)))
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
    pub run: Run,
    pub hle: Hle,
    pub jit: Jit,
    pub gpu: Gpu,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// Scripted pad input: `SECONDS:BUTTONS` steps separated by commas.
    pub input: Option<String>,
    /// How often code and read-only data are compared with their contents at start.
    pub check_code: Option<Duration>,
    /// The file the `calls` log goes to from the start, and its filter.
    pub calls: Option<(PathBuf, String)>,
    /// A host folder mounted writable as `/devlog`.
    pub devlog: Option<PathBuf>,
    /// Frames `--dump-frame` writes, and the time between them.
    pub dump_series: u32,
    pub dump_every: Duration,
    /// A faulting thread waits instead of ending the process.
    pub hold_on_fault: bool,
    /// Where to write the per-run problem report on exit (in addition to stderr).
    pub report: Option<PathBuf>,
}

impl Default for Run {
    fn default() -> Self {
        Self {
            input: None,
            check_code: None,
            calls: None,
            devlog: None,
            dump_series: 1,
            dump_every: Duration::from_millis(500),
            hold_on_fault: false,
            report: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hle {
    /// A file of names, one per line, hashed to label NIDs.
    pub names: Option<PathBuf>,
    pub unimplemented: Unimplemented,
    /// `Some(true)` opens the host's audio device even headless, `Some(false)` never.
    pub audio: Option<bool>,
    /// A WAV file that receives everything the guest plays.
    pub audio_wav: Option<PathBuf>,
}

/// What a call of an import with no host function does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Unimplemented {
    #[default]
    Stop,
    ReturnZero,
    /// Returns 0 and logs the bytes behind each pointer-like argument.
    ReturnZeroDump,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Jit {
    /// Runs every block in the interpreter too and compares.
    pub verify: bool,
    /// Address ranges `[start, end)` verify mode checks; empty checks all.
    pub verify_ranges: Vec<(u64, u64)>,
    /// Runs of each block verify mode checks before it runs unchecked.
    pub verify_runs: u32,
    /// Ends blocks at every instruction boundary the guest can be stopped at,
    /// and records each block start for fault reports.
    pub trace: bool,
    pub lazy_sync: bool,
    /// Keeps `memset`, `memcpy` and `memmove` as host calls.
    pub no_natives: bool,
    /// Code cache segments in use, fewer than all to test eviction.
    pub segments: Option<usize>,
    /// Runs before a block is compiled again as a tier-2 region; `None` keeps tier 2 off.
    pub tier2: Option<u32>,
    /// Guest addresses that always run in the interpreter, logging each arrival.
    pub probes: Vec<u64>,
    /// Address ranges `[start, end)` whose blocks are never translated.
    pub interpret: Vec<(u64, u64)>,
    /// Rows in each table of the statistics report.
    pub stats_top: usize,
}

impl Default for Jit {
    fn default() -> Self {
        Self {
            verify: false,
            verify_ranges: Vec::new(),
            verify_runs: 4,
            trace: false,
            lazy_sync: false,
            no_natives: false,
            segments: None,
            tier2: None,
            probes: Vec::new(),
            interpret: Vec::new(),
            stats_top: 20,
        }
    }
}

/// `PS5SLOP_GPU_WATCH`: a range of guest memory checked after each draw and dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Watch {
    pub addr: u64,
    pub len: u64,
    pub delay: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpu {
    /// Command buffers are rendered through Metal (off with `PS5SLOP_GPU_RENDER=0`).
    pub render: bool,
    /// Frames traced from the start, first and last.
    pub trace_frames: Option<(u64, u64)>,
    pub trace_regs: bool,
    pub trace_pixels: bool,
    pub trace_user_memory: bool,
    /// A compute program whose dispatches are logged.
    pub probe: Option<u64>,
    /// Memory watched for changes by draws and dispatches, from this long after the first.
    pub watch: Option<Watch>,
    /// Where each new content of a watched range is written.
    pub watch_dump: Option<PathBuf>,
    pub dump_targets: bool,
    pub snap_inputs: bool,
    pub snap_raw: bool,
    pub shader_dump: Option<PathBuf>,
    pub msl_override: Option<PathBuf>,
    pub msl_unique: bool,
    pub labels: bool,
    pub encoder_status: bool,
    pub fence: bool,
    pub residency_set: bool,
    pub inline_uploads: bool,
    /// Commits the command buffer every this many dispatches (0: never early).
    pub commit_every: u64,
    /// Commits the command buffer when a render pass ends after this many
    /// draws, so the GPU starts while encoding goes on (0: only at sync points).
    pub commit_draws: u64,
    pub compile_wait: bool,
    pub compile_threads: Option<usize>,
    pub htile_cpu: bool,
    /// Waits for and reads colour fast-clear words (CMASK, DCC) on the CPU.
    pub fast_clear_cpu: bool,
    pub queued_writes: bool,
    pub sync_releases: bool,
    pub sync_dispatches: bool,
    pub wait_submissions: bool,
    pub cpu_indirect: bool,
    pub no_throttle: bool,
    pub probe_flat: bool,
    /// Parts of the float semantics that follow the hardware exactly; the
    /// rest is Metal's own arithmetic.
    pub exact_float: ExactFloat,
    /// Prototype: plain 2D bilinear samples filter in the program with the
    /// hardware's 8-bit weights.
    pub filter_bilinear: bool,
    /// Where hotkey captures (`.ps5frame` and `.gputrace`) are written.
    pub capture_dir: Option<PathBuf>,
    /// Folder holding local `.ps5frame` + golden pairs for `frame_replay` tests.
    pub frame_captures: Option<PathBuf>,
}

/// `PS5SLOP_FLOAT_EXACT`: parts of the float VALU semantics translated to
/// follow the hardware exactly instead of as plain Metal arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExactFloat {
    pub nan: bool,
    pub denorm: bool,
    pub round: bool,
    pub mods: bool,
}

impl ExactFloat {
    /// A comma-separated list of `nan`, `denorm`, `round`, `mods`, or `all`.
    fn parse(list: &str) -> Self {
        let mut p = Self::default();
        for part in list.split(',').map(str::trim) {
            let all = part == "all";
            p.nan |= all || part == "nan";
            p.denorm |= all || part == "denorm";
            p.round |= all || part == "round";
            p.mods |= all || part == "mods";
        }
        p
    }
}

impl Default for Gpu {
    fn default() -> Self {
        Config::from_vars(|_| None).gpu
    }
}

impl Config {
    /// Reads the switches through `var`, which maps a variable's name to its value.
    pub fn from_vars(var: impl Fn(&str) -> Option<OsString>) -> Self {
        let v = Vars(var);
        let (text, flag, path) = (|n| v.text(n), |n| v.flag(n), |n| v.path(n));
        Config {
            run: Run {
                input: text("PS5SLOP_INPUT"),
                check_code: text("PS5SLOP_CHECK_CODE")
                    .map(|v| Duration::from_millis(v.parse().unwrap_or(100))),
                calls: text("PS5SLOP_CALLS").map(|spec| {
                    let (path, filter) = spec.split_once(':').unwrap_or((&spec, ""));
                    (PathBuf::from(path), filter.to_owned())
                }),
                devlog: path("PS5SLOP_DEVLOG"),
                dump_series: v.number("PS5SLOP_DUMP_SERIES").unwrap_or(1),
                dump_every: Duration::from_millis(v.number("PS5SLOP_DUMP_EVERY_MS").unwrap_or(500)),
                hold_on_fault: flag("PS5SLOP_HOLD_ON_FAULT"),
                report: path("PS5SLOP_REPORT"),
            },
            hle: Hle {
                names: path("PS5SLOP_NAMES"),
                unimplemented: match text("PS5SLOP_UNIMPLEMENTED_ZERO").as_deref() {
                    None => Unimplemented::Stop,
                    Some("dump") => Unimplemented::ReturnZeroDump,
                    Some(_) => Unimplemented::ReturnZero,
                },
                audio: match text("PS5SLOP_AUDIO").as_deref() {
                    Some("1") => Some(true),
                    Some("0") => Some(false),
                    _ => None,
                },
                audio_wav: path("PS5SLOP_AUDIO_WAV"),
            },
            jit: Jit {
                verify: flag("PS5SLOP_JIT_VERIFY"),
                verify_ranges: ranges(text("PS5SLOP_JIT_VERIFY")),
                verify_runs: v.number("PS5SLOP_JIT_VERIFY_RUNS").unwrap_or(4),
                trace: flag("PS5SLOP_JIT_TRACE"),
                lazy_sync: flag("PS5SLOP_JIT_LAZY_SYNC"),
                no_natives: flag("PS5SLOP_JIT_NO_NATIVES"),
                segments: v.number("PS5SLOP_JIT_SEGMENTS"),
                tier2: match text("PS5SLOP_JIT_TIER2").as_deref() {
                    None | Some("off") => None,
                    Some("eager") => Some(1),
                    Some(n) => Some(n.parse().unwrap_or(1000)),
                },
                probes: text("PS5SLOP_PROBE_RIP")
                    .unwrap_or_default()
                    .split(',')
                    .filter_map(hex)
                    .collect(),
                interpret: ranges(text("PS5SLOP_JIT_INTERPRET")),
                stats_top: v.number("PS5SLOP_JIT_STATS_TOP").unwrap_or(20),
            },
            gpu: Gpu {
                render: text("PS5SLOP_GPU_RENDER").is_none_or(|v| v != "0"),
                trace_frames: text("PS5SLOP_GPU_TRACE").and_then(|v| match v.split_once('-') {
                    Some((a, b)) => Some((a.parse().ok()?, b.parse().ok()?)),
                    None => v.parse().ok().map(|n| (n, n)),
                }),
                trace_regs: flag("PS5SLOP_TRACE_REGS"),
                trace_pixels: flag("PS5SLOP_TRACE_PIXELS"),
                trace_user_memory: flag("PS5SLOP_TRACE_USER_MEMORY"),
                probe: text("PS5SLOP_GPU_PROBE").as_deref().and_then(hex),
                watch: text("PS5SLOP_GPU_WATCH").and_then(|v| {
                    let (range, secs) = v.split_once('@').unwrap_or((&v, "0"));
                    let (addr, len) = range.split_once('+').unwrap_or((range, "8"));
                    Some(Watch {
                        addr: hex(addr)?,
                        len: match len.strip_prefix("0x") {
                            Some(h) => u64::from_str_radix(h, 16).ok()?,
                            None => len.parse().ok()?,
                        }
                        .max(1),
                        delay: Duration::from_secs(secs.parse().ok()?),
                    })
                }),
                watch_dump: path("PS5SLOP_GPU_WATCH_DUMP"),
                dump_targets: flag("PS5SLOP_DUMP_TARGETS"),
                snap_inputs: flag("PS5SLOP_SNAP_INPUTS"),
                snap_raw: flag("PS5SLOP_SNAP_RAW"),
                shader_dump: path("PS5SLOP_SHADER_DUMP"),
                msl_override: path("PS5SLOP_MSL_OVERRIDE"),
                msl_unique: flag("PS5SLOP_MSL_UNIQUE"),
                labels: flag("PS5SLOP_GPU_LABELS"),
                encoder_status: flag("PS5SLOP_GPU_ENCODER_STATUS"),
                fence: flag("PS5SLOP_GPU_FENCE"),
                residency_set: flag("PS5SLOP_GPU_RESIDENCY_SET"),
                inline_uploads: flag("PS5SLOP_GPU_INLINE_UPLOADS"),
                commit_every: v.number("PS5SLOP_GPU_COMMIT_EVERY").unwrap_or(0),
                commit_draws: v.number("PS5SLOP_GPU_COMMIT_DRAWS").unwrap_or(0),
                compile_wait: flag("PS5SLOP_COMPILE_WAIT"),
                compile_threads: v.number("PS5SLOP_COMPILE_THREADS"),
                htile_cpu: flag("PS5SLOP_GPU_HTILE_CPU"),
                fast_clear_cpu: flag("PS5SLOP_GPU_FAST_CLEAR_CPU"),
                queued_writes: flag("PS5SLOP_GPU_QUEUED_WRITES"),
                sync_releases: flag("PS5SLOP_GPU_SYNC_RELEASES"),
                sync_dispatches: flag("PS5SLOP_GPU_SYNC_DISPATCHES"),
                wait_submissions: flag("PS5SLOP_GPU_WAIT_SUBMISSIONS"),
                cpu_indirect: flag("PS5SLOP_GPU_CPU_INDIRECT"),
                no_throttle: flag("PS5SLOP_GPU_NO_THROTTLE"),
                probe_flat: flag("PS5SLOP_PROBE_FLAT"),
                exact_float: ExactFloat::parse(&text("PS5SLOP_FLOAT_EXACT").unwrap_or_default()),
                filter_bilinear: flag("PS5SLOP_FILTER_8BIT"),
                capture_dir: path("PS5SLOP_CAPTURE_DIR"),
                frame_captures: path("PS5SLOP_FRAME_CAPTURES"),
            },
        }
    }
}

struct Vars<F>(F);

impl<F: Fn(&str) -> Option<OsString>> Vars<F> {
    fn text(&self, name: &str) -> Option<String> {
        (self.0)(name).map(|v| v.to_string_lossy().into_owned())
    }

    fn flag(&self, name: &str) -> bool {
        (self.0)(name).is_some()
    }

    fn path(&self, name: &str) -> Option<PathBuf> {
        (self.0)(name).map(PathBuf::from)
    }

    fn number<T: std::str::FromStr>(&self, name: &str) -> Option<T> {
        self.text(name)?.parse().ok()
    }
}

/// A hex number, with or without `0x`.
/// `START-END[,START-END...]` in hex; anything else is skipped.
fn ranges(v: Option<String>) -> Vec<(u64, u64)> {
    v.unwrap_or_default()
        .split(',')
        .filter_map(|r| {
            let (a, b) = r.split_once('-')?;
            Some((hex(a)?, hex(b)?))
        })
        .collect()
}

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(vars: &[(&str, &str)]) -> Config {
        Config::from_vars(|name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| OsString::from(v))
        })
    }

    #[test]
    fn nothing_set_gives_the_defaults() {
        let c = with(&[]);
        assert_eq!(c, Config::default());
        assert!(c.gpu.render);
        assert_eq!(c.run.dump_series, 1);
        assert_eq!(c.jit.stats_top, 20);
    }

    #[test]
    fn a_flag_is_on_whatever_its_value() {
        assert!(with(&[("PS5SLOP_JIT_VERIFY", "0")]).jit.verify);
        let ranged = with(&[("PS5SLOP_JIT_VERIFY", "400000-500000,0x10-20")]).jit;
        assert!(ranged.verify);
        assert_eq!(ranged.verify_ranges, [(0x40_0000, 0x50_0000), (0x10, 0x20)]);
    }

    #[test]
    fn rendering_is_off_only_for_zero() {
        assert!(!with(&[("PS5SLOP_GPU_RENDER", "0")]).gpu.render);
        assert!(with(&[("PS5SLOP_GPU_RENDER", "1")]).gpu.render);
    }

    #[test]
    fn traced_frames_are_one_number_or_a_range() {
        assert_eq!(
            with(&[("PS5SLOP_GPU_TRACE", "7")]).gpu.trace_frames,
            Some((7, 7))
        );
        assert_eq!(
            with(&[("PS5SLOP_GPU_TRACE", "3-9")]).gpu.trace_frames,
            Some((3, 9))
        );
        assert_eq!(with(&[("PS5SLOP_GPU_TRACE", "x")]).gpu.trace_frames, None);
    }

    #[test]
    fn a_watch_takes_a_hex_address_and_optional_length_and_seconds() {
        let watch = |addr, len, secs| {
            Some(Watch {
                addr,
                len,
                delay: Duration::from_secs(secs),
            })
        };
        assert_eq!(
            with(&[("PS5SLOP_GPU_WATCH", "0x1000@30")]).gpu.watch,
            watch(0x1000, 8, 30)
        );
        assert_eq!(
            with(&[("PS5SLOP_GPU_WATCH", "1000")]).gpu.watch,
            watch(0x1000, 8, 0)
        );
        assert_eq!(
            with(&[("PS5SLOP_GPU_WATCH", "1000+0x20000@5")]).gpu.watch,
            watch(0x1000, 0x20000, 5)
        );
        assert_eq!(
            with(&[("PS5SLOP_GPU_WATCH", "1000+100")]).gpu.watch,
            watch(0x1000, 100, 0)
        );
        assert_eq!(with(&[("PS5SLOP_GPU_WATCH", "1000@soon")]).gpu.watch, None);
    }

    #[test]
    fn exact_float_takes_a_list_of_parts_or_all() {
        let p = |v| with(&[("PS5SLOP_FLOAT_EXACT", v)]).gpu.exact_float;
        assert_eq!(
            p("nan, round"),
            ExactFloat {
                nan: true,
                round: true,
                ..ExactFloat::default()
            }
        );
        assert_eq!(
            p("all"),
            ExactFloat {
                nan: true,
                denorm: true,
                round: true,
                mods: true
            }
        );
        assert_eq!(with(&[]).gpu.exact_float, ExactFloat::default());
    }

    #[test]
    fn probes_skip_entries_that_are_not_hex() {
        assert_eq!(
            with(&[("PS5SLOP_PROBE_RIP", "0x10, 20,zz")]).jit.probes,
            vec![0x10, 0x20]
        );
    }

    #[test]
    fn tier2_is_off_eager_or_a_count() {
        assert_eq!(with(&[("PS5SLOP_JIT_TIER2", "off")]).jit.tier2, None);
        assert_eq!(with(&[("PS5SLOP_JIT_TIER2", "eager")]).jit.tier2, Some(1));
        assert_eq!(with(&[("PS5SLOP_JIT_TIER2", "50")]).jit.tier2, Some(50));
        assert_eq!(with(&[("PS5SLOP_JIT_TIER2", "lots")]).jit.tier2, Some(1000));
    }

    #[test]
    fn unimplemented_imports_return_zero_with_or_without_a_dump() {
        assert_eq!(
            with(&[("PS5SLOP_UNIMPLEMENTED_ZERO", "1")])
                .hle
                .unimplemented,
            Unimplemented::ReturnZero
        );
        assert_eq!(
            with(&[("PS5SLOP_UNIMPLEMENTED_ZERO", "dump")])
                .hle
                .unimplemented,
            Unimplemented::ReturnZeroDump
        );
    }

    #[test]
    fn the_calls_log_takes_an_optional_filter() {
        assert_eq!(
            with(&[("PS5SLOP_CALLS", "/tmp/c.log:sceKernel")]).run.calls,
            Some((PathBuf::from("/tmp/c.log"), "sceKernel".to_owned()))
        );
        assert_eq!(
            with(&[("PS5SLOP_CALLS", "/tmp/c.log")]).run.calls,
            Some((PathBuf::from("/tmp/c.log"), String::new()))
        );
    }

    #[test]
    fn code_checks_default_to_every_100_ms() {
        assert_eq!(
            with(&[("PS5SLOP_CHECK_CODE", "")]).run.check_code,
            Some(Duration::from_millis(100))
        );
    }

    #[test]
    fn capture_dirs_read_from_environment() {
        let c = with(&[
            ("PS5SLOP_CAPTURE_DIR", "/tmp/caps"),
            ("PS5SLOP_FRAME_CAPTURES", "/tmp/goldens"),
        ]);
        assert_eq!(c.gpu.capture_dir, Some(PathBuf::from("/tmp/caps")));
        assert_eq!(c.gpu.frame_captures, Some(PathBuf::from("/tmp/goldens")));
    }
}
