//! The control socket: a Unix socket that takes one command per line, for
//! poking at the running guest. Each answer ends with a line holding a single
//! `.`. `help` lists the commands; see `docs/debugging.md`.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hle::Kernel;
use memory::Access;

const HELP: &str = "\
regions                     mapped regions with names
memory                      memory use: direct memory, mappings by name, host heap, host process
mem ADDR [LEN]              hex dump (LEN up to 0x10000, default 0x100)
dump ADDR LEN PATH          write guest memory to a host file
poke ADDR HEXBYTES          write bytes, ignoring guest protection
find HEXBYTES [START END]   search readable memory (first 32 hits)
sym ADDR                    the region and offset of an address
threads                     guest threads and the host function each called last
watch ADDR LEN              log writes to a range (adds to earlier watches)
hits [N [ADDR LEN]]         writers of watched ranges (or of ADDR..+LEN) and the last N writes
unwatch                     end all watches and clear the log
trace PATH [SITE...]        append watched writes (from SITEs only) to PATH
trace off                   stop tracing
calls PATH [FILTER]         append every host call (whose name contains FILTER) to PATH
calls off                   stop logging calls
shot PATH                   write the screen as PPM
frames DIR N                write the next N flipped frames as DIR/FLIP.ppm (N decimal)
profile on|off              start a fresh profile window, or stop profiling
profile [N]                 report the window so far (top N rows, default 20)
gputrace N                  trace every draw of the next N frames on stderr (decimal)
gpusnap DIR [FIRST]         write every render pass attachment of the next frame (from pass FIRST, decimal) as PPM
gpureload                   translate every program again (and read PS5SLOP_MSL_OVERRIDE files again)
gpuskip [ADDR...]           skip draws and dispatches using these programs (none: skip nothing)
targets DIR                 write every render target and texture as PPM (the first call starts recording)
press BUTTONS [MS]          hold pad buttons (cross+up, ...) for MS ms (default 400, decimal);
                            also l2, r2 and stick pushes lup, ldown, lleft, lright, rup, ...
rates                       guest MIPS (JIT, interpreter) since the last `rates`, as the window title shows
quit                        end the guest
Numbers are hex; a 0x prefix is optional.";

/// Listens on `path` on a new thread.
pub fn serve(kernel: Arc<Kernel>, path: PathBuf) {
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ps5slop: cannot listen on {}: {e}", path.display());
            return;
        }
    };
    eprintln!("ps5slop: control socket at {}", path.display());
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(conn) = conn else { continue };
            let kernel = kernel.clone();
            std::thread::spawn(move || {
                let Ok(mut out) = conn.try_clone() else {
                    return;
                };
                for line in BufReader::new(conn).lines() {
                    let Ok(line) = line else { break };
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let answer = run(&kernel, line);
                    if writeln!(out, "{answer}\n.").is_err() {
                        break;
                    }
                }
            });
        }
    });
}

fn num(arg: Option<&str>) -> Option<u64> {
    let a = arg?;
    u64::from_str_radix(a.strip_prefix("0x").unwrap_or(a), 16).ok()
}

fn bytes(arg: Option<&str>) -> Option<Vec<u8>> {
    let h = arg?;
    if h.len() % 2 != 0 {
        return None;
    }
    (0..h.len() / 2)
        .map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).ok())
        .collect()
}

fn run(kernel: &Arc<Kernel>, line: &str) -> String {
    let mut it = line.split_whitespace();
    let cmd = it.next().unwrap_or("");
    let mem = kernel.memory();
    match cmd {
        "help" => HELP.into(),
        "regions" => mem
            .regions()
            .iter()
            .map(|r| {
                let p = |a, c| if r.prot.allows(a) { c } else { '-' };
                format!(
                    "{:#012x}-{:#012x} {}{}{}{} {}",
                    r.start,
                    r.end,
                    p(Access::Read, 'r'),
                    p(Access::Write, 'w'),
                    p(Access::Execute, 'x'),
                    if r.trap { " trap" } else { "" },
                    r.name
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        "memory" => hle::memory_report(kernel),
        "mem" => match (num(it.next()), it.next().map(|l| num(Some(l)))) {
            (Some(addr), None) => hexdump(&mem, addr, 0x100),
            (Some(addr), Some(Some(len))) => hexdump(&mem, addr, len.min(0x10000)),
            _ => "usage: mem ADDR [LEN]".into(),
        },
        "dump" => match (num(it.next()), num(it.next()), it.next()) {
            (Some(addr), Some(len), Some(path)) => {
                let mut buf = vec![0u8; len as usize];
                match mem.read(addr, &mut buf) {
                    Err(f) => format!("cannot read: {f:?}"),
                    Ok(()) => match std::fs::write(path, &buf) {
                        Ok(()) => format!("dumped {len:#x} bytes"),
                        Err(e) => format!("dump: {e}"),
                    },
                }
            }
            _ => "usage: dump ADDR LEN PATH".into(),
        },
        "poke" => match (num(it.next()), bytes(it.next())) {
            (Some(addr), Some(b)) => match mem.write_host(addr, &b) {
                Ok(()) => format!("wrote {} bytes", b.len()),
                Err(f) => format!("cannot write: {f:?}"),
            },
            _ => "usage: poke ADDR HEXBYTES".into(),
        },
        "find" => match bytes(it.next()) {
            Some(pattern) if !pattern.is_empty() => {
                let range = (num(it.next()), num(it.next()));
                find(
                    &mem,
                    &pattern,
                    range.0.unwrap_or(0),
                    range.1.unwrap_or(u64::MAX),
                )
            }
            _ => "usage: find HEXBYTES [START END]".into(),
        },
        "sym" => match num(it.next()) {
            Some(addr) => sym(kernel, addr),
            None => "usage: sym ADDR".into(),
        },
        "threads" => kernel
            .threads()
            .iter()
            .map(|t| {
                format!(
                    "{:#x} {:<24} {} calls {:>9}  last {} from {}",
                    t.tcb,
                    t.name.as_deref().unwrap_or("-"),
                    if t.running { "running" } else { "ended  " },
                    t.calls,
                    kernel
                        .hle
                        .describe(t.stub)
                        .unwrap_or_else(|| format!("{:#x}", t.stub)),
                    sym(kernel, t.caller)
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        "watch" => match (num(it.next()), num(it.next())) {
            (Some(addr), Some(len)) if len > 0 => match jit::watch(addr, addr + len) {
                Ok(()) => format!("watching {addr:#x}..{:#x}", addr + len),
                Err(e) => {
                    memory::watch::add(addr, addr + len);
                    format!(
                        "watching {addr:#x}..{:#x} (interpreter and host writes only: {e:?})",
                        addr + len
                    )
                }
            },
            _ => "usage: watch ADDR LEN".into(),
        },
        "unwatch" => {
            if jit::unwatch().is_err() {
                memory::watch::clear();
            }
            "ok".into()
        }
        "trace" => match it.next() {
            Some("off") => {
                memory::watch::trace(None, Vec::new());
                "ok".into()
            }
            Some(path) => match std::fs::File::create(path) {
                Ok(f) => {
                    memory::watch::trace(Some(f), it.filter_map(|a| num(Some(a))).collect());
                    "ok".into()
                }
                Err(e) => format!("trace: {e}"),
            },
            None => "usage: trace PATH [SITE...] | trace off".into(),
        },
        "calls" => match it.next() {
            Some("off") => {
                hle::trace_calls(None, String::new());
                "ok".into()
            }
            Some(path) => match std::fs::File::create(path) {
                Ok(f) => {
                    hle::trace_calls(Some(f), it.collect::<Vec<_>>().join(" "));
                    "ok".into()
                }
                Err(e) => format!("calls: {e}"),
            },
            None => "usage: calls PATH [FILTER] | calls off".into(),
        },
        "hits" => {
            let last = num(it.next()).unwrap_or(0x10) as usize;
            let range = match (num(it.next()), num(it.next())) {
                (Some(a), Some(l)) => (a, a + l),
                _ => (0, u64::MAX),
            };
            hits(kernel, last, range)
        }
        "shot" => match it.next() {
            Some(path) => match crate::write_frame(kernel, Path::new(path)) {
                Ok(()) => "ok".into(),
                Err(e) => e,
            },
            None => "usage: shot PATH".into(),
        },
        "frames" => match (it.next(), it.next().and_then(|n| n.parse().ok())) {
            (Some(dir), Some(n)) => {
                crate::write_frames(kernel.clone(), dir.into(), n);
                format!("writing the next {n} frames to {dir}")
            }
            _ => "usage: frames DIR N".into(),
        },
        #[cfg(target_os = "macos")]
        "gputrace" => match it.next().and_then(|n| n.parse().ok()) {
            Some(n) => {
                hle::gpu::metal::TRACE_NEXT.store(n, std::sync::atomic::Ordering::Relaxed);
                "ok".into()
            }
            None => "usage: gputrace N".into(),
        },
        #[cfg(target_os = "macos")]
        "gpusnap" => match it.next() {
            Some(dir) => {
                let from = it.next().and_then(|n| n.parse().ok()).unwrap_or(0);
                *hle::gpu::metal::SNAP
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some((dir.into(), from));
                "snapping the next frame's render passes".into()
            }
            None => "usage: gpusnap DIR [FIRST_PASS]".into(),
        },
        #[cfg(target_os = "macos")]
        "gpureload" => {
            hle::gpu::metal::RELOAD.store(true, std::sync::atomic::Ordering::Relaxed);
            "translating programs again from the next frame".into()
        }
        #[cfg(target_os = "macos")]
        "gpuskip" => {
            let addrs: Vec<u64> = it
                .filter_map(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok())
                .collect();
            let mut skip = hle::gpu::metal::SKIP
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *skip = addrs;
            format!("skipping {:x?}", *skip)
        }
        #[cfg(target_os = "macos")]
        "targets" => match it.next() {
            Some(_)
                if !hle::gpu::metal::DUMP_TARGETS
                    .swap(true, std::sync::atomic::Ordering::Relaxed)
                    && std::env::var_os("PS5SLOP_DUMP_TARGETS").is_none() =>
            {
                "recording targets from now on; run again after a frame".into()
            }
            Some(dir) => {
                let mut n = 0;
                for (addr, t) in hle::gpu::metal::all_frames() {
                    if let Some((w, h, px)) = hle::gpu::metal::read_rgba(&t) {
                        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                        ppm.extend(
                            px.as_chunks::<4>()
                                .0
                                .iter()
                                .flat_map(|p| [p[0], p[1], p[2]]),
                        );
                        if std::fs::write(Path::new(dir).join(format!("{addr:x}.ppm")), ppm).is_ok()
                        {
                            n += 1;
                        }
                    }
                }
                format!("{n} written")
            }
            None => "usage: targets DIR".into(),
        },
        "press" => match it.next().map(crate::pad_input) {
            Some(input) if input != hle::PadInput::default() => {
                let ms = it.next().and_then(|m| m.parse().ok()).unwrap_or(400);
                kernel.set_pad_input(input);
                std::thread::sleep(std::time::Duration::from_millis(ms));
                kernel.set_pad_input(hle::PadInput::default());
                "ok".into()
            }
            _ => "usage: press BUTTONS [MS]".into(),
        },
        "profile" => match it.next() {
            Some("on") => {
                PROFILE_GENERATION.store(
                    kernel.memory().generation(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                jit::profile::set(true);
                hle::gpu::timing::set(true);
                "profiling".into()
            }
            Some("off") => {
                jit::profile::set(false);
                hle::gpu::timing::set(false);
                "profiling off".into()
            }
            n => match jit::profile::report() {
                Some(r) => profile_report(kernel, &r, n.and_then(|n| n.parse().ok()).unwrap_or(20)),
                None => "profiling is off (profile on)".into(),
            },
        },
        "rates" => {
            let now = (std::time::Instant::now(), kernel.instructions());
            let mut last = RATES.lock().unwrap_or_else(|e| e.into_inner());
            let out = match *last {
                Some((at, (interp, jit))) => {
                    let secs = now.0.duration_since(at).as_secs_f64();
                    let mips = |a: u64, b: u64| (b - a) as f64 / secs / 1e6;
                    format!(
                        "{secs:.1} s: JIT {:.1} MIPS, interpreter {:.2} MIPS",
                        mips(jit, now.1.1),
                        mips(interp, now.1.0)
                    )
                }
                None => "started; ask again for the rates since now".into(),
            };
            *last = Some(now);
            out
        }
        "quit" => {
            kernel.exit(Ok(0));
            "ok".into()
        }
        _ => format!("unknown command {cmd}; try help"),
    }
}

/// Pad buttons plus the triggers and sticks pushed to their limits.
fn hexdump(mem: &memory::Memory, addr: u64, len: u64) -> String {
    let mut lines = Vec::new();
    for row in (0..len).step_by(16) {
        let at = addr + row;
        let n = (len - row).min(16) as usize;
        let mut b = vec![0u8; n];
        if mem.read(at, &mut b).is_err() {
            lines.push(format!("{at:#012x}: unreadable"));
            continue;
        }
        let hex: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
        let text: String = b
            .iter()
            .map(|&c| {
                if c.is_ascii_graphic() || c == b' ' {
                    c as char
                } else {
                    '.'
                }
            })
            .collect();
        lines.push(format!("{at:#012x}: {:<47}  {text}", hex.join(" ")));
    }
    lines.join("\n")
}

fn find(mem: &memory::Memory, pattern: &[u8], lo: u64, hi: u64) -> String {
    const CHUNK: u64 = 1 << 20;
    let mut found = Vec::new();
    let overlap = pattern.len() as u64 - 1;
    'regions: for r in mem.regions() {
        if r.trap || !r.prot.allows(Access::Read) {
            continue;
        }
        let (start, end) = (r.start.max(lo), r.end.min(hi));
        let mut at = start;
        while at < end {
            let n = (end - at).min(CHUNK + overlap);
            let mut buf = vec![0u8; n as usize];
            if mem.read(at, &mut buf).is_err() {
                break;
            }
            for (i, w) in buf.windows(pattern.len()).enumerate() {
                if w == pattern && (i as u64) < CHUNK {
                    found.push(format!("{:#x} {}", at + i as u64, r.name));
                    if found.len() == 32 {
                        break 'regions;
                    }
                }
            }
            at += CHUNK;
        }
    }
    if found.is_empty() {
        "not found".into()
    } else {
        found.join("\n")
    }
}

fn sym(kernel: &Kernel, addr: u64) -> String {
    if let Some(stub) = kernel.hle.describe(addr) {
        return stub;
    }
    let regions = kernel.memory().regions();
    match regions.iter().find(|r| r.start <= addr && addr < r.end) {
        Some(r) => format!("{addr:#x} ({}+{:#x})", r.name, addr - r.start),
        None => format!("{addr:#x} (unmapped)"),
    }
}

fn hits(kernel: &Kernel, last: usize, (lo, hi): (u64, u64)) -> String {
    let (sites, recent, total) = memory::watch::log(lo, hi);
    let mut out = vec![format!(
        "{total} writes to {}",
        memory::watch::ranges()
            .iter()
            .map(|(s, e)| format!("{s:#x}..{e:#x}"))
            .collect::<Vec<_>>()
            .join(", ")
    )];
    let site_name = |site: u64| {
        if site & memory::watch::HOST_CALL != 0 {
            format!(
                "host call from {}",
                sym(kernel, site & !memory::watch::HOST_CALL)
            )
        } else if site == 0 {
            "unknown".into()
        } else {
            sym(kernel, site)
        }
    };
    for (site, s) in &sites {
        out.push(format!(
            "{:>9} writes {:>10} bytes {:#x}..{:#x}  {}",
            s.count,
            s.bytes,
            s.lowest,
            s.highest,
            site_name(*site)
        ));
    }
    if last > 0 {
        out.push("last writes:".into());
        for h in recent.iter().rev().take(last).rev() {
            let data: Vec<String> = h.data.iter().map(|b| format!("{b:02x}")).collect();
            out.push(format!(
                "{:#x} len {:#x} [{}] {} ({})",
                h.addr,
                h.len,
                data.join(""),
                site_name(h.site),
                h.thread
            ));
        }
    }
    out.join("\n")
}

static RATES: std::sync::Mutex<Option<(std::time::Instant, (u64, u64))>> =
    std::sync::Mutex::new(None);

static PROFILE_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn profile_report(kernel: &Kernel, r: &jit::profile::Report, top: usize) -> String {
    let ms = |ns: u64| ns as f64 / 1e6;
    let names: std::collections::HashMap<String, String> = kernel
        .threads()
        .into_iter()
        .filter_map(|t| Some((format!("guest {:#x}", t.tcb), t.name?)))
        .collect();
    let changes = kernel.memory().generation()
        - PROFILE_GENERATION.load(std::sync::atomic::Ordering::Relaxed);
    let mut out = vec![format!(
        "{:.1} s; {} translations ({} of code translated before, {} bytes each on average), {} segments evicted, {} blocks in the cache, {changes} memory map changes",
        r.seconds,
        r.translations,
        r.retranslations,
        r.emitted / r.translations.max(1),
        r.evictions,
        r.live_blocks
    )];
    out.push("threads (ms: translated code, interpreter, translating, host functions, waiting to translate, none of these, CPU used; guest MIPS in translated code):".into());
    for (thread, t) in r.threads.iter().take(top) {
        let name = names.get(thread).map_or(thread.as_str(), String::as_str);
        let other = t[6].saturating_sub(t[..5].iter().sum());
        out.push(format!(
            "  {:<28} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>7.0}",
            name,
            ms(t[0]),
            ms(t[1]),
            ms(t[2]),
            ms(t[3]),
            ms(t[4]),
            ms(other),
            ms(t[5]),
            t[7] as f64 * 1e3 / t[0].max(1) as f64
        ));
    }
    out.push("JIT lock by site (times, ms waited, ms held):".into());
    for (site, n, w, h) in r.lock_sites.iter().take(top) {
        out.push(format!("  {n:>9} {:>9.0} {:>9.0}  {site}", ms(*w), ms(*h)));
    }
    let gpu: Vec<String> = hle::gpu::timing::NAMES
        .iter()
        .zip(hle::gpu::timing::totals())
        .map(|(n, ns)| format!("{n} {:.0}", ms(ns)))
        .collect();
    out.push(format!("GPU thread (ms): {}", gpu.join(", ")));
    out.push(format!(
        "hottest guest code ({} samples, {} in runtime code):",
        r.total_samples, r.runtime_samples
    ));
    let parts: Vec<String> = r.runtime.iter().map(|(k, n)| format!("{k} {n}")).collect();
    out.push(format!("  runtime samples: {}", parts.join(", ")));
    for (rip, n) in r.samples.iter().take(top) {
        out.push(format!(
            "  {:>6} {:>5.1}%  {}",
            n,
            *n as f64 * 100.0 / r.total_samples.max(1) as f64,
            sym(kernel, *rip)
        ));
    }
    out.push("host functions (ms, calls):".into());
    for (addr, calls, ns) in r.host.iter().take(top) {
        let name = kernel
            .hle
            .describe(*addr)
            .unwrap_or_else(|| format!("{addr:#x}"));
        out.push(format!(
            "  {:>9.1} {:>9}  {}",
            ms(*ns),
            calls,
            name.trim_start_matches("the stub of ")
        ));
    }
    out.push("host functions by thread (ms, calls):".into());
    for (thread, addr, calls, ns) in r.host_by_thread.iter().take(top) {
        let name = kernel
            .hle
            .describe(*addr)
            .unwrap_or_else(|| format!("{addr:#x}"));
        let t = names.get(thread).map_or(thread.as_str(), String::as_str);
        out.push(format!(
            "  {:>9.1} {:>9}  {:<24} {}",
            ms(*ns),
            calls,
            t,
            name.trim_start_matches("the stub of ")
        ));
    }
    let (ops, rips, exits) = jit::interpreted(top);
    out.push(format!(
        "exits: misses {} ({} at translated blocks, {} at untranslatable code), untranslated {}, host calls {}, faults {}, cancels {}, slow entries {}",
        exits[0], exits[5], exits[6], exits[1], exits[2], exits[3], exits[4], exits[8]
    ));
    let causes = jit::exit_causes(top);
    let total: u64 = causes.iter().map(|c| c.1).sum();
    out.push(format!("VM exits by cause ({total}):"));
    for (cause, n, distinct, rips) in &causes {
        out.push(format!(
            "  {:>10} {:>5.1}%  {} ({distinct} addresses)",
            n,
            *n as f64 * 100.0 / total.max(1) as f64,
            cause.name()
        ));
        if *cause == jit::Cause::Canceled {
            continue;
        }
        for (rip, k) in rips {
            let at = match cause {
                jit::Cause::HostCall => kernel
                    .hle
                    .describe(*rip)
                    .map(|d| d.trim_start_matches("the stub of ").to_string()),
                _ => None,
            }
            .unwrap_or_else(|| sym(kernel, *rip));
            out.push(format!("      {k:>10}  {at}"));
        }
    }
    out.push("interpreted opcodes (map, opcode, /ext):".into());
    for ((map, op, ext), n) in ops {
        let ext = if ext == 0xff {
            String::new()
        } else {
            format!(" /{ext}")
        };
        out.push(format!("  {n:>10}  map {map} {op:02x}{ext}"));
    }
    out.push("interpreted addresses:".into());
    for (rip, n) in rips {
        out.push(format!("  {n:>10}  {}", sym(kernel, rip)));
    }
    out.join("\n")
}
