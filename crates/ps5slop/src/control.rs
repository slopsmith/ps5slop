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
gputrace N                  trace every draw of the next N frames on stderr (decimal)
targets DIR                 write every render target and texture as PPM (needs PS5SLOP_DUMP_TARGETS)
press BUTTONS [MS]          hold pad buttons (cross+up, ...) for MS ms (default 400, decimal)
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
                let Ok(mut out) = conn.try_clone() else { return };
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

fn run(kernel: &Kernel, line: &str) -> String {
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
                find(&mem, &pattern, range.0.unwrap_or(0), range.1.unwrap_or(u64::MAX))
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
                    kernel.hle.describe(t.stub).unwrap_or_else(|| format!("{:#x}", t.stub)),
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
                    format!("watching {addr:#x}..{:#x} (interpreter and host writes only: {e:?})", addr + len)
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
        #[cfg(target_os = "macos")]
        "gputrace" => match it.next().and_then(|n| n.parse().ok()) {
            Some(n) => {
                hle::gpu::metal::TRACE_NEXT.store(n, std::sync::atomic::Ordering::Relaxed);
                "ok".into()
            }
            None => "usage: gputrace N".into(),
        },
        #[cfg(target_os = "macos")]
        "targets" => match it.next() {
            Some(dir) => {
                let mut n = 0;
                for (addr, t) in hle::gpu::metal::all_frames() {
                    if let Some((w, h, px)) = hle::gpu::metal::read_rgba(&t) {
                        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                        ppm.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
                        if std::fs::write(Path::new(dir).join(format!("{addr:x}.ppm")), ppm).is_ok() {
                            n += 1;
                        }
                    }
                }
                format!("{n} written")
            }
            None => "usage: targets DIR".into(),
        },
        "press" => match it.next().map(crate::pad_buttons) {
            Some(buttons) if buttons != 0 => {
                let ms = it.next().and_then(|m| m.parse().ok()).unwrap_or(400);
                kernel.set_pad_input(hle::PadInput { buttons, ..Default::default() });
                std::thread::sleep(std::time::Duration::from_millis(ms));
                kernel.set_pad_input(hle::PadInput::default());
                "ok".into()
            }
            _ => "usage: press BUTTONS [MS]".into(),
        },
        "quit" => {
            kernel.exit(Ok(0));
            "ok".into()
        }
        _ => format!("unknown command {cmd}; try help"),
    }
}

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
            .map(|&c| if c.is_ascii_graphic() || c == b' ' { c as char } else { '.' })
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
    if found.is_empty() { "not found".into() } else { found.join("\n") }
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
            format!("host call from {}", sym(kernel, site & !memory::watch::HOST_CALL))
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
