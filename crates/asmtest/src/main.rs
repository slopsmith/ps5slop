//! Runs FEX-Emu's x86-64 ASM unit tests against our interpreter.
//!
//! FEX runs the same tests natively on x86-64 hardware ("host" mode), so the
//! register values each test expects have been checked on a real CPU. We set
//! up memory and registers the way FEX's host runner does: code at 0x10000,
//! every register zero, RFLAGS 0x202, and the test ends at `hlt`. Tests that
//! FEX disables or marks as failing on hardware are not run. See
//! `docs/cpu.md`.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use memory::{Memory, PAGE_SIZE, Prot};
use serde_json::Value;
use x86::cpu::flags;
use x86::decode::Map;
use x86::{Cpu, CpuFault, Exit};

const CODE_START: u64 = 0x1_0000;
const STEP_LIMIT: u64 = 50_000_000;
const GPR_NAMES: [&str; 16] = [
    "RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI", "R8", "R9", "R10", "R11", "R12", "R13",
    "R14", "R15",
];
/// Features the tests may ask for that the PS5's Zen 2 cores lack, or that
/// need a Linux kernel underneath.
const NOT_ON_ZEN2: [&str; 11] = [
    "3DNOW",
    "EMMI",
    "AVX_VNNI",
    "AVX_IFMA",
    "AVX_NE_CONVERT",
    "AVX_VNNI_INT8",
    "AVX_VNNI_INT16",
    "AES256",
    "SHA512",
    "SERIALIZE",
    "LINUX",
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Pass,
    Wrong(String),
    /// Expects a register we do not model yet (XMM, MMX, FS/GS).
    Unchecked(String),
    Unimplemented(String),
    Fault(String),
    Timeout,
    Skipped(String),
    AsmError(String),
}

impl Outcome {
    fn label(&self) -> &'static str {
        match self {
            Outcome::Pass => "pass",
            Outcome::Wrong(_) => "wrong",
            Outcome::Unchecked(_) => "unchecked",
            Outcome::Unimplemented(_) => "unimplemented",
            Outcome::Fault(_) => "fault",
            Outcome::Timeout => "timeout",
            Outcome::Skipped(_) => "skipped",
            Outcome::AsmError(_) => "asm-error",
        }
    }

    fn detail(&self) -> &str {
        match self {
            Outcome::Wrong(s)
            | Outcome::Unchecked(s)
            | Outcome::Unimplemented(s)
            | Outcome::Fault(s)
            | Outcome::Skipped(s)
            | Outcome::AsmError(s) => s,
            _ => "",
        }
    }
}

struct Options {
    root: PathBuf,
    filter: Option<String>,
    report: PathBuf,
    verbose: bool,
}

fn parse_args() -> Result<Options, String> {
    let mut opts = Options {
        root: PathBuf::from("references/src/FEX/unittests/ASM"),
        filter: None,
        report: PathBuf::from("target/asmtest/report.tsv"),
        verbose: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--filter" => opts.filter = Some(args.next().ok_or("--filter needs a value")?),
            "--report" => opts.report = PathBuf::from(args.next().ok_or("--report needs a value")?),
            "--verbose" | "-v" => opts.verbose = true,
            "--help" | "-h" => {
                return Err("usage: ps5slop-asmtest [ASM_DIR] [--filter SUBSTRING] [--report PATH] [--verbose]\n\
                     Run scripts/fetch-references first; needs nasm (nix develop)."
                    .into());
            }
            _ => opts.root = PathBuf::from(a),
        }
    }
    Ok(opts)
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut tests = Vec::new();
    collect(&opts.root, &opts.root, &mut tests);
    tests.sort();
    if let Some(f) = &opts.filter {
        tests.retain(|t| t.contains(f.as_str()));
    }
    if tests.is_empty() {
        eprintln!(
            "no tests found under {}; run scripts/fetch-references",
            opts.root.display()
        );
        return ExitCode::FAILURE;
    }
    let skip = skip_list(&opts.root);
    let build = PathBuf::from("target/asmtest/build");

    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(tests.len()));
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(name) = tests.get(i) else { break };
                    let outcome = if let Some(reason) = skip.get(normalise(name).as_str()) {
                        Outcome::Skipped(reason.clone())
                    } else {
                        run_test(&opts.root, &build, name)
                    };
                    results.lock().unwrap().push((name.clone(), outcome));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    report(&opts, &results)
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n != "Includes") {
                collect(root, &p, out);
            }
        } else if p.extension().is_some_and(|x| x == "asm") {
            out.push(p.strip_prefix(root).unwrap().to_string_lossy().into_owned());
        }
    }
}

/// FEX's lists name tests as `Test_<dir>/<file>` or `<dir>/<file>`.
fn normalise(name: &str) -> String {
    name.strip_prefix("Test_").unwrap_or(name).to_owned()
}

/// Tests FEX does not run on hardware, or knows to fail there, with the
/// list they came from.
fn skip_list(root: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for list in [
        "Disabled_Tests",
        "Disabled_Tests_host",
        "Known_Failures",
        "Known_Failures_host",
    ] {
        let Ok(text) = std::fs::read_to_string(root.join(list)) else {
            continue;
        };
        for line in text.lines().map(str::trim) {
            if !line.is_empty() && !line.starts_with('#') {
                out.insert(normalise(line), format!("in FEX's {list}"));
            }
        }
    }
    out
}

fn config(text: &str) -> Result<Value, String> {
    let json = text
        .split("%ifdef CONFIG")
        .nth(1)
        .and_then(|rest| rest.split("%endif").next())
        .ok_or("no CONFIG block")?;
    let v: Value = serde_json::from_str(json).map_err(|e| format!("bad CONFIG json: {e}"))?;
    let map = v.as_object().ok_or("CONFIG is not an object")?;
    Ok(Value::Object(
        map.iter()
            .map(|(k, v)| (k.to_uppercase(), v.clone()))
            .collect(),
    ))
}

fn int(v: &Value) -> Option<u64> {
    let s = v.as_str()?;
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}

/// FEX's MemoryData strings: space-separated groups of hex digits, each
/// written most significant byte first and stored least significant first.
fn hex_bytes(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    for group in s.split_whitespace() {
        let mut g = group.strip_prefix("0x").unwrap_or(group);
        while !g.is_empty() {
            let take = if g.len() >= 2 { 2 } else { 1 };
            let (rest, byte) = g.split_at(g.len() - take);
            out.push(u8::from_str_radix(byte, 16).ok()?);
            g = rest;
        }
    }
    Some(out)
}

fn assemble(root: &Path, build: &Path, name: &str, text: &str) -> Result<Vec<u8>, String> {
    let src = build.join(name).with_extension("asm");
    let bin = build.join(name).with_extension("bin");
    std::fs::create_dir_all(src.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&src, format!("BITS 64\n{text}\nret\n")).map_err(|e| e.to_string())?;
    let dir = root.join(name).parent().unwrap().to_path_buf();
    let out = Command::new("nasm")
        .arg("-i")
        .arg(root.join("Includes/"))
        .arg("-i")
        .arg(format!("{}/", dir.display()))
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .output()
        .map_err(|e| format!("cannot run nasm: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr)
            .lines()
            .next()
            .unwrap_or("nasm failed")
            .to_owned());
    }
    std::fs::read(&bin).map_err(|e| e.to_string())
}

/// Maps every page of `[addr, addr + len)` that is not mapped yet.
fn map_range(mem: &mut Memory, addr: u64, len: u64, prot: Prot) {
    let start = addr / PAGE_SIZE * PAGE_SIZE;
    let end = (addr + len).next_multiple_of(PAGE_SIZE);
    let mut page = start;
    while page < end {
        let _ = mem.map(page, PAGE_SIZE, prot, "test");
        page += PAGE_SIZE;
    }
}

fn run_test(root: &Path, build: &Path, name: &str) -> Outcome {
    let text = match std::fs::read_to_string(root.join(name)) {
        Ok(t) => t,
        Err(e) => return Outcome::AsmError(e.to_string()),
    };
    let cfg = match config(&text) {
        Ok(c) => c,
        Err(e) => return Outcome::AsmError(e),
    };
    if cfg
        .get("MODE")
        .and_then(Value::as_str)
        .is_some_and(|m| m.eq_ignore_ascii_case("32BIT"))
    {
        return Outcome::Skipped("32-bit mode".into());
    }
    if let Some(features) = cfg.get("HOSTFEATURES").and_then(Value::as_array) {
        for f in features.iter().filter_map(Value::as_str) {
            let f = f.to_uppercase().replace('.', "_");
            if NOT_ON_ZEN2.contains(&f.as_str()) {
                return Outcome::Skipped(format!("needs {f}, which Zen 2 lacks"));
            }
        }
    }
    let code = match assemble(root, build, name, &text) {
        Ok(c) => c,
        Err(e) => return Outcome::AsmError(e),
    };

    let mut mem = Memory::new(1 << 36).expect("reserve guest memory");
    let all = Prot::new(true, true, true);
    map_range(&mut mem, CODE_START, code.len() as u64, all);
    mem.write_host(CODE_START, &code)
        .expect("code was just mapped");
    map_range(&mut mem, 0xe000_0000, 10 * PAGE_SIZE, Prot::RW);
    map_range(&mut mem, 0xe800_0000 - PAGE_SIZE, 2 * PAGE_SIZE, Prot::RW);
    if let Some(regions) = cfg.get("MEMORYREGIONS").and_then(Value::as_object) {
        for (base, size) in regions {
            let (Some(base), Some(size)) = (int(&Value::String(base.clone())), int(size)) else {
                return Outcome::AsmError("bad MemoryRegions".into());
            };
            map_range(&mut mem, base, size, Prot::RW);
        }
    }
    if let Some(data) = cfg.get("MEMORYDATA").and_then(Value::as_object) {
        for (addr, bytes) in data {
            let (Some(addr), Some(bytes)) = (
                int(&Value::String(addr.clone())),
                bytes.as_str().and_then(hex_bytes),
            ) else {
                return Outcome::AsmError("bad MemoryData".into());
            };
            if mem.write_host(addr, &bytes).is_err() {
                return Outcome::AsmError(format!("MemoryData at {addr:#x} is not mapped"));
            }
        }
    }

    let mut cpu = Cpu {
        rip: CODE_START,
        rflags: 0x202,
        ..Cpu::default()
    };
    debug_assert_eq!(cpu.rflags & flags::FIXED, flags::FIXED);
    let mut steps = 0u64;
    loop {
        match x86::step(&mut cpu, &mut mem) {
            Exit::Continue => {}
            Exit::Fault {
                rip,
                kind: CpuFault::Privileged,
            } if fetch_byte(&mem, rip) == Some(0xf4) => break,
            Exit::Fault {
                kind: CpuFault::Unimplemented { map, op },
                ..
            } => {
                let prefix = match map {
                    Map::Primary => "",
                    Map::M0F => "0F ",
                    Map::M0F38 => "0F38 ",
                    Map::M0F3A => "0F3A ",
                };
                return Outcome::Unimplemented(format!("{prefix}{op:02X}"));
            }
            Exit::Fault {
                kind: CpuFault::UnsupportedEncoding,
                ..
            } => {
                return Outcome::Unimplemented("VEX/EVEX".into());
            }
            Exit::Fault { rip, kind } => return Outcome::Fault(format!("{kind:?} at {rip:#x}")),
            Exit::Syscall => return Outcome::Skipped("makes a Linux system call".into()),
            other => return Outcome::Fault(format!("{other:?}")),
        }
        steps += 1;
        if steps >= STEP_LIMIT {
            return Outcome::Timeout;
        }
    }
    compare(&cpu, &cfg)
}

fn fetch_byte(mem: &Memory, addr: u64) -> Option<u8> {
    let mut b = [0u8];
    mem.fetch(addr, &mut b).ok().map(|_| b[0])
}

fn compare(cpu: &Cpu, cfg: &Value) -> Outcome {
    let Some(expected) = cfg.get("REGDATA").and_then(Value::as_object) else {
        return Outcome::Pass;
    };
    let mut unchecked = Vec::new();
    let mut wrong = Vec::new();
    for (reg, want) in expected {
        let reg = reg.to_uppercase();
        let got = if let Some(i) = GPR_NAMES.iter().position(|n| *n == reg) {
            cpu.regs[i]
        } else if reg == "RIP" {
            cpu.rip
        } else if let Some(n) = reg
            .strip_prefix("XMM")
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|&n| n < 16)
        {
            let Some(list) = want.as_array() else {
                wrong.push(format!("{reg}: expected a list of 64-bit values"));
                continue;
            };
            for (k, v) in list.iter().enumerate() {
                match (int(v), cpu.xmm[n].get(k)) {
                    (Some(w), Some(&g)) if w == g => {}
                    (Some(w), Some(&g)) => {
                        wrong.push(format!("{reg}[{k}]={g:#x}, expected {w:#x}"))
                    }
                    _ => wrong.push(format!("{reg}[{k}]: unreadable")),
                }
            }
            continue;
        } else {
            unchecked.push(reg);
            continue;
        };
        let Some(want) = int(want) else {
            wrong.push(format!("{reg}: unreadable expected value"));
            continue;
        };
        if got != want {
            wrong.push(format!("{reg}={got:#x}, expected {want:#x}"));
        }
    }
    if !wrong.is_empty() {
        Outcome::Wrong(wrong.join("; "))
    } else if !unchecked.is_empty() {
        Outcome::Unchecked(unchecked.join(","))
    } else {
        Outcome::Pass
    }
}

fn report(opts: &Options, results: &[(String, Outcome)]) -> ExitCode {
    let mut totals: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_dir: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut missing: BTreeMap<String, usize> = BTreeMap::new();
    let mut tsv = String::from("test\toutcome\tdetail\n");
    for (name, o) in results {
        *totals.entry(o.label()).or_default() += 1;
        if !matches!(o, Outcome::Skipped(_)) {
            let dir = Path::new(name)
                .parent()
                .map_or(String::new(), |p| p.to_string_lossy().into_owned());
            let e = by_dir
                .entry(if dir.is_empty() { ".".into() } else { dir })
                .or_default();
            e.1 += 1;
            if *o == Outcome::Pass {
                e.0 += 1;
            }
        }
        if let Outcome::Unimplemented(op) = o {
            *missing.entry(op.clone()).or_default() += 1;
        }
        tsv += &format!("{name}\t{}\t{}\n", o.label(), o.detail());
        if opts.verbose && !matches!(o, Outcome::Pass | Outcome::Skipped(_)) {
            println!("{:13} {name}: {}", o.label(), o.detail());
        }
    }
    if let Some(parent) = opts.report.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&opts.report, tsv);

    let ran: usize = by_dir.values().map(|v| v.1).sum();
    let passed: usize = by_dir.values().map(|v| v.0).sum();
    println!(
        "{} tests: {passed} of {ran} run passed ({:.1}%)",
        results.len(),
        100.0 * passed as f64 / ran.max(1) as f64
    );
    for (label, n) in &totals {
        println!("  {label:13} {n}");
    }
    println!("\nby directory (passed / run):");
    for (dir, (p, n)) in &by_dir {
        println!("  {dir:28} {p:5} / {n}");
    }
    let mut top: Vec<_> = missing.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("\nmost wanted unimplemented opcodes (tests blocked):");
    for (op, n) in top.iter().take(25) {
        println!("  {op:10} {n}");
    }
    println!("\nper-test results: {}", opts.report.display());
    let failed: HashSet<&str> = ["wrong", "fault", "asm-error"].into_iter().collect();
    if results.iter().any(|(_, o)| failed.contains(o.label())) {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
