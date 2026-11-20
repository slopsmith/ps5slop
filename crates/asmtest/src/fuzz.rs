//! Differential fuzzing of single instructions: random register states run
//! natively (Rosetta 2 on Apple Silicon) by `fuzz/harness.c` and in our
//! interpreter, then compared. Only register forms (ModRM mod = 3) are made,
//! and no encoding names RSP. See `docs/cpu.md`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use memory::{Memory, PAGE_SIZE, Prot};
use x86::{Cpu, CpuFault, Exit};

const CODE: u64 = 0x1_0000;
const STATE_SIZE: usize = 0x290;
const ARITH_FLAGS: u64 = 0x8d5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Random bits, small numbers, or edge values, to reach corner cases.
    fn value(&mut self) -> u64 {
        match self.below(6) {
            0 => self.below(64),
            1 => [0, u64::MAX, 1 << 63, 0x8000_0000, 0xffff_ffff, 0x7fff_ffff][self.below(6) as usize],
            _ => self.next(),
        }
    }

    fn reg(&mut self) -> u8 {
        loop {
            let r = self.below(16) as u8;
            if r != 4 {
                return r;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct State {
    r: [u64; 16],
    flags: u64,
    status: u64,
    y: [[u64; 4]; 16],
}

impl State {
    fn random(rng: &mut Rng) -> Self {
        let mut s = State {
            r: [0; 16],
            flags: 0x202 | (rng.next() & ARITH_FLAGS),
            status: 0,
            y: [[0; 4]; 16],
        };
        for r in &mut s.r {
            *r = rng.value();
        }
        for y in &mut s.y {
            for l in y.iter_mut() {
                *l = rng.value();
            }
        }
        s
    }

    fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(STATE_SIZE);
        for v in self.r.iter().chain([&self.flags, &self.status]) {
            b.extend(v.to_le_bytes());
        }
        for y in &self.y {
            for l in y {
                b.extend(l.to_le_bytes());
            }
        }
        b
    }

    fn parse(b: &[u8]) -> Self {
        let q = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        let mut s = State {
            r: [0; 16],
            flags: q(16),
            status: q(17),
            y: [[0; 4]; 16],
        };
        for i in 0..16 {
            s.r[i] = q(i);
            for l in 0..4 {
                s.y[i][l] = q(18 + i * 4 + l);
            }
        }
        s
    }
}

/// Primary opcodes safe to run alone in register form: no control flow, stack,
/// string, I/O or x87.
fn primary_ok(op: u8) -> bool {
    matches!(op,
        0x00..=0x05 | 0x08..=0x0d | 0x10..=0x15 | 0x18..=0x1d | 0x20..=0x25 | 0x28..=0x2d
        | 0x30..=0x35 | 0x38..=0x3d | 0x63 | 0x69 | 0x6b | 0x80..=0x8b | 0x91..=0x93 | 0x95..=0x99
        | 0x9e | 0x9f | 0xa8 | 0xa9 | 0xb0..=0xb3 | 0xb5..=0xbb | 0xbd..=0xbf | 0xc0 | 0xc1 | 0xc6 | 0xc7 | 0xd0..=0xd3
        | 0xf5 | 0xf6 | 0xf7 | 0xf8 | 0xf9 | 0xfe)
}

/// Two-byte (0F) opcodes to leave out: system, control flow, stack, timing,
/// randomness, the escapes, and `maskmov`, which stores to `[rdi]`.
fn twobyte_skip(op: u8) -> bool {
    matches!(op,
        0x00 | 0x01 | 0x05..=0x09 | 0x0d | 0x0f | 0x30..=0x38 | 0x3a | 0x78 | 0x79 | 0x80..=0x8f | 0xa0..=0xa2
        | 0xa8..=0xaa | 0xae | 0xc7 | 0xcc | 0xf7)
}

fn encodings(rng: &mut Rng) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let tail = |rng: &mut Rng, mut b: Vec<u8>| {
        for _ in 0..8 {
            b.push(rng.next() as u8);
        }
        b
    };
    let modrm = |rng: &mut Rng| -> (u8, u8, u8) {
        let (reg, rm) = (rng.reg(), rng.reg());
        (0xc0 | (reg & 7) << 3 | (rm & 7), reg >> 3, rm >> 3)
    };
    for map in 1..=3u8 {
        for pp in 0..4u8 {
            for l in 0..2u8 {
                for w in 0..2u8 {
                    for op in 0..=255u8 {
                        if map == 1 && op == 0xf7 {
                            continue;
                        }
                        let (m, r, b) = modrm(rng);
                        let v = rng.reg();
                        let b1 = ((r ^ 1) << 7) | (1 << 6) | ((b ^ 1) << 5) | map;
                        let b2 = (w << 7) | ((!v & 15) << 3) | (l << 2) | pp;
                        out.push(tail(rng, vec![0xc4, b1, b2, op, m]));
                    }
                }
            }
        }
    }
    for prefix in [None, Some(0x66u8), Some(0xf3), Some(0xf2)] {
        for w in 0..2u8 {
            for escape in [&[0x0fu8][..], &[0x0f, 0x38], &[0x0f, 0x3a]] {
                for op in 0..=255u8 {
                    if escape.len() == 1 && twobyte_skip(op) {
                        continue;
                    }
                    let (m, r, b) = modrm(rng);
                    let mut e: Vec<u8> = prefix.into_iter().collect();
                    e.push(0x40 | w << 3 | r << 2 | b);
                    e.extend(escape);
                    e.extend([op, m]);
                    out.push(tail(rng, e));
                }
            }
            for op in (0..=255u8).filter(|&o| primary_ok(o)) {
                let (m, r, b) = modrm(rng);
                let mut e: Vec<u8> = prefix.filter(|&p| p == 0x66).into_iter().collect();
                e.push(0x40 | w << 3 | r << 2 | b);
                e.extend([op, m]);
                out.push(tail(rng, e));
            }
        }
    }
    out
}

/// What our interpreter makes of one state.
enum Ours {
    Ran(Box<State>),
    Unimplemented,
    Invalid,
    Other(String),
}

fn run_ours(mem: &Memory, code: &[u8], s: &State) -> Ours {
    mem.write_host(CODE, code).unwrap();
    let mut cpu = Cpu {
        regs: s.r,
        rip: CODE,
        rflags: s.flags,
        xmm: s.y,
        ..Cpu::default()
    };
    cpu.regs[4] = 0x8000;
    match x86::step(&mut cpu, mem) {
        Exit::Continue if cpu.rip == CODE + code.len() as u64 => {
            let mut r = cpu.regs;
            r[4] = s.r[4];
            Ours::Ran(Box::new(State {
                r,
                flags: cpu.rflags,
                status: 0,
                y: cpu.xmm,
            }))
        }
        Exit::Continue => Ours::Other(format!("jumped to {:#x}", cpu.rip)),
        Exit::Fault {
            kind: CpuFault::Unimplemented { .. } | CpuFault::UnsupportedEncoding,
            ..
        } => Ours::Unimplemented,
        Exit::Fault {
            kind: CpuFault::InvalidOpcode,
            ..
        } => Ours::Invalid,
        e => Ours::Other(format!("{e:?}")),
    }
}

fn describe(want: &State, got: &State, before: &State) -> String {
    let mut d = Vec::new();
    for i in 0..16 {
        if i != 4 && want.r[i] != got.r[i] {
            d.push(format!("r{i}: want {:#x} got {:#x} (was {:#x})", want.r[i], got.r[i], before.r[i]));
        }
    }
    for i in 0..16 {
        if want.y[i] != got.y[i] {
            d.push(format!("ymm{i}: want {:x?} got {:x?} (was {:x?})", want.y[i], got.y[i], before.y[i]));
        }
    }
    let f = (want.flags ^ got.flags) & ARITH_FLAGS;
    if f != 0 {
        d.push(format!("flags: want {:#x} got {:#x} (differ {f:#x})", want.flags & ARITH_FLAGS, got.flags & ARITH_FLAGS));
    }
    d.join("; ")
}

fn build_harness(dir: &Path) -> Result<PathBuf, String> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/harness.c");
    let bin = dir.join("harness");
    let ok = Command::new("/usr/bin/clang")
        .args(["-arch", "x86_64", "-O1", "-mavx2", "-o"])
        .arg(&bin)
        .arg(&src)
        .status()
        .map_err(|e| e.to_string())?
        .success();
    if ok { Ok(bin) } else { Err("cannot build the harness".into()) }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let states: u32 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(8);
    let seed: u64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(0x5eed_1234_abcd_0001);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fuzz");
    std::fs::create_dir_all(&dir).expect("make target/fuzz");
    let harness = match build_harness(&dir) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("ps5slop-fuzz: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mem = Memory::new(1 << 32).expect("reserve guest memory");
    mem.map(CODE, PAGE_SIZE, Prot::new(true, true, true), "code").unwrap();
    mem.map(0, PAGE_SIZE, Prot::new(true, true, false), "stack").unwrap();
    let mut rng = Rng(seed);
    let mut cases = Vec::new();
    let mut input = Vec::new();
    let mut unimplemented: BTreeMap<String, u32> = BTreeMap::new();
    for enc in encodings(&mut rng) {
        let fetch = |i: usize| enc.get(i).copied().ok_or(());
        let Ok(insn) = x86::decode::decode(fetch) else { continue };
        let code = enc[..insn.len as usize].to_vec();
        let before: Vec<State> = (0..states).map(|_| State::random(&mut rng)).collect();
        let mut head = [0u8; 16];
        head[0] = code.len() as u8;
        head[1..1 + code.len()].copy_from_slice(&code);
        input.extend(head);
        input.extend(states.to_le_bytes());
        for s in &before {
            input.extend(s.bytes());
        }
        cases.push((code, before));
    }
    let (inp, outp) = (dir.join("cases.bin"), dir.join("results.bin"));
    std::fs::write(&inp, &input).expect("write cases");
    let ran = Command::new("arch").arg("-x86_64").arg(&harness).arg(&inp).arg(&outp).status();
    if !ran.is_ok_and(|s| s.success()) {
        eprintln!("ps5slop-fuzz: the harness failed");
        return ExitCode::FAILURE;
    }
    let results = std::fs::read(&outp).expect("read results");
    let mut at = 0;
    let mut report = Vec::new();
    let (mut value_bad, mut flag_bad, mut same, mut accepted_invalid, mut rejected_valid, mut other) = (0, 0, 0, 0, 0, 0);
    let mut both_reject = 0;
    'cases: for (code, before) in &cases {
        let start = at + 20;
        at = start + before.len() * STATE_SIZE;
        let hex: String = code.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
        let mut first: Option<String> = None;
        let mut flags_only = true;
        for (i, s) in before.iter().enumerate() {
            let o = start + i * STATE_SIZE;
            let want = State::parse(&results[o..o + STATE_SIZE]);
            match (want.status, run_ours(&mem, code, s)) {
                (0, Ours::Ran(got)) => {
                    let got = *got;
                    let mut w = want;
                    w.status = 0;
                    if w.r == got.r && w.y == got.y && (w.flags ^ got.flags) & ARITH_FLAGS == 0 {
                        continue;
                    }
                    if w.r != got.r || w.y != got.y {
                        flags_only = false;
                    }
                    first.get_or_insert_with(|| describe(&w, &got, s));
                }
                (0, Ours::Unimplemented) => {
                    *unimplemented.entry(hex.clone()).or_default() += 1;
                    continue 'cases;
                }
                (0, Ours::Invalid) => {
                    rejected_valid += 1;
                    report.push(format!("{hex}: we reject a valid instruction"));
                    continue 'cases;
                }
                (0, Ours::Other(e)) => {
                    other += 1;
                    report.push(format!("{hex}: ours {e}"));
                    continue 'cases;
                }
                (sig, Ours::Ran(_)) => {
                    accepted_invalid += 1;
                    report.push(format!("{hex}: native signal {sig}, we ran it"));
                    continue 'cases;
                }
                _ => {
                    both_reject += 1;
                    continue 'cases;
                }
            }
        }
        match first {
            Some(d) => {
                if flags_only { flag_bad += 1 } else { value_bad += 1 }
                report.push(format!("{hex}: {}{d}", if flags_only { "[flags] " } else { "" }));
            }
            None => same += 1,
        }
    }
    let path = dir.join("report.txt");
    let mut f = std::fs::File::create(&path).expect("write report");
    for l in &report {
        let _ = writeln!(f, "{l}");
    }
    let _ = writeln!(f, "\nunimplemented (native ran them):");
    for k in unimplemented.keys() {
        let _ = writeln!(f, "{k}");
    }
    println!(
        "{} encodings: {same} agree, {value_bad} wrong values, {flag_bad} wrong flags only, \
         {both_reject} invalid in both, {accepted_invalid} we accept but native rejects, {rejected_valid} we reject, {other} misbehave, \
         {} unimplemented; see {}",
        cases.len(),
        unimplemented.len(),
        path.display()
    );
    ExitCode::SUCCESS
}
