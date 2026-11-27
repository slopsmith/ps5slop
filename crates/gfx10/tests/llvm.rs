//! Checks the decoder and printer against `llvm-mc --disassemble -mcpu=gfx1013`.
//! Skipped when llvm-mc is not on PATH (it is in the Nix dev shell).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use gfx10::ags::Ags;
use gfx10::opinfo::{self, Fmt};
use gfx10::{Syntax, decode, decode_all, print, print_checked};

fn have_llvm_mc() -> bool {
    Command::new("llvm-mc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// LLVM's text for `words`, one line per instruction, or `None` if it rejects them.
fn llvm(words: &[u32], wave64: bool) -> Option<Vec<String>> {
    let attr = if wave64 {
        "+wavefrontsize64"
    } else {
        "+wavefrontsize32"
    };
    let mut child = Command::new("llvm-mc")
        .args([
            "--disassemble",
            "-triple=amdgcn",
            "-mcpu=gfx1013",
            &format!("-mattr={attr}"),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("llvm-mc");
    let input: Vec<String> = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|b| format!("0x{b:02x}"))
        .collect();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{}", input.join(",")).unwrap();
    drop(stdin);
    let out = child.wait_with_output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("invalid instruction encoding") || err.contains("potentially undefined") {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    if text.contains("Invalid") || text.contains("invalid_") {
        return None;
    }
    Some(
        text.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
    )
}

/// Whether LLVM's assembler turns `text` back into `words`. When it does not, LLVM's
/// disassembler printed a fallback for an encoding it does not really support.
fn reassembles(text: &str, words: &[u32], wave64: bool) -> bool {
    let attr = if wave64 {
        "+wavefrontsize64"
    } else {
        "+wavefrontsize32"
    };
    let mut child = Command::new("llvm-mc")
        .args([
            "-show-encoding",
            "-triple=amdgcn",
            "-mcpu=gfx1013",
            &format!("-mattr={attr}"),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("llvm-mc");
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{text}").unwrap();
    drop(stdin);
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let Some(enc) = text
        .split("encoding: [")
        .nth(1)
        .and_then(|e| e.split(']').next())
    else {
        return false;
    };
    let bytes: Vec<u8> = enc
        .split(',')
        .filter_map(|b| u8::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    let want: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes == want
}

fn ours(words: &[u32], wave64: bool) -> Option<(String, usize)> {
    let i = decode(words).ok()?;
    Some((print_checked(&i, Syntax { wave64 })?, i.words as usize))
}

#[test]
fn balatro_shaders_match_llvm() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../games/Balatro/shaders");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: no {}", dir.display());
        return;
    };
    if !have_llvm_mc() {
        eprintln!("skipped: no llvm-mc");
        return;
    }
    let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    let (mut total, mut matched) = (0, 0);
    let mut bad = Vec::new();
    for path in paths
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "ags"))
    {
        let file = std::fs::read(path).unwrap();
        let ags = Ags::parse(&file).unwrap();
        let words = ags.code_words();
        let wave64 = ags.wave64();
        let (insts, err) = decode_all(&words);
        assert!(err.is_none(), "{}: decode error {err:?}", path.display());
        let theirs = llvm(&words, wave64).expect("llvm-mc rejected a shader");
        assert_eq!(
            theirs.len(),
            insts.len(),
            "{}: instruction count",
            path.display()
        );
        for ((pc, inst), t) in insts.iter().zip(&theirs) {
            total += 1;
            let o = print(inst, Syntax { wave64 });
            if &o == t {
                matched += 1;
            } else {
                bad.push(format!(
                    "{}+{:#x}: ours `{o}` llvm `{t}`",
                    path.file_name().unwrap().display(),
                    pc * 4
                ));
            }
        }
    }
    eprintln!("Balatro: {matched}/{total} instructions match llvm-mc");
    for b in bad.iter().take(40) {
        eprintln!("  {b}");
    }
    assert!(bad.is_empty());
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    /// A source operand code, biased toward the interesting ones.
    fn src9(&mut self) -> u32 {
        match self.below(8) {
            0 => 256 + self.below(256),
            1 => 128 + self.below(81),
            2 => 240 + self.below(9),
            3 => 255,
            4 => [
                106, 107, 124, 125, 126, 127, 251, 252, 253, 254, 235, 236, 237, 238, 239,
            ][self.below(15) as usize],
            5 => self.below(106),
            _ => self.below(512),
        }
    }

    fn src8(&mut self) -> u32 {
        self.src9() & 0xFF
    }
}

/// Builds a random instruction of `fmt` with opcode `code` (dwords, enough for any literal).
fn random_inst(rng: &mut Rng, fmt: Fmt, code: u32) -> Vec<u32> {
    let r = rng.next();
    let r1 = rng.next();
    let lit = rng.next();
    match fmt {
        Fmt::Sop2 => vec![
            2 << 30 | code << 23 | (rng.below(128)) << 16 | rng.src8() << 8 | rng.src8(),
            lit,
        ],
        Fmt::Sopk => vec![
            0xB << 28 | code << 23 | rng.below(128) << 16 | (r & 0xFFFF),
            lit,
        ],
        Fmt::Sop1 => vec![
            0x17D << 23 | rng.below(128) << 16 | code << 8 | rng.src8(),
            lit,
        ],
        Fmt::Sopc => vec![0x17E << 23 | code << 16 | rng.src8() << 8 | rng.src8(), lit],
        Fmt::Sopp => {
            let imm = if rng.below(2) == 0 {
                r & 0xFFFF
            } else {
                r & 0x3F
            };
            vec![0x17F << 23 | code << 16 | imm]
        }
        Fmt::Smem => {
            let w = 0x3D << 26
                | code << 18
                | (r & (1 << 16 | 1 << 14))
                | rng.below(128) << 6
                | rng.below(64);
            let off = if rng.below(2) == 0 {
                r1 & 0x1F_FFFF
            } else {
                r1 & 0xFF
            };
            let soff = if rng.below(2) == 0 {
                125
            } else {
                rng.below(128)
            };
            vec![w, soff << 25 | off]
        }
        Fmt::Vop1 => {
            let s = match rng.below(6) {
                0 => 249,
                1 => 250,
                2 => 233 + rng.below(2),
                _ => rng.src9(),
            };
            vec![0x3F << 25 | rng.below(256) << 17 | code << 9 | s, r1, lit]
        }
        Fmt::Vop2 => {
            let s = match rng.below(6) {
                0 => 249,
                1 => 250,
                2 => 233 + rng.below(2),
                _ => rng.src9(),
            };
            vec![
                code << 25 | rng.below(256) << 17 | rng.below(256) << 9 | s,
                r1,
                lit,
            ]
        }
        Fmt::Vopc => {
            let s = match rng.below(5) {
                0 => 249,
                1 => 250,
                _ => rng.src9(),
            };
            vec![0x3E << 25 | code << 17 | rng.below(256) << 9 | s, r1, lit]
        }
        Fmt::Vop3 => {
            let mods = if rng.below(2) == 0 { r & 0xFF00 } else { 0 };
            let w = 0x35 << 26 | code << 16 | mods | rng.below(256);
            let hi = if rng.below(2) == 0 {
                r1 & 0xF800_0000
            } else {
                0
            };
            vec![w, hi | rng.src9() << 18 | rng.src9() << 9 | rng.src9(), lit]
        }
        Fmt::Vop3p => {
            let w = 0x33 << 26 | code << 16 | (r & 0xFF00) | rng.below(256);
            vec![
                w,
                (r1 & 0xF800_0000) | rng.src9() << 18 | rng.src9() << 9 | rng.src9(),
                lit,
            ]
        }
        Fmt::Vintrp => vec![0x32 << 26 | rng.below(256) << 18 | code << 16 | (r & 0xFFFF)],
        Fmt::Ds => vec![0x36 << 26 | code << 18 | (r & 0x2FFFF), r1],
        Fmt::Mubuf => {
            let w = 0x38 << 26 | (code >> 7) << 25 | (code & 0x7F) << 18 | (r & 0x1FFFF);
            vec![w, (r1 & !(1 << 21)) & 0xFFDF_FFFF]
        }
        Fmt::Mtbuf => {
            let w = 0x3A << 26 | rng.below(128) << 19 | (code & 7) << 16 | (r & 0xFFFF);
            vec![w, (r1 & !(1 << 21)) | (code >> 3) << 21]
        }
        Fmt::Mimg => {
            let nsa = if rng.below(2) == 0 { 0 } else { rng.below(4) };
            let w = 0x3C << 26 | (code & 0x7F) << 18 | (code >> 7) | nsa << 1 | (r & 0x0203_BFB8);
            let mut v = vec![w, r1 & 0xC3FF_FFFF];
            for _ in 0..nsa {
                v.push(rng.next());
            }
            v
        }
        Fmt::Exp => vec![0x3E << 26 | (r & 0x1FFF), r1],
        Fmt::Flat | Fmt::Global | Fmt::Scratch => {
            let seg = match fmt {
                Fmt::Flat => 0,
                Fmt::Scratch => 1,
                _ => 2,
            };
            let saddr = if rng.below(2) == 0 {
                0x7D
            } else {
                rng.below(128)
            };
            vec![
                0x37 << 26 | code << 18 | seg << 14 | (r & 0x33FFF),
                (r1 & 0xFF80_FFFF) | saddr << 16,
            ]
        }
    }
}

const FORMATS: [Fmt; 20] = [
    Fmt::Sop2,
    Fmt::Sopk,
    Fmt::Sop1,
    Fmt::Sopc,
    Fmt::Sopp,
    Fmt::Smem,
    Fmt::Vop1,
    Fmt::Vop2,
    Fmt::Vopc,
    Fmt::Vop3,
    Fmt::Vop3p,
    Fmt::Vintrp,
    Fmt::Ds,
    Fmt::Mubuf,
    Fmt::Mtbuf,
    Fmt::Mimg,
    Fmt::Exp,
    Fmt::Flat,
    Fmt::Global,
    Fmt::Scratch,
];

fn opcode_bits(fmt: Fmt) -> u32 {
    match fmt {
        Fmt::Sop2 | Fmt::Sopc | Fmt::Sopp | Fmt::Vop3p | Fmt::Flat | Fmt::Global | Fmt::Scratch => {
            7
        }
        Fmt::Sopk => 5,
        Fmt::Sop1 | Fmt::Smem | Fmt::Vop1 | Fmt::Vopc | Fmt::Ds | Fmt::Mubuf | Fmt::Mimg => 8,
        Fmt::Vop2 => 6,
        Fmt::Vop3 => 10,
        Fmt::Vintrp => 2,
        Fmt::Mtbuf => 4,
        Fmt::Exp => 0,
    }
}

#[derive(Default)]
struct Tally {
    cases: u32,
    matched: u32,
    both_invalid: u32,
    only_llvm: Vec<String>,
    llvm_fallback: u32,
    fallback_text: Vec<String>,
    only_ours: Vec<String>,
    differ: Vec<String>,
}

fn hex(words: &[u32]) -> String {
    words
        .iter()
        .map(|w| format!("{w:08x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compares one case. `words` holds enough dwords for any literal; we pass LLVM as many as we consume.
fn check(t: &mut Tally, words: &[u32], wave64: bool) {
    t.cases += 1;
    match ours(words, wave64) {
        Some((text, n)) => match llvm(&words[..n], wave64) {
            Some(l) if l.len() == 1 && l[0] == text => t.matched += 1,
            Some(l) if l.len() == 1 && !reassembles(&l[0], &words[..n], wave64) => {
                let line = format!("{} ours `{text}` llvm `{}`", hex(&words[..n]), l[0]);
                if reassembles(&text, &words[..n], wave64) {
                    t.llvm_fallback += 1;
                    t.fallback_text.push(line);
                } else {
                    t.differ.push(line);
                }
            }
            Some(l) => t.differ.push(format!(
                "{} ours `{text}` llvm `{}`",
                hex(&words[..n]),
                l.join(" | ")
            )),
            None => t
                .only_ours
                .push(format!("{} ours `{text}`", hex(&words[..n]))),
        },
        None => match llvm(words, wave64) {
            Some(l) if !l.is_empty() && !reassembles(&l[0], words, wave64) => {
                t.llvm_fallback += 1;
                t.fallback_text
                    .push(format!("{} ours invalid, llvm `{}`", hex(words), l[0]));
            }
            Some(l) if !l.is_empty() => t.only_llvm.push(format!("{} llvm `{}`", hex(words), l[0])),
            _ => t.both_invalid += 1,
        },
    }
}

/// Every `ds_swizzle_b32` offset prints as LLVM prints it.
#[test]
fn swizzle_offsets_match_llvm() {
    if !have_llvm_mc() {
        eprintln!("skipped: no llvm-mc");
        return;
    }
    let op = opinfo::by_name("ds_swizzle_b32").unwrap().code as u32;
    let words: Vec<u32> = (0..=0xFFFFu32)
        .flat_map(|off| [0xD800_0000 | op << 18 | off, 0x0A00_0001])
        .collect();
    let theirs = llvm(&words, true).expect("llvm-mc takes every offset");
    assert_eq!(theirs.len(), 0x10000);
    let differ: Vec<String> = words
        .chunks(2)
        .zip(&theirs)
        .filter_map(|(w, l)| {
            let mine = print(&decode(w).unwrap(), Syntax { wave64: true });
            (mine != *l).then(|| format!("{:#06x}: ours `{mine}` llvm `{l}`", w[0] & 0xFFFF))
        })
        .collect();
    assert!(
        differ.is_empty(),
        "{} differ: {:#?}",
        differ.len(),
        &differ[..differ.len().min(10)]
    );
}

/// Random encodings per format: known opcodes with random fields, and random opcode numbers.
#[test]
fn generated_encodings_match_llvm() {
    if !have_llvm_mc() {
        eprintln!("skipped: no llvm-mc");
        return;
    }
    let per_format: u32 = std::env::var("GFX10_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let mut cases: Vec<(Fmt, Vec<u32>, bool)> = Vec::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for fmt in FORMATS {
        let known: Vec<u32> = opinfo::all()
            .iter()
            .filter(|o| o.fmt == fmt)
            .map(|o| o.code as u32)
            .collect();
        let bits = opcode_bits(fmt);
        for i in 0..per_format {
            let code = if i % 4 == 3 || known.is_empty() {
                rng.below(1 << bits)
            } else {
                known[rng.below(known.len() as u32) as usize]
            };
            let code = if fmt == Fmt::Vop3 && i % 4 != 3 && rng.below(2) == 0 {
                rng.below(0x200)
            } else {
                code
            };
            let wave64 = rng.below(4) != 0;
            cases.push((fmt, random_inst(&mut rng, fmt, code), wave64));
        }
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()) * 2;
    let chunk = cases.len().div_ceil(threads);
    let tallies: Vec<BTreeMap<String, Tally>> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut m: BTreeMap<String, Tally> = BTreeMap::new();
                    for (fmt, words, wave64) in part {
                        check(m.entry(format!("{fmt:?}")).or_default(), words, *wave64);
                    }
                    m
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut all: BTreeMap<String, Tally> = BTreeMap::new();
    for m in tallies {
        for (k, t) in m {
            let a = all.entry(k).or_default();
            a.cases += t.cases;
            a.matched += t.matched;
            a.both_invalid += t.both_invalid;
            a.llvm_fallback += t.llvm_fallback;
            a.fallback_text.extend(t.fallback_text);
            a.only_llvm.extend(t.only_llvm);
            a.only_ours.extend(t.only_ours);
            a.differ.extend(t.differ);
        }
    }
    let mut failed = false;
    eprintln!(
        "{:8} {:>6} {:>8} {:>8} {:>9} {:>9} {:>9} {:>7}",
        "format", "cases", "matched", "invalid", "fallback", "only-llvm", "only-ours", "differ"
    );
    let mut sum = Tally::default();
    for (k, t) in &all {
        eprintln!(
            "{k:8} {:>6} {:>8} {:>8} {:>9} {:>9} {:>9} {:>7}",
            t.cases,
            t.matched,
            t.both_invalid,
            t.llvm_fallback,
            t.only_llvm.len(),
            t.only_ours.len(),
            t.differ.len()
        );
        sum.cases += t.cases;
        sum.matched += t.matched;
        sum.both_invalid += t.both_invalid;
        sum.llvm_fallback += t.llvm_fallback;
        failed |= !t.only_llvm.is_empty() || !t.differ.is_empty();
    }
    eprintln!(
        "{:8} {:>6} {:>8} {:>8} {:>9}",
        "total", sum.cases, sum.matched, sum.both_invalid, sum.llvm_fallback
    );
    let limit: usize = std::env::var("GFX10_SHOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    for (k, t) in &all {
        for line in t.differ.iter().take(limit) {
            eprintln!("{k} differ: {line}");
        }
        for line in t
            .fallback_text
            .iter()
            .take(std::env::var("GFX10_SHOW_FALLBACK").map_or(0, |_| limit))
        {
            eprintln!("{k} fallback: {line}");
        }
        for line in t.only_llvm.iter().take(limit) {
            eprintln!("{k} only llvm: {line}");
        }
        for line in t.only_ours.iter().take(limit) {
            eprintln!("{k} only ours: {line}");
        }
    }
    assert!(!failed);
}

/// A few simple encodings of opcode `code`, with plain register operands and the
/// unused fields in each of their likely shapes.
fn simple_forms(fmt: Fmt, code: u32) -> Vec<Vec<u32>> {
    let lit = 0x1234_5678;
    match fmt {
        Fmt::Sop2 => vec![vec![2 << 30 | code << 23 | 4 << 16 | 12 << 8 | 8]],
        Fmt::Sopk => vec![vec![0xB << 28 | code << 23 | 4 << 16 | 0x10, lit]],
        Fmt::Sop1 => vec![
            vec![0x17D << 23 | 4 << 16 | code << 8 | 8],
            vec![0x17D << 23 | code << 8],
        ],
        Fmt::Sopc => vec![vec![0x17E << 23 | code << 16 | 12 << 8 | 8]],
        Fmt::Sopp => vec![vec![0x17F << 23 | code << 16]],
        Fmt::Smem => vec![
            vec![0x3D << 26 | code << 18 | 8 << 6 | 2, 125 << 25 | 0x10],
            vec![0x3D << 26 | code << 18, 0],
        ],
        Fmt::Vop1 => vec![
            vec![0x3F << 25 | 1 << 17 | code << 9 | 258],
            vec![0x3F << 25 | code << 9],
        ],
        Fmt::Vop2 => vec![vec![code << 25 | 1 << 17 | 3 << 9 | 258, lit]],
        Fmt::Vopc => vec![vec![0x3E << 25 | code << 17 | 3 << 9 | 258]],
        Fmt::Vop3 => (1..=3)
            .flat_map(|n| {
                let srcs = [258u32, 259, 260];
                let mut w1 = 0;
                for (i, s) in srcs.iter().take(n).enumerate() {
                    w1 |= s << (9 * i);
                }
                [
                    vec![0x35 << 26 | code << 16 | 1, w1],
                    vec![0x35 << 26 | code << 16 | 126, w1],
                ]
            })
            .chain([
                vec![0x35 << 26 | code << 16, 0],
                vec![0x35 << 26 | code << 16 | 1, 12 << 9 | 8],
            ])
            .collect(),
        Fmt::Vop3p => (2..=3)
            .map(|n| {
                let w1 = if n == 2 {
                    259 << 9 | 258 | 3 << 27
                } else {
                    260 << 18 | 259 << 9 | 258 | 3 << 27
                };
                vec![0x33 << 26 | code << 16 | 1 << 14 | 1, w1]
            })
            .collect(),
        Fmt::Vintrp => vec![vec![0x32 << 26 | 1 << 18 | code << 16 | 2]],
        Fmt::Ds => {
            let mut v = Vec::new();
            for vdst in [0u32, 4] {
                for d0 in [0u32, 2] {
                    for d1 in [0u32, 3] {
                        for addr in [0u32, 1] {
                            for gds in [0u32, 1] {
                                v.push(vec![
                                    0x36 << 26 | code << 18 | gds << 17,
                                    vdst << 24 | d1 << 16 | d0 << 8 | addr,
                                ]);
                            }
                        }
                    }
                }
            }
            v
        }
        Fmt::Mubuf => vec![
            vec![
                0x38 << 26 | (code >> 7) << 25 | (code & 0x7F) << 18 | 1 << 12,
                3 << 24 | 2 << 16 | 2 << 8 | 1,
            ],
            vec![0x38 << 26 | (code >> 7) << 25 | (code & 0x7F) << 18, 0],
        ],
        Fmt::Mtbuf => vec![vec![
            0x3A << 26 | 1 << 19 | (code & 7) << 16 | 1 << 12,
            3 << 24 | (code >> 3) << 21 | 2 << 16 | 2 << 8 | 1,
        ]],
        Fmt::Mimg => [1u32, 0xF]
            .iter()
            .flat_map(|&dmask| {
                [0u32, 1].map(|dim| {
                    vec![
                        0x3C << 26 | (code & 0x7F) << 18 | (code >> 7) | dmask << 8 | dim << 3,
                        3 << 21 | 2 << 16 | 4 << 8 | 1,
                    ]
                })
            })
            .chain([vec![
                0x3C << 26 | (code & 0x7F) << 18 | (code >> 7) | 0xF << 8 | 1 << 12 | 1 << 15,
                2 << 16 | 4 << 8 | 1,
            ]])
            .collect(),
        Fmt::Exp => vec![vec![0x3E << 26 | 0xF, 0x0403_0201]],
        Fmt::Flat | Fmt::Global | Fmt::Scratch => {
            let seg = match fmt {
                Fmt::Flat => 0,
                Fmt::Scratch => 1,
                _ => 2,
            };
            [(4u32, 0u32), (0, 2), (4, 2)]
                .iter()
                .map(|&(vdst, data)| {
                    vec![
                        0x37 << 26 | code << 18 | seg << 14 | 1 << 16,
                        vdst << 24 | 0x7D << 16 | data << 8 | 2,
                    ]
                })
                .collect()
        }
    }
}

/// Every opcode number of every format, in simple forms: which opcodes LLVM knows
/// that we do not, and the reverse.
#[test]
#[ignore = "about a minute; run with --ignored"]
fn every_opcode_matches_llvm() {
    if !have_llvm_mc() {
        eprintln!("skipped: no llvm-mc");
        return;
    }
    let mut cases = Vec::new();
    for fmt in FORMATS {
        let limit = match fmt {
            Fmt::Sop2 => 0x60,
            Fmt::Sopk => 0x1D,
            Fmt::Vop2 => 0x3E,
            _ => 1u32 << opcode_bits(fmt),
        };
        for code in 0..limit {
            cases.push((fmt, code));
        }
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()) * 2;
    let chunk = cases.len().div_ceil(threads);
    type Found = (Vec<String>, Vec<String>, Vec<String>, u32, u32);
    let results: Vec<Found> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let (mut missing, mut extra, mut differ, mut known, mut shared) =
                        (vec![], vec![], vec![], 0, 0);
                    for &(fmt, code) in part {
                        let ours_known = opinfo::all()
                            .iter()
                            .any(|o| o.fmt == fmt && o.code as u32 == code)
                            || (fmt == Fmt::Vop3 && gfx10::decode::vop3_op(code as u16).is_some());
                        let mut llvm_known = false;
                        for words in simple_forms(fmt, code) {
                            let theirs = llvm(&words, true);
                            let mine = ours(&words, true);
                            if let Some(l) = &theirs {
                                llvm_known |= !l.is_empty();
                            }
                            if let (Some(l), Some((text, n))) = (&theirs, &mine)
                                && l.len() == 1
                                && *n <= words.len()
                                && l[0] != *text
                                && reassembles(&l[0], &words[..*n], true)
                            {
                                differ.push(format!(
                                    "{fmt:?} {code:#x}: ours `{text}` llvm `{}`",
                                    l[0]
                                ));
                            }
                        }
                        known += llvm_known as u32;
                        shared += (llvm_known && ours_known) as u32;
                        if llvm_known && !ours_known {
                            missing.push(format!("{fmt:?} {code:#x}"));
                        }
                        if ours_known && !llvm_known {
                            let name = opinfo::lookup(fmt, code as u16).map_or("?", |o| o.name);
                            extra.push(format!("{fmt:?} {code:#x} {name}"));
                        }
                    }
                    (missing, extra, differ, known, shared)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut missing, mut extra, mut differ, mut known, mut shared) =
        (vec![], vec![], vec![], 0, 0);
    for r in results {
        missing.extend(r.0);
        extra.extend(r.1);
        differ.extend(r.2);
        known += r.3;
        shared += r.4;
    }
    eprintln!("opcodes llvm-mc decodes: {known}, also in our table: {shared}");
    eprintln!("missing from ours: {missing:?}");
    eprintln!("ours, not decoded by llvm-mc in these forms: {extra:?}");
    for d in differ.iter().take(30) {
        eprintln!("differ: {d}");
    }
    assert!(missing.is_empty() && differ.is_empty());
}
