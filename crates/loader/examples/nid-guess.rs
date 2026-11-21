//! Guesses names for unknown NIDs: a prefix plus up to N words in CamelCase,
//! hashed on every core. Run through `scripts/nid-guess`, which documents
//! the options.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use sha1::{Digest, Sha1};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
const SUFFIX: [u8; 16] = [
    0x51, 0x8d, 0x64, 0xa6, 0x35, 0xde, 0xd8, 0xc1, 0xe6, 0xb0, 0x39, 0xb1, 0xc3, 0xe5, 0x52, 0x30,
];
const VERBS: &[&str] = &[
    "Get",
    "Set",
    "Create",
    "Delete",
    "Open",
    "Close",
    "Init",
    "Initialize",
    "Term",
    "Terminate",
    "Is",
    "Add",
    "Remove",
    "Register",
    "Unregister",
    "Start",
    "Stop",
    "Read",
    "Write",
    "Map",
    "Unmap",
    "Allocate",
    "Release",
    "Check",
    "Wait",
    "Query",
    "Load",
    "Unload",
];

struct Options {
    prefixes: BTreeSet<String>,
    words: BTreeSet<String>,
    suffixes: Vec<String>,
    max_words: usize,
    own: Option<bool>,
    jobs: usize,
    nids: Vec<(String, u64)>,
}

fn decode(nid: &str) -> Option<u64> {
    let b = nid.as_bytes();
    if b.len() != 11 {
        return None;
    }
    let digit = |c: u8| ALPHABET.iter().position(|&a| a == c).map(|p| p as u64);
    let mut v = 0u64;
    for &c in &b[..10] {
        v = (v << 6) | digit(c)?;
    }
    Some((v << 4) | (digit(b[10])? >> 2))
}

fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|s| s.starts_with(|c: char| c.is_ascii_alphabetic()))
}

fn camel_parts(ident: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for piece in ident.split('_') {
        let c: Vec<char> = piece.chars().collect();
        let mut start = 0;
        for i in 1..c.len() {
            let boundary = (c[i].is_ascii_uppercase() && c[i - 1].is_ascii_lowercase())
                || (c[i].is_ascii_uppercase()
                    && c[i - 1].is_ascii_uppercase()
                    && c.get(i + 1).is_some_and(|n| n.is_ascii_lowercase()))
                || (c[i].is_ascii_digit() != c[i - 1].is_ascii_digit());
            if boundary {
                parts.push(c[start..i].iter().collect());
                start = i;
            }
        }
        if start < c.len() {
            parts.push(c[start..].iter().collect());
        }
    }
    parts
}

fn own_names(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut dirs = vec![root.join("crates/hle/src")];
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|x| x == "rs")
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                for quoted in text.split('"').skip(1).step_by(2) {
                    let b = quoted.as_bytes();
                    if quoted.starts_with("sce")
                        && b.get(3).is_some_and(u8::is_ascii_uppercase)
                        && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
                    {
                        names.insert(quoted.to_string());
                    }
                }
            }
        }
    }
    names
}

fn own_prefixes(names: &BTreeSet<String>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for n in names {
        let parts = camel_parts(n);
        if parts.len() < 3 || parts[0] != "sce" {
            continue;
        }
        out.insert(format!("sce{}", parts[1]));
        if let Some(v) = parts
            .iter()
            .skip(2)
            .position(|p| VERBS.contains(&p.as_str()))
        {
            out.insert(parts[..v + 2].concat());
        }
    }
    out
}

fn usage() -> ! {
    eprintln!(
        "usage: nid-guess [-p PREFIX] [-w FILE] [-s FILE] [-x SUFFIX] [-n N] [--own|--no-own] [-j N] NID..."
    );
    std::process::exit(2);
}

fn parse() -> Options {
    let mut o = Options {
        prefixes: BTreeSet::new(),
        words: BTreeSet::new(),
        suffixes: vec![String::new()],
        max_words: 2,
        own: None,
        jobs: std::thread::available_parallelism().map_or(4, |n| n.get()),
        nids: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    let read = |p: String| std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "-p" | "--prefix" => {
                o.prefixes.insert(val());
            }
            "-w" | "--words" => {
                for w in read(val()).lines().map(str::trim).filter(|w| !w.is_empty()) {
                    let mut cap = w.to_string();
                    cap[..1].make_ascii_uppercase();
                    o.words.insert(w.to_string());
                    o.words.insert(cap);
                }
            }
            "-s" | "--split" => {
                for id in identifiers(&read(val())) {
                    o.words.extend(camel_parts(id));
                }
            }
            "-x" | "--suffix" => o.suffixes.push(val()),
            "-n" | "--max-words" => o.max_words = val().parse().unwrap_or_else(|_| usage()),
            "-j" | "--jobs" => o.jobs = val().parse().unwrap_or_else(|_| usage()),
            "--no-own" => o.own = Some(false),
            "--own" => o.own = Some(true),
            "-h" | "--help" => usage(),
            nid => match decode(nid) {
                Some(v) => o.nids.push((nid.to_string(), v)),
                None => {
                    eprintln!("not a NID: {nid}");
                    usage()
                }
            },
        }
    }
    if o.nids.is_empty() {
        usage();
    }
    o
}

fn main() {
    let mut o = parse();
    let given = !o.words.is_empty();
    if o.own.unwrap_or(!given) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let names = own_names(&root);
        for n in &names {
            o.words
                .extend(camel_parts(n).into_iter().filter(|p| p != "sce"));
        }
        if o.prefixes.is_empty() {
            o.prefixes = own_prefixes(&names);
        }
    }
    if o.prefixes.is_empty() {
        o.prefixes.insert(String::new());
    }
    let words: Vec<&[u8]> = o.words.iter().map(|w| w.as_bytes()).collect();
    let prefixes: Vec<&[u8]> = o.prefixes.iter().map(|p| p.as_bytes()).collect();
    let wanted: Vec<u64> = o.nids.iter().map(|n| n.1).collect();
    let per_prefix: u128 = (1..=o.max_words as u32)
        .map(|k| (words.len() as u128).pow(k))
        .sum();
    let total = per_prefix * prefixes.len() as u128 * o.suffixes.len() as u128;
    let rate = 10_000_000 * o.jobs as u128;
    eprintln!(
        "nid-guess: {} prefixes, {} words, {total} candidates, {} threads, roughly {} s at full speed",
        prefixes.len(),
        words.len(),
        o.jobs,
        total / rate
    );

    let tasks = prefixes.len() * words.len();
    let next = AtomicUsize::new(0);
    let found: Mutex<Vec<(u64, String)>> = Mutex::new(Vec::new());
    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..o.jobs {
            s.spawn(|| {
                let mut name = Vec::with_capacity(128);
                let mut idx = vec![0usize; o.max_words];
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= tasks {
                        break;
                    }
                    if t.is_multiple_of(1000) {
                        eprint!("nid-guess: {}%\r", t * 100 / tasks);
                    }
                    let (p, first) = (prefixes[t / words.len()], words[t % words.len()]);
                    for rest in 0..o.max_words {
                        idx[..rest].fill(0);
                        loop {
                            name.clear();
                            name.extend_from_slice(p);
                            name.extend_from_slice(first);
                            for &i in &idx[..rest] {
                                name.extend_from_slice(words[i]);
                            }
                            let stem = name.len();
                            for sfx in &o.suffixes {
                                name.truncate(stem);
                                name.extend_from_slice(sfx.as_bytes());
                                let d = Sha1::new()
                                    .chain_update(&name)
                                    .chain_update(SUFFIX)
                                    .finalize();
                                let v = u64::from_le_bytes(d[..8].try_into().unwrap());
                                if wanted.contains(&v) {
                                    let n = String::from_utf8_lossy(&name).into_owned();
                                    let mut f = found.lock().unwrap();
                                    if !f.iter().any(|(_, m)| *m == n) {
                                        let nid = &o.nids.iter().find(|x| x.1 == v).unwrap().0;
                                        println!("{nid} {n}");
                                        f.push((v, n));
                                    }
                                }
                            }
                            let mut k = 0;
                            while k < rest {
                                idx[k] += 1;
                                if idx[k] < words.len() {
                                    break;
                                }
                                idx[k] = 0;
                                k += 1;
                            }
                            if k == rest {
                                break;
                            }
                        }
                    }
                }
            });
        }
    });
    eprintln!(
        "nid-guess: done in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let found = found.into_inner().unwrap();
    for (nid, v) in &o.nids {
        if !found.iter().any(|f| f.0 == *v) {
            println!("{nid} ?");
        }
    }
}
