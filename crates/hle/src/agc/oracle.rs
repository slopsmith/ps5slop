//! Checks our command-buffer builders against packet bytes the real library
//! wrote on a PS5, measured by the prosper project. The measurements are not
//! ours to commit: they are read from `references/` (`scripts/fetch-references`)
//! and the test does nothing without them.

use memory::{Memory, PAGE_SIZE, Prot};

use crate::{Cpu, Env, Hle, Kernel, Outcome, Thread, TlsTemplate, Vfs};

const GOLDEN: &str =
    "../../references/src/prosper/prosper/tests/data/console_oracle/agcbuild.golden.tsv";
const DESCRIPTOR: u64 = 0x30_0000;
const BUFFER: u64 = 0x31_0000;
const BUFFER_DWORDS: u64 = 1024;

/// Builders whose output differs from the console on purpose, and why.
const KNOWN_DIFFERENT: &[(&str, &str)] = &[
    (
        "sceAgcCbBranch",
        "most of its arguments are on the stack, which held garbage when it was measured",
    ),
    (
        "sceAgcDcbResetQueue",
        "the console's reset sequence depends on the mode in ways one measurement does not show",
    ),
];

struct Case {
    id: String,
    func: String,
    args: Vec<u64>,
    expect: Vec<u8>,
}

fn parse_int(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn cases(text: &str) -> Vec<Case> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            if cols.len() < 9 || cols[5] != "ok" || cols[7] != "ptr+0" {
                return None;
            }
            let args = cols[3]
                .split(',')
                .skip(1)
                .map(|a| a.strip_prefix("i:").and_then(parse_int))
                .collect::<Option<Vec<u64>>>()?;
            let (_, data) = cols[8].strip_prefix("a0=")?.split_once(':')?;
            let expect = (0..data.len() / 2)
                .map(|i| u8::from_str_radix(&data[2 * i..2 * i + 2], 16).ok())
                .collect::<Option<Vec<u8>>>()?;
            Some(Case {
                id: cols[0].to_owned(),
                func: cols[2].to_owned(),
                args,
                expect,
            })
        })
        .collect()
}

/// Runs `case` through our builder; the bytes written to the buffer, or
/// `None` when we do not implement the builder.
fn run(case: &Case) -> Option<Vec<u8>> {
    let f = super::functions()
        .chain(super::stubs())
        .find(|f| f.name == case.func)?;
    let mem = Memory::new(1 << 40).unwrap();
    mem.map(0x10_0000, PAGE_SIZE, Prot::RW, "stack").unwrap();
    mem.map(DESCRIPTOR, PAGE_SIZE, Prot::RW, "descriptor")
        .unwrap();
    mem.map(BUFFER, 4 * PAGE_SIZE, Prot::RW, "buffer").unwrap();
    mem.write_u64(DESCRIPTOR + 0x10, BUFFER).unwrap();
    mem.write_u64(DESCRIPTOR + 0x18, BUFFER + 4 * BUFFER_DWORDS)
        .unwrap();
    let hle = Hle::new();
    hle.map(&mem).unwrap();
    let kernel = Kernel::new(
        hle,
        mem,
        TlsTemplate::default(),
        Box::new(std::io::sink()),
        Box::new(std::io::sink()),
        Vfs::default(),
    );
    let mut cpu = Cpu::default();
    cpu.regs[x86::cpu::reg::RSP] = 0x10_0000 + PAGE_SIZE - 8;
    const ORDER: [usize; 6] = [7, 6, 2, 1, 8, 9];
    cpu.regs[ORDER[0]] = DESCRIPTOR;
    for (i, &a) in case.args.iter().take(5).enumerate() {
        cpu.regs[ORDER[i + 1]] = a;
    }
    let mem = kernel.memory();
    let mut env = Env {
        cpu: &mut cpu,
        mem: &mem,
        kernel: &kernel,
        thread: Thread { tcb: 0 },
    };
    let out = (f.func)(&mut env).ok()?;
    let mut bytes = vec![0u8; case.expect.len()];
    mem.read(BUFFER, &mut bytes).unwrap();
    match out {
        Outcome::Return(r) if r == BUFFER => Some(bytes),
        _ => Some(Vec::new()),
    }
}

#[test]
fn builders_write_what_the_console_wrote() {
    let Ok(text) = std::fs::read_to_string(GOLDEN) else {
        eprintln!("skipped: {GOLDEN} not fetched");
        return;
    };
    let mut failed = Vec::new();
    let mut checked = 0;
    for case in cases(&text) {
        if KNOWN_DIFFERENT.iter().any(|(f, _)| *f == case.func) {
            continue;
        }
        let Some(ours) = run(&case) else { continue };
        checked += 1;
        if ours != case.expect {
            failed.push(format!(
                "{}: ours {ours:02x?}, console {:02x?}",
                case.id, case.expect
            ));
        }
    }
    eprintln!("checked {checked} measured builder calls");
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
