//! Disassembles raw GFX10 machine code from a file, for local analysis of a
//! title's own programs (see docs/legal.md).
//! Usage: `cargo run -p gfx10 --example raw -- FILE [OFFSET]`; stops at the
//! last `s_endpgm` before an `s_code_end`, a zero word or the end of the file.

use gfx10::{Syntax, decode, print};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: raw FILE [OFFSET]");
    let offset = args.next().map_or(0, |o| {
        usize::from_str_radix(o.trim_start_matches("0x"), 16).expect("hex offset")
    });
    let bytes = std::fs::read(&path).expect("read file");
    let words: Vec<u32> = bytes[offset..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let mut pc = 0;
    while pc < words.len() && words[pc] != 0 && words[pc] != 0xBF9F_0000 {
        let Ok(inst) = decode(&words[pc..]) else {
            println!("{:#06x}: {:08x} (not decoded)", pc * 4, words[pc]);
            pc += 1;
            continue;
        };
        let text = print(&inst, Syntax { wave64: true });
        let target = match (
            inst.op.name.starts_with("s_branch") || inst.op.name.starts_with("s_cbranch"),
            words[pc] as u16,
        ) {
            (true, simm) => format!("  -> {:#06x}", (pc as i64 + 1 + simm as i16 as i64) * 4),
            _ => String::new(),
        };
        println!("{:#06x}: {text}{target}", pc * 4);
        pc += inst.words as usize;
    }
}
