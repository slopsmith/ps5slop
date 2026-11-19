//! Prints an `.ags` file's disassembly and its MSL translation.
//! Usage: `cargo run -p gfx10 --example ags -- file.ags`

use gfx10::ags::Ags;
use gfx10::{Syntax, decode_all, print, translate};

fn main() {
    let path = std::env::args().nth(1).expect("usage: ags FILE.ags");
    let file = std::fs::read(&path).expect("read");
    let ags = Ags::parse(&file).expect("not an .ags file");
    let words = ags.code_words();
    let (insts, _) = decode_all(&words);
    for (pc, inst) in &insts {
        println!(
            "{:#06x}: {}",
            pc * 4,
            print(
                inst,
                Syntax {
                    wave64: ags.wave64()
                }
            )
        );
    }
    match translate(&ags.request(&words)) {
        Ok(t) => {
            println!("\n// resources: {:?}", t.resources);
            println!(
                "// vertex outputs {}, colour outputs {:#x}, guest memory {}",
                t.vertex_outputs, t.color_outputs, t.guest_memory
            );
            println!("{}", t.msl);
        }
        Err(e) => println!("\ntranslation failed: {e}"),
    }
}
