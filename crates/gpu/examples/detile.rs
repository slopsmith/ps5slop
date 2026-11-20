//! Detiles a dump of guest memory into a PPM, for comparing what a title left
//! in memory with what the renderer shows.
//! Usage: `cargo run -p gpu --example detile -- IN OUT WIDTH HEIGHT MODE [LEVELS]`
//! (32-bit texels; MODE is the raw swizzle mode, e.g. 27 for 0x1b).

use gpu::tiling::{Layout, TileMode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("input");
    let (w, h): (u32, u32) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    let mode = TileMode::from_bits(a[5].parse().unwrap()).expect("mode");
    let levels: u32 = a.get(6).map_or(1, |l| l.parse().unwrap());
    let layout = Layout::new(mode, 4, w, h).expect("layout");
    let base = layout.mip0_offset(levels);
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            let o = (base + layout.offset(x, y)) as usize;
            out.extend_from_slice(data.get(o..o + 3).unwrap_or(&[0, 0, 0]));
        }
    }
    std::fs::write(&a[2], out).expect("output");
}
