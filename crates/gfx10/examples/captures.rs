//! Translates every program a title ran, from the files `PS5SLOP_SHADER_DUMP`
//! wrote, and counts why the others fail.
//! Usage: `cargo run -p gfx10 --example captures -- DIR [NAME]`; with NAME,
//! prints that program's disassembly and translation.

use std::collections::BTreeMap;

use gfx10::capture::Capture;
use gfx10::{Syntax, print, translate};

#[cfg(target_os = "macos")]
fn compile(msl: &str) -> Result<(), String> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
    device
        .newLibraryWithSource_options_error(&NSString::from_str(msl), None)
        .map(|_| ())
        .map_err(|e| e.localizedDescription().to_string())
}

#[cfg(not(target_os = "macos"))]
fn compile(_: &str) -> Result<(), String> {
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: captures DIR [NAME]");
    let only = args.next();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    paths.sort();
    let mut reasons: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ok = 0;
    for path in &paths {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if only.as_ref().is_some_and(|o| !name.contains(o.as_str())) {
            continue;
        }
        let Some(c) = Capture::from_bytes(&std::fs::read(path).unwrap()) else {
            continue;
        };
        let req = c.request();
        let result = translate(&req);
        if only.is_some() {
            if let Ok(insts) = gfx10::translate::program(&c.code) {
                for (pc, inst) in &insts {
                    println!("{:#06x}: {}", pc, print(inst, Syntax { wave64: true }));
                }
            }
            println!("// {c:x?}");
            match &result {
                Ok(t) => println!("// {:?}\n{}", t.resources, t.msl),
                Err(e) => println!("failed: {e}"),
            }
        }
        let failure = match result {
            Ok(t) => compile(&t.msl).err().map(|e| format!("Metal: {}", e.lines().find(|l| l.contains("error")).unwrap_or(&e))),
            Err(e) => Some(e.message.clone()),
        };
        match failure {
            None => ok += 1,
            Some(why) => reasons.entry(why).or_default().push(name),
        }
    }
    let mut sorted: Vec<_> = reasons.into_iter().collect();
    sorted.sort_by_key(|r| std::cmp::Reverse(r.1.len()));
    println!("{ok} of {} translate", ok + sorted.iter().map(|r| r.1.len()).sum::<usize>());
    for (why, names) in sorted {
        println!("{:4} {why}: {}", names.len(), names.iter().take(4).cloned().collect::<Vec<_>>().join(" "));
    }
}
