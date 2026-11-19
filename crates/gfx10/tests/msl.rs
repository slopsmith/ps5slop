//! Translates Balatro's shaders to MSL and compiles them with Metal's runtime compiler.
//! Skipped when the game files are absent.

use std::path::Path;

use gfx10::ags::Ags;
use gfx10::translate;

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

#[test]
fn balatro_shaders_translate_and_compile() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../games/Balatro/shaders");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: no {}", dir.display());
        return;
    };
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ags"))
        .collect();
    paths.sort();
    let mut failures = Vec::new();
    for path in &paths {
        let file = std::fs::read(path).unwrap();
        let ags = Ags::parse(&file).unwrap();
        let words = ags.code_words();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        match translate(&ags.request(&words)) {
            Ok(t) => {
                if let Err(e) = compile(&t.msl) {
                    failures.push(format!("{name}: Metal: {e}"));
                }
            }
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    eprintln!(
        "{} of {} shaders translate and compile",
        paths.len() - failures.len(),
        paths.len()
    );
    for f in &failures {
        eprintln!("  {}", f.lines().take(6).collect::<Vec<_>>().join("\n    "));
    }
    assert!(failures.is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn the_metal_check_rejects_bad_source() {
    assert!(compile("this is not MSL").is_err());
}
