//! What filtering in the program (`PS5SLOP_FILTER_8BIT`) costs: the MSL of
//! a sampling-heavy compute program without and with it, and the GPU time of
//! its dispatch under the current environment. Run it once without the
//! variable and once with it set.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    let samples = 256;
    let groups = 256;
    let case = hwcheck::suite::sampling_heavy(samples, groups);
    let code = hwcheck::Memory::code(&case, case.program);
    for on in [false, true] {
        let mut req = hwcheck::metal::request(&case, &code);
        req.filter_bilinear = on;
        let t = gfx10::translate::translate(&req).map_err(|e| e.message)?;
        println!(
            "MSL {} filtering in the program: {} bytes",
            if on { "with" } else { "without" },
            t.msl.len()
        );
    }
    let ms = hwcheck::metal::time(&case, 200)?;
    let n = (samples * groups * 64) as f64;
    println!(
        "PS5SLOP_FILTER_8BIT {}: {ms:.3} ms per dispatch of {n} samples, {:.2} ns per sample",
        if std::env::var_os("PS5SLOP_FILTER_8BIT").is_some() {
            "on"
        } else {
            "off"
        },
        ms * 1e6 / n
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("measuring needs macOS and Metal");
}
