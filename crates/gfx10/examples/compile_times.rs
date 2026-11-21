//! Times Metal's compiler on the programs a title ran, from the files
//! `PS5SLOP_SHADER_DUMP` wrote (or `.metal` files). Each source gets a
//! unique trailing comment, so Metal's shader cache never answers, unless
//! `--warm` is given. Programs compile in safe math mode, as the renderer
//! compiles them, unless `--fast` is given.
//! Usage: `cargo run --release -p gfx10 --example compile_times -- DIR
//! [--threads N] [--size] [--fast] [--strip-comments] [--min-bytes N] [--warm]
//! [--dump OUT]`; `--dump` writes each translation to OUT instead.

#[cfg(target_os = "macos")]
fn main() {
    use gfx10::capture::Capture;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibraryOptimizationLevel,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: compile_times DIR [options]");
    let (mut threads, mut size, mut strip, mut min_bytes) = (1usize, false, false, 0usize);
    let mut dump = None;
    let mut warm = false;
    let mut fast = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = args.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "--size" => size = true,
            "--strip-comments" => strip = true,
            "--dump" => dump = args.next(),
            "--warm" => warm = true,
            "--fast" => fast = true,
            "--min-bytes" => min_bytes = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            _ => panic!("unknown option {a}"),
        }
    }
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    paths.sort();
    let mut jobs = Vec::new();
    for path in &paths {
        let mut msl = if path.extension().is_some_and(|e| e == "metal") {
            std::fs::read_to_string(path).unwrap()
        } else {
            let Some(c) = Capture::from_bytes(&std::fs::read(path).unwrap()) else {
                continue;
            };
            let Ok(t) = gfx10::translate(&c.request()) else {
                continue;
            };
            t.msl
        };
        if strip {
            msl = msl
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .map(|l| format!("{l}\n"))
                .collect();
        }
        if let Some(d) = &dump {
            let name = path.file_stem().unwrap().to_string_lossy();
            std::fs::write(format!("{d}/{name}.metal"), &msl).unwrap();
            continue;
        }
        if msl.len() >= min_bytes {
            jobs.push((path.file_name().unwrap().to_string_lossy().to_string(), msl));
        }
    }
    let total_bytes: usize = jobs.iter().map(|j| j.1.len()).sum();
    let jobs = Arc::new(Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>()));
    let results = Arc::new(Mutex::new(Vec::new()));
    let start = Instant::now();
    let workers: Vec<_> = (0..threads)
        .map(|_| {
            let (jobs, results) = (jobs.clone(), results.clone());
            std::thread::spawn(move || {
                let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
                let options = MTLCompileOptions::new();
                if size {
                    options.setOptimizationLevel(MTLLibraryOptimizationLevel::Size);
                }
                if !fast {
                    options.setMathMode(objc2_metal::MTLMathMode::Safe);
                }
                loop {
                    let Some((i, (name, msl))) = jobs.lock().unwrap().pop() else {
                        break;
                    };
                    let src = if warm {
                        msl.clone()
                    } else {
                        format!("{msl}\n// {i} {:?}\n", std::time::SystemTime::now())
                    };
                    let t = Instant::now();
                    let ok = device
                        .newLibraryWithSource_options_error(
                            &NSString::from_str(&src),
                            Some(&options),
                        )
                        .is_ok();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    results.lock().unwrap().push((name, msl.len(), ms, ok));
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    let wall = start.elapsed().as_secs_f64();
    let mut r = results.lock().unwrap().clone();
    r.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (name, bytes, ms, ok) in r.iter().take(10) {
        println!(
            "{ms:8.1} ms {bytes:8} bytes {name}{}",
            if *ok { "" } else { " FAILED" }
        );
    }
    let sum: f64 = r.iter().map(|x| x.2).sum();
    println!(
        "{} programs, {} KB of MSL, {threads} threads: wall {wall:.2} s, compile time summed {:.2} s, {} failed",
        r.len(),
        total_bytes / 1000,
        sum / 1e3,
        r.iter().filter(|x| !x.3).count()
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {}
