//! The hardware oracle for GFX10 instructions: generates single-instruction
//! test programs and runs them through ps5slop's Metal path (on the Mac), then
//! runs the same programs on a real AMD GPU (on Linux) and compares.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use hwcheck::Case;

/// Checks ps5slop's GFX10 instruction translation against real AMD hardware.
#[derive(Parser)]
#[command(name = "ps5slop-hwcheck")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generates the instruction tests, runs each through ps5slop's Metal
    /// path and writes them with Metal's results into DIR (macOS only).
    Gen {
        /// Directory for the test files (created if missing).
        dir: PathBuf,
        /// Only opcodes whose name contains this text (repeat for several).
        #[arg(long)]
        filter: Vec<String>,
    },
    /// Runs test files on the AMD GPU and compares the results with Metal's
    /// (Linux with amdgpu only).
    Run {
        /// Test files, or directories of them.
        #[arg(required = true)]
        tests: Vec<PathBuf>,
        /// The amdgpu render node.
        #[arg(long, default_value = "/dev/dri/renderD128")]
        device: PathBuf,
        /// The GPU's ISA (e.g. gfx1033); read from the kernel when left out.
        #[arg(long)]
        cpu: Option<String>,
        /// How many differing lanes to print per test.
        #[arg(long, default_value_t = 4)]
        show: usize,
        /// Seconds to wait for a dispatch before calling it hung.
        #[arg(long, default_value_t = 10)]
        timeout: u64,
        /// Writes one line per test (opcode, variant, outcome) to this file.
        #[arg(long)]
        results: Option<PathBuf>,
        /// Writes the hardware's and Metal's render targets of draws that
        /// differ into this directory, as PPM images.
        #[arg(long)]
        images: Option<PathBuf>,
        /// Keeps every test's hardware result in this directory, for `check`.
        #[arg(long)]
        save: Option<PathBuf>,
    },
    /// Generates tests, runs them through ps5slop's Metal path and compares
    /// them with hardware results kept by `run --save`, without the GPU.
    Check {
        /// Directory of kept hardware results.
        #[arg(long)]
        hardware: PathBuf,
        /// Only opcodes whose name contains this text (repeat for several).
        #[arg(long)]
        filter: Vec<String>,
        /// How many differing records to print per test.
        #[arg(long, default_value_t = 4)]
        show: usize,
        /// Merges the outcomes into this results file (as `run --results` writes them).
        #[arg(long)]
        merge: Option<PathBuf>,
        /// Writes the hardware's and Metal's render targets of draws that
        /// differ into this directory, as PPM images and raw bytes.
        #[arg(long)]
        images: Option<PathBuf>,
    },
    /// Lists, per encoding format, which opcodes the translator takes and which it refuses.
    Coverage,
    /// Writes the instruction status page (HTML) from the translator and a results file.
    Chart {
        /// Results written by `run --results`.
        results: PathBuf,
        /// The HTML page to write.
        output: PathBuf,
        /// Where the results come from, shown on the page (GPU, date, commit).
        #[arg(long, default_value = "")]
        source: String,
        /// The GPU the results come from, to tell which forms it cannot test.
        #[arg(long, default_value = "gfx1033")]
        cpu: String,
    },
    /// Measures Metal's texture filtering against a model of the hardware's
    /// 8-bit weights, or against kept hardware results where there are
    /// some: lane counts for the image tests, and PNGs and error
    /// statistics for the `draw texture` tests (macOS only).
    Filter {
        /// Directory for the PNGs and `filter.tsv` (created if missing).
        out: PathBuf,
        /// Directory of kept hardware results.
        #[arg(long)]
        hardware: Option<PathBuf>,
    },
    /// Prints what test files hold.
    Info {
        #[arg(required = true)]
        tests: Vec<PathBuf>,
    },
}

/// Test files named by `paths`, directories expanded, sorted.
fn test_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let entries =
                std::fs::read_dir(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
            for e in entries.flatten() {
                if e.path().extension().is_some_and(|x| x == "ps5itest") {
                    out.push(e.path());
                }
            }
        } else {
            out.push(p.clone());
        }
    }
    out.sort();
    Ok(out)
}

/// A render target of 4-byte pixels as an RGB image, from its dwords.
fn write_ppm(path: &Path, case: &Case, dwords: &[u32]) -> Result<(), String> {
    let (_, w, h, pitch) = case.target;
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = dwords.get(y * pitch as usize / 4 + x).copied().unwrap_or(0);
            out.extend(&p.to_le_bytes()[..3]);
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// The hardware's result for `case`: the kept one when it is for this
/// version of the test, otherwise a fresh run.
#[cfg(target_os = "linux")]
fn rerun(
    device: &hwcheck::amdgpu::Device,
    case: &Case,
    kept: &Option<hwcheck::hardware::Hardware>,
    timeout: u64,
) -> Result<Option<Vec<u32>>, String> {
    if let Some(h) = kept {
        return Ok(Some(h.words.clone()));
    }
    hwcheck::run::run(device, case, std::time::Duration::from_secs(timeout))
}

/// Where `run` writes what it found besides printing it.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct Outputs<'a> {
    results: Option<&'a Path>,
    images: Option<&'a Path>,
    save: Option<&'a Path>,
}

/// The hardware's and Metal's render targets of a draw as PPM images and
/// as the raw bytes of the checked memory.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn write_images(dir: &Path, case: &Case, hw: &[u32]) -> Result<(), String> {
    let metal = case.metal_checked();
    let hw = &hw[hw.len().saturating_sub(metal.len())..];
    let stem = file_name(case);
    write_ppm(&dir.join(format!("{stem}.hardware.ppm")), case, hw)?;
    write_ppm(&dir.join(format!("{stem}.metal.ppm")), case, &metal)?;
    for (kind, words) in [("hardware", hw), ("metal", &metal[..])] {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let path = dir.join(format!("{stem}.{kind}.bin"));
        std::fs::write(&path, bytes)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn check(
    hardware: &Path,
    filter: &[String],
    show: usize,
    merge: Option<&Path>,
    images: Option<&Path>,
) -> Result<bool, String> {
    use hwcheck::hardware::{Hardware, fingerprint};
    use std::fmt::Write as _;
    let cases: Vec<Case> = hwcheck::suite::all_cases()
        .into_iter()
        .filter(|c| filter.is_empty() || filter.iter().any(|f| c.opcode.contains(f.as_str())))
        .collect();
    let (mut agree, mut differ, mut skipped) = (0, 0, 0);
    let mut log = String::new();
    for mut case in cases {
        let name = case.name();
        let fp = fingerprint(&case);
        hwcheck::metal::run(&mut case)?;
        let path = hardware.join(format!("{}.hw", file_name(&case)));
        let outcome = if let Some(why) = &case.untranslated {
            skipped += 1;
            format!("untranslated\t{why}")
        } else {
            match Hardware::load(&path) {
                Err(_) => {
                    skipped += 1;
                    "unchecked\tno hardware result kept".to_string()
                }
                Ok(h) if h.fingerprint != fp => {
                    skipped += 1;
                    println!(
                        "{name}: the kept hardware result is for an older version of the test"
                    );
                    "unchecked\tthe kept hardware result is stale".to_string()
                }
                Ok(h) => {
                    let fused = if h.fused { "\tfused on this GPU" } else { "" };
                    let o = case.compare(&h.words);
                    if o.agrees() {
                        agree += 1;
                        format!("agree{fused}")
                    } else {
                        differ += 1;
                        println!("{name}: {}\n  {}", case.text, o.report(show).trim_end());
                        if let Some(dir) = images.filter(|_| case.graphics) {
                            write_images(dir, &case, &h.words)?;
                        }
                        let memory = if o.memory.is_empty() {
                            String::new()
                        } else {
                            format!("; {} stored dwords value", o.memory.len())
                        };
                        format!(
                            "differ\t{} of {}: {}{memory}{fused}",
                            o.diffs.len(),
                            o.records,
                            o.kinds()
                        )
                    }
                }
            }
        };
        let _ = writeln!(log, "{}\t{}\t{outcome}", case.opcode, case.variant);
        for op in &case.covers {
            let _ = writeln!(log, "{op}\t{}: {}\t{outcome}", case.opcode, case.variant);
        }
    }
    println!("{agree} agree, {differ} differ, {skipped} not compared");
    if let Some(path) = merge {
        let kept = std::fs::read_to_string(path).unwrap_or_default();
        std::fs::write(path, hwcheck::merge_results(&kept, &log))
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(differ == 0)
}

#[cfg(not(target_os = "macos"))]
fn check(
    _: &Path,
    _: &[String],
    _: usize,
    _: Option<&Path>,
    _: Option<&Path>,
) -> Result<bool, String> {
    Err("checking needs macOS and Metal".into())
}

fn file_name(case: &Case) -> String {
    let variant: String = case
        .variant
        .replace('+', "plus")
        .replace('-', "minus")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("{}.{variant}.ps5itest", case.opcode)
}

#[cfg(target_os = "macos")]
fn generate(dir: &Path, filter: &[String]) -> Result<bool, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let cases: Vec<Case> = hwcheck::suite::all_cases()
        .into_iter()
        .filter(|c| filter.is_empty() || filter.iter().any(|f| c.opcode.contains(f.as_str())))
        .collect();
    let mut untranslated = 0;
    for (i, mut case) in cases.into_iter().enumerate() {
        hwcheck::metal::run(&mut case)?;
        if let Some(why) = &case.untranslated {
            untranslated += 1;
            println!("{}: not translated: {why}", case.name());
        }
        case.save(&dir.join(file_name(&case)))?;
        if (i + 1) % 100 == 0 {
            println!("{} tests written", i + 1);
        }
    }
    println!("done; {untranslated} not translated");
    Ok(true)
}

#[cfg(target_os = "macos")]
fn filter(out: &Path, hardware: Option<&Path>) -> Result<bool, String> {
    use std::fmt::Write as _;
    if let Some(dir) = hardware {
        println!("{}", hwcheck::filter::report::validate(dir, file_name)?);
    }
    let stats = hwcheck::filter::report::draws(out, hardware, file_name)?;
    let mut tsv = String::from("test\tpixels");
    for pair in ["Metal-model", "Metal-hardware", "model-hardware"] {
        for column in [
            "max error",
            "mean error",
            "pixels differing",
            "pixels differing at 8 bits",
        ] {
            write!(tsv, "\t{pair} {column}").unwrap();
        }
    }
    tsv.push('\n');
    for s in &stats {
        write!(tsv, "{}\t{}", s.name, s.pixels).unwrap();
        for c in [s.metal_model, s.metal_hardware, s.model_hardware] {
            match c {
                Some(c) => write!(
                    tsv,
                    "\t{:.3e}\t{:.3e}\t{}\t{}",
                    c.max, c.mean, c.differ, c.differ8
                ),
                None => write!(tsv, "\t-\t-\t-\t-"),
            }
            .unwrap();
        }
        tsv.push('\n');
    }
    let path = out.join("filter.tsv");
    std::fs::write(&path, &tsv).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    print!("{tsv}");
    println!("wrote {} and the PNGs beside it", path.display());
    Ok(true)
}

#[cfg(not(target_os = "macos"))]
fn filter(_: &Path, _: Option<&Path>) -> Result<bool, String> {
    Err("measuring filtering needs macOS and Metal".into())
}

#[cfg(not(target_os = "macos"))]
fn generate(_: &Path, _: &[String]) -> Result<bool, String> {
    Err("generating tests needs macOS and Metal".into())
}

#[cfg(target_os = "linux")]
fn gpu_target() -> Option<String> {
    let nodes = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes").ok()?;
    for node in nodes.flatten() {
        let props = std::fs::read_to_string(node.path().join("properties")).ok()?;
        let version = props
            .lines()
            .find_map(|l| l.strip_prefix("gfx_target_version "))?
            .trim()
            .parse::<u32>()
            .ok()?;
        if version != 0 {
            let (major, minor, step) = (version / 10000, version / 100 % 100, version % 100);
            return Some(format!("gfx{major}{minor:x}{step:x}"));
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn run(
    tests: &[PathBuf],
    device: &Path,
    cpu: Option<String>,
    show: usize,
    timeout: u64,
    outputs: Outputs,
) -> Result<bool, String> {
    use std::fmt::Write as _;
    let Outputs {
        results,
        images,
        save,
    } = outputs;
    if let Some(dir) = save {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let device = hwcheck::amdgpu::Device::open(device)?;
    let cpu = cpu
        .or_else(gpu_target)
        .ok_or("cannot tell which GPU this is; pass --cpu")?;
    println!("on {} ({cpu})", device.name());
    let (mut agree, mut differ, mut skipped) = (0, 0, 0);
    let mut log = String::new();
    for path in test_files(tests)? {
        let mut case = Case::load(&path)?;
        let name = case.name();
        let fingerprint = hwcheck::hardware::fingerprint(&case);
        let stem = file_name(&case);
        let kept = save
            .and_then(|dir| hwcheck::hardware::Hardware::load(&dir.join(format!("{stem}.hw"))).ok())
            .filter(|h| h.fingerprint == fingerprint);
        let keep = |hw: &[u32], fused: bool| -> Result<(), String> {
            let Some(dir) = save else {
                return Ok(());
            };
            hwcheck::hardware::Hardware {
                fingerprint,
                gpu: cpu.clone(),
                fused,
                words: hw.to_vec(),
            }
            .save(&dir.join(format!("{stem}.hw")))
        };
        let outcome = if let Some(why) = case.untranslated.clone() {
            skipped += 1;
            if save.is_some() {
                let mut ported = case.clone();
                let program = ported.program;
                let port = if ported.graphics {
                    Some(0)
                } else {
                    hwcheck::compat::port_program(&mut ported, program, &cpu)
                        .ok()
                        .filter(|p| p.differences.is_empty())
                        .map(|p| p.fused)
                };
                if let Some(fused) = port
                    && kept.is_none()
                    && let Some(hw) = hwcheck::run::run(
                        &device,
                        &ported,
                        std::time::Duration::from_secs(timeout),
                    )?
                {
                    keep(&hw, fused > 0)?;
                }
            }
            format!("untranslated\t{why}")
        } else if case.graphics {
            match rerun(&device, &case, &kept, timeout)? {
                None => {
                    differ += 1;
                    println!("{name}: did not finish in {timeout} s");
                    "hung".to_string()
                }
                Some(hw) => {
                    keep(&hw, false)?;
                    let o = case.compare(&hw);
                    if o.agrees() {
                        agree += 1;
                        "agree".to_string()
                    } else {
                        differ += 1;
                        println!("{name}: {}\n  {}", case.text, o.report(show).trim_end());
                        if let Some(dir) = images {
                            write_images(dir, &case, &hw)?;
                        }
                        format!("differ\t{} target dwords value", o.memory.len())
                    }
                }
            }
        } else {
            let program = case.program;
            match hwcheck::compat::port_program(&mut case, program, &cpu) {
                Err(e) => {
                    skipped += 1;
                    println!("{name}: cannot run on {cpu}: {e}");
                    format!("unrunnable\t{e}")
                }
                Ok(p) if !p.differences.is_empty() => {
                    skipped += 1;
                    let (ps5, other) = &p.differences[0];
                    println!("{name}: {cpu} reads it differently ({ps5} / {other})");
                    format!("unrunnable\t{cpu} reads {ps5} as {other}")
                }
                Ok(p) => {
                    let fused = if p.fused > 0 {
                        "\tfused on this GPU"
                    } else {
                        ""
                    };
                    match rerun(&device, &case, &kept, timeout)? {
                        None => {
                            differ += 1;
                            println!("{name}: did not finish in {timeout} s");
                            format!("hung{fused}")
                        }
                        Some(hw) => {
                            keep(&hw, p.fused > 0)?;
                            let o = case.compare(&hw);
                            if o.agrees() {
                                agree += 1;
                                format!("agree{fused}")
                            } else {
                                differ += 1;
                                println!("{name}: {}\n  {}", case.text, o.report(show).trim_end());
                                let memory = if o.memory.is_empty() {
                                    String::new()
                                } else {
                                    format!("; {} stored dwords value", o.memory.len())
                                };
                                format!(
                                    "differ\t{} of {}: {}{memory}{fused}",
                                    o.diffs.len(),
                                    o.records,
                                    o.kinds()
                                )
                            }
                        }
                    }
                }
            }
        };
        let _ = writeln!(log, "{}\t{}\t{outcome}", case.opcode, case.variant);
        for op in &case.covers {
            let _ = writeln!(log, "{op}\t{}: {}\t{outcome}", case.opcode, case.variant);
        }
    }
    println!("{agree} agree, {differ} differ, {skipped} not run");
    if let Some(path) = results {
        std::fs::write(path, log).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(differ == 0)
}

#[cfg(not(target_os = "linux"))]
fn run(
    _: &[PathBuf],
    _: &Path,
    _: Option<String>,
    _: usize,
    _: u64,
    _: Outputs,
) -> Result<bool, String> {
    Err("running tests needs Linux with an amdgpu device".into())
}

fn info(tests: &[PathBuf]) -> Result<bool, String> {
    for path in test_files(tests)? {
        let case = Case::load(&path)?;
        println!("{}: {}", case.name(), case.text);
        if let Some(why) = &case.untranslated {
            println!("  not translated: {why}");
        }
        println!(
            "  {} records, RSRC1 {:#010x}, Metal changed {} pages",
            case.records(),
            case.reg(0x212),
            case.output.len()
        );
    }
    Ok(true)
}

fn coverage() -> Result<bool, String> {
    use hwcheck::coverage::Coverage;
    let mut by_fmt: std::collections::BTreeMap<String, (usize, Vec<&str>, Vec<&str>)> =
        Default::default();
    for (op, c) in hwcheck::coverage::scan() {
        let e = by_fmt.entry(format!("{:?}", op.fmt)).or_default();
        match c {
            Coverage::Translated => e.0 += 1,
            Coverage::Untranslated(_) => e.1.push(op.name),
            Coverage::Unknown => e.2.push(op.name),
        }
    }
    for (fmt, (done, missing, unknown)) in by_fmt {
        let total = done + missing.len() + unknown.len();
        println!("{fmt}: {done} of {total} translated");
        if !missing.is_empty() {
            println!("  not translated: {}", missing.join(" "));
        }
        if !unknown.is_empty() {
            println!("  not scanned: {}", unknown.join(" "));
        }
    }
    Ok(true)
}

fn chart(results: &Path, output: &Path, source: &str, cpu: &str) -> Result<bool, String> {
    let text = std::fs::read_to_string(results)
        .map_err(|e| format!("cannot read {}: {e}", results.display()))?;
    let forms = hwcheck::chart::forms(&text, cpu);
    std::fs::write(output, hwcheck::chart::html(&forms, source))
        .map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    println!("wrote {}", output.display());
    Ok(true)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Gen { dir, filter } => generate(&dir, &filter),
        Command::Run {
            tests,
            device,
            cpu,
            show,
            timeout,
            results,
            images,
            save,
        } => run(
            &tests,
            &device,
            cpu,
            show,
            timeout,
            Outputs {
                results: results.as_deref(),
                images: images.as_deref(),
                save: save.as_deref(),
            },
        ),
        Command::Check {
            hardware,
            filter,
            show,
            merge,
            images,
        } => check(
            &hardware,
            &filter,
            show,
            merge.as_deref(),
            images.as_deref(),
        ),
        Command::Filter { out, hardware } => filter(&out, hardware.as_deref()),
        Command::Info { tests } => info(&tests),
        Command::Coverage => coverage(),
        Command::Chart {
            results,
            output,
            source,
            cpu,
        } => chart(&results, &output, &source, &cpu),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
