//! Measurements of Metal's filtering against the model and, where kept,
//! the hardware: per image test lane counts, and per texture draw PNGs
//! and error statistics.

use std::fmt::Write as _;
use std::path::Path;

use super::{Sampler, TexelFormat, Texture, Weights, gather, lod, sample};
use crate::hardware::{Hardware, fingerprint};
use crate::suite::{
    IMAGE_DESC, TEXTURE_DRAW_BIAS, TEXTURE_DRAW_DESC, TEXTURE_DRAW_LOD, TextureDraw, all_cases,
    texture_draws,
};
use crate::{Case, read_pages, words};

fn descriptors(case: &Case, at: u64) -> ([u32; 8], [u32; 4]) {
    let mut b = [0u8; 48];
    read_pages(&[&case.input], at, &mut b);
    let w = words(&b);
    (w[..8].try_into().unwrap(), w[8..12].try_into().unwrap())
}

fn kept(case: &Case, dir: &Path, stem: &str) -> Option<Vec<u32>> {
    Hardware::load(&dir.join(format!("{stem}.hw")))
        .ok()
        .filter(|h| h.fingerprint == fingerprint(case))
        .map(|h| h.words)
}

/// How close one source of results comes to another.
#[derive(Default)]
struct Tally {
    exact: usize,
    max: f64,
}

impl Tally {
    fn add(&mut self, a: &[f32], b: &[f32]) {
        let mut same = true;
        for (&x, &y) in a.iter().zip(b) {
            if x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()) {
                continue;
            }
            same = false;
            let d = (x as f64 - y as f64).abs();
            if d.is_finite() {
                self.max = self.max.max(d);
            }
        }
        self.exact += same as usize;
    }
}

/// Compares the model (with 8-bit and with exact weights) and Metal with the
/// kept hardware results of `image_sample_l` on 2D textures, lane by lane.
pub fn validate(hardware: &Path, file_name: impl Fn(&Case) -> String) -> Result<String, String> {
    let mut out = String::from(
        "test\tlanes\tmodel exact\tmodel max error\texact-weight model exact\texact-weight max error\tMetal exact\tMetal max error\n",
    );
    for mut case in all_cases() {
        if case.opcode != "image_sample_l"
            || !case.variant.starts_with("2d ")
            || case.variant.starts_with("2d array")
            || case.variant.contains("dmask")
            || case.variant.ends_with(" unorm")
        {
            continue;
        }
        let (image, words) = descriptors(&case, IMAGE_DESC);
        let s = Sampler::from_words(&words);
        let memory = case.input.clone();
        let Some(t) = Texture::from_descriptor(&memory, &image) else {
            continue;
        };
        if !s.modelled() {
            continue;
        }
        let Some(hw) = kept(&case, hardware, &file_name(&case)) else {
            continue;
        };
        crate::metal::run(&mut case)?;
        let inputs = case.record_inputs();
        let metal = case.metal_outputs();
        let (iw, ow) = (case.in_words as usize, case.out_words as usize);
        let (mut model, mut exact, mut ours) =
            (Tally::default(), Tally::default(), Tally::default());
        let lanes = case.records() as usize;
        for i in 0..lanes {
            let rec = &inputs[i * iw..];
            let (u, v) = (f32::from_bits(rec[0]), f32::from_bits(rec[1]));
            let l = f32::from_bits(rec[2]) as f64 + s.lod_bias;
            let h: Vec<f32> = hw[i * ow..i * ow + 4]
                .iter()
                .map(|&w| f32::from_bits(w))
                .collect();
            let m: Vec<f32> = metal[i * ow..i * ow + 4]
                .iter()
                .map(|&w| f32::from_bits(w))
                .collect();
            let a = sample(&t, &s, u, v, l, Weights::Hardware).map(|c| c as f32);
            let b = sample(&t, &s, u, v, l, Weights::Exact).map(|c| c as f32);
            model.add(&a, &h);
            exact.add(&b, &h);
            ours.add(&m, &h);
        }
        writeln!(
            out,
            "{}\t{lanes}\t{}\t{:.3e}\t{}\t{:.3e}\t{}\t{:.3e}",
            case.name(),
            model.exact,
            model.max,
            exact.exact,
            exact.max,
            ours.exact,
            ours.max
        )
        .unwrap();
    }
    Ok(out)
}

/// The value the model gives pixel `(x, y)` of a texture draw.
fn model_pixel(d: &TextureDraw, t: &Texture, s: &Sampler, x: u32, y: u32) -> [f64; 4] {
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let u = py.mul_add(d.u[1], px * d.u[0]) + d.u[2];
    let v = py.mul_add(d.v[1], px * d.v[0]) + d.v[2];
    if d.op == "image_gather4" {
        return gather(t, s, u, v, 0);
    }
    let implicit = lod(
        (t.width, t.height),
        [d.u[0] as f64, d.u[1] as f64],
        [d.v[0] as f64, d.v[1] as f64],
    );
    let l = match d.op {
        "image_sample_l" => TEXTURE_DRAW_LOD as f64,
        "image_sample_b" => implicit + TEXTURE_DRAW_BIAS as f64,
        _ => implicit,
    } + s.lod_bias;
    sample(t, s, u, v, l, Weights::Hardware)
}

/// A render target's pixels as values.
fn target_pixels(words: &[u32], case: &Case, float: bool) -> Vec<[f64; 4]> {
    let (_, w, h, pitch) = case.target;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = &bytes[y * pitch as usize..];
            out.push(if float {
                std::array::from_fn(|c| {
                    let at = x * 16 + c * 4;
                    f32::from_le_bytes(p[at..at + 4].try_into().unwrap()) as f64
                })
            } else {
                std::array::from_fn(|c| p[x * 4 + c] as f64 / 255.0)
            });
        }
    }
    out
}

fn unorm8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn image(pixels: &[[f64; 4]], show: impl Fn(f64) -> f64) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|p| {
            [
                unorm8(show(p[0])),
                unorm8(show(p[1])),
                unorm8(show(p[2])),
                255,
            ]
        })
        .collect()
}

/// How far one image of a draw is from another.
#[derive(Clone, Copy, Debug)]
pub struct Compare {
    pub max: f64,
    pub mean: f64,
    /// Pixels with any component not bit for bit equal.
    pub differ: usize,
    /// Pixels whose value in an 8-bit unorm target would change.
    pub differ8: usize,
}

fn compare(a: &[[f64; 4]], b: &[[f64; 4]]) -> (Compare, Vec<[f64; 4]>) {
    let diff: Vec<[f64; 4]> = a
        .iter()
        .zip(b)
        .map(|(a, b)| std::array::from_fn(|c| (a[c] - b[c]).abs()))
        .collect();
    let errors = diff.iter().flatten().filter(|e| e.is_finite());
    let (count, sum, max) = errors.fold((0usize, 0.0, 0.0f64), |(n, s, m), &e| {
        (n + 1, s + e, m.max(e))
    });
    let differ8 = a
        .iter()
        .zip(b)
        .filter(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .any(|(&x, &y)| unorm8(x) != unorm8(y))
        })
        .count();
    let c = Compare {
        max,
        mean: sum / count.max(1) as f64,
        differ: diff.iter().filter(|p| p.iter().any(|&e| e != 0.0)).count(),
        differ8,
    };
    (c, diff)
}

/// One texture draw's measurements.
pub struct DrawStats {
    pub name: String,
    pub pixels: usize,
    pub metal_model: Option<Compare>,
    pub metal_hardware: Option<Compare>,
    pub model_hardware: Option<Compare>,
}

/// Runs every texture draw through Metal and writes, per draw, PNGs of
/// Metal's target, the model's and the kept hardware result's where there
/// are, and of each pair's difference times 64 and 4096, into `out`.
pub fn draws(
    out: &Path,
    hardware: Option<&Path>,
    file_name: impl Fn(&Case) -> String,
) -> Result<Vec<DrawStats>, String> {
    std::fs::create_dir_all(out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    let mut stats = Vec::new();
    for mut d in texture_draws() {
        let stem = file_name(&d.case).replace(".ps5itest", "");
        let hw = hardware.and_then(|dir| kept(&d.case, dir, &file_name(&d.case)));
        crate::metal::run(&mut d.case)?;
        if let Some(why) = &d.case.untranslated {
            return Err(format!("{}: {why}", d.case.name()));
        }
        let (image_desc, sampler_words) = descriptors(&d.case, TEXTURE_DRAW_DESC);
        let t = Texture::from_descriptor(&d.case.input, &image_desc)
            .ok_or("texture draws use formats the model reads")?;
        let s = Sampler::from_words(&sampler_words);
        let (_, w, h, _) = d.case.target;
        let metal = d.case.metal_checked();
        let ours = target_pixels(&metal, &d.case, d.float_target);
        let hw = hw.map(|words| {
            target_pixels(&words[words.len() - metal.len()..], &d.case, d.float_target)
        });
        let model: Option<Vec<[f64; 4]>> = s.modelled().then(|| {
            let q = |v: f64| {
                let v = v as f32 as f64;
                if d.float_target {
                    v
                } else {
                    unorm8(v) as f64 / 255.0
                }
            };
            (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| model_pixel(&d, &t, &s, x, y).map(q))
                .collect()
        });
        let show = |v: f64| {
            if t.format == TexelFormat::Half {
                (v + 1.0) / 3.0
            } else {
                v
            }
        };
        let png = |suffix: &str, rgba: &[u8]| {
            gpu::replay::write_png(&out.join(format!("{stem}.{suffix}.png")), w, h, rgba)
        };
        png("metal", &image(&ours, show))?;
        let pair = |a: &[[f64; 4]],
                    b: Option<&Vec<[f64; 4]>>,
                    name: &str|
         -> Result<Option<Compare>, String> {
            let Some(b) = b else {
                return Ok(None);
            };
            let (c, diff) = compare(a, b);
            for k in [64.0, 4096.0] {
                png(&format!("{name}.diff{k}"), &image(&diff, |v| v * k))?;
            }
            Ok(Some(c))
        };
        let metal_model = pair(&ours, model.as_ref(), "metal-model")?;
        let metal_hardware = pair(&ours, hw.as_ref(), "metal-hardware")?;
        let model_hardware = match &model {
            Some(m) => pair(m, hw.as_ref(), "model-hardware")?,
            None => None,
        };
        if let Some(m) = &model {
            png("model", &image(m, show))?;
        }
        if let Some(hw) = &hw {
            png("hardware", &image(hw, show))?;
        }
        stats.push(DrawStats {
            name: d.case.name(),
            pixels: ours.len(),
            metal_model,
            metal_hardware,
            model_hardware,
        });
    }
    Ok(stats)
}
