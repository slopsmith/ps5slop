//! A model of how the texture unit filters 2D linear textures with 8-bit
//! sub-texel weights: the fraction of each bilinear (and trilinear) weight
//! is rounded to 1/256 before the texels are blended exactly. Used as the
//! reference for Metal's filtering where no hardware result exists yet.
//!
//! Checked against gfx1033 (`ps5slop-hwcheck filter --hardware`): within
//! 3e-7 of every `image_sample_l` lane on RGBA8 and RGBA16F textures under
//! point, bilinear, trilinear and mip-point samplers, and bit for bit on
//! magnified draws into 8-bit targets. Assumed: implicit LOD is the textbook
//! isotropic one. Not modelled: anisotropic filtering.

use std::collections::BTreeMap;

use crate::read_pages;

#[cfg(target_os = "macos")]
pub mod report;

/// How a texture's texels are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexelFormat {
    Unorm8,
    Srgb8,
    Half,
}

impl TexelFormat {
    /// The format of `IMG_FORMAT` `code`, if the model reads it.
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            56 => Some(TexelFormat::Unorm8),
            130 => Some(TexelFormat::Srgb8),
            71 => Some(TexelFormat::Half),
            _ => None,
        }
    }

    fn bytes(self) -> u32 {
        if self == TexelFormat::Half { 8 } else { 4 }
    }
}

/// A linear 2D texture in memory: the mips from the smallest up, rows
/// padded to 256 bytes.
pub struct Texture<'a> {
    pub memory: &'a BTreeMap<u64, Vec<u8>>,
    pub address: u64,
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub format: TexelFormat,
}

impl Texture<'_> {
    /// The texture an image descriptor describes, if the model reads its format.
    pub fn from_descriptor<'a>(
        memory: &'a BTreeMap<u64, Vec<u8>>,
        desc: &[u32; 8],
    ) -> Option<Texture<'a>> {
        let address = (desc[0] as u64) << 8 | ((desc[1] & 0xFF) as u64) << 40;
        let format = TexelFormat::from_code(desc[1] >> 20 & 0x1FF)?;
        let width = (desc[1] >> 30 | (desc[2] & 0x3FFF) << 2) + 1;
        let height = (desc[2] >> 14 & 0x3FFF) + 1;
        let levels = (desc[3] >> 16 & 0xF) + 1;
        Some(Texture {
            memory,
            address,
            width,
            height,
            levels,
            format,
        })
    }

    fn size(&self, level: u32) -> (u32, u32) {
        ((self.width >> level).max(1), (self.height >> level).max(1))
    }

    fn level_start(&self, level: u32) -> u64 {
        (level + 1..self.levels)
            .map(|l| {
                let (w, h) = self.size(l);
                ((w * self.format.bytes()).next_multiple_of(256) * h) as u64
            })
            .sum()
    }

    /// Texel `(x, y)` of `level`, decoded.
    pub fn texel(&self, level: u32, x: u32, y: u32) -> [f64; 4] {
        let (w, _) = self.size(level);
        let bpp = self.format.bytes();
        let row = (w * bpp).next_multiple_of(256) as u64;
        let at = self.address + self.level_start(level) + y as u64 * row + (x * bpp) as u64;
        let mut b = [0u8; 8];
        read_pages(&[self.memory], at, &mut b[..bpp as usize]);
        match self.format {
            TexelFormat::Unorm8 => b[..4]
                .iter()
                .map(|&c| c as f64 / 255.0)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            TexelFormat::Srgb8 => {
                let lin = |c: u8| {
                    let c = c as f64 / 255.0;
                    let exact = if c <= 0.04045 {
                        c / 12.92
                    } else {
                        ((c + 0.055) / 1.055).powf(2.4)
                    };
                    significant_bits(exact, 8)
                };
                [lin(b[0]), lin(b[1]), lin(b[2]), b[3] as f64 / 255.0]
            }
            TexelFormat::Half => {
                std::array::from_fn(|i| half_to_f64(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])))
            }
        }
    }
}

/// `v` rounded to `bits` significant bits. The hardware decodes sRGB to the
/// exact curve rounded to 8 (`image_load` of every byte value on gfx1033).
fn significant_bits(v: f64, bits: i32) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let scale = 2f64.powi(bits - 1 - v.abs().log2().floor() as i32);
    (v * scale).round() / scale
}

fn half_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10 & 0x1F) as i32;
    let man = (h & 0x3FF) as f64;
    sign * match exp {
        0 => man * 2f64.powi(-24),
        31 if man == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        e => (1.0 + man / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The fields of a sampler descriptor the model uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    pub clamp: [u32; 2],
    pub mag_linear: bool,
    pub min_linear: bool,
    /// `MIP_FILTER`: 0 none, 1 point, 2 linear.
    pub mip: u32,
    pub border: u32,
    pub aniso: u32,
    pub lod_bias: f64,
    pub unnormalized: bool,
    pub compare: u32,
}

impl Sampler {
    pub fn from_words(w: &[u32; 4]) -> Sampler {
        let bias = ((w[2] & 0x3FFF) << 18) as i32 >> 18;
        Sampler {
            clamp: [w[0] & 7, w[0] >> 3 & 7],
            mag_linear: w[2] >> 20 & 1 != 0,
            min_linear: w[2] >> 22 & 1 != 0,
            mip: w[2] >> 26 & 3,
            border: w[3] >> 30,
            aniso: w[0] >> 9 & 7,
            lod_bias: bias as f64 / 256.0,
            unnormalized: w[0] >> 15 & 1 != 0,
            compare: w[0] >> 12 & 7,
        }
    }

    /// Whether the model covers this sampler.
    pub fn modelled(&self) -> bool {
        self.aniso == 0 && !self.unnormalized && self.compare == 0 && self.mip != 3
    }

    fn border_color(&self) -> [f64; 4] {
        match self.border {
            0 => [0.0; 4],
            1 => [0.0, 0.0, 0.0, 1.0],
            _ => [1.0; 4],
        }
    }
}

/// The texel index `i` of a row of `n` under `CLAMP` mode `mode`, or `None`
/// for the border colour.
fn address(i: i64, n: u32, mode: u32) -> Option<u32> {
    let n = n as i64;
    let j = match mode {
        0 => i.rem_euclid(n),
        1 => {
            let m = i.rem_euclid(2 * n);
            if m < n { m } else { 2 * n - 1 - m }
        }
        3 | 5 | 7 => (if i < 0 { -1 - i } else { i }).clamp(0, n - 1),
        6 if !(0..n).contains(&i) => return None,
        _ => i.clamp(0, n - 1),
    };
    Some(j as u32)
}

/// A texel coordinate as the texture unit keeps it: 8 fraction bits, rounded,
/// so a coordinate just below a texel boundary snaps onto it.
pub fn weight(f: f64) -> f64 {
    (f * 256.0).round() / 256.0
}

/// How weights are rounded: the hardware's 8 bits, or exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weights {
    Hardware,
    Exact,
}

impl Weights {
    fn apply(self, f: f64) -> f64 {
        match self {
            Weights::Hardware => weight(f),
            Weights::Exact => f,
        }
    }
}

fn fetch(t: &Texture, s: &Sampler, level: u32, x: i64, y: i64) -> [f64; 4] {
    let (w, h) = t.size(level);
    match (address(x, w, s.clamp[0]), address(y, h, s.clamp[1])) {
        (Some(x), Some(y)) => t.texel(level, x, y),
        _ => s.border_color(),
    }
}

/// The four texels around `(u, v)` on `level` and the weights' fractions.
fn footprint(t: &Texture, level: u32, u: f32, v: f32, weights: Weights) -> ([i64; 2], [f64; 2]) {
    let (w, h) = t.size(level);
    let x = weights.apply(u as f64 * w as f64 - 0.5);
    let y = weights.apply(v as f64 * h as f64 - 0.5);
    (
        [x.floor() as i64, y.floor() as i64],
        [x - x.floor(), y - y.floor()],
    )
}

fn sample_level(
    t: &Texture,
    s: &Sampler,
    level: u32,
    linear: bool,
    u: f32,
    v: f32,
    weights: Weights,
) -> [f64; 4] {
    let (w, h) = t.size(level);
    if !linear {
        let x = weights.apply(u as f64 * w as f64).floor() as i64;
        let y = weights.apply(v as f64 * h as f64).floor() as i64;
        return fetch(t, s, level, x, y);
    }
    let ([x, y], [fx, fy]) = footprint(t, level, u, v, weights);
    let t00 = fetch(t, s, level, x, y);
    let t10 = fetch(t, s, level, x + 1, y);
    let t01 = fetch(t, s, level, x, y + 1);
    let t11 = fetch(t, s, level, x + 1, y + 1);
    std::array::from_fn(|c| {
        (1.0 - fy) * ((1.0 - fx) * t00[c] + fx * t10[c]) + fy * ((1.0 - fx) * t01[c] + fx * t11[c])
    })
}

/// A sample at `(u, v)` with `lod` (already biased).
pub fn sample(t: &Texture, s: &Sampler, u: f32, v: f32, lod: f64, weights: Weights) -> [f64; 4] {
    let last = (t.levels - 1) as f64;
    let linear = if lod > 0.0 {
        s.min_linear
    } else {
        s.mag_linear
    };
    let lod = lod.clamp(0.0, last);
    match s.mip {
        0 => sample_level(t, s, 0, linear, u, v, weights),
        1 => sample_level(t, s, (lod + 0.5).floor() as u32, linear, u, v, weights),
        _ => {
            let l0 = lod.floor();
            let f = weights.apply(lod - l0);
            let a = sample_level(t, s, l0 as u32, linear, u, v, weights);
            if f == 0.0 {
                return a;
            }
            let b = sample_level(
                t,
                s,
                (l0 as u32 + 1).min(t.levels - 1),
                linear,
                u,
                v,
                weights,
            );
            std::array::from_fn(|c| (1.0 - f) * a[c] + f * b[c])
        }
    }
}

/// `image_gather4` of component `comp` on level 0: the four texels a
/// bilinear sample would blend, in the ISA's order.
pub fn gather(t: &Texture, s: &Sampler, u: f32, v: f32, comp: usize) -> [f64; 4] {
    let ([x, y], _) = footprint(t, 0, u, v, Weights::Hardware);
    [
        fetch(t, s, 0, x, y + 1)[comp],
        fetch(t, s, 0, x + 1, y + 1)[comp],
        fetch(t, s, 0, x + 1, y)[comp],
        fetch(t, s, 0, x, y)[comp],
    ]
}

/// The LOD of a texture of `size` texels whose coordinates change by `du`
/// and `dv` (per x and per y pixel step): log2 of the larger singular value
/// of the derivatives in texels, as RDNA2's `image_get_lod` answers.
pub fn lod(size: (u32, u32), du: [f64; 2], dv: [f64; 2]) -> f64 {
    let (w, h) = (size.0 as f64, size.1 as f64);
    let (a, b, c, d) = (du[0] * w, du[1] * w, dv[0] * h, dv[1] * h);
    let s = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    (0.5 * (s + (s * s - 4.0 * det * det).max(0.0).sqrt())).log2() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_wrap_mirror_and_clamp_as_the_clamp_modes_say() {
        assert_eq!(address(-1, 8, 0), Some(7));
        assert_eq!(address(8, 8, 1), Some(7));
        assert_eq!(address(-1, 8, 1), Some(0));
        assert_eq!(address(-3, 8, 3), Some(2));
        assert_eq!(address(9, 8, 2), Some(7));
        assert_eq!(address(8, 8, 6), None);
    }

    #[test]
    fn weights_keep_eight_bits() {
        assert_eq!(weight(0.5), 0.5);
        assert_eq!(weight(1.0 / 512.0 + 1e-9), 1.0 / 256.0);
        assert_eq!(weight(2.999), 3.0);
        assert_eq!(weight(0.3), 77.0 / 256.0);
    }

    #[test]
    fn significant_bits_round_to_nearest() {
        assert_eq!(significant_bits(0.2158605, 8), 0.2158203125);
        assert_eq!(significant_bits(1.0, 8), 1.0);
        assert_eq!(significant_bits(0.0, 8), 0.0);
    }

    #[test]
    fn half_floats_decode() {
        assert_eq!(half_to_f64(0x3C00), 1.0);
        assert_eq!(half_to_f64(0xC000), -2.0);
        assert_eq!(half_to_f64(0x0001), 2f64.powi(-24));
    }
}
