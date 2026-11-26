//! GPU formats and states as Metal's.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Number {
    Float,
    Uint,
    Sint,
}

/// Whether Metal format `f` holds floats (including normalised) or integers.
pub(super) fn number(f: MTLPixelFormat) -> Number {
    use MTLPixelFormat as P;
    if [
        P::R8Uint,
        P::R16Uint,
        P::RG8Uint,
        P::R32Uint,
        P::RG16Uint,
        P::RGBA8Uint,
        P::RGB10A2Uint,
        P::RG32Uint,
        P::RGBA16Uint,
        P::RGBA32Uint,
    ]
    .contains(&f)
    {
        Number::Uint
    } else if [
        P::R8Sint,
        P::R16Sint,
        P::RG8Sint,
        P::R32Sint,
        P::RG16Sint,
        P::RGBA8Sint,
        P::RG32Sint,
        P::RGBA16Sint,
        P::RGBA32Sint,
    ]
    .contains(&f)
    {
        Number::Sint
    } else {
        Number::Float
    }
}

pub(super) fn write_mask(bits: u32) -> usize {
    let (r, g, b, a) = (bits & 1, (bits >> 1) & 1, (bits >> 2) & 1, (bits >> 3) & 1);
    let m = |on: u32, bit: MTLColorWriteMask| if on != 0 { bit.0 } else { 0 };
    m(r, MTLColorWriteMask::Red)
        | m(g, MTLColorWriteMask::Green)
        | m(b, MTLColorWriteMask::Blue)
        | m(a, MTLColorWriteMask::Alpha)
}

/// Whether a color target is drawn through unsigned integers of its pixel's
/// size, packed by the pixel program (`Request::packed_targets`): formats
/// Metal lacks, and ones whose Metal format stores other bits. AMD names list
/// the channels from the top bit down, so of the packed 32-bit layouts only
/// 10_11_11 floats and 2_10_10_10 unorm are Metal's (Metal's RGB10A2Uint and
/// B5G6R5Unorm cannot be read back), and Metal's snorm formats store -1 as
/// -127 where the CB keeps the -128 already in memory.
pub(super) fn packed_target(f: regs::ColorFormat) -> bool {
    gfx10::translate::packed_widths(f.layout).is_some()
        && (direct_color_format(f).is_none()
            || f.number == 1
            || matches!(f.layout, 6..=9) && !matches!((f.layout, f.number), (6, 7) | (9, 0)))
}

/// The unsigned integer format a packed target is drawn through.
fn packed_format(f: regs::ColorFormat) -> Option<MTLPixelFormat> {
    use MTLPixelFormat as P;
    let bits: u32 = gfx10::translate::packed_widths(f.layout)?.iter().sum();
    Some(match bits {
        8 => P::R8Uint,
        16 => P::R16Uint,
        32 => P::R32Uint,
        64 => P::RG32Uint,
        _ => P::RGBA32Uint,
    })
}

/// A packed target's first two dwords for a clear colour of 0s and 1s, as
/// the pixel program's `cb_pack` makes them.
pub(super) fn packed_word(f: regs::ColorFormat, colour: [f64; 4]) -> [u32; 2] {
    let widths = gfx10::translate::packed_widths(f.layout).unwrap_or([0; 4]);
    let swizzle = export_swizzle(f);
    let mut at = 0;
    let mut words = [0u64; 2];
    for (c, w) in widths.into_iter().enumerate() {
        let x = colour[(swizzle >> (2 * c) & 3) as usize].clamp(0.0, 1.0);
        let max = ((1u64 << w) - 1) as f64;
        let field = match f.number {
            1 => (x * (max / 2.0).floor()).round_ties_even() as u64,
            6 if c < 3 => (x * 255.0).round_ties_even() as u64,
            7 if w == 32 => u64::from((x as f32).to_bits()),
            7 if w == 16 && x > 0.0 => 0x3C00,
            7 if w >= 10 && x > 0.0 => 15 << (w - 5),
            4 | 5 | 7 => x as u64,
            _ => (x * max).round_ties_even() as u64,
        };
        if at < 64 {
            words[at / 32] |= (field & max as u64) << (at % 32);
        }
        at += w as usize;
    }
    [words[0] as u32, words[1] as u32]
}

/// The Metal format of a color target.
pub(super) fn color_format(f: regs::ColorFormat) -> Option<MTLPixelFormat> {
    if packed_target(f) {
        packed_format(f)
    } else {
        direct_color_format(f)
    }
}

/// The Metal format that holds a color target's bits as the CB writes them.
fn direct_color_format(f: regs::ColorFormat) -> Option<MTLPixelFormat> {
    use MTLPixelFormat as P;
    let bgr = f.swap == 1;
    Some(match (f.layout, f.number) {
        (10, 6) if bgr => P::BGRA8Unorm_sRGB,
        (10, 6) => P::RGBA8Unorm_sRGB,
        (10, 0) if bgr => P::BGRA8Unorm,
        (10, 0) => P::RGBA8Unorm,
        (10, 1) => P::RGBA8Snorm,
        (10, 4) => P::RGBA8Uint,
        (10, 5) => P::RGBA8Sint,
        (1, 0) => P::R8Unorm,
        (1, 4) => P::R8Uint,
        (2, 0) => P::R16Unorm,
        (2, 4) => P::R16Uint,
        (2, 7) => P::R16Float,
        (3, 0) => P::RG8Unorm,
        (3, 4) => P::RG8Uint,
        (4, 4) => P::R32Uint,
        (4, 5) => P::R32Sint,
        (4, 7) => P::R32Float,
        (5, 0) => P::RG16Unorm,
        (5, 4) => P::RG16Uint,
        (5, 7) => P::RG16Float,
        (6, _) => P::RG11B10Float,
        (9, _) if bgr => P::BGR10A2Unorm,
        (9, _) => P::RGB10A2Unorm,
        (11, 4) => P::RG32Uint,
        (11, 7) => P::RG32Float,
        (12, 0) => P::RGBA16Unorm,
        (12, 4) => P::RGBA16Uint,
        (12, 7) => P::RGBA16Float,
        (14, 4) => P::RGBA32Uint,
        (14, 7) => P::RGBA32Float,
        _ => return None,
    })
}

/// The mip levels read of a sampled image: its whole chain, as far as
/// Metal allows for its size; level 0 alone for views from a later level.
pub(super) fn sampled_levels(image: Image) -> u32 {
    if image.base_level != 0 {
        return 1;
    }
    let full = 32 - image.width.max(image.height).leading_zeros();
    (image.last_level + 1).clamp(1, full)
}

/// The mip levels of a 3D image's texture: its chain, as far as Metal
/// allows for its size.
pub(super) fn volume_levels(image: Image) -> u32 {
    let full = 32
        - image
            .width
            .max(image.height)
            .max(image.depth)
            .leading_zeros();
    (image.last_level + 1).clamp(1, full)
}

/// A 3D image of one texel, whose layout cannot differ from a 2D one.
pub(super) fn one_texel_volume(image: Image) -> bool {
    image.width == 1 && image.height == 1 && image.depth == 1
}

/// A storage texture for an image whose channel selects swap red and blue:
/// Metal cannot write through a swizzled view, so the swap goes into the format.
pub(super) fn store_swizzled(tex: Retained<Texture>, sel: [u8; 4]) -> Option<Retained<Texture>> {
    use MTLPixelFormat as P;
    if sel[..3] != [6, 5, 4] {
        return Some(tex);
    }
    let swapped = match tex.pixelFormat() {
        P::RGBA8Unorm => P::BGRA8Unorm,
        P::BGRA8Unorm => P::RGBA8Unorm,
        _ => return Some(tex),
    };
    tex.newTextureViewWithPixelFormat(swapped).or(Some(tex))
}

/// A texture with its sRGB conversion dropped, for raw access: image
/// stores write the value as UNORM (measured on RDNA2), and loads decode it
/// themselves.
pub(super) fn without_srgb(tex: Retained<Texture>) -> Option<Retained<Texture>> {
    use MTLPixelFormat as P;
    let linear = match tex.pixelFormat() {
        P::R8Unorm_sRGB => P::R8Unorm,
        P::RG8Unorm_sRGB => P::RG8Unorm,
        P::RGBA8Unorm_sRGB => P::RGBA8Unorm,
        P::BGRA8Unorm_sRGB => P::BGRA8Unorm,
        P::BC1_RGBA_sRGB => P::BC1_RGBA,
        P::BC2_RGBA_sRGB => P::BC2_RGBA,
        P::BC3_RGBA_sRGB => P::BC3_RGBA,
        P::BC7_RGBAUnorm_sRGB => P::BC7_RGBAUnorm,
        _ => return Some(tex),
    };
    tex.newTextureViewWithPixelFormat(linear)
}

// Metal allows views of the same bytes per pixel; packed float formats
// only as 32-bit integers.
pub(super) fn view_compatible(from: MTLPixelFormat, to: MTLPixelFormat) -> bool {
    use MTLPixelFormat as P;
    let bytes = |f: MTLPixelFormat| -> Option<u32> {
        Some(match f {
            P::R8Unorm | P::R8Unorm_sRGB | P::R8Snorm | P::R8Uint | P::R8Sint => 1,
            P::R16Unorm
            | P::R16Snorm
            | P::R16Uint
            | P::R16Sint
            | P::R16Float
            | P::RG8Unorm
            | P::RG8Unorm_sRGB
            | P::RG8Snorm
            | P::RG8Uint
            | P::RG8Sint => 2,
            P::R32Uint
            | P::R32Sint
            | P::R32Float
            | P::RG16Unorm
            | P::RG16Snorm
            | P::RG16Uint
            | P::RG16Sint
            | P::RG16Float => 4,
            P::RGBA8Unorm
            | P::RGBA8Unorm_sRGB
            | P::RGBA8Snorm
            | P::RGBA8Uint
            | P::RGBA8Sint
            | P::BGRA8Unorm
            | P::BGRA8Unorm_sRGB => 4,
            P::RGB10A2Unorm | P::BGR10A2Unorm | P::RG11B10Float | P::RGB9E5Float => 4,
            P::RG32Uint
            | P::RG32Sint
            | P::RG32Float
            | P::RGBA16Unorm
            | P::RGBA16Snorm
            | P::RGBA16Uint
            | P::RGBA16Sint
            | P::RGBA16Float => 8,
            P::RGBA32Uint | P::RGBA32Sint | P::RGBA32Float => 16,
            _ => return None,
        })
    };
    let packed = |f: MTLPixelFormat| matches!(f, P::RG11B10Float | P::RGB9E5Float);
    if from == to {
        return true;
    }
    match (bytes(from), bytes(to)) {
        (Some(a), Some(b)) if a == b => {
            if packed(from) || packed(to) {
                let other = if packed(from) { to } else { from };
                matches!(other, P::R32Uint | P::R32Sint)
            } else {
                true
            }
        }
        _ => false,
    }
}

/// The Metal format and bytes per element of a GFX10 image format.
pub(super) fn image_format(format: u32) -> Option<(MTLPixelFormat, u32)> {
    use MTLPixelFormat as P;
    Some(match format {
        1 => (P::R8Unorm, 1),
        2 => (P::R8Snorm, 1),
        5 => (P::R8Uint, 1),
        6 => (P::R8Sint, 1),
        7 => (P::R16Unorm, 2),
        8 => (P::R16Snorm, 2),
        11 => (P::R16Uint, 2),
        12 => (P::R16Sint, 2),
        13 => (P::R16Float, 2),
        14 => (P::RG8Unorm, 2),
        15 => (P::RG8Snorm, 2),
        18 => (P::RG8Uint, 2),
        19 => (P::RG8Sint, 2),
        20 => (P::R32Uint, 4),
        21 => (P::R32Sint, 4),
        22 => (P::R32Float, 4),
        23 => (P::RG16Unorm, 4),
        24 => (P::RG16Snorm, 4),
        27 => (P::RG16Uint, 4),
        28 => (P::RG16Sint, 4),
        29 => (P::RG16Float, 4),
        36 => (P::RG11B10Float, 4),
        50 => (P::RGB10A2Unorm, 4),
        56 => (P::RGBA8Unorm, 4),
        57 => (P::RGBA8Snorm, 4),
        60 => (P::RGBA8Uint, 4),
        61 => (P::RGBA8Sint, 4),
        62 => (P::RG32Uint, 8),
        63 => (P::RG32Sint, 8),
        64 => (P::RG32Float, 8),
        65 => (P::RGBA16Unorm, 8),
        66 => (P::RGBA16Snorm, 8),
        69 => (P::RGBA16Uint, 8),
        70 => (P::RGBA16Sint, 8),
        71 => (P::RGBA16Float, 8),
        75 => (P::RGBA32Uint, 16),
        76 => (P::RGBA32Sint, 16),
        77 => (P::RGBA32Float, 16),
        128 => (P::R8Unorm_sRGB, 1),
        129 => (P::RG8Unorm_sRGB, 2),
        130 => (P::RGBA8Unorm_sRGB, 4),
        169 => (P::BC1_RGBA, 8),
        170 => (P::BC1_RGBA_sRGB, 8),
        171 => (P::BC2_RGBA, 16),
        172 => (P::BC2_RGBA_sRGB, 16),
        173 => (P::BC3_RGBA, 16),
        174 => (P::BC3_RGBA_sRGB, 16),
        175 => (P::BC4_RUnorm, 8),
        176 => (P::BC4_RSnorm, 8),
        177 => (P::BC5_RGUnorm, 16),
        178 => (P::BC5_RGSnorm, 16),
        179 => (P::BC6H_RGBUfloat, 16),
        180 => (P::BC6H_RGBFloat, 16),
        181 => (P::BC7_RGBAUnorm, 16),
        182 => (P::BC7_RGBAUnorm_sRGB, 16),
        _ => return None,
    })
}

/// How many channels GFX10 image format `format` stores.
pub(super) fn channels(format: u32) -> u8 {
    match format {
        1..=13 | 20..=22 | 128 | 131 | 140 | 175 | 176 | 285 => 1,
        14..=19 | 23..=29 | 62..=64 | 129 | 137 | 141..=146 | 177 | 178 => 2,
        30..=43 | 72..=74 | 132 | 133 | 179 | 180 => 3,
        _ => 4,
    }
}

/// An image's channel selects (`DST_SEL_X..W`) as the hardware applies them:
/// selecting a channel the format does not store gives channel `c mod n`
/// of its `n` channels (measured on RDNA2).
pub(super) fn channel_selects(image: &Image) -> [u8; 4] {
    let n = channels(image.format);
    image
        .swizzle
        .map(|s| if s >= 4 { 4 + (s - 4) % n } else { s })
}

/// The texture for an image of a format Metal lacks, read and written raw:
/// unsigned integers of the element's size, which programs convert.
pub(super) fn coded_format(format: u32) -> Option<(MTLPixelFormat, u32)> {
    use MTLPixelFormat as P;
    let bytes = match format {
        1..=6 | 128 | 131 | 137 => 1,
        7..=19 | 129 | 133 | 135 | 136 => 2,
        20..=61 | 130 | 132 | 140..=144 => 4,
        62..=71 | 145 | 146 => 8,
        75..=77 => 16,
        _ => return None,
    };
    Some(match bytes {
        1 => (P::R8Uint, 1),
        2 => (P::R16Uint, 2),
        4 => (P::R32Uint, 4),
        8 => (P::RG32Uint, 8),
        _ => (P::RGBA32Uint, 16),
    })
}

/// `tex`, holding an image of format `format`, viewed as unsigned integers
/// of the element's size, for programs that unpack its bits.
pub(super) fn bits_view(tex: Retained<Texture>, format: u32) -> Option<Retained<Texture>> {
    let (want, _) = coded_format(format)?;
    if tex.pixelFormat() == want {
        Some(tex)
    } else {
        tex.newTextureViewWithPixelFormat(want)
    }
}

/// The Metal format and bytes per element of an image read or written
/// raw: its own format where Metal has one, otherwise [`coded_format`].
pub(super) fn raw_format(format: u32) -> Option<(MTLPixelFormat, u32)> {
    image_format(format).or_else(|| coded_format(format))
}

pub(super) fn is_depth_format(f: MTLPixelFormat) -> bool {
    matches!(
        f,
        MTLPixelFormat::Depth32Float
            | MTLPixelFormat::Depth16Unorm
            | MTLPixelFormat::Depth32Float_Stencil8
    )
}

/// Whether swizzle mode `raw` is one of the `_Z` (depth) modes: 4 KiB and
/// 64 KiB, plain, `_T` and `_X`.
pub(super) fn is_depth_layout(raw: u32) -> bool {
    matches!(raw, 0x04 | 0x08 | 0x10 | 0x14 | 0x18)
}

/// Bytes per texel of an uncompressed format, from the image format table.
pub(super) fn texel_bytes(f: MTLPixelFormat) -> Option<usize> {
    use MTLPixelFormat as P;
    (0..256)
        .find_map(|n| {
            image_format(n)
                .filter(|(m, _)| *m == f)
                .map(|(_, b)| b as usize)
        })
        .or(match f {
            P::BGRA8Unorm | P::BGRA8Unorm_sRGB | P::RGBA8Unorm_sRGB | P::BGR10A2Unorm => Some(4),
            P::R8Unorm_sRGB => Some(1),
            P::RG8Unorm_sRGB => Some(2),
            _ => None,
        })
}

/// Formats made with atomic usage, for image atomics.
pub(super) fn atomic_capable(format: MTLPixelFormat) -> bool {
    matches!(format, MTLPixelFormat::R32Uint | MTLPixelFormat::R32Sint)
}

pub(super) fn is_compressed(f: MTLPixelFormat) -> bool {
    (MTLPixelFormat::BC1_RGBA.0..=MTLPixelFormat::BC7_RGBAUnorm_sRGB.0).contains(&f.0)
}

pub(super) fn compare(f: u32) -> MTLCompareFunction {
    use MTLCompareFunction as C;
    match f {
        0 => C::Never,
        1 => C::Less,
        2 => C::Equal,
        3 => C::LessEqual,
        4 => C::Greater,
        5 => C::NotEqual,
        6 => C::GreaterEqual,
        _ => C::Always,
    }
}

pub(super) fn stencil_op(op: u32) -> MTLStencilOperation {
    use MTLStencilOperation as S;
    match op {
        1 => S::Zero,
        3 | 4 => S::Replace,
        5 => S::IncrementClamp,
        6 => S::DecrementClamp,
        7 => S::Invert,
        8 => S::IncrementWrap,
        9 => S::DecrementWrap,
        _ => S::Keep,
    }
}
