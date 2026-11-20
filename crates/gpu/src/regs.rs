//! GFX10 register offsets and the state a draw reads from them. Offsets
//! are relative to their bank (context `0xA000`, shader `0x2C00`,
//! user-config `0xC000`), as in SharpEmu-mac's `RegisterOffsets.cs`
//! (GPL-2.0-or-later); field layouts follow AMD's RDNA register
//! documentation as Mesa and SharpEmu-mac decode it.

use crate::tiling::TileMode;

pub mod context {
    pub const DB_RENDER_CONTROL: usize = 0x000;
    pub const DB_DEPTH_VIEW: usize = 0x002;
    pub const DB_DEPTH_SIZE_XY: usize = 0x007;
    pub const DB_STENCIL_CLEAR: usize = 0x00A;
    pub const DB_DEPTH_CLEAR: usize = 0x00B;
    pub const DB_Z_INFO: usize = 0x010;
    pub const DB_STENCIL_INFO: usize = 0x011;
    pub const DB_Z_WRITE_BASE: usize = 0x014;
    pub const DB_STENCIL_WRITE_BASE: usize = 0x015;
    pub const DB_Z_WRITE_BASE_HI: usize = 0x01C;
    pub const DB_STENCIL_CONTROL: usize = 0x10B;
    pub const DB_STENCIL_REF_MASK: usize = 0x10C;
    pub const DB_STENCIL_REF_MASK_BF: usize = 0x10D;
    pub const DB_DEPTH_CONTROL: usize = 0x200;
    pub const CB_TARGET_MASK: usize = 0x08E;
    pub const PA_SC_SCREEN_SCISSOR_TL: usize = 0x00C;
    pub const PA_SC_SCREEN_SCISSOR_BR: usize = 0x00D;
    pub const PA_SC_WINDOW_SCISSOR_TL: usize = 0x081;
    pub const PA_SC_WINDOW_SCISSOR_BR: usize = 0x082;
    pub const PA_SC_GENERIC_SCISSOR_TL: usize = 0x090;
    pub const PA_SC_GENERIC_SCISSOR_BR: usize = 0x091;
    pub const PA_SC_VPORT_SCISSOR_0_TL: usize = 0x094;
    pub const PA_SC_VPORT_SCISSOR_0_BR: usize = 0x095;
    pub const PA_SC_VPORT_ZMIN_0: usize = 0x0B4;
    pub const PA_SC_VPORT_ZMAX_0: usize = 0x0B5;
    pub const CB_BLEND_RED: usize = 0x105;
    pub const PA_CL_VPORT_XSCALE: usize = 0x10F;
    pub const SPI_PS_INPUT_CNTL_0: usize = 0x191;
    pub const SPI_VS_OUT_CONFIG: usize = 0x1B1;
    pub const SPI_PS_INPUT_ENA: usize = 0x1B3;
    pub const SPI_PS_INPUT_ADDR: usize = 0x1B4;
    pub const SPI_SHADER_COL_FORMAT: usize = 0x1C5;
    pub const CB_BLEND0_CONTROL: usize = 0x1E0;
    pub const PA_CL_CLIP_CNTL: usize = 0x204;
    pub const PA_SU_SC_MODE_CNTL: usize = 0x205;
    pub const PA_CL_VTE_CNTL: usize = 0x206;
    pub const VGT_SHADER_STAGES_EN: usize = 0x2D5;
    pub const CB_COLOR0_BASE: usize = 0x318;
    pub const CB_COLOR0_VIEW: usize = 0x31B;
    pub const CB_COLOR0_INFO: usize = 0x31C;
    pub const CB_COLOR0_BASE_EXT: usize = 0x390;
    pub const CB_COLOR0_ATTRIB2: usize = 0x3B0;
    pub const CB_COLOR0_ATTRIB3: usize = 0x3B8;
    /// Distance between consecutive color targets' registers.
    pub const CB_COLOR_STRIDE: usize = 0xF;
}

pub mod sh {
    pub const SPI_SHADER_PGM_LO_PS: usize = 0x008;
    pub const SPI_SHADER_PGM_RSRC2_PS: usize = 0x00B;
    pub const SPI_SHADER_USER_DATA_PS_0: usize = 0x00C;
    pub const SPI_SHADER_PGM_LO_ES: usize = 0x0C8;
    pub const SPI_SHADER_PGM_LO_GS: usize = 0x088;
    pub const SPI_SHADER_PGM_RSRC2_GS: usize = 0x08B;
    pub const SPI_SHADER_USER_DATA_GS_0: usize = 0x08C;
    pub const SPI_SHADER_PGM_LO_VS: usize = 0x048;
    pub const SPI_SHADER_PGM_RSRC2_VS: usize = 0x04B;
    pub const SPI_SHADER_USER_DATA_VS_0: usize = 0x04C;
    pub const COMPUTE_NUM_THREAD_X: usize = 0x207;
    pub const COMPUTE_PGM_LO: usize = 0x20C;
    pub const COMPUTE_PGM_RSRC2: usize = 0x213;
    pub const COMPUTE_USER_DATA_0: usize = 0x240;
    /// User data registers per stage.
    pub const USER_DATA_COUNT: usize = 32;
    pub const COMPUTE_USER_DATA_COUNT: usize = 16;
}

pub mod uconfig {
    pub const VGT_PRIMITIVE_TYPE: usize = 0x242;
    pub const VGT_INDEX_TYPE: usize = 0x243;
}

/// A color target's channel layout, number type and component order
/// (`CB_COLORn_INFO`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorFormat {
    pub layout: u32,
    pub number: u32,
    pub swap: u32,
}

impl ColorFormat {
    /// Bytes per pixel, for layouts we know.
    pub fn bytes(self) -> Option<u32> {
        Some(match self.layout {
            1 => 1,
            2 | 3 | 16..=19 => 2,
            4..=10 | 20..=22 => 4,
            11 | 12 => 8,
            14 => 16,
            _ => return None,
        })
    }
}

/// A bound color target, decoded from its registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub address: u64,
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub tile: Option<TileMode>,
    pub raw_tile: u32,
    /// The surface's last mip level (`CB_COLOR_ATTRIB2.MAX_MIP`): level 0
    /// lies after the smaller levels in memory.
    pub max_mip: u32,
    /// The level drawn to (`CB_COLOR_VIEW.MIP_LEVEL`).
    pub mip_level: u32,
}

impl ColorTarget {
    /// Color target `n`, or `None` when it has no address or format.
    pub fn read(c: &[u32], n: usize) -> Option<Self> {
        use context::*;
        let r = |reg: usize| c[reg + n * CB_COLOR_STRIDE];
        let ext = c[CB_COLOR0_BASE_EXT + n];
        let address = ((r(CB_COLOR0_BASE) as u64) | ((ext as u64 & 0xFF) << 32)) << 8;
        let info = r(CB_COLOR0_INFO);
        let attrib2 = c[CB_COLOR0_ATTRIB2 + n];
        let attrib3 = c[CB_COLOR0_ATTRIB3 + n];
        let format = ColorFormat {
            layout: (info >> 2) & 0x1F,
            number: (info >> 8) & 7,
            swap: (info >> 11) & 3,
        };
        if address == 0 || format.layout == 0 {
            return None;
        }
        let raw_tile = (attrib3 >> 14) & 0x1F;
        Some(Self {
            address,
            width: ((attrib2 >> 14) & 0x3FFF) + 1,
            height: (attrib2 & 0x3FFF) + 1,
            format,
            tile: TileMode::from_bits(raw_tile),
            raw_tile,
            max_mip: attrib2 >> 28,
            mip_level: (r(CB_COLOR0_VIEW) >> 26) & 0xF,
        })
    }
}

/// The bound depth (and stencil) target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthTarget {
    pub address: u64,
    /// 0 none, 1 16-bit unorm, 3 32-bit float.
    pub format: u32,
    pub stencil: bool,
    /// From `DB_DEPTH_SIZE_XY`; zero when the title did not write it.
    pub width: u32,
    pub height: u32,
    /// The array slice drawn to and the last slice of the view
    /// (`DB_DEPTH_VIEW`).
    pub slice: u32,
    pub slice_max: u32,
}

impl DepthTarget {
    pub fn read(c: &[u32]) -> Option<Self> {
        use context::*;
        let format = c[DB_Z_INFO] & 3;
        let address = ((c[DB_Z_WRITE_BASE] as u64) | ((c[DB_Z_WRITE_BASE_HI] as u64 & 0xFF) << 32)) << 8;
        if format == 0 || address == 0 {
            return None;
        }
        let size = c[DB_DEPTH_SIZE_XY];
        let (w, h) = if size == 0 { (0, 0) } else { ((size & 0x3FFF) + 1, ((size >> 16) & 0x3FFF) + 1) };
        Some(Self {
            address,
            format,
            stencil: c[DB_STENCIL_INFO] & 1 != 0,
            width: w,
            height: h,
            slice: (c[DB_DEPTH_VIEW] & 0x7FF) | ((c[DB_DEPTH_VIEW] >> 25 & 3) << 11),
            slice_max: ((c[DB_DEPTH_VIEW] >> 13) & 0x7FF) | ((c[DB_DEPTH_VIEW] >> 27 & 3) << 11),
        })
    }
}

/// Depth and stencil tests: `DB_DEPTH_CONTROL` and the stencil registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthState {
    pub control: u32,
    pub stencil_control: u32,
    pub ref_mask: u32,
    pub ref_mask_back: u32,
}

impl DepthState {
    pub fn read(c: &[u32]) -> Self {
        use context::*;
        Self {
            control: c[DB_DEPTH_CONTROL],
            stencil_control: c[DB_STENCIL_CONTROL],
            ref_mask: c[DB_STENCIL_REF_MASK],
            ref_mask_back: c[DB_STENCIL_REF_MASK_BF],
        }
    }
    pub fn depth_test(&self) -> bool {
        self.control & 2 != 0
    }
    pub fn depth_write(&self) -> bool {
        self.control & 4 != 0
    }
    pub fn depth_func(&self) -> u32 {
        (self.control >> 4) & 7
    }
    pub fn stencil(&self) -> bool {
        self.control & 1 != 0
    }
}

/// `CB_BLENDn_CONTROL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Blend {
    pub enable: bool,
    pub color_src: u32,
    pub color_op: u32,
    pub color_dst: u32,
    pub alpha_src: u32,
    pub alpha_op: u32,
    pub alpha_dst: u32,
}

impl Blend {
    pub fn read(c: &[u32], n: usize) -> Self {
        let v = c[context::CB_BLEND0_CONTROL + n];
        let separate = v & (1 << 29) != 0;
        let (cs, co, cd) = (v & 0x1F, (v >> 5) & 7, (v >> 8) & 0x1F);
        Self {
            enable: v & (1 << 30) != 0,
            color_src: cs,
            color_op: co,
            color_dst: cd,
            alpha_src: if separate { (v >> 16) & 0x1F } else { cs },
            alpha_op: if separate { (v >> 21) & 7 } else { co },
            alpha_dst: if separate { (v >> 24) & 0x1F } else { cd },
        }
    }
}

/// The viewport transform, `PA_CL_VPORT_*` for viewport 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x_scale: f32,
    pub x_offset: f32,
    pub y_scale: f32,
    pub y_offset: f32,
    pub z_scale: f32,
    pub z_offset: f32,
}

impl Viewport {
    pub fn read(c: &[u32]) -> Self {
        let f = |i: usize| f32::from_bits(c[context::PA_CL_VPORT_XSCALE + i]);
        Self {
            x_scale: f(0),
            x_offset: f(1),
            y_scale: f(2),
            y_offset: f(3),
            z_scale: f(4),
            z_offset: f(5),
        }
    }
}

/// How the depths a draw can produce land in Metal's `[0, 1]`, as
/// `depth * scale + offset`. With z clipping, depth stays in the viewport's
/// z range (`[offset - scale, offset + scale]` for GL clip space), clamped
/// to `PA_SC_VPORT_ZMIN_0..ZMAX_0`. A float depth buffer keeps values outside
/// `[0, 1]` (Isaac draws at -1..1 with a +-10000 clamp range) where Metal
/// clamps them, so a range reaching outside `[0, 1]` is mapped onto it.
/// Draws whose ranges differ map differently; no title seen so far mixes
/// such ranges in one depth buffer.
pub fn depth_range(c: &[u32], gl_clip: bool) -> (f32, f32) {
    let vp = Viewport::read(c);
    let near = if gl_clip { vp.z_offset - vp.z_scale } else { vp.z_offset };
    let far = vp.z_offset + vp.z_scale;
    let (mut lo, mut hi) = (near.min(far), near.max(far));
    let (zmin, zmax) = (
        f32::from_bits(c[context::PA_SC_VPORT_ZMIN_0]),
        f32::from_bits(c[context::PA_SC_VPORT_ZMAX_0]),
    );
    if zmax > zmin {
        (lo, hi) = (lo.max(zmin), hi.min(zmax));
    }
    let usable = hi > lo && (hi - lo).is_finite();
    if !usable || (lo >= 0.0 && hi <= 1.0) {
        return (1.0, 0.0);
    }
    (1.0 / (hi - lo), -lo / (hi - lo))
}

/// The scissor rectangle a draw is clipped to: the intersection of the
/// screen, window, generic and viewport-0 scissors, as `(x0, y0, x1, y1)`.
/// An empty one counts as unset: our register banks start at zero, while
/// the hardware's reset values are full-screen, and titles that rely on
/// those (Isaac) never write them.
pub fn scissor(c: &[u32]) -> (u32, u32, u32, u32) {
    use context::*;
    let tl = |v: u32| (v & 0x7FFF, (v >> 16) & 0x7FFF);
    let br = |v: u32| (v & 0x7FFF, (v >> 16) & 0x7FFF);
    let mut r = (0u32, 0u32, 0x7FFFu32, 0x7FFFu32);
    let mut clip = |a: (u32, u32), b: (u32, u32), offset: (u32, u32)| {
        r.0 = r.0.max(a.0 + offset.0);
        r.1 = r.1.max(a.1 + offset.1);
        r.2 = r.2.min(b.0 + offset.0);
        r.3 = r.3.min(b.1 + offset.1);
    };
    for (t, b) in [
        (PA_SC_SCREEN_SCISSOR_TL, PA_SC_SCREEN_SCISSOR_BR),
        (PA_SC_WINDOW_SCISSOR_TL, PA_SC_WINDOW_SCISSOR_BR),
        (PA_SC_GENERIC_SCISSOR_TL, PA_SC_GENERIC_SCISSOR_BR),
        (PA_SC_VPORT_SCISSOR_0_TL, PA_SC_VPORT_SCISSOR_0_BR),
    ] {
        let (a, bb) = (tl(c[t]), br(c[b]));
        if bb.0 > a.0 && bb.1 > a.1 {
            clip(a, bb, (0, 0));
        }
    }
    r
}

/// An image descriptor (T#), the fields a 2D sample needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Image {
    pub address: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// `DST_SEL_X..W`: 0 zero, 1 one, 4..7 the x..w channels.
    pub swizzle: [u8; 4],
    pub tile: Option<TileMode>,
    pub raw_tile: u32,
    pub kind: u32,
    pub base_level: u32,
    pub last_level: u32,
    /// Array slices (`DEPTH` + 1), or the depth of a 3D image.
    pub depth: u32,
    /// The first array slice the descriptor shows (`BASE_ARRAY`).
    pub base_array: u32,
}

impl Image {
    pub fn read(d: &[u32]) -> Self {
        let address = ((d[0] as u64) | ((d[1] as u64 & 0xFF) << 32)) << 8;
        let raw_tile = (d[3] >> 20) & 0x1F;
        Self {
            address,
            format: (d[1] >> 20) & 0x1FF,
            width: ((d[1] >> 30) | ((d[2] & 0xFFF) << 2)) + 1,
            height: ((d[2] >> 14) & 0x3FFF) + 1,
            swizzle: [0, 3, 6, 9].map(|s| ((d[3] >> s) & 7) as u8),
            tile: TileMode::from_bits(raw_tile),
            raw_tile,
            kind: d[3] >> 28,
            base_level: (d[3] >> 12) & 0xF,
            last_level: (d[3] >> 16) & 0xF,
            depth: d.get(4).map_or(0, |w| w & 0x1FFF) + 1,
            base_array: d.get(4).map_or(0, |w| (w >> 16) & 0x1FFF),
        }
    }
}

/// A sampler descriptor (S#), the fields Metal samplers have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sampler {
    /// `CLAMP_X`, `CLAMP_Y`, `CLAMP_Z`.
    pub clamp: [u8; 3],
    pub mag_linear: bool,
    pub min_linear: bool,
    pub mip_linear: bool,
}

impl Sampler {
    pub fn read(d: &[u32]) -> Self {
        Self {
            clamp: [0, 3, 6].map(|s| ((d[0] >> s) & 7) as u8),
            mag_linear: (d[2] >> 20) & 3 != 0,
            min_linear: (d[2] >> 22) & 3 != 0,
            mip_linear: (d[2] >> 26) & 3 == 2,
        }
    }
}

/// A buffer descriptor (V#).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Buffer {
    pub address: u64,
    pub stride: u32,
    pub records: u32,
}

impl Buffer {
    pub fn read(d: &[u32]) -> Self {
        Self {
            address: d[0] as u64 | ((d[1] as u64 & 0xFFFF) << 32),
            stride: (d[1] >> 16) & 0x3FFF,
            records: d[2],
        }
    }

    /// Bytes the descriptor covers.
    pub fn size(&self) -> u64 {
        if self.stride == 0 {
            self.records as u64
        } else {
            self.records as u64 * self.stride as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balatro_canvas_descriptor_decodes() {
        let t = Image::read(&[0x108f_7900, 0xc380_0000, 0x010d_c1df, 0x91b0_0fac, 0, 0x70_0000, 0, 0]);
        assert_eq!(t.address, 0x10_8f79_0000);
        assert_eq!((t.width, t.height, t.format), (1920, 1080, 56));
        assert_eq!(t.swizzle, [4, 5, 6, 7]);
        assert_eq!(t.tile, Some(TileMode::RenderTarget));
        assert_eq!(t.kind, 9);
        let s = Sampler::read(&[0x192, 0xfff000, 0x150_0000, 0]);
        assert!(s.mag_linear && s.min_linear && !s.mip_linear);
    }

    #[test]
    fn balatro_back_buffer_registers_decode() {
        let mut c = vec![0u32; 0x400];
        c[context::CB_COLOR0_BASE] = 0x1080_0100;
        c[context::CB_COLOR0_INFO] = 0x8828;
        c[context::CB_COLOR0_ATTRIB2] = 0x1dfc437;
        c[context::CB_COLOR0_ATTRIB3] = 0x4dc6c000;
        let t = ColorTarget::read(&c, 0).unwrap();
        assert_eq!(t.address, 0x10_8001_0000);
        assert_eq!((t.width, t.height), (1920, 1080));
        assert_eq!(t.format, ColorFormat { layout: 10, number: 0, swap: 1 });
        assert_eq!(t.tile, Some(TileMode::RenderTarget));
        c[context::CB_BLEND0_CONTROL] = 0x6501_0504;
        let b = Blend::read(&c, 0);
        assert!(b.enable);
        assert_eq!((b.color_src, b.color_dst, b.alpha_src, b.alpha_dst), (4, 5, 1, 5));
    }
}
