//! Where an element of a GFX10 surface lives in memory, for the swizzle
//! modes PS5 titles use. The bit equations are SharpEmu-mac's
//! (GPL-2.0-or-later, `TileGeometry.Blocks.cs` and `MetalTilerShaders.cs`);
//! the PS5's render-target mode matches the frame layout ScreenTester5
//! writes (see `docs/gpu.md`).

/// A surface's swizzle mode (`SW_MODE` in image descriptors, `TILE_MODE`
/// in render-target registers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileMode {
    Linear,
    /// `SW_4KB_S`: 4 KiB standard blocks, the first 4 KiB of a `SW_64KB_S` block.
    Standard4K,
    /// `SW_64KB_S`: 64 KiB standard blocks.
    Standard64K,
    /// `SW_64KB_R_X` as the PS5 lays it out: render targets and display buffers.
    RenderTarget,
}

impl TileMode {
    pub fn from_bits(mode: u32) -> Option<Self> {
        match mode {
            0 => Some(Self::Linear),
            0x05 => Some(Self::Standard4K),
            0x09 => Some(Self::Standard64K),
            0x1B => Some(Self::RenderTarget),
            _ => None,
        }
    }
}

/// `(axis, shift, mask)`: coordinate `axis` (0 = x, 1 = y) contributes
/// `(c << shift) & mask` to the byte offset inside a 64 KiB block.
type Term = (u8, u8, u32);

const fn x(shift: u8, mask: u32) -> Term {
    (0, shift, mask)
}

const fn y(shift: u8, mask: u32) -> Term {
    (1, shift, mask)
}

static STANDARD_64K: [&[Term]; 5] = [
    &[x(0, 0x0f), x(5, 0x200), x(6, 0x800), x(7, 0x2000), x(8, 0x8000), y(4, 0x1f0), y(5, 0x400), y(6, 0x1000), y(7, 0x4000)],
    &[x(1, 0x00e), x(4, 0x080), x(5, 0x200), x(6, 0x800), x(7, 0x2000), x(8, 0x8000), y(4, 0x070), y(5, 0x100), y(6, 0x400), y(7, 0x1000), y(8, 0x4000)],
    &[x(2, 0x00c), x(5, 0x080), x(6, 0x200), x(7, 0x800), x(8, 0x2000), x(9, 0x8000), y(4, 0x070), y(5, 0x100), y(6, 0x400), y(7, 0x1000), y(8, 0x4000)],
    &[x(3, 0x008), x(5, 0x0c0), x(6, 0x200), x(7, 0x800), x(8, 0x2000), x(9, 0x8000), y(4, 0x030), y(6, 0x100), y(7, 0x400), y(8, 0x1000), y(9, 0x4000)],
    &[x(6, 0x0c0), x(7, 0x200), x(8, 0x800), x(9, 0x2000), x(10, 0x8000), y(4, 0x030), y(6, 0x100), y(7, 0x400), y(8, 0x1000), y(9, 0x4000)],
];

static STANDARD_4K: [&[Term]; 5] = [
    &[x(0, 0x0f), x(5, 0x200), x(6, 0x800), y(4, 0x1f0), y(5, 0x400)],
    &[x(1, 0x00e), x(4, 0x080), x(5, 0x200), x(6, 0x800), y(4, 0x070), y(5, 0x100), y(6, 0x400)],
    &[x(2, 0x00c), x(5, 0x080), x(6, 0x200), x(7, 0x800), y(4, 0x070), y(5, 0x100), y(6, 0x400)],
    &[x(3, 0x008), x(5, 0x0c0), x(6, 0x200), x(7, 0x800), y(4, 0x030), y(6, 0x100), y(7, 0x400)],
    &[x(6, 0x0c0), x(7, 0x200), x(8, 0x800), y(4, 0x030), y(6, 0x100), y(7, 0x400)],
];

static RENDER_TARGET: [&[Term]; 5] = [
    &[y(2, 0x008), y(4, 0x010), y(3, 0x0a0), y(5, 0xf00), y(6, 0x1000), y(7, 0x4000), x(0, 0x007), x(3, 0x040), x(5, 0x300), x(4, 0x400), x(6, 0x800), x(7, 0x2000), x(8, 0x8000)],
    &[y(4, 0x070), y(5, 0xf00), y(8, 0x5000), x(1, 0x00e), x(4, 0x480), x(5, 0x300), x(6, 0x800), x(7, 0x2000), x(8, 0x8000)],
    &[y(4, 0x070), y(5, 0xf00), y(9, 0x1000), y(8, 0x4000), x(2, 0x00c), x(5, 0x380), x(4, 0x400), x(6, 0x800), x(9, 0xa000)],
    &[y(4, 0x010), y(6, 0x080), y(5, 0xf00), y(10, 0x5000), x(3, 0x008), x(4, 0x460), x(5, 0x300), x(6, 0x800), x(10, 0x2000), x(9, 0x8000)],
    &[x(4, 0x410), x(5, 0x340), x(6, 0x800), x(11, 0xa000), y(5, 0xf20), y(6, 0x080), y(10, 0x1000), y(11, 0x4000)],
];

/// log2 of a 64 KiB thin block's width and height, by log2 of the element size.
const BLOCK_64K: [(u32, u32); 5] = [(8, 8), (8, 7), (7, 7), (7, 6), (6, 6)];

/// One mip level of a 2D surface: enough to find any element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    pub mode: TileMode,
    /// Bytes per element: a texel, or a 4x4 block of a compressed format.
    pub bpe: u32,
    /// Size in elements.
    pub width: u32,
    pub height: u32,
    /// Row pitch in elements, padded to the block width.
    pub pitch: u32,
}

impl Layout {
    pub fn new(mode: TileMode, bpe: u32, width: u32, height: u32) -> Option<Self> {
        if !matches!(bpe, 1 | 2 | 4 | 8 | 16) || width == 0 || height == 0 {
            return None;
        }
        let (bw, _) = Self::block(mode, bpe);
        let pitch = width.div_ceil(bw) * bw;
        Some(Self {
            mode,
            bpe,
            width,
            height,
            pitch,
        })
    }

    /// Block width and height in elements; linear surfaces pad rows to 256 bytes.
    fn block(mode: TileMode, bpe: u32) -> (u32, u32) {
        match mode {
            TileMode::Linear => ((256 / bpe).max(1), 1),
            TileMode::Standard4K => {
                let (w, h) = BLOCK_64K[bpe.trailing_zeros() as usize];
                (1 << (w - 2), 1 << (h - 2))
            }
            _ => {
                let (w, h) = BLOCK_64K[bpe.trailing_zeros() as usize];
                (1 << w, 1 << h)
            }
        }
    }

    fn terms(&self) -> &'static [Term] {
        let i = self.bpe.trailing_zeros() as usize;
        match self.mode {
            TileMode::Linear => &[],
            TileMode::Standard4K => STANDARD_4K[i],
            TileMode::Standard64K => STANDARD_64K[i],
            TileMode::RenderTarget => RENDER_TARGET[i],
        }
    }

    /// The surface's size in bytes.
    pub fn size(&self) -> u64 {
        let (_, bh) = Self::block(self.mode, self.bpe);
        self.pitch as u64 * self.height.div_ceil(bh) as u64 * bh as u64 * self.bpe as u64
    }

    /// Where mip level 0 starts in a 64 KiB-block surface of `levels`
    /// levels: smaller levels come first, the smallest ones packed into one
    /// trailing block (the mip tail).
    pub fn mip0_offset(&self, levels: u32) -> u64 {
        if self.mode == TileMode::Linear || levels <= 1 {
            return 0;
        }
        let (bw, bh) = Self::block(self.mode, self.bpe);
        let tail_max = 12;
        let first_tail = (0..levels)
            .find(|&l| {
                self.width.div_ceil(1 << l) <= bw / 2 && self.height.div_ceil(1 << l) <= bh && levels - l <= tail_max
            })
            .unwrap_or(levels);
        let mut offset = if first_tail < levels { self.block_bytes() } else { 0 };
        for l in 1..first_tail {
            let pw = self.width.div_ceil(1 << l).max(1).next_multiple_of(bw) as u64;
            let ph = self.height.div_ceil(1 << l).max(1).next_multiple_of(bh) as u64;
            offset += pw * ph * self.bpe as u64;
        }
        offset
    }

    /// The byte offset of element (`ex`, `ey`).
    pub fn offset(&self, ex: u32, ey: u32) -> u64 {
        if self.mode == TileMode::Linear {
            return (ey as u64 * self.pitch as u64 + ex as u64) * self.bpe as u64;
        }
        let (bw, bh) = Self::block(self.mode, self.bpe);
        let block = (ey / bh) as u64 * (self.pitch / bw) as u64 + (ex / bw) as u64;
        let mut inner = 0u32;
        for &(axis, shift, mask) in self.terms() {
            let c = if axis == 0 { ex } else { ey };
            inner ^= c.wrapping_shl(shift as u32) & mask;
        }
        block * self.block_bytes() + inner as u64
    }

    pub fn block_bytes(&self) -> u64 {
        if self.mode == TileMode::Standard4K { 0x1000 } else { 0x10000 }
    }

    /// MSL for `ulong surface_offset(uint x, uint y, uint pitch)`, the same
    /// function as [`Layout::offset`] for this mode and element size.
    pub fn msl_offset(&self) -> String {
        if self.mode == TileMode::Linear {
            return format!(
                "ulong surface_offset(uint x, uint y, uint pitch) {{ return (ulong(y) * pitch + x) * {}; }}\n",
                self.bpe
            );
        }
        let (bw, bh) = Self::block(self.mode, self.bpe);
        let mut s = format!(
            "ulong surface_offset(uint x, uint y, uint pitch) {{\n    ulong block = ulong(y / {bh}u) * (pitch / {bw}u) + x / {bw}u;\n    uint inner = 0;\n"
        );
        for &(axis, shift, mask) in self.terms() {
            let c = if axis == 0 { "x" } else { "y" };
            s += &format!("    inner ^= ({c} << {shift}) & {mask:#x}u;\n");
        }
        s += &format!("    return block * {}ul + inner;\n}}\n", self.block_bytes());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame layout ScreenTester5 writes, as recorded in `docs/gpu.md`:
    /// a 32-bit pixel's offset bits inside its 128x128 tile.
    fn screentester_offset(px: u32, py: u32) -> u64 {
        let bit = |v: u32, b: u32| (v >> b) & 1;
        let mut o = 0u32;
        for (xb, ob) in [(0, 2), (1, 3), (2, 7), (3, 8), (4, 9), (4, 13), (5, 11), (6, 10), (6, 15)] {
            o ^= bit(px, xb) << ob;
        }
        for (yb, ob) in [(0, 4), (1, 5), (2, 6), (3, 8), (3, 12), (4, 9), (5, 10), (6, 11), (6, 14)] {
            o ^= bit(py, yb) << ob;
        }
        o as u64
    }

    #[test]
    fn render_target_mode_matches_the_display_layout() {
        let l = Layout::new(TileMode::RenderTarget, 4, 1920, 1080).unwrap();
        assert_eq!(l.pitch, 1920);
        for (px, py) in [(0, 0), (1, 0), (5, 3), (127, 127), (64, 9), (100, 77)] {
            assert_eq!(l.offset(px, py), screentester_offset(px, py));
        }
        assert_eq!(l.offset(128, 0), 0x10000);
        assert_eq!(l.offset(0, 128), 15 * 0x10000);
    }

    #[test]
    fn every_block_offset_is_used_once() {
        for mode in [TileMode::Standard4K, TileMode::Standard64K, TileMode::RenderTarget] {
            for bpe in [1, 2, 4, 8, 16] {
                let l = Layout::new(mode, bpe, 256, 256).unwrap();
                let (bw, bh) = Layout::block(mode, bpe);
                let bytes = l.block_bytes();
                let mut seen = vec![false; bytes as usize / bpe as usize];
                for ey in 0..bh {
                    for ex in 0..bw {
                        let o = l.offset(ex, ey);
                        assert!(o < bytes && o.is_multiple_of(bpe as u64));
                        let i = (o / bpe as u64) as usize;
                        assert!(!seen[i], "{mode:?} {bpe}: ({ex}, {ey}) collides");
                        seen[i] = true;
                    }
                }
            }
        }
    }

    #[test]
    fn standard_4k_is_the_start_of_standard_64k() {
        for bpe in [1, 2, 4, 8, 16] {
            let small = Layout::new(TileMode::Standard4K, bpe, 1024, 1024).unwrap();
            let big = Layout::new(TileMode::Standard64K, bpe, 1024, 1024).unwrap();
            let (bw, bh) = Layout::block(TileMode::Standard4K, bpe);
            for ey in 0..bh {
                for ex in 0..bw {
                    assert_eq!(small.offset(ex, ey), big.offset(ex, ey));
                }
            }
            assert_eq!(small.offset(bw, 0), 0x1000);
        }
    }

    #[test]
    fn linear_rows_are_padded_to_256_bytes() {
        let l = Layout::new(TileMode::Linear, 4, 100, 10).unwrap();
        assert_eq!(l.pitch, 128);
        assert_eq!(l.offset(1, 2), (2 * 128 + 1) * 4);
    }
}
