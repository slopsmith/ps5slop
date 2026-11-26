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
    /// `SW_256B_S`: 256-byte blocks, the first 256 bytes of a `SW_4KB_S` block.
    Standard256,
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
            0x01 => Some(Self::Standard256),
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
    &[
        x(0, 0x0f),
        x(5, 0x200),
        x(6, 0x800),
        x(7, 0x2000),
        x(8, 0x8000),
        y(4, 0x1f0),
        y(5, 0x400),
        y(6, 0x1000),
        y(7, 0x4000),
    ],
    &[
        x(1, 0x00e),
        x(4, 0x080),
        x(5, 0x200),
        x(6, 0x800),
        x(7, 0x2000),
        x(8, 0x8000),
        y(4, 0x070),
        y(5, 0x100),
        y(6, 0x400),
        y(7, 0x1000),
        y(8, 0x4000),
    ],
    &[
        x(2, 0x00c),
        x(5, 0x080),
        x(6, 0x200),
        x(7, 0x800),
        x(8, 0x2000),
        x(9, 0x8000),
        y(4, 0x070),
        y(5, 0x100),
        y(6, 0x400),
        y(7, 0x1000),
        y(8, 0x4000),
    ],
    &[
        x(3, 0x008),
        x(5, 0x0c0),
        x(6, 0x200),
        x(7, 0x800),
        x(8, 0x2000),
        x(9, 0x8000),
        y(4, 0x030),
        y(6, 0x100),
        y(7, 0x400),
        y(8, 0x1000),
        y(9, 0x4000),
    ],
    &[
        x(6, 0x0c0),
        x(7, 0x200),
        x(8, 0x800),
        x(9, 0x2000),
        x(10, 0x8000),
        y(4, 0x030),
        y(6, 0x100),
        y(7, 0x400),
        y(8, 0x1000),
        y(9, 0x4000),
    ],
];

static STANDARD_4K: [&[Term]; 5] = [
    &[
        x(0, 0x0f),
        x(5, 0x200),
        x(6, 0x800),
        y(4, 0x1f0),
        y(5, 0x400),
    ],
    &[
        x(1, 0x00e),
        x(4, 0x080),
        x(5, 0x200),
        x(6, 0x800),
        y(4, 0x070),
        y(5, 0x100),
        y(6, 0x400),
    ],
    &[
        x(2, 0x00c),
        x(5, 0x080),
        x(6, 0x200),
        x(7, 0x800),
        y(4, 0x070),
        y(5, 0x100),
        y(6, 0x400),
    ],
    &[
        x(3, 0x008),
        x(5, 0x0c0),
        x(6, 0x200),
        x(7, 0x800),
        y(4, 0x030),
        y(6, 0x100),
        y(7, 0x400),
    ],
    &[
        x(6, 0x0c0),
        x(7, 0x200),
        x(8, 0x800),
        y(4, 0x030),
        y(6, 0x100),
        y(7, 0x400),
    ],
];

static RENDER_TARGET: [&[Term]; 5] = [
    &[
        y(2, 0x008),
        y(4, 0x010),
        y(3, 0x0a0),
        y(5, 0xf00),
        y(6, 0x1000),
        y(7, 0x4000),
        x(0, 0x007),
        x(3, 0x040),
        x(5, 0x300),
        x(4, 0x400),
        x(6, 0x800),
        x(7, 0x2000),
        x(8, 0x8000),
    ],
    &[
        y(4, 0x070),
        y(5, 0xf00),
        y(8, 0x5000),
        x(1, 0x00e),
        x(4, 0x480),
        x(5, 0x300),
        x(6, 0x800),
        x(7, 0x2000),
        x(8, 0x8000),
    ],
    &[
        y(4, 0x070),
        y(5, 0xf00),
        y(9, 0x1000),
        y(8, 0x4000),
        x(2, 0x00c),
        x(5, 0x380),
        x(4, 0x400),
        x(6, 0x800),
        x(9, 0xa000),
    ],
    &[
        y(4, 0x010),
        y(6, 0x080),
        y(5, 0xf00),
        y(10, 0x5000),
        x(3, 0x008),
        x(4, 0x460),
        x(5, 0x300),
        x(6, 0x800),
        x(10, 0x2000),
        x(9, 0x8000),
    ],
    &[
        x(4, 0x410),
        x(5, 0x340),
        x(6, 0x800),
        x(11, 0xa000),
        y(5, 0xf20),
        y(6, 0x080),
        y(10, 0x1000),
        y(11, 0x4000),
    ],
];

/// log2 of a 64 KiB thin block's width and height, by log2 of the element size.
const BLOCK_64K: [(u32, u32); 5] = [(8, 8), (8, 7), (7, 7), (7, 6), (6, 6)];

/// Where each level of the mip tail starts inside the tail block, in
/// elements, by log2 of the element size (SharpEmu-mac's `TileGeometry`).
const TAIL_4K: [[(u32, u32); 8]; 5] = [
    [
        (32, 0),
        (16, 32),
        (0, 48),
        (0, 32),
        (16, 16),
        (16, 0),
        (0, 16),
        (0, 0),
    ],
    [
        (32, 0),
        (16, 16),
        (0, 24),
        (0, 16),
        (16, 8),
        (16, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (16, 0),
        (8, 16),
        (0, 24),
        (0, 16),
        (8, 8),
        (8, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (16, 0),
        (8, 8),
        (0, 12),
        (0, 8),
        (8, 4),
        (8, 0),
        (0, 4),
        (0, 0),
    ],
    [
        (8, 0),
        (4, 8),
        (0, 12),
        (0, 8),
        (4, 4),
        (4, 0),
        (0, 4),
        (0, 0),
    ],
];
const TAIL_64K: [[(u32, u32); 12]; 5] = [
    [
        (128, 0),
        (0, 128),
        (64, 0),
        (0, 64),
        (32, 0),
        (16, 32),
        (0, 48),
        (0, 32),
        (16, 16),
        (16, 0),
        (0, 16),
        (0, 0),
    ],
    [
        (128, 0),
        (0, 64),
        (64, 0),
        (0, 32),
        (32, 0),
        (16, 16),
        (0, 24),
        (0, 16),
        (16, 8),
        (16, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (64, 0),
        (0, 64),
        (32, 0),
        (0, 32),
        (16, 0),
        (8, 16),
        (0, 24),
        (0, 16),
        (8, 8),
        (8, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (64, 0),
        (0, 32),
        (32, 0),
        (0, 16),
        (16, 0),
        (8, 8),
        (0, 12),
        (0, 8),
        (8, 4),
        (8, 0),
        (0, 4),
        (0, 0),
    ],
    [
        (32, 0),
        (0, 32),
        (16, 0),
        (0, 16),
        (8, 0),
        (4, 8),
        (0, 12),
        (0, 8),
        (4, 4),
        (4, 0),
        (0, 4),
        (0, 0),
    ],
];

/// A 3D surface in `SW_4KB_S`, whose 4 KiB blocks span 3D boxes of elements.
/// Bit patterns and mip placement follow AMD's addrlib in Mesa (MIT,
/// `gfx10SwizzlePattern.h` `SW_4K_S3`, `Gfx10Lib::ComputeSurfaceInfoMacroTiled`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Volume {
    bpe_log2: u32,
    /// Block size in elements.
    pub block: (u32, u32, u32),
    /// Per level: size, padded pitch and height in elements, and offset in a slab.
    pub levels: Vec<VolumeLevel>,
    /// Bytes of one slab: the whole mip chain for `block.2` slices.
    slab: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolumeLevel {
    pub size: (u32, u32, u32),
    pitch: u32,
    offset: u64,
}

/// Address bit n of an element inside a `SW_4KB_S` 3D block, per log2 of
/// the element size: (axis, bit), axis 3 for a byte-address bit.
const STANDARD_3D: [[(u8, u8); 12]; 5] = {
    const B: (u8, u8) = (3, 0);
    const fn x(n: u8) -> (u8, u8) {
        (0, n)
    }
    const fn y(n: u8) -> (u8, u8) {
        (1, n)
    }
    const fn z(n: u8) -> (u8, u8) {
        (2, n)
    }
    [
        [
            x(0),
            x(1),
            z(0),
            y(0),
            z(1),
            y(1),
            x(2),
            z(2),
            y(2),
            x(3),
            z(3),
            y(3),
        ],
        [
            B,
            x(0),
            z(0),
            y(0),
            z(1),
            y(1),
            x(1),
            z(2),
            y(2),
            x(2),
            z(3),
            y(3),
        ],
        [
            B,
            B,
            x(0),
            y(0),
            z(0),
            y(1),
            x(1),
            z(1),
            y(2),
            x(2),
            z(2),
            y(3),
        ],
        [
            B,
            B,
            B,
            x(0),
            z(0),
            y(0),
            x(1),
            z(1),
            y(1),
            x(2),
            z(2),
            y(2),
        ],
        [B, B, B, B, z(0), y(0), x(0), z(1), y(1), x(1), z(2), y(2)],
    ]
};

impl Volume {
    /// The layout of a `width` x `height` x `depth` surface with `levels`
    /// mip levels; `None` for element sizes other than 1 to 16 bytes, or when
    /// level 0 is already in the mip tail. Levels in the tail are left out.
    pub fn new(bpe: u32, width: u32, height: u32, depth: u32, levels: u32) -> Option<Self> {
        if !bpe.is_power_of_two() || bpe > 16 {
            return None;
        }
        let bpe_log2 = bpe.trailing_zeros();
        let pattern = &STANDARD_3D[bpe_log2 as usize];
        let bits = |axis| pattern.iter().filter(|b| b.0 == axis).count() as u32;
        let block = (1 << bits(0), 1 << bits(1), 1 << bits(2));
        // With mips, a thick 4 KiB block halves its tail height and holds at most 5 levels.
        let tail = (block.0, block.1 / 2);
        let mut sizes = Vec::new();
        for i in 0..levels {
            let size = (
                (width >> i).max(1),
                (height >> i).max(1),
                (depth >> i).max(1),
            );
            if levels > 1 && size.0 <= tail.0 && size.1 <= tail.1 && levels - i <= 5 {
                break;
            }
            sizes.push(size);
        }
        if sizes.is_empty() {
            return None;
        }
        let tail_bytes = if (sizes.len() as u32) < levels {
            4096
        } else {
            0
        };
        let slab_bytes = |s: (u32, u32, u32)| {
            u64::from(s.0.next_multiple_of(block.0))
                * u64::from(s.1.next_multiple_of(block.1))
                * u64::from(block.2)
                * u64::from(bpe)
        };
        let mut offset = tail_bytes;
        let mut out = vec![None; sizes.len()];
        for (i, &size) in sizes.iter().enumerate().rev() {
            out[i] = Some(VolumeLevel {
                size,
                pitch: size.0.next_multiple_of(block.0),
                offset,
            });
            offset += slab_bytes(size);
        }
        Some(Self {
            bpe_log2,
            block,
            levels: out.into_iter().flatten().collect(),
            slab: offset,
        })
    }

    /// Bytes the whole surface spans.
    pub fn bytes(&self) -> u64 {
        u64::from(self.levels[0].size.2.div_ceil(self.block.2)) * self.slab
    }

    /// Byte offset of element (x, y, z) of `level` from the surface's start.
    pub fn offset(&self, level: usize, x: u32, y: u32, z: u32) -> u64 {
        let l = &self.levels[level];
        let (bw, bh, bd) = self.block;
        let block = u64::from(y / bh) * u64::from(l.pitch / bw) + u64::from(x / bw);
        let within = STANDARD_3D[self.bpe_log2 as usize]
            .iter()
            .enumerate()
            .filter(|(_, b)| b.0 < 3)
            .fold(0u64, |a, (bit, &(axis, n))| {
                a | u64::from([x, y, z][axis as usize] >> n & 1) << bit
            });
        u64::from(z / bd) * self.slab + l.offset + block * 4096 + within
    }
}

/// One mip level of a tiled 2D surface, from the start of its array slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mip {
    pub offset: u64,
    /// The level's size in elements, with its pitch.
    pub layout: Layout,
    /// Levels in the mip tail start at this element of the tail block.
    pub origin: (u32, u32),
}

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
            TileMode::Standard256 => {
                let (w, h) = BLOCK_64K[bpe.trailing_zeros() as usize];
                (1 << (w - 4), 1 << (h - 4))
            }
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
            TileMode::Standard256 | TileMode::Standard4K => STANDARD_4K[i],
            TileMode::Standard64K => STANDARD_64K[i],
            TileMode::RenderTarget => RENDER_TARGET[i],
        }
    }

    /// The surface's size in bytes.
    pub fn size(&self) -> u64 {
        let (_, bh) = Self::block(self.mode, self.bpe);
        self.pitch as u64 * self.height.div_ceil(bh) as u64 * bh as u64 * self.bpe as u64
    }

    /// Where mip level 0 starts in a surface of `levels` levels whose
    /// level 0 is this layout.
    pub fn mip0_offset(&self, levels: u32) -> u64 {
        self.mip_chain(levels, (self.width, self.height), 1).0[0].offset
    }

    /// Every level of a tiled surface of `levels` levels whose level 0 is
    /// this layout, and the size of the whole chain (an array slice). Per
    /// slice the mip tail (the smallest levels packed into one block) comes
    /// first, then the other levels from the smallest up, each padded to
    /// whole blocks. `texels` is level 0's size in texels and `block` the
    /// texels per element side (4 for block-compressed formats). Linear
    /// surfaces have no tail: their levels go from the smallest up, rows
    /// padded to 256 bytes (measured on RDNA2).
    pub fn mips(&self, levels: u32, texels: (u32, u32), block: u32) -> Vec<Mip> {
        self.mip_chain(levels, texels, block).0
    }

    /// The size of a whole chain of `levels` levels: the distance between array slices.
    pub fn chain_size(&self, levels: u32) -> u64 {
        self.mip_chain(levels, (self.width, self.height), 1).1
    }

    fn mip_chain(&self, levels: u32, texels: (u32, u32), block: u32) -> (Vec<Mip>, u64) {
        if levels <= 1 {
            let mip = Mip {
                offset: 0,
                layout: *self,
                origin: (0, 0),
            };
            return (vec![mip], self.size());
        }
        let (bw, bh) = Self::block(self.mode, self.bpe);
        let i = self.bpe.trailing_zeros() as usize;
        // 256-byte blocks have no mip tail.
        let tail: &[(u32, u32)] = match self.mode {
            TileMode::Linear | TileMode::Standard256 => &[],
            TileMode::Standard4K => &TAIL_4K[i],
            _ => &TAIL_64K[i],
        };
        let (w0, h0) = (self.width, self.height);
        let first_tail = (0..levels)
            .find(|&l| {
                w0.div_ceil(1 << l) <= bw / 2
                    && h0.div_ceil(1 << l) <= bh
                    && levels - l <= tail.len() as u32
            })
            .unwrap_or(levels);
        let level_size = |l: u32| {
            (
                (texels.0 >> l).max(1).div_ceil(block),
                (texels.1 >> l).max(1).div_ceil(block),
            )
        };
        let mut mips = vec![
            Mip {
                offset: 0,
                layout: *self,
                origin: (0, 0),
            };
            levels as usize
        ];
        let mut offset = 0;
        if first_tail < levels {
            for l in first_tail..levels {
                let (w, h) = level_size(l);
                mips[l as usize] = Mip {
                    offset: 0,
                    layout: Layout {
                        mode: self.mode,
                        bpe: self.bpe,
                        width: w,
                        height: h,
                        pitch: bw,
                    },
                    origin: tail[(l - first_tail) as usize],
                };
            }
            offset = self.block_bytes();
        }
        for l in (0..first_tail).rev() {
            let (w, h) = level_size(l);
            let pitch = w0.div_ceil(1 << l).max(1).next_multiple_of(bw);
            let padded_height = h0.div_ceil(1 << l).max(1).next_multiple_of(bh);
            mips[l as usize] = Mip {
                offset,
                layout: Layout {
                    mode: self.mode,
                    bpe: self.bpe,
                    width: w,
                    height: h,
                    pitch,
                },
                origin: (0, 0),
            };
            offset += pitch as u64 * padded_height as u64 * self.bpe as u64;
        }
        (mips, offset)
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
        block * self.block_bytes() + (inner as u64 & (self.block_bytes() - 1))
    }

    pub fn block_bytes(&self) -> u64 {
        match self.mode {
            TileMode::Linear | TileMode::Standard256 => 0x100,
            TileMode::Standard4K => 0x1000,
            _ => 0x10000,
        }
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
        s += &format!(
            "    return block * {}ul + (inner & {:#x}u);\n}}\n",
            self.block_bytes(),
            self.block_bytes() - 1
        );
        s
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn standard_3d_blocks_hold_4_kib_of_distinct_elements() {
        for bpe in [1, 2, 4, 8, 16] {
            let v = Volume::new(bpe, 64, 64, 64, 1).unwrap();
            let (bw, bh, bd) = v.block;
            assert_eq!(bw * bh * bd * bpe, 4096);
            let mut seen = std::collections::HashSet::new();
            for z in 0..bd {
                for y in 0..bh {
                    for x in 0..bw {
                        let o = v.offset(0, x, y, z);
                        assert!(o < 4096 && o.is_multiple_of(u64::from(bpe)));
                        assert!(seen.insert(o));
                    }
                }
            }
        }
    }

    #[test]
    fn standard_3d_mip_chains_put_small_levels_first_in_each_slab() {
        // GTA V's 64x64x32 RGBA8 volume with 3 levels: 8x16x8 blocks.
        let v = Volume::new(4, 64, 64, 32, 3).unwrap();
        let level2 = 16 * 16 * 8 * 4;
        let level1 = 32 * 32 * 8 * 4;
        let level0 = 64 * 64 * 8 * 4;
        assert_eq!(v.offset(2, 0, 0, 0), 0);
        assert_eq!(v.offset(1, 0, 0, 0), level2);
        assert_eq!(v.offset(0, 0, 0, 0), level2 + level1);
        assert_eq!(v.offset(0, 8, 0, 0), level2 + level1 + 4096);
        let slab = level2 + level1 + level0;
        assert_eq!(v.offset(0, 0, 0, 8), slab + level2 + level1);
        assert_eq!(v.offset(1, 0, 0, 8), slab + level2);
        assert_eq!(v.bytes(), 4 * slab);
    }

    #[test]
    fn standard_3d_levels_in_the_mip_tail_are_left_out() {
        let v = Volume::new(4, 16, 16, 16, 5).unwrap();
        assert_eq!(v.levels.len(), 1);
        assert_eq!(v.offset(0, 0, 0, 0), 4096);
        assert_eq!(Volume::new(4, 8, 8, 8, 2), None);
        assert_eq!(Volume::new(1, 4, 4, 16, 1).unwrap().offset(0, 0, 0, 1), 4);
    }
    use super::*;

    /// The frame layout ScreenTester5 writes, as recorded in `docs/gpu.md`:
    /// a 32-bit pixel's offset bits inside its 128x128 tile.
    fn screentester_offset(px: u32, py: u32) -> u64 {
        let bit = |v: u32, b: u32| (v >> b) & 1;
        let mut o = 0u32;
        for (xb, ob) in [
            (0, 2),
            (1, 3),
            (2, 7),
            (3, 8),
            (4, 9),
            (4, 13),
            (5, 11),
            (6, 10),
            (6, 15),
        ] {
            o ^= bit(px, xb) << ob;
        }
        for (yb, ob) in [
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 8),
            (3, 12),
            (4, 9),
            (5, 10),
            (6, 11),
            (6, 14),
        ] {
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
        for mode in [
            TileMode::Standard256,
            TileMode::Standard4K,
            TileMode::Standard64K,
            TileMode::RenderTarget,
        ] {
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
    fn mip_chains_put_the_tail_first_and_level_0_last() {
        // A 256x256 BC6H cube level in SW_4KB_S: 64x64 blocks of 16 bytes.
        let l = Layout::new(TileMode::Standard4K, 16, 64, 64).unwrap();
        let mips = l.mips(9, (256, 256), 4);
        // Levels 3 to 8 (8x8 blocks and smaller) fit the tail: half a block wide.
        assert_eq!(mips[3].offset, 0);
        assert_eq!(mips[3].origin, TAIL_4K[4][0]);
        assert_eq!(mips[3].layout.pitch, 16);
        assert_eq!(mips[8].origin, TAIL_4K[4][5]);
        assert_eq!(mips[2].offset, 0x1000);
        assert_eq!(mips[2].layout.width, 16);
        assert_eq!(mips[1].offset, 0x1000 + 16 * 16 * 16);
        assert_eq!(mips[0].offset, mips[1].offset + 32 * 32 * 16);
        assert_eq!(l.chain_size(9), mips[0].offset + 64 * 64 * 16);
        assert_eq!(l.mip0_offset(9), mips[0].offset);
        // Every tail level's elements stay inside the tail block.
        for m in &mips[3..] {
            let (x, y) = m.origin;
            let last = m
                .layout
                .offset(x + m.layout.width - 1, y + m.layout.height - 1);
            assert!(last < l.block_bytes(), "{m:?}");
        }
    }

    #[test]
    fn standard_256_is_the_start_of_standard_4k() {
        for bpe in [1, 2, 4, 8, 16] {
            let small = Layout::new(TileMode::Standard256, bpe, 64, 64).unwrap();
            let big = Layout::new(TileMode::Standard4K, bpe, 64, 64).unwrap();
            let (bw, bh) = Layout::block(TileMode::Standard256, bpe);
            assert_eq!(bw * bh * bpe, 256);
            for ey in 0..bh {
                for ex in 0..bw {
                    assert_eq!(small.offset(ex, ey), big.offset(ex, ey));
                }
            }
            assert_eq!(small.offset(bw, 0), 0x100);
        }
    }

    #[test]
    fn standard_256_chains_have_no_tail() {
        // A 64x64 BC1 texture with 5 levels: 16x16 blocks of 8 bytes, 8x4 per 256-byte block.
        let l = Layout::new(TileMode::Standard256, 8, 16, 16).unwrap();
        let mips = l.mips(5, (64, 64), 4);
        assert_eq!(mips[4].offset, 0);
        assert_eq!(mips[3].offset, 0x100);
        assert_eq!(mips[2].offset, 0x200);
        assert_eq!(mips[1].offset, 0x300);
        assert_eq!(mips[0].offset, 0x300 + 8 * 8 * 8);
        assert!(mips.iter().all(|m| m.origin == (0, 0)));
    }

    #[test]
    fn a_surface_of_one_level_is_level_0_alone() {
        let l = Layout::new(TileMode::Standard64K, 4, 300, 200).unwrap();
        assert_eq!(l.mips(1, (300, 200), 1)[0].offset, 0);
        assert_eq!(l.chain_size(1), l.size());
    }

    #[test]
    fn linear_levels_go_from_the_smallest_up() {
        let l = Layout::new(TileMode::Linear, 4, 64, 64).unwrap();
        let mips = l.mips(3, (64, 64), 1);
        assert_eq!(mips[2].offset, 0);
        assert_eq!(mips[1].offset, 16 * 256);
        assert_eq!(mips[0].offset, 16 * 256 + 32 * 256);
        assert_eq!(mips[1].layout.pitch, 64);
    }

    #[test]
    fn linear_rows_are_padded_to_256_bytes() {
        let l = Layout::new(TileMode::Linear, 4, 100, 10).unwrap();
        assert_eq!(l.pitch, 128);
        assert_eq!(l.offset(1, 2), (2 * 128 + 1) * 4);
    }
}
