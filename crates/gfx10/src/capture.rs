//! A translation request saved to a file, so programs a title runs can be
//! translated again offline (`PS5SLOP_SHADER_DUMP`, the `captures` example).
//! Files are little-endian dwords, written by [`Capture::to_bytes`]; files of
//! older versions lack the fields added after them (`PGM_RSRC1` becomes
//! [`DEFAULT_RSRC1`]).

use crate::translate::{Callee, DEFAULT_RSRC1, Mesh, Request, Stage, program};

const MAGIC: u32 = u32::from_le_bytes(*b"g10c");
const VERSION: u32 = 12;
const GRID_BOUNDS: u32 = 1;
const DROP_DEPTH_EXPORT: u32 = 2;
const DUAL_SOURCE: u32 = 4;

/// A request with its own storage.
#[derive(Clone, Debug)]
pub struct Capture {
    pub stage: Stage,
    pub user_sgprs: u32,
    pub ngg: bool,
    pub ps_input_ena: u32,
    pub ps_input_addr: u32,
    pub ps_input_cntl: Vec<u32>,
    pub col_format: u32,
    pub compute_rsrc2: u32,
    pub threads: [u32; 3],
    pub mesh: Mesh,
    pub uint_textures: u32,
    pub sint_textures: u32,
    pub coded_textures: u32,
    pub address: u64,
    pub grid_bounds: bool,
    pub drop_depth_export: bool,
    pub half_targets: u8,
    pub srgb_targets: u8,
    pub dual_source: bool,
    pub blends: [u32; 8],
    pub logic_op: u8,
    pub logic_targets: [u8; 8],
    pub swizzles: [u8; 8],
    pub packed_targets: [u16; 8],
    pub scratch_wave_bytes: u32,
    pub rsrc1: u32,
    pub calls: Vec<Callee>,
    pub code: Vec<u32>,
}

impl Capture {
    /// Copies `req`, keeping the code up to the end of the program when it
    /// decodes and calls no code after its end.
    pub fn new(req: &Request) -> Self {
        let calls = |p: &[(usize, crate::Inst)]| {
            p.iter()
                .any(|(_, i)| matches!(i.op.name, "s_call_b64" | "s_swappc_b64" | "s_setpc_b64"))
        };
        let len = program(req.code)
            .ok()
            .filter(|p| !calls(p))
            .and_then(|p| p.last().map(|(pc, i)| pc + i.words as usize))
            .unwrap_or(req.code.len());
        Capture {
            stage: req.stage,
            user_sgprs: req.user_sgprs,
            ngg: req.ngg,
            ps_input_ena: req.ps_input_ena,
            ps_input_addr: req.ps_input_addr,
            ps_input_cntl: req.ps_input_cntl.to_vec(),
            col_format: req.col_format,
            compute_rsrc2: req.compute_rsrc2,
            threads: req.threads,
            mesh: req.mesh,
            uint_textures: req.uint_textures,
            sint_textures: req.sint_textures,
            coded_textures: req.coded_textures,
            address: req.address,
            grid_bounds: req.grid_bounds,
            drop_depth_export: req.drop_depth_export,
            half_targets: req.half_targets,
            srgb_targets: req.srgb_targets,
            dual_source: req.dual_source,
            blends: req.blends,
            logic_op: req.logic_op,
            logic_targets: req.logic_targets,
            swizzles: req.swizzles,
            packed_targets: req.packed_targets,
            scratch_wave_bytes: req.scratch_wave_bytes,
            rsrc1: req.rsrc1,
            calls: req.calls.to_vec(),
            code: req.code[..len.min(req.code.len())].to_vec(),
        }
    }

    pub fn request(&self) -> Request<'_> {
        Request {
            stage: self.stage,
            code: &self.code,
            address: self.address,
            user_sgprs: self.user_sgprs,
            ngg: self.ngg,
            ps_input_ena: self.ps_input_ena,
            ps_input_addr: self.ps_input_addr,
            ps_input_cntl: &self.ps_input_cntl,
            col_format: self.col_format,
            compute_rsrc2: self.compute_rsrc2,
            threads: self.threads,
            mesh: self.mesh,
            uint_textures: self.uint_textures,
            sint_textures: self.sint_textures,
            coded_textures: self.coded_textures,
            grid_bounds: self.grid_bounds,
            drop_depth_export: self.drop_depth_export,
            half_targets: self.half_targets,
            srgb_targets: self.srgb_targets,
            dual_source: self.dual_source,
            blends: self.blends,
            logic_op: self.logic_op,
            logic_targets: self.logic_targets,
            swizzles: self.swizzles,
            packed_targets: self.packed_targets,
            probe_flat: false,
            filter_bilinear: false,
            plain_float: 0,
            scratch_wave_bytes: self.scratch_wave_bytes,
            rsrc1: self.rsrc1,
            calls: &self.calls,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let stage = match self.stage {
            Stage::Vertex => 0,
            Stage::Pixel => 1,
            Stage::Compute => 2,
            Stage::Mesh => 3,
        };
        let mut w = vec![
            MAGIC,
            VERSION,
            stage,
            self.user_sgprs,
            self.ngg as u32,
            self.ps_input_ena,
            self.ps_input_addr,
            self.col_format,
            self.compute_rsrc2,
            self.threads[0],
            self.threads[1],
            self.threads[2],
            self.uint_textures,
            self.sint_textures,
            self.ps_input_cntl.len() as u32,
        ];
        w.extend_from_slice(&self.ps_input_cntl);
        let flags = if self.grid_bounds { GRID_BOUNDS } else { 0 }
            | if self.drop_depth_export {
                DROP_DEPTH_EXPORT
            } else {
                0
            }
            | if self.dual_source { DUAL_SOURCE } else { 0 }
            | u32::from(self.half_targets) << 8
            | u32::from(self.srgb_targets) << 16;
        w.extend_from_slice(&[self.address as u32, (self.address >> 32) as u32, flags]);
        let (t, s) = (self.logic_targets, self.swizzles);
        w.extend_from_slice(&[
            self.logic_op.into(),
            u32::from_le_bytes([t[0], t[1], t[2], t[3]]),
            u32::from_le_bytes([t[4], t[5], t[6], t[7]]),
            u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            u32::from_le_bytes([s[4], s[5], s[6], s[7]]),
            self.scratch_wave_bytes,
            self.coded_textures,
            self.rsrc1,
        ]);
        w.extend_from_slice(&self.blends);
        let p = self.packed_targets;
        w.extend((0..4).map(|i| u32::from(p[2 * i]) | u32::from(p[2 * i + 1]) << 16));
        w.push(self.calls.len() as u32);
        for c in &self.calls {
            w.extend([
                c.site as u32,
                (c.site >> 32) as u32,
                c.target as u32,
                (c.target >> 32) as u32,
                c.code.len() as u32,
            ]);
            w.extend_from_slice(&c.code);
        }
        let m = self.mesh;
        w.extend([
            u32::from(m.vertices) | u32::from(m.primitives) << 16,
            m.topology.into(),
            m.lds_bytes,
        ]);
        w.extend_from_slice(&self.code);
        w.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let w: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        if w.len() < 13 || w[0] != MAGIC || !(1..=VERSION).contains(&w[1]) {
            return None;
        }
        let stage = match w[2] {
            0 => Stage::Vertex,
            1 => Stage::Pixel,
            2 => Stage::Compute,
            3 => Stage::Mesh,
            _ => return None,
        };
        let (masks, at) = if w[1] >= 2 {
            ([*w.get(12)?, *w.get(13)?], 14)
        } else {
            ([0, 0], 12)
        };
        let n = *w.get(at)? as usize;
        let cntl = w.get(at + 1..at + 1 + n)?.to_vec();
        let mut code_at = at + 1 + n;
        let mut address = 0;
        if w[1] >= 3 {
            address = *w.get(code_at)? as u64 | (*w.get(code_at + 1)? as u64) << 32;
            code_at += 2;
        }
        let mut flags = 0;
        if w[1] >= 4 {
            flags = *w.get(code_at)?;
            code_at += 1;
        }
        let (mut logic_op, mut logic_targets, mut swizzles) = (0xCC, [0; 8], [0; 8]);
        if w[1] >= 5 {
            let bytes = |at: usize| -> Option<[u8; 8]> {
                let (lo, hi) = (w.get(at)?.to_le_bytes(), w.get(at + 1)?.to_le_bytes());
                Some([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
            };
            logic_op = *w.get(code_at)? as u8;
            logic_targets = bytes(code_at + 1)?;
            swizzles = bytes(code_at + 3)?;
            code_at += 5;
        }
        let mut scratch_wave_bytes = 0;
        if w[1] >= 6 {
            scratch_wave_bytes = *w.get(code_at)?;
            code_at += 1;
        }
        let mut coded = 0;
        if w[1] >= 7 {
            coded = *w.get(code_at)?;
            code_at += 1;
        }
        let mut rsrc1 = DEFAULT_RSRC1;
        if w[1] >= 8 {
            rsrc1 = *w.get(code_at)?;
            code_at += 1;
        }
        let mut blends = [0; 8];
        if w[1] >= 9 {
            blends.copy_from_slice(w.get(code_at..code_at + 8)?);
            code_at += 8;
        }
        let mut packed_targets = [0; 8];
        if w[1] >= 10 {
            for (i, &x) in w.get(code_at..code_at + 4)?.iter().enumerate() {
                packed_targets[2 * i] = x as u16;
                packed_targets[2 * i + 1] = (x >> 16) as u16;
            }
            code_at += 4;
        }
        let mut calls = Vec::new();
        if w[1] >= 11 {
            let n = *w.get(code_at)?;
            code_at += 1;
            for _ in 0..n {
                let h = w.get(code_at..code_at + 5)?;
                let len = h[4] as usize;
                calls.push(Callee {
                    site: h[0] as u64 | (h[1] as u64) << 32,
                    target: h[2] as u64 | (h[3] as u64) << 32,
                    code: w.get(code_at + 5..code_at + 5 + len)?.to_vec(),
                });
                code_at += 5 + len;
            }
        }
        let mut mesh = Mesh::default();
        if w[1] >= 12 {
            let m = w.get(code_at..code_at + 3)?;
            mesh = Mesh {
                vertices: m[0] as u16,
                primitives: (m[0] >> 16) as u16,
                topology: m[1] as u8,
                lds_bytes: m[2],
            };
            code_at += 3;
        }
        Some(Capture {
            stage,
            user_sgprs: w[3],
            ngg: w[4] != 0,
            ps_input_ena: w[5],
            ps_input_addr: w[6],
            col_format: w[7],
            compute_rsrc2: w[8],
            threads: [w[9], w[10], w[11]],
            mesh,
            uint_textures: masks[0],
            sint_textures: masks[1],
            coded_textures: coded,
            ps_input_cntl: cntl,
            address,
            grid_bounds: flags & GRID_BOUNDS != 0,
            drop_depth_export: flags & DROP_DEPTH_EXPORT != 0,
            half_targets: (flags >> 8) as u8,
            srgb_targets: (flags >> 16) as u8,
            dual_source: flags & DUAL_SOURCE != 0,
            blends,
            logic_op,
            logic_targets,
            swizzles,
            packed_targets,
            scratch_wave_bytes,
            rsrc1,
            calls,
            code: w[code_at..].to_vec(),
        })
    }
}
