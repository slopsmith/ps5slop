//! A translation request saved to a file, so programs a title runs can be
//! translated again offline (`PS5SLOP_SHADER_DUMP`, the `captures` example).
//!
//! The file is little-endian dwords: magic, version, stage, user SGPR count,
//! NGG flag, `SPI_PS_INPUT_ENA`, `SPI_PS_INPUT_ADDR`, `SPI_SHADER_COL_FORMAT`,
//! `COMPUTE_PGM_RSRC2`, the three thread counts, the unsigned and signed
//! integer texture masks, the number of `SPI_PS_INPUT_CNTL` values and those
//! values, the program address (two dwords), the flags (bit 0 grid bounds,
//! bit 1 dropped depth export), then the code. Version 1 files have no
//! texture masks, version 2 files no address, version 3 files no flags.

use crate::translate::{Request, Stage, program};

const MAGIC: u32 = u32::from_le_bytes(*b"g10c");
const VERSION: u32 = 4;
const GRID_BOUNDS: u32 = 1;
const DROP_DEPTH_EXPORT: u32 = 2;

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
    pub uint_textures: u32,
    pub sint_textures: u32,
    pub address: u64,
    pub grid_bounds: bool,
    pub drop_depth_export: bool,
    pub code: Vec<u32>,
}

impl Capture {
    /// Copies `req`, keeping the code up to the end of the program when it decodes.
    pub fn new(req: &Request) -> Self {
        let len = program(req.code)
            .ok()
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
            uint_textures: req.uint_textures,
            sint_textures: req.sint_textures,
            address: req.address,
            grid_bounds: req.grid_bounds,
            drop_depth_export: req.drop_depth_export,
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
            uint_textures: self.uint_textures,
            sint_textures: self.sint_textures,
            grid_bounds: self.grid_bounds,
            drop_depth_export: self.drop_depth_export,
            probe_flat: false,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let stage = match self.stage {
            Stage::Vertex => 0,
            Stage::Pixel => 1,
            Stage::Compute => 2,
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
            };
        w.extend_from_slice(&[self.address as u32, (self.address >> 32) as u32, flags]);
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
        Some(Capture {
            stage,
            user_sgprs: w[3],
            ngg: w[4] != 0,
            ps_input_ena: w[5],
            ps_input_addr: w[6],
            col_format: w[7],
            compute_rsrc2: w[8],
            threads: [w[9], w[10], w[11]],
            uint_textures: masks[0],
            sint_textures: masks[1],
            ps_input_cntl: cntl,
            address,
            grid_bounds: flags & GRID_BOUNDS != 0,
            drop_depth_export: flags & DROP_DEPTH_EXPORT != 0,
            code: w[code_at..].to_vec(),
        })
    }
}
