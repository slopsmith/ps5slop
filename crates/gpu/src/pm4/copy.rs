//! Memory and GDS copy/fill packets (`DMA_DATA`, `COPY_DATA`).

use super::*;

impl Processor {
    /// COPY_DATA from an immediate or memory into memory. Selectors are the
    /// compute form; the graphics form has the same low bits (Mesa's `sid.h`).
    pub(super) fn copy_data(&mut self, host: &dyn Host, p: &[u32]) {
        let control = p[0];
        let (src_sel, dst_sel) = (control & 0xF, (control >> 8) & 0xF);
        let wide = control & (1 << 16) != 0;
        let n = if wide { 2 } else { 1 };
        if !matches!(dst_sel, 1 | 2 | 5) {
            unhandled(format_args!(
                "COPY_DATA from select {src_sel} to select {dst_sel} at {:#x}",
                p[3]
            ));
            return;
        }
        let dest = addr(p[3], p[4]);
        let values = match src_sel {
            5 => vec![p[1], p[2]][..n].to_vec(),
            1 | 2 => {
                self.wait_gpu(host, "COPY_DATA");
                read_dwords(host, addr(p[1], p[2]) & !3, n).unwrap_or_else(|_| vec![0; n])
            }
            9 => {
                let clock = gpu_clock();
                vec![clock as u32, (clock >> 32) as u32][..n].to_vec()
            }
            _ => return,
        };
        self.write_values(host, dest, &values, false);
    }

    /// `DMA_DATA`: copies or fills between memory, the GDS and the packet's
    /// own data dword. Done on the CPU in command order, after the GPU work
    /// before it (which may write the source) has finished.
    pub(super) fn dma_data(&mut self, host: &dyn Host, p: &[u32]) {
        let control = p[0];
        let src_sel = (control >> 29) & 3;
        let dst_sel = (control >> 20) & 3;
        let command = p[5];
        let bytes = (command & 0x3FF_FFFF) as usize;
        let (src_register, dst_register) = (command >> 26 & 1 != 0, command >> 27 & 1 != 0);
        let (src_fixed, dst_fixed) = (command >> 28 & 1 != 0, command >> 29 & 1 != 0);
        if bytes == 0 || bytes > DMA_LIMIT || dst_sel == 2 || src_register || dst_register {
            return;
        }
        let src = addr(p[1], p[2]);
        let dst = addr(p[3], p[4]);
        if src_sel == 2
            && dst_sel != 1
            && bytes.is_multiple_of(4)
            && bytes <= DEFERRED_FILL
            && self.backend.is_some()
            && !diag::config().gpu.sync_releases
        {
            tracing::debug!(
                target: "gpu::dma",
                "fill {dst:#x} {bytes:#x} bytes with {:#x}, queued",
                p[1]
            );
            let n = if dst_fixed { 1 } else { bytes / 4 };
            self.write_values(host, dst, &vec![p[1]; n], dst_fixed);
            return;
        }
        self.wait_gpu(host, "DMA_DATA");
        let mut data = vec![0u8; bytes];
        match src_sel {
            2 => {
                for c in data.chunks_mut(4) {
                    c.copy_from_slice(&p[1].to_le_bytes()[..c.len()]);
                }
            }
            1 => {
                let at = src as usize;
                self.with_gds(&mut |gds| {
                    for (i, b) in data.iter_mut().enumerate() {
                        let a = if src_fixed { at + i % 4 } else { at + i };
                        *b = gds.get(a).copied().unwrap_or(0);
                    }
                });
            }
            _ if src_fixed => {
                let mut word = [0u8; 4];
                host.read(src, &mut word);
                for c in data.chunks_mut(4) {
                    c.copy_from_slice(&word[..c.len()]);
                }
            }
            _ => {
                host.read(src, &mut data);
            }
        }
        tracing::debug!(
            target: "gpu::dma",
            "src_sel {src_sel} {src:#x} dst_sel {dst_sel} {dst:#x} {bytes:#x} bytes, first word {:#x}",
            u32::from_le_bytes([
                data[0],
                *data.get(1).unwrap_or(&0),
                *data.get(2).unwrap_or(&0),
                *data.get(3).unwrap_or(&0)
            ])
        );
        if dst_fixed {
            let n = data.len().min(4);
            data.drain(..data.len() - n);
        }
        if dst_sel == 1 {
            self.with_gds(&mut |gds| {
                let start = (dst as usize).min(gds.len());
                let end = (start + data.len()).min(gds.len());
                gds[start..end].copy_from_slice(&data[..end - start]);
            });
        } else {
            host.write(dst, &data);
        }
    }
}
