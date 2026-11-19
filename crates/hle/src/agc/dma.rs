//! DMA and copy packets. Ported from SharpEmu's `AgcExports.DmaPackets.cs`
//! (GPL-2.0-or-later); DMA_DATA follows the console measurements instead,
//! which show the hardware packet. See `docs/gpu.md`.

use super::cmdbuf::{arg_n, emit, pm4};
use super::packets::size;
use super::ret;
use crate::{CallError, Env, Function, Outcome};

pub(super) static FUNCTIONS: &[Function] = &[
    f("sceAgcDcbCopyData", |e| copy_data(e, false)),
    f("sceAgcDcbCopyDataGetSize", size::<6>),
    f("sceAgcAcbCopyData", |e| copy_data(e, true)),
    f("sceAgcAcbCopyDataGetSize", size::<6>),
    f("sceAgcDcbDmaData", dcb_dma_data),
    f("sceAgcDcbDmaDataGetSize", size::<7>),
    f("sceAgcAcbDmaData", acb_dma_data),
    f("sceAgcAcbDmaDataGetSize", size::<7>),
];

pub(super) static STUBS: &[Function] = &[];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

const IT_COPY_DATA: u32 = 0x40;
const IT_DMA_DATA: u32 = 0x50;

/// `(buffer, dst_sel, dst_cache, destination, src_sel, src_cache, source,
/// count_sel, confirm)`. `Dcb` selectors carry the engine in bit 0.
fn copy_data(env: &mut Env, raw: bool) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let (source, count, confirm) = (arg_n(env, 6)?, arg_n(env, 7)?, arg_n(env, 8)?);
    if buffer == 0 {
        return ret(0);
    }
    let (dst, src) = (env.arg(1) as u32 & 0xFF, env.arg(4) as u32 & 0xFF);
    let selectors = if raw {
        (src & 0xF) | ((dst & 0xF) << 8)
    } else {
        ((src >> 1) & 0xF) | (((dst >> 1) & 0xF) << 8) | ((src & 1) << 30)
    };
    let destination = env.arg(3);
    let p = [
        pm4(6, IT_COPY_DATA, 0),
        selectors
            | ((env.arg(5) as u32 & 3) << 13)
            | ((count as u32 & 1) << 16)
            | ((confirm as u32 & 1) << 20)
            | ((env.arg(2) as u32 & 3) << 25),
        source as u32,
        (source >> 32) as u32,
        destination as u32,
        (destination >> 32) as u32,
    ];
    ret(emit(env, buffer, &p)?)
}

/// The fields of a DMA_DATA. Selectors keep the hardware selector in bits
/// 0-1 and the address-space flag in bit 2, as the console shows.
struct Dma {
    engine: u32,
    dst_sel: u32,
    dst_cache: u32,
    destination: u64,
    src_sel: u32,
    src_cache: u32,
    source: u64,
    bytes: u32,
    wait: u32,
    sync: u32,
}

fn dma_packet(d: &Dma) -> [u32; 7] {
    [
        pm4(7, IT_DMA_DATA, 0),
        (d.engine & 1)
            | ((d.src_cache & 3) << 13)
            | ((d.dst_sel & 3) << 20)
            | ((d.dst_cache & 3) << 25)
            | ((d.src_sel & 3) << 29)
            | ((d.sync & 1) << 31),
        d.source as u32,
        (d.source >> 32) as u32,
        d.destination as u32,
        (d.destination >> 32) as u32,
        (d.bytes & 0x03FF_FFFF)
            | (((d.src_sel >> 2) & 1) << 26)
            | (((d.dst_sel >> 2) & 1) << 27)
            | ((d.wait & 1) << 30),
    ]
}

/// Arguments: engine, dst_sel, dst_cache, destination, src_sel, src_cache,
/// source, bytes, wait, confirm, block_engine (the last three unverified).
fn dcb_dma_data(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let d = Dma {
        engine: env.arg(1) as u32,
        dst_sel: env.arg(2) as u32,
        dst_cache: env.arg(3) as u32,
        destination: env.arg(4),
        src_sel: env.arg(5) as u32,
        src_cache: arg_n(env, 6)? as u32,
        source: arg_n(env, 7)?,
        bytes: arg_n(env, 8)? as u32,
        wait: arg_n(env, 9)? as u32,
        sync: arg_n(env, 11)? as u32,
    };
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, &dma_packet(&d))?)
}

/// `sceAgcAcbDmaData(buffer, dst_sel, dst_cache, destination, src_sel,
/// src_cache, source, bytes, wait, confirm)`.
fn acb_dma_data(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let d = Dma {
        engine: 0,
        dst_sel: env.arg(1) as u32,
        dst_cache: env.arg(2) as u32,
        destination: env.arg(3),
        src_sel: env.arg(4) as u32,
        src_cache: env.arg(5) as u32,
        source: arg_n(env, 6)?,
        bytes: arg_n(env, 7)? as u32,
        wait: arg_n(env, 8)? as u32,
        sync: 0,
    };
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, &dma_packet(&d))?)
}

#[cfg(test)]
mod tests {
    use super::super::packets::golden;
    use super::*;

    fn names() -> Vec<&'static str> {
        FUNCTIONS.iter().map(|f| f.name).collect()
    }

    #[test]
    fn builders_match_the_console() {
        let mut ours = names();
        ours.retain(|n| *n != "sceAgcDcbDmaData");
        let n = golden::builders(&ours, &[]);
        eprintln!("dma: checked {n} measured builder calls");
    }

    #[test]
    fn sizes_match_the_console() {
        let n = golden::sizes(&names());
        eprintln!("dma: checked {n} measured sizes");
    }
}
