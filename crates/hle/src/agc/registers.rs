//! Command-buffer builders that set registers (`sceAgc*Set*Register*`).
//! Ported from SharpEmu's `AgcExports.RegisterPackets.cs` (GPL-2.0-or-later)
//! and changed where console measurements show other output; see
//! `docs/gpu.md`.

use super::cmdbuf::{emit, pm4, read_dwords};
use super::packets::size;
use super::ret;
use crate::{CallError, Env, Function, Outcome};

pub(super) static FUNCTIONS: &[Function] = &[
    f("sceAgcDcbSetShRegisterDirect", |e| direct(e, IT_SET_SH_REG)),
    f("sceAgcDcbSetShRegisterDirectGetSize", size::<3>),
    f("sceAgcDcbSetCxRegisterDirect", |e| {
        direct(e, IT_SET_CONTEXT_REG)
    }),
    f("sceAgcDcbSetCxRegisterDirectGetSize", size::<3>),
    f("sceAgcDcbSetUcRegisterDirect", |e| {
        direct(e, IT_SET_UCONFIG_REG)
    }),
    f("sceAgcDcbSetUcRegisterDirectGetSize", size::<3>),
    f("sceAgcDcbSetCfRegisterDirect", |e| {
        direct(e, IT_SET_CONFIG_REG)
    }),
    f("sceAgcDcbSetShRegistersIndirect", |e| {
        indirect(e, IT_SET_SH_REG_INDIRECT)
    }),
    f("sceAgcDcbSetShRegistersIndirectGetSize", size::<5>),
    f("sceAgcDcbSetCxRegistersIndirect", |e| {
        indirect(e, IT_SET_CONTEXT_REG_INDIRECT)
    }),
    f("sceAgcDcbSetCxRegistersIndirectGetSize", size::<5>),
    f("sceAgcDcbSetUcRegistersIndirect", |e| {
        indirect(e, IT_SET_UCONFIG_REG_INDIRECT)
    }),
    f("sceAgcDcbSetUcRegistersIndirectGetSize", size::<5>),
    f("sceAgcCbSetShRegisterRangeDirect", |e| {
        range(e, IT_SET_SH_REG)
    }),
    f("sceAgcCbSetShRegisterRangeDirectGetSize", range_size),
    f("sceAgcCbSetUcRegisterRangeDirect", |e| {
        range(e, IT_SET_UCONFIG_REG)
    }),
    f("sceAgcCbSetUcRegisterRangeDirectGetSize", range_size),
    f("sceAgcDcbSetCfRegisterRangeDirect", |e| {
        range(e, IT_SET_CONFIG_REG)
    }),
    f("sceAgcCbSetShRegistersDirect", |e| pairs(e, IT_SET_SH_REG)),
    f("sceAgcCbSetShRegistersDirectGetSize", pairs_size),
    f("sceAgcCbSetUcRegistersDirect", |e| {
        pairs(e, IT_SET_UCONFIG_REG)
    }),
    f("sceAgcCbSetUcRegistersDirectGetSize", pairs_size),
];

pub(super) static STUBS: &[Function] = &[];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

const IT_SET_SH_REG_INDIRECT: u32 = 0x63;
const IT_SET_UCONFIG_REG_INDIRECT: u32 = 0x64;
const IT_SET_CONFIG_REG: u32 = 0x68;
const IT_SET_CONTEXT_REG: u32 = 0x69;
const IT_SET_SH_REG: u32 = 0x76;
const IT_SET_UCONFIG_REG: u32 = 0x79;
const IT_SET_CONTEXT_REG_INDIRECT: u32 = 0x9F;

/// The most values one SET_*_REG packet carries.
const MAX_RANGE: u32 = 0x3FFE;

/// `(buffer, register)`, the register packed as offset (low half) and value
/// (high half).
fn direct(env: &mut Env, op: u32) -> Result<Outcome, CallError> {
    let (buffer, packed) = (env.arg(0), env.arg(1));
    if buffer == 0 {
        return ret(0);
    }
    let p = [pm4(3, op, 0), packed as u32 & 0xFFFF, (packed >> 32) as u32];
    ret(emit(env, buffer, &p)?)
}

/// `(buffer, table, count)`: points the GPU at a table of registers.
fn indirect(env: &mut Env, op: u32) -> Result<Outcome, CallError> {
    let (buffer, table, count) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if buffer == 0 {
        return ret(0);
    }
    let p = [
        pm4(5, op, 0),
        table as u32 & !3,
        (table >> 32) as u32,
        0x8000_0000,
        count & 0x3FFF,
    ];
    ret(emit(env, buffer, &p)?)
}

/// `(buffer, offset, values, count)`: one packet of consecutive registers.
/// A null `values` leaves the payload zeroed for the caller to fill.
fn range(env: &mut Env, op: u32) -> Result<Outcome, CallError> {
    let (buffer, offset, values, n) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3) as u32);
    if buffer == 0 || n > MAX_RANGE {
        return ret(0);
    }
    let mut p = vec![pm4(n + 2, op, 0), offset as u32 & 0xFFFF];
    if values == 0 {
        p.resize(n as usize + 2, 0);
    } else {
        p.extend(read_dwords(env, values, n as usize)?);
    }
    ret(emit(env, buffer, &p)?)
}

fn range_size(env: &mut Env) -> Result<Outcome, CallError> {
    ret(8 + 4 * (env.arg(0) as u32 as u64))
}

/// `(buffer, pairs, count)`: one 3-dword packet per `{offset, value}` pair,
/// in the given order, as the console's size answer implies.
fn pairs(env: &mut Env, op: u32) -> Result<Outcome, CallError> {
    let (buffer, table, n) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if buffer == 0 || table == 0 || n == 0 || n > 0x1000 {
        return ret(0);
    }
    let pairs = read_dwords(env, table, 2 * n as usize)?;
    let p: Vec<u32> = pairs
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|&[o, v]| [pm4(3, op, 0), o & 0xFFFF, v])
        .collect();
    ret(emit(env, buffer, &p)?)
}

fn pairs_size(env: &mut Env) -> Result<Outcome, CallError> {
    ret(12 * (env.arg(0) as u32 as u64))
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
        let n = golden::builders(&names(), &[]);
        eprintln!("registers: checked {n} measured builder calls");
    }

    #[test]
    fn sizes_match_the_console() {
        let n = golden::sizes(&names());
        eprintln!("registers: checked {n} measured sizes");
    }
}
