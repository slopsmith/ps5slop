//! Semaphores and atomics in command buffers. MemSemaphore and the AMM
//! semaphore memory are ported from SharpEmu (GPL-2.0-or-later); SharpEmu
//! has no atomics, so those follow the console measurements and the public
//! PM4 field layout (unverified where the measurements are silent). See
//! `docs/gpu.md`.

use std::sync::Mutex;

use super::cmdbuf::{arg_n, emit, pm4};
use super::packets::{ACB, marker_begin, size};
use super::{ERROR_INVALID_ARGUMENT, ret};
use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

pub(super) static FUNCTIONS: &[Function] = &[
    f("sceAgcCbMemSemaphore", cb_mem_semaphore),
    f("sceAgcDcbMemSemaphore", mem_semaphore),
    f("sceAgcAcbMemSemaphore", mem_semaphore),
    f("sceAgcDcbAtomicMem", dcb_atomic_mem),
    f("sceAgcDcbAtomicMemGetSize", size::<9>),
    f("sceAgcAcbAtomicMem", acb_atomic_mem),
    f("sceAgcAcbAtomicMemGetSize", size::<9>),
    f("sceAgcDcbAtomicGds", dcb_atomic_gds),
    f("sceAgcDcbAtomicGdsGetSize", size::<11>),
    f("sceAgcAcbAtomicGds", acb_atomic_gds),
    f("sceAgcAcbAtomicGdsGetSize", size::<11>),
    f("sceAgcSetAmmSemaphoreMemory", set_amm_semaphore_memory),
    f("sceAgcGetSemaphoreLabel", semaphore_label),
];

pub(super) static STUBS: &[Function] = &[];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

const IT_ATOMIC_GDS: u32 = 0x1D;
const IT_ATOMIC_MEM: u32 = 0x1E;
const IT_MEM_SEMAPHORE: u32 = 0x39;

/// The AMM semaphore block (base, bytes). One per process, not per kernel,
/// because `Agc` is not ours to extend.
static AMM: Mutex<Option<(u64, u64)>> = Mutex::new(None);
const AMM_ALIGN: u64 = 0x4000;
const AMM_SLOT: u64 = 32;

/// MEM_SEMAPHORE; `op` and `signal` fill the two control bits the console
/// showed (bits 30-31 and 20).
fn semaphore(address: u64, op: u64, signal: u64) -> [u32; 4] {
    [
        pm4(4, IT_MEM_SEMAPHORE, 0),
        address as u32 & !7,
        (address >> 32) as u32,
        ((op as u32 & 3) << 30) | ((signal as u32 & 1) << 20),
    ]
}

/// `sceAgcCbMemSemaphore(buffer, address, op, signal, value)`.
fn cb_mem_semaphore(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let p = semaphore(env.arg(1), env.arg(2), env.arg(3));
    ret(emit(env, buffer, &p)?)
}

/// `sceAgc{Dcb,Acb}MemSemaphore(buffer, address, op, signal, value)`: the
/// semaphore after a trace marker naming its address.
fn mem_semaphore(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, address) = (env.arg(0), env.arg(1));
    if buffer == 0 {
        return ret(0);
    }
    let mut p = marker_begin(ACB, 0xC902_0000, address).to_vec();
    p.extend(semaphore(address, env.arg(2), env.arg(3)));
    ret(emit(env, buffer, &p)?)
}

/// ATOMIC_MEM: the control word, then address, source, compare value and
/// loop interval.
fn atomic_mem(control: u32, address: u64, source: u64, compare: u64, interval: u32) -> [u32; 9] {
    [
        pm4(9, IT_ATOMIC_MEM, 0),
        control,
        address as u32,
        (address >> 32) as u32,
        source as u32,
        (source >> 32) as u32,
        compare as u32,
        (compare >> 32) as u32,
        interval,
    ]
}

fn atomic_control(op: u64, command: u64, cache: u64) -> u32 {
    (op as u32 & 0x7F) | ((command as u32 & 0xF) << 8) | ((cache as u32 & 3) << 25)
}

/// `sceAgcDcbAtomicMem(buffer, engine, op, command, cache_policy, address,
/// source, compare, interval)`.
fn dcb_atomic_mem(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let (source, compare, interval) = (arg_n(env, 6)?, arg_n(env, 7)?, arg_n(env, 8)?);
    if buffer == 0 {
        return ret(0);
    }
    let control =
        atomic_control(env.arg(2), env.arg(3), env.arg(4)) | ((env.arg(1) as u32 & 3) << 30);
    let interval = interval.min(0xFFF) as u32;
    let p = atomic_mem(control, env.arg(5), source, compare, interval);
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcAcbAtomicMem(buffer, op, command, address, cache_policy, source,
/// compare)`.
fn acb_atomic_mem(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let compare = arg_n(env, 6)?;
    if buffer == 0 {
        return ret(0);
    }
    let control = atomic_control(env.arg(1), env.arg(2), env.arg(4));
    let p = atomic_mem(control, env.arg(3), env.arg(5), compare, 0);
    ret(emit(env, buffer, &p)?)
}

/// `first` is where `dmode` sits. The console also sets op-dependent bits
/// 16-20 of the control word, which we cannot derive yet.
fn atomic_gds(env: &Env, control: u32, first: usize) -> Result<[u32; 11], CallError> {
    let a = |n| arg_n(env, first + n);
    let (src0, src1) = (a(6)?, a(7)?);
    Ok([
        pm4(11, IT_ATOMIC_GDS, 0),
        control,
        (a(1)? as u32 & 0x3F) | ((a(0)? as u32 & 1) << 8),
        a(2)? as u32 & 0xFFFF,
        a(3)? as u32 & 0xFFFF,
        a(4)? as u32 & 0x00FF_00FF,
        a(5)? as u32,
        src0 as u32,
        (src0 >> 32) as u32,
        src1 as u32,
        (src1 >> 32) as u32,
    ])
}

/// `sceAgcDcbAtomicGds(buffer, engine, op, ?, dmode, increment, base, size,
/// offsets, destination, source0, source1)`.
fn dcb_atomic_gds(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let control = (env.arg(2) as u32 & 0x7F) | ((env.arg(1) as u32 & 3) << 30);
    let p = atomic_gds(env, control, 4)?;
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcAcbAtomicGds(buffer, op, ?, dmode, increment, base, size, offsets,
/// destination, source0, source1)`.
fn acb_atomic_gds(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let p = atomic_gds(env, env.arg(1) as u32 & 0x7F, 3)?;
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcSetAmmSemaphoreMemory(base, size)`: takes a 16 KiB-aligned block
/// of 32-byte semaphore slots, once, and zeroes each slot's counter.
fn set_amm_semaphore_memory(env: &mut Env) -> Result<Outcome, CallError> {
    let (base, size) = (env.arg(0), env.arg(1));
    if base == 0
        || size < AMM_ALIGN
        || base % AMM_ALIGN != 0
        || size % AMM_ALIGN != 0
        || base.checked_add(size).is_none()
    {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    let mut amm = lock(&AMM);
    if amm.is_some() {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    for slot in (0..size).step_by(AMM_SLOT as usize) {
        env.mem.write_u64(base + slot, 0)?;
    }
    *amm = Some((base, size));
    ret(0)
}

/// `sceAgcGetSemaphoreLabel(slot, out)`: the address of an AMM semaphore
/// slot.
fn semaphore_label(env: &mut Env) -> Result<Outcome, CallError> {
    let (slot, out) = (env.arg(0) as u32 as u64, env.arg(1));
    let Some((base, size)) = *lock(&AMM) else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    if out == 0 || slot >= size / AMM_SLOT {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    env.mem.write_u64(out, base + slot * AMM_SLOT)?;
    ret(0)
}

pub(super) fn saved_amm() -> Option<(u64, u64)> {
    *lock(&AMM)
}

pub(super) fn restore_amm(amm: Option<(u64, u64)>) {
    *lock(&AMM) = amm;
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
        ours.retain(|n| !n.contains("AtomicGds"));
        let n = golden::builders(&ours, &[]);
        eprintln!("sync: checked {n} measured builder calls");
    }

    #[test]
    fn sizes_match_the_console() {
        let n = golden::sizes(&names());
        eprintln!("sync: checked {n} measured sizes");
    }
}
