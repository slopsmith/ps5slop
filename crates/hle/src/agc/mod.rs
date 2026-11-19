//! libSceAgc and libSceAgcDriver, the PS5's graphics libraries, in HLE.
//! Ported from SharpEmu (GPL-2.0-or-later) and checked against console
//! measurements where we have them; see `docs/gpu.md`.

mod cmdbuf;
mod control;
mod defaults;
mod dma;
mod draw;
mod driver;
mod events;
#[cfg(test)]
mod oracle;
mod packets;
mod registers;
mod shader;
mod submit;
mod sync;

use std::collections::HashMap;
use std::sync::Mutex;

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};
use defaults::CompactDefaults;

/// Error codes as SharpEmu uses them (unverified).
const ERROR_INVALID_ARGUMENT: u64 = 0x8002_0003;

/// Guest addresses of the register-default blocks made so far, by version:
/// (public, internal).
#[derive(Default)]
pub struct Agc {
    defaults: Mutex<HashMap<u32, (u64, u64)>>,
    /// Shader headers by code address, from `sceAgcCreateShader`.
    shaders: Mutex<HashMap<u64, u64>>,
    registry: Mutex<driver::Registry>,
    gpu: submit::Gpu,
}

/// Every libSceAgc and libSceAgcDriver function, real ones first.
pub fn functions() -> impl Iterator<Item = &'static Function> {
    [
        FUNCTIONS,
        driver::FUNCTIONS,
        submit::FUNCTIONS,
        packets::FUNCTIONS,
        control::FUNCTIONS,
        dma::FUNCTIONS,
        draw::FUNCTIONS,
        events::FUNCTIONS,
        registers::FUNCTIONS,
        sync::FUNCTIONS,
    ]
    .into_iter()
    .flatten()
}

/// The placeholders among them.
pub fn stubs() -> impl Iterator<Item = &'static Function> {
    [
        STUBS,
        driver::STUBS,
        packets::STUBS,
        control::STUBS,
        dma::STUBS,
        draw::STUBS,
        events::STUBS,
        registers::STUBS,
        sync::STUBS,
    ]
    .into_iter()
    .flatten()
}

pub static FUNCTIONS: &[Function] = &[
    // AGC init; SharpEmu binds this NID as its `sceAgcInit`.
    f("#23LRUSvYu1M", init),
    f("sceAgcGetRegisterDefaults2", register_defaults),
    f(
        "sceAgcGetRegisterDefaults2Internal",
        register_defaults_internal,
    ),
    f("sceAgcCreateShader", shader::create_shader),
    // SharpEmu's `sceAgcCreateInterpolantMapping`.
    f("#HV4j+E0MBHE", shader::create_interpolant_mapping),
    // SharpEmu's `sceAgcCreateInterpolantMapping2`.
    f("#dbOlWdppb4o", shader::create_interpolant_mapping2),
    f("sceAgcCreatePrimState", shader::create_prim_state),
    f("sceAgcUpdatePrimState", shader::update_prim_state),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[f("sceAgcGetIsTrinityMode", ok)];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

fn ok(_: &mut Env) -> Result<Outcome, CallError> {
    ret(0)
}

/// The defaults for register-default version `version` (public, internal);
/// 12 uses 10's and 13 uses 11's, as SharpEmu does.
fn defaults_for(version: u32) -> Option<(&'static CompactDefaults, &'static CompactDefaults)> {
    match version {
        10 | 12 => Some((&defaults::PUBLIC_V10, &defaults::INTERNAL_V10)),
        11 | 13 => Some((&defaults::PUBLIC_V11, &defaults::INTERNAL_V11)),
        _ => None,
    }
}

/// `sceAgcInit(state, version)`: writes the version and a zero word.
fn init(env: &mut Env) -> Result<Outcome, CallError> {
    let (state, version) = (env.arg(0), env.arg(1) as u32);
    if state == 0 || defaults_for(version).is_none() {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    env.mem.write(state, &version.to_le_bytes())?;
    env.mem.write(state + 4, &0u32.to_le_bytes())?;
    ret(0)
}

fn register_defaults(env: &mut Env) -> Result<Outcome, CallError> {
    defaults_block(env, false)
}

fn register_defaults_internal(env: &mut Env) -> Result<Outcome, CallError> {
    defaults_block(env, true)
}

/// `sceAgcGetRegisterDefaults2(version)` and its internal twin: the address
/// of a block laid out by [`build`], made once per version; 0 if unknown.
fn defaults_block(env: &mut Env, internal: bool) -> Result<Outcome, CallError> {
    let version = env.arg(0) as u32;
    let Some((public, private)) = defaults_for(version) else {
        return ret(0);
    };
    let mut made = lock(&env.kernel.agc.defaults);
    let addresses = match made.get(&version) {
        Some(&a) => a,
        None => {
            let a = (place(env, public)?, place(env, private)?);
            made.insert(version, a);
            a
        }
    };
    ret(if internal { addresses.1 } else { addresses.0 })
}

/// Copies `d`'s block into guest memory and returns its address.
fn place(env: &Env, d: &CompactDefaults) -> Result<u64, CallError> {
    let len = build(d, 0).len() as u64;
    let at = env
        .kernel
        .heap
        .alloc_aligned(env.mem, len, 0x1000)
        .ok_or(CallError::Abort(
            "no guest memory for AGC register defaults",
        ))?;
    env.mem.write(at, &build(d, at))?;
    Ok(at)
}

/// The register-default block for `d` at guest address `base`, laid out as
/// SharpEmu lays it out.
fn build(d: &CompactDefaults, base: u64) -> Vec<u8> {
    const HEADER: usize = 0x40;
    let tables = [
        (d.table0, d.table0_ptr),
        (d.table1, d.table1_ptr),
        (d.table2, d.table2_ptr),
        (d.table3, d.table3_ptr),
    ];
    let mut ptr_at = [0usize; 4];
    let mut at = HEADER;
    for (k, (_, ptrs)) in tables.iter().enumerate() {
        ptr_at[k] = at;
        at += ptrs.len() * 8;
    }
    let mut regs_at = [0usize; 4];
    for (k, (regs, _)) in tables.iter().enumerate() {
        regs_at[k] = at;
        at += regs.len() * 4;
    }
    let types_at = at;
    let mut b = vec![0u8; types_at + d.types.len() * 4];
    let put32 =
        |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let put64 =
        |b: &mut Vec<u8>, at: usize, v: u64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    for (k, (regs, ptrs)) in tables.iter().enumerate() {
        let table = if ptrs.is_empty() {
            0
        } else {
            base + ptr_at[k] as u64
        };
        put64(&mut b, 8 * k, table);
        put32(&mut b, 0x20 + 4 * k, (regs.len() / 2) as u32);
        for (i, &p) in ptrs.iter().enumerate() {
            put64(
                &mut b,
                ptr_at[k] + 8 * i,
                base + (regs_at[k] + p as usize * 8) as u64,
            );
        }
        for (i, &r) in regs.iter().enumerate() {
            put32(&mut b, regs_at[k] + 4 * i, r);
        }
    }
    put64(&mut b, 0x30, base + types_at as u64);
    put32(&mut b, 0x38, (d.types.len() / 3) as u32);
    for (i, &t) in d.types.iter().enumerate() {
        put32(&mut b, types_at + 4 * i, t);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tables_are_well_formed() {
        for d in [
            &defaults::PUBLIC_V10,
            &defaults::INTERNAL_V10,
            &defaults::PUBLIC_V11,
            &defaults::INTERNAL_V11,
        ] {
            assert_eq!(d.types.len() % 3, 0);
            for (regs, ptrs) in [
                (d.table0, d.table0_ptr),
                (d.table1, d.table1_ptr),
                (d.table2, d.table2_ptr),
                (d.table3, d.table3_ptr),
            ] {
                assert_eq!(regs.len() % 2, 0);
                assert_eq!(ptrs.is_empty(), regs.is_empty());
                assert!(ptrs.iter().all(|&p| (p as usize) < regs.len() / 2));
            }
        }
    }

    #[test]
    fn blocks_point_inside_themselves() {
        let base = 0x1_0000;
        let b = build(&defaults::PUBLIC_V10, base);
        let read64 = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        let end = base + b.len() as u64;
        for k in 0..4 {
            let t = read64(8 * k);
            assert!(t == 0 || (base..end).contains(&t));
        }
        assert!((base..end).contains(&read64(0x30)));
    }
}
