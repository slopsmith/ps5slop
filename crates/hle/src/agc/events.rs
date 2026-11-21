//! Event and end-of-pipe packets (`EventWrite`, `ReleaseMem`, flips, the
//! wait for a display buffer) and the graphics event-queue functions of
//! libSceAgcDriver. Ported from SharpEmu's `AgcExports.Events.cs` and
//! `sceAgcCbReleaseMem` (GPL-2.0-or-later), changed where console
//! measurements (`oracle.rs`) disagree; see `docs/gpu.md`.

use super::cmdbuf::{IT_NOP, emit, pm4};
use super::control::{IT_RELEASE_MEM, arg, read_dword, size};
use super::ret;
use crate::{CallError, Env, Function, Outcome};

const IT_EVENT_WRITE: u32 = 0x46;

/// SharpEmu's tags for packets the console encodes in ways not measured
/// (the GPU thread reads them).
const R_WAIT_FLIP_DONE: u32 = 0x06;
const R_FLIP: u32 = 0x17;

/// `EVFILT_GRAPHICS_CORE`, as the GPU thread triggers it.
const GRAPHICS_FILTER: i16 = -14;
const ERROR_NOT_FOUND: u64 = 0x8002_0002;

pub(super) static FUNCTIONS: &[Function] = &[
    f("sceAgcDcbEventWrite", dcb_event_write),
    f("sceAgcAcbEventWrite", acb_event_write),
    f("sceAgcDcbEventWriteGetSize", event_write_size),
    size!("sceAgcAcbEventWriteGetSize", 8),
    f("sceAgcDcbWaitUntilSafeForRendering", wait_until_safe),
    f("sceAgcAcbWaitUntilSafeForRendering", wait_until_safe),
    f("sceAgcDcbSetFlip", set_flip),
    f("sceAgcAcbSetFlip", set_flip),
    f("sceAgcCbReleaseMem", release_mem),
    size!("sceAgcCbQueueEndOfPipeActionGetSize", 0x20),
    size!("sceAgcDcbQueueEndOfShaderActionGetSize", 0x20),
    size!("sceAgcAcbQueueEndOfShaderActionGetSize", 0x20),
    driver("sceAgcDriverAddEqEvent", add_eq_event),
    driver("sceAgcDriverDeleteEqEvent", delete_eq_event),
    driver("sceAgcDriverGetEqEventType", eq_event_type),
    driver("sceAgcDriverGetEqContextId", eq_context_id),
];

/// Placeholders: the console returned null for the measured (bogus)
/// arguments, so what they write is unknown. Ours write a 2-dword NOP.
pub(super) static STUBS: &[Function] = &[
    f("sceAgcDcbSetWorkloadsActive", nop),
    f("sceAgcAcbSetWorkloadsActive", nop),
    f("sceAgcDcbSetWorkloadComplete", nop),
    f("sceAgcAcbSetWorkloadComplete", nop),
    f("sceAgcDcbSetWorkloadStreamInactive", nop),
    f("sceAgcAcbSetWorkloadStreamInactive", nop),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

const fn driver(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgcDriver",
        name,
        func,
    }
}

fn put(env: &mut Env, packet: &[u32]) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, packet)?)
}

fn nop(env: &mut Env) -> Result<Outcome, CallError> {
    put(env, &[pm4(2, IT_NOP, 0), 0])
}

/// Event types whose EVENT_WRITE carries an address.
fn has_address(event: u32) -> bool {
    matches!(event, 0x38 | 0x39)
}

/// `(buffer, type, address)`: EVENT_WRITE, 4 dwords with an address for
/// the types that take one, else 2.
fn event_write(env: &mut Env, index: impl Fn(u32) -> u32) -> Result<Outcome, CallError> {
    let (event, address) = (env.arg(1) as u32 & 0xFF, env.arg(2));
    if event > 0x3F {
        return ret(0);
    }
    if has_address(event) {
        return put(
            env,
            &[
                pm4(4, IT_EVENT_WRITE, 0),
                event | 0x100,
                address as u32 & !7,
                (address >> 32) as u32,
            ],
        );
    }
    put(
        env,
        &[pm4(2, IT_EVENT_WRITE, 0), event | (index(event) << 8)],
    )
}

fn dcb_event_write(env: &mut Env) -> Result<Outcome, CallError> {
    event_write(env, |e| {
        if matches!(e, 0x07 | 0x0F | 0x10) {
            4
        } else {
            0
        }
    })
}

fn acb_event_write(env: &mut Env) -> Result<Outcome, CallError> {
    event_write(env, |_| 0)
}

/// `(type)`: the bytes `sceAgcDcbEventWrite` writes for it.
fn event_write_size(env: &mut Env) -> Result<Outcome, CallError> {
    ret(match env.arg(0) as u32 & 0xFF {
        e if has_address(e) => 16,
        e if e <= 0x3F => 8,
        _ => 0,
    })
}

/// `(buffer, video out handle, display buffer index)`. The console writes
/// nothing for an unknown handle; we cannot tell, so always write.
fn wait_until_safe(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, index) = (env.arg(1) as u32, env.arg(2) as u32);
    put(
        env,
        &[pm4(7, IT_NOP, R_WAIT_FLIP_DONE), handle, index, 0, 0, 0, 0],
    )
}

/// `(buffer, video out handle, display buffer index, flip mode, flip arg)`.
fn set_flip(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, index, mode) = (env.arg(1) as u32, env.arg(2) as u32, env.arg(3) as u32);
    let flip_arg = env.arg(4);
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let packet = [
        pm4(6, IT_NOP, R_FLIP),
        handle,
        index,
        mode,
        flip_arg as u32,
        (flip_arg >> 32) as u32,
    ];
    let at = emit(env, buffer, &packet)?;
    env.kernel.agc.gpu.flip_built(at);
    ret(at)
}

/// RELEASE_MEM. Args: event, gcr, destination, cache policy, address, data
/// selection, data, gds offset, gds size, interrupt, interrupt context.
fn release_mem(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(0) == 0 {
        return ret(0);
    }
    let a = (0..12)
        .map(|n| arg(env, n))
        .collect::<Result<Vec<_>, _>>()?;
    let event = a[1] as u32 & 0x3F;
    let gcr = a[2] as u32 & 0xFFF;
    let (dst, cache, address) = (a[3] as u32 & 3, a[4] as u32 & 3, a[5]);
    let (data_sel, data, interrupt) = (a[6] as u32 & 7, a[7], a[10] as u32 & 7);
    put(
        env,
        &[
            pm4(8, IT_RELEASE_MEM, 0),
            event | (5 << 8) | (gcr << 12) | (cache << 25),
            (dst << 16) | (interrupt << 24) | (data_sel << 29),
            address as u32 & !3,
            (address >> 32) as u32,
            data as u32,
            (data >> 32) as u32,
            a[11] as u32 & 0x07FF_FFFF,
        ],
    )
}

/// `(queue, ident, user data)`: registers a graphics event.
fn add_eq_event(env: &mut Env) -> Result<Outcome, CallError> {
    let (queue, ident, udata) = (env.arg(0), env.arg(1), env.arg(2));
    let ok = env.kernel.add_event(queue, ident, GRAPHICS_FILTER, udata);
    ret(if ok { 0 } else { ERROR_NOT_FOUND })
}

fn delete_eq_event(env: &mut Env) -> Result<Outcome, CallError> {
    let (queue, ident) = (env.arg(0), env.arg(1));
    let ok = env.kernel.remove_event(queue, ident, GRAPHICS_FILTER);
    ret(if ok { 0 } else { ERROR_NOT_FOUND })
}

/// `(event)`: the low 32 bits of the event's ident.
fn eq_event_type(env: &mut Env) -> Result<Outcome, CallError> {
    ret(read_dword(env, env.arg(0))? as u64)
}

/// `(event)`: the low 27 bits of the event's data.
fn eq_context_id(env: &mut Env) -> Result<Outcome, CallError> {
    ret(env.mem.read_u64(env.arg(0) + 0x10)? & 0x07FF_FFFF)
}

#[cfg(test)]
mod tests {
    use super::super::draw::tests::{call, check_sizes, measured};

    #[test]
    fn sizes_match_the_console() {
        eprintln!("checked {} sizes", check_sizes(super::FUNCTIONS));
    }

    /// RELEASE_MEM takes most arguments on the stack, which held garbage on
    /// the console; check the dwords that come from registers.
    #[test]
    fn release_mem_matches_the_console_in_register_fields() {
        let Some(rows) = measured("sceAgcCbReleaseMem") else {
            return;
        };
        for (args, expect) in rows {
            let (r, ours) = call(super::release_mem, &args, expect.len());
            assert_eq!(r, u64::MAX);
            for i in [0, 1, 3, 4] {
                assert_eq!(ours[i], expect[i], "dword {i} for {args:x?}");
            }
            assert_eq!(ours[2] & 0x3_0000, expect[2] & 0x3_0000, "{args:x?}");
        }
    }
}
