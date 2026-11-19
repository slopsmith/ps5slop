//! Command-buffer builders (`sceAgcDcb*`, `sceAgcAcb*`, `sceAgcCb*`): each
//! appends PM4 packets and returns where they went, or null when the buffer
//! is full. Ported from SharpEmu's `AgcExports.CommandPackets.cs` and
//! changed where console measurements (`oracle.rs`) show other output.
//!
//! The real library brackets some packets with thread-trace markers: writes
//! of the uconfig register `SQ_THREAD_TRACE_USERDATA_2` (0x342), the register
//! Mesa's radv also uses for trace markers. We write them too, so the bytes
//! match the console.

use super::cmdbuf::{IT_NOP, arg_n, emit, pm4, read_dwords};
use super::ret;
use crate::{CallError, Env, Function, Outcome};

pub(super) static FUNCTIONS: &[Function] = &[
    f("sceAgcDcbResetQueue", dcb_reset_queue),
    f("#qj7QZpgr9Uw", dcb_context_state_op),
    f("sceAgcAcbResetQueue", acb_reset_queue),
    f("sceAgcCbNop", cb_nop),
    f("sceAgcCbNopGetSize", cb_nop_size),
    f("sceAgcCbDispatch", cb_dispatch),
    f("sceAgcCbDispatchGetSize", size::<5>),
    f("sceAgcDcbDispatchIndirect", dcb_dispatch_indirect),
    f("sceAgcDcbDispatchIndirectGetSize", size::<3>),
    f("sceAgcAcbDispatchIndirect", acb_dispatch_indirect),
    f("sceAgcAcbDispatchIndirectGetSize", size::<4>),
    f("sceAgcDcbAcquireMem", dcb_acquire_mem),
    f("sceAgcDcbAcquireMemGetSize", size::<8>),
    f("sceAgcAcbAcquireMem", acb_acquire_mem),
    f("sceAgcAcbAcquireMemGetSize", size::<8>),
    f("sceAgcCbReleaseMem", cb_release_mem),
    f("sceAgcDcbWaitRegMem", dcb_wait_reg_mem),
    f("sceAgcAcbWaitRegMem", acb_wait_reg_mem),
    f("sceAgcDcbWaitOnAddressGetSize", wait_on_address_size),
    f("sceAgcAcbWaitOnAddressGetSize", wait_on_address_size),
    f("sceAgcDcbWriteData", write_data),
    f("sceAgcAcbWriteData", write_data),
    f("sceAgcDcbWriteDataGetSize", write_data_size),
    f("sceAgcCbCondWrite", cb_cond_write),
    f("sceAgcCbCondWriteGetSize", size::<9>),
    f(
        "sceAgcDcbStallCommandBufferParser",
        dcb_stall_command_buffer_parser,
    ),
    f("sceAgcDcbStallCommandBufferParserGetSize", size::<2>),
    f("sceAgcDcbSetBaseIndirectArgs", dcb_set_base_indirect_args),
    f("sceAgcDcbSetBaseDispatchIndirectArgsGetSize", size::<4>),
    f("sceAgcDcbSetBaseDrawIndirectArgsGetSize", size::<4>),
    f("sceAgcDcbGetLodStats", dcb_get_lod_stats),
    f("sceAgcDcbGetLodStatsGetSize", size::<5>),
    f("sceAgcDcbPushMarker", push_marker),
    f("sceAgcAcbPushMarker", push_marker),
    f("sceAgcDcbSetMarker", push_marker),
    f("sceAgcAcbSetMarker", push_marker),
    f("sceAgcDcbPopMarker", dcb_pop_marker),
    f("sceAgcAcbPopMarker", acb_pop_marker),
    // SharpEmu's `GetDataPacketPayloadAddress`.
    f("#V++UgBtQhn0", data_packet_payload_address),
];

pub(super) static STUBS: &[Function] = &[];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

const IT_SET_BASE: u32 = 0x11;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_WRITE_DATA: u32 = 0x37;
const IT_WAIT_REG_MEM: u32 = 0x3C;
const IT_PFP_SYNC_ME: u32 = 0x42;
const IT_COND_WRITE: u32 = 0x45;
const IT_RELEASE_MEM: u32 = 0x49;
const IT_ACQUIRE_MEM: u32 = 0x58;
const IT_SET_UCONFIG_REG: u32 = 0x79;
const IT_GET_LOD_STATS: u32 = 0x8E;
const IT_WAIT_REG_MEM64: u32 = 0x93;

/// `SQ_THREAD_TRACE_USERDATA_2` as a uconfig register offset.
const TRACE_USERDATA: u32 = 0x342;

/// The header flag the console sets on some `Dcb` trace markers (bit 2),
/// and its absence.
pub(super) const DCB: u32 = 1;
pub(super) const ACB: u32 = 0;

/// The longest NOP `sceAgcCbNop` writes, in dwords (SharpEmu's limit).
const MAX_NOP_DWORDS: u32 = 0x4001;

/// The trace marker opening a packet sequence: a code and a 48-bit address.
pub(super) fn marker_begin(flag: u32, code: u32, address: u64) -> [u32; 4] {
    [
        pm4(4, IT_SET_UCONFIG_REG, flag),
        TRACE_USERDATA,
        code | ((address >> 32) as u32 & 0xFFFF),
        address as u32,
    ]
}

/// A one-value trace marker, such as the one closing a sequence.
pub(super) fn marker(flag: u32, code: u32) -> [u32; 3] {
    [pm4(3, IT_SET_UCONFIG_REG, flag), TRACE_USERDATA, code]
}

/// A `GetSize` answering a fixed packet of `N` dwords.
pub(super) fn size<const N: u64>(_: &mut Env) -> Result<Outcome, CallError> {
    ret(4 * N)
}

/// `sceAgcDcbResetQueue(buffer, queue, mode)`: a NOP marking the reset,
/// shorter than the console's mode-dependent sequence (see `oracle.rs`).
fn dcb_reset_queue(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    const R_DRAW_RESET: u32 = 0x05;
    ret(emit(env, buffer, &[pm4(2, IT_NOP, R_DRAW_RESET), 0])?)
}

/// `sceAgcAcbResetQueue(buffer, ...)`: the two trace markers the console
/// wrote for the one measured call. What the payload means is unknown.
fn acb_reset_queue(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let mut p = vec![
        pm4(4, IT_SET_UCONFIG_REG, ACB),
        TRACE_USERDATA,
        0xCE20_0001,
        8,
    ];
    p.extend(marker(ACB, 0xCEA0_0000));
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcCbNop(buffer, dwords)`: a NOP of that many dwords; one dword is
/// the header-only form with the count field all ones.
fn cb_nop(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, n) = (env.arg(0), env.arg(1) as u32);
    if buffer == 0 || n == 0 || n > MAX_NOP_DWORDS {
        return ret(0);
    }
    let mut p = vec![0u32; n as usize];
    p[0] = if n == 1 {
        0xFFFF_0000 | (IT_NOP << 8)
    } else {
        pm4(n, IT_NOP, 0)
    };
    ret(emit(env, buffer, &p)?)
}

fn cb_nop_size(env: &mut Env) -> Result<Outcome, CallError> {
    let n = env.arg(0) as u32;
    ret(if (1..=MAX_NOP_DWORDS).contains(&n) {
        4 * n as u64
    } else {
        0
    })
}

/// The dispatch initiator: the caller's modifier bits SharpEmu keeps, plus
/// COMPUTE_SHADER_EN and FORCE_START_AT_000.
fn dispatch_initiator(modifier: u64) -> u32 {
    (modifier as u32 & 0xA038) | 0x41
}

/// `sceAgcCbDispatch(buffer, x, y, z, modifier)`.
fn cb_dispatch(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let p = [
        pm4(5, IT_DISPATCH_DIRECT, 0),
        env.arg(1) as u32,
        env.arg(2) as u32,
        env.arg(3) as u32,
        dispatch_initiator(env.arg(4)),
    ];
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcDcbDispatchIndirect(buffer, offset, modifier)`: the offset is from
/// the base `sceAgcDcbSetBaseIndirectArgs` set.
fn dcb_dispatch_indirect(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let p = [
        pm4(3, IT_DISPATCH_INDIRECT, 0),
        env.arg(1) as u32,
        dispatch_initiator(env.arg(2)),
    ];
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcAcbDispatchIndirect(buffer, args, modifier)`: compute queues take
/// a full address.
fn acb_dispatch_indirect(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, args) = (env.arg(0), env.arg(1));
    if buffer == 0 {
        return ret(0);
    }
    let p = [
        pm4(4, IT_DISPATCH_INDIRECT, 0),
        args as u32,
        (args >> 32) as u32,
        dispatch_initiator(env.arg(2)),
    ];
    ret(emit(env, buffer, &p)?)
}

/// ACQUIRE_MEM's range in 256-byte units (base, size), enclosing the byte
/// range. A null base with size 0 is the console's "everything" encoding.
fn acquire_range(base: u64, size: u64) -> (u64, u64) {
    if base == 0 && size == 0 {
        return (1, 0xFF_FFFF_FFFE);
    }
    let units = ((base & 0xFF) as u128 + size as u128 + 0xFF) >> 8;
    (base >> 8, units as u64)
}

fn acquire_mem(coher_cntl: u32, base: u64, size: u64, poll: u64, gcr: u64) -> [u32; 8] {
    let (base, units) = acquire_range(base, size);
    [
        pm4(8, IT_ACQUIRE_MEM, 0),
        coher_cntl,
        units as u32,
        (units >> 32) as u32 & 0xFF,
        base as u32,
        (base >> 32) as u32,
        (poll as u32 / 40).min(0xFFFF),
        gcr as u32 & 0x7_FFFF,
    ]
}

/// `sceAgcDcbAcquireMem(buffer, engine, cb_db_op, gcr_cntl, base, size,
/// poll)`.
fn dcb_acquire_mem(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let poll = arg_n(env, 6)?;
    if buffer == 0 {
        return ret(0);
    }
    let coher = ((env.arg(1) as u32 & 1) << 31) | (env.arg(2) as u32 & 0x0600_7FC0);
    let p = acquire_mem(coher, env.arg(4), env.arg(5), poll, env.arg(3));
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcAcbAcquireMem(buffer, gcr_cntl, base, size, poll)`.
fn acb_acquire_mem(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let p = acquire_mem(0, env.arg(2), env.arg(3), env.arg(4), env.arg(1));
    ret(emit(env, buffer, &p)?)
}

/// Arguments: action, gcr_cntl, dst_sel, cache_policy, address, data_sel,
/// data, gds_offset, gds_size, interrupt, context_id (GDS unused).
fn cb_release_mem(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let data_sel = arg_n(env, 6)?;
    let data = arg_n(env, 7)?;
    let interrupt = arg_n(env, 10)?;
    let context = arg_n(env, 11)?;
    if buffer == 0 {
        return ret(0);
    }
    let address = env.arg(5);
    let p = [
        pm4(8, IT_RELEASE_MEM, 0),
        (env.arg(1) as u32 & 0x3F)
            | 0x500
            | ((env.arg(2) as u32 & 0xFFF) << 12)
            | ((env.arg(4) as u32 & 3) << 25),
        ((env.arg(3) as u32 & 3) << 16)
            | ((interrupt as u32 & 7) << 24)
            | ((data_sel as u32 & 7) << 29),
        address as u32 & !3,
        (address >> 32) as u32,
        data as u32,
        (data >> 32) as u32,
        context as u32,
    ];
    ret(emit(env, buffer, &p)?)
}

/// The fields of a memory wait.
struct Wait {
    size: u32,
    compare: u32,
    operation: u32,
    cache: u32,
    address: u64,
    reference: u64,
    mask: u64,
    poll: u32,
}

/// The console matched SharpEmu's 64-bit control word; the 32-bit wait
/// reuses it (unverified).
fn wait_packets(flag: u32, w: &Wait) -> Option<Vec<u32>> {
    let control = 0x10
        | (w.compare & 7)
        | ((w.operation & 1) << 8)
        | ((w.operation & 6) << 5)
        | ((w.cache & 3) << 25);
    let poll = (w.poll >> 4).min(0xFFFF);
    let mut p = marker_begin(flag, 0xC802_0000, w.address).to_vec();
    match w.size {
        0 => p.extend([
            pm4(7, IT_WAIT_REG_MEM, 0),
            control,
            w.address as u32 & !3,
            (w.address >> 32) as u32 & 0xFFFF,
            w.reference as u32,
            w.mask as u32,
            poll,
        ]),
        1 => p.extend([
            pm4(9, IT_WAIT_REG_MEM64, 0),
            control,
            w.address as u32 & !7,
            (w.address >> 32) as u32 & 0xFFFF,
            w.reference as u32,
            (w.reference >> 32) as u32,
            w.mask as u32,
            (w.mask >> 32) as u32,
            poll,
        ]),
        _ => return None,
    }
    p.extend(marker(flag, 0xC800_0000));
    Some(p)
}

/// `sceAgcDcbWaitRegMem(buffer, size, compare, operation, cache_policy,
/// address, reference, mask, poll)`.
fn dcb_wait_reg_mem(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let w = Wait {
        size: env.arg(1) as u32 & 0xFF,
        compare: env.arg(2) as u32,
        operation: env.arg(3) as u32,
        cache: env.arg(4) as u32,
        address: env.arg(5),
        reference: arg_n(env, 6)?,
        mask: arg_n(env, 7)?,
        poll: arg_n(env, 8)? as u32,
    };
    match wait_packets(DCB, &w) {
        Some(p) if buffer != 0 => ret(emit(env, buffer, &p)?),
        _ => ret(0),
    }
}

/// `sceAgcAcbWaitRegMem(buffer, size, compare, cache_policy, address,
/// reference, mask, poll)`.
fn acb_wait_reg_mem(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let w = Wait {
        size: env.arg(1) as u32 & 0xFF,
        compare: env.arg(2) as u32,
        operation: 0,
        cache: env.arg(3) as u32,
        address: env.arg(4),
        reference: env.arg(5),
        mask: arg_n(env, 6)?,
        poll: arg_n(env, 7)? as u32,
    };
    match wait_packets(ACB, &w) {
        Some(p) if buffer != 0 => ret(emit(env, buffer, &p)?),
        _ => ret(0),
    }
}

/// The bytes of a wait for a 32-bit label (0) or a 64-bit one (other).
fn wait_on_address_size(env: &mut Env) -> Result<Outcome, CallError> {
    ret(if env.arg(0) as u32 == 0 { 0x38 } else { 0x40 })
}

/// Arguments: dst_sel, cache_policy, address, data, dwords, increment,
/// confirm. Their mapping onto WRITE_DATA's fields is unverified.
fn write_data(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, address, data, n) = (env.arg(0), env.arg(3), env.arg(4), env.arg(5) as u32);
    let increment = arg_n(env, 6)? as u32;
    let confirm = arg_n(env, 7)? as u32;
    if buffer == 0 || data == 0 || n > 0x3FFC {
        return ret(0);
    }
    let mut p = vec![
        pm4(n + 4, IT_WRITE_DATA, 0),
        ((env.arg(1) as u32 & 0xF) << 8)
            | ((increment & 1) << 16)
            | ((confirm & 1) << 20)
            | ((env.arg(2) as u32 & 3) << 25),
        address as u32,
        (address >> 32) as u32,
    ];
    p.extend(read_dwords(env, data, n as usize)?);
    ret(emit(env, buffer, &p)?)
}

fn write_data_size(env: &mut Env) -> Result<Outcome, CallError> {
    ret(16 + 4 * (env.arg(0) as u32 as u64))
}

/// `sceAgcCbCondWrite(buffer, compare, write_space, write_address, value,
/// poll_address, reference, mask)`.
fn cb_cond_write(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let reference = arg_n(env, 6)? as u32;
    let mask = arg_n(env, 7)? as u32;
    if buffer == 0 {
        return ret(0);
    }
    let (write, poll) = (env.arg(3), env.arg(5));
    let p = [
        pm4(9, IT_COND_WRITE, 0),
        0x10 | (env.arg(1) as u32 & 7) | ((env.arg(2) as u32 & 3) << 8),
        poll as u32,
        (poll >> 32) as u32,
        reference,
        mask,
        write as u32,
        (write >> 32) as u32,
        env.arg(4) as u32,
    ];
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcDcbStallCommandBufferParser(buffer, ...)`: PFP_SYNC_ME, whatever
/// the other arguments.
fn dcb_stall_command_buffer_parser(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, &[pm4(2, IT_PFP_SYNC_ME, 0), 0])?)
}

/// `sceAgcDcbSetBaseIndirectArgs(buffer, index, address)`: SET_BASE. The
/// console ORs `index << 1` into the header unmasked, and so do we.
fn dcb_set_base_indirect_args(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, address) = (env.arg(0), env.arg(2));
    if buffer == 0 {
        return ret(0);
    }
    let p = [
        pm4(4, IT_SET_BASE, 0) | (env.arg(1) as u32).wrapping_shl(1),
        1,
        address as u32 & !7,
        (address >> 32) as u32,
    ];
    ret(emit(env, buffer, &p)?)
}

/// `sceAgcDcbGetLodStats(buffer, cache_policy, address, control, counters,
/// reset, enable, select)`.
fn dcb_get_lod_stats(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    let enable = arg_n(env, 6)? as u32 & 1;
    let select = arg_n(env, 7)? as u32 & 0xFF;
    if buffer == 0 {
        return ret(0);
    }
    let address = env.arg(2);
    let p = [
        pm4(5, IT_GET_LOD_STATS, 0),
        env.arg(3) as u32,
        address as u32 & !0x3F,
        (address >> 32) as u32,
        ((env.arg(1) as u32 & 3) << 28)
            | (enable << 19)
            | ((env.arg(5) as u32 & 1) << 18)
            | ((env.arg(4) as u32 & 0xFF) << 10)
            | (select << 2),
    ];
    ret(emit(env, buffer, &p)?)
}

/// SharpEmu's NOP carrying the name; the console's form is unmeasured.
fn push_marker(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let mut name = env.mem.read_cstr(env.arg(1), 4095)?;
    name.resize(name.len() / 4 * 4 + 4, 0);
    const R_PUSH_MARKER: u32 = 0x0B;
    let mut p = vec![pm4(name.len() as u32 / 4 + 1, IT_NOP, R_PUSH_MARKER)];
    p.extend(
        name.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c)),
    );
    ret(emit(env, buffer, &p)?)
}

fn dcb_pop_marker(env: &mut Env) -> Result<Outcome, CallError> {
    pop_marker(env, DCB)
}

fn acb_pop_marker(env: &mut Env) -> Result<Outcome, CallError> {
    pop_marker(env, ACB)
}

/// `sceAgc{Dcb,Acb}PopMarker(buffer)`: one trace marker.
fn pop_marker(env: &mut Env, flag: u32) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(env, buffer, &marker(flag, 0xC200_0000))?)
}

/// SharpEmu's reading of `(out, packet, type)`, unverified.
fn data_packet_payload_address(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, packet, kind) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if out == 0 || packet == 0 {
        return ret(super::ERROR_INVALID_ARGUMENT);
    }
    let payload = if kind == 0 {
        let header = read_dwords(env, packet, 1)?[0];
        if header & 0x3FFF_0000 == 0x3FFF_0000 {
            0
        } else {
            packet + 4
        }
    } else {
        packet + 8
    };
    env.mem.write_u64(out, payload)?;
    ret(0)
}

/// SharpEmu's `DcbContextStateOperation(buffer, op)`: op 1 (push) as the
/// console wrote it, the others in SharpEmu's NOP-tagged form.
fn dcb_context_state_op(env: &mut Env) -> Result<Outcome, CallError> {
    const IT_CONTEXT_CONTROL: u32 = 0x28;
    const IT_CLEAR_STATE: u32 = 0x12;
    const NOP_CONTEXT_STATE: u32 = 0x1A;
    let (buffer, op) = (env.arg(0), env.arg(1) as u32);
    if buffer == 0 || op > 3 {
        return ret(0);
    }
    if op == 1 {
        return ret(emit(
            env,
            buffer,
            &[pm4(3, IT_CONTEXT_CONTROL, 0), 0, 0x8001_8001, pm4(2, IT_CLEAR_STATE, 0), 1],
        )?);
    }
    let mut packet = vec![pm4(5, IT_NOP, NOP_CONTEXT_STATE), op, 0, 0, 0];
    if op == 3 {
        packet.extend([pm4(5, IT_NOP, 0), 0, 0, 0, 0]);
    }
    ret(emit(env, buffer, &packet)?)
}

#[cfg(test)]
pub(super) mod golden {
    //! `oracle.rs` with a stack like the measuring harness's (0, 1, then a
    //! large value), so stack arguments get checked too; plus `GetSize`.

    use memory::{Memory, PAGE_SIZE, Prot};

    use crate::{Cpu, Env, Hle, Kernel, Outcome, Thread, TlsTemplate, Vfs};

    const DIR: &str = "../../references/src/prosper/prosper/tests/data/console_oracle/";
    const DESCRIPTOR: u64 = 0x30_0000;
    const BUFFER: u64 = 0x31_0000;
    const STACK: u64 = 0x10_0000;

    fn parse_int(s: &str) -> Option<u64> {
        match s.strip_prefix("0x") {
            Some(h) => u64::from_str_radix(h, 16).ok(),
            None => s.parse().ok(),
        }
    }

    fn rows(file: &str) -> Option<Vec<Vec<String>>> {
        let text = std::fs::read_to_string(format!("{DIR}{file}")).ok()?;
        Some(
            text.lines()
                .filter(|l| !l.starts_with('#'))
                .map(|l| l.split('\t').map(str::to_owned).collect())
                .collect(),
        )
    }

    /// Calls `name` with `args` after the buffer descriptor (if `buffer`);
    /// its return value and the start of the buffer.
    fn call(name: &str, args: &[u64], buffer: bool) -> Option<(u64, Vec<u8>)> {
        let f = super::super::functions().find(|f| f.name == name)?;
        let mem = Memory::new(1 << 40).unwrap();
        mem.map(STACK, 2 * PAGE_SIZE, Prot::RW, "stack").unwrap();
        mem.map(DESCRIPTOR, PAGE_SIZE, Prot::RW, "descriptor")
            .unwrap();
        mem.map(BUFFER, 4 * PAGE_SIZE, Prot::RW, "buffer").unwrap();
        mem.write_u64(DESCRIPTOR + 0x10, BUFFER).unwrap();
        mem.write_u64(DESCRIPTOR + 0x18, BUFFER + 4 * PAGE_SIZE)
            .unwrap();
        let rsp = STACK + PAGE_SIZE - 8;
        for (i, v) in [0u64, 1, 0xFFFF_FFF0].iter().enumerate() {
            mem.write_u64(rsp + 8 * (i as u64 + 1), *v).unwrap();
        }
        let hle = Hle::new();
        hle.map(&mem).unwrap();
        let kernel = Kernel::new(
            hle,
            mem,
            TlsTemplate::default(),
            Box::new(std::io::sink()),
            Box::new(std::io::sink()),
            Vfs::default(),
        );
        let mut cpu = Cpu::default();
        cpu.regs[x86::cpu::reg::RSP] = rsp;
        const ORDER: [usize; 6] = [7, 6, 2, 1, 8, 9];
        let skip = usize::from(buffer);
        if buffer {
            cpu.regs[ORDER[0]] = DESCRIPTOR;
        }
        for (i, &a) in args.iter().take(6 - skip).enumerate() {
            cpu.regs[ORDER[i + skip]] = a;
        }
        let mem = kernel.memory();
        let mut env = Env {
            cpu: &mut cpu,
            mem: &mem,
            kernel: &kernel,
            thread: Thread { tcb: 0 },
        };
        let Ok(Outcome::Return(r)) = (f.func)(&mut env) else {
            panic!("{name} failed");
        };
        let mut bytes = vec![0u8; 0x100];
        mem.read(BUFFER, &mut bytes).unwrap();
        Some((r, bytes))
    }

    /// Checks the measured calls of `ours`, less `skip`; returns how many.
    pub(in super::super) fn builders(ours: &[&str], skip: &[&str]) -> usize {
        let Some(rows) = rows("agcbuild.golden.tsv") else {
            eprintln!("skipped: console measurements not fetched");
            return 0;
        };
        let mut checked = 0;
        let mut failed = Vec::new();
        for c in rows.iter().filter(|c| c.len() >= 8 && c[5] == "ok") {
            if !ours.contains(&c[2].as_str()) || skip.contains(&c[0].as_str()) {
                continue;
            }
            let Some(args) = c[3]
                .split(',')
                .skip(1)
                .map(|a| a.strip_prefix("i:").and_then(parse_int))
                .collect::<Option<Vec<u64>>>()
            else {
                continue;
            };
            let Some((r, bytes)) = call(&c[2], &args, true) else {
                continue;
            };
            checked += 1;
            match c[7].as_str() {
                "null" if r != 0 => failed.push(format!("{}: not null", c[0])),
                "ptr+0" => {
                    let (_, data) = c[8].strip_prefix("a0=").unwrap().split_once(':').unwrap();
                    let expect: Vec<u8> = (0..data.len() / 2)
                        .map(|i| u8::from_str_radix(&data[2 * i..2 * i + 2], 16).unwrap())
                        .collect();
                    if r != BUFFER || bytes[..expect.len()] != expect[..] {
                        failed.push(format!(
                            "{}: ours {:02x?}, console {expect:02x?}",
                            c[0],
                            &bytes[..expect.len()]
                        ));
                    }
                }
                _ => {}
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
        checked
    }

    /// Checks every measured `GetSize` answer for the functions `ours` names.
    pub(in super::super) fn sizes(ours: &[&str]) -> usize {
        let Some(rows) = rows("agc.golden.tsv") else {
            eprintln!("skipped: console measurements not fetched");
            return 0;
        };
        let mut checked = 0;
        for c in rows.iter().filter(|c| c.len() >= 7 && c[5] == "ok") {
            if !ours.contains(&c[2].as_str()) {
                continue;
            }
            let args: Vec<u64> = match c[3].as_str() {
                "-" => vec![0],
                a => a
                    .split(',')
                    .filter_map(|a| a.strip_prefix("i:").and_then(parse_int))
                    .collect(),
            };
            let (r, _) = call(&c[2], &args, false).unwrap();
            assert_eq!(r, parse_int(&c[6]).unwrap(), "{} {}", c[2], c[3]);
            checked += 1;
        }
        checked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<&'static str> {
        FUNCTIONS.iter().map(|f| f.name).collect()
    }

    #[test]
    fn builders_match_the_console_with_stack_arguments() {
        let skip = [
            "agcbuild_CbReleaseMem_a",
            "agcbuild_CbReleaseMem_b",
            "agcbuild_CbReleaseMem_c",
        ];
        let mut ours = names();
        ours.retain(|n| *n != "sceAgcDcbResetQueue");
        let n = golden::builders(&ours, &skip);
        eprintln!("packets: checked {n} measured builder calls");
    }

    #[test]
    fn sizes_match_the_console() {
        let n = golden::sizes(&names());
        eprintln!("packets: checked {n} measured sizes");
    }

    #[test]
    fn acquire_ranges_enclose_the_bytes() {
        assert_eq!(acquire_range(0x2, 3), (0, 1));
        assert_eq!(acquire_range(0xFF, 2), (0, 2));
        assert_eq!(acquire_range(0, 0), (1, 0xFF_FFFF_FFFE));
    }
}
