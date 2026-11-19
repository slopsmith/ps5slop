//! Command-buffer control flow (branches, conditional execution, jumps,
//! rewinds, predication) and the `*Patch*` functions that fill in packets
//! after they were built. Ported from SharpEmu's `AgcExports.ControlFlow.cs`
//! and `AgcExports.PacketPatching.cs` (GPL-2.0-or-later); builders checked
//! against console measurements (`oracle.rs`), see `docs/gpu.md`.
//!
//! The patch functions take the address a builder returned. For packets
//! built in other families they accept the hardware PM4 form the console
//! writes and SharpEmu's NOP-tagged form.

use super::cmdbuf::{IT_NOP, emit, pm4};
use super::{ERROR_INVALID_ARGUMENT, ret};
use crate::{CallError, Env, Function, Outcome};

const ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

const IT_SET_PREDICATION: u32 = 0x20;
const IT_COND_EXEC: u32 = 0x22;
const IT_WRITE_DATA: u32 = 0x37;
const IT_WAIT_REG_MEM: u32 = 0x3C;
const IT_INDIRECT_BUFFER: u32 = 0x3F;
pub(super) const IT_RELEASE_MEM: u32 = 0x49;
const IT_DMA_DATA: u32 = 0x50;
const IT_REWIND: u32 = 0x59;
const IT_SET_SH_REG_INDIRECT: u32 = 0x63;
const IT_SET_UCONFIG_REG_INDIRECT: u32 = 0x64;
pub(super) const IT_SET_UCONFIG_REG: u32 = 0x79;
const IT_WAIT_REG_MEM64: u32 = 0x93;
const IT_SET_CONTEXT_REG_INDIRECT: u32 = 0x9F;

/// SharpEmu's tags in the header of its NOP-encoded packets.
const R_WAIT_MEM32: u32 = 0x0A;
const R_SH_REGS_INDIRECT: u32 = 0x11;
const R_CX_REGS_INDIRECT: u32 = 0x12;
const R_UC_REGS_INDIRECT: u32 = 0x13;
const R_WRITE_DATA: u32 = 0x15;
const R_WAIT_MEM64: u32 = 0x16;
const R_DMA_DATA: u32 = 0x19;

const REWIND_VALID: u32 = 1 << 31;
const REWIND_OFFLOAD_ENABLE: u32 = 1 << 24;

/// A `...GetSize` query answering `bytes`, the console's measured size.
macro_rules! size {
    ($name:literal, $bytes:literal) => {
        f($name, |_| super::ret($bytes))
    };
}
pub(super) use size;

pub(super) static FUNCTIONS: &[Function] = &[
    f("sceAgcDcbCondExec", cond_exec),
    f("sceAgcAcbCondExec", cond_exec),
    size!("sceAgcDcbCondExecGetSize", 0x14),
    size!("sceAgcAcbCondExecGetSize", 0x14),
    f(
        "sceAgcCondExecPatchSetCommandAddress",
        cond_exec_patch_address,
    ),
    f(
        "sceAgcAsyncCondExecPatchSetCommandAddress",
        cond_exec_patch_address,
    ),
    f("sceAgcCondExecPatchSetEnd", cond_exec_patch_end),
    f("sceAgcAsyncCondExecPatchSetEnd", cond_exec_patch_end),
    f("sceAgcDcbJump", dcb_jump),
    f("sceAgcAcbJump", acb_jump),
    size!("sceAgcDcbJumpGetSize", 0x10),
    size!("sceAgcAcbJumpGetSize", 0x10),
    f("sceAgcDcbRewind", rewind),
    f("sceAgcAcbRewind", rewind),
    size!("sceAgcDcbRewindGetSize", 8),
    size!("sceAgcAcbRewindGetSize", 8),
    f("sceAgcRewindPatchSetRewindState", rewind_patch_state),
    f("sceAgcAsyncRewindPatchSetRewindState", rewind_patch_state),
    f("sceAgcCbBranch", cb_branch),
    size!("sceAgcCbBranchGetSize", 0x38),
    f("sceAgcBranchPatchSetCompareAddress", branch_patch_compare),
    f("sceAgcBranchPatchSetThenTarget", branch_patch_then),
    f("sceAgcBranchPatchSetElseTarget", branch_patch_else),
    f("sceAgcDcbSetPredication", set_predication),
    size!("sceAgcDcbSetPredicationDisableGetSize", 0x10),
    size!("sceAgcDcbSetBoolPredicationEnableGetSize", 0x10),
    size!("sceAgcDcbSetZPassPredicationEnableGetSize", 0x10),
    size!("sceAgcDcbBeginOcclusionQueryGetSize", 0x10),
    size!("sceAgcDcbEndOcclusionQueryGetSize", 0x10),
    f("sceAgcSetPacketPredication", set_packet_predication),
    f("sceAgcSetRangePredication", set_range_predication),
    f("sceAgcGetPacketSize", packet_size),
    f("sceAgcSetNop", set_nop),
    f("sceAgcSetCxRegIndirectPatchSetAddress", cx_patch_address),
    f("sceAgcSetShRegIndirectPatchSetAddress", sh_patch_address),
    f("sceAgcSetUcRegIndirectPatchSetAddress", uc_patch_address),
    f("sceAgcSetCxRegIndirectPatchSetNumRegisters", cx_patch_count),
    f("sceAgcSetShRegIndirectPatchSetNumRegisters", sh_patch_count),
    f("sceAgcSetUcRegIndirectPatchSetNumRegisters", uc_patch_count),
    f("sceAgcSetCxRegIndirectPatchAddRegisters", cx_patch_add),
    f("sceAgcSetShRegIndirectPatchAddRegisters", sh_patch_add),
    f("sceAgcSetUcRegIndirectPatchAddRegisters", uc_patch_add),
    f("sceAgcDmaDataPatchSetDstAddressOrOffset", dma_patch_dst),
    f(
        "sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate",
        dma_patch_src,
    ),
    f(
        "sceAgcWriteDataPatchSetAddressOrOffset",
        write_data_patch_address,
    ),
    f("sceAgcWriteDataPatchSetDst", write_data_patch_dst),
    f("sceAgcWriteDataPatchSetCachePolicy", write_data_patch_cache),
    f("sceAgcWaitRegMemPatchAddress", wait_patch_address),
    f("sceAgcWaitRegMemPatchCompareFunction", wait_patch_function),
    f("sceAgcWaitRegMemPatchReference", wait_patch_reference),
    f("sceAgcWaitRegMemPatchMask", wait_patch_mask),
    f("sceAgcQueueEndOfPipeActionPatchAddress", eop_patch_address),
    f("sceAgcQueueEndOfPipeActionPatchData", eop_patch_data),
    f("sceAgcQueueEndOfPipeActionPatchGcrCntl", eop_patch_gcr),
    f("sceAgcQueueEndOfPipeActionPatchType", eop_patch_type),
];

pub(super) static STUBS: &[Function] = &[];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
        name,
        func,
    }
}

/// The `n`th integer argument (0-based), including those on the stack.
pub(super) fn arg(env: &Env, n: usize) -> Result<u64, CallError> {
    if n < 6 {
        return Ok(env.arg(n));
    }
    let rsp = env.cpu.regs[x86::cpu::reg::RSP];
    Ok(env.mem.read_u64(rsp + 8 * (n as u64 - 5))?)
}

pub(super) fn read_dword(env: &Env, at: u64) -> Result<u32, CallError> {
    let mut b = [0u8; 4];
    env.mem.read(at, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub(super) fn write_dword(env: &Env, at: u64, v: u32) -> Result<(), CallError> {
    Ok(env.mem.write(at, &v.to_le_bytes())?)
}

/// Replaces the bits of `mask` in the dword at `at` with those of `v`.
fn patch_bits(env: &Env, at: u64, mask: u32, v: u32) -> Result<(), CallError> {
    let old = read_dword(env, at)?;
    write_dword(env, at, (old & !mask) | (v & mask))
}

fn write_split(env: &Env, at: u64, v: u64) -> Result<(), CallError> {
    write_dword(env, at, v as u32)?;
    write_dword(env, at + 4, (v >> 32) as u32)
}

const fn opcode(header: u32) -> u32 {
    (header >> 8) & 0xFF
}

/// The tag SharpEmu keeps in a header's shader-type/index field.
const fn tag(header: u32) -> u32 {
    (header >> 2) & 0x3F
}

/// Length in dwords of the packet starting with `header`: type-3 packets
/// carry it, a type-2 filler is one dword.
const fn packet_dwords(header: u32) -> u32 {
    if header >> 30 == 3 {
        ((header >> 16) & 0x3FFF) + 2
    } else {
        1
    }
}

/// The header at `packet`, or `None` for a null packet.
fn header(env: &Env, packet: u64) -> Result<Option<u32>, CallError> {
    if packet == 0 {
        return Ok(None);
    }
    Ok(Some(read_dword(env, packet)?))
}

/// The status a patch function returns.
fn status(ok: bool) -> Result<Outcome, CallError> {
    ret(if ok { 0 } else { ERROR_INVALID_ARGUMENT })
}

/// `(buffer, address, dwords)`: a COND_EXEC that skips the next `dwords`
/// when the 32-bit value at `address` is zero. The ACB form is the same.
fn cond_exec(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, address, count) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(
        env,
        buffer,
        &[
            pm4(5, IT_COND_EXEC, 0),
            address as u32 & !7,
            (address >> 32) as u32,
            0,
            count & 0x3FFF,
        ],
    )?)
}

fn cond_exec_patch_address(env: &mut Env) -> Result<Outcome, CallError> {
    let (packet, address) = (env.arg(0), env.arg(1));
    if header(env, packet)?.map(opcode) != Some(IT_COND_EXEC) {
        return status(false);
    }
    write_split(env, packet + 4, address & !7)?;
    status(true)
}

/// `(packet, end)`: makes the COND_EXEC at `packet` cover everything up to
/// `end`.
fn cond_exec_patch_end(env: &mut Env) -> Result<Outcome, CallError> {
    let (packet, end) = (env.arg(0), env.arg(1));
    let Some(h) = header(env, packet)? else {
        return status(false);
    };
    let body = packet + 4 * packet_dwords(h) as u64;
    if opcode(h) != IT_COND_EXEC || end < body {
        return status(false);
    }
    patch_bits(env, packet + 16, 0x3FFF, ((end - body) / 4) as u32)?;
    status(true)
}

/// The INDIRECT_BUFFER control dword: `base` holds the fixed bits.
fn ib_control(base: u32, cache_policy: u32, dwords: u32) -> u32 {
    base | ((cache_policy & 3) << 28) | (dwords & 0xF_FFFF)
}

fn jump(env: &mut Env, buffer: u64, target: u64, control: u32) -> Result<Outcome, CallError> {
    if buffer == 0 {
        return ret(0);
    }
    ret(emit(
        env,
        buffer,
        &[
            pm4(4, IT_INDIRECT_BUFFER, 0),
            target as u32 & !3,
            (target >> 32) as u32,
            control,
        ],
    )?)
}

/// `(buffer, mode, cache policy, target, dwords)`: mode 1 chains.
fn dcb_jump(env: &mut Env) -> Result<Outcome, CallError> {
    let mode = env.arg(1) as u32 & 1;
    let control = ib_control(
        0x0F20_0000 | (mode << 20),
        env.arg(2) as u32,
        env.arg(4) as u32,
    );
    jump(env, env.arg(0), env.arg(3), control)
}

/// `(buffer, cache policy, target, dwords)`: an ACB jump always chains.
fn acb_jump(env: &mut Env) -> Result<Outcome, CallError> {
    let control = ib_control(0x0F90_0000, env.arg(1) as u32, env.arg(3) as u32);
    jump(env, env.arg(0), env.arg(2), control)
}

/// `(buffer, flags)`: bit 0 valid, bit 1 offload enable. A parser waits at
/// an invalid REWIND until the title patches it valid.
fn rewind(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, flags) = (env.arg(0), env.arg(1));
    if buffer == 0 {
        return ret(0);
    }
    let mut body = 0;
    if flags & 1 != 0 {
        body |= REWIND_VALID;
    }
    if flags & 2 != 0 {
        body |= REWIND_OFFLOAD_ENABLE;
    }
    ret(emit(env, buffer, &[pm4(2, IT_REWIND, 0), body])?)
}

/// `(packet, valid)`: sets or clears the valid bit of a REWIND.
fn rewind_patch_state(env: &mut Env) -> Result<Outcome, CallError> {
    let (packet, valid) = (env.arg(0), env.arg(1) & 1 != 0);
    if packet == 0 || (packet as i64) < 0 {
        return status(false);
    }
    let body = if opcode(read_dword(env, packet)?) == IT_REWIND {
        packet + 4
    } else {
        packet
    };
    patch_bits(
        env,
        body,
        REWIND_VALID,
        if valid { REWIND_VALID } else { 0 },
    )?;
    status(true)
}

/// COND_INDIRECT_BUFFER. Args: mode, function, compare address, mask,
/// reference, then (cache policy, target, dwords), else (same three).
fn cb_branch(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let a = (0..12)
        .map(|n| arg(env, n))
        .collect::<Result<Vec<_>, _>>()?;
    ret(emit(
        env,
        buffer,
        &[
            pm4(14, IT_INDIRECT_BUFFER, 0),
            (a[1] as u32 & 3) | ((a[2] as u32 & 7) << 8),
            a[3] as u32 & !7,
            (a[3] >> 32) as u32,
            a[4] as u32,
            (a[4] >> 32) as u32,
            a[5] as u32,
            (a[5] >> 32) as u32,
            a[7] as u32 & !3,
            (a[7] >> 32) as u32,
            ib_control(0, a[6] as u32, a[8] as u32),
            a[10] as u32 & !3,
            (a[10] >> 32) as u32,
            ib_control(0, a[9] as u32, a[11] as u32),
        ],
    )?)
}

/// True when `packet` is the 14-dword COND_INDIRECT_BUFFER of [`cb_branch`].
fn is_branch(env: &Env, packet: u64) -> Result<bool, CallError> {
    Ok(header(env, packet)?.map(|h| h & !1) == Some(pm4(14, IT_INDIRECT_BUFFER, 0)))
}

fn branch_patch_compare(env: &mut Env) -> Result<Outcome, CallError> {
    let (packet, address) = (env.arg(0), env.arg(1));
    if !is_branch(env, packet)? {
        return status(false);
    }
    write_split(env, packet + 8, address & !7)?;
    status(true)
}

/// Writes target `(address, dwords)` at `at`, keeping the cache policy.
fn branch_patch_target(env: &mut Env, at: u64) -> Result<Outcome, CallError> {
    let (packet, target, dwords) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if !is_branch(env, packet)? {
        return status(false);
    }
    write_split(env, packet + at, target & !3)?;
    patch_bits(env, packet + at + 8, 0xF_FFFF, dwords)?;
    status(true)
}

fn branch_patch_then(env: &mut Env) -> Result<Outcome, CallError> {
    branch_patch_target(env, 32)
}

fn branch_patch_else(env: &mut Env) -> Result<Outcome, CallError> {
    branch_patch_target(env, 44)
}

/// `(buffer, condition, operation, wait, address)`: SET_PREDICATION.
fn set_predication(env: &mut Env) -> Result<Outcome, CallError> {
    let buffer = env.arg(0);
    if buffer == 0 {
        return ret(0);
    }
    let condition = env.arg(1) as u32 & 1;
    let operation = env.arg(2) as u32 & 7;
    let wait = env.arg(3) as u32 & 1;
    let address = env.arg(4);
    ret(emit(
        env,
        buffer,
        &[
            pm4(4, IT_SET_PREDICATION, 0),
            (condition << 8) | (wait << 12) | (operation << 16),
            address as u32 & !0xF,
            (address >> 32) as u32,
        ],
    )?)
}

/// `(packet, enable)`: the predicate bit of a packet header.
fn set_packet_predication(env: &mut Env) -> Result<Outcome, CallError> {
    let (packet, enable) = (env.arg(0), env.arg(1) == 1);
    if packet == 0 {
        return status(false);
    }
    patch_bits(env, packet, 1, enable as u32)?;
    status(true)
}

/// `(begin, end, enable)`: the predicate bit of every type-3 packet from
/// `begin` up to `end`. Unverified: written from a one-line description.
fn set_range_predication(env: &mut Env) -> Result<Outcome, CallError> {
    let (mut at, end, enable) = (env.arg(0), env.arg(1), env.arg(2) & 0xFF != 0);
    if at == 0 || end < at {
        return status(false);
    }
    while at < end {
        let h = read_dword(env, at)?;
        if h >> 30 == 3 {
            write_dword(env, at, (h & !1) | enable as u32)?;
        }
        at += 4 * packet_dwords(h) as u64;
    }
    status(true)
}

/// `(packet)`: the packet's length in dwords, from its header. Unverified:
/// the unit could be bytes.
fn packet_size(env: &mut Env) -> Result<Outcome, CallError> {
    match header(env, env.arg(0))? {
        Some(h) => ret(packet_dwords(h) as u64),
        None => ret(0),
    }
}

/// `(packet)`: turns the packet into a NOP of the same length, so the
/// parser skips it. Unverified: written from a one-line description.
fn set_nop(env: &mut Env) -> Result<Outcome, CallError> {
    let packet = env.arg(0);
    let Some(h) = header(env, packet)? else {
        return status(false);
    };
    if h >> 30 == 3 {
        write_dword(env, packet, pm4(packet_dwords(h), IT_NOP, 0))?;
    }
    status(true)
}

/// Where a SET_*_REG_INDIRECT keeps its address and register count:
/// (address offset, count offset, count mask).
fn regs_indirect_layout(
    env: &Env,
    packet: u64,
    (op, legacy): (u32, u32),
) -> Result<Option<(u64, u64, u32)>, CallError> {
    let Some(h) = header(env, packet)? else {
        return Ok(None);
    };
    Ok(match (opcode(h), packet_dwords(h)) {
        (o, 5) if o == op => Some((4, 16, 0x3FFF)),
        (IT_NOP, 4) if tag(h) == legacy => Some((8, 4, u32::MAX)),
        _ => None,
    })
}

const CX: (u32, u32) = (IT_SET_CONTEXT_REG_INDIRECT, R_CX_REGS_INDIRECT);
const SH: (u32, u32) = (IT_SET_SH_REG_INDIRECT, R_SH_REGS_INDIRECT);
const UC: (u32, u32) = (IT_SET_UCONFIG_REG_INDIRECT, R_UC_REGS_INDIRECT);

/// `(packet, address)`: where the register values are read from.
fn regs_patch_address(env: &mut Env, kind: (u32, u32)) -> Result<Outcome, CallError> {
    let (packet, address) = (env.arg(0), env.arg(1));
    let Some((at, _, _)) = regs_indirect_layout(env, packet, kind)? else {
        return status(false);
    };
    patch_bits(env, packet + at, !3, address as u32)?;
    write_dword(env, packet + at + 4, (address >> 32) as u32)?;
    status(true)
}

/// `(packet, count)`: sets the register count, or adds to it.
fn regs_patch_count(env: &mut Env, kind: (u32, u32), add: bool) -> Result<Outcome, CallError> {
    let (packet, count) = (env.arg(0), env.arg(1) as u32);
    let Some((_, at, mask)) = regs_indirect_layout(env, packet, kind)? else {
        return status(false);
    };
    let old = read_dword(env, packet + at)?;
    let new = if add {
        (old & mask).wrapping_add(count)
    } else {
        count
    };
    write_dword(env, packet + at, (old & !mask) | (new & mask))?;
    status(true)
}

fn cx_patch_address(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_address(env, CX)
}

fn sh_patch_address(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_address(env, SH)
}

fn uc_patch_address(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_address(env, UC)
}

fn cx_patch_count(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_count(env, CX, false)
}

fn sh_patch_count(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_count(env, SH, false)
}

fn uc_patch_count(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_count(env, UC, false)
}

fn cx_patch_add(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_count(env, CX, true)
}

fn sh_patch_add(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_count(env, SH, true)
}

fn uc_patch_add(env: &mut Env) -> Result<Outcome, CallError> {
    regs_patch_count(env, UC, true)
}

/// Offsets of (destination, source) in a DMA packet.
fn dma_layout(env: &Env, packet: u64) -> Result<Option<(u64, u64)>, CallError> {
    let Some(h) = header(env, packet)? else {
        return Ok(None);
    };
    Ok(match (opcode(h), packet_dwords(h)) {
        (IT_DMA_DATA, 7) => Some((16, 8)),
        (IT_NOP, 7) if tag(h) == R_DMA_DATA => Some((4, 12)),
        (IT_NOP, 8) if tag(h) == R_DMA_DATA => Some((16, 24)),
        _ => None,
    })
}

fn dma_patch(env: &mut Env, source: bool) -> Result<Outcome, CallError> {
    let (packet, value) = (env.arg(0), env.arg(1));
    let Some((dst, src)) = dma_layout(env, packet)? else {
        return status(false);
    };
    write_split(env, packet + if source { src } else { dst }, value)?;
    status(true)
}

fn dma_patch_dst(env: &mut Env) -> Result<Outcome, CallError> {
    dma_patch(env, false)
}

fn dma_patch_src(env: &mut Env) -> Result<Outcome, CallError> {
    dma_patch(env, true)
}

fn is_write_data(env: &Env, packet: u64) -> Result<bool, CallError> {
    Ok(match header(env, packet)? {
        Some(h) => opcode(h) == IT_WRITE_DATA || (opcode(h) == IT_NOP && tag(h) == R_WRITE_DATA),
        None => false,
    })
}

/// The (packet, value) of a WriteData patch call; SharpEmu saw titles pass
/// them in either order.
fn write_data_args(env: &Env) -> Result<Option<(u64, u64)>, CallError> {
    let (a, b) = (env.arg(0), env.arg(1));
    if is_write_data(env, a)? {
        Ok(Some((a, b)))
    } else if is_write_data(env, b)? {
        Ok(Some((b, a)))
    } else {
        Ok(None)
    }
}

/// `(packet, address)`: the WriteData destination.
fn write_data_patch_address(env: &mut Env) -> Result<Outcome, CallError> {
    match write_data_args(env)? {
        Some((packet, address)) => {
            write_split(env, packet + 8, address)?;
            status(true)
        }
        None => status(false),
    }
}

/// Replaces byte `byte` of the WriteData control dword.
fn write_data_patch_byte(env: &mut Env, byte: u32) -> Result<Outcome, CallError> {
    let Some((packet, value)) = write_data_args(env)? else {
        return status(false);
    };
    let shift = 8 * byte;
    patch_bits(env, packet + 4, 0xFF << shift, (value as u32) << shift)?;
    status(true)
}

fn write_data_patch_dst(env: &mut Env) -> Result<Outcome, CallError> {
    write_data_patch_byte(env, 0)
}

fn write_data_patch_cache(env: &mut Env) -> Result<Outcome, CallError> {
    write_data_patch_byte(env, 1)
}

/// The fields of a wait packet, as offsets from the packet address.
struct WaitLayout {
    function: u64,
    address: u64,
    /// Low address bits the packet cannot hold.
    align: u64,
    reference: u64,
    mask: u64,
    wide: bool,
}

/// Finds the wait in the packet a WaitRegMem builder returned: the console
/// puts a SET_UCONFIG_REG marker before WAIT_REG_MEM64.
fn wait_layout(env: &Env, packet: u64) -> Result<Option<WaitLayout>, CallError> {
    let Some(mut h) = header(env, packet)? else {
        return Ok(None);
    };
    let mut at = 0;
    if opcode(h) == IT_SET_UCONFIG_REG {
        at = 4 * packet_dwords(h) as u64;
        h = read_dword(env, packet + at)?;
    }
    let l = |function, address, align, reference, mask, wide| WaitLayout {
        function: at + function,
        address: at + address,
        align,
        reference: at + reference,
        mask: at + mask,
        wide,
    };
    Ok(match opcode(h) {
        IT_WAIT_REG_MEM64 => Some(l(4, 8, 7, 16, 24, true)),
        IT_WAIT_REG_MEM => Some(l(4, 8, 3, 16, 20, false)),
        IT_NOP if tag(h) == R_WAIT_MEM32 => Some(l(20, 4, 3, 16, 12, false)),
        IT_NOP if tag(h) == R_WAIT_MEM64 => Some(l(28, 4, 7, 20, 12, true)),
        _ => None,
    })
}

fn wait_patch_address(env: &mut Env) -> Result<Outcome, CallError> {
    let (packet, address) = (env.arg(0), env.arg(1));
    let Some(w) = wait_layout(env, packet)? else {
        return status(false);
    };
    write_split(env, packet + w.address, address & !w.align)?;
    status(true)
}

fn wait_patch_function(env: &mut Env) -> Result<Outcome, CallError> {
    let (packet, function) = (env.arg(0), env.arg(1) as u32);
    let Some(w) = wait_layout(env, packet)?.filter(|_| function <= 7) else {
        return status(false);
    };
    patch_bits(env, packet + w.function, 7, function)?;
    status(true)
}

/// Writes the value argument at the field `at` picks: 64 bits in the wide
/// forms, else 32.
fn wait_patch_value(env: &mut Env, at: fn(&WaitLayout) -> u64) -> Result<Outcome, CallError> {
    let (packet, value) = (env.arg(0), env.arg(1));
    let Some(w) = wait_layout(env, packet)? else {
        return status(false);
    };
    if w.wide {
        write_split(env, packet + at(&w), value)?;
    } else {
        write_dword(env, packet + at(&w), value as u32)?;
    }
    status(true)
}

fn wait_patch_reference(env: &mut Env) -> Result<Outcome, CallError> {
    wait_patch_value(env, |w| w.reference)
}

fn wait_patch_mask(env: &mut Env) -> Result<Outcome, CallError> {
    wait_patch_value(env, |w| w.mask)
}

/// Runs `patch` on the RELEASE_MEM `sceAgcCbReleaseMem` wrote, with the
/// value argument; a fault becomes an error code.
fn eop_patch(
    env: &mut Env,
    patch: fn(&Env, u64, u64) -> Result<bool, CallError>,
) -> Result<Outcome, CallError> {
    let (packet, value) = (env.arg(0), env.arg(1));
    if header(env, packet)?.map(opcode) != Some(IT_RELEASE_MEM) {
        return status(false);
    }
    match patch(env, packet, value) {
        Ok(ok) => status(ok),
        Err(CallError::Memory(_)) => ret(ERROR_MEMORY_FAULT),
        Err(e) => Err(e),
    }
}

fn eop_patch_address(env: &mut Env) -> Result<Outcome, CallError> {
    eop_patch(env, |env, p, v| {
        write_split(env, p + 12, v & !3).map(|_| true)
    })
}

fn eop_patch_data(env: &mut Env) -> Result<Outcome, CallError> {
    eop_patch(env, |env, p, v| write_split(env, p + 20, v).map(|_| true))
}

fn eop_patch_gcr(env: &mut Env) -> Result<Outcome, CallError> {
    eop_patch(env, |env, p, v| {
        patch_bits(env, p + 4, 0xFFF << 12, (v as u32) << 12).map(|_| true)
    })
}

/// `(packet, data selection)`: what the end-of-pipe action writes.
fn eop_patch_type(env: &mut Env) -> Result<Outcome, CallError> {
    eop_patch(env, |env, p, v| {
        if v > 3 {
            return Ok(false);
        }
        patch_bits(env, p + 8, 7 << 29, (v as u32) << 29).map(|_| true)
    })
}

#[cfg(test)]
mod tests {
    use super::super::draw::tests::check_sizes;

    #[test]
    fn sizes_match_the_console() {
        eprintln!("checked {} sizes", check_sizes(super::FUNCTIONS));
    }
}
