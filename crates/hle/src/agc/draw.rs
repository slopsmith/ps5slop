//! Command-buffer builders for draws, dispatches, index buffers and
//! indirect arguments. Ported from SharpEmu's `AgcExports.DrawPackets.cs`
//! and the dispatch builders of `AgcExports.CommandPackets.cs`
//! (GPL-2.0-or-later), changed where console measurements (`oracle.rs`)
//! disagree; see `docs/gpu.md`.

use super::cmdbuf::emit;
use super::cmdbuf::pm4;
use super::control::{IT_SET_UCONFIG_REG, arg, size};
use super::ret;
use crate::{CallError, Env, Function, Outcome};

const IT_SET_BASE: u32 = 0x11;
const IT_INDEX_BUFFER_SIZE: u32 = 0x13;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_DRAW_INDIRECT: u32 = 0x24;
const IT_DRAW_INDEX_INDIRECT: u32 = 0x25;
const IT_INDEX_BASE: u32 = 0x26;
const IT_DRAW_INDEX_2: u32 = 0x27;
const IT_DRAW_INDIRECT_MULTI: u32 = 0x2C;
const IT_DRAW_INDEX_AUTO: u32 = 0x2D;
const IT_NUM_INSTANCES: u32 = 0x2F;
const IT_DRAW_INDEX_OFFSET_2: u32 = 0x35;
const IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
const IT_DRAW_INDEX_MULTI_INST: u32 = 0x3A;
const IT_SET_UCONFIG_REG_INDEX: u32 = 0x7A;
const IT_INDEX_ATTRIBUTES_INDIRECT: u32 = 0x91;
const VGT_INDEX_TYPE: u32 = 0x243;

/// The user-config register the console writes around multi draws, and the
/// values: `DRAW_KIND | kind` before, `DRAW_KIND` after (meaning unknown).
const DRAW_KIND_REGISTER: u32 = 0x342;
const DRAW_KIND: u32 = 0xC600_0000;

/// The register-location value meaning "none".
const NO_REGISTER: u32 = 0x280;

/// Draw-modifier enable bits.
const START_VERTEX: u32 = 1 << 0;
const MODIFIER_BIT1: u32 = 1 << 1;
const START_INSTANCE: u32 = 1 << 2;
const DRAW_INDEX: u32 = 1 << 3;
const USER_VGPRS: u32 = 1 << 4;

pub(super) static FUNCTIONS: &[Function] = &[
    f("sceAgcDcbSetIndexSize", set_index_size),
    // SharpEmu's DriverUnknownKRzWekV120: SetIndexSize plus a fourth argument.
    f("#-KRzWekV120", set_index_size_instance_id),
    size!("sceAgcDcbSetIndexSizeGetSize", 0xC),
    f("sceAgcDcbSetIndexCount", set_index_count),
    size!("sceAgcDcbSetIndexCountGetSize", 8),
    f("sceAgcDcbSetIndexBuffer", set_index_buffer),
    size!("sceAgcDcbSetIndexBufferGetSize", 0xC),
    f("sceAgcDcbSetIndexIndirectArgs", set_index_indirect_args),
    size!("sceAgcDcbSetIndexIndirectArgsGetSize", 0x10),
    f("sceAgcDcbSetNumInstances", set_num_instances),
    size!("sceAgcDcbSetNumInstancesGetSize", 8),
    f("sceAgcDcbSetBaseIndirectArgs", set_base_indirect_args),
    size!("sceAgcDcbSetBaseDrawIndirectArgsGetSize", 0x10),
    size!("sceAgcDcbSetBaseDispatchIndirectArgsGetSize", 0x10),
    f("sceAgcDcbDrawIndex", draw_index),
    size!("sceAgcDcbDrawIndexGetSize", 0x18),
    f("sceAgcDcbDrawIndexAuto", draw_index_auto),
    size!("sceAgcDcbDrawIndexAutoGetSize", 0xC),
    f("sceAgcDcbDrawIndexOffset", draw_index_offset),
    size!("sceAgcDcbDrawIndexOffsetGetSize", 0x14),
    f("sceAgcDcbDrawIndirect", draw_indirect),
    size!("sceAgcDcbDrawIndirectGetSize", 0x14),
    f("sceAgcDcbDrawIndexIndirect", draw_index_indirect),
    size!("sceAgcDcbDrawIndexIndirectGetSize", 0x14),
    f("sceAgcDcbDrawIndexIndirectMulti", draw_index_indirect_multi),
    size!("sceAgcDcbDrawIndexIndirectMultiGetSize", 0x40),
    f("sceAgcDcbDrawIndirectMulti", draw_indirect_multi),
    size!("sceAgcDcbDrawIndirectMultiGetSize", 0x40),
    f(
        "sceAgcDcbDrawIndexMultiInstanced",
        draw_index_multi_instanced,
    ),
    size!("sceAgcDcbDrawIndexMultiInstancedGetSize", 0x3C),
    f("sceAgcCbDispatch", cb_dispatch),
    size!("sceAgcCbDispatchGetSize", 0x14),
    f("sceAgcDcbDispatchIndirect", dcb_dispatch_indirect),
    size!("sceAgcDcbDispatchIndirectGetSize", 0xC),
    f("sceAgcAcbDispatchIndirect", acb_dispatch_indirect),
    size!("sceAgcAcbDispatchIndirectGetSize", 0x10),
];

pub(super) static STUBS: &[Function] = &[];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAgc",
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

/// The draw initiator: `base` plus the modifier's bit 8 as bit 5, unless
/// modifier bit 32 is set.
fn initiator(modifier: u64, base: u32) -> u32 {
    if modifier & (1 << 32) != 0 {
        base
    } else {
        base | ((modifier as u32 >> 3) & 0x20)
    }
}

/// The shader register of the user-data slot `enable` turns on: five bits
/// at `shift`, counted from the frontmost stage's user data (bits 29-31).
fn slot_register(modifier: u32, enable: u32, shift: u32) -> u32 {
    if modifier & (enable | USER_VGPRS) != enable {
        return NO_REGISTER;
    }
    let user_data = if matches!(modifier >> 29, 3 | 5) {
        0x10C
    } else {
        0x8C
    };
    user_data + ((modifier >> shift) & 0x1F)
}

/// `(buffer, size, cache policy)`: VGT_INDEX_TYPE.
fn set_index_size(env: &mut Env) -> Result<Outcome, CallError> {
    index_type(env, 0)
}

/// `(buffer, size, cache policy, per-instance object id support)`.
fn set_index_size_instance_id(env: &mut Env) -> Result<Outcome, CallError> {
    index_type(env, env.arg(3) as u32 & 1)
}

fn index_type(env: &mut Env, instance_id: u32) -> Result<Outcome, CallError> {
    let (size, cache) = (env.arg(1) as u32 & 3, env.arg(2) as u32 & 3);
    put(
        env,
        &[
            pm4(3, IT_SET_UCONFIG_REG_INDEX, 0),
            0x2000_0000 | VGT_INDEX_TYPE,
            0x400 | size | (cache << 6) | (instance_id << 14),
        ],
    )
}

fn set_index_count(env: &mut Env) -> Result<Outcome, CallError> {
    put(env, &[pm4(2, IT_INDEX_BUFFER_SIZE, 0), env.arg(1) as u32])
}

fn set_index_buffer(env: &mut Env) -> Result<Outcome, CallError> {
    let address = env.arg(1);
    put(
        env,
        &[
            pm4(3, IT_INDEX_BASE, 0),
            address as u32 & !1,
            (address >> 32) as u32,
        ],
    )
}

/// `(buffer, address, index)`: INDEX_ATTRIBUTES_INDIRECT.
fn set_index_indirect_args(env: &mut Env) -> Result<Outcome, CallError> {
    let (address, index) = (env.arg(1), env.arg(2) as u32);
    put(
        env,
        &[
            pm4(4, IT_INDEX_ATTRIBUTES_INDIRECT, 0),
            address as u32 & !0xF,
            (address >> 32) as u32,
            index & 0xFFFF,
        ],
    )
}

fn set_num_instances(env: &mut Env) -> Result<Outcome, CallError> {
    put(env, &[pm4(2, IT_NUM_INSTANCES, 0), env.arg(1) as u32])
}

/// `(buffer, base index, address)`: SET_BASE. The console ORs the base
/// index into the header unmasked, as SharpEmu does.
fn set_base_indirect_args(env: &mut Env) -> Result<Outcome, CallError> {
    let (index, address) = (env.arg(1) as u32, env.arg(2));
    put(
        env,
        &[
            pm4(4, IT_SET_BASE, 0) | (index << 1),
            1,
            address as u32 & !7,
            (address >> 32) as u32,
        ],
    )
}

/// `(buffer, count, index address, modifier)`: DRAW_INDEX_2 only; the
/// console writes no INDEX_BASE with it.
fn draw_index(env: &mut Env) -> Result<Outcome, CallError> {
    let (count, address) = (env.arg(1) as u32, env.arg(2));
    let init = initiator(env.arg(3), 0);
    put(
        env,
        &[
            pm4(6, IT_DRAW_INDEX_2, 0),
            count,
            address as u32,
            (address >> 32) as u32,
            count,
            init,
        ],
    )
}

fn draw_index_auto(env: &mut Env) -> Result<Outcome, CallError> {
    let count = env.arg(1) as u32;
    let init = initiator(env.arg(2), 2);
    put(env, &[pm4(3, IT_DRAW_INDEX_AUTO, 0), count, init])
}

/// `(buffer, first index, count, modifier)`: DRAW_INDEX_OFFSET_2.
fn draw_index_offset(env: &mut Env) -> Result<Outcome, CallError> {
    let (offset, count) = (env.arg(1) as u32, env.arg(2) as u32);
    let init = initiator(env.arg(3), 0);
    put(
        env,
        &[
            pm4(5, IT_DRAW_INDEX_OFFSET_2, 0),
            count.max(1),
            offset,
            count,
            init,
        ],
    )
}

/// `(buffer, argument offset, modifier)`.
fn draw_indirect(env: &mut Env) -> Result<Outcome, CallError> {
    let (offset, modifier) = (env.arg(1) as u32, env.arg(2));
    let bits = modifier as u32;
    put(
        env,
        &[
            pm4(5, IT_DRAW_INDIRECT, 0),
            offset,
            slot_register(bits, START_VERTEX, 9),
            slot_register(bits, START_INSTANCE, 19),
            initiator(modifier, 2),
        ],
    )
}

/// `(buffer, argument offset, modifier)`. Modifier bit 1 names a third
/// slot (bits 14-18) the console puts beside the base vertex's.
fn draw_index_indirect(env: &mut Env) -> Result<Outcome, CallError> {
    let (offset, modifier) = (env.arg(1) as u32, env.arg(2));
    let bits = modifier as u32;
    let mut vertex = slot_register(bits, START_VERTEX, 9);
    let mut instance = slot_register(bits, START_INSTANCE, 19);
    let third = slot_register(bits, MODIFIER_BIT1, 14);
    if third != NO_REGISTER {
        vertex |= third << 16;
        instance |= 1 << 28;
    }
    put(
        env,
        &[
            pm4(5, IT_DRAW_INDEX_INDIRECT, 0),
            offset,
            vertex,
            instance,
            initiator(modifier, 0),
        ],
    )
}

/// The SET_UCONFIG_REG marking the kind of the multi draw that follows.
fn draw_kind(kind: u32) -> [u32; 3] {
    [
        pm4(3, IT_SET_UCONFIG_REG, 1),
        DRAW_KIND_REGISTER,
        DRAW_KIND | kind,
    ]
}

/// `(buffer, argument offset, flags, draw count, count address, stride,
/// modifier)`, both multi draws. Flag bit 0 reads the count from memory.
fn indirect_multi(env: &mut Env, op: u32, kind: u32, base: u32) -> Result<Outcome, CallError> {
    let offset = env.arg(1) as u32;
    let from_memory = env.arg(2) as u32 & 1;
    let (count, count_address, stride) = (env.arg(3) as u32, env.arg(4), env.arg(5) as u32);
    let modifier = arg(env, 6)?;
    let bits = modifier as u32;
    let mut draw_index = slot_register(bits, DRAW_INDEX, 24) | (from_memory << 30);
    if draw_index & 0xFFFF != NO_REGISTER {
        draw_index |= 1 << 31;
    }
    let mut packet = draw_kind(kind).to_vec();
    packet.extend([
        pm4(10, op, 0),
        offset,
        slot_register(bits, START_VERTEX, 9),
        slot_register(bits, START_INSTANCE, 19),
        draw_index,
        count,
        count_address as u32 & !3,
        (count_address >> 32) as u32,
        stride,
        initiator(modifier, base),
    ]);
    packet.extend(draw_kind(0));
    put(env, &packet)
}

fn draw_index_indirect_multi(env: &mut Env) -> Result<Outcome, CallError> {
    indirect_multi(env, IT_DRAW_INDEX_INDIRECT_MULTI, 5, 0)
}

fn draw_indirect_multi(env: &mut Env) -> Result<Outcome, CallError> {
    indirect_multi(env, IT_DRAW_INDIRECT_MULTI, 8, 2)
}

/// `(buffer, count, index address, data address, instances, modifier)`.
fn draw_index_multi_instanced(env: &mut Env) -> Result<Outcome, CallError> {
    let (count, index, data) = (env.arg(1) as u32, env.arg(2), env.arg(3));
    let instances = env.arg(4) as u32;
    let mut packet = draw_kind(6).to_vec();
    packet.extend([
        pm4(9, IT_DRAW_INDEX_MULTI_INST, 0),
        count,
        index as u32,
        (index >> 32) as u32,
        instances.max(1),
        data as u32,
        (data >> 32) as u32,
        instances,
        initiator(env.arg(5), 0x80),
    ]);
    packet.extend(draw_kind(0));
    put(env, &packet)
}

/// The dispatch initiator: COMPUTE_SHADER_EN and FORCE_START_AT_000 plus
/// the modifier bits the console keeps.
fn dispatch_initiator(modifier: u64) -> u32 {
    (modifier as u32 & 0xA038) | 0x41
}

/// `(buffer, x, y, z, modifier)`: DISPATCH_DIRECT in thread groups.
fn cb_dispatch(env: &mut Env) -> Result<Outcome, CallError> {
    let (x, y, z) = (env.arg(1) as u32, env.arg(2) as u32, env.arg(3) as u32);
    let init = dispatch_initiator(env.arg(4));
    put(env, &[pm4(5, IT_DISPATCH_DIRECT, 0), x, y, z, init])
}

/// `(buffer, argument offset, modifier)`.
fn dcb_dispatch_indirect(env: &mut Env) -> Result<Outcome, CallError> {
    let offset = env.arg(1) as u32;
    let init = dispatch_initiator(env.arg(2));
    put(env, &[pm4(3, IT_DISPATCH_INDIRECT, 0), offset, init])
}

/// `(buffer, argument address, modifier)`: the ACB has no base address.
fn acb_dispatch_indirect(env: &mut Env) -> Result<Outcome, CallError> {
    let address = env.arg(1);
    let init = dispatch_initiator(env.arg(2));
    put(
        env,
        &[
            pm4(4, IT_DISPATCH_INDIRECT, 0),
            address as u32,
            (address >> 32) as u32,
            init,
        ],
    )
}

#[cfg(test)]
pub(super) mod tests {
    use memory::{Memory, PAGE_SIZE, Prot};

    use crate::{Cpu, Env, Function, Hle, Kernel, Outcome, Thread, TlsTemplate, Vfs};

    const SIZES: &str =
        "../../references/src/prosper/prosper/tests/data/console_oracle/agc.golden.tsv";
    const BUILDS: &str =
        "../../references/src/prosper/prosper/tests/data/console_oracle/agcbuild.golden.tsv";
    const DESCRIPTOR: u64 = 0x30_0000;
    const BUFFER: u64 = 0x31_0000;
    const STACK: u64 = 0x10_0000;

    /// Runs `func` with `args` (after the descriptor) and zeroed stack
    /// arguments; (return value, the first `dwords` of the buffer).
    pub fn call(func: crate::HostFn, args: &[u64], dwords: usize) -> (u64, Vec<u32>) {
        let mem = Memory::new(1 << 40).unwrap();
        mem.map(STACK, 2 * PAGE_SIZE, Prot::RW, "stack").unwrap();
        mem.map(DESCRIPTOR, PAGE_SIZE, Prot::RW, "descriptor")
            .unwrap();
        mem.map(BUFFER, 4 * PAGE_SIZE, Prot::RW, "buffer").unwrap();
        mem.write_u64(DESCRIPTOR + 0x10, BUFFER).unwrap();
        mem.write_u64(DESCRIPTOR + 0x18, BUFFER + 4 * PAGE_SIZE)
            .unwrap();
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
        cpu.regs[x86::cpu::reg::RSP] = STACK + PAGE_SIZE;
        const ORDER: [usize; 6] = [7, 6, 2, 1, 8, 9];
        cpu.regs[ORDER[0]] = DESCRIPTOR;
        for (i, &a) in args.iter().take(5).enumerate() {
            cpu.regs[ORDER[i + 1]] = a;
        }
        let mem = kernel.memory();
        let mut env = Env {
            cpu: &mut cpu,
            mem: &mem,
            kernel: &kernel,
            thread: Thread { tcb: 0 },
        };
        let Ok(Outcome::Return(r)) = func(&mut env) else {
            panic!("builder failed");
        };
        let mut b = vec![0u8; 4 * dwords];
        mem.read(BUFFER, &mut b).unwrap();
        let d = b
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (if r == BUFFER { u64::MAX } else { r }, d)
    }

    fn int(s: &str) -> Option<u64> {
        match s.strip_prefix("0x") {
            Some(h) => u64::from_str_radix(h, 16).ok(),
            None => s.parse().ok(),
        }
    }

    /// The measured `(args, dwords)` of builder `name`, where it returned
    /// the start of the buffer.
    pub fn measured(name: &str) -> Option<Vec<(Vec<u64>, Vec<u32>)>> {
        let text = std::fs::read_to_string(BUILDS).ok()?;
        let rows = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|c| c.len() >= 9 && c[2] == name && c[7] == "ptr+0")
            .map(|c| {
                let args = c[3]
                    .split(',')
                    .skip(1)
                    .map(|a| int(a.strip_prefix("i:").unwrap()).unwrap())
                    .collect();
                let hex = c[8].split_once(':').unwrap().1;
                let dwords = (0..hex.len() / 8)
                    .map(|i| {
                        let b = u32::from_str_radix(&hex[8 * i..8 * i + 8], 16).unwrap();
                        b.swap_bytes()
                    })
                    .collect();
                (args, dwords)
            })
            .collect();
        Some(rows)
    }

    /// Checks every implemented `...GetSize` in `functions` against the
    /// console's answers; how many it checked.
    pub fn check_sizes(functions: &[Function]) -> usize {
        let Ok(text) = std::fs::read_to_string(SIZES) else {
            eprintln!("skipped: {SIZES} not fetched");
            return 0;
        };
        let mut checked = 0;
        for c in text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split('\t').collect::<Vec<_>>())
        {
            if c.len() < 7 || c[5] != "ok" {
                continue;
            }
            let Some(func) = functions.iter().find(|f| f.name == c[2]) else {
                continue;
            };
            let args: Vec<u64> = match c[3] {
                "-" => Vec::new(),
                a => a
                    .split(',')
                    .map(|a| int(a.strip_prefix("i:").unwrap()).unwrap())
                    .collect(),
            };
            let mut cpu = Cpu::default();
            const ORDER: [usize; 6] = [7, 6, 2, 1, 8, 9];
            for (i, &a) in args.iter().enumerate() {
                cpu.regs[ORDER[i]] = a;
            }
            let mem = Memory::new(1 << 32).unwrap();
            let hle = Hle::new();
            let kernel = Kernel::new(
                hle,
                mem,
                TlsTemplate::default(),
                Box::new(std::io::sink()),
                Box::new(std::io::sink()),
                Vfs::default(),
            );
            let mem = kernel.memory();
            let mut env = Env {
                cpu: &mut cpu,
                mem: &mem,
                kernel: &kernel,
                thread: Thread { tcb: 0 },
            };
            let Ok(Outcome::Return(r)) = (func.func)(&mut env) else {
                panic!("{} failed", c[2]);
            };
            assert_eq!(r, int(c[6]).unwrap(), "{} {}", c[2], c[3]);
            checked += 1;
        }
        checked
    }

    #[test]
    fn sizes_match_the_console() {
        eprintln!("checked {} sizes", check_sizes(super::FUNCTIONS));
    }

    /// The multi draws take the modifier on the stack, which the shared
    /// oracle leaves unmapped; the console's was garbage without enables.
    #[test]
    fn multi_draws_match_the_console_with_a_zero_modifier() {
        for (name, func) in [
            (
                "sceAgcDcbDrawIndexIndirectMulti",
                super::draw_index_indirect_multi as crate::HostFn,
            ),
            ("sceAgcDcbDrawIndirectMulti", super::draw_indirect_multi),
        ] {
            let Some(rows) = measured(name) else { return };
            for (args, expect) in rows {
                let (r, ours) = call(func, &args, expect.len());
                assert_eq!(r, u64::MAX, "{name}");
                assert_eq!(ours, expect, "{name} {args:x?}");
            }
        }
    }
}
