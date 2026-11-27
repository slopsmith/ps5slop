//! `sceAgcCreateShader`: fixes up a shader object the title loaded, so its
//! pointers are absolute and its program-address registers point at the
//! code. Ported from SharpEmu's `AgcExports.ShaderState.cs`.

use super::{ERROR_INVALID_ARGUMENT, ret};
use crate::pthread::lock;
use crate::{CallError, Env, Outcome};

const FILE_HEADER: u32 = 0x3433_3231;
const VERSION: u32 = 0x18;

/// Offsets in the shader header.
const USER_DATA: u64 = 0x08;
const CODE: u64 = 0x10;
const CX_REGISTERS: u64 = 0x18;
const SH_REGISTERS: u64 = 0x20;
const SPECIALS: u64 = 0x28;
const INPUT_SEMANTICS: u64 = 0x30;
const OUTPUT_SEMANTICS: u64 = 0x38;
const NUM_INPUT_SEMANTICS: u64 = 0x50;
const NUM_OUTPUT_SEMANTICS: u64 = 0x56;
const TYPE: u64 = 0x5A;
const NUM_SH_REGISTERS: u64 = 0x5C;
/// Size of a shader header.
const HEADER_SIZE: usize = 0x60;

/// Shader types.
const COMPUTE: u8 = 0;
const PS: u8 = 1;
const GS: u8 = 2;
const HS: u8 = 3;
const GS_FRONT: u8 = 4;
const HS_FRONT: u8 = 5;
const GS_BACK: u8 = 6;
const HS_BACK: u8 = 7;
const FUNCTION: u8 = 8;

/// Returned when the registers lack the program address (from SharpEmu).
const ERROR_INCOMPLETE_REGISTERS: u64 = 0x8A6C_0005;

/// Program address registers (low, high), as SH register offsets.
const COMPUTE_PGM: (u32, u32) = (0x20C, 0x20D);
const PS_PGM: (u32, u32) = (0x8, 0x9);
const VS_PGM: (u32, u32) = (0x48, 0x49);
const ES_PGM: (u32, u32) = (0xC8, 0xC9);
const GS_PGM: (u32, u32) = (0x88, 0x89);
const HS_PGM: (u32, u32) = (0x108, 0x109);
const LS_PGM: (u32, u32) = (0x148, 0x149);
const PGM_PAIRS: [(u32, u32); 7] = [COMPUTE_PGM, PS_PGM, VS_PGM, ES_PGM, GS_PGM, HS_PGM, LS_PGM];

fn read_u32(env: &Env, at: u64) -> Result<u32, CallError> {
    let mut b = [0u8; 4];
    env.mem.read(at, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u8(env: &Env, at: u64) -> Result<u8, CallError> {
    let mut b = [0u8; 1];
    env.mem.read(at, &mut b)?;
    Ok(b[0])
}

/// Turns the offset stored at `field` (relative to `field`) into an address.
fn relocate(env: &Env, field: u64) -> Result<(), CallError> {
    let rel = env.mem.read_u64(field)?;
    if rel != 0 {
        env.mem.write_u64(field, field.wrapping_add(rel))?;
    }
    Ok(())
}

/// `sceAgcCreateShader(&shader, header, code)`.
pub(super) fn create_shader(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, header, code) = (env.arg(0), env.arg(1), env.arg(2));
    if header == 0 || code == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    if read_u32(env, header)? != FILE_HEADER || read_u32(env, header + 4)? != VERSION {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    for field in [
        CX_REGISTERS,
        SH_REGISTERS,
        USER_DATA,
        SPECIALS,
        INPUT_SEMANTICS,
        OUTPUT_SEMANTICS,
    ] {
        relocate(env, header + field)?;
    }
    env.mem.write_u64(header + CODE, code)?;
    let user_data = env.mem.read_u64(header + USER_DATA)?;
    if user_data != 0 {
        for k in 0..5 {
            relocate(env, user_data + 8 * k)?;
        }
    }
    let r = patch_program_registers(env, header, code)?;
    if r != 0 {
        return ret(r);
    }
    if out != 0 {
        env.mem.write_u64(out, header)?;
    }
    lock(&env.kernel.agc.shaders).insert(code, header);
    ret(0)
}

/// Writes the code address into the shader's program-address register pair:
/// the one its type uses, else any pair it has. Returns an error code or 0.
fn patch_program_registers(env: &Env, header: u64, code: u64) -> Result<u64, CallError> {
    let regs = env.mem.read_u64(header + SH_REGISTERS)?;
    let kind = read_u8(env, header + TYPE)?;
    let count = read_u8(env, header + NUM_SH_REGISTERS)? as u64;
    if kind > FUNCTION {
        return Ok(ERROR_INVALID_ARGUMENT);
    }
    let may_lack = matches!(kind, GS_FRONT | HS_FRONT | FUNCTION);
    if count == 0 {
        return Ok(if may_lack {
            0
        } else {
            ERROR_INCOMPLETE_REGISTERS
        });
    }
    if regs == 0 {
        return Ok(ERROR_INCOMPLETE_REGISTERS);
    }
    let offsets: Vec<u32> = (0..count)
        .map(|i| read_u32(env, regs + 8 * i))
        .collect::<Result<_, _>>()?;
    if kind == FUNCTION {
        return Ok(0);
    }
    let preferred = match kind {
        COMPUTE => COMPUTE_PGM,
        PS => PS_PGM,
        GS | GS_BACK => ES_PGM,
        HS => VS_PGM,
        GS_FRONT => GS_PGM,
        HS_FRONT => HS_PGM,
        HS_BACK => LS_PGM,
        _ => return Ok(ERROR_INVALID_ARGUMENT),
    };
    let find = |reg: u32| offsets.iter().position(|&o| o == reg);
    let pair = match (find(preferred.0), find(preferred.1)) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        _ => offsets.iter().enumerate().find_map(|(lo, &o)| {
            let &(_, high) = PGM_PAIRS.iter().find(|p| p.0 == o)?;
            Some((lo, find(high)?))
        }),
    };
    let Some((lo, hi)) = pair else {
        let has_any = offsets
            .iter()
            .any(|o| PGM_PAIRS.iter().any(|p| p.0 == *o || p.1 == *o));
        return Ok(if may_lack && !has_any {
            0
        } else {
            ERROR_INCOMPLETE_REGISTERS
        });
    };
    env.mem.write(
        regs + 8 * lo as u64 + 4,
        &((code >> 8) as u32).to_le_bytes(),
    )?;
    env.mem.write(
        regs + 8 * hi as u64 + 4,
        &(((code >> 40) & 0xff) as u32).to_le_bytes(),
    )?;
    Ok(0)
}

/// `SPI_PS_INPUT_CNTL_0`, the first of 32 per-attribute pixel input controls.
const SPI_PS_INPUT_CNTL_0: u32 = 0x191;
const INTERPOLANTS: usize = 32;

/// Replaces the two bits at `shift` in `value` with the low bits of `field`.
fn two_bits(value: u32, field: u32, shift: u32) -> u32 {
    (value & !(3 << shift)) | ((field & 3) << shift)
}

/// The `SPI_PS_INPUT_CNTL` value for pixel input semantic `ps`, given the
/// matching geometry output semantic `gs` (same semantic id in the low byte).
fn interpolant(ps: u32, gs: Option<u32>) -> u32 {
    let value = if ps & 0x0030_0000 != 0 {
        let mut v = (ps << 4) & 0x0300_0000;
        match gs {
            None => v |= 0x0018_0020,
            Some(gs) => {
                let common = ps & gs;
                v &= 0xFFF7_FFDF;
                v |= (common >> 15) & 0x20;
                v ^= 0x0008_0020;
                v = (v & !0x0010_0000) | ((!common >> 1) & 0x0010_0000);
            }
        }
        two_bits(v, ps >> 30, 21)
    } else if ps & 0x0100_0000 != 0 || gs.is_none() {
        0x20
    } else {
        0
    };
    let value = match gs {
        Some(gs) => {
            let flat = if ps & 0x0140_0000 != 0 { 0x400 } else { 0 };
            (value & !0x41F) | ((gs >> 8) & 0x1F) | flat
        }
        None => value & !0x41F,
    };
    two_bits(value, ps >> 28, 8)
}

/// SharpEmu's `sceAgcCreateInterpolantMapping(registers, geometry, pixel)`:
/// 32 (register, value) pairs for `SPI_PS_INPUT_CNTL_0..31`, mapping each
/// pixel shader input to the geometry shader output with its semantic.
pub(super) fn create_interpolant_mapping(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, geometry, pixel) = (env.arg(0), env.arg(1), env.arg(2));
    if out == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    let mut values: Vec<u32> = (0..INTERPOLANTS as u32).collect();
    let (inputs, count) = match pixel {
        0 => (0, 0),
        p => (
            env.mem.read_u64(p + INPUT_SEMANTICS)?,
            (read_u32(env, p + NUM_INPUT_SEMANTICS)? as usize).min(INTERPOLANTS),
        ),
    };
    if inputs != 0 && count != 0 {
        if geometry == 0 {
            return ret(ERROR_INVALID_ARGUMENT);
        }
        let outputs = env.mem.read_u64(geometry + OUTPUT_SEMANTICS)?;
        let mut n = [0u8; 2];
        env.mem.read(geometry + NUM_OUTPUT_SEMANTICS, &mut n)?;
        let n = if outputs == 0 {
            0
        } else {
            u16::from_le_bytes(n) as u64
        };
        let gs: Vec<u32> = (0..n)
            .map(|i| read_u32(env, outputs + 4 * i))
            .collect::<Result<_, _>>()?;
        for (i, v) in values.iter_mut().take(count).enumerate() {
            let ps = read_u32(env, inputs + 4 * i as u64)?;
            let matched = gs.iter().copied().find(|&g| g as u8 == ps as u8);
            *v = interpolant(ps, matched);
        }
    }
    let bytes: Vec<u8> = values
        .iter()
        .enumerate()
        .flat_map(|(i, v)| [SPI_PS_INPUT_CNTL_0 + i as u32, *v])
        .flat_map(u32::to_le_bytes)
        .collect();
    env.mem.write(out, &bytes)?;
    ret(0)
}

/// Offsets in a shader's "specials" block of (register, value) pairs.
const SPECIAL_GE_CNTL: u64 = 0x00;
const SPECIAL_STAGES_EN: u64 = 0x08;
const SPECIAL_GS_OUT_PRIM_TYPE: u64 = 0x20;
const SPECIAL_GE_USER_VGPR_EN: u64 = 0x28;
const VGT_SHADER_STAGES_GS_EN: u32 = 1 << 5;
const VGT_GS_OUT_PRIM_TYPE: u32 = 0x29B;
const VGT_PRIMITIVE_TYPE: u32 = 0x242;

/// `VGT_GS_OUT_PRIM_TYPE` for primitive type `prim` (points, lines,
/// rectangles or triangles).
fn gs_out(prim: u32) -> u32 {
    match prim {
        1 => 0,
        2 | 3 | 10 | 11 | 18 => 1,
        7 => 3,
        0x11 => 4,
        _ => 2,
    }
}

fn write_u32(env: &Env, at: u64, v: u32) -> Result<(), CallError> {
    env.mem.write(at, &v.to_le_bytes())?;
    Ok(())
}

fn copy_pair(env: &Env, from: u64, to: u64) -> Result<(), CallError> {
    write_u32(env, to, read_u32(env, from)?)?;
    write_u32(env, to + 4, read_u32(env, from + 4)?)
}

/// `sceAgcCreatePrimState(cx, uc, hull, geometry, prim)`: the context pairs
/// (shader stages, GS output primitive) and uconfig pairs (GE control, user
/// VGPRs, primitive type) for drawing `prim` with these shaders.
pub(super) fn create_prim_state(env: &mut Env) -> Result<Outcome, CallError> {
    let (cx, uc, hull, geometry, prim) = (
        env.arg(0),
        env.arg(1),
        env.arg(2),
        env.arg(3),
        env.arg(4) as u32,
    );
    if cx == 0 && uc == 0 {
        return ret(0);
    }
    if geometry == 0 || !matches!(read_u8(env, geometry + TYPE)?, GS | GS_BACK) {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    let specials = env.mem.read_u64(geometry + SPECIALS)?;
    let hull_specials = match hull {
        0 => 0,
        h if read_u8(env, h + TYPE)? == HS => env.mem.read_u64(h + SPECIALS)?,
        _ => return ret(ERROR_INVALID_ARGUMENT),
    };
    if specials == 0 || (hull != 0 && hull_specials == 0) {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    if cx != 0 {
        let stages_reg = read_u32(env, specials + SPECIAL_STAGES_EN)?;
        let mut stages = read_u32(env, specials + SPECIAL_STAGES_EN + 4)?;
        let (mut out_reg, mut out) = if stages & VGT_SHADER_STAGES_GS_EN != 0 {
            (
                read_u32(env, specials + SPECIAL_GS_OUT_PRIM_TYPE)?,
                read_u32(env, specials + SPECIAL_GS_OUT_PRIM_TYPE + 4)?,
            )
        } else {
            (VGT_GS_OUT_PRIM_TYPE, gs_out(prim))
        };
        if hull_specials != 0 {
            stages |= read_u32(env, hull_specials + SPECIAL_STAGES_EN + 4)?;
            if stages & VGT_SHADER_STAGES_GS_EN == 0 {
                out_reg = read_u32(env, hull_specials + SPECIAL_GS_OUT_PRIM_TYPE)?;
                out = read_u32(env, hull_specials + SPECIAL_GS_OUT_PRIM_TYPE + 4)?;
            }
        }
        for (k, v) in [stages_reg, stages, out_reg, out].into_iter().enumerate() {
            write_u32(env, cx + 4 * k as u64, v)?;
        }
    }
    if uc != 0 {
        copy_pair(env, specials + SPECIAL_GE_CNTL, uc)?;
        let vgprs = if hull_specials != 0 {
            hull_specials
        } else {
            specials
        };
        copy_pair(env, vgprs + SPECIAL_GE_USER_VGPR_EN, uc + 8)?;
        write_u32(env, uc + 16, VGT_PRIMITIVE_TYPE)?;
        write_u32(env, uc + 20, prim)?;
    }
    ret(0)
}

/// `sceAgcUpdatePrimState(cx, uc, prim)`: changes the primitive type in
/// state made by [`create_prim_state`].
pub(super) fn update_prim_state(env: &mut Env) -> Result<Outcome, CallError> {
    let (cx, uc, prim) = (env.arg(0), env.arg(1), env.arg(2) as u32);
    if cx != 0 {
        let stages = read_u32(env, cx + 4)?;
        if stages & 0x24 == 0 {
            let out = read_u32(env, cx + 12)?;
            write_u32(env, cx + 12, (out & !7) | gs_out(prim))?;
        }
    }
    if uc != 0 {
        let input = read_u32(env, uc + 20)?;
        write_u32(env, uc + 20, (input & !0x1F) | prim)?;
    }
    ret(0)
}

/// The `SPI_PS_INPUT_CNTL` value `CreateInterpolantMapping2` gives pixel
/// input `ps`, with the matching geometry output semantic `gs`, if any.
fn interpolant2(ps: u32, gs: Option<u32>) -> u32 {
    let mode = (ps >> 20) & 3;
    let mut f = if mode == 0 {
        let f = (((ps >> 24) & 1) | u32::from(gs.is_none())) << 5;
        two_bits(f, ps >> 28, 8)
    } else {
        let mut f = ((ps << 4) & 0x0300_0000) + 0x0008_0000;
        if mode == 2 {
            f &= 0xFFEF_FFDF;
            f |= gs.map_or(0x20, |m| (!(m & ps) >> 16) & 0x20);
            f = two_bits(f, ps >> 30, 8);
        } else {
            match gs {
                Some(m) => {
                    let masked = m & ps;
                    f = (f & 0xFFFF_FFDF) | ((masked >> 15) & 0x20);
                    f ^= 0x20;
                    f = (f & 0xFFEF_FFFF) | ((!masked >> 1) & 0x0010_0000);
                }
                None => f |= 0x0010_0020,
            }
            f = two_bits(f, ps >> 28, 8);
        }
        two_bits(f, ps >> 30, 21)
    };
    f = match gs {
        Some(m) => {
            f = (f & 0xFFFF_FFE0) | ((m >> 8) & 0x1F);
            let flat = if ps & 0x0040_0000 != 0 {
                0x400
            } else {
                (ps >> 14) & 0x400
            };
            (f & 0xFFFF_FBFF) | flat
        }
        None => f & 0xFFFF_FBE0,
    };
    f
}

/// SharpEmu's `sceAgcCreateInterpolantMapping2(registers, geometry, pixel)`:
/// like [`create_interpolant_mapping`], with the newer value encoding.
pub(super) fn create_interpolant_mapping2(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, geometry, pixel) = (env.arg(0), env.arg(1), env.arg(2));
    if out == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    let mut values: Vec<u32> = (0..INTERPOLANTS as u32).collect();
    let (inputs, count) = match pixel {
        0 => (0, 0),
        p => (
            env.mem.read_u64(p + INPUT_SEMANTICS)?,
            read_u32(env, p + NUM_INPUT_SEMANTICS)? as usize,
        ),
    };
    if count != 0 {
        if inputs == 0 || geometry == 0 || count > INTERPOLANTS {
            return ret(ERROR_INVALID_ARGUMENT);
        }
        let outputs = env.mem.read_u64(geometry + OUTPUT_SEMANTICS)?;
        let mut n = [0u8; 2];
        env.mem.read(geometry + NUM_OUTPUT_SEMANTICS, &mut n)?;
        let n = u16::from_le_bytes(n) as u64;
        if n != 0 && outputs == 0 {
            return ret(ERROR_INVALID_ARGUMENT);
        }
        let gs: Vec<u32> = (0..n)
            .map(|i| read_u32(env, outputs + 4 * i))
            .collect::<Result<_, _>>()?;
        for (i, v) in values.iter_mut().take(count).enumerate() {
            let ps = read_u32(env, inputs + 4 * i as u64)?;
            *v = interpolant2(ps, gs.iter().copied().find(|&g| g as u8 == ps as u8));
        }
    }
    let bytes: Vec<u8> = values
        .iter()
        .enumerate()
        .flat_map(|(i, v)| [SPI_PS_INPUT_CNTL_0 + i as u32, *v])
        .flat_map(u32::to_le_bytes)
        .collect();
    env.mem.write(out, &bytes)?;
    ret(0)
}

/// SH registers a fused shader takes over from its halves.
const SPI_SHADER_PGM_CHKSUM_GS: u32 = 0x80;
const SPI_SHADER_PGM_CHKSUM_HS: u32 = 0x100;
const SPI_SHADER_PGM_RSRC1_GS: u32 = 0x8A;
const SPI_SHADER_PGM_RSRC1_HS: u32 = 0x10A;
/// Wave32 bits of `VGT_SHADER_STAGES_EN`, which both halves must agree on.
const STAGES_GS_W32: u32 = 1 << 22;
const STAGES_HS_W32: u32 = 1 << 21;

/// Whether `front` and `back` are the two halves of one geometry or hull
/// shader, and if so, whether it is a geometry pair.
fn fused_pair(env: &Env, front: u64, back: u64) -> Result<Option<bool>, CallError> {
    Ok(
        match (read_u8(env, front + TYPE)?, read_u8(env, back + TYPE)?) {
            (GS_FRONT, GS_BACK) => Some(true),
            (HS_FRONT, HS_BACK) => Some(false),
            _ => None,
        },
    )
}

/// SharpEmu's `sceAgcGetFusedShaderSize(out, front, back)`: the size and
/// alignment of the memory `sceAgcFuseShaderHalves` copies the back half's
/// SH registers into.
pub(super) fn fused_shader_size(env: &mut Env) -> Result<Outcome, CallError> {
    let (out, front, back) = (env.arg(0), env.arg(1), env.arg(2));
    if out == 0 || front == 0 || back == 0 || fused_pair(env, front, back)?.is_none() {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    let count = read_u8(env, back + NUM_SH_REGISTERS)? as u64;
    env.mem.write_u64(out, count * 8)?;
    env.mem.write_u64(out + 8, 4)?;
    ret(0)
}

/// The `n`th (register, value) pair for SH register `reg` in `count` pairs
/// at `regs`.
fn find_register(
    env: &Env,
    regs: u64,
    count: u64,
    reg: u32,
    n: usize,
) -> Result<Option<u64>, CallError> {
    if regs == 0 {
        return Ok(None);
    }
    let mut seen = 0;
    for k in 0..count {
        if read_u32(env, regs + 8 * k)? == reg {
            if seen == n {
                return Ok(Some(regs + 8 * k));
            }
            seen += 1;
        }
    }
    Ok(None)
}

/// Replaces the field `mask << shift` of `into` with the larger of its value
/// and `from`'s.
fn max_field(into: u32, from: u32, shift: u32, mask: u32) -> u32 {
    let m = ((into >> shift) & mask).max((from >> shift) & mask);
    (into & !(mask << shift)) | (m << shift)
}

/// Merges the front half's `SPI_SHADER_PGM_RSRC1/2` into the fused copy.
fn merge_resources(
    env: &Env,
    front: u64,
    regs: u64,
    count: u64,
    geometry: bool,
) -> Result<(), CallError> {
    let rsrc1 = if geometry {
        SPI_SHADER_PGM_RSRC1_GS
    } else {
        SPI_SHADER_PGM_RSRC1_HS
    };
    let front_regs = env.mem.read_u64(front + SH_REGISTERS)?;
    let front_count = read_u8(env, front + NUM_SH_REGISTERS)? as u64;
    let pairs = |r: u64, c: u64| -> Result<Option<(u64, u64)>, CallError> {
        Ok(find_register(env, r, c, rsrc1, 0)?.zip(find_register(env, r, c, rsrc1 + 1, 0)?))
    };
    let (Some((f1, f2)), Some((b1, b2))) = (pairs(front_regs, front_count)?, pairs(regs, count)?)
    else {
        return Ok(());
    };
    let (front1, front2) = (read_u32(env, f1 + 4)?, read_u32(env, f2 + 4)?);
    let (back1, back2) = (read_u32(env, b1 + 4)?, read_u32(env, b2 + 4)?);
    let front_private = ((front1 & 0x3F) + 1) * 4;
    let back_private = ((back1 & 0x3F) + 1) * 4;
    let front_total = front_private + (front2 >> 28) * 8;
    let back_total = back_private + (back2 >> 28) * 8;
    let most = front_total.max(back_total);
    let shared = if front_private.max(back_private) >= most {
        0
    } else {
        (most - front_total.min(back_total) + 7) / 64
    };
    let merged1 = max_field(back1, front1, 0, 0x3F);
    let merged1 = max_field(merged1, front1, if geometry { 29 } else { 28 }, 3);
    let mut merged2 = (back2 & 0x0FFF_FFFF) | ((shared & 15) << 28);
    if geometry {
        merged2 = max_field(merged2, front2, 16, 3);
        merged2 = (merged2 & !(1 << 18)) | (front2 & (1 << 18));
    }
    const FRONT_OWNED: u32 = 0x0800_003E;
    merged2 = (merged2 & !FRONT_OWNED) | (front2 & FRONT_OWNED);
    write_u32(env, b1 + 4, merged1)?;
    write_u32(env, b2 + 4, merged2)
}

/// SharpEmu's `sceAgcFuseShaderHalves(fused, front, back, memory)`: one
/// shader from the two halves of a geometry or hull shader. It is the back
/// half's header with its SH registers copied to `memory`, the front half's
/// resource needs and checksums merged in, and the program address set to
/// the front half's code, which runs first. The renderer does not yet run
/// the back half after it.
pub(super) fn fuse_shader_halves(env: &mut Env) -> Result<Outcome, CallError> {
    let (fused, front, back, memory) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    if fused == 0 || front == 0 || back == 0 {
        return ret(ERROR_INVALID_ARGUMENT);
    }
    let Some(geometry) = fused_pair(env, front, back)? else {
        return ret(ERROR_INVALID_ARGUMENT);
    };
    let (front_specials, back_specials) = (
        env.mem.read_u64(front + SPECIALS)?,
        env.mem.read_u64(back + SPECIALS)?,
    );
    if front_specials != 0 && back_specials != 0 {
        let w32 = if geometry {
            STAGES_GS_W32
        } else {
            STAGES_HS_W32
        };
        let front_stages = read_u32(env, front_specials + SPECIAL_STAGES_EN + 4)?;
        let back_stages = read_u32(env, back_specials + SPECIAL_STAGES_EN + 4)?;
        if (front_stages ^ back_stages) & w32 != 0 {
            return ret(ERROR_INVALID_ARGUMENT);
        }
    }
    let back_regs = env.mem.read_u64(back + SH_REGISTERS)?;
    let count = read_u8(env, back + NUM_SH_REGISTERS)? as u64;
    let front_code = env.mem.read_u64(front + CODE)?;
    let mut header = [0u8; HEADER_SIZE];
    env.mem.read(back, &mut header)?;
    env.mem.write(fused, &header)?;
    let regs = if memory != 0 && back_regs != 0 && count != 0 {
        let mut bytes = vec![0u8; count as usize * 8];
        env.mem.read(back_regs, &mut bytes)?;
        env.mem.write(memory, &bytes)?;
        memory
    } else {
        back_regs
    };
    env.mem
        .write(fused + TYPE, &[if geometry { GS } else { HS }])?;
    env.mem.write_u64(fused + USER_DATA, 0)?;
    env.mem.write_u64(fused + SH_REGISTERS, regs)?;
    merge_resources(env, front, regs, count, geometry)?;
    let checksum = if geometry {
        SPI_SHADER_PGM_CHKSUM_GS
    } else {
        SPI_SHADER_PGM_CHKSUM_HS
    };
    let front_regs = env.mem.read_u64(front + SH_REGISTERS)?;
    let front_count = read_u8(env, front + NUM_SH_REGISTERS)? as u64;
    for n in 0..2 {
        if let (Some(to), Some(from)) = (
            find_register(env, regs, count, checksum, n)?,
            find_register(env, front_regs, front_count, checksum, n)?,
        ) {
            write_u32(env, to + 4, read_u32(env, from + 4)?)?;
        }
    }
    let lo = if geometry { ES_PGM.0 } else { LS_PGM.0 };
    if let Some(at) = find_register(env, regs, count, lo, 0)?
        && at + 8 < regs + count * 8
        && read_u32(env, at + 8)? == lo + 1
    {
        let hi = read_u32(env, at + 12)?;
        write_u32(env, at + 4, (front_code >> 8) as u32)?;
        write_u32(
            env,
            at + 12,
            (hi & 0xFFFF_FF00) | ((front_code >> 40) & 0xFF) as u32,
        )?;
    }
    ret(0)
}

#[cfg(test)]
mod tests {
    use memory::{Memory, PAGE_SIZE, Prot};

    use super::*;
    use crate::{Cpu, Hle, Kernel, Thread, TlsTemplate, Vfs};

    const AREA: u64 = 0x30_0000;
    const FRONT: u64 = AREA;
    const BACK: u64 = AREA + 0x100;
    const FUSED: u64 = AREA + 0x200;
    const FRONT_REGS: u64 = AREA + 0x300;
    const BACK_REGS: u64 = AREA + 0x400;
    const COPY: u64 = AREA + 0x500;
    const FRONT_CODE: u64 = 0x12_3456_7890_0000;
    const BACK_CODE: u64 = 0x99_8877_6600;

    fn header(mem: &Memory, at: u64, kind: u8, code: u64, regs: u64, pairs: &[(u32, u32)]) {
        mem.write_u64(at + CODE, code).unwrap();
        mem.write_u64(at + SH_REGISTERS, regs).unwrap();
        mem.write(at + TYPE, &[kind]).unwrap();
        mem.write(at + NUM_SH_REGISTERS, &[pairs.len() as u8])
            .unwrap();
        for (k, &(reg, value)) in pairs.iter().enumerate() {
            let bytes = [reg.to_le_bytes(), value.to_le_bytes()].concat();
            mem.write(regs + 8 * k as u64, &bytes).unwrap();
        }
    }

    fn pair(mem: &Memory, at: u64) -> (u32, u32) {
        let mut b = [0u8; 8];
        mem.read(at, &mut b).unwrap();
        (
            u32::from_le_bytes(b[..4].try_into().unwrap()),
            u32::from_le_bytes(b[4..].try_into().unwrap()),
        )
    }

    fn call(f: crate::HostFn, kernel: &Kernel, args: &[u64]) -> u64 {
        let mut cpu = Cpu::default();
        const ORDER: [usize; 4] = [7, 6, 2, 1];
        for (k, &a) in args.iter().enumerate() {
            cpu.regs[ORDER[k]] = a;
        }
        let mem = kernel.memory();
        let mut env = Env {
            cpu: &mut cpu,
            mem: &mem,
            kernel,
            thread: Thread { tcb: 0 },
        };
        match f(&mut env) {
            Ok(Outcome::Return(r)) => r,
            _ => panic!("call failed"),
        }
    }

    #[test]
    fn fusing_geometry_halves_keeps_the_back_and_starts_at_the_front() {
        let mem = Memory::new(1 << 40).unwrap();
        mem.map(AREA, PAGE_SIZE, Prot::RW, "shaders").unwrap();
        header(
            &mem,
            FRONT,
            GS_FRONT,
            FRONT_CODE,
            FRONT_REGS,
            &[
                (SPI_SHADER_PGM_RSRC1_GS, 0x0000_0009),
                (SPI_SHADER_PGM_RSRC1_GS + 1, 0x2000_0002),
                (SPI_SHADER_PGM_CHKSUM_GS, 0xF00D),
            ],
        );
        let back_pairs = [
            (ES_PGM.0, 0),
            (ES_PGM.1, 0xAB00),
            (SPI_SHADER_PGM_RSRC1_GS, 0x0000_0003),
            (SPI_SHADER_PGM_RSRC1_GS + 1, 0x1000_0000),
            (SPI_SHADER_PGM_CHKSUM_GS, 0xBEEF),
        ];
        header(&mem, BACK, GS_BACK, BACK_CODE, BACK_REGS, &back_pairs);
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
        let mem = kernel.memory();

        let size_out = AREA + 0x600;
        assert_eq!(
            call(fused_shader_size, &kernel, &[size_out, FRONT, BACK]),
            0
        );
        assert_eq!(mem.read_u64(size_out).unwrap(), 8 * back_pairs.len() as u64);
        assert_eq!(
            call(fused_shader_size, &kernel, &[size_out, BACK, FRONT]),
            ERROR_INVALID_ARGUMENT
        );

        assert_eq!(
            call(fuse_shader_halves, &kernel, &[FUSED, FRONT, BACK, COPY]),
            0
        );
        let mut kind = [0u8];
        mem.read(FUSED + TYPE, &mut kind).unwrap();
        assert_eq!(kind[0], GS);
        assert_eq!(mem.read_u64(FUSED + CODE).unwrap(), BACK_CODE);
        assert_eq!(mem.read_u64(FUSED + SH_REGISTERS).unwrap(), COPY);
        assert_eq!(pair(&mem, COPY), (ES_PGM.0, (FRONT_CODE >> 8) as u32));
        assert_eq!(pair(&mem, COPY + 8), (ES_PGM.1, 0xAB34));
        assert_eq!(pair(&mem, COPY + 16).1 & 0x3F, 9);
        assert_eq!(pair(&mem, COPY + 32).1, 0xF00D);
        assert_eq!(pair(&mem, BACK_REGS + 32).1, 0xBEEF);
    }
}
