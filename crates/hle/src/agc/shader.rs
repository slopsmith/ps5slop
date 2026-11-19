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
const TYPE: u64 = 0x5A;
const NUM_SH_REGISTERS: u64 = 0x5C;

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
    for field in [CX_REGISTERS, SH_REGISTERS, USER_DATA, SPECIALS, INPUT_SEMANTICS, OUTPUT_SEMANTICS] {
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
        return Ok(if may_lack { 0 } else { ERROR_INCOMPLETE_REGISTERS });
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
        let has_any = offsets.iter().any(|o| PGM_PAIRS.iter().any(|p| p.0 == *o || p.1 == *o));
        return Ok(if may_lack && !has_any { 0 } else { ERROR_INCOMPLETE_REGISTERS });
    };
    env.mem.write(regs + 8 * lo as u64 + 4, &((code >> 8) as u32).to_le_bytes())?;
    env.mem.write(regs + 8 * hi as u64 + 4, &(((code >> 40) & 0xff) as u32).to_le_bytes())?;
    Ok(0)
}
