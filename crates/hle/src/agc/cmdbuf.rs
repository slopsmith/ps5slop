//! AGC command buffers: the title owns a descriptor with an upward cursor
//! (where the next packet goes), a downward cursor (the end of free space),
//! a callback for when it is full, and a reserved tail. Builders append
//! PM4 packets at the upward cursor. Ported from SharpEmu's `AgcExports.cs`.

use std::sync::Mutex;

use crate::{CallError, Env};

/// Offsets in the descriptor.
const CURSOR_UP: u64 = 0x10;
const CURSOR_DOWN: u64 = 0x18;
const CALLBACK: u64 = 0x20;
const USER_DATA: u64 = 0x28;
const RESERVED_DWORDS: u64 = 0x30;

/// Serialises appends to one buffer from several guest threads.
static LOCKS: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];

fn stripe(buffer: u64) -> &'static Mutex<()> {
    &LOCKS[(buffer.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize]
}

/// A PM4 type-3 packet header: `len` dwords in all, opcode `op`, and the
/// low bits of `reg` in the shader-type/index field.
pub(super) const fn pm4(len: u32, op: u32, reg: u32) -> u32 {
    0xC000_0000 | (((len - 2) & 0x3FFF) << 16) | ((op & 0xFF) << 8) | ((reg & 0x3F) << 2)
}

pub(super) const IT_NOP: u32 = 0x10;

fn read_u32(env: &Env, at: u64) -> Result<u32, CallError> {
    let mut b = [0u8; 4];
    env.mem.read(at, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Free dwords: the cursor distance truncated to 32 bits, less the reserved
/// tail. Titles rely on the truncation (a null down cursor still has room).
fn free_dwords(up: u64, down: u64, reserved: u32) -> u32 {
    let available = ((down.wrapping_sub(up) as i64) / 4) as u32;
    available.max(reserved) - reserved
}

/// Reserves `dwords` at buffer `buffer`'s upward cursor and advances it,
/// calling the title's buffer-full callback when there is no room. `None`
/// when there is still no room afterwards.
pub(super) fn reserve(env: &mut Env, buffer: u64, dwords: u32) -> Result<Option<u64>, CallError> {
    if dwords == 0 {
        return Ok(None);
    }
    let lock = stripe(buffer);
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let state = |env: &Env| -> Result<(u64, u64, u32), CallError> {
        Ok((
            env.mem.read_u64(buffer + CURSOR_UP)?,
            env.mem.read_u64(buffer + CURSOR_DOWN)?,
            read_u32(env, buffer + RESERVED_DWORDS)?,
        ))
    };
    let (mut up, mut down, mut reserved) = state(env)?;
    if dwords > free_dwords(up, down, reserved) {
        let callback = env.mem.read_u64(buffer + CALLBACK)?;
        let user = env.mem.read_u64(buffer + USER_DATA)?;
        if callback == 0 {
            return Ok(None);
        }
        drop(guard);
        env.call_guest(callback, &[buffer, (dwords + reserved) as u64, user])?;
        guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        (up, down, reserved) = state(env)?;
        if dwords > free_dwords(up, down, reserved) {
            return Ok(None);
        }
    }
    env.mem
        .write_u64(buffer + CURSOR_UP, up + 4 * dwords as u64)?;
    drop(guard);
    Ok(Some(up))
}

/// Appends `packet` to `buffer` and returns where it went (0 if full).
pub(super) fn emit(env: &mut Env, buffer: u64, packet: &[u32]) -> Result<u64, CallError> {
    let Some(at) = reserve(env, buffer, packet.len() as u32)? else {
        return Ok(0);
    };
    let bytes: Vec<u8> = packet.iter().flat_map(|d| d.to_le_bytes()).collect();
    env.mem.write(at, &bytes)?;
    Ok(at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_space_truncates_and_keeps_the_reserved_tail() {
        assert_eq!(free_dwords(0x1000, 0x1100, 0), 0x40);
        assert_eq!(free_dwords(0x1000, 0x1100, 0x10), 0x30);
        assert_eq!(free_dwords(0x1000, 0x1100, 0x80), 0);
        assert!(free_dwords(0x1000, 0, 0) > 0);
    }

    #[test]
    fn headers_encode_length_opcode_and_register() {
        assert_eq!(pm4(2, IT_NOP, 0), 0xC000_1000);
        assert_eq!(pm4(3, 0x76, 0), 0xC001_7600);
    }
}
