//! libSceAjm batches: jobs are appended to the title's batch buffer as the
//! library would, and batches finish when they start. MP3 (codec 0) streams
//! are split into whole frames and each frame yields its samples as silence;
//! other codecs consume all input and write silence. Layouts and sizes follow
//! SharpEmu-mac's `AjmExports.cs` (GPL-2.0-or-later).

mod mpeg;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::{CallError, Env, Function, Outcome};

const ERROR_INVALID_PARAMETER: u64 = 0x8093_0005;
const ERROR_JOB_CREATION: u64 = 0x8093_0012;
/// Status written to a job result for an unknown instance.
const RESULT_INVALID_PARAMETER: i32 = 4;

/// `{ buffer, offset, size, last_good_job, last_good_job_ra }`, all u64.
const INFO_OFFSET: u64 = 8;
const INFO_SIZE: u64 = 16;
const INFO_LAST_JOB: u64 = 24;
const INFO_LAST_JOB_RA: u64 = 32;
const INFO_BYTES: usize = 40;
const JOB_CONTROL_SIZE: u64 = 48;
const JOB_RUN_SIZE: u64 = 64;
const JOB_RUN_SPLIT_SIZE: u64 = 32;
/// Result, stream and frame-count sidebands of a decode.
const DECODE_RESULT_BYTES: usize = 32;
/// The batch error record `BatchStart` and `BatchWait` clear.
const BATCH_ERROR_BYTES: usize = 0x20;
const MAX_CLEAR: u64 = 64 << 20;

static NEXT_BATCH: AtomicU32 = AtomicU32::new(1);
static NEXT_INSTANCE: AtomicU32 = AtomicU32::new(1);
static INSTANCES: LazyLock<Mutex<HashMap<u32, Instance>>> = LazyLock::new(Default::default);

const CODEC_MP3: u32 = 0;
/// The codec sits above the slot in an instance id, as in SharpEmu-mac.
const INSTANCE_SLOT_BITS: u32 = 14;
/// An MP3 frame is at most 2881 bytes; more pending input is not a stream.
const MAX_PENDING: usize = 4096;

#[derive(Default)]
struct Instance {
    flags: u64,
    /// Input bytes of a frame not yet complete.
    pending: Vec<u8>,
    decoded: u64,
}

impl Instance {
    /// Bytes per output sample: instance flags bits 7 to 9 pick S16, S32 or
    /// float (SharpEmu-mac's reading).
    fn sample_bytes(&self) -> usize {
        match (self.flags >> 7) & 7 {
            1 | 2 => 4,
            _ => 2,
        }
    }
}

struct Decoded {
    consumed: usize,
    written: usize,
    frames: u32,
}

/// Splits `input` (after what an earlier job left pending) into whole frames
/// and writes each frame's samples as silence while `out_size` has room.
/// Input is consumed whole unless the output fills first; an incomplete last
/// frame waits for the next job.
fn decode_mp3(inst: &mut Instance, input: &[u8], out_size: usize) -> Decoded {
    let old = inst.pending.len();
    let mut data = std::mem::take(&mut inst.pending);
    data.extend_from_slice(input);
    let mut at = 0;
    let mut written = 0;
    let mut frames = 0;
    let mut full = false;
    while at + 4 <= data.len() {
        let Some(h) = mpeg::parse(&data[at..]) else {
            at += 1;
            continue;
        };
        if at + h.len > data.len() {
            break;
        }
        let bytes = h.samples * h.channels * inst.sample_bytes();
        if written + bytes > out_size {
            full = true;
            break;
        }
        at += h.len;
        written += bytes;
        frames += 1;
        inst.decoded += h.samples as u64;
    }
    let consumed = if full {
        inst.pending = data[at.min(old)..old].to_vec();
        input.len() - (data.len() - at.max(old))
    } else {
        inst.pending = data[at..].to_vec();
        if inst.pending.len() > MAX_PENDING {
            inst.pending.clear();
        }
        input.len()
    };
    Decoded {
        consumed,
        written,
        frames,
    }
}

/// Placeholders: no audio is decoded.
pub static STUBS: &[Function] = &[
    f("sceAjmInstanceCreate", instance_create),
    f("sceAjmBatchInitialize", batch_initialize),
    f("sceAjmBatchJobInitialize", job_initialize),
    f("sceAjmBatchJobDecode", job_decode),
    f("sceAjmBatchJobSetGaplessDecode", job_set_gapless),
    f("sceAjmBatchJobGetCodecInfo", job_codec_info),
    f("sceAjmBatchStart", batch_start),
    f("sceAjmBatchWait", batch_wait),
    f("sceAjmBatchCancel", ok),
    f("sceAjmBatchErrorDump", ok),
    f("sceAjmInstanceDestroy", instance_destroy),
    f("sceAjmModuleUnregister", ok),
    f("sceAjmFinalize", ok),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAjm",
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

fn zero(env: &Env, at: u64, len: u64) -> Result<(), CallError> {
    if at != 0 && len != 0 && len <= MAX_CLEAR {
        env.mem.write(at, &vec![0u8; len as usize])?;
    }
    Ok(())
}

/// Takes `size` bytes for a job from the batch at `info`, as the library's
/// job builders do; false when the batch buffer is full.
fn append_job(env: &Env, info: u64, size: u64) -> Result<bool, CallError> {
    let buffer = env.mem.read_u64(info)?;
    let offset = env.mem.read_u64(info + INFO_OFFSET)?;
    let total = env.mem.read_u64(info + INFO_SIZE)?;
    if buffer == 0 || offset > total || total - offset < size {
        return Ok(false);
    }
    zero(env, buffer + offset, size)?;
    env.mem.write_u64(info + INFO_LAST_JOB, buffer + offset)?;
    env.mem.write_u64(info + INFO_LAST_JOB_RA, 0)?;
    env.mem.write_u64(info + INFO_OFFSET, offset + size)?;
    Ok(true)
}

fn write_status(env: &Env, at: u64, status: i32) -> Result<(), CallError> {
    if at != 0 {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&status.to_le_bytes());
        env.mem.write(at, &b)?;
    }
    Ok(())
}

/// `sceAjmInstanceCreate(context, codec, flags, *instance)`.
fn instance_create(env: &mut Env) -> Result<Outcome, CallError> {
    let (codec, flags, out) = (env.arg(1) as u32, env.arg(2), env.arg(3));
    if out == 0 || codec >= 1 << (32 - INSTANCE_SLOT_BITS) {
        return ret(ERROR_INVALID_PARAMETER);
    }
    let slot = NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed) & ((1 << INSTANCE_SLOT_BITS) - 1);
    let id = codec << INSTANCE_SLOT_BITS | slot;
    env.mem.write(out, &id.to_le_bytes())?;
    INSTANCES.lock().unwrap().insert(
        id,
        Instance {
            flags,
            ..Default::default()
        },
    );
    ret(0)
}

/// `sceAjmInstanceDestroy(context, instance)`.
fn instance_destroy(env: &mut Env) -> Result<Outcome, CallError> {
    INSTANCES.lock().unwrap().remove(&(env.arg(1) as u32));
    ret(0)
}

/// `sceAjmBatchInitialize(buffer, size, info)`.
fn batch_initialize(env: &mut Env) -> Result<Outcome, CallError> {
    let (buffer, size, info) = (env.arg(0), env.arg(1), env.arg(2));
    if buffer == 0 || info == 0 {
        return ret(ERROR_INVALID_PARAMETER);
    }
    let mut b = [0u8; INFO_BYTES];
    b[0..8].copy_from_slice(&buffer.to_le_bytes());
    b[16..24].copy_from_slice(&size.to_le_bytes());
    env.mem.write(info, &b)?;
    ret(0)
}

/// `sceAjmBatchJobInitialize(info, instance, params, size, result)`.
fn job_initialize(env: &mut Env) -> Result<Outcome, CallError> {
    if !append_job(env, env.arg(0), JOB_CONTROL_SIZE)? {
        return ret(ERROR_JOB_CREATION);
    }
    write_status(env, env.arg(4), status(env.arg(1)))?;
    ret(0)
}

/// `sceAjmBatchJobSetGaplessDecode(info, instance, gapless, reset, result)`.
fn job_set_gapless(env: &mut Env) -> Result<Outcome, CallError> {
    if !append_job(env, env.arg(0), JOB_CONTROL_SIZE)? {
        return ret(ERROR_JOB_CREATION);
    }
    write_status(env, env.arg(4), status(env.arg(1)))?;
    ret(0)
}

/// Instance ids are never 0; anything else is taken as valid.
fn status(instance: u64) -> i32 {
    if instance as u32 == 0 {
        RESULT_INVALID_PARAMETER
    } else {
        0
    }
}

/// `sceAjmBatchJobDecode(info, instance, input, in_size, output, out_size,
/// result)`. The result is `{ status, internal, consumed, written,
/// total decoded samples }`.
fn job_decode(env: &mut Env) -> Result<Outcome, CallError> {
    let (info, instance, input, in_size, output, out_size) = (
        env.arg(0),
        env.arg(1) as u32,
        env.arg(2),
        env.arg(3),
        env.arg(4),
        env.arg(5),
    );
    let result = env.arg_n(6)?;
    if !append_job(env, info, JOB_RUN_SIZE)? {
        return ret(ERROR_JOB_CREATION);
    }
    zero(env, output, out_size)?;
    let mut d = Decoded {
        consumed: in_size.min(i32::MAX as u64) as usize,
        written: 0,
        frames: u32::from(in_size != 0 || out_size != 0),
    };
    let mut decoded = 0;
    let is_mp3 = instance >> INSTANCE_SLOT_BITS == CODEC_MP3 && instance != 0;
    if is_mp3 && input != 0 && in_size <= MAX_CLEAR && out_size <= MAX_CLEAR {
        let mut bytes = vec![0u8; in_size as usize];
        env.mem.read(input, &mut bytes)?;
        let mut instances = INSTANCES.lock().unwrap();
        let inst = instances.entry(instance).or_default();
        d = decode_mp3(inst, &bytes, out_size as usize);
        decoded = inst.decoded;
    }
    if result != 0 {
        let mut b = [0u8; DECODE_RESULT_BYTES];
        b[0..4].copy_from_slice(&status(u64::from(instance)).to_le_bytes());
        b[8..12].copy_from_slice(&(d.consumed as i32).to_le_bytes());
        b[12..16].copy_from_slice(&(d.written as i32).to_le_bytes());
        b[16..24].copy_from_slice(&decoded.to_le_bytes());
        b[24..28].copy_from_slice(&d.frames.to_le_bytes());
        env.mem.write(result, &b)?;
    }
    ret(0)
}

/// `sceAjmBatchJobGetCodecInfo(info, instance, sideband, size)`: a zero
/// status and zero codec information.
fn job_codec_info(env: &mut Env) -> Result<Outcome, CallError> {
    let (info, instance, sideband, size) = (env.arg(0), env.arg(1), env.arg(2), env.arg(3));
    if !append_job(env, info, JOB_RUN_SPLIT_SIZE)? {
        return ret(ERROR_JOB_CREATION);
    }
    zero(env, sideband, size)?;
    if size >= 8 {
        write_status(env, sideband, status(instance))?;
    }
    ret(0)
}

/// `sceAjmBatchStart(context, info, priority, error, *batch)`.
fn batch_start(env: &mut Env) -> Result<Outcome, CallError> {
    let (info, error, out) = (env.arg(1), env.arg(3), env.arg(4));
    if info == 0 || out == 0 {
        return ret(ERROR_INVALID_PARAMETER);
    }
    zero(env, error, BATCH_ERROR_BYTES as u64)?;
    let id = NEXT_BATCH.fetch_add(1, Ordering::Relaxed);
    env.mem.write(out, &id.to_le_bytes())?;
    ret(0)
}

/// `sceAjmBatchWait(context, batch, timeout, error)`.
fn batch_wait(env: &mut Env) -> Result<Outcome, CallError> {
    zero(env, env.arg(3), BATCH_ERROR_BYTES as u64)?;
    ret(0)
}

pub(crate) fn save(e: &mut savestate::Encoder) {
    e.put(&NEXT_BATCH.load(Ordering::Relaxed));
}

pub(crate) fn load(d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
    NEXT_BATCH.store(d.get()?, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: [u8; 4] = [0xFF, 0xFA, 0x64, 0xC0];

    fn stream(frames: usize) -> Vec<u8> {
        let mut v = vec![0u8; 240 * frames];
        for f in 0..frames {
            v[f * 240..f * 240 + 4].copy_from_slice(&FRAME);
        }
        v
    }

    #[test]
    fn chunks_carry_partial_frames() {
        let s = stream(10);
        let mut inst = Instance::default();
        let a = decode_mp3(&mut inst, &s[..700], 0x2400);
        assert_eq!((a.consumed, a.written, a.frames), (700, 2 * 2304, 2));
        let b = decode_mp3(&mut inst, &s[700..1400], 0x2400);
        assert_eq!((b.consumed, b.written, b.frames), (700, 3 * 2304, 3));
        let c = decode_mp3(&mut inst, &s[1400..], 0x2400);
        assert_eq!((c.consumed, c.frames), (760, 4));
        assert_eq!(inst.decoded, 9 * 1152);
        assert!(inst.pending.is_empty());
    }

    #[test]
    fn full_output_leaves_input() {
        let s = stream(4);
        let mut inst = Instance::default();
        let a = decode_mp3(&mut inst, &s, 2 * 2304);
        assert_eq!((a.consumed, a.written, a.frames), (480, 4608, 2));
        assert!(inst.pending.is_empty());
    }
}
