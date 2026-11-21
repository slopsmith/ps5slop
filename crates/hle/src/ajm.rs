//! libSceAjm batches without decoders: jobs are appended to the title's
//! batch buffer as the library would, a decode consumes all its input and
//! writes silence, and batches finish when they start. Layouts and sizes
//! follow SharpEmu-mac's `AjmExports.cs` (GPL-2.0-or-later), which does the
//! same for codecs it cannot decode.

use std::sync::atomic::{AtomicU32, Ordering};

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

/// Placeholders: no audio is decoded.
pub static STUBS: &[Function] = &[
    f("sceAjmBatchInitialize", batch_initialize),
    f("sceAjmBatchJobInitialize", job_initialize),
    f("sceAjmBatchJobDecode", job_decode),
    f("sceAjmBatchJobSetGaplessDecode", job_set_gapless),
    f("sceAjmBatchJobGetCodecInfo", job_codec_info),
    f("sceAjmBatchStart", batch_start),
    f("sceAjmBatchWait", batch_wait),
    f("sceAjmBatchCancel", ok),
    f("sceAjmBatchErrorDump", ok),
    f("sceAjmInstanceDestroy", ok),
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
/// result)`: all input consumed, silence written, one frame.
fn job_decode(env: &mut Env) -> Result<Outcome, CallError> {
    let (info, instance, in_size, output, out_size) =
        (env.arg(0), env.arg(1), env.arg(3), env.arg(4), env.arg(5));
    let result = env.mem.read_u64(env.cpu.regs[x86::cpu::reg::RSP] + 8)?;
    if !append_job(env, info, JOB_RUN_SIZE)? {
        return ret(ERROR_JOB_CREATION);
    }
    zero(env, output, out_size)?;
    if result != 0 {
        let mut b = [0u8; DECODE_RESULT_BYTES];
        b[0..4].copy_from_slice(&status(instance).to_le_bytes());
        b[8..12].copy_from_slice(&(in_size.min(i32::MAX as u64) as i32).to_le_bytes());
        let frames = u32::from(in_size != 0 || out_size != 0);
        b[24..28].copy_from_slice(&frames.to_le_bytes());
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
