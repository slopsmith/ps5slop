//! libSceAjm batches: jobs are appended to the title's batch buffer as the
//! library would, and batches finish when they start. MP3 (codec 0) streams
//! are split into whole frames and decoded by FFmpeg ([`mp3`]); other codecs
//! consume all input and write silence. Layouts and sizes follow
//! SharpEmu-mac's `AjmExports.cs` (GPL-2.0-or-later).

mod mp3;
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
const MAX_SPLIT_BUFFERS: u64 = 64;

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
    decoder: mp3::Decoder,
}

impl Instance {
    /// Instance flags bits 7 to 9 pick S16, S32 or float (SharpEmu-mac's reading).
    fn format(&self) -> mp3::Format {
        match (self.flags >> 7) & 7 {
            1 => mp3::Format::S32,
            2 => mp3::Format::F32,
            _ => mp3::Format::S16,
        }
    }
}

struct Decoded {
    consumed: usize,
    pcm: Vec<u8>,
    frames: u32,
}

/// Input is consumed whole unless the output fills first; an incomplete last
/// frame waits for the next job.
fn decode_mp3(inst: &mut Instance, input: &[u8], out_size: usize) -> Decoded {
    let old = inst.pending.len();
    let mut data = std::mem::take(&mut inst.pending);
    data.extend_from_slice(input);
    let mut at = 0;
    let mut pcm = Vec::new();
    let mut frames = 0;
    let format = inst.format();
    let mut full = false;
    while at + 4 <= data.len() {
        let Some(h) = mpeg::parse(&data[at..]) else {
            at += 1;
            continue;
        };
        if at + h.len > data.len() {
            break;
        }
        if pcm.len() + h.samples * h.channels * format.bytes() > out_size {
            full = true;
            break;
        }
        let frame = &data[at..at + h.len];
        inst.decoder
            .frame(frame, h.samples, h.channels, format, &mut pcm);
        at += h.len;
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
        pcm,
        frames,
    }
}

/// MP3 is decoded; other codecs are placeholders.
pub static STUBS: &[Function] = &[
    f("sceAjmInstanceCreate", instance_create),
    f("sceAjmBatchInitialize", batch_initialize),
    f("sceAjmBatchJobInitialize", job_initialize),
    f("sceAjmBatchJobDecode", job_decode),
    f("sceAjmBatchJobDecodeSplit", job_decode_split),
    f("sceAjmBatchJobSetGaplessDecode", job_set_gapless),
    f("sceAjmBatchJobClearContext", job_clear_context),
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

/// `sceAjmBatchJobClearContext(info, instance, result)`: the instance starts
/// a new stream with its flags kept.
fn job_clear_context(env: &mut Env) -> Result<Outcome, CallError> {
    if !append_job(env, env.arg(0), JOB_CONTROL_SIZE)? {
        return ret(ERROR_JOB_CREATION);
    }
    let instance = env.arg(1) as u32;
    if let Some(inst) = INSTANCES.lock().unwrap().get_mut(&instance) {
        *inst = Instance {
            flags: inst.flags,
            ..Default::default()
        };
    }
    write_status(env, env.arg(2), status(env.arg(1)))?;
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
    let (info, instance) = (env.arg(0), env.arg(1) as u32);
    let input = [(env.arg(2), env.arg(3))];
    let output = [(env.arg(4), env.arg(5))];
    let result = env.arg_n(6)?;
    decode_job(env, info, instance, &input, &output, result, JOB_RUN_SIZE)
}

/// `sceAjmBatchJobDecodeSplit(info, instance, inputs, input count, outputs,
/// output count, result)`; buffers are `{ address, size }` pairs.
fn job_decode_split(env: &mut Env) -> Result<Outcome, CallError> {
    let (info, instance) = (env.arg(0), env.arg(1) as u32);
    let (Some(input), Some(output)) = (
        buffer_list(env, env.arg(2), env.arg(3))?,
        buffer_list(env, env.arg(4), env.arg(5))?,
    ) else {
        return ret(ERROR_INVALID_PARAMETER);
    };
    let result = env.arg_n(6)?;
    decode_job(
        env,
        info,
        instance,
        &input,
        &output,
        result,
        JOB_RUN_SPLIT_SIZE,
    )
}

fn buffer_list(env: &Env, at: u64, count: u64) -> Result<Option<Vec<(u64, u64)>>, CallError> {
    if count > MAX_SPLIT_BUFFERS || (at == 0 && count != 0) {
        return Ok(None);
    }
    let mut list = Vec::with_capacity(count as usize);
    for k in 0..count {
        let entry = at + 16 * k;
        list.push((env.mem.read_u64(entry)?, env.mem.read_u64(entry + 8)?));
    }
    Ok(Some(list))
}

fn decode_job(
    env: &Env,
    info: u64,
    instance: u32,
    input: &[(u64, u64)],
    output: &[(u64, u64)],
    result: u64,
    job_size: u64,
) -> Result<Outcome, CallError> {
    if !append_job(env, info, job_size)? {
        return ret(ERROR_JOB_CREATION);
    }
    let in_size: u64 = input.iter().map(|b| b.1).sum();
    let out_size: u64 = output.iter().map(|b| b.1).sum();
    for &(at, size) in output {
        zero(env, at, size)?;
    }
    let mut d = Decoded {
        consumed: in_size.min(i32::MAX as u64) as usize,
        pcm: Vec::new(),
        frames: u32::from(in_size != 0 || out_size != 0),
    };
    let mut decoded = 0;
    let is_mp3 = instance >> INSTANCE_SLOT_BITS == CODEC_MP3 && instance != 0;
    if is_mp3 && input.iter().all(|b| b.0 != 0) && in_size <= MAX_CLEAR && out_size <= MAX_CLEAR {
        let mut bytes = Vec::with_capacity(in_size as usize);
        for &(at, size) in input {
            let mut part = vec![0u8; size as usize];
            env.mem.read(at, &mut part)?;
            bytes.extend_from_slice(&part);
        }
        let mut instances = INSTANCES.lock().unwrap();
        let inst = instances.entry(instance).or_default();
        d = decode_mp3(inst, &bytes, out_size as usize);
        decoded = inst.decoded;
        drop(instances);
        let mut rest = d.pcm.as_slice();
        for &(at, size) in output {
            let (part, tail) = rest.split_at(rest.len().min(size as usize));
            env.mem.write(at, part)?;
            rest = tail;
        }
    }
    if result != 0 {
        let mut b = [0u8; DECODE_RESULT_BYTES];
        b[0..4].copy_from_slice(&status(u64::from(instance)).to_le_bytes());
        b[8..12].copy_from_slice(&(d.consumed as i32).to_le_bytes());
        b[12..16].copy_from_slice(&(d.pcm.len() as i32).to_le_bytes());
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
        assert_eq!((a.consumed, a.pcm.len(), a.frames), (700, 2 * 2304, 2));
        let b = decode_mp3(&mut inst, &s[700..1400], 0x2400);
        assert_eq!((b.consumed, b.pcm.len(), b.frames), (700, 3 * 2304, 3));
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
        assert_eq!((a.consumed, a.pcm.len(), a.frames), (480, 4608, 2));
        assert!(inst.pending.is_empty());
    }

    #[test]
    fn decodes_a_sine_in_jobs() {
        let mp3 = include_bytes!("ajm/testdata/sine-1khz.mp3");
        let mut inst = Instance::default();
        let mut pcm = Vec::new();
        for job in mp3.chunks(960) {
            let d = decode_mp3(&mut inst, job, 0x2400);
            assert_eq!(d.consumed, job.len());
            pcm.extend_from_slice(&d.pcm);
        }
        assert_eq!(inst.decoded, 10 * 1152);
        let s: Vec<i16> = pcm
            .chunks(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(s.len(), 10 * 1152);
        let tail = &s[2 * 1152..8 * 1152];
        let rms =
            (tail.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(rms > 1000.0, "decoded audio is near silent (rms {rms})");
        let crossings = tail.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count();
        let hz = crossings as f64 / 2.0 / (tail.len() as f64 / 48000.0);
        assert!(
            (hz - 1000.0).abs() < 20.0,
            "decoded tone has the wrong pitch ({hz} Hz)"
        );
    }

    #[test]
    fn float_output() {
        let mp3 = include_bytes!("ajm/testdata/sine-1khz.mp3");
        let mut inst = Instance {
            flags: 2 << 7,
            ..Default::default()
        };
        let d = decode_mp3(&mut inst, mp3, 1 << 20);
        assert_eq!(d.pcm.len(), 10 * 1152 * 4);
        let peak = d
            .pcm
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()).abs())
            .fold(0.0, f32::max);
        assert!(
            peak > 0.05 && peak <= 1.0,
            "float output is silent or out of range (peak {peak})"
        );
    }

    const AREA: u64 = 0x40_0000;
    const BATCH: u64 = AREA;
    const INFO: u64 = AREA + 0x1000;
    const LISTS: u64 = AREA + 0x1100;
    const RESULT: u64 = AREA + 0x1200;
    const ID: u64 = AREA + 0x1300;
    const STACK: u64 = AREA + 0x2000;
    const INPUT: u64 = AREA + 0x4000;
    const OUTPUT: u64 = AREA + 0x10_0000;

    /// Runs `test` with an `Env` over 2 MiB of guest memory at [`AREA`].
    fn with_env(test: impl FnOnce(&mut Env, &memory::Memory)) {
        use crate::{Cpu, Hle, Kernel, Thread, TlsTemplate, Vfs};
        use memory::{Memory, PAGE_SIZE, Prot};
        let mem = Memory::new(1 << 40).unwrap();
        mem.map(AREA, 0x20_0000, Prot::RW, "test").unwrap();
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
        let mut cpu = Cpu::default();
        let mut env = Env {
            cpu: &mut cpu,
            mem: &mem,
            kernel: &kernel,
            thread: Thread {
                tcb: AREA + PAGE_SIZE,
            },
        };
        test(&mut env, &mem);
    }

    fn call(env: &mut Env, f: crate::HostFn, args: &[u64]) {
        for (k, &a) in args.iter().enumerate() {
            env.cpu.regs[[7, 6, 2, 1, 8, 9][k]] = a;
        }
        assert!(matches!(f(env), Ok(Outcome::Return(0))));
    }

    #[test]
    fn clearing_the_context_drops_a_partial_frame() {
        with_env(|env, mem| {
            call(env, instance_create, &[0, u64::from(CODEC_MP3), 7, ID]);
            call(env, batch_initialize, &[BATCH, 0x1000, INFO]);
            let id = mem.read_u64(ID).unwrap() & 0xFFFF_FFFF;
            INSTANCES
                .lock()
                .unwrap()
                .get_mut(&(id as u32))
                .unwrap()
                .pending = vec![1, 2, 3];
            call(env, job_clear_context, &[INFO, id, RESULT]);
            let instances = INSTANCES.lock().unwrap();
            let inst = &instances[&(id as u32)];
            assert!(inst.pending.is_empty());
            assert_eq!(inst.flags, 7);
            assert_eq!(mem.read_u64(RESULT).unwrap(), 0);
        });
    }

    #[test]
    fn a_split_job_decodes_the_joined_input_across_its_outputs() {
        let mp3 = include_bytes!("ajm/testdata/sine-1khz.mp3");
        let mut whole = Instance::default();
        let expected = decode_mp3(&mut whole, mp3, 1 << 20).pcm;
        with_env(|env, mem| {
            mem.write(INPUT, mp3).unwrap();
            call(env, instance_create, &[0, u64::from(CODEC_MP3), 0, ID]);
            call(env, batch_initialize, &[BATCH, 0x1000, INFO]);
            let id = mem.read_u64(ID).unwrap() & 0xFFFF_FFFF;
            let half = mp3.len() as u64 / 2;
            let out_half = expected.len() as u64 / 2 + 6;
            for (k, (at, len)) in [
                (INPUT, half),
                (INPUT + half, mp3.len() as u64 - half),
                (OUTPUT, out_half),
                (OUTPUT + 0x8_0000, 1 << 18),
            ]
            .into_iter()
            .enumerate()
            {
                mem.write_u64(LISTS + 16 * k as u64, at).unwrap();
                mem.write_u64(LISTS + 16 * k as u64 + 8, len).unwrap();
            }
            mem.write_u64(STACK + 8, RESULT).unwrap();
            env.cpu.regs[4] = STACK;
            call(env, job_decode_split, &[INFO, id, LISTS, 2, LISTS + 32, 2]);

            let mut out = vec![0u8; expected.len()];
            mem.read(OUTPUT, &mut out[..out_half as usize]).unwrap();
            mem.read(OUTPUT + 0x8_0000, &mut out[out_half as usize..])
                .unwrap();
            assert!(out == expected, "split output differs from a whole decode");
            let written = mem.read_u64(RESULT + 8).unwrap() >> 32;
            assert_eq!(written as usize, expected.len());
        });
    }
}
