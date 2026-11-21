//! libSceAudioOut: ports take the title's sample buffers, mix them to
//! stereo and feed the host's output device, a WAVE dump, or neither.
//! `sceAudioOutOutput` blocks for about one buffer's time, as on a console,
//! because titles use it as their audio clock. Formats and the volume call
//! follow SharpEmu-mac's `AudioOutExports.cs` (GPL-2.0-or-later).

#[cfg(target_os = "macos")]
mod coreaudio;
mod input;
mod out2;
mod propagation;
mod wav;

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::pthread::lock;
use crate::{CallError, Env, Function, Outcome};

const ERROR_INVALID_PORT: u64 = 0x8026_0003;
const ERROR_INVALID_POINTER: u64 = 0x8026_0004;
const ERROR_PORT_FULL: u64 = 0x8026_0005;
const ERROR_INVALID_FORMAT: u64 = 0x8026_0007;
const ERROR_INVALID_SAMPLE_FREQ: u64 = 0x8026_0008;

/// The host device's rate. Orbis ports run at 48 kHz.
pub const RATE: u32 = 48_000;
/// `SCE_AUDIO_VOLUME_0DB`.
const UNITY_VOLUME: f32 = 32768.0;
/// Frames a port may queue for the device before its writer waits.
const QUEUED_FRAMES: usize = 2048;
/// Longest a writer waits for the device before dropping what it queued.
const DEVICE_STALL: Duration = Duration::from_millis(250);

pub use out2::FUNCTIONS as OUT2_FUNCTIONS;

pub use input::FUNCTIONS as INPUT_FUNCTIONS;

pub use propagation::STUBS as PROPAGATION_STUBS;

pub static FUNCTIONS: &[Function] = &[
    f("sceAudioOutInit", ok),
    f("sceAudioOutOpen", open),
    f("sceAudioOutOutput", output),
    f("sceAudioOutOutputs", outputs),
    f("sceAudioOutClose", close),
    f("sceAudioOutSetVolume", set_volume),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceAudioOut",
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

struct Port {
    channels: usize,
    float: bool,
    frames: usize,
    period: Duration,
    next: Instant,
    volume: [f32; 8],
    wav: Option<wav::Wav>,
}

impl Port {
    /// The port's buffer at `at` as stereo frames, with its volume applied.
    fn stereo(&self, env: &Env, at: u64) -> Result<Vec<f32>, CallError> {
        let bytes = if self.float { 4 } else { 2 };
        let mut raw = vec![0u8; self.frames * self.channels * bytes];
        env.mem.read(at, &mut raw)?;
        let samples: Vec<f32> = raw
            .chunks_exact(bytes)
            .enumerate()
            .map(|(i, b)| {
                let x = if self.float {
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                } else {
                    i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0
                };
                if x.is_finite() {
                    x * self.volume[i % self.channels]
                } else {
                    0.0
                }
            })
            .collect();
        Ok(to_stereo(&samples, self.channels))
    }
}

/// Folds mono or 8-channel (front, centre, LFE, sides, backs) frames down
/// to stereo; other layouts keep their first two channels.
fn to_stereo(samples: &[f32], channels: usize) -> Vec<f32> {
    const HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut out = Vec::with_capacity(samples.len() / channels * 2);
    for s in samples.chunks_exact(channels) {
        let (l, r) = match channels {
            1 => (s[0], s[0]),
            8 => (
                s[0] + HALF * (s[2] + s[4] + s[6]),
                s[1] + HALF * (s[2] + s[5] + s[7]),
            ),
            _ => (s[0], s[1]),
        };
        out.extend([l, r]);
    }
    out
}

/// What each open port has queued for the host device, as stereo frames.
#[derive(Default)]
pub struct Mixer {
    queues: Mutex<HashMap<u64, VecDeque<f32>>>,
    drained: Condvar,
}

impl Mixer {
    /// Fills `out` (stereo) with the sum of every port's queue; ports that
    /// run short contribute silence.
    pub fn pull(&self, out: &mut [f32]) {
        out.fill(0.0);
        let mut queues = lock(&self.queues);
        for q in queues.values_mut() {
            let n = q.len().min(out.len());
            for (o, x) in out.iter_mut().zip(q.drain(..n)) {
                *o += x;
            }
        }
        drop(queues);
        self.drained.notify_all();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Device {
    Off,
    Wanted,
    Running,
}

/// Audio ports and where their samples go.
pub struct Audio {
    next: Mutex<u64>,
    ports: Mutex<HashMap<u64, Port>>,
    mixer: Arc<Mixer>,
    device: Mutex<Device>,
    wav: Option<PathBuf>,
    out2: out2::Out2,
}

impl Default for Audio {
    fn default() -> Self {
        let device = match std::env::var("PS5SLOP_AUDIO").as_deref() {
            Ok("1") => Device::Wanted,
            _ => Device::Off,
        };
        Self {
            next: Mutex::new(0),
            ports: Default::default(),
            mixer: Default::default(),
            device: Mutex::new(device),
            wav: std::env::var_os("PS5SLOP_AUDIO_WAV").map(PathBuf::from),
            out2: Default::default(),
        }
    }
}

impl Audio {
    /// Asks for the host's output device, opened with the first port.
    /// `PS5SLOP_AUDIO=0` keeps it closed and `PS5SLOP_AUDIO=1` opens it
    /// either way.
    pub fn use_device(&self) {
        if std::env::var("PS5SLOP_AUDIO").as_deref() == Ok("0") {
            return;
        }
        let mut d = lock(&self.device);
        if *d == Device::Off {
            *d = Device::Wanted;
        }
    }

    fn start_device(&self) {
        let mut d = lock(&self.device);
        if *d != Device::Wanted {
            return;
        }
        #[cfg(target_os = "macos")]
        match coreaudio::start(self.mixer.clone()) {
            Ok(()) => *d = Device::Running,
            Err(status) => {
                eprintln!("ps5slop: no audio output (AudioQueue error {status})");
                *d = Device::Off;
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            *d = Device::Off;
        }
    }

    /// A new queue in the host mixer, numbered like AudioOut's ports.
    fn new_queue(&self) -> u64 {
        let id = {
            let mut next = lock(&self.next);
            *next += 1;
            *next
        };
        lock(&self.mixer.queues).insert(id, VecDeque::new());
        id
    }

    /// The WAVE dump for queue `id`, if dumps were asked for.
    fn open_wav(&self, id: u64) -> Option<wav::Wav> {
        let p = self.wav_path(id)?;
        match wav::Wav::create(&p, 2, RATE) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("ps5slop: cannot write {}: {e}", p.display());
                None
            }
        }
    }

    fn device_running(&self) -> bool {
        *lock(&self.device) == Device::Running
    }

    /// The WAVE file port number `id` is dumped to: the given path for
    /// the first port, `<stem>-<id>.<ext>` beside it for the rest.
    fn wav_path(&self, id: u64) -> Option<PathBuf> {
        let path = self.wav.as_ref()?;
        if id == 1 {
            return Some(path.clone());
        }
        let stem = path.file_stem()?.to_string_lossy();
        let ext = path
            .extension()
            .map_or(String::new(), |e| format!(".{}", e.to_string_lossy()));
        Some(path.with_file_name(format!("{stem}-{id}{ext}")))
    }
}

/// `sceAudioOutOpen(user, type, index, frames, freq, param)`; the low
/// byte of `param` is the format.
fn open(env: &mut Env) -> Result<Outcome, CallError> {
    let (frames, freq, format) = (
        env.arg(3) as u32,
        env.arg(4) as u32,
        env.arg(5) as u32 & 0xff,
    );
    if freq == 0 {
        return ret(ERROR_INVALID_SAMPLE_FREQ);
    }
    let channels = match format {
        0 | 3 => 1,
        1 | 4 => 2,
        2 | 5..=7 => 8,
        _ => return ret(ERROR_INVALID_FORMAT),
    };
    if frames == 0 {
        return ret(ERROR_INVALID_FORMAT);
    }
    let audio = &env.kernel.audio;
    audio.start_device();
    let id = {
        let mut next = lock(&audio.next);
        *next += 1;
        *next
    };
    let wav = audio
        .wav_path(id)
        .and_then(|p| match wav::Wav::create(&p, 2, freq) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("ps5slop: cannot write {}: {e}", p.display());
                None
            }
        });
    lock(&audio.ports).insert(
        id,
        Port {
            channels,
            float: matches!(format, 3 | 4 | 5 | 7),
            frames: frames as usize,
            period: Duration::from_secs_f64(frames as f64 / freq as f64),
            next: Instant::now(),
            volume: [1.0; 8],
            wav,
        },
    );
    lock(&audio.mixer.queues).insert(id, VecDeque::new());
    ret(id)
}

/// Queues one buffer of port `handle` (none for a null `at`) and returns
/// when the caller should wait until, or an error code.
fn submit(env: &Env, handle: u64, at: u64) -> Result<Result<Instant, u64>, CallError> {
    let audio = &env.kernel.audio;
    let mut ports = lock(&audio.ports);
    let Some(p) = ports.get_mut(&handle) else {
        return Ok(Err(ERROR_INVALID_PORT));
    };
    let wake = p.next;
    p.next = wake.max(Instant::now()) + p.period;
    if at == 0 {
        return Ok(Ok(wake));
    }
    let stereo = p.stereo(env, at)?;
    if let Some(w) = &mut p.wav
        && let Err(e) = w.append(&stereo)
    {
        eprintln!("ps5slop: audio dump stopped: {e}");
        p.wav = None;
    }
    drop(ports);
    if audio.device_running()
        && let Some(q) = lock(&audio.mixer.queues).get_mut(&handle)
    {
        q.extend(stereo);
    }
    Ok(Ok(wake))
}

/// Waits for port `handle`: until the device has taken enough of its
/// queue when one is running, otherwise until `wake`.
fn wait(env: &Env, handle: u64, wake: Instant) {
    let audio = &env.kernel.audio;
    if !audio.device_running() {
        if let Some(left) = wake.checked_duration_since(Instant::now()) {
            std::thread::sleep(left);
        }
        return;
    }
    let deadline = Instant::now() + DEVICE_STALL;
    let mut queues = lock(&audio.mixer.queues);
    loop {
        let Some(q) = queues.get_mut(&handle) else {
            return;
        };
        if q.len() / 2 <= QUEUED_FRAMES {
            return;
        }
        let now = Instant::now();
        if now >= deadline {
            let excess = q.len() - QUEUED_FRAMES * 2;
            q.drain(..excess);
            return;
        }
        queues = audio
            .mixer
            .drained
            .wait_timeout(queues, deadline - now)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}

/// `sceAudioOutOutput(port, data)`: returns the port's frame count.
fn output(env: &mut Env) -> Result<Outcome, CallError> {
    let handle = env.arg(0) as u32 as u64;
    let wake = match submit(env, handle, env.arg(1))? {
        Ok(w) => w,
        Err(code) => return ret(code),
    };
    wait(env, handle, wake);
    let frames = lock(&env.kernel.audio.ports)
        .get(&handle)
        .map_or(0, |p| p.frames);
    ret(frames as u64)
}

/// Ports one `sceAudioOutOutputs` call may name.
const MAX_OUTPUTS: u64 = 25;

/// `sceAudioOutOutputs(params, count)`: several ports at once, each
/// `{ i32 port, u64 data }`.
fn outputs(env: &mut Env) -> Result<Outcome, CallError> {
    let (list, count) = (env.arg(0), env.arg(1) as u32 as u64);
    if count == 0 || count > MAX_OUTPUTS {
        return ret(ERROR_PORT_FULL);
    }
    if list == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let mut waits = Vec::new();
    for i in 0..count {
        let handle = env.mem.read_u64(list + 16 * i)? as u32 as u64;
        let data = env.mem.read_u64(list + 16 * i + 8)?;
        match submit(env, handle, data)? {
            Ok(w) => waits.push((handle, w)),
            Err(code) => return ret(code),
        }
    }
    for &(handle, wake) in &waits {
        wait(env, handle, wake);
    }
    let frames = lock(&env.kernel.audio.ports)
        .get(&waits[0].0)
        .map_or(0, |p| p.frames);
    ret(frames as u64)
}

fn close(env: &mut Env) -> Result<Outcome, CallError> {
    let audio = &env.kernel.audio;
    let handle = env.arg(0) as u32 as u64;
    lock(&audio.mixer.queues).remove(&handle);
    match lock(&audio.ports).remove(&handle) {
        Some(_) => ret(0),
        None => ret(ERROR_INVALID_PORT),
    }
}

/// `sceAudioOutSetVolume(port, channel_mask, volumes)`: one volume per
/// channel bit, 32768 is unity.
fn set_volume(env: &mut Env) -> Result<Outcome, CallError> {
    let (handle, mask, at) = (env.arg(0) as u32 as u64, env.arg(1) as u32, env.arg(2));
    if at == 0 {
        return ret(ERROR_INVALID_POINTER);
    }
    let mut ports = lock(&env.kernel.audio.ports);
    let Some(p) = ports.get_mut(&handle) else {
        return ret(ERROR_INVALID_PORT);
    };
    for ch in 0..p.channels {
        if mask & 1 << ch != 0 {
            let mut v = [0u8; 4];
            env.mem.read(at + 4 * ch as u64, &mut v)?;
            p.volume[ch] = (i32::from_le_bytes(v) as f32 / UNITY_VOLUME).clamp(0.0, 1.0);
        }
    }
    ret(0)
}

impl crate::Kernel {
    /// Plays audio on the host's output device (see [`Audio::use_device`]).
    pub fn use_audio_device(&self) {
        self.audio.use_device();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_layouts_keep_their_first_two_channels() {
        assert_eq!(to_stereo(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 6), [0.1, 0.2]);
    }

    #[test]
    fn surround_folds_down_to_stereo() {
        assert_eq!(to_stereo(&[0.5], 1), [0.5, 0.5]);
        let s = to_stereo(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8);
        assert_eq!(s, [1.0, 0.0]);
    }

    #[test]
    fn the_mixer_sums_ports_and_pads_with_silence() {
        let m = Mixer::default();
        lock(&m.queues).insert(1, [0.25, 0.5].into());
        lock(&m.queues).insert(2, [0.25, 0.25, 1.0, 1.0].into());
        let mut out = [9.0; 6];
        m.pull(&mut out);
        assert_eq!(out, [0.5, 0.75, 1.0, 1.0, 0.0, 0.0]);
    }
}
