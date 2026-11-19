//! libSceVideoOut: one display port with registered frame buffers, flips
//! and a 60 Hz vblank. The host shows the front buffer through
//! [`Kernel::frame`]. Prototypes and the frame layout follow how
//! ScreenTester5 drives VideoOut (see `docs/gpu.md`); unverified on a PS5.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::pthread::{POLL, lock};
use crate::{CallError, Env, Function, Kernel, Outcome};

const ERROR_INVALID_VALUE: u64 = 0x8029_0001;
const ERROR_INVALID_HANDLE: u64 = 0x8029_000B;
const ERROR_INVALID_INDEX: u64 = 0x8029_0006;
const ERROR_INVALID_EVENT_QUEUE: u64 = 0x8029_000C;
const HANDLE: u64 = 1;
const VBLANK: Duration = Duration::from_nanos(1_000_000_000 / 60);
const MAX_BUFFERS: usize = 16;
/// `struct { void *data; void *metadata; void *reserved[2]; }`.
const BUFFER_ENTRY_SIZE: u64 = 32;
/// Flip events: `EVFILT_DISPLAY` from the OpenOrbis headers, ident 0.
pub const EVENT_FILTER: i16 = -13;
const FLIP_EVENT: u64 = 0;

/// The attribute block `sceVideoOutSetBufferAttribute2` fills is opaque to
/// the guest, so it holds our own layout: format, tiling, width, height.
const ATTRIBUTE_TAG: u32 = 0x7073_3576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub width: u32,
    pub height: u32,
    /// Tiling mode 0, which ScreenTester5 fills in 128x128-pixel tiles.
    pub tiled: bool,
}

#[derive(Default)]
struct Port {
    open: bool,
    buffers: Vec<Option<(u64, Layout)>>,
    front: Option<usize>,
    flips: u64,
    /// The last flip's argument and when it happened.
    last_arg: u64,
    last_flip: Option<Instant>,
}

pub struct VideoOut {
    port: Mutex<Port>,
    start: Instant,
    listener: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    on_open: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

impl Default for VideoOut {
    fn default() -> Self {
        Self {
            port: Mutex::default(),
            start: Instant::now(),
            listener: Mutex::new(None),
            on_open: Mutex::new(None),
        }
    }
}

/// A frame for the host to show: RGBA8 pixels, row by row.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Counts flips, so the host can tell new frames from old ones.
    pub flips: u64,
    /// The GPU's texture for the buffer, when the GPU rendered it; `rgba`
    /// is empty then.
    #[cfg(target_os = "macos")]
    pub texture: Option<gpu::metal::SharedTexture>,
}

impl Kernel {
    /// Calls `listener` (from the guest thread) when the title opens its
    /// video output.
    pub fn set_open_listener(&self, listener: Box<dyn Fn() + Send + Sync>) {
        *lock(&self.videoout.on_open) = Some(listener);
    }

    /// Calls `listener` (from the flipping guest thread) after every flip.
    pub fn set_flip_listener(&self, listener: Box<dyn Fn() + Send + Sync>) {
        *lock(&self.videoout.listener) = Some(listener);
    }

    /// The guest address of registered display buffer `index`.
    pub fn display_buffer(&self, index: i32) -> Option<u64> {
        let port = lock(&self.videoout.port);
        port.buffers.get(usize::try_from(index).ok()?).copied().flatten().map(|(addr, _)| addr)
    }

    /// The buffer on screen, read from guest memory and untiled.
    pub fn frame(&self) -> Option<Frame> {
        let (addr, layout, flips) = {
            let port = lock(&self.videoout.port);
            let (addr, layout) = port.buffers.get(port.front?).copied().flatten()?;
            (addr, layout, port.flips)
        };
        #[cfg(target_os = "macos")]
        if let Some(texture) = gpu::metal::frame(addr) {
            return Some(Frame {
                width: layout.width,
                height: layout.height,
                rgba: Vec::new(),
                flips,
                texture: Some(texture),
            });
        }
        let mut raw = vec![0u8; buffer_len(layout)];
        self.memory().read(addr, &mut raw).ok()?;
        let rgba = if layout.tiled {
            untile(&raw, layout)
        } else {
            raw
        };
        Some(Frame {
            width: layout.width,
            height: layout.height,
            rgba,
            flips,
            #[cfg(target_os = "macos")]
            texture: None,
        })
    }
}

/// Bytes per 128x128-pixel tile of 32-bit pixels.
const TILE_BYTES: usize = 128 * 128 * 4;

fn buffer_len(l: Layout) -> usize {
    let (w, h) = (l.width as usize, l.height as usize);
    if l.tiled {
        w.div_ceil(128) * h.div_ceil(128) * TILE_BYTES
    } else {
        w * h * 4
    }
}

/// For each coordinate bit (x or y, bit number), the bit of the byte offset
/// inside the tile it flips; the offset is the XOR of all of them.
const TILE_BITS: [(char, u32, u32); 18] = [
    ('x', 0, 2),
    ('x', 1, 3),
    ('x', 2, 7),
    ('x', 3, 8),
    ('x', 4, 9),
    ('x', 4, 13),
    ('x', 5, 11),
    ('x', 6, 10),
    ('x', 6, 15),
    ('y', 0, 4),
    ('y', 1, 5),
    ('y', 2, 6),
    ('y', 3, 8),
    ('y', 3, 12),
    ('y', 4, 9),
    ('y', 5, 10),
    ('y', 6, 11),
    ('y', 6, 14),
];

fn tile_offset(x: u32, y: u32) -> usize {
    TILE_BITS.iter().fold(0, |o, &(axis, from, to)| {
        let v = if axis == 'x' { x } else { y };
        o ^ (((v >> from) & 1) << to) as usize
    })
}

fn untile(raw: &[u8], l: Layout) -> Vec<u8> {
    let (w, h) = (l.width as usize, l.height as usize);
    let tiles_per_row = w.div_ceil(128);
    let offsets: Vec<usize> = (0..128 * 128)
        .map(|i| tile_offset(i % 128, i / 128))
        .collect();
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let tile = (y / 128) * tiles_per_row + x / 128;
            let src = tile * TILE_BYTES + offsets[(y % 128) * 128 + x % 128];
            let dst = (y * w + x) * 4;
            out[dst..dst + 4].copy_from_slice(&raw[src..src + 4]);
        }
    }
    out
}

pub static FUNCTIONS: &[Function] = &[
    f("sceVideoOutOpen", open),
    f("sceVideoOutClose", close),
    f("sceVideoOutGetFlipStatus", flip_status),
    f("sceVideoOutSetFlipRate", with_handle),
    f("sceVideoOutSetBufferAttribute2", set_buffer_attribute2),
    f("sceVideoOutRegisterBuffers2", register_buffers2),
    f("sceVideoOutSubmitFlip", submit_flip),
    f("sceVideoOutWaitVblank", wait_vblank),
    f("sceVideoOutAddFlipEvent", add_flip_event),
    f("sceVideoOutDeleteFlipEvent", delete_flip_event),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[f("sceVideoOutGetOutputStatus", output_status)];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libSceVideoOut",
        name,
        func,
    }
}

fn ret(v: u64) -> Result<Outcome, CallError> {
    Ok(Outcome::Return(v))
}

/// `sceVideoOutOpen(user, bus, index, param)`: one port, so a second open fails.
fn open(env: &mut Env) -> Result<Outcome, CallError> {
    let mut port = lock(&env.kernel.videoout.port);
    if port.open {
        return ret(ERROR_INVALID_VALUE);
    }
    *port = Port {
        open: true,
        ..Port::default()
    };
    drop(port);
    if let Some(listener) = lock(&env.kernel.videoout.on_open).as_ref() {
        listener();
    }
    ret(HANDLE)
}

fn is_open(env: &Env) -> bool {
    env.arg(0) == HANDLE && lock(&env.kernel.videoout.port).open
}

fn close(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    *lock(&env.kernel.videoout.port) = Port::default();
    ret(0)
}

fn with_handle(env: &mut Env) -> Result<Outcome, CallError> {
    ret(if is_open(env) {
        0
    } else {
        ERROR_INVALID_HANDLE
    })
}

/// `sceVideoOutSetBufferAttribute2(attr, format, tiling, width, height,
/// option, dcc_control, dcc_clear_color)`.
fn set_buffer_attribute2(env: &mut Env) -> Result<Outcome, CallError> {
    let (attr, format, tiling, width, height) = (
        env.arg(0),
        env.arg(1),
        env.arg(2) as u32,
        env.arg(3) as u32,
        env.arg(4) as u32,
    );
    let mut b = [0u8; 32];
    b[0..4].copy_from_slice(&ATTRIBUTE_TAG.to_le_bytes());
    b[8..16].copy_from_slice(&format.to_le_bytes());
    b[16..20].copy_from_slice(&tiling.to_le_bytes());
    b[20..24].copy_from_slice(&width.to_le_bytes());
    b[24..28].copy_from_slice(&height.to_le_bytes());
    env.mem.write(attr, &b)?;
    ret(0)
}

fn read_attribute(env: &Env, attr: u64) -> Result<Option<Layout>, CallError> {
    let mut b = [0u8; 32];
    env.mem.read(attr, &mut b)?;
    let word = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"));
    if word(0) != ATTRIBUTE_TAG {
        return Ok(None);
    }
    let layout = Layout {
        width: word(20),
        height: word(24),
        tiled: word(16) == 0,
    };
    Ok((layout.width > 0 && layout.height > 0).then_some(layout))
}

/// `sceVideoOutRegisterBuffers2(handle, set, first, buffers, count, attr,
/// category, option)`.
fn register_buffers2(env: &mut Env) -> Result<Outcome, CallError> {
    let (first, list, count, attr) = (
        env.arg(2) as usize,
        env.arg(3),
        env.arg(4) as usize,
        env.arg(5),
    );
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    if count == 0 || first + count > MAX_BUFFERS {
        return ret(ERROR_INVALID_INDEX);
    }
    let Some(layout) = read_attribute(env, attr)? else {
        return ret(ERROR_INVALID_VALUE);
    };
    let mut addrs = Vec::with_capacity(count);
    for i in 0..count as u64 {
        addrs.push(env.mem.read_u64(list + i * BUFFER_ENTRY_SIZE)?);
    }
    let mut port = lock(&env.kernel.videoout.port);
    if port.buffers.len() < first + count {
        port.buffers.resize(first + count, None);
    }
    for (i, addr) in addrs.into_iter().enumerate() {
        port.buffers[first + i] = Some((addr, layout));
    }
    ret(0)
}

/// `sceVideoOutSubmitFlip(handle, index, mode, arg)`: the buffer goes on
/// screen at once (index -1 blanks it) and flip events fire with `arg`.
fn submit_flip(env: &mut Env) -> Result<Outcome, CallError> {
    let (index, arg) = (env.arg(1) as i32, env.arg(3));
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    match env.kernel.flip(index, arg) {
        true => ret(0),
        false => ret(ERROR_INVALID_INDEX),
    }
}

impl Kernel {
    /// Puts buffer `index` on screen (-1 blanks it) and fires flip events
    /// with `arg`; false if no buffer is registered at `index`.
    pub(crate) fn flip(&self, index: i32, arg: u64) -> bool {
        {
            let mut port = lock(&self.videoout.port);
            let front = usize::try_from(index).ok();
            if front.is_some_and(|i| port.buffers.get(i).copied().flatten().is_none()) {
                return false;
            }
            port.front = front;
            port.flips += 1;
            port.last_arg = arg;
            port.last_flip = Some(Instant::now());
        }
        self.trigger_event(FLIP_EVENT, EVENT_FILTER, arg);
        if let Some(listener) = lock(&self.videoout.listener).as_ref() {
            listener();
        }
        true
    }
}

/// `sceVideoOutWaitVblank(handle)`: sleeps until the next 1/60 s boundary.
fn wait_vblank(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    let start = env.kernel.videoout.start;
    let n = start.elapsed().as_nanos() / VBLANK.as_nanos() + 1;
    let next = start + VBLANK * n as u32;
    while let Some(left) = next.checked_duration_since(Instant::now()) {
        if env.kernel.has_exited() {
            break;
        }
        std::thread::sleep(left.min(POLL));
    }
    ret(0)
}

/// `sceVideoOutAddFlipEvent(queue, handle, udata)`.
fn add_flip_event(env: &mut Env) -> Result<Outcome, CallError> {
    let (queue, handle, udata) = (env.arg(0), env.arg(1), env.arg(2));
    if handle != HANDLE || !lock(&env.kernel.videoout.port).open {
        return ret(ERROR_INVALID_HANDLE);
    }
    let added = env.kernel.add_event(queue, FLIP_EVENT, EVENT_FILTER, udata);
    ret(if added { 0 } else { ERROR_INVALID_EVENT_QUEUE })
}

fn delete_flip_event(env: &mut Env) -> Result<Outcome, CallError> {
    let removed = env
        .kernel
        .remove_event(env.arg(0), FLIP_EVENT, EVENT_FILTER);
    ret(if removed {
        0
    } else {
        ERROR_INVALID_EVENT_QUEUE
    })
}

/// `sceVideoOutGetOutputStatus(handle, status)`: first field 1, which
/// Minecraft reads as 1080p; the rest of the layout is unknown.
fn output_status(env: &mut Env) -> Result<Outcome, CallError> {
    env.mem.write(env.arg(1), &1u32.to_le_bytes())?;
    Ok(Outcome::Return(0))
}

/// `sceVideoOutGetFlipStatus(handle, status)` in OpenOrbis's
/// `OrbisVideoOutFlipStatus` layout; flips are never pending.
fn flip_status(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    let start = env.kernel.videoout.start;
    let (count, arg, time, front) = {
        let port = lock(&env.kernel.videoout.port);
        let time = port.last_flip.map_or(0, |t| t.duration_since(start).as_micros() as u64);
        (port.flips, port.last_arg, time, port.front.map_or(-1, |i| i as i32))
    };
    let mut b = [0u8; 0x40];
    b[0..8].copy_from_slice(&count.to_le_bytes());
    b[8..16].copy_from_slice(&time.to_le_bytes());
    b[16..24].copy_from_slice(&time.to_le_bytes());
    b[24..32].copy_from_slice(&arg.to_le_bytes());
    b[0x38..0x3c].copy_from_slice(&front.to_le_bytes());
    env.mem.write(env.arg(1), &b)?;
    ret(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_offsets_cover_every_pixel_of_a_tile_once() {
        let mut seen = vec![false; 128 * 128];
        for y in 0..128 {
            for x in 0..128 {
                let o = tile_offset(x, y);
                assert!(o.is_multiple_of(4) && !seen[o / 4]);
                seen[o / 4] = true;
            }
        }
        assert_eq!(tile_offset(1, 0), 0x4);
        assert_eq!(tile_offset(0, 1), 0x10);
        assert_eq!(tile_offset(16, 0), 0x2200);
        assert_eq!(tile_offset(0, 8), 0x1100);
        assert_eq!(tile_offset(64, 64), 0x8400 ^ 0x4800);
    }

    #[test]
    fn untiling_puts_each_pixel_back_in_place() {
        let layout = Layout {
            width: 200,
            height: 130,
            tiled: true,
        };
        let mut raw = vec![0u8; buffer_len(layout)];
        let points = [(0usize, 0usize), (199, 0), (5, 129), (130, 70)];
        for (x, y) in points {
            let tile = (y / 128) * 2 + x / 128;
            let at = tile * TILE_BYTES + tile_offset(x as u32 % 128, y as u32 % 128);
            raw[at..at + 4].copy_from_slice(&[x as u8, y as u8, 1, 2]);
        }
        let out = untile(&raw, layout);
        for (x, y) in points {
            let at = (y * 200 + x) * 4;
            assert_eq!(&out[at..at + 4], &[x as u8, y as u8, 1, 2]);
        }
    }
}
