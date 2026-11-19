//! The host's default output device through an AudioToolbox output queue:
//! stereo 32-bit float at 48 kHz, filled from [`Mixer`] on the queue's own
//! thread.

use std::ffi::c_void;
use std::sync::Arc;

use super::{Mixer, RATE};

#[repr(C)]
struct StreamDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct QueueBuffer {
    capacity: u32,
    data: *mut c_void,
    size: u32,
    user_data: *mut c_void,
    packet_capacity: u32,
    packets: *mut c_void,
    packet_count: u32,
}

type Queue = *mut c_void;
type OutputCallback = extern "C" fn(*mut c_void, Queue, *mut QueueBuffer);

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioQueueNewOutput(
        format: *const StreamDescription,
        callback: OutputCallback,
        user_data: *mut c_void,
        run_loop: *const c_void,
        run_loop_mode: *const c_void,
        flags: u32,
        queue: *mut Queue,
    ) -> i32;
    fn AudioQueueAllocateBuffer(queue: Queue, size: u32, buffer: *mut *mut QueueBuffer) -> i32;
    fn AudioQueueEnqueueBuffer(
        queue: Queue,
        buffer: *mut QueueBuffer,
        packets: u32,
        descs: *const c_void,
    ) -> i32;
    fn AudioQueueStart(queue: Queue, at: *const c_void) -> i32;
}

const FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
const FLAG_FLOAT: u32 = 1;
const FLAG_PACKED: u32 = 8;
const BUFFERS: usize = 3;
const BUFFER_FRAMES: u32 = 512;

extern "C" fn fill(user: *mut c_void, queue: Queue, buffer: *mut QueueBuffer) {
    // SAFETY: `user` is the leaked `Arc<Mixer>` from `start`, and the queue
    // hands back one of its own buffers.
    unsafe {
        let mixer = &*(user as *const Mixer);
        let b = &mut *buffer;
        let frames = b.capacity as usize / 8;
        let out = std::slice::from_raw_parts_mut(b.data as *mut f32, frames * 2);
        mixer.pull(out);
        b.size = (frames * 8) as u32;
        AudioQueueEnqueueBuffer(queue, buffer, 0, std::ptr::null());
    }
}

/// Opens the default output device and starts pulling from `mixer`.
/// Returns the failing `OSStatus`.
pub fn start(mixer: Arc<Mixer>) -> Result<(), i32> {
    let format = StreamDescription {
        sample_rate: RATE as f64,
        format_id: FORMAT_LINEAR_PCM,
        format_flags: FLAG_FLOAT | FLAG_PACKED,
        bytes_per_packet: 8,
        frames_per_packet: 1,
        bytes_per_frame: 8,
        channels_per_frame: 2,
        bits_per_channel: 32,
        reserved: 0,
    };
    let user = Arc::into_raw(mixer) as *mut c_void;
    let mut queue: Queue = std::ptr::null_mut();
    // SAFETY: plain AudioToolbox calls with valid pointers; the queue and
    // the mixer reference live for the rest of the process.
    unsafe {
        let check = |status: i32| if status == 0 { Ok(()) } else { Err(status) };
        let null = std::ptr::null();
        check(AudioQueueNewOutput(
            &format, fill, user, null, null, 0, &mut queue,
        ))?;
        for _ in 0..BUFFERS {
            let mut buffer = std::ptr::null_mut();
            check(AudioQueueAllocateBuffer(
                queue,
                BUFFER_FRAMES * 8,
                &mut buffer,
            ))?;
            fill(user, queue, buffer);
        }
        check(AudioQueueStart(queue, null))
    }
}
