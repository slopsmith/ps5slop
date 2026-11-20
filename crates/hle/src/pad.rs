//! libScePad: one controller, for the one user, fed by the host (the
//! window maps keys to it). `ScePadData` follows OpenOrbis's
//! `OrbisPadData` (the PS4 layout; unverified for the PS5).

use std::sync::Mutex;
use std::time::Instant;

use crate::pthread::lock;
use crate::services::USER_ID;
use crate::{CallError, Env, Function, Kernel, Outcome};

const ERROR_INVALID_ARG: u64 = 0x8092_0001;
const ERROR_INVALID_HANDLE: u64 = 0x8092_0003;
const ERROR_ALREADY_OPENED: u64 = 0x8092_0004;
const HANDLE: u64 = 1;
const PAD_DATA_SIZE: usize = 120;

/// Controller state, with buttons as `ORBIS_PAD_BUTTON_*` bits and sticks
/// from 0 to 255 (128 is centred).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadInput {
    pub buttons: u32,
    pub left: (u8, u8),
    pub right: (u8, u8),
    pub l2: u8,
    pub r2: u8,
}

impl Default for PadInput {
    fn default() -> Self {
        Self {
            buttons: 0,
            left: (128, 128),
            right: (128, 128),
            l2: 0,
            r2: 0,
        }
    }
}

pub struct Pad {
    input: Mutex<PadInput>,
    open: Mutex<bool>,
    reads: Mutex<u8>,
    start: Instant,
}

impl Default for Pad {
    fn default() -> Self {
        Self {
            input: Mutex::default(),
            open: Mutex::new(false),
            reads: Mutex::new(0),
            start: Instant::now(),
        }
    }
}

impl Kernel {
    /// Sets what the guest reads from its controller.
    pub fn set_pad_input(&self, input: PadInput) {
        *lock(&self.pad.input) = input;
    }
}

pub static FUNCTIONS: &[Function] = &[
    f("scePadInit", ok),
    f("scePadOpen", open),
    f("scePadGetHandle", get_handle),
    f("scePadClose", close),
    f("scePadReadState", read_state),
    f("scePadRead", read),
    f("scePadGetControllerInformation", controller_information),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f("scePadSetVibration", with_handle),
    f("scePadSetLightBar", with_handle),
    f("scePadResetLightBar", with_handle),
    f("scePadSetMotionSensorState", with_handle),
    f("scePadSetTiltCorrectionState", with_handle),
    f("scePadSetAngularVelocityDeadbandState", with_handle),
    f("scePadResetOrientation", with_handle),
    f("scePadSetVibrationMode", with_handle),
    f("scePadSetTriggerEffect", with_handle),
    f(
        "scePadDeviceClassGetExtendedInformation",
        device_class_information,
    ),
];

const fn f(name: &'static str, func: crate::HostFn) -> Function {
    Function {
        library: "libScePad",
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

/// `scePadOpen(user, type, index, param)`.
fn open(env: &mut Env) -> Result<Outcome, CallError> {
    if env.arg(0) as i32 != USER_ID {
        return ret(ERROR_INVALID_ARG);
    }
    let mut open = lock(&env.kernel.pad.open);
    if *open {
        return ret(ERROR_ALREADY_OPENED);
    }
    *open = true;
    ret(HANDLE)
}

fn get_handle(env: &mut Env) -> Result<Outcome, CallError> {
    let open = *lock(&env.kernel.pad.open);
    ret(if open && env.arg(0) as i32 == USER_ID {
        HANDLE
    } else {
        ERROR_INVALID_HANDLE
    })
}

fn is_open(env: &Env) -> bool {
    env.arg(0) == HANDLE && *lock(&env.kernel.pad.open)
}

fn close(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    *lock(&env.kernel.pad.open) = false;
    ret(0)
}

/// Functions that only check the handle, such as vibration and the light bar.
fn with_handle(env: &mut Env) -> Result<Outcome, CallError> {
    ret(if is_open(env) {
        0
    } else {
        ERROR_INVALID_HANDLE
    })
}

/// `scePadDeviceClassGetExtendedInformation(handle, info)`: a standard
/// controller, device class 0 then 4 reserved bytes (shadPS4's PS4 layout,
/// unverified; the class data after them is left alone).
fn device_class_information(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    env.mem.write(env.arg(1), &[0u8; 8])?;
    ret(0)
}

fn pad_data(input: PadInput, timestamp: u64, count: u8) -> [u8; PAD_DATA_SIZE] {
    let mut d = [0u8; PAD_DATA_SIZE];
    d[0..4].copy_from_slice(&input.buttons.to_le_bytes());
    d[4..10].copy_from_slice(&[
        input.left.0,
        input.left.1,
        input.right.0,
        input.right.1,
        input.l2,
        input.r2,
    ]);
    d[24..28].copy_from_slice(&1f32.to_le_bytes());
    d[76] = 1;
    d[80..88].copy_from_slice(&timestamp.to_le_bytes());
    d[104] = count;
    d
}

/// `scePadReadState(handle, data)`.
fn read_state(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    write_sample(env)?;
    ret(0)
}

/// `scePadRead(handle, data, count)`: the samples since the last call. We
/// keep no history, so this is always one sample of the current state.
fn read(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    if env.arg(2) as i32 <= 0 {
        return ret(ERROR_INVALID_ARG);
    }
    write_sample(env)?;
    ret(1)
}

fn write_sample(env: &Env) -> Result<(), CallError> {
    let pad = &env.kernel.pad;
    let input = *lock(&pad.input);
    let count = {
        let mut reads = lock(&pad.reads);
        *reads = reads.wrapping_add(1);
        *reads
    };
    let timestamp = pad.start.elapsed().as_micros() as u64;
    env.mem
        .write(env.arg(1), &pad_data(input, timestamp, count))?;
    Ok(())
}

/// `scePadGetControllerInformation(handle, info)`, OpenOrbis's layout:
/// a connected controller with PS4 touch pad values (unverified for PS5).
fn controller_information(env: &mut Env) -> Result<Outcome, CallError> {
    let mut info = [0u8; 28];
    info[0..4].copy_from_slice(&44.86f32.to_le_bytes());
    info[4..6].copy_from_slice(&1920u16.to_le_bytes());
    info[6..8].copy_from_slice(&943u16.to_le_bytes());
    info[8] = 13;
    info[9] = 13;
    info[11] = 1;
    info[12..16].copy_from_slice(&1i32.to_le_bytes());
    env.mem.write(env.arg(1), &info)?;
    Ok(Outcome::Return(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_data_puts_buttons_sticks_and_connection_where_orbis_expects() {
        let input = PadInput {
            buttons: 0x4000,
            left: (0, 255),
            ..PadInput::default()
        };
        let d = pad_data(input, 7, 3);
        assert_eq!(&d[0..8], &[0x00, 0x40, 0, 0, 0, 255, 128, 128]);
        assert_eq!((d[76], d[80], d[104]), (1, 7, 3));
    }
}
