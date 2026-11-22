//! libScePad: one controller, for the one user, fed by the host (the
//! window maps keys and controllers to it). `ScePadData` follows OpenOrbis's
//! `OrbisPadData` (the PS4 layout; unverified for the PS5). What the guest
//! asks of the controller (vibration, light bar, trigger effects) is kept in
//! [`PadOutput`] for the host to carry out. Parameter layouts follow shadPS4's
//! and SharpEmu-mac's `pad` code (see docs/references.md), unverified on a PS5.

use std::sync::Mutex;

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
        Self::REST
    }
}

impl PadInput {
    /// Nothing pressed, sticks centred.
    pub const REST: Self = Self {
        buttons: 0,
        left: (128, 128),
        right: (128, 128),
        l2: 0,
        r2: 0,
    };

    /// Both inputs at once: buttons of either, each stick axis and trigger
    /// from whichever is pushed further.
    pub fn merge(self, other: Self) -> Self {
        let axis = |a: u8, b: u8| {
            if (i16::from(b) - 128).abs() > (i16::from(a) - 128).abs() {
                b
            } else {
                a
            }
        };
        Self {
            buttons: self.buttons | other.buttons,
            left: (
                axis(self.left.0, other.left.0),
                axis(self.left.1, other.left.1),
            ),
            right: (
                axis(self.right.0, other.right.0),
                axis(self.right.1, other.right.1),
            ),
            l2: self.l2.max(other.l2),
            r2: self.r2.max(other.r2),
        }
    }

    /// This input without what `other` presses: its buttons are released, and
    /// every stick axis or trigger it moves goes back to rest.
    pub fn without(self, other: Self) -> Self {
        let axis = |a: u8, b: u8, rest: u8| if b != rest { rest } else { a };
        Self {
            buttons: self.buttons & !other.buttons,
            left: (
                axis(self.left.0, other.left.0, 128),
                axis(self.left.1, other.left.1, 128),
            ),
            right: (
                axis(self.right.0, other.right.0, 128),
                axis(self.right.1, other.right.1, 128),
            ),
            l2: axis(self.l2, other.l2, 0),
            r2: axis(self.r2, other.r2, 0),
        }
    }
}

/// What the guest wants the controller to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PadOutput {
    /// Large (low-frequency, left) and small (high-frequency, right) motor,
    /// 0 to 255.
    pub vibration: (u8, u8),
    /// Red, green and blue; `None` is the system's own colour.
    pub light_bar: Option<[u8; 3]>,
    /// L2, then R2.
    pub triggers: [TriggerEffect; 2],
}

/// A DualSense adaptive trigger effect, as `scePadSetTriggerEffect` gives it.
/// Positions run from 0 (released) to 9 (fully pulled), strengths and
/// amplitudes from 1 to 8 (0 is no force in the per-zone forms).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TriggerEffect {
    #[default]
    Off,
    Feedback {
        position: u8,
        strength: u8,
    },
    Weapon {
        start: u8,
        end: u8,
        strength: u8,
    },
    Vibration {
        position: u8,
        amplitude: u8,
        frequency: u8,
    },
    MultiplePositionFeedback {
        strengths: [u8; 10],
    },
    SlopeFeedback {
        start: u8,
        end: u8,
        start_strength: u8,
        end_strength: u8,
    },
    MultiplePositionVibration {
        frequency: u8,
        amplitudes: [u8; 10],
    },
}

const TRIGGER_PARAM_SIZE: usize = 120;
const TRIGGER_COMMAND_SIZE: usize = 56;

impl TriggerEffect {
    /// One 56-byte command: a `u32` mode, 4 bytes of padding, then the
    /// mode's parameters. `None` for a mode we do not know.
    fn decode(c: &[u8]) -> Option<Self> {
        let mode = u32::from_le_bytes(c[0..4].try_into().unwrap());
        let p = &c[8..];
        let ten = |at: usize| -> [u8; 10] { p[at..at + 10].try_into().unwrap() };
        Some(match mode {
            0 => Self::Off,
            1 => Self::Feedback {
                position: p[0],
                strength: p[1],
            },
            2 => Self::Weapon {
                start: p[0],
                end: p[1],
                strength: p[2],
            },
            3 => Self::Vibration {
                position: p[0],
                amplitude: p[1],
                frequency: p[2],
            },
            4 => Self::MultiplePositionFeedback { strengths: ten(0) },
            5 => Self::SlopeFeedback {
                start: p[0],
                end: p[1],
                start_strength: p[2],
                end_strength: p[3],
            },
            6 => Self::MultiplePositionVibration {
                frequency: p[0],
                amplitudes: ten(1),
            },
            _ => return None,
        })
    }
}

/// `ScePadTriggerEffectParam`: a trigger mask (bit 0 L2, bit 1 R2), 7 bytes
/// of padding, then a command for L2 and one for R2. Returns the new effects
/// for the triggers the mask selects.
fn decode_trigger_param(d: &[u8; TRIGGER_PARAM_SIZE]) -> Option<[Option<TriggerEffect>; 2]> {
    let mask = d[0];
    if mask & !3 != 0 {
        return None;
    }
    let mut out = [None; 2];
    for (i, slot) in out.iter_mut().enumerate() {
        let at = 8 + i * TRIGGER_COMMAND_SIZE;
        let effect = TriggerEffect::decode(&d[at..at + TRIGGER_COMMAND_SIZE])?;
        if mask & (1 << i) != 0 {
            *slot = Some(effect);
        }
    }
    Some(out)
}

pub struct Pad {
    input: Mutex<PadInput>,
    output: Mutex<PadOutput>,
    open: Mutex<bool>,
    reads: Mutex<u8>,
    pub(crate) start: crate::process::Uptime,
}

impl Default for Pad {
    fn default() -> Self {
        Self {
            input: Mutex::default(),
            output: Mutex::default(),
            open: Mutex::new(false),
            reads: Mutex::new(0),
            start: Default::default(),
        }
    }
}

impl Kernel {
    /// Sets what the guest reads from its controller.
    pub fn set_pad_input(&self, input: PadInput) {
        *lock(&self.pad.input) = input;
    }

    /// What the guest last asked of its controller.
    pub fn pad_output(&self) -> PadOutput {
        *lock(&self.pad.output)
    }
}

/// Changes the output state, and logs the change.
fn update_output(env: &Env, call: &str, change: impl FnOnce(&mut PadOutput)) {
    let mut out = lock(&env.kernel.pad.output);
    let before = *out;
    change(&mut out);
    if *out != before {
        tracing::debug!("{call}: {:?}", *out);
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
    f("scePadSetVibration", set_vibration),
    f("scePadSetLightBar", set_light_bar),
    f("scePadResetLightBar", reset_light_bar),
    f("scePadSetTriggerEffect", set_trigger_effect),
];

/// Placeholders: they answer without doing the real work, and say so on
/// stderr when called.
pub static STUBS: &[Function] = &[
    f("scePadSetMotionSensorState", with_handle),
    f("scePadSetTiltCorrectionState", with_handle),
    f("scePadSetAngularVelocityDeadbandState", with_handle),
    f("scePadResetOrientation", with_handle),
    f("scePadSetVibrationMode", with_handle),
    f(
        "scePadDeviceClassGetExtendedInformation",
        device_class_information,
    ),
    // Names unknown. Deltarune calls both every frame after
    // scePadSetVibration, with the handle and a pointer to one byte.
    f("#fCWdlnmB1Ks", with_handle),
    f("#oARLr0ca9N8", with_handle),
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
    update_output(env, "scePadClose", |o| *o = PadOutput::default());
    ret(0)
}

/// `scePadSetVibration(handle, param)`, `param` being
/// `{ u8 largeMotor; u8 smallMotor; }`.
fn set_vibration(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    if env.arg(1) == 0 {
        return ret(ERROR_INVALID_ARG);
    }
    let mut p = [0u8; 2];
    env.mem.read(env.arg(1), &mut p)?;
    update_output(env, "scePadSetVibration", |o| o.vibration = (p[0], p[1]));
    ret(0)
}

/// `scePadSetLightBar(handle, param)`, `param` being `{ u8 r, g, b, reserved; }`.
fn set_light_bar(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    if env.arg(1) == 0 {
        return ret(ERROR_INVALID_ARG);
    }
    let mut p = [0u8; 3];
    env.mem.read(env.arg(1), &mut p)?;
    update_output(env, "scePadSetLightBar", |o| o.light_bar = Some(p));
    ret(0)
}

fn reset_light_bar(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    update_output(env, "scePadResetLightBar", |o| o.light_bar = None);
    ret(0)
}

/// `scePadSetTriggerEffect(handle, param)`.
fn set_trigger_effect(env: &mut Env) -> Result<Outcome, CallError> {
    if !is_open(env) {
        return ret(ERROR_INVALID_HANDLE);
    }
    if env.arg(1) == 0 {
        return ret(ERROR_INVALID_ARG);
    }
    let mut p = [0u8; TRIGGER_PARAM_SIZE];
    env.mem.read(env.arg(1), &mut p)?;
    let Some(effects) = decode_trigger_param(&p) else {
        return ret(ERROR_INVALID_ARG);
    };
    update_output(env, "scePadSetTriggerEffect", |o| {
        for (slot, effect) in o.triggers.iter_mut().zip(effects) {
            if let Some(e) = effect {
                *slot = e;
            }
        }
    });
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

impl Pad {
    pub(crate) fn save(&self, e: &mut savestate::Encoder) {
        e.put(&*lock(&self.open))
            .put(&*lock(&self.reads))
            .put(&self.start.elapsed());
    }

    pub(crate) fn load(&self, d: &mut savestate::Decoder) -> Result<(), savestate::Error> {
        *lock(&self.open) = d.get()?;
        *lock(&self.reads) = d.get()?;
        self.start.set(d.get()?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_inputs_press_both_and_push_each_axis_the_further_way() {
        let walk = PadInput {
            left: (128, 0),
            ..PadInput::REST
        };
        let aim = PadInput {
            buttons: 0x100,
            right: (200, 128),
            l2: 255,
            ..PadInput::REST
        };
        let both = walk.merge(aim);
        assert_eq!(
            both,
            PadInput {
                buttons: 0x100,
                left: (128, 0),
                right: (200, 128),
                l2: 255,
                r2: 0,
            }
        );
    }

    #[test]
    fn releasing_part_of_an_input_keeps_the_rest() {
        let held = PadInput {
            buttons: 0x100 | 0x4000,
            left: (128, 0),
            l2: 255,
            ..PadInput::REST
        };
        let aim = PadInput {
            buttons: 0x100,
            l2: 255,
            ..PadInput::REST
        };
        let left = held.without(aim);
        assert_eq!(
            left,
            PadInput {
                buttons: 0x4000,
                left: (128, 0),
                ..PadInput::REST
            }
        );
        assert_eq!(left.without(left), PadInput::REST);
    }

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

    fn trigger_param(mask: u8, l2: (u32, &[u8]), r2: (u32, &[u8])) -> [u8; TRIGGER_PARAM_SIZE] {
        let mut d = [0u8; TRIGGER_PARAM_SIZE];
        d[0] = mask;
        for (at, (mode, params)) in [(8, l2), (64, r2)] {
            d[at..at + 4].copy_from_slice(&mode.to_le_bytes());
            d[at + 8..at + 8 + params.len()].copy_from_slice(params);
        }
        d
    }

    #[test]
    fn trigger_effects_decode_only_for_masked_triggers() {
        let d = trigger_param(2, (1, &[3, 5]), (2, &[2, 6, 8]));
        assert_eq!(
            decode_trigger_param(&d),
            Some([
                None,
                Some(TriggerEffect::Weapon {
                    start: 2,
                    end: 6,
                    strength: 8
                })
            ])
        );
    }

    #[test]
    fn per_zone_trigger_effects_take_ten_values() {
        let zones = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8];
        let mut vib = vec![40];
        vib.extend_from_slice(&zones);
        let d = trigger_param(3, (4, &zones), (6, &vib));
        assert_eq!(
            decode_trigger_param(&d),
            Some([
                Some(TriggerEffect::MultiplePositionFeedback { strengths: zones }),
                Some(TriggerEffect::MultiplePositionVibration {
                    frequency: 40,
                    amplitudes: zones
                })
            ])
        );
    }

    #[test]
    fn unknown_trigger_modes_and_mask_bits_are_rejected() {
        assert_eq!(
            decode_trigger_param(&trigger_param(1, (7, &[]), (0, &[]))),
            None
        );
        assert_eq!(
            decode_trigger_param(&trigger_param(4, (0, &[]), (0, &[]))),
            None
        );
    }
}
