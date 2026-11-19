//! Controllers through Apple's GameController framework, which knows the
//! DualShock 4 and DualSense by name. Read by polling on the main thread.

use hle::PadInput;
use objc2::rc::Retained;
use objc2_game_controller::{
    GCController, GCControllerButtonInput, GCControllerDirectionPad, GCDualSenseGamepad,
    GCDualShockGamepad, GCExtendedGamepad,
};

const UP: u32 = 0x0010;
const RIGHT: u32 = 0x0020;
const DOWN: u32 = 0x0040;
const LEFT: u32 = 0x0080;
const L3: u32 = 0x0002;
const R3: u32 = 0x0004;
const OPTIONS: u32 = 0x0008;
const L2: u32 = 0x0100;
const R2: u32 = 0x0200;
const L1: u32 = 0x0400;
const R1: u32 = 0x0800;
const TRIANGLE: u32 = 0x1000;
const CIRCLE: u32 = 0x2000;
const CROSS: u32 = 0x4000;
const SQUARE: u32 = 0x8000;
const TOUCH_PAD: u32 = 0x10_0000;

/// Keeps reporting controller input while the window is in the background.
pub fn init() {
    unsafe { GCController::setShouldMonitorBackgroundEvents(true) };
}

/// The state of the first connected controller with a full set of controls.
pub fn read() -> Option<PadInput> {
    let controllers = unsafe { GCController::controllers() };
    let pad = controllers
        .iter()
        .find_map(|c| unsafe { c.extendedGamepad() })?;
    Some(unsafe { state(pad) })
}

/// A stick axis from -1..1 (up positive) to the pad's 0..255 (down positive
/// when `flip`).
fn axis(v: f32, flip: bool) -> u8 {
    let v = if flip { -v } else { v };
    ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8
}

fn stick(s: Retained<GCControllerDirectionPad>) -> (u8, u8) {
    unsafe {
        (
            axis(s.xAxis().value(), false),
            axis(s.yAxis().value(), true),
        )
    }
}

fn pressed(b: Option<Retained<GCControllerButtonInput>>) -> bool {
    b.is_some_and(|b| unsafe { b.isPressed() })
}

unsafe fn touchpad(pad: &Retained<GCExtendedGamepad>) -> bool {
    let pad = pad.clone();
    match pad.downcast::<GCDualSenseGamepad>() {
        Ok(ds) => unsafe { ds.touchpadButton().isPressed() },
        Err(pad) => match pad.downcast::<GCDualShockGamepad>() {
            Ok(ds4) => pressed(unsafe { ds4.touchpadButton() }),
            Err(_) => false,
        },
    }
}

unsafe fn state(pad: Retained<GCExtendedGamepad>) -> PadInput {
    let (l2, r2) = unsafe { (pad.leftTrigger().value(), pad.rightTrigger().value()) };
    let dpad = unsafe { pad.dpad() };
    let held = unsafe {
        [
            (dpad.up().isPressed(), UP),
            (dpad.right().isPressed(), RIGHT),
            (dpad.down().isPressed(), DOWN),
            (dpad.left().isPressed(), LEFT),
            (pad.buttonA().isPressed(), CROSS),
            (pad.buttonB().isPressed(), CIRCLE),
            (pad.buttonX().isPressed(), SQUARE),
            (pad.buttonY().isPressed(), TRIANGLE),
            (pad.leftShoulder().isPressed(), L1),
            (pad.rightShoulder().isPressed(), R1),
            (pad.leftTrigger().isPressed(), L2),
            (pad.rightTrigger().isPressed(), R2),
            (pressed(pad.leftThumbstickButton()), L3),
            (pressed(pad.rightThumbstickButton()), R3),
            (pad.buttonMenu().isPressed(), OPTIONS),
            (touchpad(&pad), TOUCH_PAD),
        ]
    };
    PadInput {
        buttons: held
            .iter()
            .filter(|(on, _)| *on)
            .fold(0, |b, (_, bit)| b | bit),
        left: stick(unsafe { pad.leftThumbstick() }),
        right: stick(unsafe { pad.rightThumbstick() }),
        l2: (l2.clamp(0.0, 1.0) * 255.0).round() as u8,
        r2: (r2.clamp(0.0, 1.0) * 255.0).round() as u8,
    }
}

/// Combines controller and keyboard input: buttons from both, sticks and
/// triggers from whichever is pushed further.
pub fn merge(pad: Option<PadInput>, keys: PadInput) -> PadInput {
    let Some(pad) = pad else { return keys };
    let further = |a: u8, b: u8| {
        if a.abs_diff(128) >= b.abs_diff(128) {
            a
        } else {
            b
        }
    };
    PadInput {
        buttons: pad.buttons | keys.buttons,
        left: (
            further(pad.left.0, keys.left.0),
            further(pad.left.1, keys.left.1),
        ),
        right: (
            further(pad.right.0, keys.right.0),
            further(pad.right.1, keys.right.1),
        ),
        l2: pad.l2.max(keys.l2),
        r2: pad.r2.max(keys.r2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stick_axes_map_up_to_the_top_of_the_range() {
        assert_eq!(
            (axis(-1.0, false), axis(0.0, false), axis(1.0, false)),
            (0, 128, 255)
        );
        assert_eq!(axis(1.0, true), 0);
    }

    #[test]
    fn keyboard_and_controller_input_combine() {
        let pad = PadInput {
            buttons: CROSS,
            left: (200, 128),
            l2: 10,
            ..PadInput::default()
        };
        let keys = PadInput {
            buttons: UP,
            l2: 255,
            ..PadInput::default()
        };
        let m = merge(Some(pad), keys);
        assert_eq!((m.buttons, m.left, m.l2), (CROSS | UP, (200, 128), 255));
    }
}
