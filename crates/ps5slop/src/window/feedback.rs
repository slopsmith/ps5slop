//! What the guest asks of its controller, carried out through
//! GameController: vibration through Core Haptics engines, the light bar
//! through `GCDeviceLight` and DualSense trigger effects through
//! `GCDualSenseAdaptiveTrigger`. Runs on the main thread with the polling.

use hle::{PadOutput, TriggerEffect};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{AnyThread, msg_send};
use objc2_core_haptics::{
    CHHapticEngine, CHHapticEvent, CHHapticEventParameter, CHHapticEventParameterIDHapticIntensity,
    CHHapticEventParameterIDHapticSharpness, CHHapticEventTypeHapticContinuous, CHHapticPattern,
    CHHapticPatternPlayer, CHHapticTimeImmediate,
};
use objc2_foundation::{NSArray, NSString};
use objc2_game_controller::{
    GCColor, GCController, GCDevice, GCDeviceHaptics, GCDualSenseAdaptiveTrigger,
    GCDualSenseAdaptiveTriggerPositionalAmplitudes,
    GCDualSenseAdaptiveTriggerPositionalResistiveStrengths, GCDualSenseGamepad,
    GCHapticDurationInfinite, GCHapticsLocality, GCHapticsLocalityDefault,
    GCHapticsLocalityLeftHandle, GCHapticsLocalityRightHandle,
};

/// The light bar colour after `scePadResetLightBar`: the PS5's blue for the
/// first user (unverified). Until the guest sets a colour, the system's stays.
const DEFAULT_LIGHT: [u8; 3] = [0, 0, 255];

#[derive(Default)]
pub struct Feedback {
    controller: Option<Retained<GCController>>,
    /// What the controller was last set to; `None` when nothing was set yet.
    applied: Option<PadOutput>,
    rumble: Option<Rumble>,
}

/// Either one motor per handle (large left, small right) or, when the
/// controller only offers its default locality, one motor for both.
enum Rumble {
    Handles(Option<Motor>, Option<Motor>),
    Single(Option<Motor>),
}

struct Motor {
    engine: Retained<CHHapticEngine>,
    sharpness: f32,
    player: Option<Retained<ProtocolObject<dyn CHHapticPatternPlayer>>>,
}

impl Feedback {
    /// Brings `controller` in line with `out`. A controller that was not
    /// seen before gets the whole state.
    pub fn apply(&mut self, controller: Option<&Retained<GCController>>, out: PadOutput) {
        if controller != self.controller.as_ref() {
            self.reset();
            if let Some(c) = controller {
                tracing::info!("controller: {}", unsafe { describe(c) });
            } else if self.controller.is_some() {
                tracing::info!("controller disconnected");
            }
            self.controller = controller.cloned();
            self.applied = None;
        }
        let Some(c) = self.controller.clone() else {
            return;
        };
        let before = self.applied;
        if before.is_none_or(|b| b.vibration != out.vibration) {
            self.vibrate(&c, out.vibration);
        }
        let was = before.and_then(|b| b.light_bar);
        let colour = match out.light_bar {
            Some(rgb) if was != Some(rgb) => Some(rgb),
            None if was.is_some() => Some(DEFAULT_LIGHT),
            _ => None,
        };
        if let Some(rgb) = colour {
            unsafe { light(&c, rgb) };
        }
        if before.is_none_or(|b| b.triggers != out.triggers) {
            unsafe { triggers(&c, out.triggers) };
        }
        self.applied = Some(out);
    }

    fn vibrate(&mut self, c: &GCController, (large, small): (u8, u8)) {
        if large == 0 && small == 0 && self.rumble.is_none() {
            return;
        }
        let rumble = self.rumble.get_or_insert_with(|| unsafe { Rumble::new(c) });
        match rumble {
            Rumble::Handles(left, right) => {
                Motor::set(left, large);
                Motor::set(right, small);
            }
            Rumble::Single(motor) => Motor::set(motor, large.max(small)),
        }
    }

    /// Stops the motors and the trigger effects of the current controller.
    fn reset(&mut self) {
        if let Some(r) = self.rumble.take() {
            match r {
                Rumble::Handles(a, b) => [a, b].into_iter().flatten().for_each(Motor::stop),
                Rumble::Single(m) => m.into_iter().for_each(Motor::stop),
            }
        }
        if let Some(c) = &self.controller {
            unsafe { triggers(c, [TriggerEffect::Off; 2]) };
        }
    }
}

impl Drop for Feedback {
    fn drop(&mut self) {
        self.reset();
    }
}

/// The controller's kind, such as "DualShock 4".
pub fn kind(c: &GCController) -> String {
    unsafe { c.productCategory() }.to_string()
}

unsafe fn describe(c: &GCController) -> String {
    let name = |s: Option<Retained<NSString>>| s.map(|s| s.to_string()).unwrap_or_default();
    let (vendor, category) = unsafe { (name(c.vendorName()), c.productCategory().to_string()) };
    let haptics = unsafe { c.haptics() }.map_or("no haptics".into(), |h| {
        let all = unsafe { h.supportedLocalities() };
        let mut names: Vec<String> = all.iter().map(|l| l.to_string()).collect();
        names.sort();
        format!("haptics at {}", names.join(", "))
    });
    let light = if unsafe { c.light() }.is_some() {
        "light bar"
    } else {
        "no light bar"
    };
    format!("{vendor} ({category}), {haptics}, {light}")
}

impl Rumble {
    unsafe fn new(c: &GCController) -> Self {
        let Some(haptics) = (unsafe { c.haptics() }) else {
            tracing::info!("controller: no haptics, vibration ignored");
            return Self::Single(None);
        };
        let supported = unsafe { haptics.supportedLocalities() };
        let (left, right) = unsafe { (GCHapticsLocalityLeftHandle, GCHapticsLocalityRightHandle) };
        if supported.containsObject(left) && supported.containsObject(right) {
            unsafe {
                Self::Handles(
                    Motor::new(&haptics, left, 0.0),
                    Motor::new(&haptics, right, 1.0),
                )
            }
        } else {
            Self::Single(unsafe { Motor::new(&haptics, GCHapticsLocalityDefault, 0.5) })
        }
    }
}

impl Motor {
    unsafe fn new(
        haptics: &GCDeviceHaptics,
        at: &GCHapticsLocality,
        sharpness: f32,
    ) -> Option<Self> {
        let engine: Option<Retained<CHHapticEngine>> =
            unsafe { msg_send![haptics, createEngineWithLocality: at] };
        let Some(engine) = engine else {
            tracing::warn!("controller: no haptic engine at {at}");
            return None;
        };
        if let Err(e) = unsafe { engine.startAndReturnError() } {
            tracing::warn!("controller: haptics at {at}: {}", e.localizedDescription());
            return None;
        }
        Some(Self {
            engine,
            sharpness,
            player: None,
        })
    }

    /// Sets the intensity from 0 to 255 by replacing the player with one
    /// whose endless continuous event has that intensity. Changing a playing
    /// event's intensity control had no effect on a DualShock 4, and the
    /// advanced player fails in `gamecontrollerd` (see docs/input.md).
    /// A motor that fails is dropped.
    fn set(slot: &mut Option<Motor>, level: u8) {
        let Some(m) = slot else { return };
        let r = unsafe {
            m.player
                .take()
                .map_or(Ok(()), |p| p.stopAtTime_error(CHHapticTimeImmediate))
                .and_then(|()| {
                    if level == 0 {
                        return Ok(());
                    }
                    let params = NSArray::from_retained_slice(&[
                        CHHapticEventParameter::initWithParameterID_value(
                            CHHapticEventParameter::alloc(),
                            CHHapticEventParameterIDHapticIntensity,
                            level as f32 / 255.0,
                        ),
                        CHHapticEventParameter::initWithParameterID_value(
                            CHHapticEventParameter::alloc(),
                            CHHapticEventParameterIDHapticSharpness,
                            m.sharpness,
                        ),
                    ]);
                    let event = CHHapticEvent::initWithEventType_parameters_relativeTime_duration(
                        CHHapticEvent::alloc(),
                        CHHapticEventTypeHapticContinuous,
                        &params,
                        0.0,
                        GCHapticDurationInfinite as f64,
                    );
                    let pattern = CHHapticPattern::initWithEvents_parameters_error(
                        CHHapticPattern::alloc(),
                        &NSArray::from_retained_slice(&[event]),
                        &NSArray::new(),
                    )?;
                    let player = m.engine.createPlayerWithPattern_error(&pattern)?;
                    player.startAtTime_error(CHHapticTimeImmediate)?;
                    m.player = Some(player);
                    Ok(())
                })
        };
        if let Err(e) = r {
            tracing::warn!("controller: vibration: {}", e.localizedDescription());
            *slot = None;
        }
    }

    fn stop(self) {
        unsafe {
            if let Some(p) = self.player {
                let _ = p.stopAtTime_error(CHHapticTimeImmediate);
            }
            self.engine.stopWithCompletionHandler(std::ptr::null_mut());
        }
    }
}

unsafe fn light(c: &GCController, [r, g, b]: [u8; 3]) {
    let Some(l) = (unsafe { c.light() }) else {
        return;
    };
    let v = |x: u8| x as f32 / 255.0;
    unsafe {
        l.setColor(&GCColor::initWithRed_green_blue(
            GCColor::alloc(),
            v(r),
            v(g),
            v(b),
        ))
    };
}

unsafe fn triggers(c: &GCController, effects: [TriggerEffect; 2]) {
    let Some(pad) = (unsafe { c.extendedGamepad() }) else {
        return;
    };
    let Ok(ds) = pad.downcast::<GCDualSenseGamepad>() else {
        return;
    };
    unsafe {
        set_trigger(&ds.leftTrigger(), effects[0]);
        set_trigger(&ds.rightTrigger(), effects[1]);
    }
}

/// Zone 0 to 9 as a trigger position from 0 to 1.
fn position(p: u8) -> f32 {
    p.min(9) as f32 / 9.0
}

/// Strength or amplitude 1 to 8 as 0 to 1.
fn strength(s: u8) -> f32 {
    s.min(8) as f32 / 8.0
}

/// The guest's vibration frequency (Hz, up to 255) as GameController's 0 to 1.
fn frequency(f: u8) -> f32 {
    f as f32 / 255.0
}

unsafe fn set_trigger(t: &GCDualSenseAdaptiveTrigger, e: TriggerEffect) {
    let zones = |v: [u8; 10]| v.map(strength);
    unsafe {
        match e {
            TriggerEffect::Off => t.setModeOff(),
            TriggerEffect::Feedback {
                position: p,
                strength: s,
            } => t.setModeFeedbackWithStartPosition_resistiveStrength(position(p), strength(s)),
            TriggerEffect::Weapon {
                start,
                end,
                strength: s,
            } => t.setModeWeaponWithStartPosition_endPosition_resistiveStrength(
                position(start),
                position(end),
                strength(s),
            ),
            TriggerEffect::Vibration {
                position: p,
                amplitude,
                frequency: f,
            } => t.setModeVibrationWithStartPosition_amplitude_frequency(
                position(p),
                strength(amplitude),
                frequency(f),
            ),
            TriggerEffect::MultiplePositionFeedback { strengths } => t
                .setModeFeedbackWithResistiveStrengths(
                    GCDualSenseAdaptiveTriggerPositionalResistiveStrengths {
                        values: zones(strengths),
                    },
                ),
            TriggerEffect::SlopeFeedback {
                start,
                end,
                start_strength,
                end_strength,
            } => t.setModeSlopeFeedbackWithStartPosition_endPosition_startStrength_endStrength(
                position(start),
                position(end),
                strength(start_strength),
                strength(end_strength),
            ),
            TriggerEffect::MultiplePositionVibration {
                frequency: f,
                amplitudes,
            } => t.setModeVibrationWithAmplitudes_frequency(
                GCDualSenseAdaptiveTriggerPositionalAmplitudes {
                    values: zones(amplitudes),
                },
                frequency(f),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_scales_map_to_zero_to_one() {
        assert_eq!((position(0), position(9), position(12)), (0.0, 1.0, 1.0));
        assert_eq!((strength(4), strength(8)), (0.5, 1.0));
        assert_eq!(frequency(255), 1.0);
    }
}
