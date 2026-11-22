# Input and controller output

How a title's pad (libScePad) is fed and what it can make the controller do. The HLE side is in `crates/hle/src/pad.rs`, the host side in `crates/ps5slop/src/window/gamepad.rs` (input) and `feedback.rs` (output). Both host parts live in the window loop: with `--headless` there is no controller, only the control socket.

## Sources of input

All three are merged into one pad for user 1 (buttons from all, sticks and triggers from whichever is pushed further):

- **Controllers** through Apple's GameController framework (`GCController`). The first connected controller with an extended gamepad profile is used: DualSense, DualShock 4, Xbox and other MFi pads, over USB or Bluetooth. The list is asked for on every poll (every 8 ms and after window events), so plugging in or removing a controller needs nothing else; the log says `ps5slop: controller: ...` when the controller changes. Buttons, both sticks, analogue L2/R2 and the touch pad click (DualSense and DualShock 4) are mapped. Touch points and motion sensors are not.
- **Keyboard** in the window: arrows, Z/X/C/V for square/cross/circle/triangle, Q/E for L1/R1, 1/3 for L2/R2, Enter for Options, T for the touch pad. Debug capture hotkeys: F9 saves a `.ps5frame` GPU frame capture, F10 saves a Metal `.gputrace` bundle (with `METAL_CAPTURE_ENABLED=1`); see [debugging.md](debugging.md#gpu-frame-capture-and-replay).
- **Control socket** `press BUTTONS [MS]` (see [debugging.md](debugging.md)), which also works headless.

Background events are on (`shouldMonitorBackgroundEvents`), so a controller keeps working while another window has focus.

## Controller output

The guest's calls only record what it wants in `PadOutput`; the window applies changes to the current controller on its next poll, and applies the whole state again when a different controller appears. Nothing happens when no controller is connected, and calls still succeed. `scePadClose` and the window closing stop the motors and switch trigger effects off. `RUST_LOG=hle::pad=debug` logs each change of `PadOutput` with the call that caused it.

| Call | Parameter (unverified on a PS5) | Host |
| --- | --- | --- |
| `scePadSetVibration(handle, param)` | `{ u8 largeMotor; u8 smallMotor; }` | Core Haptics engines from `GCController.haptics`: large motor on the left handle, small on the right, when the controller has both localities; otherwise one engine on the default locality driven by the stronger of the two. Each change of a motor's value stops its player and starts a new one with one endless continuous event at that intensity (0..255 to 0..1); 0 only stops it (see below for why) |
| `scePadSetLightBar(handle, param)` | `{ u8 r, g, b, reserved; }` | `GCController.light` (DualSense and DualShock 4) |
| `scePadResetLightBar(handle)` | | Blue (our choice; the console's colour depends on the user). Until a title sets a colour, the light bar keeps the system's |
| `scePadSetTriggerEffect(handle, param)` | 120 bytes: `u8` trigger mask (bit 0 L2, bit 1 R2), 7 padding, then a 56-byte command for L2 and one for R2: `u32` mode, 4 padding, the mode's bytes | `GCDualSenseAdaptiveTrigger` on a DualSense; ignored on other controllers |

Trigger modes and their bytes, and how they are mapped (positions 0..9 become 0..1, strengths and amplitudes 1..8 become 0..1, frequency /255):

| Mode | Bytes | GameController |
| --- | --- | --- |
| 0 off | | `setModeOff` |
| 1 feedback | position, strength | `setModeFeedbackWithStartPosition:resistiveStrength:` |
| 2 weapon | start, end, strength | `setModeWeaponWithStartPosition:endPosition:resistiveStrength:` |
| 3 vibration | position, amplitude, frequency | `setModeVibrationWithStartPosition:amplitude:frequency:` |
| 4 multiple-position feedback | strength per zone (10) | `setModeFeedbackWithResistiveStrengths:` |
| 5 slope feedback | start, end, start strength, end strength | `setModeSlopeFeedbackWithStartPosition:endPosition:startStrength:endStrength:` |
| 6 multiple-position vibration | frequency, amplitude per zone (10) | `setModeVibrationWithAmplitudes:frequency:` |

A mask with other bits set, or an unknown mode, fails with `0x80920001` (invalid argument). Sources: the vibration and light bar layouts from shadPS4's `pad.h`, the trigger effect layout and mode numbers from SharpEmu-mac's `PadExports.cs` (both GPL-2.0-or-later, see [references.md](references.md)). Neither is checked on a PS5. How the guest's frequency unit compares to GameController's 0..1 range is a guess.

Still placeholders: `scePadSetVibrationMode`, `scePadGetTriggerEffectState` (no effect engaged), motion sensor and orientation calls.

## Core Haptics on macOS: what works

Found with a small probe against a Bluetooth DualShock 4 on 2026-11-20 (macOS 27, Darwin 27.3), with vic holding the controller:

- `createEngineWithLocality:` is marked iOS-only in objc2-game-controller 0.3, but exists on macOS; we call it with `msg_send!`.
- A plain player (`createPlayerWithPattern:`) plays a continuous event, also with `GCHapticDurationInfinite`. Felt.
- An advanced player (`createAdvancedPlayerWithPattern:`) fails with "Couldn't communicate with a helper application". The system log shows `gamecontrollerd` rejecting `loadHapticSequenceFromEvents:` because `AVHapticEvent` is not among the classes it allows to be decoded. This looks like a bug in macOS, not ours.
- Changing a playing event's intensity with `CHHapticDynamicParameterIDHapticIntensityControl` succeeds but the motors do not change. Starting a new player at a lower event intensity does work (felt: full, then 15%).

## Entitlements and permissions

None beyond the hypervisor entitlement. GameController works in an unbundled, unsandboxed command-line binary with no `Info.plist`: on 2026-11-20 a Bluetooth DualShock 4 was listed with handle haptics and a light bar from the ad-hoc signed `ps5slop`. Bluetooth controllers are paired in System Settings beforehand; ps5slop does not start wireless discovery.

## What is verified

- Parameter decoding: unit tests in `pad.rs` (trigger mask, every mode's bytes, rejection of unknown modes) and the scale mapping in `feedback.rs`.
- Isaac (2026-11-20, headless, `RUST_LOG=hle::pad=debug`): at start it calls `scePadSetVibrationMode(handle, 2)`; a bomb explosion in a run gives `scePadSetVibration` with `(255, 255)` and then `(0, 0)`. It does not set the light bar or trigger effects.
- Isaac with the window and a Bluetooth DualShock 4 (2026-11-20): the controller is detected and described; a bomb explosion rumbles the controller (felt by vic) and stops afterwards.

- GTA V (2026-11-21, headless, control socket): its pad thread reads `scePadReadState` on handle 1 about 75 times a second and gates on the connected byte at offset 0x4C; `press cross` reached its per-pad state and moved its menus (see [hle-kernel.md](hle-kernel.md)).

Not yet checked on hardware: the light bar (Isaac never sets it), DualSense trigger effects (no title we run sets them yet), the split between left and right handle motors, DualSense and Xbox controllers, and USB connections.
