# Input and controller output

How a title's pad (libScePad) is fed and what it can make the controller do. The HLE side is `crates/hle/src/pad.rs`; the host side is `crates/ps5slop/src/window/gamepad.rs` (input) and `feedback.rs` (output). The key list players see is in [docs/user/playing.md](user/playing.md).

## Shape

- The window runs on the main thread (AppKit needs its event loop there) and the guest's threads run beside it. The window polls controllers every 8 ms and after window events.
- The HLE side holds two plain values: `PadInput`, which the host sets with `Kernel::set_pad_input`, and `PadOutput`, which the guest's calls record. The HLE never calls host APIs, so headless runs, the control socket and tests use the same code.
- With `--headless` there is no window and no controller; only the control socket's `press`, `tap` and `hold` and `PS5SLOP_INPUT` feed the pad ([debugging.md](debugging.md)).

## Sources of input

All sources merge into one pad for user 1: buttons from all of them, each stick axis and trigger from whichever source is pushed further from rest.

- **Controllers** through Apple's GameController framework. The first connected controller with an extended gamepad profile is used: DualSense, DualShock 4, Xbox and other MFi pads, over USB or Bluetooth. The list is read on every poll, so plugging and unplugging need nothing else. Buttons, sticks, analogue triggers and the touch pad click are mapped; touch points and motion sensors are not. Background events are on, so a controller keeps working while another window has focus.
- **Keyboard** in the window: arrows, face buttons, shoulders, triggers, Options and the touch pad click (`button()` in `window/mod.rs`). No sticks. F9 and F10 are capture hotkeys ([debugging.md](debugging.md)).
- **Control socket and `PS5SLOP_INPUT`**, which also work headless.

GameController was chosen because it maps almost one-to-one onto the console's pad data, names the PlayStation buttons, and can drive the light bar and rumble, which cross-platform crates cannot do well on macOS. It needs no entitlement and works in an unbundled, unsandboxed command-line binary. Bluetooth controllers are paired in System Settings beforehand; ps5slop does not start discovery.

## What the guest reads

- `scePadReadState` and `scePadRead` report the current `PadInput` in OpenOrbis's `OrbisPadData` layout, the PS4's (unverified for the PS5). `scePadRead` keeps no history and returns one sample. Touch points and motion read as zero.
- One controller is always connected for user 1, with PS4 touch pad values in `scePadGetControllerInformation`.

## Controller output

The guest's calls only record what it wants in `PadOutput`. The window applies changes to the current controller on its next poll, and applies the whole state again when a different controller appears. Nothing happens without a controller, and the calls still succeed. `scePadClose` and closing the window stop the motors and turn trigger effects off. `RUST_LOG=hle::pad=debug` logs each change with the call that caused it.

Mapping rules:

- **Vibration** (`scePadSetVibration`, large and small motor): Core Haptics engines from `GCController.haptics`, the large motor on the left handle and the small on the right when the controller has both localities, otherwise one engine driven by the stronger motor. Intensity 0..255 maps to 0..1.
- **Light bar** (`scePadSetLightBar`): `GCController.light`. `scePadResetLightBar` sets blue (our choice; a console's colour depends on the user). Until a title sets a colour, the light bar keeps the system's.
- **Trigger effects** (`scePadSetTriggerEffect`): `GCDualSenseAdaptiveTrigger` on a DualSense, ignored on other controllers. Each mode maps to the GameController mode of the same shape; positions 0..9 and strengths 1..8 scale to 0..1. An unknown mode or mask bit fails with invalid argument. How the guest's frequency unit compares to GameController's is a guess.
- Placeholders: `scePadSetVibrationMode`, `scePadGetTriggerEffectState` (no effect engaged), motion and orientation calls.

Interface sources, none checked on a PS5: the vibration and light bar parameter layouts from shadPS4's `pad.h`; the trigger effect layout and mode numbers from SharpEmu-mac's `PadExports.cs` ([references.md](references.md)).

## Core Haptics on macOS

Found with a probe against a Bluetooth DualShock 4 on macOS 27:

- `createEngineWithLocality:` is marked iOS-only in objc2-game-controller but exists on macOS; we call it with `msg_send!`.
- A plain player with one endless continuous event works.
- The advanced player fails: `gamecontrollerd` refuses to decode `AVHapticEvent`, which looks like a macOS bug.
- Changing a playing event's intensity is accepted but does nothing. So each change of a motor's value stops its player and starts a new one at the new intensity; 0 only stops it.

## Verified, and how

- Parameter decoding and scale mapping: unit tests in `pad.rs` and `feedback.rs`.
- Rumble from a title felt on a Bluetooth DualShock 4 (vic), starting and stopping with the game event.
- Pad input reaching titles' own pad state, through the control socket, headless.
- Not checked on hardware: the light bar, DualSense trigger effects, the left and right motor split, DualSense and Xbox controllers, USB connections.
