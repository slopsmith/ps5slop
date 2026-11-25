# Playing games

## Controllers

Any game controller your Mac knows works: DualSense, DualShock 4, Xbox controllers and other Mac-compatible pads, over USB or Bluetooth. Pair a Bluetooth controller in **System Settings > Bluetooth** first. You can connect a controller while a game is running.

On a DualSense or DualShock 4, vibration and the light bar work. On a DualSense, the adaptive triggers work too. The touch pad works as a button, but touch movement and motion controls do not work yet.

## Keyboard

The keyboard works as a controller, and you can use it together with one. Click the game window first so it gets your key presses.

| Key | PS5 button |
| --- | --- |
| Arrow keys | D-pad |
| X | Cross |
| C | Circle |
| Z | Square |
| V | Triangle |
| Q / E | L1 / R1 |
| 1 / 3 | L2 / R2 |
| Enter | Options |
| T | Touch pad button |

The keyboard cannot move the sticks or press L3 and R3, so most 3D games need a controller.

## The game window

The window's title shows the game's name, its frame rate and the controller in use. Close the window or press ⌘Q to quit the game.

**F9** saves a picture of the game's graphics work for one frame (a frame capture). It is useful when we ask you for one to fix a graphics bug. See [Reporting problems](reporting.md).

## Saves

Games save the same way they do on a console, from their own menus. Each game keeps its saves and settings in its own folder:

```
~/Library/Application Support/ps5slop/TITLE_ID
```

`TITLE_ID` is the game's ID, for example `PPSA04263`. The launcher shows it under the game's name. To back up your saves, copy that folder. To start a game fresh, move it somewhere else.

**ps5slop > Settings** has a **Data folder** setting. Leave it empty. If you set it, every game uses that one folder, and games can mix up each other's files.

## Sound

Games play through your Mac's current sound output. Change the output in macOS as usual, for example in Control Center.

## Speed

How fast a game runs depends on the game and on your Mac. Small games often run at full speed. Large 3D games may run far below their speed on a PS5. Many games have a graphics mode setting, often Fidelity and Performance, and the Performance mode is usually faster in ps5slop too.
