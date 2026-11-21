<!--
Rules for this file:
- One row per title, sorted by status (best first), then by name.
- Status is one of the six below, with its emoji.
- Notes stay short: one line, about 120 characters at most. Say what works or where it stops, and the version tested.
  Longer findings go in the docs (gpu.md, hle-kernel.md, ...), not here.
- Update a row whenever a title's status or its main blocker changes.
-->

# Compatibility

What each title does under ps5slop, from our own runs on an Apple Silicon Mac.

| Status | Meaning |
| --- | --- |
| ⭐ Perfect | Plays through with no known problems. |
| 🟢 Playable | Can be played start to end; may have glitches or slowdowns. |
| 🟡 Ingame | Gets into gameplay, but problems keep it from being played properly. |
| 🟠 Menus | Reaches the title screen or menus, not gameplay. |
| 🔴 Boots | Starts and shows something (a logo, a loading screen), no menus. |
| ⚫ Nothing | Stops or crashes before showing anything. |

| Title ID | Name | Status | Notes |
| --- | --- | --- | --- |
| PPSA21402 | Balatro | ⭐ Perfect | 01.015.000: plays normally with sound. |
| PPSA17221 | Minecraft | 🟢 Playable | v26.10: worlds load in about 25 s and render correctly. Vibrant Visuals: water and sky fixed, about 20 fps (24 without); particles above water are solid squares and far water darkens toward the horizon. Settings' left column has gaps between buttons because a game feature toggle is on while offline (not an emulator bug). Long sessions no longer run out of direct memory (it used to stop after about 13 minutes in a world). Menu music and sounds play. |
| PPSA03311 | The Binding of Isaac: Repentance | 🟢 Playable | 01.000: plays at about 55 fps. The screen filter setting needed `v_cvt_off_f32_i4` (now translated, not yet rechecked); pad vibration reaches the controller (felt on a DualShock 4, see [docs/input.md](docs/input.md)). Some rooms, including the starting room at times, render black (not yet investigated). |
| PPSA06084 | Superliminal | 🟡 Ingame | 01.010 (.pkg): menu after ~50 s, lit like the loading picture; New Game reaches level 1. A new build stalls ~30 s compiling shaders; movies skipped. |
| PPSA21564 | Astro Bot | ⚫ Nothing | v1.019 (.pkg): plays its logo movie (as black) and starts the title level after ~95 s; screen black, many shaders untranslated. Needs `scripts/fetch-fonts`. |
| PPSA04263 | Grand Theft Auto V | 🟠 Menus | EU v1.011.001 (.pkg, 103.6 GB): title screen after about 5 min, landing page and menus work; loading Story Mode stalls on a GPU fence. Game audio silent. |
| CUSA52463 | Deltarune (PS4) | ⚫ Nothing | v1.12 (patch over base .pkg): game loop runs at about 30 fps; nothing drawn, since PS4 Gnm rendering is out of scope for now. |
