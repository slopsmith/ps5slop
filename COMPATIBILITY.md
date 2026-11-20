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
| PPSA17221 | Minecraft | 🟢 Playable | v26.10: worlds load in about 25 s and render correctly without Vibrant Visuals; with it, water is broken and it runs slower. Settings' left column is spaced too widely. Long sessions no longer run out of direct memory (it used to stop after about 13 minutes in a world). |
| PPSA03311 | The Binding of Isaac: Repentance | 🟢 Playable | 01.000: plays at about 55 fps. The screen filter setting needed `v_cvt_off_f32_i4` (now translated, not yet rechecked); pad vibration reaches the controller (felt on a DualShock 4, see [docs/input.md](docs/input.md)). Some rooms, including the starting room at times, render black (not yet investigated). |
| PPSA21564 | Astro Bot | ⚫ Nothing | v1.019 (.pkg): loads and starts flipping (with `PS5SLOP_UNIMPLEMENTED_ZERO=1`; otherwise stops at a libSceAudioPropagation call); most compute programs do not translate yet. Needs `scripts/fetch-fonts`. |
| PPSA04263 | Grand Theft Auto V | ⚫ Nothing | EU v1.011.001 (.pkg, 103.6 GB): submits frames, screen black (a pixel program does not decode); stops at an assert in its file streamer. |
| CUSA52463 | Deltarune (PS4) | ⚫ Nothing | v1.12 (patch over base .pkg): game loop runs at about 30 fps; nothing drawn, since PS4 Gnm rendering is out of scope for now. |
| PPSA06084 | Superliminal | ⚫ Nothing | 01.010 (.pkg): loads and runs its frame loop, screen stays black (unsupported image formats, an MSL error). |
