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
| PPSA17221 | Minecraft | 🟡 Ingame | v26.10: menus work, a world loads in about 25 s and renders at 10 to 16 fps, but block textures have wrong colours and some menu letters go missing. |
| PPSA03311 | The Binding of Isaac: Repentance | 🟢 Playable | 01.000: plays at about 55 fps. The screen filter setting needed `v_cvt_off_f32_i4` (now translated, not yet rechecked); pad vibration is a stub. |
| PPSA21564 | Astro Bot | ⚫ Nothing | v1.019 (.pkg): loads and starts flipping (with `PS5SLOP_UNIMPLEMENTED_ZERO=1`; otherwise stops at a libSceAudioPropagation call); most compute programs do not translate yet. Needs `scripts/fetch-fonts`. |
| CUSA52463 | Deltarune (PS4) | ⚫ Nothing | v1.12 (patch over base .pkg): game loop runs at about 30 fps; nothing drawn yet (no Gnm renderer). |
| PPSA06084 | Superliminal | ⚫ Nothing | 01.010 (.pkg): loads and runs its frame loop, screen stays black (unsupported image formats, an MSL error). |
