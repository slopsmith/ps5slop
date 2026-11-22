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
| PPSA17221 | Minecraft | 🟢 Playable | v26.10: worlds and Vibrant Visuals render (VV about 20 fps, particle and far-water glitches); sound plays. |
| PPSA03311 | The Binding of Isaac: Repentance | 🟢 Playable | 01.000: plays at about 52 fps. Screen filter setting not rechecked. |
| PPSA04263 | Grand Theft Auto V | 🟡 Ingame | EU v1.011.001 (.pkg): prologue plays at ~2 fps; depot, player, NPCs, reflections drawn; lighting dim; no audio. |
| PPSA06084 | Superliminal | 🟡 Ingame | 01.010 (.pkg): menu ~20 s; level 1 passes the rook and looks as on PC (lighting, AO, no glows). |
| PPSA21564 | Astro Bot | 🟠 Menus | v1.019 (.pkg): title at 50-70 s, logo as on PS5, backdrop pink; ship and Astro missing. Needs `scripts/fetch-fonts`. |
| CUSA52463 | Deltarune (PS4) | ⚫ Nothing | v1.12 (patch over base .pkg): game loop runs; nothing drawn, PS4 Gnm rendering is out of scope. |
