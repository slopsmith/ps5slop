# Running games

You supply your own games, dumped from your own console ([legal.md](../legal.md)). The [compatibility list](https://slopsmith.github.io/ps5slop/compat/) says how far each title gets; reports go to [slopsmith/ps5slop-compat](https://github.com/slopsmith/ps5slop-compat).

## Building

Run everything inside `nix develop`, which provides the toolchain. `cargo run` signs the binary with the hypervisor entitlement the JIT needs. Astro Bot also needs `scripts/fetch-fonts` once, which downloads open-source stand-ins for the system fonts.

## Starting a game

```sh
cargo run --release -p ps5slop -- run GAME.pkg           # a package, read in place
cargo run --release -p ps5slop -- run games/TITLE        # an extracted app folder holding eboot.bin
cargo run --release -p ps5slop -- run --headless GAME.pkg   # no window
cargo run --release -p ps5slop -- run --load-state X.state GAME.pkg   # go on from a save state
```

`ps5slop run --help` lists every option. The common ones:

| Option | What it does |
| --- | --- |
| `--headless` | No window: nothing is shown and no controller is read |
| `--data DIR` | The folder the game sees as `/data`, with its saves next to it (default: one per title in `~/Library/Application Support/ps5slop`) |
| `--load-state STATE` | Starts from a save state instead of the title's start. It must be the same game, and the same `--data` as when it was saved |
| `--patch PKG` | A PS4 patch package to lay over the base package |
| `--passcode X` | A fake-signed PS4 package's passcode (all zeros by default) |
| `--interpreter` | Runs the game's code in the interpreter instead of the JIT (very slow, for comparing) |
| `--dump-frame PATH --after SECONDS` | Writes the screen to a PPM file after that many seconds and stops |

`ps5slop info FOLDER...` lists the games it finds in `.pkg` files and app folders (name, title id, version, compatibility status; `--json` and `--icons DIR` for launchers).

Closing the window ends the game. The window title shows the game, its frame rate (flips a second), frame size, the instruction rate in the JIT and the interpreter, and which controller is in use.

## Controls

Any controller macOS knows works as the pad: DualSense, DualShock 4, Xbox and other MFi pads, over USB or Bluetooth (pair it in System Settings first). It can be connected while a game runs. Vibration works, the light bar on DualSense and DualShock 4, trigger effects on DualSense ([input.md](../input.md)).

The keyboard works too, together with a controller:

| Key | Button |
| --- | --- |
| Arrow keys | D-pad |
| X / C / Z / V | Cross / Circle / Square / Triangle |
| Q / E | L1 / R1 |
| 1 / 3 | L2 / R2 |
| Enter | Options |
| T | Touch pad button |
| F9 | Capture the next GPU frame (`.ps5frame`) |
| F10 | Capture a Metal `.gputrace` (start with `METAL_CAPTURE_ENABLED=1`) |

## Save states

A save state stores the whole running game in a file, so a later run can start from that exact moment. They are for debugging, not a replacement for the game's own saves. Save with the control socket: every run prints its socket path at start, and

```sh
echo 'save /Users/me/states/gta-guard.state' | nc -U SOCKET   # SOCKET: the path printed at start
```

writes one while the game keeps running. Load it with `--load-state` (above). States break when the ps5slop build changes the host-function layout; details are in [savestates.md](../savestates.md).

## Reporting a problem

- **Problem report:** start with `PS5SLOP_REPORT=report.txt` to save the end-of-run summary of skipped draws, missing functions and JIT fallbacks.
- **Visual bugs:** press F9 at the broken spot. Captures go to `~/Library/Application Support/ps5slop/captures/` (or `PS5SLOP_CAPTURE_DIR`), and the path is printed. Press it several times for several captures. Turn one into a PNG with:

  ```sh
  cargo run --release -p gpu --bin ps5slop-replay -- render CAPTURE.ps5frame final.png
  ```

  `ps5slop-replay --help` lists the other commands ([debugging.md](../debugging.md#gpu-frame-capture-and-replay)).
- **Hard-to-reach spots:** save a state there, so the bug can be reproduced without playing.

Captures, states and reports contain game data. Keep them out of the repository.

## Environment variables

All of them are listed in [environment.md](environment.md).
