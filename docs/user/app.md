# Launcher app

`ps5slop.app` is a Mac app that shows your games as a library and starts them. It is a front end: every game runs in its own `ps5slop run` process with its own window, exactly as from the terminal ([running.md](running.md)). The app's code is in `apps/macos` (SwiftUI).

## Building

```sh
scripts/bundle-app
open target/ps5slop.app
```

The app needs macOS 26 or later (it uses Liquid Glass). The script needs Xcode (Swift 6.2 or later) and Nix. It builds `ps5slop` in release mode inside `nix develop`, builds the launcher with Xcode's Swift, and puts both in `target/ps5slop.app`. FFmpeg's three libraries (libavcodec, libavformat, libavutil, from the slim build in `flake.nix`) are copied from the Nix store into `Contents/Frameworks`, so the app runs without Nix (about 28 MB in all); the script stops if anything still links into `/nix/store`. Everything is signed ad hoc, `ps5slop` with the hypervisor entitlement. The app is not notarised: on another Mac, right-click it and choose Open the first time.

For work on the app itself, `cd apps/macos && xcrun swift run` starts it without a bundle. It then finds `ps5slop` through `PS5SLOP_BIN` or `target/release/ps5slop`. A plain `cargo build` does not sign that binary with the hypervisor entitlement; `cargo run` once (or `scripts/hv-run.sh target/release/ps5slop --help`) does. `xcrun swift test` runs the launcher's tests.

## Using it

- **Library:** add folders with File > Add Games Folder (⌘O), the + under the sidebar, or by dropping folders or `.pkg` files on the window. A folder can hold `.pkg` files, app folders (with `eboot.bin`) or folders of those. The sidebar filters by folder; right-click one to remove it.
- **Views:** icons or a list (toolbar). Both show the icon, name, title ID and the [compatibility](../../COMPATIBILITY.md) status; the list also has the version and note. Search matches the name or the title ID.
- **Details:** the inspector (toolbar button on the right) shows the selected game's icon, details, compatibility note and Play button.
- **Playing:** double-click a game, press Return on it, or press Play. A bar at the bottom shows the running game and how long it has run. Stop asks the game to quit over its control socket and sends SIGTERM to that process if it does not. One game runs at a time.
- **Log:** the inspector shows the game's output (the last 5000 lines) below its details while it runs and after it exits, with copy and clear buttons.
- **Settings (⌘,):** the data folder (`--data`), the fonts folder (`--fonts`) with a button that downloads the open-source stand-in fonts (`scripts/fetch-fonts`, which needs the network), and Open Captures Folder (also under View).

The app keeps its files in `~/Library/Application Support/ps5slop/launcher`: `library.json` (the folders), `games-cache.json` (the last scan, shown at once on the next start) and `icons/`. Refresh (⌘R) scans again; adding a folder scans too.

Launch options for development: `--fixture FILE` shows a saved `ps5slop info --json` result instead of the library (nothing is scanned or saved), `--select TITLE_ID` selects a game, `--appearance light|dark` and `--window WxH` set the look and size, and `--snapshot FILE.png` saves the window as a PNG once it is drawn and quits. `apps/macos/Tools/screenshots [DIR]` uses them to save preview screenshots of the test library (grid in light and dark, inspector, empty state). The screen must be unlocked: the capture goes through ScreenCaptureKit, and without the window server the fallback drawing misses Liquid Glass, the sidebar and the inspector.

## Not done yet

- Only one game at a time, and no per-game options (patches, passcodes, save states, `--interpreter`).
- No controller or keyboard settings; the game window handles input itself.
- The game window is a separate process, so it may show up as a second app in the Dock (not checked yet).
- No automatic rescans when folders change, and no notarised or universal build.
