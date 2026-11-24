# Launcher app

`ps5slop.app` is a Mac app that shows your games as a library and starts them. It is a front end: every game runs in its own `ps5slop run` process with its own window, exactly as from the terminal ([running.md](running.md)). The app's code is in `apps/macos` (SwiftUI).

## Downloading

Nightly builds are at https://github.com/slopsmith/ps5slop/releases/tag/nightly, and https://github.com/slopsmith/ps5slop/releases/download/nightly/ps5slop-macos-arm64.zip always points at the newest one. A nightly is built at 03:00 UTC when `trunk` has changed since the last one; the release is replaced each time, so older builds are not kept. Unzip it and move `ps5slop.app` to Applications. The app is signed ad hoc, not notarised: the first time, open it, then go to System Settings > Privacy & Security and choose Open Anyway.

## Building

```sh
scripts/bundle-app
open target/ps5slop.app
```

The app needs macOS 26 or later (it uses Liquid Glass). The script needs Xcode (Swift 6.2 or later) and Nix. It builds `ps5slop` in release mode inside `nix develop`, builds the launcher with Xcode's Swift, and puts both in `target/ps5slop.app`. The shared steps come from `slop-app` in [slopsmith/ci](https://github.com/slopsmith/ci), pinned in `flake.lock` (`nix flake update slopsmith-ci` moves it). FFmpeg's three libraries (libavcodec, libavformat, libavutil, from the slim build in `flake.nix`) are copied from the Nix store into `Contents/Frameworks`, so the app runs without Nix (about 28 MB in all). Everything is signed ad hoc, `ps5slop` with the hypervisor entitlement.

The script ends with the upload checks the nightly workflow also runs (`slop-app check`): every binary is arm64 and nothing links into `/nix/store`, the signatures verify and `ps5slop` has the hypervisor entitlement, the bundled `ps5slop --version` runs and names the same commit as the app, and every bundled library is listed in the licenses.

Every build is stamped with the commit it was built from: `SLOP_COMMIT` if set (the nightly workflow sets it), else `git rev-parse HEAD`. `scripts/bundle-app` adds `-dirty` when tracked files differ from that commit, and a plain `cargo build` adds `-dev`. `ps5slop --version`, the first log line, Info.plist's `SlopCommit` and the About window show it.

The licenses in `apps/macos/Licenses` (and ps5slop's own `LICENSE` and the Rust crates' notices, which `cargo about` writes) go into `Contents/Resources/Licenses`. A new bundled library needs its license text there and a line in its `README.txt`.

For work on the app itself, `cd apps/macos && xcrun swift run` starts it without a bundle. It then finds `ps5slop` through `PS5SLOP_BIN` or `target/release/ps5slop`. A plain `cargo build` does not sign that binary with the hypervisor entitlement; `cargo run` once (or `scripts/hv-run.sh target/release/ps5slop --help`) does. `xcrun swift test` runs the launcher's tests.

## Using it

- **Library:** add folders with File > Add Games Folder (⌘O), the + under the sidebar, or by dropping folders or `.pkg` files on the window. A folder can hold `.pkg` files, app folders (with `eboot.bin`) or folders of those. The sidebar filters by folder; right-click one to remove it.
- **Views:** icons or a list (toolbar). Both show the icon, name, title ID and the [compatibility](https://slopsmith.github.io/ps5slop/compat/) status; the list also has the version and note. Search matches the name or the title ID.
- **Compatibility:** the app downloads `https://slopsmith.github.io/ps5slop/compat.json` when it starts and on Refresh, and keeps the last copy for offline starts. A game shows the newest report from a Mac with the same chip, or else the newest report, and the note says which Mac and ps5slop commit it came from. Games without a report show as untested.
- **Details:** the inspector (toolbar button on the right) shows the selected game's icon, details, compatibility note and Play button.
- **Playing:** double-click a game, press Return on it, or press Play. A bar at the bottom shows the running game and how long it has run. Stop asks the game to quit over its control socket and sends SIGTERM to that process if it does not. One game runs at a time.
- **Log:** the inspector shows the game's output (the last 5000 lines) below its details while it runs and after it exits, with copy and clear buttons.
- **About and updates:** ps5slop > About ps5slop shows the version, the commit (linked on GitHub, with a copy button) and the licenses of everything in the app. A nightly checks for a newer nightly when it starts and offers to download it; ps5slop > Check for Updates… checks at once. Builds made locally never check. The launch check can be turned off in Settings.
- **Settings (⌘,):** the data folder (`--data`), the fonts folder (`--fonts`) with a button that downloads the open-source stand-in fonts (`scripts/fetch-fonts`, which needs the network), and Open Captures Folder (also under View).

The app keeps its files in `~/Library/Application Support/ps5slop/launcher`: `library.json` (the folders), `games-cache.json` (the last scan, shown at once on the next start), `compat.json` (the last compatibility list) and `icons/`. Refresh (⌘R) scans again; adding a folder scans too.

Launch options for development: `--fixture FILE` shows a saved `ps5slop info --json` result instead of the library (nothing is scanned or saved), `--select TITLE_ID` selects a game, `--appearance light|dark` and `--window WxH` set the look and size, `--about` opens only the About window, and `--snapshot FILE.png` saves the window as a PNG once it is drawn and quits. `apps/macos/Tools/screenshots [DIR]` uses them to save preview screenshots of the test library (grid in light and dark, inspector, empty state). The screen must be unlocked: the capture goes through ScreenCaptureKit, and without the window server the fallback drawing misses Liquid Glass, the sidebar and the inspector.

## Not done yet

- Only one game at a time, and no per-game options (patches, passcodes, save states, `--interpreter`).
- No controller or keyboard settings; the game window handles input itself.
- The game window is a separate process, so it may show up as a second app in the Dock (not checked yet).
- No automatic rescans when folders change, and no notarised or universal build.
- Updates are downloaded in the browser and installed by hand.
