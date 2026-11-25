# Launcher app

`ps5slop.app` is a SwiftUI front end in `apps/macos`. It shows a game library and starts each game as its own `ps5slop run` child process, with its own window, exactly as from the terminal. It stops a game over the control socket, then with SIGTERM. How players use it is in [user/getting-started.md](user/getting-started.md).

## Building

```sh
scripts/bundle-app
open target/ps5slop.app
```

The app needs macOS 26 or later (Liquid Glass). The script needs Xcode (Swift 6.2 or later) and Nix. It builds `ps5slop` in release mode inside `nix develop`, builds the launcher with Xcode's Swift and assembles `target/ps5slop.app`. The shared steps come from `slop-app` in [slopsmith/ci](https://github.com/slopsmith/ci), pinned in `flake.lock` (`nix flake update slopsmith-ci` moves it). FFmpeg's libraries from the slim build in `flake.nix` are copied into `Contents/Frameworks`, so the app runs without Nix. Everything is signed ad hoc, `ps5slop` with the hypervisor entitlement. The script ends with the same upload checks the nightly workflow runs (`slop-app check`).

Nightlies are built at 03:00 UTC when `trunk` changed, and replace the single `nightly` release.

Every build is stamped with the commit it was built from: `SLOP_COMMIT` if set (the nightly workflow sets it), else `git rev-parse HEAD`. `scripts/bundle-app` adds `-dirty` when tracked files differ from that commit, and a plain `cargo build` adds `-dev`.

Licenses of everything bundled live in `apps/macos/Licenses` and go into `Contents/Resources/Licenses`. A new bundled library needs its license text there and a line in its `README.txt`.

## Working on the app

`cd apps/macos && xcrun swift run` starts it without a bundle. It then finds `ps5slop` through `PS5SLOP_BIN` or `target/release/ps5slop`. A plain `cargo build` does not sign that binary with the hypervisor entitlement; `cargo run` once (or `scripts/hv-run.sh target/release/ps5slop --help`) does. `xcrun swift test` runs the launcher's tests.

`DevOptions.swift` holds launch options for development: a fixture instead of the library, a selected game, appearance and window size, the About window or the report sheet alone, and `--snapshot FILE.png`, which saves the window once drawn and quits. `apps/macos/Tools/screenshots` uses them for preview screenshots; the screen must be unlocked, since the capture goes through ScreenCaptureKit.

## Compatibility data

The app reads `https://slopsmith.github.io/ps5slop/compat.json` at start and on Refresh, and keeps the last copy for offline starts. A game shows the newest report from a Mac with the same chip, else the newest report. Reports are filed as prefilled issues in [slopsmith/ps5slop-compat](https://github.com/slopsmith/ps5slop-compat); the website builds `compat.json` from the accepted ones.

## Files

The app keeps its own files in `~/Library/Application Support/ps5slop/launcher`: the library folders, the last scan (shown at once on the next start), the last `compat.json` and the game icons.
