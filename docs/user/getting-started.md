# Getting started

ps5slop runs PS5 games on Macs with Apple Silicon. It is early software: many games do not start yet, and the ones that do often have bugs or run slowly. The [compatibility list](https://slopsmith.github.io/ps5slop/compat/) shows how far each game gets.

## What you need

- A Mac with an Apple Silicon chip (M1 or newer). Intel Macs are not supported.
- macOS 26 or newer.
- Your own games, dumped from your own console. ps5slop does not come with any games, and it cannot download or buy them for you.

## Install

1. Download the newest build: [ps5slop-macos-arm64.zip](https://github.com/slopsmith/ps5slop/releases/download/nightly/ps5slop-macos-arm64.zip). A new build comes out most days.
2. Open the zip and move `ps5slop.app` to your Applications folder.
3. Open the app. The first time, macOS says it cannot check the app for malware, because the app is not notarised by Apple. Go to **System Settings > Privacy & Security**, scroll down and click **Open Anyway** next to the message about ps5slop. You only do this once.

## Add your games

ps5slop can open two kinds of game:

- **Fake-signed packages** (`.pkg` files made by dumping tools, often called FPKG).
- **Extracted game folders**: a folder with an `eboot.bin` file in it.

Retail packages from the PlayStation Store are encrypted and cannot be opened. PS4 games are not supported yet.

To add games, choose **File > Add Games Folder** (⌘O) and pick the folder your games are in, or drop folders and `.pkg` files on the window. A folder can hold packages, game folders, or folders of those. ps5slop looks through it and shows every game it finds. Choose **Refresh** (⌘R) after you add more games to a folder.

## Play

Double-click a game, or select it and press **Play**. The game opens in its own window. Only one game can run at a time.

To quit a game, close its window, press ⌘Q in it, or press **Stop** in the launcher.

The first minutes of a game can stutter while ps5slop prepares the game's graphics for your Mac. This gets better the next time you play.

See [Playing games](playing.md) for controls and where your saves are kept.

## Fonts

Some games, such as Astro Bot, show no text without the console's system fonts. ps5slop cannot include those, but it can download free fonts that look similar. Open **ps5slop > Settings** (⌘,) and click **Download Fonts**. This needs an internet connection and only has to be done once.

## Updates

The app checks for a newer build when it starts and offers to download it. You can also choose **ps5slop > Check for Updates…**. Updates are installed by hand: download the new zip and replace the app in Applications. Your games, saves and settings are kept.

## If something goes wrong

See [Troubleshooting](troubleshooting.md). To tell us how a game runs, see [Reporting problems](reporting.md).
