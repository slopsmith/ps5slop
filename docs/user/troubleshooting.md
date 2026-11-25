# Troubleshooting

## The app does not open

macOS blocks apps that Apple has not notarised. Open the app once, then go to **System Settings > Privacy & Security** and click **Open Anyway** next to the message about ps5slop.

If it still does not open, check that your Mac has an Apple Silicon chip (**Apple menu > About This Mac** shows "Chip: Apple M…") and runs macOS 26 or newer.

## A game does not show up in the library

- Check that the folder holds a `.pkg` file or a folder with `eboot.bin` in it. The folder you added can also hold folders of those.
- Retail packages from the PlayStation Store are encrypted, and ps5slop cannot read them. Neither can it read PS4 games yet.
- Choose **Refresh** (⌘R) after you add games to a folder.

## A game does not start, or quits at once

Look at the **Output** in the launcher's inspector. The last lines usually say what stopped the game. Many games do not start yet: check the [compatibility list](https://slopsmith.github.io/ps5slop/compat/) for your game. If it is not there, or a report says it works further than it does for you, please [report it](reporting.md).

## The screen stays black, or parts of the picture are missing

- Wait a little. In the first minutes ps5slop is still preparing the game's graphics, and some things appear late.
- If text is missing, download the fonts: **ps5slop > Settings > Download Fonts**.
- If things stay missing, it is a bug in ps5slop. [Report it](reporting.md) and say where in the game it happens.

## The game is slow

- Try the game's Performance graphics mode, if it has one.
- Close other heavy apps.
- Large 3D games are often slow in ps5slop for now. This is expected at this stage.

## No sound

Check that macOS plays sound through the device you expect (Control Center > Sound). Some games' sound does not work yet; say so in a [report](reporting.md).

## The controller does nothing

- Click the game window, so it has focus.
- Check that macOS sees the controller: **System Settings > Game Controllers** lists the connected ones.
- Only the first connected controller is used. Disconnect the others.

## My saves are gone

Saves are kept per game in `~/Library/Application Support/ps5slop/TITLE_ID`. If you set a **Data folder** in Settings, the games used that folder instead. Clear the setting to go back to the per-game folders. See [Playing games](playing.md#saves).
