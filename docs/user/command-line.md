# Command line

You do not need the command line to play. The launcher does everything here. This page is for people who want to start games from Terminal or a script.

The `ps5slop` program is inside the app. To call it as `ps5slop`, add an alias to your shell, for example in `~/.zshrc`:

```sh
alias ps5slop=/Applications/ps5slop.app/Contents/MacOS/ps5slop
```

## Start a game

```sh
ps5slop run GAME.pkg        # a package
ps5slop run GAME_FOLDER     # an extracted game folder with eboot.bin in it
```

Useful options:

| Option | What it does |
| --- | --- |
| `--data FOLDER` | Keep the game's saves and settings in `FOLDER` instead of its usual folder |
| `--fonts FOLDER` | Where the downloaded stand-in fonts are (the launcher's **Download Fonts** puts them in `~/Library/Application Support/ps5slop/fonts`, which is also the default here) |
| `--headless` | Run without a window. Nothing is shown and no controller is read |

`ps5slop run --help` lists every option. Most of the others are for developers.

## List games

```sh
ps5slop info FOLDER
```

prints the name, ID, version and compatibility status of every game in `FOLDER`. `--json` prints the same as JSON.

## Save the problem report

```sh
PS5SLOP_REPORT=report.txt ps5slop run GAME.pkg
```

writes the problem report that ends the game's output to `report.txt` as well.

## Version

`ps5slop --version` prints the build, which names the exact ps5slop version. Put it in reports you file by hand.
