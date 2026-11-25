# Reporting problems

Reports tell us which games work and what to fix next. Every report is a public issue on GitHub, so you need a GitHub account.

## Report how a game runs

In the launcher, right-click a game and choose **Report Compatibility…**, or select the game and press the report button in the inspector on the right.

The launcher asks up to six yes or no questions, one at a time: does the game show anything, does it reach a menu, does it reach gameplay, does it keep going without stopping you, does it show any problems, and did you play all of it. Each question says what counts as a yes. At the end you see the result your answers add up to, and you can change any answer and add notes.

**Open on GitHub** opens a new issue with your answers, your notes, the game's ID and version, your Mac's chip and macOS version, and the ps5slop build already filled in. Check it and submit it. Reports with every field filled in are accepted automatically and show up on the [compatibility list](https://slopsmith.github.io/ps5slop/compat/) after the site updates.

Report the same game again when a newer build changes how it runs. The newest report counts.

You can also file a report without the launcher, with the form at [slopsmith/ps5slop-compat](https://github.com/slopsmith/ps5slop-compat/issues/new/choose).

## Good notes

The notes are the most useful part of a report. Say:

- **What you did:** for example "started a new game, the crash happens when the first cutscene ends".
- **What went wrong:** for example "characters' faces are missing", "sound crackles in menus", "the game freezes on the loading screen".
- **How often:** every time, or only sometimes.
- **How fast it runs**, if you know it. The game window's title shows the frame rate.

## The game's output

The launcher's inspector shows what ps5slop printed while the game ran, under **Output**. When a game quits, the output ends with a short problem report. If a game crashes or stops, click **Copy Output** and paste the last part of it into your notes, starting from the error.

## Frame captures

For a graphics bug, F9 in the game window saves a frame capture to the captures folder (**View > Open Captures Folder** in the launcher). A capture lets a developer replay that frame and find what drew wrong.

A capture holds parts of the game's own data. Do not attach it to a public issue. Keep it, mention in your notes that you have one, and wait until a developer asks for it.
