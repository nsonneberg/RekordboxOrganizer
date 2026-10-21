# Rekordbox Organizer

A small app for Windows and macOS. Tick the boxes for how you want your rekordbox
library organized, press **Run**, and the playlists show up in rekordbox.

It can also listen to your music, score each track's **energy from 1 to 5**, and
set the track's star rating to match.

Your music files are never moved, renamed or edited.

## What it can do

**Your own playlists:** choose how many playlists you want, name each one, and
optionally give it rules:

| Playlist name | Genre has    | BPM     | Energy |
|---------------|--------------|---------|--------|
| House         | house        | 118-126 |        |
| Techno        | techno       | 125-150 |        |
| Hip Hop       | hip hop, rap |         |        |
| Peak Time     |              |         | 4-5    |

The app goes through every song in your library and puts it in the **first** playlist
it matches, working top to bottom. Leave a box empty to match anything. Genre words
match whole words, so "rap" matches "UK Rap" but not "Trap". Songs that match nothing
go into an **Everything Else** playlist, so no song is left out. If a rule uses energy,
the app listens to your music automatically. Your playlist setup is saved between runs.

**Energy:** listens to every track and measures loudness, rhythmic activity,
brightness, hi-hat/percussion content and BPM. Tracks are ranked against the rest
of your library: 1 = your calmest 20%, 5 = your most energetic 20%. You get a
"By Energy" playlist folder (1 - Chill, 2 - Warm-up, 3 - Groove, 4 - Driving,
5 - Peak Time). You can also set each track's star rating to its energy level, and
leave alone tracks you've already rated. Results are cached, so after the first run
only new or changed tracks are analysed.

**Automatic playlists by:** Genre · BPM range (step size you choose) · Key (Camelot, in wheel order) ·
Artist · Record label · Release year · Star rating · Track color · Month added

**Cleanup playlists:** Possible duplicates · Missing files · Missing genre / key / BPM · Never played

Everything goes into one playlist folder (named "Organized" by default). If you run
the app again, it replaces that folder.

## How to use

### Straight from rekordbox (default)

1. **Close rekordbox.**
2. Open Rekordbox Organizer, tick what you want, and press **Run**.
3. Open rekordbox. The new folder is in your playlists, and ratings are updated.

The app reads and writes rekordbox's own library database (rekordbox 6 and 7).
Before changing anything it saves a backup next to the library, named
`master.organizer-backup-<date>.db`. It keeps the 3 most recent backups.

To undo a run: close rekordbox, rename your current `master.db`, and rename the
backup to `master.db`. The library is in `%APPDATA%\Pioneer\rekordbox\` on Windows
and `~/Library/Pioneer/rekordbox/` on macOS.

### From an XML export

If you'd rather not let the app touch your library directly:

1. In rekordbox: **File → Export Collection in xml format**.
2. In the app, choose **From an XML export file**, pick the file, and press **Run**.
   It writes `<name>_organized.xml` next to your export.
3. Back in rekordbox:
   - **Preferences → Advanced → Database → rekordbox xml**: set *Imported Library* to the new file.
   - **Preferences → View → Layout**: turn on **rekordbox xml**.
   - In the sidebar, open **rekordbox xml**, right-click the **Organized** folder, and choose **Import Playlist**.

This route reliably brings in playlists, including "By Energy". It may not
update star ratings on tracks already in your collection, so use direct mode for ratings.

## Audio formats

MP3, WAV, AIFF, FLAC and OGG always work. M4A/AAC/ALAC work when PyAV is available
(it's bundled in the built apps) or when `ffmpeg` is installed. On macOS they also work
through the built-in `afconvert`.

## Running from source

You need Python 3.9 or newer, with Tkinter. The python.org installers include it.

```
python -m venv .venv
.venv\Scripts\activate          # macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py                   # launch the app
python -m unittest -v           # run the tests
```

## Building the double-click apps

```
pip install -r requirements.txt pyinstaller
# Windows -> dist/Rekordbox Organizer.exe
pyinstaller --onefile --windowed --collect-all pyrekordbox --name "Rekordbox Organizer" app.py
# macOS   -> dist/Rekordbox Organizer.app
pyinstaller --windowed --collect-all pyrekordbox --name "Rekordbox Organizer" app.py
```

PyInstaller can only build for the OS it runs on. To get both builds from one place,
push this repo to GitHub: `.github/workflows/build.yml` builds both on every `v*` tag,
or you can start it by hand from the Actions tab.

The macOS build isn't signed. The first time you open it, right-click the app and
choose **Open**.

## Credits

Reading and writing the rekordbox database uses [pyrekordbox](https://github.com/dylanljones/pyrekordbox).
