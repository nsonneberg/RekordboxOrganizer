# Rekordbox Organizer

A small app for Windows and macOS. Tick the boxes for how you want your rekordbox
library organized, press **Run**, and the playlists show up in rekordbox.

It can also listen to your music, score each track's **energy from 1 to 5**, and
set the track's star rating to match.

Your music files are never moved, renamed or edited.

## Installing

You don't need Python or anything else. The app is a single download.

### Get the download

Download the newest version for your computer:

- **Windows 10 / 11:** [Rekordbox-Organizer-Windows.exe](../../releases/latest/download/Rekordbox-Organizer-Windows.exe)
- **Mac with Apple Silicon (M1 or newer):** [Rekordbox-Organizer-macOS.zip](../../releases/latest/download/Rekordbox-Organizer-macOS.zip)

All versions are on the [Releases page](../../releases) (also linked on the right of
the repository's front page).

### Windows

1. Move `Rekordbox-Organizer-Windows.exe` from **Downloads** somewhere permanent, e.g.
   your **Documents** folder. You can rename it to `Rekordbox Organizer.exe`.
2. If your browser warns that the file "isn't commonly downloaded", choose **Keep**
   (in Edge: **…** → **Keep** → **Keep anyway**).
3. Double-click it. The first time, Windows may show *"Windows protected your PC"*.
   Click **More info → Run anyway**. It takes a few seconds to start.
4. Optional: to add it to your desktop, right-click the `.exe` and choose
   **Show more options → Send to → Desktop (create shortcut)**.

If **Smart App Control** blocks it outright and offers no "Run anyway" button, use
[Running from source](#running-from-source) instead. Smart App Control blocks apps
that aren't signed by a known publisher.

### macOS

1. Double-click `Rekordbox-Organizer-macOS.zip` in **Downloads** to get
   **Rekordbox Organizer.app**. Safari may already have unzipped it for you.
2. Drag **Rekordbox Organizer.app** into your **Applications** folder.
3. The first time, open it like this. The app isn't signed by Apple, so double-clicking
   only shows a warning.
   - **macOS 14 Sonoma or older:** right-click (or Control-click) the app → **Open** →
     **Open**.
   - **macOS 15 Sequoia or newer:** double-click the app and click **Done** on the
     warning. Then go to **System Settings → Privacy & Security**, scroll down, click
     **Open Anyway** next to "Rekordbox Organizer", and confirm with your password.
4. After that, it opens normally from Applications or Launchpad. If macOS asks for
   access to your Music or other folders, click **Allow**. The app needs it to read
   your tracks.

If macOS says the app *"is damaged and can't be opened"*, open **Terminal** and run
this, then try again:

```
xattr -dr com.apple.quarantine "/Applications/Rekordbox Organizer.app"
```

**Intel Macs:** the download only runs on Apple Silicon. Use
[Running from source](#running-from-source) instead.

### Updating

You only download the app once. After that it updates itself:

1. When the app opens, it checks GitHub for a newer version. This takes a second and
   happens in the background.
2. If there is one, a green **"Version X is available"** line appears under the title,
   with an **Update now** button.
3. Click **Update now** and confirm. The app shows what's new, downloads the new
   version, replaces itself and restarts. Your settings and playlists are kept.

The version you have is shown in the top-right corner of the window. If you're offline,
the check is skipped and tried again next time. If an update fails, the app offers to
open the download page so you can update by hand.

On a Mac, the app must be in your **Applications** folder to update itself.

**Older versions:** 1.0.0 and 1.1.0 can't update themselves (1.1.0's button stays
hidden), so download the newest version once using the links above. Updating *from*
1.1.1 works, but the app may show "Failed to load Python DLL" instead of restarting.
Click OK and open the app again: it's already updated. Every later version restarts
by itself.

**Running from source?** Update with `git pull` in the app's folder, or download the
ZIP again.

### Uninstalling

Delete the `.exe` or `.app`. To also remove its settings and energy cache, delete
`%APPDATA%\RekordboxOrganizer` on Windows or
`~/Library/Application Support/RekordboxOrganizer` on macOS. Your rekordbox library
and the backups the app made are left alone.

## What it can do

**Your own playlists:** choose how many playlists you want, name each one, and
give it rules:

| Playlist name | Genre has    | BPM     | Energy |
|---------------|--------------|---------|--------|
| House         | house        | 118-130 |        |
| Techno        | techno       | 130-145 |        |
| Hip Hop       | hip hop      |         |        |

- A song is added only if it matches **every** rule you fill in. A Techno song at 128 BPM
  stays out of a `130-145` Techno playlist. An empty box means "any".
- Genre matching is relaxed. `house` also catches Progressive House, Tech-House and
  Deephouse; `hip hop` catches Hip-Hop/Rap, Rap, Trap and Drill; `dnb` catches Drum & Bass.
  Short words only match whole words, so `rap` never catches "Trap".
- If a song fits more than one playlist, it goes to the best genre match.
  "Melodic House & Techno" goes to Techno, because the main genre is usually the last word.
  Rows with a genre win over rows without one. On a tie, the playlist higher in the list wins.
- Songs that don't fit any playlist are left alone. You can tick a box to also collect
  them in an **Everything Else** playlist.
- Your playlists go straight into the main folder (named "Organized" by default), and
  your setup is saved between runs.

**Genre guessing:** rekordbox's analysis doesn't include genre. It measures BPM, key and
a colour waveform (loudness plus bass, mid and treble). For songs with no genre tag, or a
vague one like "Electronic" or "Dance & EDM", the app estimates the genre from those
measurements. A song's own tag still comes first: if one of your playlists names it
(say an "EDM" playlist for songs tagged "Dance & EDM"), the song goes there rather
than where its guess points. The estimate works like this:

1. **Learned from your library.** Songs with a clear genre tag are used as examples.
   Each untagged song gets the genre of the tagged songs it sounds most like (BPM,
   bass, brightness, punch, loudness). This needs at least 6 tagged songs across 2 or
   more genres. The more you tag, the better it gets.
2. **BPM and sound rules** otherwise, for example 160–185 BPM → Drum & Bass and
   118–135 → House. In the 136–155 range it only names Dubstep, Techno or Trance on a
   clear signal; otherwise it says EDM.

Your own playlists use these guesses, so a "Dance & EDM" track at 125 BPM can land in
your House playlist. Turn guessing off to match on tags only.

**Auto genre detection:** makes an **Auto Genres** folder with a folder for every genre
family found in your library (House, Techno, Drum & Bass, Hip Hop…). Tags that fit no
known family get their own folder. Every song is sorted into one family:

```
Organized
└─ Auto Genres
   ├─ House
   │  ├─ All House
   │  ├─ House
   │  ├─ Progressive House
   │  └─ House (detected)      ← guessed songs, kept apart so you can check them
   ├─ Drum & Bass
   └─ Unknown Genre            ← no tag and no analysis to guess from
```

**Hot cues at phrase changes:** puts hot cues where each section of a track starts
(Intro, Build-up, Drop, Breakdown, Chorus, Verse, Bridge, Outro), each named after
its section. The positions come from rekordbox's own phrase analysis, so they land on
the beat. Only empty pads (A–H) are used. Hot cues you set yourself are never moved
or replaced, and running again doesn't add duplicates. If a track has more sections
than empty pads, drops and build-ups win. Works straight from rekordbox only, not with
an XML export. Tracks need phrase analysis, which is on by default; otherwise turn on
**Preferences → Analysis → Phrase** in rekordbox and re-analyse them.

**Energy:** judges each track by its drops and choruses (found with rekordbox's phrase
analysis), not by intros and breakdowns. Without phrase analysis, the loudest stretches
are used. It measures:

- how loud the main sections are
- how punchy the beat is
- how much of the track runs at full power
- bass weight and hi-hats
- tempo (half-speed readings of Drum & Bass, e.g. 87 instead of 174 BPM, are doubled)
- rekordbox's own mood rating

Working straight from rekordbox, it reads the colour waveform rekordbox made when it
analysed the track, so it's fast and works for every format. With an XML export, it
listens to the audio files instead.

Tracks are ranked against the rest of your library. One unusual track can't skew the
others, and copies of the same song get the same level: 1 = your calmest 20%,
5 = your most energetic 20%. You get a "By Energy" playlist folder (1 - Chill,
2 - Warm-up, 3 - Groove, 4 - Driving, 5 - Peak Time).

**Star ratings:** each track's stars can be set to its energy level, so your
highest-energy songs get 5 stars and your lowest get 1. The full range is always used,
even in a small library. Tick "Keep ratings I've already set" to leave songs you've
rated yourself alone. Measurements are cached, so after the first run only new or
changed tracks are analysed.

The app only sees measurements. A track mastered quieter than the rest can come out
lower than it feels. Rate it yourself and keep your ratings if so.

**Extra playlists (optional, all off by default):** Genre · BPM range (step size you choose) · Key (Camelot, in wheel order) ·
Artist · Record label · Release year · Star rating · Track color · Month added

**Cleanup playlists:** Possible duplicates · Missing files · Missing genre / key / BPM · Never played

Everything goes into one playlist folder (named "Organized" by default). If you run
the app again, it **adds to** folders and playlists that already have the same name
(ignoring capitals): new songs go at the end, and songs already there aren't added
twice. Untick "Add to existing folders / playlists" to replace the folder instead.

**Songs move when they fit better:** when you re-run, songs already in your playlists
are checked again. If a song now fits a *different* one of your playlists best (for
example, you added an "EDM" playlist and it's tagged EDM), it's moved there: taken out
of the old playlist and added to the new one. The log lists every move. Songs that
still fit, or that fit none of your playlists (including ones you put somewhere by
hand), stay where they are. Only your own playlists in the main folder are affected;
a playlist you've removed from your list is left alone. Untick "move songs to the
playlist they now fit best" in *My playlists* to turn this off.

**Only new songs:** under "Songs to organize" pick *All*, *Added since last run* (the
app remembers the date of each successful run), or *Added from … to …* (either date
can be left empty). Only those songs are sorted, rated and added; the rest of the
library is still read for comparison, so energy levels and genre guesses stay
consistent with earlier runs.

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

Use this if the download is blocked (e.g. by Windows Smart App Control) or you
have an Intel Mac.

1. Install Python 3.9 or newer from [python.org](https://www.python.org/downloads/).
   Its installers include Tkinter, which the app needs. On Windows, tick
   **Add python.exe to PATH** in the installer.
2. Download this repository: green **Code** button → **Download ZIP**, then unzip it.
3. Open a terminal in the unzipped folder (Windows: right-click the folder →
   **Open in Terminal**; macOS: open **Terminal**, type `cd `, drag the folder in, and
   press Return). Then run:

**Windows**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**macOS**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

To start it again later, open a terminal in the folder and run only the `activate`
line and `python app.py`. To run the tests: `python -m unittest -v`.

## Building the double-click apps

```
pip install -r requirements.txt pyinstaller
# Windows -> dist/Rekordbox Organizer.exe
pyinstaller --onefile --windowed --collect-all pyrekordbox --name "Rekordbox Organizer" app.py
# macOS   -> dist/Rekordbox Organizer.app
pyinstaller --windowed --collect-all pyrekordbox --name "Rekordbox Organizer" app.py
```

PyInstaller can only build for the OS it runs on, so GitHub Actions builds both
(`.github/workflows/build.yml`). To publish a new version as a Release, tag it and
push the tag:

```
git tag v1.1.0
git push origin v1.1.0
```

Both apps are built and tested, then attached to a new Release with notes listing the
changes since the last one. You can also start a build by hand from the Actions tab;
that gives downloadable artifacts (kept 90 days) but no Release.

The builds aren't signed, so Windows and macOS warn the first time they're opened.
See [Installing](#installing) for how to get past that.

## Credits

Reading and writing the rekordbox database uses [pyrekordbox](https://github.com/dylanljones/pyrekordbox).

## License

[MIT](LICENSE): free to use, copy, change and share, including commercially, as long as
the copyright notice is kept.
