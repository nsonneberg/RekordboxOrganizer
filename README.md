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

**Genre guessing:** rekordbox's analysis doesn't include genre. For songs with no genre
tag, or a vague one like "Electronic" or "Dance & EDM", the app estimates the genre. A
song's own tag still comes first: if one of your playlists names it (say an "EDM"
playlist for songs tagged "Dance & EDM"), the song goes there rather than where its
guess points. The estimate weighs these kinds of evidence:

1. **A tagged copy.** Another copy of the same recording (same artist, title and length)
   already has a genre. Copies are matched by name, which can be wrong, so this is only
   ever a suggestion for you to approve.
2. **An audio genre model** (optional, on the Genres tab). The app listens to up to
   four 10-second excerpts of each song: rekordbox's drops and choruses first, then
   points 30%, 50% and 70% of the way through, never just the intro. Pick one model:
   - **Discogs-EffNet** by the [MTG](https://essentia.upf.edu/models.html), trained on
     Discogs releases. It knows 400 Discogs styles, including Tech House, Deep House,
     Progressive House, Techno, Minimal, Psy-Trance, Bassline and UK Garage. 18 MB.
     **Licence: CC BY-NC-SA 4.0, free for non-commercial use only** (the MTG sells a
     commercial licence). It is the more accurate of the two for electronic music.
   - **[YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)**
     by Google, Apache-2.0 (any use). It knows about 50 broad styles (House, Techno,
     Drum and bass…), not subgenres. 16 MB.

   Either model is downloaded once from a fixed version, checked against its SHA-256,
   and then runs on your computer (CPU only, no internet needed, about 0.3 s per song).
   Results are saved per model, so only new or changed songs are listened to, and
   switching models never mixes their results.
3. **Your tagged songs.** A small classifier is trained on your own tagged songs, using
   what the model hears (or else rekordbox's waveform). This needs at least 3 tagged
   recordings in each of 2 or more genres. Copies of a song count once. It can only
   answer with genres you've tagged, so for anything else it defers to the model.
   Subgenres (Tech House, Liquid…) are named only when the model's styles or your
   nearest tagged songs clearly agree.
4. **Release info** (optional, off by default, needs internet). The app looks the song
   up on [MusicBrainz](https://musicbrainz.org), and on [Discogs](https://www.discogs.com)
   only if you also tick **Also Discogs**. The first time you switch lookups on, the app
   asks you, and says what is sent: each song's artist, title and length (for songs being
   guessed and your tagged ones), nothing else. No audio, file names or other library
   data leave your computer, and the audio analysis always runs offline. A match must have the same title, an
   overlapping artist and the same version: "Song (X Remix)" only matches X's remix,
   never the original. Genres come from the recording itself, else its single or EP at
   lower weight, never from the artist or a various-artists compilation. If two equally
   good matches disagree, neither is used. A song that isn't found counts as no
   evidence, never as evidence against a genre. MusicBrainz answers are cached (matches
   for 180 days, "not found" for 30); Discogs answers are kept for 6 hours at most and
   never used to train the app. Requests are paced within each site's limits (MusicBrainz
   about one a second; Discogs 25 a minute, 60 with a token), and if the internet is down
   the app gives up after 3 failures and carries on without it. A Discogs
   [personal token](https://www.discogs.com/settings/developers) is optional; it's kept in
   Windows Credential Manager or the macOS Keychain, never in the settings file (a token
   saved there by an earlier version is moved on first start). Genres from MusicBrainz
   are CC BY-NC-SA 3.0 and the review window credits them; Discogs data is shown as
   "Data provided by Discogs" (this app isn't affiliated with or endorsed by Discogs).
5. **Tempo and the vague tag**, as supporting evidence only. A BPM read at half or double
   speed (87 for 174 Drum & Bass) counts as a match. Tempo can make a genre less likely,
   but **BPM alone never decides a genre.**

**How it combines them:** the sources' answers are pooled (multiplied, in log space) with
a weight for each source, and for a genre with at least 10 tagged songs, a per-genre
weight too. The weights are fitted on your own tagged songs, using each source's answers
for songs it hadn't learnt from (each song hidden in turn, with its other versions), so
a source that agrees with the others because it's a copy of them, or that's often
wrong, ends up counting less. A source with nothing to say (no match found, a genre it
doesn't know) is left out rather than counted against anything. How often you've tagged
each genre is used while fitting but doesn't vote on new songs: the songs being guessed
are the ones you *didn't* tag, so "most of my tagged songs are House" isn't evidence that
an untagged one is. The result is the combining model's confidence: a probability it
fitted on your tagged songs, not a measured hit rate. On the author's library it was
often under-confident (guesses at 70-90% were nearly always right) and over-confident
below 50%, and a guess can be wrong even near 100%, for example when every source hears
the same thing (the review window shows ">99%", never 100%). Separately, the app works out
how confident a guess must be to count: the lowest level at which, on songs held out
from fitting, the estimated error was at most 1 in 10 (with at least 8 such songs). That
is an operating point, not a guarantee. On top of that, each genre must prove itself: a
genre is only tagged automatically when its own held-out guesses make a low error rate
likely, meaning the 95% upper bound on their error rate is at most 10%. With no mistakes
that takes 29 held-out guesses of that genre; with one mistake, 46. Until then its clear
guesses are **suggestions**: shown in the review window and in a "(suggested)" playlist
inside Needs Review, for you to approve, never applied by themselves. Unclear guesses, or
ones where the sources disagree, go to **Needs Review**. With fewer than 10 tagged songs
nothing can be measured, so every guess is at most a suggestion. The review window says
how often held-out guesses like each one were right. Even the 95% bound only describes
songs like the ones you've tagged, judged against your own (possibly inconsistent) tags.
A subgenre is assigned only once subgenre guesses have met the same bar; otherwise it's
shown as a suggestion. Nothing is written to a song's genre tag until you save it in
Review genres, and songs that already have a specific genre are never listed there. Genre names are
normalised through a taxonomy (genre family → subgenre, with synonyms such as "UKG" →
UK Garage and "Psychedelic Trance" → Psytrance). To add your own subgenres or synonyms,
put a `taxonomy.json` next to the app's settings with the same shape as `DEFAULT` in
`taxonomy.py`; it is merged on top.

The app only names a genre when the evidence clearly points one way. Otherwise the song
is marked **uncertain** and goes to a **Needs Review** folder. That includes songs where
two reliable sources disagree (say the model hears House but the release is listed as
Techno). Without a model, and with few tagged songs, most songs will be uncertain; that's
deliberate. The confidences in the review window are the model's estimates, fitted on
your own tagged songs, not guarantees: if your tags are inconsistent, so are they. Selecting a song shows each
source's answer, how much it counted, any suggested subgenre and the release it matched.

Your own playlists use only guesses that passed every check above (never suggestions), so a "Dance & EDM" track that clearly sounds
like house can land in your House playlist. Turn guessing off to match on tags only.

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
   │  └─ House (detected)      ← confident guesses, kept apart so you can check them
   ├─ Drum & Bass
   ├─ Needs Review             ← untagged songs the app isn't sure about
   └─ Unknown Genre            ← no tag and nothing to guess from
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
analysed the track. With an XML export, it listens to the audio files instead.

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

**Experimental energy algorithm (off by default):** a newer algorithm that scores each
track on a fixed 0-100 scale from the decoded audio (rhythm, low-end hits, beat
strength, density), so levels don't change as the library grows. Tick "Use the
experimental algorithm" to try it. Against one listener's own ratings of 72 tracks it
did much worse than the default: its ranking was no better than chance and it put
almost every track at level 3 or 4. Songs it can't judge are left unrated.

**M3 energy model (opt-in, off by default):** a model calibrated to one listener's
ratings of one library. The classic algorithm stays the default. Tick "Use M3 instead"
on the Energy tab to try it; it can't be combined with the experimental algorithm.

- **Fixed levels:** a track's level comes from that track's own measurements, so it
  doesn't change as your library grows, and levels needn't be spread evenly from 1 to 5.
- **Straight from rekordbox only:** it needs rekordbox's analysis as well as the decoded
  audio. With an XML export it's switched off and the run says so, using the classic
  algorithm (or the experimental one, if ticked). Your choice comes back when you switch
  back to rekordbox.
- **Skipped songs:** a song with no analysis, audio that can't be decoded, no BPM or a
  measurement that couldn't be made gets no level and keeps its rating, and the log says
  why. No other algorithm stands in for it.

It was tested once, on songs that shared no artist or recording with the songs it was
built from, against criteria set in advance. The same listener rated them by ear from a
sheet that showed no model's levels. Lower error is better, and the brackets are a 95% range:

| 76 songs (65 artist groups)    | M3    | Classic | Tempo only |
|--------------------------------|-------|---------|------------|
| Average error, in levels       | 0.474 | 0.855   | 0.855      |
| Exact level                    | 55%   | 39%     | 34%        |
| Within one level               | 97%   | 80%     | 83%        |
| Rank agreement (Spearman)      | 0.88  | 0.71    | 0.78       |

M3's average error was 0.382 lower than the classic algorithm's [0.192 to 0.575 lower],
and all the set criteria passed. What that doesn't show:

- **It may not suit your ears.** It describes one listener and one library, and that
  listener's ratings also trained it. How it does for anyone else hasn't been tested.
- **It isn't shown to be better on calm songs.** On songs rated 1-2 it was not
  meaningfully worse than the classic algorithm, but the result doesn't show it's better.
- **It has known weak spots.** It often gave 1 to songs rated 2 (9 of 16), and 4 to songs
  rated 5 (6 of 19). Only 8 songs were rated 4.
- **Unusual songs are untested.** There were too few fast-but-calm and slow-but-intense
  songs (2 of each) to say anything about them.

The test can't be reused: those songs are now spent, so testing any change to M3 needs
new ratings. The scoring script ran once. Its results were saved before it crashed
printing them, because the Windows console couldn't show a symbol in an artist's name;
it exited with code 1, and the saved report was read instead of running it again.

**Extra playlists (optional, all off by default):** Genre · BPM range (step size you choose) · Key (Camelot, in wheel order) ·
Artist · Record label · Release year · Star rating · Track color · Month added

**Cleanup playlists:** Possible duplicates · Missing files · Missing genre / key / BPM · Never played

**Duplicates:** copies of a song are found by name, ignoring track numbers and
download junk like "[FREE DL]", with lengths within 2 seconds. Identical files count
too. Mix names still matter, so an Extended Mix and a Radio Edit aren't duplicates.
"Possible Duplicates" shows every copy. "Duplicates - Extra Copies" holds only the
copies you can remove: for each song, the copy kept is the one whose file exists and
that has the most cues and plays. In rekordbox, select everything in that playlist and
choose **Remove from Collection**. The app never deletes anything. If a copy you'd
remove has cues the kept copy lacks, the log tells you.

Everything goes into one playlist folder (named "Organized" by default). If you run
the app again, it **adds to** folders and playlists that already have the same name
(ignoring capitals): new songs go at the end, and songs already there aren't added
twice. Untick "Add to existing folders / playlists" to replace the folder instead.

**Set builder:** builds one playlist in play order from your whole library. Pick a
name, a length, a start and end energy (e.g. 2 → 5 for a warm-up into peak time), and
optionally a genre and BPM range. Each song is chosen so it mixes into the one
before: a compatible key on the Camelot wheel (same key, one step round, or the
relative major/minor), a BPM within a few percent (half and double time count), and
an energy close to where the set should be at that point. Sets go in the **Sets**
folder and are rewritten in order each time you run, rather than added to. The log
mentions any key clashes or big BPM jumps it couldn't avoid. Energy needs to be
measured for this, which happens automatically.

**Review genres:** on the Genres tab, **Review genres...** lists every song with no
genre or a vague one ("Dance & EDM", "Electronic") next to the app's guess. Select
songs, pick the right genre, and press **Save to rekordbox** (close rekordbox first; a
backup is made). Uncertain songs show "?" with their likeliest genres; select one to see
why. Copies of a song change together, and an existing genre with the same name is
reused. Genres you pick become real tags that later guesses learn from. A suggestion
you save unchanged is remembered as accepted rather than confirmed, so the app doesn't
learn from its own guesses. It never changes a genre tag without you saving it here.

To see how well guessing works on your own tagged songs, run
`python evaluate_genres.py` (it only reads your library). Each tagged song is hidden in
turn, together with its copies and other versions of it, and guessed from the rest.
`python benchmark_genres.py` compares the old classifier, Discogs-EffNet alone, the
classifier trained on your tags, release info alone and both combination methods, all on
the same held-out songs: accuracy and macro-F1 with confidence intervals, per-genre
precision/recall, confusions, the confusable pairs (Deep vs Tech House and so on),
automatic coverage against error, calibration and runtime. It uses only cached release
info unless you add `--metadata`. `python dry_run_genres.py` lists what the app would
suggest for every song (tag, review or leave unchanged) without changing anything. Both
reports go to `benchmark_results/`, which git ignores because it lists your songs.

Your rekordbox tags aren't verified, so results measured against them are an estimate.
For a real measurement, verify some songs by hand with `benchmark_labels.py`:

```
python benchmark_labels.py candidates [--playlist NAME]   # add songs to check
python benchmark_labels.py export labels.csv              # fill in verified_genre + evidence
python benchmark_labels.py import labels.csv --by YOURNAME
python benchmark_labels.py stats                          # how many per genre (aim: 25)
python benchmark_genres.py --truth verified               # develop on the dev split
python benchmark_genres.py --truth verified --split test --final   # once, at the end
```

About 30% of songs (chosen by title, so every version of a song lands on the same side)
form a test split that the app never learns from or tunes on; each test run is logged.

To check the energy levels (all read-only; results go to `benchmark_results/`):

```
python benchmark_energy.py synthetic           # designed test signals (no music needed)
python benchmark_energy.py measure             # measure your library in every mode
python benchmark_energy.py library             # checks that need no ratings: agreement
                                               # between modes, stability, per genre
python benchmark_energy.py template 60         # a CSV of songs to rate 1-5 by ear
python benchmark_energy.py evaluate labels.csv # agreement with your ratings
python energy_labelling.py select              # more songs to rate, in batches (see below)
python energy_labelling.py status              # how many of them you've rated
```

`energy_labelling.py select` puts every song you haven't rated into batches of 40,
calm-sounding candidates first (picked several ways, not only by the current algorithms),
and sets about a quarter of them aside as a held-out test set before anything is rated.
The sheets show only artist and title. It never overwrites sheets that exist, and
`template` won't overwrite a template that already holds ratings.

Rate by listening, without looking at the app's ratings, and judge intensity rather
than how much you like a song or how loud the file is. `evaluate` reports rank
correlation (with a confidence interval), exact and within-one-level agreement,
weighted kappa, a 1-5 confusion table, results per genre and per analysis mode,
agreement between raters if there are several, and the old algorithm for comparison.
The synthetic expectations are the author's, not listeners'.

**Songs move when they fit better:** when you re-run, songs already in your playlists
are checked again. If a song now fits a *different* one of your playlists best (for
example, you added an "EDM" playlist and it's tagged EDM), it's moved there: taken out
of the old playlist and added to the new one. The log lists every move. Songs that
still fit, or that fit none of your playlists (including ones you put somewhere by
hand), stay where they are. Only your own playlists in the main folder are affected;
a playlist you've removed from your list is left alone.

The folders the app makes itself (By Energy, Auto Genres, By BPM and the other extra
playlists) are kept up to date the same way. A song whose energy went from 3 to 2 moves
from "3 - Groove" to "2 - Warm-up", and a corrected BPM moves it to the right BPM
playlist. Only folders you've ticked for this run are updated, and only for the songs
being organized. Untick "move songs to the playlist they now fit best" in
*My playlists* to turn all moving off.

**Only new songs:** under "Songs to organize" pick *All*, *Added since last run* (the
app remembers the date of each successful run), or *Added from … to …* (either date
can be left empty). Only those songs are sorted, rated and added; the rest of the
library is still read for comparison, so energy levels and genre guesses stay
consistent with earlier runs.

## How to use

### Straight from rekordbox (default)

1. Open Rekordbox Organizer and tick what you want on each tab (Library, My
   playlists, Extra playlists, Energy, Cues, Genres, Set builder). The line next to
   the Run button sums up what's ticked.
2. Press **Preview** to see exactly what would change: songs moved, playlists created
   or added to, star ratings and hot cues. Nothing is changed, and rekordbox can stay open.
3. **Close rekordbox** and press **Run** (or **Run now** in the preview).
4. Open rekordbox. The new folder is in your playlists, and ratings are updated.

The app reads and writes rekordbox's own library database (rekordbox 6 and 7).
Before changing anything it saves a backup next to the library, named
`master.organizer-backup-<date>.db`. It keeps the 3 most recent backups.

**Undo:** close rekordbox, press **Undo last run...**, pick the run, and press
**Restore**. Your library goes back to how it was just before that run. Anything you
changed in rekordbox after it is undone too. Your library as it was before the undo is
saved as `master.before-undo-<date>.db`, so you can change your mind. The library is in
`%APPDATA%\Pioneer\rekordbox\` on Windows and `~/Library/Pioneer/rekordbox/` on macOS.

Every backup (and every restore) is checked byte for byte against what it copied, and is
only given its final name once it matches: a backup that fails or is cut off is never
listed, and nothing is written without a good backup.

**If a change was interrupted** (a crash, power cut, or the app killed while saving):
rekordbox keeps your playlists in two files, `master.db` and `masterPlaylists6.xml`, and
they are saved one after the other, so an interrupted save can leave them out of step.
The app notes each change in `master.organizer-pending.json` (next to the library) before
writing and removes it once both files are saved. If it finds that note, it says so and
won't write anything else until you either press **Undo last run...** and restore the
backup it names (the safe choice; an interrupted Undo can simply be run again), or check
your library in rekordbox and choose **Keep my library as it is**. Nothing is restored
automatically. To recover by hand instead: close rekordbox, copy the whole folder
somewhere safe first, then copy `master.organizer-backup-<date>.db` over `master.db`,
`master.organizer-backup-<date>.masterPlaylists6.xml` over `masterPlaylists6.xml`, delete
any `master.db-wal` / `master.db-shm` next to `master.db` (put back the backup's own ones
if it has them), and delete `master.organizer-pending.json`. Files ending in `.partial`
or `.restoring` are unfinished copies and can be deleted.

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

MP3, WAV, AIFF, FLAC and OGG always work. M4A/AAC/ALAC work on macOS through the
built-in `afconvert`. On Windows they need `ffmpeg` installed (on your PATH), or running
from source with PyAV installed (`pip install av`). PyAV isn't bundled in the built apps
because its wheels include GPL-licensed video encoders (x264, x265); see Credits. Songs
that can't be decoded are skipped (no energy or model guess), never changed; for
M4A/AAC/ALAC without ffmpeg the log says ffmpeg is needed.

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
line and `python app.py`. To run the tests: `python -m unittest -v`. The library-safety
tests (`test_library_safety.py`) run every write path against a throwaway master.db and
XML built in a temporary folder. To check direct mode against your own library's exact
format without touching it, close rekordbox and run `python check_database_copy.py`: it
works on a temporary copy, then confirms your real files are byte-for-byte unchanged.
A built app can test itself the same way on a throwaway library:
`"Rekordbox Organizer.exe" --selftest <empty folder> [<folder of audio files>]` runs
Preview, a real run, a genre save, the interrupted-change guard and Undo on a synthetic
library inside that folder, tries to decode each audio file, and writes `selftest.json`
there (see `selftest.py`). It never opens your real library.

## Building the double-click apps

```
pip install -r requirements.txt pyinstaller
# Windows -> dist/Rekordbox Organizer.exe
pyinstaller --onefile --windowed --collect-all pyrekordbox --collect-binaries onnxruntime --collect-submodules keyring.backends --exclude-module av --add-data "THIRD_PARTY_LICENSES.md:." --add-data "energy_m3_model.json:." --name "Rekordbox Organizer" app.py
# macOS   -> dist/Rekordbox Organizer.app
pyinstaller --windowed --collect-all pyrekordbox --collect-binaries onnxruntime --collect-submodules keyring.backends --exclude-module av --add-data "THIRD_PARTY_LICENSES.md:." --add-data "energy_m3_model.json:." --name "Rekordbox Organizer" app.py
```

`--exclude-module av` keeps PyAV's bundled FFmpeg (with GPL x264/x265) out of the build;
the workflow fails if any such library ends up in it anyway.

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

Licences of the optional parts (check them before any commercial use or redistribution):

- **Discogs-EffNet** model (MTG, Universitat Pompeu Fabra): CC BY-NC-SA 4.0, free for
  non-commercial use only; MTG offers a commercial licence on request. The app
  downloads it on first use rather than shipping it.
- **YAMNet** (Google; ONNX conversion): Apache-2.0. AudioSet class names: CC BY 4.0.
- **MusicBrainz** genre and tag data: CC BY-NC-SA 3.0 (core data is CC0).
- **Discogs** data: subject to the Discogs API Terms of Use. Data provided by Discogs.
- Optional token storage uses [keyring](https://github.com/jaraco/keyring) (MIT).
- The builds bundle libsndfile (LGPL-2.1) and other permissively licensed components;
  see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). PyAV/FFmpeg is deliberately
  left out of the builds (its Windows wheel includes GPL x264/x265), and the build checks
  that no such library slipped in. Running from source with PyAV installed is fine for
  your own use.

## License

[MIT](LICENSE): free to use, copy, change and share, including commercially, as long as
the copyright notice is kept.
