import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import organizer
import rbdb
from test_organizer import SAMPLE

QUIET = dict(log=lambda m: None)


def sample_file(test) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(SAMPLE)
    out = organizer.default_output_path(path)
    test.addCleanup(lambda: [os.remove(p) for p in (path, out) if os.path.exists(p)])
    return path


def by_bpm(**kw):
    return organizer.Options(bpm=True, bpm_step=5, guess_genres=False, **kw)


class PreviewTests(unittest.TestCase):
    def test_preview_saves_nothing_and_lists_every_change(self):
        path = sample_file(self)
        changes = organizer.Changes()
        returned = organizer.run(by_bpm(), path, preview=True, changes=changes, **QUIET)
        self.assertEqual(returned, path)
        self.assertFalse(os.path.exists(organizer.default_output_path(path)))
        self.assertIn("Organized / By BPM / 125-129 BPM", changes.created)
        self.assertIn(("Organized / By BPM / 140-144 BPM", "4"), changes.added)
        text = organizer.summarize_changes(changes, {"4": "Song D"})
        self.assertIn("NEW PLAYLISTS", text)
        self.assertIn("Song D", text)

    def test_preview_of_a_finished_library(self):
        path = sample_file(self)
        out = organizer.run(by_bpm(), path, **QUIET)
        changes = organizer.Changes()
        organizer.run(by_bpm(), out, preview=True, changes=changes, **QUIET)
        self.assertTrue(changes.empty())
        self.assertIn("Nothing would change", organizer.summarize_changes(changes, {}))


class GeneratedFolderTests(unittest.TestCase):
    """The app's own folders follow the songs' current values on a re-run."""

    def setUp(self):
        path = sample_file(self)
        self.out = organizer.run(by_bpm(), path, **QUIET)
        tree = ET.parse(self.out)
        root = tree.getroot()
        # Song A's BPM was corrected in rekordbox since the last run...
        root.find("COLLECTION/TRACK[@TrackID='1']").set("AverageBpm", "141.00")
        # ...and the user keeps a playlist of their own in the main folder.
        folder = root.find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        mine = ET.SubElement(folder, "NODE", Name="Warm-up picks", Type="1",
                             KeyType="0", Entries="1")
        ET.SubElement(mine, "TRACK", Key="1")
        tree.write(self.out, encoding="UTF-8", xml_declaration=True)

    def rerun(self, **kw):
        second = organizer.run(by_bpm(**kw), self.out, **QUIET)
        self.addCleanup(os.remove, second)
        folder = ET.parse(second).getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        found = {}
        for pl in folder.iter("NODE"):
            if pl.get("Type") == "1":
                found[pl.get("Name")] = [t.get("Key") for t in pl.findall("TRACK")]
        return found

    def test_song_follows_its_new_bpm(self):
        found = self.rerun()
        self.assertEqual(found["125-129 BPM"], ["2"])
        self.assertEqual(found["140-144 BPM"], ["4", "1"])
        self.assertEqual(found["Warm-up picks"], ["1"])      # the user's own: untouched

    def test_not_when_moving_is_off(self):
        found = self.rerun(move_mismatched=False)
        self.assertEqual(found["125-129 BPM"], ["1", "2"])

    def test_only_songs_in_the_chosen_dates(self):
        found = self.rerun(date_from="2024-06-02")   # Song A was added 2024-06-01
        self.assertEqual(found["125-129 BPM"], ["1", "2"])


class UndoTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.db = os.path.join(self.dir, "master.db")

    def write(self, name, text):
        with open(os.path.join(self.dir, name), "w") as f:
            f.write(text)

    def read(self, name):
        with open(os.path.join(self.dir, name)) as f:
            return f.read()

    def test_restore_puts_back_the_library_and_keeps_the_current_one(self):
        self.write("master.db", "before")
        self.write("master.db-wal", "before-journal")
        self.write("masterPlaylists6.xml", "before-index")
        saved = rbdb.backup(self.db)
        self.write("master.db", "after")
        self.write("master.db-wal", "after-journal")
        self.write("master.db-shm", "after-shm")
        self.write("masterPlaylists6.xml", "after-index")

        keep = rbdb.restore_backup(saved, self.db)
        self.assertEqual(self.read("master.db"), "before")
        self.assertEqual(self.read("master.db-wal"), "before-journal")
        self.assertFalse(os.path.exists(self.db + "-shm"))  # never mixed with the backup
        self.assertEqual(self.read("masterPlaylists6.xml"), "before-index")
        with open(keep) as f:
            self.assertEqual(f.read(), "after")             # the undo can be undone
        with open(keep + "-wal") as f:
            self.assertEqual(f.read(), "after-journal")

    def test_backups_newest_first_and_old_ones_pruned(self):
        self.write("master.db", "x")
        self.write("master.db-wal", "j")
        for n in range(rbdb.KEEP_BACKUPS):
            self.write(f"master.organizer-backup-20200101-00000{n}.db", "old")
            self.write(f"master.organizer-backup-20200101-00000{n}.db-wal", "old")
        newest = rbdb.backup(self.db)
        backups = rbdb.list_backups(self.dir)
        self.assertEqual(len(backups), rbdb.KEEP_BACKUPS)
        self.assertEqual(backups[0], newest)
        self.assertFalse(os.path.exists(
            os.path.join(self.dir, "master.organizer-backup-20200101-000000.db-wal")))
        self.assertEqual(rbdb.backup_time(backups[-1]).year, 2020)


def song(i, name, artist="", length=200, size=0, path="", plays=0, cues=0, added="2026-01-01"):
    return organizer.Track(str(i), name, artist, "", "", 124.0, "", 0, 0, "", added, plays, path,
                           length=length, size=size, cues=cues)


class DuplicateTests(unittest.TestCase):
    def ids(self, tracks):
        return [[t.id for t in g] for g in organizer.find_duplicates(tracks)]

    def test_junk_in_titles_is_ignored(self):
        for junk in ("12. Latch (Lost Kings Remix)", "Latch (Lost Kings Remix) [FREE DL]",
                     "Latch (Lost Kings Remix) / 'Buy' = FREE MP3!", "latch (lost kings remix).mp3"):
            self.assertEqual(organizer.song_key(song(1, junk))[1], "latch lost kings remix", junk)
        # Mix names are part of the song.
        self.assertNotEqual(organizer.song_key(song(1, "Latch (Extended Mix)")),
                            organizer.song_key(song(2, "Latch (Radio Edit)")))

    def test_same_song_needs_a_similar_length(self):
        tracks = [song(1, "Latch", length=238), song(2, "LATCH", length=239),
                  song(3, "Latch", length=300)]                 # a different edit
        self.assertEqual(self.ids(tracks), [["1", "2"]])

    def test_same_file_under_another_name(self):
        tracks = [song(1, "Track 07", size=4900489, length=238),
                  song(2, "Latch", size=4900489, length=238), song(3, "Other", size=1, length=9)]
        self.assertEqual(sorted(self.ids(tracks)[0]), ["1", "2"])

    def test_keeps_the_copy_with_cues_then_plays_then_the_older_one(self):
        self.assertEqual(self.ids([song(1, "A"), song(2, "A", cues=3)]), [["2", "1"]])
        self.assertEqual(self.ids([song(1, "A", plays=1), song(2, "A")]), [["1", "2"]])
        self.assertEqual(self.ids([song(1, "A", added="2026-02-01"),
                                   song(2, "A", added="2026-01-01")]), [["2", "1"]])

    def test_cleanup_playlists(self):
        tracks = [song(1, "A"), song(2, "A", plays=2), song(3, "B")]
        plan = dict(organizer.build_plan(tracks, organizer.Options(duplicates=True,
                                                                   guess_genres=False)))
        self.assertEqual(dict(plan["Cleanup"]), {organizer.DUPLICATES_NAME: ["2", "1"],
                                                 organizer.EXTRA_COPIES_NAME: ["1"]})


if __name__ == "__main__":
    unittest.main()
