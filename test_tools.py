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


if __name__ == "__main__":
    unittest.main()
