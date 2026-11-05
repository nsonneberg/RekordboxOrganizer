"""Safety of everything that writes to a library, on synthetic libraries only.

Direct database mode runs against a throwaway master.db built here with
pyrekordbox's own table definitions (unencrypted, which pyrekordbox supports
for test databases), so the real write code - pyrekordbox's sessions,
playlists, genres and commits - is exercised without touching a real library.
XML mode runs on a small sample export in a temporary folder.
"""

import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import organizer
import rbdb
from selftest import make_master_db
from test_organizer import SAMPLE

QUIET = dict(log=lambda m: None)


class FixedClock(__import__("datetime").datetime):
    """Every backup "made in the same second"."""
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 2, 3, 4, 5)


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class DatabaseCase(unittest.TestCase):
    """Points rbdb at a disposable master.db, with rekordbox reported closed."""

    def setUp(self):
        import pyrekordbox
        self.dir = tempfile.mkdtemp(prefix="rbo-safety-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.db = make_master_db(self.dir)

        real = pyrekordbox.MasterDatabase

        def master_database(path=None):     # stands in for finding the real library
            return real(path or self.db, unlock=False)
        self.running = False
        patches = [mock.patch.object(pyrekordbox, "MasterDatabase", master_database),
                   mock.patch.object(rbdb, "rekordbox_running", lambda: self.running),
                   mock.patch("pyrekordbox.masterdb.database.get_rekordbox_pid",
                              lambda: 1234 if self.running else 0)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def state(self) -> dict:
        """Everything a run can change: genres, ratings, playlists and their songs."""
        db = rbdb.open_database()
        try:
            tracks, _ = rbdb.load_tracks(db)
            lists = {}
            for pl in db.get_playlist().all():
                if pl.rb_local_deleted:
                    continue
                songs = sorted((s.TrackNo, str(s.ContentID))
                               for s in db.get_playlist_songs(PlaylistID=pl.ID).all())
                lists[(str(pl.ParentID), pl.Name, pl.Attribute)] = songs
            return {"tracks": {t.id: (t.name, t.genre, t.rating) for t in tracks},
                    "playlists": lists}
        finally:
            db.close()

    def backups(self):
        return rbdb.list_backups(self.dir)


def by_bpm(**kw):
    return organizer.Options(bpm=True, bpm_step=5, guess_genres=False, **kw)


class DatabasePreviewTests(DatabaseCase):
    def test_preview_never_writes_and_matches_the_real_run(self):
        before, digest = self.state(), sha(self.db)
        preview = organizer.Changes()
        organizer.run(by_bpm(), preview=True, changes=preview, **QUIET)
        self.assertEqual(sha(self.db), digest)              # not a byte changed
        self.assertEqual(self.state(), before)
        self.assertEqual(self.backups(), [])                # previews need no backup
        self.assertFalse(preview.empty())

        real = organizer.Changes()
        organizer.run(by_bpm(), changes=real, **QUIET)
        for field in ("created", "added", "removed", "ratings", "genres", "sets"):
            self.assertEqual(getattr(preview, field), getattr(real, field), field)
        self.assertNotEqual(self.state(), before)

    def test_preview_works_while_rekordbox_is_open(self):
        self.running = True
        digest = sha(self.db)
        organizer.run(by_bpm(), preview=True, changes=organizer.Changes(), **QUIET)
        self.assertEqual(sha(self.db), digest)

    def test_genre_save_preview_changes_nothing(self):
        digest = sha(self.db)
        changes = organizer.Changes()
        organizer.save_genres({"2": "House"}, preview=True, changes=changes, **QUIET)
        self.assertEqual(sha(self.db), digest)
        self.assertEqual(changes.genres, [("2", "Dance & EDM", "House")])
        self.assertEqual(self.backups(), [])


class DatabaseWriteTests(DatabaseCase):
    def test_backup_is_made_before_writing_and_undo_restores(self):
        original, digest = self.state(), sha(self.db)
        organizer.run(by_bpm(), **QUIET)
        [backup] = self.backups()
        self.assertEqual(sha(backup), digest)               # the library as it was
        changed = self.state()
        self.assertNotEqual(changed, original)
        self.assertIn(("root", "Organized", 1), changed["playlists"])
        # Your own playlist and tags are untouched.
        self.assertEqual(changed["playlists"][("root", "My Set", 0)], [(1, "1")])
        self.assertEqual(changed["tracks"], original["tracks"])

        keep = rbdb.restore_backup(backup, self.db)
        self.assertEqual(self.state(), original)
        self.assertEqual(sha(self.db), digest)
        shutil.copy2(keep, self.db)                         # ...and the undo can be undone
        self.assertEqual(self.state(), changed)

    def test_rerun_adds_nothing_twice(self):
        organizer.run(by_bpm(), **QUIET)
        once = self.state()
        organizer.run(by_bpm(), **QUIET)
        self.assertEqual(self.state()["playlists"], once["playlists"])

    def test_refuses_while_rekordbox_is_open(self):
        self.running = True
        digest = sha(self.db)
        with self.assertRaisesRegex(ValueError, "rekordbox is open"):
            organizer.run(by_bpm(), **QUIET)
        with self.assertRaisesRegex(ValueError, "rekordbox is open"):
            organizer.save_genres({"2": "House"}, **QUIET)
        self.assertEqual(sha(self.db), digest)
        self.assertEqual(self.backups(), [])

    def test_rekordbox_opened_mid_run_changes_nothing(self):
        before = self.state()
        calls = iter([False, True])         # closed at the start, open before writing
        with mock.patch.object(rbdb, "rekordbox_running", lambda: next(calls)):
            with self.assertRaisesRegex(ValueError, "Nothing was changed"):
                organizer.run(by_bpm(), **QUIET)
        self.assertEqual(self.state(), before)
        self.assertEqual(self.backups(), [])

    def test_failure_part_way_through_saves_nothing(self):
        before = self.state()
        real = rbdb.write_changes

        def half_then_fail(db, *a, **kw):
            real(db, *a, **{**kw, "commit": False})          # playlists made, not saved
            raise RuntimeError("disk on fire")
        with mock.patch.object(rbdb, "write_changes", half_then_fail):
            with self.assertRaisesRegex(RuntimeError, "disk on fire"):
                organizer.run(by_bpm(), **QUIET)
        self.assertEqual(self.state(), before)
        self.assertEqual(len(self.backups()), 1)             # the backup was still made first

    def test_genre_save_only_changes_the_chosen_songs(self):
        before = self.state()["tracks"]
        with mock.patch("genres.record_labels") as remembered:
            organizer.save_genres({"2": "House", "3": "Drum & Bass"},
                                  suggested={"2": "House"}, **QUIET)
        after = self.state()["tracks"]
        self.assertEqual(after["2"][1], "House")
        self.assertEqual(after["3"][1], "Drum & Bass")
        for tid in ("1", "4"):                               # your own tags: untouched
            self.assertEqual(after[tid], before[tid])
        self.assertEqual(len(self.backups()), 1)
        labels = remembered.call_args[0][0]
        self.assertEqual(labels["2"], ("House", "accepted", "Dance & EDM"))   # not learnt from
        self.assertEqual(labels["3"], ("Drum & Bass", "corrected", ""))

    def test_genre_save_failure_saves_nothing(self):
        before = self.state()
        real = rbdb.write_genres

        def fail_after(db, contents, new, *a, **kw):
            real(db, contents, new, *a, **kw)
            raise RuntimeError("interrupted")
        with mock.patch.object(rbdb, "write_genres", fail_after):
            with self.assertRaises(RuntimeError):
                organizer.save_genres({"2": "House", "3": "Trance"}, **QUIET)
        self.assertEqual(self.state(), before)

    def test_existing_genre_name_is_reused(self):
        with mock.patch("genres.record_labels"):
            organizer.save_genres({"3": "techno"}, **QUIET)
        self.assertEqual(self.state()["tracks"]["3"][1], "Techno")

    def test_review_prefills_only_genres_that_passed_the_gate(self):
        import genres
        P = genres.GenrePrediction
        preds = {"2": P("confident", "House", candidates=[("House", 0.99)], method="legacy"),
                 "3": P("suggested", "Drum & Bass", candidates=[("Drum & Bass", 0.97)],
                        method="calibrated")}
        with mock.patch.object(organizer, "predict", return_value=preds):
            rows, _ = organizer.genre_review(opt=organizer.Options(), **QUIET)
        by_id = {r.id: r for r in rows}
        self.assertEqual(by_id["2"].guess, "")                 # legacy: never pre-filled
        self.assertEqual(by_id["3"].guess, "")                 # a suggestion: needs approval
        self.assertEqual(by_id["3"].suggestion, "Drum & Bass")

    def test_no_write_if_the_backup_fails(self):
        before = sha(self.db)
        with mock.patch.object(rbdb, "backup", side_effect=OSError("disk full")):
            with self.assertRaises(ValueError) as cm:
                organizer.run(by_bpm(), **QUIET)
        self.assertIn("nothing was changed", str(cm.exception))
        self.assertEqual(sha(self.db), before)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        with mock.patch.object(rbdb, "backup", side_effect=OSError("disk full")):
            with self.assertRaises(ValueError):
                organizer.save_genres({"1": "House"}, **QUIET)
        self.assertEqual(sha(self.db), before)

    def test_crash_after_the_commit_is_detected_next_time(self):
        before = sha(self.db)
        # The database commit goes through, then the app dies before finishing.
        with mock.patch.object(rbdb, "end_change", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                organizer.save_genres({"1": "House"}, **QUIET)
        self.assertNotEqual(sha(self.db), before)
        info = rbdb.interrupted_change(self.db)
        self.assertEqual(info["what"], "write")
        after_crash = sha(self.db)
        for attempt in (lambda: organizer.run(by_bpm(), **QUIET),
                        lambda: organizer.save_genres({"2": "Techno"}, **QUIET)):
            with self.assertRaises(ValueError) as cm:
                attempt()
            self.assertIn("didn't finish", str(cm.exception))
        self.assertEqual(sha(self.db), after_crash)            # refused: nothing written
        # Preview still works (it only reads a copy).
        organizer.run(by_bpm(), preview=True, **QUIET)
        # Undo with the backup named in the marker puts back the exact library.
        backup = os.path.join(os.path.dirname(self.db), info["backup"])
        rbdb.restore_backup(backup, self.db)
        self.assertEqual(sha(self.db), before)
        self.assertIsNone(rbdb.interrupted_change(self.db))

    def test_failure_before_the_commit_leaves_no_marker(self):
        before = sha(self.db)
        with mock.patch.object(rbdb, "write_genres", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                organizer.save_genres({"1": "House"}, **QUIET)
        self.assertEqual(sha(self.db), before)
        self.assertIsNone(rbdb.interrupted_change(self.db))

    def test_genre_review_lists_only_untagged_and_vague_songs(self):
        with mock.patch.object(organizer, "predict", return_value={}):
            rows, _ = organizer.genre_review(opt=organizer.Options(), **QUIET)
        self.assertEqual(sorted(r.id for r in rows), ["2", "3"])
        self.assertEqual(self.backups(), [])


class LeftoverJournalTests(DatabaseCase):
    """A crashed rekordbox leaves saved changes in master.db-wal. SQLite moves them into
    master.db when its last connection closes, so looking at a library must not connect to it."""

    def setUp(self):
        super().setUp()
        con = sqlite3.connect(self.db)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA wal_autocheckpoint=0")
        con.execute("UPDATE djmdContent SET Title='From the journal' WHERE ID='2'")
        con.commit()
        crashed = os.path.join(self.dir, "crashed")
        os.mkdir(crashed)
        for suffix in ("", "-wal"):
            shutil.copy2(self.db + suffix, os.path.join(crashed, "master.db" + suffix))
        con.close()
        self.db = os.path.join(crashed, "master.db")
        self.before = self.files()

    def files(self):
        folder = os.path.dirname(self.db)
        return {name: (sha(os.path.join(folder, name)),
                       os.stat(os.path.join(folder, name)).st_mtime_ns)
                for name in os.listdir(folder)}

    def test_genre_review_leaves_it_alone_and_still_sees_the_journal(self):
        with mock.patch.object(organizer, "predict", return_value={}):
            rows, _ = organizer.genre_review(opt=organizer.Options(), **QUIET)
        self.assertEqual({r.id: r.name for r in rows}["2"], "From the journal")
        self.assertEqual(self.files(), self.before)

    def test_preview_leaves_it_alone(self):
        organizer.run(by_bpm(), preview=True, changes=organizer.Changes(), **QUIET)
        self.assertEqual(self.files(), self.before)

    def test_finding_the_library_leaves_it_alone(self):
        self.assertEqual(rbdb.library_path(), self.db)
        self.assertEqual(self.files(), self.before)

    def test_opening_the_app_leaves_it_alone(self):
        import app
        import tk_test_root
        root = tk_test_root.open_root(self)
        settings = os.path.join(self.dir, "settings.json")
        with mock.patch.object(app, "settings_path", lambda: settings),                 mock.patch.object(app.App, "check_for_update", lambda _: None):
            app.App(root)
            root.update()
        self.assertEqual(self.files(), self.before)


class BackupFailureTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbo-safety-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.db = os.path.join(self.dir, "master.db")
        for name, text in (("master.db", "library"), ("master.db-wal", "journal")):
            with open(os.path.join(self.dir, name), "w") as f:
                f.write(text)

    def read(self, name):
        with open(os.path.join(self.dir, name)) as f:
            return f.read()

    def test_interrupted_backup_is_never_listed_or_pruned_into(self):
        for n in range(rbdb.KEEP_BACKUPS):
            with open(os.path.join(self.dir, f"master.organizer-backup-20200101-00000{n}.db"),
                      "w") as f:
                f.write("old")
        real = shutil.copy2

        def fail_on_database(src, dst, *a, **kw):
            if src == self.db:
                with open(dst, "w") as f:
                    f.write("half")                          # a partial copy...
                raise OSError("disk full")
            return real(src, dst, *a, **kw)
        with mock.patch.object(shutil, "copy2", fail_on_database):
            with self.assertRaises(OSError):
                rbdb.backup(self.db)
        backups = rbdb.list_backups(self.dir)
        self.assertEqual(len(backups), rbdb.KEEP_BACKUPS)    # ...isn't one; none pruned
        self.assertTrue(all(self.read(os.path.basename(b)) == "old" for b in backups))

    def test_interrupted_restore_leaves_the_library_alone(self):
        saved = rbdb.backup(self.db)
        with open(self.db, "w") as f:
            f.write("changed")
        real = shutil.copy2

        def fail_on_backup(src, dst, *a, **kw):
            if src == saved:
                raise OSError("disk full")
            return real(src, dst, *a, **kw)
        with mock.patch.object(shutil, "copy2", fail_on_backup):
            with self.assertRaises(OSError):
                rbdb.restore_backup(saved, self.db)
        self.assertEqual(self.read("master.db"), "changed")
        self.assertEqual(self.read("master.db-wal"), "journal")   # its own journal kept
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".restoring")])

    def test_restore_when_the_file_cant_be_renamed_over(self):
        saved = rbdb.backup(self.db)
        with open(self.db, "w") as f:
            f.write("changed")
        real = os.replace
        live = {self.db, self.db + "-wal", self.db + "-shm"}

        def held_open(src, dst):
            if dst in live:                  # only the library itself is held open
                raise PermissionError("in use")
            return real(src, dst)
        with mock.patch.object(os, "replace", held_open):
            rbdb.restore_backup(saved, self.db)
        self.assertEqual(self.read("master.db"), "library")
        self.assertEqual(self.read("master.db-wal"), "journal")
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".restoring")])
        self.assertIsNone(rbdb.interrupted_change(self.db))

    def test_backups_made_in_the_same_second_keep_their_order(self):
        with mock.patch.object(rbdb.datetime, "datetime", FixedClock):
            made = []
            for n in range(rbdb.KEEP_BACKUPS + 1):
                with open(self.db, "w") as f:
                    f.write(f"library {n}")
                made.append(rbdb.backup(self.db))
        # Newest first, and pruning dropped the oldest (not the newest "-2" etc.).
        self.assertEqual(rbdb.list_backups(self.dir), made[::-1][:rbdb.KEEP_BACKUPS])
        self.assertEqual(self.read(os.path.basename(rbdb.list_backups(self.dir)[0])),
                         f"library {rbdb.KEEP_BACKUPS}")

    def test_pieces_of_a_cut_off_backup_are_cleared_by_the_next_one(self):
        stale = os.path.join(self.dir, "master.organizer-backup-20200101-000000.db.partial")
        with open(stale, "w") as f:
            f.write("half")
        rbdb.backup(self.db)
        self.assertFalse(os.path.exists(stale))
        self.assertEqual(self.read("master.db"), "library")      # the library is never touched

    def test_backup_is_checked_against_the_original(self):
        saved = rbdb.backup(self.db)
        self.assertEqual(rbdb._sha256(saved), rbdb._sha256(self.db))
        self.assertEqual(self.read(os.path.basename(saved) + "-wal"), "journal")

    def test_corrupt_backup_copy_is_refused_and_not_listed(self):
        real = shutil.copy2

        def flip(src, dst, *a, **kw):
            real(src, dst, *a, **kw)
            if src == self.db:
                with open(dst, "a") as f:
                    f.write("!")                     # the copy came out different
        with mock.patch.object(shutil, "copy2", flip):
            with self.assertRaises(OSError):
                rbdb.backup(self.db)
        self.assertEqual(rbdb.list_backups(self.dir), [])
        self.assertEqual(sorted(os.listdir(self.dir)), ["master.db", "master.db-wal"])

    def test_interrupted_undo_is_flagged_and_can_be_rerun(self):
        saved = rbdb.backup(self.db)
        with open(self.db, "w") as f:
            f.write("changed")
        real = os.replace
        calls = []

        def crash_after_database(src, dst):
            real(src, dst)
            if dst == self.db:
                calls.append(dst)
                raise KeyboardInterrupt          # stopped right after the database swap
        with mock.patch.object(os, "replace", crash_after_database):
            with self.assertRaises(KeyboardInterrupt):
                rbdb.restore_backup(saved, self.db)
        info = rbdb.interrupted_change(self.db)
        self.assertEqual((info["what"], info["backup"]), ("undo", os.path.basename(saved)))
        self.assertIn("Undo again", rbdb.interrupted_message(info))
        # The backup was only read, and the library before the undo was saved first.
        self.assertEqual(self.read(os.path.basename(saved)), "library")
        before = [n for n in os.listdir(self.dir) if n.startswith("master.before-undo")
                  and n.endswith(".db")]
        self.assertEqual([self.read(n) for n in before], ["changed"])
        rbdb.restore_backup(saved, self.db)          # just run it again
        self.assertEqual((self.read("master.db"), self.read("master.db-wal")),
                         ("library", "journal"))
        self.assertIsNone(rbdb.interrupted_change(self.db))

    def test_damaged_marker_still_counts_as_interrupted(self):
        with open(os.path.join(self.dir, rbdb.PENDING), "w") as f:
            f.write("{not json")
        self.assertEqual(rbdb.interrupted_change(self.db), {"what": "unknown"})
        self.assertIn("didn't finish", rbdb.interrupted_message({"what": "unknown"}))


class PlaylistIndexTests(unittest.TestCase):
    def test_playlist_index_is_saved_whole_or_not_at_all(self):
        folder = tempfile.mkdtemp(prefix="rbo-xml-")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "masterPlaylists6.xml")
        with open(path, "w") as f:
            f.write("original")

        class FakeXml:
            _changed = True

            def __init__(self):
                self.path = path

            def to_string(self, indent=None):
                raise RuntimeError("crashed while building the text")
        xml = FakeXml()
        rbdb._atomic_playlist_xml(xml)
        with self.assertRaises(RuntimeError):
            xml.save()
        with open(path) as f:
            self.assertEqual(f.read(), "original")
        xml.to_string = lambda indent=None: "<new/>"
        real = os.replace
        with mock.patch.object(os, "replace", side_effect=OSError("power cut")):
            with self.assertRaises(OSError):
                xml.save()
        with open(path) as f:
            self.assertEqual(f.read(), "original")           # never cut short
        self.assertEqual(os.listdir(folder), ["masterPlaylists6.xml"])
        os.replace = real
        xml.save()
        with open(path) as f:
            self.assertEqual(f.read(), "<new/>")


class XmlSafetyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbo-safety-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "library.xml")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        self.out = organizer.default_output_path(self.path)

    def tracks(self, path):
        return {t.get("TrackID"): dict(t.attrib)
                for t in ET.parse(path).getroot().iterfind("COLLECTION/TRACK")}

    def test_source_export_is_never_written(self):
        digest = sha(self.path)
        organizer.run(organizer.Options(auto_genres=True, genre=True, energy=False), self.path,
                      **QUIET)
        self.assertEqual(sha(self.path), digest)
        self.assertTrue(os.path.exists(self.out))
        self.assertNotEqual(self.out, self.path)

    def test_tags_are_carried_over_unchanged(self):
        # Even with genre guessing on, XML mode never rewrites a song's tags.
        organizer.run(organizer.Options(auto_genres=True, guess_genres=True), self.path,
                      **QUIET)
        self.assertEqual(self.tracks(self.out), self.tracks(self.path))

    def test_preview_writes_no_file(self):
        digest = sha(self.path)
        organizer.run(organizer.Options(genre=True), self.path, preview=True,
                      changes=organizer.Changes(), **QUIET)
        self.assertEqual(sha(self.path), digest)
        self.assertEqual(sorted(os.listdir(self.dir)), ["library.xml"])

    def test_interrupted_write_keeps_the_previous_output(self):
        organizer.run(organizer.Options(genre=True), self.path, **QUIET)
        previous = sha(self.out)

        def fail(self_tree, path, *a, **kw):
            with open(path, "w") as f:
                f.write("<half")
            raise OSError("disk full")
        with mock.patch.object(ET.ElementTree, "write", fail):
            with self.assertRaises(OSError):
                organizer.run(organizer.Options(genre=True, bpm=True), self.path, **QUIET)
        self.assertEqual(sha(self.out), previous)
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".partial")])

    def test_replaced_output_is_kept_as_previous(self):
        organizer.run(organizer.Options(genre=True), self.path, **QUIET)
        first = sha(self.out)
        organizer.run(organizer.Options(genre=True, bpm=True), self.path, **QUIET)
        self.assertEqual(sha(organizer.previous_output_path(self.out)), first)
        self.assertNotEqual(sha(self.out), first)


class DryRunTests(DatabaseCase):
    def run_dry(self, argv):
        import dry_run_genres
        out = tempfile.mkdtemp(prefix="rbo-dry-")
        self.addCleanup(shutil.rmtree, out, True)
        with mock.patch.object(dry_run_genres, "OUT_DIR", out), \
                mock.patch.object(organizer, "model_genres", return_value={}), \
                mock.patch("builtins.print"):
            dry_run_genres.main(argv)
        return out

    def test_dry_run_never_writes_to_the_database(self):
        before, digest = self.state(), sha(self.db)
        out = self.run_dry([])
        self.assertEqual(sha(self.db), digest)
        self.assertEqual(self.state(), before)
        self.assertEqual(self.backups(), [])
        self.assertTrue(os.path.exists(os.path.join(out, "dry_run_genres.csv")))

    def test_dry_run_never_writes_to_the_xml(self):
        path = os.path.join(self.dir, "library.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        digest = sha(path)
        self.run_dry([path])
        self.assertEqual(sha(path), digest)
        self.assertFalse(os.path.exists(organizer.default_output_path(path)))

    def test_dry_run_has_no_write_calls(self):
        import dry_run_genres
        with open(dry_run_genres.__file__, encoding="utf-8") as f:
            src = f.read()
        for call in ("write_changes", "write_genres", "save_genres", "write_library",
                     "commit(", "backup(", "restore_backup", "add_hot_cues",
                     "write_energy_tags", "organizer.run("):
            self.assertNotIn(call, src)


if __name__ == "__main__":
    unittest.main()
