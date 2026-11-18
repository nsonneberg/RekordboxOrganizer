"""Copies of the live library must be of one moment, and Undo must never lose a change.

The app copies master.db and its journal as files, one after another. If
rekordbox writes, or folds its log into master.db, between two of those copies,
the copy mixes two moments and reads as "database disk image is malformed" (the
library itself is fine). These tests make a stand-in for rekordbox write in that
gap, on a synthetic library only, and check the app notices, stops, never reads
the mixed copy and leaves the library as it was. Undo is checked the same way,
including a program that still has the old files open while they're swapped
(a Mac allows renaming over an open file; Windows doesn't).
"""

import contextlib
import datetime
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import organizer
import rbdb
from test_library_safety import PLAYLIST_XML, QUIET, DatabaseCase, by_bpm, sha
from test_organizer import SAMPLE

NOW = datetime.datetime(2024, 1, 1, 12, 0, 0)


class StandInRekordbox:
    """A second program writing to the library like rekordbox: one long-lived
    connection, write-ahead log on, and checkpoints only when asked."""

    def __init__(self, path: str, first_id: int = 704823000):
        from sqlalchemy import create_engine
        self.engine = create_engine(f"sqlite:///{path}")
        self.conn = self.engine.connect()
        self.conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        self.conn.exec_driver_sql("PRAGMA wal_autocheckpoint=0")
        self.conn.commit()
        self.next_id = first_id

    def add_artists(self, n: int, tag: str) -> list[str]:
        from pyrekordbox.masterdb import models
        from sqlalchemy.orm import Session
        ids = []
        with Session(bind=self.conn) as s:
            for _ in range(n):
                self.next_id += 1
                kw = {c.name: NOW for c in models.DjmdArtist.__table__.columns
                      if "date" in type(c.type).__name__.lower()}
                s.add(models.DjmdArtist(ID=str(self.next_id), Name=f"{tag} {self.next_id} " + "x" * 80,
                                        UUID=f"a{self.next_id}", rb_local_deleted=0, **kw))
                ids.append(str(self.next_id))
            s.commit()
        return ids

    def credit_songs(self, artist_ids: list[str]) -> None:
        from pyrekordbox.masterdb import models
        from sqlalchemy.orm import Session
        with Session(bind=self.conn) as s:
            for cid, aid in zip(("1", "2", "3", "4"), artist_ids):
                s.get(models.DjmdContent, cid).ArtistID = aid
            s.commit()

    def checkpoint(self) -> None:
        self.conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.commit()          # no transaction left open (Windows would lock the files)

    def close(self) -> None:
        self.conn.close()
        self.engine.dispose()


def write_and_quit(path: str, n: int = 5, tag: str = "late") -> None:
    """rekordbox opening the library at path, saving a few artists and quitting."""
    rb = StandInRekordbox(path, first_id=704900000)
    try:
        rb.add_artists(n, tag)
    finally:
        rb.close()


class LiveLibraryCase(DatabaseCase):
    def setUp(self):
        super().setUp()
        self.rb = StandInRekordbox(self.db)
        self.addCleanup(self.rb.close)
        self.first = self.rb.add_artists(1500, "first")     # in the log, not yet in master.db
        self.after_write = None

    def keep_working(self):
        """What rekordbox does mid-copy: checkpoint, then write again on the new log."""
        self.rb.checkpoint()
        second = self.rb.add_artists(1500, "second")
        self.rb.credit_songs([self.first[10], self.first[700], self.first[1400], second[-1]])
        self.after_write = self.library_files()

    APP_FILES = ("organizer-backup", "before-undo", ".partial", ".restoring", "organizer-pending")

    def library_files(self) -> dict:
        """The library's own files and their contents: not the app's backups, and not
        the -shm index (SQLite rebuilds it and any reader may update it)."""
        folder = os.path.dirname(self.db)
        return {name: sha(os.path.join(folder, name)) for name in sorted(os.listdir(folder))
                if not name.endswith("-shm") and not any(a in name for a in self.APP_FILES)}

    def app_files(self) -> list:
        return sorted(n for n in os.listdir(os.path.dirname(self.db))
                      if any(a in n for a in self.APP_FILES))

    def add_playlist_index(self):
        with open(os.path.join(self.dir, "masterPlaylists6.xml"), "w", encoding="utf-8") as f:
            f.write(PLAYLIST_XML)

    def artists(self, tag: str) -> int:
        con = sqlite3.connect(self.db)
        try:
            return con.execute("SELECT count(*) FROM djmdArtist WHERE Name LIKE ?",
                               (tag + " %",)).fetchone()[0]
        finally:
            con.close()

    @contextlib.contextmanager
    def during_copy(self, action, copier="_copy_hashed", after=None):
        """Run action once, straight after a file is copied: the first one, or the
        one named `after`."""
        real = getattr(rbdb, copier)
        done = []

        def copy_then_act(src, dst):
            out = real(src, dst)
            if not done and (after is None or os.path.basename(src) == after):
                done.append(src)
                action()
            return out
        with mock.patch.object(rbdb, copier, copy_then_act):
            yield done
        self.assertTrue(done, "the stand-in never got to write")

    @contextlib.contextmanager
    def watching_copies(self):
        """Records what is read and checks the temporary copy is gone afterwards."""
        reads, folders = [], []
        real_load, real_mkdtemp = rbdb.load_tracks, tempfile.mkdtemp

        def load(db):
            reads.append(rbdb.database_path(db))
            return real_load(db)

        def mkdtemp(*a, **k):
            folders.append(real_mkdtemp(*a, **k))
            return folders[-1]
        with mock.patch.object(rbdb, "load_tracks", load), \
                mock.patch.object(tempfile, "mkdtemp", mkdtemp):
            yield reads
        for folder in folders:
            self.assertFalse(os.path.exists(folder), "the copy was left behind")

    def preview(self):
        return organizer.run(by_bpm(), preview=True, changes=organizer.Changes(), **QUIET)

    def assert_library_as_rekordbox_left_it(self, app_files=()):
        self.assertEqual(self.library_files(), self.after_write)   # nothing changed
        self.assertEqual(self.app_files(), sorted(app_files))      # no backup or pieces left
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assert_healthy()

    def assert_healthy(self):
        con = sqlite3.connect(self.db)
        try:
            self.assertEqual(con.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            con.close()


class TornCopyTests(LiveLibraryCase):
    def test_the_scenario_really_tears_a_copy(self):
        """Control: with the check switched off, the same write gives the reported error."""
        from sqlalchemy.exc import DatabaseError
        with mock.patch.object(rbdb, "_changed_since", return_value=False), \
                self.during_copy(self.keep_working):
            with self.assertRaises(ValueError) as ctx:      # "couldn't read a consistent copy"
                self.preview()
        self.assertIsInstance(ctx.exception.__cause__, DatabaseError)
        self.assertIn("database disk image is malformed", str(ctx.exception.__cause__))

    def test_preview_stops_and_never_reads_the_copy(self):
        with self.watching_copies() as reads, self.during_copy(self.keep_working):
            with self.assertRaises(rbdb.LibraryChanged) as ctx:
                self.preview()
        self.assertIn("changed while the app was copying it", str(ctx.exception))
        self.assertIn("Nothing in your library was changed", str(ctx.exception))
        self.assertEqual(reads, [])
        self.assert_library_as_rekordbox_left_it()

    def test_review_genres_stops(self):
        with self.watching_copies() as reads, self.during_copy(self.keep_working), \
                mock.patch.object(organizer, "predict", return_value={}):
            with self.assertRaises(rbdb.LibraryChanged):
                organizer.genre_review(opt=organizer.Options(), **QUIET)
        self.assertEqual(reads, [])
        self.assert_library_as_rekordbox_left_it()

    def test_genre_save_preview_stops(self):
        with self.watching_copies() as reads, self.during_copy(self.keep_working):
            with self.assertRaises(rbdb.LibraryChanged):
                organizer.save_genres({"2": "House"}, preview=True, changes=organizer.Changes(),
                                      **QUIET)
        self.assertEqual(reads, [])
        self.assert_library_as_rekordbox_left_it()

    def test_rekordbox_starting_mid_copy_stops_it(self):
        def opens():
            self.running = True
            self.after_write = self.library_files()
        with self.watching_copies() as reads, self.during_copy(opens):
            with self.assertRaises(rbdb.LibraryChanged) as ctx:
                self.preview()
        self.assertIn("rekordbox was opened while the app was copying", str(ctx.exception))
        self.assertEqual(reads, [])
        self.assert_library_as_rekordbox_left_it()

    def test_run_stops_if_the_library_changes_during_the_backup(self):
        before = self.state()
        with self.during_copy(self.keep_working, copier="_copy_verified"):
            with self.assertRaisesRegex(ValueError, "Couldn't back up your library, so nothing "
                                                    "was changed: .*changed while"):
                organizer.run(by_bpm(), **QUIET)
        self.assert_library_as_rekordbox_left_it()
        playlists = self.state()["playlists"]
        self.assertEqual(playlists, before["playlists"])          # nothing was written

    def test_undo_stops_if_the_library_changes_while_it_is_saved(self):
        saved = rbdb.backup(self.db)          # made while the library was still
        backup_files = self.app_files()
        with self.during_copy(self.keep_working, copier="_copy_verified"):
            with self.assertRaises(rbdb.LibraryChanged):
                rbdb.restore_backup(saved, self.db)
        # The library wasn't restored over, the backup is untouched, and no saved copy
        # ("before-undo") or staged file is left.
        self.assert_library_as_rekordbox_left_it(app_files=backup_files)
        self.assertEqual(self.backups(), [saved])


class LogAndIndexTests(LiveLibraryCase):
    """The log appearing or going, or only the playlist index changing, mid-copy."""

    def setUp(self):
        super().setUp()
        self.add_playlist_index()            # copied last, so changes can come after it

    def stops_preview(self, action):
        with self.watching_copies() as reads, \
                self.during_copy(action, after="masterPlaylists6.xml"):
            with self.assertRaises(rbdb.LibraryChanged):
                self.preview()
        self.assertEqual(reads, [])
        self.after_write = self.after_write or self.library_files()
        self.assert_library_as_rekordbox_left_it()

    def test_a_log_appearing_after_it_was_looked_for(self):
        self.rb.close()                      # rekordbox quit: its log folded in and deleted
        self.assertFalse(os.path.exists(self.db + "-wal"))

        def opens_and_saves():               # ...and starts again mid-copy
            rb = StandInRekordbox(self.db, first_id=704900000)
            self.addCleanup(rb.close)
            rb.add_artists(5, "late")
            self.after_write = self.library_files()
        self.stops_preview(opens_and_saves)
        self.assertTrue(os.path.exists(self.db + "-wal"))

    def test_the_log_going_away(self):
        self.assertTrue(os.path.exists(self.db + "-wal"))
        self.stops_preview(self.rb.close)    # rekordbox quits mid-copy
        self.assertFalse(os.path.exists(self.db + "-wal"))

    def test_only_the_playlist_index_changing(self):
        def save_index():
            with open(os.path.join(self.dir, "masterPlaylists6.xml"), "a", encoding="utf-8") as f:
                f.write("\n")
        self.stops_preview(save_index)

    def test_exactly_which_changes_count(self):
        files = rbdb._library_files(self.db)
        copied = {p: rbdb._sha256(p) for p in files if os.path.exists(p)}
        self.assertFalse(rbdb._changed_since(copied, files))
        log = files[1]
        self.assertTrue(rbdb._changed_since({p: d for p, d in copied.items() if p != log},
                                            files))                     # appeared
        other = os.path.join(self.dir, "other-file")
        with open(other, "w") as f:
            f.write("x")
        copied = {other: rbdb._sha256(other)}
        self.assertFalse(rbdb._changed_since(copied, [other]))
        os.remove(other)
        self.assertTrue(rbdb._changed_since(copied, [other]))           # gone
        # The -shm index isn't part of the library's moment: it may change freely.
        self.assertNotIn(self.db + "-shm", files)


class ReadErrorTests(LiveLibraryCase):
    def test_a_file_that_cant_be_read_again_is_reported_as_that(self):
        def locked(path):
            raise PermissionError(13, "locked by another program", path)
        with self.watching_copies() as reads, mock.patch.object(rbdb, "_sha256", locked):
            with self.assertRaisesRegex(OSError, "Couldn't read master.db again to check the "
                                                 "copy \\(locked by another program\\)") as ctx:
                self.preview()
        self.assertNotIsInstance(ctx.exception, rbdb.LibraryChanged)
        self.assertEqual(reads, [])

    def test_backup_says_so_too(self):
        real = rbdb._sha256
        checking = []

        def locked_when_checking(path):
            if checking:
                raise PermissionError(13, "locked by another program", path)
            return real(path)
        real_check = rbdb._check_snapshot

        def check(*a, **k):
            checking.append(1)
            return real_check(*a, **k)
        with mock.patch.object(rbdb, "_sha256", locked_when_checking), \
                mock.patch.object(rbdb, "_check_snapshot", check):
            with self.assertRaisesRegex(OSError, "Couldn't read .* again") as ctx:
                rbdb.backup(self.db)
        self.assertNotIsInstance(ctx.exception, rbdb.LibraryChanged)
        self.assertEqual(self.app_files(), [])


class StillLibraryTests(LiveLibraryCase):
    def test_preview_of_a_library_with_a_log_works_and_skips_the_index(self):
        self.rb.credit_songs(self.first[:4])          # still only in the log
        copied = []
        real = rbdb._copy_hashed

        def record(src, dst):
            copied.append(os.path.basename(src))
            return real(src, dst)
        changes = organizer.Changes()
        with mock.patch.object(rbdb, "_copy_hashed", record), self.watching_copies() as reads:
            organizer.run(by_bpm(), preview=True, changes=changes, **QUIET)
        self.assertEqual(copied, ["master.db", "master.db-wal"])     # no -shm
        self.assertEqual(len(reads), 1)
        self.assertFalse(changes.empty())

    def test_backup_of_a_still_library_works_without_the_index(self):
        saved = rbdb.backup(self.db)
        self.assertEqual(self.backups(), [saved])
        self.assertTrue(os.path.exists(saved + "-wal"))
        self.assertFalse(os.path.exists(saved + "-shm"))


class ShmIsIgnoredTests(DatabaseCase):
    def test_reads_updating_the_index_dont_stop_a_backup(self):
        shm = self.db + "-shm"
        with open(shm, "wb") as f:
            f.write(b"\0" * 32768)
        real = rbdb._copy_verified
        calls = []

        def copy_then_touch_index(src, dst):
            out = real(src, dst)
            calls.append(src)
            with open(shm, "r+b") as f:        # what a program merely reading does
                f.write(os.urandom(64))
            return out
        with mock.patch.object(rbdb, "_copy_verified", copy_then_touch_index):
            saved = rbdb.backup(self.db)
        self.assertEqual(self.backups(), [saved])
        self.assertNotIn(shm, calls)


class UnfinishedBackupTests(DatabaseCase):
    """A backup that didn't pass every check is never offered by Undo."""

    def assert_no_backup(self):
        self.assertEqual(self.backups(), [])
        self.assertEqual(rbdb._newest_first(rbdb._matching(self.dir, "master.before-undo-*.db")), [])

    def test_the_running_check_failing_after_the_copy(self):
        with mock.patch.object(rbdb, "rekordbox_running",
                               side_effect=RuntimeError("couldn't list processes")):
            with self.assertRaisesRegex(RuntimeError, "couldn't list processes"):
                rbdb.backup(self.db)
        self.assert_no_backup()
        self.assertEqual(sorted(os.listdir(self.dir)), ["master.db"])

    def test_the_running_check_failing_after_undos_copy(self):
        saved = rbdb.backup(self.db)
        answers = iter([False])

        def running():
            try:
                return next(answers)
            except StopIteration:
                raise RuntimeError("couldn't list processes") from None
        with mock.patch.object(rbdb, "rekordbox_running", running):
            with self.assertRaisesRegex(RuntimeError, "couldn't list processes"):
                rbdb.restore_backup(saved, self.db)
        self.assertEqual(self.backups(), [saved])
        self.assertEqual([n for n in os.listdir(self.dir) if "before-undo" in n], [])

    def test_a_piece_that_cant_be_removed_doesnt_hide_the_reason(self):
        with open(self.db + "-wal", "wb") as f:
            f.write(b"journal")
        real = rbdb._remove
        refused = []

        def held(path):
            if not refused:
                refused.append(os.path.basename(path))
                raise PermissionError(13, "in use", path)
            return real(path)
        with mock.patch.object(rbdb, "_changed_since", return_value=True), \
                mock.patch.object(rbdb, "_remove", held):
            with self.assertRaises(rbdb.LibraryChanged) as ctx:
                rbdb.backup(self.db)
        self.assertIn("changed while the app was copying it", str(ctx.exception))
        self.assertIn("Couldn't remove " + refused[0], " ".join(ctx.exception.__notes__))
        self.assert_no_backup()
        # The next backup clears what's left, and is listed.
        saved = rbdb.backup(self.db)
        self.assertEqual(self.backups(), [saved])
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith((".unchecked", ".partial"))])


class UndoNeverLosesAChangeTests(LiveLibraryCase):
    """A write in the window between Undo's last check and the swap, or to the old
    files during it, is never lost: Undo stops and the library is put back with it."""

    def setUp(self):
        super().setUp()
        self.saved = rbdb.backup(self.db)
        self.rb.close()                      # rekordbox closed, as Undo asks
        self.before = self.library_files()

    def assert_put_back_with_the_change(self, ctx, late=5):
        self.assertIn("nothing was lost", str(ctx.exception))
        self.assert_library_back(late)

    def assert_library_back(self, late=5):
        self.assertEqual(self.artists("late"), late)
        self.assertEqual(self.artists("first"), 1500)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.backups(), [self.saved])
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(
            (".old", ".new", ".restored", ".found", ".unchecked", ".partial"))])
        self.assert_healthy()

    def test_a_write_just_before_the_swap(self):
        real = rbdb.begin_change

        def write_then_begin(*a, **k):        # after every check, before any file moves
            write_and_quit(self.db)
            return real(*a, **k)
        with mock.patch.object(rbdb, "begin_change", write_then_begin):
            with self.assertRaises(rbdb.LibraryChanged) as ctx:
                rbdb.restore_backup(self.saved, self.db)
        self.assert_put_back_with_the_change(ctx)

    def test_a_write_to_the_old_files_during_the_swap(self):
        # What a program still holding the old files does on a Mac: its writes go to
        # them, now under their second name.
        def write_to_the_old_files(step):
            if step == "database in":
                old = [n for n in os.listdir(self.dir) if n.endswith(".db.old")]
                write_and_quit(os.path.join(self.dir, old[0]))
        with mock.patch.object(rbdb, "_after_step", write_to_the_old_files):
            with self.assertRaises(rbdb.LibraryChanged) as ctx:
                rbdb.restore_backup(self.saved, self.db)
        self.assert_put_back_with_the_change(ctx)

    def test_rekordbox_starting_during_the_swap(self):
        # It may have the files open, so Undo moves nothing more: the files stay where
        # they are and the marker says so, until rekordbox is closed.
        def opens(step):
            if step == "database in":
                self.running = True
        with mock.patch.object(rbdb, "_after_step", opens):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "rekordbox was opened while Undo"
                                        ".*left every file where it is") as ctx:
                rbdb.restore_backup(self.saved, self.db)
        self.assertIn("press Undo again", str(ctx.exception))
        self.assertEqual(rbdb.interrupted_change(self.db)["what"], "undo")
        self.assertEqual(rbdb._sha256(self.db), rbdb._sha256(self.saved))   # not moved back
        with self.assertRaises(rbdb.LibraryChanged):
            rbdb.recover_interrupted_undo(self.db)       # still open: still nothing moves
        self.running = False
        self.assertEqual(rbdb.recover_interrupted_undo(self.db), [])
        self.assertEqual(self.library_files(), self.before)        # exactly as it was
        self.assert_library_back(late=0)

    def test_a_good_undo_keeps_the_old_files_as_the_saved_copy(self):
        keep = rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(rbdb._sha256(self.db), rbdb._sha256(self.saved))
        self.assertEqual(rbdb._sha256(keep), self.before["master.db"])
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertFalse([n for n in os.listdir(self.dir)
                          if n.endswith((".old", ".new", ".restored", ".found"))])
        self.assert_healthy()


class UndoWithRekordboxStillOpenTests(LiveLibraryCase):
    def test_its_writes_are_never_lost(self):
        """rekordbox keeps the library open through Undo, unnoticed by name. On
        Windows the swap itself is refused; on a Mac, Undo sees that a program has
        the library open (lsof) and stops before moving anything. Either way: no loss."""
        saved = rbdb.backup(self.db)
        wrote = []

        def write(step):
            if step == "database in":
                self.rb.add_artists(5, "late")       # into the files it still has open
                wrote.append(1)
        with mock.patch.object(rbdb, "_after_step", write):
            with self.assertRaisesRegex(rbdb.LibraryChanged,
                                        "nothing was lost|before changing anything"):
                rbdb.restore_backup(saved, self.db)
        self.assertEqual(wrote, [])                  # stopped before the swap
        self.rb.close()
        self.assertEqual(self.artists("late"), 5 if wrote else 0)
        self.assertEqual(self.artists("first"), 1500)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assert_healthy()


class RekordboxAgentTests(DatabaseCase):
    @contextlib.contextmanager
    def only_the_agent_running(self):
        with mock.patch.object(rbdb, "running_writers", lambda: ["rekordboxAgent"]), \
                mock.patch.object(rbdb, "rekordbox_running", lambda: True):
            yield

    def test_changes_wait_for_it_and_say_how(self):
        digest = sha(self.db)
        with self.only_the_agent_running():
            with self.assertRaisesRegex(ValueError, "rekordboxAgent.*is running.*quit it.*"
                                                    "press Run again"):
                organizer.run(by_bpm(), **QUIET)
            with self.assertRaisesRegex(ValueError, "rekordboxAgent"):
                organizer.save_genres({"2": "House"}, **QUIET)
            with self.assertRaisesRegex(rbdb.LibraryChanged, "rekordboxAgent"):
                rbdb.restore_backup(os.path.join(self.dir, "master.organizer-backup-x.db"),
                                    self.db)
        self.assertEqual(sha(self.db), digest)
        self.assertEqual(self.backups(), [])

    def test_reading_a_checked_copy_doesnt_need_it_closed(self):
        with self.only_the_agent_running():
            changes = organizer.Changes()
            organizer.run(by_bpm(), preview=True, changes=changes, **QUIET)
        self.assertFalse(changes.empty())

    def test_both_are_named(self):
        with mock.patch.object(rbdb, "running_writers", lambda: ["rekordbox", "rekordboxAgent"]):
            self.assertIn("close rekordbox and quit rekordboxAgent", rbdb.open_message())


class ProcessCheckTests(unittest.TestCase):
    def test_processes_are_matched_by_exact_name(self):
        import psutil

        class Proc:
            def __init__(self, name):
                self._name = name

            def name(self):
                if isinstance(self._name, Exception):
                    raise self._name
                return self._name
        procs = [Proc("explorer.exe"), Proc(psutil.AccessDenied()), Proc("rekordboxAgent"),
                 Proc("rekordbox.exe"), Proc("rekordbox-helper"), Proc(psutil.NoSuchProcess(1))]
        with mock.patch.object(psutil, "process_iter", lambda: iter(procs)):
            self.assertEqual(rbdb.running_writers(), ["rekordbox", "rekordboxAgent"])
            self.assertTrue(rbdb.rekordbox_running())
            self.assertTrue(rbdb.rekordbox_app_running())
        with mock.patch.object(psutil, "process_iter", lambda: iter(procs[2:3])):
            self.assertTrue(rbdb.rekordbox_running())          # the helper alone blocks changes
            self.assertFalse(rbdb.rekordbox_app_running())     # but not reading a checked copy


class XmlNeverTouchesTheDatabaseTests(unittest.TestCase):
    def test_xml_preview_and_run(self):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        self.addCleanup(os.remove, path)
        out = organizer.default_output_path(path)
        self.addCleanup(lambda: os.path.exists(out) and os.remove(out))
        forbidden = ("open_database", "open_preview_copy", "backup", "rekordbox_running",
                     "rekordbox_app_running", "running_writers", "restore_backup")
        with mock.patch.multiple(rbdb, **{n: mock.Mock(side_effect=AssertionError(n))
                                          for n in forbidden}):
            organizer.run(organizer.Options(genre=True), path, preview=True,
                          changes=organizer.Changes(), **QUIET)
            self.assertEqual(organizer.run(organizer.Options(genre=True), path, **QUIET), out)


if __name__ == "__main__":
    unittest.main()
