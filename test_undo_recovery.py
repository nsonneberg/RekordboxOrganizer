"""Undo never loses a change, and can always be finished or undone after stopping.

Synthetic libraries only (selftest.make_master_db, in temp folders). The library
at Undo time is as rekordbox leaves it: recent changes still in master.db-wal,
plus the playlist index masterPlaylists6.xml. A stand-in writer saves that index
the way careful programs do (write a new file, rename it into place); a real
child process keeps the library open through SQLite; and Undo is stopped, or made
to fail, between every two of its file operations.
"""

import contextlib
import datetime
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import rbdb
from selftest import make_master_db
from test_library_copy import StandInRekordbox
from test_library_safety import PLAYLIST_XML

LIBRARY = ("master.db", "master.db-wal", "masterPlaylists6.xml")
FORWARD_STEPS = ["old database named", "old wal out", "database in", "log in",
                 "old playlist index out", "playlist index in", "validated"]


def sha(path):
    return rbdb._sha256(path)


def crashed_library(folder: str, scratch: str, tag: str, n: int) -> None:
    """Put in folder a library as rekordbox leaves it when it stops: master.db with
    its latest changes still in master.db-wal, and a playlist index."""
    work = tempfile.mkdtemp(dir=scratch)
    db = make_master_db(work)
    rb = StandInRekordbox(db, first_id=704823000 + (0 if tag == "then" else 100000))
    try:
        rb.add_artists(n, tag)
        os.makedirs(folder, exist_ok=True)
        for suffix in ("", "-wal"):
            shutil.copyfile(db + suffix, os.path.join(folder, "master.db" + suffix))
    finally:
        rb.close()
    with open(os.path.join(folder, "masterPlaylists6.xml"), "w", encoding="utf-8") as f:
        f.write(PLAYLIST_XML.replace("</MASTER_PLAYLIST>", f"<!-- {tag} -->\n</MASTER_PLAYLIST>"))


class UndoCase(unittest.TestCase):
    def setUp(self):
        self.running = False
        for name, value in (("rekordbox_running", lambda: self.running),
                            ("running_writers", lambda: ["rekordbox"] if self.running else [])):
            p = mock.patch.object(rbdb, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.make()

    def make(self):
        """A fresh library with a backup of how it was ("then"), now changed ("now")."""
        self.root = tempfile.mkdtemp(prefix="rbo-undo-")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.dir = os.path.join(self.root, "library")
        self.db = os.path.join(self.dir, "master.db")
        crashed_library(self.dir, self.root, "then", 20)
        self.saved = rbdb.backup(self.db)
        self.backup_files = {"master.db": sha(self.saved), "master.db-wal": sha(self.saved + "-wal"),
                             "masterPlaylists6.xml": sha(self.saved[:-3] + ".masterPlaylists6.xml")}
        crashed_library(self.dir, self.root, "now", 30)
        self.before = self.library()
        self.assertNotEqual(self.before, self.backup_files)

    def library(self) -> dict:
        return {n: sha(os.path.join(self.dir, n)) for n in LIBRARY
                if os.path.exists(os.path.join(self.dir, n))}

    def leftovers(self) -> list:
        return sorted(n for n in os.listdir(self.dir)
                      if n.endswith((".old", ".new", ".restored", ".found", ".partial",
                                     ".unchecked")) or "-wal.found" in n)

    def query(self, db_path: str, sql: str):
        """Run sql on a copy of the library at db_path (reading one directly would
        fold its log in and change it)."""
        tmp = tempfile.mkdtemp(dir=self.root)
        for suffix in ("", "-wal"):
            if os.path.exists(db_path + suffix):
                shutil.copyfile(db_path + suffix, os.path.join(tmp, "master.db" + suffix))
        con = sqlite3.connect(os.path.join(tmp, "master.db"))
        try:
            return con.execute(sql).fetchone()[0]
        finally:
            con.close()

    def artists(self, tag: str, db_path=None) -> int:
        return self.query(db_path or self.db,
                          f"SELECT count(*) FROM djmdArtist WHERE Name LIKE '{tag} %'")

    def assert_healthy(self, db_path=None):
        self.assertEqual(self.query(db_path or self.db, "PRAGMA integrity_check"), "ok")

    @contextlib.contextmanager
    def at_steps(self, action):
        """Call action(step) at each point between two of Undo's file operations,
        noting any point where master.db was missing."""
        self.missing = []

        def hook(step):
            if not os.path.exists(self.db):
                self.missing.append(step)
            action(step)
        with mock.patch.object(rbdb, "_after_step", hook):
            yield
        self.assertEqual(self.missing, [], "master.db was missing")

    def steps_of(self, *patches) -> list:
        """The steps an Undo goes through (with these patches)."""
        seen = []
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with self.at_steps(seen.append):
                with contextlib.suppress(OSError):
                    rbdb.restore_backup(self.saved, self.db)
        return seen


def save_index_by_rename(folder: str, note: str) -> None:
    """How a careful program saves masterPlaylists6.xml: a new file, renamed in."""
    path = os.path.join(folder, "masterPlaylists6.xml")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(PLAYLIST_XML.replace("</MASTER_PLAYLIST>", f"<!-- {note} -->\n</MASTER_PLAYLIST>"))
    os.replace(path + ".tmp", path)


class SaveByRenameDuringUndoTests(UndoCase):
    def test_every_step_of_the_swap(self):
        self.assertEqual(self.steps_of()[:len(FORWARD_STEPS)], FORWARD_STEPS)
        for step in FORWARD_STEPS:
            with self.subTest(step):
                self.make()
                note = f"saved during Undo, at {step}"

                def save(at, step=step, note=note):
                    if at == step:
                        save_index_by_rename(self.dir, note)
                with self.at_steps(save):
                    try:
                        rbdb.restore_backup(self.saved, self.db)
                        finished = True
                    except rbdb.LibraryChanged as exc:
                        self.assertIn("nothing was lost", str(exc))
                        finished = False
                # The save is where it was put, whatever Undo did.
                with open(os.path.join(self.dir, "masterPlaylists6.xml"), encoding="utf-8") as f:
                    self.assertIn(note, f.read())
                self.assertIsNone(rbdb.interrupted_change(self.db))
                self.assertEqual(self.leftovers(), [])
                expected = self.backup_files if finished else self.before
                self.assertEqual(self.library()["master.db"], expected["master.db"])
                self.assertEqual(finished, step == "validated")
                self.assert_healthy()


class InterruptionTests(UndoCase):
    """Undo stopped (a crash) or failing between any two file operations: master.db
    is always there, and the next Undo puts everything right first."""

    def check_recovery(self, expected):
        self.assertTrue(os.path.exists(self.db))
        info = rbdb.interrupted_change(self.db)
        self.assertEqual(info["what"], "undo")
        self.assertIn("press 'Put my library back'", rbdb.interrupted_message(info))
        self.assertEqual(rbdb.recover_interrupted_undo(self.db), [])
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.library(), expected)
        self.assertEqual(self.leftovers(), [])
        self.assert_healthy()
        rbdb.restore_backup(self.saved, self.db)            # and Undo works again
        self.assertEqual(self.library(), self.backup_files)

    def stop_at(self, step, error):
        def stop(at):
            if at == step:
                raise error
        return stop

    def test_stopping_at_each_step_of_the_swap(self):
        for step in FORWARD_STEPS:
            with self.subTest(step):
                self.make()
                with self.at_steps(self.stop_at(step, KeyboardInterrupt())):
                    with self.assertRaises(KeyboardInterrupt):
                        rbdb.restore_backup(self.saved, self.db)
                # Finished if it had passed its checks; otherwise undone, exactly.
                self.check_recovery(self.backup_files if step == "validated" else self.before)

    def rollback_steps(self):
        steps = self.steps_of(mock.patch.object(rbdb, "_undo_problem", return_value="Simulated."))
        rollback = [s for s in steps if s not in FORWARD_STEPS]
        for expected in ("restored log out", "database back", "wal back",
                         "restored playlist index out", "playlist index back"):
            self.assertIn(expected, rollback)
        return rollback

    def test_stopping_at_each_step_of_putting_it_back(self):
        for step in self.rollback_steps():
            with self.subTest(step):
                self.make()
                with mock.patch.object(rbdb, "_undo_problem", return_value="Simulated."), \
                        self.at_steps(self.stop_at(step, KeyboardInterrupt())):
                    with self.assertRaises(KeyboardInterrupt):
                        rbdb.restore_backup(self.saved, self.db)
                self.check_recovery(self.before)

    def test_failing_at_each_step_of_putting_it_back(self):
        for step in self.rollback_steps():
            with self.subTest(step):
                self.make()
                with mock.patch.object(rbdb, "_undo_problem", return_value="Simulated."), \
                        self.at_steps(self.stop_at(step, PermissionError(13, "in use"))):
                    with self.assertRaisesRegex(OSError, "couldn't put your library back yet.*"
                                                         "Nothing has been deleted.*press Undo "
                                                         "again") as ctx:
                        rbdb.restore_backup(self.saved, self.db)
                self.assertNotIsInstance(ctx.exception, rbdb.LibraryChanged)
                self.check_recovery(self.before)

    def test_nothing_moves_while_rekordbox_is_open(self):
        with self.at_steps(self.stop_at("database in", KeyboardInterrupt())):
            with self.assertRaises(KeyboardInterrupt):
                rbdb.restore_backup(self.saved, self.db)
        mid_way = sorted(os.listdir(self.dir))
        self.running = True
        with self.assertRaisesRegex(rbdb.LibraryChanged, "rekordbox is open"):
            rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(sorted(os.listdir(self.dir)), mid_way)
        self.running = False
        self.check_recovery(self.before)


class BackupIdentityTests(UndoCase):
    def test_a_failed_backups_piece_is_never_taken_into_a_later_one(self):
        fixed = datetime.datetime(2026, 1, 1, 12, 0, 0)

        class Clock(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed
        real_remove = rbdb._remove

        def log_piece_stuck(path):
            if path.endswith("-wal") and "organizer-backup" in path:
                raise PermissionError(13, "in use", path)
            real_remove(path)
        with mock.patch.object(rbdb.datetime, "datetime", Clock):
            with mock.patch.object(rbdb, "_changed_since", return_value=True), \
                    mock.patch.object(rbdb, "_remove", log_piece_stuck):
                with self.assertRaises(rbdb.LibraryChanged):
                    rbdb.backup(self.db)
            stuck = os.path.join(self.dir, "master.organizer-backup-20260101-120000.db-wal")
            self.assertTrue(os.path.exists(stuck))
            # The library then loses its log (rekordbox folded it in) and is backed up
            # again in the same second.
            con = sqlite3.connect(self.db)
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()
            self.assertFalse(os.path.exists(self.db + "-wal"))
            now = self.library()
            again = rbdb.backup(self.db)
        self.assertNotEqual(os.path.basename(again), "master.organizer-backup-20260101-120000.db")
        self.assertFalse(os.path.exists(again + "-wal"))
        rbdb.restore_backup(again, self.db)
        self.assertEqual(self.library(), now)               # exactly what was backed up
        self.assertEqual(self.artists("now"), 30)
        self.assert_healthy()

    def test_a_stray_file_beside_a_backup_is_not_restored_with_it(self):
        con = sqlite3.connect(self.db)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
        now = self.library()
        saved = rbdb.backup(self.db)                        # no log in it
        shutil.copyfile(self.saved + "-wal", saved + "-wal")   # a log from another moment
        crashed_library(self.dir, self.root, "later", 5)
        rbdb.restore_backup(saved, self.db)
        self.assertEqual(self.library(), now)
        self.assertEqual(self.artists("then"), 0)
        self.assert_healthy()

    def refused(self, damage, message):
        damage()
        with self.assertRaisesRegex(OSError, message):
            rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library(), self.before)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.leftovers(), [])

    def test_a_backup_file_changed_since_is_refused(self):
        def change():
            with open(self.saved[:-3] + ".masterPlaylists6.xml", "a", encoding="utf-8") as f:
                f.write("\n")
        self.refused(change, "doesn't match the record saved with the backup.*Nothing was changed")

    def test_a_damaged_record_is_refused(self):
        def damage():
            with open(self.saved[:-3] + ".manifest.json", "w", encoding="utf-8") as f:
                f.write("{not json")
        self.refused(damage, "record saved with .* is damaged")

    def test_a_missing_piece_is_refused(self):
        self.refused(lambda: os.remove(self.saved + "-wal"), "files are missing")

    def test_a_missing_backup_from_before_records_is_refused(self):
        def gone():
            for part in rbdb._backup_parts(self.saved):
                os.remove(part) if os.path.exists(part) else None
        self.refused(gone, "is missing, so it wasn't restored. Nothing was changed")

    def test_a_backup_from_before_records_still_restores(self):
        os.remove(self.saved[:-3] + ".manifest.json")
        rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library(), self.backup_files)


CHILD = r"""
import sqlite3, sys
con = sqlite3.connect(sys.argv[1], isolation_level=None)
con.execute("PRAGMA journal_mode=WAL")
con.execute("PRAGMA wal_autocheckpoint=0")
if sys.argv[2:] == ["quiet"]:                    # open the files, write nothing yet
    con.execute("SELECT count(*) FROM sqlite_master").fetchone()
else:
    con.execute("CREATE TABLE IF NOT EXISTS child_rows (x)")
    con.execute("INSERT INTO child_rows VALUES ('start')")
print("ready", flush=True)
for line in sys.stdin:
    if line.strip() == "write":
        con.execute("CREATE TABLE IF NOT EXISTS child_rows (x)")
        con.execute("INSERT INTO child_rows VALUES ('write')")
        print("ok", flush=True)
    elif line.strip() == "quit":
        break
con.close()
print("closed", flush=True)
"""


class OpenByAnotherProcessTests(UndoCase):
    """A real second process keeping the library open through SQLite, as rekordbox
    does: it's reported as running once `detectable` is set."""

    def setUp(self):
        super().setUp()
        self.children = []
        self.detectable = False
        alive = lambda: self.detectable and any(c.poll() is None for c in self.children)  # noqa: E731
        for name, value in (("rekordbox_running", alive),
                            ("running_writers", lambda: ["rekordbox"] if alive() else [])):
            p = mock.patch.object(rbdb, name, value)
            p.start()
            self.addCleanup(p.stop)

    def open_library(self, path, quiet=False):
        child = subprocess.Popen([sys.executable, "-c", CHILD, path] + (["quiet"] if quiet else []),
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.children.append(child)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.assertEqual(child.stdout.readline().strip(), "ready")
        return child

    def tell(self, child, command, answer):
        child.stdin.write(command + "\n")
        child.stdin.flush()
        self.assertEqual(child.stdout.readline().strip(), answer)
        if command == "quit":
            child.wait(30)

    def test_it_had_the_old_files_open(self):
        """rekordbox starts just before the swap. Windows refuses to move files it
        has open; a Mac moves them, so Undo leaves everything where it is. Either way,
        what it writes next ends up in the library once Undo is pressed again."""
        child = self.open_library(self.db)

        def starts(step):
            if step == "old database named":
                self.detectable = True
        with self.at_steps(starts):
            with self.assertRaises(rbdb.LibraryChanged):
                rbdb.restore_backup(self.saved, self.db)
        self.tell(child, "write", "ok")
        self.tell(child, "quit", "closed")
        rbdb.recover_interrupted_undo(self.db)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.query(self.db, "SELECT count(*) FROM child_rows"), 2)
        self.assertEqual(self.artists("now"), 30)
        self.assert_healthy()

    def test_it_opened_the_restored_files_and_something_was_wrong(self):
        """rekordbox opens the restored library just as Undo finds a problem: nothing
        is moved under it. Once it's closed, the next Undo puts the library back and
        keeps the restored file it wrote to."""
        started = []

        def opens(step):
            if step == "playlist index in":
                started.append(self.open_library(self.db))     # writes to it straight away
                self.detectable = True
        with self.at_steps(opens):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "left every file where it is"):
                rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(rbdb.interrupted_change(self.db)["what"], "undo")
        restored = [n for n in os.listdir(self.dir) if n.endswith(".db.restored")]
        self.assertTrue(os.path.samefile(self.db, os.path.join(self.dir, restored[0])))
        self.tell(started[0], "write", "ok")
        self.tell(started[0], "quit", "closed")
        kept = rbdb.recover_interrupted_undo(self.db)
        self.assertEqual(self.library(), self.before)
        evidence = [k for k in kept if k.endswith(".db.restored")]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(self.query(evidence[0], "SELECT count(*) FROM child_rows"), 2)
        self.assert_healthy()


class LateWriteAndLeftoverTests(UndoCase):
    def test_a_late_write_to_the_old_files_keeps_the_saved_copy(self):
        holder = os.path.join(self.root, "held-open.db")
        os.link(self.db, holder)               # stands in for a handle on the old file
        keep = rbdb.restore_backup(self.saved, self.db)
        self.assertTrue(os.path.samefile(keep, holder))     # the old file is the saved copy
        with open(holder, "ab") as f:                        # ...so a late write lands there
            f.write(b"written late")
        self.assertFalse(rbdb._unchanged_copy(keep))
        for _ in range(rbdb.KEEP_BACKUPS + 1):
            rbdb.restore_backup(self.saved, self.db)
        copies = rbdb._matching(self.dir, "master.before-undo-*.db")
        self.assertIn(keep, copies)                          # never pruned
        self.assertEqual(len(copies), rbdb.KEEP_BACKUPS + 1)
        with open(keep, "rb") as f:
            self.assertTrue(f.read().endswith(b"written late"))

    def test_only_certain_duplicates_are_tidied(self):
        keep = rbdb.restore_backup(self.saved, self.db)
        second_name = os.path.join(self.dir, "master.before-undo-20200101-000000.db.old")
        os.link(self.db, second_name)                        # Windows couldn't remove it
        duplicate = keep + ".old"
        shutil.copyfile(keep, duplicate)                     # identical to its saved copy
        unknown = os.path.join(self.dir, "master.before-undo-20200101-000001.db.old")
        with open(unknown, "wb") as f:
            f.write(b"may hold something written during that Undo")
        found = keep[:-3] + ".masterPlaylists6.xml.found"
        with open(found, "w", encoding="utf-8") as f:
            f.write("a save found in the way")
        rbdb.restore_backup(self.saved, self.db)
        self.assertFalse(os.path.exists(second_name))
        self.assertFalse(os.path.exists(duplicate))
        self.assertTrue(os.path.exists(unknown))
        self.assertTrue(os.path.exists(found))


class BeginChangeFailureTests(UndoCase):
    def test_staged_copies_go_and_nothing_else_changes(self):
        with mock.patch.object(rbdb, "begin_change", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library(), self.before)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.leftovers(), [])
        keeps = rbdb._matching(self.dir, "master.before-undo-*.db")
        self.assertEqual(len(keeps), 1)                      # the checked copy stays
        self.assertTrue(rbdb._unchanged_copy(keeps[0]))


class StoppedUndoIsPutRightTests(UndoCase):
    """An Undo cut off part-way is put right from its plan, never just 'kept'."""

    def stop_at(self, step):
        def stop(at):
            if at == step:
                raise KeyboardInterrupt
        with self.at_steps(stop):
            with self.assertRaises(KeyboardInterrupt):
                rbdb.restore_backup(self.saved, self.db)

    def test_keep_as_is_is_refused_after_the_swap_and_after_moving_the_playlist_index(self):
        for step in ("database in", "old playlist index out"):
            with self.subTest(step):
                self.make()
                self.stop_at(step)
                half_way = sorted(os.listdir(self.dir)), self.library()
                if step == "database in":       # the backup's database, without its log
                    self.assertEqual(half_way[1]["master.db"], self.backup_files["master.db"])
                    self.assertNotIn("master.db-wal", half_way[1])
                else:                           # no playlist index at all
                    self.assertNotIn("masterPlaylists6.xml", half_way[1])
                info = rbdb.interrupted_change(self.db)
                self.assertFalse(rbdb.keep_as_is_allowed(info, self.db))
                with self.assertRaisesRegex(OSError, "can't just be kept as it is.*Nothing was "
                                                     "changed"):
                    rbdb.keep_library_as_is(self.db)
                self.assertEqual(rbdb.interrupted_change(self.db), info)    # plan kept
                self.assertEqual((sorted(os.listdir(self.dir)), self.library()), half_way)
                done = rbdb.put_right_stopped_undo(self.db)
                self.assertIn("back as it was before the Undo that didn't finish", done)
                # Exactly the database, log and playlist index from before the Undo.
                self.assertEqual(self.library(), self.before)
                self.assertEqual(set(self.library()), set(LIBRARY))
                self.assertIsNone(rbdb.interrupted_change(self.db))
                self.assertEqual(self.leftovers(), [])
                self.assertEqual(self.artists("now"), 30)
                self.assert_healthy()

    def test_one_that_had_passed_its_checks_is_finished(self):
        self.stop_at("validated")
        done = rbdb.put_right_stopped_undo(self.db)
        self.assertIn("has been completed", done)
        self.assertEqual(self.library(), self.backup_files)
        self.assertIsNone(rbdb.interrupted_change(self.db))

    def test_other_unfinished_changes_can_still_be_kept(self):
        rbdb.begin_change(self.db, self.saved)                      # a run that stopped
        self.assertTrue(rbdb.keep_as_is_allowed(rbdb.interrupted_change(self.db), self.db))
        rbdb.keep_library_as_is(self.db)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.library(), self.before)

    def test_a_damaged_marker_with_undo_files_about_cant_be_kept(self):
        with open(os.path.join(self.dir, rbdb.PENDING), "w") as f:
            f.write("{not json")
        info = rbdb.interrupted_change(self.db)
        self.assertTrue(rbdb.keep_as_is_allowed(info, self.db))
        with open(os.path.join(self.dir, "master.before-undo-20260101-000000.db.old"), "wb"):
            pass
        self.assertFalse(rbdb.keep_as_is_allowed(info, self.db))
        with self.assertRaises(OSError):
            rbdb.keep_library_as_is(self.db)
        self.assertIsNotNone(rbdb.interrupted_change(self.db))

    def test_an_undo_marker_without_a_plan_is_restored_over(self):
        # Left by an older version: how far it got isn't known, so only a whole backup
        # may replace the library.
        rbdb.begin_change(self.db, self.saved, "undo")
        info = rbdb.interrupted_change(self.db)
        self.assertFalse(rbdb.keep_as_is_allowed(info, self.db))
        self.assertIn("restore a backup", rbdb.interrupted_message(info))
        with self.assertRaisesRegex(OSError, "no record of how far"):
            rbdb.put_right_stopped_undo(self.db)
        self.assertEqual(self.library(), self.before)
        rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library(), self.backup_files)


class UndoWindowTests(UndoCase):
    """The Undo window offers only choices that keep the library recoverable."""

    def setUp(self):
        super().setUp()
        import app
        import tk_test_root
        self.app_mod = app
        self.tk = tk_test_root.open_root(self)
        settings = os.path.join(self.root, "settings.json")       # the test's temp folder
        for target, name, value in ((app, "settings_path", lambda: settings),
                                    (app.App, "check_for_update", lambda _: None),
                                    (rbdb, "library_path", lambda: self.db)):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.app = app.App(self.tk)

    def undo_window(self):
        import tkinter as tk
        self.app.undo()
        self.tk.update()
        wins = [w for w in self.tk.winfo_children() if isinstance(w, tk.Toplevel)]
        self.addCleanup(lambda: [w.destroy() for w in wins if w.winfo_exists()])
        buttons, todo = {}, list(wins)
        while todo:
            w = todo.pop()
            todo.extend(w.winfo_children())
            if w.winfo_class() == "TButton":
                buttons[str(w.cget("text"))] = w
        return buttons

    def test_after_an_undo_stopped_part_way(self):
        with self.at_steps(lambda at: (_ for _ in ()).throw(KeyboardInterrupt)
                           if at == "old playlist index out" else None):
            with self.assertRaises(KeyboardInterrupt):
                rbdb.restore_backup(self.saved, self.db)
        buttons = self.undo_window()
        self.assertNotIn("Keep my library as it is", buttons)
        self.assertIn("Put my library back", buttons)
        self.assertIn("Restore", buttons)
        with mock.patch.object(self.app_mod.messagebox, "showinfo") as said:
            buttons["Put my library back"].invoke()
        self.assertIn("back as it was", said.call_args[0][1])
        self.assertEqual(self.library(), self.before)
        self.assertIsNone(rbdb.interrupted_change(self.db))

    def test_after_an_older_versions_undo_stopped(self):
        rbdb.begin_change(self.db, self.saved, "undo")       # no plan: only Restore
        buttons = self.undo_window()
        self.assertNotIn("Keep my library as it is", buttons)
        self.assertNotIn("Put my library back", buttons)
        self.assertIn("Restore", buttons)

    def test_after_another_change_stopped(self):
        rbdb.begin_change(self.db, self.saved)
        buttons = self.undo_window()
        self.assertIn("Keep my library as it is", buttons)
        self.assertNotIn("Put my library back", buttons)


class KeptFilesTests(UndoCase):
    """A file kept because it may hold changes is named in full, and never deleted."""

    def write_to_the_restored_log(self, at):
        if at == "log in":
            with open(self.db + "-wal", "ab") as f:          # someone wrote to it
                f.write(b"written during the Undo")

    def assert_kept(self, message):
        kept = [n for n in os.listdir(self.dir) if n.endswith("-wal.new")]
        self.assertEqual(len(kept), 1)
        path = os.path.join(self.dir, kept[0])
        self.assertIn(path, message)                         # its full path
        self.assertIn("won't delete them", message)
        return path

    def assert_still_there(self, path):
        with open(path, "rb") as f:
            self.assertTrue(f.read().endswith(b"written during the Undo"))

    def test_named_when_undo_puts_the_library_back(self):
        with self.at_steps(self.write_to_the_restored_log):
            with self.assertRaises(rbdb.LibraryChanged) as ctx:
                rbdb.restore_backup(self.saved, self.db)
        path = self.assert_kept(str(ctx.exception))
        self.assertEqual(self.library(), self.before)
        # Nothing later deletes it: more Undos (with tidying and pruning), backups, a
        # stopped Undo put right.
        for _ in range(rbdb.KEEP_BACKUPS + 1):
            rbdb.restore_backup(self.saved, self.db)
        rbdb.backup(self.db)
        with self.at_steps(lambda at: (_ for _ in ()).throw(KeyboardInterrupt)
                           if at == "database in" else None):
            with self.assertRaises(KeyboardInterrupt):
                rbdb.restore_backup(self.saved, self.db)
        rbdb.put_right_stopped_undo(self.db)
        self.assert_still_there(path)

    def test_named_when_a_stopped_undo_is_put_right(self):
        def write_then_stop(at):
            self.write_to_the_restored_log(at)
            if at == "playlist index in":
                raise KeyboardInterrupt
        with self.at_steps(write_then_stop):
            with self.assertRaises(KeyboardInterrupt):
                rbdb.restore_backup(self.saved, self.db)
        done = rbdb.put_right_stopped_undo(self.db)
        path = self.assert_kept(done)
        self.assert_still_there(path)
        self.assertEqual(self.library(), self.before)

    def test_named_when_undo_again_puts_it_right_first(self):
        def write_then_stop(at):
            self.write_to_the_restored_log(at)
            if at == "playlist index in":
                raise KeyboardInterrupt
        with self.at_steps(write_then_stop):
            with self.assertRaises(KeyboardInterrupt):
                rbdb.restore_backup(self.saved, self.db)
        with self.assertRaisesRegex(OSError, "put back as it was before the Undo that didn't "
                                             "finish.*press Undo again") as ctx:
            rbdb.restore_backup(self.saved, self.db)
        path = self.assert_kept(str(ctx.exception))
        self.assertEqual(self.library(), self.before)
        rbdb.restore_backup(self.saved, self.db)            # and then it restores
        self.assertEqual(self.library(), self.backup_files)
        self.assert_still_there(path)


# --- programs that have the library open: the Mac case ----------------------------------
#
# On a Mac, a file can be renamed or replaced while a program has it open; the program
# carries on with it under its new name, and SQLite, closing, deletes master.db-wal and
# -shm by name. Two ways to test that:
#  - on a Mac (macOS CI), a real SQLite process and the real lsof check;
#  - on Windows, a stand-in that opens the files the way a Mac lets every program (the
#    Windows share-delete mode: they can be renamed and deleted under it), a rename
#    that replaces an open file as a Mac does, and closes like SQLite: deleting the log
#    and index by name. It tells the test which files it has open, standing in for lsof.

HOLDER = r"""
import ctypes, json, os, sys
from ctypes import wintypes
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
k32.SetFilePointer.argtypes = [wintypes.HANDLE, ctypes.c_long, ctypes.c_void_p, wintypes.DWORD]
class INFO(ctypes.Structure):
    _fields_ = [("attrs", wintypes.DWORD), ("c", wintypes.FILETIME), ("a", wintypes.FILETIME),
                ("w", wintypes.FILETIME), ("serial", wintypes.DWORD), ("hi", wintypes.DWORD),
                ("lo", wintypes.DWORD), ("links", wintypes.DWORD), ("ihi", wintypes.DWORD),
                ("ilo", wintypes.DWORD)]
def ident(h):
    i = INFO()
    k32.GetFileInformationByHandle(h, ctypes.byref(i))
    return [i.serial, (i.ihi << 32) | i.ilo]
held = {}
for path in sys.argv[1:]:
    h = k32.CreateFileW(path, 0xC0000000, 7, None, 3, 0, None)    # read/write, share all
    if h != wintypes.HANDLE(-1).value:
        held[path] = h
print("ready", flush=True)
for line in sys.stdin:
    command = line.strip()
    if command == "ids":
        print(json.dumps([ident(h) for h in held.values()]), flush=True)
    elif command == "write":
        h = held.get(sys.argv[1] + "-wal") or held[sys.argv[1]]
        k32.SetFilePointer(h, 0, None, 2)
        k32.WriteFile(h, b"written by the holder", 21, ctypes.byref(wintypes.DWORD()), None)
        print("ok", flush=True)
    elif command == "close":
        report = {}
        for path, h in held.items():            # SQLite deletes these by name on closing
            if path.endswith(("-wal", "-shm")):
                try:
                    report[path] = "own" if os.stat(path).st_ino == ident(h)[1] else "other"
                    os.remove(path)
                except FileNotFoundError:
                    report[path] = "none"
        for h in held.values():
            k32.CloseHandle(h)
        print(json.dumps(report), flush=True)
        break
"""


def mac_like_replace(src: str, dst: str) -> None:
    """Rename src over dst even while dst is open, as a Mac does (Windows: a rename
    with POSIX semantics; the open file carries on, nameless or by another name)."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD]
    h = k32.CreateFileW(rbdb._long(src), 0x00010000, 7, None, 3, 0, None)       # DELETE access
    if h == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        name = os.path.abspath(dst).encode("utf-16-le")
        offset = 2 * ctypes.sizeof(ctypes.c_void_p)         # Flags (padded), RootDirectory
        buf = ctypes.create_string_buffer(offset + 4 + len(name) + 2)
        ctypes.c_ulong.from_buffer(buf, 0).value = 0x1 | 0x2     # replace; POSIX semantics
        ctypes.c_ulong.from_buffer(buf, offset).value = len(name)
        ctypes.memmove(ctypes.addressof(buf) + offset + 4, name, len(name))
        if not k32.SetFileInformationByHandle(h, 22, buf, len(buf)):    # FileRenameInfoEx
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        k32.CloseHandle(h)


@unittest.skipUnless(sys.platform == "win32", "stands in for a Mac on Windows")
class MacLikeHolderTests(UndoCase):
    """A program rekordbox doesn't know by name holds the library through Undo, with a
    Mac's file rules (see above)."""

    def setUp(self):
        super().setUp()
        self.holders = []
        for name, value in (("_open_by", self.open_by), ("_replace", mac_like_replace)):
            p = mock.patch.object(rbdb, name, value)
            p.start()
            self.addCleanup(p.stop)

    def hold(self):
        paths = [self.db, self.db + "-wal", self.db + "-shm"]
        child = subprocess.Popen([sys.executable, "-c", HOLDER] + paths, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.assertEqual(child.stdout.readline().strip(), "ready")
        self.holders.append(child)
        return child

    def ask(self, child, command):
        child.stdin.write(command + "\n")
        child.stdin.flush()
        answer = child.stdout.readline().strip()
        if command == "close":
            child.wait(30)
        return answer

    def open_by(self, paths):
        """lsof's answer: which holders have any of these files open."""
        import json
        found = []
        wanted = {os.stat(p).st_ino for p in paths if os.path.exists(p)}     # one volume
        for child in self.holders:
            if child.poll() is None and wanted & {i[1] for i in json.loads(self.ask(child, "ids"))}:
                found.append(f"stand-in (process {child.pid})")
        return found

    def test_one_open_before_undo_stops_it_before_anything_moves(self):
        child = self.hold()
        with self.assertRaisesRegex(rbdb.LibraryChanged, f"process {child.pid}.*has your library "
                                                         "open.*before changing anything"):
            rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library(), self.before)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(json.loads(self.ask(child, "close"))[self.db + "-wal"], "own")

    def test_one_holding_the_old_files_through_the_swap(self):
        """It opened the library just after Undo's first check and wrote nothing, so
        the swap passes its checks. Undo still puts the library back: afterwards the
        program would write to the old files and, closing, delete the restored log."""
        started = []

        def opens(at):
            if at == "old database named":
                started.append(self.hold())
        with self.at_steps(opens):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "has your library's old files "
                                                             "open.*nothing was lost"):
                rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library(), self.before)          # its files, where it expects
        self.assertEqual(self.ask(started[0], "write"), "ok")
        with open(self.db + "-wal", "rb") as f:
            self.assertTrue(f.read().endswith(b"written by the holder"))     # in the library
        self.assertEqual(json.loads(self.ask(started[0], "close"))[self.db + "-wal"], "own")
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.leftovers(), [])

    def test_one_that_opened_the_restored_files(self):
        """Something was wrong after the swap, but a program has the restored files
        open: nothing moves under it until it's closed, then the library goes back
        and what it wrote is kept."""
        started = []

        def opens(at):
            if at == "playlist index in":
                started.append(self.hold())
        with mock.patch.object(rbdb, "_undo_problem", return_value="Simulated."), \
                self.at_steps(opens):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "has the restored files open, so "
                                                             "Undo left every file where it is"):
                rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library()["master.db"], self.backup_files["master.db"])
        with self.assertRaisesRegex(rbdb.LibraryChanged, "Nothing was moved"):
            rbdb.put_right_stopped_undo(self.db)
        self.assertEqual(self.ask(started[0], "write"), "ok")
        self.assertEqual(json.loads(self.ask(started[0], "close"))[self.db + "-wal"], "own")
        done = rbdb.put_right_stopped_undo(self.db)
        self.assertEqual(self.library(), self.before)
        kept = [n for n in os.listdir(self.dir) if n.endswith("-wal.new")]
        self.assertEqual(len(kept), 1)
        self.assertIn(os.path.join(self.dir, kept[0]), done)
        with open(os.path.join(self.dir, kept[0]), "rb") as f:
            self.assertTrue(f.read().endswith(b"written by the holder"))


def _lsof_works() -> bool:
    return sys.platform != "win32" and (shutil.which("lsof") or os.path.exists("/usr/sbin/lsof"))


@unittest.skipUnless(_lsof_works(), "a Mac's file rules and lsof (macOS CI)")
class RealSQLiteHolderTests(OpenByAnotherProcessTests):
    """The same with a real SQLite process and the real lsof check (on a Mac)."""

    def test_one_open_before_undo_stops_it_before_anything_moves(self):
        child = self.open_library(self.db, quiet=True)
        with self.assertRaisesRegex(rbdb.LibraryChanged, f"process {child.pid}.*has your library "
                                                         "open.*before changing anything"):
            rbdb.restore_backup(self.saved, self.db)
        self.assertEqual(self.library(), self.before)
        self.tell(child, "quit", "closed")

    def test_one_holding_the_old_files_through_the_swap(self):
        started = []

        def opens(at):
            if at == "old database named":
                started.append(self.open_library(self.db, quiet=True))
        with self.at_steps(opens):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "has your library's old files open"):
                rbdb.restore_backup(self.saved, self.db)
        self.tell(started[0], "write", "ok")
        self.tell(started[0], "quit", "closed")        # SQLite deletes the log by name
        self.assertEqual(self.query(self.db, "SELECT count(*) FROM child_rows"), 1)
        self.assertEqual(self.artists("now"), 30)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assert_healthy()

    def test_one_that_opened_the_restored_files(self):
        started = []

        def opens(at):
            if at == "playlist index in":
                started.append(self.open_library(self.db, quiet=True))
        with mock.patch.object(rbdb, "_undo_problem", return_value="Simulated."), \
                self.at_steps(opens):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "has the restored files open"):
                rbdb.restore_backup(self.saved, self.db)
        with self.assertRaisesRegex(rbdb.LibraryChanged, "Nothing was moved"):
            rbdb.put_right_stopped_undo(self.db)
        self.tell(started[0], "write", "ok")
        self.tell(started[0], "quit", "closed")
        done = rbdb.put_right_stopped_undo(self.db)
        self.assertEqual(self.artists("now"), 30)
        self.assert_healthy()
        evidence = [p for p in done.split("; ") if ".db.restored" in p or "-wal.new" in p]
        self.assertTrue(evidence, done)


if __name__ == "__main__":
    unittest.main()
