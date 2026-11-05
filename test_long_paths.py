"""Library folders near Windows' path limit, on synthetic files and libraries only.

Backup names are up to 59 characters longer than master.db, so a library rekordbox and
SQLite can open fine could have backups, or their temporary files, past the limit. That
failed with "[WinError 3] The system cannot find the path specified", naming no file.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import rbdb

HERE = os.path.dirname(os.path.abspath(__file__))
windows_only = unittest.skipUnless(os.name == "nt", "a Windows limit")
LIMIT = 259      # the longest path most Windows calls accept when long paths are off
FULL = {"master.db": b"the database", "masterPlaylists6.xml": b"<index/>"}


def lp(path):
    """The form of a path Windows accepts at any length (independent of rbdb's)."""
    return "\\\\?\\" + os.path.abspath(path) if os.name == "nt" else path


def sha(path):
    with open(lp(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def deep_folder(base, length):
    """A new folder inside base whose path is exactly length characters."""
    path = base
    while len(path) < length:
        room = length - len(path) - 1
        step = min(room, 60)
        if 0 < room - step < 2:      # never leave too little for one more component
            step -= 2
        path = os.path.join(path, "d" * step)
    os.makedirs(path)
    assert len(path) == length
    return path


class Case(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="rbo-long-")
        self.addCleanup(self._clean)

    def _clean(self):
        shutil.rmtree(lp(self.base), ignore_errors=True)

    def library(self, length, name_prefix=""):
        if len(self.base) + 2 > length:
            self.skipTest("the temp folder is already too long")
        folder = deep_folder(self.base, length)
        for name, data in FULL.items():
            write(os.path.join(folder, name), data)
        return folder, os.path.join(folder, "master.db")


@windows_only
class BackupsNearTheLimit(Case):
    # Every library file is reachable; a backup's playlist index and its temporary
    # file aren't, unless the app asks for them in the long form.
    AT_FAILURE = 193     # the folder length of the original failure
    AT_MOST = LIMIT - 1 - len("masterPlaylists6.xml")   # the longest a library can be

    def test_the_lengths_here_really_pass_the_limit(self):
        name = "master.organizer-backup-20260101-000000.masterPlaylists6.xml.partial"
        self.assertEqual(self.AT_FAILURE + 1 + len(name), 262)
        self.assertGreater(self.AT_MOST + 1 + len(name), LIMIT)

    def test_backup_and_list(self):
        for length in (self.AT_FAILURE, self.AT_MOST):
            with self.subTest(length=length):
                folder, db = self.library(length)
                saved = rbdb.backup(db)
                self.assertFalse(saved.startswith("\\\\?\\"))     # callers get a normal path
                self.assertEqual(rbdb.list_backups(folder), [saved])
                self.assertEqual(sha(saved), sha(db))
                self.assertEqual(sha(saved[:-3] + ".masterPlaylists6.xml"),
                                 sha(os.path.join(folder, "masterPlaylists6.xml")))
                self.assertEqual([n for n in os.listdir(lp(folder)) if n.endswith(".partial")], [])
                shutil.rmtree(lp(folder))

    def test_undo_puts_everything_back_and_keeps_the_old_library(self):
        folder, db = self.library(self.AT_MOST)
        saved = rbdb.backup(db)
        write(db, b"changed database")
        write(os.path.join(folder, "masterPlaylists6.xml"), b"<changed/>")
        keep = rbdb.restore_backup(saved, db)
        with open(db, "rb") as f:
            self.assertEqual(f.read(), FULL["master.db"])
        with open(os.path.join(folder, "masterPlaylists6.xml"), "rb") as f:
            self.assertEqual(f.read(), FULL["masterPlaylists6.xml"])
        with open(lp(keep), "rb") as f:
            self.assertEqual(f.read(), b"changed database")
        self.assertIsNone(rbdb.interrupted_change(db))

    def test_old_backups_are_still_tidied_away(self):
        folder, db = self.library(self.AT_MOST)
        for n in range(rbdb.KEEP_BACKUPS + 2):
            rbdb.backup(db)
        self.assertEqual(len(rbdb.list_backups(folder)), rbdb.KEEP_BACKUPS)
        xmls = [n for n in os.listdir(lp(folder)) if n.endswith(".masterPlaylists6.xml")]
        self.assertEqual(len(xmls), rbdb.KEEP_BACKUPS)

    def test_a_cut_off_backups_pieces_are_cleared(self):
        folder, db = self.library(self.AT_MOST)
        stale = os.path.join(folder, "master.organizer-backup-20200101-000000.masterPlaylists6.xml.partial")
        write(lp(stale), b"cut off")
        rbdb.backup(db)
        self.assertFalse(os.path.exists(lp(stale)))

    def test_the_change_marker_works_at_the_limit(self):
        _, db = self.library(self.AT_MOST)
        rbdb.begin_change(db, "backup.db")
        self.assertEqual(rbdb.interrupted_change(db)["backup"], "backup.db")
        rbdb.end_change(db)
        self.assertIsNone(rbdb.interrupted_change(db))

    def test_a_backups_age_can_be_read_without_its_time_in_the_name(self):
        folder, _ = self.library(self.AT_MOST)
        odd = os.path.join(folder, "m" * 25 + ".db")
        write(lp(odd), b"x")
        self.assertGreater(rbdb.backup_time(odd).year, 2000)

    def test_playlist_index_is_saved_whole(self):
        folder, _ = self.library(self.AT_MOST)
        target = os.path.join(folder, "masterPlaylists6.xml")
        xml = mock.Mock(path=target)
        xml.to_string.return_value = "<new/>"
        rbdb._atomic_playlist_xml(xml)
        xml.save()
        with open(target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "<new/>")
        self.assertEqual([n for n in os.listdir(lp(folder)) if n.endswith(".partial")], [])

    def test_a_failed_backup_leaves_no_pieces_and_says_which_file(self):
        folder, db = self.library(self.AT_MOST)
        write(db + "-wal", b"journal")
        real = shutil.copy2

        def fail_on_database(src, dst, **kw):
            if os.path.basename(src) == "master.db":
                raise FileNotFoundError(3, "The system cannot find the path specified")
            return real(src, dst, **kw)

        with mock.patch.object(shutil, "copy2", fail_on_database):
            with self.assertRaises(OSError) as caught:
                rbdb.backup(db)
        self.assertIn("master.organizer-backup-", str(caught.exception))
        self.assertEqual(sorted(os.listdir(lp(folder))),
                         ["master.db", "master.db-wal", "masterPlaylists6.xml"])


class OddFolderNames(Case):
    def test_backups_are_found_in_a_folder_with_glob_characters(self):
        folder = os.path.join(self.base, "My [Music] Library")
        os.makedirs(folder)
        for name, data in FULL.items():
            write(os.path.join(folder, name), data)
        saved = rbdb.backup(os.path.join(folder, "master.db"))
        self.assertEqual(rbdb.list_backups(folder), [saved])

    def test_no_backups_in_a_missing_folder(self):
        self.assertEqual(rbdb.list_backups(os.path.join(self.base, "gone")), [])


class LongForm(unittest.TestCase):
    @windows_only
    def test_forms(self):
        tail = "\\" + "x" * 300 + ".db"
        self.assertEqual(rbdb._long("C:" + tail), "\\\\?\\C:" + tail)
        self.assertEqual(rbdb._long("C:" + tail.replace("\\", "/")), "\\\\?\\C:" + tail)
        self.assertEqual(rbdb._long("\\\\server\\share" + tail), "\\\\?\\UNC\\server\\share" + tail)
        self.assertEqual(rbdb._long("\\\\?\\C:" + tail), "\\\\?\\C:" + tail)

    @windows_only
    def test_paths_within_the_limit_are_left_exactly_as_they_are(self):
        """So a normal library makes the same calls it always did."""
        for path in (r"C:\a\b.db", "C:/a/b.db", r"\\server\share\b.db", "b.db", "C:\\" + "x" * 256):
            self.assertEqual(rbdb._long(path), path)
        self.assertEqual(rbdb._long("C:\\" + "x" * 257), "\\\\?\\C:\\" + "x" * 257)

    def test_unchanged_on_other_systems(self):
        with mock.patch.object(rbdb.os, "name", "posix"):
            self.assertEqual(rbdb._long("/a/b.db"), "/a/b.db")


@windows_only
class WholeApp(Case):
    """The app's own self-test on a synthetic library, at each length that matters."""

    def _selftest(self, folder_length):
        if len(self.base) + 2 > folder_length:
            self.skipTest("the temp folder is already too long")
        parent = deep_folder(self.base, folder_length - 4)
        out = os.path.join(parent, "out")
        self.assertEqual(len(out), folder_length)
        done = subprocess.run([sys.executable, "-B", os.path.join(HERE, "app.py"), "--selftest", out],
                              capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
        with open(os.path.join(out, "selftest.json"), encoding="utf-8") as f:
            report = json.load(f)
        failed = [(r["check"], r["detail"][:300]) for r in report["results"] if r["ok"] is not True]
        self.assertEqual(failed, [])
        self.assertTrue(report["passed"])
        self.assertEqual(done.returncode, 0)
        self.assertIn("backup is the library as it was", [r["check"] for r in report["results"]])

    def test_the_length_that_failed(self):
        self._selftest(185)

    def test_the_longest_library(self):
        self._selftest(BackupsNearTheLimit.AT_MOST - len(os.sep + "library"))


if __name__ == "__main__":
    unittest.main()
