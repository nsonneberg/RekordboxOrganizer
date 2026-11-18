"""Run and Preview on an encrypted library whose files can't be read, and every way a
run can stop between reading and writing.

Synthetic only: a small library (selftest.make_master_db) encrypted with SQLCipher
under a made-up key, as rekordbox's own is, with many artists. Two ways its files can
fail to read with "database disk image is malformed", as rekordbox libraries can:
  - damaged: one page of master.db, holding artist 704823498, is overwritten;
  - mismatched: a healthy master.db next to a master.db-wal from another version of
    it (what an unchecked restore or a mixed copy can leave).
Each can also have later changes waiting in master.db-wal, which SQLite folds into
master.db when its last connection to the file closes. So a run must never connect
to the library before its backup: every connection Organizer opens is recorded at
the DBAPI level (sqlite3 and sqlcipher3), and the library's files are compared byte
for byte.
"""

import contextlib
import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

import organizer
import rbdb
from selftest import make_master_db
from test_library_safety import QUIET, by_bpm

try:
    import sqlcipher3
    from sqlcipher3 import dbapi2 as cipher_dbapi
except ImportError:              # pragma: no cover - a dependency of pyrekordbox
    sqlcipher3 = cipher_dbapi = None

KEY = "402fd" + "0" * 59         # made up: pyrekordbox only checks how it starts
WRONG_KEY = "402fd" + "1" * 59
ARTIST = "704823498"
PAGE = 4096                      # SQLCipher's page size
LIBRARY_FILES = ("master.db", "master.db-wal", "masterPlaylists6.xml")


def same_path(a, b) -> bool:
    """Whether two paths name the same place (symlinks resolved, as macOS's /var)."""
    return (os.path.normcase(os.path.realpath(str(a)))
            == os.path.normcase(os.path.realpath(str(b))))


def connect(path):
    con = sqlcipher3.connect(path)
    con.execute(f"PRAGMA key = '{KEY}'")
    return con


def encrypted_library(folder: str) -> str:
    """folder/library/master.db: encrypted, with 1200 artists and a playlist index."""
    plain_dir = tempfile.mkdtemp(dir=folder)
    plain = make_master_db(plain_dir, playlist_xml=True)
    con = sqlite3.connect(plain)
    con.executemany("INSERT INTO djmdArtist (ID, Name, UUID, rb_local_deleted, created_at, "
                    "updated_at) VALUES (?, ?, ?, 0, '2024-01-01', '2024-01-01')",
                    [(str(704823000 + i), f"Artist {i} " + "x" * 300, f"a{i}")
                     for i in range(1200)])
    con.execute("UPDATE djmdContent SET ArtistID = '704823001'")
    con.execute("UPDATE djmdContent SET ArtistID = ? WHERE ID = '2'", (ARTIST,))
    con.commit()
    con.close()
    lib = os.path.join(folder, "library")
    os.makedirs(lib)
    db = os.path.join(lib, "master.db")
    con = sqlcipher3.connect(plain)
    con.execute("ATTACH DATABASE ? AS enc KEY ?", (db, KEY))
    con.execute("SELECT sqlcipher_export('enc')")
    con.execute("DETACH DATABASE enc")
    con.close()
    con = connect(db)
    con.execute("PRAGMA journal_mode=WAL")          # as rekordbox keeps its library
    con.close()
    shutil.copyfile(os.path.join(plain_dir, "masterPlaylists6.xml"),
                    os.path.join(lib, "masterPlaylists6.xml"))
    shutil.rmtree(plain_dir, True)
    return db


def reads(path) -> str:
    """'ok', or the error reading the songs and artist ARTIST gives."""
    con = connect(path)
    try:
        con.execute("SELECT count(*) FROM djmdCue").fetchone()
        con.execute("SELECT count(*) FROM djmdContent").fetchone()
        con.execute("SELECT Name FROM djmdArtist WHERE ID = ?", (ARTIST,)).fetchone()
        return "ok"
    except sqlcipher3.DatabaseError as exc:
        return str(exc)
    finally:
        con.close()


def damage_the_artists_page(db: str) -> None:
    """Overwrite part of the page holding ARTIST (found by trying each page)."""
    for page in range(1, os.path.getsize(db) // PAGE):     # page 0 holds the header
        trial = db + ".trial"
        shutil.copyfile(db, trial)
        with open(trial, "r+b") as f:
            f.seek(page * PAGE + 2000)
            f.write(b"\xff" * 16)
        if "malformed" in reads(trial):
            os.replace(trial, db)
            return
        os.remove(trial)
    raise AssertionError("no page holds the artist")


def log_from(db: str, statements, base=None) -> None:
    """Give db a master.db-wal holding statements, made on a copy of base (default db)
    and left unfolded, as when rekordbox stops without folding its log in."""
    work = tempfile.mkdtemp(dir=os.path.dirname(os.path.dirname(db)))
    shutil.copyfile(base or db, os.path.join(work, "master.db"))
    con = connect(os.path.join(work, "master.db"))
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA wal_autocheckpoint=0")
        for sql in statements:
            con.execute(sql)
        con.commit()
        shutil.copyfile(os.path.join(work, "master.db-wal"), db + "-wal")
    finally:
        con.close()
    shutil.rmtree(work, True)


LATER = "UPDATE djmdContent SET Title = 'Changed later' WHERE ID = '3'"


def mismatched_log(db: str) -> None:
    """A log from a later version of db (a third of the artists removed and the file
    compacted), next to db as it was: each file is fine on its own."""
    later = db + ".later"
    shutil.copyfile(db, later)
    con = connect(later)
    con.execute("DELETE FROM djmdArtist WHERE CAST(ID AS INTEGER) % 3 = 0")
    con.commit()
    con.execute("VACUUM")
    con.close()
    log_from(db, [f"UPDATE djmdArtist SET Name = Name || 'y' WHERE ID = '{ARTIST}'", LATER],
             base=later)
    os.remove(later)


def build(folder: str, kind: str) -> str:
    os.makedirs(folder)
    db = encrypted_library(folder)
    if kind == "damaged":
        damage_the_artists_page(db)
    elif kind == "damaged-with-log":
        damage_the_artists_page(db)
        log_from(db, [LATER])
    elif kind == "mismatched":
        mismatched_log(db)
        check = os.path.join(folder, "check")       # reading folds the log: use a copy
        os.makedirs(check)
        shutil.copyfile(db, os.path.join(check, "master.db"))
        assert reads(os.path.join(check, "master.db")) == "ok", "on its own it must read"
        shutil.copyfile(db + "-wal", os.path.join(check, "master.db-wal"))
        assert "malformed" in reads(os.path.join(check, "master.db")), "with its log, not"
    elif kind == "healthy-with-log":
        log_from(db, [LATER])
    return os.path.dirname(db)


@unittest.skipIf(sqlcipher3 is None, "needs sqlcipher3 (installed with pyrekordbox)")
class EncryptedLibraryCase(unittest.TestCase):
    KINDS: tuple = ()

    @classmethod
    def setUpClass(cls):
        cls.store = tempfile.mkdtemp(prefix="rbo-cipher-")
        cls.templates = {}
        for kind in cls.KINDS:
            cls.templates[kind] = build(os.path.join(cls.store, kind), kind)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.store, True)

    def use(self, kind: str):
        import pyrekordbox
        root = tempfile.mkdtemp(prefix="rbo-cipher-case-")
        self.addCleanup(shutil.rmtree, root, True)
        self.dir = os.path.join(root, "library")
        shutil.copytree(self.templates[kind], self.dir)
        self.db = os.path.join(self.dir, "master.db")
        # The app's temporary copies go in a folder of this test's own, so nothing
        # else on the computer can add to (or seem to leave behind in) it.
        self.temp = os.path.join(root, "temp")
        os.mkdir(self.temp)
        p = mock.patch.object(tempfile, "tempdir", self.temp)
        p.start()
        self.addCleanup(p.stop)
        real = self.real_master_database = pyrekordbox.MasterDatabase
        self.writers = []
        self.connects = []          # (path, backups there then, marker there then)
        self.recording = False
        for target, attr, value in (
                (pyrekordbox, "MasterDatabase",
                 lambda path=None: real(path or self.db, key=KEY)),   # stands in for finding it
                (rbdb, "running_writers", lambda: list(self.writers)),
                ("pyrekordbox.masterdb.database", "get_rekordbox_pid", lambda: 0)):
            p = mock.patch(f"{target}.{attr}", value) if isinstance(target, str) \
                else mock.patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)
        for module in (cipher_dbapi, sqlite3):
            p = mock.patch.object(module, "connect", self.recorder(module.connect))
            p.start()
            self.addCleanup(p.stop)
        self.labels = os.path.join(root, "labels.json")     # never the app's own
        p = mock.patch("genres.labels_path", lambda: self.labels)
        p.start()
        self.addCleanup(p.stop)
        self.before = self.files()

    def recorder(self, real):
        def connect_(database, *args, **kwargs):
            if self.recording:
                self.connects.append((self.same_file(database),
                                      len(rbdb.list_backups(self.dir)),
                                      rbdb.interrupted_change(self.db) is not None))
            return real(database, *args, **kwargs)
        return connect_

    def same_file(self, path) -> bool:
        try:
            return same_path(path, self.db)
        except (TypeError, ValueError, OSError):
            return False

    def live_connects(self) -> list:
        return [c for c in self.connects if c[0]]

    def run_organizer(self, **kw):
        self.recording = True
        try:
            return organizer.run(by_bpm(), **QUIET, **kw)
        finally:
            self.recording = False

    def save_genres(self, preview=False):
        """Review genres' Save: song 2 (artist ARTIST's) becomes House."""
        self.recording = True
        try:
            return organizer.save_genres({"2": "House"}, preview=preview,
                                         changes=organizer.Changes(), **QUIET)
        finally:
            self.recording = False

    def genre_of(self, track_id: str) -> str:
        db = rbdb.open_database(self.db)
        try:
            return {t.id: t.genre for t in rbdb.load_tracks(db)[0]}[track_id]
        finally:
            db.close()

    @contextlib.contextmanager
    def not_recorded(self):
        """For the test's own stand-in writers: they are not Organizer."""
        was, self.recording = self.recording, False
        try:
            yield
        finally:
            self.recording = was

    def write_to_library(self, title: str):
        with self.not_recorded():
            con = connect(self.db)
            con.execute(f"UPDATE djmdContent SET Title = '{title}' WHERE ID = '4'")
            con.commit()
            con.close()

    def files(self) -> dict:
        out = {}
        for name in sorted(os.listdir(self.dir)):
            with open(os.path.join(self.dir, name), "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
        return out

    def library_files(self) -> dict:
        return {n: d for n, d in self.files().items() if n in LIBRARY_FILES}

    def as_copied(self, files: dict) -> dict:
        """files in the form rbdb's checks take: {library file path: SHA-256}."""
        return {os.path.join(self.dir, n): d for n, d in files.items() if n in LIBRARY_FILES}

    def temp_left(self) -> list:
        """Whatever the app left in this test's temporary folder (nothing, normally)."""
        return sorted(os.listdir(self.temp))

    def assert_untouched(self):
        """Every file byte for byte (none added: not even -shm), no backup or marker,
        no temp copy left, and Organizer never connected to the library."""
        self.assertEqual(self.files(), self.before)
        self.assertEqual(rbdb.list_backups(self.dir), [])
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.temp_left(), [])
        self.assertEqual(self.live_connects(), [])

    def assert_unreadable_message(self, exc):
        message = str(exc)
        self.assertIn("Organizer couldn't read a consistent copy of your rekordbox library",
                      message)
        self.assertIn("database disk image is malformed", message)
        self.assertIn("nothing in it was changed and no backup was made", message)
        self.assertNotIn("damaged", message.lower())       # not established
        self.assertIn("malformed", str(exc.__cause__))


class UnreadableLibraryTests(EncryptedLibraryCase):
    KINDS = ("damaged", "damaged-with-log", "mismatched")

    def check_stops(self, kind, preview=False):
        self.use(kind)
        if kind != "damaged":
            self.assertIn("master.db-wal", self.before)
        with self.assertRaises(ValueError) as ctx:
            self.run_organizer(preview=preview)
        self.assert_unreadable_message(ctx.exception)
        self.assert_untouched()
        self.assertTrue(self.connects, "the copy was read")     # just not the library
        return str(ctx.exception)

    def test_run_damaged_page(self):
        self.assertIn("reading djmdArtist", self.check_stops("damaged"))

    def test_run_damaged_page_with_changes_waiting_in_the_log(self):
        self.assertIn("reading djmdArtist", self.check_stops("damaged-with-log"))

    def test_run_mismatched_log(self):
        self.check_stops("mismatched")

    def check_genre_save_stops(self, kind, preview=False):
        self.use(kind)
        with self.assertRaises(ValueError) as ctx:
            self.save_genres(preview=preview)
        self.assert_unreadable_message(ctx.exception)
        self.assert_untouched()
        self.assertFalse(os.path.exists(self.labels))          # no genre remembered
        return str(ctx.exception)

    def test_genre_save_damaged_page_with_changes_waiting_in_the_log(self):
        self.assertIn("reading djmdArtist",
                      self.check_genre_save_stops("damaged-with-log"))

    def test_genre_save_damaged_page(self):
        self.check_genre_save_stops("damaged")

    def test_genre_save_mismatched_log(self):
        self.check_genre_save_stops("mismatched")

    def test_genre_save_preview(self):
        self.check_genre_save_stops("mismatched", preview=True)

    def test_control_the_old_genre_save_changed_the_library(self):
        """What 1.3.4's genre save did: read the library itself before its backup. The
        failed read's connection, closing, folded the log in and deleted it."""
        self.use("damaged-with-log")
        db = rbdb.open_database()
        try:
            with self.assertRaisesRegex(Exception, "malformed"):
                rbdb.load_tracks(db)
        finally:
            rbdb.close_preview_copy(db, None)          # as its finally did
        self.assertNotIn("master.db-wal", self.files())
        self.assertEqual(rbdb.list_backups(self.dir), [])

    def test_a_wrong_key_is_not_called_damage(self):
        """A copy that can't be decrypted gives "file is not a database": the message
        says that's usually the key or SQLCipher, and still claims no damage."""
        import pyrekordbox
        self.use("damaged")
        real = self.real_master_database

        def wrong_key(path=None):
            return real(path or self.db, key=WRONG_KEY)
        for call in (self.run_organizer, self.save_genres):
            with self.subTest(call.__name__), \
                    mock.patch.object(pyrekordbox, "MasterDatabase", wrong_key):
                with self.assertRaises(ValueError) as ctx:
                    call()
                message = str(ctx.exception)
                self.assertIn("Organizer couldn't read a consistent copy", message)
                self.assertIn('"file is not a database"', message)
                self.assertIn("couldn't be decrypted", message)
                self.assertIn("right key", message)
                self.assertIn("SQLCipher", message)
                self.assertNotIn("damaged", message.lower())
                self.assert_untouched()

    def test_preview_damaged_page(self):
        self.check_stops("damaged-with-log", preview=True)

    def test_preview_mismatched_log(self):
        self.check_stops("mismatched", preview=True)

    def test_a_stable_library_fails_the_same_way_every_time(self):
        """Not a copying race: the same files, unchanged, give the same failure."""
        self.use("damaged-with-log")
        messages = []
        for _ in range(3):
            with self.assertRaises(ValueError) as ctx:
                self.run_organizer()
            messages.append(str(ctx.exception))
        self.assertEqual(len(set(messages)), 1)
        self.assert_untouched()

    def test_control_closing_a_connection_to_the_library_changes_it(self):
        """Why the run must not connect: reading this library directly fails, and
        closing that connection folds the waiting log into master.db and deletes it,
        with no backup. (This is what 1.3.4's Run did.)"""
        self.use("damaged-with-log")
        db = rbdb.open_database(self.db)
        with self.assertRaisesRegex(Exception, "malformed"):
            rbdb.load_tracks(db)
        db.close()
        after = self.files()
        self.assertNotIn("master.db-wal", after)
        self.assertNotEqual(after["master.db"], self.before["master.db"])

    def test_locating_the_library_never_connects(self):
        self.use("damaged-with-log")
        self.recording = True
        self.assertTrue(same_path(rbdb.locate_library(), self.db))
        self.recording = False
        self.assertEqual(self.connects, [])
        self.assertEqual(self.files(), self.before)


class HealthyLibraryTests(EncryptedLibraryCase):
    KINDS = ("healthy-with-log",)

    def setUp(self):
        self.use("healthy-with-log")

    def assert_nothing_written(self, library=None):
        """No backup, marker or temp copy; Organizer never connected to the library;
        its files are as they were (or as the test's own writer left them)."""
        self.assertEqual(self.library_files(), library or
                         {n: d for n, d in self.before.items() if n in LIBRARY_FILES})
        self.assertEqual(rbdb.list_backups(self.dir), [])
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.temp_left(), [])
        self.assertEqual(self.live_connects(), [])

    def test_run_connects_to_the_library_only_after_a_checked_backup(self):
        self.run_organizer()
        live = self.live_connects()
        self.assertTrue(live)
        for _, backups, marker in live:
            self.assertEqual(backups, 1)
            self.assertTrue(marker)
        saved = rbdb.list_backups(self.dir)
        self.assertEqual(len(saved), 1)
        self.assertTrue(rbdb.backup_is_of(saved[0], self.db, self.as_copied(self.before)))
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(reads(self.db), "ok")

    def test_a_change_while_it_is_copied_stops_it(self):
        real, done = rbdb._copy_hashed, []

        def copy_while_rekordbox_saves(src, dst):
            digest = real(src, dst)
            if not done:
                done.append(1)
                self.write_to_library("Saved mid-copy")
            return digest
        with mock.patch.object(rbdb, "_copy_hashed", copy_while_rekordbox_saves):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "changed while the app was copying"):
                self.run_organizer()
        self.assertEqual(rbdb.list_backups(self.dir), [])
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.live_connects(), [])

    def test_rekordbox_or_its_helper_running_stops_it_before_anything(self):
        for running in (["rekordbox"], ["rekordboxAgent"]):
            with self.subTest(running):
                self.writers[:] = running
                with mock.patch.object(rbdb, "open_database",
                                       side_effect=AssertionError("located or opened")):
                    with self.assertRaisesRegex(ValueError, running[0]):
                        self.run_organizer()
                self.assert_nothing_written()

    def test_failing_while_planning_changes_nothing(self):
        with mock.patch.object(organizer, "_organize", side_effect=RuntimeError("planning")):
            with self.assertRaisesRegex(RuntimeError, "planning"):
                self.run_organizer()
        self.assert_nothing_written()

    def test_rekordbox_starting_before_the_backup_changes_nothing(self):
        real = organizer._plan_run

        def plan_then_rekordbox_starts(*a, **kw):
            out = real(*a, **kw)
            self.writers[:] = ["rekordboxAgent"]
            return out
        with mock.patch.object(organizer, "_plan_run", plan_then_rekordbox_starts):
            with self.assertRaisesRegex(ValueError, "Nothing was changed"):
                self.run_organizer()
        self.assert_nothing_written()

    def test_a_change_after_it_was_read_stops_it_before_the_backup(self):
        real = organizer._plan_run

        def plan_then_rekordbox_saves(*a, **kw):
            out = real(*a, **kw)
            self.write_to_library("Saved after reading")
            return out
        with mock.patch.object(organizer, "_plan_run", plan_then_rekordbox_saves):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "changed after Organizer read it"):
                self.run_organizer()
        self.assertNotEqual(self.library_files(), {n: d for n, d in self.before.items()
                                                   if n in LIBRARY_FILES})
        self.assert_nothing_written(library=self.library_files())

    def test_a_change_just_before_the_backup_is_caught_by_the_backups_record(self):
        real = rbdb.backup

        def rekordbox_saves_then_backup(db_path, **kw):
            self.write_to_library("Saved just before the backup")
            return real(db_path, **kw)
        with mock.patch.object(rbdb, "backup", rekordbox_saves_then_backup):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "is kept"):
                self.run_organizer()
        self.assertEqual(len(rbdb.list_backups(self.dir)), 1)      # of the library as it is
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.live_connects(), [])

    def test_a_change_after_the_backup_stops_it_before_the_library_is_opened(self):
        real = rbdb.begin_change

        def marker_then_rekordbox_saves(*a, **kw):
            out = real(*a, **kw)
            self.write_to_library("Saved after the backup")
            return out
        with mock.patch.object(rbdb, "begin_change", marker_then_rekordbox_saves):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "changed after Organizer read it"):
                self.run_organizer()
        saved = rbdb.list_backups(self.dir)
        self.assertEqual(len(saved), 1)                     # of the library as it was read
        self.assertTrue(rbdb.backup_is_of(saved[0], self.db, self.as_copied(self.before)))
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.live_connects(), [])

    def test_reading_the_library_itself_failing_after_the_backup_is_explained(self):
        from sqlalchemy.exc import DatabaseError
        real = rbdb.load_tracks

        def library_unreadable(db):
            if same_path(rbdb.database_path(db), self.db):
                raise DatabaseError("SELECT ...", {},
                                    Exception("database disk image is malformed"))
            return real(db)
        with mock.patch.object(rbdb, "load_tracks", library_unreadable):
            with self.assertRaises(ValueError) as ctx:
                self.run_organizer()
        saved = rbdb.list_backups(self.dir)
        self.assertEqual(len(saved), 1)
        message = str(ctx.exception)
        self.assertIn(f"backed up your library to {os.path.basename(saved[0])}", message)
        self.assertIn('"database disk image is malformed"', message)
        self.assertIn("nothing was changed", message)
        self.assertIn("use Undo to put that backup back", message)
        self.assertIn("Please don't press Run again", message)
        self.assertNotIn("SELECT", message)                    # no raw query
        self.assertIsNone(rbdb.interrupted_change(self.db))    # nothing was saved
        self.assertTrue(rbdb.backup_is_of(saved[0], self.db, self.as_copied(self.before)))
        rbdb.restore_backup(saved[0], self.db)
        self.assertEqual(self.library_files(),
                         {n: d for n, d in self.before.items() if n in LIBRARY_FILES})

    def make_older_backups(self) -> list:
        for _ in range(rbdb.KEEP_BACKUPS):
            rbdb.backup(self.db)
        older = rbdb.list_backups(self.dir)
        self.assertEqual(len(older), rbdb.KEEP_BACKUPS)
        return older

    def test_a_backup_of_a_changed_library_never_pushes_out_an_older_one(self):
        older = self.make_older_backups()
        real = rbdb.backup

        def rekordbox_saves_then_backup(db_path, **kw):
            self.write_to_library("Saved just before the backup")
            return real(db_path, **kw)
        with mock.patch.object(rbdb, "backup", rekordbox_saves_then_backup):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "is kept"):
                self.run_organizer()
        now = rbdb.list_backups(self.dir)
        self.assertEqual(len(now), rbdb.KEEP_BACKUPS + 1)
        for path in older:                              # all still there, and intact
            self.assertIn(path, now)
            self.assertTrue(rbdb.backup_is_of(path, self.db, self.as_copied(self.before)))
        self.assertEqual(self.live_connects(), [])

    def test_control_a_matching_backup_still_prunes_as_usual(self):
        older = self.make_older_backups()
        self.run_organizer()
        now = rbdb.list_backups(self.dir)
        self.assertEqual(len(now), rbdb.KEEP_BACKUPS)
        self.assertNotIn(older[-1], now)                # the oldest went

    def test_failing_after_the_backup_before_the_first_write_is_recoverable(self):
        with mock.patch.object(rbdb, "write_changes", side_effect=RuntimeError("before writing")):
            with self.assertRaisesRegex(RuntimeError, "before writing"):
                self.run_organizer()
        saved = rbdb.list_backups(self.dir)
        self.assertEqual(len(saved), 1)
        self.assertTrue(all(b == 1 and m for _, b, m in self.live_connects()))
        self.assertIsNone(rbdb.interrupted_change(self.db))        # nothing was saved
        # Closing may have folded the log in, but the checked backup holds the library
        # exactly as it was, and Undo puts it back byte for byte.
        self.assertTrue(rbdb.backup_is_of(saved[0], self.db, self.as_copied(self.before)))
        rbdb.restore_backup(saved[0], self.db)
        self.assertEqual(self.library_files(),
                         {n: d for n, d in self.before.items() if n in LIBRARY_FILES})


class HealthyGenreSaveTests(EncryptedLibraryCase):
    KINDS = ("healthy-with-log",)

    def setUp(self):
        self.use("healthy-with-log")

    assert_nothing_written = HealthyLibraryTests.assert_nothing_written

    def test_saves_and_connects_only_after_a_checked_backup(self):
        self.save_genres()
        live = self.live_connects()
        self.assertTrue(live)
        self.assertTrue(all(b == 1 and m for _, b, m in live))
        saved = rbdb.list_backups(self.dir)
        self.assertEqual(len(saved), 1)
        self.assertTrue(rbdb.backup_is_of(saved[0], self.db, self.as_copied(self.before)))
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.genre_of("2"), "House")
        self.assertTrue(os.path.exists(self.labels))           # remembered once saved

    def test_preview_changes_nothing_and_returns_the_librarys_path(self):
        self.assertTrue(same_path(self.save_genres(preview=True), self.db))
        self.assert_nothing_written()

    def test_a_change_after_it_was_read_stops_it_before_the_backup(self):
        real = organizer._change_library

        def change_library(log, plan, save, again):
            def plan_then_rekordbox_saves(*a):
                out = plan(*a)
                self.write_to_library("Saved after reading")
                return out
            return real(log, plan_then_rekordbox_saves, save, again)
        with mock.patch.object(organizer, "_change_library", change_library):
            with self.assertRaisesRegex(rbdb.LibraryChanged,
                                        "changed after Organizer read it.*then save again"):
                self.save_genres()
        self.assert_nothing_written(library=self.library_files())
        self.assertFalse(os.path.exists(self.labels))

    def test_rekordbox_starting_before_the_backup_changes_nothing(self):
        real = organizer._load_copy

        def load_then_rekordbox_starts(db):
            out = real(db)
            self.writers[:] = ["rekordboxAgent"]
            return out
        with mock.patch.object(organizer, "_load_copy", load_then_rekordbox_starts):
            with self.assertRaisesRegex(ValueError, "Nothing was changed - close it and save "
                                                    "again"):
                self.save_genres()
        self.assert_nothing_written()

    def test_a_change_just_before_the_backup_is_caught_by_the_backups_record(self):
        real = rbdb.backup

        def rekordbox_saves_then_backup(db_path, **kw):
            self.write_to_library("Saved just before the backup")
            return real(db_path, **kw)
        with mock.patch.object(rbdb, "backup", rekordbox_saves_then_backup):
            with self.assertRaisesRegex(rbdb.LibraryChanged, "is kept"):
                self.save_genres()
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertEqual(self.live_connects(), [])

    def test_failing_after_the_backup_before_the_first_write_is_recoverable(self):
        with mock.patch.object(rbdb, "write_genres", side_effect=RuntimeError("before writing")):
            with self.assertRaisesRegex(RuntimeError, "before writing"):
                self.save_genres()
        saved = rbdb.list_backups(self.dir)
        self.assertEqual(len(saved), 1)
        self.assertIsNone(rbdb.interrupted_change(self.db))
        self.assertFalse(os.path.exists(self.labels))
        self.assertTrue(rbdb.backup_is_of(saved[0], self.db, self.as_copied(self.before)))
        rbdb.restore_backup(saved[0], self.db)
        self.assertEqual(self.library_files(),
                         {n: d for n, d in self.before.items() if n in LIBRARY_FILES})


if __name__ == "__main__":
    unittest.main()
