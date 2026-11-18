"""Work directly with rekordbox's own library database (master.db).

No XML export needed: tracks are read straight from rekordbox, and playlists and
star ratings are written straight back. rekordbox must be closed while writing,
and a backup of the database is made before every change.
"""

from __future__ import annotations

import datetime
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from typing import Callable, Optional
from uuid import uuid4

from organizer import (COLOUR_ORDER, SETS_FOLDER, Plan, Track, folder_path,
                       same_name, to_camelot)

KEEP_BACKUPS = 3
PLAYLIST, FOLDER = 0, 1   # DjmdPlaylist.Attribute (smart playlists are 4)
# rekordbox's built-in sampler sounds (older and newer install locations)
SAMPLER_MARKERS = ("/pioneerdj/sampler/", "/rekordbox/sampler/")


def _pyrekordbox():
    import pyrekordbox
    logging.getLogger("pyrekordbox").setLevel(logging.ERROR)
    return pyrekordbox


# Programs that keep the library open and write to it: rekordbox, and its background
# helper rekordboxAgent (which can run without rekordbox's window).
WRITERS = ("rekordbox", "rekordboxAgent")


def running_writers() -> list[str]:
    """Which of WRITERS are running, matched by exact process name (".exe" aside)."""
    import psutil
    _pyrekordbox()
    found = set()
    for proc in psutil.process_iter():
        try:
            name = os.path.splitext(proc.name())[0]
        except psutil.Error:            # gone meanwhile, or not ours to look at
            continue
        if name in WRITERS:
            found.add(name)
    return [name for name in WRITERS if name in found]


def rekordbox_running() -> bool:
    """Whether anything that writes to the library is running: checked before any change."""
    return bool(running_writers())


def rekordbox_app_running() -> bool:
    """Whether rekordbox itself is running: checked before copying the library just to
    read it. rekordboxAgent alone doesn't block that; a write it makes during the copy
    is caught by the copy check (see _check_snapshot)."""
    return "rekordbox" in running_writers()


def _running_names() -> list[str]:
    try:
        return running_writers()
    except Exception:          # a message mustn't fail; fall back to the usual one
        return []


def open_message(then: str = "try again", reading: bool = False) -> str:
    """What to say when rekordbox_running() (or, reading, rekordbox_app_running()) said yes."""
    names = [] if reading else _running_names()
    if names == ["rekordboxAgent"]:
        return ("rekordboxAgent, rekordbox's background helper, is running, and it can change "
                "your library. Please quit it (in Activity Monitor on a Mac, or Task Manager on "
                f"Windows), then {then}.")
    if len(names) == 2:
        return ("rekordbox and its background helper, rekordboxAgent, are running. Please close "
                "rekordbox and quit rekordboxAgent (in Activity Monitor on a Mac, or Task Manager "
                f"on Windows), then {then}.")
    return f"rekordbox is open. Please close it first, then {then}."


def who_started() -> str:
    """The start of a message saying rekordbox (or its helper) started while working."""
    if _running_names() == ["rekordboxAgent"]:
        return "rekordboxAgent, rekordbox's background helper, started"
    return "rekordbox was opened"


def open_database(path: Optional[str] = None):
    pyrekordbox = _pyrekordbox()
    try:
        db = pyrekordbox.MasterDatabase(path) if path else pyrekordbox.MasterDatabase()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Couldn't find your rekordbox library on this computer. "
            "Is rekordbox 6 or 7 installed and has it been opened at least once?"
        ) from exc
    # pyrekordbox's close() ends the session but keeps the file open in a
    # connection pool; let go of it too, so Undo can swap the file afterwards.
    session_close = db.close

    def close():
        try:
            session_close()
        finally:
            db.engine.dispose()
    db.close = close
    # Remember whether a commit was attempted: after that, a failure may have
    # saved the database but not its playlist index (see begin_change).
    session_commit = db.commit
    db.commit_started = False

    def commit(*args, **kwargs):
        db.commit_started = True
        return session_commit(*args, **kwargs)
    db.commit = commit
    try:
        if db.playlist_xml is not None:
            _atomic_playlist_xml(db.playlist_xml)
    except BaseException:
        db.close()
        raise
    return db


def _atomic_playlist_xml(xml) -> None:
    """Make pyrekordbox save masterPlaylists6.xml whole or not at all.

    Its own save() truncates the file and writes it in place, so a crash
    mid-write would leave it cut short. Same text and encoding, written to a
    temporary file first and then renamed over the old one.
    """
    def save(path=None, indent=None):
        target = str(path or xml.path)
        text = xml.to_string(indent)
        partial = target + ".partial"
        try:
            with open(_long(partial), "w") as fh:      # same open() as pyrekordbox's own save
                fh.write(text)
            _replace(partial, target)
        finally:
            if _exists(partial):
                _remove(partial)
        xml._changed = False
    xml.save = save


def database_path(db) -> str:
    return str(db.engine.url.database)


def locate_library() -> str:
    """Where rekordbox's master.db is, found without ever connecting to it.

    This relies on pyrekordbox's MasterDatabase never connecting while it's being
    created. In the pinned versions (requirements.txt: >=0.5,<0.6; checked against
    0.5.0) its constructor only reads pyrekordbox's settings, builds an SQLAlchemy
    engine (which connects lazily, on the first query), parses masterPlaylists6.xml
    as a plain XML file and starts a Session (also lazy). test_damaged_library checks
    this at the DBAPI level: locating the library opens no sqlite3 or sqlcipher3
    connection to it. If a new pyrekordbox changes that, those tests fail.

    The engine is then told to refuse any connection before it's let go, so nothing
    here can open the file: closing a connection to the library can fold
    master.db-wal into master.db (SQLite does that when its last connection closes),
    and a run mustn't change anything before its backup."""
    from sqlalchemy import event
    db = open_database()
    try:
        def refuse(*args, **kwargs):
            raise RuntimeError("the live library is only located here, never opened")
        event.listen(db.engine, "do_connect", refuse)
        return database_path(db)
    finally:
        db.close()


def library_path() -> str:
    """Where rekordbox's master.db is."""
    db = open_database()
    try:
        return database_path(db)
    finally:
        db.close()


def _int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def load_tracks(db) -> tuple[list[Track], dict]:
    """Return (tracks, {track id: DjmdContent})."""
    from pyrekordbox.masterdb import models
    tracks, contents = [], {}
    cue_counts: dict[str, int] = {}
    for cue in db.query(models.DjmdCue.ContentID).filter(models.DjmdCue.rb_local_deleted == 0):
        cue_counts[str(cue.ContentID)] = cue_counts.get(str(cue.ContentID), 0) + 1
    query = db.query(models.DjmdContent).filter(models.DjmdContent.rb_local_deleted == 0)
    for c in query:
        folder = (c.FolderPath or "").replace("\\", "/").casefold()
        if any(marker in folder for marker in SAMPLER_MARKERS):
            continue
        colour = c.ColorName or ""
        try:
            waveform = str(db.get_anlz_paths(c).get("EXT") or "")
        except Exception:    # no analysis yet
            waveform = ""
        tracks.append(Track(
            id=str(c.ID),
            name=(c.Title or "").strip(),
            artist=(c.ArtistName or "").strip(),
            genre=(c.GenreName or "").strip(),
            label=(c.LabelName or "").strip(),
            bpm=_int(c.BPM) / 100,
            key=to_camelot(c.KeyName or ""),
            year=_int(c.ReleaseYear),
            rating=_int(c.Rating),
            colour=colour if colour in COLOUR_ORDER else "",
            # StockDate = when it was added to rekordbox (DateCreated is the file's date)
            date_added=c.StockDate or (c.created_at.date().isoformat() if c.created_at else ""),
            play_count=_int(c.DJPlayCount),
            path=os.path.normpath(c.FolderPath) if c.FolderPath else "",
            waveform=waveform if waveform and os.path.exists(waveform) else "",
            length=_int(c.Length),
            size=_int(c.FileSize),
            cues=cue_counts.get(str(c.ID), 0),
        ))
        contents[str(c.ID)] = c
    return tracks, contents


# rekordbox keeps recent changes in these journal files next to master.db
# (SQLite's write-ahead log, and its index) until it folds them in. Copies take
# only the log (see _library_files); the index goes when a library is replaced.
JOURNALS = ("-wal", "-shm")
BACKUP_GLOB = "master.organizer-backup-*.db"


def _long(path: str) -> str:
    """path, in the form Windows accepts at any length if it's past the usual limit of
    259 characters. Backup names are up to 59 characters longer than master.db, so a
    library that opens fine can have backups past it (unless Windows has long paths on)."""
    if os.name != "nt":
        return path
    full = os.path.abspath(path)
    if len(full) <= 259 or full.startswith("\\\\?\\"):
        return path
    if full.startswith("\\\\"):
        return "\\\\?\\UNC\\" + full[2:]
    return "\\\\?\\" + full


def _exists(path: str) -> bool:
    return os.path.exists(_long(path))


def _remove(path: str) -> None:
    os.remove(_long(path))


def _replace(src: str, dst: str) -> None:
    os.replace(_long(src), _long(dst))


def _matching(folder: str, pattern: str) -> list[str]:
    """The files in folder that match pattern. glob would read a "[" in the folder's
    name, or the "?" of the long form, as part of the pattern."""
    try:
        names = os.listdir(_long(folder or os.curdir))
    except OSError:
        return []
    return [os.path.join(folder, n) for n in fnmatch.filter(names, pattern)]


def _backup_parts(path: str) -> list[str]:
    return ([path] + [path + j for j in JOURNALS]
            + [path[:-3] + ".masterPlaylists6.xml", path[:-3] + ".manifest.json"])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(_long(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _discard(paths, error: Optional[BaseException] = None) -> None:
    """Remove whichever of paths exist, as far as possible. Never raises, so the error
    that led here is the one reported; what can't be removed is noted on that error."""
    stuck = []
    for path in paths:
        try:
            if _exists(path):
                _remove(path)
        except OSError:
            stuck.append(os.path.basename(path))
    if stuck and error is not None:
        error.add_note("Couldn't remove " + ", ".join(stuck) + "; it isn't used for anything.")


def _copy_verified(src: str, dst: str) -> str:
    """Copy src to dst via a temporary name; dst appears only once its bytes match src.
    Returns their SHA-256."""
    partial = dst + ".partial"
    try:
        try:
            shutil.copy2(_long(src), _long(partial))
        except OSError as exc:
            exc.filename = exc.filename or partial      # Windows' copy error names no file
            raise
        digest = _sha256(partial)
        if digest != _sha256(src):
            raise OSError(f"the copy of {os.path.basename(src)} doesn't match the original")
        _replace(partial, dst)
        return digest
    finally:
        _discard([partial])


def _copy_hashed(src: str, dst: str) -> str:
    """Copy src to dst; returns the SHA-256 of exactly the bytes copied."""
    h = hashlib.sha256()
    with open(_long(src), "rb") as fin, open(_long(dst), "wb") as fout:
        for block in iter(lambda: fin.read(1 << 20), b""):
            h.update(block)
            fout.write(block)
    return h.hexdigest()


class LibraryChanged(OSError):
    """The library changed (or rekordbox started) while the app was copying or
    restoring it, so it stopped; nothing in the library was changed or lost.

    Copies are plain file copies, and rekordbox can write, or fold its log into
    master.db, between one file and the next: a copy like that reads as "database
    disk image is malformed" even though the library itself is fine.
    """


CHANGED_WHILE_COPYING = (
    "Your rekordbox library changed while the app was copying it, so the copy wasn't used "
    "(it could be a mix of before and after). Nothing in your library was changed. Close "
    "rekordbox, wait a moment for it to finish saving, then try again.")


def _library_files(db_path: str) -> list[str]:
    """The files that together are the library at one moment: the database, its log
    and the playlist index. Not master.db-shm: that's an index SQLite rebuilds from
    the log when it's next opened, and any program merely reading the library updates
    it, so copying or comparing it would only cause false alarms."""
    return [db_path, db_path + "-wal",
            os.path.join(os.path.dirname(db_path), "masterPlaylists6.xml")]


def _changed_since(copied: dict[str, str], files: list[str]) -> bool:
    """Whether any of files differs from when it was copied (by SHA-256), has gone,
    or has appeared since (a new log, say). A file that can't be read raises OSError
    saying so, rather than being called a change."""
    for path in files:
        if path not in copied:
            if _exists(path):
                return True
            continue
        try:
            digest = _sha256(path)
        except FileNotFoundError:
            return True                       # gone: the log was folded in and deleted
        except OSError as exc:
            raise OSError(f"Couldn't read {os.path.basename(path)} again to check the copy "
                          f"({exc.strerror or exc}). Another program may be using your "
                          "library; close rekordbox and try again. Nothing in your library "
                          "was changed.") from exc
        if digest != copied[path]:
            return True
    return False


def _check_snapshot(copied: dict[str, str], files: list[str],
                    running: Optional[Callable[[], bool]] = None) -> None:
    """Raise LibraryChanged unless the files just copied are one moment's library.

    Called after every file has been copied. Any write between the first copy
    and this check changes a byte that was copied before it, so it shows up
    here, whichever program made it; a running check alone can't see rekordbox
    start mid-copy, or a process it doesn't recognise.
    """
    if (running or rekordbox_running)():
        raise LibraryChanged(f"{who_started()} while the app was copying your library, so the "
                             "copy wasn't used. Nothing in your library was changed. Close it, "
                             "then try again.")
    if _changed_since(copied, files):
        raise LibraryChanged(CHANGED_WHILE_COPYING)


def _write_json(path: str, data: dict) -> None:
    """Write path whole or not at all."""
    with open(_long(path + ".partial"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    _replace(path + ".partial", path)


def _manifest_path(copy: str) -> str:
    """The record saved with a backup (or before-undo copy): which files belong to
    it and their SHA-256, under an id of its own."""
    return copy[:-3] + ".manifest.json"


def _read_manifest(copy: str) -> Optional[dict]:
    """A copy's record; None for one made before records existed. Raises ValueError
    if it's damaged."""
    path = _manifest_path(copy)
    if not _exists(path):
        return None
    with open(_long(path), encoding="utf-8") as f:
        data = json.load(f)
    pieces = data.get("pieces") if isinstance(data, dict) else None
    if not isinstance(pieces, dict) or not all(
            isinstance(e, dict) and isinstance(e.get("name"), str)
            and os.path.basename(e["name"]) == e["name"] and isinstance(e.get("sha256"), str)
            for e in pieces.values()):
        raise ValueError("damaged record")
    return data


def _free_name(folder: str, stem: str) -> str:
    """folder/stem.db, or stem-2.db and so on: the first name that no file at all
    starts with, so a new copy can never take over a piece left by an earlier one."""
    try:
        names = os.listdir(_long(folder or os.curdir))
    except OSError:
        names = []
    n, candidate = 1, stem
    while any(name.startswith(candidate + ".") for name in names):
        n += 1
        candidate = f"{stem}-{n}"
    return os.path.join(folder, candidate + ".db")


def _save_snapshot(db_path: str, dest: str) -> dict[str, str]:
    """Copy the library (database, log, playlist index) to dest and its siblings,
    checked as one moment's library. Returns the SHA-256 of each library file copied.

    Its record (see _manifest_path) lists exactly the files copied, so a stray file
    with a matching name is never taken for part of it. The database is copied as
    dest + ".unchecked", which nothing lists, and renamed to dest only once every
    check has passed and the record is saved: a cut-off or failed copy is never
    offered as a backup, even if cleaning it up fails. On any failure, what was
    copied is removed as far as possible and the original error is raised.
    """
    files = _library_files(db_path)
    db, wal, xml = files
    names = {wal: dest + "-wal", xml: dest[:-3] + ".masterPlaylists6.xml",
             db: dest + ".unchecked"}            # the database last
    manifest = _manifest_path(dest)
    copied = {}
    try:
        for src, dst in names.items():
            if _exists(src):
                copied[src] = _copy_verified(src, dst)
        _check_snapshot(copied, files)
        roles = {db: "db", wal: "wal", xml: "xml"}
        _write_json(manifest, {
            "id": uuid4().hex,
            "made": datetime.datetime.now().isoformat(timespec="seconds"),
            "pieces": {roles[src]: {"name": os.path.basename(dest if src == db else names[src]),
                                    "sha256": digest} for src, digest in copied.items()}})
        _replace(names[db], dest)
    except BaseException as exc:
        _discard(list(names.values()) + [manifest, manifest + ".partial"], exc)
        raise
    return copied


def _unchanged_copy(path: str) -> bool:
    """Whether every file of a backup (or before-undo copy) is still exactly as saved.
    One made before records existed is a plain copy nothing writes to: True."""
    try:
        record = _read_manifest(path)
        if record is None:
            return True
        folder = os.path.dirname(path)
        return all(_exists(os.path.join(folder, e["name"]))
                   and _sha256(os.path.join(folder, e["name"])) == e["sha256"]
                   for e in record["pieces"].values())
    except (OSError, ValueError):
        return False


def _prune(paths: list[str]) -> list[str]:
    """Remove these backups (or before-undo copies) with all their files, except any
    that changed after they were saved: an Undo's copy can receive changes from a
    program that still had the library's old files open, so it's kept rather than
    lost. Returns the ones kept."""
    kept = []
    for path in paths:
        if _unchanged_copy(path):
            _discard(_backup_parts(path))
        else:
            kept.append(path)
    return kept


def backup(db_path: str, expected: Optional[dict[str, str]] = None) -> str:
    """Copy master.db with its log (and the playlist index); keep the newest few.

    expected: {library file: SHA-256} of the checked copy a change was planned from.
    If the backup isn't of exactly those files (the library changed in between), it's
    still kept, as a backup of the library as it is now, but no older backup is
    removed for it: one that doesn't match what the app read never pushes out a
    good one. The caller then stops (see organizer._begin_write).

    Every file is checked byte-for-byte against the original, and once all are
    copied, the library is checked again: if anything changed meanwhile (rekordbox
    opened, say) the files could be from different moments, so the backup is
    thrown away and this raises LibraryChanged. The database only gets its listed
    name once all of that has passed (see _save_snapshot), so a failed backup is
    never listed (or used) as one; on any failure this raises and nothing has been
    changed, so the caller must not go on to write.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.dirname(db_path)
    # Pieces left by a backup that was cut off or failed its checks (never a usable copy).
    _discard(_matching(folder, "master.organizer-backup-*.partial")
             + _matching(folder, "master.organizer-backup-*.unchecked"))
    dest = _free_name(folder, f"master.organizer-backup-{stamp}")
    copied = _save_snapshot(db_path, dest)
    if expected is None or copied == expected:
        _prune(list_backups(folder)[KEEP_BACKUPS:])
    return dest


def _newest_first(paths: list[str]) -> list[str]:
    """Sort backups by time and then by the -2, -3... added for ones made in the same
    second (plain name order would put "-2" before the first one)."""
    def key(path):
        m = re.search(r"-(\d{8}-\d{6})(?:-(\d+))?\.db$", os.path.basename(path))
        return (m.group(1), int(m.group(2) or 1)) if m else ("", 0)
    return sorted(paths, key=key, reverse=True)


def list_backups(folder: str) -> list[str]:
    """The app's backups in folder, newest first."""
    return _newest_first(_matching(folder, BACKUP_GLOB))


def backup_time(path: str) -> datetime.datetime:
    m = re.search(r"backup-(\d{8}-\d{6})", os.path.basename(path))
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
    return datetime.datetime.fromtimestamp(os.path.getmtime(_long(path)))


# A change to the library in progress: written (next to master.db) before the
# first write and removed only once everything, the database and its playlist
# index, is saved. Still there means the app stopped part-way (a crash, power
# cut...), and the database and masterPlaylists6.xml may not match.
PENDING = "master.organizer-pending.json"


def _pending_path(db_path: str) -> str:
    return os.path.join(os.path.dirname(db_path), PENDING)


def begin_change(db_path: str, backup_path: str, what: str = "write",
                 plan: Optional[dict] = None) -> dict:
    """Write the marker; an Undo saves its plan in it (see restore_backup). Returns
    what was written, to update later."""
    data = {"what": what, "backup": os.path.basename(backup_path),
            "started": datetime.datetime.now().isoformat(timespec="seconds")}
    if plan is not None:
        data["plan"] = plan
    _write_json(_pending_path(db_path), data)
    return data


def end_change(db_path: str) -> None:
    path = _pending_path(db_path)
    if _exists(path):
        _remove(path)


def interrupted_change(db_path: str) -> Optional[dict]:
    """The change that didn't finish, or None. A damaged marker still counts."""
    path = _pending_path(db_path)
    if not _exists(path):
        return None
    try:
        with open(_long(path), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"what": "unknown"}
    except (OSError, ValueError):
        return {"what": "unknown"}


def interrupted_message(info: dict) -> str:
    when = f" (started {str(info['started']).replace('T', ' ')})" if info.get("started") else ""
    backup_name = info.get("backup", "")
    if info.get("what") == "undo":
        if not isinstance(info.get("plan"), dict):
            return (f"The last Undo{when} didn't finish, and your library may be part-way "
                    "between two versions. Close rekordbox, then restore a backup with Undo: "
                    f"that puts back a whole library ({backup_name} was being restored). "
                    "Nothing else can be saved until then.")
        return (f"The last Undo{when} didn't finish, and your library may be part-way between "
                "two versions. Close rekordbox, then press 'Put my library back': it puts "
                "your library back as it was before that Undo (or finishes it, if it had got "
                "that far). Using Undo again does that first too. Nothing else can be saved until then.")
    return (f"The last change to your library{when} didn't finish, so the database and its "
            "playlist index (masterPlaylists6.xml) may not match. Use Undo to put back the "
            f"backup made just before it ({backup_name or 'the newest one'}). Or, if you've "
            "checked in rekordbox that everything is fine, choose 'Keep my library as it is' "
            "in the Undo window. Nothing else can be saved until then.")


def keep_as_is_allowed(info: dict, db_path: str) -> bool:
    """Whether 'Keep my library as it is' may clear this marker. Never for an Undo:
    one that stopped part-way can leave the library half-swapped (the backup's
    database without its log, or no playlist index), and its marker holds the plan
    that puts it right. Nor for a damaged marker while an Undo's working files are
    about, since it may have been an Undo's."""
    what = info.get("what")
    if what == "undo":
        return False
    if what not in ("write",):
        folder = os.path.dirname(db_path)
        return not any(_matching(folder, f"master.before-undo-*{end}")
                       for end in (".old", ".new", ".restored"))
    return True


def keep_library_as_is(db_path: str) -> None:
    """Clear the marker of a change that didn't finish, when keep_as_is_allowed."""
    info = interrupted_change(db_path)
    if info is None:
        return
    if not keep_as_is_allowed(info, db_path):
        raise OSError("The last Undo didn't finish, so your library may be part-way between "
                      "two versions, and it can't just be kept as it is. Press 'Put my "
                      "library back' (or Restore) instead. Nothing was changed.")
    end_change(db_path)


def put_right_stopped_undo(db_path: str) -> str:
    """Finish, or undo, an Undo that stopped part-way (recover_interrupted_undo), and
    say what happened, naming in full every file kept because it may hold changes."""
    info = interrupted_change(db_path) or {}
    state = (info.get("plan") or {}).get("state") if isinstance(info.get("plan"), dict) else None
    kept = recover_interrupted_undo(db_path)
    if kept is None:
        raise OSError("There's no record of how far that Undo got, so it can't be put right "
                      "step by step. Restore a backup instead: that puts back a whole "
                      "library. Nothing was changed.")
    if state == "validated":
        done = "The Undo that didn't finish has been completed: the backup is your library."
    else:
        done = "Your library is back as it was before the Undo that didn't finish."
    return done + (" " + _kept_note(kept) if kept else "")


def restore_backup(backup_path: str, db_path: str) -> str:
    """Put a backup back as the library. rekordbox and rekordboxAgent must be closed.

    1. An Undo that stopped part-way is finished, or undone, first (see
       recover_interrupted_undo), so pressing Undo again is always safe.
    2. The current library is saved as master.before-undo-<time>.db, checked as
       one moment's library like a backup, so the undo itself can be undone.
    3. The backup's own files (those its record lists, checked against it) are
       copied next to the library and the library is checked again. Then every
       name the swap will use is saved in the "change in progress" marker, before
       any library file moves.
    4. The swap. The old files always keep a name: master.db gets a second name
       (a hard link) and is replaced in one step, so it's never missing; its log,
       index and playlist index are renamed away (whatever file is there moves,
       so a save that just landed is kept, not overwritten). The backup's files
       go in by hard link, which fails rather than overwrite a file that appeared
       meanwhile. Just before, no program may have the library open (_open_by): on a
       Mac it would carry on with the old files after the swap.
    5. The check: the old files are still exactly as saved (a program that had them
       open - a Mac lets an open file be renamed - would have written there), the
       library's files are the very ones Undo put there and unchanged, no program
       has the old files open (any program, not only rekordbox: see _open_by), and,
       after all that hashing, rekordbox still isn't running.

    If step 5 finds anything, the library's own files are put back, with whatever
    was written to them, and this raises LibraryChanged: nothing was lost. If
    rekordbox (or its helper) is running by then, nothing more is moved, since it
    may have the files open, and likewise if any program has the restored files
    open; the marker stays, and the next Undo puts them back.
    If all is well, the old files become the saved copy, so even a late write by a
    program still holding them lands there; such a copy is never pruned.
    Returns the saved copy's path.
    """
    if rekordbox_running():
        raise LibraryChanged(open_message("try Undo again"))
    recovered = recover_interrupted_undo(db_path)
    if recovered:
        raise OSError("Your library was first put back as it was before the Undo that didn't "
                      "finish. " + _kept_note(recovered) + "The backup you picked hasn't "
                      "been restored yet: press Undo again to restore it.")
    folder = os.path.dirname(db_path)
    if not _exists(db_path):
        raise OSError(f"There's no {os.path.basename(db_path)} in your rekordbox folder, so "
                      "Undo can't save your library first, and stopped. Nothing was changed.")
    _tidy_undo_leftovers(db_path)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keep = _free_name(folder, f"master.before-undo-{stamp}")
    snapshot = _save_snapshot(db_path, keep)
    files = _library_files(db_path)
    copied = {role: snapshot.get(path) for role, path in zip(("db", "wal", "xml"), files)}
    names = _undo_names(db_path, keep)
    pieces = _backup_pieces(backup_path)

    staged, restored = [], {}
    try:
        for role, (src, expected) in pieces.items():
            staged.append(names["new_" + role])
            restored[role] = _copy_verified(src, names["new_" + role])
            if expected is not None and restored[role] != expected:
                raise OSError(f"{os.path.basename(src)} doesn't match the record saved with "
                              "the backup, so it wasn't restored. Nothing was changed.")
        _check_snapshot(snapshot, files)                # nothing changed while copying it
        holders = _open_by(files + [names["shm"]])
        if holders != []:
            raise LibraryChanged(_held_message(holders, "your library") + " Undo stopped "
                                 "before changing anything. Close it, then try Undo again.")
        plan = {"keep": os.path.basename(keep), "copied": copied, "restored": restored,
                "state": "swapping"}
        marker = begin_change(db_path, backup_path, "undo", plan=plan)
    except BaseException as exc:
        _discard(staged, exc)                           # only fresh copies of the backup
        raise

    moved = []
    try:
        _swap_in(names, restored, moved)
        # Who has the old files open is asked before they're checked: a program that
        # closes after this has already done whatever closing does (SQLite deletes the
        # log by name) when the check runs, so the check sees it.
        held = _open_by(_old_files(names))
        problem = _undo_problem(names, copied, restored)
        if problem is None and held != []:
            problem = _held_message(held, "your library's old files") + (
                " After the swap it would carry on with them, not with your library, and "
                "on closing could delete the restored library's log.")
    except Exception as exc:          # a crash (or Ctrl+C) leaves the marker instead
        problem = f"Undo couldn't swap the files safely ({exc})."
    writer = _writer_running()        # again, after all the hashing above
    if problem is None:
        plan["state"] = "validated"
        _write_json(_pending_path(db_path), marker)
        _after_step("validated")
        if not writer:                # leave alone any file a program may have open
            _finish_undo(names, restored)
        end_change(db_path)
        _prune(_newest_first(_matching(folder, "master.before-undo-*.db"))[KEEP_BACKUPS:])
        return keep
    in_use = _open_by(_restored_files(names)) if moved and not writer else []
    if moved and (writer or in_use != []):
        who = (f"{_running_label()} is running now and may have your library's files open"
               if writer else _held_message(in_use, "the restored files")[:-1])
        raise LibraryChanged(
            f"{problem} {who}, so Undo left every file where it is rather than move one in "
            f"use. Nothing is lost: your library from before the Undo is also saved as "
            f"{keep}. Close rekordbox, rekordboxAgent and any program using the library, "
            "then press Undo again: it first puts your library back as it was.")
    try:
        kept = _roll_back(names, copied, restored)
    except Exception as exc:
        raise OSError(
            f"{problem} Undo stopped part-way and couldn't put your library back yet ({exc}). "
            f"Nothing has been deleted, and {os.path.basename(db_path)} is in place; your "
            f"library from before the Undo is also saved as {os.path.basename(keep)}. Close "
            "rekordbox, then press Undo again: it first finishes putting your library back.") from exc
    end_change(db_path)
    raise LibraryChanged(
        f"{problem} Undo was stopped and your library was put back as it is now, with any "
        "change just made, so nothing was lost. " + _kept_note(kept)
        + "Make sure rekordbox and rekordboxAgent are closed, then try again.")


def recover_interrupted_undo(db_path: str) -> Optional[list[str]]:
    """Finish, or undo, an Undo that stopped part-way, from the plan in its marker.

    One that had passed its checks is finished; any other is undone, so the library
    is its own files again (with anything written to them). Returns the files kept
    because they may hold something written during it, or None if there was nothing
    to recover (no marker, or one from before plans were saved).
    """
    info = interrupted_change(db_path)
    plan = info.get("plan") if info and info.get("what") == "undo" else None
    if not isinstance(plan, dict) or os.path.basename(str(plan.get("keep"))) != plan.get("keep"):
        return None
    if rekordbox_running():
        raise LibraryChanged(open_message("press Undo again"))
    names = _undo_names(db_path, os.path.join(os.path.dirname(db_path), plan["keep"]))
    holders = _open_by([names[k] for k in ("db", "wal", "shm", "xml")]
                       + _old_files(names) + _restored_files(names))
    if holders != []:
        raise LibraryChanged(_held_message(holders, "your library's files") + " Nothing was "
                             "moved. Close it, then press Undo again.")
    restored = plan.get("restored") or {}
    if plan.get("state") == "validated":
        _finish_undo(names, restored)
        kept = []
    else:
        kept = _roll_back(names, plan.get("copied") or {}, restored)
    end_change(db_path)
    return kept


def _after_step(step: str) -> None:
    """A point between two of Undo's file operations. Does nothing; tests use it to
    stop the app there, or write to the library there."""


def _writer_running() -> bool:
    """rekordbox_running(), taking "couldn't tell" as yes: then nothing is moved."""
    try:
        return rekordbox_running()
    except Exception:
        return True


def _open_by(paths: list[str]) -> Optional[list[str]]:
    """The programs that have any of these files open other than just to read them,
    as "name (process N)": [] if none, None if that couldn't be checked.

    On a Mac a file can be renamed or replaced while a program has it open; the
    program carries on with the file under its new name. So a program holding the
    library through Undo's swap would write to the old files, and SQLite, closing,
    deletes master.db-wal and -shm *by name*, whatever file is there by then: the
    restored library's log. Any program counts, not just those Undo knows by name,
    so this asks lsof (part of macOS), which finds a file by what it is, whatever
    name it was opened by. A program run by another user (or the system) isn't
    visible to it; rekordbox runs as you. Files opened only to be read are left
    out: such a program can't write or delete anything (Spotlight, backups).

    Windows won't rename or replace a file SQLite (or rekordbox) has open, so the
    swap stops there instead (see the tests); this returns [] on Windows.
    """
    if sys.platform == "win32":
        return []
    paths = [p for p in paths if _exists(p)]
    if not paths:
        return []
    lsof = shutil.which("lsof") or "/usr/sbin/lsof"
    try:
        done = subprocess.run([lsof, "-n", "-P", "-w", "+c", "0", "-F", "pcan", "--", *paths],
                              capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    # 1 also means "some of the files aren't open"; anything on stderr is an error.
    if done.returncode not in (0, 1) or done.stderr.strip():
        return None
    found, pid, command, mode = {}, None, "", None
    for line in done.stdout.splitlines():
        tag, value = line[:1], line[1:]
        if tag == "p":
            pid, command = value, ""
        elif tag == "c":
            command = value
        elif tag == "f":
            mode = None                     # no access mode given: counted as writing
        elif tag == "a":
            mode = value.strip()
        elif tag == "n" and pid is not None and mode != "r":
            found[pid] = command
    return [f"{command or 'a program'} (process {pid})" for pid, command in found.items()]


def _held_message(holders: Optional[list[str]], what: str) -> str:
    if holders is None:
        return (f"Undo couldn't check whether a program has {what} open (it asks macOS's "
                "lsof), and moving a file in use could lose what that program saves.")
    return f"{', '.join(holders)} {'has' if len(holders) == 1 else 'have'} {what} open."


def _old_files(names: dict) -> list[str]:
    """The library's own files while the backup's are in place."""
    return [names[k] for k in ("old_db", "old_wal", "old_shm", "old_xml")]


def _restored_files(names: dict) -> list[str]:
    """The backup's files, by Undo's own names for them (wherever else they are)."""
    return [names[k] for k in ("restored_db", "new_db", "new_wal", "new_xml")]


def _running_label() -> str:
    names = _running_names()
    return "rekordboxAgent, rekordbox's background helper," if names == ["rekordboxAgent"] \
        else "rekordbox (or its helper, rekordboxAgent)"


def _kept_note(kept: list[str]) -> str:
    """Names in full every file kept because it may hold something saved during an
    Undo. This app never deletes them: no clean-up matches their names, or deletes
    an Undo file that isn't certainly a duplicate (see _tidy_undo_leftovers)."""
    if not kept:
        return ""
    return ("These files were kept because they may hold something saved during the Undo; "
            "this app won't delete them: " + "; ".join(kept) + ". ")


def _undo_names(db_path: str, keep: str) -> dict[str, str]:
    """Every name an Undo uses, fixed (and saved in its marker) before it starts."""
    db, wal, xml = _library_files(db_path)
    keep_xml = keep[:-3] + ".masterPlaylists6.xml"
    return {"db": db, "wal": wal, "shm": db_path + "-shm", "xml": xml, "keep": keep,
            "keep_wal": keep + "-wal", "keep_xml": keep_xml,
            # the library's own files, while the backup's are in place
            "old_db": keep + ".old", "old_wal": keep + "-wal.old", "old_shm": keep + "-shm.old",
            "old_xml": keep_xml + ".old",
            # the backup's files, copied next to the library first
            "new_db": keep + ".new", "new_wal": keep + "-wal.new", "new_xml": keep_xml + ".new",
            # a second name for the restored database, and for files found in the way
            "restored_db": keep + ".restored", "found_wal": keep + "-wal.found",
            "found_xml": keep_xml + ".found"}


def _backup_pieces(backup_path: str) -> dict[str, tuple[str, Optional[str]]]:
    """The files that make up a backup, by role, with the SHA-256 its record gives
    them: only those its record lists, so a stray file with a matching name (a piece
    left by a failed backup, say) is never restored with it. A backup made before
    records existed is taken by its file names, unchecked."""
    folder = os.path.dirname(backup_path)
    name = os.path.basename(backup_path)
    try:
        record = _read_manifest(backup_path)
    except (OSError, ValueError) as exc:
        raise OSError(f"The record saved with {name} is damaged, so it wasn't restored. "
                      "Nothing was changed.") from exc
    if record is None:
        if not _exists(backup_path):
            raise OSError(f"{name} is missing, so it wasn't restored. Nothing was changed.")
        by_name = {"db": backup_path, "wal": backup_path + "-wal",
                   "xml": backup_path[:-3] + ".masterPlaylists6.xml"}
        return {role: (path, None) for role, path in by_name.items() if _exists(path)}
    pieces = {role: (os.path.join(folder, e["name"]), e["sha256"])
              for role, e in record["pieces"].items() if role in ("db", "wal", "xml")}
    if "db" not in pieces or not all(_exists(path) for path, _ in pieces.values()):
        raise OSError(f"Some of {name}'s files are missing, so it wasn't restored. "
                      "Nothing was changed.")
    return pieces


def _same_file(a: str, b: str) -> bool:
    try:
        return _exists(a) and _exists(b) and os.path.samefile(_long(a), _long(b))
    except OSError:
        return False


def _is_copy(path: str, digest: Optional[str]) -> bool:
    """Whether path holds exactly the bytes with this SHA-256 (so it's a duplicate)."""
    try:
        return digest is not None and _sha256(path) == digest
    except OSError:
        return False


def _drop_second_name(path: str) -> None:
    """Remove a name that is only a second name of a file still in place: the data
    stays either way. Windows refuses while the file is open; then it's left for a
    later Undo to tidy (see _tidy_undo_leftovers)."""
    _discard([path])


def _rename_aside(src: str, dst: str) -> None:
    """Rename src to the unused name dst. A rename moves whatever file is at src at
    that moment, so nothing that just landed there is lost."""
    if _exists(dst):
        raise FileExistsError(f"{os.path.basename(dst)} is in the way")
    os.rename(_long(src), _long(dst))


def _link_new(src: str, dst: str) -> None:
    """Give src the name dst. Fails if anything is at dst: never overwrites."""
    os.link(_long(src), _long(dst))


def _free_evidence(path: str) -> str:
    n, candidate = 1, path
    while _exists(candidate):
        n += 1
        candidate = f"{path}-{n}"
    return candidate


def _swap_in(names: dict, restored: dict, moved: list) -> None:
    """Swap the backup's files in (step 4 of restore_backup)."""
    os.link(_long(names["db"]), _long(names["old_db"]))
    _after_step("old database named")
    # The library's own log and index go first: they must never be paired with the
    # backup's database (a database without them is older but consistent).
    for role in ("wal", "shm"):
        if _exists(names[role]):
            _rename_aside(names[role], names["old_" + role])
            moved.append(role)
            _after_step(f"old {role} out")
    _link_new(names["new_db"], names["restored_db"])      # to know it again later
    _replace(names["new_db"], names["db"])                # one step: never missing
    moved.append("db")
    _after_step("database in")
    if "wal" in restored:
        _link_new(names["new_wal"], names["wal"])
        _after_step("log in")
    if "xml" in restored:
        if _exists(names["xml"]):
            _rename_aside(names["xml"], names["old_xml"])
            moved.append("xml")
            _after_step("old playlist index out")
        _link_new(names["new_xml"], names["xml"])
        _after_step("playlist index in")


def _undo_problem(names: dict, copied: dict, restored: dict) -> Optional[str]:
    """What, if anything, touched the library during the swap; None if nothing did."""
    if _writer_running():
        return f"{who_started()} while Undo was swapping the files."
    changed = "Your library changed while Undo was swapping the files."
    # The library's old files are exactly as saved before the swap...
    roles = ("db", "wal", "xml") if "xml" in restored else ("db", "wal")
    for role in roles:
        old = names["old_" + role]
        if copied.get(role) is None:
            if _exists(old):
                return changed                  # appeared after it was saved
        elif not _is_copy(old, copied[role]):
            return changed
    # ...and the library's files now are the very ones Undo put there, unchanged.
    anchors = {"db": names["restored_db"], "wal": names["new_wal"], "xml": names["new_xml"]}
    for role, digest in restored.items():
        if not (_same_file(names[role], anchors[role]) and _is_copy(names[role], digest)):
            return changed
    if "wal" not in restored and _exists(names["wal"]):
        return changed                          # something started a log on it
    return None


def _finish_undo(names: dict, restored: dict) -> None:
    """After a good swap: the library's old files become the saved copy (identical
    to it, as just checked), so a program that still had them open writes there,
    not into nothing; Undo's extra names go. Tidying only: a step that fails leaves
    a file under its Undo name (never deleted unchecked). Safe to run again."""
    for old, target in (("old_db", "keep"), ("old_wal", "keep_wal"), ("old_xml", "keep_xml")):
        try:
            if _exists(names[old]):
                _replace(names[old], names[target])
        except OSError:
            pass
    _discard([names["old_shm"]])                         # an index, rebuilt by SQLite
    for extra, role in (("restored_db", "db"), ("new_db", "db"), ("new_wal", "wal"),
                        ("new_xml", "xml")):
        if _same_file(names[extra], names[role]) or _is_copy(names[extra], restored.get(role)):
            _discard([names[extra]])


def _roll_back(names: dict, copied: dict, restored: dict) -> list[str]:
    """Put the library's own files back after a swap that stopped part-way or failed
    its checks. Safe to run again from any point: each step looks at what's there.

    master.db is never missing: the old database goes back in one step, over the
    restored one (which keeps a second name). Nothing unknown is deleted: a log no
    one restored, a playlist index saved meanwhile, or a restored file something
    wrote to is kept. Returns the files kept for that reason.
    """
    kept = []
    # 1. While the restored database is in place, a log or index next to it isn't
    #    the library's own; it comes out first, as the old database must never
    #    meet it. (Otherwise the library's own database never left, or is back:
    #    then its log and index are its own and stay.)
    if _exists(names["old_db"]) and not _same_file(names["db"], names["old_db"]):
        live = names["wal"]
        if _exists(live):
            if _same_file(live, names["new_wal"]):
                _remove(live)                           # a second name of Undo's copy
            else:
                found = _free_evidence(names["found_wal"])
                _rename_aside(live, found)
                kept.append(found)
            _after_step("restored log out")
        if _exists(names["shm"]):
            _remove(names["shm"])                       # an index; nothing has it open
            _after_step("restored index out")
    # 2. The old database back, in one step.
    if _exists(names["old_db"]):
        if _same_file(names["db"], names["old_db"]):
            _drop_second_name(names["old_db"])          # never swapped
        else:
            if _exists(names["db"]) and not _exists(names["restored_db"]):
                _link_new(names["db"], names["restored_db"])
            _replace(names["old_db"], names["db"])
        _after_step("database back")
    # 3. Its log and index back, never over something that appeared meanwhile.
    for role in ("wal", "shm"):
        old, live = names["old_" + role], names[role]
        if _exists(old):
            if _same_file(live, old):
                _drop_second_name(old)
            elif _exists(live):
                raise OSError(f"{os.path.basename(live)} appeared while Undo was putting your "
                              f"library back; the library's own is {os.path.basename(old)}")
            else:
                _link_new(old, live)
                _drop_second_name(old)
            _after_step(f"{role} back")
    # 4. The playlist index. One saved while Undo ran is newer than the old one, so
    #    it stays; the old one is then a duplicate of the saved copy, or kept.
    old, live = names["old_xml"], names["xml"]
    if _exists(old):
        if _same_file(live, old):
            _drop_second_name(old)
        else:
            theirs = None
            if _exists(live):
                aside = _free_evidence(names["found_xml"])
                _rename_aside(live, aside)
                if _same_file(aside, names["new_xml"]) or _is_copy(aside, restored.get("xml")):
                    _remove(aside)                      # Undo's own copy
                else:
                    theirs = aside
                _after_step("restored playlist index out")
            if theirs is None:
                _link_new(old, live)
                _drop_second_name(old)
            else:
                _link_new(theirs, live)
                _drop_second_name(theirs)
                if _is_copy(old, copied.get("xml")):
                    _discard([old])                     # the same as the saved copy's
                else:
                    kept.append(old)
            _after_step("playlist index back")
    # 5. Undo's copies of the backup go, unless something wrote to them.
    for extra, role in (("new_db", "db"), ("restored_db", "db"), ("new_wal", "wal"),
                        ("new_xml", "xml")):
        if _exists(names[extra]):
            if _is_copy(names[extra], restored.get(role)):
                _remove(names[extra])
            else:
                kept.append(names[extra])
    if not _exists(names["db"]):
        raise OSError(f"{os.path.basename(names['db'])} is missing")
    return sorted(set(kept))


def _tidy_undo_leftovers(db_path: str) -> None:
    """Remove files an earlier Undo couldn't (Windows won't delete a file that's in
    use), but only those that are certainly duplicates: a second name of one of the
    library's own files, or an old file identical to the saved copy it belongs to.
    Anything else may hold something written during that Undo, so it stays."""
    folder = os.path.dirname(db_path)
    live = [p for p in _library_files(db_path) + [db_path + "-shm"] if _exists(p)]
    for path in _matching(folder, "master.before-undo-*.old") \
            + _matching(folder, "master.before-undo-*.new") \
            + _matching(folder, "master.before-undo-*.restored"):
        if any(_same_file(path, p) for p in live) or path.endswith("-shm.old"):
            _discard([path])                      # a second name, or an index SQLite rebuilds
        elif path.endswith(".old"):
            target = path[:-4]                    # the saved copy's file it duplicates
            copy = (target if target.endswith(".db") else
                    target[:-4] if target.endswith("-wal") else
                    target[:-len(".masterPlaylists6.xml")] + ".db")
            try:
                record = _read_manifest(copy)
            except (OSError, ValueError):
                continue
            digests = {e["name"]: e["sha256"] for e in (record or {}).get("pieces", {}).values()}
            if _is_copy(path, digests.get(os.path.basename(target))):
                _discard([path])


def _copy_library(src: str, work: str, running: Callable[[], bool]) -> dict[str, str]:
    """Copy the library's files into work as one moment's library (see _check_snapshot):
    {library file: SHA-256 of the bytes copied}. Raises LibraryChanged if it changed."""
    files = _library_files(src)
    copied = {}
    for path, name in zip(files, ("master.db", "master.db-wal", "masterPlaylists6.xml")):
        if _exists(path):
            copied[path] = _copy_hashed(path, os.path.join(work, name))
    _check_snapshot(copied, files, running)
    return copied


def _open_copy(src: str, work: str):
    import pathlib
    copy = open_database(os.path.join(work, "master.db"))
    copy._share_dir = pathlib.Path(os.path.dirname(src)) / "share"    # analysis files
    return copy


def open_snapshot(db_path: str) -> tuple[object, str, dict[str, str]]:
    """A run's checked copy of the library at db_path: (database, temp folder,
    {library file: SHA-256}), found and copied by path alone, so the library itself
    is never opened (see locate_library). Close it with close_preview_copy. Before
    writing, the library must still match the hashes (unchanged_since) and its
    backup be of exactly them (backup_is_of). As open_preview_copy, it raises
    LibraryChanged if the library changed while it was copied, and leaves no temp
    folder behind on any failure."""
    import tempfile
    copy = work = None
    try:
        work = tempfile.mkdtemp(prefix="rbo-snapshot-")
        copied = _copy_library(db_path, work, rekordbox_app_running)
        copy = _open_copy(db_path, work)
    except BaseException:
        close_preview_copy(copy, work)
        raise
    return copy, work, copied


def unchanged_since(db_path: str, copied: dict[str, str]) -> bool:
    """Whether the library's files are still exactly those copied (none gone or new)."""
    return not _changed_since(copied, _library_files(db_path))


def backup_is_of(backup_path: str, db_path: str, copied: dict[str, str]) -> bool:
    """Whether a backup's record lists exactly the library files copied, with the same
    SHA-256, and its files on disk still match the record."""
    try:
        record = _read_manifest(backup_path)
    except (OSError, ValueError):
        return False
    if record is None:
        return False
    roles = dict(zip(("db", "wal", "xml"), _library_files(db_path)))
    want = {role: copied[path] for role, path in roles.items() if path in copied}
    have = {role: entry["sha256"] for role, entry in record["pieces"].items()}
    return have == want and _unchanged_copy(backup_path)


def open_preview_copy(db) -> tuple[object, str]:
    """Swap the open library for a copy to read or try a run on: (database, temp folder).

    Always closes db, even on failure, and a failed copy leaves no temp folder behind.
    Analysis files are still read from the real library's folder. Anything that only
    looks at the library should use this: if rekordbox crashed, saved changes wait in
    master.db-wal, and SQLite moves them into master.db when its last connection closes.

    The copy must be of one moment: rekordbox must be closed, and once everything is
    copied the library is checked again (see _check_snapshot). If it changed, the copy
    is thrown away unopened and this raises LibraryChanged. master.db-shm isn't copied:
    SQLite rebuilds it from the log, and a copied one could describe a different log.
    """
    import tempfile

    copy = work = None
    try:
        try:
            if rekordbox_app_running():
                raise LibraryChanged(open_message(reading=True))
            src = database_path(db)
            work = tempfile.mkdtemp(prefix="rbo-preview-")
            _copy_library(src, work, rekordbox_app_running)
            copy = _open_copy(src, work)
        finally:
            db.close()
    except BaseException:
        close_preview_copy(copy, work)
        raise
    return copy, work


def close_preview_copy(db, work: Optional[str]) -> None:
    """Close db, if there is one, and remove its temp folder even if closing fails."""
    try:
        if db is not None:
            db.close()
    finally:
        if work:
            if db is not None and db.session is not None:   # close failed: Windows can't delete an open file
                try:
                    db.session.invalidate()
                    db.engine.dispose()
                except Exception:
                    pass
            shutil.rmtree(work, ignore_errors=True)


def write_changes(db, contents: dict, plan: Plan, folder_name: str,
                  ratings: dict[str, int], log: Callable[[str], None] = print,
                  merge: bool = True, evict=None,
                  changes=None, commit: bool = True) -> tuple[int, int, int, list]:
    """Apply ratings and the plan.

    changes: an organizer.Changes to record every change in (for previews).
    commit: False leaves everything unsaved (the caller throws it away).

    Returns (playlists created, playlists added to, tracks added,
    [(folder path, playlist name, track id) taken out]). evict: an organizer.Evictor saying
    which songs to take out of existing playlists.

    With merge, same-named folders and playlists (ignoring case) are reused and
    new tracks are appended; nothing already in them is removed. Smart playlists
    are never touched.
    """
    from pyrekordbox.masterdb import models

    changed = 0
    for track_id, stars in ratings.items():
        content = contents[track_id]
        if _int(content.Rating) != stars:
            if changes is not None:
                changes.ratings.append((track_id, _int(content.Rating), stars))
            content.Rating = stars
            changed += 1
    if ratings:
        log(f"Updated star ratings on {changed} tracks.")

    created = updated = added = 0
    removed: list = []
    if plan:
        def child(parent_id: str, name: str, attribute: int):
            matches = [p for p in db.get_playlist(ParentID=parent_id).all()
                       if p.Attribute == attribute and not p.rb_local_deleted
                       and same_name(p.Name, name)]
            return min(matches, key=lambda p: p.Seq or 0) if matches else None

        top = child("root", folder_name, FOLDER)
        if top is not None and not merge:
            log(f"Replacing the existing '{top.Name}' folder.")
            db.delete_playlist(top)
            db.flush()
            top = None
        if top is not None:
            log(f"Adding to your existing '{top.Name}' folder.")
        else:
            top = db.create_playlist_folder(folder_name)
        folders = {(): top}

        def take_out(pl, path: tuple):
            ids = evict.ids(path, pl.Name or "")
            if not ids:
                return
            # Like db.remove_from_playlist(), but without its commit per song,
            # so the whole run is still saved at once.
            songs = sorted(db.get_playlist_songs(PlaylistID=pl.ID).all(),
                           key=lambda s: s.TrackNo or 0)
            keep, now = [], datetime.datetime.now()
            for song in songs:
                if str(song.ContentID) in ids:
                    db.delete(song)
                    removed.append((path, pl.Name, str(song.ContentID)))
                    if changes is not None:
                        changes.removed.append((" / ".join((top.Name,) + path + (pl.Name,)),
                                                str(song.ContentID)))
                else:
                    keep.append(song)
            if len(keep) < len(songs):
                renumbered = []
                with db.registry.disabled():
                    for n, song in enumerate(keep, 1):
                        if song.TrackNo != n:
                            song.TrackNo, song.updated_at = n, now
                            renumbered.append(song)
                if renumbered:
                    db.registry.on_move(renumbered)
                db.flush()

        def walk(folder, path: tuple):
            for item in db.get_playlist(ParentID=folder.ID).all():
                if item.rb_local_deleted:
                    continue
                if item.Attribute == PLAYLIST:
                    take_out(item, path)
                elif item.Attribute == FOLDER and evict.scans(path + (item.Name or "",)):
                    walk(item, path + (item.Name or "",))

        if evict:
            walk(top, ())

        def folder_for(path: tuple):
            if path not in folders:
                parent = folder_for(path[:-1])
                existing = child(parent.ID, path[-1], FOLDER)
                folders[path] = (existing if existing is not None
                                 else db.create_playlist_folder(path[-1], parent=parent))
            return folders[path]

        for folder_title, lists in plan:
            path = folder_path(folder_title)
            sub = folder_for(path)
            for title, ids in lists:
                where = " / ".join((top.Name,) + path + (title,))
                pl = child(sub.ID, title, PLAYLIST)
                if path == (SETS_FOLDER,):
                    # A set is in play order: write it afresh rather than adding on.
                    if changes is not None:
                        changes.sets.append((where, list(ids), pl is not None))
                    if pl is None:
                        pl = db.create_playlist(title, parent=sub)
                        created += 1
                    else:
                        for song in db.get_playlist_songs(PlaylistID=pl.ID).all():
                            db.delete(song)
                        db.flush()
                        updated += 1
                    now = datetime.datetime.now()
                    for n, track_id in enumerate(ids, 1):
                        db.add(models.DjmdSongPlaylist.create(
                            ID=str(uuid4()), PlaylistID=str(pl.ID), ContentID=str(track_id),
                            TrackNo=n, UUID=str(uuid4()), created_at=now, updated_at=now))
                    continue
                if pl is None:
                    pl = db.create_playlist(title, parent=sub)
                    have, last = set(), 0
                    created += 1
                    if changes is not None:
                        changes.created.append(where)
                else:
                    songs = db.get_playlist_songs(PlaylistID=pl.ID).all()
                    have = {str(s.ContentID) for s in songs}
                    last = max((s.TrackNo or 0 for s in songs), default=0)
                    if any(str(i) not in have for i in ids):
                        updated += 1
                now = datetime.datetime.now()
                # Same as db.add_to_playlist(), minus a COUNT query per track.
                for track_id in ids:
                    if str(track_id) in have:
                        continue
                    last += 1
                    db.add(models.DjmdSongPlaylist.create(
                        ID=str(uuid4()), PlaylistID=str(pl.ID), ContentID=str(track_id),
                        TrackNo=last, UUID=str(uuid4()), created_at=now, updated_at=now))
                    have.add(str(track_id))
                    added += 1
                    if changes is not None:
                        changes.added.append((where, str(track_id)))

    if commit:
        db.commit()
    else:
        db.flush()
    return created, updated, added, removed


def write_genres(db, contents: dict, new: dict[str, str], log: Callable[[str], None] = print,
                 changes=None) -> int:
    """Set track genres: {track id: genre name}. Returns how many changed.

    An existing rekordbox genre with the same name (ignoring capitals) is reused,
    so "house" goes in as your existing "House".
    """
    from pyrekordbox.masterdb import models

    genres = {}
    for g in db.query(models.DjmdGenre).all():
        if g.Name:
            genres.setdefault(" ".join(g.Name.split()).casefold(), g)
    changed = 0
    now = datetime.datetime.now()
    for track_id, name in new.items():
        name = " ".join(name.split())
        content = contents.get(track_id)
        if not name or content is None:
            continue
        genre = genres.get(name.casefold())
        if genre is None:
            genre = db.add_genre(name)
            genres[name.casefold()] = genre
        if str(content.GenreID or "") == str(genre.ID):
            continue
        if changes is not None:
            changes.genres.append((track_id, (content.GenreName or "").strip(), genre.Name))
        content.GenreID, content.updated_at = genre.ID, now
        changed += 1
    log(f"Saved the genre of {changed} track{'s' if changed != 1 else ''}.")
    return changed


# My Tag: rekordbox has four tag columns (DjmdMyTag rows with Attribute 1 under
# "root"), each holding tags (Attribute 0). Energy goes in a column named
# "Energy", as tags "Energy 1" ... "Energy 5".
ENERGY_COLUMN = "Energy"
SPARE_COLUMN = "Untitled Column"   # rekordbox's name for the column it leaves empty


def energy_tag_column(db, log: Callable[[str], None] = print):
    """The "Energy" My Tag column, taking over the spare one if needed; None if impossible."""
    from pyrekordbox.masterdb import models

    columns = [c for c in db.query(models.DjmdMyTag).filter(
        models.DjmdMyTag.ParentID == "root", models.DjmdMyTag.Attribute == 1,
        models.DjmdMyTag.rb_local_deleted == 0)]
    for c in columns:
        if same_name(c.Name, ENERGY_COLUMN):
            return c
    for c in columns:
        if same_name(c.Name, SPARE_COLUMN):
            tags = [t.ID for t in db.query(models.DjmdMyTag).filter(
                models.DjmdMyTag.ParentID == c.ID, models.DjmdMyTag.rb_local_deleted == 0)]
            used = tags and db.query(models.DjmdSongMyTag).filter(
                models.DjmdSongMyTag.MyTagID.in_(tags),
                models.DjmdSongMyTag.rb_local_deleted == 0).count()
            if not used:
                c.Name, c.updated_at = ENERGY_COLUMN, datetime.datetime.now()
                log(f"Renamed your unused My Tag column '{SPARE_COLUMN}' to '{ENERGY_COLUMN}'.")
                return c
    log("Energy My Tags skipped: all your My Tag columns are in use. In rekordbox, rename "
        f"one to '{ENERGY_COLUMN}' and run again.")
    return None


def write_energy_tags(db, levels: dict[str, int], log: Callable[[str], None] = print,
                      changes=None) -> None:
    """Tag each track with "Energy <level>" (one per track; old levels are taken off)."""
    from pyrekordbox.masterdb import models

    column = energy_tag_column(db, log)
    if column is None or not levels:
        return
    now = datetime.datetime.now()
    tags = {t.Name: t for t in db.query(models.DjmdMyTag).filter(
        models.DjmdMyTag.ParentID == column.ID, models.DjmdMyTag.rb_local_deleted == 0)}
    seq = max((t.Seq or 0 for t in tags.values()), default=0)
    for level in range(1, 6):
        name = f"Energy {level}"
        if name not in tags:
            seq += 1
            tags[name] = models.DjmdMyTag.create(
                ID=str(db.generate_unused_id(models.DjmdMyTag, is_28_bit=False)), Seq=seq,
                Name=name, Attribute=0, ParentID=column.ID, UUID=str(uuid4()),
                created_at=now, updated_at=now)
            db.add(tags[name])
            db.flush()
    by_id = {t.ID: name for name, t in tags.items() if name.startswith("Energy ")}

    current: dict[str, list] = {}
    counts: dict[str, int] = {}
    for row in db.query(models.DjmdSongMyTag).filter(
            models.DjmdSongMyTag.MyTagID.in_(list(by_id)),
            models.DjmdSongMyTag.rb_local_deleted == 0):
        current.setdefault(str(row.ContentID), []).append(row)
        counts[row.MyTagID] = max(counts.get(row.MyTagID, 0), row.TrackNo or 0)

    changed = 0
    for track_id, level in levels.items():
        if not level:
            continue
        want = tags[f"Energy {level}"].ID
        rows = current.get(str(track_id), [])
        if [r.MyTagID for r in rows] == [want]:
            continue
        for r in rows:
            if r.MyTagID != want:
                db.delete(r)
        if all(r.MyTagID != want for r in rows):
            counts[want] = counts.get(want, 0) + 1
            db.add(models.DjmdSongMyTag.create(
                ID=str(uuid4()), MyTagID=want, ContentID=str(track_id), TrackNo=counts[want],
                UUID=str(uuid4()), created_at=now, updated_at=now))
        if changes is not None:
            old = ", ".join(sorted(by_id[r.MyTagID] for r in rows)) or ""
            changes.tags.append((str(track_id), old, f"Energy {level}"))
        changed += 1
    db.flush()
    log(f"Energy My Tags: updated {changed} track{'s' if changed != 1 else ''}.")


# Hot cue pads A-H as stored in DjmdCue.Kind (0 is a memory cue; there's no 4).
# Checked against hot cues made in rekordbox 7: those live only in master.db
# (the ANLZ files' cue tags stay empty), with InFrame = InMsec * 150 // 1000.
HOT_CUE_KINDS = (1, 2, 3, 5, 6, 7, 8, 9)


MEMORY_CUE = 0


def plan_hot_cues(db, tracks: list[Track], log: Callable[[str], None] = print,
                  hot: bool = True, memory: bool = False) -> dict:
    """{track id: [(DjmdCue.Kind, Phrase)]} for the cues to add, from phrase analysis.

    hot: hot cues where sections start, in empty pads. memory: memory cues
    where the intro ends and the outro starts.
    """
    import cues
    from pyrekordbox.masterdb import models

    existing_hot: dict[str, list] = {}
    existing_memory: dict[str, list] = {}
    for c in db.query(models.DjmdCue).filter(models.DjmdCue.rb_local_deleted == 0):
        if c.Kind in HOT_CUE_KINDS:
            existing_hot.setdefault(str(c.ContentID), []).append(c)
        elif c.Kind == MEMORY_CUE:
            existing_memory.setdefault(str(c.ContentID), []).append(_int(c.InMsec))

    plan, unanalysed, done = {}, 0, 0
    for t in tracks:
        phrases = cues.read_phrases(t.waveform)
        if not phrases:
            unanalysed += 1
            continue
        picks = []
        if hot:
            have = existing_hot.get(t.id, [])
            picks += [(HOT_CUE_KINDS[pad], p) for pad, p in cues.choose_hot_cues(
                phrases, {HOT_CUE_KINDS.index(c.Kind) for c in have},
                [_int(c.InMsec) for c in have])]
        if memory:
            picks += [(MEMORY_CUE, p) for p in
                      cues.choose_memory_cues(phrases, existing_memory.get(t.id, []))]
        if picks:
            plan[t.id] = picks
        else:
            done += 1
    n_hot = sum(kind != MEMORY_CUE for p in plan.values() for kind, _ in p)
    n_memory = sum(kind == MEMORY_CUE for p in plan.values() for kind, _ in p)
    parts = ([f"{n_hot} hot cues"] if hot else []) + ([f"{n_memory} memory cues"] if memory else [])
    log(f"Cues: {' and '.join(parts)} to add on {len(plan)} track{'s' if len(plan) != 1 else ''}"
        + (f"; {done} already have them or no empty pads" if done else "") + ".")
    if unanalysed:
        log(f"  {unanalysed} track{'s have' if unanalysed != 1 else ' has'} no phrase analysis. "
            "In rekordbox, turn on Preferences > Analysis > Phrase, re-analyse them, "
            "and run again.")
    return plan


def add_hot_cues(db, contents: dict, plan: dict, log: Callable[[str], None] = print,
                 changes=None, colours: bool = False) -> None:
    """Add the planned cues (saved by the next commit). colours: colour hot cues by section."""
    import cues
    from pyrekordbox.masterdb import models

    now = datetime.datetime.now()
    taken: set[int] = set()     # IDs handed out but not yet in the database
    for track_id, picks in plan.items():
        content = contents[track_id]
        for kind, phrase in picks:
            cue_id = db.generate_unused_id(models.DjmdCue, is_28_bit=False)
            while cue_id in taken:
                cue_id = db.generate_unused_id(models.DjmdCue, is_28_bit=False)
            taken.add(cue_id)
            ms = int(phrase.start_ms)
            memory = kind == MEMORY_CUE
            colour = None if memory or not colours else cues.colour_index(phrase.label)
            if changes is not None:
                where = "Memory" if memory else cues.PADS[HOT_CUE_KINDS.index(kind)]
                changes.cues.append((track_id, where, phrase.label, ms))
            # Same fields rekordbox fills in for a cue set by hand.
            db.add(models.DjmdCue.create(
                ID=str(cue_id), ContentID=str(content.ID), ContentUUID=content.UUID,
                InMsec=ms, InFrame=ms * 150 // 1000, InMpegFrame=0, InMpegAbs=0,
                OutMsec=-1, OutFrame=0, OutMpegFrame=0, OutMpegAbs=0,
                Kind=kind, Color=-1, ColorTableIndex=colour, Comment=phrase.label,
                UUID=str(uuid4()), created_at=now, updated_at=now))
    if plan:
        log(f"Added {sum(len(p) for p in plan.values())} cues.")
