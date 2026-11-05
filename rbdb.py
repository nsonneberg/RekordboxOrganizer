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
from typing import Callable, Optional
from uuid import uuid4

# SQLAlchemy's compiled speed-ups can be blocked by Windows Smart App Control;
# the pure-Python fallback is plenty fast for this.
os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")

from organizer import (COLOUR_ORDER, SETS_FOLDER, Plan, Track, folder_path,  # noqa: E402
                       same_name, to_camelot)

KEEP_BACKUPS = 3
PLAYLIST, FOLDER = 0, 1   # DjmdPlaylist.Attribute (smart playlists are 4)
# rekordbox's built-in sampler sounds (older and newer install locations)
SAMPLER_MARKERS = ("/pioneerdj/sampler/", "/rekordbox/sampler/")


def _pyrekordbox():
    import pyrekordbox
    logging.getLogger("pyrekordbox").setLevel(logging.ERROR)
    return pyrekordbox


def rekordbox_running() -> bool:
    from pyrekordbox.utils import get_rekordbox_pid
    _pyrekordbox()
    return bool(get_rekordbox_pid())


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
    if db.playlist_xml is not None:
        _atomic_playlist_xml(db.playlist_xml)
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
# (SQLite's write-ahead log) until it folds them in, so they belong together.
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
    return [path] + [path + j for j in JOURNALS] + [path[:-3] + ".masterPlaylists6.xml"]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(_long(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _copy_verified(src: str, dst: str) -> None:
    """Copy src to dst via a temporary name; dst appears only once its bytes match src."""
    partial = dst + ".partial"
    try:
        try:
            shutil.copy2(_long(src), _long(partial))
        except OSError as exc:
            exc.filename = exc.filename or partial      # Windows' copy error names no file
            raise
        if _sha256(partial) != _sha256(src):
            raise OSError(f"the copy of {os.path.basename(src)} doesn't match the original")
        _replace(partial, dst)
    finally:
        if _exists(partial):
            _remove(partial)


def backup(db_path: str) -> str:
    """Copy master.db with its journal files (and the playlist index); keep the newest few.

    Every file is checked byte-for-byte against the original. The database goes
    in last, so an interrupted or failed backup is never listed (or used) as
    one; on any failure this raises and nothing has been changed, so the caller
    must not go on to write.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.dirname(db_path)
    # Pieces left by a backup that was cut off (never renamed, so never a usable copy).
    for stale in _matching(folder, "master.organizer-backup-*.partial"):
        _remove(stale)
    dest = os.path.join(folder, f"master.organizer-backup-{stamp}.db")
    n = 1
    while _exists(dest):   # never overwrite an earlier backup
        n += 1
        dest = os.path.join(folder, f"master.organizer-backup-{stamp}-{n}.db")
    parts = [(db_path + j, dest + j) for j in JOURNALS if _exists(db_path + j)]
    xml = os.path.join(folder, "masterPlaylists6.xml")
    if _exists(xml):
        parts.append((xml, dest[:-3] + ".masterPlaylists6.xml"))
    try:
        for src, dst in parts + [(db_path, dest)]:
            _copy_verified(src, dst)
    except OSError:
        for _, dst in parts:          # don't leave pieces of a backup that isn't one
            if _exists(dst) and not _exists(dest):
                _remove(dst)
        raise

    for path in list_backups(folder)[KEEP_BACKUPS:]:
        for p in _backup_parts(path):
            if _exists(p):
                _remove(p)
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


def begin_change(db_path: str, backup_path: str, what: str = "write") -> None:
    path = _pending_path(db_path)
    data = {"what": what, "backup": os.path.basename(backup_path),
            "started": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(_long(path + ".partial"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    _replace(path + ".partial", path)


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
        return (f"The last Undo{when} didn't finish. Your library may be part-restored. "
                f"Use Undo again and pick the same backup ({backup_name}); the backup itself "
                "is untouched. Nothing else can be saved until then.")
    return (f"The last change to your library{when} didn't finish, so the database and its "
            "playlist index (masterPlaylists6.xml) may not match. Use Undo to put back the "
            f"backup made just before it ({backup_name or 'the newest one'}). Or, if you've "
            "checked in rekordbox that everything is fine, choose 'Keep my library as it is' "
            "in the Undo window. Nothing else can be saved until then.")


def restore_backup(backup_path: str, db_path: str) -> str:
    """Put a backup back as the library. rekordbox must be closed.

    The current library (with its journal files) is first saved as
    master.before-undo-<time>.db, so the undo itself can be undone. The
    restored database is checked byte-for-byte against the backup. If it stops
    part-way, the "change in progress" marker says so and Undo can simply be
    run again (the backup is only ever read). Returns the saved copy's path.
    """
    folder = os.path.dirname(db_path)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keep = os.path.join(folder, f"master.before-undo-{stamp}.db")
    n = 1
    while _exists(keep):
        n += 1
        keep = os.path.join(folder, f"master.before-undo-{stamp}-{n}.db")
    xml = os.path.join(folder, "masterPlaylists6.xml")
    current = [(db_path + j, keep + j) for j in JOURNALS if _exists(db_path + j)]
    if _exists(xml):
        current.append((xml, keep[:-3] + ".masterPlaylists6.xml"))
    for src, dst in current + [(db_path, keep)]:
        _copy_verified(src, dst)

    # Copy everything next to the library first, so a failed copy (disk full...)
    # leaves the library untouched; then swap the files in by renaming.
    saved_xml = backup_path[:-3] + ".masterPlaylists6.xml"
    sources = [(backup_path, db_path)] + [(backup_path + j, db_path + j) for j in JOURNALS]
    if _exists(saved_xml):
        sources.append((saved_xml, xml))
    staged = {}
    try:
        for src, dst in sources:
            if _exists(src):
                _copy_verified(src, dst + ".restoring")
                staged[dst] = dst + ".restoring"
    except OSError:
        for tmp in staged.values():
            _remove(tmp)
        raise
    begin_change(db_path, backup_path, "undo")
    # Order matters if this stops part-way: the library's own journals go first
    # (they must never be paired with the backup's database), then the database,
    # then the backup's journals (a backup database without them is older but
    # consistent; a backup journal next to the wrong database is not).
    for j in JOURNALS:
        if _exists(db_path + j):
            _remove(db_path + j)
    for dst in [db_path] + [db_path + j for j in JOURNALS] + [xml]:
        if dst not in staged:
            continue
        try:
            _replace(staged[dst], dst)
        except PermissionError:
            # Windows refuses to rename over a file another program holds open;
            # copy over it in place instead (the staged copy proved the space).
            shutil.copyfile(_long(staged[dst]), _long(dst))
            _remove(staged[dst])
    if _sha256(db_path) != _sha256(backup_path):
        raise OSError("The restored library doesn't match the backup. Run Undo again; "
                      f"your library from before the undo is saved as {os.path.basename(keep)}.")
    end_change(db_path)

    undos = _newest_first(_matching(folder, "master.before-undo-*.db"))
    for path in undos[KEEP_BACKUPS:]:
        for p in _backup_parts(path):
            if _exists(p):
                _remove(p)
    return keep


def open_preview_copy(db) -> tuple[object, str]:
    """A copy of the open library to try a run on: (database, temp folder).

    Analysis files are still read from the real library's folder.
    """
    import pathlib
    import tempfile

    src = database_path(db)
    work = tempfile.mkdtemp(prefix="rbo-preview-")
    for suffix in ("",) + JOURNALS:
        if os.path.exists(src + suffix):
            shutil.copy2(src + suffix, os.path.join(work, "master.db" + suffix))
    xml = os.path.join(os.path.dirname(src), "masterPlaylists6.xml")
    if os.path.exists(xml):
        shutil.copy2(xml, work)
    copy = open_database(os.path.join(work, "master.db"))
    copy._share_dir = pathlib.Path(os.path.dirname(src)) / "share"
    return copy, work


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
