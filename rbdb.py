"""Work directly with rekordbox's own library database (master.db).

No XML export needed: tracks are read straight from rekordbox, and playlists and
star ratings are written straight back. rekordbox must be closed while writing,
and a backup of the database is made before every change.
"""

from __future__ import annotations

import datetime
import glob
import logging
import os
import re
import shutil
from typing import Callable, Optional
from uuid import uuid4

# SQLAlchemy's compiled speed-ups can be blocked by Windows Smart App Control;
# the pure-Python fallback is plenty fast for this.
os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")

from organizer import COLOUR_ORDER, Plan, Track, folder_path, same_name, to_camelot  # noqa: E402

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
        return pyrekordbox.MasterDatabase(path) if path else pyrekordbox.MasterDatabase()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Couldn't find your rekordbox library on this computer. "
            "Is rekordbox 6 or 7 installed and has it been opened at least once?"
        ) from exc


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
        ))
        contents[str(c.ID)] = c
    return tracks, contents


# rekordbox keeps recent changes in these journal files next to master.db
# (SQLite's write-ahead log) until it folds them in, so they belong together.
JOURNALS = ("-wal", "-shm")
BACKUP_GLOB = "master.organizer-backup-*.db"


def _backup_parts(path: str) -> list[str]:
    return [path] + [path + j for j in JOURNALS] + [path[:-3] + ".masterPlaylists6.xml"]


def backup(db_path: str) -> str:
    """Copy master.db with its journal files (and the playlist index); keep the newest few."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.dirname(db_path)
    dest = os.path.join(folder, f"master.organizer-backup-{stamp}.db")
    n = 1
    while os.path.exists(dest):   # never overwrite an earlier backup
        n += 1
        dest = os.path.join(folder, f"master.organizer-backup-{stamp}-{n}.db")
    shutil.copy2(db_path, dest)
    for j in JOURNALS:
        if os.path.exists(db_path + j):
            shutil.copy2(db_path + j, dest + j)
    xml = os.path.join(folder, "masterPlaylists6.xml")
    if os.path.exists(xml):
        shutil.copy2(xml, dest[:-3] + ".masterPlaylists6.xml")

    for path in list_backups(folder)[KEEP_BACKUPS:]:
        for p in _backup_parts(path):
            if os.path.exists(p):
                os.remove(p)
    return dest


def list_backups(folder: str) -> list[str]:
    """The app's backups in folder, newest first."""
    return sorted(glob.glob(os.path.join(folder, BACKUP_GLOB)), reverse=True)


def backup_time(path: str) -> datetime.datetime:
    m = re.search(r"backup-(\d{8}-\d{6})", os.path.basename(path))
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def restore_backup(backup_path: str, db_path: str) -> str:
    """Put a backup back as the library. rekordbox must be closed.

    The current library (with its journal files) is first saved as
    master.before-undo-<time>.db, so the undo itself can be undone.
    Returns that file's path.
    """
    folder = os.path.dirname(db_path)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keep = os.path.join(folder, f"master.before-undo-{stamp}.db")
    shutil.copy2(db_path, keep)
    for j in JOURNALS:
        if os.path.exists(db_path + j):
            shutil.copy2(db_path + j, keep + j)
    xml = os.path.join(folder, "masterPlaylists6.xml")
    if os.path.exists(xml):
        shutil.copy2(xml, keep[:-3] + ".masterPlaylists6.xml")

    # Journals first: the library's own must never be paired with the backup.
    for j in JOURNALS:
        if os.path.exists(db_path + j):
            os.remove(db_path + j)
    shutil.copy2(backup_path, db_path)
    for j in JOURNALS:
        if os.path.exists(backup_path + j):
            shutil.copy2(backup_path + j, db_path + j)
    saved_xml = backup_path[:-3] + ".masterPlaylists6.xml"
    if os.path.exists(saved_xml):
        shutil.copy2(saved_xml, xml)

    undos = sorted(glob.glob(os.path.join(folder, "master.before-undo-*.db")), reverse=True)
    for path in undos[KEEP_BACKUPS:]:
        for p in _backup_parts(path):
            if os.path.exists(p):
                os.remove(p)
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
    [(playlist name, track id) taken out]). evict: an organizer.Evictor saying
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
                    removed.append((pl.Name, str(song.ContentID)))
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


# Hot cue pads A-H as stored in DjmdCue.Kind (0 is a memory cue; there's no 4).
# Checked against hot cues made in rekordbox 7: those live only in master.db
# (the ANLZ files' cue tags stay empty), with InFrame = InMsec * 150 // 1000.
HOT_CUE_KINDS = (1, 2, 3, 5, 6, 7, 8, 9)


def plan_hot_cues(db, tracks: list[Track], log: Callable[[str], None] = print) -> dict:
    """{track id: [(pad 0-7, Phrase)]} for the hot cues to add, from phrase analysis."""
    import cues
    from pyrekordbox.masterdb import models

    hot: dict[str, list] = {}
    for c in db.query(models.DjmdCue).filter(models.DjmdCue.rb_local_deleted == 0):
        if c.Kind in HOT_CUE_KINDS:
            hot.setdefault(str(c.ContentID), []).append(c)

    plan, unanalysed, done = {}, 0, 0
    for t in tracks:
        phrases = cues.read_phrases(t.waveform)
        if not phrases:
            unanalysed += 1
            continue
        existing = hot.get(t.id, [])
        picks = cues.choose_hot_cues(phrases, {HOT_CUE_KINDS.index(c.Kind) for c in existing},
                                     [_int(c.InMsec) for c in existing])
        if picks:
            plan[t.id] = picks
        else:
            done += 1
    added = sum(len(p) for p in plan.values())
    log(f"Hot cues: {added} to add on {len(plan)} track{'s' if len(plan) != 1 else ''}"
        + (f"; {done} already have them or no empty pads" if done else "") + ".")
    if unanalysed:
        log(f"  {unanalysed} track{'s have' if unanalysed != 1 else ' has'} no phrase analysis. "
            "In rekordbox, turn on Preferences > Analysis > Phrase, re-analyse them, "
            "and run again.")
    return plan


def add_hot_cues(db, contents: dict, plan: dict, log: Callable[[str], None] = print,
                 changes=None) -> None:
    """Add the planned hot cues (saved by the next commit)."""
    from pyrekordbox.masterdb import models

    now = datetime.datetime.now()
    taken: set[int] = set()     # IDs handed out but not yet in the database
    for track_id, picks in plan.items():
        content = contents[track_id]
        for pad, phrase in picks:
            cue_id = db.generate_unused_id(models.DjmdCue, is_28_bit=False)
            while cue_id in taken:
                cue_id = db.generate_unused_id(models.DjmdCue, is_28_bit=False)
            taken.add(cue_id)
            ms = int(phrase.start_ms)
            if changes is not None:
                changes.cues.append((track_id, "ABCDEFGH"[pad], phrase.label, ms))
            # Same fields rekordbox fills in for a hot cue set on a pad.
            db.add(models.DjmdCue.create(
                ID=str(cue_id), ContentID=str(content.ID), ContentUUID=content.UUID,
                InMsec=ms, InFrame=ms * 150 // 1000, InMpegFrame=0, InMpegAbs=0,
                OutMsec=-1, OutFrame=0, OutMpegFrame=0, OutMpegAbs=0,
                Kind=HOT_CUE_KINDS[pad], Color=-1, Comment=phrase.label,
                UUID=str(uuid4()), created_at=now, updated_at=now))
    if plan:
        log(f"Added {sum(len(p) for p in plan.values())} hot cues.")
