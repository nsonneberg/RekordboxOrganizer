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


def backup(db_path: str) -> str:
    """Copy master.db (and the playlist index next to it); keep the newest few."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.dirname(db_path)
    dest = os.path.join(folder, f"master.organizer-backup-{stamp}.db")
    n = 1
    while os.path.exists(dest):   # never overwrite an earlier backup
        n += 1
        dest = os.path.join(folder, f"master.organizer-backup-{stamp}-{n}.db")
    shutil.copy2(db_path, dest)
    xml = os.path.join(folder, "masterPlaylists6.xml")
    if os.path.exists(xml):
        shutil.copy2(xml, dest[:-3] + ".masterPlaylists6.xml")

    old = sorted(glob.glob(os.path.join(folder, "master.organizer-backup-*.db")))
    for path in old[:-KEEP_BACKUPS]:
        for p in (path, path[:-3] + ".masterPlaylists6.xml"):
            if os.path.exists(p):
                os.remove(p)
    return dest


def write_changes(db, contents: dict, plan: Plan, folder_name: str,
                  ratings: dict[str, int], log: Callable[[str], None] = print,
                  merge: bool = True) -> tuple[int, int, int]:
    """Apply ratings and the plan. Returns (playlists created, playlists added to, tracks added).

    With merge, same-named folders and playlists (ignoring case) are reused and
    new tracks are appended; nothing already in them is removed. Smart playlists
    are never touched.
    """
    from pyrekordbox.masterdb import models

    changed = 0
    for track_id, stars in ratings.items():
        content = contents[track_id]
        if _int(content.Rating) != stars:
            content.Rating = stars
            changed += 1
    if ratings:
        log(f"Updated star ratings on {changed} tracks.")

    created = updated = added = 0
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

        def folder_for(path: tuple):
            if path not in folders:
                parent = folder_for(path[:-1])
                existing = child(parent.ID, path[-1], FOLDER)
                folders[path] = (existing if existing is not None
                                 else db.create_playlist_folder(path[-1], parent=parent))
            return folders[path]

        for folder_title, lists in plan:
            sub = folder_for(folder_path(folder_title))
            for title, ids in lists:
                pl = child(sub.ID, title, PLAYLIST)
                if pl is None:
                    pl = db.create_playlist(title, parent=sub)
                    have, last = set(), 0
                    created += 1
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

    db.commit()
    return created, updated, added
