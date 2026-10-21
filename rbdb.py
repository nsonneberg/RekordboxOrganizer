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

from organizer import COLOUR_ORDER, Plan, Track, to_camelot   # noqa: E402

KEEP_BACKUPS = 3
SAMPLER_MARKER = "/PioneerDJ/Sampler/"


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
        folder = (c.FolderPath or "").replace("\\", "/")
        if SAMPLER_MARKER in folder:          # rekordbox's built-in sampler sounds
            continue
        colour = c.ColorName or ""
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
            date_added=c.StockDate or c.DateCreated or "",
            play_count=_int(c.DJPlayCount),
            path=os.path.normpath(c.FolderPath) if c.FolderPath else "",
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
                  ratings: dict[str, int], log: Callable[[str], None] = print) -> None:
    from pyrekordbox.masterdb import models

    changed = 0
    for track_id, stars in ratings.items():
        content = contents[track_id]
        if _int(content.Rating) != stars:
            content.Rating = stars
            changed += 1
    if ratings:
        log(f"Updated star ratings on {changed} tracks.")

    if plan:
        for old in db.get_playlist(Name=folder_name, ParentID="root").all():
            if old.Attribute == 1 and not old.rb_local_deleted:   # 1 = folder
                log(f"Replacing the existing '{folder_name}' folder.")
                db.delete_playlist(old)
        db.flush()

        top = db.create_playlist_folder(folder_name)
        for folder_title, lists in plan:
            sub = db.create_playlist_folder(folder_title, parent=top)
            for title, ids in lists:
                pl = db.create_playlist(title, parent=sub)
                now = datetime.datetime.now()
                # Same as db.add_to_playlist(), minus a COUNT query per track.
                for n, track_id in enumerate(ids, 1):
                    db.add(models.DjmdSongPlaylist.create(
                        ID=str(uuid4()), PlaylistID=str(pl.ID), ContentID=str(track_id),
                        TrackNo=n, UUID=str(uuid4()), created_at=now, updated_at=now))

    db.commit()
