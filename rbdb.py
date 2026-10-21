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


# Hot cue pads A-H as stored in DjmdCue.Kind (0 is a memory cue).
# UNVERIFIED: checked against cues made in rekordbox before use.
HOT_CUES_VERIFIED = False
HOT_CUE_KINDS = (1, 2, 3, 5, 6, 7, 8, 9)
# Section label -> rekordbox hot cue colour (ColorTableIndex). UNVERIFIED.
CUE_COLOURS = {"Intro": 49, "Verse": 9, "Build-up": 37, "Breakdown": 9, "Bridge": 9,
               "Drop": 42, "Chorus": 42, "Outro": 49}


def plan_hot_cues(db, tracks: list[Track], log: Callable[[str], None] = print) -> dict:
    """{track id: [(pad 0-7, Phrase)]} for the hot cues to add, from phrase analysis."""
    import cues
    from pyrekordbox.masterdb import models

    if not HOT_CUES_VERIFIED:
        raise ValueError("Adding hot cues isn't ready yet - untick 'Add hot cues' and run again.")
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


def add_hot_cues(db, contents: dict, plan: dict, log: Callable[[str], None] = print) -> None:
    """Add the planned hot cues (saved by the next commit)."""
    from pyrekordbox.masterdb import models

    now = datetime.datetime.now()
    for track_id, picks in plan.items():
        content = contents[track_id]
        for pad, phrase in picks:
            ms = int(phrase.start_ms)
            db.add(models.DjmdCue.create(
                ID=str(uuid4()), ContentID=str(content.ID), ContentUUID=content.UUID,
                InMsec=ms, InFrame=round(ms * 150 / 1000), InMpegFrame=0, InMpegAbs=0,
                OutMsec=-1, OutFrame=0, OutMpegFrame=0, OutMpegAbs=0,
                Kind=HOT_CUE_KINDS[pad], Color=-1, ColorTableIndex=CUE_COLOURS.get(phrase.label, 0),
                ActiveLoop=0, Comment=phrase.label, BeatLoopSize=0, CueMicrosec=0,
                InPointSeekInfo="", OutPointSeekInfo="", UUID=str(uuid4()),
                created_at=now, updated_at=now))
    if plan:
        log(f"Added {sum(len(p) for p in plan.values())} hot cues.")
