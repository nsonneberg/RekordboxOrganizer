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
