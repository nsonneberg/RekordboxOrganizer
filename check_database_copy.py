"""Check direct database mode on a disposable copy of your real rekordbox library.

The synthetic test database (test_library_safety.py) can't cover rekordbox's
encryption, your library's exact schema version or the masterPlaylists6.xml
index. This script does, without touching your library:

1. fingerprints master.db, its journal files and masterPlaylists6.xml (SHA-256);
2. copies them into a new temporary folder and opens only the copy;
3. on the copy: previews a run (checks nothing changed), backs it up, writes
   BPM playlists into a new "RBO safety check" folder and one genre tag, checks
   that nothing else changed, then restores the backup and checks the copy is
   back exactly as it was;
4. deletes the copy and confirms your real files are byte-for-byte unchanged.

rekordbox must be closed (so the copy is consistent). The output has counts
only, no song names.

Usage:
    python check_database_copy.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

import organizer
import rbdb

FOLDER = "RBO safety check"


def fingerprints(folder: str) -> dict:
    out = {}
    for name in ("master.db", "master.db-wal", "master.db-shm", "masterPlaylists6.xml"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def state(path: str) -> dict:
    db = rbdb.open_database(path)
    try:
        tracks, _ = rbdb.load_tracks(db)
        lists = {}
        for pl in db.get_playlist().all():
            if not pl.rb_local_deleted:
                lists[str(pl.ID)] = (str(pl.ParentID), pl.Name, pl.Attribute, sorted(
                    (s.TrackNo, str(s.ContentID))
                    for s in db.get_playlist_songs(PlaylistID=pl.ID).all()))
        return {"tracks": {t.id: (t.genre, t.rating, t.name) for t in tracks}, "playlists": lists}
    finally:
        db.close()


def check(ok: bool, what: str, failures: list) -> None:
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


def main() -> int:
    if rbdb.rekordbox_running():
        sys.exit("Please close rekordbox first (the copy has to be consistent).")
    real_path = rbdb.library_path()
    real_dir = os.path.dirname(real_path)
    before = fingerprints(real_dir)
    work = tempfile.mkdtemp(prefix="rbo-db-check-")
    copy = os.path.join(work, "master.db")
    assert os.path.commonpath([work, real_dir]) != real_dir, "the copy must be elsewhere"
    failures: list = []
    try:
        for name in before:
            shutil.copy2(os.path.join(real_dir, name), os.path.join(work, name))
        print(f"Copied the library to a temporary folder ({len(before)} files).")
        original = state(copy)
        print(f"  {len(original['tracks'])} tracks, {len(original['playlists'])} playlists "
              "and folders in the copy.")

        db = rbdb.open_database(copy)
        check(os.path.samefile(rbdb.database_path(db), copy), "working on the copy", failures)
        tracks, contents = rbdb.load_tracks(db)
        opt = organizer.Options(bpm=True, bpm_step=5, guess_genres=False)
        plan = organizer.build_plan(tracks, opt)
        preview = organizer.Changes()
        rbdb.write_changes(db, contents, plan, FOLDER, {}, lambda m: None,
                           changes=preview, commit=False)
        db.session.rollback()
        db.close()
        check(state(copy) == original, "a preview changes nothing", failures)

        saved = rbdb.backup(copy)
        check(os.path.exists(saved), "backup made before writing", failures)

        db = rbdb.open_database(copy)
        tracks, contents = rbdb.load_tracks(db)
        real_run = organizer.Changes()
        rbdb.write_changes(db, contents, organizer.build_plan(tracks, opt), FOLDER, {},
                           lambda m: None, changes=real_run)
        vague = next((t.id for t in tracks if not t.genre.strip()), None)
        if vague:
            rbdb.write_genres(db, contents, {vague: "RBO Check Genre"}, lambda m: None)
            db.commit()
        db.close()
        check(real_run.created == preview.created and real_run.added == preview.added,
              f"the preview matched the real run ({len(real_run.created)} playlists, "
              f"{len(real_run.added)} entries)", failures)

        after = state(copy)
        made = [v for k, v in after["playlists"].items() if k not in original["playlists"]]
        check(any(name == FOLDER for _, name, _, _ in made), "playlists were written", failures)
        kept = all(after["playlists"].get(k) == v for k, v in original["playlists"].items())
        check(kept, "every existing playlist is unchanged", failures)
        others = {k: v for k, v in after["tracks"].items() if k != vague}
        check(others == {k: v for k, v in original["tracks"].items() if k != vague},
              "every other track's genre and rating is unchanged", failures)
        if vague:
            check(after["tracks"][vague][0] == "RBO Check Genre", "the genre tag was written",
                  failures)

        rbdb.restore_backup(saved, copy)
        check(state(copy) == original, "restoring the backup puts everything back", failures)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(fingerprints(real_dir) == before,
          "your real library files are byte-for-byte unchanged", failures)
    check(not os.path.exists(work), "the temporary copy was deleted", failures)
    print("All checks passed." if not failures else f"{len(failures)} check(s) FAILED.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
