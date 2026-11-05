"""A self-test of the app's write paths on a throwaway library, runnable from the built app.

    "Rekordbox Organizer.exe" --selftest OUTPUT_FOLDER [AUDIO_FOLDER]

Builds a small synthetic master.db (with masterPlaylists6.xml) and a small
XML export inside OUTPUT_FOLDER, then runs Preview, a real run (backup,
playlists, ratings), a genre save, the interrupted-change guard and Undo on
them, and decodes every file in AUDIO_FOLDER. The real rekordbox library is
never opened: pyrekordbox is pointed at the synthetic database and all of the
app's own files (settings, caches) go to OUTPUT_FOLDER/appdata. Results go to
OUTPUT_FOLDER/selftest.json; the exit code is 0 only if every check passed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import traceback

PLAYLIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MASTER_PLAYLIST Version="1.0.0" AutomaticSync="0">
  <PRODUCT Name="rekordbox" Version="6.8.0" Company="AlphaTheta"/>
  <PLAYLISTS>
    <NODE Id="64" ParentId="0" Attribute="0" Timestamp="1700000000000" Lib_Type="0" CheckType="0"/>
  </PLAYLISTS>
</MASTER_PLAYLIST>
"""


def sha(path: str) -> str:
    import rbdb
    with open(rbdb._long(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_master_db(folder: str, playlist_xml: bool = False) -> str:
    """A small master.db: 4 songs (two with your own genre tags) and one playlist."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from pyrekordbox.masterdb import models

    path = os.path.join(folder, "master.db")
    engine = create_engine(f"sqlite:///{path}")
    # rekordbox's own schema allows NULL in these columns (pyrekordbox relies on it).
    for table in models.Base.metadata.tables.values():
        for col in table.columns:
            if not col.primary_key:
                col.nullable = True
    models.Base.metadata.create_all(engine)
    now = datetime.datetime.now()

    def make(model, **kw):
        for col in model.__table__.columns:
            if col.name not in kw and "date" in type(col.type).__name__.lower():
                kw[col.name] = now      # pyrekordbox's date type can't store NULL
        return model(**kw)

    with Session(engine) as s:
        s.add(make(models.AgentRegistry, registry_id="localUpdateCount", int_1=0))
        s.add(make(models.DjmdGenre, ID="10", Name="Techno", UUID="g10"))
        s.add(make(models.DjmdGenre, ID="11", Name="Dance & EDM", UUID="g11"))
        songs = [("1", "Kick Drum", 12600, "10"), ("2", "Night Drive", 12800, "11"),
                 ("3", "Bassline", 17400, ""), ("4", "Warehouse", 13000, "10")]
        for cid, title, bpm, genre in songs:
            s.add(make(models.DjmdContent, ID=cid, Title=title, BPM=bpm, Rating=0,
                       GenreID=genre or None, FolderPath=f"C:/Music/{cid}.mp3",
                       StockDate="2024-01-0" + cid, rb_local_deleted=0, UUID=f"c{cid}"))
        s.add(make(models.DjmdPlaylist, ID="100", Name="My Set", Attribute=0, ParentID="root",
                   Seq=1, UUID="p100", rb_local_deleted=0))
        s.add(make(models.DjmdSongPlaylist, ID="200", PlaylistID="100", ContentID="1",
                   TrackNo=1, UUID="s200", rb_local_deleted=0))
        s.commit()
    engine.dispose()
    if playlist_xml:
        with open(os.path.join(folder, "masterPlaylists6.xml"), "w", encoding="utf-8") as f:
            f.write(PLAYLIST_XML)
    return path


def _xml_export(folder: str, audio: list[str]) -> str:
    """A rekordbox XML export whose tracks are the given audio files (unrated)."""
    from urllib.parse import quote
    from xml.sax.saxutils import quoteattr
    rows = []
    for i, path in enumerate(audio, 1):
        loc = "file://localhost/" + quote(os.path.abspath(path).replace("\\", "/"))
        rows.append(f'<TRACK TrackID="{i}" Name={quoteattr(os.path.basename(path))} '
                    f'Artist="Test" Genre="" AverageBpm="{120 + i}.00" Rating="0" '
                    f'TotalTime="12" Location={quoteattr(loc)}/>')
    text = ('<?xml version="1.0" encoding="UTF-8"?>\n<DJ_PLAYLISTS Version="1.0.0">'
            '<PRODUCT Name="rekordbox" Version="6.8.0" Company="AlphaTheta"/>'
            f'<COLLECTION Entries="{len(rows)}">' + "".join(rows) + "</COLLECTION>"
            '<PLAYLISTS><NODE Type="0" Name="ROOT" Count="0"/></PLAYLISTS></DJ_PLAYLISTS>')
    path = os.path.join(folder, "export.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run(out: str, audio_dir: str = "") -> int:
    out = os.path.abspath(out)
    lib = os.path.join(out, "library")
    os.makedirs(lib, exist_ok=False)          # always a fresh, empty folder
    os.environ["APPDATA"] = os.path.join(out, "appdata")     # settings, caches, labels
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"]
    results: list[dict] = []
    log_lines: list[str] = []

    def check(name, ok, detail=""):
        results.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    def step(name, fn):
        try:
            fn()
        except BaseException as exc:          # noqa: BLE001 - recorded, never hidden
            check(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    import pyrekordbox
    import pyrekordbox.masterdb.database as pdb
    import organizer
    import rbdb

    db = make_master_db(lib, playlist_xml=True)
    xml = os.path.join(lib, "masterPlaylists6.xml")
    original = (sha(db), sha(xml))
    real = pyrekordbox.MasterDatabase
    pyrekordbox.MasterDatabase = lambda path=None: real(path or db, unlock=False)
    rbdb.rekordbox_running = lambda: False
    pdb.get_rekordbox_pid = lambda: 0
    log = log_lines.append
    opt = organizer.Options(bpm=True, bpm_step=5, guess_genres=False)

    def preview():
        changes = organizer.Changes()
        organizer.run(opt, preview=True, changes=changes, log=log)
        check("preview changes nothing", (sha(db), sha(xml)) == original)
        check("preview makes no backup", rbdb.list_backups(lib) == [])
        check("preview lists changes", not changes.empty(), organizer.summarize_changes(changes))

    def apply():
        organizer.run(opt, log=log)
        backups = rbdb.list_backups(lib)
        check("run made one backup", len(backups) == 1, backups)
        b = backups[0]
        check("backup is the library as it was",
              (sha(b), sha(b[:-3] + ".masterPlaylists6.xml")) == original)
        check("run changed the database", sha(db) != original[0])
        import xml.etree.ElementTree as ET
        nodes = ET.parse(xml).getroot().findall(".//NODE")
        check("playlist index updated and valid", len(nodes) > 1, f"{len(nodes)} playlists")
        check("no change left marked in progress", rbdb.interrupted_change(db) is None)

    def genres():
        organizer.save_genres({"3": "Drum & Bass"}, log=log)
        d = rbdb.open_database()
        try:
            tracks, _ = rbdb.load_tracks(d)
        finally:
            d.close()
        by_id = {t.id: t.genre for t in tracks}
        check("genre save wrote only the chosen song",
              by_id == {"1": "Techno", "2": "Dance & EDM", "3": "Drum & Bass", "4": "Techno"},
              by_id)
        check("genre save made a backup", len(rbdb.list_backups(lib)) == 2)

    def interrupted():
        before = sha(db)
        rbdb.begin_change(db, rbdb.list_backups(lib)[0])
        try:
            organizer.run(opt, log=log)
            check("an unfinished change blocks the next write", False, "it ran")
        except ValueError as exc:
            check("an unfinished change blocks the next write",
                  "didn't finish" in str(exc) and sha(db) == before, exc)
        rbdb.end_change(db)

    def undo():
        oldest = rbdb.list_backups(lib)[-1]
        keep = rbdb.restore_backup(oldest, db)
        check("undo restores the exact database and playlist index",
              (sha(db), sha(xml)) == original)
        check("undo kept the library from before it", os.path.exists(rbdb._long(keep)))
        check("nothing left marked in progress", rbdb.interrupted_change(db) is None)

    def unlocked():
        stuck = []
        for name in os.listdir(lib):
            p = os.path.join(lib, name)
            try:
                os.replace(rbdb._long(p), rbdb._long(p + ".moved"))
                os.replace(rbdb._long(p + ".moved"), rbdb._long(p))
            except OSError as exc:
                stuck.append(f"{name}: {exc}")
        check("no library file left open or locked", not stuck, stuck)
        leftovers = [n for n in os.listdir(lib) if n.endswith((".partial", ".restoring"))]
        check("no temporary files left behind", not leftovers, leftovers)

    def m3():
        import energy_m3
        model = energy_m3.load_model()      # the packaged file, in the built app
        check("M3 model file loads and is the frozen model", True, energy_m3.model_path())
        score, level = energy_m3.predict(model, model["mu"])
        check("M3 scores a known input", score == model["intercept"] and level == 3, (score, level))
        song = organizer.Track("1", "Song", "", "", "", 128.0, "", 0, 2, "", "", 0, "")
        lines: list[str] = []
        new = organizer.apply_energy([song], organizer.Options(energy=True, energy_m3=True), {},
                                     lines.append)
        check("M3 skips a track without measurements, explains why and keeps its rating",
              not new and song.energy == 0 and song.rating == 2
              and any("no rekordbox analysis" in line for line in lines), lines)
        check("classic energy stays the default",
              not organizer.Options().energy_m3 and not organizer.Options().energy_experimental)

    def audio():
        import shutil as sh
        import energy
        files = sorted(os.path.join(audio_dir, n) for n in os.listdir(audio_dir))
        check("ffmpeg on PATH", True, sh.which("ffmpeg") or "not found")
        for path in files:
            try:
                samples, sr = energy.decode(path)
                check(f"decode {os.path.basename(path)}", True, f"{len(samples) / sr:.1f}s")
            except Exception as exc:
                check(f"decode {os.path.basename(path)}", None, exc)
                results[-1]["ok"] = None        # reported, judged by the caller
        export = _xml_export(out, files)
        export_sha = sha(export)
        xml_opt = organizer.Options(energy=True, energy_ratings=True, guess_genres=False)
        lines: list[str] = []
        organizer.run(xml_opt, export, log=lines.append)
        log_lines.extend(lines)
        import xml.etree.ElementTree as ET
        got = ET.parse(organizer.default_output_path(export)).getroot()
        rating = {t.get("Name"): t.get("Rating") for t in got.iterfind("COLLECTION/TRACK")}
        decoded = {os.path.basename(p) for p in files} - {
            r["check"][len("decode "):] for r in results
            if r["check"].startswith("decode ") and r["ok"] is None}
        check("decoded songs got an energy rating, undecodable ones were left unrated",
              all((rating[n] != "0") == (n in decoded) for n in rating), rating)
        check("source export untouched", sha(export) == export_sha)
        results.append({"check": "xml run log", "ok": None, "detail": "\n".join(lines)})

    for name, fn in [("preview", preview), ("apply", apply), ("genres", genres),
                     ("interrupted", interrupted), ("undo", undo), ("unlocked", unlocked), ("m3", m3)]:
        step(name, fn)
    if audio_dir:
        step("audio", audio)

    failed = [r for r in results if r["ok"] is False]
    report = {"frozen": bool(getattr(sys, "frozen", False)), "executable": sys.executable,
              "passed": not failed, "results": results, "log": log_lines}
    with open(os.path.join(out, "selftest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return 0 if not failed else 1
