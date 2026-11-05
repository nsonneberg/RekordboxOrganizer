"""Manually verified genre labels for benchmarking (kept out of the repository).

rekordbox tags are what the app learns from, but they're unverified: a typo,
a label name in the genre field, or a quick guess all count. A benchmark
needs ground truth someone has actually checked. This module keeps that list
in benchmark_labels.json in the app's data folder (never in the repo: it
lists your songs).

Workflow:

    python benchmark_labels.py candidates [--playlist NAME] [--xml FILE]
        Adds tracks as *candidates* (unverified), with their current tag as a
        hint. Nothing is ground truth yet.
    python benchmark_labels.py export FILE.csv
        Writes candidates to a spreadsheet: fill in verified_genre (family or
        subgenre, e.g. "Tech House") and evidence (how you checked: listened
        and compared with the label's / Beatport's genre for this exact
        version, etc.). Leave verified_genre empty to skip a row.
    python benchmark_labels.py import FILE.csv --by "your name"
        Rows with a verified_genre and evidence become verified labels.
    python benchmark_labels.py verify ID --genre "Tech House" --evidence "..." --by NAME
    python benchmark_labels.py reject ID
    python benchmark_labels.py stats
        Verified recordings per genre and split, and how many more each
        major genre needs to reach TARGET_PER_GENRE.

Splits: each label is assigned "dev" or "test" from a hash of its base title
(artist / version text removed), so remixes, edits and copies of one song
always share a split. The test split is for the final measurement only:
benchmark_genres.py evaluates it only with --split test --final, and every
such run is logged here (test_runs), so it's visible if it has been looked at
more than intended. Nothing in the app is trained or tuned on it.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

VERSION = 1
TEST_SHARE = 0.3                # share of song groups held out for the final test
TARGET_PER_GENRE = 25
MAJOR = ("House", "Techno", "Trance", "Drum & Bass", "Dubstep", "Hip Hop", "Pop")
FOCUS = ("Deep House", "Tech House", "Progressive House", "Melodic Techno", "Psytrance",
         "UK Garage", "Afro House")          # confusable subgenres worth extra examples
CSV_FIELDS = ("id", "artist", "title", "current_tag", "verified_genre", "evidence")


def path() -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), "benchmark_labels.json")


def load(file: Optional[str] = None) -> dict:
    try:
        with open(file or path(), encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == VERSION:
            data.setdefault("entries", {})
            data.setdefault("test_runs", [])
            return data
    except (OSError, ValueError):
        pass
    return {"version": VERSION, "entries": {}, "test_runs": []}


def save(data: dict, file: Optional[str] = None) -> None:
    file = file or path()
    os.makedirs(os.path.dirname(file), exist_ok=True)
    tmp = file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, file)


def split_of(title: str) -> str:
    """'dev' or 'test', stable for every version of a song."""
    import genres
    key = genres._base_title(title) or title.casefold()
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return "test" if h % 1000 < TEST_SHARE * 1000 else "dev"


def add_candidates(data: dict, tracks, library=None) -> int:
    """Add tracks as unverified candidates. Returns how many were new.

    library: every track (default: `tracks`). Related recordings in it
    (versions, mashups, shared artists: genres.version_groups) get the same
    split as each other, and as any of them already listed.
    """
    import genres
    library = list(library or tracks)
    related = genres.version_groups(library + [t for t in tracks if t not in library])
    group_split = {}
    for t in sorted(library, key=lambda t: t.id):
        e = data["entries"].get(t.id)
        if e and e.get("split"):
            group_split.setdefault(related[t.id], e["split"])
    n = 0
    for t in tracks:
        if t.id in data["entries"]:
            continue
        split = group_split.setdefault(related[t.id], split_of(t.name))
        data["entries"][t.id] = {"artist": t.artist, "title": t.name, "length": t.length,
                                 "current_tag": t.genre, "status": "candidate",
                                 "split": split}
        n += 1
    return n


def verify(data: dict, tid: str, genre: str, evidence: str, by: str) -> None:
    import genres
    if tid not in data["entries"]:
        raise KeyError(f"no candidate with id {tid}")
    genre = " ".join(genre.split())
    if not genre or genres.is_vague(genre):
        raise ValueError(f"'{genre}' is too vague to be ground truth")
    if not genres.known_family(genres.family_of(genre)):
        raise ValueError(f"'{genre}' isn't a genre the taxonomy knows (see taxonomy.py)")
    if not evidence.strip():
        raise ValueError("say how you checked it (evidence)")
    if not by.strip():
        raise ValueError("say who checked it (--by)")
    e = data["entries"][tid]
    e.update(status="verified", genre=genre, evidence=evidence.strip(), verified_by=by.strip(),
             verified_on=datetime.date.today().isoformat())


def verified(data: dict, tracks, split: Optional[str] = None) -> dict[str, str]:
    """{track id: verified genre} for tracks in the library (matched by id, else by name)."""
    import genres
    by_id = {t.id: t for t in tracks}
    by_name = defaultdict(list)
    for t in tracks:
        by_name[(genres._norm_genre(t.artist), genres._norm_genre(t.name))].append(t)
    out = {}
    for tid, e in data["entries"].items():
        if e.get("status") != "verified" or (split and e.get("split") != split):
            continue
        if tid in by_id:
            out[tid] = e["genre"]
            continue
        for t in by_name.get((genres._norm_genre(e["artist"]), genres._norm_genre(e["title"])), []):
            if not e.get("length") or abs(t.length - e["length"]) <= 3:
                out[t.id] = e["genre"]
                break
    return out


def stats(data: dict) -> str:
    import genres
    rows = [e for e in data["entries"].values() if e.get("status") == "verified"]
    fam = Counter((genres.family_of(e["genre"]), e["split"]) for e in rows)
    sub = Counter((genres.subgenre_of(e["genre"])[1], e["split"]) for e in rows)
    cand = sum(1 for e in data["entries"].values() if e.get("status") == "candidate")
    lines = [f"{len(rows)} verified labels, {cand} unverified candidates; "
             f"test split evaluated {len(data['test_runs'])} time(s).", "",
             f"{'Genre':<20}{'dev':>6}{'test':>6}{'needed':>8}"]
    for f in MAJOR:
        have = fam[(f, 'dev')] + fam[(f, 'test')]
        lines.append(f"{f:<20}{fam[(f, 'dev')]:>6}{fam[(f, 'test')]:>6}"
                     f"{max(0, TARGET_PER_GENRE - have):>8}")
    lines += ["", "Confusable subgenres:"]
    for s in FOCUS:
        lines.append(f"{s:<20}{sub[(s, 'dev')]:>6}{sub[(s, 'test')]:>6}"
                     f"{max(0, TARGET_PER_GENRE - sub[(s, 'dev')] - sub[(s, 'test')]):>8}")
    return "\n".join(lines)


def log_test_run(data: dict, note: str) -> None:
    data["test_runs"].append({"date": datetime.datetime.now().isoformat(timespec="seconds"),
                              "note": note})


def _tracks(args):
    """(the tracks to add, the whole library)."""
    import organizer
    if args.xml:
        library = organizer.load_library(args.xml)[1]
        return library, library
    import rbdb
    db = rbdb.open_database()
    try:
        library = rbdb.load_tracks(db)[0]
        tracks = library
        if args.playlist:
            ids = set()
            for pl in db.get_playlist():
                if pl.Name == args.playlist and pl.Attribute == 0:
                    ids |= {str(c.ID) for c in db.get_playlist_contents(pl)}
            if not ids:
                sys.exit(f"No playlist called '{args.playlist}' (or it's empty).")
            tracks = [t for t in library if t.id in ids]
        return tracks, library
    finally:
        db.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates", help="add library tracks as unverified candidates")
    c.add_argument("--playlist")
    c.add_argument("--xml")
    e = sub.add_parser("export", help="write candidates to a CSV to fill in")
    e.add_argument("csv")
    i = sub.add_parser("import", help="read verified genres back from the CSV")
    i.add_argument("csv")
    i.add_argument("--by", required=True)
    v = sub.add_parser("verify")
    v.add_argument("id")
    v.add_argument("--genre", required=True)
    v.add_argument("--evidence", required=True)
    v.add_argument("--by", required=True)
    r = sub.add_parser("reject")
    r.add_argument("id")
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    data = load()
    if args.cmd == "candidates":
        n = add_candidates(data, *_tracks(args))
        save(data)
        print(f"Added {n} candidates (unverified). Next: export a CSV and verify them.")
    elif args.cmd == "export":
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, CSV_FIELDS)
            w.writeheader()
            for tid, en in sorted(data["entries"].items()):
                if en.get("status") == "candidate":
                    w.writerow({"id": tid, "artist": en["artist"], "title": en["title"],
                                "current_tag": en.get("current_tag", ""),
                                "verified_genre": "", "evidence": ""})
        print(f"Wrote {args.csv}. It lists your songs: keep it private.")
    elif args.cmd == "import":
        done, skipped = 0, []
        with open(args.csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if not (row.get("verified_genre") or "").strip():
                    continue
                try:
                    verify(data, row["id"], row["verified_genre"], row.get("evidence", ""), args.by)
                    done += 1
                except (KeyError, ValueError) as exc:
                    skipped.append(f"{row['id']}: {exc}")
        save(data)
        print(f"Verified {done}.", *(["Skipped:"] + skipped if skipped else []), sep="\n")
    elif args.cmd == "verify":
        verify(data, args.id, args.genre, args.evidence, args.by)
        save(data)
        print("Verified.")
    elif args.cmd == "reject":
        data["entries"].pop(args.id, None)
        save(data)
        print("Removed.")
    print(stats(data) if args.cmd == "stats" else "")


if __name__ == "__main__":
    main()
