"""A read-only genre dry run: what the app would suggest for every song.

Nothing is written to rekordbox (no tags, playlists or database changes). The
library file is fingerprinted (size, modified time, SHA-256) before and after
and the report says whether it changed.

For each song the report lists the existing tag, the predicted family and
subgenre, the confidence, every source's own answer, the release-metadata
match, and a recommended action:

  accept           untagged / vague tag, a confident guess for a genre that has
                   proved reliable (it would go in a "(detected)" playlist and be
                   pre-filled in the review window)
  suggest          untagged / vague tag, a clear guess for a genre that hasn't
                   proved reliable yet: shown for the user to approve
  review           untagged / vague tag, but unclear or the sources disagree
  leave unchanged  already has a specific tag (manual tags are never overwritten;
                   the prediction is shown for comparison, made with the song's
                   own tag hidden), or there's nothing to go on

Usage:
    python dry_run_genres.py [library.xml] [--model yamnet|discogs-effnet] [--lookup]

--lookup looks songs up on MusicBrainz online (artist, title, length are sent);
without it only cached answers are used and nothing is sent.

The report goes to benchmark_results/ (git-ignored): it lists your songs, so
keep it private.
"""

from __future__ import annotations

import csv
import hashlib
import html
import os
import sys
import time

import genres
import organizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "benchmark_results")


def fingerprint(path: str) -> tuple:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns, h.hexdigest()


def load(paths):
    if paths:
        return organizer.load_library(paths[0])[1], paths[0]
    import rbdb
    db = rbdb.open_database()
    try:
        return rbdb.load_tracks(db)[0], rbdb.database_path(db)
    finally:
        db.close()


def source_text(p, source):
    for s, fam, share, weight in p.evidence:
        if s == source:
            return f"{fam} ({share:.0%}, counts {weight:.2f})"
    return ""


def main(argv):
    paths = [a for a in argv if not a.startswith("--")]
    backend = "discogs-effnet"
    if "--model" in argv:
        backend = argv[argv.index("--model") + 1]
        paths = [a for a in paths if a != backend]
    online = "--lookup" in argv
    tracks, library_file = load(paths)
    before = fingerprint(library_file)
    t0 = time.time()
    log = print
    features = organizer.analyse_tracks(tracks, log)
    results = organizer.model_genres(tracks, log, backend=backend)
    import metadata
    from version import VERSION
    meta, stats = metadata.lookup_tracks([t for t in tracks if t.name],
                                         [metadata.MusicBrainz(version=VERSION)], log,
                                         offline=not online)
    labels = genres.load_labels()
    trusted = genres.training_labels(tracks, labels)

    # Untagged / vague songs: exactly what the app would do.
    predictions = genres.predict_genres(tracks, features, log, results, labels, metadata=meta)
    # Tagged songs: the prediction with the song (and its other versions) hidden.
    held_out = {}
    ev = genres.evaluate(tracks, features, results, labels, metadata=meta)
    for probe, _, p in ev.rows:
        held_out[probe] = p
    group = genres._groups(tracks)
    probe_of = {group[pid]: pid for pid in held_out}

    rows = []
    for t in sorted(tracks, key=lambda t: (t.artist.casefold(), t.name.casefold())):
        vague = genres.is_vague(t.genre)
        if vague:
            p = predictions.get(t.id)
            kind = "untagged" if not t.genre.strip() else "vague tag"
        else:
            p = held_out.get(t.id) or held_out.get(probe_of.get(group[t.id], ""))
            kind = "tagged" if t.id in trusted else "tagged (not used)"
        if p is None:
            action = "leave unchanged"
        elif not vague:
            action = "leave unchanged"
        elif not p.candidates or set(p.sources) <= {"tempo", "tag"}:
            action = "leave unchanged"
        elif p.confident:
            action = "accept"
        elif p.suggested:
            action = "suggest"
        else:
            action = "review"
        m = meta.get(t.id)
        note = ""
        if not vague and p is not None and p.candidates and \
                p.candidates[0][0] != genres.family_of(t.genre) and genres.known_family(
                    genres.family_of(t.genre)):
            note = "prediction differs from the tag (tag kept)"
        rows.append({
            "title": t.name, "artist": t.artist, "existing tag": t.genre, "kind": kind,
            "predicted family": p.family if p else "",
            "subgenre (assigned)": p.subgenre if p else "",
            "subgenre (suggested)": (p.suggested_subgenre if p else ""),
            "confidence": f"{p.score:.2f}" if p and p.candidates else "",
            "status": p.status if p else "",
            "auto threshold": "" if not p or p.threshold is None else f"{p.threshold:.2f}",
            "audio model": source_text(p, "model") if p else "",
            "your tagged songs": source_text(p, "library") if p else "",
            "release metadata": source_text(p, "metadata") if p else "",
            "metadata match": (f"{m.describe} [{m.source}, {m.level}, quality {m.quality:.2f}]"
                               if m else ("not found" if online else "not looked up / no cached match")),
            "disagreement": "yes" if p and p.disagreement else "",
            "reason": p.reason if p else "",
            "recommended action": action, "note": note})

    threshold, eligible = genres.auto_eligibility(tracks, features, results, labels, meta)
    gate = (f"Automatic-tagging threshold {threshold:.2f}. " if threshold is not None
            else "No automatic-tagging threshold (too few tagged songs, or none met the target). ")
    gate += ("Per genre, held-out guesses at or above it: " + "; ".join(
        f"{f}: {k} checks, {w} wrong, error up to {ub:.0%} (95% bound) -> "
        f"{'automatic' if ok else 'suggestions only'}" for f, k, w, ub, ok in eligible)
        if eligible else "No genre has held-out checks.")
    gate += " Genres not listed have no checks: suggestions only."

    after = fingerprint(library_file)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "dry_run_genres.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["recommended action"]] = counts.get(r["recommended action"], 0) + 1
    unchanged = before == after
    summary = (f"{len(rows)} songs: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
               + f". Model: {backend}. Metadata: {'online lookups' if online else 'cache only'} "
               f"({stats.matched} matched). Library file {'UNCHANGED' if unchanged else 'CHANGED'}"
               f" (SHA-256 {after[2][:16]}...). {time.time() - t0:.0f} s.")
    html_path = os.path.join(OUT_DIR, "dry_run_genres.html")
    cols = list(rows[0])
    body = "".join("<tr class='%s'>%s</tr>" % (
        r["recommended action"].split()[0],
        "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols)) for r in rows)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><meta charset="utf-8"><title>Genre dry run</title>
<style>body{{font:13px system-ui,sans-serif;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:3px 6px;vertical-align:top}}th{{background:#eee;position:sticky;top:0}}
tr.accept td{{background:#e8f6e8}}tr.suggest td{{background:#e8f0fb}}tr.review td{{background:#fff6dc}}</style>
<h1>Genre dry run (private: lists your songs)</h1><p>{html.escape(summary)}</p>
<p>{html.escape(gate)}</p>
<p>Nothing was written to rekordbox. Green: would be tagged automatically. Blue: suggested
(needs your approval). Yellow: needs review.
Confidence is estimated from held-out checks on your own tagged songs (unverified tags).</p>
<table><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in cols)}</tr>{body}</table>""")
    print(summary)
    print(gate)
    print(f"Report: {csv_path}\n        {html_path}")
    if not unchanged:
        sys.exit("The library file changed during the dry run (rekordbox running?). "
                 "This script doesn't write to it; check whether rekordbox was open.")


if __name__ == "__main__":
    main(sys.argv[1:])
