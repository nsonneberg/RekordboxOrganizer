"""Pick songs from your library to rate for energy, in batches (read-only).

    python energy_labelling.py select [FOLDER]   build the candidate pool and rating sheets
    python energy_labelling.py status [FOLDER]   how many songs are rated, per batch and split

Ratings made elsewhere (stars you set in rekordbox) can be added to the folder as
import_<name>.csv with track_id and rating columns; they keep the frozen split.

The first rating round (benchmark_energy.py template) picked songs at random per
genre family and never looked at energy, so calm songs were rare. This picks
broadly instead: songs either baseline rates 1-2, songs the baselines disagree
on, slower or quieter songs, fast songs that look soft, songs resembling ones you
rated 1-2, and songs from parts of the library that have few ratings so far. Songs
matching none of these are still included, in the last batches, so the ratings
also cover the library as it is. None of the criteria says what a song's energy
is: they only decide what gets listened to first.

The rating sheets (batch_NN.csv) show only id, artist and title, in shuffled
order: no genre, stars, predictions or criteria. Those are in manifest.csv,
which you shouldn't open while rating. Batch 1 holds the low-energy candidates
plus a few other songs, so it doesn't announce what it's for.

Before anything is rated, about 30% of the new songs are set aside as a held-out
test set, by artist cluster (every song sharing an artist or a recording lands on
the same side) and stratified per batch. Songs linked to ones you've already rated
are always development songs. The split is fixed in manifest.csv, whose hash is
kept in summary.json; ratings are only read through load_ratings(), which returns
the held-out ones only when asked for them by name.

Nothing is written to your library: results go to benchmark_results/ (private,
not committed). Existing sheets are never overwritten, so ratings can't be lost.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

import benchmark_energy as bench
import energy

ORIGINAL_LABELS = os.path.join(bench.OUT_DIR, "energy_labels_template.csv")
DEFAULT_DIR = os.path.join(bench.OUT_DIR, "energy_rating_round2")

LOW_SHARE = 0.30        # "slow" / "quiet" / "soft": the library's lowest 30%
HELD_OUT_SHARE = 0.30   # share of new artist clusters set aside for testing
LOW_BATCH = 34          # low-energy candidates in batch 1 (at least 30 when there are)
CONTROLS = 6            # other songs mixed into batch 1
BATCH_SIZE = 40
SALT = "energy-round2"  # fixes the shuffles and the split

# Feature columns compared when looking for songs like the ones you rated 1-2.
SIMILAR_ON = ("felt_bpm", "lufs", "density", "onsets", "pulse", "w_punch", "w_power", "w_highs")

CRITERIA = {
    "old_waveform_1_2": "old waveform algorithm rates it 1-2",
    "tempo_only_1_2": "tempo-only baseline rates it 1-2",
    "baselines_disagree": "old waveform and tempo-only differ by 2+ levels",
    "slow": "tempo among the library's slowest 30%",
    "quiet": "loudness among the library's quietest 30%",
    "fast_but_soft": "faster than the median, but quiet, little full-level sound or thin",
    "like_your_1_2": "the most similar song you've rated is a 1 or 2",
    "underrepresented": "from a genre family with few ratings so far",
}
LOW_SIGNALS = ("old_waveform_1_2", "tempo_only_1_2", "slow", "quiet", "fast_but_soft",
               "like_your_1_2")

# Files in the round's folder that hold ratings: the batch sheets, and ratings made
# elsewhere (e.g. stars set in rekordbox) imported as import_<name>.csv with the same
# track_id / rating columns. import_<name>.skipped.csv records what wasn't imported.
SHEET = re.compile(r"batch_\d+\.csv")
IMPORT = re.compile(r"import_[\w-]+\.csv")

# Round 3: an independent test set, kept apart from every earlier rating (see load_round3).
ROUND3_DIR = os.path.join(bench.OUT_DIR, "energy_rating_round3")
ROUND3_SHEET = re.compile(r"round3_sheet\.csv")
ROUND3_IMPORT = re.compile(r"round3_import_[\w-]+\.csv")
ROUND3_STATUS = ("scorable", "unscorable", "excluded_copy")
ROUND3_COLUMNS = ["track_id", "artist", "title", "status", "copy_of", "unit", "copies"]

_ARTIST_SPLIT = re.compile(r"[,&，]| x | vs\.? | feat\.? | ft\.? ")
_NOT_AN_ARTIST = {"", "various artists", "unknown artist", "unknown"}


def _hash(*parts) -> str:
    return hashlib.sha256("|".join(map(str, (SALT,) + parts)).encode("utf-8")).hexdigest()


def artist_names(artist: str) -> set[str]:
    return {n.strip().casefold() for n in _ARTIST_SPLIT.split(f" {artist} ")} - _NOT_AN_ARTIST


def units(rows: list[dict]) -> dict[str, str]:
    """Track id -> artist cluster: songs sharing an artist name or a recording."""
    parent = {r["id"]: r["id"] for r in rows}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for r in rows:
        for key in [("g", r["group"])] + [("a", n) for n in artist_names(r["artist"])]:
            if key in first:
                parent[root(r["id"])] = root(first[key])
            else:
                first[key] = r["id"]
    return {i: root(i) for i in parent}


def tempo_only_fit(rows: list[dict], rated: dict[str, float]):
    """The tempo-only baseline of the cross-validation: a straight line from felt
    BPM to your rating, fitted on the songs already rated (development ratings)."""
    pts = [(r["felt_bpm"], rated[r["id"]]) for r in rows if r["id"] in rated and r["felt_bpm"]]
    if len(pts) < 3:
        return None
    x, y = np.array(pts).T
    b, a = np.polyfit(x, y, 1)
    return lambda bpm: int(np.clip(np.rint(a + b * bpm), 1, 5)) if bpm else None


def _quantile(rows, key, q):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.quantile(v, q)) if v else None


def criteria_for(rows: list[dict], rated: dict[str, float], eligible: list[dict]) -> dict:
    """Track id -> the criteria it meets (see CRITERIA). Thresholds come from the
    whole library; "like_your_1_2" and the tempo-only line from rated songs."""
    measured = [r for r in rows if r.get("felt_bpm") is not None]
    slow = _quantile(measured, "felt_bpm", LOW_SHARE)
    median_bpm = _quantile(measured, "felt_bpm", 0.5)
    quiet = _quantile(measured, "lufs", LOW_SHARE)
    weak_power = _quantile(measured, "w_power", LOW_SHARE)
    thin = _quantile(measured, "density", LOW_SHARE)
    tempo_level = tempo_only_fit(rows, rated)

    # Rated share per family: a family is underrepresented when its share of
    # rated songs is below the library's, or it has fewer than 5 ratings.
    fam_all = Counter(r["family"] for r in rows)
    fam_rated = Counter(r["family"] for r in rows if r["id"] in rated)
    overall = len(rated) / max(1, len(rows))
    under = {f for f, n in fam_all.items() if fam_rated[f] < 5 or fam_rated[f] / n < overall}

    # Standardised features for similarity to the songs you rated.
    stats = {}
    for k in SIMILAR_ON:
        v = np.array([r[k] for r in measured if r.get(k) is not None], float)
        if len(v) > 1 and v.std() > 0:
            stats[k] = (v.mean(), v.std())
    refs = [r for r in measured if r["id"] in rated]

    def distance(a, b):
        d = [((a[k] - b[k]) / s) ** 2 for k, (_, s) in stats.items()
             if a.get(k) is not None and b.get(k) is not None]
        return float(np.sqrt(np.mean(d))) if d else float("inf")

    out = {}
    for r in eligible:
        met = set()
        if r["family"] in under:
            met.add("underrepresented")
        if r.get("felt_bpm") is not None:
            old = r.get("old_waveform_level")
            tempo = tempo_level(r["felt_bpm"]) if tempo_level else None
            r["tempo_only_level"] = tempo
            if old is not None and old <= 2:
                met.add("old_waveform_1_2")
            if tempo is not None and tempo <= 2:
                met.add("tempo_only_1_2")
            if old is not None and tempo is not None and abs(old - tempo) >= 2:
                met.add("baselines_disagree")
            if r["felt_bpm"] <= slow:
                met.add("slow")
            if r.get("lufs") is not None and r["lufs"] <= quiet:
                met.add("quiet")
            if r["felt_bpm"] >= median_bpm and (
                    (r.get("lufs") is not None and r["lufs"] <= quiet)
                    or (r.get("w_power") is not None and r["w_power"] <= weak_power)
                    or (r.get("density") is not None and r["density"] <= thin)):
                met.add("fast_but_soft")
            if refs:
                near = min(refs, key=lambda q: (distance(r, q), q["id"]))
                if distance(r, near) < float("inf") and rated[near["id"]] <= 2:
                    met.add("like_your_1_2")
        out[r["id"]] = met
    return out


def diverse_order(items: list[dict], priority, unit_of: dict[str, str]) -> list[dict]:
    """Highest priority first, but each pick counts against songs from the same
    artist cluster, genre family and 8-BPM band, so the top isn't 30 lookalikes."""
    left, picked = list(items), []
    n_unit, n_fam, n_bpm = Counter(), Counter(), Counter()
    band = lambda r: int((r.get("felt_bpm") or 0) // 8)
    while left:
        best = max(left, key=lambda r: (priority(r) - 1.0 * n_unit[unit_of[r["id"]]]
                                        - 0.25 * n_fam[r["family"]] - 0.25 * n_bpm[band(r)],
                                        _hash(r["id"])))
        left.remove(best)
        picked.append(best)
        n_unit[unit_of[best["id"]]] += 1
        n_fam[best["family"]] += 1
        n_bpm[band(best)] += 1
    return picked


def select(rows: list[dict], rated: dict[str, float]) -> dict:
    """The candidate pool, its batches and the development / held-out split.

    rows: every library track as a dict (id, artist, title, family, group,
    path_ok and the measured features, None when missing). rated: track id ->
    your rating so far. Every track ends up either in the pool or excluded with
    a reason; nothing that meets a criterion is dropped to hit a target size.
    """
    by_id = {r["id"]: r for r in rows}
    excluded = {}
    rated_groups = {by_id[i]["group"] for i in rated if i in by_id}
    kept_group = {}
    for r in sorted(rows, key=lambda r: (not r["path_ok"], r["id"])):
        if r["id"] in rated:
            excluded[r["id"]] = "already rated"
        elif not r["path_ok"]:
            excluded[r["id"]] = "audio file not found, so it can't be listened to"
        elif r["group"] in rated_groups:
            excluded[r["id"]] = "a copy of a recording you've already rated"
        elif r["group"] in kept_group:
            excluded[r["id"]] = f"a copy of {kept_group[r['group']]}, which is in the pool"
        else:
            kept_group[r["group"]] = r["id"]
    eligible = [r for r in rows if r["id"] not in excluded]
    met = criteria_for(rows, rated, eligible)
    unit_of = units(rows)
    low_signals = lambda r: sum(c in met[r["id"]] for c in LOW_SIGNALS)

    low = diverse_order([r for r in eligible if low_signals(r)], low_signals, unit_of)
    tier = lambda r: (2 * ("baselines_disagree" in met[r["id"]])
                      + ("underrepresented" in met[r["id"]]))
    others = diverse_order([r for r in eligible if not low_signals(r)], tier, unit_of)
    controls = diverse_order(others, lambda r: 0, unit_of)[:CONTROLS]
    first = low[:LOW_BATCH] + controls
    rest = low[LOW_BATCH:] + [r for r in others if r not in controls]
    batches = [first] + [rest[i:i + BATCH_SIZE] for i in range(0, len(rest), BATCH_SIZE)]
    batch_of = {r["id"]: n for n, b in enumerate(batches, 1) for r in b}

    # Held-out split: whole artist clusters, none touching an already-rated song,
    # about HELD_OUT_SHARE of the clusters first seen in each batch.
    linked = {unit_of[i] for i in rated if i in unit_of}
    first_batch = {}
    for r in eligible:
        u = unit_of[r["id"]]
        first_batch[u] = min(first_batch.get(u, 99), batch_of[r["id"]])
    held_out = set()
    for b in range(1, len(batches) + 1):
        free = sorted((u for u, fb in first_batch.items() if fb == b and u not in linked), key=_hash)
        held_out |= set(free[:round(HELD_OUT_SHARE * len(free))])

    pool = []
    for r in eligible:
        u = unit_of[r["id"]]
        pool.append({**r, "batch": batch_of[r["id"]], "unit": u,
                     "split": "held_out" if u in held_out else "dev",
                     "criteria": sorted(met[r["id"]]), "low_signals": low_signals(r),
                     "low_batch": r in low[:LOW_BATCH]})
    pool.sort(key=lambda p: (p["batch"], _hash(p["id"])))
    missing = [r["id"] for r in eligible if met[r["id"]] and r["id"] not in batch_of]
    assert not missing, missing   # every eligible track meeting a criterion is in the pool
    return {"pool": pool, "excluded": excluded, "low_candidates": len(low),
            "counts": {c: sum(c in m for m in met.values()) for c in CRITERIA},
            "any_criterion": sum(bool(m) for m in met.values()),
            "no_criterion": sum(not m for m in met.values())}


# --------------------------------------------------------------------------- #
# Reading the library and writing the sheets
# --------------------------------------------------------------------------- #

def library_rows(paths=()) -> list[dict]:
    """Every track with what the selection needs. Reads only."""
    import genres
    tracks = bench._load_library(paths)
    feats = bench.load_features()
    group = genres._groups(tracks)
    ids = [i for i in feats if feats[i].get("waveform_v3")]
    old = energy.relative_levels(bench.v3_scores({i: feats[i]["waveform_v3"] for i in ids},
                                                 {i: feats[i]["bpm"] for i in ids},
                                                 {i: feats[i]["genre"] for i in ids}))
    rows = []
    for t in tracks:
        f = feats.get(t.id, {})
        ap, wv, v3w = f.get("audio_phrases") or f.get("audio") or {}, f.get("waveform") or {}, f.get("waveform_v3") or {}
        measured = bool(ap or wv)
        rows.append({
            "id": t.id, "artist": t.artist, "title": t.name, "group": group[t.id],
            "family": genres.family_of(t.genre) or "(none)",
            "path_ok": bool(t.path) and os.path.exists(t.path),
            "felt_bpm": energy.effective_bpm(t.bpm, t.genre) if measured and t.bpm else None,
            "lufs": ap.get("x_lufs"), "density": ap.get("x_density"), "onsets": ap.get("x_onsets"),
            "pulse": ap.get("x_pulse"), "w_punch": wv.get("x_punch"),
            "w_power": v3w.get("e_power"), "w_highs": v3w.get("e_highs"),
            "old_waveform_level": old.get(t.id),
        })
    return rows


def _rated_so_far() -> dict[str, float]:
    labels = bench.load_labels(ORIGINAL_LABELS) if os.path.exists(ORIGINAL_LABELS) else {}
    return {i: float(np.mean([x for x, _ in v])) for i, v in labels.items()}


def write_selection(result: dict, folder: str) -> dict:
    if os.path.exists(folder) and os.listdir(folder):
        raise SystemExit(f"{folder} already exists and isn't empty: its sheets may hold ratings, "
                         "so it's left alone. Pass another folder name.")
    os.makedirs(folder, exist_ok=True)
    pool = result["pool"]
    for b in sorted({p["batch"] for p in pool}):
        sheet = sorted((p for p in pool if p["batch"] == b), key=lambda p: _hash("sheet", p["id"]))
        with open(os.path.join(folder, f"batch_{b:02d}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "artist", "title", "rating", "notes"])
            for p in sheet:
                w.writerow([p["id"], p["artist"], p["title"], "", ""])
    manifest = os.path.join(folder, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "artist", "title", "batch", "split", "unit", "low_batch",
                    "low_signals", "criteria", "old_waveform_level", "tempo_only_level", "family"])
        for p in pool:
            w.writerow([p["id"], p["artist"], p["title"], p["batch"], p["split"], p["unit"],
                        int(p["low_batch"]), p["low_signals"], ";".join(p["criteria"]),
                        p.get("old_waveform_level") or "", p.get("tempo_only_level") or "",
                        p["family"]])
    with open(os.path.join(folder, "excluded.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "reason"])
        for i, why in sorted(result["excluded"].items()):
            w.writerow([i, why])
    with open(manifest, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    summary = {
        "manifest_sha256": digest,
        "pool": len(pool), "excluded": Counter(result["excluded"].values()),
        "criteria": result["counts"], "any_criterion": result["any_criterion"],
        "no_criterion": result["no_criterion"], "low_candidates": result["low_candidates"],
        "low_batch": sum(p["low_batch"] for p in pool),
        "batches": dict(sorted(Counter(p["batch"] for p in pool).items())),
        "split": Counter(p["split"] for p in pool),
        "held_out_per_batch": dict(sorted(Counter(p["batch"] for p in pool
                                                  if p["split"] == "held_out").items())),
    }
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(folder, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Rate the songs in batch_01.csv, batch_02.csv, ... by listening: fill in 'rating'.\n"
                "Don't open manifest.csv or summary.json while rating: they hold the selection\n"
                "criteria and predictions. Leave a song blank if you can't judge it.\n\n"
                + bench.RATING_GUIDE + "\n")
    return summary


def load_ratings(folder: str, split: str, imports: bool = True,
                 round3: str | None = ROUND3_DIR) -> dict[str, list[tuple[float, str]]]:
    """Your ratings in one split: "dev" (the first round plus new development
    songs) or "held_out". Imported ratings count too (imports=False leaves them
    out: you picked those songs yourself, the batches didn't). The split always
    comes from manifest.csv, whoever rated a song and wherever; refuses if
    manifest.csv was changed after selection, or if a round-3 test song (or a
    copy of one) has a rating in this folder or the first-round template."""
    if split not in ("dev", "held_out"):
        raise ValueError(split)
    if round3 and os.path.exists(os.path.join(round3, "manifest.csv")):
        check_isolation(folder, round3)
    with open(os.path.join(folder, "summary.json"), encoding="utf-8") as f:
        expected = json.load(f)["manifest_sha256"]
    with open(os.path.join(folder, "manifest.csv"), "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != expected:
            raise SystemExit("manifest.csv changed since selection: the split can't be trusted.")
    with open(os.path.join(folder, "manifest.csv"), newline="", encoding="utf-8") as f:
        split_of = {row["track_id"]: row["split"] for row in csv.DictReader(f)}
    out = defaultdict(list)
    if split == "dev" and os.path.exists(ORIGINAL_LABELS):
        for i, v in bench.load_labels(ORIGINAL_LABELS).items():
            out[i] += v
    for name in sorted(os.listdir(folder)):
        if SHEET.fullmatch(name) or (imports and IMPORT.fullmatch(name)):
            for i, v in bench.load_labels(os.path.join(folder, name)).items():
                if split_of.get(i) == split:
                    out[i] += v
    return dict(out)


def development_ratings(folder: str = DEFAULT_DIR,
                        round3: str | None = ROUND3_DIR) -> dict[str, list[tuple[float, str]]]:
    """Every rating that may be used to build a model: round 2's "dev" split plus
    its "held_out" split, which was used up by the 2026-10-27 evaluation and is now
    spent development data. Never includes round-3 ratings."""
    dev = load_ratings(folder, "dev", round3=round3)
    spent = load_ratings(folder, "held_out", round3=round3)
    assert not set(dev) & set(spent), "a song in both round-2 splits"
    return {**dev, **spent}


def _rated_ids(path: str) -> Counter:
    """How often each track_id has a non-blank rating cell in a CSV, valid or not."""
    with open(path, newline="", encoding="utf-8") as f:
        return Counter((row.get("track_id") or "").strip() for row in csv.DictReader(f)
                       if (row.get("rating") or "").strip())


def _manifest_rows(folder: str) -> list[dict]:
    with open(os.path.join(folder, "summary.json"), encoding="utf-8") as f:
        expected = json.load(f)["manifest_sha256"]
    with open(os.path.join(folder, "manifest.csv"), "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != expected:
            raise SystemExit(f"{folder}: manifest.csv changed since it was created.")
    with open(os.path.join(folder, "manifest.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_round3_rows(rows: list[dict]) -> None:
    ids = [r["track_id"] for r in rows]
    if len(ids) != len(set(ids)) or not all(ids):
        raise SystemExit("round-3 manifest: blank or duplicate track ids")
    by_id = {r["track_id"]: r for r in rows}
    for r in rows:
        if r["status"] not in ROUND3_STATUS:
            raise SystemExit(f"round-3 manifest: {r['track_id']} has status {r['status']!r}")
        if (r["status"] == "excluded_copy") != bool(r["copy_of"]):
            raise SystemExit(f"round-3 manifest: {r['track_id']}: copy_of must be set for, and only for, copies")
        if r["status"] == "excluded_copy":
            kept = by_id.get(r["copy_of"])
            if not kept or kept["status"] == "excluded_copy" or r["track_id"] not in kept["copies"].split(";"):
                raise SystemExit(f"round-3 manifest: {r['track_id']} isn't a known copy of an included recording")


def round3_manifest(folder: str = ROUND3_DIR) -> list[dict]:
    """The round-3 manifest, after checking its hash and its statuses and copies."""
    rows = _manifest_rows(folder)
    _check_round3_rows(rows)
    return rows


def round3_protected(folder: str = ROUND3_DIR) -> set[str]:
    """Every round-3 track id, plus every known copy of one."""
    out = set()
    for r in round3_manifest(folder):
        out |= {r["track_id"]} | {c for c in r["copies"].split(";") if c}
    return out


def _round2_rated(folder: str) -> dict[str, list[str]]:
    """track_id -> the round-2 rating files (the ones load_ratings reads) rating it."""
    out = defaultdict(list)
    files = [ORIGINAL_LABELS] if os.path.exists(ORIGINAL_LABELS) else []
    files += [os.path.join(folder, n) for n in sorted(os.listdir(folder))
              if SHEET.fullmatch(n) or IMPORT.fullmatch(n)]
    for path in files:
        for i in _rated_ids(path):
            out[i].append(os.path.basename(path))
    return out


def check_isolation(folder: str = DEFAULT_DIR, round3: str = ROUND3_DIR) -> None:
    """Refuses if a round-3 song or a known copy of one has a rating anywhere round 2
    reads (template, batch sheets, imports), or if a round-3 recording is a known
    copy of a round-2 rated song."""
    rated = _round2_rated(folder)
    clash = sorted(i for i in round3_protected(round3) if i in rated)
    if clash:
        raise SystemExit(f"round-3 test songs rated in round-2 files: "
                         + ", ".join(f"{i} ({'/'.join(rated[i])})" for i in clash))


def load_round3(folder: str = ROUND3_DIR, round2: str = DEFAULT_DIR) -> dict[str, tuple[int, str]]:
    """Round-3 test ratings: {track_id: (rating, file)}, one rating per recording.
    Read only from round3_sheet.csv and round3_import_*.csv in the round-3 folder.
    Refuses, rather than skips: any other rating file in the folder, ids not in the
    manifest, ratings of excluded copies, a recording rated twice (in one file or
    across files), ratings that aren't whole numbers 1-5, and songs rated in round 2."""
    rows = {r["track_id"]: r for r in round3_manifest(folder)}
    check_isolation(round2, folder)
    out, problems = {}, []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not name.lower().endswith(".csv") or name == "manifest.csv":
            continue
        if not (ROUND3_SHEET.fullmatch(name) or ROUND3_IMPORT.fullmatch(name)):
            with open(path, newline="", encoding="utf-8") as f:
                if "rating" in (csv.DictReader(f).fieldnames or []):
                    problems.append(f"{name}: rating file with an unexpected name")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                i, v = (row.get("track_id") or "").strip(), (row.get("rating") or "").strip()
                if not v:
                    continue
                if i not in rows:
                    problems.append(f"{name}: {i} isn't in the round-3 manifest")
                elif rows[i]["status"] == "excluded_copy":
                    problems.append(f"{name}: {i} is an excluded copy")
                elif v not in ("1", "2", "3", "4", "5"):
                    problems.append(f"{name}: {i} has an invalid rating")
                elif i in out:
                    problems.append(f"{i} rated twice ({out[i][1]}, {name})")
                else:
                    out[i] = (int(v), name)
    if problems:
        raise SystemExit("round-3 ratings refused:\n  " + "\n  ".join(problems))
    return out


def write_round3(rows: list[dict], folder: str, info: dict) -> dict:
    """Writes the round-3 manifest, the blind sheet (every recording except excluded
    copies, in a fixed shuffled order: id, artist, title only), summary.json with
    both hashes, and a README. Refuses a non-empty folder."""
    if os.path.exists(folder) and os.listdir(folder):
        raise SystemExit(f"{folder} already exists and isn't empty; left alone.")
    _check_round3_rows(rows)
    os.makedirs(folder, exist_ok=True)
    manifest = os.path.join(folder, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ROUND3_COLUMNS)
        w.writeheader()
        w.writerows({k: r[k] for k in ROUND3_COLUMNS} for r in rows)
    sheet_rows = sorted((r for r in rows if r["status"] != "excluded_copy"),
                        key=lambda r: hashlib.sha256(f"round3|{r['track_id']}".encode()).hexdigest())
    sheet = os.path.join(folder, "round3_sheet.csv")
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "artist", "title", "rating", "notes"])
        for r in sheet_rows:
            w.writerow([r["track_id"], r["artist"], r["title"], "", ""])
    def digest(path):
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()
    summary = {**info, "manifest_sha256": digest(manifest), "sheet_sha256_when_blank": digest(sheet),
               "status": dict(Counter(r["status"] for r in rows)), "sheet_rows": len(sheet_rows)}
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(folder, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"""Round-3 energy ratings: an independent test set ({len(sheet_rows)} songs)

These songs share no artist and no recording with anything you've rated before. They
are only used once, to test one model that is frozen before you start rating.

How to rate
- Open round3_sheet.csv. Listen to each song and type a whole number 1-5 in 'rating'.
  'notes' is optional. Leave a song blank if you can't judge it.
- Don't delete, reorder or edit rows, ids, artists or titles.
- Rate here only. Many of these songs are also listed on the old batch_01-05 sheets:
  leave them blank there (the tools refuse ratings for these songs anywhere else), and
  don't set stars in rekordbox for them either.
- Don't look up the app's or rekordbox's ratings, genres or BPMs first.
- Don't open manifest.csv or summary.json while rating.

{bench.RATING_GUIDE}

Returning it: save the file in place (keep it as CSV) and say it's done.
""")
    return summary


def status(folder: str) -> str:
    """Rated / total per batch and split; no rating values."""
    with open(os.path.join(folder, "manifest.csv"), newline="", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    rated = set()
    for name in os.listdir(folder):
        if SHEET.fullmatch(name) or IMPORT.fullmatch(name):
            rated |= set(bench.load_labels(os.path.join(folder, name)))
    lines = []
    for b in sorted({int(m["batch"]) for m in man}):
        rows = [m for m in man if int(m["batch"]) == b]
        lines.append(f"batch {b:2d}: {sum(m['track_id'] in rated for m in rows):3d} of {len(rows)} rated")
    for s in ("dev", "held_out"):
        rows = [m for m in man if m["split"] == s]
        lines.append(f"{s:9}: {sum(m['track_id'] in rated for m in rows):3d} of {len(rows)} rated")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("select", "status"):
        print(__doc__)
        return 2
    folder = argv[1] if len(argv) > 1 else DEFAULT_DIR
    if argv[0] == "status":
        print(status(folder))
        return 0
    rows = library_rows()
    summary = write_selection(select(rows, _rated_so_far()), folder)
    print(json.dumps(summary, indent=1))
    print(f"\nRating sheets: {folder}\\batch_NN.csv\n\n{bench.RATING_GUIDE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
