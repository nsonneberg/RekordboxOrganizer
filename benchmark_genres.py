"""Benchmark genre detection on your own songs (reads the library only).

Every method predicts the same recordings under the same rules:

  * leave-one-out (default): each labelled recording is hidden together with
    its copies and other versions of the same song (remixes, edits), and
    predicted by a model refitted without them;
  * --split test --final: the verified test split (benchmark_labels.py) is
    hidden all at once and predicted in one go: a genuinely held-out test.

Methods:
  old                 the previous release's classifier (genres.py at git HEAD)
  zero-shot           Discogs-EffNet's top family, no thresholds (always answers)
  legacy:*            the earlier weighted-average fusion (method="legacy")
  calibrated:*        the calibrated log-linear fusion (method="calibrated", default)
  ...:model / :library / :metadata / :hybrid    which evidence it may use

Truth:
  --truth tags (default)  your rekordbox tags. UNVERIFIED: treat the numbers as
                          a development estimate, not a final result.
  --truth verified        labels verified with benchmark_labels.py.

Release metadata: by default only what's already cached is used (nothing is
sent). --metadata looks the labelled songs up on MusicBrainz (artist, title
and length are sent); add --discogs to include Discogs (its answers are kept
6 hours at most and never trained on).

Usage:
    python benchmark_genres.py [library.xml] [--truth verified] [--split dev|test --final]
                               [--metadata [--discogs]] [--no-old]

Reports go to benchmark_results/ (git-ignored: they list your songs).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict

import numpy as np

import genre_model
import genres
import organizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "benchmark_results")
BOOTSTRAP = 2000
PREVIOUS = "calibrated:hybrid (earlier: label prior votes, no per-family gate)"
POINT_GATE = "calibrated:hybrid (earlier: per-family gate by point estimate)"
COVERAGE = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
FEW = 10                    # families with fewer labelled songs are too few to judge
# Confusable pairs: (family, subgenre). Subgenre None = the whole family,
# "" = the family without the other side's subgenre (plain House vs Afro House).
PAIRS = [("Deep House vs Tech House", ("House", "Deep House"), ("House", "Tech House")),
         ("Tech House vs Techno", ("House", "Tech House"), ("Techno", None)),
         ("Progressive House vs Melodic Techno", ("House", "Progressive House"),
          ("Techno", "Melodic Techno")),
         ("Dubstep vs Drum & Bass", ("Dubstep", None), ("Drum & Bass", None)),
         ("Trance vs Psytrance", ("Trance", ""), ("Trance", "Psytrance")),
         ("UK Garage vs House", ("House", "UK Garage"), ("House", "")),
         ("House vs Afro House", ("House", ""), ("House", "Afro House")),
         ("Hip Hop vs House", ("Hip Hop", None), ("House", None))]
CAL_BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001)
FEATURES: dict = {}


def rss_mb() -> float:
    """This process's peak memory (MB), or 0 if it can't be read."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t)] + [
                    (n, ctypes.c_size_t) for n in ("a", "b", "c", "d", "e", "f")]
            c = Counters()
            c.cb = ctypes.sizeof(c)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                                    wintypes.DWORD]
            k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return c.PeakWorkingSetSize / 1e6
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1e6 if sys.platform == "darwin" else 1e3)
    except Exception:
        return 0.0


def load(paths):
    if paths:
        return organizer.load_library(paths[0])[1]
    import rbdb
    db = rbdb.open_database()
    try:
        return rbdb.load_tracks(db)[0]
    finally:
        db.close()


def old_classifier():
    src = subprocess.run(["git", "show", "HEAD:genres.py"], capture_output=True, text=True,
                         cwd=HERE, encoding="utf-8").stdout
    if "def guess_genres" not in src:
        return None
    mod = types.ModuleType("old_genres")
    exec(compile(src, "old_genres.py", "exec"), mod.__dict__)
    return mod


# --------------------------------------------------------------------------- #
# One set of folds for every method
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Fold:
    groups: list            # [(truth tag, probe id)] predicted in this fold
    tracks: list            # the library with the hidden songs' genres blanked


def make_folds(tracks, truth, holdout, exclude, has_audio):
    group = genres._groups(tracks)
    related = genres.version_groups(tracks)
    always = {related[i] for i in exclude if i in related}
    members = defaultdict(list)
    skipped = []
    for tid in sorted(truth):
        if tid not in related:
            continue
        if not genres.known_family(genres.family_of(truth[tid])):
            skipped.append(truth[tid])
            continue
        members[group[tid]].append(tid)

    def hide(held):
        return [dataclasses.replace(t, genre="") if related[t.id] in held else t for t in tracks]

    def entry(ids):
        tag = Counter(truth[i] for i in ids).most_common(1)[0][0]
        return tag, min(ids, key=lambda i: (i not in has_audio, i))

    if holdout:
        held = {related[i] for ids in members.values() for i in ids} | always
        folds = [Fold([entry(ids) for _, ids in sorted(members.items())], hide(held))]
    else:
        folds = [Fold([entry(ids)], hide({related[i] for i in ids} | always))
                 for _, ids in sorted(members.items())]
    return folds, related, sorted(set(skipped))


@dataclasses.dataclass
class Row:
    tid: str
    name: str
    tag: str
    truth: str          # family
    truth_sub: str
    top: str            # best family ("" if none)
    confidence: float   # calibrated confidence / legacy score / 1.0 for always-answering
    auto: bool          # assigned automatically
    label: str          # assigned label ("" if not assigned)
    suggested_sub: str
    group: str          # related-versions group (for the bootstrap)
    sources: list = dataclasses.field(default_factory=list)   # [(source, its top family)]
    disagreement: bool = False
    suggested: bool = False   # a clear guess offered for approval (not assigned)


def make_row(tid, name, tag, fam, sub, group, p):
    if p is None:
        return Row(tid, name, tag, fam, sub, "", 0.0, False, "", "", group)
    if isinstance(p, str):                      # old classifier: a label or ""
        top = genres.family_of(p) if p else ""
        return Row(tid, name, tag, fam, sub, top, 1.0 if p else 0.0, bool(p), p,
                   genres.subgenre_of(p)[1] if p else "", group)
    top = p.candidates[0][0] if p.candidates else ""
    if set(p.sources) <= {"tempo", "tag"}:
        top = ""        # a tempo-only hint isn't an answer (BPM never decides a genre)
    return Row(tid, name, tag, fam, sub, top, float(p.score), p.confident,
               p.label if p.confident else "", p.suggested_subgenre or p.subgenre, group,
               [(e[0], e[1]) for e in p.evidence], p.disagreement,
               getattr(p, "suggested", False))


def run_method(name, folds, related, by_id, predict):
    t0 = time.time()
    rows = []
    for fold in folds:
        preds = predict(fold.tracks, {p for _, p in fold.groups})
        for tag, probe in fold.groups:
            fam, sub = genres.subgenre_of(tag)
            rows.append(make_row(probe, by_id[probe].name, tag, fam, sub, related[probe],
                                 preds.get(probe)))
    secs = time.time() - t0
    print(f"  {name}: {secs:.1f} s", flush=True)
    return rows, secs


def genres_predictor(results, meta, use, method):
    def predict(test_tracks, probes):
        return genres.predict_genres(test_tracks, FEATURES, None, results, {},
                                     targets=set(probes), metadata=meta, use=use, method=method)
    return predict


def previous_calibrated(results, meta):
    """The calibrated hybrid as it was before the prior-shift fix and the per-family gate."""
    from unittest import mock
    import fusion
    inner = genres_predictor(results, meta, None, "calibrated")

    def predict(test_tracks, probes):
        with mock.patch.object(fusion, "PREDICT_WITH_PRIOR", True), \
                mock.patch.object(fusion, "family_reliable", lambda *a, **k: _Every()):
            return inner(test_tracks, probes)
    return predict


class _Every(dict):
    """Every family passes (no per-family gate)."""
    def get(self, key, default=None):
        return True

    def __bool__(self):
        return True


def point_gate_calibrated(results, meta):
    """The calibrated hybrid with the per-family gate judged by the posterior-mean
    error (e + 1) / (n + 2) instead of the 95% upper bound (the version before it)."""
    from unittest import mock
    import fusion
    inner = genres_predictor(results, meta, None, "calibrated")

    def predict(test_tracks, probes):
        with mock.patch.object(fusion, "error_upper_bound",
                               lambda e, n, level=0.95: fusion.error_estimate(e, n)):
            return inner(test_tracks, probes)
    return predict


def exact_ci(k: int, n: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval for a proportion k / n."""
    import fusion
    if n == 0:
        return 0.0, 1.0
    return 1 - fusion.error_upper_bound(n - k, n, 0.975), fusion.error_upper_bound(k, n, 0.975)


def zero_shot(results):
    def predict(test_tracks, probes):
        out = {}
        for tid in probes:
            r = results.get(tid)
            m = genres._model_evidence(r) if r is not None else None
            if not m or not m["shares"]:
                continue
            fam = max(m["shares"], key=lambda f: (m["shares"][f], f))
            op = genres.Opinion("model", m["shares"], 1.0, m["subs"])
            out[tid] = genres.GenrePrediction(
                genres.CONFIDENT, fam, "", genres.parent_of(fam), 1.0, 0.0, [(fam, 1.0)],
                ("model",), suggested_subgenre=genres._choose_subgenre(fam, [op], {}))
        return out
    return predict


def old_predictor(old):
    def predict(test_tracks, probes):
        guesses = old.guess_genres(test_tracks, FEATURES, lambda m: None)
        return {tid: guesses.get(tid, "") for tid in probes}
    return predict


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def f1_table(rows):
    """Per true family: n, precision, recall, F1 of the best guess (forced choice)."""
    out = {}
    for c in sorted({r.truth for r in rows}):
        tp = sum(r.truth == c and r.top == c for r in rows)
        fp = sum(r.truth != c and r.top == c for r in rows)
        fn = sum(r.truth == c and r.top != c for r in rows)
        out[c] = {"n": tp + fn, "precision": tp / (tp + fp) if tp + fp else None,
                  "recall": tp / (tp + fn) if tp + fn else None,
                  "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    return out


def macro_f1(rows, union=False):
    table = f1_table(rows)
    scores = [v["f1"] for v in table.values()]
    if union:       # the earlier report's formula: predicted-only classes count as 0
        scores += [0.0] * len({r.top or "(none)" for r in rows} - set(table))
    return float(np.mean(scores)) if scores else 0.0


def accuracy(rows):
    return sum(r.top == r.truth for r in rows) / len(rows) if rows else 0.0


def summarise(rows):
    import fusion
    n = len(rows)
    auto = [r for r in rows if r.auto]
    wrong = sum(genres.family_of(r.label) != r.truth for r in auto)
    sug = [r for r in rows if r.suggested]
    ranked = sorted(rows, key=lambda r: -r.confidence)
    curve = {}
    for c in COVERAGE:
        k = max(1, int(round(c * n)))
        curve[c] = (sum(r.top != r.truth for r in ranked[:k]), k)
    return {"n": n, "accuracy": accuracy(rows), "macro_f1": macro_f1(rows),
            "macro_f1_union": macro_f1(rows, union=True),
            "coverage": len(auto) / n if n else 0.0, "auto": len(auto), "auto_wrong": wrong,
            "auto_error": wrong / len(auto) if auto else None,
            "auto_error_upper": fusion.error_upper_bound(wrong, len(auto)) if auto else None,
            "suggested": len(sug), "suggested_wrong": sum(r.top != r.truth for r in sug),
            "risk_coverage": curve}


def family_decisions(rows) -> list[str]:
    """Per true family: best-guess accuracy (exact CI) and what happened to the songs."""
    import fusion
    out = ["| Family | n | Best guess right (95% CI) | Auto-tagged (wrong) | "
           "Suggested (wrong) | Needs review | Note |", "|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[r.truth].append(r)
    for fam, rs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        k = sum(r.top == r.truth for r in rs)
        lo, hi = exact_ci(k, len(rs))
        auto = [r for r in rs if r.auto]
        sug = [r for r in rs if r.suggested]
        out.append(f"| {fam} | {len(rs)} | {k}/{len(rs)} ({lo:.0%}-{hi:.0%}) | "
                   f"{len(auto)} ({sum(genres.family_of(r.label) != fam for r in auto)}) | "
                   f"{len(sug)} ({sum(r.top != fam for r in sug)}) | "
                   f"{len(rs) - len(auto) - len(sug)} | "
                   f"{'too few to judge' if len(rs) < FEW else ''} |")
    out += ["", "Auto-tagged songs by the genre they were given (what an automatic tag "
            "of that genre is worth):", "",
            "| Given | Auto-tagged | Wrong | Error, 95% upper bound |", "|---|---|---|---|"]
    given = defaultdict(list)
    for r in rows:
        if r.auto:
            given[genres.family_of(r.label)].append(r)
    for fam, rs in sorted(given.items(), key=lambda kv: -len(kv[1])):
        w = sum(r.truth != fam for r in rs)
        out.append(f"| {fam} | {len(rs)} | {w} | {fusion.error_upper_bound(w, len(rs)):.0%} |")
    if not given:
        out.append("| (none) | 0 | - | - |")
    return out


def _grouped(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r.group].append(r.tid)
    return groups


def bootstrap(rows, stat, seed=0):
    by = {r.tid: r for r in rows}
    groups = _grouped(rows)
    keys = sorted(groups)
    rng = random.Random(seed)
    vals = [stat([by[t] for k in (rng.choice(keys) for _ in keys) for t in groups[k]])
            for _ in range(BOOTSTRAP)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff(a_rows, b_rows, stat, seed=1):
    """stat(a) - stat(b) on the same recordings, with a 95% bootstrap interval."""
    a = {r.tid: r for r in a_rows}
    b = {r.tid: r for r in b_rows}
    common = [r for r in a_rows if r.tid in b]
    groups = _grouped(common)
    keys = sorted(groups)
    rng = random.Random(seed)
    diffs = []
    for _ in range(BOOTSTRAP):
        ids = [t for k in (rng.choice(keys) for _ in keys) for t in groups[k]]
        diffs.append(stat([a[t] for t in ids]) - stat([b[t] for t in ids]))
    point = stat([a[r.tid] for r in common]) - stat([b[r.tid] for r in common])
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def calibration(rows):
    out = []
    for lo, hi in zip(CAL_BINS, CAL_BINS[1:]):
        sel = [r for r in rows if lo <= r.confidence < hi]
        if sel:
            out.append((lo, min(hi, 1.0), len(sel), float(np.mean([r.confidence for r in sel])),
                        accuracy(sel)))
    ece = sum(n * abs(c - a) for _, _, n, c, a in out) / max(len(rows), 1)
    return out, ece


def side(spec, fam, sub, other):
    f, s = spec
    if fam != f:
        return False
    if s is None:
        return True
    if s == "":
        return sub != other[1]
    return sub == s


def pair_report(rows):
    lines = []
    for title, a, b in PAIRS:
        na = [r for r in rows if side(a, r.truth, r.truth_sub, b)]
        nb = [r for r in rows if side(b, r.truth, r.truth_sub, a)]
        if not na or not nb:
            lines.append(f"| {title} | {len(na)} labelled | {len(nb)} labelled | not measurable "
                         "(needs labelled examples of both) | |")
            continue
        aa = sum(side(a, r.top, r.suggested_sub, b) for r in na)
        ab = sum(side(b, r.top, r.suggested_sub, a) for r in na)
        bb = sum(side(b, r.top, r.suggested_sub, a) for r in nb)
        ba = sum(side(a, r.top, r.suggested_sub, b) for r in nb)
        lines.append(f"| {title} | {aa} of {len(na)} | {bb} of {len(nb)} | {ab} / {ba} | "
                     f"{len(na) - aa - ab + len(nb) - bb - ba} |")
    return lines


def pct(v):
    return "-" if v is None else f"{v:.0%}"


def option(argv, name, default):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


# --------------------------------------------------------------------------- #

def main(argv):
    global FEATURES
    truth_kind = option(argv, "--truth", "tags")
    split = option(argv, "--split", "dev")
    values = {truth_kind, split}
    paths = [a for a in argv if not a.startswith("--") and a not in values]
    flags = {a for a in argv if a.startswith("--")}
    if truth_kind not in ("tags", "verified") or split not in ("dev", "test"):
        sys.exit("--truth is tags or verified; --split is dev or test.")
    if split == "test" and "--final" not in flags:
        sys.exit("The test split is for the final measurement only. Add --final if you mean "
                 "it (every run is logged in benchmark_labels.json).")
    if split == "test" and truth_kind != "verified":
        sys.exit("The test split needs --truth verified.")
    os.makedirs(OUT_DIR, exist_ok=True)
    tracks = load(paths)
    by_id = {t.id: t for t in tracks}
    log = print
    timings = {}
    t0 = time.time()
    FEATURES = organizer.analyse_tracks(tracks, log)
    timings["rekordbox analysis"] = time.time() - t0

    import benchmark_labels
    labels_db = benchmark_labels.load()
    test_ids = set(benchmark_labels.verified(labels_db, tracks, "test"))
    if truth_kind == "verified":
        truth = benchmark_labels.verified(labels_db, tracks, split)
        if not truth:
            print(benchmark_labels.stats(labels_db))
            sys.exit(f"No verified {split} labels yet: see benchmark_labels.py for the workflow.")
        if split == "test":
            benchmark_labels.log_test_run(labels_db, "benchmark_genres.py --split test --final")
            benchmark_labels.save(labels_db)
    else:
        truth = genres.training_labels(tracks, genres.load_labels())
        truth = {t: g for t, g in truth.items() if t not in test_ids}
    exclude = set() if split == "test" else test_ids   # never learn from the test split
    holdout = split == "test"

    results = {}
    for key in ("yamnet", "discogs-effnet"):
        before = time.time()
        results[key] = organizer.model_genres(tracks, log, backend=key)
        timings[f"{key} analysis (cached + new)"] = time.time() - before

    speed = {}
    sample = [t for t in tracks if t.path and os.path.exists(t.path)][:5]
    for key in ("yamnet", "discogs-effnet"):
        b = genre_model.backend_of(key)
        clf = genre_model.Classifier(genre_model.model_path(b), b)
        tmp = os.path.join(OUT_DIR, f"speed-{key}.json")
        if os.path.exists(tmp):
            os.remove(tmp)
        items = [genre_model.TrackInput(t.id, t.path, float(t.length)) for t in sample]
        before = time.time()
        genre_model.analyse_tracks(items, lambda m: None, classifier=clf, cache_file=tmp)
        speed[key] = (time.time() - before) / max(len(items), 1)
        os.remove(tmp)

    import metadata
    from version import VERSION
    providers = [metadata.MusicBrainz(version=VERSION)]
    if "--discogs" in flags:
        providers.append(metadata.Discogs(version=VERSION))
    online = "--metadata" in flags
    before = time.time()
    meta, meta_stats = metadata.lookup_tracks([by_id[i] for i in truth if i in by_id],
                                              providers, log, offline=not online)
    timings["metadata lookups" + ("" if online else " (cache only)")] = time.time() - before

    disc, yam = results["discogs-effnet"], results["yamnet"]
    folds, related, skipped = make_folds(tracks, truth, holdout, exclude, set(disc))
    plan = []
    old = None if "--no-old" in flags else old_classifier()
    if old is not None:
        plan.append(("old", old_predictor(old)))
    zs = "zero-shot (Discogs-EffNet top family)"
    plan += [(zs, zero_shot(disc)),
             ("legacy:model", genres_predictor(disc, {}, {"model"}, "legacy")),
             ("legacy:library", genres_predictor(disc, {}, {"library"}, "legacy")),
             ("legacy:hybrid", genres_predictor(disc, meta, None, "legacy")),
             ("calibrated:model", genres_predictor(disc, {}, {"model"}, "calibrated")),
             ("calibrated:library", genres_predictor(disc, {}, {"library"}, "calibrated")),
             ("calibrated:metadata", genres_predictor({}, meta, {"metadata"}, "calibrated")),
             ("calibrated:hybrid", genres_predictor(disc, meta, None, "calibrated")),
             (PREVIOUS, previous_calibrated(disc, meta)),
             (POINT_GATE, point_gate_calibrated(disc, meta)),
             ("calibrated:hybrid (YAMNet)", genres_predictor(yam, meta, None, "calibrated"))]
    methods, secs = {}, {}
    for name, predict in plan:
        methods[name], secs[name] = run_method(name, folds, related, by_id, predict)

    report = {k: summarise(v) for k, v in methods.items()}
    new, legacy = "calibrated:hybrid", "legacy:hybrid"
    n = report[new]["n"]
    fams = Counter(r.truth for r in methods[new])
    if truth_kind == "verified" and split == "test":
        what = "verified labels, held-out test split (never learned from or tuned on)"
    elif truth_kind == "verified":
        what = "verified labels, dev split (a development estimate)"
    else:
        what = ("UNVERIFIED rekordbox tags. A DEVELOPMENT ESTIMATE, NOT A TEST RESULT: the "
                "method's design was chosen while looking at results on these same songs")
    lines = ["# Genre benchmark", "",
             "PRIVATE: this file lists your songs. Don't commit or share it.", "",
             f"Truth: {what}. {n} recordings scored ({len(tracks)} tracks in the library); "
             f"{'one held-out test set' if holdout else 'leave-one-out'}; copies and other "
             "versions of a song are always held out together; every method predicts the same "
             "recordings."
             + (f" Skipped {len(skipped)} label(s) that aren't a genre: {', '.join(skipped)}."
                if skipped else ""), "",
             "Families: " + ", ".join(f"{f} {c}" for f, c in fams.most_common()), "",
             "## How this was measured", "",
             "- For each held-out song group, everything learned is refitted from the other "
             "songs only: the classifier trained on tags, the fusion weights (fitted on inner "
             "5-fold out-of-fold opinions), the automatic-tagging threshold and the per-family "
             "and subgenre gates (from inner out-of-fold predictions). The held-out songs' "
             "labels are never used.",
             "- Related recordings are held out together: copies, remixes and edits, mashups "
             "and their sources, and songs whose names share a part (which includes a shared "
             "artist; stricter than versions alone).",
             "- Label-free steps use the whole library (the embedding projection and the "
             "z-scores of rekordbox's analysis). No labels are involved, but it isn't strictly "
             "inductive.",
             "- Not independent: the taxonomy, the fusion design, its fixed constants and the "
             "subgenre and prior-shift changes were all chosen while looking at results on "
             "these songs. Expect a genuinely new test set to score lower. Only "
             "`--truth verified --split test --final` is a test result.",
             "- Intervals: 95% percentile bootstrap over song groups (2000 resamples); "
             "unreliable for families with fewer than about 5 songs.",
             "- Coverage vs error ranks songs by each method's own confidence. Methods without "
             "one (old, zero-shot) tie everywhere, so their curve order is arbitrary.",
             "- Upper bounds on error are one-sided 95% Clopper-Pearson (exact binomial); "
             "per-family accuracy intervals are two-sided 95% Clopper-Pearson. Neither accounts "
             "for the design having been developed on these songs.", "",
             "## Overall", "",
             "Accuracy and F1 use each method's best guess for every song. 'Auto-tagged' is what "
             "it would tag automatically; 'Suggested' is offered for approval; the rest goes "
             "to review.", "",
             "| Method | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-F1, old formula | "
             "Auto-tagged | Wrong among auto (95% upper bound) | Suggested (wrong) | Seconds |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in report.items():
        lo, hi = bootstrap(methods[name], accuracy)
        flo, fhi = bootstrap(methods[name], macro_f1)
        lines.append(f"| {name} | {m['accuracy']:.0%} ({lo:.0%}-{hi:.0%}) | "
                     f"{m['macro_f1']:.2f} ({flo:.2f}-{fhi:.2f}) | {m['macro_f1_union']:.2f} | "
                     f"{m['auto']}/{m['n']} ({m['coverage']:.0%}) | "
                     f"{m['auto_wrong']} ({pct(m['auto_error'])}; up to "
                     f"{pct(m['auto_error_upper'])}) | "
                     f"{m['suggested']} ({m['suggested_wrong']}) | {secs[name]:.0f} |")
    lines += ["", "Macro-F1 averages over the true families. The 'old formula' (used in the "
              "earlier report) also averaged in families that were only ever predicted, as zeros.",
              "", "## Paired differences", "",
              "Same recordings; 95% bootstrap interval, resampling whole song groups.", ""]
    for other in (POINT_GATE, PREVIOUS, legacy, "calibrated:model", zs, "old"):
        if other in methods:
            d, lo, hi = paired_diff(methods[new], methods[other], accuracy)
            fd, flo, fhi = paired_diff(methods[new], methods[other], macro_f1)
            lines.append(f"- {new} minus {other}: accuracy {d:+.0%} ({lo:+.0%} to {hi:+.0%}), "
                         f"macro-F1 {fd:+.2f} ({flo:+.2f} to {fhi:+.2f})")
    lines += ["", "## Automatic coverage versus error", "",
              "Wrong best guesses among each method's most confident X% of songs (wrong / "
              "songs), and at its automatic-tagging point. With 5 to 49 songs per cell, one "
              "song moves these by 2 to 20 points.", "",
              "| Method | " + " | ".join(f"{c:.0%}" for c in COVERAGE) + " | Auto point |",
              "|---|" + "---|" * (len(COVERAGE) + 1)]
    for name, m in report.items():
        c = m["risk_coverage"]
        lines.append(f"| {name} | " + " | ".join(f"{c[x][0]}/{c[x][1]}" for x in COVERAGE)
                     + f" | {m['coverage']:.0%} covered, {pct(m['auto_error'])} wrong |")
    bins, ece = calibration(methods[new])
    lines += ["", f"## Calibration ({new})", "",
              "| Confidence | Songs | Mean confidence | Actually right |", "|---|---|---|---|"]
    for lo, hi, k, c, a in bins:
        lines.append(f"| {lo:.2f}-{hi:.2f} | {k} | {c:.0%} | {a:.0%} |")
    lines += ["", f"Expected calibration error: {ece:.3f}. (Legacy scores are relative shares, "
              "not confidences, so they aren't meant to be calibrated.)", "",
              f"## Per family: decisions and uncertainty ({new})", ""] + \
        family_decisions(methods[new]) + [""]
    for name in (new, legacy, "old"):
        if name not in methods:
            continue
        table = f1_table(methods[name])
        lines += [f"## Per family: {name}", "", "| Family | n | Precision | Recall | F1 |",
                  "|---|---|---|---|---|"]
        for f, v in sorted(table.items(), key=lambda kv: -kv[1]["n"]):
            lines.append(f"| {f} | {v['n']} | {pct(v['precision'])} | {pct(v['recall'])} | "
                         f"{v['f1']:.2f} |")
        conf = Counter((r.truth, r.top or "(none)") for r in methods[name])
        cols = sorted({p for _, p in conf})
        lines += ["", "Confusion (rows: truth, columns: best guess):", "",
                  "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for t in sorted({t for t, _ in conf}):
            lines.append(f"| {t} | " + " | ".join(str(conf.get((t, c), "")) for c in cols) + " |")
        lines.append("")
    lines += [f"## Confusable pairs ({new})", "",
              "Subgenres are judged by the suggested subgenre (assigned or not).", "",
              "| Pair (A vs B) | A recognised | B recognised | A taken for B / B for A | "
              "Neither |", "|---|---|---|---|---|"] + pair_report(methods[new])
    lines += ["", "## Subgenres (recordings labelled with a subgenre)", ""]
    for name in (new, legacy, zs):
        rows = [r for r in methods[name] if r.truth_sub]
        sug = [r for r in rows if r.suggested_sub]
        ass = [r for r in rows if r.auto and r.label != genres.family_of(r.label)]
        lines.append(f"- {name}: suggested a subgenre for {len(sug)} of {len(rows)} "
                     f"({sum(r.suggested_sub == r.truth_sub for r in sug)} right); assigned one "
                     f"automatically for {len(ass)} ({sum(r.label == r.truth_sub for r in ass)} "
                     "right)")
    by_sub = Counter(r.truth_sub for r in methods[new] if r.truth_sub)
    lines.append("- labelled subgenres: " + ", ".join(f"{s} {c}" for s, c in by_sub.most_common()))
    present, seen, agree = Counter(), Counter(), Counter()
    for r in methods[new]:
        tops = dict(r.sources)
        present.update(tops.keys())
        names = sorted(tops)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                seen[(names[i], names[j])] += 1
                agree[(names[i], names[j])] += tops[names[i]] == tops[names[j]]
    lines += ["", f"## Sources ({new})", "",
              "- present for: " + ", ".join(f"{s} {c}/{n}" for s, c in present.most_common())]
    for (a, b), k in sorted(seen.items()):
        lines.append(f"- {a} and {b} agree on the family for {agree[(a, b)]} of {k}")
    lines.append(f"- sent to review because sources disagreed: "
                 f"{sum(r.disagreement for r in methods[new])}")
    matched = [r for r in methods["calibrated:metadata"] if r.top and
               any(s == "metadata" for s, _ in r.sources)]
    lines += ["", "## Release metadata", "",
              f"- {meta_stats.matched} of {meta_stats.asked} labelled tracks matched "
              f"({meta_stats.matched / max(meta_stats.asked, 1):.0%}); {meta_stats.not_found} "
              f"not found, {meta_stats.failed} errors, {meta_stats.not_looked_up} not looked up "
              f"({'online' if online else 'cache only; --metadata looks them up'}); by provider: "
              f"{meta_stats.by_provider}",
              f"- where matched, its best family was right for "
              f"{sum(r.top == r.truth for r in matched)} of {len(matched)}"]
    lines += ["", "## Few-shot families", ""]
    table = f1_table(methods[new])
    for f, c in sorted(fams.items(), key=lambda kv: kv[1]):
        if c < 10:
            lines.append(f"- {f}: {c} labelled, recall {pct(table[f]['recall'])} "
                         "(too few to judge reliably)")
    lines += ["", "## Runtime and memory", ""]
    lines += [f"- {k}: {v:.1f} s" for k, v in timings.items()]
    lines += [f"- {k}, uncached: {v:.2f} s per track (decode + up to 4 excerpts)"
              for k, v in speed.items()]
    lines.append(f"- peak memory of this process: {rss_mb():.0f} MB")
    lines.append(f"- model downloads: YAMNet {genre_model.MODEL['size'] / 1e6:.1f} MB, "
                 f"Discogs-EffNet {genre_model.DISCOGS_MODEL['size'] / 1e6:.1f} MB")
    lines += ["", f"## Per recording ({new})", "",
              "| Song | Tag | Best guess | Confidence | Auto | Suggested subgenre |",
              "|---|---|---|---|---|---|"]
    for r in methods[new]:
        lines.append(f"| {r.name[:50]} | {r.tag} | {r.top} | {r.confidence:.2f} | "
                     f"{'yes' if r.auto else 'suggested' if r.suggested else 'review'} | "
                     f"{r.suggested_sub} |")
    text = "\n".join(lines) + "\n"
    base = os.path.join(OUT_DIR, f"genre_benchmark-{truth_kind}-{split}")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(text)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"report": {k: {**v, "risk_coverage": {str(c): list(e) for c, e in
                                                         v["risk_coverage"].items()}}
                              for k, v in report.items()},
                   "timings": timings, "speed": speed,
                   "rows": {k: [dataclasses.asdict(r) for r in v] for k, v in methods.items()}},
                  f, indent=1)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
