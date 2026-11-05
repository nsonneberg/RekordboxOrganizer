"""Benchmark the energy rating (energy.py).

    python benchmark_energy.py synthetic            synthetic signals, designed expectations
    python benchmark_energy.py measure              measure your library in both modes (read-only)
    python benchmark_energy.py library              label-free checks on those measurements
    python benchmark_energy.py template [N]         a CSV of songs for listeners to rate
    python benchmark_energy.py evaluate LABELS.csv  compare with listeners' ratings

Results go to benchmark_results/ (private, not committed): they hold your track
ids and, for the template, titles.

The synthetic expectations are the author's design, not listeners' judgements:
they check that the algorithm does what it was built to do (e.g. it isn't fooled
by mastering level or BPM), not that it agrees with people. Agreement with
people can only be measured with `evaluate`, on ratings made by listening
without seeing the app's ratings. The old (v3) algorithm is kept here, frozen,
as the baseline those ratings are compared against.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

import energy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "benchmark_results")
FEATURES_FILE = os.path.join(OUT_DIR, "energy_features.json")
SR = 22050


# --------------------------------------------------------------------------- #
# Synthetic signals
# --------------------------------------------------------------------------- #

def _kick(level: float) -> np.ndarray:
    n = int(0.15 * SR)
    t = np.arange(n) / SR
    freq = 45 + 75 * np.exp(-t / 0.03)                 # 120 -> 45 Hz sweep
    return level * np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.05)


def _snare(rng, level: float) -> np.ndarray:
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n) * np.exp(-t / 0.04)
    return level * (0.7 * noise + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.03))


def _hat(rng, level: float) -> np.ndarray:
    n = int(0.05 * SR)
    noise = np.diff(rng.standard_normal(n + 1))          # roughly high-passed
    return level * 0.5 * noise * np.exp(-np.arange(n) / (0.012 * SR))


def _pad(seconds: float, freqs, level: float, swell: float = 8.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * t / swell)
    return level * lfo * sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate(freqs)) / len(freqs)


def _saw(seconds: float, freq: float, level: float, drive: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    saw = 2 * ((t * freq) % 1) - 1
    return level * np.tanh(drive * saw) / np.tanh(drive)


def _place(out: np.ndarray, sound: np.ndarray, every: float, offset: float = 0.0) -> None:
    for start in np.arange(offset, len(out) / SR, every):
        i = int(start * SR)
        j = min(len(out), i + len(sound))
        out[i:j] += sound[:j - i]


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    return x / (np.abs(x).max() + 1e-12) * peak


def peak_time(seconds=60, bpm=140, seed=1):
    """Kick every beat, snare on 2 and 4, 16th hats, distorted saw bass."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    out = _saw(seconds, 55, 0.35, 4.0) + _pad(seconds, (220, 277, 330), 0.1)
    _place(out, _kick(1.0), beat)
    _place(out, _snare(rng, 0.6), 2 * beat, beat)
    _place(out, _hat(rng, 0.35), beat / 4)
    return _norm(out)


def house_groove(seconds=60, bpm=124, seed=2):
    """Kick every beat, off-beat hats, a round bass and a pad."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    out = _pad(seconds, (55,), 0.25) + _pad(seconds, (262, 330, 392), 0.15)
    _place(out, _kick(0.8), beat)
    _place(out, _hat(rng, 0.25), beat, beat / 2)
    return _norm(out, 0.7)


def fast_ambient(seconds=60, seed=3):
    """Slow swelling pads with soft, irregular grains: tagged 170 BPM, no beat."""
    rng = np.random.default_rng(seed)
    out = _pad(seconds, (196, 247, 294, 392), 0.4, swell=13) + _pad(seconds, (98,), 0.2, 21)
    for start in np.sort(rng.uniform(0, seconds - 1, 40)):
        i = int(start * SR)
        grain = _pad(0.4, (rng.uniform(800, 2000),), 0.04, swell=0.8) * np.hanning(int(0.4 * SR))
        out[i:i + len(grain)] += grain
    return _norm(out, 0.5)


def slow_forceful(seconds=60, bpm=85, seed=4):
    """Half-time: huge kicks and snares, a distorted bass chopped in 8ths, 16th hats."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    t = np.arange(int(seconds * SR)) / SR
    gate = ((t / (beat / 2)) % 1 < 0.6).astype(float)            # bass stabs on 8ths
    out = _saw(seconds, 41, 0.5, 6.0) * gate
    _place(out, _kick(1.0), 2 * beat)
    _place(out, _snare(rng, 1.0), 2 * beat, beat)
    _place(out, _hat(rng, 0.4), beat / 4)
    return _norm(np.tanh(1.2 * out))


def distorted_wall(seconds=60, bpm=85, seed=9):
    """Half-time drums buried in a constant, heavily clipped bass: forceful to a
    listener, but with hardly any rise and fall for the onset measures to find."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    out = _saw(seconds, 41, 0.6, 9.0)
    _place(out, _kick(1.0), 2 * beat)
    _place(out, _snare(rng, 1.0), 2 * beat, beat)
    _place(out, _hat(rng, 0.4), beat / 2)
    return _norm(np.tanh(1.5 * out))


def fast_restrained(seconds=60, bpm=128, seed=5):
    """A soft kick under a sustained pad, nothing else."""
    out = _pad(seconds, (220, 262, 330), 0.5, swell=16)
    _place(out, _kick(0.08), 60 / bpm)                         # ~8 dB under the pad
    return _norm(out, 0.6)


def calm_ballad(seconds=60, bpm=70, seed=6):
    """Piano-like notes, one a beat, ringing out."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR))
    beat = 60 / bpm
    for k, start in enumerate(np.arange(0, seconds - 2, beat)):
        f = [262, 294, 330, 392, 440][rng.integers(5)]
        t = np.arange(int(2 * SR)) / SR
        note = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3)) * np.exp(-t / 0.6)
        i = int(start * SR)
        out[i:i + len(note)] += 0.3 * note
    return _norm(out, 0.5)


def drone(seconds=60, seed=7):
    """A sustained tone with no rhythm at all."""
    return _norm(_pad(seconds, (110, 165), 0.5, swell=30), 0.4)


def breakdown_then_drop(seconds=60, seed=8):
    """A long quiet breakdown, then the peak-time section."""
    half = seconds // 2
    return np.concatenate([_norm(_pad(half, (220, 277, 330), 0.3), 0.25),
                           peak_time(seconds - half, seed=seed)])


@dataclass
class Synthetic:
    name: str
    signal: np.ndarray
    bpm: float
    genre: str
    expect: tuple          # (lowest, highest) acceptable level, or "review"
    what: str
    known_weakness: bool = False   # reported, but doesn't fail the run


def synthetic_tracks() -> list[Synthetic]:
    loud = peak_time()
    return [
        Synthetic("peak_time", loud, 140, "Techno", (4, 5), "dense, hard-hitting, 140 BPM"),
        Synthetic("peak_time_quiet", loud * 0.03, 140, "Techno", (4, 5),
                  "the same track mastered 30 dB quieter"),
        Synthetic("house_groove", house_groove(), 124, "House", (3, 4), "steady groove"),
        Synthetic("fast_ambient", fast_ambient(), 170, "Ambient", (1, 2),
                  "tagged 170 BPM but no beat"),
        Synthetic("slow_forceful", slow_forceful(), 85, "Hip Hop", (4, 5),
                  "85 BPM half-time, distorted, hard-hitting"),
        Synthetic("distorted_wall", distorted_wall(), 85, "Dubstep", (4, 5),
                  "85 BPM, drums buried in constant clipped bass", known_weakness=True),
        Synthetic("fast_restrained", fast_restrained(), 128, "House", (1, 3),
                  "128 BPM, soft kick under a pad"),
        Synthetic("calm_ballad", calm_ballad(), 70, "Pop", (1, 2), "soft piano-like notes"),
        Synthetic("drone", drone(), 0, "", (1, 1), "no rhythm, no BPM"),
        Synthetic("breakdown_then_drop", breakdown_then_drop(), 140, "Techno", (4, 5),
                  "half breakdown, half peak-time: judged on the drop"),
        Synthetic("too_short", peak_time(15), 140, "Techno", "review", "15 s clip"),
        Synthetic("silence", np.zeros(SR * 60), 0, "", "review", "silence"),
    ]


# Designed orderings: (higher, lower)
SYNTHETIC_ORDER = [("slow_forceful", "fast_restrained"), ("peak_time", "house_groove"),
                   ("house_groove", "fast_ambient"), ("house_groove", "calm_ballad"),
                   ("peak_time_quiet", "house_groove"), ("calm_ballad", "drone")]


def to_waveform(signal: np.ndarray) -> np.ndarray:
    """An approximate PWV5 colour waveform of a synthetic signal, for testing
    waveform mode. Height: RMS per 1/150 s on a log scale scaled to the track's
    peak (rekordbox's exact scaling isn't documented); colours: share of energy
    below 200 Hz / 200 Hz-2.5 kHz / above, 0-7. An approximation, not rekordbox."""
    hop = SR // energy.WAVEFORM_RATE
    n = len(signal) // hop
    frames = signal[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms / rms.max())
    height = np.clip(np.round(31 * (1 + db / 48)), 0, 31).astype(int)
    spec = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1 / SR)
    parts = [spec[:, freqs < 200].sum(1), spec[:, (freqs >= 200) & (freqs < 2500)].sum(1),
             spec[:, freqs >= 2500].sum(1)]
    total = sum(parts) + 1e-12
    low, mid, high = (np.clip(np.round(7 * p / total), 0, 7).astype(int) for p in parts)
    return (low << 13) | (mid << 10) | (high << 7) | (height << 2)


def run_synthetic(verbose: bool = True) -> dict:
    """Score the synthetic tracks in audio mode (and waveform mode where possible)."""
    rows = []
    for s in synthetic_tracks():
        feats = energy.measure(s.signal.astype(np.float32), SR)
        r = energy.score_track(feats, s.bpm, s.genre)
        wave = None
        if s.signal.any():
            try:
                wf = energy.measure_waveform(to_waveform(s.signal))
                wave = energy.score_track(wf, s.bpm, s.genre)
            except ValueError:
                wave = None
        ok = (r.review if s.expect == "review"
              else (not r.review and s.expect[0] <= r.level <= s.expect[1]))
        rows.append({"name": s.name, "what": s.what, "expect": s.expect, "ok": ok,
                     "known_weakness": s.known_weakness,
                     "score": r.score, "level": r.level, "review": r.review,
                     "warnings": r.warnings,
                     "waveform_score": wave.score if wave else None,
                     "waveform_level": wave.level if wave else None,
                     "components": {k: v[1] for k, v in r.components.items()}})
    by = {r["name"]: r for r in rows}
    order = [(a, b, (by[a]["score"] or 0) > (by[b]["score"] or 0)) for a, b in SYNTHETIC_ORDER]
    wave_order = [(a, b, (by[a]["waveform_score"] or 0) > (by[b]["waveform_score"] or 0))
                  for a, b in SYNTHETIC_ORDER]
    result = {"rows": rows, "order": order, "waveform_order": wave_order}
    if verbose:
        print("Synthetic signals (audio mode; waveform mode from an approximate waveform)")
        print(f"{'track':22} {'expect':>8} {'score':>6} {'level':>5} {'wave':>6}  ok")
        for r in rows:
            exp = r["expect"] if r["expect"] == "review" else f"{r['expect'][0]}-{r['expect'][1]}"
            sc = "-" if r["score"] is None else f"{r['score']:.0f}"
            ws = "-" if r["waveform_score"] is None else f"{r['waveform_score']:.0f}"
            print(f"{r['name']:22} {exp:>8} {sc:>6} {r['level']:>5} {ws:>6}  "
                  f"{'yes' if r['ok'] else 'NO'}{' (known weakness)' if r['known_weakness'] else ''}")
        print("Designed orderings (audio / waveform):")
        for (a, b, ok), (_, _, wok) in zip(order, wave_order):
            print(f"  {a} > {b}: {'yes' if ok else 'NO'} / {'yes' if wok else 'NO'}")
    return result


# --------------------------------------------------------------------------- #
# The old (v3) algorithm, frozen as a baseline
# --------------------------------------------------------------------------- #

V3_WEIGHTS = {"e_loud": 0.24, "e_punch": 0.20, "e_power": 0.14, "e_bass": 0.10,
              "e_highs": 0.10, "tempo": 0.17, "mood": 0.05}


def v3_audio(samples: np.ndarray, sr: int) -> dict:
    n = len(samples) // 2048
    frames = samples[: n * 2048].reshape(n, 2048).astype(np.float32)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    spec = np.abs(np.fft.rfft(frames * np.hanning(2048).astype(np.float32), axis=1))
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    power = spec ** 2
    total = power.sum(axis=1) + 1e-12
    flux = np.concatenate([[0.0], np.maximum(np.diff(np.log1p(spec * 100), axis=0), 0).mean(1)])
    lows = power[:, freqs < 150].sum(axis=1) / total
    highs = power[:, freqs >= 5000].sum(axis=1) / total
    main = db >= np.percentile(db, 60)
    return {"e_loud": float(np.percentile(db[main], 75)), "e_punch": float(flux[main].mean()),
            "e_power": float((db >= np.percentile(db, 95) - 3).mean()),
            "e_bass": float((lows * rms)[main].mean()), "e_highs": float(highs[main].mean()),
            "mood": 0}


def v3_waveform(entries: np.ndarray, sections=None, mood: int = 0) -> dict:
    height = ((entries >> 2) & 0x1F).astype(float)
    low = ((entries >> 13) & 7).astype(float)
    mid = ((entries >> 10) & 7).astype(float)
    high = ((entries >> 7) & 7).astype(float)
    sounding = height > 0
    total = low + mid + high + 1e-9
    flux = np.maximum(np.diff(height, prepend=height[0]), 0)
    main = np.zeros_like(sounding)
    for start, end in sections or []:
        main[int(start * 150):int(end * 150)] = True
    main &= sounding
    if main.sum() < 600:
        main = sounding & (height >= np.percentile(height[sounding], 60))
    peak = np.percentile(height[sounding], 95)
    return {"e_loud": float(np.percentile(height[main], 75)),
            "e_punch": float(flux[main].mean() / (height[main].mean() + 1e-9)),
            "e_power": float((height[sounding] >= 0.8 * peak).mean()),
            "e_bass": float((low * height)[main].mean()),
            "e_highs": float((high / total)[main].mean()), "mood": mood}


def _v3_percentiles(values: list) -> np.ndarray:
    known = [(v, i) for i, v in enumerate(values) if v is not None]
    out = np.full(len(values), 0.5)
    if len(known) >= 2:
        out[[i for _, i in known]] = energy._percentiles([v for v, _ in known])
    return out


def v3_scores(features: dict[str, dict], bpms: dict, genres: dict) -> dict[str, float]:
    """The v3 score: each measurement's rank in the library, weighted."""
    ids = list(features)
    score = np.zeros(len(ids))
    for name in ("e_loud", "e_punch", "e_power", "e_bass", "e_highs"):
        score += V3_WEIGHTS[name] * _v3_percentiles([features[i].get(name) for i in ids])
    tempo = [energy.effective_bpm(bpms.get(i, 0), genres.get(i, "")) for i in ids]
    score += V3_WEIGHTS["tempo"] * _v3_percentiles([b if b > 0 else None for b in tempo])
    score += V3_WEIGHTS["mood"] * np.array([{1: 1.0, 2: 0.5, 3: 0.0}.get(
        int(features[i].get("mood") or 0), 0.5) for i in ids])
    return dict(zip(ids, score.tolist()))


# --------------------------------------------------------------------------- #
# Statistics (no scipy)
# --------------------------------------------------------------------------- #

def ranks(values) -> np.ndarray:
    """Average ranks (ties share), 1-based."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> Optional[float]:
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau_b(a, b) -> Optional[float]:
    n = len(a)
    if n < 3:
        return None
    conc = disc = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    denom = np.sqrt((conc + disc + ties_a) * (conc + disc + ties_b))
    return float((conc - disc) / denom) if denom else None


def weighted_kappa(a, b, k: int = 5) -> Optional[float]:
    """Quadratic-weighted Cohen's kappa between two 1..k ratings."""
    if len(a) < 2:
        return None
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[int(x) - 1, int(y) - 1] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else None


def confusion(truth, pred, k: int = 5) -> list[list[int]]:
    m = [[0] * k for _ in range(k)]
    for t, p in zip(truth, pred):
        m[int(t) - 1][int(p) - 1] += 1
    return m


def bootstrap_ci(fn, a, b, n: int = 1000, seed: int = 0) -> Optional[tuple]:
    if len(a) < 5:
        return None
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        v = fn(a[i], b[i])
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else None


def agreement(truth, pred) -> dict:
    """Rank and ordinal agreement between listener levels and predicted levels/scores."""
    truth = list(truth)
    pred = list(pred)
    t_lv = [int(round(x)) for x in truth]
    exact = float(np.mean([a == b for a, b in zip(t_lv, pred)])) if truth else None
    within = float(np.mean([abs(a - b) <= 1 for a, b in zip(t_lv, pred)])) if truth else None
    return {"n": len(truth), "spearman": spearman(truth, pred),
            "spearman_ci": bootstrap_ci(spearman, truth, pred),
            "kendall_tau_b": kendall_tau_b(truth, pred), "exact": exact, "within_1": within,
            "weighted_kappa": weighted_kappa(t_lv, pred), "confusion": confusion(t_lv, pred)}


# --------------------------------------------------------------------------- #
# Measuring a library (read-only)
# --------------------------------------------------------------------------- #

def _load_library(paths):
    import organizer
    if paths:
        return organizer.load_library(paths[0])[1]
    import rbdb
    db = rbdb.open_database()
    try:
        return rbdb.load_tracks(db)[0]
    finally:
        db.close()


# How the app can measure a track: decoded audio judged in rekordbox's drop /
# chorus phrases (direct mode), decoded audio on its own (XML mode), and
# rekordbox's colour waveform (direct mode, for files that can't be decoded).
MODES = ("audio_phrases", "audio", "waveform")


def _measure_audio(path: str, sections, mood) -> dict:
    samples, sr = energy.decode(path)
    out = {"audio": energy.measure(samples, sr), "audio_v3": v3_audio(samples, sr)}
    if sections is not None:
        out["audio_phrases"] = energy.measure(samples, sr, sections, mood)
    return out


def measure_library(paths=()) -> dict:
    """Every mode (MODES) and the v3 baseline for every track. Nothing is written
    to the library; results go to benchmark_results/energy_features.json."""
    from concurrent.futures import ThreadPoolExecutor
    import genres
    tracks = _load_library(paths)
    rows, hints = {}, {}
    for t in tracks:
        row = {"family": genres.family_of(t.genre) or "", "genre": t.genre, "bpm": t.bpm,
               "length": t.length}
        if t.waveform:
            try:
                entries, sections, mood = energy.read_analysis(t.waveform)
                hints[t.id] = (sections, mood)
                row["waveform"] = energy.measure_waveform(entries, sections, mood)
                row["waveform_v3"] = v3_waveform(entries, sections, mood)
            except (OSError, ValueError) as exc:
                row["waveform_error"] = str(exc)
        rows[t.id] = row
    with ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1))) as pool:
        futures = {t.id: pool.submit(_measure_audio, t.path, *hints.get(t.id, (None, 0)))
                   for t in tracks if t.path and os.path.exists(t.path)}
        for n, (tid, fut) in enumerate(futures.items(), 1):
            try:
                rows[tid].update(fut.result())
            except Exception as exc:
                rows[tid]["audio_error"] = str(exc)[:200]
            if n % 25 == 0:
                print(f"  decoded {n}/{len(futures)}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(FEATURES_FILE, "w", encoding="utf-8") as f:
        json.dump({"cache_version": energy.CACHE_VERSION, "tracks": rows}, f)
    return rows


def load_features() -> dict:
    with open(FEATURES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("cache_version") != energy.CACHE_VERSION:
        raise SystemExit("energy_features.json is from another version: run `measure` again.")
    return data["tracks"]


# --------------------------------------------------------------------------- #
# Label-free checks on a real library
# --------------------------------------------------------------------------- #

def _rate(rows: dict, mode: str, relative: bool = False, ids=None) -> dict:
    ids = [i for i in (ids or rows) if rows[i].get(mode)]
    return energy.rate_tracks({i: rows[i][mode] for i in ids},
                              {i: rows[i]["bpm"] for i in ids},
                              {i: rows[i]["genre"] for i in ids}, relative)


# Values a signal with none of each quality measures (steady noise, a pure tone,
# a drone; see `anchors` in the report): the 0 points of energy.CALIBRATION.
NO_QUALITY = {"x_onsets": 0.0, "x_punch": 0.5, "x_impact": 1.2, "x_pulse": 0.15,
              "x_density": 0.0}


def derive_anchors(rows: dict, mode: str = "audio") -> dict:
    """energy.CALIBRATION's rule, re-derived from a measured library.

    Audio: 0 = NO_QUALITY, 1 = the library's 90th percentile (sustain, which has
    no natural zero: its 10th to 90th percentile). Waveform: matched to the
    audio scale on the same tracks, so each measurement's 10th and 90th
    percentiles land where the audio ones do.
    """
    def pct(m, k, q):
        v = [r[m][k] for r in rows.values() if r.get(m) and r[m].get(k) is not None]
        return float(np.percentile(v, q)) if v else None
    audio = {}
    for k in ("x_onsets", "x_punch", "x_impact", "x_pulse", "x_density", "x_sustain"):
        lo = NO_QUALITY.get(k, pct(mode, k, 10))
        audio[k] = (round(lo, 3), round(pct(mode, k, 90), 3))
    wave = {}
    for k in ("x_punch", "x_impact", "x_pulse", "x_density"):
        a_lo, a_hi = audio[k]
        target = (pct(mode, k, 10) - a_lo) / (a_hi - a_lo)        # audio's P10, scaled
        w10, w90 = pct("waveform", k, 10), pct("waveform", k, 90)
        if w10 is None:
            continue
        span = (w90 - w10) / (1 - target)
        wave[k] = (round(w90 - span, 3), round(w90, 3))
    return {"audio": audio, "waveform": wave}


def library_report(rows: dict, seed: int = 0) -> dict:
    out: dict = {"tracks": len(rows), "derived_anchors": derive_anchors(rows),
                 "anchors_in_use": energy.CALIBRATION}
    rated = {m: _rate(rows, m) for m in MODES}
    for mode, res in rated.items():
        levels = Counter(r.level for r in res.values())
        reasons = Counter(w for r in res.values() if r.review for w in r.warnings[-1:])
        warn = Counter(w.split(" (")[0].split(";")[0] for r in res.values() for w in r.warnings)
        scores = [r.score for r in res.values() if r.score is not None]
        out[mode] = {"measured": len(res), "levels": {str(k): levels.get(k, 0) for k in range(6)},
                     "review": sum(r.review for r in res.values()),
                     "review_reasons": dict(reasons), "warnings": dict(warn.most_common(8)),
                     "score_pcts": [round(float(np.percentile(scores, q)), 1)
                                    for q in (5, 25, 50, 75, 95)] if scores else None,
                     "sections": dict(Counter(rows[i][mode].get("x_sections") for i in res))}
        fam = defaultdict(list)
        for i, r in res.items():
            if r.score is not None:
                fam[rows[i]["family"] or "(no genre)"].append(r.score)
        out[mode]["by_family"] = {f: {"n": len(v), "median": round(float(np.median(v)), 1),
                                      "iqr": [round(float(np.percentile(v, 25)), 1),
                                              round(float(np.percentile(v, 75)), 1)]}
                                  for f, v in sorted(fam.items(), key=lambda kv: -len(kv[1]))
                                  if len(v) >= 3}
        keys = tuple(k for k in ("x_onsets", "x_punch", "x_impact", "x_pulse", "x_density",
                                 "x_sustain") if any(rows[i][mode].get(k) is not None for i in res))
        ids = [i for i in res if all(rows[i][mode].get(k) is not None for k in keys)]
        out[mode]["feature_correlation"] = {
            f"{a}~{b}": round(spearman([rows[i][mode][a] for i in ids],
                                       [rows[i][mode][b] for i in ids]) or 0, 2)
            for n, a in enumerate(keys) for b in keys[n + 1:]}
        out[mode]["feature_pcts"] = {k: [round(float(np.percentile(
            [rows[i][mode][k] for i in ids], q)), 3) for q in (5, 50, 90, 95)] for k in keys}

    out["mode_agreement"] = {}
    for m1, m2 in (("audio_phrases", "audio"), ("audio_phrases", "waveform"),
                   ("audio", "waveform")):
        both = [i for i in rows if i in rated[m1] and i in rated[m2]
                and rated[m1][i].score is not None and rated[m2][i].score is not None]
        x = [rated[m1][i].score for i in both]
        y = [rated[m2][i].score for i in both]
        l1 = [rated[m1][i].level for i in both]
        l2 = [rated[m2][i].level for i in both]
        out["mode_agreement"][f"{m1} vs {m2}"] = {
            "n": len(both), "spearman": spearman(x, y), "spearman_ci": bootstrap_ci(spearman, x, y),
            "exact_level": float(np.mean([a == b for a, b in zip(l1, l2)])) if both else None,
            "within_1": float(np.mean([abs(a - b) <= 1 for a, b in zip(l1, l2)])) if both else None,
            "mean_abs_score_diff": float(np.mean(np.abs(np.subtract(x, y)))) if both else None,
            "confusion": confusion(l1, l2)}

    # Stability: rate a random half of the library on its own and compare with the
    # whole library. Absolute levels can't move; relative ones can.
    rng = random.Random(seed)
    out["stability"] = {}
    for mode in MODES:
        ids = [i for i in rated[mode] if rated[mode][i].score is not None]
        half = rng.sample(ids, len(ids) // 2) if len(ids) >= 4 else ids
        alone_abs = _rate(rows, mode, ids=half)
        alone_rel = _rate(rows, mode, True, ids=half)
        full_rel = _rate(rows, mode, True)
        unrelated = dict({i: rows[i][mode] for i in half})
        synth = {f"synthetic:{s.name}": energy.measure(s.signal.astype(np.float32), SR)
                 for s in synthetic_tracks()[:8]} if mode.startswith("audio") else {}
        with_synth = energy.rate_tracks(unrelated | synth,
                                        {i: rows[i]["bpm"] for i in half}, {}, relative=True)
        out["stability"][mode] = {
            "n": len(half),
            "absolute_changed": float(np.mean([alone_abs[i].level != rated[mode][i].level
                                               for i in half])) if half else None,
            "relative_changed_half_vs_full": float(np.mean([alone_rel[i].level != full_rel[i].level
                                                            for i in half])) if half else None,
            "relative_changed_after_adding_synthetic": (float(np.mean(
                [with_synth[i].level != alone_rel[i].level for i in half])) if synth else None)}

    # The v3 baseline on the same tracks, for orientation (not a target).
    out["vs_v3"] = {}
    for mode in MODES:
        base = "waveform_v3" if mode == "waveform" else "audio_v3"
        ids = [i for i in rated[mode] if rated[mode][i].score is not None and rows[i].get(base)]
        v3 = v3_scores({i: rows[i][base] for i in ids}, {i: rows[i]["bpm"] for i in ids},
                       {i: rows[i]["genre"] for i in ids})
        out["vs_v3"][mode] = {"n": len(ids), "spearman": spearman(
            [v3[i] for i in ids], [rated[mode][i].score for i in ids])}
        # Loudness: does the score just follow mastering level?
        lo = [rows[i][mode].get("x_lufs") for i in ids]
        if mode.startswith("audio") and all(v is not None for v in lo) and ids:
            out["vs_v3"][f"{mode}_score_vs_lufs"] = spearman(lo, [rated[mode][i].score for i in ids])
            out["vs_v3"][f"{mode}_v3_vs_lufs"] = spearman(lo, [v3[i] for i in ids])
    return out


# --------------------------------------------------------------------------- #
# Human ratings
# --------------------------------------------------------------------------- #

def write_template(n: int = 60, paths=(), seed: int = 0) -> str:
    """A CSV of up to n songs (spread over genres) for listeners to rate 1-5.

    It deliberately shows no ratings from the app, so listeners aren't anchored.
    A template that already holds ratings is never overwritten.
    """
    import genres
    path = os.path.join(OUT_DIR, "energy_labels_template.csv")
    if os.path.exists(path) and load_labels(path):
        raise SystemExit(f"{path} already holds ratings, so it's left alone. "
                         "Move it elsewhere first, or use energy_labelling.py for more songs.")
    tracks = [t for t in _load_library(paths) if t.path]
    rng = random.Random(seed)
    by_fam = defaultdict(list)
    for t in tracks:
        by_fam[genres.family_of(t.genre) or ""].append(t)
    for v in by_fam.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < min(n, len(tracks)):
        for fam in sorted(by_fam, key=lambda f: len(by_fam[f])):
            if by_fam[fam] and len(chosen) < n:
                chosen.append(by_fam[fam].pop())
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "artist", "title", "genre", "rating", "rater", "notes"])
        for t in chosen:
            w.writerow([t.id, t.artist, t.name, t.genre, "", "", ""])
    return path


RATING_GUIDE = """How to rate (1-5), by listening to the whole track's main sections:
1 Chill: little drive, soft or sparse     2 Warm-up: relaxed but moving
3 Groove: steady, medium drive            4 Driving: strong, insistent
5 Peak time: as intense as music gets in your sets
Judge intensity, not how much you like it, how loud the file is, or its genre."""


def load_labels(path: str) -> dict[str, list[tuple[float, str]]]:
    labels: dict[str, list] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rating = float(row["rating"])
            except (KeyError, ValueError, TypeError):
                continue
            if 1 <= rating <= 5:
                labels[row["track_id"].strip()].append((rating, (row.get("rater") or "").strip()))
    return dict(labels)


def evaluate(labels_path: str, rows: dict) -> dict:
    labels = load_labels(labels_path)
    truth = {i: float(np.mean([r for r, _ in v])) for i, v in labels.items() if i in rows}
    out: dict = {"labelled": len(truth), "raters": sorted({r for v in labels.values()
                                                          for _, r in v if r})}
    # Listener agreement with each other: the ceiling for any algorithm.
    pairs = defaultdict(dict)
    for i, v in labels.items():
        for rating, rater in v:
            pairs[rater or "?"][i] = rating
    raters = sorted(pairs)
    out["inter_rater"] = {}
    for a_i, a in enumerate(raters):
        for b in raters[a_i + 1:]:
            common = sorted(set(pairs[a]) & set(pairs[b]))
            if len(common) >= 3:
                out["inter_rater"][f"{a} vs {b}"] = agreement(
                    [pairs[a][i] for i in common], [int(pairs[b][i]) for i in common])
    for mode in MODES:
        ids = [i for i in truth if rows[i].get(mode)]
        if not ids:
            continue
        res = _rate(rows, mode)
        rel = _rate(rows, mode, True)
        rated = [i for i in ids if res[i].score is not None]
        block = {"labelled_measured": len(ids), "flagged_for_review": len(ids) - len(rated),
                 "absolute": agreement([truth[i] for i in rated], [res[i].level for i in rated]),
                 "absolute_score_spearman": spearman([truth[i] for i in rated],
                                                     [res[i].score for i in rated]),
                 "relative": agreement([truth[i] for i in rated], [rel[i].level for i in rated])}
        base = "waveform_v3" if mode == "waveform" else "audio_v3"
        v3_ids = [i for i in ids if rows[i].get(base)]
        if v3_ids:
            allv3 = [i for i in rows if rows[i].get(base)]
            v3 = v3_scores({i: rows[i][base] for i in allv3},
                           {i: rows[i]["bpm"] for i in allv3}, {i: rows[i]["genre"] for i in allv3})
            v3_levels = energy.relative_levels(v3)
            block["v3_baseline"] = agreement([truth[i] for i in v3_ids],
                                             [v3_levels[i] for i in v3_ids])
        fam = defaultdict(list)
        for i in rated:
            fam[rows[i]["family"] or "(no genre)"].append(i)
        block["by_family"] = {f: agreement([truth[i] for i in v], [res[i].level for i in v])
                              for f, v in fam.items() if len(v) >= 5}
        # Stability: the labelled songs rated on their own vs. in the whole library.
        alone = _rate(rows, mode, ids=rated)
        alone_rel = _rate(rows, mode, True, ids=rated)
        block["stability"] = {
            "absolute_changed": float(np.mean([alone[i].level != res[i].level for i in rated]))
            if rated else None,
            "relative_changed": float(np.mean([alone_rel[i].level != rel[i].level for i in rated]))
            if rated else None}
        out[mode] = block
    return out


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("synthetic", "measure", "library", "template", "evaluate"):
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "synthetic":
        res = run_synthetic()
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "energy_synthetic.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
        return 0 if all(r["ok"] or r["known_weakness"] for r in res["rows"]) and all(o for *_, o in res["order"]) else 1
    if cmd == "measure":
        rows = measure_library(rest)
        print(f"Measured {len(rows)} tracks -> {FEATURES_FILE}")
        return 0
    if cmd == "template":
        n = int(rest[0]) if rest and rest[0].isdigit() else 60
        path = write_template(n, [r for r in rest if not r.isdigit()])
        print(f"Wrote {path}. Fill in 'rating' (and 'rater') by listening.\n\n{RATING_GUIDE}")
        return 0
    rows = load_features()
    if cmd == "library":
        rep = library_report(rows)
        with open(os.path.join(OUT_DIR, "energy_library.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=str)
        print(json.dumps(rep, indent=1, default=str))
        return 0
    if not rest:
        print("usage: python benchmark_energy.py evaluate LABELS.csv")
        return 2
    rep = evaluate(rest[0], rows)
    with open(os.path.join(OUT_DIR, "energy_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
