"""Listen to tracks and score their energy on a 1-5 scale.

Each track is measured for loudness, rhythmic activity (flux), brightness and
high-frequency (hi-hat / percussion) content, either from rekordbox's own colour
waveform (direct mode, no decoding) or by decoding the audio file. Those
measurements, plus BPM, are combined into one score per track. Scores are then
ranked against the rest of the library: 1 = calmest 20%, 5 = most energetic 20%.

Measurements are cached on disk so tracks are only decoded once.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np

CACHE_VERSION = 3   # 2: added 'lows'; 3: energy measured in the drops (e_*)
# Energy features. "loudness", "flux", "brightness", "highs" and "lows" are
# whole-track measurements kept for genre guessing.
FEATURES = ("e_loud", "e_punch", "e_power", "e_bass", "e_highs")
WEIGHTS = {"e_loud": 0.24, "e_punch": 0.20, "e_power": 0.14, "e_bass": 0.10,
           "e_highs": 0.10, "tempo": 0.17, "mood": 0.05}
MAIN_SECTIONS = ("Drop", "Chorus")      # where a track's energy is judged
# rekordbox often measures these at half speed (87 instead of 174 BPM)
DOUBLE_TIME = ("drum and bass", "dnb", "jungle", "dubstep", "neurofunk", "liquid")

WAVEFORM_RATE = 150    # PWV5 waveform entries per second
MAX_SECONDS = 420      # analyse at most 7 minutes of each track
FRAME = 2048


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #

def _decode_soundfile(path: str):
    import soundfile as sf
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        data = f.read(frames=sr * MAX_SECONDS, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _decode_pyav(path: str):
    import av   # optional: may be missing or blocked
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=22050)
        chunks, total = [], 0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                chunks.append(arr)
                total += len(arr)
            if total >= 22050 * MAX_SECONDS:
                break
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32), 22050


def _decode_external(path: str):
    """Fall back to ffmpeg (if installed) or afconvert (built into macOS)."""
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", path, "-t", str(MAX_SECONDS),
                   "-ac", "1", "-ar", "22050", tmp]
        elif sys.platform == "darwin" and shutil.which("afconvert"):
            cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", "-c", "1", path, tmp]
        else:
            raise RuntimeError("no decoder available for this format")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(cmd, check=True, capture_output=True, creationflags=flags)
        return _decode_soundfile(tmp)
    finally:
        os.remove(tmp)


def decode(path: str):
    errors = []
    for fn in (_decode_soundfile, _decode_pyav, _decode_external):
        try:
            samples, sr = fn(path)
            if len(samples) > sr:   # at least a second of audio
                return samples, sr
            errors.append(f"{fn.__name__}: too short")
        except Exception as exc:
            errors.append(f"{fn.__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #

def measure(samples: np.ndarray, sr: int) -> dict:
    """Return raw energy features for one track."""
    n = len(samples) // FRAME
    frames = samples[: n * FRAME].reshape(n, FRAME).astype(np.float32)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    body = db >= np.median(db)          # ignore quiet intros / breakdowns

    spec = np.abs(np.fft.rfft(frames * np.hanning(FRAME).astype(np.float32), axis=1))
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    power = spec ** 2
    total = power.sum(axis=1) + 1e-12

    log_spec = np.log1p(spec * 100)
    flux = np.maximum(np.diff(log_spec, axis=0), 0).mean(axis=1)
    flux = np.concatenate([[0.0], flux])

    lows = power[:, freqs < 150].sum(axis=1) / total
    highs = power[:, freqs >= 5000].sum(axis=1) / total
    main = db >= np.percentile(db, 60)  # the loudest stretches stand in for the drops
    return {
        "loudness": float(db[body].mean()),
        "flux": float(flux[body].mean()),
        "brightness": float(((power * freqs).sum(axis=1) / total)[body].mean()),
        "highs": float(highs[body].mean()),
        "lows": float(lows[body].mean()),
        # energy (same meaning as in measure_waveform)
        "e_loud": float(np.percentile(db[main], 75)),
        "e_punch": float(flux[main].mean()),
        "e_power": float((db >= np.percentile(db, 95) - 3).mean()),
        "e_bass": float((lows * rms)[main].mean()),
        "e_highs": float(highs[main].mean()),
        "mood": 0,
    }


def analyse_file(path: str) -> dict:
    samples, sr = decode(path)
    return measure(samples, sr)


# --------------------------------------------------------------------------- #
# rekordbox waveforms (no audio decoding needed)
# --------------------------------------------------------------------------- #

def read_waveform(ext_path: str) -> np.ndarray:
    """Return the colour waveform (PWV5 tag) from a rekordbox ANLZ .EXT file.

    150 entries per second; each 16-bit entry packs red (lows), green (mids),
    blue (highs) as 3 bits each, plus a 5-bit height.
    """
    with open(ext_path, "rb") as f:
        data = f.read()
    if data[:4] != b"PMAI":
        raise ValueError("not a rekordbox analysis file")
    pos = int.from_bytes(data[4:8], "big")
    while pos + 12 <= len(data):
        kind = data[pos:pos + 4]
        len_header = int.from_bytes(data[pos + 4:pos + 8], "big")
        len_tag = int.from_bytes(data[pos + 8:pos + 12], "big")
        if kind == b"PWV5":
            n = int.from_bytes(data[pos + 16:pos + 20], "big")
            return np.frombuffer(data, ">u2", count=n, offset=pos + len_header).astype(np.int32)
        if len_tag <= 0:
            break
        pos += len_tag
    raise ValueError("no colour waveform in analysis file")


def measure_waveform(entries: np.ndarray, sections: Optional[list] = None, mood: int = 0) -> dict:
    """Features from a PWV5 colour waveform (150 entries a second).

    sections: [(start s, end s)] of the track's drops / choruses, from
    rekordbox's phrase analysis; without them the loudest stretches stand in.
    """
    height = ((entries >> 2) & 0x1F).astype(float)
    low = ((entries >> 13) & 7).astype(float)
    mid = ((entries >> 10) & 7).astype(float)
    high = ((entries >> 7) & 7).astype(float)
    sounding = height > 0
    if sounding.sum() < 150:
        raise ValueError("waveform too short")
    body = sounding & (height >= np.median(height[sounding]))   # skip intros / breakdowns
    total = low + mid + high + 1e-9
    flux = np.maximum(np.diff(height, prepend=height[0]), 0)

    main = np.zeros_like(sounding)
    for start, end in sections or []:
        main[int(start * WAVEFORM_RATE):int(end * WAVEFORM_RATE)] = True
    main &= sounding
    if main.sum() < WAVEFORM_RATE * 4:       # no usable phrase analysis
        main = sounding & (height >= np.percentile(height[sounding], 60))
    peak = np.percentile(height[sounding], 95)
    return {
        "loudness": float(height[body].mean()),
        "flux": float(flux[body].mean()),
        "brightness": float(((mid + 2 * high) / total)[body].mean()),
        "highs": float((high / total)[body].mean()),
        "lows": float((low / total)[body].mean()),
        # energy, judged in the drops
        "e_loud": float(np.percentile(height[main], 75)),
        "e_punch": float(flux[main].mean() / (height[main].mean() + 1e-9)),
        "e_power": float((height[sounding] >= 0.8 * peak).mean()),
        "e_bass": float((low * height)[main].mean()),
        "e_highs": float((high / total)[main].mean()),
        "mood": mood,
    }


def main_sections(phrases: list, n_entries: int) -> list[tuple[float, float]]:
    """[(start s, end s)] of the drops / choruses among rekordbox's phrases."""
    end_of_track = n_entries / WAVEFORM_RATE
    out = []
    for i, p in enumerate(phrases):
        if p.label in MAIN_SECTIONS:
            end = phrases[i + 1].start_ms / 1000 if i + 1 < len(phrases) else end_of_track
            out.append((p.start_ms / 1000, end))
    return out


def analyse_waveforms(paths: dict[str, str], log: Callable[[str], None] = print,
                      progress: Optional[Callable[[int, int], None]] = None) -> dict[str, dict]:
    """Measure tracks from rekordbox's own analysis. paths = {track id: .EXT path}."""
    log(f"Energy: reading rekordbox's analysis of {len(paths)} tracks ...")
    results, failures = {}, 0
    import cues
    for n, (tid, path) in enumerate(paths.items(), 1):
        try:
            entries = read_waveform(path)
            mood, phrases = cues.read_structure(path)
            results[tid] = measure_waveform(entries, main_sections(phrases, len(entries)), mood)
        except (OSError, ValueError):
            failures += 1
        if progress:
            progress(n, len(paths))
    if failures:
        log(f"  {failures} tracks have no usable analysis (analyse them in rekordbox first).")
    return results


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _percentiles(values: list) -> np.ndarray:
    """Each value's place among the others, 0 (lowest) to 1 (highest); ties share it.

    Unlike averages and spreads, one extreme track can't skew everyone else.
    Missing values (None) get the middle, 0.5.
    """
    known = [(v, i) for i, v in enumerate(values) if v is not None]
    out = np.full(len(values), 0.5)
    if len(known) < 2:
        return out
    known.sort()
    n, j = len(known), 0
    while j < n:
        k = j
        while k + 1 < n and known[k + 1][0] == known[j][0]:
            k += 1
        for _, i in known[j:k + 1]:
            out[i] = ((j + k) / 2) / (n - 1)
        j = k + 1
    return out


def effective_bpm(bpm: float, genre: str) -> float:
    """BPM as felt: half-speed readings of Drum & Bass etc. are doubled."""
    norm = " ".join(genre.casefold().replace("&", " and ").split())
    if 0 < bpm < 100 and any(word in norm for word in DOUBLE_TIME):
        return bpm * 2
    return bpm


def energy_scores(features: dict[str, dict], bpms: dict[str, float],
                  genres: Optional[dict[str, str]] = None) -> dict[str, float]:
    """One energy score per track (higher = more energetic), compared across all of them."""
    ids = list(features)
    genres = genres or {}
    score = np.zeros(len(ids))
    for name in FEATURES:
        score += WEIGHTS[name] * _percentiles([features[i].get(name) for i in ids])
    tempo = [effective_bpm(bpms.get(i, 0), genres.get(i, "")) for i in ids]
    score += WEIGHTS["tempo"] * _percentiles([b if b > 0 else None for b in tempo])
    # rekordbox's mood: 1 = high, 2 = mid, 3 = low
    score += WEIGHTS["mood"] * np.array([{1: 1.0, 2: 0.5, 3: 0.0}.get(
        int(features[i].get("mood") or 0), 0.5) for i in ids])
    return dict(zip(ids, score.tolist()))


def levels_from_scores(scores: dict[str, float]) -> dict[str, int]:
    """1-5 by rank: the most energetic fifth gets 5, the calmest fifth 1.

    The highest-scoring track is always 5 and the lowest 1. Tracks with the
    same score (e.g. two copies of a song) get the same level.
    """
    ids = list(scores)
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: 3}
    place = _percentiles([round(scores[i], 9) for i in ids])
    return {i: min(5, 1 + int(p * 5)) for i, p in zip(ids, place)}


def _zscores(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    return np.zeros_like(arr) if std == 0 else (arr - arr.mean()) / std


def levels_from_features(features: dict[str, dict], bpms: dict[str, float],
                         genres: Optional[dict[str, str]] = None) -> dict[str, int]:
    """Turn raw features into 1-5 levels, ranked across all given tracks."""
    return levels_from_scores(energy_scores(features, bpms, genres))


# --------------------------------------------------------------------------- #
# Cache + batch
# --------------------------------------------------------------------------- #

def cache_path() -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), "energy_cache.json")


def _load_cache(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["tracks"] if data.get("version") == CACHE_VERSION else {}
    except (OSError, ValueError, KeyError):
        return {}


def _save_cache(path: str, tracks: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "tracks": tracks}, f)
    os.replace(tmp, path)


def analyse_library(
    paths: dict[str, str],
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[int, int], None]] = None,
    cache_file: Optional[str] = None,
) -> dict[str, dict]:
    """Measure every track. paths = {track id: file path}. Returns {track id: features}."""
    cache_file = cache_file or cache_path()
    cache = _load_cache(cache_file)
    results: dict[str, dict] = {}
    todo = []
    for tid, path in paths.items():
        try:
            st = os.stat(path)
        except OSError:
            continue
        stamp = [st.st_size, int(st.st_mtime)]
        hit = cache.get(path)
        if hit and hit.get("stamp") == stamp:
            results[tid] = hit["features"]
        else:
            todo.append((tid, path, stamp))

    skipped = len(paths) - len(results) - len(todo)
    log(f"Energy: {len(results)} tracks already analysed, {len(todo)} to listen to"
        + (f", {skipped} files not found." if skipped else "."))

    lock = threading.Lock()
    done = 0
    failures = 0
    workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyse_file, p): (tid, p, stamp) for tid, p, stamp in todo}
        for fut in as_completed(futures):
            tid, path, stamp = futures[fut]
            try:
                feats = fut.result()
                results[tid] = feats
                with lock:
                    cache[path] = {"stamp": stamp, "features": feats}
            except Exception as exc:
                failures += 1
                if failures <= 10:
                    log(f"  Couldn't read {os.path.basename(path)}: {str(exc)[:120]}")
            done += 1
            if progress:
                progress(done, len(todo))
            if done % 50 == 0:
                log(f"  Listened to {done}/{len(todo)} ...")
                _save_cache(cache_file, cache)

    if failures > 10:
        log(f"  ... and {failures - 10} more files couldn't be read.")
    if todo:
        _save_cache(cache_file, cache)
    return results
