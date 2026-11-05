"""Listen to tracks and rate their energy 1-5.

Two algorithms:

- classic (the default): the committed v3 algorithm, restored unchanged. A few
  measurements of the main sections (loudness, punch, share at full level, bass,
  highs), tempo and rekordbox's mood, each ranked against the rest of the
  library and weighted (CLASSIC_WEIGHTS); levels are library fifths, so
  1 = your calmest 20%, 5 = your most energetic 20%. Measured from rekordbox's
  colour waveform in direct mode and from the decoded audio in XML mode. See
  classic_scores. Against one listener's ratings (72 tracks) it ranked tracks
  far better than the experimental algorithm below.
- experimental (opt-in): everything below. Against the same ratings its
  ranking was no better than chance and it put almost every track at 3-4.

How the experimental algorithm works (details in each function):

1. Measure. Each track is measured either from rekordbox's own colour waveform
   (direct mode, no decoding) or from the decoded audio (XML mode). Energy is
   judged in the track's representative sections: rekordbox's drops/choruses
   when its phrase analysis is there and plausible, otherwise its most intense
   stretches (which needn't be "drops"; a track without one is judged as it is).
2. Describe. The measurements are things a listener hears as intensity and are
   all independent of how loud the file was mastered: how busy the rhythm is
   (onsets a second, how sharply the sound rises), how hard the low end hits,
   how steady and strong the beat is, how dense/noisy the sound is, and how much
   of the track runs at full intensity. Tempo counts a little, and only as far
   as there's a steady beat to carry it (fast ambient isn't energetic).
3. Score. Each measurement is mapped onto 0-1 with fixed calibration anchors
   (CALIBRATION, per analysis mode) and combined into a 0-100 base score. It's
   an absolute scale: a track's score doesn't depend on the rest of the library.
   Levels come from fixed thresholds (LEVEL_THRESHOLDS); the old ranking
   against the library ("relative") is still available as an option.
4. Explain. Every result lists the measurements, what each contributed, and
   any data-quality warnings. Tracks without enough to go on are flagged for
   review and not rated, rather than given a made-up level. There is no
   confidence number: nothing here has been calibrated against listeners.

The weights and anchors are documented priors, checked on synthetic signals and
on one real (unlabelled) library; they have NOT been fitted to human ratings.
benchmark_energy.py measures them against human ratings once some exist.

Whole-track measurements ("loudness", "flux", "brightness", "highs", "lows")
are also used by genre guessing and are kept exactly as before. Measurements
of decoded audio are cached on disk (CACHE_VERSION), so tracks are only decoded
once; a cache from an older version is never reused.
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
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# 2: added 'lows'; 3: energy measured in the drops (e_*); 4: new energy
# measurements (x_*), absolute scoring.
CACHE_VERSION = 4
MAIN_SECTIONS = ("Drop", "Chorus")      # rekordbox phrases that are a track's peak
# The classic (v3) algorithm: its measurements and weights, as committed.
CLASSIC_FEATURES = ("e_loud", "e_punch", "e_power", "e_bass", "e_highs")
CLASSIC_WEIGHTS = {"e_loud": 0.24, "e_punch": 0.20, "e_power": 0.14, "e_bass": 0.10,
                   "e_highs": 0.10, "tempo": 0.17, "mood": 0.05}
# rekordbox often measures these at half speed (87 instead of 174 BPM)
DOUBLE_TIME = ("drum and bass", "dnb", "jungle", "dubstep", "neurofunk", "liquid")

WAVEFORM_RATE = 150    # PWV5 waveform entries per second
MAX_SECONDS = 420      # analyse at most 7 minutes of each track
FRAME = 2048           # whole-track (genre) measurements
N_FFT = 2048           # energy measurements: ~93 ms windows ...
HOP = 512              # ... every ~23 ms at the analysis rate (22.05 / 24 kHz)
MIN_SECONDS = 30       # less sounding audio than this can't be rated
SECTION_SECONDS = 4    # representative sections are picked in blocks this long
FLUX_FLOOR = 2e-4      # rises are measured above this x the track's average bin power
MIN_BAND_BINS = 4      # narrowest flux band, in FFT bins (narrower ones flicker on noise)
AUDIO_ONSET_DB = 2.0   # smallest average spectral rise (dB) that counts as an onset:
                       # real hits rise 4-25 dB, steady noise stays under ~1.5 dB
WAVEFORM_ONSET = 0.05  # the same for waveform height (share of the average height)
LOW_FLOOR = 0.01       # low-end rises are measured above this x the track's average
                       # total power (-20 dB): a kick only counts as impact once it's
                       # a sizeable part of the sound, not a tiny bump in an empty band


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
    if os.path.splitext(path)[1].lower() in NEEDS_FFMPEG and not shutil.which("ffmpeg"):
        raise RuntimeError("this format (AAC / ALAC / M4A) needs FFmpeg installed to be "
                           "analysed; the song was skipped and left unchanged")
    raise RuntimeError("; ".join(errors))


# Formats soundfile (libsndfile) can't read; on Windows they need FFmpeg on the PATH.
NEEDS_FFMPEG = {".m4a", ".mp4", ".aac", ".alac", ".m4b"}


# --------------------------------------------------------------------------- #
# Shared helpers: sections, onsets, beat
# --------------------------------------------------------------------------- #

def _moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples (shorter at the ends)."""
    n = max(1, int(n))
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    i = np.arange(len(x))
    lo = np.clip(i - n // 2, 0, len(x))
    hi = np.clip(i + n // 2 + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def hit_strength(env: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """How big the biggest rises are compared with a typical frame: the 97th
    percentile of the onset curve minus its median, within mask. Steady sound
    (even loud, even noisy) scores low; hard, separated hits score high."""
    if not mask.any():
        return None
    v = env[mask]
    return float(np.percentile(v, 97) - np.median(v))


def representative_sections(level: np.ndarray, rate: float, sounding: np.ndarray,
                            preferred: Optional[np.ndarray] = None
                            ) -> tuple[np.ndarray, str, list[str]]:
    """(mask of the frames to judge energy on, where it came from, warnings).

    level: a per-frame intensity on any scale that only needs to be comparable
    within the track (dB, waveform height). preferred: rekordbox's drop/chorus
    phrases. They're used only if they cover 8+ seconds and are louder than at
    least half of the track (strictly: a mostly-flat track whose 'drop' is no
    louder than its breakdown doesn't qualify); otherwise phrase analysis is
    treated as unreliable.
    Without them the track's most intense 40% (in 4-second blocks) is used:
    whatever its loudest parts are, drop or not. Very short tracks use it all.
    """
    warnings: list[str] = []
    if preferred is not None and preferred.any():
        p = preferred & sounding
        if p.sum() >= rate * 8 and np.mean(level[sounding] < np.median(level[p])) >= 0.5:
            return p, "phrases", warnings
        warnings.append("rekordbox's drop/chorus sections looked unreliable (too short or "
                        "quieter than the rest), so the most intense parts were used")
    win = int(rate * SECTION_SECONDS)
    nwin = len(level) // win if win else 0
    if nwin < 2:
        return sounding.copy(), "whole", warnings
    s = sounding[:nwin * win].reshape(nwin, win)
    w = level[:nwin * win].reshape(nwin, win)
    usable = s.mean(axis=1) >= 0.5
    if usable.sum() < 2:
        return sounding.copy(), "whole", warnings
    means = np.where(usable, (w * s).sum(axis=1) / np.maximum(s.sum(axis=1), 1), -np.inf)
    keep = np.argsort(-means, kind="stable")[:max(2, int(np.ceil(0.4 * usable.sum())))]
    mask = np.zeros(len(level), bool)
    for i in keep:
        mask[i * win:(i + 1) * win] = s[i]
    return mask, "intensity", warnings


def _rhythmless(env: np.ndarray, mask: np.ndarray, floor: float) -> bool:
    """True if (almost) nothing in mask rises by at least floor (a drone, noise, silence)."""
    return float(np.percentile(env[mask], 99)) < floor


def onset_rate(env: np.ndarray, rate: float, mask: np.ndarray,
               floor: float) -> Optional[float]:
    """Onsets (new sounds: hits, notes) per second within mask.

    env: an onset-strength curve (how sharply the sound rises each frame). A
    peak counts if it rises by at least floor (in env's units) and stands above
    the local (1 s) average by half the curve's spread, at least 50 ms after
    the previous one. None with < 4 s to look at.
    """
    if mask.sum() < rate * 4:
        return None
    if _rhythmless(env, mask, floor):
        return 0.0
    spread = float(np.std(env[mask]))
    thr = np.maximum(_moving_mean(env, rate) + 0.5 * spread, floor)
    prev = np.concatenate([[np.inf], env[:-1]])
    nxt = np.concatenate([env[1:], [-np.inf]])
    peaks = np.flatnonzero((env > thr) & (env >= prev) & (env > nxt) & mask)
    gap = max(1, int(round(0.05 * rate)))
    count, last = 0, -gap
    for i in peaks:
        if i - last >= gap:
            count += 1
            last = i
    return count / (mask.sum() / rate)


def beat_strength(env: np.ndarray, rate: float, mask: np.ndarray, floor: float,
                  fallback: Optional[np.ndarray] = None
                  ) -> tuple[Optional[float], Optional[float]]:
    """(how strong and steady the beat is, 0-1; the beat heard, BPM in 80-180).

    The onset curve's autocorrelation in 12-second stretches of the sections:
    its highest peak between 50 and 200 BPM, 0 = no repeating pulse, 1 =
    perfectly regular. The median over stretches. The tempo is only reported
    when the beat is clear (>= 0.15), folded into 80-180 BPM because half and
    double time can't be told apart this way. 0 if nothing rises by at least
    floor (no rhythm to speak of). If mask has no stretch of 6 s or more (the
    most intense parts can be scattered), fallback (the whole sounding track)
    is used instead; (None, None) if that has none either.
    """
    if mask.sum() >= rate * 6 and _rhythmless(env, mask, floor):
        return 0.0, None
    values, lags = _autocorrelation_peaks(env, rate, mask)
    if not values and fallback is not None:
        values, lags = _autocorrelation_peaks(env, rate, fallback)
    if not values:
        return None, None
    strength = float(np.median(values))
    tempo = None
    if strength >= 0.15 and lags:
        tempo = 60 * rate / float(np.median(lags))
        while tempo < 80:
            tempo *= 2
        while tempo >= 180:
            tempo /= 2
    return strength, tempo


def _autocorrelation_peaks(env: np.ndarray, rate: float, mask: np.ndarray) -> tuple[list, list]:
    """Per 12-second stretch of mask (6 s or longer): the highest normalised
    autocorrelation of env between 50 and 200 BPM, and its lag. Gaps in mask
    shorter than a second (the silences between stabs) don't break a stretch."""
    win = int(rate * 12)
    lo_lag, hi_lag = int(rate * 60 / 200), int(rate * 60 / 50)
    runs, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if runs and start - runs[-1][1] < rate:
                start = runs.pop()[0]
            runs.append((start, i))
            start = None
    values, lags = [], []
    for a, b in runs:
        for s in range(a, b, win):
            e = env[s:min(s + win, b)]
            if len(e) < rate * 6 or len(e) <= hi_lag + 1:
                continue
            e = e - e.mean()
            if not e.any():
                values.append(0.0)
                continue
            f = np.fft.rfft(e, 2 * len(e))
            ac = np.fft.irfft(f * np.conj(f))[:len(e)]
            ac = ac / ac[0] * len(e) / (len(e) - np.arange(len(e)))   # unbiased
            seg = ac[lo_lag:hi_lag + 1]
            j = int(np.argmax(seg))
            values.append(float(max(0.0, min(1.0, seg[j]))))
            lags.append(lo_lag + j)
    return values, lags


# --------------------------------------------------------------------------- #
# Decoded audio
# --------------------------------------------------------------------------- #

WHOLE_TRACK = ("loudness", "flux", "brightness", "highs", "lows")


def _whole_track(samples: np.ndarray, sr: int) -> dict:
    """Whole-track measurements used by genre guessing (unchanged since v3)."""
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
    flux = np.concatenate([[0.0], np.maximum(np.diff(log_spec, axis=0), 0).mean(axis=1)])
    return {
        "loudness": float(db[body].mean()),
        "flux": float(flux[body].mean()),
        "brightness": float(((power * freqs).sum(axis=1) / total)[body].mean()),
        "highs": float((power[:, freqs >= 5000].sum(axis=1) / total)[body].mean()),
        "lows": float((power[:, freqs < 150].sum(axis=1) / total)[body].mean()),
    }


def _analysis_rate(samples: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Halve the sample rate (averaging pairs) down to at most 32 kHz."""
    x = np.asarray(samples, dtype=np.float32)
    while sr > 32000 and len(x) >= 2:
        x = 0.5 * (x[0:len(x) // 2 * 2:2] + x[1:len(x) // 2 * 2:2])
        sr //= 2
    return x, sr


def _k_weighting(freqs: np.ndarray) -> np.ndarray:
    """Power gain of the ITU-R BS.1770 K-weighting filter at each frequency."""
    def gain(b, a):
        z = np.exp(-2j * np.pi * freqs / 48000)
        return np.abs(np.polyval(b[::-1], z) / np.polyval(a[::-1], z)) ** 2
    shelf = gain([1.53512485958697, -2.69169618940638, 1.19839281085285],
                 [1.0, -1.69065929318241, 0.73248077421585])
    highpass = gain([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])
    return shelf * highpass


def _log_bands(freqs: np.ndarray, lo: float, hi: float, n: int,
               min_bins: int = 1) -> np.ndarray:
    """(bins x bands) 0/1 matrix of about n log-spaced bands, each at least
    min_bins FFT bins wide (narrower ones are merged with the next)."""
    edges = np.geomspace(lo, hi, n + 1)
    cols = []
    start = 0
    for i in range(1, len(edges)):
        sel = (freqs >= edges[start]) & (freqs < edges[i])
        if sel.sum() >= min_bins:
            cols.append(sel.astype(float))
            start = i
    return np.stack(cols, axis=1)


def spectral_frames(x: np.ndarray, sr: int) -> dict:
    """Per-frame measurements (one every HOP samples), computed in chunks.

    kms: K-weighted mean square (for loudness). flux / low_flux: how much the
    spectrum rose since the previous frame, in dB, averaged over log-spaced
    bands of 30 Hz-11 kHz / 30-150 Hz (bands, not single FFT bins, so steady
    noise doesn't look like a stream of hits). In dB above a floor proportional
    to the track's own level, so they're the same for a track mastered quietly
    or loudly; the 30-150 Hz floor is LOW_FLOOR of the track's whole level. flat: spectral flatness over 200 Hz-8 kHz (0 = pure tones, ~0.56
    = white noise).
    """
    n = 1 + (len(x) - N_FFT) // HOP if len(x) >= N_FFT else 0
    out = {k: np.zeros(n) for k in ("kms", "flux", "low_flux", "flat")}
    out["rate"] = sr / HOP
    if n == 0:
        return out
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    kgain = _k_weighting(freqs) * np.where((freqs == 0) | (freqs == freqs[-1]), 1.0, 2.0)
    norm = 1.0 / (N_FFT * float(np.sum(win.astype(float) ** 2)))
    floor = FLUX_FLOOR * (float(np.mean(x.astype(float) ** 2)) + 1e-20) * float(np.sum(win.astype(float) ** 2))
    bands = _log_bands(freqs, 30, 11000, 32, MIN_BAND_BINS)
    low_bands = _log_bands(freqs, 30, 150, 3)
    total = floor / FLUX_FLOOR * N_FFT / 2           # average power of a whole frame
    floors = np.concatenate([floor * bands.sum(axis=0),
                             np.full(low_bands.shape[1], LOW_FLOOR * total / low_bands.shape[1])])
    flat_band = (freqs >= 200) & (freqs < 8000)
    prev = None
    for s in range(0, n, 512):
        m = min(512, n - s)
        idx = (s + np.arange(m))[:, None] * HOP + np.arange(N_FFT)
        spec = np.abs(np.fft.rfft(x[idx] * win, axis=1))
        power = spec.astype(float) ** 2
        out["kms"][s:s + m] = (power * kgain).sum(axis=1) * norm
        logm = 10 * np.log10(power @ np.hstack([bands, low_bands]) + floors)
        rise = np.maximum(np.diff(np.vstack([logm[:1] if prev is None else prev, logm]),
                                  axis=0), 0)
        out["flux"][s:s + m] = rise[:, :bands.shape[1]].mean(axis=1)
        out["low_flux"][s:s + m] = rise[:, bands.shape[1]:].mean(axis=1)
        # Floor 40 dB below each frame's own band level: a band holding only
        # leakage from bass reads as not flat, and the result doesn't depend on level.
        p = power[:, flat_band]
        p = p + 1e-4 * p.mean(axis=1, keepdims=True) + 1e-30
        out["flat"][s:s + m] = np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)
        prev = logm[-1:]
    return out


def loudness_blocks(kms: np.ndarray, rate: float, seconds: float = 0.4,
                    step: float = 0.1) -> np.ndarray:
    """Loudness (LUFS) of overlapping blocks, as in BS.1770 (0.4 s every 0.1 s)."""
    n, hop = max(1, int(round(rate * seconds))), max(1, int(round(rate * step)))
    if len(kms) < n:
        return np.array([-0.691 + 10 * np.log10(kms.mean() + 1e-20)]) if len(kms) else np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(kms)])
    starts = np.arange(0, len(kms) - n + 1, hop)
    return -0.691 + 10 * np.log10((c[starts + n] - c[starts]) / n + 1e-20)


def integrated_loudness(blocks: np.ndarray) -> Optional[float]:
    """BS.1770 gated loudness (-70 LUFS absolute, -10 LU relative gate)."""
    on = blocks[blocks > -70]
    if not len(on):
        return None
    rel = 10 * np.log10(np.mean(10 ** (on / 10))) - 10
    on = on[on > rel]
    return float(10 * np.log10(np.mean(10 ** (on / 10)))) if len(on) else None


def _classic_audio(samples: np.ndarray, sr: int) -> dict:
    """The classic algorithm's measurements of decoded audio (v3, unchanged): the
    loudest 40% of frames stand in for the drops."""
    n = len(samples) // FRAME
    frames = samples[: n * FRAME].reshape(n, FRAME).astype(np.float32)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    spec = np.abs(np.fft.rfft(frames * np.hanning(FRAME).astype(np.float32), axis=1))
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    power = spec ** 2
    total = power.sum(axis=1) + 1e-12
    flux = np.maximum(np.diff(np.log1p(spec * 100), axis=0), 0).mean(axis=1)
    flux = np.concatenate([[0.0], flux])
    lows = power[:, freqs < 150].sum(axis=1) / total
    highs = power[:, freqs >= 5000].sum(axis=1) / total
    main = db >= np.percentile(db, 60)
    return {
        "e_loud": float(np.percentile(db[main], 75)),
        "e_punch": float(flux[main].mean()),
        "e_power": float((db >= np.percentile(db, 95) - 3).mean()),
        "e_bass": float((lows * rms)[main].mean()),
        "e_highs": float(highs[main].mean()),
    }


def measure(samples: np.ndarray, sr: int, sections: Optional[list] = None,
            mood: int = 0) -> dict:
    """All measurements of one decoded track (see measure_audio_energy).

    sections / mood: rekordbox's drop/chorus phrases [(start s, end s)] and
    mood, when the track has been analysed in rekordbox (direct mode).
    """
    out = _whole_track(samples, sr)
    out.update(_classic_audio(samples, sr))
    out.update(measure_audio_energy(samples, sr, sections))
    out["mood"] = mood
    return out


def _section_mask(sections: Optional[list], n: int, rate: float) -> Optional[np.ndarray]:
    if not sections:
        return None
    mask = np.zeros(n, bool)
    for start, end in sections:
        mask[int(start * rate):int(end * rate)] = True
    return mask


def measure_audio_energy(samples: np.ndarray, sr: int,
                         sections: Optional[list] = None) -> dict:
    """Energy measurements (x_*) of decoded audio. Same meaning as in waveform mode.

    x_onsets   onsets a second in the representative sections
    x_punch    how big the biggest spectral rises are there, dB (hit_strength)
    x_impact   the same in 30-150 Hz only (kick / bass hits, not bass amount)
    x_pulse    beat strength 0-1 (beat_strength); x_tempo the beat heard (BPM)
    x_density  spectral flatness there (dense / distorted / noisy vs. pure)
    x_sustain  share of the track within 2 LU of its loud level (95th pct.)
    x_lufs, x_lra  integrated loudness and loudness range: reported, not scored
    """
    x, sr = _analysis_rate(samples, sr)
    st = spectral_frames(x, sr)
    rate = st["rate"]
    blocks = loudness_blocks(st["kms"], rate)
    lufs = integrated_loudness(blocks)
    frame_lufs = -0.691 + 10 * np.log10(st["kms"] + 1e-20)
    sounding = frame_lufs > max(-70.0, (lufs if lufs is not None else -70.0) - 20)
    sections, source, warnings = representative_sections(
        frame_lufs, rate, sounding, _section_mask(sections, len(frame_lufs), rate))
    strength, tempo = beat_strength(st["flux"], rate, sections, AUDIO_ONSET_DB, sounding)
    gated = blocks[blocks > ((lufs - 10) if lufs is not None else -70)]
    short = loudness_blocks(st["kms"], rate, 3.0, 1.0)
    short = short[short > ((lufs - 20) if lufs is not None else -70)]
    any_sections = bool(sections.any())
    return {
        "x_mode": "audio",
        "x_seconds": float(sounding.sum() / rate),
        "x_sections": source,
        "x_warnings": warnings,
        "x_onsets": onset_rate(st["flux"], rate, sections, AUDIO_ONSET_DB),
        "x_punch": hit_strength(st["flux"], sections),
        "x_impact": hit_strength(st["low_flux"], sections),
        "x_pulse": strength,
        "x_tempo": tempo,
        "x_density": float(st["flat"][sections].mean()) if any_sections else None,
        "x_sustain": (float((gated >= np.percentile(gated, 95) - 2).mean())
                      if len(gated) else None),
        "x_lufs": lufs,
        "x_lra": (float(np.percentile(short, 95) - np.percentile(short, 10))
                  if len(short) >= 3 else None),
    }


def analyse_file(path: str, sections: Optional[list] = None, mood: int = 0) -> dict:
    samples, sr = decode(path)
    return measure(samples, sr, sections, mood)


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
    """Measurements from a PWV5 colour waveform (150 entries a second).

    sections: [(start s, end s)] of the track's drops / choruses, from
    rekordbox's phrase analysis (see representative_sections). The waveform
    is coarse (5-bit height, 3-bit colours), so these are rougher than the
    decoded-audio ones and have their own calibration. Energy measurements
    have the same names and meanings as measure_audio_energy's; x_density is
    the share of mids and highs in the colours (the waveform has no spectrum),
    x_sustain the share of the track within 80% of its peak height.
    """
    height = ((entries >> 2) & 0x1F).astype(float)
    low = ((entries >> 13) & 7).astype(float)
    mid = ((entries >> 10) & 7).astype(float)
    high = ((entries >> 7) & 7).astype(float)
    sounding = height > 0
    if sounding.sum() < WAVEFORM_RATE:
        raise ValueError("waveform too short")
    body = sounding & (height >= np.median(height[sounding]))   # skip intros / breakdowns
    total = low + mid + high + 1e-9
    flux = np.maximum(np.diff(height, prepend=height[0]), 0)
    out = {
        "loudness": float(height[body].mean()),
        "flux": float(flux[body].mean()),
        "brightness": float(((mid + 2 * high) / total)[body].mean()),
        "highs": float((high / total)[body].mean()),
        "lows": float((low / total)[body].mean()),
        "mood": mood,
    }
    # The classic algorithm's measurements (v3, unchanged), judged in the drops.
    main = np.zeros_like(sounding)
    for start, end in sections or []:
        main[int(start * WAVEFORM_RATE):int(end * WAVEFORM_RATE)] = True
    main &= sounding
    if main.sum() < WAVEFORM_RATE * 4:       # no usable phrase analysis
        main = sounding & (height >= np.percentile(height[sounding], 60))
    top = np.percentile(height[sounding], 95)
    out.update({
        "e_loud": float(np.percentile(height[main], 75)),
        "e_punch": float(flux[main].mean() / (height[main].mean() + 1e-9)),
        "e_power": float((height[sounding] >= 0.8 * top).mean()),
        "e_bass": float((low * height)[main].mean()),
        "e_highs": float((high / total)[main].mean()),
    })

    preferred = _section_mask(sections, len(height), WAVEFORM_RATE)
    rate = WAVEFORM_RATE
    sec, source, warnings = representative_sections(height, rate, sounding, preferred)
    env = flux / (height[sounding].mean() + 1e-9)            # independent of overall height
    lows = low * height
    low_env = np.maximum(np.diff(lows, prepend=lows[0]), 0)
    strength, tempo = beat_strength(env, rate, sec, WAVEFORM_ONSET, sounding)
    peak = np.percentile(height[sounding], 95)
    out.update({
        "x_mode": "waveform",
        "x_seconds": float(sounding.sum() / rate),
        "x_sections": source,
        "x_warnings": warnings,
        "x_onsets": None,       # a 150 Hz, 5-bit envelope can't resolve single hits
        "x_punch": hit_strength(env, sec),
        "x_impact": hit_strength(low_env / (lows[sec].mean() + 1e-9), sec),
        "x_pulse": strength,
        "x_tempo": tempo,
        "x_density": float(((mid + high) / total)[sec].mean()),
        "x_sustain": float((height[sounding] >= 0.8 * peak).mean()),
        "x_lufs": None,
        "x_lra": None,
    })
    return out


def main_sections(phrases: list, n_entries: int) -> list[tuple[float, float]]:
    """[(start s, end s)] of the drops / choruses among rekordbox's phrases."""
    end_of_track = n_entries / WAVEFORM_RATE
    out = []
    for i, p in enumerate(phrases):
        if p.label in MAIN_SECTIONS:
            end = phrases[i + 1].start_ms / 1000 if i + 1 < len(phrases) else end_of_track
            out.append((p.start_ms / 1000, end))
    return out


def read_analysis(ext_path: str) -> tuple[np.ndarray, list, int]:
    """(colour waveform, drop/chorus sections, mood) from rekordbox's analysis files."""
    import cues
    entries = read_waveform(ext_path)
    mood, phrases = cues.read_structure(ext_path)
    return entries, main_sections(phrases, len(entries)), mood


def analyse_waveforms(paths: dict[str, str], log: Callable[[str], None] = print,
                      progress: Optional[Callable[[int, int], None]] = None) -> dict[str, dict]:
    """Measure tracks from rekordbox's own analysis only. paths = {track id: .EXT path}.

    Rougher than decoded audio (see measure_waveform); analyse_rekordbox prefers
    the audio and uses this only for files that can't be decoded.
    """
    log(f"Energy: reading rekordbox's analysis of {len(paths)} tracks ...")
    results, failures = {}, 0
    for n, (tid, path) in enumerate(paths.items(), 1):
        try:
            results[tid] = measure_waveform(*read_analysis(path))
        except (OSError, ValueError):
            failures += 1
        if progress:
            progress(n, len(paths))
    if failures:
        log(f"  {failures} tracks have no usable analysis (analyse them in rekordbox first).")
    return results


def analyse_rekordbox(audio: dict[str, str], analysis: dict[str, str],
                      log: Callable[[str], None] = print,
                      progress: Optional[Callable[[int, int], None]] = None,
                      cache_file: Optional[str] = None) -> dict[str, dict]:
    """Direct mode: measure the decoded audio, judged in the sections rekordbox's
    phrase analysis marks as drops/choruses (and with its mood). Tracks whose
    audio can't be decoded fall back to rekordbox's colour waveform, a rougher
    measurement that's marked as such. audio / analysis: {track id: path}."""
    hints: dict[str, tuple] = {}
    entries: dict[str, np.ndarray] = {}
    for tid, ext in analysis.items():
        try:
            entries[tid], sections, mood = read_analysis(ext)
            hints[tid] = (sections, mood)
        except (OSError, ValueError):
            pass
    results = analyse_library(audio, log, progress, cache_file, hints)
    for tid in list(results):
        # Genre guessing compares the whole-track measurements across the library,
        # so they must all be of one kind: rekordbox's waveform, as before. So do
        # the classic energy measurements: in direct mode it ranks waveforms only.
        f = {k: v for k, v in results[tid].items()
             if k not in WHOLE_TRACK and k not in CLASSIC_FEATURES}
        if tid in entries:
            try:
                wave = measure_waveform(entries[tid], *hints[tid])
                f.update({k: wave[k] for k in WHOLE_TRACK + CLASSIC_FEATURES})
            except ValueError:
                pass
        results[tid] = f
    fallback = 0
    for tid in entries:
        if tid not in results:
            try:
                results[tid] = measure_waveform(entries[tid], *hints[tid])
                fallback += 1
            except ValueError:
                pass
    if fallback:
        log(f"  {fallback} tracks couldn't be decoded, so rekordbox's waveform was used "
            "for them (a rougher estimate).")
    return results


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

# Each component: (weight, what it says). Correlated measurements are grouped
# into one component so they aren't counted twice (see COMPONENT_FEATURES).
WEIGHTS = {
    "activity": 0.30,   # how busy and hard-hitting the rhythm is
    "impact": 0.15,     # how hard the low end hits
    "pulse": 0.15,      # how strong and steady the beat is
    "density": 0.15,    # how dense / distorted / noisy the sound is
    "sustain": 0.10,    # how much of the track runs at full intensity
    "tempo": 0.10,      # felt tempo, only as far as there's a beat to carry it
    "mood": 0.05,       # rekordbox's own mood (waveform mode only)
}
_SHARED = {"impact": ("x_impact",), "pulse": ("x_pulse",), "density": ("x_density",),
           "sustain": ("x_sustain",)}
COMPONENT_FEATURES = {"audio": {"activity": ("x_onsets", "x_punch"), **_SHARED},
                      # the waveform can't resolve single hits: activity is punch only
                      "waveform": {"activity": ("x_punch",), **_SHARED}}
CORE = ("activity", "impact", "pulse", "density", "sustain")   # all needed for a score
# Calibration anchors per analysis mode: the measurement that maps to 0 and to 1
# (linear in between, clipped). Fixed numbers, so a track's score never depends
# on the other tracks in the library.
CALIBRATION = {
    "audio": {"x_onsets": (0.0, 6.46), "x_punch": (0.5, 8.59), "x_impact": (1.2, 6.64),
              "x_pulse": (0.15, 0.83), "x_density": (0.0, 0.25), "x_sustain": (0.31, 0.65)},
    "waveform": {"x_punch": (0.14, 1.49), "x_impact": (0.07, 1.14),
                 "x_pulse": (0.0, 0.47), "x_density": (0.33, 0.70), "x_sustain": (0.10, 0.45)},
}
CALIBRATION_NOTES = """How the anchors were set (benchmark_energy.derive_anchors):
- audio 0: what a signal with none of the quality measures (steady noise, a pure
  tone, a drone): no onsets, punch 0.5, low-end impact 1.2, beat 0.15, density 0.
  Sustain has no natural zero: its 10th percentile.
- audio 1: the 90th percentile of one 261-track library (mostly EDM, house, D&B,
  dubstep; measured 2026-10). Re-derived on two random halves of it, the 1
  points moved by under 4% except punch (7.6 vs 9.9: a long tail).
- waveform: matched to the audio scale on the same tracks (10th and 90th
  percentiles land where the audio ones do); the waveform is only the fallback
  for files that can't be decoded.
These set where 'level 5' sits; they were not fitted to listeners' ratings (there
are none yet). benchmark_energy.py evaluate measures that once there are."""
TEMPO_RANGE = (70.0, 175.0)            # felt BPM mapped to 0 and 1
LEVEL_THRESHOLDS = (20.0, 40.0, 60.0, 80.0)   # score at which levels 2, 3, 4, 5 start
MOOD_VALUE = {1: 1.0, 2: 0.5, 3: 0.0}  # rekordbox's mood: 1 high, 2 mid, 3 low


@dataclass
class EnergyResult:
    """One track's energy, with everything needed to explain it."""
    score: Optional[float]                  # 0-100; None when it couldn't be rated
    level: int                              # 1-5; 0 = not rated (see review)
    mode: str = ""                          # "audio" or "waveform"
    components: dict = field(default_factory=dict)  # name: (measured, 0-1, weight)
    warnings: list = field(default_factory=list)    # data-quality problems
    notes: list = field(default_factory=list)       # e.g. close to a level boundary
    review: bool = False                    # not enough to go on: check it by ear
    quality: str = "good"                   # "good", "limited" or "poor"; not a probability

    def explain(self) -> str:
        if self.score is None:
            return "Not rated: " + "; ".join(self.warnings)
        parts = [f"{name} {norm:.2f}" for name, (_, norm, _) in self.components.items()
                 if norm is not None]
        text = f"Energy {self.score:.0f}/100 (level {self.level}): " + ", ".join(parts)
        if self.warnings:
            text += ". Check: " + "; ".join(self.warnings)
        return text


def _scale(value: Optional[float], lo: float, hi: float) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def effective_bpm(bpm: float, genre: str) -> float:
    """BPM as felt: half-speed readings of Drum & Bass etc. are doubled."""
    norm = " ".join(genre.casefold().replace("&", " and ").split())
    if 0 < bpm < 100 and any(word in norm for word in DOUBLE_TIME):
        return bpm * 2
    return bpm


def _fold(bpm: float) -> float:
    while bpm < 80:
        bpm *= 2
    while bpm >= 180:
        bpm /= 2
    return bpm


def felt_tempo(bpm: float, genre: str, heard: Optional[float]) -> tuple[Optional[float], list]:
    """(the tempo to score, warnings). rekordbox's BPM first; the beat heard in
    the audio only stands in when the BPM is missing or implausible."""
    warnings = []
    if not bpm or not 50 <= bpm <= 220:
        if heard:
            warnings.append("no usable BPM; the beat heard in the audio was used")
            return heard, warnings
        warnings.append("no usable BPM and no clear beat: tempo not counted")
        return None, warnings
    felt = effective_bpm(bpm, genre)
    if heard:
        ratio = _fold(felt) / heard
        if not any(abs(ratio - r) < 0.04 * r for r in (1.0, 2 / 3, 1.5)):
            warnings.append(f"the BPM ({bpm:.0f}) doesn't match the beat heard "
                            f"(~{heard:.0f}); it may be wrong")
    return felt, warnings


def score_track(feats: dict, bpm: float = 0.0, genre: str = "") -> EnergyResult:
    """The absolute 0-100 energy score of one track, with its explanation.

    A missing measurement is never filled in with an average: the components
    in CORE are all needed (otherwise the track is flagged for review, not
    rated); tempo and mood are left out when unknown and the rest re-weighted.
    """
    mode = feats.get("x_mode")
    anchors = CALIBRATION.get(mode or "")
    warnings = list(feats.get("x_warnings") or [])
    if anchors is None:
        return EnergyResult(None, 0, mode or "", warnings=warnings + [
            "measured by an older version of the app; analyse it again"], review=True,
            quality="poor")
    components: dict[str, tuple] = {}
    for name, keys in COMPONENT_FEATURES[mode].items():
        raw = [feats.get(k) for k in keys]
        scaled = [_scale(v, *anchors[k]) for k, v in zip(keys, raw)]
        known = [s for s in scaled if s is not None]
        norm = float(np.mean(known)) if len(known) == len(keys) else None
        components[name] = (raw[0] if len(raw) == 1 else raw, norm, WEIGHTS[name])

    tempo, tempo_warnings = felt_tempo(bpm, genre, feats.get("x_tempo"))
    warnings += tempo_warnings
    pulse = components["pulse"][1]
    if tempo is not None and pulse is not None:
        components["tempo"] = (tempo, _scale(tempo, *TEMPO_RANGE) * pulse, WEIGHTS["tempo"])
    else:
        components["tempo"] = (tempo, None, WEIGHTS["tempo"])
    if pulse is not None and pulse < 0.25 and tempo is not None:
        warnings.append("little steady beat, so tempo barely counts")
    # Running at full level only adds energy when there's activity to sustain:
    # a constant drone is 100% "sustained" but not energetic.
    sustain, activity = components["sustain"], components["activity"][1]
    if sustain[1] is not None and activity is not None:
        components["sustain"] = (sustain[0], sustain[1] * activity, sustain[2])
    mood = MOOD_VALUE.get(int(feats.get("mood") or 0))
    components["mood"] = (feats.get("mood") or None, mood, WEIGHTS["mood"])

    missing = [n for n in CORE if components[n][1] is None]
    seconds = feats.get("x_seconds") or 0.0
    problems = []
    if seconds < MIN_SECONDS:
        problems.append(f"only {seconds:.0f} s of sound to judge")
    if missing:
        problems.append("couldn't measure " + ", ".join(missing))
    if problems:
        return EnergyResult(None, 0, mode, components, warnings + problems, review=True,
                            quality="poor")
    used = {n: c for n, c in components.items() if c[1] is not None}
    score = 100 * sum(c[1] * c[2] for c in used.values()) / sum(c[2] for c in used.values())
    level = absolute_level(score)
    notes = [f"close to the boundary between levels {i + 1} and {i + 2}"
             for i, t in enumerate(LEVEL_THRESHOLDS) if abs(score - t) < 2.5]
    return EnergyResult(round(score, 1), level, mode, components, warnings, notes,
                        quality="limited" if warnings else "good")


def absolute_level(score: float) -> int:
    """1-5 from the fixed LEVEL_THRESHOLDS (doesn't depend on any other track)."""
    return 1 + sum(score >= t for t in LEVEL_THRESHOLDS)


def _percentiles(values: list) -> np.ndarray:
    """Each value's place among the others, 0 (lowest) to 1 (highest); ties share it."""
    known = sorted((v, i) for i, v in enumerate(values))
    out = np.full(len(values), 0.5)
    if len(known) < 2:
        return out
    n, j = len(known), 0
    while j < n:
        k = j
        while k + 1 < n and known[k + 1][0] == known[j][0]:
            k += 1
        for _, i in known[j:k + 1]:
            out[i] = ((j + k) / 2) / (n - 1)
        j = k + 1
    return out


def relative_levels(scores: dict[str, float]) -> dict[str, int]:
    """1-5 by rank in this library: the most energetic fifth 5, the calmest fifth 1.

    Changes when the library changes, by design. The highest-scoring track is
    always 5 and the lowest 1; tracks with the same score share a level.
    """
    ids = list(scores)
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: 3}
    place = _percentiles([round(scores[i], 9) for i in ids])
    return {i: min(5, 1 + int(p * 5)) for i, p in zip(ids, place)}


def rate_tracks(features: dict[str, dict], bpms: dict[str, float],
                genres: Optional[dict[str, str]] = None,
                relative: bool = False) -> dict[str, EnergyResult]:
    """{track id: EnergyResult}. relative: levels by rank in this library instead
    of the absolute thresholds (the score itself is the same either way)."""
    genres = genres or {}
    results = {tid: score_track(f, bpms.get(tid, 0.0), genres.get(tid, ""))
               for tid, f in features.items()}
    if relative:
        ranked = relative_levels({t: r.score for t, r in results.items() if r.score is not None})
        for tid, level in ranked.items():
            results[tid].level = level
            results[tid].notes = []
    return results


def energy_scores(features: dict[str, dict], bpms: dict[str, float],
                  genres: Optional[dict[str, str]] = None) -> dict[str, float]:
    """{track id: 0-100 score} of the tracks that could be rated."""
    return {t: r.score for t, r in rate_tracks(features, bpms, genres).items()
            if r.score is not None}


def levels_from_features(features: dict[str, dict], bpms: dict[str, float],
                         genres: Optional[dict[str, str]] = None,
                         relative: bool = False) -> dict[str, int]:
    """{track id: 1-5, or 0 when it couldn't be rated}."""
    return {t: r.level for t, r in rate_tracks(features, bpms, genres, relative).items()}


# --------------------------------------------------------------------------- #
# The classic (v3) algorithm's scoring, restored unchanged
# --------------------------------------------------------------------------- #

def _classic_percentiles(values: list) -> np.ndarray:
    """Each value's place among the others, 0 (lowest) to 1 (highest); ties share
    it. Missing values (None) get the middle, 0.5."""
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


def classic_scores(features: dict[str, dict], bpms: dict[str, float],
                   genres: Optional[dict[str, str]] = None) -> dict[str, float]:
    """The classic score of every track that has its measurements: each
    measurement's rank among these tracks, weighted (CLASSIC_WEIGHTS)."""
    genres = genres or {}
    ids = [i for i in features if features[i].get("e_loud") is not None]
    score = np.zeros(len(ids))
    for name in CLASSIC_FEATURES:
        score += CLASSIC_WEIGHTS[name] * _classic_percentiles([features[i].get(name) for i in ids])
    tempo = [effective_bpm(bpms.get(i, 0), genres.get(i, "")) for i in ids]
    score += CLASSIC_WEIGHTS["tempo"] * _classic_percentiles([b if b > 0 else None for b in tempo])
    # rekordbox's mood: 1 = high, 2 = mid, 3 = low
    score += CLASSIC_WEIGHTS["mood"] * np.array([{1: 1.0, 2: 0.5, 3: 0.0}.get(
        int(features[i].get("mood") or 0), 0.5) for i in ids])
    return dict(zip(ids, score.tolist()))


def classic_levels(features: dict[str, dict], bpms: dict[str, float],
                   genres: Optional[dict[str, str]] = None
                   ) -> tuple[dict[str, float], dict[str, int]]:
    """(scores, 1-5 levels): by rank, the most energetic fifth 5, the calmest 1."""
    scores = classic_scores(features, bpms, genres)
    return scores, relative_levels(scores)


# --------------------------------------------------------------------------- #
# Cache + batch
# --------------------------------------------------------------------------- #

def cache_path() -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), "energy_cache.json")


REQUIRED = ("loudness", "flux", "brightness", "highs", "lows", "x_mode", "x_onsets",
            "x_punch", "x_impact", "x_pulse", "x_density", "x_sustain", "x_seconds"
            ) + CLASSIC_FEATURES


def _load_cache(path: str) -> dict:
    """The cached measurements, or {} if the cache is from another version.

    Entries missing any measurement this version needs are dropped too, so
    nothing stale or partial is ever reused.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != CACHE_VERSION:
            return {}
        return {p: e for p, e in data["tracks"].items()
                if isinstance(e, dict) and all(k in e.get("features", {}) for k in REQUIRED)}
    except (OSError, ValueError, KeyError, AttributeError):
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
    hints: Optional[dict[str, tuple]] = None,
) -> dict[str, dict]:
    """Measure every track. paths = {track id: file path}. Returns {track id: features}.

    hints: {track id: (drop/chorus sections, mood)} from rekordbox. A cached
    measurement is reused only if the file and these hints are unchanged.
    """
    hints = hints or {}
    cache_file = cache_file or cache_path()
    cache = _load_cache(cache_file)
    results: dict[str, dict] = {}
    todo = []
    for tid, path in paths.items():
        try:
            st = os.stat(path)
        except OSError:
            continue
        sections, mood = hints.get(tid, ([], 0))
        stamp = [st.st_size, int(st.st_mtime), [[round(a, 2), round(b, 2)] for a, b in sections],
                 mood]
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
        futures = {pool.submit(analyse_file, p, *hints.get(tid, ([], 0))): (tid, p, stamp)
                   for tid, p, stamp in todo}
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
