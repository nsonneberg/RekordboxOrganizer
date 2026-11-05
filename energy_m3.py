"""M3, the opt-in energy model: a ridge regression frozen after a pre-registered test.

Calibrated to one listener's ratings of one library, so its levels mean "how that
listener hears energy". Inference only: the parameters in energy_m3_model.json are used
exactly as frozen and nothing here fits, ranks or adapts to the library, so a track's
level doesn't depend on the other tracks.

It needs measurements from both rekordbox's analysis and the decoded audio, so it only
works straight from rekordbox. A track missing any of the 22 inputs is skipped with the
reason; there's no estimate and no fallback to another algorithm.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Optional

import numpy as np

MODEL_FILE = "energy_m3_model.json"
# Of the parameters only (see _fingerprint), so line endings can't trip it.
MODEL_FINGERPRINT = "f11e62783c7a4498368e048498489105f0f43b53eff60d6f0ae84f1dae30bcd3"
PARAMETERS = ("features", "mu", "sd", "intercept", "beta", "cuts")
CLASSIC = ("e_loud", "e_punch", "e_power", "e_bass", "e_highs")

# Every input in the model's order, except felt_bpm: (measured from, measurement).
INPUTS = {
    "lufs": ("audio", "x_lufs"), "lra": ("audio", "x_lra"), "density": ("audio", "x_density"),
    "onsets": ("audio", "x_onsets"), "pulse": ("audio", "x_pulse"), "punch": ("audio", "x_punch"),
    "impact": ("audio", "x_impact"), "sustain": ("audio", "x_sustain"),
    "w_punch": ("waveform", "x_punch"), "w_density": ("waveform", "x_density"),
    "w_sustain": ("waveform", "x_sustain"),
    **{f"v3a_{k}": ("audio", k) for k in CLASSIC},
    **{f"v3w_{k}": ("waveform", k) for k in CLASSIC},
}
FEATURES = ("felt_bpm", *INPUTS)


class ModelError(ValueError):
    """The packaged model file is missing or isn't the frozen model."""


def model_path() -> str:
    # In the built app, data files are unpacked to sys._MEIPASS.
    return os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
                        MODEL_FILE)


def _fingerprint(model: dict) -> str:
    params = {k: model[k] for k in PARAMETERS}
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
                          ).hexdigest()


def load_model(path: Optional[str] = None) -> dict:
    """The frozen parameters, or ModelError if the file is missing or was changed."""
    try:
        with open(path or model_path(), encoding="utf-8") as f:
            model = json.load(f)
        if _fingerprint(model) != MODEL_FINGERPRINT:
            raise ValueError("its parameters aren't the frozen ones")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelError(f"Couldn't load {MODEL_FILE}: {exc}") from exc
    return model


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def model_inputs(bpm: float, genre: str, audio: dict, waveform: dict) -> tuple[list, list]:
    """(the inputs in the model's order, the names of any missing or invalid ones).

    genre is the track's own tag, never a guess: felt BPM was frozen on that.
    """
    import energy

    sources = {"audio": audio, "waveform": waveform}
    bpm = _number(bpm)
    values = [energy.effective_bpm(bpm, genre) if bpm and bpm > 0 else None]
    values += [_number(sources[source].get(name)) for source, name in INPUTS.values()]
    return values, [name for name, value in zip(FEATURES, values) if value is None]


def predict(model: dict, values: list) -> tuple[float, int]:
    """(score, level 1-5) of one complete set of inputs."""
    x = (np.array(values, float) - np.array(model["mu"])) / np.array(model["sd"])
    score = float(model["intercept"] + x @ np.array(model["beta"]))
    return score, 1 + int(np.sum(score >= np.array(model["cuts"])))


def _measurements(tracks: list, features: dict[str, dict], cache_file: Optional[str]
                  ) -> tuple[dict, dict, dict]:
    """({id: audio measurements}, {id: waveform measurements}, {id: why it has none}).

    energy.analyse_rekordbox keeps only some of each kind in `features`, so they're
    read again: the same analysis files, and the audio from its cache where it can.
    """
    import energy

    waves, hints, skipped = {}, {}, {}
    audio_paths = {}
    for t in tracks:
        if not t.waveform:
            skipped[t.id] = "no rekordbox analysis (needs direct mode and an analysed track)"
        elif t.id not in features:
            skipped[t.id] = "it couldn't be measured"
        elif features[t.id].get("x_mode") != "audio" or not t.path:
            skipped[t.id] = "its audio couldn't be decoded"
        else:
            try:
                entries, sections, mood = energy.read_analysis(t.waveform)
                waves[t.id] = energy.measure_waveform(entries, sections, mood)
                hints[t.id] = (sections, mood)
                audio_paths[t.id] = t.path
            except (OSError, ValueError):
                skipped[t.id] = "its rekordbox analysis couldn't be read"
    audio = energy.analyse_library(audio_paths, lambda _: None, None, cache_file, hints)
    for tid in audio_paths:
        if tid not in audio:
            skipped[tid] = "its audio couldn't be decoded"
    return audio, waves, skipped


def score_tracks(tracks: list, features: dict[str, dict], model: dict,
                 cache_file: Optional[str] = None) -> tuple[dict, dict]:
    """({id: (score, level)}, {id: why it was skipped}) for every track."""
    audio, waves, skipped = _measurements(tracks, features, cache_file)
    scored = {}
    for t in tracks:
        if t.id in skipped:
            continue
        values, bad = model_inputs(t.bpm, t.genre, audio[t.id], waves[t.id])
        if bad:
            skipped[t.id] = "no BPM" if bad == ["felt_bpm"] else (
                "missing or invalid measurements: " + ", ".join(bad))
        else:
            scored[t.id] = predict(model, values)
    return scored, skipped
