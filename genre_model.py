"""Optional audio genre models, run locally with onnxruntime.

Two backends (pick one in the app):

* YAMNet (Google, Apache-2.0, 16 MB): tags 0.96 s frames of 16 kHz audio with
  521 AudioSet classes, about 50 of them broad music genres ("House music",
  "Techno", "Drum and bass"...), plus a 1024-number embedding per frame.
* Discogs-EffNet (Music Technology Group, UPF; CC BY-NC-SA 4.0, i.e.
  non-commercial use only, proprietary licence on request; 18 MB): predicts
  400 Discogs styles ("Electronic---Tech House", "Electronic---Deep House",
  "Electronic---Drum n Bass"...) from 128-frame mel-spectrogram patches
  (about 2 s), plus a 1280-number embedding. Essentia's own audio front end
  is reproduced in numpy (melspectrogram()), checked against Essentia's
  published reference values (test_genre_model.EssentiaFrontEndTests).

Both run on this computer; the model file is downloaded once, from a fixed
version, and checked against its SHA-256. Scores are independent per-class
activations between 0 and 1: not probabilities, and they don't add up to 1.
genres.py turns them into evidence; nothing here decides a genre.

Per track:
1. Pick up to MAX_SEGMENTS excerpts of SEGMENT_SECONDS: rekordbox's drops /
   choruses first (from its phrase analysis), then fixed points in the body
   of the track (30% / 50% / 70%), never the very start or end. Short tracks
   use one excerpt.
2. Decode, convert to 16 kHz mono, skip excerpts that are silent.
3. Run the model; per excerpt, average the frame / patch scores and
   embeddings. The track's embedding is the mean over its excerpts.

Results are cached per file (size + modification time + chosen excerpts), one
cache per backend, under a key naming the model file and the preprocessing
version, so a new model or a change in preprocessing re-analyses everything.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

REVISION = "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd"
MODEL = {
    "name": "YAMNet",
    "license": "Apache-2.0 (Google, tensorflow/models); AudioSet labels CC BY 4.0",
    "url": f"https://huggingface.co/audiomagic/yamnet-onnx/resolve/{REVISION}/yamnet.onnx",
    "sha256": "d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e",
    "size": 16093355,
}
MODEL_ID = f"yamnet-onnx@{REVISION[:12]}"
PREPROCESS_VERSION = 1      # bump when segment choice / resampling / pooling changes
CACHE_KEY = f"{MODEL_ID}|pre{PREPROCESS_VERSION}"

SR = 16000
SEGMENT_SECONDS = 10.0
MAX_SEGMENTS = 4
MIN_SECONDS = 3.0           # shorter audio isn't worth classifying
SILENCE_DB = -50.0          # an excerpt quieter than this (RMS, dBFS) is skipped
FIXED_POINTS = (0.3, 0.5, 0.7)

MUSIC_INDEX = 132
# YAMNet class index -> AudioSet display name, for the music genre classes
# (yamnet_class_map.csv at the pinned revision; checked by the real-model test).
GENRE_CLASSES = {
    211: "Pop music", 212: "Hip hop music", 214: "Rock music", 215: "Heavy metal",
    216: "Punk rock", 217: "Grunge", 218: "Progressive rock", 219: "Rock and roll",
    220: "Psychedelic rock", 221: "Rhythm and blues", 222: "Soul music", 223: "Reggae",
    224: "Country", 225: "Swing music", 226: "Bluegrass", 227: "Funk", 228: "Folk music",
    229: "Middle Eastern music", 230: "Jazz", 231: "Disco", 232: "Classical music",
    233: "Opera", 234: "Electronic music", 235: "House music", 236: "Techno", 237: "Dubstep",
    238: "Drum and bass", 239: "Electronica", 240: "Electronic dance music",
    241: "Ambient music", 242: "Trance music", 243: "Music of Latin America",
    244: "Salsa music", 245: "Flamenco", 246: "Blues", 248: "New-age music",
    251: "Music of Africa", 252: "Afrobeat", 253: "Christian music", 254: "Gospel music",
    255: "Music of Asia", 256: "Carnatic music", 257: "Music of Bollywood", 258: "Ska",
    259: "Traditional music", 269: "Dance music",
}


DISCOGS_FILE = "discogs-effnet-bsdynamic-1.onnx"
DISCOGS_MODEL = {
    "name": "Discogs-EffNet",
    "license": "CC BY-NC-SA 4.0 (Music Technology Group, Universitat Pompeu Fabra): "
               "non-commercial use; a proprietary licence is available from the MTG",
    "url": f"https://essentia.upf.edu/models/feature-extractors/discogs-effnet/{DISCOGS_FILE}",
    "sha256": "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c",
    "size": 18027718,
}
DISCOGS_KEEP = 0.01         # styles scoring below this in an excerpt aren't stored
PATCH_FRAMES, PATCH_HOP, MEL_BANDS = 128, 62, 96


@dataclass(frozen=True)
class Backend:
    key: str                # "yamnet" / "discogs-effnet"
    model: dict             # name, license, url, sha256, size
    filename: str
    preprocess: int         # bump when this backend's preprocessing changes
    cache_name: str
    commercial_use: bool

    @property
    def cache_key(self) -> str:
        return f"{self.key}@{self.model['sha256'][:12]}|pre{self.preprocess}"


YAMNET = Backend("yamnet", MODEL, f"yamnet-{REVISION[:12]}.onnx", PREPROCESS_VERSION,
                 "genre_model_cache.json", True)
DISCOGS = Backend("discogs-effnet", DISCOGS_MODEL,
                  f"discogs-effnet-1-{DISCOGS_MODEL['sha256'][:12]}.onnx", 1,
                  "genre_model_cache-discogs-effnet.json", False)
BACKENDS = {b.key: b for b in (YAMNET, DISCOGS)}


def backend_of(key) -> Backend:
    if isinstance(key, Backend):
        return key
    return BACKENDS.get(key or "yamnet", YAMNET)


@dataclass
class ModelResult:
    """The model's view of one track."""
    segments: list[float]                  # excerpt start times (s) that were used
    genre_scores: list[dict[str, float]]   # per excerpt: model class name -> score
    music: list[float]                     # per excerpt: how sure it is that this is music
    embedding: Optional[np.ndarray] = None  # mean embedding, unit length
    model: str = "yamnet"                  # which backend produced it

    def to_json(self) -> dict:
        emb = None
        if self.embedding is not None:
            emb = base64.b64encode(self.embedding.astype("<f2").tobytes()).decode("ascii")
        return {"segments": self.segments, "genre_scores": self.genre_scores,
                "music": self.music, "embedding": emb, "model": self.model}

    @classmethod
    def from_json(cls, data: dict) -> "ModelResult":
        emb = data.get("embedding")
        if emb:
            emb = np.frombuffer(base64.b64decode(emb), "<f2").astype(np.float32)
        return cls(list(data["segments"]), [dict(s) for s in data["genre_scores"]],
                   list(data["music"]), emb if emb is not None and len(emb) else None,
                   data.get("model", "yamnet"))


@dataclass
class TrackInput:
    id: str
    path: str
    length: float = 0.0                    # seconds, 0 if unknown
    sections: list = field(default_factory=list)   # [(start s, end s)] drops / choruses


# --------------------------------------------------------------------------- #
# Model file and runtime
# --------------------------------------------------------------------------- #

def runtime_problem() -> str:
    """'' if onnxruntime can be used here, else why not."""
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:     # missing, or blocked by the OS (e.g. Smart App Control)
        return f"onnxruntime couldn't be loaded ({type(exc).__name__}: {exc})"
    return ""


def model_path(backend=None) -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), "models", backend_of(backend).filename)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_model(log: Callable[[str], None] = print,
                 progress: Optional[Callable[[int, int], None]] = None, backend=None) -> str:
    """The model file's path, downloading it (once) if needed. Raises on failure."""
    b = backend_of(backend)
    model = b.model
    path = model_path(b)
    if os.path.exists(path) and os.path.getsize(path) == model["size"]:
        return path
    import ssl
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"Downloading the {model['name']} genre model ({model['size'] // 1_000_000} MB, once) ...")
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    part = path + ".part"
    request = urllib.request.Request(model["url"], headers={"User-Agent": "RekordboxOrganizer"})
    with urllib.request.urlopen(request, context=context, timeout=60) as resp, \
            open(part, "wb") as out:
        done = 0
        while True:
            block = resp.read(1 << 16)
            if not block:
                break
            out.write(block)
            done += len(block)
            if progress:
                progress(done, model["size"])
    if _sha256(part) != model["sha256"]:
        os.remove(part)
        raise RuntimeError("the downloaded genre model didn't match its checksum")
    os.replace(part, path)
    return path


class Classifier:
    """The loaded model. run() is safe to call from several threads."""

    def __init__(self, path: Optional[str] = None, backend=None):
        import onnxruntime as ort
        self.backend = backend_of(backend)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1      # several tracks run in parallel instead
        self.session = ort.InferenceSession(path or model_path(self.backend), options,
                                            providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def run(self, wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """16 kHz mono float32 audio -> (per-frame / patch scores, embeddings)."""
        if self.backend is DISCOGS:
            mel = patches(melspectrogram(wave))
            if mel is None:
                return np.zeros((0, 400), np.float32), np.zeros((0, 1280), np.float32)
            scores, embeddings = self.session.run(None, {self.input: mel})[:2]
            return scores, embeddings
        scores, embeddings = self.session.run(None, {self.input: wave.astype(np.float32)})[:2]
        return scores, embeddings

    def summarise(self, mean_scores: np.ndarray) -> tuple[dict[str, float], float]:
        """An excerpt's mean scores -> ({class name: score} worth keeping, music score)."""
        return summarise(self.backend, mean_scores)


def summarise(backend: Backend, mean: np.ndarray) -> tuple[dict[str, float], float]:
    if backend is DISCOGS:
        from discogs_styles import STYLES
        kept = {STYLES[i]: round(float(mean[i]), 5) for i in np.flatnonzero(mean >= DISCOGS_KEEP)}
        non_music = max((float(mean[i]) for i, s in enumerate(STYLES)
                         if s.startswith("Non-Music---")), default=0.0)
        return kept, round(max(0.0, float(mean.max()) - non_music), 5)
    return ({name: round(float(mean[i]), 5) for i, name in GENRE_CLASSES.items()},
            round(float(mean[MUSIC_INDEX]), 5))


# --------------------------------------------------------------------------- #
# Essentia's mel front end (TensorflowInputMusiCNN), as used by Discogs-EffNet
# --------------------------------------------------------------------------- #
# Parameters from essentia/src/algorithms/spectral/tensorflowinputmusicnn.cpp:
# 512-sample Hann frames (symmetric), hop 256, magnitude spectrum, 96 mel bands
# 0-8000 Hz (Slaney mel scale, linear-weighted triangles, unit_tri area
# normalisation) on the power spectrum, then log10(1 + 10000 x).

def _hz2mel(hz):
    hz = np.asarray(hz, float)
    return np.where(hz < 1000, hz * 3 / 200,
                    15 + np.log(np.maximum(hz, 1e-9) / 1000) / (np.log(6.4) / 27))


def _mel2hz(mel):
    mel = np.asarray(mel, float)
    return np.where(mel < 15, mel * 200 / 3, 1000 * np.exp((mel - 15) * np.log(6.4) / 27))


def _mel_filters(n_bands: int = MEL_BANDS, sr: int = SR, size: int = 257) -> np.ndarray:
    edges = _mel2hz(np.linspace(_hz2mel(0), _hz2mel(sr / 2), n_bands + 2))
    step = (sr / 2) / (size - 1)
    bank = np.zeros((n_bands, size))
    for i in range(n_bands):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        for j in range(int(np.ceil(lo / step)), int(np.floor(hi / step)) + 1):
            f = j * step
            bank[i, j] = (f - lo) / (mid - lo) if f < mid else (hi - f) / (hi - mid)
        bank[i] /= ((mid - lo) + (hi - mid)) / 2
    return bank


_FILTERS = None
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 511)


def melspectrogram(wave: np.ndarray) -> np.ndarray:
    """16 kHz mono audio -> (frames, 96) log-mel, matching Essentia's front end."""
    global _FILTERS
    if _FILTERS is None:
        _FILTERS = _mel_filters().T
    x = np.concatenate([np.zeros(256), np.asarray(wave, np.float64), np.zeros(512)])
    n = max(0, 1 + (len(x) - 768) // 256)
    if n == 0:
        return np.zeros((0, MEL_BANDS), np.float32)
    idx = np.arange(512)[None, :] + 256 * np.arange(n)[:, None]
    spectrum = np.abs(np.fft.rfft(x[idx] * _WINDOW, axis=1))
    return np.log10(1 + 10000 * (spectrum ** 2) @ _FILTERS).astype(np.float32)


def patches(mel: np.ndarray) -> Optional[np.ndarray]:
    """(frames, 96) -> (n, 128, 96) patches, hop 62 frames; None if under one patch."""
    n = 1 + (len(mel) - PATCH_FRAMES) // PATCH_HOP
    if len(mel) < PATCH_FRAMES:
        return None
    return np.stack([mel[i * PATCH_HOP:i * PATCH_HOP + PATCH_FRAMES] for i in range(n)])


def resample_essentia(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """FFT resampling with a 7.0-7.8 kHz roll-off, like the resampler Essentia's
    MonoLoader uses (fitted to Essentia's reference values; see the tests)."""
    if len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x.astype(np.float64))
    keep = n_out // 2 + 1
    spec = spec[:keep] if len(spec) >= keep else np.concatenate(
        [spec, np.zeros(keep - len(spec), complex)])
    freq = np.arange(keep) * sr_in / len(x)
    gain = np.clip((7800.0 - freq) / 800.0, 0, 1)
    spec = spec * (0.5 - 0.5 * np.cos(np.pi * gain))
    return (np.fft.irfft(spec, n_out) * (n_out / len(x))).astype(np.float32)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #

def choose_segments(length: float, sections: Optional[list] = None) -> list[float]:
    """Excerpt start times (s) for a track of `length` seconds. Deterministic.

    Drops / choruses first (the middle of the longest few, in time order),
    then fixed points in the body of the track, keeping excerpts apart.
    """
    if length <= 0:
        return []
    if length < SEGMENT_SECONDS * 1.5:
        return [0.0]                     # short track: one excerpt from the start
    last = length - SEGMENT_SECONDS
    picks: list[float] = []

    def add(start: float) -> None:
        start = round(min(max(start, 0.0), last), 2)
        if len(picks) < MAX_SEGMENTS and all(abs(start - p) >= SEGMENT_SECONDS for p in picks):
            picks.append(start)

    usable = [(s, e) for s, e in (sections or []) if e - s >= 4 and s < length]
    for s, e in sorted(usable, key=lambda se: (-(se[1] - se[0]), se[0]))[:MAX_SEGMENTS - 1]:
        add((s + min(e, length)) / 2 - SEGMENT_SECONDS / 2)
    for point in FIXED_POINTS:
        add(length * point - SEGMENT_SECONDS / 2)
    return sorted(picks)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Band-limited resampling via the FFT (deterministic; fine for short excerpts)."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x.astype(np.float64))
    keep = n_out // 2 + 1
    if len(spec) >= keep:
        spec = spec[:keep]
    else:
        spec = np.concatenate([spec, np.zeros(keep - len(spec), complex)])
    return (np.fft.irfft(spec, n_out) * (n_out / len(x))).astype(np.float32)


def excerpts(samples: np.ndarray, sr: int, starts: list[float],
             backend=None) -> list[tuple[float, np.ndarray]]:
    """[(start, 16 kHz excerpt)] for the starts that have sound."""
    convert = resample_essentia if backend_of(backend) is DISCOGS else resample
    out = []
    for start in starts:
        a = int(start * sr)
        piece = samples[a:a + int(SEGMENT_SECONDS * sr)]
        if len(piece) < MIN_SECONDS * sr:
            continue
        rms = float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)) + 1e-20))
        if 20 * np.log10(rms) < SILENCE_DB:
            continue
        out.append((start, convert(piece, sr)))
    return out


def analyse_audio(samples: np.ndarray, sr: int, starts: list[float],
                  classifier: Classifier) -> ModelResult:
    """Run the model on the chosen excerpts. Raises ValueError if nothing is usable."""
    backend = getattr(classifier, "backend", YAMNET)
    used = excerpts(samples, sr, starts, backend)
    if not used:
        raise ValueError("silent or too short")
    seg_scores, music, embeddings, kept_starts = [], [], [], []
    for start, wave in used:
        scores, emb = classifier.run(wave)
        if len(scores) == 0:
            continue
        kept, is_music = summarise(backend, scores.mean(axis=0))
        seg_scores.append(kept)
        music.append(is_music)
        embeddings.append(emb.mean(axis=0))
        kept_starts.append(start)
    if not seg_scores:
        raise ValueError("too short for the model")
    emb = np.mean(embeddings, axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return ModelResult(kept_starts, seg_scores, music, emb.astype(np.float32), backend.key)


# --------------------------------------------------------------------------- #
# Cache + batch
# --------------------------------------------------------------------------- #

def cache_path(backend=None) -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), backend_of(backend).cache_name)


def _cache_key(backend: Backend) -> str:
    return CACHE_KEY if backend is YAMNET else backend.cache_key


def _load_cache(path: str, key: str = CACHE_KEY) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["tracks"] if data.get("key") == key else {}
    except (OSError, ValueError, KeyError):
        return {}


def _save_cache(path: str, tracks: dict, key: str = CACHE_KEY) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"key": key, "tracks": tracks}, f)
    os.replace(tmp, path)


def analyse_tracks(items: list[TrackInput], log: Callable[[str], None] = print,
                   progress: Optional[Callable[[int, int], None]] = None,
                   classifier: Optional[Classifier] = None,
                   cache_file: Optional[str] = None, backend=None) -> dict[str, ModelResult]:
    """Model results for every track that can be read: {track id: ModelResult}.

    Problems (missing files, unreadable audio, silence, model errors) are
    counted and logged; those tracks are simply left out.
    """
    import energy   # decoders
    b = getattr(classifier, "backend", None) or backend_of(backend)
    key = _cache_key(b)
    cache_file = cache_file or cache_path(b)
    cache = _load_cache(cache_file, key)
    results: dict[str, ModelResult] = {}
    todo = []
    problems: dict[str, int] = {}
    for item in items:
        try:
            st = os.stat(item.path)
        except (OSError, ValueError):
            problems["file not found"] = problems.get("file not found", 0) + 1
            continue
        stamp = [st.st_size, int(st.st_mtime)]
        starts = choose_segments(item.length, item.sections) if item.length > 0 else None
        hit = cache.get(item.path)
        if hit and hit.get("stamp") == stamp and (starts is None or hit.get("starts") == starts):
            try:
                results[item.id] = ModelResult.from_json(hit["result"])
                continue
            except (KeyError, TypeError, ValueError):
                pass
        todo.append((item, stamp, starts))

    if todo:
        log(f"Genre model ({b.model['name']}): {len(results)} tracks already analysed, "
            f"{len(todo)} to listen to.")
    if todo and classifier is None:
        classifier = Classifier(backend=b)

    lock = threading.Lock()

    def work(item: TrackInput, starts: Optional[list[float]]):
        samples, sr = energy.decode(item.path)
        if starts is None:          # length unknown until decoded
            starts = choose_segments(len(samples) / sr, item.sections)
        if len(samples) < MIN_SECONDS * sr:
            raise ValueError("too short")
        return analyse_audio(samples, sr, starts, classifier), starts

    done = 0
    workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, item, starts): (item, stamp, starts)
                   for item, stamp, starts in todo}
        for fut in as_completed(futures):
            item, stamp, starts = futures[fut]
            try:
                result, used_starts = fut.result()
                results[item.id] = result
                with lock:
                    cache[item.path] = {"stamp": stamp, "starts": starts if starts is not None
                                        else used_starts, "result": result.to_json()}
            except ValueError as exc:
                key = str(exc) if "short" in str(exc) or "silent" in str(exc) else "unreadable"
                problems[key] = problems.get(key, 0) + 1
            except Exception:       # decoder or model failure on this file
                problems["unreadable"] = problems.get("unreadable", 0) + 1
            done += 1
            if progress:
                progress(done, len(todo))
    if todo:
        _save_cache(cache_file, cache, key)
    if problems:
        log("  Genre model skipped: " + ", ".join(f"{n} {why}" for why, n in sorted(problems.items())))
    return results
