"""Genre families, genre guessing, and auto genre folders.

rekordbox doesn't detect genre, so genres for untagged or vaguely tagged songs
("Electronic", "Dance & EDM") are estimated by combining evidence:

  copy      Another copy of the same recording has a trusted genre.
  model     The optional audio model (genre_model.py) on several excerpts:
            Discogs-EffNet's 400 Discogs styles, or YAMNet's broad genres.
  library   A small classifier trained on the songs you've tagged, using the
            audio model's embeddings (what the songs sound like to it).
  sound     Without the model: tagged songs that rekordbox's waveform
            measurements say are similar.
  metadata  Optional online release metadata (metadata.py: MusicBrainz,
            Discogs), weighted by how well the release matched.
  tempo/tag Secondary only: a BPM that doesn't fit a family's usual range
            (half / double speed counts as fitting), or a vague electronic tag,
            lowers that family's score. They can never raise one, so BPM alone
            never decides a genre.

Fusion (_fuse): each source gives relative shares over genre families and a
quality for this track (how clear the model was, how well the release
matched). A source's weight is quality x its *measured reliability*: how
often its best family was right on your own tagged songs, held out one
recording at a time for sources that learn from them (reliability()). With
no tagged songs every source starts at a neutral 0.5.

The result is a ranked list of families with relative scores. They are
heuristic shares that add up to 1, not calibrated probabilities. A track is
"confident" only when the best family clearly leads (MIN_SCORE, MIN_MARGIN),
enough reliable evidence backs it (MIN_WEIGHT), and no two reliable sources
point at different families; otherwise it's "uncertain" and needs review.

Genres are hierarchical (taxonomy.py, configurable): a parent ("Electronic"),
the family ("House"), and a subgenre ("Tech House") only when the evidence
for it is clear.

Learned only from trusted labels (training_labels): genres you picked in the
review window, plus specific rekordbox tags, but not suggestions this app
wrote that you saved unchanged. Copies of a recording count once, and related
versions (remixes / edits of the same title) are held out together when
measuring.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import taxonomy
from organizer import Track, _norm_genre, genre_score
from taxonomy import ELECTRONIC_PARENT

UNKNOWN = "Unknown Genre"
DETECTED_SUFFIX = " (detected)"
SUGGESTED_SUFFIX = " (suggested)"
NEEDS_REVIEW = "Needs Review"

# Tags that say little about the actual style.
ELECTRONIC_VAGUE = {
    "electronic", "electronica", "dance", "edm", "dance and edm", "dance electronic",
    "electronic dance", "electronic dance music", "club",
}
VAGUE_TAGS = ELECTRONIC_VAGUE | {"other", "unknown", "misc", "various", "none", "genre", "music"}

TEMPO_MISMATCH = 0.6        # a family whose usual tempo doesn't fit keeps 60% of its score
AUDIOSET_ELECTRONIC = ("Electronic music", "Electronica", "Dance music")

# Decision thresholds (heuristic; chosen to keep false confidence rare, see README).
MIN_SCORE = 0.45            # the best family's share of the evidence
MIN_MARGIN = 0.15           # ...and its lead over the runner-up
MIN_WEIGHT = 0.35           # total reliability-weighted evidence behind the result
DISAGREE_WEIGHT = 0.25      # sources at least this heavy must not point elsewhere
MIN_MODEL_EVIDENCE = 0.05   # summed model genre activation; below this the model has no opinion
CLEAR_EVIDENCE = {"yamnet": 0.3, "discogs-effnet": 0.6}   # activation sum counted as fully clear
MIN_AGREEMENT = 0.5         # share of excerpts whose own best family matches the result
MIN_MUSIC = 0.2             # below this "music" score the model doesn't think it's music
K_NEIGHBOURS = 5
MIN_GROUPS = 3              # labelled recordings per family before the library classifier uses it
SUBGENRE_SHARE = 0.6        # neighbours (of the chosen family) that must agree on a subgenre
SUB_MIN, SUB_MARGIN = 0.5, 0.15   # subgenre share / lead needed among subgenre votes
SUB_SUPPORT = 0.35          # ...and the winner's share of the weight of sources naming subgenres
RELIABILITY_PRIOR = (1, 1)  # (right, wrong) pseudo-counts: 0.5 with no evidence
MAX_RELIABILITY_CHECKS = 200

# confident: assigned automatically. suggested: a clear guess for a genre that
# hasn't proved reliable enough to assign, shown for the user to approve.
# uncertain: unclear or the sources disagree (Needs Review).
CONFIDENT, SUGGESTED, UNCERTAIN = "confident", "suggested", "uncertain"
SUGGEST_MIN = 0.5           # without a threshold: suggest only guesses likelier than the rest together
SOURCES = ("copy", "model", "library", "sound", "metadata")


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #

FAMILIES: list = []         # [(family, [words])], in taxonomy order
PARENTS: dict = {}          # family -> parent
ELECTRONIC: set = set()     # families under "Electronic"
SUBGENRES: dict = {}        # family -> [subgenres]
TYPICAL_BPM: dict = {}      # family -> (low, high)
AUDIOSET_FAMILY: dict = {}  # AudioSet class -> family
_SYNONYMS: dict = {}        # squashed name -> (family, subgenre)
_LABEL_CACHE: dict = {}


def _squash(norm: str) -> str:
    return norm.replace(" ", "")


def refresh_taxonomy(reload: bool = False) -> None:
    """Rebuild the lookups from taxonomy.load() (after taxonomy.json changes)."""
    global FAMILIES, PARENTS, ELECTRONIC, SUBGENRES, TYPICAL_BPM, AUDIOSET_FAMILY
    tax = taxonomy.load(reload)
    FAMILIES = [(name, list(spec["words"])) for name, spec in tax["families"].items()]
    PARENTS = {name: spec["parent"] for name, spec in tax["families"].items() if spec["parent"]}
    ELECTRONIC = {name for name, parent in PARENTS.items() if parent == ELECTRONIC_PARENT}
    SUBGENRES = {name: list(spec["subgenres"]) for name, spec in tax["families"].items()}
    TYPICAL_BPM = {name: tuple(span) for name, span in tax["typical_bpm"].items()}
    AUDIOSET_FAMILY = dict(tax["audioset"])
    _SYNONYMS.clear()
    for name, spec in tax["families"].items():
        for sub, synonyms in spec["subgenres"].items():
            for text in [sub] + list(synonyms):
                _SYNONYMS.setdefault(_squash(_norm_genre(text)), (name, sub))
    _LABEL_CACHE.clear()


refresh_taxonomy()


def is_vague(tag: str) -> bool:
    norm = _norm_genre(tag)
    return not norm or norm in VAGUE_TAGS


def family_of(tag: str) -> str:
    """Genre family for a tag; a tag that fits no known family becomes its own family."""
    norm = _norm_genre(tag)
    if not norm:
        return UNKNOWN
    best, best_score = None, None
    for family, words in FAMILIES:
        for word in words:
            score = genre_score(word, norm)
            if score and (best_score is None or score > best_score):
                best, best_score = family, score
    return best or " ".join(tag.split())


def known_family(family: str) -> bool:
    return family in SUBGENRES


def subgenre_of(tag: str) -> tuple[str, str]:
    """(family, subgenre) for a tag; subgenre is '' when the tag only names the family.

    Known subgenres and their synonyms get the standard spelling
    ("tech-house" -> "Tech House", "UKG" -> "UK Garage"); other specific tags
    keep their own ("Melodic House & Techno").
    """
    norm = _norm_genre(tag)
    if not norm:
        return UNKNOWN, ""
    hit = _SYNONYMS.get(_squash(norm))
    if hit:
        return hit
    family = family_of(tag)
    if not known_family(family):
        return family, ""
    first = next((ws[0] for f, ws in FAMILIES if f == family and ws), "")
    if norm in (_norm_genre(family), _norm_genre(first)):
        return family, ""
    return family, " ".join(tag.split())


def parent_of(family: str) -> str:
    return PARENTS.get(family, "")


def model_label(model: str, name: str) -> list[tuple[str, str, float]]:
    """An audio model's class name -> [(family, subgenre, weight)] ([] = says nothing useful)."""
    key = (model, name)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    tax = taxonomy.load()
    out: list = []
    if model == "discogs-effnet":
        if name in tax["discogs"]:
            out = taxonomy.label_targets(tax["discogs"][name])
        else:
            genre, _, style = name.partition("---")
            family, sub = subgenre_of(style)
            if known_family(family):
                out = [(family, sub, 1.0)]
            elif genre in tax["discogs_genres"]:
                out = [(tax["discogs_genres"][genre], "", 1.0)]
    elif name in AUDIOSET_FAMILY:
        out = [(AUDIOSET_FAMILY[name], "", 1.0)]
    _LABEL_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #

@dataclass
class GenrePrediction:
    status: str                       # CONFIDENT, SUGGESTED or UNCERTAIN
    family: str = ""                  # best family ("" if there's no evidence at all)
    subgenre: str = ""                # only when the evidence for it is clear
    parent: str = ""                  # "Electronic" for electronic families
    score: float = 0.0                # the family's relative share of the evidence (not a probability)
    margin: float = 0.0               # its lead over the runner-up
    candidates: list = field(default_factory=list)   # [(family, relative score)], best first
    sources: tuple = ()               # evidence used
    reason: str = ""
    evidence: list = field(default_factory=list)     # [(source, its best family, share, weight)]
    disagreement: bool = False        # reliable sources pointed at different families
    matched: str = ""                 # the release metadata came from, if any
    suggested_subgenre: str = ""      # best subgenre guess, even when not assigned
    method: str = "legacy"            # "calibrated" or "legacy"
    threshold: Optional[float] = None  # confidence needed to assign automatically (calibrated)

    @property
    def confident(self) -> bool:
        return self.status == CONFIDENT

    @property
    def suggested(self) -> bool:
        return self.status == SUGGESTED

    @property
    def label(self) -> str:
        """The most specific supported name: subgenre, else family."""
        return self.subgenre or self.family


@dataclass
class Opinion:
    """One source's view of one track."""
    source: str
    shares: dict                      # family -> relative share (adds up to 1)
    quality: float                    # 0-1: how usable this source's input is for this track
    subs: dict = field(default_factory=dict)   # family -> {subgenre or "": share}
    note: str = ""

    @property
    def top(self) -> str:
        return max(self.shares, key=lambda f: (self.shares[f], f)) if self.shares else ""


def tempo_fits(family: str, bpm: float) -> Optional[bool]:
    """Whether a BPM fits the family's usual tempo (half / double speed allowed); None = can't say."""
    span = TYPICAL_BPM.get(family)
    if span is None or bpm <= 0:
        return None
    lo, hi = span
    return any(lo - 3 <= b <= hi + 3 for b in (bpm, bpm * 2, bpm / 2))


def _tempo_vector(bpm: float) -> np.ndarray:
    """BPM on a circle where 87 and 174 (half / double speed) are the same point."""
    if bpm <= 0:
        return np.zeros(2)
    angle = 2 * np.pi * np.log2(bpm)
    return np.array([np.cos(angle), np.sin(angle)])


def _rule_guess(bpm: float, z: dict[str, float], electronic_only: bool) -> Optional[str]:
    """Rough style rules from BPM plus sound. Only ever used as a *candidate*, never confident."""
    if bpm <= 0:
        return None
    folded = bpm
    while folded >= 190:
        folded /= 2
    if 160 <= folded < 190 or 80 <= folded < 92 and z.get("lows", 0) > 0.5:
        return "Drum & Bass"
    if 136 <= folded <= 155:
        if z.get("flux", 0) < -0.8 and z.get("lows", 0) > 0.8:
            return "Dubstep"
        if z.get("highs", 0) > 1.0 and z.get("brightness", 0) < 0:
            return "Techno"
        if z.get("brightness", 0) > 1.0 and z.get("flux", 0) < 0:
            return "Trance"
        return "EDM"
    if 118 <= folded < 136:
        return "House"
    if electronic_only:
        return "EDM"
    if 60 <= folded < 118:
        return "Hip Hop" if z.get("lows", 0) > -0.3 else "Downtempo"
    return None


def _model_evidence(result) -> Optional[dict]:
    """Family / subgenre shares and strength from a genre_model.ModelResult, or None."""
    model = getattr(result, "model", "yamnet")
    per_segment, sub_votes = [], defaultdict(lambda: defaultdict(float))
    for scores in result.genre_scores:
        fam: dict[str, float] = defaultdict(float)
        for name, value in scores.items():
            for family, sub, weight in model_label(model, name):
                fam[family] += value * weight
                sub_votes[family][sub] += value * weight
        per_segment.append(fam)
    if not per_segment:
        return None
    families = sorted({f for seg in per_segment for f in seg})
    mean = {f: float(np.mean([seg.get(f, 0.0) for seg in per_segment])) for f in families}
    total = sum(mean.values())
    music = float(np.mean(result.music)) if result.music else 0.0
    if model == "yamnet":
        electronic = float(np.mean([max(s.get(n, 0.0) for n in AUDIOSET_ELECTRONIC)
                                    for s in result.genre_scores]))
    else:
        electronic = sum(v for f, v in mean.items() if f in ELECTRONIC)
    if total < MIN_MODEL_EVIDENCE and music < MIN_MUSIC:
        return None
    shares = {f: v / total for f, v in mean.items()} if total > 0 else {}
    top = max(shares, key=lambda f: (shares[f], f)) if shares else ""
    agree = (sum(1 for seg in per_segment if seg and max(seg, key=lambda f: (seg[f], f)) == top)
             / len(per_segment)) if top else 0.0
    subs = {}
    for family, votes in sub_votes.items():
        s = sum(votes.values())
        if s > 0:
            subs[family] = {k: v / s for k, v in votes.items()}
    return {"shares": shares, "strength": total, "agreement": agree, "electronic": electronic,
            "music": music, "subs": subs, "model": model}


@dataclass
class _Example:
    group: str
    family: str
    subgenre: str
    embedding: Optional[np.ndarray]
    sound: Optional[np.ndarray]


def _groups(tracks: list[Track]) -> dict[str, str]:
    """Track id -> recording group: copies of the same recording share one."""
    import organizer
    group = {t.id: t.id for t in tracks}
    for copies in organizer.find_duplicates(tracks):
        for t in copies:
            group[t.id] = copies[0].id
    return group


_BRACKETS = re.compile(r"[\(\[].*?[\)\]]")


def _base_title(name: str) -> str:
    """The title without artist prefix, (Remix) / [Edit] parts and punctuation."""
    import organizer
    text = _BRACKETS.sub(" ", name)
    if " - " in text:
        text = text.split(" - ", 1)[1]
    return organizer._norm_title(text)


_MIX_WORDS = re.compile(r"\b(?:remix|rmx|edit|mix|bootleg|flip|vip|rework|remake|refix|"
                        r"version|extended|mashup|cover|premiere)\b")
_FEATURING = re.compile(r"\b(?:ft|feat|featuring)\b.*$")
_PREFIX = re.compile(r"^[^|]*\|\s*")          # "PREMIERE | Artist - Title"


def _title_keys(name: str) -> set[str]:
    """Every name part that could identify the song: title, artists, mashup parts.

    Downloaded names come as "Artist - Title", "Title - Artist",
    "12. Remixer - Artist - Title - Remixer Remix" and so on, and which part
    is the title can't be told reliably. So every part counts, except mix
    descriptors ("Ori Blake Remix"), track numbers, download junk, "PREMIERE |"
    prefixes and "ft." credits. Mashups ("A x B") give each part.
    """
    import organizer
    text = organizer._TITLE_JUNK.sub(" ", name)
    text = _PREFIX.sub("", text) if "|" in text else text
    text = _BRACKETS.sub(" ", text)
    keys = set()
    for seg in text.split(" - "):
        seg = _FEATURING.sub("", organizer._norm_title(seg)).strip()
        parts = [p.strip() for p in re.split(r"\s+x\s+", seg)]
        for part in parts:
            if len(part) >= 4 and not _MIX_WORDS.search(part):
                keys.add(part)
    return keys


def version_groups(tracks: list[Track]) -> dict[str, str]:
    """Track id -> group of related recordings: copies, versions, mashups, shared artists.

    Two recordings are related if their names share any part (see
    _title_keys): remixes and edits of a title, mashups and their sources,
    and, because "Artist - Title" can't be told from "Title - Artist", songs
    whose names share an artist. Used to hold related recordings out together
    when measuring accuracy, so a remix can't give away the original's genre;
    grouping too much only makes the measurement stricter. (Not used for "copy" evidence:
    a D&B remix of a house track is still D&B.)
    """
    group = _groups(tracks)
    parent = {g: g for g in group.values()}

    def root(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    by_key: dict[str, str] = {}
    for t in sorted(tracks, key=lambda t: t.id):
        g = group[t.id]
        for key in sorted(_title_keys(t.name)):
            if key in by_key:
                parent[root(g)] = root(by_key[key])
            else:
                by_key[key] = g
    return {t.id: root(group[t.id]) for t in tracks}


def _zscores(rows: list[dict], keys) -> dict[str, np.ndarray]:
    out = {}
    for k in keys:
        col = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        known = ~np.isnan(col)
        z = np.zeros(len(col))
        if known.sum() > 1 and col[known].std() > 0:
            z[known] = (col[known] - col[known].mean()) / col[known].std()
        out[k] = z
    return out


SOUND = ("lows", "highs", "brightness", "flux", "loudness")


def _library_votes(examples: list[_Example], query, kind: str, balanced: bool = False):
    """(family shares, {family: neighbours}, {family: Counter(subgenre)}) or None."""
    usable = [e for e in examples if getattr(e, kind) is not None]
    if query is None or not usable:
        return None
    if kind == "embedding":
        sims = np.array([float(np.dot(query, e.embedding)) for e in usable])
        weights = np.maximum(sims, 0.0) ** 2
        order = np.lexsort(([e.group for e in usable], -sims))
    else:
        dist = np.array([float(np.linalg.norm(query - e.sound)) for e in usable])
        weights = 1 / (dist + 0.25)
        order = np.lexsort(([e.group for e in usable], dist))
    size = Counter(e.family for e in usable)
    near = order[:min(K_NEIGHBOURS, len(usable))]
    votes: dict[str, float] = defaultdict(float)
    groups: dict[str, int] = defaultdict(int)
    subs: dict[str, Counter] = defaultdict(Counter)
    for j in near:
        e = usable[j]
        votes[e.family] += weights[j] / (size[e.family] if balanced else 1)
        groups[e.family] += 1
        subs[e.family][e.subgenre] += 1
    total = sum(votes.values())
    if total <= 0:
        return None
    return {f: v / total for f, v in votes.items()}, groups, subs


# --------------------------------------------------------------------------- #
# The library classifier: logistic regression on the model's embeddings
# --------------------------------------------------------------------------- #

PCA_DIMS = 24


class LibraryClassifier:
    """Class-balanced multinomial logistic regression on the audio model's embeddings.

    Features: the embeddings projected onto their top PCA_DIMS principal
    components, fitted on every analysed track (labels aren't used for that,
    so nothing leaks), scaled to unit average length. Trained only on families with at least MIN_GROUPS
    labelled recordings. Deterministic (zero start, fixed steps). Its output
    is relative scores over the families it knows, not probabilities.
    """

    def __init__(self, embeddings: dict[str, np.ndarray]):
        ids = sorted(embeddings)
        self.dims = 0
        if len(ids) < 3:
            return
        x = np.array([embeddings[i] for i in ids], dtype=np.float64)
        self.mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.dims = int(min(PCA_DIMS, len(ids) - 1, int((s > 1e-8).sum())))
        self.components = vt[:self.dims]
        proj = (x - self.mean) @ self.components.T
        # One scale for all components (not per component: whitening would blow
        # the near-empty directions up to the size of the real ones).
        self.scale = float(np.sqrt((proj ** 2).sum(axis=1).mean())) + 1e-9

    def features(self, emb: np.ndarray) -> np.ndarray:
        return ((emb - self.mean) @ self.components.T) / self.scale

    def fit(self, xs: list[np.ndarray], ys: list[str], iters: int = 300, rate: float = 0.5,
            l2: float = 0.05):
        """Weights for these examples (or None if there aren't two usable families)."""
        classes = sorted(c for c, n in Counter(ys).items() if n >= MIN_GROUPS)
        if self.dims == 0 or len(classes) < 2:
            return None
        keep = [i for i, y in enumerate(ys) if y in classes]
        x = np.array([xs[i] for i in keep])
        y = np.array([classes.index(ys[i]) for i in keep])
        n, k = len(y), len(classes)
        onehot = np.eye(k)[y]
        weight = (n / (k * np.bincount(y, minlength=k)))[y][:, None]
        w, b = np.zeros((x.shape[1], k)), np.zeros(k)
        for _ in range(iters):
            logits = x @ w + b
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            g = (p - onehot) * weight
            w -= rate * (x.T @ g / n + l2 * w)
            b -= rate * g.mean(axis=0)
        return classes, w, b

    @staticmethod
    def shares(model, x: np.ndarray) -> dict[str, float]:
        classes, w, b = model
        logits = x @ w + b
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {c: float(v) for c, v in zip(classes, p)}


# --------------------------------------------------------------------------- #
# Trusted labels
# --------------------------------------------------------------------------- #

def training_labels(tracks: list[Track], labels: Optional[dict] = None,
                    include_tags: bool = True) -> dict[str, str]:
    """{track id: genre} that may be learned from and measured against.

    Genres the user chose in the review window ("corrected"), plus (unless
    include_tags is False) specific rekordbox tags, except ones this app wrote
    from a suggestion the user saved unchanged ("accepted") that are still
    unchanged.
    """
    labels = labels or {}
    out = {}
    for t in tracks:
        if is_vague(t.genre):
            continue
        rec = labels.get(t.id)
        same = rec and _norm_genre(rec.get("genre", "")) == _norm_genre(t.genre)
        if same and rec.get("source") == "accepted":
            continue
        if not include_tags and not (same and rec.get("source") == "corrected"):
            continue
        out[t.id] = t.genre
    return out


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

class _Context:
    """Everything predict_genres needs about the library, built once."""

    def __init__(self, tracks, features, model_results, trusted, metadata, use,
                 method: str = "legacy"):
        self.tracks = tracks
        self.by_id = {t.id: t for t in tracks}
        self.features = features
        self.model_results = model_results
        self.metadata = metadata
        self.use = use
        self.group = _groups(tracks)
        rows = [features.get(t.id, {}) for t in tracks]
        self.z = _zscores(rows, SOUND)
        self.index = {t.id: i for i, t in enumerate(tracks)}

        embedded = sorted(i for i, r in model_results.items()
                          if r.embedding is not None and i in self.by_id)
        self.centre = (np.mean([model_results[i].embedding for i in embedded], axis=0)
                       if embedded else None)
        self.embeddings = {i: v for i in embedded
                           if (v := self._centred(model_results[i].embedding)) is not None}
        self.classifier = LibraryClassifier(self.embeddings) if "library" in use else None

        members: dict[str, list[str]] = defaultdict(list)
        for tid in sorted(trusted):
            members[self.group[tid]].append(tid)
        self.labelled_tag = {g: Counter(trusted[i] for i in ids).most_common(1)[0][0]
                             for g, ids in members.items()}
        self.examples: list[_Example] = []
        for g, ids in sorted(members.items()):
            family, sub = subgenre_of(self.labelled_tag[g])
            if not known_family(family):
                continue        # e.g. a record label typed into the genre field
            embs = [self.embeddings[i] for i in ids if i in self.embeddings]
            emb = None
            if embs:
                emb = np.mean(embs, axis=0)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
            sounds = [v for v in (self.sound_vector(i) for i in ids) if v is not None]
            self.examples.append(_Example(g, family, sub, emb,
                                          np.mean(sounds, axis=0) if sounds else None))
        self.per_family = Counter(e.family for e in self.examples)
        self.comparable = sum(1 for n in self.per_family.values() if n >= MIN_GROUPS) >= 2
        self._fits: dict = {}
        self.reliability = (reliability(self) if method == "legacy"
                            and any(s in use for s in SOURCES) else {})

    def _centred(self, emb) -> Optional[np.ndarray]:
        if emb is None or self.centre is None:
            return None
        v = emb - self.centre
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    def sound_vector(self, tid: str) -> Optional[np.ndarray]:
        if tid not in self.features:
            return None
        i = self.index[tid]
        return np.concatenate([_tempo_vector(self.by_id[tid].bpm), [self.z[k][i] for k in SOUND]])

    def library_fit(self, exclude=None):
        """The library classifier trained without some recording groups (cached)."""
        exclude = _as_set(exclude)
        if exclude in self._fits:
            return self._fits[exclude]
        fit = None
        if self.classifier is not None and self.classifier.dims:
            usable = [e for e in self.examples if e.embedding is not None
                      and e.group not in exclude]
            if usable:
                xs = [self.classifier.features(e.embedding) for e in usable]
                fit = self.classifier.fit(xs, [e.family for e in usable])
        self._fits[exclude] = fit
        return fit

    def opinions(self, tid: str, exclude=None, raw: bool = False
                 ) -> tuple[list[Opinion], list[str]]:
        """Every source's opinion of one track (learned ones without the groups in `exclude`).

        raw: the library classifier's own scores over the families it knows,
        not mixed with the model's (the calibrated fusion weighs them itself).
        """
        exclude = _as_set(exclude)
        ops, notes = [], []
        result = self.model_results.get(tid)
        m = _model_evidence(result) if result is not None and "model" in self.use else None
        model_op = None
        if m is not None:
            clear = CLEAR_EVIDENCE.get(m["model"], 0.3)
            quality = min(1.0, m["strength"] / clear) * min(1.0, m["agreement"] / MIN_AGREEMENT)
            if m["strength"] < MIN_MODEL_EVIDENCE:
                quality = 0.0
                notes.append("the audio model hears music but no clear genre")
            elif m["agreement"] < MIN_AGREEMENT:
                notes.append("different parts of the track point to different genres")
            if m["shares"]:
                model_op = Opinion("model", m["shares"], quality, m["subs"])
                ops.append(model_op)

        emb = self.embeddings.get(tid)
        examples = [e for e in self.examples if e.group not in exclude]
        if emb is not None and "library" in self.use:
            fit = self.library_fit(exclude)
            subs = self._library_subs(examples, emb, "embedding")
            if fit is not None:
                shares = LibraryClassifier.shares(fit, self.classifier.features(emb))
                # It can only answer with families it was trained on. So, as a
                # mixture: the audio model's share for those families is split
                # the classifier's way, and for families it doesn't know it
                # defers to the model (a D&B track mustn't be forced into
                # Dubstep because too few D&B songs are tagged to train on).
                if not raw and model_op is not None and model_op.quality > 0:
                    known = sum(model_op.shares.get(f, 0.0) for f in shares)
                    mixed = {f: known * v for f, v in shares.items()}
                    for f, v in model_op.shares.items():
                        if f not in shares:
                            mixed[f] = v
                    shares = mixed
                ops.append(Opinion("library", shares, 1.0, subs))
            elif subs:          # subgenre evidence only (e.g. one family tagged so far)
                ops.append(Opinion("library", {}, 1.0, subs))
        elif m is None and "sound" in self.use:
            sound = self.sound_vector(tid)
            votes = _library_votes(examples, sound, "sound", balanced=True)
            if votes:
                comparable = sum(1 for n in Counter(e.family for e in examples).values()
                                 if n >= MIN_GROUPS) >= 2
                if comparable:
                    ops.append(Opinion("sound", votes[0], 1.0,
                                       self._library_subs(examples, sound, "sound")))
                else:
                    notes.append("too few tagged songs to compare with")

        meta = self.metadata.get(tid)
        if meta is not None and "metadata" in self.use:
            op = metadata_opinion(meta)
            if op is not None:
                ops.append(op)
        return ops, notes

    @staticmethod
    def _library_subs(examples, query, kind) -> dict:
        """family -> {subgenre: share} from the nearest labelled songs of each family.

        Only songs tagged with a subgenre take part: a plain family tag
        ("House") says the subgenre wasn't given, not that there is none, so
        it isn't a vote against Tech House.
        """
        out = {}
        for family in sorted({e.family for e in examples}):
            own = [e for e in examples if e.family == family and e.subgenre]
            if len({e.subgenre for e in own}) < 2:
                continue        # fewer than two subgenres tagged: nothing to compare
            if kind == "embedding":
                shares = _centroid_shares(own, query)
                if shares:
                    out[family] = shares
                continue
            votes = _library_votes(own, query, kind)
            if votes is None:
                continue
            _, groups, subs = votes
            counts = subs.get(family, Counter())
            n = sum(counts.values())
            sub, top = counts.most_common(1)[0] if counts else ("", 0)
            if n and top >= 2 and top / n >= SUBGENRE_SHARE:
                out[family] = {s: c / n for s, c in counts.items()}
        return out


SUB_TEMPERATURE = 0.05      # cosine-similarity scale for subgenre centroid shares
SUB_MIN_EXAMPLES = 2        # tagged recordings a subgenre needs to get a centroid


def _centroid_shares(examples: list, query) -> dict:
    """{subgenre: share} by similarity to each subgenre's average embedding.

    Works with very few examples (2-3 per subgenre), where "3 of the 5
    nearest songs agree" can't happen. Subgenres with fewer than
    SUB_MIN_EXAMPLES tagged recordings take no part.
    """
    by_sub = defaultdict(list)
    for e in examples:
        if e.embedding is not None:
            by_sub[e.subgenre].append(e.embedding)
    by_sub = {k: v for k, v in by_sub.items() if len(v) >= SUB_MIN_EXAMPLES}
    if len(by_sub) < 2 or query is None:
        return {}
    names = sorted(by_sub)
    sims = []
    for k in names:
        c = np.mean(by_sub[k], axis=0)
        sims.append(float(np.dot(query, c) / (np.linalg.norm(c) + 1e-9)))
    z = np.array(sims) / SUB_TEMPERATURE
    p = np.exp(z - z.max())
    p /= p.sum()
    return {k: float(v) for k, v in zip(names, p)}


def _as_set(exclude) -> frozenset:
    if exclude is None:
        return frozenset()
    if isinstance(exclude, str):
        return frozenset([exclude])
    return frozenset(exclude)


def metadata_opinion(meta) -> Optional[Opinion]:
    """A metadata.MetadataResult -> Opinion (or None if its genres say nothing)."""
    shares: dict[str, float] = defaultdict(float)
    subs: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for name, weight in meta.genres:
        if is_vague(name) or weight <= 0:
            continue
        family, sub = subgenre_of(name)
        if not known_family(family):
            continue
        shares[family] += weight
        subs[family][sub] += weight
    total = sum(shares.values())
    if total <= 0 or meta.quality <= 0:
        return None
    sub_shares = {f: {s: v / sum(d.values()) for s, v in d.items()} for f, d in subs.items()}
    return Opinion("metadata", {f: v / total for f, v in shares.items()}, float(meta.quality),
                   sub_shares, f"matched {meta.source}: {meta.describe}")


def reliability(ctx: _Context) -> dict[str, float]:
    """How often each source's best family was right on the labelled recordings.

    Sources that learn from labels are asked with the recording held out. A
    Beta(1, 1) prior keeps it at 0.5 when there's nothing to go on. Only the
    first MAX_RELIABILITY_CHECKS recordings (by group id) are checked.
    """
    right: Counter = Counter()
    seen: Counter = Counter()
    for e in ctx.examples[:MAX_RELIABILITY_CHECKS]:
        probe = next((i for i in sorted(ctx.by_id) if ctx.group[i] == e.group
                      and (i in ctx.model_results or i in ctx.features or i in ctx.metadata)), None)
        if probe is None:
            continue
        ops, _ = ctx.opinions(probe, exclude=e.group)
        for op in ops:
            if op.quality <= 0 or not op.shares:
                continue
            seen[op.source] += 1
            right[op.source] += op.top == e.family
    a, b = RELIABILITY_PRIOR
    return {s: (right[s] + a) / (seen[s] + a + b) for s in SOURCES}


def _fuse(t: Track, ops: list[Opinion], notes: list[str], reliab: dict) -> GenrePrediction:
    evidence: dict[str, float] = defaultdict(float)
    weights = []
    for op in ops:
        w = reliab.get(op.source, 0.5) * op.quality if op.shares else 0.0
        weights.append(w)
        if w <= 0:
            continue
        for f, s in op.shares.items():
            evidence[f] += w * s
    total_w = sum(weights)
    sources = [op.source for op, w in zip(ops, weights) if w > 0]
    shown = [(op.source, op.top, round(op.shares.get(op.top, 0.0), 3), round(w, 3))
             for op, w in zip(ops, weights) if op.shares]
    matched = next((op.note for op in ops if op.source == "metadata" and op.note), "")

    electronic_tag = _norm_genre(t.genre) in ELECTRONIC_VAGUE
    if not evidence:
        guess = _rule_guess(t.bpm, {}, electronic_tag)
        candidates = [(guess, 0.0)] if guess else []
        if notes:
            reason = "; ".join(notes)
            reason = reason[:1].upper() + reason[1:] + "."
        elif guess:
            reason = "Only the tempo is known, which isn't enough to tell the genre."
        else:
            reason = "Nothing to go on: no audio analysis or BPM."
        return GenrePrediction(UNCERTAIN, guess or "", "", parent_of(guess or ""), 0.0, 0.0,
                               candidates, tuple(sources) + (("tempo",) if guess else ()),
                               reason, shown, matched=matched)

    # Secondary evidence can only lower a family's score.
    for f in list(evidence):
        if tempo_fits(f, t.bpm) is False:
            evidence[f] *= TEMPO_MISMATCH
        if electronic_tag and f not in ELECTRONIC:
            evidence[f] *= TEMPO_MISMATCH
    if t.bpm > 0:
        sources.append("tempo")
    if electronic_tag:
        sources.append("tag")

    total = sum(evidence.values())
    ranked = sorted(((f, v / total) for f, v in evidence.items() if v > 0),
                    key=lambda fv: (-fv[1], fv[0]))
    family, score = ranked[0]
    margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    strong = [op for op, w in zip(ops, weights)
              if w >= DISAGREE_WEIGHT and op.shares.get(op.top, 0) >= 0.5]
    disagree = len({op.top for op in strong}) > 1
    ok = score >= MIN_SCORE and margin >= MIN_MARGIN and total_w >= MIN_WEIGHT and not disagree
    status = CONFIDENT if ok else UNCERTAIN
    if disagree:
        tops = sorted({f"{op.source} says {op.top}" for op in strong})
        notes.append("the sources disagree (" + ", ".join(tops) + ")")
    elif status == UNCERTAIN:
        if total_w < MIN_WEIGHT and score >= MIN_SCORE and margin >= MIN_MARGIN:
            notes.append("the evidence is weak")
        elif len(ranked) > 1 and margin < MIN_MARGIN:
            notes.append(f"{family} only narrowly leads {ranked[1][0]}")
        elif len(ranked) > 1:
            notes.append(f"{family} leads, but the evidence is spread over several genres "
                         f"(next: {ranked[1][0]})")
        else:
            notes.append("the evidence is weak")

    subgenre = ""
    if status == CONFIDENT:
        subgenre = _choose_subgenre(family, ops, reliab)
        if subgenre and "library" not in sources and any(
                op.source == "library" and family in op.subs for op in ops):
            sources.insert(1, "library")
    reason = "; ".join(notes)
    return GenrePrediction(status, family, subgenre, parent_of(family), round(score, 4),
                           round(margin, 4), [(f, round(s, 4)) for f, s in ranked[:5]],
                           tuple(dict.fromkeys(sources)),
                           reason[:1].upper() + reason[1:] + "." if reason else "",
                           shown, disagree, matched)


def _choose_subgenre(family: str, ops: list[Opinion], reliab: dict) -> str:
    """A subgenre for a confident family, or "" when the evidence isn't clear.

    Only sources that name subgenres take part. Subgenres are compared with
    each other (a plain family label isn't evidence against them); the winner
    needs SUB_MIN of the subgenre votes, a SUB_MARGIN lead, and SUB_SUPPORT of
    the weight of the sources taking part.
    """
    votes: dict[str, float] = defaultdict(float)
    weight = 0.0
    for op in ops:
        subs = {k: v for k, v in (op.subs.get(family) or {}).items() if k}
        if not subs:
            continue
        w = reliab.get(op.source, 0.5) * op.quality
        weight += w
        for sub, share in subs.items():
            votes[sub] += w * share
    total = sum(votes.values())
    if total <= 0 or weight <= 0:
        return ""
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    sub, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if top / total < SUB_MIN or (top - second) / total < SUB_MARGIN or top / weight < SUB_SUPPORT:
        return ""
    return sub


# --------------------------------------------------------------------------- #
# Calibrated fusion (the default): see fusion.py
# --------------------------------------------------------------------------- #

FOLDS = 5
EVIDENCE_SOURCES = ("model", "library", "sound", "metadata")   # tempo / tag only support
STRONG_SHARE = 0.5          # a source this sure of another family means the sources disagree
SHOWN_CANDIDATE = 0.02      # candidates below this confidence aren't listed
METHOD = "calibrated"       # or "legacy" (the earlier weighted average; kept for comparison)
_NEUTRAL, _AGAINST = 1.0, 0.3679     # tempo / tag shares: e^0 and e^-1


def _trainable_metadata(meta):
    """The part of a metadata match that may be learned from.

    Discogs data is only used to predict, never to fit weights: Discogs' terms
    restrict using its content to develop machine-learning software and limit
    how long it may be kept (see metadata.py).
    """
    if meta is None:
        return None
    parts = getattr(meta, "parts", None) or [meta]
    keep = [r for r in parts if "Discogs" not in r.source]
    if not keep:
        return None
    if len(keep) == len(parts):
        return meta
    import metadata as md
    return md.merge(keep)


class _Calibrated:
    """Log-linear fusion fitted on out-of-fold opinions of the labelled songs."""

    def __init__(self, ctx: _Context):
        import fusion
        self.ctx = ctx
        self.fusion_mod = fusion
        related = version_groups(ctx.tracks)
        self.items = []                 # (example, probe track id)
        for e in ctx.examples:
            probe = next((i for i in sorted(ctx.by_id) if ctx.group[i] == e.group
                          and (i in ctx.model_results or i in ctx.features
                               or i in ctx.metadata)), None)
            if probe is not None:
                self.items.append((e, probe))
        keys = [related[p] for _, p in self.items]
        k = min(FOLDS, len(set(keys)))
        self.rows = [None] * len(self.items)
        self.ops = [None] * len(self.items)
        self.fold = [0] * len(self.items)
        if k >= 2:
            fold_of = fusion.folds(keys, k)
            self.fold = [fold_of[key] for key in keys]
            for f in range(k):
                members = [i for i in range(len(self.items)) if self.fold[i] == f]
                hidden = frozenset(self.items[i][0].group for i in members)
                for i in members:
                    probe = self.items[i][1]
                    ops, _ = ctx.opinions(probe, exclude=hidden, raw=True)
                    self.ops[i] = ops
                    self.rows[i] = self.row(ctx.by_id[probe], ops, train=True)
        used = [i for i in range(len(self.items)) if self.rows[i] is not None]
        self.fusion = fusion.build([self.rows[i] for i in used],
                                   [self.items[i][0].family for i in used],
                                   extra_classes=[f for f, _ in FAMILIES])
        self.threshold: Optional[float] = fusion.UNFITTED_THRESHOLD
        self.checked = 0                # held-out checks behind the threshold
        self.sub_ok: dict[str, bool] = {}
        self.family_ok: Optional[dict[str, bool]] = None    # None: not measured (unfitted)
        self.checks: list = []          # held-out (family guessed, confidence, right, group)
        if not self.fusion.fitted:
            return
        conf, right, guessed, held = [], [], [], []
        for f in range(k):
            train = [i for i in used if self.fold[i] != f]
            test = [i for i in used if self.fold[i] == f]
            if not test:
                continue
            inner = fusion.build([self.rows[i] for i in train],
                                 [self.items[i][0].family for i in train],
                                 extra_classes=self.fusion.classes)
            for i, probs in zip(test, inner.predict([self.rows[i] for i in test])):
                fam = max(probs, key=lambda c: (probs[c], c))
                conf.append(probs[fam])
                right.append(fam == self.items[i][0].family)
                guessed.append(fam)
                held.append(keys[i])        # its related-recordings group
        self.checked = len(conf)
        self.checks = list(zip(guessed, conf, right, held))
        self.threshold = fusion.choose_threshold(conf, right)
        self.family_ok = fusion.family_reliable(guessed, conf, right, self.threshold, held)
        # Subgenres are named automatically only for families where the
        # held-out subgenre suggestions proved accurate enough.
        tried: Counter = Counter()
        wrong: Counter = Counter()
        for i in used:
            e = self.items[i][0]
            if not e.subgenre:
                continue
            guess = _choose_subgenre(e.family, self.ops[i], {})
            if guess:
                tried[e.family] += 1
                wrong[e.family] += guess != e.subgenre
        self.sub_ok = {f: n >= fusion.MIN_AUTO_CHECKS
                       and fusion.error_upper_bound(wrong[f], n) <= fusion.AUTO_MAX_ERROR
                       for f, n in tried.items()}

    def family_counts(self) -> dict[str, tuple[int, int]]:
        """{family: (held-out checks at the threshold, wrong)}, related recordings once."""
        if not self.checks:
            return {}
        g, c, r, h = (list(x) for x in zip(*self.checks))
        return self.fusion_mod.family_checks(g, c, r, self.threshold, h)

    def row(self, t: Track, ops: list[Opinion], train: bool = False):
        r = self.fusion_mod.Row(partial=("library", "sound"))
        for op in ops:
            if not op.shares or (op.source == "model" and op.quality <= 0):
                continue
            if op.source == "metadata":
                if train:
                    meta = _trainable_metadata(self.ctx.metadata.get(t.id))
                    op = metadata_opinion(meta) if meta is not None else None
                    if op is None:
                        continue
                r.strength["metadata"] = op.quality
            r.shares[op.source] = dict(op.shares)
        if t.bpm > 0:
            fits = {f: (_AGAINST if tempo_fits(f, t.bpm) is False else _NEUTRAL)
                    for f, _ in FAMILIES}
            if any(v < _NEUTRAL for v in fits.values()):
                r.shares["tempo"] = fits
        if _norm_genre(t.genre) in ELECTRONIC_VAGUE:
            r.shares["tag"] = {f: (_NEUTRAL if f in ELECTRONIC else _AGAINST)
                               for f, _ in FAMILIES}
        return r

    def predict(self, t: Track) -> GenrePrediction:
        ops, notes = self.ctx.opinions(t.id, raw=True)
        row = self.row(t, ops)
        matched = next((op.note for op in ops if op.source == "metadata" and op.note), "")
        present = [s for s in EVIDENCE_SOURCES if s in row.shares]
        if not present:
            guess = _rule_guess(t.bpm, {}, _norm_genre(t.genre) in ELECTRONIC_VAGUE)
            if notes:
                reason = "; ".join(notes)
            elif guess:
                reason = "only the tempo is known, which isn't enough to tell the genre"
            else:
                reason = "nothing to go on: no audio analysis or BPM"
            return GenrePrediction(UNCERTAIN, guess or "", "", parent_of(guess or ""), 0.0, 0.0,
                                   [(guess, 0.0)] if guess else [],
                                   ("tempo",) if guess else (),
                                   reason[:1].upper() + reason[1:] + ".",
                                   matched=matched, method="calibrated")
        probs = self.fusion.predict([row])[0]
        ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
        family, conf = ranked[0]
        margin = conf - (ranked[1][1] if len(ranked) > 1 else 0.0)
        evidence, against = [], []
        for s in present:
            shares = row.shares[s]
            top = max(shares, key=lambda f: (shares[f], f))
            strength = row.strength.get(s, 1.0)
            evidence.append((s, top, round(shares[top], 3),
                             round(self.fusion.weight(s, family) * strength, 3)))
            if (top != family and shares[top] >= STRONG_SHARE and strength >= 0.5
                    and self.fusion.weight(s, top) > 0.1):
                against.append(f"{s} says {top}")
        disagree = bool(against)
        # Unfitted (too few tagged songs): nothing is measured, so nothing is automatic.
        family_ok = bool(self.family_ok and self.family_ok.get(family))
        auto = (self.threshold is not None and conf >= self.threshold and not disagree
                and family_ok)
        suggest = not auto and not disagree and conf >= (self.threshold or SUGGEST_MIN)
        if disagree:
            notes.append(f"the sources disagree ({', '.join(sorted(against))}; "
                         f"combined: {family})")
        elif self.threshold is None:
            notes.append("your tagged songs don't yet show the guesses are accurate enough "
                         "to tag automatically")
        elif conf < self.threshold:
            notes.append(f"confidence {conf:.0%} is below the {self.threshold:.0%} needed "
                         "to tag automatically")
        elif self.family_ok is None:
            notes.append("there are too few tagged songs to measure how accurate guesses "
                         "are, so it's only a suggestion")
        elif not family_ok:
            n, wrong = self.family_counts().get(family, (0, 0))
            bound = self.fusion_mod.error_upper_bound(wrong, n)
            notes.append(f"{family} guesses haven't yet proved reliable enough to tag "
                         f"automatically ({n} held-out check{'s' if n != 1 else ''}, "
                         f"{wrong} wrong: the error rate could be as high as {bound:.0%}, "
                         f"above the {self.fusion_mod.AUTO_MAX_ERROR:.0%} allowed), so it's "
                         "only a suggestion")
        level = conf if self.threshold is None else min(conf, self.threshold)
        seen = [ok for f, c, ok, _ in self.checks if f == family and c >= level]
        if seen:
            how = ("at least this confident" if level == conf
                   else f"at or above the {level:.0%} threshold")
            notes.append(f"held-out {family} guesses {how} were right "
                         f"{sum(seen)} of {len(seen)} times")
        suggested = _choose_subgenre(family, ops, {})
        sub = suggested if auto and self.sub_ok.get(family) else ""
        sources = present + [s for s in ("tempo", "tag") if s in row.shares]
        reason = "; ".join(notes)
        status = CONFIDENT if auto else SUGGESTED if suggest else UNCERTAIN
        return GenrePrediction(status, family, sub, parent_of(family),
                               round(conf, 4), round(margin, 4),
                               [(f, round(v, 4)) for f, v in ranked[:5]
                                if v >= SHOWN_CANDIDATE or f == family],
                               tuple(sources),
                               reason[:1].upper() + reason[1:] + "." if reason else "",
                               evidence, disagree, matched, suggested, "calibrated",
                               self.threshold)


def predict_genres(tracks: list[Track], features: dict[str, dict],
                   log: Optional[Callable[[str], None]] = print,
                   model_results: Optional[dict] = None, labels: Optional[dict] = None,
                   targets: Optional[set] = None, metadata: Optional[dict] = None,
                   use: Optional[set] = None, include_tags: bool = True,
                   method: Optional[str] = None) -> dict[str, GenrePrediction]:
    """A GenrePrediction for every untagged / vaguely tagged track (or the given targets).

    use: the evidence sources allowed (default: all of SOURCES).
    method: "calibrated" (default, see fusion.py) or "legacy".
    """
    method = method or METHOD
    model_results = model_results or {}
    use = set(SOURCES if use is None else use)
    trusted = training_labels(tracks, labels, include_tags)
    if targets is None:
        targets = {t.id for t in tracks if is_vague(t.genre)}
    targets -= set(trusted)
    if not targets:
        return {}
    ctx = _Context(tracks, features, model_results, trusted, metadata or {}, use, method)
    calibrated = _Calibrated(ctx) if method == "calibrated" else None

    out: dict[str, GenrePrediction] = {}
    for tid in sorted(targets):
        t = ctx.by_id[tid]
        g = ctx.group[tid]
        if "copy" in use and g in ctx.labelled_tag:     # a copy of a labelled recording
            # Only suggested: "a copy" is matched by name and length, which can
            # be wrong, and it has no measured accuracy of its own.
            family, sub = subgenre_of(ctx.labelled_tag[g])
            out[tid] = GenrePrediction(SUGGESTED, family, sub, parent_of(family), 1.0, 1.0,
                                       [(family, 1.0)], ("copy",),
                                       "Another copy of this recording (same artist, title "
                                       f"and length) is tagged '{ctx.labelled_tag[g]}'; "
                                       "approve it to use the same genre.")
            continue
        if calibrated is not None:
            out[tid] = calibrated.predict(t)
            continue
        ops, notes = ctx.opinions(tid)
        out[tid] = _fuse(t, ops, notes, ctx.reliability)

    if log:
        confident = [p for p in out.values() if p.confident]
        by_source = Counter(next((s for s in SOURCES if s in p.sources), "other")
                            for p in confident)
        how = {"copy": "from a tagged copy", "model": "from the audio model",
               "library": "from your tagged songs (by sound)",
               "sound": "learned from your tagged songs", "metadata": "from release info"}
        parts = [f"{n} {how.get(s, s)}" for s, n in sorted(by_source.items())]
        suggested = sum(1 for p in out.values() if p.suggested)
        unsure = len(out) - len(confident) - suggested
        log(f"Guessed genres for {len(confident)} of {len(out)} untagged/vague songs"
            + (f" ({', '.join(parts)})" if parts else "")
            + (f"; {suggested} suggested for you to approve" if suggested else "")
            + (f"; {unsure} need review" if unsure else "") + "."
            + ("" if model_results or ctx.per_family else
               " Tag more songs, or turn on the audio genre model, to let it learn."))
    return out


def auto_eligibility(tracks: list[Track], features: dict[str, dict],
                     model_results: Optional[dict] = None, labels: Optional[dict] = None,
                     metadata: Optional[dict] = None) -> tuple[Optional[float], list[tuple]]:
    """Why each genre may or may not be tagged automatically (calibrated method).

    Returns (threshold, [(family, held-out checks at the threshold (related
    recordings counted once), wrong,
    95% upper bound on their error rate, eligible)]), most checked first.
    The threshold is None when unfitted or when no level met the target.
    """
    import fusion
    trusted = training_labels(tracks, labels)
    ctx = _Context(tracks, features, model_results or {}, trusted, metadata or {},
                   set(SOURCES), "calibrated")
    cal = _Calibrated(ctx)
    if not cal.fusion.fitted:
        return None, []
    counts = cal.family_counts()
    rows = [(f, k, w, fusion.error_upper_bound(w, k), bool(cal.family_ok and cal.family_ok.get(f)))
            for f, (k, w) in sorted(counts.items(), key=lambda kv: (-kv[1][0], kv[0]))]
    return cal.threshold, rows


def guess_genres(tracks: list[Track], features: dict[str, dict],
                 log: Callable[[str], None] = print, model_results: Optional[dict] = None,
                 labels: Optional[dict] = None, metadata: Optional[dict] = None) -> dict[str, str]:
    """Confident genre guesses for untagged / vaguely tagged tracks: {id: subgenre or family}.

    Uncertain tracks are left out (see predict_genres for everything).
    """
    return confident_labels(predict_genres(tracks, features, log, model_results, labels,
                                           metadata=metadata))


def auto_allowed(p: Optional[GenrePrediction]) -> bool:
    """The one rule for using a guess without the user approving it.

    Only a calibrated prediction that passed every gate (threshold, sources
    agreeing, the genre's own held-out error bound) qualifies. Anything else
    (legacy scores, suggestions, missing data) fails closed.
    """
    return (p is not None and p.status == CONFIDENT and p.method == "calibrated"
            and bool(p.label))


def confident_labels(predictions: dict[str, GenrePrediction]) -> dict[str, str]:
    return {tid: p.label for tid, p in predictions.items() if auto_allowed(p)}


def effective_genre(t: Track, guesses: dict[str, str]) -> str:
    """The tag, unless it's missing or vague and a guess exists."""
    if t.id in guesses and is_vague(t.genre):
        return guesses[t.id]
    return t.genre


# --------------------------------------------------------------------------- #
# Labels the user has saved
# --------------------------------------------------------------------------- #

def labels_path() -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), "genre_labels.json")


def load_labels(path: Optional[str] = None) -> dict[str, dict]:
    """{track id: {"genre", "source": "corrected" | "accepted", "date", "original"}}."""
    try:
        with open(path or labels_path(), encoding="utf-8") as f:
            data = json.load(f)
        return dict(data.get("tracks", {})) if data.get("version") == 1 else {}
    except (OSError, ValueError, AttributeError):
        return {}


def record_labels(saved: dict[str, tuple], path: Optional[str] = None) -> None:
    """Remember genres saved from the review window: {track id: (genre, source[, original])}.

    source "corrected": the user chose it (a confirmed label, learned from).
    "accepted": the app's suggestion saved unchanged (not learned from while it
    stays as is). original: the rekordbox tag it replaced, kept for reference.
    """
    import datetime
    path = path or labels_path()
    labels = load_labels(path)
    today = datetime.date.today().isoformat()
    for tid, saved_as in saved.items():
        genre, source = saved_as[0], saved_as[1]
        record = {"genre": genre, "source": source, "date": today}
        if len(saved_as) > 2:
            record["original"] = saved_as[2]
        labels[tid] = record
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "tracks": labels}, f, indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class Evaluation:
    recordings: int                 # labelled recordings tested (copies count once)
    confident: int                  # ...that got a confident prediction
    correct_confident: int          # ...that were right
    correct_top: int                # recordings whose best candidate was right (confident or not)
    macro_f1: float                 # over families, confident predictions only
    confusion: dict = field(default_factory=dict)   # (true, predicted) -> count
    rows: list = field(default_factory=list)        # (track id, truth tag, GenrePrediction)
    skipped: int = 0                # truth labels that aren't a known genre

    @property
    def coverage(self) -> float:
        return self.confident / self.recordings if self.recordings else 0.0

    @property
    def precision(self) -> float:
        return self.correct_confident / self.confident if self.confident else 0.0

    def summary(self) -> str:
        return (f"{self.recordings} labelled recordings: {self.confident} confident "
                f"({self.coverage:.0%}), {self.correct_confident} of those right "
                f"({self.precision:.0%}); best candidate right for {self.correct_top} "
                f"({self.correct_top / max(self.recordings, 1):.0%}); macro-F1 {self.macro_f1:.2f}")


def macro_f1(confusion: dict) -> float:
    families = sorted({a for a, _ in confusion} | {b for _, b in confusion})
    f1s = []
    for f in families:
        tp = confusion.get((f, f), 0)
        fp = sum(n for (a, b), n in confusion.items() if b == f and a != f)
        fn = sum(n for (a, b), n in confusion.items() if a == f and b != f)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(tracks: list[Track], features: dict[str, dict],
             model_results: Optional[dict] = None, labels: Optional[dict] = None,
             metadata: Optional[dict] = None, use: Optional[set] = None,
             include_tags: bool = True, method: Optional[str] = None,
             truth: Optional[dict] = None, holdout: bool = False,
             exclude: Optional[set] = None) -> Evaluation:
    """Measure predictions against labels the predictor never sees.

    Default (leave-one-out): each labelled recording is hidden along with its
    copies and related versions (remixes / edits of the same title), then
    predicted from the rest; everything learned (classifier, fusion weights,
    threshold) is refitted without it.

    truth: {track id: genre} to measure against (default: the trusted tags).
    holdout: hide every truth recording (and its versions) at once and predict
        them together: a held-out test set nothing was learned from.
    exclude: track ids whose song groups are never learned from in any fold
        (e.g. the verified test split while measuring the dev split).
    Truth labels that aren't a known genre (a record label typed into the
    genre field) are skipped and counted in `skipped`.
    """
    import dataclasses
    trusted = training_labels(tracks, labels, include_tags)
    truth = dict(trusted if truth is None else truth)
    group = _groups(tracks)
    related = version_groups(tracks)
    always = {related[i] for i in (exclude or ()) if i in related}
    members: dict[str, list[str]] = defaultdict(list)
    skipped = set()
    for tid in sorted(truth):
        if tid not in related:
            continue
        if not known_family(family_of(truth[tid])):
            skipped.add(group[tid])
            continue
        members[group[tid]].append(tid)
    model_results = model_results or {}
    metadata = metadata or {}

    def probe_of(ids):
        # Predict the copy with the most to go on (another copy's file may be missing).
        return min(ids, key=lambda i: (i not in model_results, i not in features,
                                       i not in metadata, i))

    def hide(held):
        return [dataclasses.replace(t, genre="") if related[t.id] in held else t
                for t in tracks]

    predictions: dict[str, GenrePrediction] = {}
    if holdout:
        held = {related[i] for ids in members.values() for i in ids} | always
        probes = {probe_of(ids) for ids in members.values()}
        if probes:
            predictions = predict_genres(hide(held), features, None, model_results, labels,
                                         targets=probes, metadata=metadata, use=use,
                                         include_tags=include_tags, method=method)
    confusion: Counter = Counter()
    confident = correct_confident = correct_top = 0
    rows = []
    for g, ids in sorted(members.items()):
        probe = probe_of(ids)
        if holdout:
            p = predictions[probe]
        else:
            held = {related[i] for i in ids} | always
            p = predict_genres(hide(held), features, None, model_results, labels,
                               targets={probe}, metadata=metadata, use=use,
                               include_tags=include_tags, method=method)[probe]
        truth_tag = Counter(truth[i] for i in ids).most_common(1)[0][0]
        family = family_of(truth_tag)
        rows.append((probe, truth_tag, p))
        if p.candidates and p.candidates[0][0] == family:
            correct_top += 1
        if p.confident:
            confident += 1
            correct_confident += p.family == family
            confusion[(family, p.family)] += 1
    return Evaluation(len(members), confident, correct_confident, correct_top,
                      macro_f1(confusion), dict(confusion), rows, len(skipped))


# --------------------------------------------------------------------------- #
# Auto genre folders
# --------------------------------------------------------------------------- #

AUTO_FOLDER = "Auto Genres"


def auto_genre_plan(tracks: list[Track], guesses: dict[str, str],
                    review=None):
    """A folder per genre family, a playlist per sub-genre tag, plus guessed songs.

    guesses: confident {id: subgenre or family}. review: ids of untagged /
    vague songs whose genre is uncertain; they go in a "Needs Review" folder.
    It can be {id: suggested family or ""}: suggested songs go in a
    "<family> (suggested)" playlist inside that folder (still unconfirmed).
    Returns plan entries [((AUTO_FOLDER, family), [(playlist, [ids])])].
    """
    review = review or set()
    suggestion = review if isinstance(review, dict) else {}
    tree: dict[str, dict[str, list[Track]]] = defaultdict(lambda: defaultdict(list))
    spelling: dict[str, str] = {}
    for t in tracks:
        if t.id in guesses and is_vague(t.genre):
            family = family_of(guesses[t.id])
            name = guesses[t.id] + DETECTED_SUFFIX
        elif not is_vague(t.genre):
            family = family_of(t.genre)
            key = _norm_genre(t.genre)
            name = spelling.setdefault(key, " ".join(t.genre.split()))
        elif t.id in review:
            family = NEEDS_REVIEW
            name = (suggestion[t.id] + SUGGESTED_SUFFIX) if suggestion.get(t.id) else NEEDS_REVIEW
        elif t.genre.strip():
            family = family_of(t.genre)     # vague tag, no guess possible
            name = spelling.setdefault(_norm_genre(t.genre), " ".join(t.genre.split()))
        else:
            family, name = UNKNOWN, "No genre"
        tree[family][name].append(t)

    def order(t: Track):
        return (t.bpm, t.artist.casefold(), t.name.casefold())

    plan = []
    for family in sorted(tree, key=lambda f: (f in (UNKNOWN, NEEDS_REVIEW), f == UNKNOWN,
                                              f.casefold())):
        subs = tree[family]
        names = sorted(subs, key=lambda n: (n.endswith(DETECTED_SUFFIX), n.casefold()))
        lists = [(n, [t.id for t in sorted(subs[n], key=order)]) for n in names]
        if len(lists) > 1:
            everything = sorted((t for n in names for t in subs[n]), key=order)
            lists.insert(0, (f"All {family}", [t.id for t in everything]))
        plan.append(((AUTO_FOLDER, family), lists))
    return plan
