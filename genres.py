"""Genre families, genre guessing from rekordbox's analysis, and auto genre folders.

rekordbox's analysis doesn't include genre - it measures BPM and a colour
waveform (loudness plus bass / mid / treble content). Genres for untagged or
vaguely tagged songs ("Electronic", "Dance & EDM") are therefore *estimated*:

1. Learned from the library: songs that do have a clear genre tag are used as
   examples, and each untagged song gets the genre of the songs it sounds most
   like (BPM, bass, brightness, punch, loudness).
2. If there aren't enough tagged examples, simple BPM + sound rules are used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np

from organizer import Track, _norm_genre, genre_score

UNKNOWN = "Unknown Genre"
DETECTED_SUFFIX = " (detected)"

# Tags that say little about the actual style.
ELECTRONIC_VAGUE = {
    "electronic", "electronica", "dance", "edm", "dance and edm", "dance electronic",
    "electronic dance", "electronic dance music", "club",
}
VAGUE_TAGS = ELECTRONIC_VAGUE | {"other", "unknown", "misc", "various", "none", "genre", "music"}

# Family name -> words that belong to it. Order matters only for ties.
FAMILIES = [
    ("House", ["house", "garage", "ukg", "amapiano", "jackin"]),
    ("Techno", ["techno", "tekno", "minimal"]),
    ("Trance", ["trance", "psytrance", "psy", "goa"]),
    ("Drum & Bass", ["drum and bass", "dnb", "d and b", "drum n bass", "jungle", "liquid",
                     "neurofunk"]),
    ("Dubstep", ["dubstep", "riddim", "brostep", "bass music"]),
    ("Breaks", ["breaks", "breakbeat", "breakbeats"]),
    ("Hardstyle", ["hardstyle", "hardcore", "hard dance", "gabber"]),
    ("EDM", ["edm", "big room", "electro", "dance", "future bass", "festival"]),
    ("Hip Hop", ["hip hop", "rap", "trap", "drill", "boom bap", "grime"]),
    ("R&B", ["r and b", "rnb", "soul", "neo soul"]),
    ("Pop", ["pop", "k pop"]),
    ("Latin", ["latin", "reggaeton", "dembow", "salsa", "bachata", "cumbia", "merengue"]),
    ("Afro", ["afrobeats", "afrobeat", "afro"]),
    ("Disco & Funk", ["disco", "nu disco", "funk", "boogie"]),
    ("Reggae & Dancehall", ["reggae", "dancehall", "dub"]),
    ("Downtempo", ["downtempo", "chill", "chillout", "ambient", "lofi", "lo fi", "trip hop"]),
    ("Rock", ["rock", "indie", "metal", "punk", "alternative"]),
    ("Jazz", ["jazz", "swing"]),
    ("Country", ["country", "bluegrass"]),
]


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


# --------------------------------------------------------------------------- #
# Guessing
# --------------------------------------------------------------------------- #

SOUND = ("lows", "highs", "brightness", "flux", "loudness")
ELECTRONIC = {"House", "Techno", "Trance", "Drum & Bass", "Dubstep", "Breaks", "Hardstyle", "EDM"}
MIN_EXAMPLES = 6       # tagged songs needed before learning from the library
MIN_PER_FAMILY = 2


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


def _rule_guess(bpm: float, z: dict[str, float], electronic_only: bool) -> Optional[str]:
    """Rough style rules from BPM plus how the track sounds compared with the library."""
    if bpm <= 0:
        return None
    if 160 <= bpm <= 185:
        return "Drum & Bass"
    if 186 <= bpm <= 220:
        return "Hardstyle" if bpm < 200 else "Drum & Bass"
    if 136 <= bpm <= 155:
        # Only name a specific style on a clear signal; otherwise stay broad.
        if z["flux"] < -0.8 and z["lows"] > 0.8:
            return "Dubstep"           # half-time feel, heavy bass
        if z["highs"] > 1.0 and z["brightness"] < 0:
            return "Techno"            # dark, percussive
        if z["brightness"] > 1.0 and z["flux"] < 0:
            return "Trance"            # bright, smooth
        return "EDM"
    if 118 <= bpm < 136:
        return "House"
    if electronic_only:
        return "EDM"
    if 60 <= bpm < 118:
        return "Hip Hop" if z["lows"] > -0.3 else "Downtempo"
    return None


def guess_genres(tracks: list[Track], features: dict[str, dict],
                 log: Callable[[str], None] = print) -> dict[str, str]:
    """Guess a genre family for every track with no or a vague tag. Returns {id: family}."""
    targets = [t for t in tracks if is_vague(t.genre)]
    if not targets:
        return {}

    rows = [features.get(t.id, {}) | {"bpm": t.bpm if t.bpm > 0 else np.nan} for t in tracks]
    z = _zscores(rows, SOUND + ("bpm",))
    index = {t.id: i for i, t in enumerate(tracks)}

    # Examples: clearly tagged tracks with measurements.
    examples = [(index[t.id], family_of(t.genre)) for t in tracks
                if not is_vague(t.genre) and t.id in features and t.bpm > 0]
    counts = Counter(fam for _, fam in examples)
    examples = [(i, fam) for i, fam in examples if counts[fam] >= MIN_PER_FAMILY]
    learn = len(examples) >= MIN_EXAMPLES and len({fam for _, fam in examples}) >= 2

    def vector(i):
        # BPM counts double: it's the strongest single clue to genre.
        return np.array([2 * z["bpm"][i]] + [z[k][i] for k in SOUND])

    if learn:
        ex_vectors = np.array([vector(i) for i, _ in examples])
        ex_families = [fam for _, fam in examples]

    guesses: dict[str, str] = {}
    learned = 0
    for t in targets:
        i = index[t.id]
        electronic_only = _norm_genre(t.genre) in ELECTRONIC_VAGUE   # "Dance & EDM" etc.
        guess = None
        if learn and t.id in features and t.bpm > 0:
            dist = np.linalg.norm(ex_vectors - vector(i), axis=1)
            near = np.argsort(dist)[: min(5, len(dist))]
            votes: dict[str, float] = defaultdict(float)
            for j in near:
                if not electronic_only or ex_families[j] in ELECTRONIC:
                    votes[ex_families[j]] += 1 / (dist[j] + 0.25)
            if votes:
                fam, weight = max(votes.items(), key=lambda kv: kv[1])
                if weight / sum(votes.values()) >= 0.6:
                    guess = fam
                    learned += 1
        if guess is None:
            guess = _rule_guess(t.bpm, {k: z[k][i] for k in SOUND}, electronic_only)
        if guess:
            guesses[t.id] = guess

    how = (f"{learned} learned from your tagged songs, {len(guesses) - learned} from BPM/sound rules"
           if learn else "from BPM and sound (tag more songs to let it learn your library)")
    log(f"Guessed genres for {len(guesses)} of {len(targets)} untagged/vague songs ({how}).")
    return guesses


def effective_genre(t: Track, guesses: dict[str, str]) -> str:
    """The tag, unless it's missing or vague and a guess exists."""
    if t.id in guesses and is_vague(t.genre):
        return guesses[t.id]
    return t.genre


# --------------------------------------------------------------------------- #
# Auto genre folders
# --------------------------------------------------------------------------- #

AUTO_FOLDER = "Auto Genres"


def auto_genre_plan(tracks: list[Track], guesses: dict[str, str]):
    """A folder per genre family, a playlist per sub-genre tag, plus guessed songs.

    Returns plan entries [((AUTO_FOLDER, family), [(playlist, [ids])])].
    """
    tree: dict[str, dict[str, list[Track]]] = defaultdict(lambda: defaultdict(list))
    spelling: dict[str, str] = {}
    for t in tracks:
        if t.id in guesses and is_vague(t.genre):
            family = guesses[t.id]
            name = family + DETECTED_SUFFIX
        elif not is_vague(t.genre):
            family = family_of(t.genre)
            key = _norm_genre(t.genre)
            name = spelling.setdefault(key, " ".join(t.genre.split()))
        elif t.genre.strip():
            family = family_of(t.genre)     # vague tag, no guess possible
            name = spelling.setdefault(_norm_genre(t.genre), " ".join(t.genre.split()))
        else:
            family, name = UNKNOWN, "No genre"
        tree[family][name].append(t)

    def order(t: Track):
        return (t.bpm, t.artist.casefold(), t.name.casefold())

    plan = []
    for family in sorted(tree, key=lambda f: (f == UNKNOWN, f.casefold())):
        subs = tree[family]
        names = sorted(subs, key=lambda n: (n.endswith(DETECTED_SUFFIX), n.casefold()))
        lists = [(n, [t.id for t in sorted(subs[n], key=order)]) for n in names]
        if len(lists) > 1:
            everything = sorted((t for n in names for t in subs[n]), key=order)
            lists.insert(0, (f"All {family}", [t.id for t in everything]))
        plan.append(((AUTO_FOLDER, family), lists))
    return plan

