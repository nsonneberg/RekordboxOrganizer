"""Core logic: read a rekordbox library, build organized playlists, write them back.

The library comes either straight from rekordbox's database (see rbdb.py) or
from an XML export. With an XML export the output is a new XML file to import
back into rekordbox. The music files themselves are never modified.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import dataclasses
import datetime
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import unquote, urlparse


# --------------------------------------------------------------------------- #
# Lookup tables
# --------------------------------------------------------------------------- #

# Classic key notation -> Camelot. rekordbox writes either form depending on
# the user's "key display format" preference.
_CLASSIC_TO_CAMELOT = {
    "Abm": "1A", "G#m": "1A", "Ebm": "2A", "D#m": "2A", "Bbm": "3A", "A#m": "3A",
    "Fm": "4A", "Cm": "5A", "Gm": "6A", "Dm": "7A", "Am": "8A", "Em": "9A",
    "Bm": "10A", "F#m": "11A", "Gbm": "11A", "Dbm": "12A", "C#m": "12A",
    "B": "1B", "Cb": "1B", "F#": "2B", "Gb": "2B", "Db": "3B", "C#": "3B",
    "Ab": "4B", "G#": "4B", "Eb": "5B", "D#": "5B", "Bb": "6B", "A#": "6B",
    "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B",
}
_CAMELOT_TO_CLASSIC = {
    "1A": "Abm", "2A": "Ebm", "3A": "Bbm", "4A": "Fm", "5A": "Cm", "6A": "Gm",
    "7A": "Dm", "8A": "Am", "9A": "Em", "10A": "Bm", "11A": "F#m", "12A": "Dbm",
    "1B": "B", "2B": "F#", "3B": "Db", "4B": "Ab", "5B": "Eb", "6B": "Bb",
    "7B": "F", "8B": "C", "9B": "G", "10B": "D", "11B": "A", "12B": "E",
}

# rekordbox track colours (Colour attribute) -> name.
_COLOURS = {
    "0XFF007F": "Pink", "0XFF0000": "Red", "0XFFA500": "Orange",
    "0XFFFF00": "Yellow", "0X00FF00": "Green", "0X25FDE9": "Aqua",
    "0X0000FF": "Blue", "0X660099": "Purple",
}
COLOUR_ORDER = ["Pink", "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple"]

ENERGY_NAMES = {1: "Chill", 2: "Warm-up", 3: "Groove", 4: "Driving", 5: "Peak Time"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    id: str
    name: str
    artist: str
    genre: str
    label: str
    bpm: float
    key: str          # Camelot notation, "" if unknown
    year: int
    rating: int       # 0-5 stars
    colour: str       # colour name, "" if none
    date_added: str   # "YYYY-MM-DD"
    play_count: int
    path: str         # local file path decoded from Location
    energy: int = 0   # 1-5 once analysed, 0 if unknown
    waveform: str = ""   # rekordbox analysis file (.EXT), direct mode only
    length: int = 0      # seconds, 0 if unknown
    size: int = 0        # file size in bytes, 0 if unknown
    cues: int = 0        # hot cues and memory cues set on it


@dataclass
class CustomPlaylist:
    """A playlist the user names. Each filled-in rule must match; empty = any."""
    name: str
    genres: str = ""   # comma-separated words, e.g. "hip hop, rap"
    bpm: str = ""      # e.g. "120-126", "128+", "174"
    energy: str = ""   # e.g. "4-5", "3"


@dataclass
class Options:
    custom_enabled: bool = False   # sort the library into the user's own playlists
    custom: list[CustomPlaylist] = field(default_factory=list)
    custom_leftovers: bool = False  # songs matching no rule go to "Everything Else"
    guess_genres: bool = True    # estimate genres for untagged / vaguely tagged songs
    date_from: str = ""          # only organize songs added on/after this date (YYYY-MM-DD)
    date_to: str = ""            # ...and on/before this one; "" = no limit
    merge_existing: bool = True  # add to same-named folders/playlists instead of replacing
    move_mismatched: bool = True  # ...and move songs that now fit another playlist better
    auto_genres: bool = False    # auto genre detection: a folder per genre family
    energy: bool = False         # listen to the music and score energy 1-5
    energy_ratings: bool = True  # ...and set the star rating to the energy level
    keep_ratings: bool = True    # ...except on tracks that already have a rating
    energy_tags: bool = False    # ...and/or tag it "Energy 1-5" in My Tag (direct mode)
    phrase_cues: bool = False    # hot cues where sections start (direct mode only)
    cue_colours: bool = True     # ...coloured by section
    memory_cues: bool = False    # memory cues where the intro ends / outro starts
    # Set builder: one playlist in mixable order following an energy curve.
    set_build: bool = False
    set_name: str = "My Set"
    set_start: int = 2
    set_end: int = 5
    set_length: int = 20
    set_genres: str = ""
    set_bpm: str = ""
    # Automatic playlists: all off unless asked for.
    genre: bool = False
    bpm: bool = False
    bpm_step: int = 5
    key: bool = False
    artist: bool = False
    label: bool = False
    min_tracks: int = 3          # minimum tracks for Artist / Label playlists
    year: bool = False
    rating: bool = False
    colour: bool = False
    date_added: bool = False
    duplicates: bool = False
    missing_files: bool = False
    missing_info: bool = False
    never_played: bool = False
    folder_name: str = "Organized"


# Options whose rekordbox data format hasn't been checked against rekordbox's own
# yet. They stay off (and hidden in the app) until it has.
NOT_YET = ("cue_colours", "memory_cues", "energy_tags")


# [(folder, [(playlist, [track ids])])]. folder is a name, TOP_LEVEL (straight into the
# main folder), or a tuple of names for nested folders, e.g. ("Auto Genres", "House").
Plan = list[tuple[object, list[tuple[str, list[str]]]]]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

@dataclass
class Changes:
    """Every change a run makes, for the preview."""
    ratings: list = field(default_factory=list)   # (track id, old stars, new stars)
    created: list = field(default_factory=list)   # "Organized / House"
    added: list = field(default_factory=list)     # (playlist, track id)
    removed: list = field(default_factory=list)   # (playlist, track id)
    cues: list = field(default_factory=list)      # (track id, pad, label, ms)
    genres: list = field(default_factory=list)    # (track id, old genre, new genre)
    tags: list = field(default_factory=list)      # (track id, old energy tag, new)
    sets: list = field(default_factory=list)      # (playlist, [track ids in order], replaced)
    names: dict = field(default_factory=dict)     # track id -> song name, for showing them

    def empty(self) -> bool:
        return not any(v for k, v in vars(self).items() if k != "names")


def _int(value: Optional[str]) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _float(value: Optional[str]) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def location_to_path(location: str) -> str:
    """Convert a rekordbox Location URL to a local file path."""
    if not location:
        return ""
    path = unquote(urlparse(location).path)
    if re.match(r"^/[A-Za-z]:", path):   # Windows: "/C:/Music/..." -> "C:/Music/..."
        path = path[1:]
    return os.path.normpath(path)


def to_camelot(tonality: str) -> str:
    tonality = (tonality or "").strip()
    if not tonality:
        return ""
    upper = tonality.upper()
    if re.fullmatch(r"(1[0-2]|[1-9])[AB]", upper):
        return upper
    return _CLASSIC_TO_CAMELOT.get(tonality, "")


def load_library(xml_path: str) -> tuple[ET.ElementTree, list[Track]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "DJ_PLAYLISTS":
        raise ValueError("This doesn't look like a rekordbox XML export (missing DJ_PLAYLISTS).")
    collection = root.find("COLLECTION")
    if collection is None:
        raise ValueError("No COLLECTION found in the XML file.")

    tracks = []
    for el in collection.findall("TRACK"):
        a = el.attrib
        tracks.append(Track(
            id=a.get("TrackID", ""),
            name=a.get("Name", "").strip(),
            artist=a.get("Artist", "").strip(),
            genre=a.get("Genre", "").strip(),
            label=a.get("Label", "").strip(),
            bpm=_float(a.get("AverageBpm")),
            key=to_camelot(a.get("Tonality", "")),
            year=_int(a.get("Year")),
            rating=round(_int(a.get("Rating")) / 51),
            colour=_COLOURS.get(a.get("Colour", "").upper(), ""),
            date_added=a.get("DateAdded", ""),
            play_count=_int(a.get("PlayCount")),
            path=location_to_path(a.get("Location", "")),
            length=_int(a.get("TotalTime")),
            size=_int(a.get("Size")),
            cues=len(el.findall("POSITION_MARK")),
        ))
    return tree, tracks


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #

def _group(tracks, keyfn, min_tracks=1, case_insensitive=False):
    """Group tracks by keyfn(track) -> (sort_key, name) | None.

    Returns [(name, [track ids])] in sort order.
    """
    groups: dict = defaultdict(list)
    names: dict = {}
    sort_keys: dict = {}
    for t in tracks:
        result = keyfn(t)
        if result is None:
            continue
        sort_key, name = result
        ident = name.casefold() if case_insensitive else name
        names.setdefault(ident, name)          # first spelling seen wins
        sort_keys.setdefault(ident, sort_key)
        groups[ident].append(t)

    out = []
    for ident in sorted(groups, key=lambda i: sort_keys[i]):
        members = groups[ident]
        if len(members) < min_tracks:
            continue
        members.sort(key=lambda t: (t.bpm, t.artist.casefold(), t.name.casefold()))
        out.append((names[ident], [t.id for t in members]))
    return out


def _text_key(value: str, fallback: Optional[str]):
    value = " ".join(value.split())
    if not value:
        return None if fallback is None else ("\uffff", fallback)   # fallback sorts last
    return (value.casefold(), value)


def _norm_title(s: str) -> str:
    s = s.casefold()
    s = re.sub(r"\(\s*original mix\s*\)", "", s)
    s = re.sub(r"[^\w]+", " ", s)
    return " ".join(s.split())


# Junk that download sites and rippers add to titles; never part of the song's name.
_TITLE_JUNK = re.compile(
    r"^\s*\d{1,3}\s*[.\-)]\s+"                                # "12. ", "07 - "
    r"|[\[(][^\])]*\b(?:free|download|dl|buy|out now)\b[^\])]*[\])]"  # "[FREE DL]"
    r"|/?\s*'?buy'?\s*=\s*free\b.*$"                          # "/ 'Buy' = FREE MP3!"
    r"|\b(?:free (?:download|dl|mp3))\b.*$"
    r"|\.(?:mp3|m4a|wav|aiff?|flac)$", re.I)

DUPLICATES_NAME = "Possible Duplicates"
EXTRA_COPIES_NAME = "Duplicates - Extra Copies"
LENGTH_SLACK = 2   # seconds two copies of a song may differ by


def song_key(t: Track) -> tuple[str, str]:
    """(artist, title) for spotting copies of one song: case, punctuation and junk ignored.

    Mix names stay ("Extended Mix" and "Radio Edit" are different versions).
    """
    title = t.name
    for _ in range(3):
        cleaned = _TITLE_JUNK.sub("", title).strip()
        if cleaned == title:
            break
        title = cleaned
    return _norm_title(t.artist), _norm_title(title or t.name)


def _keep_score(t: Track) -> tuple:
    """Higher = the better copy to keep."""
    present = bool(t.path) and os.path.exists(t.path)
    return (present, t.cues, t.play_count, t.rating, bool(t.waveform), t.size,
            # the older entry (it's what playlists and history point to), then a tidier path
            [-ord(c) for c in t.date_added[:19]], -len(t.path))


def find_duplicates(tracks: list[Track]) -> list[list[Track]]:
    """Groups of copies of the same song, the copy to keep first.

    Copies: the same artist and title (ignoring junk like track numbers and
    "[FREE DL]") with lengths within LENGTH_SLACK seconds; or files of exactly
    the same size and length.
    """
    parent = {t.id: t.id for t in tracks}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        parent[root(a)] = root(b)

    by_name: dict = defaultdict(list)
    by_file: dict = defaultdict(list)
    for t in tracks:
        if t.name:
            by_name[song_key(t)].append(t)
        if t.size > 0 and t.length > 0:
            by_file[(t.size, t.length)].append(t)
    for same in by_name.values():
        for i, a in enumerate(same):
            for b in same[i + 1:]:
                if not a.length or not b.length or abs(a.length - b.length) <= LENGTH_SLACK:
                    join(a.id, b.id)
    for same in by_file.values():
        for b in same[1:]:
            join(same[0].id, b.id)

    groups: dict = defaultdict(list)
    for t in tracks:
        groups[root(t.id)].append(t)
    found = [sorted(g, key=_keep_score, reverse=True) for g in groups.values() if len(g) > 1]
    return sorted(found, key=lambda g: song_key(g[0]))


def parse_date(text: str) -> str:
    """'2026-10-21', '2026/10/21', '10/21/2026' or '' -> 'YYYY-MM-DD' or ''. Raises ValueError."""
    t = text.strip()
    if not t:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y"):
        try:
            return datetime.datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"'{text}' isn't a date. Use YYYY-MM-DD, e.g. 2026-10-21.")


def select_tracks(tracks: list[Track], opt: Options) -> list[Track]:
    """The tracks to organize: those added within the chosen dates (all if none set)."""
    start, end = parse_date(opt.date_from), parse_date(opt.date_to)
    if start and end and end < start:
        raise ValueError(f"The end date ({end}) is before the start date ({start}).")
    if not start and not end:
        return list(tracks)
    return [t for t in tracks
            if t.date_added[:10] and (not start or t.date_added[:10] >= start)
            and (not end or t.date_added[:10] <= end)]


LEFTOVERS_NAME = "Everything Else"
SETS_FOLDER = "Sets"   # set builder playlists: rewritten in order on every run
TOP_LEVEL = ""   # plan folder name meaning "directly inside the main folder"

# Genre words that belong together. Typing any word in a group also matches the
# others, so "hip hop" catches "Trap" and "dnb" catches "Drum & Bass". Sub-genres
# that contain the word ("Progressive House", "Tech-House") match anyway.
GENRE_GROUPS = [
    ["hip hop", "rap", "trap", "drill", "boom bap"],
    ["drum and bass", "dnb", "d and b", "drum n bass", "jungle", "liquid", "neurofunk"],
    ["dubstep", "riddim", "brostep"],
    ["techno", "tekno"],
    ["trance", "psytrance", "psy", "goa"],
    ["edm", "dance", "big room", "electro"],
    ["r and b", "rnb"],
    ["garage", "ukg", "speed garage"],
    ["afrobeats", "afrobeat", "afro", "amapiano"],
    ["latin", "reggaeton", "dembow", "salsa", "bachata", "cumbia", "merengue"],
]


def _norm_genre(text: str) -> str:
    """'Drum & Bass' -> 'drum and bass'; 'Hip-Hop/Rap' -> 'hip hop rap'."""
    text = text.casefold().replace("&", " and ").replace("+", " and ")
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


_ALIASES: dict[str, set[str]] = {}
for _group_words in GENRE_GROUPS:
    for _w in _group_words:
        _ALIASES.setdefault(_w, set()).update(_group_words)


def genre_score(term: str, genre: str) -> Optional[tuple[int, float]]:
    """How well a normalized genre tag fits a normalized search term (None = not at all).

    Returns (quality, position): quality 3 = exact tag, 2 = contains the word,
    1 = related word or compound like "deephouse". Position favours matches near
    the end of the tag, where the main genre usually is ("Melodic House & Techno").
    """
    tokens = genre.split()
    if not tokens:
        return None
    if genre == term:
        return (3, 1.0)
    padded = f" {genre} "
    best = None
    for cand in [term] + sorted(_ALIASES.get(term, set()) - {term}):
        quality = 2 if cand == term else 1
        i = padded.find(f" {cand} ")
        if i >= 0:
            end = len(padded[: i + len(cand) + 1].split()) - 1
            score = (quality, (end + 1) / len(tokens))
            best = score if best is None else max(best, score)
            continue
        compact = cand.replace(" ", "")
        if len(compact) < 4:       # short words ("rap") must match whole, never "trap"
            continue
        for idx, tok in enumerate(tokens):
            if tok == compact or tok.startswith(compact) or tok.endswith(compact):
                score = (quality if tok == compact else 1, (idx + 1) / len(tokens))
                best = score if best is None else max(best, score)
    return best


def parse_range(text: str) -> tuple[Optional[int], Optional[int]]:
    """'' / 'any' -> (None, None); '120-126'; '128+'; '174' -> (174, 174)."""
    t = text.strip().lower().replace(" ", "")
    if t in ("", "any", "*"):
        return (None, None)
    m = re.fullmatch(r"(\d+)(?:-(\d+)|(\+))?", t)
    if not m:
        raise ValueError(text)
    lo = int(m.group(1))
    if m.group(3):
        return (lo, None)
    hi = int(m.group(2)) if m.group(2) else lo
    if hi < lo:
        raise ValueError(text)   # probably a typo ("100-50" for "100-150"); don't guess
    return (lo, hi)


@dataclass
class _Rule:
    name: str
    terms: list[str]       # normalized genre words
    bpm: tuple[Optional[int], Optional[int]]
    energy: tuple[Optional[int], Optional[int]]

    def score(self, t: Track, genre: str) -> Optional[tuple[int, float]]:
        """None if the track doesn't fit; otherwise how well it fits (higher = better)."""
        for value, (lo, hi) in ((round(t.bpm), self.bpm), (t.energy, self.energy)):
            if lo is None and hi is None:
                continue
            if value <= 0 or (lo is not None and value < lo) or (hi is not None and value > hi):
                return None
        if not self.terms:
            return (0, 0.0)    # no genre rule: fits, but less specifically than a genre match
        scores = [s for s in (genre_score(term, genre) for term in self.terms) if s]
        return max(scores) if scores else None


def compile_custom(opt: Options) -> list[_Rule]:
    """Validate the user's playlists. Raises ValueError with a friendly message."""
    if not opt.custom_enabled:
        return []
    rules = []
    for i, cp in enumerate(opt.custom, 1):
        name = cp.name.strip() or f"Playlist {i}"
        try:
            bpm = parse_range(cp.bpm)
        except ValueError:
            backwards = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", cp.bpm)
            if backwards and int(backwards.group(2)) < int(backwards.group(1)):
                lo, hi = int(backwards.group(1)), int(backwards.group(2))
                fix = f"{lo}-{hi + 100}" if hi < 100 <= lo < hi + 100 else f"{hi}-{lo}"
                raise ValueError(f"Playlist '{name}': in BPM '{cp.bpm}' the second number is "
                                 f"lower than the first. Did you mean {fix}?") from None
            raise ValueError(f"Playlist '{name}': BPM '{cp.bpm}' isn't a valid range. "
                             "Try 120-126, 128+ or 174, or leave it empty.") from None
        try:
            energy = parse_range(cp.energy)
        except ValueError:
            raise ValueError(f"Playlist '{name}': energy '{cp.energy}' isn't valid. "
                             "Try 4-5, 3+ or 2, or leave it empty.") from None
        if any(v is not None and not 1 <= v <= 5 for v in energy):
            raise ValueError(f"Playlist '{name}': energy goes from 1 to 5.")
        if (energy[0] or 1) <= 1 and (energy[1] is None or energy[1] >= 5):
            energy = (None, None)          # "1-5" means any energy
        terms = [_norm_genre(w) for w in cp.genres.split(",") if _norm_genre(w)]
        if not terms and bpm == (None, None) and energy == (None, None):
            raise ValueError(f"Playlist '{name}' has no rules, so it would take every song. "
                             "Give it a genre, BPM or energy, or lower 'How many'.")
        rules.append(_Rule(name, terms, bpm, energy))
    return rules


def rules_need_energy(rules: list[_Rule]) -> bool:
    return any(r.energy != (None, None) for r in rules)


def sort_into_custom(tracks: list[Track], rules: list[_Rule], leftovers: bool,
                     guesses: Optional[dict[str, str]] = None,
                     by_guess: Optional[set] = None):
    """Put each track in the playlist it fits best; tracks that fit none are left alone.

    guesses: {track id: guessed genre} for songs with no or a vague tag. The
    song's own tag comes first: a playlist naming it (e.g. 'EDM' for a song
    tagged 'Dance & EDM') beats one that only matches the guess.
    by_guess: if given, filled with the ids placed because of their guess.
    Returns [(name, [ids])]. Ties go to the playlist higher in the list.
    """
    guesses = guesses or {}
    lists: list[tuple[str, list[str]]] = [(r.name, []) for r in rules]
    rest: list[str] = []
    for t in sorted(tracks, key=lambda t: (t.bpm, t.artist.casefold(), t.name.casefold())):
        tag = _norm_genre(t.genre)
        guess = _norm_genre(guesses[t.id]) if t.id in guesses else ""
        best_i, best_key = None, None
        for i, rule in enumerate(rules):
            if not rule.terms:
                s = rule.score(t, tag)
                key = None if s is None else (0, *s)
            else:
                s = rule.score(t, tag) if tag else None
                key = (1, *s) if s is not None else None
                if key is None and guess:
                    s = rule.score(t, guess)
                    key = None if s is None else (0, *s, "guess")
            if key is not None and (best_key is None or key[:3] > best_key[:3]):
                best_i, best_key = i, key
        if best_i is None:
            rest.append(t.id)
        else:
            lists[best_i][1].append(t.id)
            if by_guess is not None and best_key[-1] == "guess":
                by_guess.add(t.id)
    if leftovers and rest:
        lists.append((LEFTOVERS_NAME, rest))
    return lists


def evictions(plan: Plan, rules: list[_Rule]) -> dict[str, set[str]]:
    """{playlist name: track ids to take out of it} when re-running into existing playlists.

    A song leaves one of the user's playlists (or 'Everything Else') only when it
    now fits a *different* one of their playlists best; it's added there by the
    plan. Songs that fit none stay where they are.
    """
    best: dict[str, str] = {}
    names = {r.name for r in rules}
    for folder, lists in plan:
        if folder_path(folder) == ():
            for name, ids in lists:
                if name in names:
                    for i in ids:
                        best[i] = name
    if not best:
        return {}
    out = {}
    for name in [r.name for r in rules] + [LEFTOVERS_NAME]:
        leaving = {i for i, target in best.items() if not same_name(target, name)}
        if leaving:
            out[name] = leaving
    return out


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()


class Evictor:
    """Which songs to take out of existing playlists in the main folder on a re-run.

    - The user's own playlists (directly in the main folder): a song leaves only
      when it now fits a different one of their playlists best (evictions()).
    - Folders the app generates (By Energy, Auto Genres, By BPM, Cleanup...):
      they show values the app works out, so for the songs being organized
      they're brought up to date - a song whose energy went from 3 to 2 leaves
      "3 - Groove". Only folders this run generates are touched.
    """

    def __init__(self, top: dict[str, set[str]], plan: Plan, chosen: set[str], sync: bool):
        self.top = {_key(name): ids for name, ids in top.items()}
        self.placed: dict[tuple, set[str]] = {}
        self.roots: set[str] = set()
        self.chosen = chosen
        if sync:
            for folder, lists in plan:
                path = folder_path(folder)
                if path and path[0] != SETS_FOLDER:
                    self.roots.add(_key(path[0]))
                    for name, ids in lists:
                        key = (tuple(_key(p) for p in path), _key(name))
                        self.placed.setdefault(key, set()).update(ids)

    def __bool__(self) -> bool:
        return bool(self.top or self.roots)

    def ids(self, path: tuple, name: str) -> set[str]:
        """Track ids to take out of playlist `name` in folder `path` (below the main folder)."""
        if not path:
            return self.top.get(_key(name), set())
        if _key(path[0]) not in self.roots:
            return set()
        return self.chosen - self.placed.get((tuple(_key(p) for p in path), _key(name)), set())

    def scans(self, path: tuple) -> bool:
        """Whether to look inside the folder at `path`."""
        return not path or _key(path[0]) in self.roots


def build_plan(tracks: list[Track], opt: Options, rules: Optional[list[_Rule]] = None,
               guesses: Optional[dict[str, str]] = None,
               by_guess: Optional[set] = None) -> Plan:
    plan: Plan = []

    if rules is None:
        rules = compile_custom(opt)
    if rules:
        plan.append((TOP_LEVEL, sort_into_custom(tracks, rules, opt.custom_leftovers, guesses,
                                                 by_guess)))

    if opt.auto_genres:
        import genres
        plan.extend(genres.auto_genre_plan(tracks, guesses or {}))

    if opt.energy:
        plan.append(("By Energy", _group(
            tracks, lambda t: (t.energy, f"{t.energy} - {ENERGY_NAMES[t.energy]}")
            if t.energy else None)))

    if opt.genre:
        plan.append(("By Genre", _group(
            tracks, lambda t: _text_key(t.genre, "No Genre"), case_insensitive=True)))

    if opt.bpm:
        step = max(1, int(opt.bpm_step))

        def bpm_key(t):
            if t.bpm <= 0:
                return None
            lo = int(round(t.bpm)) // step * step
            return (lo, f"{lo}-{lo + step - 1} BPM" if step > 1 else f"{lo} BPM")
        plan.append(("By BPM", _group(tracks, bpm_key)))

    if opt.key:
        def key_key(t):
            if not t.key:
                return None
            num, mode = int(t.key[:-1]), t.key[-1]
            return ((num, mode), f"{t.key} ({_CAMELOT_TO_CLASSIC[t.key]})")
        plan.append(("By Key", _group(tracks, key_key)))

    if opt.artist:
        plan.append(("By Artist", _group(
            tracks, lambda t: _text_key(t.artist, None),
            min_tracks=opt.min_tracks, case_insensitive=True)))

    if opt.label:
        plan.append(("By Label", _group(
            tracks, lambda t: _text_key(t.label, None),
            min_tracks=opt.min_tracks, case_insensitive=True)))

    if opt.year:
        plan.append(("By Year", _group(
            tracks, lambda t: (-t.year, str(t.year)) if t.year > 0 else (1, "Unknown Year"))))

    if opt.rating:
        plan.append(("By Rating", _group(
            tracks, lambda t: (-t.rating, f"{t.rating} Star{'s' if t.rating != 1 else ''}")
            if t.rating else (1, "Unrated"))))

    if opt.colour:
        plan.append(("By Color", _group(
            tracks, lambda t: (COLOUR_ORDER.index(t.colour), t.colour) if t.colour else None)))

    if opt.date_added:
        def month_key(t):
            m = re.match(r"(\d{4})-(\d{2})", t.date_added)
            if not m:
                return None
            ym = f"{m.group(1)}-{m.group(2)}"
            return (tuple(-int(x) for x in ym.split("-")), ym)   # newest first
        plan.append(("By Date Added", _group(tracks, month_key)))

    cleanup: list[tuple[str, list[str]]] = []
    if opt.duplicates:
        groups = find_duplicates(tracks)
        if groups:
            cleanup.append((DUPLICATES_NAME, [t.id for g in groups for t in g]))
            cleanup.append((EXTRA_COPIES_NAME, [t.id for g in groups for t in g[1:]]))
    if opt.missing_files:
        missing = [t.id for t in tracks if t.path and not os.path.exists(t.path)]
        if missing:
            cleanup.append(("Missing Files", missing))
    if opt.missing_info:
        for title, test in (("Missing Genre", lambda t: not t.genre),
                            ("Missing Key", lambda t: not t.key),
                            ("Missing BPM", lambda t: t.bpm <= 0)):
            ids = [t.id for t in tracks if test(t)]
            if ids:
                cleanup.append((title, ids))
    if opt.never_played:
        ids = [t.id for t in tracks if t.play_count == 0]
        if ids:
            cleanup.append(("Never Played", ids))
    if cleanup:
        plan.append(("Cleanup", cleanup))

    return [(folder, playlists) for folder, playlists in plan if playlists]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def folder_path(folder) -> tuple:
    """Plan folder (name, TOP_LEVEL or tuple) -> tuple of folder names."""
    if isinstance(folder, tuple):
        return folder
    return () if folder == TOP_LEVEL else (folder,)


def _folder(parent: ET.Element, name: str) -> ET.Element:
    return ET.SubElement(parent, "NODE", Type="0", Name=name, Count="0")


def _set_count(folder: ET.Element) -> None:
    folder.set("Count", str(len(folder.findall("NODE"))))


def same_name(a: Optional[str], b: str) -> bool:
    return " ".join((a or "").split()).casefold() == " ".join(b.split()).casefold()


def write_library(tree: ET.ElementTree, plan: Plan, folder_name: str, out_path: str,
                  ratings: Optional[dict[str, int]] = None, merge: bool = True,
                  log: Callable[[str], None] = print,
                  evict: Optional["Evictor"] = None,
                  changes: Optional[Changes] = None) -> tuple[int, int, int, list]:
    """Write the plan into the XML.

    evict: which songs to take out of existing playlists (see Evictor).
    Returns (playlists created, playlists added to, tracks added,
    [(folder path, playlist name, track id) taken out]).
    """
    root = tree.getroot()
    if ratings:
        for el in root.iterfind("COLLECTION/TRACK"):
            stars = ratings.get(el.get("TrackID", ""))
            if stars is not None:
                old = _int(el.get("Rating")) // 51
                if changes is not None and old != stars:
                    changes.ratings.append((el.get("TrackID"), old, stars))
                el.set("Rating", str(stars * 51))


    playlists = root.find("PLAYLISTS")
    if playlists is None:
        playlists = ET.SubElement(root, "PLAYLISTS")
    top = playlists.find("NODE")
    if top is None:
        top = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="0")

    def child(parent: ET.Element, name: str, kind: str) -> Optional[ET.Element]:
        for node in parent.findall("NODE"):
            if node.get("Type") == kind and same_name(node.get("Name"), name):
                return node
        return None

    organized = child(top, folder_name, "0")
    if organized is not None and not merge:
        top.remove(organized)
        organized = None
    if organized is not None:
        log(f"Adding to your existing '{organized.get('Name')}' folder.")
    else:
        organized = _folder(top, folder_name)
    folders = {(): organized}

    def folder_for(path: tuple) -> ET.Element:
        if path not in folders:
            parent = folder_for(path[:-1])
            existing = child(parent, path[-1], "0")   # (an empty Element is falsy: no `or`)
            folders[path] = existing if existing is not None else _folder(parent, path[-1])
        return folders[path]

    removed = []

    def take_out(node: ET.Element, path: tuple):
        for el in node.findall("NODE"):
            name = el.get("Name") or ""
            if el.get("Type") == "1":
                ids = evict.ids(path, name)
                for t in el.findall("TRACK"):
                    if t.get("Key") in ids:
                        el.remove(t)
                        removed.append((path, name, t.get("Key")))
                        if changes is not None:
                            changes.removed.append((" / ".join(
                                (organized.get("Name"),) + path + (name,)), t.get("Key")))
                el.set("Entries", str(len(el.findall("TRACK"))))
            elif el.get("Type") == "0" and evict.scans(path + (name,)):
                take_out(el, path + (name,))

    if evict:
        take_out(organized, ())

    created = updated = added = 0
    for folder_title, lists in plan:
        path = folder_path(folder_title)
        sub = folder_for(path)
        for title, ids in lists:
            where = " / ".join((organized.get("Name"),) + path + (title,))
            pl = child(sub, title, "1")
            if path == (SETS_FOLDER,):
                # A set is in play order: write it afresh rather than adding on.
                if changes is not None:
                    changes.sets.append((where, list(ids), pl is not None))
                if pl is None:
                    pl = ET.SubElement(sub, "NODE", Name=title, Type="1", KeyType="0")
                    created += 1
                else:
                    for t in pl.findall("TRACK"):
                        pl.remove(t)
                    updated += 1
                for track_id in ids:
                    ET.SubElement(pl, "TRACK", Key=track_id)
                pl.set("Entries", str(len(ids)))
                continue
            if pl is None:
                pl = ET.SubElement(sub, "NODE", Name=title, Type="1", KeyType="0", Entries="0")
                created += 1
                have = set()
                if changes is not None:
                    changes.created.append(where)
            else:
                have = {t.get("Key") for t in pl.findall("TRACK")}
                if any(i not in have for i in ids):
                    updated += 1
            for track_id in ids:
                if track_id not in have:
                    ET.SubElement(pl, "TRACK", Key=track_id)
                    have.add(track_id)
                    added += 1
                    if changes is not None:
                        changes.added.append((where, track_id))
            pl.set("Entries", str(len(pl.findall("TRACK"))))
    for node in organized.iter("NODE"):
        if node.get("Type") == "0":
            _set_count(node)
    _set_count(top)

    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    return created, updated, added, removed


def describe_moves(removed: list, plan: Plan, tracks: list[Track],
                   log: Callable[[str], None]) -> None:
    """Log the songs that were moved to a playlist they fit better."""
    if not removed:
        return
    target = {(tuple(_key(p) for p in folder_path(folder)), i): name
              for folder, lists in plan for name, ids in lists for i in ids}
    names = {t.id: t.name or t.path or t.id for t in tracks}
    log(f"Moved {len(removed)} song{'s' if len(removed) != 1 else ''} to the playlist "
        f"{'they fit' if len(removed) != 1 else 'it fits'} better:")
    for path, playlist, track_id in removed[:30]:
        new = target.get((tuple(_key(p) for p in path), track_id), "removed (no longer fits)")
        where = f" ({' / '.join(path)})" if path else ""
        log(f"    {names.get(track_id, track_id)}: {playlist} -> {new}{where}")
    if len(removed) > 30:
        log(f"    ... and {len(removed) - 30} more.")


def describe_changes(created: int, updated: int, added: int, removed: Optional[list] = None) -> str:
    parts = [f"{created} new playlist{'s' if created != 1 else ''}"]
    if updated:
        parts.append(f"{updated} existing playlist{'s' if updated != 1 else ''} added to")
    return ", ".join(parts) + f"; {added} track{'s' if added != 1 else ''} added in total."


def app_data_dir() -> str:
    """Per-user folder for the energy cache and saved settings."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "RekordboxOrganizer")


def default_output_path(xml_path: str) -> str:
    base, _ = os.path.splitext(xml_path)
    return base + "_organized.xml"


def analyse_tracks(tracks: list[Track], log: Callable[[str], None],
                   progress: Optional[Callable[[int, int], None]] = None) -> dict[str, dict]:
    """Measure every track (for energy and genre guessing). Returns {id: features}."""
    import energy   # needs numpy + an audio decoder; only loaded when used

    # Prefer rekordbox's own analysis (works for every format, no decoding).
    # Never mix the two sources: their measurements aren't on the same scale.
    waveforms = {t.id: t.waveform for t in tracks if t.waveform}
    if waveforms:
        return energy.analyse_waveforms(waveforms, log, progress)
    return energy.analyse_library({t.id: t.path for t in tracks if t.path}, log, progress)


def apply_energy(tracks: list[Track], opt: Options, features: dict[str, dict],
                 log: Callable[[str], None],
                 guesses: Optional[dict[str, str]] = None) -> dict[str, int]:
    """Set Track.energy from the measurements; return {track id: new star rating}.

    Star ratings follow the level: the most energetic songs get 5 stars and the
    calmest 1.
    """
    import energy

    guesses = guesses or {}
    genre_of = {t.id: f"{t.genre} {guesses.get(t.id, '')}" for t in tracks}
    scores = energy.energy_scores(features, {t.id: t.bpm for t in tracks}, genre_of)
    levels = energy.levels_from_scores(scores)
    ratings: dict[str, int] = {}
    counts = [0] * 6
    for t in tracks:
        level = levels.get(t.id, 0)
        t.energy = level
        counts[level] += 1
        if level and opt.energy and opt.energy_ratings and not (opt.keep_ratings and t.rating):
            t.rating = level
            ratings[t.id] = level
    log("Energy levels: " + ", ".join(f"{i}: {counts[i]}" for i in range(1, 6))
        + (f" ({counts[0]} not analysed)" if counts[0] else ""))
    if scores:
        names = {t.id: t.name or t.id for t in tracks}
        top = max(scores, key=scores.get)
        bottom = min(scores, key=scores.get)
        log(f"  Most energetic: {names.get(top, top)}. Calmest: {names.get(bottom, bottom)}.")
    return ratings


def set_request(opt: Options):
    import setbuilder
    return setbuilder.SetRequest(opt.set_name.strip() or "My Set", opt.set_start, opt.set_end,
                                 opt.set_length, opt.set_genres, opt.set_bpm)


def _organize(tracks: list[Track], opt: Options, log: Callable[[str], None],
              progress: Optional[Callable[[int, int], None]]
              ) -> tuple[Plan, dict[str, int], list[Track]]:
    """Shared by both modes: (plan, new star ratings, the tracks chosen to organize)."""
    rules = compile_custom(opt)          # validate before any slow work
    library = tracks
    tracks = select_tracks(library, opt)
    if len(tracks) < len(library):
        start, end = parse_date(opt.date_from), parse_date(opt.date_to)
        span = (f"between {start} and {end}" if start and end
                else f"on or after {start}" if start else f"on or before {end}")
        log(f"{len(tracks)} of {len(library)} tracks were added {span}; only those will be "
            "organized (the rest of the library is only used for comparison).")
        if not tracks:
            raise ValueError(f"No songs were added {span}.")
    if opt.set_build:
        import setbuilder
        setbuilder.check(set_request(opt))
    want_energy = opt.energy or rules_need_energy(rules) or opt.set_build
    want_guesses = opt.guess_genres and (opt.auto_genres or any(r.terms for r in rules))

    features: dict[str, dict] = {}
    if want_energy or want_guesses:
        if want_energy and not opt.energy:
            log("Your playlist rules or set use energy, so the app will measure your music.")
        features = analyse_tracks(library, log, progress)
    guesses: dict[str, str] = {}
    if want_guesses:
        import genres
        guesses = genres.guess_genres(library, features, log)
    ratings = apply_energy(library, opt, features, log, guesses) if want_energy else {}
    chosen = {t.id for t in tracks}
    ratings = {i: stars for i, stars in ratings.items() if i in chosen}

    by_guess: set = set()
    plan = build_plan(tracks, opt, rules, guesses, by_guess)
    if opt.duplicates:
        for group in find_duplicates(tracks):
            keep = group[0]
            for t in group[1:]:
                if t.cues > keep.cues:
                    missing = t.path and not os.path.exists(t.path)
                    log(f"  Note: the extra copy of '{t.name}' has {t.cues} cues the kept copy "
                        "doesn't" + (" (its file is missing, so it can't play)" if missing else "")
                        + ". Copy them over in rekordbox before removing it.")
    if opt.set_build:
        req = set_request(opt)
        chain = setbuilder.build_set(library, req, guesses)   # from the whole library
        log(setbuilder.describe(chain, req))
        if chain:
            plan.append((SETS_FOLDER, [(req.name, [t.id for t in chain])]))
    if (not plan and not ratings and not opt.phrase_cues and not opt.memory_cues
            and not (opt.energy and opt.energy_tags)):
        raise ValueError("Nothing to do - tick at least one option.")

    shown_parents = set()
    for folder, lists in plan:
        path = folder_path(folder)
        if path == ():
            placed = sum(len(ids) for name, ids in lists if name != LEFTOVERS_NAME)
            used = sum(1 for name, ids in lists if name != LEFTOVERS_NAME
                       for i in ids if i in by_guess)
            log(f"Sorted {placed} of {len(tracks)} tracks into your playlists"
                + (f" ({used} by guessed genre):" if used else ":"))
            for name, ids in lists:
                log(f"    {name}: {len(ids)} track{'s' if len(ids) != 1 else ''}")
            if placed < len(tracks) and not opt.custom_leftovers:
                log(f"  The other {len(tracks) - placed} tracks didn't fit any playlist "
                    "and were left alone.")
        elif len(path) > 1:
            if path[:-1] not in shown_parents:
                shown_parents.add(path[:-1])
                log(f"  {' / '.join(path[:-1])}:")
            count = len(lists[0][1])    # the "All ..." playlist, or the only one
            log(f"    {path[-1]}: {count} track{'s' if count != 1 else ''}, "
                f"{len(lists)} playlist{'s' if len(lists) != 1 else ''}")
        else:
            log(f"  {path[0]}: {len(lists)} playlist{'s' if len(lists) != 1 else ''}")
    return plan, ratings, tracks


def _evictor(plan: Plan, opt: Options, chosen: list[Track]) -> Evictor:
    move = opt.merge_existing and opt.move_mismatched
    top = evictions(plan, compile_custom(opt)) if move and opt.custom_enabled else {}
    return Evictor(top, plan, {t.id for t in chosen}, move)


def run(opt: Options, xml_path: Optional[str] = None, log: Callable[[str], None] = print,
        progress: Optional[Callable[[int, int], None]] = None,
        preview: bool = False, changes: Optional[Changes] = None) -> str:
    """Do the whole job.

    With xml_path, reads that export and writes a new XML next to it; returns its path.
    Without, works directly on rekordbox's database; returns the database path.
    preview: do everything on a throwaway copy and save nothing; fill `changes`
    to see what a real run would do. Works while rekordbox is open.
    """
    folder_name = opt.folder_name.strip() or "Organized"
    opt = dataclasses.replace(opt, **{name: False for name in NOT_YET})
    if changes is None:
        changes = Changes()
    if xml_path:
        log(f"Reading {os.path.basename(xml_path)} ...")
        tree, tracks = load_library(xml_path)
        log(f"Found {len(tracks)} tracks.")
        if not tracks:
            raise ValueError("The collection in this XML file is empty.")
        if opt.phrase_cues or opt.memory_cues:
            log("Cues need rekordbox's phrase analysis, so they're only added when "
                "working straight from rekordbox. Skipping them.")
            opt = dataclasses.replace(opt, phrase_cues=False, memory_cues=False)
        if opt.energy_tags:
            log("My Tags can only be written straight into rekordbox. Skipping them.")
            opt = dataclasses.replace(opt, energy_tags=False)
        changes.names.update((t.id, t.name) for t in tracks)
        plan, ratings, chosen = _organize(tracks, opt, log, progress)
        out_path = default_output_path(xml_path)
        if preview:
            import tempfile
            fd, out_path = tempfile.mkstemp(suffix=".xml")
            os.close(fd)
        try:
            counts = write_library(tree, plan, folder_name, out_path, ratings,
                                   opt.merge_existing, log, _evictor(plan, opt, chosen),
                                   changes)
        finally:
            if preview:
                os.remove(out_path)
        describe_moves(counts[3], plan, tracks, log)
        log(describe_changes(*counts))
        if preview:
            return xml_path
        log(f"Saved: {out_path}")
        return out_path

    import rbdb
    if not preview and rbdb.rekordbox_running():
        raise ValueError("rekordbox is open. Please close it first, then press Run again.")
    log("Opening your rekordbox library ...")
    db = rbdb.open_database()
    work = None
    if preview:
        real, (db, work) = db, rbdb.open_preview_copy(db)
        real.close()
    try:
        db_path = rbdb.database_path(db)
        tracks, contents = rbdb.load_tracks(db)
        log(f"Found {len(tracks)} tracks.")
        if not tracks:
            raise ValueError("Your rekordbox collection is empty.")
        changes.names.update((t.id, t.name) for t in tracks)
        plan, ratings, chosen = _organize(tracks, opt, log, progress)
        hot_cues = (rbdb.plan_hot_cues(db, chosen, log, opt.phrase_cues, opt.memory_cues)
                    if opt.phrase_cues or opt.memory_cues else {})

        if not preview:
            if rbdb.rekordbox_running():
                raise ValueError("rekordbox was opened while working. Nothing was changed - "
                                 "close rekordbox and press Run again.")
            log(f"Backed up your library to {os.path.basename(rbdb.backup(db_path))}")
        rbdb.add_hot_cues(db, contents, hot_cues, log, changes,   # saved with the rest
                          colours=opt.cue_colours)
        if opt.energy and opt.energy_tags:
            rbdb.write_energy_tags(db, {t.id: t.energy for t in chosen}, log, changes)
        counts = rbdb.write_changes(db, contents, plan, folder_name, ratings, log,
                                    opt.merge_existing, _evictor(plan, opt, chosen),
                                    changes, commit=not preview)
        describe_moves(counts[3], plan, tracks, log)
        if plan:
            log(describe_changes(*counts))
        return db_path
    finally:
        db.close()
        if work:
            import shutil
            shutil.rmtree(work, ignore_errors=True)


def summarize_changes(changes: Changes, names: Optional[dict[str, str]] = None) -> str:
    """A readable list of what a run would change. names: {track id: song name}."""
    names = names or changes.names

    def song(track_id):
        return names.get(track_id, track_id)

    lines: list[str] = []
    if changes.empty():
        return "Nothing would change: everything is already organized this way."

    def group(playlist: str) -> str:
        # "Organized / House" -> "Organized"; "Organized / By Energy / 3 - Groove"
        # -> "Organized / By Energy": a move stays within one of these.
        parts = playlist.split(" / ")
        return " / ".join(parts[:2]) if len(parts) > 2 else parts[0]

    moved_from = {(tid, group(pl)): pl for pl, tid in changes.removed}
    moved_to = {(tid, group(pl)): pl for pl, tid in changes.added
                if (tid, group(pl)) in moved_from}
    if moved_from:
        lines.append(f"MOVED ({len(moved_from)} songs)")
        for (tid, where), old in moved_from.items():
            new = moved_to.get((tid, where))
            lines.append(f"  {song(tid)}:  {old.split(' / ')[-1]} -> "
                         + (new.split(" / ")[-1] if new else "removed (no longer fits)")
                         + (f"   [{where.split(' / ', 1)[1]}]" if " / " in where else ""))
        lines.append("")

    if changes.created:
        lines.append(f"NEW PLAYLISTS ({len(changes.created)})")
        lines.extend(f"  {p}" for p in changes.created)
        lines.append("")

    added = [(pl, tid) for pl, tid in changes.added if (tid, group(pl)) not in moved_to]
    if added:
        by_playlist: dict[str, list[str]] = defaultdict(list)
        for pl, tid in added:
            by_playlist[pl].append(tid)
        lines.append(f"ADDED TO PLAYLISTS ({len(added)} entries)")
        for pl, ids in by_playlist.items():
            shown = ", ".join(song(i) for i in ids[:5])
            more = f" and {len(ids) - 5} more" if len(ids) > 5 else ""
            lines.append(f"  {pl}: +{len(ids)}  ({shown}{more})")
        lines.append("")

    for where, ids, replaced in changes.sets:
        lines.append(f"SET: {where} ({len(ids)} songs, "
                     + ("replacing the old order)" if replaced else "new playlist)"))
        lines.extend(f"  {n:>2}. {song(i)}" for n, i in enumerate(ids, 1))
        lines.append("")

    if changes.ratings:
        lines.append(f"STAR RATINGS ({len(changes.ratings)} songs)")
        for tid, old, new in sorted(changes.ratings, key=lambda r: -r[2]):
            was = f"{old} star{'s' if old != 1 else ''}" if old else "unrated"
            lines.append(f"  {song(tid)}:  {was} -> {new} star{'s' if new != 1 else ''}")
        lines.append("")

    if changes.tags:
        lines.append(f"ENERGY MY TAGS ({len(changes.tags)} songs)")
        for tid, old, new in changes.tags:
            lines.append(f"  {song(tid)}:  {old or '(none)'} -> {new}")
        lines.append("")

    if changes.genres:
        lines.append(f"GENRE TAGS ({len(changes.genres)} songs)")
        for tid, old, new in changes.genres:
            lines.append(f"  {song(tid)}:  {old or '(none)'} -> {new}")
        lines.append("")

    if changes.cues:
        by_track: dict[str, list[str]] = defaultdict(list)
        for tid, pad, label, ms in changes.cues:
            by_track[tid].append(f"{pad} {label} {ms // 60000}:{ms // 1000 % 60:02d}")
        lines.append(f"CUES ({len(changes.cues)} on {len(by_track)} songs)")
        for tid, cues in by_track.items():
            lines.append(f"  {song(tid)}:  " + ", ".join(cues))
        lines.append("")
    return "\n".join(lines).rstrip()


@dataclass
class GenreSuggestion:
    id: str
    name: str
    artist: str
    bpm: float
    tag: str      # the genre it has now ("" if none)
    guess: str    # the app's guess ("" if it couldn't tell)


def genre_review(xml_path: Optional[str] = None, log: Callable[[str], None] = print,
                 progress: Optional[Callable[[int, int], None]] = None
                 ) -> tuple[list[GenreSuggestion], list[str]]:
    """Guesses for every song with no genre or a vague one ("Dance & EDM"), to check by hand.

    Returns (suggestions, genre names to offer: the app's genre families plus
    the library's own specific tags). Only reads the library.
    """
    import genres
    if xml_path:
        _, tracks = load_library(xml_path)
    else:
        import rbdb
        db = rbdb.open_database()
        try:
            tracks, _ = rbdb.load_tracks(db)
        finally:
            db.close()
    choices: dict[str, str] = {fam.casefold(): fam for fam, _ in genres.FAMILIES}
    for t in tracks:
        tag = " ".join(t.genre.split())
        if tag and not genres.is_vague(tag):
            choices.setdefault(tag.casefold(), tag)
    names = sorted(choices.values(), key=str.casefold)
    vague = [t for t in tracks if genres.is_vague(t.genre)]
    if not vague:
        return [], names
    features = analyse_tracks(tracks, log, progress)
    guesses = genres.guess_genres(tracks, features, log)
    return [GenreSuggestion(t.id, t.name, t.artist, t.bpm, t.genre, guesses.get(t.id, ""))
            for t in sorted(vague, key=lambda t: (t.artist.casefold(), t.name.casefold()))], names


def save_genres(new: dict[str, str], log: Callable[[str], None] = print,
                preview: bool = False, changes: Optional[Changes] = None) -> str:
    """Write {track id: genre} into rekordbox (backed up first). Returns the library path."""
    import rbdb
    if not preview and rbdb.rekordbox_running():
        raise ValueError("rekordbox is open. Please close it first, then save again.")
    db = rbdb.open_database()
    db_path = rbdb.database_path(db)
    work = None
    if preview:
        real, (db, work) = db, rbdb.open_preview_copy(db)
        real.close()
    try:
        _, contents = rbdb.load_tracks(db)
        if not preview:
            log(f"Backed up your library to {os.path.basename(rbdb.backup(db_path))}")
        rbdb.write_genres(db, contents, new, log, changes)
        if preview:
            db.rollback()
        else:
            db.commit()
        return db_path
    finally:
        db.close()
        if work:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
