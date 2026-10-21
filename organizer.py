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
    auto_genres: bool = False    # auto genre detection: a folder per genre family
    energy: bool = False         # listen to the music and score energy 1-5
    energy_ratings: bool = True  # ...and set the star rating to the energy level
    keep_ratings: bool = True    # ...except on tracks that already have a rating
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


# [(folder, [(playlist, [track ids])])]. folder is a name, TOP_LEVEL (straight into the
# main folder), or a tuple of names for nested folders, e.g. ("Auto Genres", "House").
Plan = list[tuple[object, list[tuple[str, list[str]]]]]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

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


LEFTOVERS_NAME = "Everything Else"
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
                     guesses: Optional[dict[str, str]] = None):
    """Put each track in the playlist it fits best; tracks that fit none are left alone.

    guesses: {track id: guessed genre} for songs with no or a vague tag.
    Returns [(name, [ids])]. Ties go to the playlist higher in the list.
    """
    guesses = guesses or {}
    lists: list[tuple[str, list[str]]] = [(r.name, []) for r in rules]
    rest: list[str] = []
    for t in sorted(tracks, key=lambda t: (t.bpm, t.artist.casefold(), t.name.casefold())):
        genre = _norm_genre(guesses.get(t.id, t.genre))
        best_i, best_score = None, None
        for i, rule in enumerate(rules):
            s = rule.score(t, genre)
            if s is not None and (best_score is None or s > best_score):
                best_i, best_score = i, s
        if best_i is None:
            rest.append(t.id)
        else:
            lists[best_i][1].append(t.id)
    if leftovers and rest:
        lists.append((LEFTOVERS_NAME, rest))
    return lists


def build_plan(tracks: list[Track], opt: Options, rules: Optional[list[_Rule]] = None,
               guesses: Optional[dict[str, str]] = None) -> Plan:
    plan: Plan = []

    if rules is None:
        rules = compile_custom(opt)
    if rules:
        plan.append((TOP_LEVEL, sort_into_custom(tracks, rules, opt.custom_leftovers, guesses)))

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
        seen: dict = defaultdict(list)
        for t in tracks:
            if t.name:
                seen[(_norm_title(t.artist), _norm_title(t.name))].append(t)
        dupes = [t for k in sorted(seen) if len(seen[k]) > 1 for t in seen[k]]
        if dupes:
            cleanup.append(("Possible Duplicates", [t.id for t in dupes]))
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


def write_library(tree: ET.ElementTree, plan: Plan, folder_name: str, out_path: str,
                  ratings: Optional[dict[str, int]] = None) -> None:
    root = tree.getroot()
    if ratings:
        for el in root.iterfind("COLLECTION/TRACK"):
            stars = ratings.get(el.get("TrackID", ""))
            if stars is not None:
                el.set("Rating", str(stars * 51))


    playlists = root.find("PLAYLISTS")
    if playlists is None:
        playlists = ET.SubElement(root, "PLAYLISTS")
    top = playlists.find("NODE")
    if top is None:
        top = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="0")

    # Replace a folder left over from a previous run.
    for node in top.findall("NODE"):
        if node.get("Name") == folder_name:
            top.remove(node)

    organized = _folder(top, folder_name)
    folders = {(): organized}

    def folder_for(path: tuple) -> ET.Element:
        if path not in folders:
            folders[path] = _folder(folder_for(path[:-1]), path[-1])
        return folders[path]

    for folder_title, lists in plan:
        sub = folder_for(folder_path(folder_title))
        for title, ids in lists:
            pl = ET.SubElement(sub, "NODE", Name=title, Type="1", KeyType="0",
                               Entries=str(len(ids)))
            for track_id in ids:
                ET.SubElement(pl, "TRACK", Key=track_id)
    for node in folders.values():
        _set_count(node)
    _set_count(top)

    tree.write(out_path, encoding="UTF-8", xml_declaration=True)


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
                 log: Callable[[str], None]) -> dict[str, int]:
    """Set Track.energy from the measurements; return {track id: new star rating}."""
    import energy

    levels = energy.levels_from_features(features, {t.id: t.bpm for t in tracks})
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
    return ratings


def _organize(tracks: list[Track], opt: Options, log: Callable[[str], None],
              progress: Optional[Callable[[int, int], None]]) -> tuple[Plan, dict[str, int], int]:
    """Shared by both modes: (plan, new star ratings, number of playlists)."""
    rules = compile_custom(opt)          # validate before any slow work
    want_energy = opt.energy or rules_need_energy(rules)
    want_guesses = opt.guess_genres and (opt.auto_genres or any(r.terms for r in rules))

    features: dict[str, dict] = {}
    if want_energy or want_guesses:
        if want_energy and not opt.energy:
            log("Your playlist rules use energy, so the app will measure your music.")
        features = analyse_tracks(tracks, log, progress)
    ratings = apply_energy(tracks, opt, features, log) if want_energy else {}
    guesses: dict[str, str] = {}
    if want_guesses:
        import genres
        guesses = genres.guess_genres(tracks, features, log)

    plan = build_plan(tracks, opt, rules, guesses)
    if not plan and not ratings:
        raise ValueError("Nothing to do - tick at least one option.")

    total = 0
    shown_parents = set()
    for folder, lists in plan:
        path = folder_path(folder)
        if path == ():
            placed = sum(len(ids) for name, ids in lists if name != LEFTOVERS_NAME)
            used = sum(1 for name, ids in lists if name != LEFTOVERS_NAME
                       for i in ids if i in guesses)
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
        total += len(lists)
    return plan, ratings, total


def run(opt: Options, xml_path: Optional[str] = None, log: Callable[[str], None] = print,
        progress: Optional[Callable[[int, int], None]] = None) -> str:
    """Do the whole job.

    With xml_path, reads that export and writes a new XML next to it; returns its path.
    Without, works directly on rekordbox's database; returns the database path.
    """
    folder_name = opt.folder_name.strip() or "Organized"
    if xml_path:
        log(f"Reading {os.path.basename(xml_path)} ...")
        tree, tracks = load_library(xml_path)
        log(f"Found {len(tracks)} tracks.")
        if not tracks:
            raise ValueError("The collection in this XML file is empty.")
        plan, ratings, total = _organize(tracks, opt, log, progress)
        out_path = default_output_path(xml_path)
        write_library(tree, plan, folder_name, out_path, ratings)
        log(f"Created {total} playlists.")
        log(f"Saved: {out_path}")
        return out_path

    import rbdb
    if rbdb.rekordbox_running():
        raise ValueError("rekordbox is open. Please close it first, then press Run again.")
    log("Opening your rekordbox library ...")
    db = rbdb.open_database()
    try:
        db_path = rbdb.database_path(db)
        tracks, contents = rbdb.load_tracks(db)
        log(f"Found {len(tracks)} tracks.")
        if not tracks:
            raise ValueError("Your rekordbox collection is empty.")
        plan, ratings, total = _organize(tracks, opt, log, progress)

        if rbdb.rekordbox_running():
            raise ValueError("rekordbox was opened while working. Nothing was changed - "
                             "close rekordbox and press Run again.")
        log(f"Backed up your library to {os.path.basename(rbdb.backup(db_path))}")
        rbdb.write_changes(db, contents, plan, folder_name, ratings, log)
        log(f"Created {total} playlists in rekordbox.")
        return db_path
    finally:
        db.close()
