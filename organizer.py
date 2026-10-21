"""Core logic: read a rekordbox XML export, build organized playlists, write a new XML.

The music files themselves are never touched. The output is a rekordbox XML file
containing the full collection plus a new playlist folder that can be imported
back into rekordbox.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
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
_COLOUR_ORDER = ["Pink", "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple"]


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


@dataclass
class Options:
    genre: bool = True
    bpm: bool = True
    bpm_step: int = 5
    key: bool = True
    artist: bool = False
    label: bool = False
    min_tracks: int = 3          # minimum tracks for Artist / Label playlists
    year: bool = False
    rating: bool = False
    colour: bool = False
    date_added: bool = False
    duplicates: bool = True
    missing_files: bool = True
    missing_info: bool = True
    never_played: bool = False
    folder_name: str = "Organized"


Plan = list[tuple[str, list[tuple[str, list[str]]]]]  # [(folder, [(playlist, [track ids])])]


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


def build_plan(tracks: list[Track], opt: Options) -> Plan:
    plan: Plan = []

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
            tracks, lambda t: (_COLOUR_ORDER.index(t.colour), t.colour) if t.colour else None)))

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

def _folder(parent: ET.Element, name: str) -> ET.Element:
    return ET.SubElement(parent, "NODE", Type="0", Name=name, Count="0")


def _set_count(folder: ET.Element) -> None:
    folder.set("Count", str(len(folder.findall("NODE"))))


def write_library(tree: ET.ElementTree, plan: Plan, folder_name: str, out_path: str) -> None:
    root = tree.getroot()
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
    for folder_title, lists in plan:
        sub = _folder(organized, folder_title)
        for title, ids in lists:
            pl = ET.SubElement(sub, "NODE", Name=title, Type="1", KeyType="0",
                               Entries=str(len(ids)))
            for track_id in ids:
                ET.SubElement(pl, "TRACK", Key=track_id)
        _set_count(sub)
    _set_count(organized)
    _set_count(top)

    tree.write(out_path, encoding="UTF-8", xml_declaration=True)


def default_output_path(xml_path: str) -> str:
    base, _ = os.path.splitext(xml_path)
    return base + "_organized.xml"


def run(xml_path: str, opt: Options, log: Callable[[str], None] = print) -> str:
    """Do the whole job. Returns the path of the written XML file."""
    log(f"Reading {os.path.basename(xml_path)} ...")
    tree, tracks = load_library(xml_path)
    log(f"Found {len(tracks)} tracks.")
    if not tracks:
        raise ValueError("The collection in this XML file is empty.")

    plan = build_plan(tracks, opt)
    if not plan:
        raise ValueError("Nothing to create - tick at least one option.")
    total = 0
    for folder, lists in plan:
        log(f"  {folder}: {len(lists)} playlist{'s' if len(lists) != 1 else ''}")
        total += len(lists)

    out_path = default_output_path(xml_path)
    write_library(tree, plan, opt.folder_name.strip() or "Organized", out_path)
    log(f"Created {total} playlists.")
    log(f"Saved: {out_path}")
    return out_path
