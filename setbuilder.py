"""Set builder: put songs in a mixable order that follows an energy curve.

Each next song should be in a compatible key (Camelot wheel), within a few
percent of the previous BPM (half/double time counts), and close to the energy
the set should have at that point - rising from the start level to the end
level (or falling, if the end is lower).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import organizer

BEAM = 6             # partial sets kept at each step; more = better order, slower
BPM_EASY = 0.03      # BPM changes up to 3% are free (pitch faders cope easily)
WEIGHTS = {"key": 1.0, "bpm": 1.0, "energy": 1.5}


@dataclass
class SetRequest:
    name: str = "My Set"
    start: int = 2        # energy level of the first song
    end: int = 5          # ...and of the last
    length: int = 20      # number of songs
    genres: str = ""      # comma-separated words; "" = any genre
    bpm: str = ""         # range like "120-128"; "" = any


def camelot(key: str) -> Optional[tuple[int, str]]:
    m = re.fullmatch(r"(\d{1,2})([AB])", key or "")
    if not m or not 1 <= int(m.group(1)) <= 12:
        return None
    return int(m.group(1)), m.group(2)


def key_cost(a: str, b: str) -> float:
    """0 = mixes perfectly; 3 = clashes. Unknown keys get a middling 1."""
    ka, kb = camelot(a), camelot(b)
    if ka is None or kb is None:
        return 1.0
    steps = min((ka[0] - kb[0]) % 12, (kb[0] - ka[0]) % 12)
    if ka[1] == kb[1]:
        # same key or one step round the wheel; two steps is a usable energy boost
        return {0: 0.0, 1: 0.0, 2: 1.0}.get(steps, 3.0)
    return {0: 0.25, 1: 1.5}.get(steps, 3.0)     # relative major/minor; diagonal


def bpm_cost(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 1.0
    change = min(abs(b / a - 1), abs(2 * b / a - 1), abs(b / (2 * a) - 1))
    return 0.0 if change <= BPM_EASY else (change - BPM_EASY) * 50   # 5% -> 1, 9% -> 3


def targets(start: int, end: int, length: int) -> list[float]:
    if length == 1:
        return [float(start)]
    return [start + (end - start) * i / (length - 1) for i in range(length)]


def check(req: SetRequest) -> None:
    """Raise ValueError with a friendly message for impossible settings."""
    if not 1 <= req.start <= 5 or not 1 <= req.end <= 5:
        raise ValueError("Set builder: energy goes from 1 to 5.")
    if not 2 <= req.length <= 200:
        raise ValueError("Set builder: a set needs between 2 and 200 songs.")
    try:
        organizer.parse_range(req.bpm)
    except ValueError:
        raise ValueError(f"Set builder: BPM '{req.bpm}' isn't a valid range. "
                         "Try 120-128, 125+ or leave it empty.") from None


def candidates(tracks: list[organizer.Track], req: SetRequest,
               guesses: Optional[dict[str, str]] = None) -> list[organizer.Track]:
    """Songs that may go in the set: analysed, matching the filters, one copy of each."""
    guesses = guesses or {}
    lo, hi = organizer.parse_range(req.bpm)
    terms = [organizer._norm_genre(w) for w in req.genres.split(",")
             if organizer._norm_genre(w)]
    best: dict[tuple, organizer.Track] = {}
    for t in tracks:
        if not t.energy:
            continue
        if (lo is not None or hi is not None) and not (
                t.bpm > 0 and (lo is None or round(t.bpm) >= lo)
                and (hi is None or round(t.bpm) <= hi)):
            continue
        if terms:
            tag = organizer._norm_genre(t.genre)
            guess = organizer._norm_genre(guesses.get(t.id, ""))
            if not any(organizer.genre_score(term, g) for term in terms
                       for g in (tag, guess) if g):
                continue
        song = (organizer._norm_title(t.artist), organizer._norm_title(t.name))
        have = best.get(song)
        # Of several copies, keep the one with the most information.
        if have is None or (bool(t.key), t.bpm > 0) > (bool(have.key), have.bpm > 0):
            best[song] = t
    return list(best.values())


def build_set(tracks: list[organizer.Track], req: SetRequest,
              guesses: Optional[dict[str, str]] = None) -> list[organizer.Track]:
    """The set, in play order. May be shorter than asked if there aren't enough songs."""
    pool = candidates(tracks, req, guesses)
    length = min(req.length, len(pool))
    if not length:
        return []
    goal = targets(req.start, req.end, length)
    rising = req.end >= req.start

    def first_cost(t):
        # Start near the start level; a rising set opens at the slower end.
        speed = t.bpm if t.bpm > 0 else 999
        return WEIGHTS["energy"] * abs(t.energy - goal[0]) + speed / 1000 * (1 if rising else -1)

    beams = [(first_cost(t), [t]) for t in pool]
    beams = sorted(beams, key=lambda b: b[0])[:BEAM]
    for step in range(1, length):
        grown = []
        for cost, chain in beams:
            used = {t.id for t in chain}
            last = chain[-1]
            options = []
            for t in pool:
                if t.id in used:
                    continue
                c = (WEIGHTS["key"] * key_cost(last.key, t.key)
                     + WEIGHTS["bpm"] * bpm_cost(last.bpm, t.bpm)
                     + WEIGHTS["energy"] * abs(t.energy - goal[step]))
                options.append((c, t))
            options.sort(key=lambda o: o[0])
            grown.extend((cost + c, chain + [t]) for c, t in options[:BEAM])
        beams = sorted(grown, key=lambda b: b[0])[:BEAM]
    return beams[0][1]


def describe(chain: list[organizer.Track], req: SetRequest) -> str:
    if not chain:
        return f"Set '{req.name}': no analysed songs match, so it wasn't made."
    clashes = sum(key_cost(a.key, b.key) >= 3 for a, b in zip(chain, chain[1:]))
    jumps = sum(bpm_cost(a.bpm, b.bpm) >= 1 for a, b in zip(chain, chain[1:]))
    bpms = [t.bpm for t in chain if t.bpm > 0]
    text = (f"Set '{req.name}': {len(chain)} songs, energy {chain[0].energy} -> "
            f"{chain[-1].energy}")
    if bpms:
        text += f", {bpms[0]:.0f} -> {bpms[-1]:.0f} BPM"
    if len(chain) < req.length:
        text += f" (only {len(chain)} songs fit; you asked for {req.length})"
    notes = []
    if clashes:
        notes.append(f"{clashes} key clash{'es' if clashes != 1 else ''}")
    if jumps:
        notes.append(f"{jumps} big BPM jump{'s' if jumps != 1 else ''}")
    return text + (". Watch for " + " and ".join(notes) + "." if notes else ".")
