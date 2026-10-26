"""Hot cues at phrase changes, from rekordbox's own phrase analysis.

rekordbox splits each analysed track into phrases (its PSSI analysis tag) and
labels them by the track's mood:
  high (most dance music): Intro, Up, Down, Chorus, Outro
  mid / low:               Intro, Verse 1-6, Bridge, Chorus, Outro
Each phrase starts on a beat number; the beat grid (PQTZ tag) turns that into
a time. A hot cue goes where each new section begins, i.e. at the end of the
phrase before it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# PSSI phrase kind -> label, per mood (1 = high, 2 = mid, 3 = low)
_VERSES = {kind: "Verse" for kind in range(2, 8)}
LABELS = {
    1: {1: "Intro", 2: "Build-up", 3: "Breakdown", 5: "Drop", 6: "Outro"},
    2: {1: "Intro", **_VERSES, 8: "Bridge", 9: "Chorus", 10: "Outro"},
    3: {1: "Intro", **_VERSES, 8: "Bridge", 9: "Chorus", 10: "Outro"},
}
# With more phrases than free pads, the most useful ones win.
PRIORITY = {"Drop": 5, "Chorus": 5, "Build-up": 4, "Breakdown": 3, "Bridge": 3,
            "Intro": 2, "Outro": 2, "Verse": 1}
PADS = "ABCDEFGH"
MIN_GAP_MS = 1000   # don't put a cue this close to one that's already there


@dataclass
class Phrase:
    start_ms: int
    label: str


def read_phrases(ext_path: str) -> list[Phrase]:
    """The track's sections in order, or [] if rekordbox hasn't phrase-analysed it.

    A run of phrases with the same label (e.g. Chorus, Chorus) counts as one
    section, starting where the run starts.
    """
    return read_structure(ext_path)[1]


def read_structure(ext_path: str) -> tuple[int, list[Phrase]]:
    """(rekordbox's mood: 1 high, 2 mid, 3 low, 0 unknown; the sections, as read_phrases)."""
    from pyrekordbox import AnlzFile

    dat_path = os.path.splitext(ext_path)[0] + ".DAT"
    if not ext_path or not os.path.exists(ext_path) or not os.path.exists(dat_path):
        return 0, []
    ext = AnlzFile.parse_file(ext_path)
    if "PSSI" not in ext.tag_types:
        return 0, []
    dat = AnlzFile.parse_file(dat_path)
    if "PQTZ" not in dat.tag_types:
        return 0, []
    beats = [int(b.time) for b in dat.get_tag("PQTZ").content.entries]   # ms per beat
    pssi = ext.get_tag("PSSI").content
    names = LABELS.get(int(pssi.mood), LABELS[2])

    phrases: list[Phrase] = []
    for entry in sorted(pssi.entries, key=lambda e: e.beat):
        label = names.get(int(entry.kind))
        if label is None or not 1 <= entry.beat <= len(beats):
            continue
        if phrases and phrases[-1].label == label:
            continue
        phrases.append(Phrase(beats[entry.beat - 1], label))
    mood = int(pssi.mood)
    return (mood if mood in LABELS else 0), phrases


def choose_hot_cues(phrases: list[Phrase], used_pads: set[int],
                    existing_ms: list[int]) -> list[tuple[int, Phrase]]:
    """[(pad 0-7, phrase)] for the empty pads, in time order.

    Pads that already hold a cue are never touched, and nothing is added
    within MIN_GAP_MS of an existing cue (so running again adds nothing new).
    """
    free = [pad for pad in range(len(PADS)) if pad not in used_pads]
    wanted = [p for p in phrases
              if all(abs(p.start_ms - ms) >= MIN_GAP_MS for ms in existing_ms)]
    if len(wanted) > len(free):
        best = sorted(wanted, key=lambda p: (-PRIORITY.get(p.label, 0), p.start_ms))
        wanted = sorted(best[:len(free)], key=lambda p: p.start_ms)
    return list(zip(free, wanted))
