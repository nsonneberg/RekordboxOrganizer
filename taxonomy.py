"""The genre taxonomy: families, their parent, subgenres and synonyms, and how
the audio models' labels map onto it.

Everything here can be changed without touching code by putting a
taxonomy.json file in the app's data folder (see load()). Example:

    {
      "families": {
        "House": {"subgenres": {"Organic House": ["organic", "downtempo house"]}},
        "Bass Music": {"parent": "Electronic", "words": ["bass music", "freeform bass"]}
      },
      "discogs": {"Electronic---Donk": "EDM"},
      "audioset": {"Disco": "House"}
    }

Families are matched by their words (see genres.family_of); a subgenre name
or any of its synonyms resolves to that subgenre. A model label maps to
"Family", "Family/Subgenre", a {"Family/Subgenre": weight} split for labels
that sit between two families, or null for labels that say nothing useful.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Optional

ELECTRONIC_PARENT = "Electronic"

DEFAULT: dict = {
    # name -> {"parent", "words" (what tags mean this family), "subgenres": {name: [synonyms]}}
    # Order matters only for ties in family_of.
    "families": {
        "House": {"parent": ELECTRONIC_PARENT,
                  "words": ["house", "garage", "ukg", "amapiano", "jackin", "bassline"],
                  "subgenres": {
                      "Deep House": [], "Tech House": [], "Progressive House": ["prog house"],
                      "Bass House": [], "Afro House": ["afro tech"], "Electro House": [],
                      "Future House": [], "Melodic House": [], "Acid House": [],
                      "Tribal House": [], "Tropical House": [], "Hard House": [],
                      "Disco House": ["nu disco house"],
                      "UK Garage": ["ukg", "2 step", "two step", "garage"],
                      "Speed Garage": [], "Bassline": ["bassline house", "niche"]}},
        "Techno": {"parent": ELECTRONIC_PARENT, "words": ["techno", "tekno", "minimal"],
                   "subgenres": {"Melodic Techno": [],
                                 "Minimal": ["minimal techno"], "Hard Techno": ["schranz"],
                                 "Peak Time Techno": ["peak time", "driving techno"],
                                 "Deep Techno": [], "Dub Techno": [], "Acid Techno": ["acid"],
                                 "Industrial Techno": []}},
        "Trance": {"parent": ELECTRONIC_PARENT, "words": ["trance", "psytrance", "psy", "goa"],
                   "subgenres": {"Progressive Trance": [],
                                 "Psytrance": ["psychedelic trance", "psy trance", "psy"],
                                 "Goa Trance": ["goa"], "Uplifting Trance": [],
                                 "Hard Trance": [], "Tech Trance": []}},
        "Drum & Bass": {"parent": ELECTRONIC_PARENT,
                        "words": ["drum and bass", "dnb", "d and b", "drum n bass", "jungle",
                                  "liquid", "neurofunk", "jump up"],
                        "subgenres": {"Liquid": ["liquid funk", "liquid dnb"], "Neurofunk": [],
                                      "Jungle": [], "Jump Up": [], "Halftime": []}},
        "Dubstep": {"parent": ELECTRONIC_PARENT,
                    "words": ["dubstep", "riddim", "brostep", "bass music", "freeform"],
                    "subgenres": {"Riddim": [], "Brostep": [],
                                  "Freeform Bass": ["freeform"], "Melodic Dubstep": []}},
        "Breaks": {"parent": ELECTRONIC_PARENT, "words": ["breaks", "breakbeat", "breakbeats"],
                   "subgenres": {"Big Beat": [], "Progressive Breaks": []}},
        "Hardstyle": {"parent": ELECTRONIC_PARENT,
                      "words": ["hardstyle", "hardcore", "hard dance", "gabber", "donk"],
                      "subgenres": {"Hardcore": [], "Gabber": [], "Happy Hardcore": [],
                                    "Rawstyle": [], "Jumpstyle": [], "Donk": []}},
        "EDM": {"parent": ELECTRONIC_PARENT,
                "words": ["edm", "big room", "electro", "dance", "future bass", "festival"],
                "subgenres": {"Big Room": [], "Future Bass": [], "Eurodance": [],
                              "Hands Up": [], "Moombahton": []}},
        "Hip Hop": {"parent": "", "words": ["hip hop", "rap", "trap", "drill", "boom bap", "grime"],
                    "subgenres": {"Trap": [], "Drill": [], "Boom Bap": [], "Grime": []}},
        "R&B": {"parent": "", "words": ["r and b", "rnb", "soul", "neo soul"], "subgenres": {}},
        "Pop": {"parent": "", "words": ["pop", "k pop"],
                "subgenres": {"Dance Pop": ["dance-pop", "dance pop"], "Synth-pop": []}},
        "Latin": {"parent": "", "words": ["latin", "reggaeton", "dembow", "salsa", "bachata",
                                          "cumbia", "merengue", "flamenco"],
                  "subgenres": {"Reggaeton": []}},
        "Afro": {"parent": "", "words": ["afrobeats", "afrobeat", "afro"], "subgenres": {}},
        "Disco & Funk": {"parent": "", "words": ["disco", "nu disco", "funk", "boogie"],
                         "subgenres": {"Nu Disco": ["nu-disco"], "Italo Disco": ["italo-disco"]}},
        "Reggae & Dancehall": {"parent": "", "words": ["reggae", "dancehall", "dub", "ska"],
                               "subgenres": {}},
        "Downtempo": {"parent": ELECTRONIC_PARENT,
                      "words": ["downtempo", "chill", "chillout", "ambient", "lofi", "lo fi",
                                "trip hop"],
                      "subgenres": {"Ambient": [], "Trip Hop": [], "Chillwave": []}},
        "Rock": {"parent": "", "words": ["rock", "indie", "metal", "punk", "alternative", "grunge"],
                 "subgenres": {}},
        "Jazz": {"parent": "", "words": ["jazz", "swing"], "subgenres": {}},
        "Country": {"parent": "", "words": ["country", "bluegrass"], "subgenres": {}},
        "Blues": {"parent": "", "words": ["blues"], "subgenres": {}},
        "Folk": {"parent": "", "words": ["folk"], "subgenres": {}},
        "Classical": {"parent": "", "words": ["classical", "opera", "orchestral"],
                      "subgenres": {}},
        "Gospel": {"parent": "", "words": ["gospel", "christian"], "subgenres": {}},
    },

    # Discogs style ("Genre---Style") -> taxonomy. Styles not listed are mapped
    # by name (genres.subgenre_of), then by their Discogs genre (DISCOGS_GENRES).
    "discogs": {
        "Electronic---House": "House", "Electronic---Deep House": "House/Deep House",
        "Electronic---Tech House": "House/Tech House",
        "Electronic---Progressive House": "House/Progressive House",
        "Electronic---Electro House": "House/Electro House",
        "Electronic---Acid House": "House/Acid House", "Electronic---Garage House": "House",
        "Electronic---Ghetto House": "House", "Electronic---Euro House": "House",
        "Electronic---Hard House": "House/Hard House", "Electronic---Italo House": "House",
        "Electronic---Tribal House": "House/Tribal House",
        "Electronic---Tropical House": "House/Tropical House", "Electronic---Hip-House": "House",
        "Electronic---Bassline": "House/Bassline", "Electronic---UK Garage": "House/UK Garage",
        "Electronic---Speed Garage": "House/Speed Garage",
        "Electronic---Techno": "Techno", "Electronic---Minimal Techno": "Techno/Minimal",
        "Electronic---Minimal": "Techno/Minimal", "Electronic---Deep Techno": "Techno/Deep Techno",
        "Electronic---Dub Techno": "Techno/Dub Techno", "Electronic---Hard Techno": "Techno/Hard Techno",
        "Electronic---Schranz": "Techno/Hard Techno", "Electronic---Acid": "Techno/Acid Techno",
        "Electronic---Trance": "Trance", "Electronic---Progressive Trance": "Trance/Progressive Trance",
        "Electronic---Psy-Trance": "Trance/Psytrance", "Electronic---Goa Trance": "Trance/Goa Trance",
        "Electronic---Hard Trance": "Trance/Hard Trance", "Electronic---Tech Trance": "Trance/Tech Trance",
        "Electronic---Drum n Bass": "Drum & Bass", "Electronic---Jungle": "Drum & Bass/Jungle",
        "Electronic---Halftime": {"Drum & Bass/Halftime": 0.5, "Dubstep": 0.5},
        "Electronic---Dubstep": "Dubstep", "Electronic---Grime": "Hip Hop/Grime",
        "Electronic---Breakbeat": "Breaks", "Electronic---Breaks": "Breaks",
        "Electronic---Big Beat": "Breaks/Big Beat", "Electronic---Progressive Breaks":
            "Breaks/Progressive Breaks", "Electronic---Broken Beat": "Breaks",
        "Electronic---Hardstyle": "Hardstyle", "Electronic---Hardcore": "Hardstyle/Hardcore",
        "Electronic---Gabber": "Hardstyle/Gabber", "Electronic---Happy Hardcore":
            "Hardstyle/Happy Hardcore", "Electronic---Jumpstyle": "Hardstyle/Jumpstyle",
        "Electronic---Donk": "Hardstyle/Donk", "Electronic---Makina": "Hardstyle",
        "Electronic---Speedcore": "Hardstyle",
        "Electronic---Electro": "EDM", "Electronic---Eurodance": "EDM/Eurodance",
        "Electronic---Hands Up": "EDM/Hands Up", "Electronic---Italodance": "EDM/Eurodance",
        "Electronic---Dance-pop": "Pop/Dance Pop", "Electronic---Synth-pop": "Pop/Synth-pop",
        "Electronic---Disco": "Disco & Funk", "Electronic---Nu-Disco": "Disco & Funk/Nu Disco",
        "Electronic---Italo-Disco": "Disco & Funk/Italo Disco", "Electronic---Euro-Disco":
            "Disco & Funk", "Electronic---Hi NRG": "Disco & Funk", "Electronic---Disco Polo": "Pop",
        "Electronic---Ambient": "Downtempo/Ambient", "Electronic---Dark Ambient": "Downtempo/Ambient",
        "Electronic---Downtempo": "Downtempo", "Electronic---Trip Hop": "Downtempo/Trip Hop",
        "Electronic---Chillwave": "Downtempo/Chillwave", "Electronic---Hip Hop": "Hip Hop",
        "Electronic---Latin": "Latin", "Electronic---Juke": "House",
        "Hip Hop---Trap": "Hip Hop/Trap", "Hip Hop---Grime": "Hip Hop/Grime",
        "Hip Hop---Drill": "Hip Hop/Drill", "Hip Hop---Boom Bap": "Hip Hop/Boom Bap",
        "Latin---Reggaeton": "Latin/Reggaeton", "Reggae---Dancehall": "Reggae & Dancehall",
        "Funk / Soul---Disco": "Disco & Funk", "Funk / Soul---Funk": "Disco & Funk",
        "Funk / Soul---Afrobeat": "Afro",
    },
    "discogs_genres": {   # Discogs top-level genre -> family, for styles not mapped above
        "Hip Hop": "Hip Hop", "Pop": "Pop", "Rock": "Rock", "Jazz": "Jazz", "Latin": "Latin",
        "Reggae": "Reggae & Dancehall", "Funk / Soul": "R&B", "Blues": "Blues",
        "Classical": "Classical", "Folk, World, & Country": "Folk",
    },

    # YAMNet / AudioSet genre classes. "Electronic music", "Electronica" and
    # "Dance music" only say it's electronic, not which style.
    "audioset": {
        "House music": "House", "Techno": "Techno", "Trance music": "Trance",
        "Drum and bass": "Drum & Bass", "Dubstep": "Dubstep", "Electronic dance music": "EDM",
        "Ambient music": "Downtempo", "New-age music": "Downtempo",
        "Hip hop music": "Hip Hop", "Pop music": "Pop", "Rhythm and blues": "R&B",
        "Soul music": "R&B", "Disco": "Disco & Funk", "Funk": "Disco & Funk",
        "Reggae": "Reggae & Dancehall", "Ska": "Reggae & Dancehall",
        "Music of Latin America": "Latin", "Salsa music": "Latin", "Flamenco": "Latin",
        "Afrobeat": "Afro", "Music of Africa": "Afro",
        "Rock music": "Rock", "Heavy metal": "Rock", "Punk rock": "Rock", "Grunge": "Rock",
        "Progressive rock": "Rock", "Rock and roll": "Rock", "Psychedelic rock": "Rock",
        "Jazz": "Jazz", "Swing music": "Jazz", "Country": "Country", "Bluegrass": "Country",
        "Blues": "Blues", "Folk music": "Folk", "Classical music": "Classical",
        "Opera": "Classical", "Gospel music": "Gospel", "Christian music": "Gospel",
        "Middle Eastern music": "World", "Music of Asia": "World", "Carnatic music": "World",
        "Music of Bollywood": "World", "Traditional music": "World",
    },

    # Usual tempo per family (secondary evidence only; half / double speed counts too).
    "typical_bpm": {
        "House": [112, 135], "Techno": [120, 150], "Trance": [125, 145],
        "Drum & Bass": [160, 182], "Dubstep": [135, 152], "Breaks": [120, 140],
        "Hardstyle": [145, 160], "EDM": [100, 150], "Hip Hop": [60, 115],
        "Downtempo": [60, 120], "Disco & Funk": [95, 130], "Reggae & Dancehall": [60, 110],
        "Latin": [85, 130], "Afro": [95, 125],
    },
}


def path() -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), "taxonomy.json")


def merge(base: dict, override: dict) -> dict:
    """The default taxonomy with a user's changes applied (see the module docstring)."""
    out = copy.deepcopy(base)
    for name, spec in (override.get("families") or {}).items():
        if not isinstance(spec, dict):
            continue
        fam = out["families"].setdefault(name, {"parent": "", "words": [name.casefold()],
                                                "subgenres": {}})
        if "parent" in spec:
            fam["parent"] = str(spec["parent"] or "")
        if "words" in spec:
            fam["words"] = [str(w) for w in spec["words"]]
        for sub, synonyms in (spec.get("subgenres") or {}).items():
            fam["subgenres"][str(sub)] = [str(s) for s in (synonyms or [])]
    for section in ("discogs", "discogs_genres", "audioset", "typical_bpm"):
        out[section].update(override.get(section) or {})
    return out


_loaded: Optional[dict] = None
problem = ""            # why the user's taxonomy.json was ignored, if it was


def load(reload: bool = False) -> dict:
    """The taxonomy in use: DEFAULT plus the user's taxonomy.json, if it's valid."""
    global _loaded, problem
    if _loaded is not None and not reload:
        return _loaded
    _loaded, problem = DEFAULT, ""
    try:
        with open(path(), encoding="utf-8") as f:
            override = json.load(f)
        if not isinstance(override, dict):
            raise ValueError("it isn't a JSON object")
        _loaded = merge(DEFAULT, override)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        problem = f"taxonomy.json was ignored ({exc})"
    return _loaded


def label_targets(target) -> list[tuple[str, str, float]]:
    """A mapping value -> [(family, subgenre, weight)]."""
    if not target:
        return []
    if isinstance(target, dict):
        items = list(target.items())
    else:
        items = [(target, 1.0)]
    out = []
    for name, weight in items:
        family, _, sub = str(name).partition("/")
        out.append((family, sub, float(weight)))
    return out
