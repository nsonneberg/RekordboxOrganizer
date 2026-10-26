import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import organizer
import setbuilder
from setbuilder import SetRequest, bpm_cost, build_set, key_cost
from test_organizer import SAMPLE


def track(i, energy, key="8A", bpm=124.0, name=None, artist="DJ", genre="House"):
    return organizer.Track(str(i), name or f"Song {i}", artist, genre, "", bpm, key, 0, 0, "",
                           "2026-01-01", 0, "", energy=energy)


class RuleTests(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(key_cost("8A", "8A"), 0)
        self.assertEqual(key_cost("8A", "9A"), 0)
        self.assertEqual(key_cost("12A", "1A"), 0)     # round the wheel
        self.assertEqual(key_cost("8A", "8B"), 0.25)   # relative major
        self.assertEqual(key_cost("8A", "3A"), 3)      # clash
        self.assertEqual(key_cost("", "8A"), 1)        # unknown

    def test_bpm(self):
        self.assertEqual(bpm_cost(124, 126), 0)
        self.assertEqual(bpm_cost(87, 174), 0)         # double time
        self.assertGreater(bpm_cost(124, 140), 3)

    def test_bad_settings(self):
        for req in (SetRequest(start=0), SetRequest(length=1), SetRequest(bpm="fast")):
            with self.assertRaises(ValueError):
                setbuilder.check(req)


class BuildTests(unittest.TestCase):
    def test_follows_the_energy_curve(self):
        tracks = [track(i, e) for i, e in enumerate([5, 1, 3, 2, 4, 3, 2, 4, 5, 1])]
        chain = build_set(tracks, SetRequest(start=1, end=5, length=5))
        self.assertEqual([t.energy for t in chain], [1, 2, 3, 4, 5])
        down = build_set(tracks, SetRequest(start=5, end=1, length=5))
        self.assertEqual([t.energy for t in down], [5, 4, 3, 2, 1])

    def test_prefers_mixable_keys_and_tempos(self):
        tracks = [track(1, 3, "8A", 124), track(2, 3, "3B", 124),   # clashing key
                  track(3, 3, "9A", 125), track(4, 3, "8A", 150)]   # big tempo jump
        chain = build_set(tracks, SetRequest(start=3, end=3, length=2))
        self.assertEqual({t.id for t in chain}, {"1", "3"})

    def test_one_copy_of_each_song_and_filters(self):
        tracks = [track(1, 2, name="Same"), track(2, 2, name="Same", key=""),
                  track(3, 3, genre="Techno", bpm=135), track(4, 4)]
        chain = build_set(tracks, SetRequest(start=2, end=4, length=10))
        self.assertEqual([t.id for t in chain], ["1", "3", "4"])     # the copy with a key
        house = build_set(tracks, SetRequest(start=2, end=4, length=10, genres="house"))
        self.assertEqual([t.id for t in house], ["1", "4"])
        slow = build_set(tracks, SetRequest(start=2, end=4, length=10, bpm="120-130"))
        self.assertNotIn("3", [t.id for t in slow])
        self.assertEqual(build_set([track(5, 0)], SetRequest()), [])   # not analysed

    def test_describe(self):
        chain = [track(1, 2, "8A", 124), track(2, 5, "3A", 124)]
        text = setbuilder.describe(chain, SetRequest(name="Friday", length=2))
        self.assertIn("Set 'Friday': 2 songs, energy 2 -> 5", text)
        self.assertIn("1 key clash", text)


class WriteTests(unittest.TestCase):
    def test_set_is_rewritten_in_order(self):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        out = path + ".out.xml"
        self.addCleanup(lambda: [os.remove(p) for p in (path, out) if os.path.exists(p)])

        def write(src, ids):
            changes = organizer.Changes()
            organizer.write_library(ET.parse(src), [(organizer.SETS_FOLDER, [("Friday", ids)])],
                                    "Organized", out, log=lambda m: None,
                                    evict=organizer.Evictor({}, [], set(), True),
                                    changes=changes)
            node = ET.parse(out).getroot().find(
                "PLAYLISTS/NODE/NODE[@Name='Organized']/NODE[@Name='Sets']/NODE[@Name='Friday']")
            return [t.get("Key") for t in node.findall("TRACK")], changes

        first, changes = write(path, ["3", "1", "2"])
        self.assertEqual(first, ["3", "1", "2"])
        self.assertFalse(changes.sets[0][2])
        again, changes = write(out, ["2", "4"])
        self.assertEqual(again, ["2", "4"])
        self.assertTrue(changes.sets[0][2])
        self.assertIn("SET: Organized / Sets / Friday (2 songs, replacing",
                      organizer.summarize_changes(changes))


if __name__ == "__main__":
    unittest.main()
