import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import genres
import organizer
import test_organizer


def track(tid, genre, bpm):
    return organizer.Track(tid, f"t{tid}", "a", genre, "", bpm, "", 0, 0, "", "", 0, "")


def sound(lows, highs=0.2, brightness=1.0, flux=1.0, loudness=15.0):
    return {"lows": lows, "highs": highs, "brightness": brightness, "flux": flux,
            "loudness": loudness}


class FamilyTests(unittest.TestCase):
    def test_family_of(self):
        cases = {"Progressive House": "House", "Tech-House": "House", "Deep House": "House",
                 "Melodic House & Techno": "Techno", "Drum & Bass": "Drum & Bass",
                 "Hip-Hop/Rap": "Hip Hop", "Trap": "Hip Hop", "Dance & EDM": "EDM",
                 "Loop Samples": "Loop Samples", "": genres.UNKNOWN}
        for tag, family in cases.items():
            self.assertEqual(genres.family_of(tag), family, tag)

    def test_is_vague(self):
        for tag in ("", "Electronic", "Dance & EDM", "  dance/electronic ", "Other"):
            self.assertTrue(genres.is_vague(tag), tag)
        for tag in ("House", "Progressive House", "Hip Hop"):
            self.assertFalse(genres.is_vague(tag), tag)


class GuessTests(unittest.TestCase):
    def test_rules_when_too_few_examples(self):
        tracks = [track("1", "", 174), track("2", "", 124), track("3", "Dance & EDM", 95),
                  track("4", "", 90), track("5", "House", 125), track("6", "", 0)]
        logs = []
        guesses = genres.guess_genres(tracks, {}, logs.append)
        self.assertEqual(guesses, {"1": "Drum & Bass", "2": "House", "3": "EDM", "4": "Hip Hop"})
        self.assertIn("from BPM and sound", logs[-1])

    def test_learns_from_tagged_songs(self):
        tracks, feats = [], {}
        for i in range(4):   # tagged examples: two clear clusters
            tracks.append(track(f"h{i}", "Tech House", 124 + i))
            feats[f"h{i}"] = sound(lows=0.50, highs=0.20)
            tracks.append(track(f"t{i}", "Techno", 132 + i))
            feats[f"t{i}"] = sound(lows=0.20, highs=0.60, brightness=0.6)
        # Sounds like the techno examples although the BPM is in house range.
        tracks.append(track("x", "Electronic", 131))
        feats["x"] = sound(lows=0.21, highs=0.58, brightness=0.6)
        logs = []
        guesses = genres.guess_genres(tracks, feats, logs.append)
        self.assertEqual(guesses, {"x": "Techno"})
        self.assertIn("1 learned from your tagged songs", logs[-1])

    def test_effective_genre(self):
        guesses = {"1": "House"}
        self.assertEqual(genres.effective_genre(track("1", "", 124), guesses), "House")
        self.assertEqual(genres.effective_genre(track("2", "Techno", 124), guesses), "Techno")


class AutoGenreTests(unittest.TestCase):
    def test_folder_per_family(self):
        tracks = [track("1", "Progressive House", 128), track("2", "House", 124),
                  track("3", "house", 126), track("4", "", 122), track("5", "Drum & Bass", 174),
                  track("6", "", 0), track("7", "Loop Samples", 120)]
        plan = dict(genres.auto_genre_plan(tracks, {"4": "House"}))
        self.assertEqual(list(plan), [("Auto Genres", "Drum & Bass"), ("Auto Genres", "House"),
                                      ("Auto Genres", "Loop Samples"),
                                      ("Auto Genres", genres.UNKNOWN)])
        self.assertEqual(plan[("Auto Genres", "House")], [
            ("All House", ["4", "2", "3", "1"]),
            ("House", ["2", "3"]),                 # "House" and "house" merged
            ("Progressive House", ["1"]),
            ("House (detected)", ["4"]),           # guesses listed last, clearly marked
        ])
        self.assertEqual(plan[("Auto Genres", "Drum & Bass")], [("Drum & Bass", ["5"])])
        self.assertEqual(plan[("Auto Genres", genres.UNKNOWN)], [("No genre", ["6"])])


class IntegrationTests(unittest.TestCase):
    def test_custom_playlists_use_guessed_genres(self):
        tracks = [track("1", "Dance & EDM", 125), track("2", "", 174), track("3", "Techno", 125)]
        rules = organizer.compile_custom(organizer.Options(custom_enabled=True, custom=[
            organizer.CustomPlaylist("House", "house", "118-130")]))
        guesses = genres.guess_genres(tracks, {}, lambda m: None)
        self.assertEqual(organizer.sort_into_custom(tracks, rules, False, guesses),
                         [("House", ["1"])])               # tagged Techno stays out
        self.assertEqual(organizer.sort_into_custom(tracks, rules, False),
                         [("House", [])])                  # guessing off: tags only

    def test_xml_run_writes_nested_genre_folders(self):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(test_organizer.SAMPLE)
        out = organizer.default_output_path(path)
        self.addCleanup(lambda: [os.remove(p) for p in (path, out) if os.path.exists(p)])

        organizer.run(organizer.Options(auto_genres=True), path, log=lambda m: None)
        organized = ET.parse(out).getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        auto = organized.find("NODE[@Name='Auto Genres']")
        self.assertEqual(auto.get("Type"), "0")
        families = [n.get("Name") for n in auto]
        self.assertEqual(families, ["House", "Techno", genres.UNKNOWN])   # untagged, no BPM
        self.assertEqual(auto.get("Count"), "3")
        house = auto.find("NODE[@Name='House']")
        self.assertEqual([(n.get("Name"), n.get("Entries")) for n in house], [("Tech House", "2")])


if __name__ == "__main__":
    unittest.main()
