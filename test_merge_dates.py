import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import organizer
from test_organizer import SAMPLE

QUIET = dict(log=lambda m: None)


def techno(**kw):
    return organizer.Options(custom_enabled=True, guess_genres=False,
                             custom=[organizer.CustomPlaylist("Techno", "techno")], **kw)


class DateTests(unittest.TestCase):
    def test_parse_date(self):
        for text in ("2026-10-21", "2026/10/21", "10/21/2026", " 10-21-2026 "):
            self.assertEqual(organizer.parse_date(text), "2026-10-21", text)
        self.assertEqual(organizer.parse_date(""), "")
        for bad in ("yesterday", "2026-13-01", "21/10/2026"):
            with self.assertRaises(ValueError):
                organizer.parse_date(bad)

    def test_select_tracks(self):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        self.addCleanup(os.remove, path)
        _, tracks = organizer.load_library(path)
        ids = lambda **kw: [t.id for t in organizer.select_tracks(tracks, organizer.Options(**kw))]
        self.assertEqual(ids(), ["1", "2", "3", "4"])
        self.assertEqual(ids(date_from="2024-06-01"), ["2", "3"])
        self.assertEqual(ids(date_to="2024-05-31"), ["1", "4"])
        self.assertEqual(ids(date_from="2024-05-01", date_to="2024-06-02"), ["1", "2"])
        with self.assertRaises(ValueError):
            ids(date_from="2024-06-02", date_to="2024-05-01")


class XmlMergeTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        self.out = organizer.default_output_path(self.path)
        self.addCleanup(lambda: [os.remove(p) for p in (self.path, self.out) if os.path.exists(p)])

    def organized(self, path=None):
        top = ET.parse(path or self.out).getroot().find("PLAYLISTS/NODE")
        return [n for n in top.findall("NODE") if n.get("Name", "").casefold() == "organized"]

    def test_adds_to_existing_folder_and_playlist_without_duplicates(self):
        # A previous result, then a manual edit: an extra track in "Techno",
        # and an empty sub-folder the user made themselves.
        organizer.run(techno(), self.path, **QUIET)
        tree = ET.parse(self.out)
        folder = tree.getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        folder.set("Name", "ORGANIZED")                       # different capitals
        ET.SubElement(folder.find("NODE[@Name='Techno']"), "TRACK", Key="1")
        ET.SubElement(folder, "NODE", Type="0", Name="Auto Genres", Count="0")
        tree.write(self.out, encoding="UTF-8", xml_declaration=True)

        logs = []
        second = organizer.run(techno(auto_genres=True), self.out, log=logs.append)
        self.addCleanup(os.remove, second)
        [folder] = self.organized(second)                     # still exactly one
        self.assertEqual(folder.get("Name"), "ORGANIZED")
        playlist = folder.find("NODE[@Name='Techno']")
        self.assertEqual([t.get("Key") for t in playlist], ["4", "1"])   # kept, not doubled
        self.assertEqual(playlist.get("Entries"), "2")
        [auto] = folder.findall("NODE[@Name='Auto Genres']")  # the empty one was reused
        self.assertEqual(auto.get("Count"), str(len(auto.findall("NODE"))))
        self.assertTrue(any("Adding to your existing 'ORGANIZED' folder" in m for m in logs))

    def test_replace_when_merging_is_off(self):
        organizer.run(techno(), self.path, **QUIET)
        tree = ET.parse(self.out)
        ET.SubElement(tree.getroot().find(
            "PLAYLISTS/NODE/NODE[@Name='Organized']/NODE[@Name='Techno']"), "TRACK", Key="1")
        tree.write(self.out, encoding="UTF-8", xml_declaration=True)

        second = organizer.run(techno(merge_existing=False), self.out, **QUIET)
        self.addCleanup(os.remove, second)
        [folder] = self.organized(second)
        self.assertEqual([t.get("Key") for t in folder.find("NODE[@Name='Techno']")], ["4"])

    def test_date_range_only_organizes_new_songs(self):
        opt = organizer.Options(custom_enabled=True, guess_genres=False, date_from="2024-06-01",
                                custom=[organizer.CustomPlaylist("House", "house")])
        logs = []
        organizer.run(opt, self.path, log=logs.append)
        [folder] = self.organized()
        self.assertEqual([t.get("Key") for t in folder.find("NODE[@Name='House']")], ["2"])
        self.assertTrue(any("2 of 4 tracks were added on or after 2024-06-01" in m for m in logs))

        with self.assertRaisesRegex(ValueError, "No songs were added"):
            organizer.run(techno(date_from="2030-01-01"), self.path, **QUIET)


class TagFirstTests(unittest.TestCase):
    def test_own_tag_beats_a_guess(self):
        def track(i, genre):
            return organizer.Track(id=i, name=i, artist="", genre=genre, label="", bpm=128,
                                   key="", year=0, rating=0, colour="", date_added="",
                                   play_count=0, path="")
        tracks = [track("edm", "Dance & EDM"), track("plain", ""), track("deep", "Deep House")]
        guesses = {"edm": "House", "plain": "House"}      # vague / missing tags were guessed
        rules = organizer.compile_custom(organizer.Options(custom_enabled=True, custom=[
            organizer.CustomPlaylist("House", "house"), organizer.CustomPlaylist("EDM", "edm")]))
        by_guess = set()
        lists = dict(organizer.sort_into_custom(tracks, rules, False, guesses, by_guess))
        self.assertEqual(lists, {"House": ["deep", "plain"], "EDM": ["edm"]})
        self.assertEqual(by_guess, {"plain"})

        # Without an EDM playlist the vague tag isn't used, so the guess decides.
        lists = dict(organizer.sort_into_custom(tracks, rules[:1], False, guesses))
        self.assertEqual(sorted(lists["House"]), ["deep", "edm", "plain"])


class MoveTests(unittest.TestCase):
    """Re-running moves songs to the playlist they now fit best."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        self.out = organizer.default_output_path(self.path)
        self.addCleanup(lambda: [os.remove(p) for p in (self.path, self.out) if os.path.exists(p)])
        # First run: one broad playlist by BPM only -> tracks 1, 2 (tech house) and 4 (techno).
        organizer.run(organizer.Options(custom_enabled=True, guess_genres=False, custom=[
            organizer.CustomPlaylist("House", "", "120-145")]), self.path, **QUIET)
        # By hand, the user also put track 3 (no genre, no BPM) in House.
        tree = ET.parse(self.out)
        house = tree.getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']/NODE[@Name='House']")
        ET.SubElement(house, "TRACK", Key="3")
        tree.write(self.out, encoding="UTF-8", xml_declaration=True)

    def rerun(self, **kw):
        logs = []
        opt = organizer.Options(custom_enabled=True, guess_genres=False, custom=[
            organizer.CustomPlaylist("House", "house", "120-145"),
            organizer.CustomPlaylist("Techno", "techno")], **kw)
        second = organizer.run(opt, self.out, log=logs.append)
        self.addCleanup(os.remove, second)
        folder = ET.parse(second).getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        keys = {pl.get("Name"): [t.get("Key") for t in pl.findall("TRACK")]
                for pl in folder.findall("NODE")}
        return keys, logs

    def test_moves_song_that_fits_a_new_playlist_better(self):
        keys, logs = self.rerun()
        self.assertEqual(keys["Techno"], ["4"])
        # 4 left House; 3 fits nothing, so it stays where the user put it.
        self.assertEqual(keys["House"], ["1", "2", "3"])
        self.assertTrue(any("Moved 1 song" in m for m in logs))
        self.assertTrue(any("Song D: House -> Techno" in m for m in logs))

    def test_moving_can_be_turned_off(self):
        keys, _ = self.rerun(move_mismatched=False)
        self.assertEqual(keys["House"], ["1", "2", "4", "3"])
        self.assertEqual(keys["Techno"], ["4"])

    def test_only_songs_in_the_chosen_dates_move(self):
        keys, _ = self.rerun(date_from="2024-01-01")   # track 4 was added 2023-01-01
        self.assertIn("4", keys["House"])

    def test_leaves_everything_else_when_it_fits_somewhere(self):
        tree = ET.parse(self.out)
        folder = tree.getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        rest = ET.SubElement(folder, "NODE", Name=organizer.LEFTOVERS_NAME, Type="1",
                             KeyType="0", Entries="2")
        for key in ("3", "4"):
            ET.SubElement(rest, "TRACK", Key=key)
        tree.write(self.out, encoding="UTF-8", xml_declaration=True)
        keys, _ = self.rerun(custom_leftovers=True)
        self.assertEqual(keys[organizer.LEFTOVERS_NAME], ["3"])


if __name__ == "__main__":
    unittest.main()
