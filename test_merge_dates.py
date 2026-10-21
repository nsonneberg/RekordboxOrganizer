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


if __name__ == "__main__":
    unittest.main()
