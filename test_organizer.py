import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import organizer

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.8.0" Company="AlphaTheta"/>
  <COLLECTION Entries="4">
    <TRACK TrackID="1" Name="Song A" Artist="DJ One" Genre="Tech House" AverageBpm="126.00"
           Tonality="8A" Year="2023" Rating="255" Colour="0xFF0000" DateAdded="2024-05-01"
           PlayCount="3" Label="Lbl" Location="file://localhost/C:/Music/a%20b.mp3"/>
    <TRACK TrackID="2" Name="Song A (Original Mix)" Artist="dj one" Genre="tech house"
           AverageBpm="128.40" Tonality="Am" Year="2022" Rating="0" DateAdded="2024-06-02"
           PlayCount="0" Location="file://localhost/Users/me/Music/a.mp3"/>
    <TRACK TrackID="3" Name="Song C" Artist="Other" Genre="" AverageBpm="0" Tonality=""
           Year="0" Rating="102" DateAdded="2024-06-10" PlayCount="1" Location=""/>
    <TRACK TrackID="4" Name="Song D" Artist="Other" Genre="Techno" AverageBpm="140"
           Tonality="F#" Year="2021" Rating="153" DateAdded="2023-01-01" PlayCount="2" Location=""/>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="1">
      <NODE Name="Mine" Type="1" KeyType="0" Entries="1"><TRACK Key="1"/></NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
"""


class OrganizerTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SAMPLE)

    def tearDown(self):
        for p in (self.path, organizer.default_output_path(self.path)):
            if os.path.exists(p):
                os.remove(p)

    def test_location_to_path(self):
        self.assertEqual(organizer.location_to_path("file://localhost/C:/Music/a%20b.mp3"),
                         os.path.normpath("C:/Music/a b.mp3"))
        self.assertEqual(organizer.location_to_path("file://localhost/Users/me/x.mp3"),
                         os.path.normpath("/Users/me/x.mp3"))

    def test_keys(self):
        self.assertEqual(organizer.to_camelot("Am"), "8A")
        self.assertEqual(organizer.to_camelot("8a"), "8A")
        self.assertEqual(organizer.to_camelot("F#"), "2B")
        self.assertEqual(organizer.to_camelot("nonsense"), "")

    def test_plan(self):
        _, tracks = organizer.load_library(self.path)
        plan = dict(organizer.build_plan(tracks, organizer.Options(
            year=True, rating=True, colour=True, date_added=True, never_played=True)))

        self.assertEqual(plan["By Genre"],
                         [("Tech House", ["1", "2"]), ("Techno", ["4"]), ("No Genre", ["3"])])
        self.assertEqual(plan["By BPM"],
                         [("125-129 BPM", ["1", "2"]), ("140-144 BPM", ["4"])])
        self.assertEqual(plan["By Key"], [("2B (F#)", ["4"]), ("8A (Am)", ["1", "2"])])
        self.assertEqual([n for n, _ in plan["By Year"]], ["2023", "2022", "2021", "Unknown Year"])
        self.assertEqual([n for n, _ in plan["By Rating"]],
                         ["5 Stars", "3 Stars", "2 Stars", "Unrated"])
        self.assertEqual(plan["By Color"], [("Red", ["1"])])
        self.assertEqual([n for n, _ in plan["By Date Added"]], ["2024-06", "2024-05", "2023-01"])

        cleanup = dict(plan["Cleanup"])
        self.assertEqual(cleanup["Possible Duplicates"], ["1", "2"])
        self.assertEqual(cleanup["Missing Genre"], ["3"])
        self.assertEqual(cleanup["Never Played"], ["2"])

    def test_artist_min_tracks(self):
        _, tracks = organizer.load_library(self.path)
        plan = dict(organizer.build_plan(tracks, organizer.Options(artist=True, min_tracks=2)))
        self.assertEqual([n for n, _ in plan["By Artist"]], ["DJ One", "Other"])

    def test_run_writes_importable_xml_and_is_rerunnable(self):
        first_out = organizer.run(self.path, organizer.Options(), log=lambda m: None)
        out = organizer.run(first_out, organizer.Options(), log=lambda m: None)  # re-run on own output
        self.addCleanup(os.remove, out)

        root = ET.parse(out).getroot()
        top = root.find("PLAYLISTS/NODE")
        names = [n.get("Name") for n in top.findall("NODE")]
        self.assertEqual(names, ["Mine", "Organized"])
        self.assertEqual(top.get("Count"), "2")
        self.assertEqual(len(root.findall("COLLECTION/TRACK")), 4)

        genre = top.find("NODE[@Name='Organized']/NODE[@Name='By Genre']")
        first = genre.find("NODE")
        self.assertEqual(first.get("Entries"), str(len(first.findall("TRACK"))))


if __name__ == "__main__":
    unittest.main()
