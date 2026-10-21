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

    def test_defaults_make_nothing_extra(self):
        _, tracks = organizer.load_library(self.path)
        self.assertEqual(organizer.build_plan(tracks, organizer.Options()), [])

    def test_plan(self):
        _, tracks = organizer.load_library(self.path)
        plan = dict(organizer.build_plan(tracks, organizer.Options(
            genre=True, bpm=True, key=True, year=True, rating=True, colour=True,
            date_added=True, duplicates=True, missing_info=True, never_played=True)))

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

    def test_parse_range(self):
        self.assertEqual(organizer.parse_range(""), (None, None))
        self.assertEqual(organizer.parse_range(" Any "), (None, None))
        self.assertEqual(organizer.parse_range("120-126"), (120, 126))
        self.assertEqual(organizer.parse_range("126 - 120"), (120, 126))
        self.assertEqual(organizer.parse_range("128+"), (128, None))
        self.assertEqual(organizer.parse_range("174"), (174, 174))
        for bad in ("fast", "1-2-3", "12.5"):
            with self.assertRaises(ValueError):
                organizer.parse_range(bad)

    def _custom(self, tracks, playlists, leftovers=False):
        opt = organizer.Options(custom_enabled=True, custom_leftovers=leftovers,
                                custom=[organizer.CustomPlaylist(*p) for p in playlists])
        return dict(organizer.sort_into_custom(tracks, organizer.compile_custom(opt), leftovers))

    @staticmethod
    def _tracks(*specs):
        """specs: (id, genre, bpm)"""
        return [organizer.Track(tid, f"t{tid}", "a", genre, "", bpm, "", 0, 0, "", "", 0, "")
                for tid, genre, bpm in specs]

    def test_custom_playlists_never_overlap_and_ties_go_to_first(self):
        _, tracks = organizer.load_library(self.path)
        result = self._custom(tracks, [("House", "house", "120-130"),
                                       ("Anything House", "house"),
                                       ("Fast", "", "135+")], leftovers=True)
        self.assertEqual(result["House"], ["1", "2"])
        self.assertEqual(result["Anything House"], [])          # same fit: higher row wins
        self.assertEqual(result["Fast"], ["4"])
        self.assertEqual(result["Everything Else"], ["3"])
        placed = [i for ids in result.values() for i in ids]
        self.assertEqual(sorted(placed), ["1", "2", "3", "4"])

    def test_tracks_that_fit_nothing_are_left_alone(self):
        _, tracks = organizer.load_library(self.path)
        result = self._custom(tracks, [("Fast", "", "135+")])
        self.assertEqual(result, {"Fast": ["4"]})

    def test_genre_and_bpm_must_both_match(self):
        tracks = self._tracks(("1", "Techno", 136), ("2", "Techno", 128), ("3", "House", 136),
                              ("4", "Techno", 0))
        result = self._custom(tracks, [("Techno", "techno", "130-145")])
        self.assertEqual(result["Techno"], ["1"])                # 128 too slow, house wrong genre,
                                                                 # unknown BPM can't be checked

    def test_genre_matching_is_relaxed(self):
        tracks = self._tracks(("1", "Progressive House", 128), ("2", "Tech-House", 126),
                              ("3", "Deephouse", 122), ("4", "Drum & Bass", 174),
                              ("5", "Hip-Hop/Rap", 90), ("6", "Trap", 140), ("7", "UK Rap", 92),
                              ("9", "Dance & EDM", 128))
        result = self._custom(tracks, [("House", "house"), ("DnB", "dnb"), ("Hip Hop", "hip hop")])
        self.assertEqual(result["House"], ["3", "2", "1"])       # sorted by BPM
        self.assertEqual(result["DnB"], ["4"])
        self.assertEqual(result["Hip Hop"], ["5", "7", "6"])     # trap & rap are hip hop family

    def test_rap_does_not_match_inside_other_words(self):
        tracks = self._tracks(("1", "Grapefruit Pop", 100), ("2", "Rap", 90))
        self.assertEqual(self._custom(tracks, [("Rap", "rap")]), {"Rap": ["2"]})

    def test_best_genre_fit_wins_over_list_order(self):
        tracks = self._tracks(("1", "Melodic House & Techno", 124), ("2", "House", 124),
                              ("3", "Techno", 124), ("4", "Hip Hop", 124))
        result = self._custom(tracks, [("House", "house"), ("Techno", "techno"),
                                       ("Anything 124", "", "120-130")])
        self.assertEqual(result["House"], ["2"])
        self.assertEqual(result["Techno"], ["1", "3"])           # main genre is the last word
        self.assertEqual(result["Anything 124"], ["4"])          # genre rows beat no-genre rows

    def test_custom_energy_rule(self):
        _, tracks = organizer.load_library(self.path)
        for t, e in zip(tracks, [5, 2, 0, 4]):
            t.energy = e
        result = self._custom(tracks, [("Peak", "", "", "4-5")])
        self.assertEqual(result["Peak"], ["1", "4"])
        self.assertTrue(organizer.rules_need_energy(organizer.compile_custom(organizer.Options(
            custom_enabled=True, custom=[organizer.CustomPlaylist("Peak", energy="4+")]))))

    def test_full_energy_range_means_any(self):
        rules = organizer.compile_custom(organizer.Options(custom_enabled=True, custom=[
            organizer.CustomPlaylist("House", "house", "", "1-5")]))
        self.assertFalse(organizer.rules_need_energy(rules))
        tracks = self._tracks(("1", "House", 0))                 # energy unknown
        self.assertEqual(organizer.sort_into_custom(tracks, rules, False), [("House", ["1"])])

    def test_custom_validation_messages(self):
        for playlist, text in ((organizer.CustomPlaylist("Fast", bpm="quick"), "BPM 'quick'"),
                               (organizer.CustomPlaylist("Hot", energy="9"), "1 to 5"),
                               (organizer.CustomPlaylist("Empty"), "has no rules"),
                               (organizer.CustomPlaylist("Any", energy="1-5"), "has no rules")):
            with self.assertRaises(ValueError) as ctx:
                organizer.compile_custom(organizer.Options(custom_enabled=True, custom=[playlist]))
            self.assertIn(text, str(ctx.exception))
        self.assertEqual(organizer.compile_custom(organizer.Options(
            custom=[organizer.CustomPlaylist("x", bpm="bad")])), [])   # disabled: not checked

    def test_custom_playlists_sit_directly_in_main_folder(self):
        opt = organizer.Options(custom_enabled=True,
                                custom=[organizer.CustomPlaylist("Techno", "techno")])
        out = organizer.run(opt, self.path, log=lambda m: None)
        organized = ET.parse(out).getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        self.assertEqual([(n.get("Name"), n.get("Type")) for n in organized], [("Techno", "1")])
        self.assertEqual([t.get("Key") for t in organized.find("NODE")], ["4"])

    def test_run_writes_importable_xml_and_is_rerunnable(self):
        opt = organizer.Options(genre=True)
        first_out = organizer.run(opt, self.path, log=lambda m: None)
        out = organizer.run(opt, first_out, log=lambda m: None)  # re-run on own output
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
