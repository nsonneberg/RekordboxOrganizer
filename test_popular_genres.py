"""Popular-genre playlists: a playlist per genre tag with more than 25 tracks.

All tracks here are synthetic; no real library is read.
"""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock
from xml.sax.saxutils import quoteattr

import genres
import organizer
import rbdb
import taxonomy
import test_library_safety as safety

FOLDER = genres.POPULAR_FOLDER


def track(tid, genre, bpm=124.0):
    return organizer.Track(str(tid), f"Song {tid}", "Artist", genre, "", bpm, "", 0, 0, "", "",
                           0, "")


def library(*groups):
    """groups: (genre tag or [spellings used in turn], how many songs)."""
    out, n = [], 0
    for tags, count in groups:
        tags = [tags] if isinstance(tags, str) else tags
        for i in range(count):
            n += 1
            out.append(track(n, tags[i % len(tags)]))
    return out


def playlists(plan):
    """{(family, playlist): number of songs} for the popular-genre part of a plan."""
    return {(folder[1], name): len(ids) for folder, lists in plan
            if isinstance(folder, tuple) and folder[0] == FOLDER for name, ids in lists}


class PopularGenrePlanTests(unittest.TestCase):
    def test_more_than_25_strictly(self):
        tracks = library(("Techno", 26), ("Trance", 25), ("Deep House", 26), ("Tech House", 25))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("Techno", "Techno"): 26, ("House", "Deep House"): 26})

    def test_spelling_case_and_whitespace_count_as_one_genre(self):
        spellings = ["Tech House", "tech house", "TECH-HOUSE", "  Tech   House ", "Tech/House"]
        tracks = library((spellings, 26))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("House", "Tech House"): 26})

    def test_none_of_the_spellings_qualifies_alone(self):
        # 13 + 13 is only 26 once the spellings are merged.
        tracks = library(("Afro House", 13), ("afro-house", 13))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("House", "Afro House"): 26})

    def test_taxonomy_synonyms_and_family_aliases(self):
        tracks = library((["UKG", "UK Garage", "2-step"], 26),           # subgenre synonyms
                         (["DnB", "Drum & Bass", "drum n bass", "Drum and Bass"], 26),  # aliases
                         (["prog house", "Progressive House"], 26))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("House", "UK Garage"): 26, ("House", "Progressive House"): 26,
                          ("Drum & Bass", "Drum & Bass"): 26})

    def test_subgenres_get_their_own_playlists_inside_the_family(self):
        tracks = library(("House", 30), ("Melodic House", 27), ("Tech House", 40),
                         ("Afro House", 26))
        plan = genres.popular_genre_plan(tracks)
        self.assertEqual([folder for folder, _ in plan], [(FOLDER, "House")])
        self.assertEqual([(name, len(ids)) for name, ids in plan[0][1]],
                         [("House", 30), ("Afro House", 26), ("Melodic House", 27),
                          ("Tech House", 40)])

    def test_plain_family_playlist_holds_only_plain_tags(self):
        tracks = library(("House", 26), ("Tech House", 30), ("Deep House", 10))
        plan = genres.popular_genre_plan(tracks)
        lists = dict(plan[0][1])
        house_ids = set(lists["House"])
        self.assertEqual(house_ids, {t.id for t in tracks if t.genre == "House"})
        self.assertNotIn("All House", lists)
        # Deep House (10) doesn't qualify and is not folded into House either.
        self.assertEqual(sorted(lists), ["House", "Tech House"])

    def test_subgenres_never_make_a_broad_family_playlist(self):
        # 20 + 20 + 20 House-family songs, but no single genre has more than 25.
        tracks = library(("Tech House", 20), ("Afro House", 20), ("Melodic House", 20))
        self.assertEqual(genres.popular_genre_plan(tracks), [])

    def test_family_without_its_own_tag_still_gets_a_folder_for_subgenres(self):
        tracks = library(("Melodic Techno", 26), ("Techno", 3))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("Techno", "Melodic Techno"): 26})

    def test_unfamiliar_genres_keep_their_own_name(self):
        self.assertEqual(genres.family_of("Shoegaze"), "Shoegaze")   # fits no family
        tracks = library((["Shoegaze", "shoegaze ", "SHOEGAZE"], 26))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("Shoegaze", "Shoegaze"): 26})

    def test_specific_tags_outside_the_taxonomy_stay_separate(self):
        tracks = library(("Melodic House & Techno", 26), ("Melodic House and Techno", 1),
                         ("Melodic House", 26))
        found = playlists(genres.popular_genre_plan(tracks))
        self.assertEqual(sorted(name for _, name in found),
                         ["Melodic House", "Melodic House & Techno"])
        self.assertIn(27, found.values())   # "&" and "and" are the same genre

    def test_family_words_keep_their_own_name(self):
        # "Disco" belongs to the "Disco & Funk" family but isn't the whole family.
        tracks = library(("Disco", 26), ("Disco & Funk", 26), ("Reggae", 26),
                         ("Afrobeats", 26), (["Hip-Hop", "hip hop", "HipHop"], 26))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("Disco & Funk", "Disco & Funk"): 26, ("Disco & Funk", "Disco"): 26,
                          ("Reggae & Dancehall", "Reggae"): 26, ("Afro", "Afrobeats"): 26,
                          ("Hip Hop", "Hip Hop"): 26})

    def test_underscores_count_as_spaces(self):
        # 13 + 13: only 26 once "Tech_House" counts as "Tech House".
        tracks = library(("Tech_House", 13), ("Tech House", 13),
                         (["Melodic_House", "melodic__house"], 26), ("Deep_Tech", 26))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("House", "Tech House"): 26, ("House", "Melodic House"): 26,
                          ("Deep Tech", "Deep Tech"): 26})

    def test_underscores_only_change_this_feature(self):
        # The shared normaliser (custom rules, By Genre, Auto genres) is unchanged.
        self.assertEqual(organizer._norm_genre("Tech_House"), "tech_house")

    def test_a_tag_with_commas_is_one_genre(self):
        # rekordbox stores one genre per track; "House, Techno" is that genre's name.
        tracks = library(("House", 20), ("Techno", 20), ("House, Techno", 26))
        found = playlists(genres.popular_genre_plan(tracks))
        self.assertEqual(list(found.values()), [26])          # House/Techno stay at 20
        self.assertEqual([name for _, name in found], ["House, Techno"])

    def test_compound_names_are_not_split(self):
        tracks = library(("Hip Hop/Rap", 26), ("Melodic House & Techno", 26),
                         ("Drum & Bass", 26))
        self.assertEqual(sorted(name for _, name in playlists(genres.popular_genre_plan(tracks))),
                         ["Drum & Bass", "Hip Hop/Rap", "Melodic House & Techno"])

    def test_vague_and_missing_tags_are_left_out(self):
        tracks = library(("Electronic", 40), ("Dance & EDM", 40), ("", 40), ("Unknown", 40),
                         ("   ", 40))
        self.assertEqual(genres.popular_genre_plan(tracks), [])

    def test_most_used_spelling_names_the_playlist(self):
        tracks = library((["Shoegaze"] * 3 + ["shoegaze"], 28))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("Shoegaze", "Shoegaze"): 28})

    def test_result_does_not_depend_on_track_order(self):
        tracks = library((["tech house", "Tech House"], 26), ("Afro House", 26))
        self.assertEqual(genres.popular_genre_plan(tracks),
                         genres.popular_genre_plan(list(reversed(tracks))))

    def test_custom_taxonomy_aliases_are_used(self):
        override = {"families": {"House": {"aliases": ["haus"]},
                                 "Bass Music": {"parent": "Electronic",
                                                "words": ["bass music"],
                                                "subgenres": {"Wave": ["wave music"]}}}}
        merged = taxonomy.merge(taxonomy.DEFAULT, override)
        self.addCleanup(genres.refresh_taxonomy)    # back to the taxonomy in use
        with mock.patch.object(taxonomy, "load", return_value=merged):
            genres.refresh_taxonomy()
        tracks = library((["Haus", "House"], 26), (["wave music", "Wave"], 26))
        self.assertEqual(playlists(genres.popular_genre_plan(tracks)),
                         {("House", "House"): 26, ("Bass Music", "Wave"): 26})


class PopularGenreOptionTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(organizer.Options().popular_genres)
        tracks = library(("Techno", 40))
        self.assertEqual(organizer.build_plan(tracks, organizer.Options()), [])

    def test_build_plan_uses_tags_not_guesses(self):
        tracks = library(("Techno", 26), ("", 30))
        guesses = {t.id: "Techno" for t in tracks if not t.genre}
        plan = organizer.build_plan(tracks, organizer.Options(popular_genres=True), [], guesses)
        self.assertEqual(playlists(plan), {("Techno", "Techno"): 26})


XML_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.8.0" Company="AlphaTheta"/>
  <COLLECTION Entries="{n}">
"""
XML_TAIL = """  </COLLECTION>
  <PLAYLISTS><NODE Type="0" Name="ROOT" Count="0"></NODE></PLAYLISTS>
</DJ_PLAYLISTS>
"""


class PopularGenreRunTests(unittest.TestCase):
    """End to end on a synthetic XML export (never the real library)."""

    def setUp(self):
        tracks = library(("House", 26), ("Tech House", 27), ("Afro House", 25),
                         ("Techno", 3), ("electronic", 30))
        rows = "".join(
            f'    <TRACK TrackID="{t.id}" Name="{t.name}" Artist="A" Genre={quoteattr(t.genre)} '
            f'AverageBpm="124" Location=""/>\n' for t in tracks)
        fd, self.path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(XML_HEAD.format(n=len(tracks)) + rows + XML_TAIL)
        self.genres_before = {t.id: t.genre for t in tracks}

    def tearDown(self):
        for p in (self.path, organizer.default_output_path(self.path)):
            if os.path.exists(p):
                os.remove(p)

    def run_app(self, **options):
        lines = []
        opt = organizer.Options(**options)
        # Guessing is on by default; this option must still never analyse audio or look up.
        with mock.patch.object(organizer, "analyse_tracks", side_effect=AssertionError), \
                mock.patch.object(organizer, "predict", side_effect=AssertionError):
            out = organizer.run(opt, self.path, log=lines.append)
        return out, lines

    def test_run_makes_the_playlists_and_logs_counts(self):
        out, lines = self.run_app(popular_genres=True)
        root = ET.parse(out).getroot()
        organized = root.find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        popular = organized.find(f"NODE[@Name='{FOLDER}']")
        house = popular.find("NODE[@Name='House']")
        self.assertEqual([n.get("Name") for n in popular], ["House"])
        self.assertEqual([(n.get("Name"), len(n)) for n in house],
                         [("House", 26), ("Tech House", 27)])
        log = "\n".join(lines)
        self.assertIn("more than 25 tracks", log)
        self.assertIn("    House / House: 26 tracks", log)
        self.assertIn("    House / Tech House: 27 tracks", log)
        self.assertNotIn("Afro House", log)      # 25 songs: left out
        self.assertNotIn("electronic", log.casefold())   # vague tag: left out
        # Genre tags are unchanged.
        after = {t.get("TrackID"): t.get("Genre") for t in root.find("COLLECTION")}
        self.assertEqual(after, self.genres_before)

    def test_preview_lists_the_same_counts(self):
        changes = organizer.Changes()
        lines = []
        with mock.patch.object(organizer, "analyse_tracks", side_effect=AssertionError), \
                mock.patch.object(organizer, "predict", side_effect=AssertionError):
            organizer.run(organizer.Options(popular_genres=True), self.path,
                          log=lines.append, preview=True, changes=changes)
        self.assertIn("    House / Tech House: 27 tracks", lines)
        self.assertFalse(os.path.exists(organizer.default_output_path(self.path)))

    def test_no_qualifying_genre_is_reported(self):
        lines = []
        organizer.log_popular_genres([], lines.append)
        self.assertIn("no genre has more than 25 tracks", lines[0])

    def test_off_leaves_existing_behaviour(self):
        out, lines = self.run_app(genre=True)
        organized = ET.parse(out).getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        self.assertIsNone(organized.find(f"NODE[@Name='{FOLDER}']"))
        self.assertFalse(any("Popular genres" in line for line in lines))

    def write_xml(self, groups):
        tracks = library(*groups)
        rows = "".join(
            f'    <TRACK TrackID="{t.id}" Name="{t.name}" Artist="A" Genre={quoteattr(t.genre)} '
            f'AverageBpm="124" Location=""/>\n' for t in tracks)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(XML_HEAD.format(n=len(tracks)) + rows + XML_TAIL)
        with open(self.path, "rb") as f:
            return f.read()

    def test_no_genre_meets_the_threshold_is_a_result_not_an_error(self):
        before = self.write_xml([("House", 25), ("Tech House", 25), ("electronic", 40)])
        for preview in (True, False):
            lines = []
            with self.assertRaises(organizer.NoQualifyingGenres) as ctx:
                organizer.run(organizer.Options(popular_genres=True), self.path,
                              log=lines.append, preview=preview, changes=organizer.Changes())
            self.assertIn("No genres meet the threshold", str(ctx.exception))
            self.assertIn("Nothing was changed", str(ctx.exception))
            self.assertNotIn("tick at least one option", str(ctx.exception))
            self.assertEqual(sum("No genres meet" in l or "no genre has" in l for l in lines), 0)
            self.assertFalse(os.path.exists(organizer.default_output_path(self.path)))
            with open(self.path, "rb") as f:
                self.assertEqual(f.read(), before)

    def test_nothing_ticked_still_says_nothing_to_do(self):
        with self.assertRaises(ValueError) as ctx:
            organizer.run(organizer.Options(), self.path, log=lambda m: None)
        self.assertNotIsInstance(ctx.exception, organizer.NoQualifyingGenres)
        self.assertIn("tick at least one option", str(ctx.exception))

    def test_with_other_options_the_run_goes_ahead(self):
        self.write_xml([("House", 25), ("Techno", 3)])
        out, lines = self.run_app(popular_genres=True, genre=True)
        self.assertTrue(os.path.exists(out))
        self.assertIn("no genre has more than 25 tracks", "\n".join(lines))
        organized = ET.parse(out).getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        self.assertIsNone(organized.find(f"NODE[@Name='{FOLDER}']"))

    def test_constants_match(self):
        self.assertEqual(organizer.POPULAR_FOLDER, genres.POPULAR_FOLDER)
        self.assertEqual(organizer.POPULAR_MORE_THAN, genres.POPULAR_MORE_THAN)


class PopularGenreDatabaseTests(safety.DatabaseCase):
    """Direct mode on the disposable master.db from test_library_safety (4 songs)."""

    def test_no_qualifying_genre_writes_nothing_and_makes_no_backup(self):
        before, digest = self.state(), safety.sha(self.db)
        for preview in (True, False):
            with self.assertRaises(organizer.NoQualifyingGenres):
                organizer.run(organizer.Options(popular_genres=True), preview=preview,
                              changes=organizer.Changes(), log=lambda m: None)
            self.assertEqual(safety.sha(self.db), digest)
            self.assertEqual(self.state(), before)
            self.assertEqual(self.backups(), [])
            self.assertIsNone(rbdb.interrupted_change(self.db))


if __name__ == "__main__":
    unittest.main()
