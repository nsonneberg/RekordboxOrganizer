import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import numpy as np

import genre_model
import genres
import organizer
import test_organizer


def track(tid, genre, bpm, name=None, artist="a", length=0):
    return organizer.Track(tid, name or f"t{tid}", artist, genre, "", bpm, "", 0, 0, "", "", 0, "",
                           length=length)


def sound(lows, highs=0.2, brightness=1.0, flux=1.0, loudness=15.0):
    return {"lows": lows, "highs": highs, "brightness": brightness, "flux": flux,
            "loudness": loudness}


def model(segments, embedding=None, music=0.9):
    """A fake genre_model.ModelResult: segments = [{AudioSet genre name: score}]."""
    emb = None
    if embedding is not None:
        emb = np.asarray(embedding, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
    return genre_model.ModelResult([30.0 * i for i in range(len(segments))], segments,
                                   [music] * len(segments), emb)


def direction(i, noise=0.0, dims=16):
    v = np.zeros(dims)
    v[i] = 1.0
    v[(i + 1) % dims] = noise
    return v


class HermeticLabels(unittest.TestCase):
    """Keep the user's real genre_labels.json out of the tests."""

    def setUp(self):
        folder = tempfile.mkdtemp()
        self.labels_file = os.path.join(folder, "labels.json")
        patcher = mock.patch.object(genres, "labels_path", lambda: self.labels_file)
        patcher.start()
        self.addCleanup(patcher.stop)


class LegacyMethod(HermeticLabels):
    """Runs a test class against the earlier weighted-average fusion (method="legacy").

    These tests were written for it and still describe how it behaves; the
    calibrated method (the default) has its own tests in test_fusion.py.
    """

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(genres, "METHOD", "legacy")
        patcher.start()
        self.addCleanup(patcher.stop)
        # The app never uses legacy guesses unapproved (genres.auto_allowed); here
        # guess_genres reports the legacy scorer's own confident answers.
        patcher = mock.patch.object(genres, "auto_allowed",
                                    lambda p: p is not None and p.confident and bool(p.label))
        patcher.start()
        self.addCleanup(patcher.stop)


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

    def test_hierarchy(self):
        cases = {"tech-house": ("House", "Tech House"), "TECH HOUSE": ("House", "Tech House"),
                 "House": ("House", ""), "Drum & Bass": ("Drum & Bass", ""),
                 "liquid": ("Drum & Bass", "Liquid"), "Hip-Hop": ("Hip Hop", ""),
                 "Trap": ("Hip Hop", "Trap"),
                 "Melodic House & Techno": ("Techno", "Melodic House & Techno"),
                 "": (genres.UNKNOWN, "")}
        for tag, expected in cases.items():
            self.assertEqual(genres.subgenre_of(tag), expected, tag)
        self.assertEqual(genres.parent_of("House"), "Electronic")
        self.assertEqual(genres.parent_of("Drum & Bass"), "Electronic")
        self.assertEqual(genres.parent_of("Hip Hop"), "")

    def test_tempo_fits_half_and_double_time(self):
        self.assertTrue(genres.tempo_fits("Drum & Bass", 174))
        self.assertTrue(genres.tempo_fits("Drum & Bass", 87))      # read at half speed
        self.assertFalse(genres.tempo_fits("Drum & Bass", 124))
        self.assertTrue(genres.tempo_fits("Hip Hop", 180))         # 90 read at double speed
        self.assertIsNone(genres.tempo_fits("Drum & Bass", 0))     # unknown BPM
        self.assertIsNone(genres.tempo_fits("Rock", 120))          # no usual range


class UncertaintyTests(LegacyMethod):
    def test_bpm_alone_is_never_confident(self):
        tracks = [track("1", "", 174), track("2", "", 124), track("3", "Dance & EDM", 95),
                  track("4", "", 90), track("5", "House", 125), track("6", "", 0),
                  track("7", "", 140)]
        logs = []
        predictions = genres.predict_genres(tracks, {}, logs.append)
        self.assertEqual(genres.guess_genres(tracks, {}, lambda m: None), {})
        self.assertTrue(all(not p.confident for p in predictions.values()))
        # The tempo still suggests a candidate for the review window ...
        self.assertEqual(predictions["1"].candidates[0][0], "Drum & Bass")
        self.assertEqual(predictions["2"].candidates[0][0], "House")
        self.assertEqual(predictions["3"].candidates[0][0], "EDM")   # vague electronic tag
        self.assertIn("tempo", predictions["1"].reason)
        # ... except with nothing to go on at all.
        self.assertEqual(predictions["6"].candidates, [])
        self.assertEqual(predictions["6"].family, "")
        self.assertNotIn("5", predictions)                 # tagged songs aren't predicted
        self.assertIn("6 need review", logs[-1])

    def test_rules_put_misread_bpm_in_the_right_place(self):
        z = {k: 0.0 for k in genres.SOUND}
        self.assertEqual(genres._rule_guess(174, z, False), "Drum & Bass")
        self.assertEqual(genres._rule_guess(348, z, False), "Drum & Bass")   # double time
        self.assertEqual(genres._rule_guess(87, dict(z, lows=1.0), False), "Drum & Bass")
        self.assertNotEqual(genres._rule_guess(190, z, False), "Hardstyle")
        self.assertEqual(genres._rule_guess(128, z, True), "House")
        self.assertIsNone(genres._rule_guess(0, z, False))

    def test_weak_model_evidence_needs_review(self):
        tracks = [track("1", "", 124)]
        # The model hears music, but only "Electronic music": no style.
        vague = {"1": model([{"Electronic music": 0.6, "House music": 0.01}] * 3)}
        p = genres.predict_genres(tracks, {}, None, vague)["1"]
        self.assertFalse(p.confident)
        self.assertIn("no clear genre", p.reason)
        # Excerpts that disagree with each other.
        split = {"1": model([{"House music": 0.5}, {"Techno": 0.5}, {"Trance music": 0.5}])}
        p = genres.predict_genres(tracks, {}, None, split)["1"]
        self.assertFalse(p.confident)
        self.assertEqual(len(p.candidates), 3)

    def test_close_call_is_uncertain(self):
        tracks = [track("1", "", 128)]
        close = {"1": model([{"House music": 0.40, "Techno": 0.36}] * 3)}
        p = genres.predict_genres(tracks, {}, None, close)["1"]
        self.assertFalse(p.confident)
        self.assertEqual([f for f, _ in p.candidates], ["House", "Techno"])
        self.assertLess(p.margin, genres.MIN_MARGIN)

    def test_not_forced_into_a_known_family(self):
        tracks = [track("1", "", 120)]
        speech = {"1": model([{"Pop music": 0.001}], music=0.05)}   # not music at all
        p = genres.predict_genres(tracks, {}, None, speech)["1"]
        self.assertFalse(p.confident)
        self.assertEqual(genres.guess_genres(tracks, {}, None, speech), {})

    def test_scores_are_relative_shares_not_probabilities(self):
        tracks = [track("1", "", 174)]
        p = genres.predict_genres(tracks, {}, None,
                                  {"1": model([{"Drum and bass": 0.3, "Dubstep": 0.05}] * 3)})["1"]
        self.assertAlmostEqual(sum(s for _, s in p.candidates), 1.0, places=3)
        self.assertTrue(p.confident)
        self.assertEqual(p.parent, "Electronic")


class ModelEvidenceTests(LegacyMethod):
    def test_related_genres_at_the_same_bpm(self):
        tracks = [track("h", "", 128), track("t", "Electronic", 128), track("r", "", 128)]
        results = {"h": model([{"House music": 0.5, "Techno": 0.08}] * 4),
                   "t": model([{"Techno": 0.45, "House music": 0.1}] * 4),
                   "r": model([{"Trance music": 0.4, "House music": 0.05}] * 4)}
        guesses = genres.guess_genres(tracks, {}, None, results)
        self.assertEqual(guesses, {"h": "House", "t": "Techno", "r": "Trance"})

    def test_misleading_bpm(self):
        # Half-time D&B read at 87 BPM: the audio decides, the tempo doesn't object.
        tracks = [track("1", "", 87), track("2", "", 128)]
        results = {"1": model([{"Drum and bass": 0.4, "Hip hop music": 0.05}] * 3),
                   # Sounds like hip hop although 128 BPM is "house tempo".
                   "2": model([{"Hip hop music": 0.5, "House music": 0.05}] * 3)}
        guesses = genres.guess_genres(tracks, {}, None, results)
        self.assertEqual(guesses, {"1": "Drum & Bass", "2": "Hip Hop"})

    def test_tempo_is_secondary(self):
        tracks = [track("1", "", 124)]
        # Clear audio evidence for D&B: a "house tempo" can't overrule it.
        p = genres.predict_genres(tracks, {}, None,
                                  {"1": model([{"Drum and bass": 0.5, "House music": 0.05}] * 3)})
        self.assertEqual((p["1"].family, p["1"].confident), ("Drum & Bass", True))
        # A close call: the tempo may tip the order, but it can't make it confident.
        p = genres.predict_genres(tracks, {}, None,
                                  {"1": model([{"Drum and bass": 0.3, "House music": 0.2}] * 3)})
        self.assertFalse(p["1"].confident)
        self.assertEqual({f for f, _ in p["1"].candidates}, {"Drum & Bass", "House"})

    def test_learns_subgenres_from_tagged_songs(self):
        tracks, results = [], {}
        for i in range(3):
            tracks.append(track(f"th{i}", "Tech House", 126, name=f"th{i}"))
            results[f"th{i}"] = model([{"House music": 0.5}] * 3, direction(0, 0.1 * i))
            tracks.append(track(f"dh{i}", "Deep House", 122, name=f"dh{i}"))
            results[f"dh{i}"] = model([{"House music": 0.5}] * 3, direction(4, 0.1 * i))
        tracks.append(track("x", "", 125))
        results["x"] = model([{"House music": 0.5}] * 3, direction(0, 0.05))
        p = genres.predict_genres(tracks, {}, None, results)["x"]
        self.assertTrue(p.confident)
        self.assertEqual((p.family, p.subgenre, p.label), ("House", "Tech House", "Tech House"))
        self.assertIn("library", p.sources)

    def test_family_without_subgenre_support(self):
        tracks = [track("1", "", 125)]
        p = genres.predict_genres(tracks, {}, None, {"1": model([{"House music": 0.5}] * 3)})["1"]
        self.assertEqual((p.family, p.subgenre), ("House", ""))


class LibraryLearningTests(LegacyMethod):
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

    def test_too_few_examples_stay_uncertain(self):
        tracks = [track("h", "House", 124), track("t", "Techno", 132), track("x", "", 130)]
        feats = {"h": sound(0.5), "t": sound(0.2, 0.6), "x": sound(0.21, 0.58)}
        p = genres.predict_genres(tracks, feats, None)["x"]
        self.assertFalse(p.confident)
        self.assertIn("too few tagged songs", p.reason.lower())

    def test_copies_vote_once(self):
        # Five copies of one House recording mustn't outvote three Techno recordings.
        tracks, results = [], {}
        for i in range(5):
            tracks.append(track(f"h{i}", "House", 126, name="Same Song", length=200))
            results[f"h{i}"] = model([{"Electronic music": 0.5}], direction(0, 0.3))
        for i in range(3):
            tracks.append(track(f"t{i}", "Techno", 126, name=f"techno {i}"))
            results[f"t{i}"] = model([{"Electronic music": 0.5}], direction(0, 0.2 + 0.01 * i))
        tracks.append(track("x", "", 126))
        results["x"] = model([{"Techno": 0.2, "House music": 0.15}] * 2, direction(0, 0.21))
        p = genres.predict_genres(tracks, {}, None, results)["x"]
        self.assertEqual(p.candidates[0][0], "Techno")

    def test_copy_of_a_tagged_recording(self):
        tracks = [track("1", "Deep House", 122, name="Song", length=300),
                  track("2", "", 122, name="Song", length=301)]
        p = genres.predict_genres(tracks, {}, None)["2"]
        self.assertEqual(p.status, "suggested")      # matched by name and length: approve it
        self.assertEqual((p.label, p.sources), ("Deep House", ("copy",)))

    def test_accepted_suggestions_are_not_learned_from(self):
        tracks = [track("1", "House", 124), track("2", "Techno", 130), track("3", "", 0)]
        genres.record_labels({"1": ("House", "accepted"), "2": ("Techno", "corrected")})
        labels = genres.load_labels()
        self.assertEqual(labels["1"]["source"], "accepted")
        self.assertEqual(genres.training_labels(tracks, labels), {"2": "Techno"})
        # Once the user changes the tag themselves it's theirs again.
        tracks[0] = track("1", "Tech House", 124)
        self.assertEqual(genres.training_labels(tracks, labels), {"1": "Tech House", "2": "Techno"})

    def test_effective_genre(self):
        guesses = {"1": "House"}
        self.assertEqual(genres.effective_genre(track("1", "", 124), guesses), "House")
        self.assertEqual(genres.effective_genre(track("2", "Techno", 124), guesses), "Techno")


class EvaluationTests(HermeticLabels):
    def test_copies_are_hidden_together(self):
        tracks, results = [], {}
        for i in range(3):       # each House recording has two copies
            for c in range(2):
                tracks.append(track(f"h{i}{c}", "House", 125, name=f"house {i}", length=200))
                results[f"h{i}{c}"] = model([{"House music": 0.5}] * 2, direction(0, 0.1 * i))
            tracks.append(track(f"d{i}", "Drum & Bass", 174, name=f"dnb {i}"))
            results[f"d{i}"] = model([{"Drum and bass": 0.5}] * 2, direction(3, 0.1 * i))
        for i in range(3):
            del results[f"h{i}0"]                     # one copy's file is missing
        ev = genres.evaluate(tracks, {}, results)
        self.assertEqual(ev.recordings, 6)            # copies count once
        self.assertEqual(ev.correct_top, 6)           # the copy that could be heard is used
        # Too few tagged songs to measure accuracy: nothing is assigned automatically.
        self.assertEqual(ev.confident, 0)
        # Without the model and with no sound features, nothing can be confident -
        # in particular not via the "tagged copy" shortcut.
        ev = genres.evaluate(tracks, {}, {})
        self.assertEqual(ev.confident, 0)
        self.assertIn("6 labelled recordings", ev.summary())


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

    def test_uncertain_songs_go_to_needs_review(self):
        tracks = [track("1", "House", 124), track("2", "", 128), track("3", "Electronic", 130),
                  track("4", "", 0), track("5", "", 125)]
        plan = dict(genres.auto_genre_plan(tracks, {"5": "Tech House"}, review={"2", "3"}))
        self.assertEqual(list(plan), [("Auto Genres", "House"), ("Auto Genres", genres.NEEDS_REVIEW),
                                      ("Auto Genres", genres.UNKNOWN)])
        self.assertEqual(plan[("Auto Genres", genres.NEEDS_REVIEW)],
                         [(genres.NEEDS_REVIEW, ["2", "3"])])
        self.assertIn(("Tech House (detected)", ["5"]), plan[("Auto Genres", "House")])

    def test_suggestions_stay_inside_needs_review(self):
        tracks = [track("1", "House", 124), track("2", "", 138), track("3", "", 128)]
        plan = dict(genres.auto_genre_plan(tracks, {}, review={"2": "Trance", "3": ""}))
        self.assertEqual(plan[("Auto Genres", genres.NEEDS_REVIEW)],
                         [(f"All {genres.NEEDS_REVIEW}", ["3", "2"]),
                          (genres.NEEDS_REVIEW, ["3"]), ("Trance (suggested)", ["2"])])
        self.assertNotIn(("Auto Genres", "Trance"), plan)      # not filed as Trance


class IntegrationTests(HermeticLabels):
    def test_custom_playlists_use_guessed_genres(self):
        tracks = [track("1", "Dance & EDM", 125), track("2", "", 174), track("3", "Techno", 125)]
        rules = organizer.compile_custom(organizer.Options(custom_enabled=True, custom=[
            organizer.CustomPlaylist("House", "house", "118-130")]))
        results = {"1": model([{"House music": 0.5}] * 3),
                   "2": model([{"Drum and bass": 0.5}] * 3)}
        # With no tagged songs to measure against, the House guess is only a suggestion,
        # so it doesn't place the song by itself.
        self.assertEqual(genres.guess_genres(tracks, {}, lambda m: None, results), {})
        guesses = {"1": "House", "2": "Drum & Bass"}       # confident guesses
        self.assertEqual(organizer.sort_into_custom(tracks, rules, False, guesses),
                         [("House", ["1"])])               # tagged Techno stays out
        self.assertEqual(organizer.sort_into_custom(tracks, rules, False),
                         [("House", [])])                  # guessing off: tags only
        # BPM alone (no model) isn't enough to put song 1 in House.
        guesses = genres.guess_genres(tracks, {}, lambda m: None)
        self.assertEqual(organizer.sort_into_custom(tracks, rules, False, guesses),
                         [("House", [])])

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

    def test_run_with_mocked_model(self):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(test_organizer.SAMPLE)
        out = organizer.default_output_path(path)
        self.addCleanup(lambda: [os.remove(p) for p in (path, out) if os.path.exists(p)])
        _, tracks = organizer.load_library(path)
        untagged = [t.id for t in tracks if genres.is_vague(t.genre)]
        self.assertTrue(untagged)
        fake = {i: model([{"Techno": 0.5}] * 3) for i in untagged}
        logs = []
        with mock.patch.object(organizer, "model_genres", return_value=fake) as called:
            organizer.run(organizer.Options(auto_genres=True, genre_model=True), path,
                          log=logs.append)
        called.assert_called_once()
        organized = ET.parse(out).getroot().find("PLAYLISTS/NODE/NODE[@Name='Organized']")
        # A handful of tagged songs can't show Techno guesses are reliable: they're
        # suggested inside Needs Review, never filed as detected.
        review = organized.find(f"NODE[@Name='Auto Genres']/NODE[@Name='{genres.NEEDS_REVIEW}']")
        self.assertIn("Techno (suggested)", [n.get("Name") for n in review])
        names = [n.get("Name") for n in organized.iter("NODE")]
        self.assertFalse(any(n.endswith("(detected)") for n in names))
        self.assertTrue(any("suggested for you to approve" in m for m in logs))

    def test_model_failures_fall_back(self):
        tracks = [track("1", "", 124)]
        tracks[0].path = __file__
        logs = []
        with mock.patch.object(genre_model, "runtime_problem", return_value="blocked"):
            self.assertEqual(organizer.model_genres(tracks, logs.append), {})
        self.assertIn("without it", logs[-1])
        with mock.patch.object(genre_model, "runtime_problem", return_value=""), \
                mock.patch.object(genre_model, "ensure_model", side_effect=OSError("offline")):
            self.assertEqual(organizer.model_genres(tracks, logs.append), {})
        self.assertIn("offline", logs[-1])


def discogs(segments, embedding=None):
    """A fake Discogs-EffNet ModelResult: segments = [{"Genre---Style": score}]."""
    r = model(segments, embedding)
    r.model = "discogs-effnet"
    return r


def meta(genres_, quality=0.9, source="MusicBrainz"):
    import metadata
    return metadata.MetadataResult(source, list(genres_), quality, "recording", "Artist - Song")


class TaxonomyTests(unittest.TestCase):
    def tearDown(self):
        genres.refresh_taxonomy(reload=True)

    def test_synonyms(self):
        cases = {"Psychedelic Trance": ("Trance", "Psytrance"), "psy-trance": ("Trance", "Psytrance"),
                 "UKG": ("House", "UK Garage"), "Freeform Bass": ("Dubstep", "Freeform Bass"),
                 "Liquid Funk": ("Drum & Bass", "Liquid"), "Minimal Techno": ("Techno", "Minimal")}
        for tag, expected in cases.items():
            self.assertEqual(genres.subgenre_of(tag), expected, tag)

    def test_discogs_labels(self):
        self.assertEqual(genres.model_label("discogs-effnet", "Electronic---Tech House"),
                         [("House", "Tech House", 1.0)])
        self.assertEqual(genres.model_label("discogs-effnet", "Electronic---Halftime"),
                         [("Drum & Bass", "Halftime", 0.5), ("Dubstep", "", 0.5)])
        self.assertEqual(genres.model_label("discogs-effnet", "Rock---Shoegaze"), [("Rock", "", 1.0)])
        self.assertEqual(genres.model_label("discogs-effnet", "Non-Music---Field Recording"), [])
        self.assertEqual(genres.model_label("yamnet", "Electronic music"), [])   # no style

    def test_user_taxonomy_file(self):
        import json
        import taxonomy
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "taxonomy.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"families": {"House": {"subgenres": {"Organic House": ["organic"]}},
                                    "Bass Music": {"parent": "Electronic",
                                                   "words": ["bass music", "wobble"]}},
                       "discogs": {"Electronic---Donk": "EDM"}}, f)
        with mock.patch.object(taxonomy, "path", lambda: path):
            genres.refresh_taxonomy(reload=True)
            self.assertEqual(genres.subgenre_of("organic"), ("House", "Organic House"))
            self.assertEqual(genres.family_of("Wobble"), "Bass Music")
            self.assertEqual(genres.parent_of("Bass Music"), "Electronic")
            self.assertEqual(genres.model_label("discogs-effnet", "Electronic---Donk"),
                             [("EDM", "", 1.0)])
            self.assertEqual(genres.subgenre_of("Tech House"), ("House", "Tech House"))  # kept

    def test_broken_taxonomy_file_is_ignored(self):
        import taxonomy
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "taxonomy.json")
        with open(path, "w") as f:
            f.write("[not, valid")
        with mock.patch.object(taxonomy, "path", lambda: path):
            genres.refresh_taxonomy(reload=True)
            self.assertIn("ignored", taxonomy.problem)
            self.assertEqual(genres.family_of("Tech House"), "House")


class FusionTests(LegacyMethod):
    def test_discogs_styles_give_family_and_subgenre(self):
        tracks = [track("1", "", 126)]
        r = {"1": discogs([{"Electronic---Tech House": 0.7, "Electronic---House": 0.3,
                            "Electronic---Techno": 0.1}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r)["1"]
        self.assertEqual((p.status, p.family, p.subgenre, p.parent),
                         ("confident", "House", "Tech House", "Electronic"))

    def test_tech_house_vs_techno_at_the_same_bpm(self):
        tracks = [track("h", "", 128), track("t", "", 128)]
        r = {"h": discogs([{"Electronic---Tech House": 0.6, "Electronic---Techno": 0.15}] * 3),
             "t": discogs([{"Electronic---Techno": 0.7, "Electronic---Tech House": 0.1}] * 3)}
        g = genres.guess_genres(tracks, {}, None, r)
        self.assertEqual(g, {"h": "Tech House", "t": "Techno"})

    def test_metadata_only(self):
        tracks = [track("1", "", 0)]                 # no BPM, no audio analysis
        p = genres.predict_genres(tracks, {}, None, metadata={"1": meta([("techno", 3)])})["1"]
        self.assertEqual((p.status, p.family), ("confident", "Techno"))
        self.assertEqual(p.sources[0], "metadata")

    def test_poor_metadata_match_is_not_enough(self):
        tracks = [track("1", "", 0)]
        p = genres.predict_genres(tracks, {}, None,
                                  metadata={"1": meta([("techno", 3)], quality=0.3)})["1"]
        self.assertFalse(p.confident)

    def test_vague_metadata_is_ignored(self):
        tracks = [track("1", "", 0)]
        p = genres.predict_genres(tracks, {}, None,
                                  metadata={"1": meta([("electronic", 5), ("dance", 2)])})["1"]
        self.assertFalse(p.confident)
        self.assertEqual(p.sources, ())

    def test_conflicting_sources_need_review(self):
        tracks = [track("1", "", 128)]
        r = {"1": discogs([{"Electronic---House": 0.8}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r, metadata={"1": meta([("techno", 4)])})["1"]
        self.assertFalse(p.confident)
        self.assertTrue(p.disagreement)
        self.assertIn("disagree", p.reason)
        self.assertEqual({f for f, _ in p.candidates}, {"House", "Techno"})

    def test_agreeing_sources(self):
        tracks = [track("1", "", 128)]
        r = {"1": discogs([{"Electronic---Techno": 0.5, "Electronic---House": 0.3}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r, metadata={"1": meta([("techno", 4)])})["1"]
        self.assertEqual((p.status, p.family), ("confident", "Techno"))
        self.assertEqual([e[0] for e in p.evidence], ["model", "metadata"])

    def test_missing_audio_and_metadata(self):
        tracks = [track("1", "", 0), track("2", "Electronic", 0)]
        p = genres.predict_genres(tracks, {}, None, {}, metadata={})
        self.assertTrue(all(not x.confident and not x.candidates for x in p.values()))

    def test_bpm_cannot_overrule_the_model(self):
        # The BPM (85) is wrong (half-read); the audio is clearly D&B.
        tracks = [track("1", "", 85)]
        r = {"1": discogs([{"Electronic---Drum n Bass": 0.8}] * 3)}
        self.assertEqual(genres.guess_genres(tracks, {}, None, r), {"1": "Drum & Bass"})
        # A wildly wrong BPM can lower, but not flip, clear evidence.
        tracks = [track("1", "", 100)]
        self.assertEqual(genres.guess_genres(tracks, {}, None, r), {"1": "Drum & Bass"})


class LibraryClassifierTests(HermeticLabels):
    def library(self, n_house=6, n_techno=4, n_dnb=0):
        tracks, r = [], {}
        rng = np.random.default_rng(0)
        for fam, n, axis, style in (("House", n_house, 0, "Electronic---House"),
                                    ("Techno", n_techno, 1, "Electronic---Techno"),
                                    ("Drum & Bass", n_dnb, 2, "Electronic---Drum n Bass")):
            for i in range(n):
                tid = f"{fam[:2]}{i}"
                words = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf")
                tracks.append(track(tid, fam, 128, name=f"{fam} {words[i]}"))
                emb = np.eye(16)[axis] + rng.normal(0, 0.05, 16)
                r[tid] = discogs([{style: 0.6}] * 2, emb)
        return tracks, r

    def test_learns_from_tagged_songs(self):
        tracks, r = self.library()
        tracks.append(track("x", "", 128))
        # The model can't tell; the embedding sits with the Techno songs.
        r["x"] = discogs([{"Electronic---House": 0.3, "Electronic---Techno": 0.3}] * 2,
                         np.eye(16)[1] + 0.01)
        p = genres.predict_genres(tracks, {}, None, r)["x"]
        self.assertEqual(p.candidates[0][0], "Techno")
        self.assertIn("library", p.sources)

    def test_unknown_families_are_not_forced(self):
        # Only 2 D&B songs are tagged: too few to train on. A D&B track must not
        # be pushed into House / Techno by the classifier.
        tracks, r = self.library(n_dnb=2)
        tracks.append(track("x", "", 174))
        r["x"] = discogs([{"Electronic---Drum n Bass": 0.7}] * 2, np.eye(16)[1] + 0.01)
        p = genres.predict_genres(tracks, {}, None, r)["x"]
        self.assertEqual(p.family, "Drum & Bass")

    def test_reliability_is_measured(self):
        tracks, r = self.library()
        ctx = genres._Context(tracks, {}, r, genres.training_labels(tracks), {},
                              set(genres.SOURCES))
        self.assertGreater(ctx.reliability["model"], 0.8)     # 10 of 10 right (+ prior)
        self.assertEqual(ctx.reliability["metadata"], 0.5)    # nothing to measure: neutral

    def test_confirmed_labels_only(self):
        tracks, r = self.library()
        genres.record_labels({"Ho0": ("House", "corrected", "")})
        labels = genres.load_labels()
        self.assertEqual(genres.training_labels(tracks, labels, include_tags=False), {"Ho0": "House"})
        self.assertEqual(labels["Ho0"]["original"], "")


class LeakageTests(HermeticLabels):
    def test_related_versions_are_grouped(self):
        tracks = [track("1", "House", 124, name="Artist - Song Title (Original Mix)", length=300),
                  track("2", "Drum & Bass", 174, name="Song Title (Someone D&B Remix)", length=250),
                  track("3", "House", 124, name="Different Song", length=200)]
        g = genres.version_groups(tracks)
        self.assertEqual(g["1"], g["2"])
        self.assertNotEqual(g["1"], g["3"])
        # ...but a remix is not a "copy": it doesn't give its genre away.
        self.assertNotEqual(genres._groups(tracks)["1"], genres._groups(tracks)["2"])

    def test_evaluation_holds_out_related_versions(self):
        tracks, r = [], {}
        for i in range(3):
            tracks.append(track(f"o{i}", "Techno", 130, name=f"Tune {i}", length=300))
            tracks.append(track(f"r{i}", "Techno", 130, name=f"Tune {i} (X Remix)", length=280))
            r[f"o{i}"] = model([{"House music": 0.5}] * 2)          # the model is wrong
            r[f"r{i}"] = model([{"House music": 0.5}] * 2)
        ev = genres.evaluate(tracks, {}, r)
        # If the remix leaked, the library would get these right; held out, it can't.
        self.assertEqual(ev.correct_top, 0)
        self.assertEqual(ev.recordings, 6)


if __name__ == "__main__":
    unittest.main()
