"""The verified-label workflow, held-out evaluation and benchmark metrics."""

import csv
import os
import shutil
import tempfile
import unittest
from unittest import mock

import benchmark_genres as bg
import benchmark_labels as bl
import fusion
import genres
from test_fusion import discogs
from test_genres import HermeticLabels, direction, track


class LabelWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.file = os.path.join(self.dir, "labels.json")
        self.tracks = [track("1", "House", 124, name="Artist - Song (Original Mix)", length=300),
                       track("2", "", 174, name="Song (Someone D&B Remix)", length=250),
                       track("3", "Juniper Foxglove Records", 122, name="Other Tune", length=200)]

    def test_candidates_are_not_ground_truth(self):
        data = bl.load(self.file)
        self.assertEqual(bl.add_candidates(data, self.tracks), 3)
        self.assertEqual(bl.add_candidates(data, self.tracks), 0)          # no duplicates
        self.assertEqual(bl.verified(data, self.tracks), {})

    def test_verification_rules(self):
        data = bl.load(self.file)
        bl.add_candidates(data, self.tracks)
        with self.assertRaises(ValueError):
            bl.verify(data, "1", "Electronic", "listened", "me")            # too vague
        with self.assertRaises(ValueError):
            bl.verify(data, "3", "Juniper Foxglove Records", "listened", "me")   # not a genre
        with self.assertRaises(ValueError):
            bl.verify(data, "1", "Tech House", "", "me")                    # no evidence
        with self.assertRaises(ValueError):
            bl.verify(data, "1", "Tech House", "listened", "")              # nobody
        with self.assertRaises(KeyError):
            bl.verify(data, "99", "Techno", "listened", "me")
        bl.verify(data, "1", "Tech House", "listened; label page says Tech House", "me")
        e = data["entries"]["1"]
        self.assertEqual((e["status"], e["genre"], e["verified_by"]), ("verified", "Tech House", "me"))
        self.assertTrue(e["verified_on"])

    def test_versions_of_a_song_share_a_split(self):
        names = ["Song (Original Mix)", "Artist - Song (X Remix)", "Song [Extended Edit]", "Song"]
        self.assertEqual(len({bl.split_of(n) for n in names}), 1)
        splits = {bl.split_of(f"Unrelated title number {i}") for i in range(60)}
        self.assertEqual(splits, {"dev", "test"})

    def test_matching_by_name_when_ids_change(self):
        data = bl.load(self.file)
        bl.add_candidates(data, self.tracks)
        bl.verify(data, "1", "Tech House", "listened", "me")
        moved = [track("77", "House", 124, name="Artist - Song (Original Mix)", length=301)]
        moved[0] = moved[0].__class__(**{**moved[0].__dict__, "artist": self.tracks[0].artist})
        self.assertEqual(bl.verified(data, moved), {"77": "Tech House"})
        far = [moved[0].__class__(**{**moved[0].__dict__, "length": 400})]
        self.assertEqual(bl.verified(data, far), {})                       # a different edit

    def test_split_filter_and_test_run_log(self):
        data = bl.load(self.file)
        bl.add_candidates(data, self.tracks)
        bl.verify(data, "1", "House", "listened", "me")
        split = data["entries"]["1"]["split"]
        other = "dev" if split == "test" else "test"
        self.assertEqual(bl.verified(data, self.tracks, split), {"1": "House"})
        self.assertEqual(bl.verified(data, self.tracks, other), {})
        bl.log_test_run(data, "x")
        bl.save(data, self.file)
        self.assertEqual(len(bl.load(self.file)["test_runs"]), 1)

    def test_csv_round_trip(self):
        data = bl.load(self.file)
        bl.add_candidates(data, self.tracks)
        bl.save(data, self.file)
        path = os.path.join(self.dir, "c.csv")
        with mock.patch.object(bl, "path", lambda: self.file):
            bl.main(["export", path])
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            rows[0]["verified_genre"], rows[0]["evidence"] = "Techno", "listened"
            rows[1]["verified_genre"] = "Drum & Bass"                       # no evidence: skipped
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, bl.CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            bl.main(["import", path, "--by", "me"])
            data = bl.load(self.file)
        self.assertEqual(data["entries"][rows[0]["id"]]["status"], "verified")
        self.assertEqual(data["entries"][rows[1]["id"]]["status"], "candidate")

    def test_stats_lists_what_is_missing(self):
        data = bl.load(self.file)
        text = bl.stats(data)
        self.assertIn("House", text)
        self.assertIn(str(bl.TARGET_PER_GENRE), text)


class GroupingTests(unittest.TestCase):
    def test_title_keys_drop_descriptors_numbers_and_credits(self):
        k = genres._title_keys
        self.assertEqual(k("12. Remixer Name - Some Singer - Marigold - Remixer Name Remix [FREE DL]"),
                         {"remixer name", "some singer", "marigold"})
        self.assertEqual(k("PREMIERE | Band - Roof Song (LOUD REMIX)"), {"band", "roof song"})
        self.assertEqual(k("Singer - Saltwater Ft Guest Star (dj bootleg)"), {"singer", "saltwater"})
        self.assertEqual(k("Tune One X Tune Two (Someone Mix)"), {"tune one", "tune two"})
        self.assertEqual(k("After Hours - Someone"), {"after hours", "someone"})   # "ft" in a word

    def test_related_recordings_are_grouped_and_unrelated_ones_are_not(self):
        tracks = [track("1", "House", 124, name="Singer - Marigold (Producer A Remix)"),
                  track("2", "", 124, name="12. Producer B - Singer2 - Marigold - Producer B Remix"),
                  track("3", "", 124, name="Marigold X Other Song (Someone Mix)"),
                  track("4", "House", 124, name="Totally Different Tune"),
                  track("5", "Techno", 130, name="Other Artist - Another Thing")]
        g = genres.version_groups(tracks)
        self.assertEqual(len({g["1"], g["2"], g["3"]}), 1)
        self.assertEqual(len({g["1"], g["4"], g["5"]}), 3)

    def test_versions_get_one_split_even_from_a_playlist(self):
        library = [track(str(i), "", 124, name=f"Artist{i} - Song Number {i}") for i in range(40)]
        library += [track("r", "", 124, name="Song Number 7 (Remix Person Remix)")]
        data = {"version": 1, "entries": {}, "test_runs": []}
        bl.add_candidates(data, [library[7]], library)
        bl.add_candidates(data, [library[-1]], library)
        self.assertEqual(data["entries"]["7"]["split"], data["entries"]["r"]["split"])
        bl.add_candidates(data, library, library)
        self.assertEqual({e["split"] for e in data["entries"].values()}, {"dev", "test"})


class HeldOutEvaluationTests(HermeticLabels):
    def library(self):
        tracks, r = [], {}
        for i in range(8):
            for fam, style, axis in (("House", "Electronic---House", 0),
                                     ("Techno", "Electronic---Techno", 1)):
                tid = f"{fam[:2]}{i}"
                tracks.append(track(tid, fam, 126, name=f"{fam} tune {i} distinct"))
                r[tid] = discogs([{style: 0.8}] * 2, direction(axis, 0.05 * i))
        return tracks, r

    def test_holdout_hides_the_whole_test_set_at_once(self):
        tracks, r = self.library()
        truth = {"Ho0": "House", "Te0": "Techno"}
        calls = []
        real = genres.predict_genres

        def spy(test, *a, **kw):
            calls.append({t.id for t in test if not t.genre})
            return real(test, *a, **kw)
        with mock.patch.object(genres, "predict_genres", spy):
            ev = genres.evaluate(tracks, {}, r, truth=truth, holdout=True)
        self.assertEqual(len(calls), 1)
        self.assertTrue({"Ho0", "Te0"} <= calls[0])
        self.assertEqual(ev.recordings, 2)

    def test_excluded_songs_are_never_learned_from(self):
        tracks, r = self.library()
        calls = []
        real = genres.predict_genres

        def spy(test, *a, **kw):
            calls.append({t.id for t in test if not t.genre})
            return real(test, *a, **kw)
        with mock.patch.object(genres, "predict_genres", spy):
            genres.evaluate(tracks, {}, r, exclude={"Te7"},
                            truth={"Ho0": "House", "Ho1": "House"})
        self.assertTrue(all("Te7" in hidden for hidden in calls))

    def test_non_genre_labels_are_skipped(self):
        tracks, r = self.library()
        tracks.append(track("x", "Juniper Foxglove Records", 122, name="label in genre field"))
        ev = genres.evaluate(tracks, {}, r, truth={"x": "Juniper Foxglove Records", "Ho0": "House"})
        self.assertEqual((ev.recordings, ev.skipped), (1, 1))


class ManualTagTests(HeldOutEvaluationTests.__bases__[0]):
    """Specific tags the user set are never predicted over; only vague/untagged songs are."""

    def test_only_untagged_and_vague_songs_get_suggestions(self):
        tracks, r = HeldOutEvaluationTests.library(self)
        tracks += [track("u", "", 126, name="untagged one"),
                   track("v", "Dance & EDM", 126, name="vague one"),
                   track("s", "Hard Techno", 126, name="specific one")]
        for tid in ("u", "v", "s"):
            r[tid] = discogs([{"Electronic---House": 0.8}] * 2, direction(0, 0.0))
        out = genres.predict_genres(tracks, {}, None, r, {})
        self.assertEqual(set(out) & {"u", "v", "s"}, {"u", "v"})
        self.assertEqual(next(t for t in tracks if t.id == "s").genre, "Hard Techno")


def row(truth, top, conf=0.9, auto=True, sub="", tsub="", group=None, tid=None):
    return bg.Row(tid or f"{truth}{top}{conf}{group}", "song", truth, truth, tsub, top, conf, auto,
                  top if auto else "", sub, group or f"g{id(object())}")


class MetricTests(unittest.TestCase):
    def test_macro_f1_over_true_classes(self):
        rows = [row("House", "House"), row("House", "EDM"), row("Techno", "Techno")]
        # True classes: House F1 = 2/3, Techno 1.0 -> 0.83; the old formula adds EDM as 0.
        self.assertAlmostEqual(bg.macro_f1(rows), (2 / 3 + 1) / 2)
        self.assertAlmostEqual(bg.macro_f1(rows, union=True), (2 / 3 + 1 + 0) / 3)

    def test_risk_coverage_and_auto_error(self):
        rows = [row("House", "House", 0.99), row("House", "Techno", 0.6, auto=False),
                row("Techno", "Techno", 0.95), row("Techno", "House", 0.8)]
        m = bg.summarise(rows)
        self.assertEqual((m["auto"], m["auto_wrong"]), (3, 1))
        self.assertEqual(m["risk_coverage"][0.5], (0, 2))      # (wrong, songs)
        self.assertEqual(m["risk_coverage"][1.0], (2, 4))
        self.assertAlmostEqual(m["auto_error_upper"], fusion.error_upper_bound(1, 3))

    def test_family_decisions_flag_small_families(self):
        rows = [row("House", "House", 0.99), row("Techno", "Techno", 0.9, auto=False)]
        rows[1].suggested = True
        table = "\n".join(bg.family_decisions(rows))
        self.assertIn("| Techno | 1 | 1/1 (3%-100%) | 0 (0) | 1 (0) | 0 | too few to judge |", table)
        self.assertIn("| House | 1 | 0 | 95% |", table)

    def test_exact_interval(self):
        lo, hi = bg.exact_ci(45, 49)
        self.assertTrue(0.80 < lo < 0.82 and 0.97 < hi < 0.99)   # 81.3% to 97.7%
        self.assertEqual(bg.exact_ci(0, 0), (0.0, 1.0))

    def test_pair_sides(self):
        # A Tech House song: recognised only if the suggested subgenre says so.
        self.assertTrue(bg.side(("House", "Tech House"), "House", "Tech House", ("Techno", None)))
        self.assertFalse(bg.side(("House", "Tech House"), "House", "", ("Techno", None)))
        # Plain House vs Afro House: plain means "not Afro House".
        self.assertTrue(bg.side(("House", ""), "House", "Bassline", ("House", "Afro House")))
        self.assertFalse(bg.side(("House", ""), "House", "Afro House", ("House", "Afro House")))

    def test_tempo_only_hint_is_not_an_answer(self):
        p = genres.GenrePrediction(genres.UNCERTAIN, "Drum & Bass", candidates=[("Drum & Bass", 0)],
                                   sources=("tempo",))
        r = bg.make_row("1", "s", "Drum & Bass", "Drum & Bass", "", "g", p)
        self.assertEqual(r.top, "")

    def test_bootstrap_resamples_groups(self):
        rows = [row("House", "House", group="a", tid="1"), row("House", "House", group="a", tid="2"),
                row("Techno", "House", group="b", tid="3")]
        lo, hi = bg.bootstrap(rows, bg.accuracy)
        self.assertLessEqual(lo, hi)
        self.assertIn(lo, (0.0, 2 / 3, 1.0))


if __name__ == "__main__":
    unittest.main()
