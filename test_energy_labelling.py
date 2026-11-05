"""Tests for energy_labelling.py on made-up libraries (no real music or library)."""

import csv
import os
import random
import tempfile
import unittest
from collections import Counter
from unittest import mock

import benchmark_energy as bench
import energy_labelling as lab


def library(n=200, seed=1):
    rng = random.Random(seed)
    fams = ["House"] * 4 + ["Drum & Bass"] * 3 + ["Dubstep"] * 2 + ["(none)"] * 4 + ["Pop"]
    rows = []
    for i in range(n):
        fam = rng.choice(fams)
        bpm = {"House": 124, "Drum & Bass": 174, "Dubstep": 140, "Pop": 100}.get(fam, 128)
        rows.append({
            "id": str(i), "artist": f"Artist {i % 70}", "title": f"Song {i}", "group": str(i),
            "family": fam, "path_ok": True, "felt_bpm": bpm + rng.uniform(-8, 8),
            "lufs": rng.uniform(-16, -6), "density": rng.uniform(0.2, 0.6),
            "onsets": rng.uniform(1, 6), "pulse": rng.uniform(0.2, 0.9),
            "w_punch": rng.uniform(0.1, 0.5), "w_power": rng.uniform(0.0, 0.3),
            "w_highs": rng.uniform(0.1, 0.4), "old_waveform_level": rng.randint(1, 5),
        })
    return rows


def rated_for(rows, k=50, seed=2):
    rng = random.Random(seed)
    return {r["id"]: float(rng.randint(1, 5)) for r in rng.sample(rows, k)}


class Selection(unittest.TestCase):
    def setUp(self):
        self.rows = library()
        self.rows[5]["path_ok"] = False
        self.rows[6]["group"] = self.rows[7]["group"] = "copy"
        self.rated = rated_for(self.rows)
        for i in ("5", "6", "7"):
            self.rated.pop(i, None)
        self.res = lab.select(self.rows, self.rated)

    def test_every_track_is_in_the_pool_or_excluded_with_a_reason(self):
        pooled = {p["id"] for p in self.res["pool"]}
        self.assertEqual(pooled | set(self.res["excluded"]), {r["id"] for r in self.rows})
        self.assertFalse(pooled & set(self.res["excluded"]))
        self.assertTrue(all(self.res["excluded"].values()))

    def test_rated_missing_and_copies_are_excluded(self):
        ex = self.res["excluded"]
        self.assertTrue(all(ex[i] == "already rated" for i in self.rated))
        self.assertIn("not found", ex["5"])
        self.assertEqual(sum(i in ex for i in ("6", "7")), 1)   # one copy kept

    def test_nothing_meeting_a_criterion_is_dropped(self):
        pooled = {p["id"] for p in self.res["pool"]}
        eligible = [r for r in self.rows if r["id"] not in self.res["excluded"]]
        self.assertEqual(len(pooled), len(eligible))
        self.assertEqual(self.res["any_criterion"] + self.res["no_criterion"], len(eligible))

    def test_low_batch_has_at_least_30_diverse_candidates(self):
        low = [p for p in self.res["pool"] if p["low_batch"]]
        self.assertGreaterEqual(len(low), 30)
        self.assertTrue(all(p["batch"] == 1 and p["low_signals"] >= 1 for p in low))
        self.assertLessEqual(max(Counter(p["unit"] for p in low).values()), 3)
        self.assertGreater(len({p["family"] for p in low}), 2)
        self.assertEqual(sum(p["batch"] == 1 for p in self.res["pool"]), len(low) + lab.CONTROLS)

    def test_low_candidates_beyond_batch_1_stay_in_the_pool(self):
        later = [p for p in self.res["pool"] if p["low_signals"] and not p["low_batch"]]
        self.assertEqual(len(later) + lab.LOW_BATCH, self.res["low_candidates"])
        self.assertTrue(all(p["batch"] >= 2 for p in later))

    def test_low_candidates_come_from_either_baseline_not_just_both(self):
        low = [p for p in self.res["pool"] if p["low_signals"]]
        only_one = [p for p in low if ("old_waveform_1_2" in p["criteria"])
                    != ("tempo_only_1_2" in p["criteria"])]
        self.assertTrue(only_one)

    def test_split_keeps_artist_clusters_together_and_away_from_rated_songs(self):
        unit_split = {}
        for p in self.res["pool"]:
            self.assertEqual(unit_split.setdefault(p["unit"], p["split"]), p["split"])
        units = lab.units(self.rows)
        linked = {units[i] for i in self.rated}
        for p in self.res["pool"]:
            if p["unit"] in linked:
                self.assertEqual(p["split"], "dev")
        held = sum(p["split"] == "held_out" for p in self.res["pool"])
        self.assertGreater(held, 0)

    def test_selection_is_deterministic(self):
        again = lab.select([dict(r) for r in self.rows], dict(self.rated))
        self.assertEqual([(p["id"], p["batch"], p["split"]) for p in again["pool"]],
                         [(p["id"], p["batch"], p["split"]) for p in self.res["pool"]])


class Sheets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = os.path.join(self.tmp.name, "round2")
        rows = library()
        self.res = lab.select(rows, rated_for(rows))
        self.summary = lab.write_selection(self.res, self.folder)

    def tearDown(self):
        self.tmp.cleanup()

    def sheets(self):
        return sorted(n for n in os.listdir(self.folder) if n.startswith("batch_"))

    def test_sheets_are_blind_and_unrated(self):
        for name in self.sheets():
            with open(os.path.join(self.folder, name), newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["track_id", "artist", "title", "rating", "notes"])
            self.assertTrue(all(r[3] == "" and r[4] == "" for r in rows[1:]))

    def test_sheet_order_is_not_the_priority_order(self):
        with open(os.path.join(self.folder, "batch_01.csv"), newline="", encoding="utf-8") as f:
            ids = [r["track_id"] for r in csv.DictReader(f)]
        low_first = [p["id"] for p in self.res["pool"] if p["batch"] == 1 and p["low_batch"]]
        self.assertNotEqual(ids[:len(low_first)], low_first)

    def test_existing_folder_is_never_overwritten(self):
        with self.assertRaises(SystemExit):
            lab.write_selection(self.res, self.folder)

    def rate(self, name, value):
        path = os.path.join(self.folder, name)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["rating"] = value
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["track_id", "artist", "title", "rating", "notes"])
            w.writeheader()
            w.writerows(rows)

    def test_ratings_are_read_per_split(self):
        for name in self.sheets():
            self.rate(name, "3")
        with mock.patch.object(lab, "ORIGINAL_LABELS", os.path.join(self.folder, "none.csv")):
            dev = lab.load_ratings(self.folder, "dev")
            held = lab.load_ratings(self.folder, "held_out")
        split = {p["id"]: p["split"] for p in self.res["pool"]}
        self.assertTrue(dev and held)
        self.assertFalse(set(dev) & set(held))
        self.assertTrue(all(split[i] == "dev" for i in dev))
        self.assertTrue(all(split[i] == "held_out" for i in held))
        self.assertIn("rated", lab.status(self.folder))

    def test_imported_ratings_keep_the_frozen_split(self):
        split = {p["id"]: p["split"] for p in self.res["pool"]}
        held = next(i for i, sp in split.items() if sp == "held_out")
        dev = next(i for i, sp in split.items() if sp == "dev")
        with open(os.path.join(self.folder, "import_rekordbox_test.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "uuid", "artist", "title", "rating", "rater", "source"])
            w.writerow([held, "", "", "", "1", "", "rekordbox"])
            w.writerow([dev, "", "", "", "2", "", "rekordbox"])
        with open(os.path.join(self.folder, "import_rekordbox_test.skipped.csv"), "w",
                  newline="", encoding="utf-8") as f:
            f.write("track_id,rating\n" + dev + ",5\n")
        with mock.patch.object(lab, "ORIGINAL_LABELS", os.path.join(self.folder, "none.csv")):
            self.assertEqual(lab.load_ratings(self.folder, "held_out"), {held: [(1.0, "")]})
            self.assertEqual(lab.load_ratings(self.folder, "dev"), {dev: [(2.0, "")]})
            self.assertEqual(lab.load_ratings(self.folder, "dev", imports=False), {})
        self.assertIn("dev      :   1 of", lab.status(self.folder))
        self.assertIn("held_out :   1 of", lab.status(self.folder))

    def test_changed_manifest_is_refused(self):
        with open(os.path.join(self.folder, "manifest.csv"), "a", encoding="utf-8") as f:
            f.write("x\n")
        with self.assertRaises(SystemExit):
            lab.load_ratings(self.folder, "dev")


class Round3Isolation(unittest.TestCase):
    """Round-3 test songs are also listed (unrated) on round-2 sheets, as in the real
    library; ratings must never cross between the rounds in either direction."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.r2 = os.path.join(self.tmp.name, "round2")
        self.r3 = os.path.join(self.tmp.name, "round3")
        self.template = os.path.join(self.tmp.name, "template.csv")
        rows = library()
        self.res = lab.select(rows, rated_for(rows))
        lab.write_selection(self.res, self.r2)
        dev = [p for p in self.res["pool"] if p["split"] == "dev"]
        a, b, c, d, e, f = (p["id"] for p in dev[:6])
        self.ids = dict(scorable=[a, b, c], unscorable=d, copy=e, other=f)
        mk = lambda i, status, copy_of="", copies="": {"track_id": i, "artist": f"A{i}", "title": f"T{i}",
                                                       "status": status, "copy_of": copy_of, "unit": i,
                                                       "copies": copies}
        self.rows = [mk(a, "scorable", copies=f"{a};{e};lib-copy"), mk(b, "scorable"), mk(c, "scorable"),
                     mk(d, "unscorable"), mk(e, "excluded_copy", copy_of=a)]
        self.summary = lab.write_round3(self.rows, self.r3, {"created": "test"})
        self.patch = mock.patch.object(lab, "ORIGINAL_LABELS", self.template)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def write(self, path, ratings, header=("track_id", "artist", "title", "rating", "notes")):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i, v in ratings:
                w.writerow([i, "", "", v, ""][:len(header)])

    def rate_sheet(self, values):
        path = os.path.join(self.r3, "round3_sheet.csv")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["rating"] = values.get(r["track_id"], "")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["track_id", "artist", "title", "rating", "notes"])
            w.writeheader()
            w.writerows(rows)

    # manifest and sheet
    def test_manifest_hash_statuses_and_blind_sheet(self):
        man = lab.round3_manifest(self.r3)
        self.assertEqual([r["status"] for r in man], ["scorable"] * 3 + ["unscorable", "excluded_copy"])
        with open(os.path.join(self.r3, "round3_sheet.csv"), newline="", encoding="utf-8") as f:
            sheet = list(csv.reader(f))
        self.assertEqual(sheet[0], ["track_id", "artist", "title", "rating", "notes"])
        self.assertEqual({r[0] for r in sheet[1:]}, set(self.ids["scorable"]) | {self.ids["unscorable"]})
        self.assertTrue(all(r[3] == r[4] == "" for r in sheet[1:]))
        self.assertEqual(self.summary["sheet_rows"], 4)
        self.assertEqual(lab.round3_protected(self.r3),
                         set(self.ids["scorable"]) | {self.ids["unscorable"], self.ids["copy"], "lib-copy"})

    def test_changed_manifest_is_refused_everywhere(self):
        with open(os.path.join(self.r3, "manifest.csv"), "a", encoding="utf-8") as f:
            f.write("x,y,z,scorable,,x,\n")
        for call in (lambda: lab.load_round3(self.r3, self.r2), lambda: lab.round3_manifest(self.r3),
                     lambda: lab.load_ratings(self.r2, "dev", round3=self.r3)):
            with self.assertRaises(SystemExit):
                call()

    def test_bad_manifests_are_not_written(self):
        a = self.ids["scorable"][0]
        bad = [self.rows + [dict(self.rows[1])],                                   # duplicate id
               [dict(self.rows[0], status="maybe")],                               # unknown status
               [dict(self.rows[0]), dict(self.rows[4], copy_of=self.ids["scorable"][1])],  # not a known copy
               [dict(self.rows[0], copy_of=a)]]                                    # copy_of on a recording
        for n, rows in enumerate(bad):
            with self.assertRaises(SystemExit):
                lab.write_round3(rows, os.path.join(self.tmp.name, f"bad{n}"), {})
        with self.assertRaises(SystemExit):
            lab.write_round3(self.rows, self.r3, {})                               # never overwritten

    # round 3 -> round 2
    def test_round3_song_rated_on_a_round2_sheet_is_refused(self):
        a = self.ids["scorable"][0]
        batch = next(p["batch"] for p in self.res["pool"] if p["id"] == a)
        path = os.path.join(self.r2, f"batch_{batch:02d}.csv")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["rating"] = "2" if r["track_id"] == a else ""
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["track_id", "artist", "title", "rating", "notes"])
            w.writeheader()
            w.writerows(rows)
        for call in (lambda: lab.load_ratings(self.r2, "dev", round3=self.r3),
                     lambda: lab.load_ratings(self.r2, "held_out", round3=self.r3),
                     lambda: lab.development_ratings(self.r2, self.r3),
                     lambda: lab.load_round3(self.r3, self.r2)):
            with self.assertRaises(SystemExit):
                call()

    def test_round3_rating_smuggled_in_as_a_round2_import_is_refused(self):
        self.write(os.path.join(self.r2, "import_round3_copy.csv"), [(self.ids["unscorable"], "1")])
        with self.assertRaises(SystemExit):
            lab.development_ratings(self.r2, self.r3)

    def test_known_copy_rated_in_round2_is_refused(self):
        for copy in (self.ids["copy"], "lib-copy"):
            path = os.path.join(self.r2, "import_rekordbox_x.csv")
            self.write(path, [(copy, "3")])
            with self.assertRaises(SystemExit):
                lab.load_ratings(self.r2, "dev", round3=self.r3)
            os.remove(path)

    def test_round3_song_in_the_first_round_template_is_refused(self):
        self.write(self.template, [(self.ids["scorable"][1], "4")])
        with self.assertRaises(SystemExit):
            lab.development_ratings(self.r2, self.r3)

    def test_invalid_or_blank_text_still_counts_as_a_round2_rating(self):
        self.write(os.path.join(self.r2, "import_x.csv"), [(self.ids["scorable"][2], "high")])
        with self.assertRaises(SystemExit):
            lab.load_ratings(self.r2, "dev", round3=self.r3)

    def test_development_ratings_join_both_round2_splits_without_round3(self):
        self.write(self.template, [("t1", "2")])
        held = next(p["id"] for p in self.res["pool"] if p["split"] == "held_out")
        self.write(os.path.join(self.r2, "import_y.csv"), [(held, "1"), (self.ids["other"], "5")])
        self.rate_sheet({self.ids["scorable"][0]: "3"})
        got = lab.development_ratings(self.r2, self.r3)
        self.assertEqual(set(got), {"t1", held, self.ids["other"]})
        self.assertFalse(set(got) & lab.round3_protected(self.r3))

    # round 2 -> round 3, and round-3 import rules
    def test_round3_ratings_are_read_once_per_recording(self):
        a, b, c = self.ids["scorable"]
        self.rate_sheet({a: "1", self.ids["unscorable"]: "5"})
        self.write(os.path.join(self.r3, "round3_import_extra.csv"), [(b, "4")])
        self.assertEqual(lab.load_round3(self.r3, self.r2),
                         {a: (1, "round3_sheet.csv"), b: (4, "round3_import_extra.csv"),
                          self.ids["unscorable"]: (5, "round3_sheet.csv")})

    def test_round3_refusals(self):
        a = self.ids["scorable"][0]
        cases = {"round3_import_dup.csv": [(a, "2")],                     # a is rated on the sheet too
                 "round3_import_twice.csv": [(self.ids["scorable"][1], "2"), (self.ids["scorable"][1], "2")],
                 "round3_import_copy.csv": [(self.ids["copy"], "2")],     # excluded copy
                 "round3_import_unknown.csv": [("nope", "2")],
                 "round3_import_bad.csv": [(self.ids["scorable"][2], "2.5")],
                 "batch_01.csv": [(self.ids["scorable"][2], "3")],         # round-2 style file
                 "import_rekordbox.csv": [(self.ids["scorable"][2], "3")]}
        for name, ratings in cases.items():
            self.rate_sheet({a: "3"})
            path = os.path.join(self.r3, name)
            self.write(path, ratings)
            with self.assertRaises(SystemExit, msg=name):
                lab.load_round3(self.r3, self.r2)
            os.remove(path)
        self.assertEqual(lab.load_round3(self.r3, self.r2), {a: (3, "round3_sheet.csv")})

    def test_round2_rated_song_put_into_round3_is_refused(self):
        other = self.ids["other"]
        self.write(os.path.join(self.r2, "import_z.csv"), [(other, "2")])
        rows = [dict(self.rows[1], track_id=other, unit=other)]
        folder = os.path.join(self.tmp.name, "r3b")
        lab.write_round3(rows, folder, {})
        with self.assertRaises(SystemExit):
            lab.load_round3(folder, self.r2)


class OriginalTemplate(unittest.TestCase):
    def test_rated_template_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "energy_labels_template.csv")
            content = "track_id,artist,title,genre,rating,rater,notes\n1,A,B,C,4,,\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            with mock.patch.object(bench, "OUT_DIR", tmp), \
                    mock.patch.object(bench, "_load_library", return_value=[]):
                with self.assertRaises(SystemExit):
                    bench.write_template(5)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), content)


if __name__ == "__main__":
    unittest.main()
