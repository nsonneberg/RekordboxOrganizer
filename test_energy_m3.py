"""M3, the opt-in energy model: frozen inference, skipping, and that nothing else changes.

No test reads or needs a rating. The committed tests use synthetic inputs only. Parity
with the frozen predictions of real tracks runs where the private, git-ignored files in
benchmark_results/ exist, and is skipped elsewhere.
"""

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import numpy as np
import soundfile as sf

import energy
import energy_m3
import organizer
from test_energy import SR, banger, groove, write_ext

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(HERE, "benchmark_results", "energy_experiments")
FROZEN_MODEL = os.path.join(EXPERIMENTS, "frozen_models_round3_2026-10-27.json")
FROZEN_PREDICTIONS = os.path.join(EXPERIMENTS, "round3_predictions_2026-10-27.csv")
MEASUREMENTS = os.path.join(HERE, "benchmark_results", "energy_features.json")
private = unittest.skipUnless(all(map(os.path.exists, (FROZEN_MODEL, FROZEN_PREDICTIONS, MEASUREMENTS))),
                              "the private benchmark files aren't here")

# Scores are a sum of 22 terms of size about 1 at most, so float64 rounding (2.2e-16 per
# operation) can move one by about 22 * 2.2e-16 = 5e-15; a different BLAS can add order
# effects of the same size. 1e-12 is 200x that, and far below the gap to any cut-off,
# which test_levels_dont_hinge_on_the_tolerance checks.
SCORE_TOLERANCE = 1e-12
# The frozen model's order: pre-registration section 3.
ORDER = ["felt_bpm", "lufs", "lra", "density", "onsets", "pulse", "punch", "impact", "sustain",
         "w_punch", "w_density", "w_sustain",
         "v3a_e_loud", "v3a_e_punch", "v3a_e_power", "v3a_e_bass", "v3a_e_highs",
         "v3w_e_loud", "v3w_e_punch", "v3w_e_power", "v3w_e_bass", "v3w_e_highs"]


def track(tid="1", name="Song", genre="", bpm=128.0, rating=0, path="", waveform=""):
    return organizer.Track(tid, name, "", genre, "", bpm, "", 0, rating, "", "", 0, path,
                           waveform=waveform)


class SyntheticInference(unittest.TestCase):
    """Made-up inputs: each test would fail on a wrong order, scale, sign or cut-off."""

    def setUp(self):
        self.model = energy_m3.load_model()

    def test_a_small_model_worked_by_hand(self):
        toy = {"mu": [10.0, 20.0], "sd": [2.0, 4.0], "beta": [0.5, -1.0], "intercept": 3.0,
               "cuts": [2.0, 3.0, 4.0, 5.0]}
        # z = (12 - 10) / 2 = 1 and (16 - 20) / 4 = -1, so 3 + 0.5 * 1 + -1 * -1 = 4.5
        self.assertEqual(energy_m3.predict(toy, [12.0, 16.0]), (4.5, 4))

    def test_the_average_track_scores_the_intercept(self):
        score, level = energy_m3.predict(self.model, self.model["mu"])
        self.assertEqual(score, self.model["intercept"])
        self.assertEqual(level, 3)

    def test_each_input_moves_the_score_by_its_own_weight(self):
        for i, name in enumerate(ORDER):
            values = list(self.model["mu"])
            values[i] += self.model["sd"][i]       # one standard deviation up
            score, _ = energy_m3.predict(self.model, values)
            self.assertAlmostEqual(score, self.model["intercept"] + self.model["beta"][i], 12, name)

    def test_agrees_with_a_separate_calculation(self):
        rng = np.random.default_rng(7)
        m = self.model
        for _ in range(200):
            values = (np.array(m["mu"]) + rng.normal(0, 2, 22) * np.array(m["sd"])).tolist()
            expected = m["intercept"] + math.fsum(
                b * (v - mu) / sd for b, v, mu, sd in zip(m["beta"], values, m["mu"], m["sd"]))
            score, level = energy_m3.predict(m, values)
            self.assertAlmostEqual(score, expected, 11)
            self.assertEqual(level, 1 + sum(expected >= c for c in m["cuts"]))

    def test_a_score_on_a_cut_off_takes_the_higher_level(self):
        base = {**self.model, "mu": [0.0] * 22, "sd": [1.0] * 22, "beta": [0.0] * 22}
        for k, cut in enumerate(self.model["cuts"]):
            at = {**base, "intercept": cut}
            below = {**base, "intercept": math.nextafter(cut, -math.inf)}
            self.assertEqual(energy_m3.predict(at, [0.0] * 22)[1], k + 2)
            self.assertEqual(energy_m3.predict(below, [0.0] * 22)[1], k + 1)

    def test_the_real_model_reaches_every_level_either_side_of_each_cut(self):
        m = self.model
        lever = int(np.argmax(np.abs(m["beta"])))
        seen = set()
        for k, cut in enumerate(m["cuts"]):
            for offset, level in ((-1e-6, k + 1), (1e-6, k + 2)):
                values = list(m["mu"])
                values[lever] += m["sd"][lever] * (cut + offset - m["intercept"]) / m["beta"][lever]
                self.assertEqual(energy_m3.predict(m, values)[1], level)
                seen.add(level)
        self.assertEqual(seen, {1, 2, 3, 4, 5})

    def test_a_higher_score_never_lowers_the_level(self):
        base = {**self.model, "mu": [0.0] * 22, "sd": [1.0] * 22, "beta": [0.0] * 22}
        levels = [energy_m3.predict({**base, "intercept": s}, [0.0] * 22)[1]
                  for s in np.linspace(0.0, 6.0, 601)]
        self.assertEqual(levels, sorted(levels))
        self.assertEqual((levels[0], levels[-1]), (1, 5))

    def test_the_wrong_number_of_inputs_is_refused(self):
        for values in (self.model["mu"][:-1], self.model["mu"] + [0.0]):
            with self.assertRaises(ValueError):
                energy_m3.predict(self.model, values)

    def test_inputs_and_predict_work_together(self):
        names = {key: i + 1.0 for i, key in enumerate(ORDER)}
        audio = {m: names[k] for k, (src, m) in energy_m3.INPUTS.items() if src == "audio"}
        wave = {m: names[k] for k, (src, m) in energy_m3.INPUTS.items() if src == "waveform"}
        values, bad = energy_m3.model_inputs(1.0, "", audio, wave)
        self.assertEqual((values, bad), ([names[k] for k in ORDER], []))
        self.assertEqual(energy_m3.predict(self.model, values),
                         energy_m3.predict(self.model, [i + 1.0 for i in range(22)]))


class ModelFile(unittest.TestCase):
    def setUp(self):
        self.model = energy_m3.load_model()

    def test_order_and_shape(self):
        self.assertEqual(self.model["features"], ORDER)
        self.assertEqual(list(energy_m3.FEATURES), ORDER)
        for key in ("mu", "sd", "beta"):
            self.assertEqual(len(self.model[key]), 22, key)
            self.assertTrue(all(math.isfinite(v) for v in self.model[key]), key)
        self.assertTrue(all(v > 0 for v in self.model["sd"]))

    def test_cut_offs_rise(self):
        cuts = self.model["cuts"]
        self.assertEqual(len(cuts), 4)
        self.assertEqual(cuts, sorted(set(cuts)))

    def test_fingerprint_is_of_these_parameters(self):
        self.assertEqual(energy_m3._fingerprint(self.model), energy_m3.MODEL_FINGERPRINT)

    def test_the_frozen_constants_are_pinned(self):
        """So a new file and a new fingerprint together still fail a test."""
        self.assertEqual(self.model["cuts"], [2.06238377834877, 2.50265851844396,
                                              3.104593349783584, 4.088041420961431])
        self.assertEqual(self.model["intercept"], 3.103448275862069)
        self.assertEqual(self.model["alpha"], 30)
        self.assertEqual(energy_m3.MODEL_FINGERPRINT,
                         "f11e62783c7a4498368e048498489105f0f43b53eff60d6f0ae84f1dae30bcd3")


@private
class FrozenFiles(unittest.TestCase):
    """Against the evaluation's own files. These tests only read them."""

    def setUp(self):
        with open(FROZEN_MODEL, encoding="utf-8") as f:
            self.frozen = json.load(f)
        with open(FROZEN_PREDICTIONS, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def test_parameters_are_exactly_the_frozen_ones(self):
        model = energy_m3.load_model()
        ridge = self.frozen["ridge"]
        self.assertEqual(self.frozen["selected"], "M3 ridge calibrated")
        self.assertEqual(model["features"], self.frozen["features"])
        for key in ("mu", "sd", "intercept", "beta"):
            self.assertEqual(model[key], ridge[key], key)
        self.assertEqual(model["cuts"], self.frozen["cuts"])

    def test_it_names_the_frozen_files_it_came_from(self):
        def sha(path):
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        source = energy_m3.load_model()["frozen_source_sha256"]
        self.assertEqual(source, {"model": sha(FROZEN_MODEL), "predictions": sha(FROZEN_PREDICTIONS)})

    def test_all_78_from_the_stored_measurements(self):
        """The whole route: raw measurements -> model_inputs -> predict."""
        with open(MEASUREMENTS, encoding="utf-8") as f:
            tracks = json.load(f)["tracks"]
        model = energy_m3.load_model()
        self.assertEqual(len(self.rows), 78)
        self.assertEqual(sorted(r["track_id"] for r in self.rows), sorted(self.frozen["scorable_ids"]))
        for row in self.rows:
            t = tracks[row["track_id"]]
            values, bad = energy_m3.model_inputs(t["bpm"], t["genre"], {**t["audio_v3"], **t["audio_phrases"]},
                                                 {**t["waveform_v3"], **t["waveform"]})
            self.assertEqual(bad, [])
            self.assertEqual(values[0], float(row["felt_bpm"]))
            score, level = energy_m3.predict(model, values)
            self.assertEqual(level, int(row["selected_level"]), row["track_id"])
            self.assertLessEqual(abs(score - float(row["selected_score"])), SCORE_TOLERANCE)

    def test_levels_dont_hinge_on_the_tolerance(self):
        cuts = energy_m3.load_model()["cuts"]
        gap = min(abs(float(r["selected_score"]) - cut) for r in self.rows for cut in cuts)
        self.assertGreater(gap, 1000 * SCORE_TOLERANCE)

    def test_no_track_from_the_test_is_in_a_file_git_would_commit(self):
        try:
            listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=HERE,
                                    capture_output=True, text=True, check=True).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git isn't available here")
        pattern = re.compile(r"(?<!\d)(" + "|".join(r["track_id"] for r in self.rows) + r")(?!\d)")
        found = []
        for name in listed:
            path = os.path.join(HERE, name)
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="ignore") as f:
                    if pattern.search(f.read()):
                        found.append(name)
        self.assertEqual(found, [])


class Artifact(unittest.TestCase):
    def test_holds_no_tracks_or_ratings(self):
        with open(energy_m3.model_path(), encoding="utf-8") as f:
            model = json.load(f)
        self.assertEqual(set(model), {
            "schema", "name", "note", "features", "mu", "sd", "intercept", "beta", "cuts", "alpha",
            "n_development_tracks", "frozen_source_sha256", "evaluation"})
        self.assertEqual(set(model["evaluation"]), {
            "verdict", "n_scored", "mae", "mae_classic_v3", "mae_difference", "mae_difference_ci95",
            "listeners", "libraries"})

    def test_no_measurement_fixture_is_kept_in_the_repository(self):
        self.assertFalse(os.path.exists(os.path.join(HERE, "test_energy_m3_fixture.json")))
        with open(os.path.join(HERE, ".gitignore"), encoding="utf-8") as f:
            self.assertIn("benchmark_results/", f.read().split())

    def test_lives_outside_the_ignored_folder(self):
        self.assertEqual(os.path.dirname(energy_m3.model_path()), HERE)
        with open(os.path.join(HERE, "energy_m3.py"), encoding="utf-8") as f:
            self.assertNotIn("benchmark_results", f.read())

    def test_both_builds_package_it(self):
        with open(os.path.join(HERE, ".github", "workflows", "build.yml"), encoding="utf-8") as f:
            builds = [line for line in f if "pyinstaller --noconfirm" in line]
        self.assertEqual(len(builds), 2)
        for line in builds:
            self.assertIn('--add-data "energy_m3_model.json:."', line)

    def test_found_where_the_built_app_unpacks_it(self):
        with mock.patch.object(energy_m3.sys, "_MEIPASS", "bundle", create=True):
            self.assertEqual(energy_m3.model_path(), os.path.join("bundle", energy_m3.MODEL_FILE))

    def _changed(self, change):
        with open(energy_m3.model_path(), encoding="utf-8") as f:
            model = json.load(f)
        change(model)
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "m.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f)
        return path

    def test_refuses_changed_parameters(self):
        def nudge_beta(m):
            m["beta"][3] += 1e-9

        def reorder(m):
            m["features"][0], m["features"][1] = m["features"][1], m["features"][0]

        def shift_cut(m):
            m["cuts"][0] -= 0.01

        for change in (nudge_beta, reorder, shift_cut, lambda m: m.pop("cuts")):
            with self.assertRaises(energy_m3.ModelError):
                energy_m3.load_model(self._changed(change))

    def test_refuses_a_missing_or_damaged_file(self):
        with self.assertRaises(energy_m3.ModelError):
            energy_m3.load_model(os.path.join(HERE, "no such file.json"))
        path = self._changed(lambda m: None)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(energy_m3.ModelError):
            energy_m3.load_model(path)


class Inputs(unittest.TestCase):
    def setUp(self):
        self.audio = {name: 1.0 for src, name in energy_m3.INPUTS.values() if src == "audio"}
        self.wave = {name: 2.0 for src, name in energy_m3.INPUTS.values() if src == "waveform"}

    def test_each_input_comes_from_its_own_measurement(self):
        values, bad = energy_m3.model_inputs(128.0, "", self.audio, self.wave)
        self.assertEqual(bad, [])
        self.assertEqual(values, [128.0] + [1.0] * 8 + [2.0] * 3 + [1.0] * 5 + [2.0] * 5)

    def test_invalid_values_are_named(self):
        for value in (None, math.nan, math.inf, -math.inf, "3.5", True):
            _, bad = energy_m3.model_inputs(128.0, "", {**self.audio, "x_lufs": value}, self.wave)
            self.assertEqual(bad, ["lufs"], value)
            _, bad = energy_m3.model_inputs(128.0, "", self.audio, {**self.wave, "e_bass": value})
            self.assertEqual(bad, ["v3w_e_bass"], value)

    def test_a_missing_measurement_is_named(self):
        del self.audio["e_loud"]
        self.assertEqual(energy_m3.model_inputs(128.0, "", self.audio, self.wave)[1], ["v3a_e_loud"])

    def test_no_bpm_is_invalid(self):
        for bpm in (0, 0.0, -1.0, None, math.nan):
            self.assertEqual(energy_m3.model_inputs(bpm, "", self.audio, self.wave)[1], ["felt_bpm"], bpm)

    def test_felt_bpm_doubles_half_speed_drum_and_bass(self):
        self.assertEqual(energy_m3.model_inputs(87.0, "Drum & Bass", self.audio, self.wave)[0][0], 174.0)
        self.assertEqual(energy_m3.model_inputs(87.0, "House", self.audio, self.wave)[0][0], 87.0)


class MeasuredTracks(unittest.TestCase):
    """Real measuring of synthetic audio and rekordbox analysis files."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.cache = os.path.join(self.dir, "cache.json")
        self.model = energy_m3.load_model()
        self.signals = {"1": groove(), "2": banger()}
        self.tracks, self.features = [], {}
        for tid, signal in self.signals.items():
            path = os.path.join(self.dir, f"{tid}.wav")
            sf.write(path, signal.astype(np.float32), SR)
            ext = os.path.join(self.dir, f"{tid}.EXT")
            n = 6000
            heights = (np.arange(n) % 30) + 1
            write_ext(ext, heights, (np.arange(n) % 8), (np.arange(n) % 5), (np.arange(n) % 3))
            self.tracks.append(track(tid, f"Song {tid}", path=path, waveform=ext))
            self.features[tid] = {"x_mode": "audio"}

    def _direct(self, tid):
        t = next(t for t in self.tracks if t.id == tid)
        entries, sections, mood = energy.read_analysis(t.waveform)
        samples, sr = energy.decode(t.path)
        audio = energy.measure(samples, sr, sections, mood)
        return energy_m3.model_inputs(t.bpm, t.genre, audio, energy.measure_waveform(entries, sections, mood))

    def test_matches_measuring_each_kind_directly(self):
        scored, skipped = energy_m3.score_tracks(self.tracks, self.features, self.model, self.cache)
        self.assertEqual(skipped, {})
        for tid in self.signals:
            values, bad = self._direct(tid)
            self.assertEqual(bad, [])
            self.assertEqual(scored[tid], energy_m3.predict(self.model, values))

    def test_a_tracks_level_doesnt_depend_on_the_others(self):
        both, _ = energy_m3.score_tracks(self.tracks, self.features, self.model, self.cache)
        alone, _ = energy_m3.score_tracks(self.tracks[:1], self.features, self.model, self.cache)
        self.assertEqual(alone["1"], both["1"])

    def test_measuring_again_doesnt_decode_again(self):
        energy_m3.score_tracks(self.tracks, self.features, self.model, self.cache)
        with mock.patch.object(energy, "decode", side_effect=AssertionError("decoded again")):
            scored, skipped = energy_m3.score_tracks(self.tracks, self.features, self.model, self.cache)
        self.assertEqual((len(scored), skipped), (2, {}))

    def test_each_reason_for_skipping(self):
        missing_ext = os.path.join(self.dir, "gone.EXT")
        tracks = [track("no analysis", path=self.tracks[0].path),
                  track("unmeasured", path=self.tracks[0].path, waveform=self.tracks[0].waveform),
                  track("waveform only", path=self.tracks[0].path, waveform=self.tracks[0].waveform),
                  track("bad analysis", path=self.tracks[0].path, waveform=missing_ext),
                  track("no bpm", bpm=0.0, path=self.tracks[0].path, waveform=self.tracks[0].waveform)]
        features = {"no analysis": {"x_mode": "audio"}, "waveform only": {"x_mode": "waveform"},
                    "bad analysis": {"x_mode": "audio"}, "no bpm": {"x_mode": "audio"}}
        scored, skipped = energy_m3.score_tracks(tracks, features, self.model, self.cache)
        self.assertEqual(scored, {})
        self.assertEqual(skipped, {
            "no analysis": "no rekordbox analysis (needs direct mode and an analysed track)",
            "unmeasured": "it couldn't be measured",
            "waveform only": "its audio couldn't be decoded",
            "bad analysis": "its rekordbox analysis couldn't be read",
            "no bpm": "no BPM"})

    def test_audio_that_stopped_decoding_is_skipped(self):
        os.remove(self.tracks[0].path)
        scored, skipped = energy_m3.score_tracks(self.tracks, self.features, self.model, self.cache)
        self.assertEqual(list(scored), ["2"])
        self.assertEqual(skipped, {"1": "its audio couldn't be decoded"})

    def test_undefined_measurements_are_named(self):
        with mock.patch.object(energy, "measure_waveform", return_value={"x_punch": None}):
            _, skipped = energy_m3.score_tracks(self.tracks[:1], self.features, self.model, self.cache)
        self.assertIn("w_punch", skipped["1"])
        self.assertTrue(skipped["1"].startswith("missing or invalid measurements: "))


class Applying(unittest.TestCase):
    def _apply(self, tracks, scored, skipped=None, **options):
        logs = []
        opt = organizer.Options(energy=True, energy_m3=True, **options)
        with mock.patch.object(energy_m3, "score_tracks", return_value=(scored, skipped or {})) as run:
            ratings = organizer.apply_energy(tracks, opt, {}, logs.append)
        self.run_mock = run
        return ratings, logs

    def test_off_by_default(self):
        opt = organizer.Options()
        self.assertFalse(opt.energy_m3)
        self.assertFalse(opt.energy_experimental)

    def test_classic_doesnt_touch_m3(self):
        tracks = [track("1"), track("2")]
        features = {"1": {"e_loud": 1.0}, "2": {"e_loud": 2.0}}
        with mock.patch.object(energy_m3, "load_model", side_effect=AssertionError), \
                mock.patch.object(energy_m3, "score_tracks", side_effect=AssertionError):
            organizer.apply_energy(tracks, organizer.Options(energy=True), features, lambda m: None)
        self.assertTrue(all(t.energy for t in tracks))

    def test_sets_levels_and_new_ratings(self):
        tracks = [track("1"), track("2", rating=4)]
        ratings, logs = self._apply(tracks, {"1": (4.5, 5), "2": (1.0, 1)})
        self.assertEqual((tracks[0].energy, tracks[1].energy), (5, 1))
        self.assertEqual(ratings, {"1": 5})          # "2" keeps the rating it has
        self.assertEqual(tracks[1].rating, 4)
        self.assertIn("calibrated to one listener", logs[0])

    def test_skipped_tracks_keep_ratings_and_the_log_says_why(self):
        tracks = [track("1", "Kept", rating=3), track("2", "Unrated")]
        ratings, logs = self._apply(tracks, {}, {"1": "no BPM", "2": "no BPM"}, keep_ratings=False)
        self.assertEqual(ratings, {})
        self.assertEqual((tracks[0].rating, tracks[1].rating), (3, 0))
        self.assertEqual((tracks[0].energy, tracks[1].energy), (0, 0))
        text = "\n".join(logs)
        self.assertIn("skipped 2 tracks, which keep their ratings", text)
        self.assertIn("2: no BPM, e.g. Kept; Unrated", text)

    def test_scoring_nothing_warns_about_empty_playlists_and_sets(self):
        tracks = [track("1"), track("2")]
        _, logs = self._apply(tracks, {}, {"1": "no BPM", "2": "no BPM"})
        self.assertIn("energy playlists and sets will be empty", "\n".join(logs))
        _, logs = self._apply(tracks, {"1": (4.5, 5)}, {"2": "no BPM"})
        self.assertNotIn("will be empty", "\n".join(logs))

    def test_long_lists_of_skipped_tracks_are_shortened(self):
        tracks = [track(str(i), f"Song {i}") for i in range(10)]
        _, logs = self._apply(tracks, {}, {t.id: "no BPM" for t in tracks})
        self.assertTrue(any("and 7 more" in m for m in logs), logs)

    def test_no_stars_unless_asked(self):
        tracks = [track("1")]
        self.assertEqual(self._apply(tracks, {"1": (4.5, 5)}, energy_ratings=False)[0], {})
        self.assertEqual((tracks[0].energy, tracks[0].rating), (5, 0))

    def test_rules_using_energy_get_levels_but_no_stars(self):
        logs = []
        tracks = [track("1")]
        opt = organizer.Options(energy=False, energy_m3=True)
        with mock.patch.object(energy_m3, "score_tracks", return_value=({"1": (4.5, 5)}, {})):
            self.assertEqual(organizer.apply_energy(tracks, opt, {}, logs.append), {})
        self.assertEqual((tracks[0].energy, tracks[0].rating), (5, 0))

    def test_wins_over_a_saved_experimental_tick(self):
        tracks = [track("1")]
        self._apply(tracks, {"1": (3.5, 4)}, energy_experimental=True)
        self.assertEqual(tracks[0].energy, 4)

    def test_uses_the_tracks_own_genre_not_a_guess(self):
        tracks = [track("1", genre="House")]
        opt = organizer.Options(energy=True, energy_m3=True)
        with mock.patch.object(energy_m3, "score_tracks", return_value=({}, {})) as run:
            organizer.apply_energy(tracks, opt, {}, lambda m: None, {"1": "Drum & Bass"})
        self.assertEqual(run.call_args[0][0][0].genre, "House")

    def test_a_broken_model_file_changes_nothing(self):
        logs = []
        tracks = [track("1", rating=2)]
        tracks[0].energy = 3
        opt = organizer.Options(energy=True, energy_m3=True, keep_ratings=False)
        with mock.patch.object(energy_m3, "load_model", side_effect=energy_m3.ModelError("broken")):
            self.assertEqual(organizer.apply_energy(tracks, opt, {"1": {"e_loud": 1.0}}, logs.append), {})
        self.assertEqual((tracks[0].energy, tracks[0].rating), (0, 2))
        self.assertIn("ratings stay as they are", logs[0])


class XmlRun(unittest.TestCase):
    """An XML export has no rekordbox analysis, so M3 is off and the run says so."""

    def _run(self, preview=False, **options):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        audio = os.path.join(folder, "a.wav")
        sf.write(audio, banger().astype(np.float32), SR)
        location = audio.replace(os.sep, "/").lstrip("/")
        tracks = (f'<TRACK TrackID="1" Name="Rated" Artist="A" AverageBpm="128" Rating="51" Location="file://localhost/{location}"/>'
                  f'<TRACK TrackID="2" Name="Unrated" Artist="A" AverageBpm="128" Rating="0" Location="file://localhost/{location}"/>')
        path = os.path.join(folder, "lib.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'<DJ_PLAYLISTS Version="1.0.0"><COLLECTION>{tracks}</COLLECTION>'
                    '<PLAYLISTS><NODE Type="0" Name="ROOT" Count="0"/></PLAYLISTS></DJ_PLAYLISTS>')
        logs = []
        opt = organizer.Options(energy=True, energy_m3=True, genre=False, bpm=False, key=False,
                                duplicates=False, missing_files=False, missing_info=False,
                                guess_genres=False, **options)
        with mock.patch.object(energy, "cache_path", lambda: os.path.join(folder, "cache.json")), \
                mock.patch.object(energy_m3, "score_tracks", side_effect=AssertionError("M3 ran")):
            out = organizer.run(opt, path, log=logs.append, preview=preview)
        return out, "\n".join(logs)

    def test_a_saved_m3_tick_uses_classic_and_says_why(self):
        out, logs = self._run()
        self.assertIn("M3 is off for XML exports", logs)
        self.assertIn("Using the classic algorithm", logs)
        self.assertIn("Energy levels: ", logs)
        self.assertNotIn("no rekordbox analysis", logs)
        root = ET.parse(out).getroot()
        ratings = {t.get("TrackID"): t.get("Rating") for t in root.iter("TRACK") if t.get("TrackID")}
        self.assertEqual(ratings["1"], "51")         # keeps a rating already set
        self.assertNotEqual(ratings["2"], "0")
        folder = root.find("PLAYLISTS/NODE/NODE[@Name='Organized']/NODE[@Name='By Energy']")
        self.assertEqual(sum(len(pl.findall("TRACK")) for pl in folder), 2)    # not empty

    def test_a_set_isnt_left_empty(self):
        _, logs = self._run(set_build=True, set_length=2, set_start=1, set_end=5)
        self.assertIn("M3 is off for XML exports", logs)
        self.assertIn("Set '", logs)

    def test_it_says_so_when_experimental_is_what_runs(self):
        _, logs = self._run(energy_experimental=True)
        self.assertIn("Using the experimental algorithm", logs)

    def test_preview_says_so_too(self):
        _, logs = self._run(preview=True)
        self.assertIn("M3 is off for XML exports", logs)


class Window(unittest.TestCase):
    def setUp(self):
        import tk_test_root
        self.root = tk_test_root.open_root(self)
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.settings = os.path.join(self.dir, "settings.json")
        import app
        self.app_mod = app
        p = mock.patch.object(app, "settings_path", lambda: self.settings)
        p.start()
        self.addCleanup(p.stop)

    def _make(self, saved=None, source=None):
        if saved is not None:
            with open(self.settings, "w", encoding="utf-8") as f:
                data = {"version": self.app_mod.SETTINGS_VERSION, "options": saved}
                if source:
                    data["source"] = source
                json.dump(data, f)
        app = self.app_mod.App(self.root)
        self.root.update_idletasks()
        return app

    def _state(self, widget):
        return str(widget.cget("state"))

    def test_off_in_a_new_install(self):
        self.assertFalse(self._make().collect_options().energy_m3)

    def test_older_settings_keep_their_energy_choices(self):
        app = self._make({"energy": True, "energy_experimental": True, "energy_relative": True})
        opt = app.collect_options()
        self.assertEqual((opt.energy, opt.energy_experimental, opt.energy_relative, opt.energy_m3),
                         (True, True, True, False))

    def test_the_label_says_one_listener(self):
        app = self._make()
        self.assertIn("one listener", app.energy_m3_cb.cget("text"))

    def test_ticks_are_saved(self):
        app = self._make({"energy": True, "energy_m3": True})
        self.assertTrue(app.collect_options().energy_m3)

    def test_m3_and_experimental_exclude_each_other(self):
        app = self._make({"energy": True})
        app.vars["energy_m3"].set(True)
        self.root.update()
        self.assertEqual(self._state(app.energy_experimental_cb), "disabled")
        self.assertEqual(self._state(app.energy_relative_cb), "disabled")
        self.assertEqual(self._state(app.energy_m3_cb), "normal")
        app.vars["energy_m3"].set(False)
        app.vars["energy_experimental"].set(True)
        self.root.update()
        self.assertEqual(self._state(app.energy_m3_cb), "disabled")
        self.assertEqual(self._state(app.energy_experimental_cb), "normal")

    def test_switching_source_doesnt_need_a_wider_window(self):
        """The window keeps its size when the source changes, and the XML text is longer."""
        app = self._make({"energy": True, "energy_m3": True})
        self.root.update()
        width = self.root.winfo_reqwidth()
        app.source.set("xml")
        self.root.update()
        self.assertEqual(self.root.winfo_reqwidth(), width)

    def test_text_wraps_wider_on_scaled_displays(self):
        app = self._make()
        dpi = app.winfo_fpixels("1i")
        self.assertEqual(int(str(app.energy_m3_hint.cget("wraplength"))), round(596 * dpi / 96))
        self.addCleanup(self.root.tk.call, "tk", "scaling", self.root.tk.call("tk", "scaling"))
        # Tk keeps the screen's size in whole millimetres, so the DPI it reports is only
        # close to the scale asked for, by an amount that depends on the screen's size.
        for scale in (1, 1.25, 1.5, 2, 3):
            with self.subTest(scale=scale):
                self.root.tk.call("tk", "scaling", scale * 96 / 72)
                app = self._make()
                dpi = app.winfo_fpixels("1i")
                self.assertAlmostEqual(dpi, scale * 96, delta=scale * 96 * 0.02)
                self.assertEqual(int(str(app.energy_m3_hint.cget("wraplength"))),
                                 round(596 * dpi / 96))

    def test_both_saved_ticks_can_still_be_undone(self):
        app = self._make({"energy": True, "energy_m3": True, "energy_experimental": True})
        self.root.update()
        self.assertEqual(self._state(app.energy_m3_cb), "normal")
        self.assertTrue(app.collect_options().energy_m3)

    def test_switching_to_xml_turns_it_off_and_explains(self):
        app = self._make({"energy": True, "energy_m3": True})
        self.root.update()
        self.assertTrue(app.collect_options().energy_m3)
        self.assertIn("energy (M3)", app.summary.cget("text"))
        self.assertIn("Fixed levels", app.energy_m3_hint.cget("text"))
        app.source.set("xml")
        self.root.update()
        self.assertEqual(self._state(app.energy_m3_cb), "disabled")
        self.assertFalse(app.collect_options().energy_m3)
        self.assertNotIn("M3", app.summary.cget("text"))
        hint = app.energy_m3_hint.cget("text")
        self.assertIn("Off for XML exports", hint)
        self.assertIn("classic", hint)
        self.assertEqual(self._state(app.energy_experimental_cb), "normal")

    def test_the_explanation_names_what_will_run(self):
        app = self._make({"energy": True, "energy_m3": True, "energy_experimental": True}, "xml")
        self.root.update()
        self.assertIn("experimental", app.energy_m3_hint.cget("text"))

    def test_switching_back_restores_the_choice(self):
        app = self._make({"energy": True, "energy_m3": True})
        app.source.set("xml")
        self.root.update()
        app.source.set("direct")
        self.root.update()
        self.assertEqual(self._state(app.energy_m3_cb), "normal")
        self.assertTrue(app.collect_options().energy_m3)
        self.assertIn("energy (M3)", app.summary.cget("text"))

    def test_a_saved_tick_loaded_in_xml_mode_stays_off(self):
        app = self._make({"energy": True, "energy_m3": True}, "xml")
        self.root.update()
        self.assertEqual(self._state(app.energy_m3_cb), "disabled")
        self.assertFalse(app.collect_options().energy_m3)
        self.assertIn("Off for XML exports", app.energy_m3_hint.cget("text"))
        self.assertTrue(app.vars["energy_m3"].get())

    def test_the_tick_survives_saving_in_either_mode(self):
        for source in ("direct", "xml"):
            with self.subTest(source=source):
                app = self._make({"energy": True, "energy_m3": True}, source)
                app.save_settings()
                app.destroy()
                with open(self.settings, encoding="utf-8") as f:
                    saved = json.load(f)
                self.assertTrue(saved["options"]["energy_m3"])
                self.assertEqual(saved["source"], source)
                reopened = self._make()
                self.assertTrue(reopened.vars["energy_m3"].get())
                self.assertEqual(reopened.source.get(), source)
                reopened.destroy()

    def test_unticking_is_saved(self):
        app = self._make({"energy": True, "energy_m3": True})
        app.vars["energy_m3"].set(False)
        app.save_settings()
        app.destroy()
        self.assertFalse(self._make().vars["energy_m3"].get())

    def test_v3_stays_the_default_in_saved_settings_too(self):
        app = self._make()
        app.save_settings()
        with open(self.settings, encoding="utf-8") as f:
            options = json.load(f)["options"]
        self.assertFalse(options["energy_m3"])
        self.assertFalse(options["energy_experimental"])


if __name__ == "__main__":
    unittest.main()
