import json
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import numpy as np
import soundfile as sf

import benchmark_energy as bench
import energy
import organizer

SR = 22050


def tone(seconds=40):
    """Quiet, smooth sine: low energy."""
    t = np.arange(SR * seconds) / SR
    return 0.05 * np.sin(2 * np.pi * 220 * t)


def groove(seconds=40, bpm=110):
    """Moderate kick pattern with a pad."""
    t = np.arange(SR * seconds) / SR
    out = 0.15 * np.sin(2 * np.pi * 110 * t)
    beat = int(SR * 60 / bpm)
    kick = 0.4 * np.sin(2 * np.pi * 60 * np.arange(2000) / SR) * np.exp(-np.arange(2000) / 400)
    for start in range(0, len(out) - 2000, beat):
        out[start:start + 2000] += kick
    return out


def banger(seconds=40, bpm=140):
    """Loud kicks plus noisy hi-hats: high energy."""
    rng = np.random.default_rng(0)
    out = groove(seconds, bpm) * 2
    beat = int(SR * 60 / bpm)
    hat = rng.standard_normal(600) * np.exp(-np.arange(600) / 100) * 0.5
    for start in range(beat // 2, len(out) - 600, beat // 2):
        out[start:start + 600] += hat
    return np.clip(out, -1, 1)


def write_ext(path, heights, low, mid, high):
    """Write a minimal rekordbox .EXT analysis file with a PWV5 colour waveform."""
    entries = ((np.asarray(low) << 13) | (np.asarray(mid) << 10) | (np.asarray(high) << 7)
               | (np.asarray(heights) << 2)).astype(">u2").tobytes()
    n = len(entries) // 2
    header = b"PMAI" + (28).to_bytes(4, "big") + (28 + 24 + 2 * n).to_bytes(4, "big") + bytes(16)
    tag = (b"PWV5" + (24).to_bytes(4, "big") + (24 + 2 * n).to_bytes(4, "big")
           + (2).to_bytes(4, "big") + n.to_bytes(4, "big") + bytes(4))
    with open(path, "wb") as f:
        f.write(header + tag + entries)


def measured(signal, sections=None):
    return energy.measure(np.asarray(signal, dtype=np.float32), SR, sections)


def scored(signal, bpm=0.0, genre=""):
    return energy.score_track(measured(signal), bpm, genre)


def fake(value=0.5, **extra):
    """Audio-mode measurements with every scaled component at `value` (0-1)."""
    a = energy.CALIBRATION["audio"]
    f = {k: lo + value * (hi - lo) for k, (lo, hi) in a.items()}
    return f | {"x_mode": "audio", "x_seconds": 120.0, "x_warnings": [], "x_tempo": None,
                "mood": 0} | extra


class SyntheticSignals(unittest.TestCase):
    """The designed cases of benchmark_energy.py (shortened). Expectations are
    the design's, not listeners' ratings."""

    @classmethod
    def setUpClass(cls):
        cls.peak = bench.peak_time(45)
        cls.results = {
            "peak": scored(cls.peak, 140, "Techno"),
            "quiet": scored(cls.peak * 0.03, 140, "Techno"),
            "ambient": scored(bench.fast_ambient(45), 170, "Ambient"),
            "slow": scored(bench.slow_forceful(45), 85, "Hip Hop"),
            "restrained": scored(bench.fast_restrained(45), 128, "House"),
            "drone": scored(bench.drone(45)),
        }

    def test_mastering_level_doesnt_change_the_score(self):
        loud, quiet = self.results["peak"], self.results["quiet"]
        self.assertAlmostEqual(loud.score, quiet.score, delta=1.0)
        self.assertEqual(loud.level, quiet.level)
        self.assertGreaterEqual(quiet.level, 4)       # quiet master of an intense track

    def test_fast_ambient_isnt_high(self):
        r = self.results["ambient"]
        self.assertLessEqual(r.level, 2)
        self.assertIn("little steady beat, so tempo barely counts", r.warnings)

    def test_slow_forceful_beats_fast_restrained(self):
        self.assertGreater(self.results["slow"].score, self.results["restrained"].score)
        self.assertGreater(self.results["slow"].level, self.results["restrained"].level)

    def test_a_drone_is_the_lowest_level(self):
        self.assertEqual(self.results["drone"].level, 1)
        self.assertIn("no usable BPM and no clear beat: tempo not counted",
                      self.results["drone"].warnings)

    def test_identical_tracks_get_identical_results(self):
        again = scored(self.peak, 140, "Techno")
        self.assertEqual(again.score, self.results["peak"].score)
        self.assertEqual(again.components, self.results["peak"].components)

    def test_too_short_and_silent_tracks_are_flagged_not_rated(self):
        for signal in (bench.peak_time(15), np.zeros(SR * 45)):
            r = scored(signal, 140)
            self.assertTrue(r.review)
            self.assertIsNone(r.score)
            self.assertEqual(r.level, 0)
            self.assertEqual(r.quality, "poor")

    def test_explanation_lists_the_components(self):
        text = self.results["peak"].explain()
        self.assertTrue(text.startswith("Energy "))
        for name in ("activity", "impact", "pulse", "density", "sustain", "tempo"):
            self.assertIn(name, text)
        self.assertTrue(scored(np.zeros(SR * 45)).explain().startswith("Not rated: "))


class Sections(unittest.TestCase):
    def setUp(self):
        quiet = bench.drone(30) * 0.1
        self.track = np.concatenate([quiet, bench.peak_time(30)])

    def test_phrases_are_used_when_they_point_at_the_intense_part(self):
        f = measured(self.track, [(32.0, 58.0)])
        self.assertEqual(f["x_sections"], "phrases")
        self.assertEqual(f["x_warnings"], [])

    def test_phrases_in_a_quiet_part_are_not_trusted(self):
        f = measured(self.track, [(2.0, 28.0)])
        self.assertEqual(f["x_sections"], "intensity")
        self.assertTrue(any("drop/chorus" in w for w in f["x_warnings"]), f["x_warnings"])
        self.assertAlmostEqual(f["x_onsets"], measured(self.track)["x_onsets"], places=6)

    def test_without_phrases_the_most_intense_part_is_used(self):
        whole = measured(self.track)
        drop_only = measured(bench.peak_time(30))
        self.assertEqual(whole["x_sections"], "intensity")
        self.assertAlmostEqual(whole["x_onsets"], drop_only["x_onsets"], delta=1.0)


class Scoring(unittest.TestCase):
    def test_score_is_absolute(self):
        # A track's rating doesn't depend on which other tracks are rated with it.
        mine = {"a": fake(0.3), "b": fake(0.6)}
        others = {f"x{i}": fake(v) for i, v in enumerate((0.0, 0.05, 0.9, 1.0, 1.0))}
        alone = energy.rate_tracks(mine, {"a": 120, "b": 128})
        together = energy.rate_tracks(mine | others, {"a": 120, "b": 128})
        for tid in mine:
            self.assertEqual(alone[tid].score, together[tid].score)
            self.assertEqual(alone[tid].level, together[tid].level)

    def test_levels_follow_the_documented_thresholds(self):
        self.assertEqual([energy.absolute_level(s) for s in (0, 19.9, 20, 40, 60, 79.9, 80, 100)],
                         [1, 1, 2, 3, 4, 4, 5, 5])
        r = energy.score_track(fake(0.5), 0)
        used = [c for c in r.components.values() if c[1] is not None]
        self.assertAlmostEqual(r.score, 100 * sum(c[1] * c[2] for c in used)
                               / sum(c[2] for c in used), delta=0.06)
        self.assertEqual(r.level, 3)

    def test_a_library_of_similar_tracks_isnt_spread_to_1_and_5(self):
        feats = {str(i): fake(0.45 + 0.01 * i) for i in range(10)}
        levels = energy.levels_from_features(feats, {})
        self.assertEqual(set(levels.values()), {3})
        relative = energy.levels_from_features(feats, {}, relative=True)
        self.assertEqual([relative[str(i)] for i in range(10)], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_relative_mode_ranks_within_the_library(self):
        for n in (2, 3, 7):
            levels = energy.levels_from_features({str(i): fake(i / 10) for i in range(n)}, {},
                                                 relative=True)
            self.assertEqual((levels["0"], levels[str(n - 1)]), (1, 5), n)
        self.assertEqual(energy.levels_from_features({"a": fake(0.1)}, {}, relative=True),
                         {"a": 3})
        # Same scores either way; only the levels differ.
        feats = {str(i): fake(i / 10) for i in range(10)}
        a = energy.rate_tracks(feats, {})
        r = energy.rate_tracks(feats, {}, relative=True)
        self.assertEqual({t: x.score for t, x in a.items()}, {t: x.score for t, x in r.items()})

    def test_relative_levels_change_when_tracks_are_added(self):
        feats = {str(i): fake(0.3 + i / 50) for i in range(10)}
        before = energy.levels_from_features(feats, {}, relative=True)
        louder = {f"x{i}": fake(0.9) for i in range(10)}
        after = energy.levels_from_features(feats | louder, {}, relative=True)
        self.assertLess(after["9"], before["9"])
        absolute = energy.levels_from_features(feats, {})
        self.assertEqual(absolute, {t: v for t, v in
                                    energy.levels_from_features(feats | louder, {}).items()
                                    if t in feats})

    def test_one_extreme_track_doesnt_change_the_others(self):
        feats = {str(i): fake(i / 10) for i in range(10)}
        before = energy.levels_from_features(feats, {})
        feats["odd"] = fake(1.0, x_punch=1e6)
        after = energy.levels_from_features(feats, {})
        self.assertEqual({t: after[t] for t in before}, before)

    def test_copies_of_a_song_share_a_level(self):
        feats = {str(i): fake(i / 10) for i in range(10)}
        feats["copy"] = dict(feats["4"])
        for relative in (False, True):
            levels = energy.levels_from_features(feats, {t: 128 for t in feats}, relative=relative)
            self.assertEqual(levels["copy"], levels["4"])

    def test_missing_measurements_are_flagged_not_averaged(self):
        for key in ("x_pulse", "x_onsets", "x_impact", "x_density", "x_sustain"):
            r = energy.score_track(fake(0.5, **{key: None}), 128)
            self.assertTrue(r.review, key)
            self.assertIsNone(r.score)
            self.assertEqual(r.level, 0)
            self.assertTrue(any(w.startswith("couldn't measure") for w in r.warnings))
        # Flagged tracks aren't ranked and keep level 0 in relative mode too.
        feats = {"ok1": fake(0.2), "ok2": fake(0.8), "bad": fake(0.5, x_pulse=None)}
        levels = energy.levels_from_features(feats, {}, relative=True)
        self.assertEqual(levels, {"ok1": 1, "ok2": 5, "bad": 0})

    def test_old_measurements_are_not_scored(self):
        old = {"e_loud": -8.0, "e_punch": 0.3, "e_power": 0.2, "e_bass": 0.1, "e_highs": 0.1}
        r = energy.score_track(old, 128)
        self.assertTrue(r.review)
        self.assertIn("older version", r.warnings[-1])

    def test_unknown_tempo_and_mood_are_left_out_not_guessed(self):
        r = energy.score_track(fake(0.5), 0)
        self.assertIsNone(r.components["tempo"][1])
        self.assertIsNone(r.components["mood"][1])
        self.assertFalse(r.review)
        self.assertIn("no usable BPM and no clear beat: tempo not counted", r.warnings)
        self.assertEqual(r.quality, "limited")

    def test_tempo_only_counts_as_far_as_there_is_a_beat(self):
        steady = energy.score_track(fake(0.5), 174)
        weak = energy.score_track(fake(0.5, x_pulse=energy.CALIBRATION["audio"]["x_pulse"][0]),
                                  174)
        pulse = steady.components["pulse"][1]
        self.assertAlmostEqual(steady.components["tempo"][1],
                               energy._scale(174, *energy.TEMPO_RANGE) * pulse)
        self.assertEqual(weak.components["tempo"][1], 0.0)

    def test_close_to_a_boundary_is_noted(self):
        r = energy.score_track(fake(0.405), 0)
        self.assertTrue(any("boundary between levels 3 and 4" in n or
                            "boundary between levels 2 and 3" in n for n in r.notes), r)


class Tempo(unittest.TestCase):
    def test_half_speed_drum_and_bass_counts_double(self):
        self.assertEqual(energy.effective_bpm(87, "Drum & Bass"), 174)
        self.assertEqual(energy.effective_bpm(87, "Hip Hop"), 87)
        self.assertEqual(energy.effective_bpm(174, "Drum & Bass"), 174)
        dnb = energy.score_track(fake(0.5), 87, "Drum & Bass")
        hiphop = energy.score_track(fake(0.5), 87, "Hip Hop")
        self.assertGreater(dnb.score, hiphop.score)

    def test_missing_bpm_uses_the_beat_heard(self):
        tempo, warnings = energy.felt_tempo(0, "", 126.0)
        self.assertEqual(tempo, 126.0)
        self.assertIn("beat heard in the audio was used", warnings[0])
        self.assertEqual(energy.felt_tempo(0, "", None), (None, [
            "no usable BPM and no clear beat: tempo not counted"]))

    def test_double_and_half_time_readings_are_not_mismatches(self):
        for bpm in (64.0, 128.0, 256.0 / 2):
            self.assertEqual(energy.felt_tempo(bpm, "", 128.0)[1], [], bpm)

    def test_a_bpm_far_from_the_beat_heard_is_flagged(self):
        tempo, warnings = energy.felt_tempo(100, "", 128.0)
        self.assertEqual(tempo, 100)
        self.assertIn("may be wrong", warnings[0])

    def test_heard_tempo_of_a_beat(self):
        f = measured(banger(40, 140))
        self.assertAlmostEqual(f["x_tempo"], 140, delta=3)
        self.assertGreater(f["x_pulse"], 0.5)
        self.assertEqual(measured(tone())["x_pulse"], 0.0)


class WaveformTests(unittest.TestCase):
    def test_reads_rekordbox_waveform_and_ranks_energy(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        n = 150 * 60
        beat = (np.arange(n) % 70 < 10)               # a hit every ~0.47 s (128 BPM)
        quiet = os.path.join(d, "quiet.EXT")         # steady, smooth, mostly lows
        write_ext(quiet, np.full(n, 8), np.full(n, 6), np.full(n, 2), np.zeros(n, int))
        loud = os.path.join(d, "loud.EXT")           # punchy beat, lots of mids and highs
        write_ext(loud, np.where(beat, 30, 14), np.where(beat, 6, 2), np.full(n, 5),
                  np.full(n, 5))
        broken = os.path.join(d, "broken.EXT")
        with open(broken, "wb") as f:
            f.write(b"nope")

        self.assertEqual(len(energy.read_waveform(quiet)), n)
        logs = []
        feats = energy.analyse_waveforms({"q": quiet, "l": loud, "x": broken}, logs.append)
        self.assertEqual(set(feats), {"q", "l"})
        self.assertIn("1 tracks have no usable analysis", logs[-1])
        self.assertEqual(feats["l"]["x_mode"], "waveform")
        self.assertIsNone(feats["l"]["x_onsets"])       # can't be resolved from a waveform
        res = energy.rate_tracks(feats, {"q": 128, "l": 128})
        self.assertLess(res["q"].score, res["l"].score)

    def test_energy_is_judged_in_the_drops(self):
        # 60 s: quiet and still except a loud, pumping, bass-heavy drop at 20-40 s.
        def entry(height, low, high):
            return (low << 13) | (3 << 10) | (high << 7) | (height << 2)
        i = np.arange(60 * 150)
        pump = np.where(i % 70 < 10, entry(30, 7, 4), entry(18, 3, 4))
        wave = np.where((i >= 20 * 150) & (i < 40 * 150), pump, entry(6, 2, 1)).astype(np.uint16)
        with_drop = energy.measure_waveform(wave, [(20.0, 40.0)], mood=1)
        self.assertEqual(with_drop["x_sections"], "phrases")
        self.assertEqual(with_drop["mood"], 1)
        wrong_spot = energy.measure_waveform(wave, [(0.0, 20.0)])
        self.assertEqual(wrong_spot["x_sections"], "intensity")   # not trusted
        self.assertTrue(wrong_spot["x_warnings"])
        self.assertAlmostEqual(wrong_spot["x_punch"], with_drop["x_punch"], delta=0.05)
        no_phrases = energy.measure_waveform(wave)
        self.assertEqual(no_phrases["x_sections"], "intensity")
        self.assertAlmostEqual(no_phrases["x_punch"], with_drop["x_punch"], delta=0.05)


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cache = os.path.join(self.dir, "cache.json")
        self.paths = {}
        for tid, (name, signal) in enumerate(
                [("tone", tone()), ("groove", groove()), ("banger", banger())], 1):
            path = os.path.join(self.dir, f"{name}.wav")
            sf.write(path, signal.astype(np.float32), SR)
            self.paths[str(tid)] = path

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_ordering_and_cache(self):
        logs = []
        feats = energy.analyse_library(self.paths, logs.append, cache_file=self.cache)
        self.assertEqual(set(feats), {"1", "2", "3"})
        res = energy.rate_tracks(feats, {"1": 0, "2": 110, "3": 140})
        self.assertLess(res["1"].score, res["2"].score)
        self.assertLess(res["2"].score, res["3"].score)
        self.assertEqual(res["1"].level, 1)

        logs.clear()
        again = energy.analyse_library(self.paths, logs.append, cache_file=self.cache)
        self.assertIn("3 tracks already analysed, 0 to listen to", logs[0])
        self.assertEqual(again, feats)

    def test_cache_from_another_version_is_not_reused(self):
        energy.analyse_library(self.paths, lambda m: None, cache_file=self.cache)
        with open(self.cache, encoding="utf-8") as f:
            data = json.load(f)
        data["version"] = energy.CACHE_VERSION - 1
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logs = []
        energy.analyse_library(self.paths, logs.append, cache_file=self.cache)
        self.assertIn("0 tracks already analysed, 3 to listen to", logs[0])

    def test_incomplete_cache_entries_are_not_reused(self):
        energy.analyse_library(self.paths, lambda m: None, cache_file=self.cache)
        with open(self.cache, encoding="utf-8") as f:
            data = json.load(f)
        entry = data["tracks"][self.paths["2"]]
        del entry["features"]["x_pulse"]
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logs = []
        feats = energy.analyse_library(self.paths, logs.append, cache_file=self.cache)
        self.assertIn("2 tracks already analysed, 1 to listen to", logs[0])
        self.assertIn("x_pulse", feats["2"])

    def test_changed_phrase_analysis_is_measured_again(self):
        hints = {"3": ([(5.0, 35.0)], 1)}
        energy.analyse_library(self.paths, lambda m: None, cache_file=self.cache, hints=hints)
        logs = []
        feats = energy.analyse_library(self.paths, logs.append, cache_file=self.cache,
                                       hints={"3": ([(10.0, 30.0)], 1)})
        self.assertIn("2 tracks already analysed, 1 to listen to", logs[0])
        self.assertEqual(feats["3"]["mood"], 1)

    def test_direct_mode_measures_audio_and_falls_back_to_the_waveform(self):
        n = 150 * 60
        ext = {}
        for tid in ("1", "2", "9"):
            ext[tid] = os.path.join(self.dir, f"{tid}.EXT")
            write_ext(ext[tid], np.where(np.arange(n) % 70 < 10, 28, 12), np.full(n, 4),
                      np.full(n, 3), np.full(n, 3))
        bad = os.path.join(self.dir, "bad.mp3")
        with open(bad, "wb") as f:
            f.write(b"not audio")
        logs = []
        feats = energy.analyse_rekordbox({"1": self.paths["1"], "2": self.paths["2"], "9": bad},
                                         ext, logs.append, cache_file=self.cache)
        self.assertEqual(feats["1"]["x_mode"], "audio")
        self.assertEqual(feats["9"]["x_mode"], "waveform")
        self.assertTrue(any("1 tracks couldn't be decoded" in m for m in logs), logs)
        # Genre guessing gets rekordbox's whole-track measurements for every track.
        wave = energy.measure_waveform(*energy.read_analysis(ext["1"]))
        for k in energy.WHOLE_TRACK:
            self.assertEqual(feats["1"][k], wave[k])

    def test_unreadable_file_is_skipped(self):
        bad = os.path.join(self.dir, "bad.mp3")
        with open(bad, "wb") as f:
            f.write(b"not audio")
        feats = energy.analyse_library({"9": bad, **self.paths}, lambda m: None,
                                       cache_file=self.cache)
        self.assertNotIn("9", feats)
        self.assertEqual(len(feats), 3)

    def test_m4a_without_ffmpeg_says_what_is_needed(self):
        m4a = os.path.join(self.dir, "song.m4a")
        with open(m4a, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 64)
        logs = []

        def _decode_pyav(path):
            raise ImportError("no av")              # as in the packaged app
        with mock.patch.object(energy, "_decode_pyav", _decode_pyav), \
                mock.patch.object(energy.shutil, "which", return_value=None):
            feats = energy.analyse_library({"9": m4a}, logs.append, cache_file=self.cache)
        self.assertEqual(feats, {})
        self.assertTrue(any("needs FFmpeg" in m and "left unchanged" in m for m in logs), logs)

    def _xml(self, ratings):
        tracks = "".join(
            f'<TRACK TrackID="{tid}" Name="T{tid}" Artist="A" AverageBpm="0" Rating="{rating}" '
            f'Location="file://localhost/{path.replace(os.sep, "/").lstrip("/")}"/>'
            for (tid, path), rating in zip(self.paths.items(), ratings))
        xml_path = os.path.join(self.dir, "lib.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(f'<DJ_PLAYLISTS Version="1.0.0"><COLLECTION>{tracks}</COLLECTION>'
                    f'<PLAYLISTS><NODE Type="0" Name="ROOT" Count="0"/></PLAYLISTS></DJ_PLAYLISTS>')
        return xml_path

    def _run(self, xml_path, **options):
        logs = []
        with mock.patch.object(energy, "cache_path", lambda: self.cache):
            opt = organizer.Options(energy=True, genre=False, bpm=False, key=False,
                                    duplicates=False, missing_files=False, missing_info=False,
                                    **options)
            out = organizer.run(opt, xml_path, log=logs.append)
        root = ET.parse(out).getroot()
        ratings = {t.get("TrackID"): t.get("Rating") for t in root.iter("TRACK")
                   if t.get("TrackID")}
        return root, ratings, logs

    def test_xml_run_writes_energy_ratings_and_playlists(self):
        root, ratings, logs = self._run(self._xml(["0", "0", "51"]))
        self.assertEqual(ratings["3"], "51")             # already rated: kept
        self.assertNotEqual(ratings["1"], "0")           # newly rated from energy
        folder = root.find("PLAYLISTS/NODE/NODE[@Name='Organized']/NODE[@Name='By Energy']")
        self.assertEqual(sum(len(pl.findall("TRACK")) for pl in folder), 3)
        self.assertTrue(any(m.startswith("Energy levels: ") for m in logs), logs)

    def test_xml_run_in_relative_mode(self):
        _, ratings, logs = self._run(self._xml(["0", "0", "0"]), energy_experimental=True,
                                     energy_relative=True)
        self.assertTrue(any("ranked within this library" in m for m in logs), logs)
        stars = {tid: int(r) for tid, r in ratings.items()}
        self.assertLess(stars["1"], stars["3"])
        self.assertEqual(stars["3"], 255)                # the top of this library: 5 stars

    def test_tracks_that_cant_be_rated_keep_their_rating(self):
        short = os.path.join(self.dir, "short.wav")
        sf.write(short, banger(10).astype(np.float32), SR)
        self.paths["2"] = short
        _, ratings, logs = self._run(self._xml(["0", "102", "0"]), energy_experimental=True)
        self.assertEqual(ratings["2"], "102")
        self.assertTrue(any("couldn't be rated reliably" in m for m in logs), logs)


class ClassicAlgorithm(unittest.TestCase):
    """The classic (v3) algorithm, the default, must behave exactly as committed.
    benchmark_energy keeps a frozen copy of v3 (checked identical to the commit on
    a real library), so it's the reference here."""

    def test_classic_is_the_default(self):
        self.assertFalse(organizer.Options().energy_experimental)

    def test_audio_measurements_match_v3(self):
        for signal in (tone(), groove(), banger()):
            x = np.asarray(signal, dtype=np.float32)
            got = energy.measure(x, SR)
            want = bench.v3_audio(x, SR)
            for k in energy.CLASSIC_FEATURES:
                self.assertAlmostEqual(got[k], want[k], places=9, msg=k)

    def test_waveform_measurements_match_v3(self):
        rng = np.random.default_rng(3)
        n = 150 * 90
        heights = np.clip(rng.integers(0, 31, n), 0, 31)
        entries = ((rng.integers(0, 8, n) << 13) | (rng.integers(0, 8, n) << 10)
                   | (rng.integers(0, 8, n) << 7) | (heights << 2)).astype(np.int32)
        for sections, mood in (([(20.0, 50.0)], 1), ([], 0), ([(10.0, 11.0)], 3)):
            got = energy.measure_waveform(entries, sections, mood)
            want = bench.v3_waveform(entries, sections, mood)
            for k in energy.CLASSIC_FEATURES:
                self.assertAlmostEqual(got[k], want[k], places=9, msg=(k, sections))

    def test_scores_and_levels_match_v3(self):
        rng = np.random.default_rng(5)
        feats = {str(i): {k: float(rng.normal()) for k in energy.CLASSIC_FEATURES}
                 | {"mood": int(rng.integers(0, 4))} for i in range(40)}
        feats["7"]["e_bass"] = None                         # missing: the middle, as before
        bpms = {str(i): float(rng.choice([0, 87, 124, 140, 174])) for i in range(40)}
        genres = {str(i): rng.choice(["Drum & Bass", "House", ""]) for i in range(40)}
        scores, levels = energy.classic_levels(feats, bpms, genres)
        want = bench.v3_scores(feats, bpms, genres)
        self.assertEqual(set(scores), set(want))
        for i in want:
            self.assertAlmostEqual(scores[i], want[i], places=12)
        self.assertEqual(levels, energy.relative_levels(want))
        self.assertEqual(sorted(set(levels.values())), [1, 2, 3, 4, 5])

    def test_tracks_without_classic_measurements_get_no_level(self):
        feats = {"a": {k: 1.0 for k in energy.CLASSIC_FEATURES}, "b": {"x_mode": "audio"},
                 "c": {k: 2.0 for k in energy.CLASSIC_FEATURES}}
        scores, levels = energy.classic_levels(feats, {}, {})
        self.assertEqual(set(levels), {"a", "c"})
        self.assertEqual((levels["a"], levels["c"]), (1, 5))


class ClassicRuns(unittest.TestCase):
    """Runs through organizer with the default (classic) algorithm."""
    setUp, tearDown = EnergyTests.setUp, EnergyTests.tearDown
    _xml, _run = EnergyTests._xml, EnergyTests._run

    def test_xml_run_ranks_the_library(self):
        _, ratings, logs = self._run(self._xml(["0", "0", "0"]))
        stars = {tid: int(r) for tid, r in ratings.items()}
        self.assertEqual((stars["1"], stars["3"]), (51, 255))   # calmest 1, top 5 stars
        self.assertTrue(any(m.startswith("Energy levels: ") for m in logs), logs)
        self.assertTrue(any("Most energetic: T3" in m for m in logs), logs)
        self.assertFalse(any("ranked within this library" in m for m in logs), logs)

    def test_xml_run_levels_match_v3_on_the_same_measurements(self):
        feats = energy.analyse_library(self.paths, lambda m: None, cache_file=self.cache)
        tracks = [organizer.Track(id=t, name=t, artist="", genre="", label="", bpm=b, key="",
                                  year=0, rating=0, colour="", date_added="", play_count=0,
                                  path=self.paths[t]) for t, b in (("1", 0), ("2", 110), ("3", 140))]
        organizer.apply_energy(tracks, organizer.Options(energy=True), feats, lambda m: None)
        want = energy.relative_levels(bench.v3_scores(
            {t: feats[t] for t in feats}, {"1": 0, "2": 110, "3": 140}, {t: " " for t in feats}))
        self.assertEqual({t.id: t.energy for t in tracks}, want)

    def test_direct_mode_classic_measurements_come_from_the_waveform_only(self):
        n = 150 * 60
        ext = {}
        for tid in ("1", "2"):
            ext[tid] = os.path.join(self.dir, f"{tid}.EXT")
            write_ext(ext[tid], np.where(np.arange(n) % 70 < 10, 28, 12), np.full(n, 4),
                      np.full(n, 3), np.full(n, 3))
        feats = energy.analyse_rekordbox({"1": self.paths["1"], "2": self.paths["2"],
                                          "3": self.paths["3"]}, ext, lambda m: None,
                                         cache_file=self.cache)
        wave = energy.measure_waveform(*energy.read_analysis(ext["1"]))
        for k in energy.CLASSIC_FEATURES:
            self.assertEqual(feats["1"][k], wave[k])
        # Decoded but no rekordbox analysis: not ranked, as when only waveforms were read.
        self.assertNotIn("e_loud", feats["3"])
        self.assertEqual(set(energy.classic_scores(feats, {}, {})), {"1", "2"})

    def test_cache_without_classic_measurements_is_measured_again(self):
        energy.analyse_library(self.paths, lambda m: None, cache_file=self.cache)
        with open(self.cache, encoding="utf-8") as f:
            data = json.load(f)
        del data["tracks"][self.paths["2"]]["features"]["e_loud"]
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logs = []
        feats = energy.analyse_library(self.paths, logs.append, cache_file=self.cache)
        self.assertIn("2 tracks already analysed, 1 to listen to", logs[0])
        self.assertIn("e_loud", feats["2"])


if __name__ == "__main__":
    unittest.main()
