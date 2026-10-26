import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

import energy
import organizer

SR = 22050


def tone(seconds=20):
    """Quiet, smooth sine: low energy."""
    t = np.arange(SR * seconds) / SR
    return 0.05 * np.sin(2 * np.pi * 220 * t)


def groove(seconds=20, bpm=110):
    """Moderate kick pattern with a pad."""
    t = np.arange(SR * seconds) / SR
    out = 0.15 * np.sin(2 * np.pi * 110 * t)
    beat = int(SR * 60 / bpm)
    kick = 0.4 * np.sin(2 * np.pi * 60 * np.arange(2000) / SR) * np.exp(-np.arange(2000) / 400)
    for start in range(0, len(out) - 2000, beat):
        out[start:start + 2000] += kick
    return out


def banger(seconds=20, bpm=140):
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


class WaveformTests(unittest.TestCase):
    def test_reads_rekordbox_waveform_and_ranks_energy(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        rng = np.random.default_rng(1)
        n = 150 * 60
        quiet = os.path.join(d, "quiet.EXT")       # soft, smooth, mostly lows
        write_ext(quiet, np.full(n, 8), np.full(n, 6), np.full(n, 2), np.zeros(n, int))
        loud = os.path.join(d, "loud.EXT")         # loud, punchy, lots of highs
        write_ext(loud, rng.integers(18, 31, n), np.full(n, 5), np.full(n, 3), np.full(n, 6))
        broken = os.path.join(d, "broken.EXT")
        with open(broken, "wb") as f:
            f.write(b"nope")

        self.assertEqual(len(energy.read_waveform(quiet)), n)
        logs = []
        feats = energy.analyse_waveforms({"q": quiet, "l": loud, "x": broken}, logs.append)
        self.assertEqual(set(feats), {"q", "l"})
        self.assertIn("1 tracks have no usable analysis", logs[-1])
        levels = energy.levels_from_features(feats, {})
        self.assertLess(levels["q"], levels["l"])


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
        levels = energy.levels_from_features(feats, {"1": 0, "2": 110, "3": 140})
        self.assertLess(levels["1"], levels["2"])
        self.assertLess(levels["2"], levels["3"])
        self.assertEqual(levels["3"], 5)

        logs.clear()
        energy.analyse_library(self.paths, logs.append, cache_file=self.cache)
        self.assertIn("3 tracks already analysed, 0 to listen to", logs[0])

    def test_unreadable_file_is_skipped(self):
        bad = os.path.join(self.dir, "bad.mp3")
        with open(bad, "wb") as f:
            f.write(b"not audio")
        feats = energy.analyse_library({"9": bad, **self.paths}, lambda m: None,
                                       cache_file=self.cache)
        self.assertNotIn("9", feats)
        self.assertEqual(len(feats), 3)

    @staticmethod
    def feats(value, **extra):
        return {name: value for name in energy.FEATURES} | extra

    def test_levels_spread_across_library(self):
        feats = {str(i): self.feats(i) for i in range(10)}
        levels = energy.levels_from_features(feats, {})
        self.assertEqual([levels[str(i)] for i in range(10)], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_highest_is_5_and_lowest_is_1_even_in_tiny_libraries(self):
        for n in (2, 3, 4, 7):
            levels = energy.levels_from_features({str(i): self.feats(i) for i in range(n)}, {})
            self.assertEqual((levels["0"], levels[str(n - 1)]), (1, 5), n)
        self.assertEqual(energy.levels_from_features({"a": self.feats(1)}, {}), {"a": 3})

    def test_one_extreme_track_doesnt_skew_the_rest(self):
        feats = {str(i): self.feats(i) for i in range(10)}
        feats["9"]["e_punch"] = 1e6          # e.g. an odd demo track
        levels = energy.levels_from_features(feats, {})
        self.assertEqual([levels[str(i)] for i in range(10)], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_copies_of_a_song_share_a_level(self):
        feats = {str(i): self.feats(i) for i in range(10)}
        feats["copy"] = dict(feats["4"])
        levels = energy.levels_from_features(feats, {str(i): 128 for i in range(10)} | {"copy": 128})
        self.assertEqual(levels["copy"], levels["4"])

    def test_half_speed_drum_and_bass_counts_double(self):
        self.assertEqual(energy.effective_bpm(87, "Drum & Bass"), 174)
        self.assertEqual(energy.effective_bpm(87, "Hip Hop"), 87)
        self.assertEqual(energy.effective_bpm(174, "Drum & Bass"), 174)
        feats = {"dnb": self.feats(5), "house": self.feats(5)}
        scores = energy.energy_scores(feats, {"dnb": 87, "house": 128},
                                      {"dnb": "Drum & Bass", "house": "House"})
        self.assertGreater(scores["dnb"], scores["house"])

    def test_energy_is_judged_in_the_drops(self):
        # 60 s track: quiet everywhere except a loud, bass-heavy drop at 20-40 s.
        def entry(height, low, high):
            return (low << 13) | (3 << 10) | (high << 7) | (height << 2)
        quiet, loud = entry(8, 2, 1), entry(28, 7, 4)
        wave = np.array([loud if 20 * 150 <= i < 40 * 150 else quiet for i in range(60 * 150)],
                        dtype=np.uint16)
        with_drop = energy.measure_waveform(wave, [(20.0, 40.0)], mood=1)
        self.assertEqual(with_drop["e_loud"], 28)
        self.assertEqual(with_drop["mood"], 1)
        wrong_spot = energy.measure_waveform(wave, [(0.0, 20.0)])
        self.assertLess(wrong_spot["e_loud"], with_drop["e_loud"])
        # Without phrase analysis the loudest stretch is used instead.
        self.assertEqual(energy.measure_waveform(wave)["e_loud"], 28)

    def test_xml_run_writes_energy_ratings_and_playlists(self):
        tracks = "".join(
            f'<TRACK TrackID="{tid}" Name="T{tid}" Artist="A" AverageBpm="0" Rating="{rating}" '
            f'Location="file://localhost/{path.replace(os.sep, "/").lstrip("/")}"/>'
            for (tid, path), rating in zip(self.paths.items(), ["0", "0", "51"]))
        xml_path = os.path.join(self.dir, "lib.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(f'<DJ_PLAYLISTS Version="1.0.0"><COLLECTION>{tracks}</COLLECTION>'
                    f'<PLAYLISTS><NODE Type="0" Name="ROOT" Count="0"/></PLAYLISTS></DJ_PLAYLISTS>')

        energy_cache = energy.cache_path
        energy.cache_path = lambda: self.cache
        try:
            opt = organizer.Options(energy=True, genre=False, bpm=False, key=False,
                                    duplicates=False, missing_files=False, missing_info=False)
            out = organizer.run(opt, xml_path, log=lambda m: None)
        finally:
            energy.cache_path = energy_cache

        root = ET.parse(out).getroot()
        ratings = {t.get("TrackID"): t.get("Rating") for t in root.iter("TRACK") if t.get("TrackID")}
        self.assertEqual(ratings["3"], "51")             # already rated: kept
        self.assertNotEqual(ratings["1"], "0")           # newly rated from energy
        folder = root.find("PLAYLISTS/NODE/NODE[@Name='Organized']/NODE[@Name='By Energy']")
        self.assertEqual(sum(len(pl.findall("TRACK")) for pl in folder), 3)


if __name__ == "__main__":
    unittest.main()
