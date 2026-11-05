"""Tests for genre_model.py. A fake classifier stands in for the model.

The real model is only used by RealModelTests, which run when the environment
variable RBO_REAL_MODEL=1 is set (they download the 16 MB model once):

    set RBO_REAL_MODEL=1   (Windows)   /   export RBO_REAL_MODEL=1   (macOS)
    python -m unittest test_genre_model.RealModelTests -v
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

import genre_model


class FakeClassifier:
    """Deterministic stand-in: "House music" rises with loudness; counts calls."""

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def run(self, wave):
        if self.fail:
            raise RuntimeError("model broke")
        self.calls += 1
        frames = max(1, int((len(wave) / genre_model.SR - 0.96) / 0.48) + 1)
        scores = np.zeros((frames, 521), np.float32)
        level = float(np.sqrt(np.mean(wave ** 2)))
        scores[:, genre_model.MUSIC_INDEX] = 0.9
        scores[:, 235] = min(1.0, level * 2)        # House music
        emb = np.ones((frames, 1024), np.float32) * level
        return scores, emb


def tone(seconds, sr=44100, amp=0.3, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class SegmentTests(unittest.TestCase):
    def test_body_of_the_track_not_the_intro(self):
        starts = genre_model.choose_segments(300)
        self.assertEqual(starts, [85.0, 145.0, 205.0])          # 30% / 50% / 70%, centred
        self.assertTrue(all(s >= 30 for s in starts))

    def test_drops_first(self):
        starts = genre_model.choose_segments(300, [(60, 90), (150, 200), (250, 256)])
        self.assertIn(170.0, starts)                            # middle of the longest drop
        self.assertIn(70.0, starts)
        self.assertLessEqual(len(starts), genre_model.MAX_SEGMENTS)
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(starts, starts[1:]):
            self.assertGreaterEqual(b - a, genre_model.SEGMENT_SECONDS)

    def test_short_and_unknown_length(self):
        self.assertEqual(genre_model.choose_segments(12), [0.0])
        self.assertEqual(genre_model.choose_segments(0), [])
        self.assertEqual(genre_model.choose_segments(-1), [])

    def test_deterministic(self):
        sections = [(10, 40), (100, 130)]
        self.assertEqual(genre_model.choose_segments(240, sections),
                         genre_model.choose_segments(240, list(sections)))


class AudioTests(unittest.TestCase):
    def test_resample(self):
        x = tone(1.0, 44100)
        y = genre_model.resample(x, 44100)
        self.assertEqual(len(y), 16000)
        self.assertEqual(y.dtype, np.float32)
        self.assertAlmostEqual(float(np.sqrt(np.mean(y ** 2))),
                               float(np.sqrt(np.mean(x ** 2))), places=2)
        self.assertIs(genre_model.resample(y, 16000).dtype, np.dtype(np.float32))

    def test_silence_and_short_excerpts_skipped(self):
        sr = 22050
        audio = np.concatenate([np.zeros(sr * 20, np.float32), tone(20, sr)])
        used = genre_model.excerpts(audio, sr, [0.0, 25.0, 39.0])
        self.assertEqual([s for s, _ in used], [25.0])          # silent / 1 s tail dropped
        with self.assertRaises(ValueError):
            genre_model.analyse_audio(np.zeros(sr * 30, np.float32), sr, [5.0], FakeClassifier())

    def test_pooling(self):
        sr = 22050
        audio = np.concatenate([tone(20, sr, 0.1), tone(20, sr, 0.4)])
        result = genre_model.analyse_audio(audio, sr, [2.0, 25.0], FakeClassifier())
        self.assertEqual(result.segments, [2.0, 25.0])
        self.assertEqual(len(result.genre_scores), 2)
        self.assertLess(result.genre_scores[0]["House music"], result.genre_scores[1]["House music"])
        self.assertAlmostEqual(float(np.linalg.norm(result.embedding)), 1.0, places=4)
        again = genre_model.ModelResult.from_json(json.loads(json.dumps(result.to_json())))
        self.assertEqual(again.genre_scores, result.genre_scores)
        np.testing.assert_allclose(again.embedding, result.embedding, atol=1e-3)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.cache = os.path.join(self.dir, "cache.json")

    def wav(self, name, seconds, amp=0.3):
        path = os.path.join(self.dir, name)
        sf.write(path, tone(seconds, 22050, amp), 22050)
        return path

    def test_problems_are_skipped_and_counted(self):
        good = self.wav("good.wav", 60)
        broken = os.path.join(self.dir, "broken.mp3")
        with open(broken, "wb") as f:
            f.write(b"\x00" * 4000)
        short = self.wav("short.wav", 1)
        silent = os.path.join(self.dir, "silent.wav")
        sf.write(silent, np.zeros(22050 * 30, np.float32), 22050)
        items = [genre_model.TrackInput("good", good, 60),
                 genre_model.TrackInput("gone", os.path.join(self.dir, "missing.wav"), 200),
                 genre_model.TrackInput("broken", broken, 0),
                 genre_model.TrackInput("short", short, 1),
                 genre_model.TrackInput("silent", silent, 30)]
        logs = []
        results = genre_model.analyse_tracks(items, logs.append, classifier=FakeClassifier(),
                                             cache_file=self.cache)
        self.assertEqual(list(results), ["good"])
        self.assertIn("1 file not found", logs[-1])
        self.assertIn("silent", logs[-1])
        self.assertIn("short", logs[-1])
        self.assertIn("unreadable", logs[-1])

    def test_model_failure_on_every_file(self):
        path = self.wav("a.wav", 40)
        logs = []
        results = genre_model.analyse_tracks([genre_model.TrackInput("a", path, 40)], logs.append,
                                             classifier=FakeClassifier(fail=True),
                                             cache_file=self.cache)
        self.assertEqual(results, {})
        self.assertIn("unreadable", logs[-1])

    def test_cache(self):
        path = self.wav("a.wav", 60)
        items = [genre_model.TrackInput("a", path, 60)]
        fake = FakeClassifier()
        first = genre_model.analyse_tracks(items, lambda m: None, classifier=fake,
                                           cache_file=self.cache)
        calls = fake.calls
        second = genre_model.analyse_tracks(items, lambda m: None, classifier=fake,
                                            cache_file=self.cache)
        self.assertEqual(fake.calls, calls)                       # nothing re-analysed
        self.assertEqual(first["a"].genre_scores, second["a"].genre_scores)

        # Different excerpts (rekordbox found drops) -> analysed again.
        items = [genre_model.TrackInput("a", path, 60, [(40, 50)])]
        genre_model.analyse_tracks(items, lambda m: None, classifier=fake, cache_file=self.cache)
        self.assertGreater(fake.calls, calls)

        # A new model / preprocessing version throws the whole cache away.
        calls = fake.calls
        with open(self.cache, encoding="utf-8") as f:
            data = json.load(f)
        data["key"] = "some-older-model|pre0"
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump(data, f)
        genre_model.analyse_tracks(items, lambda m: None, classifier=fake, cache_file=self.cache)
        self.assertGreater(fake.calls, calls)

        # The file changing on disk -> analysed again.
        calls = fake.calls
        sf.write(path, tone(60, 22050, 0.1), 22050)
        os.utime(path, (1_600_000_000, 1_600_000_000))
        genre_model.analyse_tracks(items, lambda m: None, classifier=fake, cache_file=self.cache)
        self.assertGreater(fake.calls, calls)

    def test_corrupt_cache_is_ignored(self):
        with open(self.cache, "w") as f:
            f.write("{not json")
        path = self.wav("a.wav", 30)
        results = genre_model.analyse_tracks([genre_model.TrackInput("a", path, 30)],
                                             lambda m: None, classifier=FakeClassifier(),
                                             cache_file=self.cache)
        self.assertIn("a", results)


class EssentiaFrontEndTests(unittest.TestCase):
    """The numpy copy of Essentia's TensorflowInputMusiCNN front end."""

    def test_filter_bank(self):
        bank = genre_model._mel_filters()
        self.assertEqual(bank.shape, (96, 257))
        self.assertTrue(np.all(bank >= 0))
        self.assertTrue(np.all(bank.max(axis=1) > 0))           # no empty band
        peaks = bank.argmax(axis=1)
        self.assertTrue(np.all(np.diff(peaks) >= 0))            # bands rise in pitch
        self.assertGreater(bank[0].max(), bank[95].max())      # unit_tri: wider bands are lower
        self.assertAlmostEqual(float(genre_model._mel2hz(genre_model._hz2mel(3000.0))), 3000.0, 3)

    def test_silence_and_shapes(self):
        mel = genre_model.melspectrogram(np.zeros(16000, np.float32))
        self.assertTrue(np.all(mel == 0))                    # log10(1 + 0)
        self.assertEqual(mel.shape[1], 96)
        self.assertEqual(len(mel), 1 + (16000 + 768 - 768) // 256)
        self.assertIsNone(genre_model.patches(mel[:100]))    # under one 128-frame patch
        p = genre_model.patches(genre_model.melspectrogram(np.zeros(16000 * 5, np.float32)))
        self.assertEqual(p.shape[1:], (128, 96))

    def test_tone_lands_in_the_right_band(self):
        sr = 16000
        t = np.arange(sr * 3) / sr
        mel = genre_model.melspectrogram(0.5 * np.sin(2 * np.pi * 1000 * t)).mean(axis=0)
        edges = genre_model._mel2hz(np.linspace(0, genre_model._hz2mel(8000), 98))
        centre = edges[1 + int(np.argmax(mel))]
        self.assertLess(abs(centre - 1000), 60)

    def test_resampler_rolls_off_near_nyquist(self):
        sr = 44100
        t = np.arange(sr) / sr
        low = genre_model.resample_essentia(np.sin(2 * np.pi * 1000 * t).astype(np.float32), sr)
        high = genre_model.resample_essentia(np.sin(2 * np.pi * 7900 * t).astype(np.float32), sr)
        self.assertEqual(len(low), 16000)
        self.assertGreater(np.std(low), 0.6)
        self.assertLess(np.std(high), 0.01)

    @unittest.skipUnless(os.environ.get("RBO_REAL_MODEL") == "1",
                         "set RBO_REAL_MODEL=1 to check against Essentia's reference values")
    def test_matches_essentia_reference(self):
        """Essentia's own regression test (test_tensorflowinputmusicnn.py) on vignesh.wav."""
        import io
        import ssl
        import urllib.request
        import certifi
        url = "https://raw.githubusercontent.com/MTG/essentia-audio/master/recorded/vignesh.wav"
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=60) as resp:
            audio, sr = sf.read(io.BytesIO(resp.read()), dtype="float32", always_2d=True)
        x16 = genre_model.resample_essentia(audio.mean(axis=1), sr)
        got = genre_model.melspectrogram(x16).mean(axis=0)
        err = np.abs(got - np.array(ESSENTIA_REFERENCE))
        self.assertLess(float(err.mean()), 0.01)      # Essentia's own test allows 1e-2
        self.assertLess(float(err.max()), 0.1)


ESSENTIA_REFERENCE = [
    0.60742337, 0.30574673, 0.45560792, 1.256332, 2.4021673, 3.4365354, 3.619476, 3.0021546,
    2.1846066, 1.5598421, 1.4810421, 2.2823677, 2.679103, 2.86526, 2.6989846, 2.2553382,
    2.219071, 2.3255587, 2.8502884, 2.9091403, 2.7634032, 2.4637504, 1.8271459, 1.522163,
    1.7100089, 1.8728845, 1.6959977, 1.3508593, 0.9608341, 0.9133418, 1.0304681, 1.493988,
    1.6636051, 1.4825928, 1.1171728, 0.93050385, 1.2989489, 1.7412357, 1.7828379, 1.5357956,
    1.0274258, 1.4541839, 1.8527577, 1.8838495, 1.4812496, 1.4385983, 2.1568356, 2.3677773,
    1.9438239, 1.5913178, 1.8563453, 1.7012404, 1.1431638, 1.0995349, 1.1092283, 0.74655735,
    0.6698305, 0.7290597, 0.47290954, 0.64479357, 0.7136836, 0.9934933, 1.3321629, 1.1683794,
    1.2097421, 1.1075293, 1.0301174, 0.9288259, 0.8876033, 0.8086145, 0.9854008, 1.0852002,
    1.2092237, 1.2816739, 1.2066866, 0.52382684, 0.1494276, 0.08070073, 0.09443883, 0.12541461,
    0.11942478, 0.1558171, 0.17869301, 0.36044103, 0.5242918, 0.7467586, 0.8322874, 0.7977463,
    0.8188014, 0.80939233, 0.74459517, 0.5341967, 0.4339693, 0.33098528, 0.10355855, 0.00549104]


class FakeDiscogsSession:
    """Stands in for the ONNX session: style 'Electronic---Techno' rises with level."""

    def __init__(self):
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        mel = next(iter(feeds.values()))
        assert mel.shape[1:] == (128, 96), mel.shape
        n = len(mel)
        scores = np.full((n, 400), 0.001, np.float32)
        from discogs_styles import STYLES
        scores[:, STYLES.index("Electronic---Techno")] = np.clip(mel.mean(axis=(1, 2)) / 3, 0, 1)
        scores[:, STYLES.index("Electronic---Tech House")] = 0.2
        return scores, np.ones((n, 1280), np.float32)


class DiscogsBackendTests(unittest.TestCase):
    def classifier(self):
        clf = genre_model.Classifier.__new__(genre_model.Classifier)
        clf.backend = genre_model.DISCOGS
        clf.session = FakeDiscogsSession()
        clf.input = "melspectrogram"
        return clf

    def test_analyse_audio(self):
        sr = 22050
        audio = tone(40, sr, 0.3)
        result = genre_model.analyse_audio(audio, sr, [2.0, 20.0], self.classifier())
        self.assertEqual(result.model, "discogs-effnet")
        self.assertEqual(len(result.genre_scores), 2)
        self.assertIn("Electronic---Techno", result.genre_scores[0])
        self.assertNotIn("Blues---Boogie Woogie", result.genre_scores[0])   # below DISCOGS_KEEP
        self.assertEqual(result.embedding.shape, (1280,))
        self.assertGreater(result.music[0], 0)

    def test_separate_caches(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "a.wav")
        sf.write(path, tone(30, 22050), 22050)
        item = [genre_model.TrackInput("a", path, 30)]
        y_cache = os.path.join(folder, "y.json")
        d_cache = os.path.join(folder, "d.json")
        genre_model.analyse_tracks(item, lambda m: None, classifier=FakeClassifier(),
                                   cache_file=y_cache)
        clf = self.classifier()
        r = genre_model.analyse_tracks(item, lambda m: None, classifier=clf, cache_file=d_cache)
        self.assertEqual(r["a"].model, "discogs-effnet")
        with open(d_cache, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["key"], genre_model.DISCOGS.cache_key)
        # A YAMNet cache file is never read as a Discogs one.
        self.assertEqual(genre_model._load_cache(y_cache, genre_model.DISCOGS.cache_key), {})
        before = clf.session.calls
        genre_model.analyse_tracks(item, lambda m: None, classifier=clf, cache_file=d_cache)
        self.assertEqual(clf.session.calls, before)

    def test_old_cache_entries_are_yamnet(self):
        old = {"segments": [1.0], "genre_scores": [{"House music": 0.5}], "music": [0.9],
               "embedding": None}
        self.assertEqual(genre_model.ModelResult.from_json(old).model, "yamnet")


@unittest.skipUnless(os.environ.get("RBO_REAL_MODEL") == "1",
                     "set RBO_REAL_MODEL=1 to test the real model (downloads 16 MB once)")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = genre_model.ensure_model(print)
        cls.classifier = genre_model.Classifier(cls.path)

    def test_checksum(self):
        self.assertEqual(genre_model._sha256(self.path), genre_model.MODEL["sha256"])

    def test_shapes_and_determinism(self):
        wave = genre_model.resample(tone(10, 22050), 22050)
        scores, emb = self.classifier.run(wave)
        self.assertEqual(scores.shape[1], 521)
        self.assertEqual(emb.shape[1], 1024)
        again, _ = self.classifier.run(wave)
        np.testing.assert_array_equal(scores, again)

    def test_class_map_matches(self):
        """GENRE_CLASSES must match the class map published with this model revision."""
        import csv
        import io
        import ssl
        import urllib.request
        import certifi
        url = genre_model.MODEL["url"].replace("yamnet.onnx", "yamnet_class_map.csv")
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=60) as resp:
            rows = list(csv.reader(io.StringIO(resp.read().decode("utf-8"))))[1:]
        names = {int(r[0]): r[2] for r in rows}
        self.assertEqual(names[genre_model.MUSIC_INDEX], "Music")
        for index, name in genre_model.GENRE_CLASSES.items():
            self.assertEqual(names[index], name)

    def test_music_is_music(self):
        sr = 22050
        t = np.arange(sr * 12) / sr
        beat = (np.sin(2 * np.pi * 55 * t) * (np.mod(t, 0.5) < 0.1)
                + 0.3 * np.sin(2 * np.pi * 440 * t) * (np.mod(t, 0.25) < 0.05)).astype(np.float32)
        result = genre_model.analyse_audio(beat * 0.5, sr, [0.0], self.classifier)
        self.assertEqual(len(result.genre_scores), 1)
        self.assertEqual(result.embedding.shape, (1024,))


if __name__ == "__main__":
    unittest.main()
