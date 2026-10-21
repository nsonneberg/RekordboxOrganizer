import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import cues
import updater
from cues import Phrase


class ChooseHotCueTests(unittest.TestCase):
    PHRASES = [Phrase(0, "Intro"), Phrase(15000, "Build-up"), Phrase(30000, "Drop"),
               Phrase(60000, "Breakdown"), Phrase(75000, "Build-up"), Phrase(90000, "Drop"),
               Phrase(120000, "Verse"), Phrase(135000, "Build-up"), Phrase(150000, "Drop"),
               Phrase(180000, "Outro")]

    def test_fills_pads_in_time_order(self):
        picks = cues.choose_hot_cues(self.PHRASES[:4], set(), [])
        self.assertEqual([(pad, p.label) for pad, p in picks],
                         [(0, "Intro"), (1, "Build-up"), (2, "Drop"), (3, "Breakdown")])

    def test_keeps_most_useful_when_pads_run_out(self):
        picks = cues.choose_hot_cues(self.PHRASES, set(), [])
        self.assertEqual(len(picks), 8)
        labels = [p.label for _, p in picks]
        self.assertNotIn("Verse", labels)                 # lowest priority dropped first
        self.assertEqual(labels.count("Drop"), 3)
        times = [p.start_ms for _, p in picks]
        self.assertEqual(times, sorted(times))

    def test_never_touches_used_pads_or_doubles_up(self):
        # Pads A and C taken; one existing cue sits right on the first drop.
        picks = cues.choose_hot_cues(self.PHRASES[:4], {0, 2}, [0, 30400])
        self.assertEqual([(pad, p.label) for pad, p in picks], [(1, "Build-up"), (3, "Breakdown")])
        self.assertEqual(cues.choose_hot_cues(self.PHRASES, set(range(8)), []), [])

    def test_missing_analysis(self):
        self.assertEqual(cues.read_phrases(""), [])
        self.assertEqual(cues.read_phrases("nope/ANLZ0000.EXT"), [])


class _Resp(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


class UpdaterTests(unittest.TestCase):
    def release(self, tag):
        return {"tag_name": tag, "body": "New stuff", "html_url": "https://example/rel",
                "assets": [{"name": name, "browser_download_url": "https://example/" + name}
                           for name in updater.ASSETS.values()]}

    def check(self, tag, current):
        payload = json.dumps(self.release(tag)).encode()
        with mock.patch.object(updater, "VERSION", current), \
             mock.patch.object(updater, "_open", lambda url, timeout: _Resp(payload)):
            return updater.check()

    def test_version_compare(self):
        self.assertIsNone(self.check("v1.0.0", "1.0.0"))
        self.assertIsNone(self.check("v0.9.5", "1.0.0"))
        self.assertIsNone(self.check("v1.0.10", "1.0.10"))
        found = self.check("v1.0.10", "1.0.9")           # numeric, not text, comparison
        self.assertEqual(found["version"], "1.0.10")
        if sys.platform in updater.ASSETS:
            self.assertTrue(found["url"].endswith(updater.ASSETS[sys.platform]))

    def test_source_runs_never_self_update(self):
        with mock.patch.object(sys, "frozen", False, create=True):
            self.assertFalse(updater.can_self_update())

    @unittest.skipUnless(sys.platform == "win32", "Windows swap")
    def test_windows_swap(self):
        folder = tempfile.mkdtemp()
        exe = os.path.join(folder, "Rekordbox Organizer.exe")
        with open(exe, "wb") as f:
            f.write(b"old version")
        with mock.patch.object(sys, "executable", exe), \
             mock.patch.object(updater, "_open", lambda url, timeout: _Resp(b"new version")), \
             mock.patch.object(updater.subprocess, "Popen") as started:
            seen = []
            updater.install("https://example/x.exe", lambda d, t: seen.append((d, t)))
        with open(exe, "rb") as f:
            self.assertEqual(f.read(), b"new version")
        with open(exe + ".old", "rb") as f:
            self.assertEqual(f.read(), b"old version")
        started.assert_called_once_with([exe], close_fds=True)
        self.assertEqual(seen[-1], (11, 11))
        with mock.patch.object(sys, "executable", exe), \
             mock.patch.object(sys, "frozen", True, create=True):
            updater.cleanup_old()
        self.assertFalse(os.path.exists(exe + ".old"))


if __name__ == "__main__":
    unittest.main()
